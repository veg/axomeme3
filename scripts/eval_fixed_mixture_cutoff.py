#!/usr/bin/env python3
"""
eval_fixed_mixture_cutoff.py
----------------------------
Evaluates Checkpoint 29 on official HyPhy benchmark empirical datasets
using FIXED theoretical LRT cutoffs derived from the MEME asymptotic null mixture distribution:
  P(LRT >= tau) = 0.5 * sf_chi1(tau) + 0.5 * sf_chi2(tau)

Fixed theoretical cutoffs:
- For p <= 0.05: Direct LRT >= 5.1384 (Model Log(LRT+1) >= 1.8146)
- For p <= 0.10: Direct LRT >= 3.8078 (Model Log(LRT+1) >= 1.5702)

Reports counts of TP, FP, FN, TN, empirical FPR, empirical TPR (Sensitivity), and Precision.
"""

import sys
import os
import glob
import json
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
from Bio import Phylo
from io import StringIO
from scipy.optimize import brentq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scripts.train_transformer_selection import (
    PhyloAxialTransformer,
    compute_mds_coordinates,
    get_patristic_distances,
    is_site_variable
)
from scripts.update_meme_db_filter import parse_nexus_gz
from scripts.predict_multitask_nexus import (
    parse_nexus_alignment_and_embedded_tree,
    get_codon_token,
    get_aa_token,
    AA_TO_IDX,
    GENETIC_CODE
)

HYPHY_TEST_DIR = "/Users/sergei/Development/hyphy/tests/data"

def get_mixture_pval(lrt_val):
    if lrt_val <= 0:
        return 1.0
    return 0.5 * stats.chi2.sf(lrt_val, df=1) + 0.5 * stats.chi2.sf(lrt_val, df=2)

def solve_lrt_cutoff(target_p):
    return brentq(lambda x: get_mixture_pval(x) - target_p, 0.001, 50.0)

def load_meme_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    content = data['MLE']['content']['0']
    pvals = [float(site_row[6]) for site_row in content]
    return np.array(pvals)

def run():
    tau_05_direct = solve_lrt_cutoff(0.05)
    tau_05_log = np.log1p(tau_05_direct)

    tau_10_direct = solve_lrt_cutoff(0.10)
    tau_10_log = np.log1p(tau_10_direct)

    device = torch.device('cpu')
    ckpt_path = os.path.join(PROJECT_ROOT, "selection_transformer_edge_best-29.pt")
    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        window_size=1,
        max_species=256
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meme_files = sorted(glob.glob(os.path.join(HYPHY_TEST_DIR, "*.MEME.json")))

    dataset_stats = []

    for json_path in meme_files:
        base_name = os.path.basename(json_path).replace('.MEME.json', '')
        nex_path = json_path.replace('.MEME.json', '')
        if not os.path.exists(nex_path):
            if os.path.exists(nex_path + '.gz'):
                nex_path = nex_path + '.gz'
            else:
                continue

        gt_pval = load_meme_json(json_path)

        if nex_path.endswith('.gz'):
            seq_dict, tree_str = parse_nexus_gz(nex_path)
        else:
            seq_dict, taxlabels, tree_str = parse_nexus_alignment_and_embedded_tree(nex_path)
            if not seq_dict or len(seq_dict) < 4:
                with open(nex_path, 'r', errors='ignore') as f:
                    content_str = f.read()
                lines = content_str.split('\n')
                in_matrix = False
                temp_seqs = {}
                for line in lines:
                    line_s = line.strip()
                    if line_s.upper().startswith('MATRIX'):
                        in_matrix = True
                        continue
                    if in_matrix:
                        if line_s == ';' or line_s.upper().startswith('END;'):
                            in_matrix = False
                            continue
                        parts = line_s.split()
                        if len(parts) >= 2:
                            tax_name = parts[0]
                            seq_val = ''.join(parts[1:])
                            temp_seqs[tax_name] = seq_val
                    if 'TREE ' in line_s.upper():
                        p = line_s.split('=', 1)
                        if len(p) > 1:
                            tree_str = p[1].strip()
                if temp_seqs:
                    seq_dict = temp_seqs

        if not seq_dict or len(seq_dict) < 4:
            continue

        species_names = list(seq_dict.keys())
        dist_matrix = {}
        if tree_str:
            try:
                clean_tree_str = tree_str.strip().rstrip(';') + ';'
                tree_obj = Phylo.read(StringIO(clean_tree_str), 'newick')
                _, dist_matrix = get_patristic_distances(tree_obj)
            except Exception:
                dist_matrix = {}

        if not dist_matrix:
            dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}

        selected_species = species_names[:256]
        num_selected = len(selected_species)
        ref_seq = seq_dict[selected_species[0]]
        total_codons = len(ref_seq) // 3

        if total_codons == 0:
            continue

        msa_tokens = torch.ones(total_codons, num_selected, 1, dtype=torch.long) * 65
        aa_tokens = torch.ones(total_codons, num_selected, 1, dtype=torch.long) * AA_TO_IDX['?']
        dist_tensor = torch.zeros(num_selected, num_selected, dtype=torch.float32)
        padding_mask = torch.zeros(num_selected, dtype=torch.bool)

        for i, s1 in enumerate(selected_species):
            for j, s2 in enumerate(selected_species):
                dist_tensor[i, j] = dist_matrix.get(s1, {}).get(s2, 0.0)

        dist_np = dist_tensor.numpy()
        mds_coords_np = compute_mds_coordinates(dist_np, n_components=4)
        mds_coords_tensor = torch.from_numpy(mds_coords_np).float()

        variable_sites_flags = []
        for site_idx in range(1, total_codons + 1):
            site_codons = []
            site_aas = []
            for s_idx, spec in enumerate(selected_species):
                seq = seq_dict.get(spec, '')
                seq_len_codons = len(seq) // 3
                if 1 <= site_idx <= seq_len_codons:
                    n_idx = (site_idx - 1) * 3
                    codon = seq[n_idx:n_idx+3]
                    msa_tokens[site_idx - 1, s_idx, 0] = get_codon_token(codon)
                    aa_tokens[site_idx - 1, s_idx, 0] = get_aa_token(codon)
                    codon_u = codon.upper()
                    if '-' not in codon_u and 'N' not in codon_u and len(codon_u) == 3:
                        site_codons.append(codon_u)
                        aa = GENETIC_CODE.get(codon_u, '?')
                        if aa != '?': site_aas.append(aa)
            variable_sites_flags.append(is_site_variable(site_codons, site_aas))

        msa_tokens = msa_tokens.to(device)
        aa_tokens = aa_tokens.to(device)
        dist_tensor = dist_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
        mds_coords_tensor = mds_coords_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
        padding_mask_tensor = padding_mask.unsqueeze(0).expand(total_codons, -1).to(device)

        pred_lrts_log = []
        with torch.no_grad():
            batch_step = 64
            for b_start in range(0, total_codons, batch_step):
                b_end = min(b_start + batch_step, total_codons)
                out_lrt, _, _, _, _ = model(
                    msa_tokens[b_start:b_end],
                    aa_tokens[b_start:b_end],
                    dist_tensor[b_start:b_end],
                    mds_coords_tensor[b_start:b_end],
                    padding_mask_tensor[b_start:b_end]
                )
                lrt_arr = out_lrt.cpu().numpy()
                for idx_b in range(len(lrt_arr)):
                    s_idx = b_start + idx_b
                    val = float(lrt_arr[idx_b]) if variable_sites_flags[s_idx] else 0.0
                    pred_lrts_log.append(val)

        n_eval = min(len(gt_pval), total_codons)
        dataset_stats.append((base_name, gt_pval[:n_eval], np.array(pred_lrts_log[:n_eval])))

    # Output for Fixed Mixture Cutoff p <= 0.05 (Direct LRT >= 5.1384)
    print(f"\n========================================================================================================================")
    print(f"FIXED MIXTURE LRT CUTOFF AT NOMINAL p <= 0.05 (Direct LRT >= {tau_05_direct:.4f} | Model Scale Log(LRT+1) >= {tau_05_log:.4f})")
    print(f"========================================================================================================================")
    header05 = f'{"Dataset":38s} | {"Sites":6s} | {"MEME Pos (p<=0.05)":18s} | {"Pred Pos (LRT>=5.14)":20s} | {"TP":4s} | {"FP":4s} | {"FN":4s} | {"TN":5s} | {"FPR":8s} | {"TPR (Sens)":10s} | {"Precision":9s}'
    print(header05)
    print('-' * len(header05))

    tot_sites, tot_meme_pos_05, tot_pred_pos_05 = 0, 0, 0
    tot_tp_05, tot_fp_05, tot_fn_05, tot_tn_05 = 0, 0, 0, 0

    for base_name, gt_pval_sub, pr_log_sub in dataset_stats:
        n_eval = len(gt_pval_sub)
        gt_pos = (gt_pval_sub <= 0.05)
        gt_neg = (~gt_pos)
        pr_pos = (pr_log_sub >= tau_05_log)

        tp = int(np.sum(gt_pos & pr_pos))
        fp = int(np.sum(gt_neg & pr_pos))
        fn = int(np.sum(gt_pos & (~pr_pos)))
        tn = int(np.sum(gt_neg & (~pr_pos)))

        n_pos = int(np.sum(gt_pos))
        n_neg = int(np.sum(gt_neg))
        n_pr = int(np.sum(pr_pos))

        tot_sites += n_eval
        tot_meme_pos_05 += n_pos
        tot_pred_pos_05 += n_pr
        tot_tp_05 += tp
        tot_fp_05 += fp
        tot_fn_05 += fn
        tot_tn_05 += tn

        fpr = (fp / n_neg) if n_neg > 0 else np.nan
        tpr = (tp / n_pos) if n_pos > 0 else np.nan
        prec = (tp / n_pr) if n_pr > 0 else np.nan

        fpr_s = f"{fpr*100:.2f}%" if not np.isnan(fpr) else "N/A"
        tpr_s = f"{tpr*100:.2f}%" if not np.isnan(tpr) else "N/A"
        prec_s = f"{prec*100:.2f}%" if not np.isnan(prec) else "N/A"

        print(f'{base_name:38s} | {n_eval:6d} | {n_pos:18d} | {n_pr:20d} | {tp:4d} | {fp:4d} | {fn:4d} | {tn:5d} | {fpr_s:8s} | {tpr_s:10s} | {prec_s:9s}')

    print('-' * len(header05))
    agg_fpr_05 = tot_fp_05 / (tot_sites - tot_meme_pos_05)
    agg_tpr_05 = tot_tp_05 / tot_meme_pos_05
    agg_prec_05 = tot_tp_05 / tot_pred_pos_05 if tot_pred_pos_05 > 0 else 0.0
    print(f'{"OVERALL AGGREGATE (p <= 0.05)":38s} | {tot_sites:6d} | {tot_meme_pos_05:18d} | {tot_pred_pos_05:20d} | {tot_tp_05:4d} | {tot_fp_05:4d} | {tot_fn_05:4d} | {tot_tn_05:5d} | {agg_fpr_05*100:7.2f}% | {agg_tpr_05*100:9.2f}% | {agg_prec_05*100:8.2f}%')

    # Output for Fixed Mixture Cutoff p <= 0.10 (Direct LRT >= 3.8078)
    print(f"\n========================================================================================================================")
    print(f"FIXED MIXTURE LRT CUTOFF AT NOMINAL p <= 0.10 (Direct LRT >= {tau_10_direct:.4f} | Model Scale Log(LRT+1) >= {tau_10_log:.4f})")
    print(f"========================================================================================================================")
    header10 = f'{"Dataset":38s} | {"Sites":6s} | {"MEME Pos (p<=0.10)":18s} | {"Pred Pos (LRT>=3.81)":20s} | {"TP":4s} | {"FP":4s} | {"FN":4s} | {"TN":5s} | {"FPR":8s} | {"TPR (Sens)":10s} | {"Precision":9s}'
    print(header10)
    print('-' * len(header10))

    tot_meme_pos_10, tot_pred_pos_10 = 0, 0
    tot_tp_10, tot_fp_10, tot_fn_10, tot_tn_10 = 0, 0, 0, 0

    for base_name, gt_pval_sub, pr_log_sub in dataset_stats:
        n_eval = len(gt_pval_sub)
        gt_pos = (gt_pval_sub <= 0.10)
        gt_neg = (~gt_pos)
        pr_pos = (pr_log_sub >= tau_10_log)

        tp = int(np.sum(gt_pos & pr_pos))
        fp = int(np.sum(gt_neg & pr_pos))
        fn = int(np.sum(gt_pos & (~pr_pos)))
        tn = int(np.sum(gt_neg & (~pr_pos)))

        n_pos = int(np.sum(gt_pos))
        n_neg = int(np.sum(gt_neg))
        n_pr = int(np.sum(pr_pos))

        tot_meme_pos_10 += n_pos
        tot_pred_pos_10 += n_pr
        tot_tp_10 += tp
        tot_fp_10 += fp
        tot_fn_10 += fn
        tot_tn_10 += tn

        fpr = (fp / n_neg) if n_neg > 0 else np.nan
        tpr = (tp / n_pos) if n_pos > 0 else np.nan
        prec = (tp / n_pr) if n_pr > 0 else np.nan

        fpr_s = f"{fpr*100:.2f}%" if not np.isnan(fpr) else "N/A"
        tpr_s = f"{tpr*100:.2f}%" if not np.isnan(tpr) else "N/A"
        prec_s = f"{prec*100:.2f}%" if not np.isnan(prec) else "N/A"

        print(f'{base_name:38s} | {n_eval:6d} | {n_pos:18d} | {n_pr:20d} | {tp:4d} | {fp:4d} | {fn:4d} | {tn:5d} | {fpr_s:8s} | {tpr_s:10s} | {prec_s:9s}')

    print('-' * len(header10))
    agg_fpr_10 = tot_fp_10 / (tot_sites - tot_meme_pos_10)
    agg_tpr_10 = tot_tp_10 / tot_meme_pos_10
    agg_prec_10 = tot_tp_10 / tot_pred_pos_10 if tot_pred_pos_10 > 0 else 0.0
    print(f'{"OVERALL AGGREGATE (p <= 0.10)":38s} | {tot_sites:6d} | {tot_meme_pos_10:18d} | {tot_pred_pos_10:20d} | {tot_tp_10:4d} | {tot_fp_10:4d} | {tot_fn_10:4d} | {tot_tn_10:5d} | {agg_fpr_10*100:7.2f}% | {agg_tpr_10*100:9.2f}% | {agg_prec_10*100:8.2f}%')

if __name__ == "__main__":
    run()
