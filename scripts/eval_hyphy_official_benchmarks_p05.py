#!/usr/bin/env python3
"""
eval_hyphy_official_benchmarks_p05.py
--------------------------------------
Evaluates Checkpoint 29 (selection_transformer_edge_best-29.pt) across all
official HyPhy empirical benchmark datasets located in:
  /Users/sergei/Development/hyphy/tests/data/

Uses MEME Ground Truth threshold p <= 0.05.
Reports TP, FP, FN, TN, Sensitivity (TPR), Specificity (TNR), and ROC-AUC.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scripts.train_transformer_selection import (
    PhyloAxialTransformer,
    compute_mds_coordinates,
    get_patristic_distances as calculate_patristic_distances,
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

def load_meme_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    content = data['MLE']['content']['0']
    lrts = []
    pvals = []
    alphas = []
    betas_pos = []
    for site_row in content:
        alphas.append(float(site_row[0]))
        betas_pos.append(float(site_row[3]))
        lrts.append(float(site_row[5]))
        pvals.append(float(site_row[6]))
    return np.array(lrts), np.array(pvals), np.array(alphas), np.array(betas_pos)

def evaluate():
    device = torch.device('cpu')
    print(f"[*] Evaluating Checkpoint-29 on Official HyPhy Benchmark Datasets at p <= 0.05 ({HYPHY_TEST_DIR}) using device: {device}\n")

    ckpt_path = os.path.join(PROJECT_ROOT, "selection_transformer_edge_best-29.pt")
    if not os.path.exists(ckpt_path):
        print(f"[!] Error: Checkpoint {ckpt_path} not found!")
        sys.exit(1)

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
    
    header = f'{"Dataset":38s} | {"Sites":6s} | {"MEME Pos (p<=0.05)":18s} | {"Axo Pos (p<=0.05)":18s} | {"TP":4s} | {"FP":4s} | {"FN":4s} | {"TN":5s} | {"Sens (TPR)":10s} | {"Spec (TNR)":10s} | {"ROC-AUC":8s}'
    print('=' * len(header))
    print(header)
    print('=' * len(header))

    all_gt_pvals = []
    all_pr_pvals = []
    all_pr_lrts = []

    for json_path in meme_files:
        base_name = os.path.basename(json_path).replace('.MEME.json', '')
        nex_path = json_path.replace('.MEME.json', '')
        if not os.path.exists(nex_path):
            if os.path.exists(nex_path + '.gz'):
                nex_path = nex_path + '.gz'
            else:
                continue

        gt_lrt, gt_pval, gt_alpha, gt_beta_pos = load_meme_json(json_path)

        seq_dict = {}
        tree_str = None

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
                _, dist_matrix = calculate_patristic_distances(tree_obj)
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

        pred_lrts, pred_pvals = [], []
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
                    if variable_sites_flags[s_idx]:
                        val = float(lrt_arr[idx_b])
                        pval = 0.5 * stats.chi2.sf(val, df=1) + 0.5 * stats.chi2.sf(val, df=2) if val > 0 else 1.0
                    else:
                        val = 0.0
                        pval = 1.0
                    pred_lrts.append(val)
                    pred_pvals.append(pval)

        n_eval = min(len(gt_pval), total_codons)
        gt_pval_sub = gt_pval[:n_eval]
        pr_lrt_sub = np.array(pred_lrts[:n_eval])
        pr_pval_sub = np.array(pred_pvals[:n_eval])

        all_gt_pvals.extend(gt_pval_sub)
        all_pr_pvals.extend(pr_pval_sub)
        all_pr_lrts.extend(pr_lrt_sub)

        gt_pos = (gt_pval_sub <= 0.05)
        pr_pos = (pr_pval_sub <= 0.05)

        tp = int(np.sum(gt_pos & pr_pos))
        fp = int(np.sum((~gt_pos) & pr_pos))
        fn = int(np.sum(gt_pos & (~pr_pos)))
        tn = int(np.sum((~gt_pos) & (~pr_pos)))

        meme_pos_count = int(np.sum(gt_pos))
        axo_pos_count = int(np.sum(pr_pos))

        sens = (tp / meme_pos_count) if meme_pos_count > 0 else np.nan
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else np.nan

        binary_gt = gt_pos.astype(int)
        if len(np.unique(binary_gt)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(binary_gt, pr_lrt_sub)
        else:
            auc = np.nan

        sens_str = f"{sens:.3f}" if not np.isnan(sens) else "N/A"
        spec_str = f"{spec:.3f}" if not np.isnan(spec) else "N/A"
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"

        print(f'{base_name:38s} | {n_eval:6d} | {meme_pos_count:18d} | {axo_pos_count:18d} | {tp:4d} | {fp:4d} | {fn:4d} | {tn:5d} | {sens_str:10s} | {spec_str:10s} | {auc_str:8s}', flush=True)

    print('=' * len(header), flush=True)

    if all_gt_pvals:
        all_gt_pvals = np.array(all_gt_pvals)
        all_pr_pvals = np.array(all_pr_pvals)
        all_pr_lrts = np.array(all_pr_lrts)

        gt_pos_all = (all_gt_pvals <= 0.05)
        pr_pos_all = (all_pr_pvals <= 0.05)

        agg_tp = int(np.sum(gt_pos_all & pr_pos_all))
        agg_fp = int(np.sum((~gt_pos_all) & pr_pos_all))
        agg_fn = int(np.sum(gt_pos_all & (~pr_pos_all)))
        agg_tn = int(np.sum((~gt_pos_all) & (~pr_pos_all)))

        agg_meme_pos = int(np.sum(gt_pos_all))
        agg_axo_pos = int(np.sum(pr_pos_all))

        agg_sens = agg_tp / agg_meme_pos if agg_meme_pos > 0 else 0.0
        agg_spec = agg_tn / (agg_tn + agg_fp) if (agg_tn + agg_fp) > 0 else 0.0
        agg_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0

        from sklearn.metrics import roc_auc_score, average_precision_score
        agg_auc = roc_auc_score(gt_pos_all.astype(int), all_pr_lrts)
        agg_pr_auc = average_precision_score(gt_pos_all.astype(int), all_pr_lrts)

        print(f"\n🌐 OVERALL AGGREGATE STATS AT MEME p <= 0.05 (Total Sites N = {len(all_gt_pvals)}):", flush=True)
        print(f"   - MEME Ground Truth Positives (p <= 0.05): {agg_meme_pos} / {len(all_gt_pvals)} ({agg_meme_pos/len(all_gt_pvals)*100:.2f}%)", flush=True)
        print(f"   - AxoMEME-29 Predicted Positives (p <= 0.05): {agg_axo_pos} / {len(all_gt_pvals)} ({agg_axo_pos/len(all_gt_pvals)*100:.2f}%)", flush=True)
        print(f"   - True Positives (TP):  {agg_tp}", flush=True)
        print(f"   - False Positives (FP): {agg_fp}", flush=True)
        print(f"   - False Negatives (FN): {agg_fn}", flush=True)
        print(f"   - True Negatives (TN):  {agg_tn}", flush=True)
        print(f"   - Sensitivity / Recall (TPR = TP / Positives): {agg_sens:.4f} ({agg_sens*100:.2f}%)", flush=True)
        print(f"   - Specificity (TNR = TN / Negatives):           {agg_spec:.4f} ({agg_spec*100:.2f}%)", flush=True)
        print(f"   - False Positive Rate (FPR = FP / Negatives):     {1.0 - agg_spec:.4f} ({(1.0 - agg_spec)*100:.2f}%)", flush=True)
        print(f"   - Precision (PPV = TP / (TP + FP)):              {agg_prec:.4f} ({agg_prec*100:.2f}%)", flush=True)
        print(f"   - Detection ROC-AUC (p <= 0.05 sites):          {agg_auc:.4f}", flush=True)
        print(f"   - Detection PR-AUC (Precision-Recall):           {agg_pr_auc:.4f}", flush=True)

if __name__ == "__main__":
    evaluate()
