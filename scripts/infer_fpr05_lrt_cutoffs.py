#!/usr/bin/env python3
"""
infer_fpr05_lrt_cutoffs.py
--------------------------
Evaluates individual HyPhy benchmark empirical datasets for Checkpoint 29.
For each dataset (and overall aggregate):
1. Finds the AxoMEME-29 LRT threshold tau that yields False Positive Rate (FPR) = 0.05 on negative sites (p_MEME > 0.05).
2. Computes the True Positive Rate (TPR / Sensitivity) achieved at that tau.
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

def load_meme_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    content = data['MLE']['content']['0']
    pvals = [float(site_row[6]) for site_row in content]
    return np.array(pvals)

def run():
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

    header = f'{"Dataset":38s} | {"Sites":6s} | {"MEME Pos":8s} | {"LRT Cutoff (FPR=0.05)":22s} | {"FP Count":8s} | {"TP Count":8s} | {"TPR (Sens)":10s} | {"ROC-AUC":8s}'
    print('=' * len(header))
    print(header)
    print('=' * len(header))

    all_gt_pvals = []
    all_pr_lrts = []

    results = []

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

        pred_lrts = []
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
                    pred_lrts.append(val)

        n_eval = min(len(gt_pval), total_codons)
        gt_pval_sub = gt_pval[:n_eval]
        pr_lrt_sub = np.array(pred_lrts[:n_eval])

        all_gt_pvals.extend(gt_pval_sub)
        all_pr_lrts.extend(pr_lrt_sub)

        gt_pos = (gt_pval_sub <= 0.05)
        gt_neg = (~gt_pos)
        n_pos = int(np.sum(gt_pos))
        n_neg = int(np.sum(gt_neg))

        if n_neg > 0:
            neg_lrts = pr_lrt_sub[gt_neg]
            tau = float(np.percentile(neg_lrts, 95.0))
        else:
            tau = 0.0

        if n_pos > 0:
            tp_count = int(np.sum(pr_lrt_sub[gt_pos] >= tau))
            tpr = tp_count / n_pos
            sens_str = f"{tpr:.3f} ({tp_count}/{n_pos})"
        else:
            tp_count = 0
            tpr = np.nan
            sens_str = "N/A (0 pos)"

        fp_count = int(np.sum(pr_lrt_sub[gt_neg] >= tau))

        binary_gt = gt_pos.astype(int)
        if len(np.unique(binary_gt)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(binary_gt, pr_lrt_sub)
            auc_str = f"{auc:.3f}"
        else:
            auc_str = "N/A"

        print(f'{base_name:38s} | {n_eval:6d} | {n_pos:8d} | {tau:22.4f} | {fp_count:8d} | {tp_count:8d} | {sens_str:10s} | {auc_str:8s}', flush=True)

    print('=' * len(header))

    if all_gt_pvals:
        all_gt_pvals = np.array(all_gt_pvals)
        all_pr_lrts = np.array(all_pr_lrts)
        gt_pos_all = (all_gt_pvals <= 0.05)
        gt_neg_all = (~gt_pos_all)

        n_pos_all = int(np.sum(gt_pos_all))
        n_neg_all = int(np.sum(gt_neg_all))

        tau_global = float(np.percentile(all_pr_lrts[gt_neg_all], 95.0))

        tp_all = int(np.sum(all_pr_lrts[gt_pos_all] >= tau_global))
        fp_all = int(np.sum(all_pr_lrts[gt_neg_all] >= tau_global))
        tpr_all = tp_all / n_pos_all

        from sklearn.metrics import roc_auc_score
        auc_global = roc_auc_score(gt_pos_all.astype(int), all_pr_lrts)

        print(f"\n🌐 OVERALL GLOBAL AGGREGATE (N = {len(all_gt_pvals)} sites across 20 genes):")
        print(f"   - Global LRT Cutoff for FPR = 0.05: tau = {tau_global:.4f}")
        print(f"   - Negatives Count (p > 0.05):       {n_neg_all}")
        print(f"   - False Positives at tau (FPR=5%):  {fp_all} (actual FPR = {fp_all/n_neg_all*100:.2f}%)")
        print(f"   - Positives Count (p <= 0.05):     {n_pos_all}")
        print(f"   - True Positives at tau:            {tp_all}")
        print(f"   - Achieved TPR (Sensitivity):       {tpr_all:.4f} ({tpr_all*100:.2f}%)")
        print(f"   - Global ROC-AUC:                   {auc_global:.4f}")

if __name__ == "__main__":
    run()
