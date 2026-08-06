#!/usr/bin/env python3
"""
eval_hyphy_benchmark_datasets.py
--------------------------------
Evaluates Checkpoint 29 (selection_transformer_edge_best-29.pt) on classic HyPhy
empirical benchmark alignments from /Users/sergei/Development/hyphy/ test suite:
  1. CD2 (CD2 receptor alignment)
  2. adh (Drosophila Alcohol Dehydrogenase)
  3. integrase_BDA (HIV-1 Integrase subtype B/D/A)
  4. fluHA_codon (Influenza Hemagglutinin codon alignment)
  5. H5N1_HA_5 (H5N1 Avian Influenza HA)

Calculates:
  - Spearman rank correlation (r_s) between MEME LRT and AxoMEME LRT
  - Spearman rank correlation (r_s) between MEME p-value and AxoMEME p-value
  - Classification ROC-AUC for identifying positive selection sites (p <= 0.10)
"""

import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
from Bio import Phylo
from io import StringIO

# Ensure project root & scripts are in path
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

HYPHY_DATASETS = [
    {
        'name': 'CD2',
        'nexus': '/Users/sergei/Development/hyphy/tests/hbltests/libv3/data/CD2.nex',
        'json': '/Users/sergei/Development/hyphy/tests/hbltests/libv3/data/CD2.nex.MEME.json'
    },
    {
        'name': 'adh',
        'nexus': '/Users/sergei/Development/hyphy/tests/hbltests/libv3/data/adh.nex',
        'json': '/tmp/adh.nex.MEME.json'
    },
    {
        'name': 'integrase_BDA',
        'nexus': '/Users/sergei/Development/hyphy/tests/hbltests/data/integrase_BDA.nex',
        'json': '/tmp/integrase_BDA.nex.MEME.json'
    },
    {
        'name': 'fluHA_codon',
        'nexus': '/Users/sergei/Development/hyphy/tests/hbltests/data/fluHA_codon.nex',
        'json': '/tmp/fluHA_codon.nex.MEME.json'
    },
    {
        'name': 'H5N1_HA_5',
        'nexus': '/Users/sergei/Development/hyphy/res/TemplateBatchFiles/Miscellaneous/phylohandbook/datasets/H5N1_HA_5.nex',
        'json': '/tmp/H5N1_HA_5.nex.MEME.json'
    }
]

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
    print(f"[*] Evaluating Checkpoint-29 on HyPhy Test Empirical Datasets using device: {device}\n")

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

    header = f'{"Dataset":14s} | {"Sites":6s} | {"MEME Sig (0.10)":16s} | {"Axo-29 Sig (0.10)":18s} | {"LRT Corr (r_s)":14s} | {"P-val Corr":12s} | {"ROC-AUC":8s}'
    print('=' * len(header))
    print(header)
    print('=' * len(header))

    all_gt_lrts = []
    all_pr_lrts = []
    all_gt_pvals = []
    all_pr_pvals = []

    for ds in HYPHY_DATASETS:
        name = ds['name']
        nex_path = ds['nexus']
        json_path = ds['json']

        if not os.path.exists(nex_path) or not os.path.exists(json_path):
            print(f"[-] Skipping {name}: missing files.")
            continue

        gt_lrt, gt_pval, gt_alpha, gt_beta_pos = load_meme_json(json_path)

        seq_dict, taxlabels, tree_str = parse_nexus_alignment_and_embedded_tree(nex_path)
        if not seq_dict or len(seq_dict) < 4:
            # Fallback to parse_nexus_gz
            seq_dict, tree_str = parse_nexus_gz(nex_path)

        if not seq_dict or len(seq_dict) < 4:
            print(f"[-] Error parsing sequences for {name}.")
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

        n_eval = min(len(gt_lrt), total_codons)
        gt_lrt_sub = gt_lrt[:n_eval]
        gt_pval_sub = gt_pval[:n_eval]
        pr_lrt = np.array(pred_lrts[:n_eval])
        pr_pval = np.array(pred_pvals[:n_eval])

        all_gt_lrts.extend(gt_lrt_sub)
        all_pr_lrts.extend(pr_lrt)
        all_gt_pvals.extend(gt_pval_sub)
        all_pr_pvals.extend(pr_pval)

        lrt_corr, _ = stats.spearmanr(gt_lrt_sub, pr_lrt)
        pval_corr, _ = stats.spearmanr(gt_pval_sub, pr_pval)

        meme_sig_count = int(np.sum(gt_pval_sub <= 0.10))
        axo_sig_count = int(np.sum(pr_pval <= 0.10))

        binary_gt = (gt_pval_sub <= 0.10).astype(int)
        if len(np.unique(binary_gt)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(binary_gt, pr_lrt)
        else:
            auc = float('nan')

        print(f'{name:14s} | {n_eval:6d} | {meme_sig_count:16d} | {axo_sig_count:18d} | {lrt_corr:14.4f} | {pval_corr:12.4f} | {auc:8.3f}')

    print('=' * len(header))

    if all_gt_lrts:
        all_gt_lrts = np.array(all_gt_lrts)
        all_pr_lrts = np.array(all_pr_lrts)
        all_gt_pvals = np.array(all_gt_pvals)
        all_pr_pvals = np.array(all_pr_pvals)

        agg_lrt_corr, _ = stats.spearmanr(all_gt_lrts, all_pr_lrts)
        agg_pval_corr, _ = stats.spearmanr(all_gt_pvals, all_pr_pvals)
        agg_binary_gt = (all_gt_pvals <= 0.10).astype(int)
        from sklearn.metrics import roc_auc_score, average_precision_score
        agg_auc = roc_auc_score(agg_binary_gt, all_pr_lrts)
        agg_pr_auc = average_precision_score(agg_binary_gt, all_pr_lrts)

        print(f'\n🌐 OVERALL AGGREGATE PERFORMANCE across HyPhy Empirical Datasets (N = {len(all_gt_lrts)} sites):')
        print(f'   - Overall Spearman Correlation (MEME LRT vs AxoMEME LRT): r_s = {agg_lrt_corr:.4f}')
        print(f'   - Overall Spearman Correlation (MEME p-val vs AxoMEME p-val): r_s = {agg_pval_corr:.4f}')
        print(f'   - Overall Detection ROC-AUC (Identifying p <= 0.10 sites): {agg_auc:.4f}')
        print(f'   - Overall Detection PR-AUC (Precision-Recall): {agg_pr_auc:.4f}')

if __name__ == "__main__":
    evaluate()
