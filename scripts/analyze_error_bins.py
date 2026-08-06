#!/usr/bin/env python3
"""
analyze_error_bins.py
---------------------
Categorizes False Positives (FPs) and False Negatives (FNs) between AxoMEME-29
and HyPhy MEME across all 20 benchmark alignments into diagnostic bins:

1. False Negatives (MEME Positives missed by AxoMEME):
   - Borderline FNs (0.03 < p_MEME <= 0.05) [Close to threshold]
   - Strong FNs     (0.001 < p_MEME <= 0.03) [Clear MEME signal]
   - Extreme FNs    (p_MEME <= 0.001)        [Complete disagreement, massive MEME signal]

2. False Positives (AxoMEME Positives missed by MEME):
   - Near-Threshold FPs (0.05 < p_MEME <= 0.15) [Subtle MEME signal]
   - Moderate FPs       (0.15 < p_MEME <= 0.50) [Weak MEME signal]
   - Egregious FPs      (p_MEME > 0.50)         [Complete disagreement, highly conserved/neutral in MEME]
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
    lrts = [float(site_row[5]) for site_row in content]
    pvals = [float(site_row[6]) for site_row in content]
    alphas = [float(site_row[0]) for site_row in content]
    betas_pos = [float(site_row[3]) for site_row in content]
    return np.array(lrts), np.array(pvals), np.array(alphas), np.array(betas_pos)

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

    records = []

    for json_path in meme_files:
        base_name = os.path.basename(json_path).replace('.MEME.json', '')
        nex_path = json_path.replace('.MEME.json', '')
        if not os.path.exists(nex_path):
            if os.path.exists(nex_path + '.gz'):
                nex_path = nex_path + '.gz'
            else:
                continue

        gt_lrt, gt_pval, gt_alpha, gt_beta_pos = load_meme_json(json_path)

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
        for s_idx in range(n_eval):
            lrt_log_val = pred_lrts_log[s_idx]
            lrt_direct_val = float(np.expm1(lrt_log_val))
            records.append({
                'gene': base_name,
                'site': s_idx + 1,
                'gt_lrt': float(gt_lrt[s_idx]),
                'gt_pval': float(gt_pval[s_idx]),
                'gt_alpha': float(gt_alpha[s_idx]),
                'gt_beta_pos': float(gt_beta_pos[s_idx]),
                'pred_lrt_log': lrt_log_val,
                'pred_lrt_direct': lrt_direct_val
            })

    df = pd.DataFrame(records)
    print(f"[*] Total parsed codon sites: {len(df):,}")

    # Use fixed mixture cutoff for p <= 0.05 (Direct LRT >= 5.1384 <=> Model Log(LRT+1) >= 1.8146)
    tau_log = np.log1p(5.1384)

    gt_pos_05 = df['gt_pval'] <= 0.05
    pr_pos_05 = df['pred_lrt_log'] >= tau_log

    df['is_tp'] = gt_pos_05 & pr_pos_05
    df['is_fp'] = (~gt_pos_05) & pr_pos_05
    df['is_fn'] = gt_pos_05 & (~pr_pos_05)
    df['is_tn'] = (~gt_pos_05) & (~pr_pos_05)

    fns = df[df['is_fn']].copy()
    fps = df[df['is_fp']].copy()

    # Binned FNs
    fn_borderline = fns[(fns['gt_pval'] > 0.03) & (fns['gt_pval'] <= 0.05)]
    fn_strong = fns[(fns['gt_pval'] > 0.001) & (fns['gt_pval'] <= 0.03)]
    fn_extreme = fns[fns['gt_pval'] <= 0.001]

    # Binned FPs
    fp_near = fps[(fps['gt_pval'] > 0.05) & (fps['gt_pval'] <= 0.15)]
    fp_mod = fps[(fps['gt_pval'] > 0.15) & (fps['gt_pval'] <= 0.50)]
    fp_egregious = fps[fps['gt_pval'] > 0.50]

    print("\n" + "="*90)
    print("📊 ERROR DISSECTION & CATEGORIZATION FOR AxoMEME-29 AT p <= 0.05")
    print("="*90)

    print(f"\n1. FALSE NEGATIVES DISSECTION (Total FNs = {len(fns)} / {sum(gt_pos_05)} MEME Positives):")
    print(f"   -----------------------------------------------------------------------------------------")
    print(f"   a) BORDERLINE / NEAR-THRESHOLD FNs (0.03 < p_MEME <= 0.05):")
    print(f"      - Count: {len(fn_borderline)} ({len(fn_borderline)/len(fns)*100:.1f}% of FNs)")
    print(f"      - Description: Moderate MEME positive signal right near significance threshold.")
    if not fn_borderline.empty:
        ex = fn_borderline.iloc[0]
        print(f"      - Sample: {ex['gene']} site {ex['site']} (MEME p={ex['gt_pval']:.4f}, MEME LRT={ex['gt_lrt']:.2f}, Axo Direct LRT={ex['pred_lrt_direct']:.2f})")

    print(f"\n   b) STRONG FNs (0.001 < p_MEME <= 0.03):")
    print(f"      - Count: {len(fn_strong)} ({len(fn_strong)/len(fns)*100:.1f}% of FNs)")
    print(f"      - Description: Clear MEME positive signal missed by model threshold.")
    if not fn_strong.empty:
        ex = fn_strong.iloc[0]
        print(f"      - Sample: {ex['gene']} site {ex['site']} (MEME p={ex['gt_pval']:.4f}, MEME LRT={ex['gt_lrt']:.2f}, Axo Direct LRT={ex['pred_lrt_direct']:.2f})")

    print(f"\n   c) EXTREME FNs / COMPLETE DISAGREEMENT (p_MEME <= 0.001, MEME LRT > 10.8):")
    print(f"      - Count: {len(fn_extreme)} ({len(fn_extreme)/len(fns)*100:.1f}% of FNs)")
    print(f"      - Description: Massive MEME positive signal (e.g. single-branch burst) missed by model.")
    if not fn_extreme.empty:
        print(f"      - Top 5 Extreme Misses:")
        for _, ex in fn_extreme.sort_values('gt_lrt', ascending=False).head(5).iterrows():
            print(f"        * {ex['gene']:35s} Site {ex['site']:4d} | MEME p={ex['gt_pval']:.6f} (LRT={ex['gt_lrt']:6.2f}) | Axo Direct LRT={ex['pred_lrt_direct']:5.2f}")


    print(f"\n2. FALSE POSITIVES DISSECTION (Total FPs = {len(fps)} / {len(fps)+sum(df['is_tn'])} MEME Negatives):")
    print(f"   -----------------------------------------------------------------------------------------")
    print(f"   a) NEAR-THRESHOLD FPs / BORDERLINE AGREEMENT (0.05 < p_MEME <= 0.15):")
    print(f"      - Count: {len(fp_near)} ({len(fp_near)/len(fps)*100:.1f}% of FPs)")
    print(f"      - Description: MEME detected sub-significant selection signal; minor threshold boundary difference.")
    if not fp_near.empty:
        ex = fp_near.iloc[0]
        print(f"      - Sample: {ex['gene']} site {ex['site']} (MEME p={ex['gt_pval']:.4f}, MEME LRT={ex['gt_lrt']:.2f}, Axo Direct LRT={ex['pred_lrt_direct']:.2f})")

    print(f"\n   b) MODERATE FPs (0.15 < p_MEME <= 0.50):")
    print(f"      - Count: {len(fp_mod)} ({len(fp_mod)/len(fps)*100:.1f}% of FPs)")
    print(f"      - Description: Weak MEME background signal elevated by AxoMEME.")
    if not fp_mod.empty:
        ex = fp_mod.iloc[0]
        print(f"      - Sample: {ex['gene']} site {ex['site']} (MEME p={ex['gt_pval']:.4f}, MEME LRT={ex['gt_lrt']:.2f}, Axo Direct LRT={ex['pred_lrt_direct']:.2f})")

    print(f"\n   c) EGREGIOUS FPs / COMPLETE DISAGREEMENT (p_MEME > 0.50 or Invariable):")
    print(f"      - Count: {len(fp_egregious)} ({len(fp_egregious)/len(fps)*100:.1f}% of FPs)")
    print(f"      - Description: Neutral or conserved sites in MEME called as positive by AxoMEME.")
    if not fp_egregious.empty:
        print(f"      - Top 5 Egregious False Positives:")
        for _, ex in fp_egregious.sort_values('pred_lrt_direct', ascending=False).head(5).iterrows():
            print(f"        * {ex['gene']:35s} Site {ex['site']:4d} | MEME p={ex['gt_pval']:.4f} (LRT={ex['gt_lrt']:5.2f}) | Axo Direct LRT={ex['pred_lrt_direct']:5.2f}")

if __name__ == "__main__":
    run()
