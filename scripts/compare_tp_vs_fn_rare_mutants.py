#!/usr/bin/env python3
"""
compare_tp_vs_fn_rare_mutants.py
--------------------------------
Fast comparative analysis between True Positives (TPs) vs False Negatives (FNs)
on low-mutant-frequency sites (<= 10% non-synonymous mutant frequency).
"""

import sys
import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from io import StringIO
from scipy.stats import ttest_ind

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from scripts.update_meme_db_filter import parse_nexus_gz
from scripts.predict_multitask_nexus import (
    parse_nexus_alignment_and_embedded_tree,
    GENETIC_CODE,
    get_codon_token,
    get_aa_token,
    AA_TO_IDX
)

HYPHY_TEST_DIR = "/Users/sergei/Development/hyphy/tests/data"

AA_PROPS = {
    'A': [-0.5,  88.6,  0], 'R': [-3.0, 173.4,  1], 'N': [-3.0, 114.1,  0],
    'D': [-3.0, 111.1, -1], 'C': [-1.0, 108.5,  0], 'Q': [-3.0, 143.8,  0],
    'E': [-3.0, 138.4, -1], 'G': [-0.5,  60.1,  0], 'H': [-0.5, 153.2,  0.5],
    'I': [ 3.0, 166.7,  0], 'L': [ 3.0, 166.7,  0], 'K': [-3.0, 168.6,  1],
    'M': [ 1.5, 162.9,  0], 'F': [ 2.5, 189.9,  0], 'P': [ 0.0, 112.7,  0],
    'S': [-0.5,  89.0,  0], 'T': [-0.5, 116.1,  0], 'W': [ 2.0, 227.8,  0],
    'Y': [-0.5, 193.6,  0], 'V': [ 1.5, 140.0,  0]
}

def get_aa_diff(aa1, aa2):
    if aa1 not in AA_PROPS or aa2 not in AA_PROPS: return 0.0, 0.0, 0.0
    p1, p2 = AA_PROPS[aa1], AA_PROPS[aa2]
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2])

def run():
    from scripts.train_transformer_selection import PhyloAxialTransformer, compute_mds_coordinates, get_patristic_distances, is_site_variable
    device = torch.device('cpu')
    ckpt_path = os.path.join(PROJECT_ROOT, "selection_transformer_edge_best-29.pt")
    model = PhyloAxialTransformer(num_tokens=66, embed_dim=256, num_heads=8, num_layers=6, window_size=1, max_species=256).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meme_files = sorted(glob.glob(os.path.join(HYPHY_TEST_DIR, "*.MEME.json")))

    tau_log = np.log1p(5.1384)
    all_sites = []

    for json_path in meme_files:
        base_name = os.path.basename(json_path).replace('.MEME.json', '')
        nex_path = json_path.replace('.MEME.json', '')
        if not os.path.exists(nex_path):
            if os.path.exists(nex_path + '.gz'): nex_path += '.gz'
            else: continue

        with open(json_path) as f: jdata = json.load(f)
        content = jdata['MLE']['content']['0']

        if nex_path.endswith('.gz'):
            seq_dict, tree_str = parse_nexus_gz(nex_path)
        else:
            seq_dict, taxlabels, tree_str = parse_nexus_alignment_and_embedded_tree(nex_path)
            if not seq_dict or len(seq_dict) < 4:
                with open(nex_path, 'r', errors='ignore') as f: content_str = f.read()
                lines = content_str.split('\n')
                in_matrix = False
                temp_seqs = {}
                for line in lines:
                    line_s = line.strip()
                    if line_s.upper().startswith('MATRIX'): in_matrix = True; continue
                    if in_matrix:
                        if line_s == ';' or line_s.upper().startswith('END;'): in_matrix = False; continue
                        parts = line_s.split()
                        if len(parts) >= 2: temp_seqs[parts[0]] = ''.join(parts[1:])
                if temp_seqs: seq_dict = temp_seqs

        if not seq_dict or len(seq_dict) < 4: continue

        species_names = list(seq_dict.keys())
        selected_species = species_names[:256]
        num_selected = len(selected_species)
        ref_seq = seq_dict[selected_species[0]]
        total_codons = len(ref_seq) // 3
        if total_codons == 0: continue

        msa_tokens = torch.ones(total_codons, num_selected, 1, dtype=torch.long) * 65
        aa_tokens = torch.ones(total_codons, num_selected, 1, dtype=torch.long) * AA_TO_IDX['?']
        dist_tensor = torch.zeros(num_selected, num_selected, dtype=torch.float32)
        padding_mask = torch.zeros(num_selected, dtype=torch.bool)
        mds_coords_tensor = torch.zeros(total_codons, num_selected, 4, dtype=torch.float32)

        variable_sites_flags = []
        for site_idx in range(1, total_codons + 1):
            site_codons = []
            site_aas = []
            for s_idx, spec in enumerate(selected_species):
                seq = seq_dict.get(spec, '')
                if 1 <= site_idx <= len(seq)//3:
                    n_idx = (site_idx - 1)*3
                    codon = seq[n_idx:n_idx+3]
                    msa_tokens[site_idx - 1, s_idx, 0] = get_codon_token(codon)
                    aa_tokens[site_idx - 1, s_idx, 0] = get_aa_token(codon)
                    codon_u = codon.upper()
                    if '-' not in codon_u and 'N' not in codon_u and len(codon_u) == 3:
                        site_codons.append(codon_u)
                        aa = GENETIC_CODE.get(codon_u, '?')
                        if aa != '?': site_aas.append(aa)
            variable_sites_flags.append(is_site_variable(site_codons, site_aas))

        msa_tokens = msa_tokens.to(device); aa_tokens = aa_tokens.to(device)
        dist_tensor = dist_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
        mds_coords_tensor = mds_coords_tensor.to(device)
        padding_mask_tensor = padding_mask.unsqueeze(0).expand(total_codons, -1).to(device)

        pred_lrts_log = []
        with torch.no_grad():
            for b_start in range(0, total_codons, 64):
                b_end = min(b_start + 64, total_codons)
                out_lrt, _, _, _, _ = model(msa_tokens[b_start:b_end], aa_tokens[b_start:b_end], dist_tensor[b_start:b_end], mds_coords_tensor[b_start:b_end], padding_mask_tensor[b_start:b_end])
                for idx_b, val in enumerate(out_lrt.cpu().numpy()):
                    pred_lrts_log.append(float(val) if variable_sites_flags[b_start+idx_b] else 0.0)

        n_eval = min(len(content), total_codons)
        for s_idx in range(n_eval):
            row = content[s_idx]
            alpha = float(row[0])
            beta_neg = float(row[1])
            p_neg = float(row[2])
            beta_pos = float(row[3])
            p_pos = float(row[4])
            meme_lrt = float(row[5])
            meme_pval = float(row[6])
            axo_log = pred_lrts_log[s_idx]
            axo_direct = float(np.expm1(axo_log))

            if meme_pval <= 0.05:
                site_aas = []
                for spec, seq in seq_dict.items():
                    if 1 <= (s_idx + 1) <= len(seq)//3:
                        c = seq[(s_idx)*3 : (s_idx+1)*3].upper()
                        if '-' not in c and 'N' not in c and len(c) == 3:
                            aa = GENETIC_CODE.get(c, '?')
                            if aa != '?': site_aas.append(aa)

                if not site_aas: continue
                aa_counts = Counter(site_aas)
                major_aa, major_cnt = aa_counts.most_common(1)[0]
                tot_valid = len(site_aas)
                mutant_cnt = tot_valid - major_cnt
                mutant_freq = mutant_cnt / tot_valid if tot_valid > 0 else 0.0

                max_dh, max_dv, max_dc = 0.0, 0.0, 0.0
                distinct_mutant_aas = 0
                for aa, cnt in aa_counts.items():
                    if aa != major_aa:
                        distinct_mutant_aas += 1
                        dh, dv, dc = get_aa_diff(major_aa, aa)
                        max_dh = max(max_dh, dh)
                        max_dv = max(max_dv, dv)
                        max_dc = max(max_dc, dc)

                is_tp = axo_log >= tau_log
                is_fn = not is_tp

                all_sites.append({
                    'gene': base_name,
                    'site': s_idx + 1,
                    'is_tp': is_tp,
                    'is_fn': is_fn,
                    'tot_species': tot_valid,
                    'num_mutants': mutant_cnt,
                    'mutant_freq': mutant_freq,
                    'distinct_mutant_aas': distinct_mutant_aas,
                    'meme_alpha': alpha,
                    'meme_beta_pos': beta_pos,
                    'meme_p_pos': p_pos,
                    'meme_lrt': meme_lrt,
                    'axo_direct_lrt': axo_direct,
                    'max_d_hydro': max_dh,
                    'max_d_vol': max_dv,
                    'max_d_charge': max_dc
                })

    df = pd.DataFrame(all_sites)
    print(f"[*] Parsed {len(df)} MEME-positive sites (TPs = {sum(df['is_tp'])}, FNs = {sum(df['is_fn'])}).")

    df_low = df[df['mutant_freq'] <= 0.10].copy()
    print(f"[*] Low Mutant Frequency Subset (<= 10% mutants): {len(df_low)} sites (TPs = {sum(df_low['is_tp'])}, FNs = {sum(df_low['is_fn'])}).\n")

    tps = df_low[df_low['is_tp']]
    fns = df_low[df_low['is_fn']]

    print("="*100)
    print("🔬 COMPARATIVE METRICS: TRUE POSITIVES (TPs) VS FALSE NEGATIVES (FNs) [Mutants <= 10%]")
    print("="*100)

    metrics = [
        ('Total Species in Alignment', 'tot_species', '.1f'),
        ('Mutant Sequence Count', 'num_mutants', '.1f'),
        ('Mutant Frequency (%)', 'mutant_freq', '.4f'),
        ('Distinct Mutant Amino Acids', 'distinct_mutant_aas', '.2f'),
        ('MEME p_pos (% branches under selection)', 'meme_p_pos', '.4f'),
        ('MEME alpha (synonymous rate)', 'meme_alpha', '.2f'),
        ('MEME beta_pos (positive selection rate)', 'meme_beta_pos', '.2f'),
        ('MEME LRT Statistic', 'meme_lrt', '.2f'),
        ('Max Hydrophobicity Shift', 'max_d_hydro', '.2f'),
        ('Max Volume Shift (Å^3)', 'max_d_vol', '.2f'),
        ('Max Charge Shift (|e|)', 'max_d_charge', '.2f')
    ]

    header_c = f"{'Metric / Property':42s} | {'True Positives (N=' + str(len(tps)) + ')':25s} | {'False Negatives (N=' + str(len(fns)) + ')':25s} | {'p-value (t-test)':16s}"
    print(header_c)
    print("-" * len(header_c))

    for label, col, fmt in metrics:
        tp_mean = tps[col].mean()
        tp_std = tps[col].std()
        fn_mean = fns[col].mean()
        fn_std = fns[col].std()
        
        pval = ttest_ind(tps[col].dropna(), fns[col].dropna(), equal_var=False).pvalue

        tp_str = f"{tp_mean:{fmt}} ± {tp_std:{fmt}}"
        fn_str = f"{fn_mean:{fmt}} ± {fn_std:{fmt}}"
        p_str = f"{pval:.4e}" if pval < 0.001 else f"{pval:.4f}"
        sig = " ***" if pval < 0.001 else (" **" if pval < 0.01 else (" *" if pval < 0.05 else ""))

        print(f"{label:42s} | {tp_str:25s} | {fn_str:25s} | {p_str:12s}{sig}")

    print("\n" + "="*100)
    print("💡 HEAD-TO-HEAD MATCHED COMPARISON: HIGH-PERFORMING TPs vs MISSED FNs")
    print("="*100)

    print("\nTop True Positive Comparators (Sites with <= 10% mutants successfully detected by AxoMEME):")
    print("-" * 115)
    header_ex = f"{'Type':4s} | {'Gene':28s} | {'Site':4s} | {'N_spec':6s} | {'Mutants':15s} | {'Dist AA':7s} | {'HydroShift':10s} | {'VolShift':8s} | {'ChgShift':8s} | {'Axo LRT':7s} | {'MEME LRT':8s}"
    print(header_ex)
    print("-" * 115)
    for _, r in tps.sort_values('axo_direct_lrt', ascending=False).head(5).iterrows():
        m_str = f"{r['num_mutants']} ({r['mutant_freq']*100:.1f}%)"
        print(f"TP   | {r['gene']:28s} | {r['site']:4d} | {r['tot_species']:6d} | {m_str:15s} | {r['distinct_mutant_aas']:7d} | {r['max_d_hydro']:10.1f} | {r['max_d_vol']:8.1f} | {r['max_d_charge']:8.1f} | {r['axo_direct_lrt']:7.2f} | {r['meme_lrt']:8.2f}")

    print("\nTop False Negative Comparators (Sites with <= 10% mutants missed by AxoMEME):")
    print("-" * 115)
    print(header_ex)
    print("-" * 115)
    for _, r in fns.sort_values('meme_lrt', ascending=False).head(5).iterrows():
        m_str = f"{r['num_mutants']} ({r['mutant_freq']*100:.1f}%)"
        print(f"FN   | {r['gene']:28s} | {r['site']:4d} | {r['tot_species']:6d} | {m_str:15s} | {r['distinct_mutant_aas']:7d} | {r['max_d_hydro']:10.1f} | {r['max_d_vol']:8.1f} | {r['max_d_charge']:8.1f} | {r['axo_direct_lrt']:7.2f} | {r['meme_lrt']:8.2f}")

if __name__ == "__main__":
    run()
