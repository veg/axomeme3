#!/usr/bin/env python3
"""
verify_extreme_fns.py
---------------------
Empirically verifies whether Extreme FNs (where MEME LRT > 10.8 / p <= 0.001 but AxoMEME-29 LRT < 5.14)
are driven by single (or very few) branch substitutions.

For each Extreme FN site across all benchmark datasets, extracts:
1. MSA codon/amino acid site pattern (number of unique non-synonymous variants, counts of mutant sequences).
2. MEME MLE parameter estimates from the HyPhy JSON:
   - alpha (synonymous rate)
   - beta_neg (conserved non-synonymous rate)
   - p_neg (proportion of branches under beta_neg)
   - beta_pos (positive selection rate)
   - p_pos / q_pos (proportion of branches under positive selection beta_pos)
   - MEME LRT and p-value
"""

import sys
import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter
from io import StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.update_meme_db_filter import parse_nexus_gz
from scripts.predict_multitask_nexus import (
    parse_nexus_alignment_and_embedded_tree,
    GENETIC_CODE
)

HYPHY_TEST_DIR = "/Users/sergei/Development/hyphy/tests/data"

# Load Extreme FN sites identified earlier
def get_extreme_fns():
    from scripts.analyze_error_bins import load_meme_json, load_meme_json
    meme_files = sorted(glob.glob(os.path.join(HYPHY_TEST_DIR, "*.MEME.json")))
    
    # Run prediction script or load predictions
    import torch
    from scripts.train_transformer_selection import PhyloAxialTransformer, compute_mds_coordinates, get_patristic_distances, is_site_variable
    from scripts.predict_multitask_nexus import get_codon_token, get_aa_token, AA_TO_IDX
    from Bio import Phylo

    device = torch.device('cpu')
    ckpt_path = os.path.join(PROJECT_ROOT, "selection_transformer_edge_best-29.pt")
    model = PhyloAxialTransformer(num_tokens=66, embed_dim=256, num_heads=8, num_layers=6, window_size=1, max_species=256).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tau_log = np.log1p(5.1384)
    extreme_fns = []

    for json_path in meme_files:
        base_name = os.path.basename(json_path).replace('.MEME.json', '')
        nex_path = json_path.replace('.MEME.json', '')
        if not os.path.exists(nex_path):
            if os.path.exists(nex_path + '.gz'): nex_path += '.gz'
            else: continue

        with open(json_path) as f:
            jdata = json.load(f)
        content = jdata['MLE']['content']['0']
        
        # MEME content fields: [alpha, beta_neg, p_neg, beta_pos, p_pos, LRT, p-value]
        
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
                    if line_s.upper().startswith('MATRIX'): in_matrix = True; continue
                    if in_matrix:
                        if line_s == ';' or line_s.upper().startswith('END;'): in_matrix = False; continue
                        parts = line_s.split()
                        if len(parts) >= 2: temp_seqs[parts[0]] = ''.join(parts[1:])
                if temp_seqs: seq_dict = temp_seqs

        if not seq_dict or len(seq_dict) < 4: continue

        species_names = list(seq_dict.keys())
        dist_matrix = {}
        if tree_str:
            try:
                tree_obj = Phylo.read(StringIO(tree_str.strip().rstrip(';') + ';'), 'newick')
                _, dist_matrix = get_patristic_distances(tree_obj)
            except: pass
        if not dist_matrix: dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}

        selected_species = species_names[:256]
        num_selected = len(selected_species)
        ref_seq = seq_dict[selected_species[0]]
        total_codons = len(ref_seq) // 3
        if total_codons == 0: continue

        msa_tokens = torch.ones(total_codons, num_selected, 1, dtype=torch.long) * 65
        aa_tokens = torch.ones(total_codons, num_selected, 1, dtype=torch.long) * AA_TO_IDX['?']
        dist_tensor = torch.zeros(num_selected, num_selected, dtype=torch.float32)
        padding_mask = torch.zeros(num_selected, dtype=torch.bool)
        for i, s1 in enumerate(selected_species):
            for j, s2 in enumerate(selected_species): dist_tensor[i, j] = dist_matrix.get(s1, {}).get(s2, 0.0)

        mds_coords_tensor = torch.from_numpy(compute_mds_coordinates(dist_tensor.numpy(), n_components=4)).float()

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
        mds_coords_tensor = mds_coords_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
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

            # Criteria for Extreme FN: MEME p <= 0.001 (LRT > 10.8) and Axo direct LRT < 5.14
            if meme_pval <= 0.001 and axo_log < tau_log:
                # Extract MSA codon & AA distribution across all species in dataset
                site_codons = []
                site_aas = []
                for spec, seq in seq_dict.items():
                    if 1 <= (s_idx + 1) <= len(seq)//3:
                        c = seq[(s_idx)*3 : (s_idx+1)*3].upper()
                        if '-' not in c and 'N' not in c and len(c) == 3:
                            site_codons.append(c)
                            aa = GENETIC_CODE.get(c, '?')
                            if aa != '?': site_aas.append(aa)
                
                aa_counts = Counter(site_aas)
                major_aa, major_cnt = aa_counts.most_common(1)[0] if aa_counts else ('?', 0)
                tot_valid = sum(aa_counts.values())
                minor_cnts = [cnt for aa, cnt in aa_counts.items() if aa != major_aa]
                num_variants = len(minor_cnts)
                tot_mutants = sum(minor_cnts)

                extreme_fns.append({
                    'gene': base_name,
                    'site': s_idx + 1,
                    'num_species': len(seq_dict),
                    'valid_aas': tot_valid,
                    'major_aa': major_aa,
                    'major_count': major_cnt,
                    'num_variants': num_variants,
                    'tot_mutants': tot_mutants,
                    'aa_counts_str': str(dict(aa_counts)),
                    'meme_alpha': alpha,
                    'meme_beta_pos': beta_pos,
                    'meme_p_pos': p_pos,
                    'meme_lrt': meme_lrt,
                    'meme_pval': meme_pval,
                    'axo_direct_lrt': axo_direct
                })

    return pd.DataFrame(extreme_fns)

def main():
    print("[*] Inspecting MSA site patterns and MEME parameters for Extreme FNs...")
    df = get_extreme_fns()
    print(f"[*] Found {len(df)} Extreme FN sites across all datasets.\n")

    print("="*100)
    print("🔬 EMPIRICAL VERIFICATION: ARE EXTREME FNs DRIVEN BY SINGLE / FEW SUBSTITUTIONS?")
    print("="*100)

    # Bin Extreme FNs by total mutant sequence count in the MSA
    df['mutant_bin'] = pd.cut(
        df['tot_mutants'],
        bins=[-1, 1, 2, 5, 10, 10000],
        labels=['Single Mutant (1 seq)', '2 Mutants (2 seqs)', '3-5 Mutants', '6-10 Mutants', '> 10 Mutants']
    )

    print("\n1. DISTRIBUTION OF EXTREME FNs BY MSA MUTANT SEQUENCE COUNT:")
    print("-----------------------------------------------------------------------------------------")
    counts = df['mutant_bin'].value_counts().sort_index()
    for bin_name, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"   - {bin_name:25s}: {cnt:3d} sites ({pct:5.1f}%)")

    # Analyze MEME proportion of branches under selection (p_pos)
    print(f"\n2. MEME ESTIMATED PROPORTION OF POSITIVE BRANCHES (p_pos):")
    print("-----------------------------------------------------------------------------------------")
    print(f"   - Median MEME p_pos across Extreme FNs: {df['meme_p_pos'].median():.4f}")
    print(f"   - Mean MEME p_pos across Extreme FNs:   {df['meme_p_pos'].mean():.4f}")
    print(f"   - Sites with p_pos <= 0.05 (<=5% of branches under selection): {sum(df['meme_p_pos'] <= 0.05)} / {len(df)} ({sum(df['meme_p_pos'] <= 0.05)/len(df)*100:.1f}%)")
    print(f"   - Sites with p_pos <= 0.02 (<=2% of branches under selection): {sum(df['meme_p_pos'] <= 0.02)} / {len(df)} ({sum(df['meme_p_pos'] <= 0.02)/len(df)*100:.1f}%)")

    print("\n3. TOP 10 EXTREME FNs: DETAILED SITE MSA & MEME DISSECTION:")
    print("-----------------------------------------------------------------------------------------")
    header = f"{'Gene':30s} | {'Site':4s} | {'N_spec':6s} | {'Major AA':8s} | {'Mutant AAs':15s} | {'MEME p_pos':10s} | {'MEME LRT':9s} | {'Axo LRT':7s}"
    print(header)
    print("-" * len(header))

    top10 = df.sort_values('meme_lrt', ascending=False).head(10)
    for _, r in top10.iterrows():
        mutants = [f"{aa}:{c}" for aa, c in eval(r['aa_counts_str']).items() if aa != r['major_aa']]
        mutants_str = ", ".join(mutants) if mutants else "None"
        if len(mutants_str) > 15: mutants_str = mutants_str[:12] + "..."
        print(f"{r['gene']:30s} | {r['site']:4d} | {r['num_species']:6d} | {r['major_aa']}:{r['major_count']:<4d} | {mutants_str:15s} | {r['meme_p_pos']:10.4f} | {r['meme_lrt']:9.2f} | {r['axo_direct_lrt']:7.2f}")

if __name__ == "__main__":
    main()
