#!/usr/bin/env python3
"""
eval_null_simulations.py
------------------------
Evaluates Checkpoint 30 vs Checkpoint 29 on 10 randomly selected NULL (MG94 neutral) simulation alignments
from silverback.temple.edu.
Verifies False Positive Rate (FPR) control on pure null data.
"""

import sys
import os
import glob
import random
import torch
import numpy as np
import scipy.stats as stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_checkpoint_30 import load_model, solve_lrt_cutoff
from scripts.train_transformer_selection import is_site_variable
from scripts.predict_multitask_nexus import (
    parse_nexus_alignment_and_embedded_tree,
    get_codon_token,
    get_aa_token,
    AA_TO_IDX,
    GENETIC_CODE
)

SIM_DIR = "/Users/sergei/Projects/TOGA_MEME/scratch/silverback_null_sims"

def eval_model_on_null_files(model, device, null_files):
    all_pred_lrts_log = []
    file_summaries = []

    with torch.no_grad():
        for idx, nex_path in enumerate(null_files, 1):
            base_name = os.path.basename(nex_path)
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
                            temp_seqs[parts[0]] = ''.join(parts[1:])
                if temp_seqs:
                    seq_dict = temp_seqs

            if not seq_dict or len(seq_dict) < 4:
                continue

            species_names = list(seq_dict.keys())
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
            mds_coords_tensor = torch.zeros(total_codons, num_selected, 4, dtype=torch.float32)

            variable_sites_flags = []
            for site_idx in range(1, total_codons + 1):
                site_codons = []
                site_aas = []
                for s_idx, spec in enumerate(selected_species):
                    seq = seq_dict.get(spec, '')
                    if 1 <= site_idx <= len(seq) // 3:
                        n_idx = (site_idx - 1) * 3
                        codon = seq[n_idx:n_idx + 3]
                        msa_tokens[site_idx - 1, s_idx, 0] = get_codon_token(codon)
                        aa_tokens[site_idx - 1, s_idx, 0] = get_aa_token(codon)
                        codon_u = codon.upper()
                        if '-' not in codon_u and 'N' not in codon_u and len(codon_u) == 3:
                            site_codons.append(codon_u)
                            aa = GENETIC_CODE.get(codon_u, '?')
                            if aa != '?':
                                site_aas.append(aa)
                variable_sites_flags.append(is_site_variable(site_codons, site_aas))

            msa_tokens = msa_tokens.to(device)
            aa_tokens = aa_tokens.to(device)
            dist_tensor = dist_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
            mds_coords_tensor = mds_coords_tensor.to(device)
            padding_mask_tensor = padding_mask.unsqueeze(0).expand(total_codons, -1).to(device)

            pred_lrts_log = []
            for b_start in range(0, total_codons, 64):
                b_end = min(b_start + 64, total_codons)
                out_lrt, _, _, _, _ = model(
                    msa_tokens[b_start:b_end],
                    aa_tokens[b_start:b_end],
                    dist_tensor[b_start:b_end],
                    mds_coords_tensor[b_start:b_end],
                    padding_mask_tensor[b_start:b_end]
                )
                for idx_b, val in enumerate(out_lrt.cpu().numpy()):
                    pred_lrts_log.append(float(val) if variable_sites_flags[b_start + idx_b] else 0.0)

            all_pred_lrts_log.extend(pred_lrts_log)
            file_summaries.append((base_name, total_codons, np.array(pred_lrts_log)))
            print(f"  [{idx:>2d}/{len(null_files)}] Evaluated {base_name}: {total_codons} null sites.", flush=True)

    return np.array(all_pred_lrts_log), file_summaries

def main():
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluation Device: {device}", flush=True)

    all_sim_files = sorted(glob.glob(os.path.join(SIM_DIR, "*.nex")))

    if len(all_sim_files) == 0:
        print(f"[!] Error: Found 0 NEXUS simulation files in {SIM_DIR}", flush=True)
        sys.exit(1)

    print(f"[*] Found {len(all_sim_files)} NULL simulation alignments from silverback in {SIM_DIR}:", flush=True)
    for f in all_sim_files:
        print(f"  - {os.path.basename(f)}", flush=True)

    # Cutoffs
    cutoff_p05_log = np.log1p(solve_lrt_cutoff(0.05)) # ~5.14
    cutoff_p10_log = np.log1p(solve_lrt_cutoff(0.10)) # ~3.81

    # 1. Evaluate Checkpoint 29
    print("\n[*] Evaluating Checkpoint 29 on 10 silverback NULL alignments...", flush=True)
    model_29 = load_model("selection_transformer_edge_best-29.pt", device)
    pred_29_log, files_29 = eval_model_on_null_files(model_29, device, all_sim_files)
    del model_29
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Checkpoint 30
    print("\n[*] Evaluating Checkpoint 30 on 10 silverback NULL alignments...", flush=True)
    model_30 = load_model("selection_transformer_edge_best-30.pt", device)
    pred_30_log, files_30 = eval_model_on_null_files(model_30, device, all_sim_files)
    del model_30

    total_sites = len(pred_30_log)

    fp_p05_29 = int(np.sum(pred_29_log >= cutoff_p05_log))
    fp_p10_29 = int(np.sum(pred_29_log >= cutoff_p10_log))
    fpr_p05_29 = (fp_p05_29 / total_sites) * 100
    fpr_p10_29 = (fp_p10_29 / total_sites) * 100

    fp_p05_30 = int(np.sum(pred_30_log >= cutoff_p05_log))
    fp_p10_30 = int(np.sum(pred_30_log >= cutoff_p10_log))
    fpr_p05_30 = (fp_p05_30 / total_sites) * 100
    fpr_p10_30 = (fp_p10_30 / total_sites) * 100

    print("\n" + "=" * 90, flush=True)
    print(f"🏆 FALSE POSITIVE RATE (FPR) ON 10 SILVERBACK NULL ALIGNMENTS ({total_sites:,d} TOTAL NULL SITES)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Cutoff Level':<25} | {'Checkpoint 29':<25} | {'Checkpoint 30':<25} | {'FPR Reduction':<15}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Nominal p <= 0.05 (LRT>=5.14)':<25} | {fp_p05_29:<5d} FPs (FPR = {fpr_p05_29:.2f}%) | {fp_p05_30:<5d} FPs (FPR = {fpr_p05_30:.2f}%) | {(fpr_p05_30 - fpr_p05_29):+.2f}%", flush=True)
    print(f"{'Nominal p <= 0.10 (LRT>=3.81)':<25} | {fp_p10_29:<5d} FPs (FPR = {fpr_p10_29:.2f}%) | {fp_p10_30:<5d} FPs (FPR = {fpr_p10_30:.2f}%) | {(fpr_p10_30 - fpr_p10_29):+.2f}%", flush=True)
    print("=" * 90, flush=True)

    print("\n--- PER-ALIGNMENT NULL BREAKDOWN ---", flush=True)
    print(f"{'Alignment File':<45} | {'Null Sites':<10} | {'Ckpt 29 FPs (p<=0.05)':<20} | {'Ckpt 30 FPs (p<=0.05)':<20}", flush=True)
    print("-" * 100, flush=True)
    for (name_29, n_s, p29), (name_30, _, p30) in zip(files_29, files_30):
        fp29 = int(np.sum(p29 >= cutoff_p05_log))
        fp30 = int(np.sum(p30 >= cutoff_p05_log))
        print(f"{name_29:<45} | {n_s:<10d} | {fp29:<5d} (FPR={fp29/n_s*100:.1f}%) | {fp30:<5d} (FPR={fp30/n_s*100:.1f}%)", flush=True)
    print("=" * 100, flush=True)

if __name__ == "__main__":
    main()
