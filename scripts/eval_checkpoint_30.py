#!/usr/bin/env python3
"""
eval_checkpoint_30.py
---------------------
Evaluates Checkpoint 30 (trained with Fused Physical Coupling + Uncertainty Loss)
on official HyPhy benchmark empirical datasets and compares it side-by-side with Checkpoint 29.
"""

import sys
import os
import glob
import json
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train_transformer_selection import (
    PhyloAxialTransformer,
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

def load_model(ckpt_filename, device):
    ckpt_path = os.path.join(PROJECT_ROOT, ckpt_filename)
    if not os.path.exists(ckpt_path):
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}", flush=True)
        return None
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    embed_dim = state_dict['codon_embedding.weight'].shape[1] * 2 if 'codon_embedding.weight' in state_dict else 256
    window_size = state_dict['pos_embedding'].shape[1] if 'pos_embedding' in state_dict else 1
    col_layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('col_layers.')]
    num_layers = max(col_layer_indices) + 1 if col_layer_indices else 6
    num_streams = 4
    if 'stream_fusion.0.weight' in state_dict:
        num_streams = state_dict['stream_fusion.0.weight'].shape[1] // embed_dim

    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=num_layers,
        window_size=window_size,
        max_species=256,
        num_streams=num_streams
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[*] Loaded model '{ckpt_filename}' (embed_dim={embed_dim}, num_layers={num_layers}, window_size={window_size}, num_streams={num_streams})", flush=True)
    return model

def eval_model_on_datasets(model, device, meme_files):
    results = {}
    for idx, json_path in enumerate(meme_files, 1):
        base_name = os.path.basename(json_path).replace('.MEME.json', '')
        nex_path = json_path.replace('.MEME.json', '')
        if not os.path.exists(nex_path):
            if os.path.exists(nex_path + '.gz'):
                nex_path += '.gz'
            else:
                continue

        with open(json_path) as f:
            jdata = json.load(f)
        content = jdata['MLE']['content']['0']
        gt_pvals = np.array([float(r[6]) for r in content])

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
        with torch.no_grad():
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

        n_eval = min(len(gt_pvals), total_codons)
        results[base_name] = {
            'gt_pvals': gt_pvals[:n_eval],
            'pred_lrts_log': np.array(pred_lrts_log[:n_eval])
        }
        print(f"  [{idx:>2d}/{len(meme_files)}] Parsed {base_name}: {n_eval} sites evaluated.", flush=True)

    return results

def compute_metrics(res_dict, target_p):
    tau_direct = solve_lrt_cutoff(target_p)
    tau_log = np.log1p(tau_direct)

    tot_sites, tot_pos, tot_pr = 0, 0, 0
    tot_tp, tot_fp, tot_fn, tot_tn = 0, 0, 0, 0

    dataset_rows = []

    for name, data in res_dict.items():
        gt_pvals = data['gt_pvals']
        pr_log = data['pred_lrts_log']

        n_sites = len(gt_pvals)
        gt_pos = (gt_pvals <= target_p)
        gt_neg = (~gt_pos)
        pr_pos = (pr_log >= tau_log)

        tp = int(np.sum(gt_pos & pr_pos))
        fp = int(np.sum(gt_neg & pr_pos))
        fn = int(np.sum(gt_pos & (~pr_pos)))
        tn = int(np.sum(gt_neg & (~pr_pos)))

        n_pos = int(np.sum(gt_pos))
        n_neg = int(np.sum(gt_neg))
        n_pr = int(np.sum(pr_pos))

        tot_sites += n_sites
        tot_pos += n_pos
        tot_pr += n_pr
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_tn += tn

        fpr = (fp / n_neg) if n_neg > 0 else 0.0
        tpr = (tp / n_pos) if n_pos > 0 else 0.0

        dataset_rows.append({
            'gene': name,
            'sites': n_sites,
            'pos': n_pos,
            'pred_pos': n_pr,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'fpr': fpr, 'tpr': tpr
        })

    agg_fpr = tot_fp / tot_tn if tot_tn > 0 else 0.0
    agg_tpr = tot_tp / tot_pos if tot_pos > 0 else 0.0
    agg_prec = tot_tp / tot_pr if tot_pr > 0 else 0.0

    return {
        'tot_sites': tot_sites,
        'tot_pos': tot_pos,
        'tot_pr': tot_pr,
        'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn, 'tn': tot_tn,
        'fpr': agg_fpr,
        'tpr': agg_tpr,
        'precision': agg_prec,
        'dataset_rows': dataset_rows
    }

def run():
    device = torch.device('cpu')
    print("[*] Loading Checkpoint 29...", flush=True)
    m29 = load_model("selection_transformer_edge_best-29.pt", device)
    print("[*] Loading Checkpoint 30...", flush=True)
    m30 = load_model("selection_transformer_edge_best-30.pt", device)

    if m30 is None:
        print("[!] Cannot evaluate Checkpoint 30: model file not found.", flush=True)
        return

    meme_files = sorted(glob.glob(os.path.join(HYPHY_TEST_DIR, "*.MEME.json")))
    print(f"[*] Found {len(meme_files)} official HyPhy benchmark datasets.\n", flush=True)

    print("[*] Running evaluation for Checkpoint 29...", flush=True)
    res29 = eval_model_on_datasets(m29, device, meme_files) if m29 else {}

    print("\n[*] Running evaluation for Checkpoint 30...", flush=True)
    res30 = eval_model_on_datasets(m30, device, meme_files)

    for p_thresh in [0.05, 0.10]:
        print("\n" + "="*110, flush=True)
        print(f"🏆 BENCHMARK EVALUATION AT NOMINAL p <= {p_thresh:.2f} (Theoretical Fixed LRT Cutoff = {solve_lrt_cutoff(p_thresh):.4f})", flush=True)
        print("="*110, flush=True)

        m30_stats = compute_metrics(res30, p_thresh)

        if res29:
            m29_stats = compute_metrics(res29, p_thresh)

            header = f"{'Metric / Benchmark Property':35s} | {'Checkpoint 29':20s} | {'Checkpoint 30 (3 Epochs)':25s} | {'Delta / Improvement':20s}"
            print(header, flush=True)
            print("-" * len(header), flush=True)

            metrics_to_show = [
                ('Total Evaluated Sites', 'tot_sites', '{:,}'),
                ('Ground Truth MEME Positives', 'tot_pos', '{:,}'),
                ('Predicted Positives (LRT >= Cutoff)', 'tot_pr', '{:,}'),
                ('True Positives (TP)', 'tp', '{:,}'),
                ('False Positives (FP)', 'fp', '{:,}'),
                ('False Negatives (FN)', 'fn', '{:,}'),
                ('True Negatives (TN)', 'tn', '{:,}'),
                ('False Positive Rate (FPR)', 'fpr', '{:.2%}'),
                ('True Positive Rate (TPR / Sensitivity)', 'tpr', '{:.2%}'),
                ('Precision (TP / Pred Pos)', 'precision', '{:.2%}')
            ]

            for label, key, fmt in metrics_to_show:
                v29 = m29_stats[key]
                v30 = m30_stats[key]
                s29 = fmt.format(v29)
                s30 = fmt.format(v30)

                if key in ['fpr', 'fp']:
                    diff = v30 - v29
                    diff_s = f"{diff:+.2%}" if key == 'fpr' else f"{int(diff):+d}"
                    status = " ✅ (Lower FP)" if diff < 0 else (" ❌ (Higher FP)" if diff > 0 else " ➖")
                elif key in ['tpr', 'tp', 'precision']:
                    diff = v30 - v29
                    diff_s = f"{diff:+.2%}" if key in ['tpr', 'precision'] else f"{int(diff):+d}"
                    status = " ✅ (Higher TPR/Prec)" if diff > 0 else (" ❌" if diff < 0 else " ➖")
                else:
                    diff_s = "—"
                    status = ""

                print(f"{label:35s} | {s29:20s} | {s30:25s} | {diff_s + status:20s}", flush=True)

if __name__ == "__main__":
    run()
