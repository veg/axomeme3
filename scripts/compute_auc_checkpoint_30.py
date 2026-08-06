#!/usr/bin/env python3
"""
compute_auc_checkpoint_30.py
----------------------------
Computes overall and PER-DATASET ROC AUC and PR AUC (Precision-Recall Area Under Curve / Average Precision)
for Checkpoint 29 vs Checkpoint 30 across all 21 official HyPhy benchmark empirical datasets.
"""

import sys
import os
import glob
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_checkpoint_30 import load_model, eval_model_on_datasets

HYPHY_TEST_DIR = "/Users/sergei/Development/hyphy/tests/data"

def compute_dataset_aucs(gt_pvals, pred_scores, target_p=0.10):
    y_true = (gt_pvals <= target_p).astype(int)
    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    
    if n_pos == 0 or n_neg == 0:
        return n_pos, len(y_true), float('nan'), float('nan')
        
    roc = roc_auc_score(y_true, pred_scores)
    pr = average_precision_score(y_true, pred_scores)
    return n_pos, len(y_true), roc, pr

def evaluate_all_aucs(res_29, res_30, target_p=0.10):
    rows = []
    
    all_gt_29, all_pred_29 = [], []
    all_gt_30, all_pred_30 = [], []
    
    dataset_names = sorted(res_29.keys())
    
    for name in dataset_names:
        gt_29 = res_29[name]['gt_pvals']
        pred_29 = res_29[name]['pred_lrts_log']
        
        gt_30 = res_30[name]['gt_pvals']
        pred_30 = res_30[name]['pred_lrts_log']
        
        all_gt_29.extend(gt_29)
        all_pred_29.extend(pred_29)
        all_gt_30.extend(gt_30)
        all_pred_30.extend(pred_30)
        
        n_pos, n_sites, roc_29, pr_29 = compute_dataset_aucs(gt_29, pred_29, target_p=target_p)
        _, _, roc_30, pr_30 = compute_dataset_aucs(gt_30, pred_30, target_p=target_p)
        
        rows.append({
            'dataset': name,
            'n_sites': n_sites,
            'n_pos': n_pos,
            'roc_29': roc_29,
            'roc_30': roc_30,
            'delta_roc': roc_30 - roc_29 if not np.isnan(roc_29) and not np.isnan(roc_30) else float('nan'),
            'pr_29': pr_29,
            'pr_30': pr_30,
            'delta_pr': pr_30 - pr_29 if not np.isnan(pr_29) and not np.isnan(pr_30) else float('nan')
        })
        
    # Global AUCs
    all_gt_29, all_pred_29 = np.array(all_gt_29), np.array(all_pred_29)
    all_gt_30, all_pred_30 = np.array(all_gt_30), np.array(all_pred_30)
    
    global_n_pos, global_sites, global_roc_29, global_pr_29 = compute_dataset_aucs(all_gt_29, all_pred_29, target_p=target_p)
    _, _, global_roc_30, global_pr_30 = compute_dataset_aucs(all_gt_30, all_pred_30, target_p=target_p)
    
    global_summary = {
        'total_sites': global_sites,
        'total_pos': global_n_pos,
        'roc_29': global_roc_29,
        'roc_30': global_roc_30,
        'pr_29': global_pr_29,
        'pr_30': global_pr_30
    }
    
    return rows, global_summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluation Device: {device}", flush=True)

    meme_files = sorted(glob.glob(os.path.join(HYPHY_TEST_DIR, "*.MEME.json")))
    if not meme_files:
        print(f"[!] Error: No .MEME.json files found in {HYPHY_TEST_DIR}", flush=True)
        sys.exit(1)

    print(f"[*] Found {len(meme_files)} official HyPhy benchmark datasets.", flush=True)

    # 1. Load & Evaluate Checkpoint 29
    print("\n[*] Running evaluation for Checkpoint 29...", flush=True)
    model_29 = load_model("selection_transformer_edge_best-29.pt", device)
    if model_29 is None:
        sys.exit(1)
    res_29 = eval_model_on_datasets(model_29, device, meme_files)
    del model_29
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Load & Evaluate Checkpoint 30
    print("\n[*] Running evaluation for Checkpoint 30...", flush=True)
    model_30 = load_model("selection_transformer_edge_best-30.pt", device)
    if model_30 is None:
        sys.exit(1)
    res_30 = eval_model_on_datasets(model_30, device, meme_files)
    del model_30

    for target_p in [0.05, 0.10]:
        rows, glob_sum = evaluate_all_aucs(res_29, res_30, target_p=target_p)
        print("\n" + "=" * 115, flush=True)
        print(f"🏆 DATASET-BY-DATASET BREAKDOWN OF ROC AUC & PR AUC AT p <= {target_p:.02f}", flush=True)
        print("=" * 115, flush=True)
        print(f"{'Dataset Name':<42} | {'Sites':<6} | {'Pos':<4} | {'ROC (Ckpt 29 -> 30)':<22} | {'PR AUC / AvgPrec (Ckpt 29 -> 30)':<26}", flush=True)
        print("-" * 115, flush=True)
        
        for r in rows:
            name = r['dataset'][:40]
            sites = r['n_sites']
            pos = r['n_pos']
            
            if np.isnan(r['roc_29']):
                roc_str = "N/A (No Positives)"
                pr_str = "N/A (No Positives)"
            else:
                roc_str = f"{r['roc_29']:.3f} -> {r['roc_30']:.3f} ({r['delta_roc']:+.3f})"
                pr_str = f"{r['pr_29']:.3f} -> {r['pr_30']:.3f} ({r['delta_pr']:+.3f})"
                
            print(f"{name:<42} | {sites:<6d} | {pos:<4d} | {roc_str:<22} | {pr_str:<26}", flush=True)
            
        print("-" * 115, flush=True)
        g_roc_str = f"{glob_sum['roc_29']:.4f} -> {glob_sum['roc_30']:.4f} ({glob_sum['roc_30'] - glob_sum['roc_29']:+.4f})"
        g_pr_str = f"{glob_sum['pr_29']:.4f} -> {glob_sum['pr_30']:.4f} ({glob_sum['pr_30'] - glob_sum['pr_29']:+.4f})"
        print(f"{'OVERALL AGGREGATE BENCHMARK':<42} | {glob_sum['total_sites']:<6d} | {glob_sum['total_pos']:<4d} | {g_roc_str:<22} | {g_pr_str:<26}", flush=True)
        print("=" * 115, flush=True)

if __name__ == "__main__":
    main()
