#!/usr/bin/env python3
import os, sys, glob, json, math, torch
import numpy as np, pandas as pd
from io import StringIO
from Bio import Phylo

PROJECT_ROOT = "/Users/sergei/Projects/TOGA_MEME"
sys.path.insert(0, PROJECT_ROOT)

from scripts.train_transformer_selection import PhyloAxialTransformer, decode_soft_ordinal_lrt, BIN_MEANS
from scripts.predict_regression_nexus import (
    parse_nexus_alignment_and_embedded_tree,
    calculate_patristic_distances,
    compute_mds_coordinates,
    get_codon_token,
    get_aa_token
)

DATA_DIR = "/Users/sergei/Development/hyphy/tests/data"
meme_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.MEME.json")))

device = torch.device('cpu')
model_path = os.path.join(PROJECT_ROOT, 'axomeme_2.0_best_checkpoint.pt')
if not os.path.exists(model_path):
    model_path = os.path.join(PROJECT_ROOT, 'selection_transformer_best.pt')

print(f" -> Loading model weights from {model_path}...")
state_dict = torch.load(model_path, map_location=device, weights_only=False)

model = PhyloAxialTransformer(
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    window_size=1,
    max_species=256,
    max_k=64
).to(device)

model_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(filtered_state_dict)
model.load_state_dict(model_dict)
model.eval()

bin_means_device = BIN_MEANS.to(device)

all_site_records = []

print("=" * 115)
print(f"🚀 EVALUATING AXOMEME 2.0 FOCAL-CORAL CHECKPOINT ON {len(meme_files)} HYPHY BENCHMARK DATASETS")
print("=" * 115)
print(f"{'Dataset Name':<25} | {'Sites':<5} | {'Trees (T)':<8} | {'MEME Sig (p<=0.05)':<18} | {'Axo2 Detected':<13} | {'Recall %':<9} | {'Spearman rho':<12}")
print("-" * 115)

total_gt_sig = 0
total_detected_sig = 0

for meme_path in meme_files:
    dataset_name = os.path.basename(meme_path).replace('.MEME.json', '')
    nex_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(nex_path):
        continue
        
    with open(meme_path) as f:
        meme_gt = json.load(f)
        
    mle_rows = meme_gt['MLE']['content']['0']
    
    try:
        seq_dict, _, embedded_tree_str = parse_nexus_alignment_and_embedded_tree(nex_path)
        if not seq_dict:
            continue
            
        spec_names = list(seq_dict.keys())[:256]
        N = len(spec_names)
        seq_len = len(seq_dict[spec_names[0]]) // 3
        
        # Parse tree
        tree_obj = None
        nwk_path = os.path.join(DATA_DIR, dataset_name.replace('.nex', '.nwk'))
        if os.path.exists(nwk_path):
            tree_obj = Phylo.read(nwk_path, 'newick')
        elif embedded_tree_str:
            clean_t = embedded_tree_str.strip()
            clean_t = clean_t.split(';')[-2] + ';' if ';' in clean_t[:-1] else clean_t
            if not clean_t.endswith(';'): clean_t += ';'
            tree_obj = Phylo.read(StringIO(clean_t), 'newick')
            
        if not tree_obj:
            continue
            
        tree_len = tree_obj.total_branch_length()
        _, dist_map, _ = calculate_patristic_distances(tree_obj)
        D = np.zeros((N, N))
        norm_map = {s: s.replace("'", "").replace('"', '').strip() for s in spec_names}
        for i, s1 in enumerate(spec_names):
            for j, s2 in enumerate(spec_names):
                n1, n2 = norm_map[s1], norm_map[s2]
                D[i, j] = dist_map.get(n1, {}).get(n2, 0.0)
                
        mds = compute_mds_coordinates(D, 4)
        msa_tokens = torch.zeros(seq_len, N, 1, dtype=torch.long)
        aa_tokens = torch.zeros(seq_len, N, 1, dtype=torch.long)
        for s_idx in range(seq_len):
            for spec_i, sname in enumerate(spec_names):
                codon = seq_dict[sname][s_idx*3:s_idx*3+3]
                msa_tokens[s_idx, spec_i, 0] = get_codon_token(codon)
                aa_tokens[s_idx, spec_i, 0] = get_aa_token(codon)
                
        dist_t = torch.from_numpy(D).unsqueeze(0).expand(seq_len, -1, -1).float()
        mds_t = torch.from_numpy(mds).unsqueeze(0).expand(seq_len, -1, -1).float()
        mask_t = torch.zeros(seq_len, N, dtype=torch.bool)
        
        with torch.no_grad():
            out_lrt, _, _, _, _ = model(msa_tokens, aa_tokens, dist_t, mds_t, mask_t)
            
        pred_continuous = out_lrt.squeeze().cpu().numpy()
                
        dataset_records = []
        for s_idx in range(min(seq_len, len(mle_rows))):
            row = mle_rows[s_idx]
            meme_lrt = float(row[5])
            meme_pval = float(row[6])
            
            pred_lrt = float(pred_continuous[s_idx])
            
            rec = {
                'dataset': dataset_name,
                'site': s_idx + 1,
                'meme_lrt': meme_lrt,
                'meme_pval': meme_pval,
                'pred_lrt': pred_lrt,
                'is_gt_sig': meme_pval <= 0.05,
                'is_pred_sig': pred_lrt >= 5.14
            }
            dataset_records.append(rec)
            all_site_records.append(rec)
            
        df_ds = pd.DataFrame(dataset_records)
        gt_sig_cnt = int(df_ds['is_gt_sig'].sum())
        pred_sig_cnt = int((df_ds['is_gt_sig'] & df_ds['is_pred_sig']).sum())
        
        recall_pct = (pred_sig_cnt / gt_sig_cnt * 100.0) if gt_sig_cnt > 0 else 0.0
        rho = df_ds['meme_lrt'].corr(df_ds['pred_lrt'], method='spearman')
        
        total_gt_sig += gt_sig_cnt
        total_detected_sig += pred_sig_cnt
        
        print(f"{dataset_name:<25} | {len(df_ds):<5} | {tree_len:<8.2f} | {gt_sig_cnt:<18} | {pred_sig_cnt:<13} | {recall_pct:<8.2f}% | {rho:<12.4f}")
    except Exception as e:
        print(f"{dataset_name:<25} | Error: {e}")

print("=" * 115)
df_all = pd.DataFrame(all_site_records)
overall_recall = (total_detected_sig / total_gt_sig * 100.0) if total_gt_sig > 0 else 0.0
overall_rho = df_all['meme_lrt'].corr(df_all['pred_lrt'], method='spearman')

gt_negatives = df_all[~df_all['is_gt_sig']]
fp_cnt = int(gt_negatives['is_pred_sig'].sum())
fpr_pct = (fp_cnt / len(gt_negatives) * 100.0) if len(gt_negatives) > 0 else 0.0

gt_positives = df_all[df_all['is_gt_sig']]
mean_gt_lrt = gt_positives['meme_lrt'].mean()
mean_pred_lrt = gt_positives['pred_lrt'].mean()

print(f"🏆 OVERALL SUMMARY:")
print(f"   • Total Ground Truth MEME Significant Sites (p <= 0.05) : {total_gt_sig}")
print(f"   • Total Correctly Detected by axoMEME 2.0 (LRT >= 5.14) : {total_detected_sig}")
print(f"   • Overall Sensitivity / Recall                         : {overall_recall:.2f}%")
print(f"   • False Positives (LRT_pred >= 5.14 on GT p > 0.05)    : {fp_cnt}/{len(gt_negatives)} (FPR: {fpr_pct:.2f}%)")
print(f"   • Mean Ground Truth LRT on Selection Sites             : {mean_gt_lrt:.2f}")
print(f"   • Mean Predicted LRT on Selection Sites                : {mean_pred_lrt:.2f}")
print(f"   • Overall Spearman Rank Correlation                    : {overall_rho:.4f}")
print("=" * 115)
