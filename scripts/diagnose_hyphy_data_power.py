#!/usr/bin/env python3
import os, sys, glob, json, re, torch
import numpy as np, pandas as pd
from io import StringIO
from Bio import Phylo

PROJECT_ROOT = "/Users/sergei/Projects/TOGA_MEME"
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_checkpoint_30 import load_model
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
model_path = os.path.join(PROJECT_ROOT, 'selection_transformer_edge_best-34.pt')
axomeme_model = load_model(model_path, device)

all_site_data = []

print("=" * 115)
print(f"🚀 DIAGNOSING AXOMEME (CKPT 34) POWER LOSS ON {len(meme_files)} HYPHY BENCHMARK DATASETS (hyphy/tests/data)")
print("=" * 115)

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
            
        spec_names = list(seq_dict.keys())[:256] # axoMEME max species context
        N = len(spec_names)
        N_total = len(seq_dict)
        seq_len = len(seq_dict[spec_names[0]]) // 3
        
        # Parse tree safely
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
            out_lrt, _, _, _, _ = axomeme_model(msa_tokens, aa_tokens, dist_t, mds_t, mask_t)
            
        axomeme_lrts = np.expm1(out_lrt.squeeze().numpy())
        
        for s_idx in range(min(seq_len, len(mle_rows))):
            row = mle_rows[s_idx]
            alpha = float(row[0])
            beta_pos = float(row[3])
            p_pos = float(row[4])
            meme_lrt = float(row[5])
            meme_pval = float(row[6])
            branches_sel = int(row[7])
            
            ax_lrt = float(axomeme_lrts[s_idx])
            
            is_meme_sig = (meme_pval <= 0.05)
            is_axomeme_sig = (ax_lrt >= 5.1384)
            
            all_site_data.append({
                'dataset': dataset_name,
                'site': s_idx + 1,
                'N_total': N_total,
                'N_eval': N,
                'tree_len': tree_len,
                'alpha': alpha,
                'beta_pos': beta_pos,
                'p_pos': p_pos,
                'meme_lrt': meme_lrt,
                'meme_pval': meme_pval,
                'branches_sel': branches_sel,
                'axomeme_lrt': ax_lrt,
                'meme_sig': is_meme_sig,
                'axomeme_sig': is_axomeme_sig,
                'detected': (is_meme_sig and is_axomeme_sig),
                'missed': (is_meme_sig and not is_axomeme_sig)
            })
    except Exception as e:
        print(f"[!] Error on {dataset_name}: {e}")

df_all = pd.DataFrame(all_site_data)
df_sig = df_all[df_all['meme_sig']]

print("\n" + "=" * 115)
print("📊 HYPHY BENCHMARK DATASETS (hyphy/tests/data) POWER & RECALL BREAKDOWN:")
print("=" * 115)
print(f"  • Total Empirical Datasets Processed:                {len(df_all['dataset'].unique())}")
print(f"  • Total Alignment Sites Evaluated:                   {len(df_all)}")
print(f"  • Ground-Truth MEME Significant Sites (p <= 0.05):   {len(df_sig)}")
print(f"  • Sites Successfully Detected by axoMEME:            {df_sig['detected'].sum()} (Overall Recall = {df_sig['detected'].mean()*100:.2f}%)")
print(f"  • Sites MISSED by axoMEME (Power Loss):               {df_sig['missed'].sum()} (Miss Rate = {df_sig['missed'].mean()*100:.2f}%)")

ds_summary = df_sig.groupby('dataset').agg(
    total_sig=('meme_sig', 'count'),
    detected=('detected', 'sum'),
    recall=('detected', 'mean'),
    mean_meme_lrt=('meme_lrt', 'mean'),
    mean_axomeme_lrt=('axomeme_lrt', 'mean'),
    mean_branches=('branches_sel', 'mean'),
    tree_len=('tree_len', 'first'),
    N_total=('N_total', 'first')
).reset_index()

print("\n📋 DATASET-BY-DATASET RECALL TABLE:")
print(ds_summary.to_string(index=False))

print("\n🔍 RECALL BREAKDOWN BY NUMBER OF SELECTED BRANCHES (# branches under selection):")
b_summary = df_sig.groupby('branches_sel').agg(
    total_sig=('meme_sig', 'count'),
    detected=('detected', 'sum'),
    recall=('detected', 'mean'),
    mean_meme_lrt=('meme_lrt', 'mean'),
    mean_axomeme_lrt=('axomeme_lrt', 'mean')
).reset_index()
print(b_summary.head(10).to_string(index=False))

# Save results
df_all.to_csv(os.path.join(PROJECT_ROOT, "hyphy_tests_data_power_breakdown.csv"), index=False)
