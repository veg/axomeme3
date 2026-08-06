#!/usr/bin/env python3
import os, sys, re, json, math, random, gzip, subprocess
import torch
import numpy as np
import pandas as pd
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

SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch", "hyphy_simulamg94_nulls")
os.makedirs(SCRATCH_DIR, exist_ok=True)

SIMULATE_BF_PATH = "/Users/sergei/Development/hyphy-analyses/SimulateMG94/SimulateMG94.bf"

def clean_newick_string(tree_obj):
    out = StringIO()
    Phylo.write(tree_obj, out, "newick")
    newick_str = out.getvalue().strip()
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str)
    newick_str = re.sub(r'\[.*?\]', '', newick_str)
    newick_str = re.sub(r'\)[0-9.]*:', '):', newick_str)
    return newick_str

def write_nexus_tree(filepath, tree_newick_str):
    with open(filepath, "w") as f:
        f.write(tree_newick_str + "\n")

def run_hyphy_meme(nex_path, json_out_path):
    cmd = ["hyphy", "meme", "--alignment", nex_path, "--tree", nex_path, "--output", json_out_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[!] HyPhy MEME error on {nex_path}: {res.stderr[:200]}")

def main():
    print("=" * 110)
    print("🚀 HYPHY SimulateMG94.bf STRICT NULL BENCHMARK (N=64, 128, 256, 512 TAXA, 200 CODON SITES)")
    print("=" * 110)
    
    tree_path = os.path.join(PROJECT_ROOT, "pruned_species_tree.nhx")
    full_tree = Phylo.read(tree_path, "newick")
    full_terminals = full_tree.get_terminals()
    
    taxon_scales = [64, 128, 256, 512]
    
    device = torch.device('cpu')
    model_path = os.path.join(PROJECT_ROOT, 'selection_transformer_edge_best-34.pt')
    axomeme_model = load_model(model_path, device)
    
    scale_results = []
    
    for n_taxa in taxon_scales:
        print(f"\n--- ⚡ Simulating with HyPhy SimulateMG94.bf: N = {n_taxa} Taxa (200 Neutral Sites) ---")
        
        # Subsample n_taxa tree
        random.seed(42)
        keep_taxa = random.sample(full_terminals, min(n_taxa, len(full_terminals)))
        keep_names = {t.name for t in keep_taxa}
        
        sub_tree = Phylo.read(tree_path, "newick")
        for leaf in sub_tree.get_terminals():
            if leaf.name not in keep_names:
                sub_tree.prune(leaf)
                
        tree_len = sub_tree.total_branch_length()
        clean_tree_str = clean_newick_string(sub_tree)
        
        tree_file_path = os.path.join(SCRATCH_DIR, f"tree_{n_taxa}taxa.nwk")
        write_nexus_tree(tree_file_path, clean_tree_str)
        
        out_prefix = os.path.join(SCRATCH_DIR, f"hyphy_sim_mg94_{n_taxa}taxa")
        
        cmd_sim = [
            "hyphy", SIMULATE_BF_PATH,
            "--tree", tree_file_path,
            "--sites", "200",
            "--output", out_prefix,
            "--seed", "42"
        ]
        
        print(f"  [*] Executing HyPhy SimulateMG94.bf for N={n_taxa}...")
        res = subprocess.run(cmd_sim, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        sim_nex_path = f"{out_prefix}.replicate.1.nex"
        if not os.path.exists(sim_nex_path):
            print(f"[!] Error: Generated alignment not found at {sim_nex_path}")
            continue
            
        print(f"  [*] Generated alignment (N={n_taxa}, T={tree_len:.2f}) at {sim_nex_path}")
        
        # 1. Run HyPhy MEME on SimulateMG94.bf output
        meme_json_path = os.path.join(SCRATCH_DIR, f"hyphy_sim_mg94_{n_taxa}taxa.MEME.json")
        print(f"  [*] Running HyPhy MEME on N={n_taxa} alignment...")
        run_hyphy_meme(sim_nex_path, meme_json_path)
        
        with open(meme_json_path) as f: meme_gt = json.load(f)
        meme_mle = meme_gt['MLE']['content']['0']
        meme_pvals = [row[6] for row in meme_mle]
        meme_fp_05 = sum(1 for p in meme_pvals if p <= 0.05)
        meme_fp_10 = sum(1 for p in meme_pvals if p <= 0.10)
        
        # 2. Run axoMEME Checkpoint 34
        seq_dict, _, _ = parse_nexus_alignment_and_embedded_tree(sim_nex_path)
        spec_names = list(seq_dict.keys())[:256] # axoMEME max species context
        N = len(spec_names)
        seq_len = len(seq_dict[spec_names[0]]) // 3
        
        t_phy = Phylo.read(sim_nex_path, 'nexus')
        _, dist_map, _ = calculate_patristic_distances(t_phy)
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
        axomeme_fp_05 = sum(1 for l in axomeme_lrts if l >= 5.1384)
        axomeme_fp_10 = sum(1 for l in axomeme_lrts if l >= 3.8078)
        
        scale_results.append({
            'Taxa (N)': n_taxa,
            'Tree Length (T)': f'{tree_len:.2f}',
            'MEME FP (p<=0.05)': meme_fp_05,
            'MEME FPR (p<=0.05)': f'{meme_fp_05 / 200 * 100:.2f}%',
            'axoMEME FP (p<=0.05)': axomeme_fp_05,
            'axoMEME FPR (p<=0.05)': f'{axomeme_fp_05 / 200 * 100:.2f}%',
            'MEME FP (p<=0.10)': meme_fp_10,
            'axoMEME FP (p<=0.10)': axomeme_fp_10,
            'axoMEME Mean LRT': f'{np.mean(axomeme_lrts):.4f}',
            'axoMEME Median LRT': f'{np.median(axomeme_lrts):.4f}',
            'axoMEME Max LRT': f'{np.max(axomeme_lrts):.4f}'
        })
        
    df_res = pd.DataFrame(scale_results)
    
    print("\n" + "=" * 120)
    print("📊 HYPHY SimulateMG94.bf STRICT NULL BENCHMARK RESULTS (MEME VS AXOMEME CHECKPOINT 34)")
    print("=" * 120)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
