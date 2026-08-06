#!/usr/bin/env python3
import os, sys, re, json, math, random, gzip, subprocess
import torch
import numpy as np
import pandas as pd
from io import StringIO
from Bio import Phylo
import pyvolve

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

SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch", "taxon_scale_null_sims")
os.makedirs(SCRATCH_DIR, exist_ok=True)

def clean_newick_string(tree_obj):
    out = StringIO()
    Phylo.write(tree_obj, out, "newick")
    newick_str = out.getvalue().strip()
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str)
    newick_str = re.sub(r'\[.*?\]', '', newick_str)
    newick_str = re.sub(r'\)[0-9.]*:', '):', newick_str)
    return newick_str

def write_nexus_alignment(filepath, sequences, tree_newick_str):
    taxa = sorted(list(sequences.keys()))
    seq_len = len(sequences[taxa[0]])
    
    lines = []
    lines.append("#NEXUS")
    lines.append("BEGIN TAXA;")
    lines.append(f"  DIMENSIONS NTAX={len(taxa)};")
    lines.append("  TAXLABELS")
    for t in taxa: lines.append(f"    '{t}'")
    lines.append("  ;")
    lines.append("END;")
    lines.append("BEGIN CHARACTERS;")
    lines.append(f"  DIMENSIONS NCHAR={seq_len};")
    lines.append("  FORMAT DATATYPE=DNA MISSING=? GAP=-;")
    lines.append("  MATRIX")
    for t in taxa: lines.append(f"    '{t}'  {sequences[t]}")
    lines.append("  ;")
    lines.append("END;")
    if tree_newick_str:
        lines.append("BEGIN TREES;")
        lines.append(f"  TREE tree = {tree_newick_str}")
        lines.append("END;")
        
    content = "\n".join(lines) + "\n"
    with open(filepath, "w") as f: f.write(content)

def run_hyphy_meme(nex_path, json_out_path):
    cmd = ["hyphy", "meme", "--alignment", nex_path, "--tree", nex_path, "--output", json_out_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[!] HyPhy MEME failed on {nex_path}: {res.stderr[:200]}")

def main():
    print("=" * 100)
    print("🚀 TAXON SCALE STRICT NULL BENCHMARK (MG94 MODEL, OMEGA = 1.0, 200 CODON SITES)")
    print("=" * 100)
    
    tree_path = os.path.join(PROJECT_ROOT, "pruned_species_tree.nhx")
    full_tree = Phylo.read(tree_path, "newick")
    full_terminals = full_tree.get_terminals()
    
    taxon_scales = [64, 128, 256, 512]
    
    device = torch.device('cpu')
    model_path = os.path.join(PROJECT_ROOT, 'selection_transformer_edge_best-34.pt')
    axomeme_model = load_model(model_path, device)
    
    scale_results = []
    
    for n_taxa in taxon_scales:
        print(f"\n--- ⚡ Simulating Taxon Scale: N = {n_taxa} Taxa (200 Neutral Sites, MG94 model) ---")
        
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
        
        pyvolve_tree = pyvolve.read_tree(tree=clean_tree_str)
        
        # Strict null MG94 model (omega = 1.0)
        m_mg94 = pyvolve.Model("MG94", {"omega": 1.0})
        p_mg94 = pyvolve.Partition(models=m_mg94, size=200)
        
        sim_ev = pyvolve.Evolver(partitions=p_mg94, tree=pyvolve_tree)
        sim_nex_path = os.path.join(SCRATCH_DIR, f"sim_null_mg94_{n_taxa}taxa.nex")
        sim_ev(seqfile=os.path.join(SCRATCH_DIR, f"tmp_{n_taxa}.fasta"), write_flag=False)
        sequences = sim_ev.get_sequences()
        write_nexus_alignment(sim_nex_path, sequences, clean_tree_str)
        
        print(f"  [*] Simulated NEXUS alignment (N={n_taxa}, T={tree_len:.2f}) written to {sim_nex_path}")
        
        # 1. Run HyPhy MEME
        meme_json_path = os.path.join(SCRATCH_DIR, f"sim_null_mg94_{n_taxa}taxa.MEME.json")
        print(f"  [*] Running HyPhy MEME on N={n_taxa} null alignment...")
        run_hyphy_meme(sim_nex_path, meme_json_path)
        
        with open(meme_json_path) as f: meme_gt = json.load(f)
        meme_mle = meme_gt['MLE']['content']['0']
        meme_pvals = [row[6] for row in meme_mle]
        meme_fp_05 = sum(1 for p in meme_pvals if p <= 0.05)
        meme_fp_10 = sum(1 for p in meme_pvals if p <= 0.10)
        
        # 2. Run axoMEME Checkpoint 34
        seq_dict, _, _ = parse_nexus_alignment_and_embedded_tree(sim_nex_path)
        spec_names = list(seq_dict.keys())[:256] # Model context capped at 256
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
    print("📊 TAXON SCALE NULL SIMULATION RESULTS (MEME VS AXOMEME CHECKPOINT 34)")
    print("=" * 120)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
