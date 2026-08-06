#!/usr/bin/env python3
import os, sys, re, json, math, time, random, gzip, subprocess
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
    compute_mds_coordinates,
    get_codon_token,
    get_aa_token
)

SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch", "progressive_pruning_nulls")
SIMULATE_BF_PATH = "/Users/sergei/Development/hyphy-analyses/SimulateMG94/SimulateMG94.bf"

emp_gz_path = os.path.join(PROJECT_ROOT, "msa", "DGCR2.gz")
tmp_emp_nex = "/tmp/DGCR2_full.nex"

with gzip.open(emp_gz_path, "rt") as f_in:
    full_content = f_in.read()

seq_dict_full, _, _ = parse_nexus_alignment_and_embedded_tree(tmp_emp_nex)
full_species_list = list(seq_dict_full.keys())
total_codons = len(seq_dict_full[full_species_list[0]]) // 3

tree_match = re.search(r'TREE\s+tree\s*=\s*(.*?);', full_content, re.IGNORECASE | re.DOTALL)
clean_tree_str_full = tree_match.group(1).strip()
master_tree = Phylo.read(StringIO(clean_tree_str_full), 'newick')
master_terminals = master_tree.get_terminals()

species_counts = [644, 594, 544, 494, 444, 394, 344, 294, 244, 194, 144, 94, 44]

device = torch.device('cpu')
model_path = os.path.join(PROJECT_ROOT, 'selection_transformer_edge_best-34.pt')
axomeme_model = load_model(model_path, device)

pruning_results = []

def clean_newick_string(tree_obj):
    out = StringIO()
    Phylo.write(tree_obj, out, 'newick')
    newick_str = out.getvalue().strip()
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str)
    newick_str = re.sub(r'\[.*?\]', '', newick_str)
    newick_str = re.sub(r'\)[0-9.]*:', '):', newick_str)
    return newick_str

def compute_fast_patristic_matrix(t, spec_names):
    root = t.root
    node_depth = {}
    def compute_depths(node, current_d):
        node_depth[node] = current_d
        for child in node.clades:
            compute_depths(child, current_d + (child.branch_length or 0.0))
    compute_depths(root, 0.0)

    parent_map = {}
    def build_parents(node):
        for child in node.clades:
            parent_map[child] = node
            build_parents(child)
    build_parents(root)

    leaf_map = {leaf.name.replace("'", "").replace('"', '').strip(): leaf for leaf in t.get_terminals()}
    lineages = {}
    for name in spec_names:
        norm_name = name.replace("'", "").replace('"', '').strip()
        curr = leaf_map[norm_name]
        path = []
        while curr is not None:
            path.append(curr)
            curr = parent_map.get(curr)
        lineages[name] = path

    N = len(spec_names)
    D = np.zeros((N, N))
    for i in range(N):
        l1 = lineages[spec_names[i]]
        set1 = set(l1)
        norm_i = spec_names[i].replace("'", "").replace('"', '').strip()
        for j in range(i + 1, N):
            l2 = lineages[spec_names[j]]
            norm_j = spec_names[j].replace("'", "").replace('"', '').strip()
            for ancestor in l2:
                if ancestor in set1:
                    lca = ancestor
                    break
            dist = node_depth[leaf_map[norm_i]] + node_depth[leaf_map[norm_j]] - 2 * node_depth[lca]
            D[i, j] = dist
            D[j, i] = dist
    return D

print("=" * 115, flush=True)
print("🚀 FAST UNTRUNCATED FULL SPECIES BENCHMARK (N=644 DOWN TO N=44)", flush=True)
print("=" * 115, flush=True)

for n_target in species_counts:
    random.seed(42)
    keep_taxa = random.sample(master_terminals, min(n_target, len(master_terminals)))
    keep_names = {t.name for t in keep_taxa}
    
    sub_tree = Phylo.read(StringIO(clean_tree_str_full), 'newick')
    for leaf in sub_tree.get_terminals():
        if leaf.name not in keep_names:
            sub_tree.prune(leaf)
            
    tree_len = sub_tree.total_branch_length()
    clean_sub_tree_str = clean_newick_string(sub_tree)
    
    tree_file_path = os.path.join(SCRATCH_DIR, f'tree_{n_target}taxa.nwk')
    with open(tree_file_path, 'w') as f:
        f.write(clean_sub_tree_str + '\n')
        
    out_prefix = os.path.join(SCRATCH_DIR, f'hyphy_sim_dgcr2_{n_target}taxa')
    sim_nex_path = f'{out_prefix}.replicate.1.nex'
    
    if not os.path.exists(sim_nex_path):
        cmd_sim = [
            'hyphy', SIMULATE_BF_PATH,
            '--tree', tree_file_path,
            '--sites', str(total_codons),
            '--base-frequencies', tmp_emp_nex,
            '--output', out_prefix,
            '--seed', '42'
        ]
        subprocess.run(cmd_sim, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    # axoMEME FULL N SPECIES EVALUATION (NO CAPPING AT 256)
    seq_dict_sim, _, _ = parse_nexus_alignment_and_embedded_tree(sim_nex_path)
    spec_names = list(seq_dict_sim.keys()) # ALL SPECIES IN THE PRUNED ALIGNMENT
    N_eval = len(spec_names)
    seq_len = len(seq_dict_sim[spec_names[0]]) // 3
    
    t_phy = Phylo.read(tree_file_path, 'newick')
    D = compute_fast_patristic_matrix(t_phy, spec_names)
    mds = compute_mds_coordinates(D, 4)
    
    msa_tokens = torch.zeros(seq_len, N_eval, 1, dtype=torch.long)
    aa_tokens = torch.zeros(seq_len, N_eval, 1, dtype=torch.long)
    for s_idx in range(seq_len):
        for spec_i, sname in enumerate(spec_names):
            codon = seq_dict_sim[sname][s_idx*3:s_idx*3+3]
            msa_tokens[s_idx, spec_i, 0] = get_codon_token(codon)
            aa_tokens[s_idx, spec_i, 0] = get_aa_token(codon)
            
    dist_t = torch.from_numpy(D).unsqueeze(0).expand(seq_len, -1, -1).float()
    mds_t = torch.from_numpy(mds).unsqueeze(0).expand(seq_len, -1, -1).float()
    mask_t = torch.zeros(seq_len, N_eval, dtype=torch.bool)
    
    with torch.no_grad():
        out_lrt, _, _, _, _ = axomeme_model(msa_tokens, aa_tokens, dist_t, mds_t, mask_t)
        
    axomeme_lrts = np.expm1(out_lrt.squeeze().numpy())
    axomeme_fp_05 = sum(1 for l in axomeme_lrts if l >= 5.1384)
    axomeme_fp_10 = sum(1 for l in axomeme_lrts if l >= 3.8078)
    
    res_dict = {
        'Species (N)': N_eval,
        'Tree Length (T)': f'{tree_len:.2f}',
        'axoMEME FP (p<=0.05)': axomeme_fp_05,
        'axoMEME FPR (p<=0.05)': f'{axomeme_fp_05 / total_codons * 100:.2f}%',
        'axoMEME FP (p<=0.10)': axomeme_fp_10,
        'axoMEME FPR (p<=0.10)': f'{axomeme_fp_10 / total_codons * 100:.2f}%',
        'axoMEME Mean LRT': f'{np.mean(axomeme_lrts):.4f}',
        'axoMEME Median LRT': f'{np.median(axomeme_lrts):.4f}',
        'axoMEME Max LRT': f'{np.max(axomeme_lrts):.4f}'
    }
    pruning_results.append(res_dict)
    
    print(f'📌 N={N_eval:3d} | T={tree_len:5.2f} | FP(p<=0.05)={axomeme_fp_05:2d} ({axomeme_fp_05/total_codons*100:5.2f}%) | FP(p<=0.10)={axomeme_fp_10:2d} ({axomeme_fp_10/total_codons*100:5.2f}%) | Mean LRT={np.mean(axomeme_lrts):.4f} | Max LRT={np.max(axomeme_lrts):.4f}', flush=True)

df_res = pd.DataFrame(pruning_results)
print('\n' + '=' * 115, flush=True)
print('📊 FAST UNTRUNCATED FULL SPECIES BENCHMARK RESULTS (N=644 DOWN TO N=44)', flush=True)
print('=' * 115, flush=True)
print(df_res.to_string(index=False), flush=True)
