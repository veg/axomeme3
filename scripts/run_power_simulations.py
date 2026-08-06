#!/usr/bin/env python3
import os, sys, re, json, math, random, gzip, subprocess
import torch
import numpy as np
import pandas as pd
from io import StringIO
from Bio import Phylo
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import pyvolve

PROJECT_ROOT = "/Users/sergei/Projects/TOGA_MEME"
sys.path.insert(0, PROJECT_ROOT)

# Load axoMEME model loading and inference helpers
from scripts.eval_checkpoint_30 import load_model
from scripts.predict_regression_nexus import (
    parse_nexus_alignment_and_embedded_tree,
    calculate_patristic_distances,
    compute_mds_coordinates,
    get_codon_token,
    get_aa_token
)

SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch", "power_simulations")
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
    for t in taxa:
        lines.append(f"    '{t}'")
    lines.append("  ;")
    lines.append("END;")
    lines.append("BEGIN CHARACTERS;")
    lines.append(f"  DIMENSIONS NCHAR={seq_len};")
    lines.append("  FORMAT DATATYPE=DNA MISSING=? GAP=-;")
    lines.append("  MATRIX")
    for t in taxa:
        lines.append(f"    '{t}'  {sequences[t]}")
    lines.append("  ;")
    lines.append("END;")
    if tree_newick_str:
        lines.append("BEGIN TREES;")
        lines.append(f"  TREE tree = {tree_newick_str}")
        lines.append("END;")
        
    content = "\n".join(lines) + "\n"
    with open(filepath, "w") as f:
        f.write(content)

def run_hyphy_meme(nex_path, json_out_path):
    cmd = [
        "hyphy", "meme",
        "--alignment", nex_path,
        "--tree", nex_path,
        "--output", json_out_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[!] MEME Error on {nex_path}: {res.stderr[:200]}")
        return None
    return json_out_path

def assign_model_to_nodes(node, model):
    node.model = model
    for child in node.children:
        assign_model_to_nodes(child, model)

def main():
    print("=" * 80)
    print("🚀 POWER SIMULATION BENCHMARK: MEME VS AXOMEME (CHECKPOINT 34)")
    print("=" * 80)
    
    tree_path = os.path.join(PROJECT_ROOT, "pruned_species_tree.nhx")
    if not os.path.exists(tree_path):
        print(f"[!] Error: Tree file not found at {tree_path}")
        return
        
    full_tree = Phylo.read(tree_path, "newick")
    terminals = full_tree.get_terminals()
    
    # Sample 128 species tree for power simulation
    random.seed(42)
    keep_taxa = random.sample(terminals, min(128, len(terminals)))
    keep_names = {t.name for t in keep_taxa}
    for leaf in terminals:
        if leaf.name not in keep_names:
            full_tree.prune(leaf)
            
    clean_tree_str = clean_newick_string(full_tree)
    print(f"[*] Pruned simulation tree: {len(full_tree.get_terminals())} species.")
    
    # We will test 3 selection strength scenarios: omega_pos in [4.0, 8.0, 16.0]
    omega_scenarios = [4.0, 8.0, 16.0]
    
    # Load axoMEME Checkpoint 34
    device = torch.device('cpu')
    model_path = os.path.join(PROJECT_ROOT, 'selection_transformer_edge_best-34.pt')
    axomeme_model = load_model(model_path, device)
    
    summary_results = []
    
    for omega_pos in omega_scenarios:
        print(f"\n--- ⚡ Simulating Scenario: Omega_Pos = {omega_pos:.1f} (40 Alt Sites, 160 Null Sites) ---")
        pyvolve_tree = pyvolve.read_tree(tree=clean_tree_str)
        def get_nodes(n):
            res = [n]
            for c in n.children: res.extend(get_nodes(c))
            return res
        all_nodes = [node for node in get_nodes(pyvolve_tree) if node.name != "root"]
        num_fg = max(1, int(len(all_nodes) * 0.20)) # 20% foreground branches under selection
        fg_nodes = set(random.sample(all_nodes, num_fg))
        
        # Models
        m_null = pyvolve.Model("GY", {"omega": 1.0})
        m_null.name = "m_null"
        
        m_bg = pyvolve.Model("GY", {"omega": 0.2})
        m_fg = pyvolve.Model("GY", {"omega": omega_pos})
        m_bg.name = "m_bg"
        m_fg.name = "m_fg"
        
        # Assign foreground branches m_fg
        for n in fg_nodes:
            n.model = m_fg
            
        part_null = pyvolve.Partition(models=m_null, size=160)
        part_alt = pyvolve.Partition(models=[m_bg, m_fg], root_model_name="m_bg", size=40)
        
        # Run pyvolve simulation
        sim_ev = pyvolve.Evolver(partitions=[part_null, part_alt], tree=pyvolve_tree)
        sim_nex_name = f"sim_omega_{int(omega_pos)}.nex"
        sim_nex_path = os.path.join(SCRATCH_DIR, sim_nex_name)
        
        sim_ev(seqfile=os.path.join(SCRATCH_DIR, "tmp_seq.fasta"), write_flag=False)
        sequences = sim_ev.get_sequences()
        
        # Write clean NEXUS alignment file with embedded tree
        write_nexus_alignment(sim_nex_path, sequences, clean_tree_str)
        print(f"  [*] Simulated alignment written to {sim_nex_path}")
        
        # 1. Run MEME Ground Truth on Simulated Alignment
        meme_json_path = os.path.join(SCRATCH_DIR, f"sim_omega_{int(omega_pos)}.MEME.json")
        print(f"  [*] Running HyPhy MEME on {sim_nex_name}...")
        run_hyphy_meme(sim_nex_path, meme_json_path)
        
        with open(meme_json_path) as f:
            meme_gt = json.load(f)
            
        meme_mle = meme_gt['MLE']['content']['0']
        meme_pvals = [row[6] for row in meme_mle]
        meme_lrts = [row[5] for row in meme_mle]
        
        # 2. Run axoMEME Checkpoint 34 on Simulated Alignment
        seq_dict, _, _ = parse_nexus_alignment_and_embedded_tree(sim_nex_path)
        spec_names = list(seq_dict.keys())[:256]
        N = len(spec_names)
        seq_len = len(seq_dict[spec_names[0]]) // 3
        
        # Calculate patristic matrix
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
        
        # Ground Truth Labels: Sites 1-160 = 0 (Null), Sites 161-200 = 1 (Positive Selection)
        true_labels = [0]*160 + [1]*40
        
        # MEME Metrics
        meme_calls_05 = [1 if p <= 0.05 else 0 for p in meme_pvals]
        meme_tp = sum(1 for c, t in zip(meme_calls_05, true_labels) if c == 1 and t == 1)
        meme_fp = sum(1 for c, t in zip(meme_calls_05, true_labels) if c == 1 and t == 0)
        meme_power = meme_tp / 40.0
        meme_fpr = meme_fp / 160.0
        meme_roc = roc_auc_score(true_labels, [-p for p in meme_pvals])
        meme_pr = average_precision_score(true_labels, [-p for p in meme_pvals])
        
        # axoMEME Metrics (threshold LRT >= 5.1384 for p <= 0.05)
        axomeme_calls_05 = [1 if l >= 5.1384 else 0 for l in axomeme_lrts]
        axomeme_tp = sum(1 for c, t in zip(axomeme_calls_05, true_labels) if c == 1 and t == 1)
        axomeme_fp = sum(1 for c, t in zip(axomeme_calls_05, true_labels) if c == 1 and t == 0)
        axomeme_power = axomeme_tp / 40.0
        axomeme_fpr = axomeme_fp / 160.0
        axomeme_roc = roc_auc_score(true_labels, axomeme_lrts)
        axomeme_pr = average_precision_score(true_labels, axomeme_lrts)
        
        summary_results.append({
            'Omega_Pos': omega_pos,
            'MEME_Power (Recall)': meme_power,
            'axoMEME_Power (Recall)': axomeme_power,
            'MEME_FPR': meme_fpr,
            'axoMEME_FPR': axomeme_fpr,
            'MEME_ROC_AUC': meme_roc,
            'axoMEME_ROC_AUC': axomeme_roc,
            'MEME_PR_AUC': meme_pr,
            'axoMEME_PR_AUC': axomeme_pr
        })
        
    df_sum = pd.DataFrame(summary_results)
    
    print("\n" + "=" * 100)
    print("📊 POWER SIMULATION BENCHMARK SUMMARY TABLE (MEME VS AXOMEME CHECKPOINT 34)")
    print("=" * 100)
    print(df_sum.to_string(index=False))

if __name__ == "__main__":
    main()
