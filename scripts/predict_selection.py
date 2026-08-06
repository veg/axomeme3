#!/usr/bin/env python3
import os
import re
import gzip
import argparse
import pandas as pd
import torch
import torch.nn as nn
import sys

# Standard genetic code dictionary
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGA':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

# Codon vocabulary setup
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

def translate_codon(codon):
    codon = codon.upper()
    if len(codon) != 3 or '-' in codon or 'N' in codon:
        return '?'
    return GENETIC_CODE.get(codon, '?')

def get_codon_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 64
    if len(codon) != 3 or 'N' in codon:
        return 65
    return CODON_TO_IDX.get(codon, 65)

# --- Tree Parsing & Patristic Distance calculation ---
class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

def parse_newick(newick_str):
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str.strip())
    newick_str = re.sub(r'\[.*?\]', '', newick_str)
    
    tokens = []
    i = 0
    while i < len(newick_str):
        c = newick_str[i]
        if c in '(),;':
            tokens.append(c)
            i += 1
        elif c == ':':
            i += 1
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;':
                i += 1
            tokens.append(('length', float(newick_str[start:i])))
        else:
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;:':
                i += 1
            tokens.append(('name', newick_str[start:i].strip()))
            
    root = TreeNode()
    current = root
    for t in tokens:
        if t == '(':
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ',':
            current = current.parent
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ')':
            current = current.parent
        elif isinstance(t, tuple) and t[0] == 'name':
            current.name = t[1]
        elif isinstance(t, tuple) and t[0] == 'length':
            current.length = t[1]
            
    return root

def get_path_to_root(node):
    path = []
    curr = node
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path

def get_leaves(node):
    if not node.children:
        return [node]
    leaves = []
    for c in node.children:
        leaves.extend(get_leaves(c))
    return leaves

def get_patristic_distances(root):
    leaves = get_leaves(root)
    leaf_paths = {}
    leaf_len_to_root = {}
    for leaf in leaves:
        if not leaf.name:
            continue
        path = get_path_to_root(leaf)
        leaf_paths[leaf.name] = path
        length = 0.0
        for node in path[:-1]:
            length += node.length
        leaf_len_to_root[leaf.name] = length
        
    names = list(leaf_len_to_root.keys())
    n = len(names)
    dist_matrix = {}
    for name in names:
        dist_matrix[name] = {name: 0.0}
        
    for i in range(n):
        name1 = names[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, n):
            name2 = names[j]
            path2 = leaf_paths[name2]
            lca = None
            for node in path2:
                if node in set1:
                    lca = node
                    break
            
            if lca is not None:
                lca_path = get_path_to_root(lca)
                lca_len = sum(node.length for node in lca_path[:-1])
                dist = leaf_len_to_root[name1] + leaf_len_to_root[name2] - 2 * lca_len
            else:
                dist = leaf_len_to_root[name1] + leaf_len_to_root[name2]
                
            dist_matrix[name1][name2] = dist
            dist_matrix[name2][name1] = dist
            
    return names, dist_matrix

# Load model structure (needs to match train script)
sys.path.append(os.path.join(os.path.dirname(__file__), "../scratch"))
try:
    from train_transformer_selection import PhyloAxialTransformer
except ImportError:
    # Inline definition in case path is unresolved
    sys.path.append(os.path.dirname(__file__))
    from train_transformer_selection import PhyloAxialTransformer

def parse_nexus(align_path):
    taxlabels = []
    sequences = []
    tree_str = None
    in_taxlabels = False
    in_matrix = False
    in_trees = False
    
    open_func = gzip.open if align_path.endswith('.gz') else open
    with open_func(align_path, 'rt') as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip.upper().startswith("TAXLABELS"):
                in_taxlabels = True
                content = line_strip[len("TAXLABELS"):].strip()
                tokens = content.replace("'", "").replace(";", "").split()
                taxlabels.extend(tokens)
                if line_strip.endswith(";"):
                    in_taxlabels = False
                continue
            if in_taxlabels:
                tokens = line_strip.replace("'", "").replace(";", "").split()
                taxlabels.extend(tokens)
                if line_strip.endswith(";"):
                    in_taxlabels = False
                continue
            if line_strip.upper().startswith("MATRIX"):
                in_matrix = True
                continue
            if in_matrix:
                if line_strip == ";":
                    in_matrix = False
                    continue
                if line_strip.endswith(";"):
                    sequences.append(line_strip[:-1].strip())
                    in_matrix = False
                    continue
                sequences.append(line_strip)
                continue
            if line_strip.upper().startswith("BEGIN TREES"):
                in_trees = True
                continue
            if in_trees:
                if line_strip.upper().startswith("TREE "):
                    parts = line_strip.split('=', 1)
                    if len(parts) > 1:
                        tree_str = parts[1].strip()
                if line_strip == "END;":
                    in_trees = False
                    
    seq_dict = {}
    for label, seq in zip(taxlabels, sequences):
        seq_dict[label] = seq
        
    return seq_dict, tree_str

def main():
    parser = argparse.ArgumentParser(description="Run trained Selection Transformer on a NEXUS alignment.")
    parser.add_argument("--alignment", required=True, help="Path to NEXUS alignment file (.gz or uncompressed)")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output", default="selection_predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--window_size", type=int, default=5, help="MSA column window size centered at site")
    parser.add_argument("--max_species", type=int, default=100, help="Maximum number of sequences to process")
    args = parser.parse_args()
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running inference on device: {device}")
    
    # 1. Parse alignment and tree
    print(f"Parsing alignment from: {args.alignment}")
    seq_dict, tree_str = parse_nexus(args.alignment)
    if not seq_dict:
        print("Error: Empty alignment or invalid NEXUS format.")
        sys.exit(1)
        
    # Get species list and distance matrix
    species_names = []
    dist_matrix = {}
    if tree_str:
        print("Parsing tree and calculating pairwise evolutionary distances...")
        try:
            tree_root = parse_newick(tree_str)
            species_names, dist_matrix = get_patristic_distances(tree_root)
        except Exception as e:
            print(f"Warning: Tree parsing failed: {e}. Defaulting to flat distance structure.")
            
    if not species_names:
        species_names = list(seq_dict.keys())
        dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}
        
    # Restrict species count
    selected_species = species_names[:args.max_species]
    num_selected = len(selected_species)
    print(f"Selected {num_selected} sequences for analysis.")
    
    # Reference sequence (default to 'hg38', 'hg', or first sequence)
    ref_key = 'hg38' if 'hg38' in seq_dict else ('hg' if 'hg' in seq_dict else next(iter(seq_dict.keys())))
    ref_seq = seq_dict[ref_key]
    total_codons = len(ref_seq) // 3
    print(f"Reference sequence: '{ref_key}' | Total sites: {total_codons} codons")
    
    # 2. Build evolutionary distance tensor
    dist_tensor = torch.zeros(1, args.max_species, args.max_species)
    for i, spec1 in enumerate(selected_species):
        for j, spec2 in enumerate(selected_species):
            dist_tensor[0, i, j] = dist_matrix.get(spec1, {}).get(spec2, 0.0)
    dist_tensor = dist_tensor.to(device)
    
    # 3. Load trained model weights
    print(f"Loading model checkpoint: {args.model}")
    # Instantiate architecture
    model = PhyloAxialTransformer(embed_dim=64, num_heads=4, num_layers=2, window_size=args.window_size).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Predict loop
    half_win = args.window_size // 2
    predictions = []
    
    print("Running codon-level positive selection predictions...")
    with torch.no_grad():
        for site_idx in range(1, total_codons + 1):
            # Build input token matrix of shape (1, max_species, window_size)
            msa_tokens = torch.ones(1, args.max_species, args.window_size, dtype=torch.long) * 65
            
            for s_idx, spec in enumerate(selected_species):
                seq = seq_dict.get(spec, "")
                seq_len_codons = len(seq) // 3
                
                for w_idx in range(args.window_size):
                    codon_pos_1based = site_idx - half_win + w_idx
                    if 1 <= codon_pos_1based <= seq_len_codons:
                        nuc_idx = (codon_pos_1based - 1) * 3
                        codon = seq[nuc_idx:nuc_idx+3]
                        msa_tokens[0, s_idx, w_idx] = get_codon_token(codon)
                        
            msa_tokens = msa_tokens.to(device)
            
            # Predict
            logits = model(msa_tokens, dist_tensor)
            prob = torch.sigmoid(logits).item()
            
            # Extract ref codon and translation
            ref_nuc_idx = (site_idx - 1) * 3
            ref_codon = ref_seq[ref_nuc_idx:ref_nuc_idx+3].upper()
            ref_aa = translate_codon(ref_codon)
            
            predictions.append({
                "codon_site": site_idx,
                "ref_codon": ref_codon,
                "ref_aa": ref_aa,
                "selection_probability": round(prob, 4)
            })
            
            if site_idx % 100 == 0:
                print(f"  Processed {site_idx}/{total_codons} sites...")
                
    # 5. Output predictions
    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv(args.output, index=False)
    print(f"\n🎉 Predictions complete! Results saved to '{args.output}'")

if __name__ == "__main__":
    main()
