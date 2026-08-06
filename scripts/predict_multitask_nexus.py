#!/usr/bin/env python3
"""
predict_multitask_nexus.py
--------------------------
Driver script to load a trained 5-head PhyloAxialTransformer selection model,
convert a NEXUS alignment and a NEXUS/Newick tree into model inputs,
and run codon-level multi-task Maximum Likelihood selection predictions.

Outputs predicted MEME parameters per site:
  - LRT_pred: Log-likelihood ratio test statistic
  - p_value_pred: Asymptotic mixture chi-squared p-value (0.5*chi2_1 + 0.5*chi2_2)
  - alpha_pred: Synonymous rate alpha
  - beta_neg_pred: Purifying non-synonymous rate beta^-
  - beta_pos_pred: Positive selection non-synonymous rate beta^+
  - p_neg_pred: Proportion of purifying branches p^-
  - p_pos_pred: Proportion of positive selection branches p^+ (1 - p^-)
  - omega_pos_pred: Positive selection omega ratio (beta^+ / alpha)
  - omega_neg_pred: Purifying omega ratio (beta^- / alpha)

Usage:
  python3 scripts/predict_multitask_nexus.py \
      --alignment msa/A1BG.gz \
      --model check_points/selection_transformer_best.pt \
      --output A1BG_multitask_predictions.csv
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import sys
import gzip
import math
import argparse
import subprocess
from io import StringIO
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import Phylo

# Ensure local scripts directory is on sys.path to import model architecture
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from train_transformer_selection import (
        PhyloAxialTransformer,
        compute_mds_coordinates,
        get_patristic_distances as calculate_patristic_distances,
        parse_newick
    )
except ImportError:
    # Standalone Fallback Definitions if train_transformer_selection is not in path
    print("[!] Warning: Importing from train_transformer_selection failed. Using standalone model definition.")

# =====================================================================
# 1. CODON VOCABULARY & GENETIC CODE DEFINITION
# =====================================================================

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
    'TGC':'C', 'TGT':'C', 'TGG':'W',
}

codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY*")
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}
AA_TO_IDX['-'] = 21
AA_TO_IDX['?'] = 22

def translate_codon(codon):
    codon = codon.upper()
    if len(codon) != 3 or '-' in codon or 'N' in codon or '?' in codon:
        return '?'
    return GENETIC_CODE.get(codon, '?')

def get_codon_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 64
    if len(codon) != 3 or 'N' in codon or '?' in codon:
        return 65
    return CODON_TO_IDX.get(codon, 65)

def get_aa_token(codon):
    aa = translate_codon(codon)
    return AA_TO_IDX.get(aa, 22)

def is_site_variable(site_codons, site_aas):
    clean_codons = [c.upper() for c in site_codons if c.upper() in CODON_TO_IDX and c.upper() not in ['---', 'NNN', '???']]
    clean_aas = [a for a in site_aas if a not in ['?', '*']]
    if len(set(clean_aas)) > 1:
        return True
    if len(set(clean_codons)) > 1:
        return True
    return False

# =====================================================================
# 2. HELPER PARSERS FOR NEXUS ALIGNMENT AND NEWICK TREES
# =====================================================================

def parse_nexus_alignment_and_embedded_tree(filepath):
    taxlabels = []
    sequences = []
    tree_str = None
    
    if filepath.endswith('.gz'):
        f = gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore')
    else:
        f = open(filepath, 'r', encoding='utf-8', errors='ignore')
        
    try:
        lines = f.readlines()
    finally:
        f.close()
        
    in_matrix = False
    in_trees = False
    seq_dict = {}
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('[') or stripped.startswith('#'):
            if 'BEGIN TREES' in line.upper():
                in_trees = True
            elif 'END;' in line.upper():
                in_matrix = False
                in_trees = False
            continue
            
        if 'BEGIN MATRIX' in line.upper() or stripped.upper() == 'MATRIX':
            in_matrix = True
            continue
        elif 'BEGIN TREES' in line.upper():
            in_trees = True
            continue
        elif stripped.upper() == 'END;':
            in_matrix = False
            in_trees = False
            continue
            
        if in_matrix:
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                name = parts[0].replace("'", "").replace('"', '').strip()
                seq = parts[1].replace(" ", "").rstrip(';')
                if name not in seq_dict:
                    seq_dict[name] = seq
                else:
                    seq_dict[name] += seq
        elif in_trees:
            if 'TREE' in stripped.upper() and '=' in stripped:
                parts = stripped.split('=', 1)
                if len(parts) == 2:
                    tree_str = parts[1].strip()
                    
    return seq_dict, taxlabels, tree_str

def parse_nexus_tree_file(tree_filepath):
    if tree_filepath.endswith('.gz'):
        f = gzip.open(tree_filepath, 'rt', encoding='utf-8', errors='ignore')
    else:
        f = open(tree_filepath, 'r', encoding='utf-8', errors='ignore')
        
    try:
        content = f.read()
    finally:
        f.close()
        
    for line in content.splitlines():
        line_s = line.strip()
        if 'TREE' in line_s.upper() and '=' in line_s:
            return line_s.split('=', 1)[1].strip()
            
    return content.strip()

# =====================================================================
# 3. MAIN INFERENCE DRIVER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Task PhyloAxialTransformer MEME Selection Predictor")
    parser.add_argument("--alignment", type=str, required=True, help="Path to NEXUS alignment file (.nex, .nexus, .gz)")
    parser.add_argument("--tree", type=str, default=None, help="Optional separate NEXUS/Newick tree file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PyTorch model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV filepath")
    parser.add_argument("--reference_seq", type=str, default=None, help="Reference species name for codon position numbering")
    parser.add_argument("--window_size", type=int, default=5, help="Transformer sliding window size (default: 5)")
    parser.add_argument("--max_species", type=int, default=256, help="Maximum species per site (default: 256)")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads (default: 8)")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer layers (default: 4)")
    args = parser.parse_args()
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[*] Running inference on: {device}")
    
    # Resolve output name
    out_file = args.output
    if not out_file:
        basename = os.path.basename(args.alignment)
        if basename.endswith('.gz'):
            basename = basename[:-3]
        if basename.endswith('.nex') or basename.endswith('.nexus'):
            basename = basename.rsplit('.', 1)[0]
        out_file = f"{basename}_multitask_predictions.csv"
        
    # Load trained model
    print(f"[*] Loading model checkpoint: {args.model}")
    if not os.path.exists(args.model):
        print(f"[!] Error: Model file not found: {args.model}")
        sys.exit(1)
        
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Auto-detect model shape parameters if present in state_dict
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    window_size = args.window_size

    if 'pos_embedding' in state_dict:
        window_size = state_dict['pos_embedding'].shape[1]
    if 'codon_embed.weight' in state_dict:
        embed_dim = state_dict['codon_embed.weight'].shape[1]
    if 'lrt_head.0.weight' in state_dict:
        fusion_in = state_dict['stream_fusion.0.weight'].shape[1]
        embed_dim = fusion_in // 3
    layer_indices = [int(m.group(1)) for k in state_dict.keys() for m in [re.search(r'layers\.(\d+)\.', k)] if m]
    if layer_indices:
        num_layers = max(layer_indices) + 1

    print(f"[*] Instantiating PhyloAxialTransformer (embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads}, window_size={window_size}, max_species={args.max_species})")

    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=window_size,
        max_species=args.max_species
    ).to(device)
    
    try:
        model.load_state_dict(state_dict)
        print("[*] Model state dict successfully loaded.")
    except Exception as e:
        print(f"[!] Error loading model checkpoint: {e}")
        sys.exit(1)
    model.eval()
    
    # Parse alignment & tree
    seq_dict, taxlabels, embedded_tree_str = parse_nexus_alignment_and_embedded_tree(args.alignment)
    if not seq_dict:
        print(f"[!] Error parsing alignment: {args.alignment}")
        sys.exit(1)
    print(f"[*] Alignment parsed successfully: {len(seq_dict)} species sequences.")
    
    tree_str = None
    if args.tree:
        tree_str = parse_nexus_tree_file(args.tree)
    else:
        tree_str = embedded_tree_str
        
    tree_obj = None
    species_names = []
    if tree_str:
        try:
            clean_tree_str = re.sub(r'\{[^}]*\}', '', tree_str)
            clean_tree_str = re.sub(r'\[.*?\]', '', clean_tree_str)
            clean_tree_str = clean_tree_str.strip().rstrip(';') + ';'
            tree_obj = Phylo.read(StringIO(clean_tree_str), 'newick')
            species_names = [leaf.name for leaf in tree_obj.get_terminals() if leaf.name]
        except Exception as e:
            print(f"[!] Warning: Tree parsing failed: {e}")
            
    # Select reference sequence
    if args.reference_seq:
        ref_key = args.reference_seq
    else:
        heuristics = ['hg', 'hg38', 'human', next(iter(seq_dict.keys()))]
        ref_key = next((h for h in heuristics if h in seq_dict), next(iter(seq_dict.keys())))
    print(f"[*] Reference species: {ref_key}")
    
    if not species_names:
        species_names = list(seq_dict.keys())
        
    name_map = {name.replace("'", "").replace('"', '').strip(): name for name in seq_dict.keys()}
    matching_species = [name_map[t.replace("'", "").replace('"', '').strip()] for t in species_names if t.replace("'", "").replace('"', '').strip() in name_map]
    
    if not matching_species:
        matching_species = list(seq_dict.keys())
        
    if ref_key in matching_species:
        matching_species.remove(ref_key)
        matching_species.insert(0, ref_key)
    else:
        matching_species.insert(0, ref_key)
        
    selected_species = matching_species[:args.max_species]
    num_selected = len(selected_species)
    print(f"[*] Analyzing {num_selected} species sequences.")
    
    # Handle patristic distances
    dist_matrix = {}
    if tree_obj:
        try:
            _, dist_matrix = calculate_patristic_distances(tree_obj)
        except Exception as e:
            print(f"[!] Patristic calculation failed: {e}")
            
    if not dist_matrix:
        dist_matrix = {s1: {s2: 0.0 for s2 in selected_species} for s1 in selected_species}
        
    ref_seq = seq_dict[ref_key]
    total_codons = len(ref_seq) // 3
    
    # Populate input tensors
    msa_tokens = torch.ones(total_codons, num_selected, args.window_size, dtype=torch.long) * 65
    aa_tokens = torch.ones(total_codons, num_selected, args.window_size, dtype=torch.long) * AA_TO_IDX['?']
    dist_tensor = torch.zeros(num_selected, num_selected, dtype=torch.float32)
    padding_mask = torch.zeros(num_selected, dtype=torch.bool)
    
    for i, spec1 in enumerate(selected_species):
        for j, spec2 in enumerate(selected_species):
            norm1 = spec1.replace("'", "").replace('"', '').strip()
            norm2 = spec2.replace("'", "").replace('"', '').strip()
            dist_tensor[i, j] = dist_matrix.get(norm1, {}).get(norm2, 0.0)
            
    dist_np = dist_tensor.numpy()
    mds_coords_np = compute_mds_coordinates(dist_np, n_components=4)
    mds_coords_tensor = torch.from_numpy(mds_coords_np).float()
    
    variable_sites_flags = []
    half_win = args.window_size // 2
    for site_idx in range(1, total_codons + 1):
        site_codons = []
        site_aas = []
        for s_idx, spec in enumerate(selected_species):
            seq = seq_dict.get(spec, "")
            seq_len_codons = len(seq) // 3
            for w_idx in range(args.window_size):
                codon_pos_1based = site_idx - half_win + w_idx
                if 1 <= codon_pos_1based <= seq_len_codons:
                    nuc_idx = (codon_pos_1based - 1) * 3
                    codon = seq[nuc_idx:nuc_idx+3]
                    msa_tokens[site_idx - 1, s_idx, w_idx] = get_codon_token(codon)
                    aa_tokens[site_idx - 1, s_idx, w_idx] = get_aa_token(codon)
            if 1 <= site_idx <= seq_len_codons:
                nuc_idx = (site_idx - 1) * 3
                codon = seq[nuc_idx:nuc_idx+3].upper()
                if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                    site_codons.append(codon)
                    aa = GENETIC_CODE.get(codon, '?')
                    if aa != '?':
                        site_aas.append(aa)
        variable_sites_flags.append(is_site_variable(site_codons, site_aas))
        
    msa_tokens = msa_tokens.to(device)
    aa_tokens = aa_tokens.to(device)
    dist_tensor = dist_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
    mds_coords_tensor = mds_coords_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
    padding_mask_tensor = padding_mask.unsqueeze(0).expand(total_codons, -1).to(device)
    
    # Batched Inference Pass
    print(f"[*] Running multi-task batched inference pass over {total_codons} codon sites...")
    with torch.no_grad():
        # Evaluate model in site batches if alignment is long
        batch_step = 64
        pred_lrts, pred_alphas, pred_beta_negs, pred_beta_poses, pred_p_negs = [], [], [], [], []
        
        for b_start in range(0, total_codons, batch_step):
            b_end = min(b_start + batch_step, total_codons)
            out_lrt, out_alpha, out_bneg, out_bpos, out_pneg = model(
                msa_tokens[b_start:b_end],
                aa_tokens[b_start:b_end],
                dist_tensor[b_start:b_end],
                mds_coords_tensor[b_start:b_end],
                padding_mask_tensor[b_start:b_end]
            )
            pred_lrts.extend(out_lrt.cpu().numpy().tolist())
            pred_alphas.extend(out_alpha.cpu().numpy().tolist())
            pred_beta_negs.extend(out_bneg.cpu().numpy().tolist())
            pred_beta_poses.extend(out_bpos.cpu().numpy().tolist())
            pred_p_negs.extend(out_pneg.cpu().numpy().tolist())
            
    # Collect predictions
    results = []
    sig_count = 0
    
    for site_idx in range(1, total_codons + 1):
        ref_nuc_idx = (site_idx - 1) * 3
        ref_codon = ref_seq[ref_nuc_idx:ref_nuc_idx+3].upper()
        ref_aa = translate_codon(ref_codon)
        is_var = variable_sites_flags[site_idx - 1]
        
        if is_var:
            lrt_log = float(pred_lrts[site_idx - 1])
            lrt = float(np.expm1(lrt_log))
            alpha = float(np.expm1(pred_alphas[site_idx - 1]))
            beta_neg = float(np.expm1(pred_beta_negs[site_idx - 1]))
            beta_pos = float(np.expm1(pred_beta_poses[site_idx - 1]))
            p_neg = float(pred_p_negs[site_idx - 1])
            p_pos = max(0.0, min(1.0, 1.0 - p_neg))
            
            # Asymptotic mixture chi-squared p-value: 0.5*chi2(df=1) + 0.5*chi2(df=2)
            p_val = 0.5 * stats.chi2.sf(lrt, df=1) + 0.5 * stats.chi2.sf(lrt, df=2) if lrt > 0 else 1.0
        else:
            lrt_log = 0.0
            lrt = 0.0
            p_val = 1.0
            alpha = 1.0
            beta_neg = 1.0
            beta_pos = 1.0
            p_neg = 1.0
            p_pos = 0.0
            
        omega_pos = beta_pos / max(1e-4, alpha)
        omega_neg = beta_neg / max(1e-4, alpha)
        
        if p_val <= 0.10 and is_var:
            sig_count += 1
            
        results.append({
            "site": site_idx,
            "ref_codon": ref_codon,
            "ref_aa": ref_aa,
            "is_variable": is_var,
            "LRT_direct_pred": round(lrt, 4),
            "LRT_log_pred": round(lrt_log, 4),
            "p_value_pred": round(p_val, 6),
            "alpha_pred": round(alpha, 4),
            "beta_neg_pred": round(beta_neg, 4),
            "beta_pos_pred": round(beta_pos, 4),
            "p_neg_pred": round(p_neg, 4),
            "p_pos_pred": round(p_pos, 4),
            "omega_pos_pred": round(omega_pos, 4),
            "omega_neg_pred": round(omega_neg, 4)
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_file, index=False)
    print(f"\n==========================================================================")
    print(f"🎉 PREDICTIONS COMPLETE: Saved results to '{out_file}'")
    print(f"   - Total Codon Sites: {total_codons}")
    print(f"   - Variable Sites: {sum(variable_sites_flags)}")
    print(f"   - Predicted Episodic Selection Sites (p <= 0.10): {sig_count}")
    print(f"==========================================================================")

if __name__ == "__main__":
    main()
