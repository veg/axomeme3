#!/usr/bin/env python3
"""
predict_regression_nexus.py
--------------------------
Driver script to load a trained continuous selection PhyloAxialTransformer model,
convert a NEXUS alignment and a NEXUS tree (embedded or separate) into model inputs,
and run codon-level Likelihood Ratio Test (LRT) statistic predictions.

Requirements:
  - Python 3
  - PyTorch
  - pandas
  - Biopython (for robust phylogenetic tree parsing and patristic distance calculations)

Usage:
  python3 predict_regression_nexus.py \
      --alignment msa/A1BG.gz \
      --model /Users/sergei/Documents/MEME_transformer_joint.pt \
      --output A1BG_predictions.csv
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import sys
import subprocess
import gzip
import math
import argparse
from io import StringIO
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import Phylo
import math

def select_species_maximize_pd(tree_obj, matching_species, ref_key, max_species):
    # Find all leaf clades in the tree
    terminals = {t.name: t for t in tree_obj.get_terminals() if t.name}
    
    # Check if tree has valid non-zero branch lengths
    has_branch_lengths = any(
        clade.branch_length is not None and clade.branch_length > 0.0 
        for clade in tree_obj.find_clades() 
        if clade != tree_obj.root
    )
    
    # Helper to get weight of a clade
    def get_weight(clade):
        if not has_branch_lengths:
            return 1.0
        return clade.branch_length if (clade.branch_length is not None and not math.isnan(clade.branch_length)) else 0.0

    # Build paths from root to each leaf node
    paths = {}
    for name, terminal in terminals.items():
        path = [tree_obj.root] + tree_obj.get_path(terminal)
        paths[name] = path

    selected_nodes = set()
    selected_species = [ref_key]
    
    norm_ref_key = ref_key.replace("'", "").replace('"', '').strip()
    
    # Pre-populate selected_nodes with the path from root to ref_key
    ref_matched_name = None
    for name in paths:
        if name.replace("'", "").replace('"', '').strip() == norm_ref_key:
            ref_matched_name = name
            break
            
    if ref_matched_name and ref_matched_name in paths:
        for node in paths[ref_matched_name]:
            selected_nodes.add(node)
    else:
        selected_nodes.add(tree_obj.root)

    # Filter out reference key from remaining candidates
    candidates = [
        name for name in matching_species 
        if name.replace("'", "").replace('"', '').strip() != norm_ref_key
    ]
    
    target_count = min(max_species, len(matching_species))
    
    # Greedy loop
    while len(selected_species) < target_count and candidates:
        max_dist = -1.0
        best_idx = -1
        
        # Calculate distance for each candidate
        for idx, name in enumerate(candidates):
            norm_name = name.replace("'", "").replace('"', '').strip()
            # Find matching leaf in tree paths
            matched_name = None
            for p_name in paths:
                if p_name.replace("'", "").replace('"', '').strip() == norm_name:
                    matched_name = p_name
                    break
                    
            dist = 0.0
            if matched_name and matched_name in paths:
                path = paths[matched_name]
                for node in reversed(path):
                    if node in selected_nodes:
                        break
                    dist += get_weight(node)
            
            if dist > max_dist:
                max_dist = dist
                best_idx = idx
                
        if best_idx > -1:
            best_name = candidates[best_idx]
            selected_species.append(best_name)
            candidates.pop(best_idx)
            
            # Add path of the chosen leaf to the selected nodes
            norm_best = best_name.replace("'", "").replace('"', '').strip()
            best_matched = None
            for p_name in paths:
                if p_name.replace("'", "").replace('"', '').strip() == norm_best:
                    best_matched = p_name
                    break
                    
            if best_matched and best_matched in paths:
                for node in paths[best_matched]:
                    selected_nodes.add(node)
        else:
            break
            
    return selected_species

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

# 64 codons + 1 gap token (64) + 1 unknown/missing token (65) = 66 tokens total
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

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

AA_LIST = "ACDEFGHIKLMNPQRSTVWY*-?"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def get_aa_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 21  # AA_TO_IDX['-']
    if len(codon) != 3 or 'N' in codon or '?' in codon:
        return 22  # AA_TO_IDX['?']
    aa = GENETIC_CODE.get(codon, '?')
    return AA_TO_IDX.get(aa, 22)

# --- BLOSUM62 & Grantham Scoring Matrices ---
_blosum_raw = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1 -2 -2  0 -3 -3  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1 -2  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -3 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

_grantham_raw = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  0 112 111 126  44  91 107  60  86  94  96 106  84  95  27  35  58 148 112  64
R 112  0  86 138 102  43  97 125  29  97 102  26  91  97 103 110  71 101  77  96
N 111  86  0  23 139  46  42  80  68 149 143  94 142 158  91  46  65 174 143 133
D 126 138  23  0 154  61  45  94  81 168 162 101 160 177 108  54  85 181 162 152
C  44 102 139 154  0 116 126 117 118 117 121 112 118 135  74  80 101 190 154 109
Q  91  43  46  61 116  0  29  87  24  93  99  53  81  93  76  68  47 130  84  96
E 107  97  42  45 126  29  0  98  40 103 107  56  87 102  93  80  65 122  86  96
G  60 125  80  94 117  87  98  0  98 135 127 127 127 153  42  56  59 184 147 109
H  86  29  68  81 118  24  40  98  0  99 105  32  87  92  77  83  47 115  83  98
I  94  97 149 168 117  93 103 135  99  0  10  97  10  21  95 142 124 103  83  29
L  96 102 143 162 121  99 107 127 105  10  0 107  15  22  95 145 130 113  92  32
K 106  26  94 101 112  53  56 127  32 97 107  0  95 102 103 121  78 110  85  97
M  84  91 142 160 118  81  87 127  87  10  15  95  0  28  87 135 121 115  95  21
F  95  97 158 177 135  93 102 153  92  21  22 102  28  0 110 155 140  40  22  50
P  27 103  91 108  74  76  93  42  77  95  95 103  87 110  0  56  74 153 110  76
S  35 110  46  54  80  68  80  56  83 142 145 121 135 155  56  0  58 177 144 124
T  58  71  65  85 101  47  65  59  47 124 130  78 121 140  74  58  0 178 134 103
W 148 101 174 181 190 130 122 184 115 103 113 110 115  40 153 177 178  0  37  88
Y 112  77 143 162 154  84  86 147  83  83  92  85  95  22 110 144 134  37  0  90
V  64  96 133 152 109  96  96 109  98  29  32  97  21  50  76 124 103  88  90  0
"""

def parse_scoring_matrices():
    lines = [line.strip().split() for line in _blosum_raw.strip().split('\n')]
    headers = lines[0]
    blosum_dict = {}
    for row in lines[1:]:
        key = row[0]
        blosum_dict[key] = {headers[i]: int(row[i+1]) for i in range(len(headers))}
        
    lines = [line.strip().split() for line in _grantham_raw.strip().split('\n')]
    headers = lines[0]
    grantham_dict = {}
    for row in lines[1:]:
        key = row[0]
        grantham_dict[key] = {headers[i]: float(row[i+1]) for i in range(len(headers))}
    return blosum_dict, grantham_dict

BLOSUM62, GRANTHAM = parse_scoring_matrices()

def compute_mds_coordinates(dist_matrix_np, n_components=4):
    N = dist_matrix_np.shape[0]
    if N <= n_components:
        coords = np.zeros((N, n_components), dtype=np.float32)
        return coords
    D2 = dist_matrix_np ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (H @ D2 @ H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Enforce sign convention: largest absolute value element is positive
    for col in range(evecs.shape[1]):
        max_abs_idx = np.argmax(np.abs(evecs[:, col]))
        sign = np.sign(evecs[max_abs_idx, col])
        if sign < 0:
            evecs[:, col] *= -1.0
            
    coords = np.zeros((N, n_components), dtype=np.float32)
    for i in range(n_components):
        val = evals[i]
        if val > 0:
            coords[:, i] = evecs[:, i] * np.sqrt(val)
    return coords

def is_site_variable(site_codons, site_aas):
    if not site_codons or not site_aas:
        return False
        
    unique_aas_set = set(site_aas)
    
    # Condition 1: Multiple amino acids
    if len(unique_aas_set) > 1:
        return True
        
    # Condition 2: Serine Island transition (synonymous but selection-relevant)
    if len(unique_aas_set) == 1 and 'S' in unique_aas_set:
        has_tcn = any(c in ('TCA', 'TCC', 'TCG', 'TCT') for c in site_codons)
        has_agy = any(c in ('AGC', 'AGT') for c in site_codons)
        if has_tcn and has_agy:
            return True
            
    return False

# =====================================================================
# 2. MODEL ARCHITECTURE (PhyloAxialTransformer)
# =====================================================================

class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.phylo_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix, padding_mask=None):
        batch_size, num_species, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Guide attention using patristic distances
        dist_guide = dist_matrix.unsqueeze(1)
        scaled_scale = torch.exp(self.phylo_scale)
        guided_bias = -scaled_scale * dist_guide
        
        scores = scores + guided_bias
        
        if padding_mask is not None:
            # padding_mask shape: [batch_size, num_species]
            # scores shape: [batch_size, num_heads, num_species, num_species]
            mask = padding_mask.unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, num_species]
            scores = scores.masked_fill(mask, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        
        # Relation-aware value scaling (soft-thresholded log-distance weight)
        epsilon = 0.05
        log_dist_weight = torch.log1p(1.0 / (dist_matrix.unsqueeze(1) + epsilon))
        attn = attn * log_dist_weight
        
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        return self.out_proj(out)


# --- Custom Multihead Attention to bypass Apple MPS bugs ---
class StableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_seq_len, _ = query.shape
        k_seq_len = key.shape[1]
        
        w_q, w_k, w_v = torch.chunk(self.in_proj_weight, 3, dim=0)
        b_q, b_k, b_v = torch.chunk(self.in_proj_bias, 3, dim=0)
        
        w_q, w_k, w_v = w_q.contiguous(), w_k.contiguous(), w_v.contiguous()
        b_q, b_k, b_v = b_q.contiguous(), b_k.contiguous(), b_v.contiguous()
        
        q_proj = F.linear(query, w_q, b_q)
        k_proj = F.linear(key, w_k, b_k)
        v_proj = F.linear(value, w_v, b_v)
        
        q_h = q_proj.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if key_padding_mask is not None:
            # key_padding_mask shape: [batch_size, k_seq_len]
            # scores shape: [batch_size, num_heads, q_seq_len, k_seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, k_seq_len]
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)
        return self.out_proj(out)


# --- Custom Stable Transformer Encoder Layer ---
class StableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = StableAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_out))
        
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src


class PhyloAxialTransformer(nn.Module):
    def __init__(self, num_tokens=66, embed_dim=128, num_heads=8, num_layers=4, window_size=1, max_species=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_species = max_species
        
        # Split embed_dim between codon and amino acid embeddings
        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)  # 23 amino acid tokens
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.mds_proj = nn.Linear(4, embed_dim)
        
        self.col_layers = nn.ModuleList([
            StableTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2*embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pool_attn = StableAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, msa_codons, msa_aas, dist_matrix, mds_coords, padding_mask=None):
        batch_size, num_species, window_size = msa_codons.shape
        
        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        
        # Project and add MDS coordinates (phylogenetic positional embedding)
        phylo_pos = self.mds_proj(mds_coords) # [batch_size, num_species, embed_dim]
        x = x + phylo_pos.unsqueeze(2) # [batch_size, num_species, window_size, embed_dim]
        
        for i in range(len(self.col_layers)):
            col_in = x.reshape(batch_size * num_species, window_size, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x = col_out.reshape(batch_size, num_species, window_size, self.embed_dim)
            
            row_in = x.transpose(1, 2).contiguous().view(batch_size * window_size, num_species, self.embed_dim)
            dist_dup = dist_matrix.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species)
            
            if padding_mask is not None:
                padding_mask_dup = padding_mask.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_species)
            else:
                padding_mask_dup = None
                
            row_out = self.row_layers[i](row_in, dist_dup, padding_mask_dup)
            row_out = self.row_norms[i](row_in + row_out)
            x = row_out.reshape(batch_size, window_size, num_species, self.embed_dim).transpose(1, 2)
            
        central_idx = window_size // 2
        site_repr = x[:, :, central_idx, :]
        
        q = self.pool_query.expand(batch_size, -1, -1).contiguous()
        pooled_repr = self.pool_attn(q, site_repr, site_repr, key_padding_mask=padding_mask)
        pooled_repr = pooled_repr.squeeze(1)
        
        logits = self.mlp(pooled_repr)
        return logits.squeeze(1)

# =====================================================================
# 3. NEXUS ALIGNMENT AND TREE PARSERS
# =====================================================================

def parse_nexus_alignment_and_embedded_tree(filepath):
    taxlabels = []
    matrix_lines = []
    tree_str = None
    in_taxlabels = False
    in_matrix = False
    in_trees = False
    
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
            
            line_strip = re.sub(r'\[.*?\]', '', line_strip).strip()
            if not line_strip:
                continue
            
            if line_strip.upper().startswith('TAXLABELS'):
                in_taxlabels = True
                content = line_strip[len('TAXLABELS'):].strip()
                tokens = content.replace("'", "").replace('"', '').replace(';', '').split()
                taxlabels.extend(tokens)
                if line_strip.endswith(';'):
                    in_taxlabels = False
                continue
            if in_taxlabels:
                tokens = line_strip.replace("'", "").replace('"', '').replace(';', '').split()
                taxlabels.extend(tokens)
                if line_strip.endswith(';'):
                    in_taxlabels = False
                continue
                
            if line_strip.upper().startswith('MATRIX'):
                in_matrix = True
                continue
            if in_matrix:
                if line_strip == ';':
                    in_matrix = False
                    continue
                if line_strip.endswith(';'):
                    matrix_lines.append(line_strip[:-1].strip())
                    in_matrix = False
                    continue
                matrix_lines.append(line_strip)
                continue
                
            if line_strip.upper().startswith('BEGIN TREES') or line_strip.upper().startswith('BEGIN TREE'):
                in_trees = True
                continue
            if in_trees:
                if line_strip.upper().startswith('TREE '):
                    parts = line_strip.split('=', 1)
                    if len(parts) > 1:
                        tree_str = parts[1].strip()
                if line_strip.upper().startswith('END;'):
                    in_trees = False
                    continue
                    
    seq_dict = {}
    for i, seq_line in enumerate(matrix_lines):
        if not seq_line:
            continue
        parts = seq_line.split(None, 1)
        if len(parts) == 2 and (parts[0] in taxlabels or parts[0].replace("'", "").replace('"', '') in taxlabels):
            label = parts[0].replace("'", "").replace('"', '')
            seq = parts[1].replace(' ', '').replace('\t', '')
            seq_dict[label] = seq
        else:
            if i < len(taxlabels):
                label = taxlabels[i]
                seq = seq_line.replace(' ', '').replace('\t', '')
                seq_dict[label] = seq
                
    return seq_dict, taxlabels, tree_str


def parse_nexus_tree_file(filepath):
    tree_str = None
    in_trees = False
    
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        first_line = f.readline()
        f.seek(0)
        
        if '#NEXUS' in first_line.upper():
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                line_strip = re.sub(r'\[.*?\]', '', line_strip).strip()
                if line_strip.upper().startswith('BEGIN TREES') or line_strip.upper().startswith('BEGIN TREE'):
                    in_trees = True
                    continue
                if in_trees:
                    if line_strip.upper().startswith('TREE '):
                        parts = line_strip.split('=', 1)
                        if len(parts) > 1:
                            tree_str = parts[1].strip()
                    if line_strip.upper().startswith('END;'):
                        in_trees = False
        else:
            content = f.read().strip()
            match = re.search(r'\(.*\);?', content)
            if match:
                tree_str = match.group(0)
                
    if tree_str and tree_str.endswith(';'):
        tree_str = tree_str[:-1]
    return tree_str


def calculate_patristic_distances(tree):
    node_to_root_dist = {}
    node_to_parent = {}
    
    def traverse(node, current_dist, parent):
        node_to_root_dist[node] = current_dist
        node_to_parent[node] = parent
        for child in node.clades:
            traverse(child, current_dist + (child.branch_length or 0.0), node)
            
    traverse(tree.root, 0.0, None)
    
    leaves = tree.get_terminals()
    leaf_names = [leaf.name for leaf in leaves if leaf.name]
    leaf_by_name = {leaf.name: leaf for leaf in leaves if leaf.name}
    
    leaf_paths = {}
    for leaf in leaves:
        if not leaf.name:
            continue
        path = []
        curr = leaf
        while curr is not None:
            path.append(curr)
            curr = node_to_parent[curr]
        leaf_paths[leaf.name] = path
        
    dist_matrix = {}
    for name in leaf_names:
        dist_matrix[name] = {name: 0.0}
        
    n = len(leaf_names)
    for i in range(n):
        name1 = leaf_names[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, n):
            name2 = leaf_names[j]
            path2 = leaf_paths[name2]
            
            lca = None
            for node in path2:
                if node in set1:
                    lca = node
                    break
                    
            if lca is not None:
                dist = node_to_root_dist[leaf_by_name[name1]] + node_to_root_dist[leaf_by_name[name2]] - 2 * node_to_root_dist[lca]
            else:
                dist = node_to_root_dist[leaf_by_name[name1]] + node_to_root_dist[leaf_by_name[name2]]
                
            dist_matrix[name1][name2] = dist
            dist_matrix[name2][name1] = dist
            
    return leaf_names, dist_matrix

# =====================================================================
# 4. MAIN INFERENCE DRIVER PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a NEXUS alignment and a NEXUS/Newick tree into model inputs, "
                    "and return codon-level selection strength (LRT) predictions using a trained regression transformer."
    )
    parser.add_argument("--alignment", required=True, help="Path to NEXUS alignment file (.gz or uncompressed)")
    parser.add_argument("--tree", help="Path to NEXUS or Newick tree file. If omitted, will try to read from the alignment.")
    parser.add_argument("--model", default="/Users/sergei/Documents/MEME_transformer_joint.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--output", help="Path to output predictions CSV. Defaults to [alignment_prefix]_regression_predictions.csv")
    parser.add_argument("--reference_seq", help="Name of reference sequence (e.g. hg, hg38). Defaults to first sequence.")
    parser.add_argument("--window_size", type=int, default=1, help="Alignment sliding window size centered at site (default: 1)")
    parser.add_argument("--max_species", type=int, default=256, help="Maximum number of sequences to feed to model (default: 256)")
    parser.add_argument("--device", help="Force run on device (cpu, mps, cuda). Auto-detected by default.")
    parser.add_argument("--tier1_percentile", type=float, default=98.0, help="Percentile threshold for Tier 1 High-Confidence calls (default: 98.0)")
    parser.add_argument("--tier2_percentile", type=float, default=97.0, help="Percentile threshold for Tier 2 Medium-Confidence calls (default: 97.0)")
    parser.add_argument("--use_zscore", action="store_true", help="Use Z-score thresholds instead of percentiles for calling tiers")
    parser.add_argument("--tier1_zscore", type=float, default=2.5, help="Z-score threshold for Tier 1 High-Confidence calls (default: 2.5)")
    parser.add_argument("--tier2_zscore", type=float, default=2.0, help="Z-score threshold for Tier 2 Medium-Confidence calls (default: 2.0)")
    parser.add_argument("--tier1_lrt_gate", type=float, default=5.0, help="Absolute predicted LRT gate for Tier 1 calls (default: 5.0, set negative to disable)")
    parser.add_argument("--tier2_lrt_gate", type=float, default=3.0, help="Absolute predicted LRT gate for Tier 2 calls (default: 3.0, set negative to disable)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Legacy: LRT threshold to report sites to stdout (default: 5.0)")
    parser.add_argument("--z_threshold", type=float, default=2.5, help="Legacy: Local Z-score threshold to report sites to stdout (default: 2.5)")
    
    args = parser.parse_args()
    
    # 1. Device Setup
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[*] Running inference on device: {device}")
    
    # 2. Parse NEXUS alignment
    print(f"[*] Parsing NEXUS alignment from: {args.alignment}")
    if not os.path.exists(args.alignment):
        print(f"[!] Error: Alignment file not found: {args.alignment}")
        sys.exit(1)
        
    seq_dict, taxlabels, embedded_tree_str = parse_nexus_alignment_and_embedded_tree(args.alignment)
    if not seq_dict:
        print("[!] Error: No sequences successfully parsed from NEXUS matrix.")
        sys.exit(1)
    print(f"[*] Alignment loaded: {len(seq_dict)} sequences, alignment length = {len(next(iter(seq_dict.values())))} nucleotides.")
    
    # 3. Parse Tree (embedded or separate)
    tree_str = None
    if args.tree:
        print(f"[*] Parsing separate tree file: {args.tree}")
        if not os.path.exists(args.tree):
            print(f"[!] Error: Tree file not found: {args.tree}")
            sys.exit(1)
        tree_str = parse_nexus_tree_file(args.tree)
    else:
        if embedded_tree_str:
            print("[*] Using tree embedded in the NEXUS alignment file.")
            tree_str = embedded_tree_str
        else:
            print("[!] Warning: No separate tree file provided and no embedded tree found in alignment.")
            
    # Load and clean tree string into Bio.Phylo tree object
    tree_obj = None
    species_names = []
    if tree_str:
        try:
            clean_tree_str = re.sub(r'\{[^}]*\}', '', tree_str)
            clean_tree_str = re.sub(r'\[.*?\]', '', clean_tree_str)
            clean_tree_str = clean_tree_str.strip().rstrip(';') + ';'
            tree_obj = Phylo.read(StringIO(clean_tree_str), 'newick')
            species_names = [leaf.name for leaf in tree_obj.get_terminals() if leaf.name]
            print(f"[*] Successfully parsed tree topology with {len(species_names)} leaves.")
        except Exception as e:
            print(f"[!] Warning: Tree parsing failed: {e}")
            
    # 4. Determine Reference Sequence and Selection of Species
    if args.reference_seq:
        ref_key = args.reference_seq
        if ref_key not in seq_dict:
            print(f"[!] Error: Reference sequence '{ref_key}' not found in alignment.")
            print(f"Available sequences: {list(seq_dict.keys())[:10]}...")
            sys.exit(1)
    else:
        heuristics = ['hg', 'hg38', 'human', next(iter(seq_dict.keys()))]
        ref_key = next((h for h in heuristics if h in seq_dict), next(iter(seq_dict.keys())))
        
    print(f"[*] Reference sequence selected: '{ref_key}'")
    
    has_matching_tree = False
    if tree_obj and species_names:
        name_map = {}
        for align_name in seq_dict.keys():
            norm = align_name.replace("'", "").replace('"', '').strip()
            name_map[norm] = align_name
            
        matching_species = []
        for tree_name in species_names:
            norm_t = tree_name.replace("'", "").replace('"', '').strip()
            if norm_t in name_map:
                matching_species.append(name_map[norm_t])
                
        print(f"[*] Matched {len(matching_species)} species between the tree and alignment.")
        if matching_species:
            has_matching_tree = True
        else:
            print("[!] Warning: Zero matching species between tree and alignment. Falling back to alignment order.")
            matching_species = list(seq_dict.keys())
    else:
        matching_species = list(seq_dict.keys())
        
    selected_species = []
    if has_matching_tree:
        selected_species = select_species_maximize_pd(tree_obj, matching_species, ref_key, args.max_species)
    else:
        # Move reference key to the front of matched list
        if ref_key in matching_species:
            matching_species.remove(ref_key)
            
        target_count = args.max_species - 1
        if len(matching_species) <= target_count:
            selected_species = list(matching_species)
        elif target_count > 0:
            for i in range(target_count):
                idx = int(i * (len(matching_species) - 1) / (target_count - 1))
                selected_species.append(matching_species[idx])
        selected_species.insert(0, ref_key)
        
    num_selected = len(selected_species)
    print(f"[*] Final model input selection: {num_selected} species (max_species cap = {args.max_species})")
    
    # 4.1. Estimate branch lengths using HyPhy if they are missing or all zero
    dist_matrix = {}
    if tree_obj:
        has_branch_lengths = any(
            clade.branch_length is not None and clade.branch_length > 0.0 
            for clade in tree_obj.find_clades() 
            if clade != tree_obj.root
        )
        
        if not has_branch_lengths:
            print("[!] Tree topology has no branch lengths. Running HyPhy to estimate branch lengths...")
            try:
                # Prune tree to only include selected_species
                common_set = set(selected_species)
                for leaf in tree_obj.get_terminals():
                    if leaf.name not in common_set:
                        tree_obj.prune(leaf)
                        
                # Format pruned tree topology Newick string (removing any placeholder/0.0 branch lengths)
                out_stream = StringIO()
                Phylo.write(tree_obj, out_stream, 'newick')
                raw_pruned_tree_str = out_stream.getvalue().strip()
                pruned_tree_str = re.sub(r':[0-9.eE-]+', '', raw_pruned_tree_str)
                
                # Write temporary alignment FASTA containing only selected species
                os.makedirs("scratch", exist_ok=True)
                temp_fasta = "scratch/temp_hyphy_align.fa"
                with open(temp_fasta, "w") as f:
                    for spec in selected_species:
                        f.write(f">{spec}\n{seq_dict[spec]}\n")
                
                # Write temporary HyPhy batch script
                temp_bf = "scratch/temp_hyphy_est.bf"
                bf_content = f"""
DataSet ds = ReadDataFile("{temp_fasta}");
DataSetFilter df = CreateFilter(ds, 1);
HarvestFrequencies(freqs, df, 1, 1, 1);
global kappa = 1.0;
HKY85RateMatrix = [
    [*, kappa*t, t, kappa*t]
    [kappa*t, *, kappa*t, t]
    [t, kappa*t, *, kappa*t]
    [kappa*t, t, kappa*t, *]
];
Model HKY85Model = (HKY85RateMatrix, freqs);
UseModel(HKY85Model);
Tree T = "{pruned_tree_str}";
LikelihoodFunction lf = (df, T);
Optimize(res, lf);
fprintf(stdout, Format(T, 1, 1));
"""
                with open(temp_bf, "w") as f:
                    f.write(bf_content)
                
                # Execute HyPhy
                res = subprocess.run(["hyphy", temp_bf], capture_output=True, text=True)
                
                # Clean up temporary files
                if os.path.exists(temp_fasta):
                    os.remove(temp_fasta)
                if os.path.exists(temp_bf):
                    os.remove(temp_bf)
                
                if res.returncode == 0 and res.stdout.strip():
                    estimated_tree_str = res.stdout.strip()
                    # Parse the tree with estimated branch lengths
                    tree_obj = Phylo.read(StringIO(estimated_tree_str), 'newick')
                    print("[*] HyPhy branch length estimation succeeded.")
                    
                    # Save the estimated tree next to the predictions CSV output
                    out_file = args.output
                    if not out_file:
                        basename = os.path.basename(args.alignment)
                        if basename.endswith('.gz'):
                            basename = basename[:-3]
                        if basename.endswith('.nex') or basename.endswith('.nexus'):
                            basename = basename.rsplit('.', 1)[0]
                        out_file = f"{basename}_regression_predictions.csv"
                    
                    tree_out_path = out_file.rsplit('.', 1)[0] + "_estimated_tree.nwk"
                    with open(tree_out_path, "w") as f:
                        f.write(estimated_tree_str + ";\n")
                    print(f"[*] Saved estimated tree to '{tree_out_path}'")
                else:
                    print(f"[!] HyPhy estimation failed (code {res.returncode}). Stderr: {res.stderr}")
                    print("[*] Falling back to flat evolutionary distance structure.")
            except Exception as ex:
                print(f"[!] Error running HyPhy branch length estimation: {ex}")
                print("[*] Falling back to flat evolutionary distance structure.")
                
        try:
            species_names, dist_matrix = calculate_patristic_distances(tree_obj)
        except Exception as e:
            print(f"[!] Error calculating patristic distances: {e}")
            dist_matrix = {}
            
    if not dist_matrix:
        dist_matrix = {s1: {s2: 0.0 for s2 in selected_species} for s1 in selected_species}
        
    # 5. Build input tensors for model
    ref_seq = seq_dict[ref_key]
    if len(ref_seq) % 3 != 0:
        print(f"[!] Warning: Reference sequence length ({len(ref_seq)} nucs) is not a multiple of 3. Truncating tail.")
    total_codons = len(ref_seq) // 3
    print(f"[*] Reference sequence '{ref_key}' length: {total_codons} codons.")
    
    msa_tokens = torch.ones(total_codons, args.max_species, args.window_size, dtype=torch.long) * 65
    aa_tokens = torch.ones(total_codons, args.max_species, args.window_size, dtype=torch.long) * AA_TO_IDX['?']
    dist_tensor = torch.zeros(args.max_species, args.max_species)
    padding_mask = torch.ones(args.max_species, dtype=torch.bool) # True means padded
    
    for i, spec1 in enumerate(selected_species):
        padding_mask[i] = False
        for j, spec2 in enumerate(selected_species):
            norm1 = spec1.replace("'", "").replace('"', '').strip()
            norm2 = spec2.replace("'", "").replace('"', '').strip()
            dist_tensor[i, j] = dist_matrix.get(norm1, {}).get(norm2, 0.0)
            
    # Compute MDS coordinates on the distance matrix
    dist_np = dist_tensor.numpy()
    mds_coords_np = compute_mds_coordinates(dist_np, n_components=4)
    mds_coords_tensor = torch.from_numpy(mds_coords_np) # [max_species, 4]
            
    variable_sites_flags = []
    
    half_win = args.window_size // 2
    for site_idx in range(1, total_codons + 1):
        site_codons = []
        site_aas = []
        spec_aas = []
        
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
                    
            # Collect codons at the central site (site_idx itself) for stats
            if 1 <= site_idx <= seq_len_codons:
                nuc_idx = (site_idx - 1) * 3
                codon = seq[nuc_idx:nuc_idx+3].upper()
                if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                    site_codons.append(codon)
                    aa = GENETIC_CODE.get(codon, '?')
                    if aa != '?':
                        site_aas.append(aa)
                        spec_aas.append(aa)
                    else:
                        spec_aas.append('?')
                else:
                    spec_aas.append('?')
            else:
                spec_aas.append('?')
                        
        variable_sites_flags.append(is_site_variable(site_codons, site_aas))
        
    msa_tokens = msa_tokens.to(device)
    aa_tokens = aa_tokens.to(device)
    dist_tensor = dist_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
    mds_coords_tensor = mds_coords_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
    padding_mask_tensor = padding_mask.unsqueeze(0).expand(total_codons, -1).to(device)
    
    # 6. Load trained model weights
    print(f"[*] Loading model checkpoint: {args.model}")
    if not os.path.exists(args.model):
        print(f"[!] Error: Model checkpoint not found: {args.model}")
        sys.exit(1)
        
    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        window_size=args.window_size,
        max_species=args.max_species
    ).to(device)
    
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[*] Model successfully loaded and state dict mapped.")
    except Exception as e:
        print(f"[!] Error loading model checkpoint: {e}")
        sys.exit(1)
        
    # 7. Batched Model Inference (No Sigmoid for Regression Model)
    model.eval()
    print("[*] Running model predictions across all sites in a single batched pass...")
    with torch.no_grad():
        logits = model(msa_tokens, aa_tokens, dist_tensor, mds_coords_tensor, padding_mask_tensor)
        pred_log_lrts = logits.cpu().numpy()
        
    # 8. Build Predictions Table
    predictions = []
    for site_idx in range(1, total_codons + 1):
        ref_nuc_idx = (site_idx - 1) * 3
        ref_codon = ref_seq[ref_nuc_idx:ref_nuc_idx+3].upper()
        ref_aa = translate_codon(ref_codon)
        
        is_var = variable_sites_flags[site_idx - 1]
        if is_var:
            pred_log_lrt = float(pred_log_lrts[site_idx - 1])
            pred_lrt = max(0.0, math.exp(pred_log_lrt) - 1.0)
        else:
            pred_log_lrt = 0.0
            pred_lrt = 0.0
        
        predictions.append({
            "codon_site": site_idx,
            "ref_codon": ref_codon,
            "ref_aa": ref_aa,
            "is_variable": int(is_var),
            "predicted_log_lrt": round(pred_log_lrt, 5),
            "predicted_lrt": round(pred_lrt, 5)
        })
        
    df_preds = pd.DataFrame(predictions)
    
    # Calculate local relative metrics for variable sites
    df_preds["local_z_score"] = 0.0
    df_preds["local_percentile"] = 0.0
    df_preds["selection_call"] = "Neutral"
    
    var_mask = df_preds["is_variable"] == 1
    if var_mask.sum() > 0:
        var_lrts = df_preds.loc[var_mask, "predicted_lrt"].values
        mean_lrt = np.mean(var_lrts)
        std_lrt = np.std(var_lrts)
        
        if std_lrt > 0:
            df_preds.loc[var_mask, "local_z_score"] = np.round((var_lrts - mean_lrt) / std_lrt, 4)
        else:
            df_preds.loc[var_mask, "local_z_score"] = 0.0
            
        ranks = df_preds.loc[var_mask, "predicted_lrt"].rank(pct=True) * 100.0
        df_preds.loc[var_mask, "local_percentile"] = np.round(ranks, 2)
        
        # Apply Tier calling
        if args.use_zscore:
            t1_cond = df_preds["local_z_score"] >= args.tier1_zscore
            t2_cond = df_preds["local_z_score"] >= args.tier2_zscore
        else:
            t1_cond = df_preds["local_percentile"] >= args.tier1_percentile
            t2_cond = df_preds["local_percentile"] >= args.tier2_percentile
            
        # Optional absolute LRT gates
        if args.tier1_lrt_gate >= 0.0:
            t1_cond = t1_cond | (df_preds["predicted_lrt"] >= args.tier1_lrt_gate)
        if args.tier2_lrt_gate >= 0.0:
            t2_cond = t2_cond | (df_preds["predicted_lrt"] >= args.tier2_lrt_gate)
            
        t1_mask = var_mask & t1_cond
        t2_mask = var_mask & t2_cond & ~t1_mask
        
        df_preds.loc[t2_mask, "selection_call"] = "Tier 2 (Medium)"
        df_preds.loc[t1_mask, "selection_call"] = "Tier 1 (High)"
        
    # 9. Output predictions
    out_file = args.output
    if not out_file:
        basename = os.path.basename(args.alignment)
        if basename.endswith('.gz'):
            basename = basename[:-3]
        if basename.endswith('.nex') or basename.endswith('.nexus'):
            basename = basename.rsplit('.', 1)[0]
        out_file = f"{basename}_regression_predictions.csv"
        
    df_preds.to_csv(out_file, index=False)
    
    # 10. Print Summary
    print("\n" + "=" * 50)
    print("✨ Regression Prediction Summary")
    print("=" * 50)
    print(f"Total codon sites predicted: {len(df_preds)}")
    print(f"Mean predicted log(LRT+1):   {df_preds['predicted_log_lrt'].mean():.4f}")
    print(f"Mean predicted raw LRT:      {df_preds['predicted_lrt'].mean():.4f}")
    print(f"Max predicted raw LRT:       {df_preds['predicted_lrt'].max():.4f}")
    
    t1_sites = df_preds[df_preds["selection_call"] == "Tier 1 (High)"]
    t2_sites = df_preds[df_preds["selection_call"] == "Tier 2 (Medium)"]
    
    gate_t1_str = f" or predicted_lrt >= {args.tier1_lrt_gate}" if args.tier1_lrt_gate >= 0.0 else ""
    gate_t2_str = f" or predicted_lrt >= {args.tier2_lrt_gate}" if args.tier2_lrt_gate >= 0.0 else ""
    
    if args.use_zscore:
        print(f"Calling selection based on Z-score and absolute LRT gates:")
        print(f"  - Tier 1 (High Confidence, Z-score >= {args.tier1_zscore}{gate_t1_str}): {len(t1_sites)} sites")
        print(f"  - Tier 2 (Medium Confidence, Z-score >= {args.tier2_zscore}{gate_t2_str}): {len(t2_sites)} sites")
    else:
        print(f"Calling selection based on local percentile and absolute LRT gates:")
        print(f"  - Tier 1 (High Confidence, Percentile >= {args.tier1_percentile}%{gate_t1_str}): {len(t1_sites)} sites")
        print(f"  - Tier 2 (Medium Confidence, Percentile >= {args.tier2_percentile}%{gate_t2_str}): {len(t2_sites)} sites")
        
    called_sites = df_preds[df_preds["selection_call"] != "Neutral"]
    if len(called_sites) > 0:
        print(f"\nPredicted positive selection sites:")
        print(called_sites.sort_values(by="local_percentile", ascending=False).to_string(index=False))
        
    print(f"\n🎉 Predictions complete! Results saved to '{out_file}'")

if __name__ == "__main__":
    main()
