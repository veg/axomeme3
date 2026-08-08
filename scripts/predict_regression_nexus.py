import torch._utils
if not hasattr(torch._utils, '_rebuild_device_tensor_from_cpu_tensor'):
    def _rebuild_device_tensor_from_cpu_tensor(data, device, *args, **kwargs):
        req_grad = kwargs.get('requires_grad', False)
        if len(args) > 1 and isinstance(args[-1], bool):
            req_grad = args[-1]
        return data.to(device).requires_grad_(bool(req_grad))
    torch._utils._rebuild_device_tensor_from_cpu_tensor = _rebuild_device_tensor_from_cpu_tensor


import subprocess
import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip
import pickle
import re
from io import StringIO
from Bio import Phylo
from collections import Counter
from train_transformer_selection import PhyloAxialTransformer, compute_mds_coordinates, decode_soft_ordinal_lrt

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
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

CODON_LIST = [
    'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT',
    'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT',
    'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
    'TAC', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGC', 'TGG', 'TGT', 'TTC', 'TTG', 'TTT'
]
CODON_TO_IDX = {c: i for i, c in enumerate(CODON_LIST)}

AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}

def get_codon_token(codon):
    return CODON_TO_IDX.get(codon.upper(), 64)

def get_aa_token(codon):
    aa = GENETIC_CODE.get(codon.upper(), '?')
    return AA_TO_IDX.get(aa, 20) if aa != '?' else 21


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
    dist_64 = dist_matrix_np.astype(np.float64)
    D2 = dist_64 ** 2
    H = np.eye(N, dtype=np.float64) - np.ones((N, N), dtype=np.float64) / float(N)
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
            coords[:, i] = (evecs[:, i] * np.sqrt(val)).astype(np.float32)
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
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)
        return self.out_proj(out)


# --- Custom Row Attention with Learnable Phylogenetic Bias & Genetic Code Biases ---
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 3-Channel Unrooted Tree Topological Attention Projection:
        # Channel 0: Patristic Path Distance D_ij
        # Channel 1: Topological Node Count N_ij
        # Channel 2: Off-Path Subtree Density S_ij
        self.tree_w1 = nn.Parameter(torch.randn(num_heads, 3) * 0.02)
        self.tree_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.tree_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        
        # Legacy fallback support for 1D distance inputs
        self.phylo_w1 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        self.phylo_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.phylo_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        
        # Explicit Genetic Code Pairwise Attention Biases
        self.nonsyn_head_bias = nn.Parameter(torch.tensor(2.0))
        self.syn_head_bias = nn.Parameter(torch.tensor(-1.0))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix, padding_mask=None, nonsyn_mask=None, syn_mask=None):
        batch_size, num_species, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Sequence Density Invariant Softmax Normalization
        if padding_mask is not None:
            active_counts = (~padding_mask).sum(dim=-1, keepdim=True).clamp(min=1.0).float()
            density_scale = torch.log(active_counts / 256.0).unsqueeze(-1).unsqueeze(-1)
            scores = scores + density_scale
            
        # Unrooted 3-Channel Tree Feature Projection
        if dist_matrix.dim() == 4: # [batch_size, num_species, num_species, 3]
            tree_proj = torch.matmul(dist_matrix, self.tree_w1.t()).permute(0, 3, 1, 2)
            tree_bias = F.softplus(tree_proj + self.tree_b1) * F.softplus(self.tree_w2)
        elif dist_matrix.dim() == 3 and dist_matrix.shape[-1] == 3: # [num_species, num_species, 3]
            tree_proj = torch.matmul(dist_matrix, self.tree_w1.t()).permute(2, 0, 1).unsqueeze(0)
            tree_bias = F.softplus(tree_proj + self.tree_b1) * F.softplus(self.tree_w2)
        else: # Legacy 1D Distance Fallback [batch_size, N, N] or [N, N]
            bias = dist_matrix.unsqueeze(1) if dist_matrix.dim() == 3 else dist_matrix.unsqueeze(0).unsqueeze(1)
            tree_bias = F.softplus(self.phylo_w1 * bias + self.phylo_b1) * F.softplus(self.phylo_w2)
            
        scores = scores - tree_bias
        
        if nonsyn_mask is not None:
            scores = scores + self.nonsyn_head_bias * nonsyn_mask.unsqueeze(1)
        if syn_mask is not None:
            scores = scores + self.syn_head_bias * syn_mask.unsqueeze(1)
            
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        return self.out_proj(out)


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


# --- Fitch Codon Parsimony and Tree Topology Utilities ---
CODON_TO_AA_DICT = {
    'TTT': 0, 'TTC': 0, 'TTA': 1, 'TTG': 1, 'TCT': 2, 'TCC': 2, 'TCA': 2, 'TCG': 2,
    'TAT': 3, 'TAC': 3, 'TAA': 20, 'TAG': 20, 'TGT': 4, 'TGC': 4, 'TGA': 20, 'TGG': 5,
    'CTT': 1, 'CTC': 1, 'CTA': 1, 'CTG': 1, 'CCT': 6, 'CCC': 6, 'CCA': 6, 'CCG': 6,
    'CAT': 7, 'CAC': 7, 'CAA': 8, 'CAG': 8, 'CGT': 9, 'CGC': 9, 'CGA': 9, 'CGG': 9,
    'ATT': 10, 'ATC': 10, 'ATA': 10, 'ATG': 11, 'ACT': 12, 'ACC': 12, 'ACA': 12, 'ACG': 12,
    'AAT': 13, 'AAC': 13, 'AAA': 14, 'AAG': 14, 'AGT': 2, 'AGC': 2, 'AGA': 9, 'AGG': 9,
    'GTT': 15, 'GTC': 15, 'GTA': 15, 'GTG': 15, 'GCT': 16, 'GCC': 16, 'GCA': 16, 'GCG': 16,
    'GAT': 17, 'GAC': 17, 'GAA': 18, 'GAG': 18, 'GGT': 19, 'GGC': 19, 'GGA': 19, 'GGG': 19
}

NUC_LIST = ['T', 'C', 'A', 'G']
SENSE_CODONS = [n1+n2+n3 for n1 in NUC_LIST for n2 in NUC_LIST for n3 in NUC_LIST if CODON_TO_AA_DICT[n1+n2+n3] < 20]
SENSE_CODON_TO_IDX = {c: i for i, c in enumerate(SENSE_CODONS)}
SERINE_TCT_SET = {SENSE_CODON_TO_IDX[c] for c in ['TCT', 'TCC', 'TCA', 'TCG']}
SERINE_AGC_SET = {SENSE_CODON_TO_IDX[c] for c in ['AGT', 'AGC']}

def build_tree_topology(newick_str, selected_species):
    clean_newick = newick_str.split(";")[0].strip() + ";" if newick_str else ""
    if not clean_newick:
        N = len(selected_species)
        num_nodes = 2 * N - 1
        parent_array = np.full(num_nodes, -1, dtype=np.int32)
        for i in range(N):
            parent_array[i] = N + (i // 2) if (N + (i // 2)) < num_nodes else num_nodes - 1
        branch_lengths = np.ones(num_nodes, dtype=np.float32) * 0.05
        return parent_array, branch_lengths
        
    try:
        root = parse_newick(clean_newick)
    except Exception:
        N = len(selected_species)
        num_nodes = 2 * N - 1
        parent_array = np.full(num_nodes, -1, dtype=np.int32)
        for i in range(N):
            parent_array[i] = N + (i // 2) if (N + (i // 2)) < num_nodes else num_nodes - 1
        branch_lengths = np.ones(num_nodes, dtype=np.float32) * 0.05
        return parent_array, branch_lengths
        
    species_to_idx = {}
    norm_selected = [s.replace("'", "").replace('"', '').strip() for s in selected_species]
    for idx, s in enumerate(norm_selected):
        species_to_idx[s] = idx
        
    N = len(selected_species)
    num_nodes = 2 * N - 1
    node_to_id = {}
    
    all_nodes = []
    stack = [root]
    while stack:
        curr = stack.pop()
        all_nodes.append(curr)
        for c in reversed(curr.children):
            stack.append(c)
            
    leaves = [n for n in all_nodes if not n.children]
    internals = [n for n in all_nodes if n.children]
    
    leaves_found = 0
    for term in leaves:
        term_name = term.name.replace("'", "").replace('"', '').strip() if term.name else ""
        if term_name in species_to_idx:
            idx = species_to_idx[term_name]
            node_to_id[term] = idx
            leaves_found += 1
            
    if leaves_found < N:
        for idx, term in enumerate(leaves):
            if idx < N and term not in node_to_id:
                node_to_id[term] = idx
                
    next_int_id = N
    for n in internals:
        node_to_id[n] = next_int_id
        next_int_id += 1
        if next_int_id >= num_nodes:
            break
            
    parent_array = np.full(num_nodes, -1, dtype=np.int32)
    branch_lengths = np.ones(num_nodes, dtype=np.float32) * 1e-3
    
    for n, n_id in node_to_id.items():
        branch_lengths[n_id] = max(float(n.length), 1e-4)
        if n.parent and n.parent in node_to_id:
            parent_array[n_id] = node_to_id[n.parent]
            
    return parent_array, branch_lengths

def build_sankoff_cost_matrix():
    cost_matrix = np.zeros((61, 61), dtype=np.float32)
    for i, c1 in enumerate(SENSE_CODONS):
        aa1 = CODON_TO_AA_DICT[c1]
        for j, c2 in enumerate(SENSE_CODONS):
            if i == j:
                cost_matrix[i, j] = 0.0
                continue
            aa2 = CODON_TO_AA_DICT[c2]
            nuc_diff = sum(1 for k in range(3) if c1[k] != c2[k])
            
            if aa1 == aa2:
                cost_matrix[i, j] = 1.0 * nuc_diff
            else:
                cost_matrix[i, j] = 2.5 * nuc_diff
    return cost_matrix

SANKOFF_COST_MATRIX = build_sankoff_cost_matrix()

def fitch_codon_parsimony(site_codon_ids, parent_array, branch_lengths, max_k=32):
    num_nodes = len(parent_array)
    num_taxa = (num_nodes + 1) // 2
    
    S = np.zeros((num_nodes, 61), dtype=np.float32)
    
    for i in range(min(num_taxa, len(site_codon_ids))):
        c_tok = site_codon_ids[i]
        if c_tok < 64:
            c_str = codons_list[c_tok] if c_tok < 64 else '???'
            s_idx = SENSE_CODON_TO_IDX.get(c_str, 61)
            if s_idx < 61:
                S[i, :] = 1e6
                S[i, s_idx] = 0.0
            else:
                S[i, :] = 0.0
        else:
            S[i, :] = 0.0
            
    children = [[] for _ in range(num_nodes)]
    for v in range(num_nodes):
        p = parent_array[v]
        if p >= 0:
            children[p].append(v)
            
    root = -1
    for v in range(num_nodes):
        if parent_array[v] < 0 and len(children[v]) > 0:
            root = v
            break
    if root < 0:
        root = num_nodes - 1
        
    # Dynamic Post-Order Bottom-Up DP (children before parent)
    post_order = []
    def get_post_order(u):
        for ch in children[u]:
            get_post_order(ch)
        post_order.append(u)
    get_post_order(root)
    
    for u in post_order:
        ch = children[u]
        if len(ch) > 0:
            node_cost = np.zeros(61, dtype=np.float32)
            for child in ch:
                ch_cost_matrix = S[child, :][np.newaxis, :] + SANKOFF_COST_MATRIX
                node_cost += np.min(ch_cost_matrix, axis=1)
            S[u, :] = node_cost
            
    # Dynamic Pre-Order Top-Down Backtracking (parent before children)
    pre_order = post_order[::-1]
    reconstructed = np.zeros(num_nodes, dtype=np.int32)
    reconstructed[root] = np.argmin(S[root, :])
    
    for u in pre_order[1:]:
        p = parent_array[u]
        p_state = reconstructed[p]
        costs = S[u, :] + SANKOFF_COST_MATRIX[p_state, :]
        reconstructed[u] = np.argmin(costs)

        
    active_edges = []
    total_syn_count = 0.0
    total_nonsyn_count = 0.0
    
    for v in range(num_nodes - 1):
        p = parent_array[v]
        if p < 0:
            continue
        c_u = reconstructed[p]
        c_v = reconstructed[v]
        
        if c_u != c_v and c_u < 61 and c_v < 61:
            b_len = max(float(branch_lengths[v]), 1e-4)
            sub_id = c_u * 61 + c_v
            
            aa_u = CODON_TO_AA_DICT[SENSE_CODONS[c_u]]
            aa_v = CODON_TO_AA_DICT[SENSE_CODONS[c_v]]
            
            str_u, str_v = SENSE_CODONS[c_u], SENSE_CODONS[c_v]
            nuc_diff = sum(1 for i in range(3) if str_u[i] != str_v[i])
            
            is_syn = 1.0 if aa_u == aa_v else 0.0
            is_nonsyn_single = 1.0 if (aa_u != aa_v and nuc_diff == 1) else 0.0
            is_nonsyn_multi = 1.0 if (aa_u != aa_v and nuc_diff > 1) else 0.0
            is_serine = 1.0 if (aa_u == aa_v and ((c_u in SERINE_TCT_SET and c_v in SERINE_AGC_SET) or (c_u in SERINE_AGC_SET and c_v in SERINE_TCT_SET))) else 0.0
            
            if is_syn == 1.0:
                total_syn_count += 1.0
            else:
                total_nonsyn_count += 1.0
                
            rate = 1.0 / b_len
            active_edges.append((is_syn, rate, sub_id, [is_syn, is_nonsyn_single, is_nonsyn_multi, is_serine], b_len))
            
    dNdS_ratio = total_nonsyn_count / (total_syn_count + 0.1)
    rates = [e[1] for e in active_edges]
    mean_rate = float(np.mean(rates)) if len(rates) > 0 else 1.0
    
    nonsyn_edges = [e for e in active_edges if e[0] == 0.0]
    syn_edges = [e for e in active_edges if e[0] == 1.0]
    
    nonsyn_edges.sort(key=lambda x: x[1], reverse=True)
    syn_edges.sort(key=lambda x: x[1], reverse=True)
    
    # Dynamic dual allocation ratio: 75% non-synonymous, 25% synonymous
    target_nonsyn = int(max_k * 0.75)
    target_syn = max_k - target_nonsyn
    
    k_nonsyn = min(target_nonsyn, len(nonsyn_edges))
    k_syn = min(target_syn, len(syn_edges))
    
    selected = nonsyn_edges[:k_nonsyn] + syn_edges[:k_syn]
    rem = nonsyn_edges[k_nonsyn:] + syn_edges[k_syn:]
    rem.sort(key=lambda x: x[1], reverse=True)
    
    if len(selected) < max_k:
        selected += rem[:(max_k - len(selected))]
        
    sub_ids = np.zeros(max_k, dtype=np.int64)
    flags = np.zeros((max_k, 8), dtype=np.float32)
    lengths = np.ones(max_k, dtype=np.float32) * 1e-4
    mask = np.zeros(max_k, dtype=np.float32)
    
    for i, (_, rate, sub_id, fl, b_len) in enumerate(selected):
        sub_ids[i] = sub_id
        # Pure Transformer: Zero out all precomputed summary heuristics (dNdS_ratio, total_nonsyn, total_syn, burst_ratio)
        flags[i] = fl + [0.0, 0.0, 0.0, 0.0]
        lengths[i] = b_len
        mask[i] = 1.0
        
    return sub_ids, flags, lengths, mask


def _build_path_ns_tensor():
    # 61x61x2 precomputed lookup table of expected (N, S) steps
    sense_codons = ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAC', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']
    code = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R', 'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 'TAC':'Y', 'TAT':'Y', 'TGC':'C', 'TGT':'C', 'TGG':'W'}
    stops = {'TAA', 'TAG', 'TGA'}
    
    import itertools
    matrix = np.zeros((61, 61, 2), dtype=np.float32)
    for i, c1 in enumerate(sense_codons):
        for j, c2 in enumerate(sense_codons):
            if c1 == c2:
                continue
            diffs = [k for k in range(3) if c1[k] != c2[k]]
            perms = list(itertools.permutations(diffs))
            valid_paths = []
            for perm in perms:
                path = [c1]
                curr = list(c1)
                valid = True
                for pos in perm:
                    curr[pos] = c2[pos]
                    nc = "".join(curr)
                    if nc in stops:
                        valid = False
                        break
                    path.append(nc)
                if valid:
                    valid_paths.append(path)
            if not valid_paths:
                for perm in perms:
                    path = [c1]
                    curr = list(c1)
                    for pos in perm:
                        curr[pos] = c2[pos]
                        path.append("".join(curr))
                    valid_paths.append(path)
            tn, ts = 0.0, 0.0
            for p in valid_paths:
                pn, ps = 0, 0
                for step in range(len(p) - 1):
                    if code.get(p[step]) == code.get(p[step+1]):
                        ps += 1
                    else:
                        pn += 1
                tn += pn
                ts += ps
            matrix[i, j, 0] = tn / len(valid_paths)
            matrix[i, j, 1] = ts / len(valid_paths)
    return torch.tensor(matrix, dtype=torch.float32)


# --- Sparse Codon Edge Token Encoder ---
class SparseCodonEdgeEncoder(nn.Module):
    def __init__(self, embed_dim=128, max_k=64, num_categories=8):
        super().__init__()
        self.max_k = max_k
        self.embed_dim = embed_dim
        
        self.register_buffer('path_ns_matrix', _build_path_ns_tensor())
        self.codon_sub_embed = nn.Embedding(3721, 64)
        self.category_proj = nn.Linear(num_categories, 32)
        
        self.b_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 32)
        )
        
        # Additional path projection layer for (exp_N, exp_S, nonsyn_ratio, nonsyn_flux, syn_flux, diff_flux)
        self.path_proj = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 16)
        )
        
        self.edge_proj = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 16, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.pool_combine = nn.Sequential(
            nn.Linear(5 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, active_sub_ids, active_flags, active_lengths, active_mask):
        if self.category_proj.in_features == 4 and active_flags.shape[-1] >= 4:
            cat_flags = active_flags[..., :4]
        elif self.category_proj.in_features == 7 and active_flags.shape[-1] >= 7:
            cat_flags = active_flags[..., :7]
        elif self.category_proj.in_features == 8 and active_flags.shape[-1] < 8:
            pad_size = 8 - active_flags.shape[-1]
            pad_tensor = torch.zeros((*active_flags.shape[:-1], pad_size), device=active_flags.device, dtype=active_flags.dtype)
            cat_flags = torch.cat([active_flags, pad_tensor], dim=-1)
        else:
            cat_flags = active_flags
            
        sub_emb = self.codon_sub_embed(active_sub_ids)
        cat_emb = self.category_proj(cat_flags)
        
        log_b = torch.log(torch.clamp(active_lengths, min=1e-4))
        b_feat = torch.stack([active_lengths, log_b], dim=-1)
        b_emb = self.b_mlp(b_feat)
        
        # Extract path-averaged (N, S) metrics from lookup matrix
        c_u = torch.clamp(active_sub_ids // 61, min=0, max=60)
        c_v = torch.clamp(active_sub_ids % 61, min=0, max=60)
        ns_vals = self.path_ns_matrix[c_u, c_v] # [B, K, 2]
        exp_N = ns_vals[..., 0] # [B, K]
        exp_S = ns_vals[..., 1] # [B, K]
        
        b_len_clamp = torch.clamp(active_lengths, min=1e-4)
        
        # Solution 1: Composition-Aware Path-Averaged Features
        nonsyn_ratio = exp_N / (exp_N + exp_S + 1e-6)
        nonsyn_flux = exp_N / b_len_clamp
        syn_flux = exp_S / b_len_clamp
        diff_flux = (exp_N - exp_S) / b_len_clamp
        
        path_feats = torch.stack([exp_N, exp_S, nonsyn_ratio, nonsyn_flux, syn_flux, diff_flux], dim=-1)
        path_emb = self.path_proj(path_feats)
        
        concat_feat = torch.cat([sub_emb, cat_emb, b_emb, path_emb], dim=-1)
        edge_vec = self.edge_proj(concat_feat)
        
        intensity = 1.0 / (torch.clamp(active_lengths, min=1e-4) + 1e-3)
        intensity_log = torch.log1p(torch.clamp(intensity, max=100.0))
        
        weighted_edge_vec = edge_vec * intensity_log.unsqueeze(-1) * active_mask.unsqueeze(-1)
        
        # 1. Max Pooling across active edges (isolates 1st highest burst)
        masked_for_max = weighted_edge_vec.masked_fill((active_mask == 0).unsqueeze(-1), -1e4)
        max_pooled = torch.relu(torch.max(masked_for_max, dim=1)[0])
        
        # 2. Mean Pooling across active edges (tree-size invariant rate average)
        active_counts = active_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pooled = weighted_edge_vec.sum(dim=1) / active_counts
        
        # 3. L2 Norm Pooling (overall mutational energy normalized by active counts)
        l2_pooled = torch.sqrt((weighted_edge_vec ** 2).sum(dim=1) / active_counts + 1e-6)
        
        # 4. Softmax Attention Pooling (weighted by edge intensity)
        attn_logits = (weighted_edge_vec.sum(dim=-1) / math.sqrt(self.embed_dim)).masked_fill(active_mask == 0, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        attn_pooled = (weighted_edge_vec * attn_weights).sum(dim=1)
        
        # 5. Top-2 Edge Pooling (isolates 2nd highest burst, zeroed if < 2 active edges)
        has_at_least_two_edges = (active_counts >= 2.0).float()
        top2_val, _ = torch.topk(masked_for_max, k=min(2, masked_for_max.shape[1]), dim=1)
        if top2_val.shape[1] >= 2:
            top2_pooled = torch.relu(top2_val[:, 1, :]) * has_at_least_two_edges
        else:
            top2_pooled = max_pooled * has_at_least_two_edges
            
        has_edges = (active_counts > 0.0).float()
        combined = torch.cat([max_pooled, mean_pooled, l2_pooled, attn_pooled, top2_pooled], dim=-1)
        return F.layer_norm(self.pool_combine(combined), (self.embed_dim,)) * has_edges



# Use imported PhyloAxialTransformer from train_transformer_selection.py
# --- Differentiable Ranking Loss via ListNet ---

def parse_nexus_alignment_and_embedded_tree(filepath):
    # Try fasta parsing first if file starts with >
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        first_char = f.read(1)
        if first_char == '>':
            f.seek(0)
            from Bio import SeqIO
            seq_dict = {}
            taxlabels = []
            for record in SeqIO.parse(f, 'fasta'):
                seq_dict[record.id] = str(record.seq).upper()
                taxlabels.append(record.id)
            return seq_dict, taxlabels, None
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
    tree_tensor_dict = {} # 3-Channel Unrooted Tree Feature Matrix
    
    for name in leaf_names:
        dist_matrix[name] = {name: 0.0}
        tree_tensor_dict[name] = {name: [0.0, 0.0, 0.0]}
        
    n = len(leaf_names)
    for i in range(n):
        name1 = leaf_names[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, n):
            name2 = leaf_names[j]
            path2 = leaf_paths[name2]
            
            lca = None
            idx1 = 0
            idx2 = 0
            for p2_idx, node in enumerate(path2):
                if node in set1:
                    lca = node
                    idx2 = p2_idx
                    idx1 = path1.index(node)
                    break
                    
            if lca is not None:
                dist = node_to_root_dist[leaf_by_name[name1]] + node_to_root_dist[leaf_by_name[name2]] - 2 * node_to_root_dist[lca]
                node_count = float(idx1 + idx2)
            else:
                dist = node_to_root_dist[leaf_by_name[name1]] + node_to_root_dist[leaf_by_name[name2]]
                node_count = float(len(path1) + len(path2))
                
            density = math.log((node_count + 1.0) / (dist + 0.1))
            
            dist_matrix[name1][name2] = dist
            dist_matrix[name2][name1] = dist
            
            tree_feats = [dist, node_count, density]
            tree_tensor_dict[name1][name2] = tree_feats
            tree_tensor_dict[name2][name1] = tree_feats
            
    return leaf_names, dist_matrix, tree_tensor_dict


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
    parser.add_argument("--model", default="/Users/sergei/Projects/TOGA_MEME/selection_transformer_edge_best-28.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--output", help="Path to output predictions CSV. Defaults to [alignment_prefix]_regression_predictions.csv")
    parser.add_argument("--reference_seq", help="Name of reference sequence (e.g. hg, hg38). Defaults to first sequence.")
    parser.add_argument("--window_size", type=int, default=1, help="Alignment sliding window size centered at site (default: 1)")
    parser.add_argument("--max_species", type=int, default=512, help="Maximum number of sequences to feed to model (default: 512)")
    parser.add_argument("--device", help="Force run on device (cpu, mps, cuda). Auto-detected by default.")
    parser.add_argument("--call_mode", choices=["pvalue", "zscore", "percentile"], default="pvalue", help="Selection calling mode: 'pvalue' (default, LRT >= 3.81/5.14), 'zscore' (local Z >= 2.0/2.5), or 'percentile' (top 3%%/5%%)")
    parser.add_argument("--use_zscore", action="store_true", help="Enable local relative Z-score calling (equivalent to --call_mode zscore)")
    parser.add_argument("--tier1_percentile", type=float, default=98.0, help="Percentile threshold for Tier 1 High-Confidence calls (default: 98.0)")
    parser.add_argument("--tier2_percentile", type=float, default=95.0, help="Percentile threshold for Tier 2 Medium-Confidence calls (default: 95.0)")
    parser.add_argument("--tier1_zscore", type=float, default=2.5, help="Z-score threshold for Tier 1 High-Confidence calls (default: 2.5)")
    parser.add_argument("--tier2_zscore", type=float, default=2.0, help="Z-score threshold for Tier 2 Medium-Confidence calls (default: 2.0)")
    parser.add_argument("--tier1_lrt_gate", type=float, default=4.45, help="Absolute predicted LRT gate for Tier 1 calls (p <= 0.05, default: 4.45)")
    parser.add_argument("--tier2_lrt_gate", type=float, default=3.12, help="Absolute predicted LRT gate for Tier 2 calls (p <= 0.10, default: 3.12)")
    parser.add_argument("--prior_shift", type=float, default=0.0, help="Bayesian prior logit shift (e.g. 2.20 for 10%% prior, 3.89 for 2%% prior) to calibrate 50:50 training loader bias")
    
    args = parser.parse_args()
    if args.use_zscore:
        args.call_mode = "zscore"
    
    # 1. Device Setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") # Default to CPU on macOS to prevent PyTorch MPS float precision bugs
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
    
    if not species_names:
        species_names = list(seq_dict.keys())
        
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
    if not matching_species:
        print("[!] Warning: Zero matching species between tree and alignment. Falling back to alignment order.")
        matching_species = list(seq_dict.keys())
        
    if ref_key in matching_species:
        matching_species.remove(ref_key)
        matching_species.insert(0, ref_key)
    # 4.1. Estimate branch lengths using HyPhy FIRST if they are missing or all zero across tree topology
    if tree_obj:
        has_branch_lengths = any(
            clade.branch_length is not None and clade.branch_length > 0.0 
            for clade in tree_obj.find_clades() 
            if clade != tree_obj.root
        )
        
        if not has_branch_lengths:
            print("[!] Tree topology has no branch lengths. Running HyPhy to estimate branch lengths on alignment...")
            try:
                # Format full tree topology Newick string (removing any placeholder/0.0 branch lengths)
                out_stream = StringIO()
                Phylo.write(tree_obj, out_stream, 'newick')
                raw_tree_str = out_stream.getvalue().strip()
                clean_pruned_tree_str = re.sub(r':[0-9.eE-]+', '', raw_tree_str)
                
                # Write temporary alignment FASTA containing matching species
                os.makedirs("scratch", exist_ok=True)
                temp_fasta = "scratch/temp_hyphy_align.fa"
                with open(temp_fasta, "w") as f:
                    for spec in matching_species:
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
Tree T = "{clean_pruned_tree_str}";
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
                    tree_obj = Phylo.read(StringIO(estimated_tree_str), 'newick')
                    print("[*] HyPhy branch length estimation succeeded.")
                    
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

    # 4.2. Select Species Subsample (Max-PD or Cap)
    if len(matching_species) > args.max_species:
        if tree_obj:
            clean_species_full, patristic_dict_full, _ = calculate_patristic_distances(tree_obj)
            N_full = len(matching_species)
            D_full = np.zeros((N_full, N_full), dtype=np.float32)
            for i, sp1 in enumerate(matching_species):
                for j, sp2 in enumerate(matching_species):
                    if sp1 in patristic_dict_full and sp2 in patristic_dict_full[sp1]:
                        D_full[i, j] = patristic_dict_full[sp1][sp2]
            
            # Max-PD Farthest-Point Traversal
            selected_local = [0]
            min_dists = D_full[0].copy()
            for _ in range(1, args.max_species):
                next_idx = np.argmax(min_dists)
                selected_local.append(next_idx)
                min_dists = np.minimum(min_dists, D_full[next_idx])
                
            selected_species = [matching_species[i] for i in selected_local]
            print(f"[*] Max-PD (Faith's PD) Selection: Selected {len(selected_species)} / {N_full} species maximizing tree branch length.")
        else:
            selected_species = matching_species[:args.max_species]
    else:
        selected_species = matching_species
        
    num_selected = len(selected_species)
    print(f"[*] Final model input selection: {num_selected} species (max_species cap = {args.max_species})")
    
    dist_matrix = {}
    if tree_obj:
        try:
            species_names, dist_matrix, tree_tensor_dict = calculate_patristic_distances(tree_obj)
        except Exception as e:
            print(f"[!] Error calculating patristic distances: {e}")
            dist_matrix = {}
            tree_tensor_dict = {}
            
    if not dist_matrix:
        dist_matrix = {s1: {s2: 0.0 for s2 in selected_species} for s1 in selected_species}
        tree_tensor_dict = {s1: {s2: [0.0, 0.0, 0.0] for s2 in selected_species} for s1 in selected_species}
        
    # 5. Build input tensors for model
    ref_seq = seq_dict[ref_key]
    if len(ref_seq) % 3 != 0:
        print(f"[!] Warning: Reference sequence length ({len(ref_seq)} nucs) is not a multiple of 3. Truncating tail.")
    total_codons = len(ref_seq) // 3
    print(f"[*] Reference sequence '{ref_key}' length: {total_codons} codons.")
    
    msa_tokens = torch.ones(total_codons, args.max_species, args.window_size, dtype=torch.long) * 65
    aa_tokens = torch.ones(total_codons, args.max_species, args.window_size, dtype=torch.long) * AA_TO_IDX['?']
    dist_tensor = torch.zeros(args.max_species, args.max_species, dtype=torch.float32)
    padding_mask = torch.ones(args.max_species, dtype=torch.bool) # True means padded
    
    for i, spec1 in enumerate(selected_species):
        padding_mask[i] = False
        for j, spec2 in enumerate(selected_species):
            norm1 = spec1.replace("'", "").replace('"', '').strip()
            norm2 = spec2.replace("'", "").replace('"', '').strip()
            dist_val = dist_matrix.get(norm1, {}).get(norm2, 0.0)
            dist_tensor[i, j] = float(dist_val)
            
    # Compute MDS coordinates on the 2D distance matrix
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
        
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    ckpt_embed_dim = checkpoint.get('embed_dim', 256) if isinstance(checkpoint, dict) else 256
    ckpt_num_layers = checkpoint.get('num_layers', 6) if isinstance(checkpoint, dict) else 6
    ckpt_num_heads = checkpoint.get('num_heads', 8) if isinstance(checkpoint, dict) else 8
    ckpt_loss_type = checkpoint.get('loss_type', 'coral') if isinstance(checkpoint, dict) else 'coral'
    ckpt_pure_coral = checkpoint.get('pure_coral', (ckpt_loss_type == 'coral')) if isinstance(checkpoint, dict) else True
    
    ckpt_window_size = args.window_size
    if 'pos_embedding' in state_dict:
        ckpt_window_size = state_dict['pos_embedding'].shape[1]
        ckpt_embed_dim = state_dict['pos_embedding'].shape[-1]
        
    layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('col_layers.')]
    if layer_indices:
        ckpt_num_layers = max(layer_indices) + 1
        
    ckpt_num_streams = 5
    if 'stream_fusion.0.block_codon.weight' in state_dict:
        ckpt_num_streams = state_dict['stream_fusion.0.block_codon.weight'].shape[1] // (ckpt_embed_dim // 2)
    elif 'stream_fusion.0.weight' in state_dict:
        ckpt_num_streams = state_dict['stream_fusion.0.weight'].shape[1] // ckpt_embed_dim
        
    ckpt_num_thresholds = 12
    if 'lrt_ordinal_head.theta_steps' in state_dict:
        ckpt_num_thresholds = state_dict['lrt_ordinal_head.theta_steps'].shape[0] + 1
        
    print(f'[*] Auto-detected checkpoint architecture: embed_dim={ckpt_embed_dim}, num_layers={ckpt_num_layers}, num_heads={ckpt_num_heads}, window_size={ckpt_window_size}, num_streams={ckpt_num_streams}, pure_coral={ckpt_pure_coral}, num_thresholds={ckpt_num_thresholds}')
    
    # Check if checkpoint is legacy 1-head (reg_head) vs 5-head multi-task (alpha_head)
    has_5_heads = ('alpha_head.0.weight' in state_dict)
    if 'reg_head.0.weight' in state_dict and not has_5_heads:
        state_dict['lrt_head.0.weight'] = state_dict['reg_head.0.weight']
        state_dict['lrt_head.0.bias'] = state_dict['reg_head.0.bias']
        state_dict['lrt_head.3.weight'] = state_dict['reg_head.3.weight']
        state_dict['lrt_head.3.bias'] = state_dict['reg_head.3.bias']
    
    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=ckpt_embed_dim,
        num_heads=ckpt_num_heads,
        num_layers=ckpt_num_layers,
        window_size=ckpt_window_size,
        max_species=args.max_species,
        num_streams=ckpt_num_streams,
        pure_coral=ckpt_pure_coral,
        num_thresholds=ckpt_num_thresholds
    ).to(device)
    
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    head_mismatch = any(k.startswith('lrt_') for k in state_dict.keys() if k not in filtered_state_dict)
    if head_mismatch:
        print(f"[!] Warning: Loaded {len(filtered_state_dict)}/{len(state_dict)} checkpoint weights. Output head shape mismatch detected!")
    else:
        print(f"[*] Successfully loaded all {len(filtered_state_dict)}/{len(state_dict)} checkpoint tensor weights into model!")
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print("[*] Running model predictions across all sites in 1-codon sliding windows...")
    pred_raw_lrts_list = []
    pred_alphas_list = []
    pred_beta_poses_list = []
    pred_p_negs_list = []
    has_5_heads = False

    model.eval()
    with torch.no_grad():
        for site_i in range(total_codons):
            msa_win = msa_tokens[site_i:site_i+1] # [1, max_species, 1]
            aa_win = aa_tokens[site_i:site_i+1]   # [1, max_species, 1]
            dist_win = dist_tensor[site_i:site_i+1] # [1, max_species, max_species]
            mds_win = mds_coords_tensor[site_i:site_i+1] # [1, max_species, 4]
            pad_win = padding_mask_tensor[site_i:site_i+1] # [1, max_species]
            
            out_eval = model(msa_win, aa_win, dist_win, mds_win, pad_win)
            
            if isinstance(out_eval, tuple) and len(out_eval) == 5:
                y_lrt_val, y_alpha, y_beta_neg, y_beta_pos, y_p_neg = out_eval
                
                # Apply optional Bayesian prior shift if specified
                if args.prior_shift != 0.0:
                    # Re-extract raw logits to apply prior shift
                    model.train()
                    raw_out = model(msa_win, aa_win, dist_win, mds_win, pad_win)
                    model.eval()
                    head0 = raw_out[0][0] if isinstance(raw_out[0], tuple) else raw_out[0]
                    logits_calibrated = head0 - args.prior_shift
                    y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_calibrated)
                    lrt_final = y_lrt_soft.item()
                else:
                    lrt_final = y_lrt_val.item() if hasattr(y_lrt_val, 'item') else float(y_lrt_val)
                    
                pred_raw_lrts_list.append(lrt_final)
                pred_alphas_list.append(np.expm1(y_alpha.item()))
                pred_beta_poses_list.append(np.expm1(y_beta_pos.item()))
                pred_p_negs_list.append(y_p_neg.item())
                has_5_heads = True
            else:
                lrt_final = out_eval[0].item() if hasattr(out_eval[0], 'item') else float(out_eval[0])
                pred_raw_lrts_list.append(lrt_final)
                has_5_heads = False

    pred_raw_lrts = np.array(pred_raw_lrts_list)
    if has_5_heads:
        pred_alphas = np.array(pred_alphas_list)
        pred_beta_poses = np.array(pred_beta_poses_list)
        pred_p_negs = np.array(pred_p_negs_list)
        
    # 8. Build Predictions Table
    predictions = []
    for site_idx in range(1, total_codons + 1):
        ref_nuc_idx = (site_idx - 1) * 3
        ref_codon = ref_seq[ref_nuc_idx:ref_nuc_idx+3].upper()
        ref_aa = translate_codon(ref_codon)
        
        is_var = variable_sites_flags[site_idx - 1] if site_idx - 1 < len(variable_sites_flags) else 1
        
        if not is_var:
            pred_log_lrt = 0.0
            pred_lrt = 0.0
            pred_alpha = 0.0
            pred_beta_pos = 0.0
            pred_p_pos = 0.0
        else:
            pred_lrt = max(0.0, float(pred_raw_lrts[site_idx - 1]))
            pred_log_lrt = math.log1p(pred_lrt)
            if has_5_heads:
                pred_alpha = max(0.0, float(pred_alphas[site_idx - 1]))
                pred_beta_pos = max(0.0, float(pred_beta_poses[site_idx - 1]))
                pred_p_pos = round(1.0 - float(pred_p_negs[site_idx - 1]), 4)
            else:
                pred_alpha = 0.0
                pred_beta_pos = 0.0
                pred_p_pos = 0.0
            
        row_dict = {
            "codon_site": site_idx,
            "ref_codon": ref_codon,
            "ref_aa": ref_aa,
            "is_variable": int(is_var),
            "predicted_log_lrt": round(pred_log_lrt, 5),
            "predicted_lrt": round(pred_lrt, 5)
        }
        if has_5_heads:
            row_dict["predicted_alpha_dS"] = round(pred_alpha, 4)
            row_dict["predicted_beta_pos_dN"] = round(pred_beta_pos, 4)
            row_dict["predicted_p_pos"] = pred_p_pos
            
        predictions.append(row_dict)
        
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
        
        # Apply Tier calling based on requested call_mode
        if args.call_mode == "zscore":
            t1_cond = df_preds["local_z_score"] >= args.tier1_zscore
            t2_cond = (df_preds["local_z_score"] >= args.tier2_zscore) & ~t1_cond
            t1_label, t2_label = f"Tier 1 (Z >= {args.tier1_zscore})", f"Tier 2 (Z >= {args.tier2_zscore})"
        elif args.call_mode == "percentile":
            t1_cond = df_preds["local_percentile"] >= args.tier1_percentile
            t2_cond = (df_preds["local_percentile"] >= args.tier2_percentile) & ~t1_cond
            t1_label, t2_label = f"Tier 1 (Top {100-args.tier1_percentile:.1f}%)", f"Tier 2 (Top {100-args.tier2_percentile:.1f}%)"
        else: # "pvalue" (default)
            t1_cond = df_preds["predicted_lrt"] >= args.tier1_lrt_gate
            t2_cond = (df_preds["predicted_lrt"] >= args.tier2_lrt_gate) & ~t1_cond
            t1_label, t2_label = "Tier 1 (p <= 0.05)", "Tier 2 (p <= 0.10)"
        
        t1_mask = var_mask & t1_cond
        t2_mask = var_mask & t2_cond & ~t1_mask
        
        df_preds.loc[t2_mask, "selection_call"] = t2_label
        df_preds.loc[t1_mask, "selection_call"] = t1_label
        
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
    
    t1_sites = df_preds[df_preds["selection_call"].str.startswith("Tier 1")]
    t2_sites = df_preds[df_preds["selection_call"].str.startswith("Tier 2")]
    
    print(f"Calling selection based on '{args.call_mode}' mode:")
    if args.call_mode == "zscore":
        print(f"  - Tier 1 (Z-score >= {args.tier1_zscore}): {len(t1_sites)} sites")
        print(f"  - Tier 2 (Z-score >= {args.tier2_zscore}): {len(t2_sites)} sites")
    elif args.call_mode == "percentile":
        print(f"  - Tier 1 (Percentile >= {args.tier1_percentile}%): {len(t1_sites)} sites")
        print(f"  - Tier 2 (Percentile >= {args.tier2_percentile}%): {len(t2_sites)} sites")
    else:
        print(f"  - Tier 1 (p <= 0.05, predicted_lrt >= {args.tier1_lrt_gate}): {len(t1_sites)} sites")
        print(f"  - Tier 2 (p <= 0.10, predicted_lrt >= {args.tier2_lrt_gate}): {len(t2_sites)} sites")
        
    called_sites = df_preds[df_preds["selection_call"] != "Neutral"]
    if len(called_sites) > 0:
        print(f"\nPredicted positive selection sites:")
        print(called_sites.sort_values(by="local_percentile", ascending=False).to_string(index=False))
        
    print(f"\n🎉 Predictions complete! Results saved to '{out_file}'")

if __name__ == "__main__":
    main()