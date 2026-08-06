import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import time
import re
import gzip
import math
import sqlite3
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr

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

# Codon vocabulary setup (64 codons + gap + unknown)
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

def get_codon_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 64
    if len(codon) != 3 or 'N' in codon:
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

# Precomputed mapping from codon to amino acid index directly
CODON_TO_AA_IDX = {}
for c in codons_list:
    aa = GENETIC_CODE.get(c, '?')
    CODON_TO_AA_IDX[c] = AA_TO_IDX.get(aa, 22)

# Create the 16.7M elements lookup tables (base 256)
CODON_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 65
AA_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 22

grid_arr = np.arange(256, dtype=np.int32)
b0_grid = grid_arr[:, None, None] * 65536
b1_grid = grid_arr[None, :, None] * 256
b2_grid = grid_arr[None, None, :]
flat_indices = (b0_grid + b1_grid + b2_grid).ravel()

b0_val = (flat_indices // 65536)
b1_val = (flat_indices // 256) % 256
b2_val = flat_indices % 256

gap_mask = (b0_val == 45) | (b1_val == 45) | (b2_val == 45)
CODON_LOOKUP[gap_mask] = 64
AA_LOOKUP[gap_mask] = 21

unknown_mask = (b0_val == 78) | (b1_val == 78) | (b2_val == 78) | \
               (b0_val == 110) | (b1_val == 110) | (b2_val == 110) | \
               (b0_val == 63) | (b1_val == 63) | (b2_val == 63)
unknown_mask = unknown_mask & (~gap_mask)
CODON_LOOKUP[unknown_mask] = 65
AA_LOOKUP[unknown_mask] = 22

for codon, c_idx in CODON_TO_IDX.items():
    if len(codon) == 3 and '-' not in codon and '?' not in codon:
        for c_str in [codon.upper(), codon.lower()]:
            b_encoded = c_str.encode('ascii')
            idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
            CODON_LOOKUP[idx] = c_idx
            aa = GENETIC_CODE.get(codon.upper(), '?')
            AA_LOOKUP[idx] = AA_TO_IDX.get(aa, 22)


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

def select_diverse_species_matrix(dist_matrix_np, k, keep_idx=0):
    n = dist_matrix_np.shape[0]
    if n <= k:
        return list(range(n))
    selected = [keep_idx]
    selected_set = {keep_idx}
    min_dists = dist_matrix_np[keep_idx].copy()
    for _ in range(k - 1):
        farthest_val = -1.0
        farthest_idx = -1
        for i in range(n):
            if i not in selected_set:
                if min_dists[i] > farthest_val:
                    farthest_val = min_dists[i]
                    farthest_idx = i
        if farthest_idx == -1:
            break
        selected.append(farthest_idx)
        selected_set.add(farthest_idx)
        min_dists = np.minimum(min_dists, dist_matrix_np[farthest_idx])
    return sorted(selected)

def compute_mds_coordinates(dist_matrix_np, n_components=4):
    N = dist_matrix_np.shape[0]
    if N <= n_components:
        return np.zeros((N, n_components), dtype=np.float32)
    D2 = dist_matrix_np ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (H @ D2 @ H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
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
    if len(unique_aas_set) > 1:
        return True
    if len(unique_aas_set) == 1 and 'S' in unique_aas_set:
        has_tcn = any(c in ('TCA', 'TCC', 'TCG', 'TCT') for c in site_codons)
        has_agy = any(c in ('AGC', 'AGT') for c in site_codons)
        if has_tcn and has_agy:
            return True
    return False

def precompute_variable_sites_worker(task):
    gene, msa_path = task
    if not os.path.exists(msa_path):
        return gene, []
    taxlabels = []
    sequences = []
    in_taxlabels = False
    in_matrix = False
    try:
        with gzip.open(msa_path, 'rt') as f:
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
    except:
        return gene, []
    seq_dict = {}
    for label, seq in zip(taxlabels, sequences):
        seq_dict[label] = seq
    any_seq = next(iter(seq_dict.values())) if seq_dict else ""
    num_codons = len(any_seq) // 3
    species_codons = []
    for spec in taxlabels:
        seq = seq_dict.get(spec, "")
        codons = [seq[idx*3:idx*3+3].upper() for idx in range(num_codons)]
        species_codons.append(codons)
    variable_sites = [False] * (num_codons + 1)
    for site_idx in range(1, num_codons + 1):
        c_idx = site_idx - 1
        site_codons = []
        site_aas = []
        for s_idx in range(len(taxlabels)):
            if s_idx < len(species_codons):
                codon = species_codons[s_idx][c_idx]
                if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                    site_codons.append(codon)
                    aa = GENETIC_CODE.get(codon, '?')
                    if aa != '?':
                        site_aas.append(aa)
        variable_sites[site_idx] = is_site_variable(site_codons, site_aas)
    return gene, variable_sites


class MSADataset(Dataset):
    def __init__(self, db_path, msa_dir, sites_list, window_size=5, max_species=256, cache_dict=None, cache_dir=None, cache_size_limit=0, subsample_species=False):
        self.db_path = db_path
        self.msa_dir = msa_dir
        self.sites = sites_list
        self.window_size = window_size
        self.max_species = max_species
        self.cache_dict = cache_dict
        self.cache_dir = cache_dir
        self.cache_size_limit = cache_size_limit
        self.subsample_species = subsample_species
        self.alignment_cache = {}

    def __len__(self):
        return len(self.sites)
        
    def _load_msa(self, gene_name):
        if gene_name in self.alignment_cache:
            return self.alignment_cache[gene_name]
        if self.cache_dir and os.path.isdir(self.cache_dir):
            npz_path = os.path.join(self.cache_dir, f"{gene_name}.npz")
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    dist_tensor_cached = torch.from_numpy(data["dist_arr"])
                    if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
                        dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
                    mds_coords_cached = torch.from_numpy(data["mds_coords"])
                    if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
                        mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
                    selected_indices = data["selected_indices"].tolist()
                    if len(selected_indices) > self.max_species:
                        selected_indices = select_diverse_species_matrix(data["dist_arr"], self.max_species, keep_idx=0)
                    self.alignment_cache[gene_name] = (
                        data["codon_ids_matrix"],
                        data["aa_ids_matrix"],
                        data["species_names"].tolist(),
                        dist_tensor_cached,
                        mds_coords_cached,
                        data["variable_sites"].tolist(),
                        selected_indices,
                        data["valid_seqs"]
                    )
                    if self.cache_size_limit > 0 and len(self.alignment_cache) > self.cache_size_limit:
                        first_key = next(iter(self.alignment_cache))
                        self.alignment_cache.pop(first_key)
                    return self.alignment_cache[gene_name]
                except Exception as e:
                    print(f"Error loading NPZ cache for {gene_name}: {e}")
        if self.cache_dict and gene_name in self.cache_dict:
            data = self.cache_dict[gene_name]
            if "codon_ids_matrix" in data:
                dist_tensor_cached = torch.from_numpy(data["dist_arr"])
                if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
                    dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
                mds_coords_cached = torch.from_numpy(data["mds_coords"])
                if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
                    mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
                selected_indices = data["selected_indices"]
                if len(selected_indices) > self.max_species:
                    selected_indices = select_diverse_species_matrix(data["dist_arr"], self.max_species, keep_idx=0)
                self.alignment_cache[gene_name] = (
                    data["codon_ids_matrix"],
                    data["aa_ids_matrix"],
                    data["species_names"],
                    dist_tensor_cached,
                    mds_coords_cached,
                    data["variable_sites"],
                    selected_indices,
                    data["valid_seqs"]
                )
                return self.alignment_cache[gene_name]
        return None

    def __getitem__(self, idx):
        gene_name, site_idx, label = self.sites[idx]
        data = self._load_msa(gene_name)
        if data is None:
            dummy_codon = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 65
            dummy_aa = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 22
            dummy_dist = torch.zeros(self.max_species, self.max_species)
            dummy_mds = torch.zeros(self.max_species, 4, dtype=torch.float32)
            dummy_mask = torch.ones(self.max_species, dtype=torch.bool)
            return dummy_codon, dummy_aa, dummy_dist, dummy_mds, dummy_mask, torch.tensor(label, dtype=torch.float)
            
        codon_ids_matrix, aa_ids_matrix, species_names, dist_tensor_cached, mds_coords_cached, _, selected_indices, valid_seqs = data
        if self.subsample_species and len(selected_indices) > 10:
            target_n = random.randint(10, len(selected_indices))
            sampled_idx = [selected_indices[0]] + sorted(random.sample(selected_indices[1:], target_n - 1))
        else:
            sampled_idx = selected_indices
            
        sub_dist = dist_tensor_cached[sampled_idx][:, sampled_idx]
        sub_mds = mds_coords_cached[sampled_idx]
        codon_tokens = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 65
        aa_tokens = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 22
        dist_tensor = torch.zeros(self.max_species, self.max_species)
        mds_coords = torch.zeros(self.max_species, 4, dtype=torch.float32)
        padding_mask = torch.ones(self.max_species, dtype=torch.bool)
        n_sel = len(sampled_idx)
        sampled_valid = valid_seqs[sampled_idx]
        padding_mask[:n_sel] = torch.from_numpy(~sampled_valid)
        
        half_win = self.window_size // 2
        start_1based = site_idx - half_win
        end_1based = site_idx + half_win
        start_idx = start_1based - 1
        end_idx = end_1based
        w_start = 0
        w_end = self.window_size
        seq_len_codons = codon_ids_matrix.shape[1]
        if start_idx < 0:
            w_start = -start_idx
            start_idx = 0
        if end_idx > seq_len_codons:
            w_end = w_end - (end_idx - seq_len_codons)
            end_idx = seq_len_codons
        if start_idx < end_idx:
            c_slice = codon_ids_matrix[sampled_idx, start_idx:end_idx]
            a_slice = aa_ids_matrix[sampled_idx, start_idx:end_idx]
            codon_tokens[:n_sel, w_start:w_end] = torch.from_numpy(c_slice.astype(np.int64))
            aa_tokens[:n_sel, w_start:w_end] = torch.from_numpy(a_slice.astype(np.int64))
        dist_tensor[:n_sel, :n_sel] = sub_dist
        mds_coords[:n_sel, :] = sub_mds
        return codon_tokens, aa_tokens, dist_tensor, mds_coords, padding_mask, torch.tensor(label, dtype=torch.float)


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
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)
        return self.out_proj(out)


class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.phylo_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix, padding_mask=None):
        batch_size, num_species, _ = x.shape
        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = dist_matrix.unsqueeze(1)
        scores = scores - torch.exp(self.phylo_scale.clamp(max=10.0)) * bias
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
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


class PhyloAxialTransformer(nn.Module):
    def __init__(self, num_tokens=66, embed_dim=128, num_heads=8, num_layers=4, window_size=5, max_species=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_species = max_species
        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)
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
        
        # Dual outputs for Multi-Task Learning
        self.mlp_reg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        self.mlp_cls = nn.Sequential(
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
        phylo_pos = self.mds_proj(mds_coords)
        x = x + phylo_pos.unsqueeze(2)
        
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
        
        logits_reg = self.mlp_reg(pooled_repr).squeeze(1)
        logits_cls = self.mlp_cls(pooled_repr).squeeze(1)
        return logits_reg, logits_cls


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, rank_weight=0.5, power=2.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        self.rank_weight = rank_weight
        self.power = power

    def forward(self, y_pred, y_true):
        raw_loss = self.huber(y_pred, y_true)
        # Apply power-law target weighting (default exponent = 2.0)
        weights = torch.pow(y_true + 1.0, self.power)
        loss_regression = (raw_loss * weights).mean()
        if self.rank_weight > 0.0 and y_pred.shape[0] > 1:
            logits_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
            labels_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)
            margin = 0.1
            mask = labels_diff > margin
            if mask.sum() > 0:
                ranking_margin = 0.1
                loss_rank = torch.relu(-logits_diff[mask] + ranking_margin).mean()
            else:
                loss_rank = torch.tensor(0.0, device=y_pred.device)
        else:
            loss_rank = torch.tensor(0.0, device=y_pred.device)
        return loss_regression + self.rank_weight * loss_rank


class MultiTaskLoss(nn.Module):
    def __init__(self, rank_weight=0.5, alpha=1.0, beta=1.0):
        super().__init__()
        self.reg_loss = WeightedHuberLoss(rank_weight=rank_weight)
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_reg, pred_cls, true_reg, true_cls):
        loss_reg = self.reg_loss(pred_reg, true_reg)
        loss_cls = self.cls_loss(pred_cls, true_cls)
        return self.alpha * loss_reg + self.beta * loss_cls


def load_warm_start_weights(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f" -> Warning: Checkpoint path '{checkpoint_path}' not found! Starting training from scratch.")
        return False
    print(f" -> Loading pre-trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    old_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("mlp."):
            new_key = key.replace("mlp.", "mlp_reg.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict, strict=False)
    print(" -> Successfully restored backbone and regression head weights.")
    return True


def train_multitask_model(db_path="meme_results.db", msa_dir="msa", epochs=5, batch_size=128, 
                          lr_cls=2e-4, lr_reg=1e-5, lr_backbone=5e-6, binary_threshold=5.99,
                          subsample_limit=None, device_override=None, cache_path=None, 
                          epoch_size_limit=100000, num_workers=0, max_species=256, 
                          cache_size_limit=0, window_size=1, subsample_species=True, 
                          rank_weight=0.5, resume_checkpoint=None, val_size_limit=0, 
                          checkpoint_path="multitask_transformer_best.pt", alpha=1.0, beta=1.0):
    print("=" * 80)
    print("🚀 MULTI-TASK WARM-START TRAINING ROUTINE")
    print("=" * 80)
    
    loaded_cache = None
    cache_is_dir = False
    
    if not cache_path and os.path.isdir("msa_cache_npz"):
        cache_path = "msa_cache_npz"
        
    if cache_path and os.path.exists(cache_path):
        if os.path.isdir(cache_path):
            print(f" -> Utilizing directory-based NPZ cache from {cache_path}...")
            cache_is_dir = True
        else:
            print(f" -> Loading precomputed alignment cache from {cache_path}...")
            t_cache0 = time.time()
            with gzip.open(cache_path, "rb") as f:
                loaded_cache = pickle.load(f)
            print(f" -> Successfully loaded cache for {len(loaded_cache):,} genes in {time.time()-t_cache0:.2f}s.")

    if device_override:
        device = torch.device(device_override)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f" -> Target training device resolved: {device}")
        
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found!")
        return
        
    print("\n" + "=" * 80)
    print("🚀 DATA PREPARATION & Homology-Aware Splitting")
    print("=" * 80)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT gene_name, site_index, lrt FROM site_results WHERE lrt IS NOT NULL")
    raw_sites = c.fetchall()
    conn.close()
    
    all_genes = sorted(list(set(row[0] for row in raw_sites)))
    random.seed(42)
    random.shuffle(all_genes)
    split_idx = int(len(all_genes) * 0.8)
    train_genes = set(all_genes[:split_idx])
    val_genes = set(all_genes[split_idx:])
    
    train_raw = [row for row in raw_sites if row[0] in train_genes]
    val_raw = [row for row in raw_sites if row[0] in val_genes]
    
    if subsample_limit:
        train_raw = random.sample(train_raw, min(len(train_raw), subsample_limit))
        val_raw = random.sample(val_raw, min(len(val_raw), subsample_limit // 4))
        
    gene_to_variable_sites = {}
    if loaded_cache:
        for gene, cache_data in loaded_cache.items():
            if "variable_sites" in cache_data:
                gene_to_variable_sites[gene] = cache_data["variable_sites"]
    elif cache_is_dir:
        pkl_path = os.path.join(cache_path, "variable_sites.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f_pkl:
                gene_to_variable_sites = pickle.load(f_pkl)
        else:
            unique_genes_in_split = list(set(row[0] for row in raw_sites))
            for gene in unique_genes_in_split:
                npz_path = os.path.join(cache_path, f"{gene}.npz")
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        gene_to_variable_sites[gene] = data["variable_sites"].tolist()
                    except:
                        pass
                        
    train_sites = []
    skipped_train = 0
    for gene, site, lrt in train_raw:
        variable_sites = gene_to_variable_sites.get(gene)
        if variable_sites is None:
            skipped_train += 1
            continue
        if 1 <= site < len(variable_sites) and variable_sites[site]:
            y = math.log(max(0.0, lrt) + 1.0)
            train_sites.append((gene, site, y))
        else:
            skipped_train += 1
            
    val_sites = []
    skipped_val = 0
    for gene, site, lrt in val_raw:
        variable_sites = gene_to_variable_sites.get(gene)
        if variable_sites is None:
            skipped_val += 1
            continue
        if 1 <= site < len(variable_sites) and variable_sites[site]:
            y = math.log(max(0.0, lrt) + 1.0)
            val_sites.append((gene, site, y))
        else:
            skipped_val += 1
            
    if val_size_limit and val_size_limit > 0 and len(val_sites) > val_size_limit:
        random.seed(42)
        val_sites = random.sample(val_sites, val_size_limit)
        
    train_dataset = MSADataset(db_path, msa_dir, train_sites, window_size=window_size, max_species=max_species, cache_dict=loaded_cache if not cache_is_dir else None, cache_dir=cache_path if cache_is_dir else None, cache_size_limit=cache_size_limit, subsample_species=subsample_species)
    val_dataset = MSADataset(db_path, msa_dir, val_sites, window_size=window_size, max_species=max_species, cache_dict=loaded_cache if not cache_is_dir else None, cache_dir=cache_path if cache_is_dir else None, cache_size_limit=cache_size_limit, subsample_species=False)
    
    drop_last_train = len(train_dataset) >= batch_size
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Instantiate architecture
    model = PhyloAxialTransformer(embed_dim=128, num_heads=8, num_layers=4, window_size=window_size, max_species=max_species).to(device)
    
    # Load Warm-Start weights
    weights_loaded = False
    if resume_checkpoint:
        weights_loaded = load_warm_start_weights(model, resume_checkpoint, device)
        
    # Setup parameters optimizer group with differential learning rates
    if weights_loaded:
        print(f" -> Setting up differential learning rates:")
        print(f"    - Classification head (mlp_cls): LR = {lr_cls}")
        print(f"    - Regression head (mlp_reg):     LR = {lr_reg}")
        print(f"    - Shared Backbone:               LR = {lr_backbone}")
        optimizer = torch.optim.AdamW([
            {"params": model.mlp_cls.parameters(), "lr": lr_cls},
            {"params": model.mlp_reg.parameters(), "lr": lr_reg},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("mlp_cls") and not n.startswith("mlp_reg")], "lr": lr_backbone}
        ], weight_decay=1e-2)
    else:
        print(" -> Warning: Starting all layers from scratch with baseline classification learning rate.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_cls, weight_decay=1e-2)
        
    criterion = MultiTaskLoss(rank_weight=rank_weight, alpha=alpha, beta=beta)
    best_spearman = -1.0
    
    print("\n" + "=" * 80)
    print("🚀 TRAINING LOOP STARTING")
    print("=" * 80)
    
    train_active = [s for s in train_sites if s[2] > 0.0]
    train_neutral = [s for s in train_sites if s[2] <= 0.0]
    
    for epoch in range(1, epochs + 1):
        if len(train_active) > 0 and len(train_neutral) > 0:
            sampled_neutral = random.sample(train_neutral, min(len(train_neutral), len(train_active)))
            epoch_sites = train_active + sampled_neutral
            if epoch_size_limit and len(epoch_sites) > epoch_size_limit:
                epoch_sites = random.sample(epoch_sites, epoch_size_limit)
            random.shuffle(epoch_sites)
            train_dataset.sites = epoch_sites
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train, num_workers=num_workers)
        
        model.train()
        train_loss = 0.0
        
        for batch_idx, (codon_tokens, aa_tokens, dist, mds_coords, padding_mask, labels_reg) in enumerate(train_loader):
            codon_tokens = codon_tokens.to(device)
            aa_tokens = aa_tokens.to(device)
            dist = dist.to(device)
            mds_coords = mds_coords.to(device)
            padding_mask = padding_mask.to(device)
            labels_reg = labels_reg.to(device)
            
            # Construct binary target on-the-fly using the threshold
            # labels_reg is log(lrt + 1) -> lrt = exp(reg_label) - 1.0
            lrt_est = torch.exp(labels_reg) - 1.0
            labels_cls = (lrt_est >= binary_threshold).float()
            
            optimizer.zero_grad()
            pred_reg, pred_cls = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
            
            loss = criterion(pred_reg, pred_cls, labels_reg, labels_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader) or batch_idx < 5:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Multi-Task Loss: {loss.item():.4f}", flush=True)
                
        avg_train_loss = train_loss / max(1, len(train_loader))
        
        # Validation Evaluation
        model.eval()
        val_preds_reg = []
        val_preds_cls = []
        val_targets_reg = []
        
        with torch.no_grad():
            for codon_tokens, aa_tokens, dist, mds_coords, padding_mask, labels_reg in val_loader:
                codon_tokens = codon_tokens.to(device)
                aa_tokens = aa_tokens.to(device)
                dist = dist.to(device)
                mds_coords = mds_coords.to(device)
                padding_mask = padding_mask.to(device)
                
                pred_reg, pred_cls = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
                val_preds_reg.extend(pred_reg.cpu().tolist())
                # Sigmoid classification output
                val_preds_cls.extend(torch.sigmoid(pred_cls).cpu().tolist())
                val_targets_reg.extend(labels_reg.tolist())
                
        val_preds_reg = np.array(val_preds_reg)
        val_preds_cls = np.array(val_preds_cls)
        val_targets_reg = np.array(val_targets_reg)
        
        # Determine classification metrics using targets
        val_lrt_est = np.exp(val_targets_reg) - 1.0
        val_targets_cls = (val_lrt_est >= binary_threshold).astype(float)
        
        mse = np.mean((val_preds_reg - val_targets_reg) ** 2)
        val_spearman = 0.0
        if len(np.unique(val_preds_reg)) > 1 and len(np.unique(val_targets_reg)) > 1:
            val_spearman = spearmanr(val_preds_reg, val_targets_reg)[0]
            if np.isnan(val_spearman):
                val_spearman = 0.0
                
        # Simple threshold classification accuracy
        cls_preds_bin = (val_preds_cls >= 0.5).astype(float)
        cls_acc = np.mean(cls_preds_bin == val_targets_cls)
        
        print(f"\n📈 Epoch {epoch} Metrics:")
        print(f"  - Avg Training Multi-Task Loss: {avg_train_loss:.4f}")
        print(f"  - Validation Regression MSE:    {mse:.4f}")
        print(f"  - Validation Spearman rho:      {val_spearman:.4f}")
        print(f"  - Validation Binary Accuracy:   {cls_acc*100.0:.2f}% (Treshold LRT: {binary_threshold})")
        
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_spearman_rho': best_spearman,
                'val_mse': mse,
                'binary_accuracy': cls_acc,
                'binary_threshold': binary_threshold
            }, checkpoint_path)
            print(f"  💾 SUCCESS: Saved best multi-task checkpoint to '{checkpoint_path}' (Spearman rho = {best_spearman:.4f})")
            
    print("\n" + "=" * 80)
    print(f"🎉 MULTI-TASK MODEL TRAINING COMPLETE! Best Spearman rho: {best_spearman:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Task Warm-Start Transfer Learning for selection detection")
    parser.add_argument("--db_path", default="meme_results.db", help="Path to SQLite database")
    parser.add_argument("--msa_dir", default="msa", help="Path to compressed MSAs directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr_cls", type=float, default=2e-4, help="Learning rate for new classification head")
    parser.add_argument("--lr_reg", type=float, default=1e-5, help="Learning rate for regression head")
    parser.add_argument("--lr_backbone", type=float, default=5e-6, help="Learning rate for pre-trained backbone")
    parser.add_argument("--binary_threshold", type=float, default=5.99, help="Raw LRT threshold for positive selection (default: 5.99)")
    parser.add_argument("--device", default=None, help="Force run on device (cpu, mps, cuda)")
    parser.add_argument("--cache_path", default=None, help="Path to precomputed MSA cache")
    parser.add_argument("--epoch_size_limit", type=int, default=200000, help="Max training sites per epoch")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max_species", type=int, default=768, help="Maximum number of species")
    parser.add_argument("--cache_size_limit", type=int, default=512, help="Memory cache size limit")
    parser.add_argument("--window_size", type=int, default=1, help="Window size around target site")
    parser.add_argument("--subsample_species", type=lambda x: (str(x).lower() == 'true'), default=False, help="Subsample species count")
    parser.add_argument("--rank_weight", type=float, default=0.5, help="Pairwise ranking loss weight")
    parser.add_argument("--resume_checkpoint", default=None, required=True, help="Path to continuous selection model checkpoint (.pt) to load pre-trained weights")
    parser.add_argument("--val_size_limit", type=int, default=100000, help="Limit number of validation sites")
    parser.add_argument("--checkpoint_path", default="multitask_transformer_best.pt", help="Output path for best checkpoint")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for regression loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for binary classification loss")
    
    args = parser.parse_args()
    
    train_multitask_model(
        db_path=args.db_path,
        msa_dir=args.msa_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_cls=args.lr_cls,
        lr_reg=args.lr_reg,
        lr_backbone=args.lr_backbone,
        binary_threshold=args.binary_threshold,
        device_override=args.device,
        cache_path=args.cache_path,
        epoch_size_limit=args.epoch_size_limit,
        num_workers=args.num_workers,
        max_species=args.max_species,
        cache_size_limit=args.cache_size_limit,
        window_size=args.window_size,
        subsample_species=args.subsample_species,
        rank_weight=args.rank_weight,
        resume_checkpoint=args.resume_checkpoint,
        val_size_limit=args.val_size_limit,
        checkpoint_path=args.checkpoint_path,
        alpha=args.alpha,
        beta=args.beta
    )
