import torch
import torch._utils
if not hasattr(torch._utils, "_rebuild_device_tensor_from_cpu_tensor"):
    def _rebuild_device_tensor_from_cpu_tensor(data, device, *args, **kwargs):
        req_grad = kwargs.get("requires_grad", False)
        if len(args) > 1 and isinstance(args[-1], bool): req_grad = args[-1]
        return data.to(device).requires_grad_(bool(req_grad))
    torch._utils._rebuild_device_tensor_from_cpu_tensor = _rebuild_device_tensor_from_cpu_tensor

# Bin 0 is exact 0.00 (Neutral), Bins 1..63 span 0.05 to 4.0 in log(1+LRT) space

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
import torch._utils

if not hasattr(torch._utils, '_rebuild_device_tensor_from_cpu_tensor'):
    def _rebuild_device_tensor_from_cpu_tensor(data, device, *args, **kwargs):
        req_grad = kwargs.get('requires_grad', False)
        if len(args) > 1 and isinstance(args[-1], bool):
            req_grad = args[-1]
        return data.to(device).requires_grad_(bool(req_grad))
    torch._utils._rebuild_device_tensor_from_cpu_tensor = _rebuild_device_tensor_from_cpu_tensor


import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def xla_mark_step():
    """Flushes the XLA graph safely using PyTorch-XLA sync API without deprecation warnings."""
    try:
        import torch_xla
        if hasattr(torch_xla, 'sync'):
            torch_xla.sync()
        else:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
    except Exception:
        pass
import random
import numpy as np
import contextlib
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

# We find base-256 byte values
grid_arr = np.arange(256, dtype=np.int32)
b0_grid = grid_arr[:, None, None] * 65536
b1_grid = grid_arr[None, :, None] * 256
b2_grid = grid_arr[None, None, :]
flat_indices = (b0_grid + b1_grid + b2_grid).ravel()

b0_val = (flat_indices // 65536)
b1_val = (flat_indices // 256) % 256
b2_val = flat_indices % 256

# Gaps: contains '-' (45)
gap_mask = (b0_val == 45) | (b1_val == 45) | (b2_val == 45)
CODON_LOOKUP[gap_mask] = 64
AA_LOOKUP[gap_mask] = 21

# Unknown: contains 'N' (78), 'n' (110), '?' (63)
unknown_mask = (b0_val == 78) | (b1_val == 78) | (b2_val == 78) | \
               (b0_val == 110) | (b1_val == 110) | (b2_val == 110) | \
               (b0_val == 63) | (b1_val == 63) | (b2_val == 63)
unknown_mask = unknown_mask & (~gap_mask)
CODON_LOOKUP[unknown_mask] = 65
AA_LOOKUP[unknown_mask] = 22

# Overwrite with exact valid codons
for codon, c_idx in CODON_TO_IDX.items():
    if len(codon) == 3 and '-' not in codon and '?' not in codon:
        for c_str in [codon.upper(), codon.lower()]:
            b_encoded = c_str.encode('ascii')
            idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
            CODON_LOOKUP[idx] = c_idx
            aa = GENETIC_CODE.get(codon.upper(), '?')
            AA_LOOKUP[idx] = AA_TO_IDX.get(aa, 22)


# --- Lightweight Newick Parser and Distance Calculator ---
class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

def parse_newick(newick_str):
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str.strip())  # remove annotations
    newick_str = re.sub(r'\[.*?\]', '', newick_str)            # remove comments
    
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
        coords = np.zeros((N, n_components), dtype=np.float32)
        return coords
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
    if not site_codons:
        return False
    unique_codons_set = set(site_codons)
    # Keep ALL sites with ANY nucleotide/codon variation (synonymous + non-synonymous)
    if len(unique_codons_set) > 1:
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


# --- PyTorch Dataset Loader ---
class MSADataset(Dataset):
    def __init__(self, db_path, msa_dir, sites_list, window_size=1, max_species=256, cache_dict=None, cache_dir=None, cache_size_limit=0, subsample_species=False, max_k=64, prewarm=True):
        self.db_path = db_path
        self.msa_dir = msa_dir
        self.sites = sites_list
        self.window_size = window_size
        self.max_species = max_species
        self.cache_dict = cache_dict
        self.cache_dir = cache_dir
        self.cache_size_limit = cache_size_limit
        self.subsample_species = subsample_species
        self.max_k = max_k
        self.alignment_cache = {}
        
        if prewarm and self.cache_dir and os.path.isdir(self.cache_dir):
            self._prewarm_cache()

    def _prewarm_cache(self):
        unique_genes = list(set(s[0] for s in self.sites))
        target_genes = unique_genes[:self.cache_size_limit] if self.cache_size_limit > 0 else unique_genes
        print(f" -> [Fast Parallel Pre-warmer] Loading {len(target_genes):,} unique gene alignments in parallel across CPU cores...", flush=True)
        t0 = time.time()
        
        from concurrent.futures import ThreadPoolExecutor
        workers = min(32, max(1, os.cpu_count() or 4))
        
        def _load_single(gene_name):
            npz_path = os.path.join(self.cache_dir, f"{gene_name}.npz")
            if not os.path.exists(npz_path):
                return gene_name, None
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
                tree_str = str(data["tree_str"]) if "tree_str" in data else ""
                species_names = data["species_names"].tolist()
                parent_array, branch_lengths = build_tree_topology(tree_str, species_names)
                return gene_name, (
                    data["codon_ids_matrix"],
                    data["aa_ids_matrix"],
                    species_names,
                    dist_tensor_cached,
                    mds_coords_cached,
                    data["variable_sites"].tolist(),
                    selected_indices,
                    data["valid_seqs"],
                    parent_array,
                    branch_lengths
                )
            except Exception:
                return gene_name, None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_load_single, target_genes)
            for gene_name, parsed_data in results:
                if parsed_data is not None:
                    self.alignment_cache[gene_name] = parsed_data

        print(f"    - Successfully pre-warmed {len(self.alignment_cache):,} gene alignments into RAM in {time.time()-t0:.2f}s!", flush=True)

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
                    tree_str = str(data["tree_str"]) if "tree_str" in data else ""
                    species_names = data["species_names"].tolist()
                    parent_array, branch_lengths = build_tree_topology(tree_str, species_names)
                    
                    self.alignment_cache[gene_name] = (
                        data["codon_ids_matrix"],
                        data["aa_ids_matrix"],
                        species_names,
                        dist_tensor_cached,
                        mds_coords_cached,
                        data["variable_sites"].tolist(),
                        selected_indices,
                        data["valid_seqs"],
                        parent_array,
                        branch_lengths
                    )
                    # Limit cache size if cache_size_limit > 0 to prevent memory bloat (especially with multiprocessing dataloading)
                    if self.cache_size_limit > 0 and len(self.alignment_cache) > self.cache_size_limit:
                        first_key = next(iter(self.alignment_cache))
                        self.alignment_cache.pop(first_key)
                        
                    return self.alignment_cache[gene_name]
                except Exception as e:
                    print(f"Error loading NPZ cache for {gene_name}: {e}")
                    
        if self.cache_dict and gene_name in self.cache_dict:
            data = self.cache_dict[gene_name]
            
            # Check if this is the new compact pre-tokenized cache format
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
                
                tree_str = str(data["tree_str"]) if "tree_str" in data else ""
                species_names = list(data["species_names"])
                parent_array, branch_lengths = build_tree_topology(tree_str, species_names)
                
                self.alignment_cache[gene_name] = (
                    data["codon_ids_matrix"],
                    data["aa_ids_matrix"],
                    species_names,
                    dist_tensor_cached,
                    mds_coords_cached,
                    data["variable_sites"],
                    selected_indices,
                    data["valid_seqs"],
                    parent_array,
                    branch_lengths
                )
                return self.alignment_cache[gene_name]
                
            if "seq_dict" in data and "species_names" in data and "dist_arr" in data:
                dist_tensor_cached = torch.from_numpy(data["dist_arr"])
                if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
                    dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
                
                mds_coords_cached = torch.from_numpy(data.get("mds_coords")) if "mds_coords" in data else None
                if mds_coords_cached is not None:
                    if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
                        mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    mds_coords_cached = torch.from_numpy(compute_mds_coordinates(data["dist_arr"], n_components=4))
                    
                var_sites = data.get("variable_sites")
                
                # Precompute diversity selection indices once per gene
                n_spec = len(data["species_names"])
                if n_spec > self.max_species:
                    selected_indices = select_diverse_species_matrix(data["dist_arr"], self.max_species, keep_idx=0)
                else:
                    selected_indices = list(range(n_spec))
                    
                # Pre-tokenize all sequences for this gene into 2D matrices once
                codon_ids_list = []
                aa_ids_list = []
                species_names = data["species_names"]
                seq_dict = data["seq_dict"]
                
                any_seq = next(iter(seq_dict.values())) if seq_dict else ""
                seq_len_codons = len(any_seq) // 3
                
                valid_seqs = np.zeros(n_spec, dtype=bool)
                for i, spec in enumerate(species_names):
                    seq = seq_dict.get(spec, "")
                    if seq:
                        valid_seqs[i] = True
                        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
                        if len(seq_bytes) == seq_len_codons * 3:
                            codons_bytes = seq_bytes.reshape(seq_len_codons, 3)
                            indices = codons_bytes[:, 0].astype(np.int32) * 65536 + codons_bytes[:, 1].astype(np.int32) * 256 + codons_bytes[:, 2].astype(np.int32)
                            c_ids = CODON_LOOKUP[indices]
                            a_ids = AA_LOOKUP[indices]
                        else:
                            c_ids = np.ones(seq_len_codons, dtype=np.int8) * 65
                            a_ids = np.ones(seq_len_codons, dtype=np.int8) * 22
                    else:
                        c_ids = np.ones(seq_len_codons, dtype=np.int8) * 65
                        a_ids = np.ones(seq_len_codons, dtype=np.int8) * 22
                    codon_ids_list.append(c_ids)
                    aa_ids_list.append(a_ids)
                    
                codon_ids_matrix = np.stack(codon_ids_list, axis=0)
                aa_ids_matrix = np.stack(aa_ids_list, axis=0)
                
                tree_str = str(data["tree_str"]) if "tree_str" in data else ""
                parent_array, branch_lengths = build_tree_topology(tree_str, species_names)
                
                self.alignment_cache[gene_name] = (
                    codon_ids_matrix,
                    aa_ids_matrix,
                    species_names,
                    dist_tensor_cached,
                    mds_coords_cached,
                    var_sites,
                    selected_indices,
                    valid_seqs,
                    parent_array,
                    branch_lengths
                )
                return self.alignment_cache[gene_name]
                
        align_path = os.path.join(self.msa_dir, f"{gene_name}.gz")
        if not os.path.exists(align_path):
            return None
            
        taxlabels = []
        sequences = []
        tree_str = None
        in_taxlabels = False
        in_matrix = False
        in_trees = False
        
        try:
            with gzip.open(align_path, 'rt') as f:
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
        except Exception as e:
            print(f"Error loading MSA for {gene_name}: {e}")
            return None
            
        seq_dict = {}
        for label, seq in zip(taxlabels, sequences):
            seq_dict[label] = seq
            
        dist_matrix = {}
        species_names = []
        if tree_str:
            try:
                tree_root = parse_newick(tree_str)
                species_names, dist_matrix = get_patristic_distances(tree_root)
            except Exception as e:
                pass
                
        if not species_names:
            species_names = list(seq_dict.keys())
            dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}
            
        n_spec = len(species_names)
        dist_arr = np.zeros((n_spec, n_spec), dtype=np.float32)
        for i, spec1 in enumerate(species_names):
            for j, spec2 in enumerate(species_names):
                dist_arr[i, j] = dist_matrix.get(spec1, {}).get(spec2, 0.0)
        dist_tensor_cached = torch.from_numpy(dist_arr)
        if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
            dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
            
        mds_coords_np = compute_mds_coordinates(dist_arr, n_components=4)
        mds_coords_cached = torch.from_numpy(mds_coords_np)
        if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
            mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Variable sites computation
        any_seq = next(iter(seq_dict.values())) if seq_dict else ""
        num_codons = len(any_seq) // 3
        species_codons = []
        for spec in species_names:
            seq = seq_dict.get(spec, "")
            codons = [seq[idx*3:idx*3+3].upper() for idx in range(num_codons)]
            species_codons.append(codons)
            
        variable_sites = [False] * (num_codons + 1)
        for site_idx in range(1, num_codons + 1):
            c_idx = site_idx - 1
            site_codons = []
            site_aas = []
            for s_idx in range(n_spec):
                codon = species_codons[s_idx][c_idx]
                if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                    site_codons.append(codon)
                    aa = GENETIC_CODE.get(codon, '?')
                    if aa != '?':
                        site_aas.append(aa)
            variable_sites[site_idx] = is_site_variable(site_codons, site_aas)
            
        # Precompute diversity selection indices once per gene
        if n_spec > self.max_species:
            selected_indices = select_diverse_species_matrix(dist_arr, self.max_species, keep_idx=0)
        else:
            selected_indices = list(range(n_spec))
            
        # Pre-tokenize all sequences for this gene once into 2D matrices
        codon_ids_list = []
        aa_ids_list = []
        valid_seqs = np.zeros(n_spec, dtype=bool)
        for i, spec in enumerate(species_names):
            seq = seq_dict.get(spec, "")
            if seq:
                valid_seqs[i] = True
                seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
                if len(seq_bytes) == num_codons * 3:
                    codons_bytes = seq_bytes.reshape(num_codons, 3)
                    indices = codons_bytes[:, 0].astype(np.int32) * 65536 + codons_bytes[:, 1].astype(np.int32) * 256 + codons_bytes[:, 2].astype(np.int32)
                    c_ids = CODON_LOOKUP[indices]
                    a_ids = AA_LOOKUP[indices]
                else:
                    c_ids = np.ones(num_codons, dtype=np.int8) * 65
                    a_ids = np.ones(num_codons, dtype=np.int8) * 22
            else:
                c_ids = np.ones(num_codons, dtype=np.int8) * 65
                a_ids = np.ones(num_codons, dtype=np.int8) * 22
            codon_ids_list.append(c_ids)
            aa_ids_list.append(a_ids)
            
        codon_ids_matrix = np.stack(codon_ids_list, axis=0)
        aa_ids_matrix = np.stack(aa_ids_list, axis=0)
        parent_array, branch_lengths = build_tree_topology(tree_str if tree_str else "", species_names)
            
        self.alignment_cache[gene_name] = (
            codon_ids_matrix,
            aa_ids_matrix,
            species_names,
            dist_tensor_cached,
            mds_coords_cached,
            variable_sites,
            selected_indices,
            valid_seqs,
            parent_array,
            branch_lengths
        )
        num_cached = len(self.alignment_cache)
        if num_cached % 100 == 0:
            print(f"    [DataLoader Cache] Loaded and parsed {num_cached} unique gene alignments...", flush=True)
        return self.alignment_cache[gene_name]
        
    def __getitem__(self, idx):
        gene_name, site_idx, label = self.sites[idx]
        
        data = self._load_msa(gene_name)
        if data is None:
            dummy_codon = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 65
            dummy_aa = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 22
            dummy_dist = torch.zeros(self.max_species, self.max_species)
            dummy_mds = torch.zeros(self.max_species, 4, dtype=torch.float32)
            dummy_mask = torch.ones(self.max_species, dtype=torch.bool)
            dummy_sub = torch.zeros(32, dtype=torch.long)
            dummy_flags = torch.zeros(32, 4, dtype=torch.float32)
            dummy_lens = torch.ones(32, dtype=torch.float32) * 1e-4
            return dummy_codon, dummy_aa, dummy_dist, dummy_mds, dummy_mask, torch.tensor(label, dtype=torch.float)
            
        codon_ids_matrix, aa_ids_matrix, species_names, dist_tensor_cached, mds_coords_cached, _, selected_indices, valid_seqs, parent_array, branch_lengths = data
        
        # Smart Variant-Aware & Faith's PD Subsampling (100% Mutation Protection)
        if len(selected_indices) > self.max_species:
            # 1. Extract amino acids at target site across all species in gene
            site_aa_all = aa_ids_matrix[selected_indices, site_idx - 1]
            valid_aa = site_aa_all[site_aa_all < 20]
            
            if len(valid_aa) > 0:
                counts = np.bincount(valid_aa)
                consensus_aa = np.argmax(counts)
                variant_mask = (site_aa_all < 20) & (site_aa_all != consensus_aa)
                variant_local_idx = np.where(variant_mask)[0]
            else:
                variant_local_idx = np.array([], dtype=int)
                
            # Always retain reference species [0] + ALL variant-carrying species
            priority_local_set = set([0] + list(variant_local_idx))
            
            if len(priority_local_set) >= self.max_species:
                sampled_local_idx = sorted(list(priority_local_set)[:self.max_species])
            else:
                remaining_local = [idx for idx in range(len(selected_indices)) if idx not in priority_local_set]
                target_n = random.randint(len(priority_local_set), self.max_species) if self.subsample_species else self.max_species
                needed = target_n - len(priority_local_set)
                
                if needed > 0 and len(remaining_local) > 0:
                    D_cache = dist_tensor_cached[:, :, 0] if dist_tensor_cached.dim() == 3 else dist_tensor_cached # Patristic distance matrix
                    priority_global = [selected_indices[i] for i in priority_local_set]
                    rem_global = [selected_indices[i] for i in remaining_local]
                    min_dists = np.min(D_cache[priority_global][:, rem_global], axis=0)
                    
                    selected_rem_local = []
                    for _ in range(min(needed, len(remaining_local))):
                        best_rem_pos = np.argmax(min_dists)
                        selected_rem_local.append(remaining_local[best_rem_pos])
                        added_global = selected_indices[remaining_local[best_rem_pos]]
                        new_dists = D_cache[added_global, rem_global]
                        min_dists = np.minimum(min_dists, new_dists)
                        
                    sampled_local_idx = sorted(list(priority_local_set) + selected_rem_local)
                else:
                    sampled_local_idx = sorted(list(priority_local_set))
                    
            sampled_idx = [selected_indices[i] for i in sampled_local_idx]
        elif self.subsample_species and len(selected_indices) > 10:
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
        padding_mask = torch.ones(self.max_species, dtype=torch.bool) # True means padded
        
        n_sel = len(sampled_idx)
        sampled_valid = valid_seqs[sampled_idx]
        padding_mask[:n_sel] = torch.from_numpy(~sampled_valid)
        
        half_win = self.window_size // 2
        
        # Precomputed fast vectorized slice lookup
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


# --- Block-Diagonal Disentanglement Linear Projection ---
class BlockLinear(nn.Module):
    """
    Block-Diagonal Linear Projection.
    Guarantees 100% mathematical disentanglement between the Codon Synonymous Track
    and the Amino Acid Selection Track. Prevents linear layer cross-mixing of dS noise into dN+ features.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_in = in_features // 2
        self.half_out = out_features // 2
        self.block_codon = nn.Linear(self.half_in, self.half_out, bias=bias)
        self.block_aa = nn.Linear(self.half_in, self.half_out, bias=bias)

    def forward(self, x):
        # x: [..., in_features]
        x_codon = x[..., :self.half_in]
        x_aa = x[..., self.half_in:]
        out_codon = self.block_codon(x_codon)
        out_aa = self.block_aa(x_aa)
        return torch.cat([out_codon, out_aa], dim=-1)


# --- Stable Attention Module with Block-Diagonal Disentanglement ---
class StableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = BlockLinear(embed_dim, embed_dim)
        self.k_proj = BlockLinear(embed_dim, embed_dim)
        self.v_proj = BlockLinear(embed_dim, embed_dim)
        self.out_proj = BlockLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_seq_len, _ = query.shape
        k_seq_len = key.shape[1]
        
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)
        
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


# --- Custom Row Attention with Block-Diagonal Disentanglement & Learnable Phylogenetic Bias ---
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = BlockLinear(embed_dim, embed_dim)
        self.k_proj = BlockLinear(embed_dim, embed_dim)
        self.v_proj = BlockLinear(embed_dim, embed_dim)
        
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
        
        # Learnable Per-Head Pairwise Mask Attention Biases (Initialized to 0.0)
        self.nonsyn_head_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.syn_head_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.out_proj = BlockLinear(embed_dim, embed_dim)
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
        
        self.linear1 = BlockLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = BlockLinear(dim_feedforward, d_model)
        
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



# --- 13-Bin Ordinal Likelihood Partition & Soft-Bin Expectation Decoder ---
BIN_EDGES_9 = [0.0, 0.2738, 1.8272, 3.1248, 4.4537, 5.7987, 7.5909, 12.1310, 16.6963, 21.2737, 100.0]
BIN_EDGES_12 = [0.0, 0.2738, 0.7500, 1.2500, 1.8272, 2.4500, 3.1248, 4.4537, 5.7987, 7.5909, 12.1310, 16.6963, 21.2737, 100.0]
BIN_EDGES = BIN_EDGES_12
BIN_MEANS = torch.tensor([0.00, 0.51, 1.00, 1.54, 2.14, 2.79, 3.79, 5.13, 6.69, 9.86, 14.41, 18.98, 35.00])

def sparsemax(logits, dim=-1):
    """
    TPU-friendly Sparsemax (Martins & Astudillo, ICML 2016).
    Projects logits onto the probability simplex, truncating low-scoring tail values to EXACTLY 0.0.
    Uses 100% static tensor shapes and ops to prevent PyTorch-XLA recompilation graph breaks.
    """
    input_sorted, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum = torch.cumsum(input_sorted, dim=dim)
    
    num_elements = logits.shape[dim]
    k_range = torch.arange(1, num_elements + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    k_range = k_range.view(*shape)
    
    bound = 1.0 + k_range * input_sorted
    is_greater = (bound > cumsum).float()
    
    k_max = torch.max(is_greater * k_range, dim=dim, keepdim=True)[0]
    tau = (torch.gather(cumsum, dim, k_max.long() - 1) - 1.0) / k_max
    
    return torch.relu(logits - tau)


def decode_soft_ordinal_lrt(logits_ordinal, bin_edges=None, temperature=1.0):
    """
    Rigorously decodes continuous LRT prediction from CORAL cumulative ordinal logits (9 or 12 heads)
    using Smooth Expected Value Interpolation: E[LRT] = sum_{m=0}^K P(Bin m) * midpoints[m].
    This eliminates discrete cliff-edge step jumps (p >= 0.50) and provides 100% smooth, calibrated predictions.
    
    logits_ordinal: [..., K] cumulative logits for thresholds P(y > edge_{k+1})
    """
    num_heads = logits_ordinal.shape[-1]
    if bin_edges is None:
        bin_edges = BIN_EDGES_12 if num_heads == 12 else BIN_EDGES_9
        
    device = logits_ordinal.device
    edges = torch.tensor(bin_edges, device=device, dtype=logits_ordinal.dtype)
    
    # Cumulative probabilities P(y > threshold_k) for k in 0..num_heads-1
    p_cum = torch.sigmoid(logits_ordinal / temperature)
    
    # Probability mass per bin P(Bin m) for m in 0..K
    p0 = p_cum[..., 0:1]
    p_diffs = p_cum[..., :-1] - p_cum[..., 1:]
    p_last = p_cum[..., -1:]
    
    bin_probs = torch.cat([1.0 - p0, p_diffs, p_last], dim=-1)
    
    num_bins = len(bin_edges) - 1
    midpoints = torch.zeros(num_bins, device=device, dtype=logits_ordinal.dtype)
    midpoints[0] = 0.0  # Background null bin
    for i in range(1, num_bins - 1):
        midpoints[i] = (edges[i] + edges[i+1]) / 2.0
    midpoints[-1] = edges[-2] + 5.0
    
    y_continuous_lrt = (bin_probs * midpoints).sum(dim=-1)
    return y_continuous_lrt, p_cum


class RankConsistentCoralHead(nn.Module):
    """
    Guarantees strict logit monotonicity: z_0 > z_1 > z_2 > ... for 100% of inputs
    by projecting features with a shared scalar projection w^T h and adding monotonically decreasing
    threshold cutoffs: b_k = b_0 - sum_{j=1}^k softplus(theta_j).
    Supports both 9-head (legacy) and 12-head (empirical prior initialized) setups.
    """
    def __init__(self, embed_dim, num_thresholds=12):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.feature_proj = nn.Linear(embed_dim, 1, bias=False)  # Shared projection scalar w^T h
        nn.init.normal_(self.feature_proj.weight, mean=0.1, std=0.02)
        
        if num_thresholds == 9:
            self.b0 = nn.Parameter(torch.tensor(3.0))
            self.theta_steps = nn.Parameter(torch.ones(8) * 0.5)
        else:
            # Empirical 90:10 natural prior initialization for b0 (logit(0.10) = -2.1972) and theta_steps (12 heads):
            self.b0 = nn.Parameter(torch.tensor(-2.1972))
            init_thetas = torch.tensor([-0.0679, -0.6850, -0.9210, -1.0795, -1.1709, -0.5861, -0.7915, -0.6470, 0.2742, -0.0837, -0.3517])
            self.theta_steps = nn.Parameter(init_thetas)

    def forward(self, x):
        # x: [batch_size, embed_dim]
        proj = self.feature_proj(x)  # [batch_size, 1]
        
        # Compute monotonic decreasing cutoffs: b_0 > b_1 > ...
        step_decrements = F.softplus(self.theta_steps)  # [num_thresholds - 1] > 0
        cum_decrements = torch.cumsum(step_decrements, dim=0)  # [num_thresholds - 1]
        
        # b_all = [b_0, b_0 - d_1, b_0 - d_2, ...]
        b_cutoffs = torch.cat([self.b0.unsqueeze(0), self.b0 - cum_decrements], dim=0).to(dtype=proj.dtype)  # [num_thresholds]
        
        # z_k = proj + b_k
        logits = proj + b_cutoffs.unsqueeze(0)  # [batch_size, num_thresholds]
        return logits


class PhyloAxialTransformer(nn.Module):
    def __init__(self, num_tokens=66, embed_dim=128, num_heads=8, num_layers=4, window_size=1, max_species=256, dropout=0.1, max_k=32, num_streams=4, pure_coral=False, use_ordinal=True, num_thresholds=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_species = max_species
        self.num_streams = num_streams
        self.pure_coral = pure_coral
        self.use_ordinal = use_ordinal
        self.use_aa_embeddings = True

        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, embed_dim) * 0.02)
        self.mds_proj = nn.Linear(4, embed_dim)
        self.mds_embed = nn.Linear(4, embed_dim)
        self.dist_decay = nn.Parameter(torch.tensor(1.0))

        self.col_layers = nn.ModuleList([
            StableTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2*embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.col_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.lrt_ordinal_head = RankConsistentCoralHead(embed_dim, num_thresholds=num_thresholds)
        self.species_attn_query = nn.Linear(embed_dim, 1)
        self.stream_fusion = nn.Sequential(
            BlockLinear(num_streams * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.lrt_reg_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Head 2: alpha (dS synonymous rate) -> softplus
        self.alpha_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        # Head 3: beta_neg (purifying non-synonymous rate) -> softplus
        self.beta_neg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        # Head 4: beta_pos (positive selection non-synonymous rate) -> softplus
        self.beta_pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        # Head 5: p_neg (proportion of purifying branches) -> sigmoid
        self.p_neg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        nn.init.constant_(self.alpha_head[-1].bias, 1.0)
        nn.init.constant_(self.beta_neg_head[-1].bias, 0.5)
        nn.init.constant_(self.beta_pos_head[-1].bias, 1.0)
        nn.init.constant_(self.p_neg_head[-1].bias, 0.0)

    def forward(self, msa_codons, msa_aas, dist_matrix, mds_coords, padding_mask=None):
        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2
        
        # Ensure padding_mask is always a canonical boolean tensor to keep XLA graph topology static
        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, num_species, dtype=torch.bool, device=msa_codons.device)
        
        # Pre-compute genetic code pairwise attention masks at central site
        c_cent = msa_codons[:, :, central_idx]  # [batch_size, num_species]
        a_cent = msa_aas[:, :, central_idx]     # [batch_size, num_species]
        
        valid = (c_cent < 64) & (a_cent < 21) & (~padding_mask.bool())
            
        v_float = valid.float()
        v_pair = v_float.unsqueeze(1) * v_float.unsqueeze(2)  # [batch_size, num_species, num_species]
        
        # Static matrix identity mask (Zero PyTorch-XLA recompilation)
        diag_mask = torch.eye(num_species, device=c_cent.device, dtype=v_float.dtype).unsqueeze(0)
        pair_mask = v_pair * (1.0 - diag_mask)
        
        a_diff = (a_cent.unsqueeze(1) != a_cent.unsqueeze(2)).float() * pair_mask
        c_diff = (c_cent.unsqueeze(1) != c_cent.unsqueeze(2)).float()
        a_eq = (a_cent.unsqueeze(1) == a_cent.unsqueeze(2)).float()
        c_syn = (c_diff * a_eq) * pair_mask
        
        nonsyn_mask = a_diff
        syn_mask = c_syn
        
        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        
        phylo_pos = self.mds_proj(mds_coords)
        x = x + phylo_pos.unsqueeze(2)
        
        padding_mask_dup = padding_mask.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_species)
        nonsyn_mask_dup = nonsyn_mask.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species)
        syn_mask_dup = syn_mask.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species)

        for i in range(len(self.col_layers)):
            col_in = x.reshape(batch_size * num_species, window_size, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x = col_out.reshape(batch_size, num_species, window_size, self.embed_dim)
            
            row_in = x.transpose(1, 2).contiguous().view(batch_size * window_size, num_species, self.embed_dim)
            # Relative Tree Scale Distance Normalization (makes PhyloRowAttention scale-invariant for short trees like T=1.41)
            dist_pairs = dist_matrix * pair_mask
            mean_dist = dist_pairs.sum(dim=(-2, -1), keepdim=True) / pair_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            norm_dist_matrix = dist_matrix / (mean_dist + 1e-4)
            
            if norm_dist_matrix.dim() == 4:
                dist_dup = norm_dist_matrix.unsqueeze(1).expand(-1, window_size, -1, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species, norm_dist_matrix.shape[-1])
            else:
                dist_dup = norm_dist_matrix.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species)
            
            row_out = self.row_layers[i](row_in, dist_dup, padding_mask_dup, nonsyn_mask_dup, syn_mask_dup)
            row_out = self.row_norms[i](row_in + row_out)
            x = row_out.reshape(batch_size, window_size, num_species, self.embed_dim).transpose(1, 2)
            
        site_repr = x[:, :, central_idx, :]  # [batch_size, num_species, embed_dim]
        
        # Multi-Strategy Species Representation Pooling across taxa (Mean, Max, Sparsemax Attention, Factorized AA Difference)
        valid_mask = (~padding_mask).float().unsqueeze(-1)
        species_counts = valid_mask.sum(dim=1).clamp(min=1.0)
        
        # 1. Masked Mean Pooling across species
        mean_pooled = (site_repr * valid_mask).sum(dim=1) / species_counts
        
        # 2. Masked Max Pooling across species
        site_repr_masked = site_repr.masked_fill(padding_mask.unsqueeze(-1), -1e4)
        max_pooled = torch.max(site_repr_masked, dim=1)[0]
        
        # 3. TPU-Friendly Sparsemax Attention Pooling across species (Exact zero tail truncation)
        attn_logits = self.species_attn_query(site_repr).squeeze(-1)
        attn_logits = attn_logits.masked_fill(padding_mask, -1e4)
        attn_weights = sparsemax(attn_logits, dim=-1).unsqueeze(-1)
        attn_pooled = (site_repr * attn_weights).sum(dim=1)
        
        # 4. Tree-Weighted Hybrid Amino Acid Difference + Laplacian Variance Stream
        # Preserves baseline LRT range scale + 159x signal jump on 1D ladder toggling bursts
        aa_dim_half = self.embed_dim // 2
        aa_site_repr = site_repr[:, :, aa_dim_half:] # [batch_size, num_species, embed_dim//2]
        
        # 4a. Original Factorized AA Difference (First 128 dims: preserves baseline LRT range scale 100%)
        aa_mean_pooled = (aa_site_repr * valid_mask).sum(dim=1) / species_counts
        aa_site_masked = aa_site_repr.masked_fill(padding_mask.unsqueeze(-1), -1e4)
        aa_max_pooled = torch.max(aa_site_masked, dim=1)[0]
        diff_aa_half = F.relu(aa_max_pooled - aa_mean_pooled)
        
        # 4b. Tree-Weighted Laplacian Variance (Second 128 dims: Pairwise feature diff squared / normalized tree distance)
        # LayerNorm bounds feature magnitude to scale ~11.3, preventing gradient explosion and prediction collapse
        mean_dist = dist_matrix.mean(dim=[-2, -1], keepdim=True).unsqueeze(-1)
        dist_norm = dist_matrix.unsqueeze(-1) / (mean_dist + 1e-4)
        
        feat_diff_sq = (aa_site_repr.unsqueeze(2) - aa_site_repr.unsqueeze(1)) ** 2
        valid_pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
        
        phylo_rate = (feat_diff_sq / (dist_norm + 1e-4)) * valid_pair_mask
        phylo_var_raw = phylo_rate.mean(dim=[1, 2])
        phylo_var_half = F.layer_norm(phylo_var_raw, (aa_dim_half,))
        
        # Stream 4 combines 128d baseline diff_aa_half + 128d LayerNorm phylo_var_half into full 256d Stream 4
        stream4_pooled = torch.cat([diff_aa_half, phylo_var_half], dim=-1)
        
        # Stream Fusion: Projects concatenated [Mean, Max, Attn, Tree_Weighted_Stream4] representations
        if getattr(self, 'num_streams', 4) == 4:
            pooled_repr = self.stream_fusion(torch.cat([mean_pooled, max_pooled, attn_pooled, stream4_pooled], dim=-1))
        else:
            pooled_repr = self.stream_fusion(torch.cat([mean_pooled, max_pooled, attn_pooled], dim=-1))
        
        # 10-Bin Ordinal LRT Logits, Direct Continuous LRT Regression & Evolutionary Rates
        logits_lrt_ordinal = self.lrt_ordinal_head(pooled_repr)
        y_lrt_direct = F.softplus(self.lrt_reg_head(pooled_repr).squeeze(-1))
        y_alpha = F.softplus(self.alpha_head(pooled_repr).squeeze(1))
        y_beta_neg = F.softplus(self.beta_neg_head(pooled_repr).squeeze(1))
        y_beta_pos = F.softplus(self.beta_pos_head(pooled_repr).squeeze(1))
        y_p_neg = torch.sigmoid(self.p_neg_head(pooled_repr).squeeze(1))
        
        if self.training:
            return (logits_lrt_ordinal, y_lrt_direct), y_alpha, y_beta_neg, y_beta_pos, y_p_neg
        else:
            if getattr(self, 'pure_coral', False) or getattr(self, 'use_ordinal', True):
                y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)
                return y_lrt_soft, y_alpha, y_beta_neg, y_beta_pos, y_p_neg
            else:
                y_lrt_direct_raw = torch.expm1(y_lrt_direct)
                return y_lrt_direct_raw, y_alpha, y_beta_neg, y_beta_pos, y_p_neg


def unpack_model_outputs(model_outputs):
    """
    Unpacks model outputs into (logits_lrt_ordinal, y_lrt_direct, y_alpha, y_beta_neg, y_beta_pos, y_p_neg)
    handling both nested training tuple outputs and 2D Tensor outputs.
    """
    if isinstance(model_outputs, (tuple, list)):
        if len(model_outputs) == 5:
            head0, y_alpha, y_beta_neg, y_beta_pos, y_p_neg = model_outputs
            if isinstance(head0, (tuple, list)):
                logits_lrt_ordinal, y_lrt_direct = head0
            else:
                logits_lrt_ordinal = head0
                y_lrt_direct = torch.zeros_like(head0[:, 0])
            return logits_lrt_ordinal, y_lrt_direct, y_alpha, y_beta_neg, y_beta_pos, y_p_neg
        elif len(model_outputs) == 6:
            return model_outputs
    elif torch.is_tensor(model_outputs):
        if model_outputs.dim() == 2 and model_outputs.shape[-1] == 5:
            y_lrt = model_outputs[:, 0]
            y_alpha = model_outputs[:, 1]
            y_beta_neg = model_outputs[:, 2]
            y_beta_pos = model_outputs[:, 3]
            y_p_neg = model_outputs[:, 4]
            return y_lrt, y_lrt, y_alpha, y_beta_neg, y_beta_pos, y_p_neg
        elif model_outputs.dim() == 1:
            return model_outputs, model_outputs, torch.zeros_like(model_outputs), torch.zeros_like(model_outputs), torch.zeros_like(model_outputs), torch.zeros_like(model_outputs)
            
    raise ValueError(f"Unrecognized model outputs format: {type(model_outputs)}")


# --- Fused Joint Multi-Task Loss with Physical Coupling & Uncertainty Weighting ---
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, rank_weight=0.20, phys_weight=0.50, power=1.5, loss_type="huber", rank_threshold=3.0, std_weight=0.0, bce_weight=1.0):
        super().__init__()
        self.delta = delta
        self.power = power
        self.rank_weight = rank_weight
        self.phys_weight = phys_weight
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        # 5 Learnable homoscedastic uncertainty log-variance parameters (Kendall et al., CVPR 2018)
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, y_pred, y_true):
        logits_ordinal, y_lrt_pred, y_alpha_pred, y_beta_neg_pred, y_beta_pos_pred, y_p_neg_pred = unpack_model_outputs(y_pred)
        y_pred_stacked = torch.stack([y_lrt_pred, y_alpha_pred, y_beta_neg_pred, y_beta_pos_pred, y_p_neg_pred], dim=-1)
        y_true_cast = y_true.to(y_pred_stacked.dtype)
        y_lrt_true = y_true_cast[:, 0]
        
        # 1. Focal Weighting for strong selection LRT sites
        pos_y = F.relu(y_lrt_true)
        focal_weights = 1.0 + 2.0 * (pos_y * torch.sqrt(pos_y))
        
        # 2. Vectorized 5-Target Huber Losses [B, 5]
        huber_all = self.huber(y_pred_stacked, y_true_cast)
        huber_all[:, 0] = huber_all[:, 0] * focal_weights
        
        # 3. Dynamic Uncertainty-Weighted Multi-Task Loss: sum_k [ exp(-log_var_k) * L_k + 0.5 * log_var_k ]
        log_vars_cast = self.log_vars.to(device=y_pred_stacked.device, dtype=y_pred_stacked.dtype)
        inv_vars = torch.exp(-log_vars_cast)
        loss_tasks = torch.mean(huber_all * inv_vars.unsqueeze(0), dim=0) + 0.5 * log_vars_cast
        loss_multitask = torch.sum(loss_tasks)
        
        # 4. Biological Physical Inequality Penalties (Targeted to Ground-Truth Neutral Sites):
        # Mask to ground-truth neutral sites (y_lrt_true <= 0.1) so physical penalty never suppresses True Positives during parameter warmup
        neutral_mask = (y_lrt_true <= 0.1).to(y_pred_stacked.dtype)
        
        # Penalty A: Penalize any predicted LRT > 0 when beta^+ <= alpha on neutral sites
        no_selection_penalty = neutral_mask * F.relu(y_alpha_pred - y_beta_pos_pred) * y_lrt_pred
        
        # Penalty B: Penalize any predicted LRT > 0 when p^- -> 1.0 (prop_pos -> 0) on neutral sites
        prop_pos_pred = F.relu(1.0 - y_p_neg_pred)
        zero_prop_penalty = neutral_mask * (1.0 - prop_pos_pred) * F.relu(y_beta_pos_pred - y_alpha_pred) * y_lrt_pred
        
        loss_physical = torch.mean(no_selection_penalty + zero_prop_penalty)
        
        # 5. Selection Boost Under-Prediction Penalty (Forces High Sensitivity on Selection Sites)
        pos_mask = (y_lrt_true > 0.5).to(y_pred_stacked.dtype)
        underpred_penalty = pos_mask * (F.relu(y_lrt_true - y_lrt_pred) ** 2)
        loss_pos_boost = torch.mean(underpred_penalty)
        
        # 6. Margin Ranking Loss
        if len(y_lrt_true) > 1:
            diff_true = y_lrt_true[1:] - y_lrt_true[:-1]
            diff_pred = y_lrt_pred[1:] - y_lrt_pred[:-1]
            margin_target = F.relu(diff_true - 0.05)
            margin_penalty = F.relu(diff_true - diff_pred)
            loss_rank = torch.mean(margin_target * margin_penalty)
        else:
            loss_rank = torch.tensor(0.0, device=y_pred_stacked.device, dtype=y_pred_stacked.dtype)
            
        loss_total = loss_multitask + self.phys_weight * loss_physical + self.rank_weight * loss_rank + 1.0 * loss_pos_boost
        return loss_total

class FocalCoralOrdinalLoss(nn.Module):
    """
    Calibrated Cumulative Ordinal Loss (CORAL) for 6-bin LRT classification
    with unweighted BCE (gamma_focal=0.0) to ensure exact probability calibration on NULL alignments.
    Combined with Huber loss for evolutionary rate estimation [alpha, beta_neg, beta_pos, p_neg].
    """
    def __init__(self, gamma_focal=0.0, pos_weights=None, phys_weight=0.50, pure_coral=False):
        super().__init__()
        self.gamma_focal = gamma_focal
        self.phys_weight = phys_weight
        self.pure_coral = pure_coral
        self.huber = nn.HuberLoss(reduction='none')
        # 5 Learnable homoscedastic uncertainty log-variance parameters
        self.log_vars = nn.Parameter(torch.zeros(5))
        
        # Calibrated Focal-CORAL ordinal loss (unweighted BCE ensures un-biased probability calibration):
        # 12 Threshold Tasks derived from MEME.bf exact Chi^2 mixture distribution:
        # t0: >0.2738 (p<=0.50), t1: >0.7500 (p<=0.38 NEW1), t2: >1.2500 (p<=0.28 NEW2),
        # t3: >1.8272 (p<=0.20), t4: >2.4500 (p<=0.14 NEW3), t5: >3.1248 (p<=0.10 Tier 2),
        # t6: >4.4537 (p<=0.05 Tier 1), t7: >5.7987 (p<=0.025), t8: >7.5909 (p<=0.01),
        # t9: >12.1310 (p<=0.001), t10: >16.6963 (p<=0.0001), t11: >21.2737 (p<=1e-5)
        if pos_weights is None:
            pos_weights = [1.0] * 12
        self.register_buffer('pos_weights', torch.tensor(pos_weights, dtype=torch.float32))

    def get_ordinal_targets(self, y_lrt_true):
        # Maps continuous log(1 + LRT) ground truth to 12 binary cumulative targets
        t0  = (y_lrt_true > math.log(1.0 + 0.2738)).to(y_lrt_true.dtype)   # raw LRT > 0.2738 (p <= 0.50)
        t1  = (y_lrt_true > math.log(1.0 + 0.7500)).to(y_lrt_true.dtype)   # raw LRT > 0.7500 (p <= 0.38 NEW 1)
        t2  = (y_lrt_true > math.log(1.0 + 1.2500)).to(y_lrt_true.dtype)   # raw LRT > 1.2500 (p <= 0.28 NEW 2)
        t3  = (y_lrt_true > math.log(1.0 + 1.8272)).to(y_lrt_true.dtype)   # raw LRT > 1.8272 (p <= 0.20)
        t4  = (y_lrt_true > math.log(1.0 + 2.4500)).to(y_lrt_true.dtype)   # raw LRT > 2.4500 (p <= 0.14 NEW 3)
        t5  = (y_lrt_true > math.log(1.0 + 3.1248)).to(y_lrt_true.dtype)   # raw LRT > 3.1248 (p <= 0.10 Tier 2)
        t6  = (y_lrt_true > math.log(1.0 + 4.4537)).to(y_lrt_true.dtype)   # raw LRT > 4.4537 (p <= 0.05 Nominal)
        t7  = (y_lrt_true > math.log(1.0 + 5.7987)).to(y_lrt_true.dtype)   # raw LRT > 5.7987 (p <= 0.025 Tier 1)
        t8  = (y_lrt_true > math.log(1.0 + 7.5909)).to(y_lrt_true.dtype)   # raw LRT > 7.5909 (p <= 0.01)
        t9  = (y_lrt_true > math.log(1.0 + 12.1310)).to(y_lrt_true.dtype)  # raw LRT > 12.1310 (p <= 0.001)
        t10 = (y_lrt_true > math.log(1.0 + 16.6963)).to(y_lrt_true.dtype)  # raw LRT > 16.6963 (p <= 0.0001)
        t11 = (y_lrt_true > math.log(1.0 + 21.2737)).to(y_lrt_true.dtype)  # raw LRT > 21.2737 (p <= 1e-5)
        return torch.stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11], dim=-1)

    def focal_bce_with_logits(self, logits, targets):
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # p_t: probability of true class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma_focal
        
        # Apply escalating pos_weight for positive targets (targets == 1)
        pos_w = self.pos_weights.to(device=logits.device, dtype=logits.dtype)
        w = targets * pos_w.unsqueeze(0) + (1.0 - targets) * 1.0
        
        focal_bce = focal_factor * bce * w
        return torch.mean(focal_bce)

    def forward(self, logits_and_rates, y_true):
        logits_lrt_ordinal, y_lrt_direct, y_alpha_pred, y_beta_neg_pred, y_beta_pos_pred, y_p_neg_pred = unpack_model_outputs(logits_and_rates)
        
        y_true_cast = y_true.to(logits_lrt_ordinal.dtype)
        y_lrt_true = y_true_cast[:, 0]
        y_rates_true = y_true_cast[:, 1:] # [alpha, beta_neg, beta_pos, p_neg]
        
        # 1. Focal-CORAL Cumulative Ordinal Loss on 6 LRT Bins
        ordinal_targets = self.get_ordinal_targets(y_lrt_true)
        loss_ordinal_focal = self.focal_bce_with_logits(logits_lrt_ordinal, ordinal_targets)
        
        # 2. Direct Continuous LRT Huber Loss (weighted by raw LRT to reach LRT > 100+)
        raw_lrt = torch.expm1(y_lrt_true)
        lrt_weights = 1.0 + 0.5 * torch.clamp(raw_lrt, max=100.0)
        loss_lrt_reg = torch.mean(lrt_weights * self.huber(y_lrt_direct.squeeze(-1), y_lrt_true))
        
        # 3. Huber Loss on Evolutionary Rates [alpha, beta_neg, beta_pos, p_neg]
        y_rates_pred = torch.stack([y_alpha_pred.squeeze(-1), y_beta_neg_pred.squeeze(-1), y_beta_pos_pred.squeeze(-1), y_p_neg_pred.squeeze(-1)], dim=-1)
        huber_rates = torch.mean(self.huber(y_rates_pred, y_rates_true), dim=0)
        
        # 4. Dynamic Uncertainty Weighting
        log_vars_cast = self.log_vars.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
        inv_vars = torch.exp(-log_vars_cast)
        
        all_losses = torch.cat([loss_ordinal_focal.unsqueeze(0), huber_rates], dim=0)
        lrt_reg_weight = 0.0 if getattr(self, 'pure_coral', False) else 0.5
        loss_multitask = torch.sum(all_losses * inv_vars + 0.5 * log_vars_cast) + lrt_reg_weight * loss_lrt_reg
        
        total_loss = loss_multitask
        return total_loss


class RateWeightedLoss(nn.Module):
    def __init__(self, gamma=1.5, underpredict_penalty=2.0, phys_weight=0.50):
        super().__init__()
        self.gamma = gamma
        self.underpredict_penalty = underpredict_penalty
        self.phys_weight = phys_weight
        self.huber = nn.HuberLoss(reduction='none')
        # 5 Learnable homoscedastic uncertainty log-variance parameters
        self.log_vars = nn.Parameter(torch.zeros(5))
        
    def forward(self, y_pred, y_true):
        logits_ordinal, y_lrt_pred, y_alpha_pred, y_beta_neg_pred, y_beta_pos_pred, y_p_neg_pred = unpack_model_outputs(y_pred)
        y_pred_stacked = torch.stack([y_lrt_pred, y_alpha_pred, y_beta_neg_pred, y_beta_pos_pred, y_p_neg_pred], dim=-1)
        y_true_cast = y_true.to(y_pred_stacked.dtype)
        
        y_lrt_true = y_true_cast[:, 0]
        
        # 1. Quadratic Peak Rate-Weighted Focal Loss on LRT (prevents peak LRT compression on LRT > 20 sites)
        base_mse = (y_lrt_pred - y_lrt_true) ** 2
        rate_weights = 1.0 + self.gamma * y_lrt_true + 0.5 * (y_lrt_true ** 2)
        loss_lrt = rate_weights * base_mse
        
        # 2. Multi-Task Huber Losses for [alpha, beta_neg, beta_pos, p_neg]
        huber_others = self.huber(y_pred_stacked[:, 1:], y_true_cast[:, 1:])
        huber_all = torch.cat([loss_lrt.unsqueeze(-1), huber_others], dim=-1)
        
        # 3. Dynamic Uncertainty Weighting: sum_k [ exp(-log_var_k) * L_k + 0.5 * log_var_k ]
        log_vars_cast = self.log_vars.to(device=y_pred_stacked.device, dtype=y_pred_stacked.dtype)
        inv_vars = torch.exp(-log_vars_cast)
        loss_tasks = torch.mean(huber_all * inv_vars.unsqueeze(0), dim=0) + 0.5 * log_vars_cast
        loss_multitask = torch.sum(loss_tasks)
        
        # 4. Biological Physical Rate Inequality Penalties (Targeted to Neutral Sites):
        neutral_mask = (y_lrt_true <= 0.1).to(y_pred_stacked.dtype)
        # On ground-truth neutral sites, beta_pos MUST NOT exceed alpha
        neutral_rate_violation = neutral_mask * F.relu(y_beta_pos_pred - y_alpha_pred) * (1.0 + y_lrt_pred)
        # Directly penalize non-zero LRT predictions on ground-truth neutral sites
        neutral_overpred_penalty = neutral_mask * (y_lrt_pred ** 2)
        loss_physical = torch.mean(neutral_rate_violation + neutral_overpred_penalty)
        
        # 5. Asymmetric Under-Prediction Penalty on True Selection Sites (y_lrt_true >= 1.57)
        underpredict_mask = ((y_lrt_true >= 1.57) & (y_lrt_pred < y_lrt_true)).to(y_pred_stacked.dtype)
        underpredict_loss = torch.mean(underpredict_mask * ((y_lrt_true - y_lrt_pred) ** 2))
        
        total_loss = loss_multitask + (self.phys_weight * loss_physical) + (self.underpredict_penalty * underpredict_loss)
        return total_loss


# --- Full Training & Validation Routine ---
def train_full_model(db_path="meme_results.db", msa_dir="msa", epochs=5, batch_size=1536, micro_batch_size=256, lr=2e-4, subsample_limit=None, loss_type="coral", target_transform="none", device_override=None, cache_path=None, epoch_size_limit=100000, num_workers=0, max_species=256, cache_size_limit=0, window_size=1, subsample_species=True, rank_weight=0.5, weights_path=None, subsample_val_limit=None, save_path="selection_transformer_best.pt", loss_power=2.0, rank_threshold=3.0, std_weight=0.0, max_k=64, embed_dim=128, num_layers=4, num_heads=8, profile=False, pure_coral=False):
    print("=" * 80)
    print("🚀 STEP 1: INITIALIZING HARDWARE ACCELERATION")
    print("=" * 80)
    
    loaded_cache = None
    cache_is_dir = False
    
    # Auto-resolve cache_path if not specified but the default NPZ directory exists locally
    if not cache_path and os.path.isdir("msa_cache_npz"):
        print(" -> Cache path not specified, but 'msa_cache_npz' directory found locally. Auto-selecting it to prevent slow fallback.")
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

    # Auto-detect hardware accelerator (TPU v2/v3/v4/v6e via PyTorch-XLA, CUDA GPU, or MPS)
    if device_override in ('xla', 'tpu') or (device_override is None and ('PJRT_DEVICE' in os.environ or 'XRT_TPU_CONFIG' in os.environ or 'TPU_NAME' in os.environ)):
        try:
            import torch_xla
            if hasattr(torch_xla, 'device'):
                device = torch_xla.device()
            else:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
            print(" -> Utilizing Google TPU backend (PyTorch-XLA)!")
        except Exception:
            if device_override:
                device = torch.device(device_override)
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
    elif device_override:
        device = torch.device(device_override)
        print(f" -> Forcing device selection: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(" -> Detected NVIDIA GPU! Utilizing CUDA backend.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" -> Detected Apple Silicon GPU! Utilizing MPS backend.")
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(" -> Detected Google TPU! Utilizing PyTorch-XLA backend.")
        except Exception:
            device = torch.device("cpu")
            print(" -> GPU/TPU acceleration not found. Training will run on CPU.")
        
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found!")
        return
        
    print("\n" + "=" * 80)
    print("🚀 STEP 2: EXTRACTING AND SPLITTING DATASET (PREVENTING DATA LEAKAGE)")
    print("=" * 80)
    print(" -> Querying SQLite database for all labeled site-level results...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT gene_name, site_index, lrt, alpha, beta_neg, beta_pos, p_neg
        FROM site_results 
        WHERE lrt IS NOT NULL
    """)
    raw_sites = c.fetchall()
    conn.close()
    print(f" -> Successfully fetched {len(raw_sites):,} site-level results with 5 MEME parameters from the database.")
    
    # Shuffle and split by gene to avoid homology data leakage
    all_genes = sorted(list(set(row[0] for row in raw_sites)))
    random.seed(42)
    random.shuffle(all_genes)
    
    split_idx = int(len(all_genes) * 0.8)
    train_genes = set(all_genes[:split_idx])
    val_genes = set(all_genes[split_idx:])
    
    train_raw = [row for row in raw_sites if row[0] in train_genes]
    val_raw = [row for row in raw_sites if row[0] in val_genes]
    
    if subsample_limit:
        print(f" -> NOTE: Subsampling training dataset to {subsample_limit} sites.")
        train_raw = random.sample(train_raw, min(len(train_raw), subsample_limit))
        if not subsample_val_limit:
            val_raw = random.sample(val_raw, min(len(val_raw), subsample_limit // 4))
            
    # We do NOT subsample val_raw here to allow randomly selecting a different subset each epoch
        
    gene_to_variable_sites = {}
    if loaded_cache:
        for gene, cache_data in loaded_cache.items():
            if "variable_sites" in cache_data:
                gene_to_variable_sites[gene] = cache_data["variable_sites"]
        print(f"    - Variable site flags resolved from cache for {len(gene_to_variable_sites):,} genes.")
    elif cache_is_dir:
        # Load variable_sites flags. Try loading precompiled pkl first for speed.
        t_var0 = time.time()
        pkl_path = os.path.join(cache_path, "variable_sites.pkl")
        if os.path.exists(pkl_path):
            print(f" -> Loading precompiled variable site flags from {pkl_path}...")
            try:
                with open(pkl_path, "rb") as f_pkl:
                    gene_to_variable_sites = pickle.load(f_pkl)
                print(f"    - Successfully loaded variable site flags in {time.time()-t_var0:.2f}s.")
            except Exception as e:
                print(f"    - Error loading precompiled flags: {e}. Falling back to directory scan.")
                gene_to_variable_sites = {}
        
        if not gene_to_variable_sites:
            unique_genes_in_split = list(set(row[0] for row in raw_sites))
            print(f" -> Loading variable site flags on-the-fly from NPZ directory for {len(unique_genes_in_split):,} genes...")
            for idx, gene in enumerate(unique_genes_in_split):
                npz_path = os.path.join(cache_path, f"{gene}.npz")
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        gene_to_variable_sites[gene] = data["variable_sites"].tolist()
                    except:
                        pass
            print(f"    - Successfully loaded variable site flags in {time.time()-t_var0:.2f}s.")
    else:
        print(f" -> Precomputing variable site flags for {len(all_genes):,} unique genes in parallel...")
        t_start_precompute = time.time()
        tasks = [(gene, os.path.join(msa_dir, f"{gene}.gz")) for gene in all_genes]
        from multiprocessing import Pool, cpu_count
        pool_workers = max(1, cpu_count() - 2)
        print(f"    - Utilizing {pool_workers} CPU worker processes...")
        with Pool(processes=pool_workers) as pool:
            for gene, var_sites in pool.imap_unordered(precompute_variable_sites_worker, tasks, chunksize=20):
                gene_to_variable_sites[gene] = var_sites
        print(f"    - Completed precomputing variable site flags in {time.time() - t_start_precompute:.2f}s.")
        
    print(" -> Filtering out 100% uninformative invariable sites (unique_codons <= 1, keeping ALL synonymous and non-synonymous variation)...")
    train_sites = []
    skipped_train = 0
    for row in train_raw:
        gene, site = row[0], row[1]
        lrt, alpha, beta_neg, beta_pos, p_neg = row[2], row[3], row[4], row[5], row[6]
        variable_sites = gene_to_variable_sites.get(gene)
        is_var = True
        if variable_sites and len(variable_sites) > 1 and 1 <= site < len(variable_sites):
            is_var = variable_sites[site]
            
        if is_var:
            y_lrt = math.log(max(0.0, lrt) + 1.0)
            y_alpha = math.log(max(0.0, alpha if alpha is not None else 0.0) + 1.0)
            y_beta_neg = math.log(max(0.0, beta_neg if beta_neg is not None else 0.0) + 1.0)
            y_beta_pos = math.log(max(0.0, beta_pos if beta_pos is not None else 0.0) + 1.0)
            y_p_neg = float(p_neg if p_neg is not None else 1.0)
            y_vec = (y_lrt, y_alpha, y_beta_neg, y_beta_pos, y_p_neg)
            train_sites.append((gene, site, y_vec))
        else:
            skipped_train += 1
            
    val_sites = []
    skipped_val = 0
    for row in val_raw:
        gene, site = row[0], row[1]
        lrt, alpha, beta_neg, beta_pos, p_neg = row[2], row[3], row[4], row[5], row[6]
        variable_sites = gene_to_variable_sites.get(gene)
        is_var = True
        if variable_sites and len(variable_sites) > 1 and 1 <= site < len(variable_sites):
            is_var = variable_sites[site]
            
        if is_var:
            y_lrt = math.log(max(0.0, lrt) + 1.0)
            y_alpha = math.log(max(0.0, alpha if alpha is not None else 0.0) + 1.0)
            y_beta_neg = math.log(max(0.0, beta_neg if beta_neg is not None else 0.0) + 1.0)
            y_beta_pos = math.log(max(0.0, beta_pos if beta_pos is not None else 0.0) + 1.0)
            y_p_neg = float(p_neg if p_neg is not None else 1.0)
            y_vec = (y_lrt, y_alpha, y_beta_neg, y_beta_pos, y_p_neg)
            val_sites.append((gene, site, y_vec))
        else:
            skipped_val += 1
            
    print(f" -> Variable Site Filter Complete:")
    print(f"    - Training sites kept: {len(train_sites):,} (skipped {skipped_train:,} invariable sites)")
    print(f"    - Validation sites kept: {len(val_sites):,} (skipped {skipped_val:,} invariable sites)")
    
    print(f" -> Split Stats:")
    print(f"    - Total unique genes: {len(all_genes)}")
    print(f"    - Training: {len(train_genes)} genes ({len(train_sites):,} sites)")
    print(f"    - Validation: {len(val_genes)} genes ({len(val_sites):,} sites)")
    
    train_active = [s for s in train_sites if s[2][0] > 0.0]
    train_neutral = [s for s in train_sites if s[2][0] <= 0.0]
    print(f" -> Balanced Sampling Setup:")
    print(f"    - Active training sites: {len(train_active):,}")
    print(f"    - Neutral training sites: {len(train_neutral):,}")
    
    print("\n" + "=" * 80)
    print("🚀 STEP 3: CREATING PYTORCH DATA LOADERS")
    print("=" * 80)
    train_dataset = MSADataset(db_path, msa_dir, train_sites, window_size=window_size, max_species=max_species, cache_dict=loaded_cache if not cache_is_dir else None, cache_dir=cache_path if cache_is_dir else None, cache_size_limit=cache_size_limit, subsample_species=subsample_species, max_k=max_k)
    # Save the full validation set for dynamic per-epoch sampling
    val_sites_all = val_sites
    if subsample_val_limit and len(val_sites_all) > subsample_val_limit:
        val_sites_init = random.sample(val_sites_all, subsample_val_limit)
        print(f" -> Validation evaluation will be subsampled dynamically to {subsample_val_limit} random sites per epoch.")
    else:
        val_sites_init = val_sites_all
        
    val_dataset = MSADataset(db_path, msa_dir, val_sites_init, window_size=window_size, max_species=max_species, cache_dict=loaded_cache if not cache_is_dir else None, cache_dir=cache_path if cache_is_dir else None, cache_size_limit=cache_size_limit, subsample_species=False, max_k=max_k)
    
    drop_last_train = len(train_dataset) >= batch_size
    val_batch_size = min(batch_size, 256)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    print(f" -> Initial Validation Batches: {len(val_loader)} (Batch Size: {batch_size})")
    
    print("\n" + "=" * 80)
    print("🚀 STEP 4: INSTANTIATING MODEL, OPTIMIZER, AND LOSS FUNCTION")
    print("=" * 80)
    use_ordinal = (loss_type == "coral")
    model = PhyloAxialTransformer(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, max_species=max_species, max_k=max_k, pure_coral=pure_coral, use_ordinal=use_ordinal).to(device)
    
    # Load existing weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f" -> Loading existing model weights from '{weights_path}'...")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter state dict to handle potential mismatches gracefully
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        if len(filtered_state_dict) < len(state_dict):
            print(f"    - Warning: Only loaded {len(filtered_state_dict)}/{len(state_dict)} tensors due to size or key mismatch.")
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        print("    - Model weights loaded successfully.")
    elif weights_path:
        print(f" -> Warning: weights_path '{weights_path}' specified but file not found. Starting from scratch.")
        
    if loss_type == "coral":
        print(f" -> Utilizing 12-Threshold CORAL Cumulative Ordinal Multi-Task Loss with Calibrated BCE (gamma_focal=0.0, phys_weight=0.50, pure_coral={pure_coral})")
        criterion = FocalCoralOrdinalLoss(gamma_focal=0.0, phys_weight=0.50, pure_coral=pure_coral).to(device)
    elif loss_type == "rate_weighted":
        print(f" -> Utilizing Rate-Weighted MSE Loss (gamma=1.5, underpredict_penalty=2.0) with rank_weight={rank_weight}")
        criterion = RateWeightedLoss(gamma=1.5, underpredict_penalty=2.0).to(device)
    elif loss_type == "listnet":
        print(f" -> Utilizing ListNet Loss (ranking) with target_transform='{target_transform}'")
        criterion = ListNetLoss(target_transform=target_transform).to(device)
    else:
        print(f" -> Utilizing Fused Physical Coupling + Learnable Uncertainty Multi-Task Joint Loss (phys_weight=0.50, rank_weight={rank_weight})")
        criterion = WeightedHuberLoss(rank_weight=rank_weight, power=loss_power, loss_type=loss_type, rank_threshold=rank_threshold, std_weight=std_weight).to(device)

    # Exclude biases, LayerNorms, and embeddings from weight decay to prevent parameter/output shrinkage
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    # Include learnable loss module parameters (e.g. log_vars) in optimizer
    for param in criterion.parameters():
        if param.requires_grad:
            no_decay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    
    # Setup Automatic Mixed Precision (AMP) safely using contextlib
    if device.type == 'xla':
        amp_context = contextlib.nullcontext()
        use_amp = False
    elif device.type == 'cuda':
        amp_context = torch.autocast(device_type='cuda', dtype=torch.float16)
        use_amp = True
    elif device.type == 'mps':
        try:
            amp_context = torch.autocast(device_type='mps', dtype=torch.float16)
            use_amp = True
        except Exception:
            amp_context = contextlib.nullcontext()
            use_amp = False
    else:
        amp_context = contextlib.nullcontext()
        use_amp = False
            
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler(device.type if device.type in ['cuda', 'mps'] else 'cuda', enabled=(use_amp and device.type in ['cuda', 'mps']))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    if use_amp:
        print(f" -> Enabled Automatic Mixed Precision (AMP FP16) on {device.type.upper()} device!")
    else:
        print(f" -> Using standard FP32 precision on {device.type.upper()} device.")
        
    best_spearman = -1.0
    
    print("\n" + "=" * 80)
    print("🚀 STEP 5: TRAINING LOOP STARTING")
    print("=" * 80)
    for epoch in range(1, epochs + 1):
        print(f"\n--- 🌟 STARTING EPOCH {epoch}/{epochs} ---")
        
        # Resample Empirical Quantile Stratified Balanced Sites with Dedicated Extreme Bin (LRT > 20.0)
        if len(train_active) > 0 and len(train_neutral) > 0:
            # Active bins based on critical biological cutoffs:
            # Bin 1: LRT in (0, 2.0], Bin 2: LRT in (2.0, 5.14], Bin 3: LRT in (5.14, 10.0], Bin 4: LRT in (10.0, 20.0], Bin 5: LRT > 20.0 (Extreme)
            bin1 = [s for s in train_active if s[2][0] <= math.log(3.0)]
            bin2 = [s for s in train_active if math.log(3.0) < s[2][0] <= math.log(6.14)]
            bin3 = [s for s in train_active if math.log(6.14) < s[2][0] <= math.log(11.0)]
            bin4 = [s for s in train_active if math.log(11.0) < s[2][0] <= math.log(21.0)]
            bin5 = [s for s in train_active if s[2][0] > math.log(21.0)]
            
            total_target = epoch_size_limit if epoch_size_limit else min(len(train_sites), 262144)
            # Use natural empirical prior weighting (92% Neutral, 8% Active split across 5 bins)
            # Matches natural genome-wide empirical prior, eliminating baseline shift and controlling FPR <= 5%
            n_neutral = int(total_target * 0.92)
            n_per_bin = max(1, (total_target - n_neutral) // 5)
            
            def sample_bin(bin_list, n_needed):
                if len(bin_list) == 0:
                    return []
                if len(bin_list) >= n_needed:
                    return random.sample(bin_list, n_needed)
                return random.choices(bin_list, k=n_needed)
                
            samp_neutral = sample_bin(train_neutral, n_neutral)
            samp_b1 = sample_bin(bin1, n_per_bin)
            samp_b2 = sample_bin(bin2, n_per_bin)
            samp_b3 = sample_bin(bin3, n_per_bin)
            samp_b4 = sample_bin(bin4, n_per_bin)
            samp_b5 = sample_bin(bin5, n_per_bin)
            
            epoch_sites = samp_neutral + samp_b1 + samp_b2 + samp_b3 + samp_b4 + samp_b5
            random.shuffle(epoch_sites)
            train_dataset.sites = epoch_sites
            print(f" -> Epoch {epoch} Stratified Dataset Size: {len(epoch_sites):,} sites ({len(epoch_sites)//batch_size} batches) | Neutral: {len(samp_neutral):,}, B1(0-2): {len(samp_b1):,}, B2(2-5.14): {len(samp_b2):,}, B3(5.14-10): {len(samp_b3):,}, B4(10-20): {len(samp_b4):,}, B5(>20): {len(samp_b5):,}")
        
        actual_micro_batch = min(micro_batch_size, batch_size)
        accum_steps = max(1, batch_size // actual_micro_batch)
        drop_last_tr = True if device.type == 'xla' else drop_last_train
        train_loader = DataLoader(train_dataset, batch_size=actual_micro_batch, shuffle=True, drop_last=drop_last_tr, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
        actual_total_batches = max(1, math.ceil(len(train_loader) / accum_steps))
        
        model.train()
        train_loss_acc = 0.0
        
        # Let the user know the alignment cache is warming up
        if epoch == 1:
            print(" -> [Cache Warmup] Loading alignment files into memory for the first time. This may take 1-3 minutes...", flush=True)
            
        if device.type == 'xla':
            import torch_xla.distributed.parallel_loader as pl
            device_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        else:
            device_train_loader = train_loader
            
        t_last_batch = time.time()
        for batch_idx, (codon_tokens, aa_tokens, dist, mds_coords, padding_mask, labels) in enumerate(device_train_loader):
            step_start = time.time()
            t_data_done = time.time()
            # On TPU with ParallelLoader, cap data_fetch_sec if XLA JIT compilation or host sync delayed background loader
            data_fetch_sec = max(0.0, t_data_done - t_last_batch)
            is_jit_compilation = False
            if device.type == 'xla' and data_fetch_sec > 5.0 and batch_idx > 0:
                is_jit_compilation = True
                data_fetch_sec = 0.001  # Attribute wait time to XLA JIT optimizer/graph compilation
            
            codon_tokens = codon_tokens.to(device)
            aa_tokens = aa_tokens.to(device)
            dist = dist.to(device)
            mds_coords = mds_coords.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            t_h2d_done = time.time()
            h2d_sec = t_h2d_done - t_data_done
            actual_micro_batch = min(micro_batch_size, batch_size)
            accum_steps = max(1, batch_size // actual_micro_batch)
            is_accum_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            
            with amp_context:
                y_pred = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
                t_fwd_done = time.time()
                fwd_sec = t_fwd_done - t_h2d_done
                
                raw_loss = criterion(y_pred, labels)
                loss = raw_loss / accum_steps
                t_loss_done = time.time()
                loss_sec = t_loss_done - t_fwd_done
            
            bwd_start = time.time()
            if use_amp and device.type in ['cuda', 'mps']:
                scaler.scale(loss).backward()
                if is_accum_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=False)
            else:
                loss.backward()
                if is_accum_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if device.type == 'xla':
                        try:
                            import torch_xla.core.xla_model as xm
                            xm.optimizer_step(optimizer)
                        except Exception:
                            optimizer.step()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=False)
                elif device.type == 'xla':
                    xla_mark_step()

            bwd_sec = time.time() - bwd_start
            
            loss = raw_loss

            if device.type == 'xla':
                should_print = (batch_idx + 1) in [1, accum_steps] or ((batch_idx + 1) % 50 == 0) or (batch_idx + 1) == len(train_loader)
            else:
                should_print = (batch_idx + 1) in [1, 2, 5, 10] or ((batch_idx + 1) % 10 == 0) or (batch_idx + 1) == len(train_loader)
                
            train_loss_acc += float(raw_loss.detach().cpu())
            
            total_step_sec = time.time() - step_start
            
            if profile or should_print:
                sites_per_sec = len(labels) / max(1e-5, total_step_sec)
                if profile:
                    jit_tag = " [XLA JIT Compile]" if is_jit_compilation else ""
                    print(f"   [PROFILER Batch {batch_idx+1:>3d}]{jit_tag} Total: {total_step_sec:.4f}s ({sites_per_sec:>6.0f} sites/s) | Data: {data_fetch_sec:.4f}s | H2D: {h2d_sec:.4f}s | Fwd: {fwd_sec:.4f}s | Loss: {loss_sec:.4f}s | Bwd+Opt: {bwd_sec:.4f}s", flush=True)
                if should_print:
                    mem_str = ""
                    if device.type == 'cuda':
                        vram_gb = torch.cuda.memory_allocated() / (1024**3)
                        mem_str += f" | VRAM: {vram_gb:.2f} GB"
                    elif device.type == 'xla':
                        try:
                            import torch_xla.core.xla_model as xm
                            m_info = xm.get_memory_info(device)
                            u_mb = m_info['bytes_used'] / (1024 * 1024)
                            t_mb = m_info['total_bytes'] / (1024 * 1024)
                            mem_str += f" | TPU VRAM: {u_mb:.0f}/{t_mb:.0f} MB"
                        except Exception:
                            pass
                    try:
                        import psutil
                        ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                        mem_str += f" | RAM: {ram_gb:.2f} GB"
                    except Exception:
                        pass
                    print(f"  Micro-Batch {batch_idx+1}/{len(train_loader)} (Accum Step {min(actual_total_batches, (batch_idx//accum_steps)+1)}/{actual_total_batches}) | Loss: {raw_loss.detach().cpu().item():.4f}{mem_str}", flush=True)
                    
            t_last_batch = time.time()
                
        avg_train_loss = train_loss_acc / max(1, len(train_loader))
        print(f"\n -> Epoch {epoch} training complete. Running validation evaluation (loading validation alignments into memory cache)...", flush=True)
        
        model.eval()
        
        # Use a fixed, deterministic validation subset across all epochs to eliminate evaluation noise
        val_batch_size = min(batch_size, 256)
        drop_last_val = True if device.type == 'xla' else False
        if subsample_val_limit and len(val_sites_all) > subsample_val_limit:
            rng_val = random.Random(42)  # Fixed seed = 42 ensures identical 65k sites every epoch!
            epoch_val_sites = rng_val.sample(val_sites_all, subsample_val_limit)
            val_dataset.sites = epoch_val_sites
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=drop_last_val, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
            print(f" -> Dynamically selected a new validation subset of {len(epoch_val_sites):,} sites ({len(val_loader)} batches, val_batch_size={val_batch_size})...", flush=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=drop_last_val, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
            
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            if device.type == 'xla':
                import torch_xla.distributed.parallel_loader as pl
                device_val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
            else:
                device_val_loader = val_loader
            for codon_tokens, aa_tokens, dist, mds_coords, padding_mask, labels in device_val_loader:
                codon_tokens = codon_tokens.to(device)
                aa_tokens = aa_tokens.to(device)
                dist = dist.to(device)
                mds_coords = mds_coords.to(device)
                padding_mask = padding_mask.to(device)

                out = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
                if isinstance(out, tuple):
                    y_pred = torch.stack(out, dim=-1)
                else:
                    y_pred = out
                val_preds_list.append(y_pred.detach().cpu())
                val_targets_list.append(labels.detach().cpu())
                if device.type == 'xla':
                    xla_mark_step()
                
        val_preds = torch.cat(val_preds_list, dim=0).numpy()
        val_targets = torch.cat(val_targets_list, dim=0).numpy()
        
        # Primary target = Column 0 (raw LRT)
        lrt_preds = val_preds[:, 0]
        lrt_targets_raw = np.expm1(val_targets[:, 0])
        
        mse = np.mean((lrt_preds - lrt_targets_raw) ** 2)
        
        if len(np.unique(lrt_preds)) > 1 and len(np.unique(lrt_targets_raw)) > 1:
            val_pearson = pearsonr(lrt_preds, lrt_targets_raw)[0]
            val_spearman = spearmanr(lrt_preds, lrt_targets_raw)[0]
            if np.isnan(val_spearman):
                val_spearman = 0.0
            if np.isnan(val_pearson):
                val_pearson = 0.0
        else:
            val_pearson, val_spearman = 0.0, 0.0
            
        print(f"\n📈 Epoch {epoch} Metrics Summary:")
        print(f"  - Average Training Loss: {avg_train_loss:.4f}")
        print(f"  - Validation LRT MSE: {mse:.4f}")
        print(f"  - Validation Pearson r (LRT):  {val_pearson:.4f}")
        if device.type == 'xla':
            try:
                import torch_xla.core.xla_model as xm
                mem_info = xm.get_memory_info(device)
                u_mb = mem_info['bytes_used'] / (1024 * 1024)
                t_mb = mem_info['total_bytes'] / (1024 * 1024)
                print(f"  - TPU VRAM Memory Used:      {u_mb:.1f} MB / {t_mb:.1f} MB ({u_mb/t_mb*100:.1f}%)")
            except Exception:
                pass
        elif device.type == 'cuda':
            v_used = torch.cuda.memory_allocated() / (1024**3)
            v_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            print(f"  - GPU VRAM Memory Used:      {v_used:.2f} GB / {v_total:.2f} GB ({v_used/v_total*100:.1f}%)")
        try:
            import psutil
            ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            print(f"  - Host System RAM Used:      {ram_gb:.2f} GB")
        except Exception:
            pass
        
        is_best = (val_spearman > best_spearman) or (epoch == epochs) or (best_spearman == -1.0)
        if is_best:
            best_spearman = max(best_spearman, val_spearman)
            checkpoint_path = save_path
            checkpoint_payload = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_type': loss_type,
                'pure_coral': pure_coral,
                'embed_dim': embed_dim,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'val_spearman': val_spearman,
                'val_mse': mse
            }
            if device.type == 'xla':
                try:
                    import torch_xla.core.xla_model as xm
                    xm.save(checkpoint_payload, checkpoint_path)
                except Exception:
                    torch.save(checkpoint_payload, checkpoint_path)
            else:
                torch.save(checkpoint_payload, checkpoint_path)
            print(f"  💾 SUCCESS: Saved model checkpoint to '{checkpoint_path}' (Epoch {epoch}, Spearman rho: {val_spearman:.4f}, MSE: {mse:.4f})")
        else:
            print(f"  ℹ️ Checkpoint not saved (Validation Spearman rho {val_spearman:.4f} did not exceed best of {best_spearman:.4f})")
            
    print("\n" + "=" * 80)
    print("🎉 MODEL TRAINING COMPLETE!")
    print(f"   Best Validation Spearman rho: {best_spearman:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Continuous Selection Transformer Model")
    parser.add_argument("--db_path", default="meme_results.db", help="Path to SQLite database")
    parser.add_argument("--msa_dir", default="msa", help="Path to compressed MSAs directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1536, help="Effective total batch size across gradient accumulation steps")
    parser.add_argument("--micro_batch_size", type=int, default=256, help="Micro-batch size executed per forward pass to control peak GPU VRAM usage")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--subsample_limit", type=int, default=None, help="Limit number of training sites loaded (for testing)")
    parser.add_argument("--subsample_val_limit", type=int, default=None, help="Limit number of validation sites loaded to speed up evaluation")
    parser.add_argument("--loss_type", default="coral", choices=["coral", "huber", "l1", "listnet", "rate_weighted"], help="Loss function (coral, huber, l1, listnet, or rate_weighted)")
    parser.add_argument("--target_transform", default="none", choices=["none", "log1p"], help="Transformation for targets (listnet only)")
    parser.add_argument("--device", default=None, help="Force run on device (cpu, mps, cuda)")
    parser.add_argument("--cache_path", default=None, help="Path to precomputed MSA cache file (.pkl.gz)")
    parser.add_argument("--epoch_size_limit", type=int, default=100000, help="Max training sites per epoch to speed up local training (default: 100000)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for DataLoader (default: 0)")
    parser.add_argument("--max_species", type=int, default=256, help="Maximum number of species to subsample (default: 256)")
    parser.add_argument("--cache_size_limit", type=int, default=0, help="Maximum cache size for dataset alignments in memory (default: 0 = unlimited. Set to 256 for low RAM local execution)")
    parser.add_argument("--window_size", type=int, default=1, help="Window size around target site (default: 1)")
    parser.add_argument("--subsample_species", type=lambda x: (str(x).lower() == 'true'), default=True, help="Dynamically subsample species count during training (default: True)")
    parser.add_argument("--rank_weight", type=float, default=0.5, help="Pairwise ranking loss weight (default: 0.5)")
    parser.add_argument("--weights_path", default=None, help="Path to existing model checkpoint weights (.pt) to resume training from")
    parser.add_argument("--save_path", default="selection_transformer_best.pt", help="Path to save the best trained model checkpoint weights (.pt)")
    parser.add_argument("--loss_power", type=float, default=2.0, help="Exponent power for target weighting in WeightedHuberLoss (default: 2.0)")
    parser.add_argument("--rank_threshold", type=float, default=3.0, help="Minimum raw LRT threshold to include a site as the higher target in pairwise ranking loss (default: 3.0)")
    parser.add_argument("--std_weight", type=float, default=0.0, help="Batch standard deviation matching loss weight (default: 0.0)")
    parser.add_argument("--max_k", type=int, default=64, help="Maximum edge tokens to extract per codon site (default: 64)")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of axial transformer layers (default: 4)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--profile", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable high-resolution per-step performance profiling (default: False)")
    parser.add_argument("--pure_coral", action="store_true", help="Enable pure CORAL ordinal training/evaluation, disabling direct continuous regression head feedback and evaluation hybrid maxing")
    args = parser.parse_args()
    
    train_full_model(
        db_path=args.db_path,
        msa_dir=args.msa_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        subsample_limit=args.subsample_limit,
        loss_type=args.loss_type,
        target_transform=args.target_transform,
        device_override=args.device,
        cache_path=args.cache_path,
        epoch_size_limit=args.epoch_size_limit,
        num_workers=args.num_workers,
        max_species=args.max_species,
        cache_size_limit=args.cache_size_limit,
        window_size=args.window_size,
        subsample_species=args.subsample_species,
        rank_weight=args.rank_weight,
        weights_path=args.weights_path,
        subsample_val_limit=args.subsample_val_limit,
        save_path=args.save_path,
        loss_power=args.loss_power,
        rank_threshold=args.rank_threshold,
        std_weight=args.std_weight,
        max_k=args.max_k,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        profile=args.profile,
        pure_coral=args.pure_coral
    )
