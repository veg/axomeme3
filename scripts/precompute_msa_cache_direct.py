#!/usr/bin/env python3
import os
# Limit threads to 1 per process to prevent oversubscription in multiprocessing Pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gzip
import re
import math
import time
import sqlite3
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

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

# standard translation
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGA':'*', 'TGC':'C', 'TGT':'C', 'TGG':'W',
}
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

AA_LIST = "ACDEFGHIKLMNPQRSTVWY*-?"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Create lookup tables locally
CODON_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 65
AA_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 22

grid = np.arange(256, dtype=np.int32)
b0 = grid[:, None, None] * 65536
b1 = grid[None, :, None] * 256
b2 = grid[None, None, :]
flat_indices = (b0 + b1 + b2).ravel()
b0_val = flat_indices // 65536
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


def process_gene_worker(args):
    gene, msa_dir, out_dir, max_species = args
    gene_path = os.path.join(out_dir, f"{gene}.npz")
    
    # Skip if completed
    if os.path.exists(gene_path) and os.path.getsize(gene_path) > 100:
        return gene, "skipped"
        
    msa_path = os.path.join(msa_dir, f"{gene}.gz")
    if not os.path.exists(msa_path):
        return gene, "missing_msa"

    taxlabels = []
    sequences = []
    tree_str = None
    in_taxlabels = False
    in_matrix = False
    in_trees = False
    
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
        return gene, f"read_error: {e}"
        
    seq_dict = {}
    for label, seq in zip(taxlabels, sequences):
        seq_dict[label] = seq
        
    dist_matrix = {}
    species_names = []
    if tree_str:
        try:
            tree_root = parse_newick(tree_str)
            species_names, dist_matrix = get_patristic_distances(tree_root)
        except Exception:
            pass
            
    if not species_names:
        species_names = list(seq_dict.keys())
        dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}
        
    n_spec = len(species_names)
    dist_arr = np.zeros((n_spec, n_spec), dtype=np.float32)
    for i, spec1 in enumerate(species_names):
        for j, spec2 in enumerate(species_names):
            dist_arr[i, j] = dist_matrix.get(spec1, {}).get(spec2, 0.0)
            
    mds_coords = compute_mds_coordinates(dist_arr, n_components=4)
    
    # Precompute variable status for all codon sites
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
        
    # Precompute diversity indices
    if n_spec > max_species:
        selected_indices = select_diverse_species_matrix(dist_arr, max_species, keep_idx=0)
    else:
        selected_indices = list(range(n_spec))
        
    # Vectorized tokenization
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
    
    # Save direct to NPZ in uncompressed format for fast training loading
    np.savez(
        gene_path,
        codon_ids_matrix=codon_ids_matrix,
        aa_ids_matrix=aa_ids_matrix,
        species_names=np.array(species_names),
        dist_arr=dist_arr,
        mds_coords=mds_coords,
        variable_sites=np.array(variable_sites),
        selected_indices=np.array(selected_indices),
        valid_seqs=valid_seqs
    )
    
    return gene, "success"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", required=True, help="Path to SQLite database")
    parser.add_argument("--msa_dir", required=True, help="Path to MSA directory")
    parser.add_argument("--out_dir", required=True, help="Path to output NPZ cache directory")
    parser.add_argument("--max_species", type=int, default=768, help="Maximum species depth threshold for diversity indices (default: 768)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Connecting to database {args.db_path}...")
    conn = sqlite3.connect(args.db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT gene_name FROM site_results WHERE lrt IS NOT NULL")
    all_genes = [row[0] for row in c.fetchall()]
    conn.close()
    
    print(f"Found {len(all_genes):,} unique genes with site results in database.")
    
    # Filter completed
    completed_genes = set()
    for filename in os.listdir(args.out_dir):
        if filename.endswith(".npz"):
            completed_genes.add(filename[:-4])
            
    remaining_genes = [g for g in all_genes if g not in completed_genes]
    print(f"Already completed: {len(completed_genes):,} genes. Remaining to process: {len(remaining_genes):,} genes.")
    
    if not remaining_genes:
        print("[*] All genes are already cached in NPZ directory!")
        return

    t0 = time.time()
    num_workers = max(1, cpu_count() - 2)
    print(f"Starting direct precomputation using {num_workers} parallel worker processes...")
    
    tasks = [(gene, args.msa_dir, args.out_dir, args.max_species) for gene in remaining_genes]
    
    processed_count = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    with Pool(processes=num_workers) as pool:
        for gene, status in pool.imap_unordered(process_gene_worker, tasks, chunksize=10):
            processed_count += 1
            if status == "success":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                error_count += 1
                
            if processed_count % 50 == 0:
                elapsed = time.time() - t0
                rate = processed_count / elapsed
                etc = (len(remaining_genes) - processed_count) / rate if rate > 0 else 0.0
                print(f"  Processed {processed_count}/{len(remaining_genes)} remaining ({processed_count/len(remaining_genes)*100:.1f}%) | Success: {success_count} | Skip/Err: {skipped_count}/{error_count} | Rate: {rate:.2f} genes/s | ETC: {etc/60:.1f}m...")
                
    t_end = time.time()
    print(f"\n[*] Precomputation Complete!")
    print(f"  Total time: {t_end-t0:.2f} seconds.")
    print(f"  Successfully cached: {success_count} genes.")
    print(f"  Errors: {error_count} genes.")

if __name__ == "__main__":
    main()
