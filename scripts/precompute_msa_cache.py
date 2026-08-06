#!/usr/bin/env python3
import os
# Limit threads to 1 per process to prevent thrashing / oversubscription in multiprocessing Pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gzip
import pickle
import sqlite3
import time
import numpy as np
from multiprocessing import Pool, cpu_count


# Ensure parent directory is in path to import train_regression
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_regression import (
    parse_newick,
    get_patristic_distances,
    compute_mds_coordinates,
    is_site_variable,
    GENETIC_CODE
)

def process_gene_worker(args):
    gene, msa_dir = args
    msa_path = os.path.join(msa_dir, f"{gene}.gz")
    if not os.path.exists(msa_path):
        return gene, None

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
        return gene, None
        
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
            
    mds_coords_np = compute_mds_coordinates(dist_arr, n_components=4)
    
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
        
    return gene, {
        "seq_dict": seq_dict,
        "species_names": species_names,
        "dist_arr": dist_arr,
        "mds_coords": mds_coords_np,
        "variable_sites": variable_sites
    }

def main():
    import argparse
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", required=True, help="Path to SQLite database")
    parser.add_argument("--msa_dir", required=True, help="Path to MSA directory")
    parser.add_argument("--out_cache", required=True, help="Path to output compressed cache file (e.g. msa_cache.pkl.gz)")
    args = parser.parse_args()

    print(f"Connecting to database {args.db_path}...")
    conn = sqlite3.connect(args.db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT gene_name FROM site_results WHERE lrt IS NOT NULL")
    all_genes = [row[0] for row in c.fetchall()]
    conn.close()
    
    print(f"Found {len(all_genes):,} unique genes with site results.")
    
    t0 = time.time()
    num_workers = max(1, cpu_count() - 2)
    print(f"Starting precomputation using {num_workers} worker processes...")
    
    tasks = [(gene, args.msa_dir) for gene in all_genes]
    
    cache_dict = {}
    processed_count = 0
    
    with Pool(processes=num_workers) as pool:
        for gene, result in pool.imap_unordered(process_gene_worker, tasks, chunksize=10):
            if result is not None:
                cache_dict[gene] = result
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count}/{len(all_genes)} genes ({processed_count/len(all_genes)*100:.1f}%) in {time.time()-t0:.1f}s...")
                
    t_end = time.time()
    print(f"Precomputation complete: processed {len(cache_dict):,} valid genes in {t_end-t0:.2f} seconds.")
    
    print(f"Saving serialized cache to {args.out_cache}...")
    t_save = time.time()
    with gzip.open(args.out_cache, "wb") as f:
        pickle.dump(cache_dict, f, protocol=4)
    print(f"Successfully saved cache to {args.out_cache} in {time.time()-t_save:.2f} seconds.")
    print(f"Cache size: {os.path.getsize(args.out_cache)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
