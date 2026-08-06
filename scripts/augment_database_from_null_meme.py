#!/usr/bin/env python3
"""
Augment Database & NPZ Cache with Full MEME Inferences on Silverback Null Simulations
-------------------------------------------------------------------------------------
Parses MEME JSON result files (*.MEME.json) from local null_simulations/ and 
silverback.temple.edu, extracts site-by-site MEME Maximum Likelihood estimates 
(LRT, alpha, beta_neg, p_neg, beta_pos, p_pos, p_value), computes NPZ features,
and populates the SQLite database (meme_results.db).
"""

import os
import sys
import glob
import gzip
import json
import math
import sqlite3
import numpy as np
import pandas as pd
from io import StringIO
from Bio import Phylo
from multiprocessing import Pool, cpu_count

from train_transformer_selection import (
    parse_newick,
    compute_mds_coordinates,
    get_codon_token,
    get_aa_token,
    is_site_variable,
    GENETIC_CODE
)
from predict_regression_nexus import (
    calculate_patristic_distances,
    translate_codon,
    parse_nexus_alignment_and_embedded_tree
)

def parse_meme_json_targets(json_path):
    """Extract site-level Maximum Likelihood MEME estimates from JSON file."""
    try:
        open_func = gzip.open if json_path.endswith('.gz') else open
        with open_func(json_path, 'rt', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            
        mle_data = data.get('MLE', {})
        mle_content = mle_data.get('content', {}).get('0', [])
        if not mle_content:
            return None
            
        # Format per site: [alpha, beta_neg, p_neg, beta_pos, p_pos, lrt, p_val, num_branches, total_branch_len]
        site_targets = {}
        for s_idx, row in enumerate(mle_content):
            alpha = float(row[0])
            beta_neg = float(row[1])
            p_neg = float(row[2])
            beta_pos = float(row[3])
            p_pos = float(row[4])
            lrt = float(row[5])
            p_val = float(row[6])
            site_targets[s_idx + 1] = {
                'alpha': alpha,
                'beta_neg': beta_neg,
                'p_neg': p_neg,
                'beta_pos': beta_pos,
                'p_pos': p_pos,
                'lrt': lrt,
                'p_value': p_val
            }
        return site_targets
    except Exception as e:
        return None


def process_null_meme_pair(args_tuple):
    """Worker function to process alignment + tree + MEME JSON into NPZ and SQLite records."""
    nex_path, tree_path, json_path, output_npz_dir = args_tuple
    
    gene_name = os.path.basename(nex_path).replace('.replicate.1.nex', '').replace('.nex', '')
    if not gene_name.startswith('null_'):
        gene_name = f"null_{gene_name}"
        
    out_npz_path = os.path.join(output_npz_dir, f"{gene_name}.npz")
    
    # 1. Parse MEME JSON targets
    meme_targets = parse_meme_json_targets(json_path)
    if not meme_targets:
        return None
        
    try:
        # 2. Parse alignment & tree
        seq_dict, _, embedded_tree_str = parse_nexus_alignment_and_embedded_tree(nex_path)
        if not seq_dict:
            return None
            
        tree_str = None
        if tree_path and os.path.exists(tree_path):
            with open(tree_path, 'r') as tf:
                tree_str = tf.read().strip()
        elif embedded_tree_str:
            tree_str = embedded_tree_str
            
        tree_obj = None
        if tree_str:
            clean_t = tree_str.split(';')[-2] + ';' if ';' in tree_str else tree_str + ';'
            try:
                tree_obj = Phylo.read(StringIO(clean_t), 'newick')
            except Exception:
                try:
                    tree_obj = parse_newick(clean_t)
                except Exception:
                    tree_obj = None
                    
        species_names = list(seq_dict.keys())
        N = len(species_names)
        if N < 4:
            return None
            
        if tree_obj:
            try:
                sp_clean, dist_dict, _ = calculate_patristic_distances(tree_obj)
                dist_arr = np.zeros((N, N), dtype=np.float32)
                name_map = {s.replace("'", "").replace('"', '').strip(): s for s in species_names}
                for i, s1 in enumerate(species_names):
                    n1 = s1.replace("'", "").replace('"', '').strip()
                    for j, s2 in enumerate(species_names):
                        n2 = s2.replace("'", "").replace('"', '').strip()
                        if n1 in dist_dict and n2 in dist_dict[n1]:
                            dist_arr[i, j] = dist_dict[n1][n2]
            except Exception:
                dist_arr = np.ones((N, N), dtype=np.float32) * 0.1
                np.fill_diagonal(dist_arr, 0.0)
        else:
            dist_arr = np.ones((N, N), dtype=np.float32) * 0.1
            np.fill_diagonal(dist_arr, 0.0)
            
        mds_coords = compute_mds_coordinates(dist_arr, n_components=4)
        
        # Build token matrices
        seq_len = len(next(iter(seq_dict.values())))
        total_codons = seq_len // 3
        
        codon_ids_matrix = np.full((N, total_codons), 65, dtype=np.int8)
        aa_ids_matrix = np.full((N, total_codons), 22, dtype=np.int8)
        
        variable_sites = [False] # 1-based indexing alignment
        
        for c_idx in range(total_codons):
            site_codons = []
            site_aas = []
            for s_idx, spec in enumerate(species_names):
                seq = seq_dict[spec]
                nuc_idx = c_idx * 3
                if nuc_idx + 3 <= len(seq):
                    codon = seq[nuc_idx:nuc_idx+3].upper()
                    c_tok = get_codon_token(codon)
                    a_tok = get_aa_token(codon)
                    codon_ids_matrix[s_idx, c_idx] = c_tok
                    aa_ids_matrix[s_idx, c_idx] = a_tok
                    
                    if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                        site_codons.append(codon)
                        aa = GENETIC_CODE.get(codon, '?')
                        if aa != '?':
                            site_aas.append(aa)
                            
            is_var = is_site_variable(site_codons, site_aas)
            variable_sites.append(bool(is_var))
            
        variable_sites = np.array(variable_sites, dtype=bool)
        selected_indices = np.arange(N, dtype=np.int64)
        valid_seqs = np.ones(N, dtype=bool)
        
        # Save NPZ file
        np.savez_compressed(
            out_npz_path,
            codon_ids_matrix=codon_ids_matrix,
            aa_ids_matrix=aa_ids_matrix,
            species_names=np.array(species_names, dtype=str),
            dist_arr=dist_arr,
            mds_coords=mds_coords,
            variable_sites=variable_sites,
            selected_indices=selected_indices,
            valid_seqs=valid_seqs
        )
        
        # Build site records using exact MEME ML targets
        site_records = []
        for site_idx in range(1, total_codons + 1):
            if variable_sites[site_idx] and site_idx in meme_targets:
                t = meme_targets[site_idx]
                is_sig = 1 if t['p_value'] <= 0.05 else 0
                rec = (
                    gene_name, site_idx, t['alpha'], t['beta_neg'], t['p_neg'], t['beta_pos'], t['p_pos'],
                    t['lrt'], t['p_value'], 0, 1.0, is_sig, 'NEUTRAL_NULL', 'NEUTRAL_NULL', 'NEUTRAL', t['p_value']
                )
                site_records.append(rec)
                
        return gene_name, N, total_codons, site_records
    except Exception as e:
        print(f"[!] Error processing {nex_path}: {e}")
        return None


def main():
    db_path = "/Users/sergei/Projects/TOGA_MEME/meme_results.db"
    npz_dir = "/Users/sergei/Projects/TOGA_MEME/msa_cache_npz"
    local_null_dir = "/Users/sergei/Projects/TOGA_MEME/null_simulations"
    
    os.makedirs(npz_dir, exist_ok=True)
    
    print("=" * 80)
    print("🚀 AUGMENTING DATASET WITH EXACT MEME INFERENCES ON SILVERBACK NULL SIMULATIONS")
    print("=" * 80)
    
    # Discover local simulation files matching NEXUS and MEME.json
    json_files = glob.glob(os.path.join(local_null_dir, "*.MEME.json"))
    print(f" -> Found {len(json_files)} local MEME JSON result files in '{local_null_dir}'")
    
    tasks = []
    for json_f in json_files:
        if os.path.getsize(json_f) < 100:
            continue
        base = os.path.basename(json_f).replace('.MEME.json', '')
        nex_f = os.path.join(local_null_dir, f"{base}.replicate.1.nex")
        if not os.path.exists(nex_f):
            nex_f = os.path.join(local_null_dir, f"{base}.nex")
            
        tree_f = os.path.join(local_null_dir, f"{base.replace('null_', '')}_tree.nwk")
        if not os.path.exists(tree_f):
            tree_f = None
            
        if os.path.exists(nex_f):
            tasks.append((nex_f, tree_f, json_f, npz_dir))
            
    print(f" -> Processing {len(tasks)} simulation alignments with full MEME targets in parallel...")
    workers = max(1, cpu_count() - 2)
    
    all_site_records = []
    results_list = []
    genes_added = 0
    total_null_sites = 0
    
    with Pool(processes=workers) as pool:
        for res in pool.imap_unordered(process_null_meme_pair, tasks, chunksize=1):
            if res:
                gene_name, num_seqs, total_codons, site_records = res
                results_list.append(res)
                all_site_records.extend(site_records)
                genes_added += 1
                total_null_sites += len(site_records)
                print(f"   [+] Processed '{gene_name}' ({num_seqs} seqs, {total_codons} codons, {len(site_records)} MEME-fitted sites)")
                
    print(f"\n -> Processed {genes_added} null simulation genes ({total_null_sites:,} MEME-fitted sites).")
    
    # Insert records into SQLite meme_results.db
    if all_site_records:
        print(f" -> Inserting {len(all_site_records):,} MEME-fitted site records into SQLite database '{db_path}'...")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("DELETE FROM site_results WHERE gene_name LIKE 'null_%'")
        c.execute("DELETE FROM gene_results WHERE gene_name LIKE 'null_%'")
        
        for gene_name, num_seqs, total_codons, site_recs in results_list:
            c.execute("""
                INSERT OR REPLACE INTO gene_results (gene_name, num_seqs, num_sites, log_likelihood, aic_c, num_selected_sites, runtime_sec)
                VALUES (?, ?, ?, 0.0, 0.0, 0, 1.0)
            """, (gene_name, num_seqs, total_codons))
            
        c.executemany("""
            INSERT INTO site_results (
                gene_name, site_index, alpha, beta_neg, p_neg, beta_pos, p_pos, lrt, p_value,
                num_branches_under_selection, total_branch_length, is_significant, classification, filter_reason, selection_type, q_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, all_site_records)
        
        conn.commit()
        conn.close()
        print(f" 🎉 Successfully inserted {len(all_site_records):,} MEME ML-inferred site targets into SQLite database!")

if __name__ == "__main__":
    main()
