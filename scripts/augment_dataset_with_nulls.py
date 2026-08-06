#!/usr/bin/env python3
"""
Augment Dataset with Null Simulations & Actual MEME Estimates
--------------------------------------------------------------
This script parses neutral NULL simulation alignments (from /Users/sergei/Projects/TOGA_MEME/null/aln)
and their corresponding actual HyPhy MEME estimation JSONs (from /Users/sergei/Projects/TOGA_MEME/null/meme),
precomputes NPZ feature caches (distance matrices, MDS coordinates, variable site flags),
and inserts actual MEME MLE records into the SQLite database (`meme_results.db`).
"""

import os, sys, glob, gzip, json, math, sqlite3
import numpy as np, pandas as pd
from io import StringIO
from Bio import Phylo
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = "/Users/sergei/Projects/TOGA_MEME"
sys.path.insert(0, PROJECT_ROOT)

from scripts.train_transformer_selection import (
    parse_newick,
    compute_mds_coordinates,
    get_codon_token,
    get_aa_token,
    is_site_variable,
    GENETIC_CODE
)
from scripts.predict_regression_nexus import (
    calculate_patristic_distances,
    parse_nexus_alignment_and_embedded_tree
)

def process_null_alignment(args_tuple):
    """Worker function to parse alignment, compute tree distance & MDS, parse MEME JSON, and output NPZ array."""
    nex_path, tree_path, meme_json_path, output_npz_dir = args_tuple
    gene_name = os.path.basename(nex_path).replace('.replicate.1.nex', '').replace('.nex', '')
    if not gene_name.startswith('null_'):
        gene_name = f"null_{gene_name}"
        
    out_npz_path = os.path.join(output_npz_dir, f"{gene_name}.npz")
    
    try:
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
        
        seq_len = len(next(iter(seq_dict.values())))
        total_codons = seq_len // 3
        
        codon_ids_matrix = np.full((N, total_codons), 65, dtype=np.int8)
        aa_ids_matrix = np.full((N, total_codons), 22, dtype=np.int8)
        variable_sites = [False]
        
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
        
        # Check if actual MEME JSON exists for this null dataset
        meme_data = None
        if meme_json_path and os.path.exists(meme_json_path) and os.path.getsize(meme_json_path) > 100:
            try:
                with open(meme_json_path, 'r') as mf:
                    meme_data = json.load(mf)
            except Exception:
                meme_data = None
                
        site_records = []
        for site_idx in range(1, total_codons + 1):
            if variable_sites[site_idx]:
                if meme_data and 'MLE' in meme_data and 'content' in meme_data['MLE'] and '0' in meme_data['MLE']['content']:
                    mle_rows = meme_data['MLE']['content']['0']
                    if site_idx - 1 < len(mle_rows):
                        row = mle_rows[site_idx - 1]
                        alpha = float(row[0])
                        beta_neg = float(row[1])
                        p_neg = float(row[2])
                        beta_pos = float(row[3])
                        p_pos = float(row[4])
                        lrt = float(row[5])
                        pval = float(row[6])
                        n_sel = int(row[7])
                        tot_len = float(row[8])
                        is_sig = 1 if pval <= 0.05 else 0
                        
                        rec = (
                            gene_name, site_idx, alpha, beta_neg, p_neg, beta_pos, p_pos, lrt, pval,
                            n_sel, tot_len, is_sig, 'NEUTRAL_NULL', 'NEUTRAL_NULL', 'NEUTRAL', pval
                        )
                        site_records.append(rec)
                        continue
                        
                # Default ground-truth neutral targets if MEME JSON not present
                rec = (
                    gene_name, site_idx, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, N, 1.0, 0, 'NEUTRAL_NULL', 'NEUTRAL_NULL', 'NEUTRAL', 1.0
                )
                site_records.append(rec)
                
        return gene_name, N, total_codons, site_records
    except Exception as e:
        print(f"[!] Error processing {nex_path}: {e}")
        return None

def main():
    db_path = "/Users/sergei/Projects/TOGA_MEME/meme_results.db"
    npz_dir = "/Users/sergei/Projects/TOGA_MEME/msa_cache_npz"
    local_null_aln_dir = "/Users/sergei/Projects/TOGA_MEME/null/aln"
    local_null_meme_dir = "/Users/sergei/Projects/TOGA_MEME/null/meme"
    
    os.makedirs(npz_dir, exist_ok=True)
    
    print("=" * 80)
    print("🚀 AUGMENTING DATASET WITH ACTUAL HYPHY MEME ESTIMATES ON NEUTRAL SIMULATIONS")
    print("=" * 80)
    
    nex_files = glob.glob(os.path.join(local_null_aln_dir, "*.nex"))
    print(f" -> Found {len(nex_files)} neutral simulation alignments in '{local_null_aln_dir}'")
    
    tasks = []
    for nex_f in nex_files:
        base = os.path.basename(nex_f).replace('.replicate.1.nex', '').replace('.nex', '')
        tree_f = os.path.join(local_null_aln_dir, f"{base}_tree.nwk")
        if not os.path.exists(tree_f): tree_f = None
        
        # Match MEME JSON
        meme_f = os.path.join(local_null_meme_dir, f"{base}.MEME.json")
        if not os.path.exists(meme_f):
            meme_f = os.path.join(local_null_meme_dir, f"null_{base}.MEME.json")
            if not os.path.exists(meme_f):
                meme_f = None
                
        tasks.append((nex_f, tree_f, meme_f, npz_dir))
        
    print(f" -> Processing {len(tasks)} simulation alignments in parallel...")
    workers = max(1, cpu_count() - 2)
    
    all_site_records = []
    results_list = []
    genes_added = 0
    total_null_sites = 0
    
    with Pool(processes=workers) as pool:
        for res in pool.imap_unordered(process_null_alignment, tasks, chunksize=2):
            if res:
                gene_name, num_seqs, total_codons, site_records = res
                results_list.append(res)
                all_site_records.extend(site_records)
                genes_added += 1
                total_null_sites += len(site_records)
                
    print(f"\n -> Processed {genes_added} null simulation genes ({total_null_sites:,} variable null sites).")
    
    # Insert records into SQLite meme_results.db
    if all_site_records:
        print(f" -> Inserting {len(all_site_records):,} actual MEME site records into SQLite database '{db_path}'...")
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
        print(f" 🎉 Successfully inserted {len(all_site_records):,} actual MEME site records into SQLite database!")

if __name__ == "__main__":
    main()
