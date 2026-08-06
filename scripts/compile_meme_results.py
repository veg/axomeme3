#!/usr/bin/env python3
"""
compile_meme_results.py
-----------------------
Scans the MEME results directory, parses completed gzipped JSON files,
and compiles results into an incremental SQLite database.
"""

import os
import re
import json
import gzip
import sqlite3
import glob
import multiprocessing as mp
from datetime import datetime

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

def translate_codon(codon):
    codon = codon.upper()
    if len(codon) != 3 or '-' in codon or 'N' in codon:
        return '?'
    return GENETIC_CODE.get(codon, '?')

# Newick Tree Parser Classes and Functions
class Node:
    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None

def parse_newick_to_tree(newick_str):
    root = Node()
    current = root
    i = 0
    while i < len(newick_str):
        c = newick_str[i]
        if c == '(':
            child = Node()
            child.parent = current
            current.children.append(child)
            current = child
            i += 1
        elif c == ',':
            parent = current.parent
            child = Node()
            child.parent = parent
            parent.children.append(child)
            current = child
            i += 1
        elif c == ')':
            current = current.parent
            i += 1
        elif c == ';' or c == ' ':
            i += 1
        else:
            match = re.match(r'[^(),;:]+', newick_str[i:])
            if match:
                name = match.group(0).strip()
                name = re.sub(r'\{[^}]*\}', '', name) # remove annotations
                current.name = name
                i += len(match.group(0))
            if i < len(newick_str) and newick_str[i] == ':':
                i += 1
                match_len = re.match(r'[0-9.eE+-]+', newick_str[i:])
                if match_len:
                    i += len(match_len.group(0))
    return root

def assign_internal_names(node, counter=[1]):
    if not node.name:
        node.name = f"ANON_NODE_{counter[0]}"
        counter[0] += 1
    for child in node.children:
        assign_internal_names(child, counter)

def build_parent_map(node, parent_map):
    for child in node.children:
        if child.name and node.name:
            parent_map[child.name] = node.name
        build_parent_map(child, parent_map)

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gene_results (
        gene_name TEXT PRIMARY KEY,
        num_seqs INTEGER,
        num_sites INTEGER,
        log_likelihood REAL,
        aic_c REAL,
        num_selected_sites INTEGER,
        runtime_sec REAL
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS site_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_name TEXT,
        site_index INTEGER,
        alpha REAL,
        beta_neg REAL,
        p_neg REAL,
        beta_pos REAL,
        p_pos REAL,
        lrt REAL,
        p_value REAL,
        q_value REAL,
        num_branches_under_selection INTEGER,
        total_branch_length REAL,
        is_significant INTEGER,
        FOREIGN KEY(gene_name) REFERENCES gene_results(gene_name)
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS site_substitutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_name TEXT,
        site_index INTEGER,
        branch_name TEXT,
        ancestral_codon TEXT,
        derived_codon TEXT,
        ancestral_aa TEXT,
        derived_aa TEXT,
        is_synonymous INTEGER,
        FOREIGN KEY(gene_name) REFERENCES gene_results(gene_name)
    )""")
    
    # Create indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_site_gene ON site_results(gene_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_site_p_val ON site_results(p_value)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subst_gene ON site_substitutions(gene_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subst_branch ON site_substitutions(branch_name)")
    conn.commit()
    conn.close()

def parse_single_file(args):
    file_path, gene_name = args
    try:
        try:
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)
        except Exception:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
        if not data or 'input' not in data:
            return None
            
        num_seqs = data['input']['number of sequences']
        num_sites = data['input']['number of sites']
        
        # Get fit statistics (Log L, AICc)
        log_l = None
        aic_c = None
        fits = data.get('fits', {})
        for fit_name, fit_data in fits.items():
            if 'Log Likelihood' in fit_data:
                log_l = fit_data['Log Likelihood']
                aic_c = fit_data.get('AIC-c', None)
                break
                
        # Parse tree to get node parent mapping
        trees = data['input'].get('trees', {})
        newick_str = trees.get('0', '')
        parent_map = {}
        if newick_str:
            root = parse_newick_to_tree(newick_str)
            root.name = 'root'
            assign_internal_names(root)
            build_parent_map(root, parent_map)
            
        # Parse timers
        timers = data.get('timers', {})
        runtime = timers.get('Total time', {}).get('timer', 0.0)
        
        # Parse MLE
        mle_data = data.get('MLE', {})
        mle_content = mle_data.get('content', {}).get('0', [])
        
        raw_sites = []
        for s_idx, row in enumerate(mle_content):
            alpha = row[0]
            beta_neg = row[1]
            p_neg = row[2]
            beta_pos = row[3]
            p_pos = row[4]
            lrt = row[5]
            p_val = row[6]
            num_branches = row[7]
            total_branch_len = row[8]
            raw_sites.append({
                's_idx': s_idx,
                'alpha': alpha,
                'beta_neg': beta_neg,
                'p_neg': p_neg,
                'beta_pos': beta_pos,
                'p_pos': p_pos,
                'lrt': lrt,
                'p_value': p_val,
                'num_branches': num_branches,
                'total_branch_len': total_branch_len
            })

        # Calculate BH Q-values for the gene's sites
        M = len(raw_sites)
        q_values = [1.0] * M
        if M > 0:
            # Sort by p_value
            sorted_indices = sorted(range(M), key=lambda i: raw_sites[i]['p_value'])
            # Raw Q-values: raw_q_i = p_val * M / (rank + 1)
            raw_qs = [raw_sites[idx]['p_value'] * M / (k + 1) for k, idx in enumerate(sorted_indices)]
            # Monotonicity check: Q_i = min_{j >= i} raw_q_j
            running_min = 1.0
            for k in range(M - 1, -1, -1):
                running_min = min(running_min, raw_qs[k])
                idx = sorted_indices[k]
                q_values[idx] = min(running_min, 1.0)

        site_records = []
        num_selected_sites = 0
        for s_idx, item in enumerate(raw_sites):
            q_val = q_values[s_idx]
            is_sig = 1 if q_val <= 0.10 else 0
            if is_sig:
                num_selected_sites += 1
            site_records.append((
                gene_name,
                item['s_idx'] + 1, # 1-based index
                item['alpha'],
                item['beta_neg'],
                item['p_neg'],
                item['beta_pos'],
                item['p_pos'],
                item['lrt'],
                item['p_value'],
                q_val,
                item['num_branches'],
                item['total_branch_len'],
                is_sig
            ))
            
        # Parse substitutions mapping if available
        subst_records = []
        subs_partition = data.get('substitutions', {}).get('0', {})
        for s_idx_str, states in subs_partition.items():
            if not states:
                continue
            s_idx = int(s_idx_str) + 1 # 1-based index
            for node_name, derived in states.items():
                if node_name == 'root':
                    continue
                # Find ancestral state
                ancestor = parent_map.get(node_name)
                while ancestor and ancestor not in states:
                    ancestor = parent_map.get(ancestor)
                    
                ancestral = states[ancestor] if ancestor else '---'
                
                # If both ancestral and derived codons are valid and different
                if ancestral != '---' and derived != '---' and ancestral != derived:
                    # Translate to amino acids
                    aa_anc = translate_codon(ancestral)
                    aa_der = translate_codon(derived)
                    
                    if aa_anc == '?' or aa_der == '?':
                        is_syn = None
                    else:
                        is_syn = 1 if aa_anc == aa_der else 0
                        
                    subst_records.append((
                        gene_name,
                        s_idx,
                        node_name,
                        ancestral,
                        derived,
                        aa_anc,
                        aa_der,
                        is_syn
                    ))
                    
        return {
            'gene': (gene_name, num_seqs, num_sites, log_l, aic_c, num_selected_sites, runtime),
            'sites': site_records,
            'substitutions': subst_records
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error parsing {file_path}: {e}")
        return None

def main():
    base_dir = "/Users/sergei/Projects/TOGA_MEME"
    if not os.path.exists(base_dir) or not os.path.exists(os.path.join(base_dir, "msa")):
        base_dir = "/Users/sergei/Dropbox/TOGA2026"
    msa_dir = os.path.join(base_dir, "msa")
    results_dir = os.path.join(base_dir, "meme_results")
    db_path = os.path.join(base_dir, "meme_results.db")
    
    # 1. Scan msa directory for genes list
    print("Scanning alignments directory for target gene list...")
    if not os.path.exists(msa_dir):
        print(f"Error: alignments directory not found at {msa_dir}!")
        return
        
    msa_files = sorted(glob.glob(os.path.join(msa_dir, "*.gz")))
    genes = []
    for path in msa_files:
        filename = os.path.basename(path)
        gene = filename[:-3] if filename.endswith(".gz") else filename
        genes.append(gene)
        
    print(f"Loaded {len(genes)} target genes from msa files.")
    
    # 2. Init SQLite DB
    print(f"Initializing database: {db_path}")
    init_db(db_path)
    
    # Check already processed genes to enable incremental compilation
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT gene_name FROM gene_results")
    processed = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    print(f"Already processed: {len(processed)} genes.")
    
    # Filter genes to process
    to_process = []
    for g in genes:
        if g not in processed:
            file_path = os.path.join(results_dir, f"{g}.MEME.json.gz")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                to_process.append((file_path, g))
                
    print(f"New completed files to process: {len(to_process)}")
    if not to_process:
        print("No new files to compile. Exiting.")
        return
        
    # 3. Process in parallel
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Processing in parallel using {num_workers} workers...")
    pool = mp.Pool(num_workers)
    
    results = []
    completed = 0
    for res in pool.imap_unordered(parse_single_file, to_process):
        if res:
            results.append(res)
        completed += 1
        if completed % 10 == 0 or completed == len(to_process):
            print(f"Parsed: {completed}/{len(to_process)} files...")
            
    pool.close()
    pool.join()
    
    # 4. Write results to database in a single transaction
    print(f"Writing {len(results)} results to SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("BEGIN TRANSACTION")
    
    for r in results:
        # Insert gene stats
        cursor.execute("""
        INSERT OR REPLACE INTO gene_results (gene_name, num_seqs, num_sites, log_likelihood, aic_c, num_selected_sites, runtime_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, r['gene'])
        
        # Insert site details
        if r['sites']:
            cursor.executemany("""
            INSERT INTO site_results (gene_name, site_index, alpha, beta_neg, p_neg, beta_pos, p_pos, lrt, p_value, q_value, num_branches_under_selection, total_branch_length, is_significant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, r['sites'])
            
        # Insert site substitutions
        if r['substitutions']:
            cursor.executemany("""
            INSERT INTO site_substitutions (gene_name, site_index, branch_name, ancestral_codon, derived_codon, ancestral_aa, derived_aa, is_synonymous)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, r['substitutions'])
            
    conn.commit()
    
    # Recalculate stats
    cursor.execute("SELECT COUNT(*) FROM gene_results")
    total_db = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(num_selected_sites) FROM gene_results")
    total_sig_sites = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(*) FROM site_substitutions")
    total_substs = cursor.fetchone()[0]
    
    conn.close()
    
    print("\nCompilation completed successfully!")
    print(f"Total compiled genes in DB: {total_db}")
    print(f"Total significant sites found (q <= 0.10): {total_sig_sites}")
    print(f"Total mapped codon substitutions: {total_substs}")

if __name__ == '__main__':
    main()
