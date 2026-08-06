#!/usr/bin/env python3
"""
compile_absrel_results.py
--------------------------
Scans the aBSREL results directory, parses completed gzipped JSON files,
and compiles results into an incremental SQLite database.
"""

import os
import re
import json
import gzip
import sqlite3
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

def get_leaves(node, leaves):
    if not node.children:
        leaves.add(node.name)
    for child in node.children:
        get_leaves(child, leaves)

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gene_results (
        gene_name TEXT PRIMARY KEY,
        transcript_id TEXT,
        num_seqs INTEGER,
        num_sites INTEGER,
        log_likelihood REAL,
        aic_c REAL,
        num_tested INTEGER,
        num_significant INTEGER,
        runtime_sec REAL
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS branch_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_name TEXT,
        branch_name TEXT,
        is_leaf INTEGER,
        lrt REAL,
        uncorrected_p_value REAL,
        corrected_p_value REAL,
        is_significant INTEGER,
        branch_length REAL,
        num_rate_classes INTEGER,
        omega_max REAL,
        proportion_max REAL,
        rate_distribution TEXT,
        FOREIGN KEY(gene_name) REFERENCES gene_results(gene_name)
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS selected_sites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_name TEXT,
        branch_name TEXT,
        site_index INTEGER,
        posterior_prob REAL,
        bayes_factor REAL,
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_branch_gene ON branch_results(gene_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_branch_name ON branch_results(branch_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sites_gene ON selected_sites(gene_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subst_gene ON site_substitutions(gene_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subst_branch ON site_substitutions(branch_name)")
    conn.commit()
    conn.close()

def parse_single_file(args):
    file_path, gene_name, transcript_id = args
    try:
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
            
        # If the file is created but empty (still running), skip
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
                # Calculate AICc if possible
                num_params = fit_data.get('Estimated Parameters', 0)
                if num_sites > 0 and (num_sites * 3 - num_params - 1) > 0:
                    # N = num_sites * 3 (alignment columns)
                    n_samples = num_sites * 3
                    aic_c = -2 * log_l + 2 * num_params + (2 * num_params * (num_params + 1)) / (n_samples - num_params - 1)
                break
                
        # Parse tree to get node parent mapping
        trees = data['input'].get('trees', {})
        newick_str = trees.get('0', '')
        parent_map = {}
        leaves = set()
        if newick_str:
            root = parse_newick_to_tree(newick_str)
            root.name = 'root'
            assign_internal_names(root)
            build_parent_map(root, parent_map)
            get_leaves(root, leaves)
            
        # Parse timers to get running time
        timers = data.get('timers', {})
        runtime = timers.get('Overall', {}).get('timer', 0.0)
        
        # Parse branch attributes
        branch_attrs = data.get('branch attributes', {}).get('0', {})
        tested_info = data.get('test results', {})
        num_tested = tested_info.get('tested', 0)
        num_significant = tested_info.get('positive test results', 0)
        
        branch_records = []
        site_records = []
        subst_records = []
        
        for branch_name, attrs in branch_attrs.items():
            # Check if this branch was tested (i.e. has a p-value)
            if 'Corrected P-value' not in attrs:
                continue
                
            lrt = attrs.get('LRT', 0.0)
            uncorrected_p = attrs.get('Uncorrected P-value', 1.0)
            corrected_p = attrs.get('Corrected P-value', 1.0)
            is_sig = 1 if corrected_p <= 0.05 else 0
            branch_len = attrs.get('Full adaptive model', 0.0)
            
            rate_dist = attrs.get('Rate Distributions', [])
            num_rates = len(rate_dist)
            omega_max = max(x[0] for x in rate_dist) if rate_dist else 0.0
            proportion_max = next(x[1] for x in rate_dist if x[0] == omega_max) if rate_dist else 0.0
            
            is_leaf = 1 if branch_name in leaves else 0
            
            branch_records.append((
                gene_name,
                branch_name,
                is_leaf,
                lrt,
                uncorrected_p,
                corrected_p,
                is_sig,
                branch_len,
                num_rates,
                omega_max,
                proportion_max,
                json.dumps(rate_dist)
            ))
            
            # If the branch is significant, extract sites with EBF >= 100 for dN/dS > 1
            if is_sig == 1 and rate_dist:
                # Find indices of selection rate classes (omega > 1)
                sel_indices = [idx for idx, x in enumerate(rate_dist) if x[0] > 1.0]
                if sel_indices:
                    # Prior probability of selection classes
                    p_sel = sum(rate_dist[idx][1] for idx in sel_indices)
                    posterior = attrs.get('posterior', [])
                    
                    # If prior is < 1.0 (purifying class exists) and posterior is available
                    if p_sel < 1.0 and posterior:
                        # For each site
                        for s_idx in range(len(posterior[0])):
                            # Sum posterior probabilities of all selection classes
                            P_s = sum(posterior[c_idx][s_idx] for c_idx in sel_indices)
                            # Compute Empirical Bayes Factor (EBF)
                            # EBF = (P_s / (1 - P_s)) / (p_sel / (1 - p_sel))
                            ebf = (P_s / (1.0 - P_s + 1e-10)) / (p_sel / (1.0 - p_sel + 1e-10))
                            
                            if ebf >= 100.0:
                                site_records.append((
                                    gene_name,
                                    branch_name,
                                    s_idx + 1, # 1-based index
                                    P_s,
                                    ebf
                                ))
                                
        # Parse substitutions mapping if available
        # substitutions[partition][site_idx] = {node_name: codon, "root": root_codon}
        subs_partition = data.get('substitutions', {}).get('0', {})
        for s_idx_str, states in subs_partition.items():
            s_idx = int(s_idx_str) + 1 # 1-based index
            for node_name, derived in states.items():
                if node_name == 'root':
                    continue
                # Find ancestral state (traverse parent map until we find a node in the dictionary)
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
            'gene': (gene_name, transcript_id, num_seqs, num_sites, log_l, aic_c, num_tested, num_significant, runtime),
            'branches': branch_records,
            'sites': site_records,
            'substitutions': subst_records
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def main():
    base_dir = "/home/sergei/Projects/TOGA"
    absrel_dir = os.path.join(base_dir, "absrel")
    db_path = os.path.join(base_dir, "absrel_results.db")
    gene_list_path = os.path.join(base_dir, "gene_list.txt")
    
    # 1. Load gene list
    print("Loading gene list...")
    if not os.path.exists(gene_list_path):
        print(f"Error: gene list not found at {gene_list_path}!")
        return
        
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(genes)} target genes.")
    
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
            file_path = os.path.join(absrel_dir, f"{g}.json.gz")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                # We pass transcript_id as None, which can be updated later if needed
                to_process.append((file_path, g, None))
                
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
        if completed % 100 == 0 or completed == len(to_process):
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
        INSERT OR REPLACE INTO gene_results (gene_name, transcript_id, num_seqs, num_sites, log_likelihood, aic_c, num_tested, num_significant, runtime_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, r['gene'])
        
        # Insert branch details
        if r['branches']:
            cursor.executemany("""
            INSERT INTO branch_results (gene_name, branch_name, is_leaf, lrt, uncorrected_p_value, corrected_p_value, is_significant, branch_length, num_rate_classes, omega_max, proportion_max, rate_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, r['branches'])
            
        # Insert selected sites
        if r['sites']:
            cursor.executemany("""
            INSERT INTO selected_sites (gene_name, branch_name, site_index, posterior_prob, bayes_factor)
            VALUES (?, ?, ?, ?, ?)
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
    cursor.execute("SELECT SUM(num_significant) FROM gene_results")
    total_sig_branches = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(*) FROM selected_sites")
    total_selected_sites = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM site_substitutions")
    total_substs = cursor.fetchone()[0]
    
    conn.close()
    
    print("\nCompilation completed successfully!")
    print(f"Total compiled genes in DB: {total_db}")
    print(f"Total significant branches found: {total_sig_branches}")
    print(f"Total site-level selection events (EBF >= 100): {total_selected_sites}")
    print(f"Total mapped codon substitutions: {total_substs}")

if __name__ == '__main__':
    main()
