#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import urllib.parse
import gzip
from http.server import HTTPServer, BaseHTTPRequestHandler

DB_PATH = "absrel_results.db"
HTML_PATH = "docs/absrel_dashboard.html"

# Global cache for summary stats
summary_cache = {}

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def cache_summary_stats():
    print("Pre-calculating summary statistics...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        return
        
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("select count(*) from gene_results")
    total_genes = c.fetchone()[0]
    
    c.execute("select count(*) from branch_results where is_significant=1")
    total_sig_branches = c.fetchone()[0]
    
    c.execute("select count(*) from selected_sites")
    total_selected_sites = c.fetchone()[0]
    
    c.execute("select count(*) from selected_sites where classification='GOLD'")
    gold_count = c.fetchone()[0]
    
    c.execute("select count(*) from selected_sites where classification='SILVER'")
    silver_count = c.fetchone()[0]
    
    c.execute("select count(*) from selected_sites where classification='LIKELY_ERROR'")
    likely_error_count = c.fetchone()[0]
    
    summary_cache['total_genes'] = total_genes
    summary_cache['total_sig_branches'] = total_sig_branches
    summary_cache['total_selected_sites'] = total_selected_sites
    summary_cache['gold_count'] = gold_count
    summary_cache['silver_count'] = silver_count
    summary_cache['likely_error_count'] = likely_error_count
    
    conn.close()
    print("Summary statistics cached successfully.")

def parse_nexus_gz(filepath):
    taxlabels = []
    sequences = []
    tree_str = None
    in_taxlabels = False
    in_matrix = False
    in_trees = False
    
    with gzip.open(filepath, 'rt') as f:
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
                    continue
                    
    mapped = {}
    for label, seq in zip(taxlabels, sequences):
        mapped[label] = seq
    return mapped, tree_str

def parse_newick(newick_str):
    if not newick_str:
        return {}
    cleaned = re.sub(r'\{[^}]*\}', '', newick_str)
    cleaned = re.sub(r':[0-9.e-]+', '', cleaned)
    
    parents = {}
    tokens = []
    i = 0
    while i < len(cleaned):
        c = cleaned[i]
        if c in '(),;':
            tokens.append(c)
            i += 1
        else:
            name = []
            while i < len(cleaned) and cleaned[i] not in '(),;':
                name.append(cleaned[i])
                i += 1
            tokens.append(''.join(name).strip())
            
    group_stack = []
    current_group = []
    anon_counter = 0
    
    token_idx = 0
    while token_idx < len(tokens):
        t = tokens[token_idx]
        if t == '(':
            group_stack.append(current_group)
            current_group = []
            token_idx += 1
        elif t == ')':
            token_idx += 1
            parent_name = ""
            if token_idx < len(tokens) and tokens[token_idx] not in '(),;':
                parent_name = tokens[token_idx]
                token_idx += 1
            else:
                anon_counter += 1
                parent_name = f"ANON_{anon_counter}"
            
            for child in current_group:
                parents[child] = parent_name
                
            parent_group = group_stack.pop()
            parent_group.append(parent_name)
            current_group = parent_group
        elif t in ',;':
            token_idx += 1
        else:
            current_group.append(t)
            token_idx += 1
            
    return parents

import re

def get_descendant_leaves(node, children_map, leaves_set):
    if node not in children_map:
        return [node]
    desc_leaves = []
    for child in children_map[node]:
        desc_leaves.extend(get_descendant_leaves(child, children_map, leaves_set))
    return desc_leaves

class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # Normalize API requests to strip trailing .json extension
        if path.startswith("/api/"):
            if path.endswith(".json"):
                path = path[:-5]
        
        # Serve main HTML page
        if path == "/" or path == "/index.html":
            if not os.path.exists(HTML_PATH):
                self.send_error(404, "aBSREL Dashboard HTML file not found!")
                return
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open(HTML_PATH, 'rb') as f:
                self.wfile.write(f.read())
            return
            
        # API: Summary Stats
        if path == "/api/summary":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(summary_cache).encode('utf-8'))
            return
            
        # API: Genes list with significant branches info
        if path == "/api/genes":
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""
                select g.gene_name, g.num_seqs, g.num_sites, g.num_tested, g.num_significant,
                       (select count(*) from selected_sites s where s.gene_name=g.gene_name) as selected_sites_count,
                       (select count(*) from selected_sites s where s.gene_name=g.gene_name and s.classification='GOLD') as gold_count,
                       (select count(*) from selected_sites s where s.gene_name=g.gene_name and s.classification='SILVER') as silver_count,
                       (select count(*) from selected_sites s where s.gene_name=g.gene_name and s.classification='LIKELY_ERROR') as likely_error_count
                from gene_results g
                where g.num_significant > 0
                order by g.gene_name asc
            """)
            genes = []
            for row in c.fetchall():
                genes.append({
                    "gene_name": row[0],
                    "num_seqs": row[1],
                    "num_sites": row[2],
                    "num_tested": row[3],
                    "num_significant": row[4],
                    "num_selected_sites": row[5],
                    "gold_sites": row[6],
                    "silver_sites": row[7],
                    "likely_error_sites": row[8]
                })
            conn.close()
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(genes).encode('utf-8'))
            return
            
        # API: Gene details with significant branches and selected sites
        if path.startswith("/api/gene/"):
            gene_name = path[len("/api/gene/"):]
            gene_name = urllib.parse.unquote(gene_name)
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get general gene stats
            c.execute("select num_seqs, num_sites, log_likelihood, aic_c, num_tested, num_significant, runtime_sec from gene_results where gene_name=?", (gene_name,))
            gene_info = c.fetchone()
            if not gene_info:
                conn.close()
                self.send_error(404, f"Gene {gene_name} not found!")
                return
                
            num_seqs, num_sites, log_likelihood, aic_c, num_tested, num_significant, runtime_sec = gene_info
            
            # Get Newick tree from alignment if it exists
            tree_str = None
            align_path = f"msa/{gene_name}.gz"
            descendants_map = {}
            if os.path.exists(align_path):
                try:
                    seqs, tree_str = parse_nexus_gz(align_path)
                    parents = parse_newick(tree_str)
                    children = {}
                    for child, parent in parents.items():
                        children.setdefault(parent, []).append(child)
                    leaves_set = set(seqs.keys())
                    
                    # Pre-calculate descendants for all nodes (to know carrying leaves)
                    all_nodes = set(parents.keys()).union(set(parents.values()))
                    for node in all_nodes:
                        if node:
                            descendants_map[node.upper()] = get_descendant_leaves(node, children, leaves_set)
                except Exception as e:
                    print(f"Error parsing alignment tree for {gene_name}: {e}")
            
            # Get significant branches on this gene
            c.execute("""
                select branch_name, is_leaf, lrt, uncorrected_p_value, corrected_p_value, 
                       branch_length, num_rate_classes, omega_max, proportion_max, rate_distribution
                from branch_results
                where gene_name=? and is_significant=1
                order by corrected_p_value asc
            """, (gene_name,))
            
            significant_branches = []
            for row in c.fetchall():
                branch_name = row[0]
                is_leaf = row[1]
                lrt = row[2]
                uncorrected_p = row[3]
                corrected_p = row[4]
                branch_len = row[5]
                num_rates = row[6]
                omega_max = row[7]
                prop_max = row[8]
                rate_dist_str = row[9]
                
                # Fetch selected sites on this branch
                c.execute("""
                    select site_index, posterior_prob, bayes_factor, classification, filter_reason
                    from selected_sites
                    where gene_name=? and branch_name=?
                    order by site_index asc
                """, (gene_name, branch_name))
                
                selected_sites = []
                for site in c.fetchall():
                    s_idx = site[0]
                    p_prob = site[1]
                    bf = site[2]
                    classification = site[3]
                    reason = site[4]
                    
                    # Fetch codon substitutions at this site on this branch
                    c.execute("""
                        select ancestral_codon, derived_codon, ancestral_aa, derived_aa, is_synonymous
                        from site_substitutions
                        where gene_name=? and site_index=? and branch_name=?
                    """, (gene_name, s_idx, branch_name))
                    subst_info = c.fetchall()
                    substs = []
                    for sub in subst_info:
                        syn_label = "syn" if sub[4] == 1 else "non-syn"
                        substs.append(f"{sub[2]}->{sub[3]}({sub[0]}->{sub[1]}:{syn_label})")
                        
                    selected_sites.append({
                        "site_index": s_idx,
                        "posterior_prob": p_prob,
                        "bayes_factor": bf,
                        "classification": classification,
                        "filter_reason": reason,
                        "substitutions": substs
                    })
                    
                significant_branches.append({
                    "branch_name": branch_name,
                    "is_leaf": is_leaf,
                    "lrt": lrt,
                    "uncorrected_p_value": uncorrected_p,
                    "corrected_p_value": corrected_p,
                    "branch_length": branch_len,
                    "num_rate_classes": num_rates,
                    "omega_max": omega_max,
                    "proportion_max": prop_max,
                    "rate_distribution": json.loads(rate_dist_str) if rate_dist_str else [],
                    "carrying_leaves": descendants_map.get(branch_name, [branch_name]),
                    "selected_sites": selected_sites
                })
                
            # Get all selected sites for the Manhattan scatter plot
            c.execute("""
                select site_index, branch_name, bayes_factor, classification
                from selected_sites
                where gene_name=?
                order by site_index asc
            """, (gene_name,))
            all_selected_sites = []
            for row in c.fetchall():
                all_selected_sites.append({
                    "site_index": row[0],
                    "branch_name": row[1],
                    "bayes_factor": row[2],
                    "classification": row[3]
                })
                
            conn.close()
            
            payload = {
                "gene_name": gene_name,
                "num_seqs": num_seqs,
                "num_sites": num_sites,
                "log_likelihood": log_likelihood,
                "aic_c": aic_c,
                "num_tested": num_tested,
                "num_significant": num_significant,
                "runtime_sec": runtime_sec,
                "tree_str": tree_str,
                "significant_branches": significant_branches,
                "all_selected_sites": all_selected_sites
            }
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return
            
        # API: Alignment window (identical to MEME dashboard, but returns carrying leaves highlighting metadata)
        if path.startswith("/api/alignment/"):
            # Format: /api/alignment/<gene_name>/<site_index>
            parts = path[len("/api/alignment/"):].split('/')
            if len(parts) < 2:
                self.send_error(400, "Invalid alignment request format")
                return
                
            gene_name = urllib.parse.unquote(parts[0])
            try:
                site_index = int(parts[1])
            except ValueError:
                self.send_error(400, "Invalid site index")
                return
                
            align_path = f"msa/{gene_name}.gz"
            if not os.path.exists(align_path):
                self.send_error(404, f"Alignment for {gene_name} not found")
                return
                
            try:
                seqs, tree_str = parse_nexus_gz(align_path)
            except Exception as e:
                self.send_error(500, f"Error parsing alignment file: {e}")
                return
                
            # Window size: ±5 codons
            win_start_codon = max(1, site_index - 5)
            first_seq = next(iter(seqs.values()))
            total_codons = len(first_seq) // 3
            win_end_codon = min(total_codons, site_index + 5)
            
            # Find the reference codon (from hg, or consensus)
            ref_seq = seqs.get('hg', None)
            if not ref_seq:
                codon_counts = {}
                nuc_start = (site_index - 1) * 3
                for seq in seqs.values():
                    if len(seq) > nuc_start + 2:
                        c = seq[nuc_start:nuc_start+3].upper()
                        if '-' not in c and 'N' not in c:
                            codon_counts[c] = codon_counts.get(c, 0) + 1
                ref_codon = max(codon_counts, key=codon_counts.get) if codon_counts else "---"
            else:
                nuc_start = (site_index - 1) * 3
                ref_codon = ref_seq[nuc_start:nuc_start+3].upper()

            # Separate other species into has-change and no-change
            has_change = []
            no_change = []
            nuc_start = (site_index - 1) * 3
            for spec, seq in seqs.items():
                if spec == 'hg':
                    continue
                codon = seq[nuc_start:nuc_start+3].upper() if len(seq) > nuc_start + 2 else "---"
                is_diff = (codon != ref_codon) and ('-' not in codon) and ('N' not in codon)
                if is_diff:
                    has_change.append(spec)
                else:
                    no_change.append(spec)
            
            has_change.sort()
            no_change.sort()
            
            sorted_species = []
            if 'hg' in seqs:
                sorted_species.append('hg')
            sorted_species.extend(has_change)
            sorted_species.extend(no_change)
            
            alignment_window = []
            for spec in sorted_species:
                seq = seqs[spec]
                row_codons = []
                for c_idx in range(win_start_codon, win_end_codon + 1):
                    nuc_start = (c_idx - 1) * 3
                    row_codons.append({
                        "site_index": c_idx,
                        "sequence": seq[nuc_start:nuc_start+3].upper()
                    })
                alignment_window.append({
                    "taxon": spec,
                    "codons": row_codons
                })
                
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"alignment": alignment_window}).encode('utf-8'))
            return
            
        self.send_error(404, "Endpoint not found")

def run(port=8083):
    cache_summary_stats()
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    print(f"\n🚀 TOGA aBSREL Dashboard is running locally on http://localhost:{port}")
    print("Press Ctrl+C to terminate the server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        sys.exit(0)

if __name__ == "__main__":
    port = 8083
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port)
