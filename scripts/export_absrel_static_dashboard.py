#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import gzip
import shutil

DB_PATH = "absrel_results.db"
HTML_PATH = "docs/absrel_dashboard.html"
EXPORT_DIR = "toga-absrel"

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

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        sys.exit(1)
        
    # 1. Recreate clean toga-absrel directory structure
    print(f"Creating export directories under: {EXPORT_DIR}")
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
        
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api", "gene"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api", "alignment"), exist_ok=True)
    
    # Connect to database
    conn = get_db_connection = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 2. Export api/summary.json
    print("Exporting summary stats...")
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
    
    summary = {
        'total_genes': total_genes,
        'total_sig_branches': total_sig_branches,
        'total_selected_sites': total_selected_sites,
        'gold_count': gold_count,
        'silver_count': silver_count,
        'likely_error_count': likely_error_count
    }
    
    with open(os.path.join(EXPORT_DIR, "api", "summary.json"), "w") as f:
        json.dump(summary, f)
        
    # 3. Export api/genes.json
    print("Exporting genes list...")
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
        
    with open(os.path.join(EXPORT_DIR, "api", "genes.json"), "w") as f:
        json.dump(genes, f)
        
    # 4. Export api/gene/<gene_name>.json and alignments
    print(f"Exporting details and alignments for {len(genes)} genes...")
    for idx, g in enumerate(genes):
        gene_name = g["gene_name"]
        
        # Get general gene stats
        c.execute("select num_seqs, num_sites, log_likelihood, aic_c, num_tested, num_significant, runtime_sec from gene_results where gene_name=?", (gene_name,))
        num_seqs, num_sites, log_likelihood, aic_c, num_tested, num_significant, runtime_sec = c.fetchone()
        
        # Parse tree structure to calculate clade descendants list
        tree_str = None
        align_path = f"msa/{gene_name}.gz"
        descendants_map = {}
        seqs = {}
        if os.path.exists(align_path):
            try:
                seqs, tree_str = parse_nexus_gz(align_path)
                parents = parse_newick(tree_str)
                children = {}
                for child, parent in parents.items():
                    children.setdefault(parent, []).append(child)
                leaves_set = set(seqs.keys())
                
                all_nodes = set(parents.keys()).union(set(parents.values()))
                for node in all_nodes:
                    if node:
                        descendants_map[node.upper()] = get_descendant_leaves(node, children, leaves_set)
            except Exception as e:
                print(f"Error parsing tree for {gene_name}: {e}")
                
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
                
                # Fetch codon substitutions
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
            
        # Get all selected sites for plot
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
        
        with open(os.path.join(EXPORT_DIR, "api", "gene", f"{gene_name}.json"), "w") as f:
            json.dump(payload, f)
            
        # Check and write alignment files
        if seqs:
            try:
                first_seq = next(iter(seqs.values()))
                total_codons = len(first_seq) // 3
                
                os.makedirs(os.path.join(EXPORT_DIR, "api", "alignment", gene_name), exist_ok=True)
                
                # Fetch all unique site indices under selection for this gene
                unique_sites = set(s["site_index"] for s in all_selected_sites)
                for site_idx in unique_sites:
                    win_start_codon = max(1, site_idx - 5)
                    win_end_codon = min(total_codons, site_idx + 5)
                    
                    # Sort species (exact same logic as in backend API)
                    ref_seq = seqs.get('hg', None)
                    if not ref_seq:
                        codon_counts = {}
                        nuc_start = (site_idx - 1) * 3
                        for seq in seqs.values():
                            if len(seq) > nuc_start + 2:
                                c = seq[nuc_start:nuc_start+3].upper()
                                if '-' not in c and 'N' not in c:
                                    codon_counts[c] = codon_counts.get(c, 0) + 1
                        ref_codon = max(codon_counts, key=codon_counts.get) if codon_counts else "---"
                    else:
                        nuc_start = (site_idx - 1) * 3
                        ref_codon = ref_seq[nuc_start:nuc_start+3].upper()

                    has_change = []
                    no_change = []
                    nuc_start = (site_idx - 1) * 3
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
                            n_start = (c_idx - 1) * 3
                            row_codons.append({
                                "site_index": c_idx,
                                "sequence": seq[n_start:n_start+3].upper()
                            })
                        alignment_window.append({
                            "taxon": spec,
                            "codons": row_codons
                        })
                        
                    with open(os.path.join(EXPORT_DIR, "api", "alignment", gene_name, f"{site_idx}.json"), "w") as f:
                        json.dump({"alignment": alignment_window}, f)
            except Exception as e:
                print(f"Error rendering MSA for gene {gene_name}: {e}")
                
        if (idx + 1) % 50 == 0 or (idx + 1) == len(genes):
            print(f"  Processed {idx + 1} / {len(genes)} genes...")
            
    conn.close()
    
    # 5. Modify and copy HTML fetches
    print("Modifying HTML fetches for static relative JSON requests...")
    with open(HTML_PATH, "r") as f:
        html = f.read()
        
    html = html.replace("fetch('api/summary.json')", "fetch('api/summary.json')")
    html = html.replace("fetch('api/genes.json')", "fetch('api/genes.json')")
    html = html.replace("fetch(`api/gene/${geneName}.json`)", "fetch(`api/gene/${geneName}.json`)") # was /api/gene
    # Fix absolute URLs
    html = html.replace("fetch(`api/gene/${geneName}.json`)", "fetch(`api/gene/${geneName}.json`)")
    html = html.replace("fetch(`api/alignment/${gene}/${siteIdx}.json`)", "fetch(`api/alignment/${gene}/${siteIdx}.json`)")
    # Make sure we replace absolute paths in html
    html = html.replace("fetch('api/summary.json')", "fetch('api/summary.json')")
    html = html.replace("fetch('api/genes.json')", "fetch('api/genes.json')")
    html = html.replace("fetch(`api/gene/${geneName}.json`)", "fetch(`api/gene/${geneName}.json`)")
    html = html.replace("fetch(`api/alignment/${gene}/${siteIdx}.json`)", "fetch(`api/alignment/${gene}/${siteIdx}.json`)")
    
    # Actually wait, in HTML_PATH we wrote:
    # fetch('api/summary.json')
    # fetch('api/genes.json')
    # fetch(`api/gene/${geneName}.json`)
    # fetch(`api/alignment/${gene}/${siteIdx}.json`)
    # So they are already relative! No modification needed unless there are absolute slash prefix issues. Let's make sure.
    # In docs/absrel_dashboard.html we wrote:
    # fetch('api/summary.json')
    # fetch('api/genes.json')
    # fetch(`api/gene/${geneName}.json`)
    # fetch(`api/alignment/${gene}/${siteIdx}.json`)
    # So yes, they are completely relative!
    
    with open(os.path.join(EXPORT_DIR, "index.html"), "w") as f:
        f.write(html)
        
    print(f"Static site export to '{EXPORT_DIR}' completed successfully!")

if __name__ == "__main__":
    main()
