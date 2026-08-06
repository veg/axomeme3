#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import gzip
import shutil

DB_PATH = "meme_results.db"
HTML_PATH = "docs/meme_dashboard.html"
EXPORT_DIR = "toga-meme"

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
    'TGC':'C', 'TGT':'C', 'TGG':'W',
}

def parse_nexus_gz(filepath):
    taxlabels = []
    sequences = []
    in_taxlabels = False
    in_matrix = False
    
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
                    
    mapped = {}
    for label, seq in zip(taxlabels, sequences):
        mapped[label] = seq
    return mapped

def export_single_gene(args):
    gene_name, db_path, export_dir = args
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get gene details
    c.execute("select num_seqs, num_sites, global_dnds, num_branch_outliers, outlier_branches, outlier_threshold, branch_lengths from gene_results where gene_name=?", (gene_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    num_seqs, num_sites, global_dnds, num_branch_outliers, outlier_branches, outlier_threshold, branch_lengths = row
    
    # Get significant sites
    c.execute("""
        select site_index, p_value, classification, filter_reason, selection_type
        from site_results
        where gene_name=? and is_significant=1
        order by site_index
    """, (gene_name,))
    sig_sites = []
    for row in c.fetchall():
        site_idx, pval, classification, reason, selection_type = row
        
        c.execute("""
            select branch_name, ancestral_codon, derived_codon, ancestral_aa, derived_aa
            from site_substitutions
            where gene_name=? and site_index=? 
            and (
                is_synonymous=0
                or (
                    (ancestral_codon in ('TCA','TCC','TCG','TCT') and derived_codon in ('AGC','AGT'))
                    or 
                    (ancestral_codon in ('AGC','AGT') and derived_codon in ('TCA','TCC','TCG','TCT'))
                )
            )
        """, (gene_name, site_idx))
        substs = []
        for sub in c.fetchall():
            substs.append(f"{sub[0]}:{sub[3]}->{sub[4]}({sub[1]}->{sub[2]})")
            
        sig_sites.append({
            "site_index": site_idx,
            "p_value": pval,
            "classification": classification,
            "filter_reason": reason,
            "selection_type": selection_type,
            "substitutions": substs
        })
        
    # Get all sites for plot
    c.execute("""
        select site_index, p_value, classification
        from site_results
        where gene_name=?
        order by site_index
    """, (gene_name,))
    all_sites = []
    for row in c.fetchall():
        all_sites.append({
            "site_index": row[0],
            "p_value": row[1],
            "classification": row[2]
        })
        
    payload = {
        "gene_name": gene_name,
        "num_seqs": num_seqs,
        "num_sites": num_sites,
        "global_dnds": global_dnds,
        "num_branch_outliers": num_branch_outliers,
        "outlier_branches": outlier_branches,
        "outlier_threshold": outlier_threshold,
        "branch_lengths": branch_lengths,
        "significant_sites": sig_sites,
        "all_sites": all_sites
    }
    
    with open(os.path.join(export_dir, "api", "gene", f"{gene_name}.json"), "w") as f:
        json.dump(payload, f)
        
    # Check alignment folder
    align_path = f"msa/{gene_name}.gz"
    if os.path.exists(align_path):
        try:
            seqs = parse_nexus_gz(align_path)
            if seqs:
                first_seq = next(iter(seqs.values()))
                total_codons = len(first_seq) // 3
                
                # Expose local window for each significant site
                os.makedirs(os.path.join(export_dir, "api", "alignment", gene_name), exist_ok=True)
                
                for s in sig_sites:
                    site_idx = s["site_index"]
                    win_start_codon = max(1, site_idx - 5)
                    win_end_codon = min(total_codons, site_idx + 5)
                    
                    ref_seq = seqs.get('hg', None)
                    if not ref_seq:
                        codon_counts = {}
                        nuc_start = (site_idx - 1) * 3
                        for seq in seqs.values():
                            if len(seq) > nuc_start + 2:
                                c_codon = seq[nuc_start:nuc_start+3].upper()
                                if '-' not in c_codon and 'N' not in c_codon:
                                    codon_counts[c_codon] = codon_counts.get(c_codon, 0) + 1
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
                                "sequence": seq[n_start:n_start+3].upper() if len(seq) > n_start + 2 else "---"
                            })
                        alignment_window.append({
                            "taxon": spec,
                            "codons": row_codons
                        })
                        
                    with open(os.path.join(export_dir, "api", "alignment", gene_name, f"{site_idx}.json"), "w") as f:
                        json.dump({"alignment": alignment_window}, f)
        except Exception as e:
            print(f"Error rendering MSA for gene {gene_name}: {e}")
            
    conn.close()
    return True

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        sys.exit(1)
        
    # 1. Recreate clean toga-meme directory structure
    print(f"Creating export directories under: {EXPORT_DIR}")
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
        
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api", "gene"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api", "alignment"), exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "api", "species"), exist_ok=True)
    
    # Connect to local database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 2. Export api/summary.json
    print("Exporting summary stats...")
    c.execute("select count(*) from gene_results")
    total_genes = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where is_significant=1")
    total_sig_sites = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where classification='GOLD'")
    gold_count = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where classification='SILVER'")
    silver_count = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where classification='LIKELY_ERROR'")
    likely_error_count = c.fetchone()[0]
    
    summary = {
        'total_genes': total_genes,
        'total_sig_sites': total_sig_sites,
        'gold_count': gold_count,
        'silver_count': silver_count,
        'likely_error_count': likely_error_count
    }
    with open(os.path.join(EXPORT_DIR, "api", "summary.json"), "w") as f:
        json.dump(summary, f)
        
    # 3. Export api/genes.json
    print("Exporting genes list...")
    c.execute("""
        select g.gene_name, g.num_seqs, g.num_sites, g.log_likelihood, 
               (select count(*) from site_results s where s.gene_name=g.gene_name and s.is_significant=1) as sig_sites,
               (select count(*) from site_results s where s.gene_name=g.gene_name and s.classification='GOLD') as gold_sites,
               (select count(*) from site_results s where s.gene_name=g.gene_name and s.classification='SILVER') as silver_sites,
               (select count(*) from site_results s where s.gene_name=g.gene_name and s.classification='LIKELY_ERROR') as likely_error_sites
        from gene_results g
        order by g.gene_name asc
    """)
    genes = []
    for row in c.fetchall():
        genes.append({
            "gene_name": row[0],
            "num_seqs": row[1],
            "num_sites": row[2],
            "log_likelihood": row[3],
            "num_sig_sites": row[4],
            "gold_sites": row[5],
            "silver_sites": row[6],
            "likely_error_sites": row[7]
        })
    with open(os.path.join(EXPORT_DIR, "api", "genes.json"), "w") as f:
        json.dump(genes, f)
        
    # 4. Export api/gene/<gene_name>.json and alignments in parallel
    print(f"Exporting details and alignments for {len(genes)} genes in parallel...")
    import multiprocessing as mp
    num_workers = max(1, mp.cpu_count() - 1)
    pool = mp.Pool(num_workers)
    
    tasks = [(g["gene_name"], DB_PATH, EXPORT_DIR) for g in genes]
    completed = 0
    for res in pool.imap_unordered(export_single_gene, tasks):
        completed += 1
        if completed % 100 == 0 or completed == len(genes):
            print(f"  Processed {completed} / {len(genes)} genes...")
    pool.close()
    pool.join()

    # 4.5. Export species list and details
    print("Exporting species list...")
    c.execute("SELECT node_id, is_leaf, node_label, descendants FROM master_nodes")
    nodes = []
    for row in c.fetchall():
        node_id, is_leaf, node_label, descendants = row
        num_leaves = len(descendants.split(',')) if descendants else 0
        if is_leaf or (num_leaves <= 10):
            nodes.append({
                "node_id": node_id,
                "is_leaf": is_leaf,
                "label": node_label,
                "num_leaves": num_leaves,
                "descendants_raw": descendants
            })
    nodes.sort(key=lambda x: (not x['is_leaf'], x['num_leaves'], x['label']))
    
    species_list_payload = []
    for n in nodes:
        species_list_payload.append({
            "node_id": n["node_id"],
            "is_leaf": n["is_leaf"],
            "label": n["label"],
            "num_leaves": n["num_leaves"]
        })
        
    with open(os.path.join(EXPORT_DIR, "api", "species.json"), "w") as f:
        json.dump(species_list_payload, f)
        
    print(f"Exporting details for {len(nodes)} species/nodes...")
    for idx, n in enumerate(nodes):
        node_id = n["node_id"]
        is_leaf = n["is_leaf"]
        node_label = n["label"]
        descendants = n["descendants_raw"]
        
        c.execute("""
            SELECT s.gene_name, s.site_index, s.ancestral_aa, s.derived_aa, s.ancestral_codon, s.derived_codon, r.classification, r.p_value, r.selection_type
            FROM branch_mappings bm
            JOIN site_substitutions s ON s.gene_name = bm.gene_name AND s.branch_name = bm.branch_name
            JOIN site_results r ON r.gene_name = s.gene_name AND r.site_index = s.site_index
            WHERE bm.master_node_id = ?
              AND r.is_significant = 1
              AND r.classification IN ('GOLD', 'SILVER')
              AND s.is_synonymous = 0
            ORDER BY s.gene_name ASC, s.site_index ASC
        """, (node_id,))
        
        signals = []
        gene_sites = {}
        gene_signal_counts = {}
        
        for row in c.fetchall():
            gene, site, anc_aa, der_aa, anc_codon, der_codon, classification, p_val, selection_type = row
            signals.append({
                "gene_name": gene,
                "site_index": site,
                "ancestral_aa": anc_aa,
                "derived_aa": der_aa,
                "ancestral_codon": anc_codon,
                "derived_codon": der_codon,
                "classification": classification,
                "p_value": p_val,
                "selection_type": selection_type
            })
            
            if gene not in gene_sites:
                gene_sites[gene] = set()
                gene_signal_counts[gene] = 0
            gene_sites[gene].add(site)
            gene_signal_counts[gene] += 1
            
        top_genes = []
        for gene, sites in gene_sites.items():
            top_genes.append({
                "gene_name": gene,
                "site_count": len(sites),
                "signal_count": gene_signal_counts[gene]
            })
        top_genes.sort(key=lambda x: x['site_count'], reverse=True)
        
        total_genes = len(gene_sites)
        total_sites = sum(len(s) for s in gene_sites.values())
        total_signals = len(signals)
        
        species_detail = {
            "node_id": node_id,
            "label": node_label,
            "is_leaf": is_leaf,
            "descendants": descendants,
            "stats": {
                "total_genes": total_genes,
                "total_sites": total_sites,
                "total_signals": total_signals
            },
            "top_genes": top_genes,
            "signals": signals
        }
        
        with open(os.path.join(EXPORT_DIR, "api", "species", f"{node_id}.json"), "w") as f:
            json.dump(species_detail, f)
            
        if (idx + 1) % 200 == 0 or (idx + 1) == len(nodes):
            print(f"  Processed {idx + 1} / {len(nodes)} species/nodes...")
            
    conn.close()
    
    # 5. Modify and copy docs/meme_dashboard.html to toga-meme/index.html
    print("Modifying HTML fetches for static relative JSON requests...")
    with open(HTML_PATH, "r") as f:
        html = f.read()
        
    # Replace absolute API URLs with local relative JSON URLs
    html = html.replace("fetch('/api/summary')", "fetch('api/summary.json')")
    html = html.replace("fetch('/api/genes')", "fetch('api/genes.json')")
    html = html.replace("fetch(`/api/gene/${geneName}`)", "fetch(`api/gene/${geneName}.json`)")
    html = html.replace("fetch(`/api/alignment/${geneName}/${siteIdx}`)", "fetch(`api/alignment/${geneName}/${siteIdx}.json`)")
    html = html.replace("fetch('/api/species')", "fetch('api/species.json')")
    html = html.replace("fetch(`/api/species/${nodeId}`)", "fetch(`api/species/${nodeId}.json`)")
    
    # Write to target toga-meme/index.html
    with open(os.path.join(EXPORT_DIR, "index.html"), "w") as f:
        f.write(html)
        
    print(f"Static site export to '{EXPORT_DIR}' completed successfully!")

if __name__ == "__main__":
    main()
