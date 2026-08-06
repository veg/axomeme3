#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import urllib.parse
import gzip
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

DB_PATH = "meme_results.db"
HTML_PATH = "docs/meme_dashboard.html"

app = FastAPI(title="TOGA MEME Dashboard API (FastAPI)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches
summary_cache = {}
species_list_cache = None
species_detail_cache = {}
genes_cache = None

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
    
    c.execute("select count(*) from site_results where is_significant=1")
    total_sig_sites = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where classification='GOLD'")
    gold_count = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where classification='SILVER'")
    silver_count = c.fetchone()[0]
    
    c.execute("select count(*) from site_results where classification='LIKELY_ERROR'")
    likely_error_count = c.fetchone()[0]
    
    summary_cache['total_genes'] = total_genes
    summary_cache['total_sig_sites'] = total_sig_sites
    summary_cache['gold_count'] = gold_count
    summary_cache['silver_count'] = silver_count
    summary_cache['likely_error_count'] = likely_error_count
    
    conn.close()
    print("Summary statistics cached successfully.")

def warm_up_cache():
    print("Starting background cache warm-up...")
    
    # 1. Warm up genes list cache
    try:
        get_genes()
        print("Genes cache warmed up.")
    except Exception as e:
        print(f"Error warming up genes cache: {e}")
        
    # 2. Warm up species list cache
    try:
        species_list = get_species()
        print("Species list cache warmed up.")
        
        # 3. Warm up species detail cache for listed nodes
        print(f"Warming up species details for {len(species_list)} nodes in background...")
        for node in species_list:
            try:
                get_species_detail(node['node_id'])
            except Exception as e:
                print(f"Error warming up detail for species node {node['node_id']}: {e}")
        print("All species details caches warmed up successfully.")
    except Exception as e:
        print(f"Error warming up species list cache: {e}")

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

def optimize_database():
    print("Verifying database indexes for performance...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        return
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_site_gene_idx ON site_results(gene_name, site_index)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_subst_gene_branch ON site_substitutions(gene_name, branch_name)")
    conn.commit()
    conn.close()
    print("Database index optimization complete.")

@app.middleware("http")
async def strip_json_extension(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path.endswith(".json"):
        # Rewrite scope path to strip .json extension for seamless routing
        request.scope["path"] = path[:-5]
    response = await call_next(request)
    return response

@app.on_event("startup")
def startup_event():
    optimize_database()
    cache_summary_stats()
    threading.Thread(target=warm_up_cache, daemon=True).start()

@app.get("/")
@app.get("/index.html")
def serve_index():
    if not os.path.exists(HTML_PATH):
        raise HTTPException(status_code=404, detail="Dashboard HTML file not found!")
    return FileResponse(HTML_PATH)

@app.get("/api/summary")
def get_summary():
    return summary_cache

@app.get("/api/species")
def get_species():
    global species_list_cache
    if species_list_cache is not None:
        return species_list_cache
        
    conn = get_db_connection()
    c = conn.cursor()
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
                "num_leaves": num_leaves
            })
    conn.close()
    nodes.sort(key=lambda x: (not x['is_leaf'], x['num_leaves'], x['label']))
    species_list_cache = nodes
    return nodes

@app.get("/api/species/{node_id:path}")
def get_species_detail(node_id: str):
    node_id = urllib.parse.unquote(node_id)
    global species_detail_cache
    if node_id in species_detail_cache:
        return species_detail_cache[node_id]
        
    conn = get_db_connection()
    c = conn.cursor()
    
    # Fetch node general info
    c.execute("SELECT node_id, is_leaf, node_label, descendants FROM master_nodes WHERE node_id = ?", (node_id,))
    node_info = c.fetchone()
    if not node_info:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Species/node {node_id} not found")
        
    node_id, is_leaf, node_label, descendants = node_info
    
    # Fetch signals using optimized JOIN order
    c.execute("""
        SELECT s.gene_name, s.site_index, s.ancestral_aa, s.derived_aa, s.ancestral_codon, s.derived_codon, r.classification, r.p_value, r.q_value, r.selection_type
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
        gene, site, anc_aa, der_aa, anc_codon, der_codon, classification, p_val, q_val, selection_type = row
        signals.append({
            "gene_name": gene,
            "site_index": site,
            "ancestral_aa": anc_aa,
            "derived_aa": der_aa,
            "ancestral_codon": anc_codon,
            "derived_codon": der_codon,
            "classification": classification,
            "p_value": p_val,
            "q_value": q_val,
            "selection_type": selection_type
        })
        
        if gene not in gene_sites:
            gene_sites[gene] = set()
            gene_signal_counts[gene] = 0
        gene_sites[gene].add(site)
        gene_signal_counts[gene] += 1
        
    conn.close()
    
    # Build top_genes
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
    
    payload = {
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
    species_detail_cache[node_id] = payload
    return payload

@app.get("/api/genes")
def get_genes():
    global genes_cache
    if genes_cache is not None:
        return genes_cache
        
    conn = get_db_connection()
    c = conn.cursor()
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
    conn.close()
    genes_cache = genes
    return genes

@app.get("/api/gene/{gene_name:path}")
def get_gene_detail(gene_name: str):
    gene_name = urllib.parse.unquote(gene_name)
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get gene general info
    c.execute("select num_seqs, num_sites, global_dnds, num_branch_outliers, outlier_branches, outlier_threshold, branch_lengths from gene_results where gene_name=?", (gene_name,))
    gene_info = c.fetchone()
    if not gene_info:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Gene {gene_name} not found!")
        
    num_seqs, num_sites, global_dnds, num_branch_outliers, outlier_branches, outlier_threshold, branch_lengths = gene_info
    
    # Load MEME JSON to extract branch-level posterior probabilities and EBFs
    results_dir = "/Users/sergei/Dropbox/TOGA2026/scratch/meme_results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{gene_name}.MEME.json.gz")
    
    # On-demand download from m3.local if not found locally
    if not os.path.exists(json_path):
        import re
        import subprocess
        if re.match(r"^[a-zA-Z0-9_\-]+$", gene_name):
            try:
                remote_src = f"m3.local:/Users/sergei/Projects/TOGA_MEME/meme_results/{gene_name}.MEME.json.gz"
                print(f"Downloading {gene_name} MEME JSON from m3.local on-demand...")
                subprocess.run(["scp", remote_src, json_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Failed to fetch remote MEME JSON for {gene_name}: {e}")
                
    meme_data = None
    if os.path.exists(json_path):
        try:
            try:
                with gzip.open(json_path, 'rt') as f:
                    meme_data = json.load(f)
            except Exception:
                with open(json_path, 'r') as f:
                    meme_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON for EBF calculation: {e}")

    # Get significant sites
    c.execute("""
        select site_index, p_value, q_value, classification, filter_reason, selection_type
        from site_results
        where gene_name=? and is_significant=1
        order by site_index
    """, (gene_name,))
    sig_sites = []
    for row in c.fetchall():
        site_idx, pval, qval, classification, reason, selection_type = row
        
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
            
        branch_ebfs = []
        if meme_data:
            try:
                mle_content = meme_data['MLE']['content']['0']
                s_idx = site_idx - 1
                if 0 <= s_idx < len(mle_content):
                    alpha = mle_content[s_idx][0]
                    beta_pos = mle_content[s_idx][3]
                    p_pos = mle_content[s_idx][4]
                    
                    if beta_pos > alpha and p_pos > 0 and p_pos < 1:
                        prior_odds = p_pos / (1 - p_pos)
                        
                        branch_attr = meme_data.get('branch attributes', {}).get('0', {})
                        for branch, attr in branch_attr.items():
                            post_probs = attr.get('Posterior prob omega class by site')
                            if post_probs and len(post_probs) == 2:
                                p_post = post_probs[1][s_idx]
                                if p_post > 0:
                                    if p_post < 1:
                                        post_odds = p_post / (1 - p_post)
                                        ebf = post_odds / prior_odds
                                    else:
                                        ebf = 10000.0
                                        
                                    if ebf >= 20:
                                        branch_name = attr.get('original name', branch)
                                        branch_ebfs.append({
                                            "branch": branch_name,
                                            "ebf": round(ebf, 1),
                                            "post_prob": round(p_post, 3)
                                        })
                        branch_ebfs = sorted(branch_ebfs, key=lambda x: x['ebf'], reverse=True)
            except Exception as e:
                print(f"Error computing EBF for {gene_name} site {site_idx}: {e}")

        sig_sites.append({
            "site_index": site_idx,
            "p_value": pval,
            "q_value": qval,
            "classification": classification,
            "filter_reason": reason,
            "selection_type": selection_type,
            "substitutions": substs,
            "branch_ebfs": branch_ebfs
        })
        
    # Get all sites for the plot
    c.execute("""
        select site_index, p_value, q_value, classification
        from site_results
        where gene_name=?
        order by site_index
    """, (gene_name,))
    all_sites = []
    for row in c.fetchall():
        all_sites.append({
            "site_index": row[0],
            "p_value": row[1],
            "q_value": row[2],
            "classification": row[3]
        })
        
    newick_tree = None
    if meme_data:
        try:
            newick_tree = meme_data.get('input', {}).get('trees', {}).get('0', None)
        except Exception:
            pass

    conn.close()
    
    return {
        "gene_name": gene_name,
        "num_seqs": num_seqs,
        "num_sites": num_sites,
        "global_dnds": global_dnds,
        "num_branch_outliers": num_branch_outliers,
        "outlier_branches": outlier_branches,
        "outlier_threshold": outlier_threshold,
        "branch_lengths": branch_lengths,
        "significant_sites": sig_sites,
        "all_sites": all_sites,
        "newick_tree": newick_tree
    }

@app.get("/api/alignment/{gene_name}/{site_index}")
def get_alignment(gene_name: str, site_index: int):
    gene_name = urllib.parse.unquote(gene_name)
    align_path = f"msa/{gene_name}.gz"
    if not os.path.exists(align_path):
        raise HTTPException(status_code=404, detail=f"Alignment for {gene_name} not found")
        
    try:
        seqs = parse_nexus_gz(align_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing alignment file: {e}")
        
    # Expose a ±5 codon window around site_index (1-based)
    win_start_codon = max(1, site_index - 5)
    # Find sequence length in codons
    first_seq = next(iter(seqs.values()))
    total_codons = len(first_seq) // 3
    win_end_codon = min(total_codons, site_index + 5)
    
    alignment_window = []
    # Sort species so that sequences with changes at site_index are at the top (under hg)
    # Find the reference codon (from hg, or consensus if hg is not present)
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
        
    return {"alignment": alignment_window}

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"\n🚀 Starting FastAPI TOGA MEME Dashboard server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
