#!/usr/bin/env python3
"""
prepare_hyphy_package.py
-------------------------
Extracts, filters, and deduplicates sequence alignments for HyPhy analysis.
Applies:
1. Excludes outlier genes and pseudogenes.
2. Strips low-coverage sequences (resolved length < 50% of alignment length).
3. Strips sequences containing premature stop codons.
4. Selects the representative genome sequence for each species, unless an alternative genome sequence
   has a coverage advantage of >= 10% (0.10).
5. Excludes alignments that end up with < 3 sequences.
6. Strips all-gap columns from the remaining alignments.
7. Saves final files to hyphy/[gene_name].fa with tree leaf headers.
8. Writes statistics and an interactive HTML dashboard.
"""

import os
import re
import json
import sqlite3
import math
import multiprocessing as mp
from datetime import datetime

# Define stop codons
STOP_CODONS = {'TAA', 'TAG', 'TGA'}

def clean_numeric(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0

def check_premature_stop(seq):
    seq_upper = seq.upper()
    codons = [seq_upper[i:i+3] for i in range(0, len(seq_upper), 3)]
    stops = []
    for idx, codon in enumerate(codons):
        if '-' not in codon and 'N' not in codon and codon in STOP_CODONS:
            stops.append(idx)
    # Find last codon containing at least one non-gap base
    last_codon_idx = -1
    for idx in range(len(codons) - 1, -1, -1):
        if any(c != '-' for c in codons[idx]):
            last_codon_idx = idx
            break
    # Return True if any stop codon is before the last codon
    return any(idx < last_codon_idx for idx in stops)

def parse_fasta(file_path):
    seqs = []
    try:
        with open(file_path, 'r') as f:
            current_name = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_name:
                        seqs.append((current_name, "".join(current_seq)))
                    current_name = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_name:
                seqs.append((current_name, "".join(current_seq)))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return seqs

def degap_alignment(sequences):
    """
    Given a list of sequence strings of equal length, removes columns
    that consist entirely of gaps ('-') in all sequences.
    Returns the cleaned list of sequence strings.
    """
    if not sequences:
        return sequences
    num_cols = len(sequences[0])
    valid_cols = []
    for col_idx in range(num_cols):
        if any(seq[col_idx] != '-' for seq in sequences):
            valid_cols.append(col_idx)
    
    # Reconstruct sequences
    return ["".join(seq[col_idx] for col_idx in valid_cols) for seq in sequences]

def process_single_alignment(args):
    """
    Worker function to process a single gene alignment.
    """
    (file_path, file_name, resolved_gene_name, transcript_id, 
     species_to_rep_assembly, species_to_leaf, assembly_to_species, out_dir) = args
    
    # 1. Parse raw sequences from FASTA
    raw_seqs = parse_fasta(file_path)
    if not raw_seqs:
        return {
            'gene': resolved_gene_name,
            'transcript': transcript_id,
            'status': 'FAILED_TO_PARSE',
            'orig_seq_count': 0,
            'retained_seq_count': 0,
            'orig_len': 0,
            'cleaned_len': 0,
            'dropped_reasons': {}
        }
        
    orig_len = len(raw_seqs[0][1])
    orig_seq_count = len(raw_seqs)
    
    # Group sequences by species
    species_groups = {}
    dropped_reasons = {
        'low_coverage': 0,
        'premature_stop': 0,
        'unmapped_species': 0,
        'not_in_tree': 0,
        'deduplicated': 0
    }
    
    for seq_name, seq_content in raw_seqs:
        # Map sequence name to species
        species = assembly_to_species.get(seq_name)
        if not species:
            dropped_reasons['unmapped_species'] += 1
            continue
            
        # Check if species is in tree
        if species not in species_to_leaf:
            dropped_reasons['not_in_tree'] += 1
            continue
            
        # Calculate coverage
        seq_upper = seq_content.upper()
        a_cnt = seq_upper.count('A')
        c_cnt = seq_upper.count('C')
        g_cnt = seq_upper.count('G')
        t_cnt = seq_upper.count('T')
        resolved_len = a_cnt + c_cnt + g_cnt + t_cnt
        coverage = resolved_len / orig_len if orig_len > 0 else 0.0
        
        # Filter 1: Low coverage (>= 50% required)
        if coverage < 0.50:
            dropped_reasons['low_coverage'] += 1
            continue
            
        # Filter 2: Premature stops
        if check_premature_stop(seq_content):
            dropped_reasons['premature_stop'] += 1
            continue
            
        species_groups.setdefault(species, []).append({
            'name': seq_name,
            'seq': seq_content,
            'coverage': coverage
        })
        
    # Deduplicate species with multiple sequences
    selected_sequences = []
    
    for species, seq_list in species_groups.items():
        if len(seq_list) == 1:
            selected_sequences.append((species, seq_list[0]))
        else:
            # We have multiple sequences for this species
            rep_assembly = species_to_rep_assembly.get(species)
            
            # Find representative sequence
            s_best = None
            for s in seq_list:
                if s['name'] == rep_assembly:
                    s_best = s
                    break
                    
            # Find alternative sequence with highest coverage
            s_alts = [s for s in seq_list if s['name'] != rep_assembly]
            s_alt = max(s_alts, key=lambda x: x['coverage']) if s_alts else None
            
            selected = None
            if s_best:
                if s_alt and s_alt['coverage'] >= s_best['coverage'] + 0.10:
                    selected = s_alt
                else:
                    selected = s_best
            else:
                selected = s_alt
                
            selected_sequences.append((species, selected))
            dropped_reasons['deduplicated'] += (len(seq_list) - 1)
            
    retained_seq_count = len(selected_sequences)
    
    # 5. Filter 3: Minimum sequence count (>= 3 required for HyPhy)
    if retained_seq_count < 3:
        return {
            'gene': resolved_gene_name,
            'transcript': transcript_id,
            'status': 'DROPPED_FEW_SEQS',
            'orig_seq_count': orig_seq_count,
            'retained_seq_count': retained_seq_count,
            'orig_len': orig_len,
            'cleaned_len': 0,
            'dropped_reasons': dropped_reasons
        }
        
    # degap alignment
    headers = [species_to_leaf[species] for species, s_data in selected_sequences]
    seq_strings = [s_data['seq'] for species, s_data in selected_sequences]
    cleaned_seq_strings = degap_alignment(seq_strings)
    cleaned_len = len(cleaned_seq_strings[0])
    
    # Write to final FASTA file
    out_path = os.path.join(out_dir, f"{resolved_gene_name}.fa")
    try:
        with open(out_path, 'w') as out_f:
            for header, seq_str in zip(headers, cleaned_seq_strings):
                out_f.write(f">{header}\n{seq_str}\n")
    except Exception as e:
        return {
            'gene': resolved_gene_name,
            'transcript': transcript_id,
            'status': 'WRITE_ERROR',
            'orig_seq_count': orig_seq_count,
            'retained_seq_count': retained_seq_count,
            'orig_len': orig_len,
            'cleaned_len': 0,
            'dropped_reasons': dropped_reasons
        }
        
    return {
        'gene': resolved_gene_name,
        'transcript': transcript_id,
        'status': 'RETAINED',
        'orig_seq_count': orig_seq_count,
        'retained_seq_count': retained_seq_count,
        'orig_len': orig_len,
        'cleaned_len': cleaned_len,
        'dropped_reasons': dropped_reasons,
        'species_list': [species for species, _ in selected_sequences]
    }

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    db_path = os.path.join(base_dir, "alignments_stats.db")
    taxon_map_path = os.path.join(base_dir, "species_to_taxon_map.tsv")
    assembly_to_species_path = os.path.join(base_dir, "assembly_to_species.tsv")
    out_dir = os.path.join(base_dir, "hyphy")
    
    os.makedirs(out_dir, exist_ok=True)
    
    start_time = datetime.now()
    
    # 1. Load Species taxonomic mapping and representative assemblies
    print("Loading species taxons and representative assemblies...")
    species_to_rep_assembly = {}
    species_to_leaf = {}
    
    with open(taxon_map_path, 'r') as f:
        f.readline() # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                sp = parts[0]
                rep_assembly = parts[2]
                leaf = parts[4]
                
                species_to_rep_assembly[sp] = rep_assembly
                species_to_leaf[sp] = leaf
                
    print(f"Loaded {len(species_to_leaf)} target species in the tree.")
    
    # 2. Load Assembly to Species mapping
    print("Loading assembly to species mapping...")
    assembly_to_species = {}
    with open(assembly_to_species_path, 'r') as f:
        f.readline() # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                acc = parts[0]
                dir_name = parts[1]
                species = parts[2]
                
                subparts = dir_name.split('__')
                if len(subparts) >= 3:
                    short_id = subparts[2]
                    assembly_to_species[short_id] = species
                    
    print(f"Loaded mapping for {len(assembly_to_species)} assembly IDs.")
    
    # 3. Query DB for active alignments (excluding outliers & pseudogenes)
    print("Connecting to alignments SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, file_name, resolved_gene_name, transcript_id 
        FROM alignments 
        WHERE is_outlier = 0 AND is_pseudogene = 0
    """)
    db_alignments = cursor.fetchall()
    conn.close()
    
    total_db_alignments = len(db_alignments)
    print(f"Found {total_db_alignments} curated alignments in database (non-outlier, non-pseudogene).")
    
    # 4. Prepare args for parallel workers
    tasks = []
    for aln_id, file_name, resolved_gene_name, transcript_id in db_alignments:
        file_path = os.path.join(base_dir, "individualAlis", file_name)
        tasks.append((
            file_path, file_name, resolved_gene_name, transcript_id,
            species_to_rep_assembly, species_to_leaf, assembly_to_species, out_dir
        ))
        
    # 5. Run parallel processing
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting parallel curation with {num_workers} workers...")
    pool = mp.Pool(num_workers)
    
    results = []
    processed_count = 0
    for res in pool.imap_unordered(process_single_alignment, tasks):
        results.append(res)
        processed_count += 1
        if processed_count % 1000 == 0 or processed_count == total_db_alignments:
            pct = (processed_count / total_db_alignments) * 100
            print(f"Curation progress: {processed_count}/{total_db_alignments} ({pct:.1f}%)...")
            
    pool.close()
    pool.join()
    
    # 6. Analyze Curation Results
    print("Analyzing and compiling curation metrics...")
    
    total_processed = len(results)
    retained_count = 0
    dropped_few_seqs = 0
    failed_count = 0
    
    retained_genes = []
    dropped_genes = []
    
    global_dropped_reasons = {
        'low_coverage': 0,
        'premature_stop': 0,
        'unmapped_species': 0,
        'not_in_tree': 0,
        'deduplicated': 0
    }
    
    # Track sequence counts and length distributions for retained alignments
    retained_seq_counts = []
    retained_codon_lengths = []
    
    # Track species representation in final dataset
    species_representation = {sp: 0 for sp in species_to_leaf.keys()}
    
    for r in results:
        status = r['status']
        if status == 'RETAINED':
            retained_count += 1
            retained_genes.append({
                'gene': r['gene'],
                'transcript': r['transcript'],
                'retained_seqs': r['retained_seq_count'],
                'orig_seqs': r['orig_seq_count'],
                'codons': int(r['cleaned_len'] / 3),
                'orig_codons': int(r['orig_len'] / 3)
            })
            retained_seq_counts.append(r['retained_seq_count'])
            retained_codon_lengths.append(int(r['cleaned_len'] / 3))
            
            for sp in r['species_list']:
                if sp in species_representation:
                    species_representation[sp] += 1
                    
        elif status == 'DROPPED_FEW_SEQS':
            dropped_few_seqs += 1
            dropped_genes.append({
                'gene': r['gene'],
                'transcript': r['transcript'],
                'reason': f"Too few sequences remaining ({r['retained_seq_count']}/3)",
                'retained_seqs': r['retained_seq_count'],
                'orig_seqs': r['orig_seq_count']
            })
        else:
            failed_count += 1
            dropped_genes.append({
                'gene': r['gene'],
                'transcript': r['transcript'],
                'reason': 'File processing or write error',
                'retained_seqs': r['retained_seq_count'],
                'orig_seqs': r['orig_seq_count']
            })
            
        # Accumulate dropped reasons
        if 'dropped_reasons' in r:
            for k, val in r['dropped_reasons'].items():
                global_dropped_reasons[k] += val
                
    # Sort genes by name for summary table
    retained_genes.sort(key=lambda x: x['gene'])
    dropped_genes.sort(key=lambda x: x['gene'])
    
    # 7. Pre-compute histograms for dashboard
    def build_histogram_bins(data, bin_count=15):
        if not data:
            return {'counts': [], 'labels': []}
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val += 1
        width = (max_val - min_val) / bin_count
        counts = [0] * bin_count
        labels = []
        for i in range(bin_count):
            start = min_val + i * width
            end = min_val + (i + 1) * width
            labels.append(f"{math.floor(start)}-{math.floor(end)}")
            
        for x in data:
            idx = int((x - min_val) / width)
            if idx >= bin_count:
                idx = bin_count - 1
            counts[idx] += 1
        return {'counts': counts, 'labels': labels}
        
    seqs_hist = build_histogram_bins(retained_seq_counts, bin_count=15)
    len_hist = build_histogram_bins(retained_codon_lengths, bin_count=15)
    
    # Species table details
    species_summary_list = []
    for sp, count in species_representation.items():
        species_summary_list.append({
            'species': sp,
            'leaf': species_to_leaf[sp],
            'rep_assembly': species_to_rep_assembly[sp],
            'gene_count': count,
            'pct_retained_genes': (count / max(1, retained_count)) * 100
        })
    species_summary_list.sort(key=lambda x: x['gene_count'], reverse=True)
    
    # Build complete stats JSON
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'curation_parameters': {
            'coverage_threshold': '>= 50%',
            'exclude_pseudogenes': True,
            'exclude_outliers': True,
            'min_retained_sequences': 3,
            'deduplication_margin': '>= 10% advantage'
        },
        'metrics': {
            'total_curated_db_alignments': total_db_alignments,
            'total_retained_genes': retained_count,
            'total_dropped_few_seqs': dropped_few_seqs,
            'total_failed_curation': failed_count,
            'pct_genes_retained': (retained_count / total_db_alignments) * 100,
            'average_retained_sequences': sum(retained_seq_counts)/len(retained_seq_counts) if retained_seq_counts else 0,
            'average_retained_codon_length': sum(retained_codon_lengths)/len(retained_codon_lengths) if retained_codon_lengths else 0
        },
        'global_sequence_filtering': {
            'dropped_low_coverage': global_dropped_reasons['low_coverage'],
            'dropped_premature_stop': global_dropped_reasons['premature_stop'],
            'dropped_unmapped_species': global_dropped_reasons['unmapped_species'],
            'dropped_not_in_tree': global_dropped_reasons['not_in_tree'],
            'dropped_deduplicated': global_dropped_reasons['deduplicated']
        },
        'histograms': {
            'sequences_per_gene': seqs_hist,
            'codons_per_gene': len_hist
        },
        'retained_genes': retained_genes,
        'dropped_genes': dropped_genes,
        'species_representation': species_summary_list
    }
    
    # Write summary.json
    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Curation summary statistics saved to {summary_json_path}")
    
    # 8. Generate HTML Dashboard
    html_dashboard_path = os.path.join(out_dir, "dashboard.html")
    generate_dashboard_html(summary_data, html_dashboard_path)
    print(f"Interactive curation dashboard generated at {html_dashboard_path}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Curation pipeline completed successfully in {duration/60:.2f} minutes!")

def generate_dashboard_html(summary_data, dest_path):
    """
    Writes a self-contained, responsive, beautifully styled HTML/CSS/JS dashboard.
    Embeds the summary data directly to avoid CORS issues.
    """
    json_data_str = json.dumps(summary_data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HyPhy Curated Dataset Portal</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-main: #060814;
            --bg-card: rgba(15, 23, 42, 0.7);
            --bg-card-hover: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.06);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.12);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.12);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.12);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.12);
            --accent: #8b5cf6;
            
            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.5;
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.08) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.06) 0px, transparent 40%);
            background-attachment: fixed;
            padding-bottom: 5rem;
        }}

        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.3);
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        .container {{
            max-width: 1500px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        header h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }}

        .badge {{
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            color: var(--primary);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            transition: var(--transition);
        }}

        .card:hover {{
            border-color: rgba(255, 255, 255, 0.12);
            transform: translateY(-2px);
        }}

        .metric-title {{
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}

        .metric-sub {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.3rem;
        }}

        .text-primary {{ color: var(--primary); }}
        .text-success {{ color: var(--success); }}
        .text-warning {{ color: var(--warning); }}
        .text-danger {{ color: var(--danger); }}

        /* Charts Section */
        .charts-row {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        @media (max-width: 1024px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            min-height: 380px;
        }}

        .chart-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            border-left: 3px solid var(--primary);
            padding-left: 0.75rem;
        }}

        .filter-stats-list {{
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            justify-content: center;
            height: calc(100% - 3rem);
        }}

        .filter-stat-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.04);
            padding-bottom: 0.6rem;
        }}

        .filter-stat-label {{
            font-size: 0.9rem;
            color: var(--text-muted);
        }}

        .filter-stat-value {{
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 1rem;
        }}

        /* Tabs and Tables */
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }}

        .section-title {{
            font-size: 1.2rem;
            font-weight: 600;
        }}

        .tabs {{
            display: flex;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.2);
            padding: 0.3rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            transition: var(--transition);
        }}

        .tab-btn:hover {{
            color: var(--text-main);
        }}

        .tab-btn.active {{
            background: var(--primary);
            color: var(--text-main);
        }}

        .search-container {{
            margin-bottom: 1.5rem;
            display: flex;
            gap: 1rem;
        }}

        .search-input {{
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-family: var(--font-sans);
            font-size: 0.9rem;
            outline: none;
            transition: var(--transition);
        }}

        .search-input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }}

        .table-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}

        th {{
            background: rgba(0, 0, 0, 0.20);
            padding: 1rem 1.5rem;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            vertical-align: middle;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.01);
        }}

        .font-mono-val {{
            font-family: var(--font-mono);
            font-size: 0.85rem;
        }}

        .pagination {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
        }}

        .pagination-info {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .pagination-btns {{
            display: flex;
            gap: 0.5rem;
        }}

        .pg-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.5rem 0.8rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: var(--transition);
        }}

        .pg-btn:hover:not(:disabled) {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.2);
        }}

        .pg-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 0.5rem;
        }}

        .param-item {{
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 0.75rem;
        }}

        .param-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .param-val {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.15rem;
            color: var(--text-main);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>HyPhy Codon-Curation Portal</h1>
                <p>Detailed overview of clean evolutionary alignments extracted for molecular analyses</p>
            </div>
            <div>
                <span class="badge">Pipeline Curated</span>
            </div>
        </header>

        <!-- Metrics Cards -->
        <div class="metrics-grid">
            <div class="card">
                <div class="metric-title">Retained Genes</div>
                <div class="metric-value text-success" id="m-retained-genes">-</div>
                <div class="metric-sub" id="m-retained-pct">-</div>
            </div>
            <div class="card">
                <div class="metric-title">Dropped (Few Seqs)</div>
                <div class="metric-value text-danger" id="m-dropped-few">-</div>
                <div class="metric-sub">Alignments with &lt; 3 species remaining</div>
            </div>
            <div class="card">
                <div class="metric-title">Mean Seqs per Gene</div>
                <div class="metric-value text-primary" id="m-mean-seqs">-</div>
                <div class="metric-sub">After species deduplication and stripping</div>
            </div>
            <div class="card">
                <div class="metric-title">Mean Codon Length</div>
                <div class="metric-value text-warning" id="m-mean-len">-</div>
                <div class="metric-sub">After degapping all-gap columns</div>
            </div>
        </div>

        <!-- Charts and Filter breakdown -->
        <div class="charts-row">
            <!-- Histogram panel -->
            <div class="chart-card">
                <div class="chart-title">Dataset Distributions</div>
                <div style="height: 300px; position: relative;">
                    <canvas id="distributionChart"></canvas>
                </div>
                <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
                    <button class="pg-btn" onclick="toggleChartType('seqs')" id="btn-chart-seqs" style="background: rgba(59, 130, 246, 0.15);">Sequences / Gene</button>
                    <button class="pg-btn" onclick="toggleChartType('len')" id="btn-chart-len">Codons / Gene</button>
                </div>
            </div>
            <!-- Filter rejects breakdown -->
            <div class="chart-card">
                <div class="chart-title">Filter Exclusion Summary</div>
                <div class="filter-stats-list">
                    <div class="filter-stat-item">
                        <span class="filter-stat-label">Low Coverage (&lt;50%)</span>
                        <span class="filter-stat-value text-danger" id="f-low-cov">-</span>
                    </div>
                    <div class="filter-stat-item">
                        <span class="filter-stat-label">Premature Stop Codons</span>
                        <span class="filter-stat-value text-danger" id="f-prem-stop">-</span>
                    </div>
                    <div class="filter-stat-item">
                        <span class="filter-stat-label">Deduplicated Assembly Seqs</span>
                        <span class="filter-stat-value text-primary" id="f-dedup">-</span>
                    </div>
                    <div class="filter-stat-item">
                        <span class="filter-stat-label">Unmapped Species</span>
                        <span class="filter-stat-value text-warning" id="f-unmapped">-</span>
                    </div>
                    <div class="filter-stat-item">
                        <span class="filter-stat-label">Not in Species Tree</span>
                        <span class="filter-stat-value text-warning" id="f-not-tree">-</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Curation Settings card -->
        <div class="card" style="margin-bottom: 2.5rem;">
            <div class="chart-title" style="margin-bottom: 1rem;">Curation Parameters Applied</div>
            <div class="param-grid">
                <div class="param-item">
                    <div class="param-label">Sequence Coverage</div>
                    <div class="param-val" id="p-cov">-</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Exclusions</div>
                    <div class="param-val">Outliers &amp; Pseudogenes</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Deduplication Method</div>
                    <div class="param-val" id="p-dedup">-</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Minimum Seqs / Gene</div>
                    <div class="param-val" id="p-min-seqs">-</div>
                </div>
            </div>
        </div>

        <!-- Detailed Database Section -->
        <div class="section-header">
            <h2 class="section-title" id="table-sec-title">Retained Genes (0)</h2>
            <div class="tabs">
                <button class="tab-btn active" id="tab-retained" onclick="switchTab('retained')">Retained Genes</button>
                <button class="tab-btn" id="tab-dropped" onclick="switchTab('dropped')">Dropped Genes</button>
                <button class="tab-btn" id="tab-species" onclick="switchTab('species')">Species Representation</button>
            </div>
        </div>

        <!-- Search Bar -->
        <div class="search-container">
            <input type="text" class="search-input" id="search-bar" placeholder="Search by gene name or transcript ID..." oninput="handleSearch()">
        </div>

        <!-- Data Table -->
        <div class="table-container">
            <table id="data-table">
                <thead>
                    <tr id="table-headers">
                        <!-- Dynamic headers -->
                    </tr>
                </thead>
                <tbody id="table-body">
                    <!-- Dynamic rows -->
                </tbody>
            </table>
        </div>

        <!-- Pagination Controls -->
        <div class="pagination">
            <div class="pagination-info" id="pagination-info">Showing 0-0 of 0 entries</div>
            <div class="pagination-btns">
                <button class="pg-btn" id="pg-prev" onclick="changePage(-1)">Previous</button>
                <button class="pg-btn" id="pg-next" onclick="changePage(1)">Next</button>
            </div>
        </div>
    </div>

    <!-- Embedded Curation Summary JSON -->
    <script>
        const data = {json_data_str};
    </script>

    <!-- UI Logic -->
    <script>
        let currentTab = 'retained';
        let searchQuery = '';
        let currentPage = 1;
        const pageSize = 15;
        let filteredData = [];
        let chartInstance = null;
        let activeChartType = 'seqs';

        // Load metrics on startup
        document.addEventListener('DOMContentLoaded', () => {{
            loadMetrics();
            switchTab('retained');
            buildChart();
        }});

        function loadMetrics() {{
            document.getElementById('m-retained-genes').innerText = data.metrics.total_retained_genes.toLocaleString();
            document.getElementById('m-retained-pct').innerText = `${{data.metrics.pct_genes_retained.toFixed(1)}}% of all curated database alignments`;
            document.getElementById('m-dropped-few').innerText = data.metrics.total_dropped_few_seqs.toLocaleString();
            document.getElementById('m-mean-seqs').innerText = data.metrics.average_retained_sequences.toFixed(1);
            document.getElementById('m-mean-len').innerText = `${{Math.round(data.metrics.average_retained_codon_length)}} codons`;
            
            // Rejects
            document.getElementById('f-low-cov').innerText = data.global_sequence_filtering.dropped_low_coverage.toLocaleString();
            document.getElementById('f-prem-stop').innerText = data.global_sequence_filtering.dropped_premature_stop.toLocaleString();
            document.getElementById('f-dedup').innerText = data.global_sequence_filtering.dropped_deduplicated.toLocaleString();
            document.getElementById('f-unmapped').innerText = data.global_sequence_filtering.dropped_unmapped_species.toLocaleString();
            document.getElementById('f-not-tree').innerText = data.global_sequence_filtering.dropped_not_in_tree.toLocaleString();

            // Curation Params
            document.getElementById('p-cov').innerText = data.curation_parameters.coverage_threshold;
            document.getElementById('p-dedup').innerText = data.curation_parameters.deduplication_margin;
            document.getElementById('p-min-seqs').innerText = `${{data.curation_parameters.min_retained_sequences}} species`;
        }}

        function buildChart() {{
            const ctx = document.getElementById('distributionChart').getContext('2d');
            
            let chartLabels = [];
            let chartCounts = [];
            let label = "";
            let color = "#3b82f6";
            
            if (activeChartType === 'seqs') {{
                chartLabels = data.histograms.sequences_per_gene.labels;
                chartCounts = data.histograms.sequences_per_gene.counts;
                label = "Number of Genes";
                color = "#3b82f6";
            }} else {{
                chartLabels = data.histograms.codons_per_gene.labels;
                chartCounts = data.histograms.codons_per_gene.counts;
                label = "Number of Genes";
                color = "#f59e0b";
            }}
            
            if (chartInstance) {{
                chartInstance.destroy();
            }}
            
            chartInstance = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: chartLabels,
                    datasets: [{{
                        label: label,
                        data: chartCounts,
                        backgroundColor: color,
                        borderWidth: 0,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{
                                display: false
                            }},
                            ticks: {{
                                color: '#94a3b8',
                                font: {{
                                    family: "'Plus Jakarta Sans', sans-serif",
                                    size: 11
                                }}
                            }}
                        }},
                        y: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.05)'
                            }},
                            ticks: {{
                                color: '#94a3b8',
                                font: {{
                                    family: "'Plus Jakarta Sans', sans-serif",
                                    size: 11
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}

        function toggleChartType(type) {{
            activeChartType = type;
            document.getElementById('btn-chart-seqs').style.background = type === 'seqs' ? 'rgba(59, 130, 246, 0.15)' : 'transparent';
            document.getElementById('btn-chart-len').style.background = type === 'len' ? 'rgba(245, 158, 11, 0.15)' : 'transparent';
            buildChart();
        }}

        function switchTab(tab) {{
            currentTab = tab;
            currentPage = 1;
            
            document.getElementById('tab-retained').classList.toggle('active', tab === 'retained');
            document.getElementById('tab-dropped').classList.toggle('active', tab === 'dropped');
            document.getElementById('tab-species').classList.toggle('active', tab === 'species');
            
            // Adjust search placeholder
            const searchBar = document.getElementById('search-bar');
            if (tab === 'species') {{
                searchBar.placeholder = "Search by species, leaf name, or NCBI accession...";
            }} else {{
                searchBar.placeholder = "Search by gene name or transcript ID...";
            }}
            
            handleSearch();
        }}

        function handleSearch() {{
            searchQuery = document.getElementById('search-bar').value.toLowerCase().strip();
            
            let rawData = [];
            if (currentTab === 'retained') {{
                rawData = data.retained_genes;
            }} else if (currentTab === 'dropped') {{
                rawData = data.dropped_genes;
            }} else if (currentTab === 'species') {{
                rawData = data.species_representation;
            }}
            
            if (!searchQuery) {{
                filteredData = [...rawData];
            }} else {{
                if (currentTab === 'species') {{
                    filteredData = rawData.filter(x => 
                        x.species.toLowerCase().includes(searchQuery) ||
                        x.leaf.toLowerCase().includes(searchQuery) ||
                        x.rep_assembly.toLowerCase().includes(searchQuery)
                    );
                }} else {{
                    filteredData = rawData.filter(x => 
                        x.gene.toLowerCase().includes(searchQuery) ||
                        x.transcript.toLowerCase().includes(searchQuery)
                    );
                }}
            }}
            
            currentPage = 1;
            renderTable();
        }}

        function renderTable() {{
            const headersRow = document.getElementById('table-headers');
            const tableBody = document.getElementById('table-body');
            
            // 1. Setup headers
            if (currentTab === 'retained') {{
                document.getElementById('table-sec-title').innerText = `Retained Genes (${{filteredData.length.toLocaleString()}})`;
                headersRow.innerHTML = `
                    <th>Gene Name</th>
                    <th>Transcript ID</th>
                    <th>Retained Seqs</th>
                    <th>Original Seqs</th>
                    <th>Cleaned Length (codons)</th>
                    <th>Original Length (codons)</th>
                `;
            }} else if (currentTab === 'dropped') {{
                document.getElementById('table-sec-title').innerText = `Dropped Genes (${{filteredData.length.toLocaleString()}})`;
                headersRow.innerHTML = `
                    <th>Gene Name</th>
                    <th>Transcript ID</th>
                    <th>Retained Seqs</th>
                    <th>Original Seqs</th>
                    <th>Exclusion Reason</th>
                `;
            }} else if (currentTab === 'species') {{
                document.getElementById('table-sec-title').innerText = `Species Taxonomic Coverage (${{filteredData.length.toLocaleString()}})`;
                headersRow.innerHTML = `
                    <th>Species Name</th>
                    <th>Tree Leaf Label</th>
                    <th>Representative Genome</th>
                    <th>Retained Genes</th>
                    <th>Dataset Coverage</th>
                `;
            }}
            
            // 2. Setup rows (paginated)
            tableBody.innerHTML = '';
            
            if (filteredData.length === 0) {{
                const colSpan = currentTab === 'species' ? 5 : (currentTab === 'dropped' ? 5 : 6);
                tableBody.innerHTML = `<tr><td colspan="${{colSpan}}" style="text-align: center; color: var(--text-muted); padding: 3rem;">No entries found.</td></tr>`;
                document.getElementById('pagination-info').innerText = 'Showing 0-0 of 0 entries';
                document.getElementById('pg-prev').disabled = true;
                document.getElementById('pg-next').disabled = true;
                return;
            }}
            
            const startIdx = (currentPage - 1) * pageSize;
            const endIdx = Math.min(startIdx + pageSize, filteredData.length);
            const pageData = filteredData.slice(startIdx, endIdx);
            
            pageData.forEach(row => {{
                let trContent = '';
                if (currentTab === 'retained') {{
                    trContent = `
                        <td><strong>${{row.gene}}</strong></td>
                        <td class="font-mono-val">${{row.transcript}}</td>
                        <td class="font-mono-val">${{row.retained_seqs}}</td>
                        <td class="font-mono-val">${{row.orig_seqs}}</td>
                        <td class="font-mono-val text-success">${{row.codons}}</td>
                        <td class="font-mono-val">${{row.orig_codons}}</td>
                    `;
                }} else if (currentTab === 'dropped') {{
                    trContent = `
                        <td><strong>${{row.gene}}</strong></td>
                        <td class="font-mono-val">${{row.transcript}}</td>
                        <td class="font-mono-val">${{row.retained_seqs}}</td>
                        <td class="font-mono-val">${{row.orig_seqs}}</td>
                        <td class="text-danger">${{row.reason}}</td>
                    `;
                }} else if (currentTab === 'species') {{
                    trContent = `
                        <td><strong>${{row.species}}</strong></td>
                        <td class="font-mono-val text-primary">${{row.leaf}}</td>
                        <td class="font-mono-val">${{row.rep_assembly}}</td>
                        <td class="font-mono-val text-success">${{row.gene_count.toLocaleString()}}</td>
                        <td>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; max-width: 100px;">
                                    <div style="width: ${{row.pct_retained_genes}}%; height: 100%; background: var(--accent);"></div>
                                </div>
                                <span class="font-mono-val" style="font-size: 0.75rem;">${{row.pct_retained_genes.toFixed(1)}}%</span>
                            </div>
                        </td>
                    `;
                }}
                
                const tr = document.createElement('tr');
                tr.innerHTML = trContent;
                tableBody.appendChild(tr);
            }});
            
            // 3. Update pagination
            document.getElementById('pagination-info').innerText = `Showing ${{startIdx + 1}}-${{endIdx}} of ${{filteredData.length.toLocaleString()}} entries`;
            document.getElementById('pg-prev').disabled = currentPage === 1;
            document.getElementById('pg-next').disabled = endIdx >= filteredData.length;
        }}

        function changePage(direction) {{
            currentPage += direction;
            renderTable();
        }}

        // Helper string clean
        String.prototype.strip = function() {{
            return this.replace(/^\s+|\s+$/g, '');
        }};
    </script>
</body>
</html>
"""
    with open(dest_path, 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    main()
