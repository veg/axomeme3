#!/usr/bin/env python3
"""
generate_dashboard.py
---------------------
Reads assemblies_and_species.tsv, parses the metadata, and writes a self-contained,
highly polished interactive HTML dashboard (dashboard.html) in the root of the workspace.
Now includes assembly counts per species, best representative selections, and problematic genome flags.
"""

import os
import json
import pandas as pd
import numpy as np

def clean_numeric(val):
    if pd.isna(val):
        return None
    try:
        f = float(val)
        if f == -1.0:
            return None
        return int(f) if f.is_integer() else f
    except ValueError:
        s = str(val).strip().upper()
        if s in ('NULL', 'NAN', ''):
            return None
        return None

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    tsv_path = os.path.join(base_dir, "assemblies_and_species.tsv")
    output_html_path = os.path.join(base_dir, "dashboard.html")
    
    print(f"Reading TSV data from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Clean up column data
    df['contig_N50_clean'] = df['contig N50 (bp)'].apply(clean_numeric)
    df['scaffold_N50_clean'] = df['scaffold N50 (bp)'].apply(clean_numeric)
    df['intact_clean'] = df['No. ancestral genes with intact ORF'].apply(clean_numeric)
    df['mut_clean'] = df['No. ancestral genes with inactivating mutations'].apply(clean_numeric)
    df['miss_clean'] = df['No. ancestral genes with missing sequences'].apply(clean_numeric)
    
    # Calculate species counts
    species_counts = df['Species'].value_counts().to_dict()
    
    # Determine "Best" genome per species
    def get_status_score(status):
        if pd.isna(status):
            return 0
        s = str(status).strip().lower()
        if 'chromosome' in s or 'complete' in s:
            return 3
        elif 'scaffold' in s:
            return 2
        elif 'contig' in s:
            return 1
        return 0

    df['status_score'] = df['NCBI assembly status'].apply(get_status_score)
    df['submission_date_dt'] = pd.to_datetime(df['Assembly submission date'], errors='coerce')
    
    df['sort_s_n50'] = df['scaffold_N50_clean'].fillna(0)
    df['sort_c_n50'] = df['contig_N50_clean'].fillna(0)
    df['sort_intact'] = df['intact_clean'].fillna(0)
    df['sort_date'] = df['submission_date_dt'].fillna(pd.Timestamp('1900-01-01'))
    
    sorted_df = df.sort_values(
        by=['status_score', 'sort_s_n50', 'sort_c_n50', 'sort_intact', 'sort_date'],
        ascending=[False, False, False, False, False]
    )
    best_accessions = set(sorted_df.groupby('Species').first()['NCBI accession'].dropna().unique())
    
    # Process taxonomic lineage
    def get_order_family(lineage_str):
        if pd.isna(lineage_str):
            return 'Unknown', 'Unknown'
        parts = [p.strip() for p in str(lineage_str).split(';') if p.strip()]
        order = parts[3] if len(parts) > 3 else 'Unknown'
        family = parts[4] if len(parts) > 4 else 'Unknown'
        return order, family
        
    orders_families = df['Taxonomic Lineage'].apply(get_order_family)
    df['Order'] = [x[0] for x in orders_families]
    df['Family'] = [x[1] for x in orders_families]
    
    # Create compact records list for the dashboard
    records = []
    for _, row in df.iterrows():
        # Handle date parsing
        date_str = str(row['Assembly submission date']).split(' ')[0] if pd.notna(row['Assembly submission date']) else None
        acc = str(row['NCBI accession']) if pd.notna(row['NCBI accession']) else ''
        
        # Sibling best representative check
        is_best = acc in best_accessions
        
        # Problematic checks
        reasons = []
        status_str = str(row['NCBI assembly status']).strip() if pd.notna(row['NCBI assembly status']) else 'Unknown'
        c_n50 = row['contig_N50_clean']
        s_n50 = row['scaffold_N50_clean']
        intact = row['intact_clean']
        miss = row['miss_clean']
        mut = row['mut_clean']
        
        if status_str.lower() == 'contig':
            reasons.append("Assembly status is Contig")
        if s_n50 is not None and s_n50 < 100000:
            reasons.append("Low Scaffold N50 (< 100 KB)")
        if c_n50 is not None and c_n50 < 10000:
            reasons.append("Low Contig N50 (< 10 KB)")
        if intact is not None:
            if intact < 8000:
                reasons.append("Low intact ORFs (< 8,000)")
            if miss is not None and miss > 2000:
                reasons.append("High missing sequences (> 2,000)")
            if mut is not None and mut > 2000:
                reasons.append("High inactivating mutations (> 2,000)")
                
        is_prob = len(reasons) > 0
        prob_reason = " | ".join(reasons) if is_prob else ""
        
        rec = {
            'dir': str(row['Directory Name']) if pd.notna(row['Directory Name']) else '',
            'sp': str(row['Species']) if pd.notna(row['Species']) else '',
            'common': str(row['Common name']) if pd.notna(row['Common name']) else '',
            'order': row['Order'],
            'family': row['Family'],
            'status': status_str,
            'acc': acc,
            'c_n50': c_n50,
            's_n50': s_n50,
            'intact': intact,
            'mut': mut,
            'miss': miss,
            'date': date_str,
            'org': str(row['Assembly submitter organization']) if pd.notna(row['Assembly submitter organization']) else 'Unknown',
            'sp_count': int(species_counts.get(str(row['Species']), 1)),
            'best': bool(is_best),
            'prob': bool(is_prob),
            'reason': prob_reason
        }
        records.append(rec)
        
    records_json = json.dumps(records, default=str)
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TOGA Genome Alignments Dashboard</title>
    <meta name="description" content="Interactive dashboard for visualizing genome assembly quality and TOGA projections.">
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #080b11;
            --bg-secondary: #101524;
            --bg-card: rgba(22, 30, 49, 0.75);
            --bg-card-hover: rgba(30, 41, 67, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.15);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #00f2fe;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-green: #10b981;
            --accent-orange: #f97316;
            --accent-red: #ef4444;
            --glass-glow: 0 8px 32px 0 rgba(0, 242, 254, 0.05);
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 242, 254, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }}

        /* Scrollbar styles */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg-primary);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}

        header {{
            border-bottom: 1px solid var(--border-color);
            background: rgba(8, 11, 17, 0.85);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 1.25rem 2rem;
            display: flex;
            align-content: center;
            justify-content: space-between;
        }}

        .brand-container {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .brand-logo {{
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 1.25rem;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        }}

        .brand-title-group h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(to right, #ffffff, var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-title-group p {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .header-stats-ticker {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }}

        .ticker-item {{
            text-align: right;
            border-right: 1px solid var(--border-color);
            padding-right: 1.5rem;
        }}

        .ticker-item:last-child {{
            border-right: none;
            padding-right: 0;
        }}

        .ticker-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        .ticker-val {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        main {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        /* KPI grid */
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: var(--transition-smooth);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(8px);
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-gradient, var(--accent-blue));
            opacity: 0.7;
        }}

        .kpi-card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-hover);
            box-shadow: var(--glass-glow);
            background: var(--bg-card-hover);
        }}

        .kpi-content {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }}

        .kpi-label {{
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .kpi-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        .kpi-subtext {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .kpi-icon-wrapper {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: var(--accent-color, var(--accent-blue));
            border: 1px solid var(--border-color);
        }}

        /* Row of charts */
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
        }}

        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            min-height: 420px;
        }}

        .chart-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .chart-title-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .chart-title-group i {{
            color: var(--accent-cyan);
        }}

        .chart-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
        }}

        .chart-canvas-container {{
            position: relative;
            flex-grow: 1;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        /* Data table container */
        .data-section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.75rem;
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}

        .filter-toolbar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }}

        .search-wrapper {{
            position: relative;
            min-width: 300px;
            flex-grow: 0.2;
        }}

        .search-wrapper i {{
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
        }}

        .search-input {{
            width: 100%;
            background: rgba(8, 11, 17, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            color: var(--text-primary);
            font-family: inherit;
            outline: none;
            transition: var(--transition-smooth);
        }}

        .search-input:focus {{
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
        }}

        .filter-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .select-filter {{
            background: rgba(8, 11, 17, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            color: var(--text-primary);
            font-family: inherit;
            outline: none;
            cursor: pointer;
            transition: var(--transition-smooth);
        }}

        .select-filter:focus, .select-filter:hover {{
            border-color: var(--accent-blue);
        }}

        /* Table design */
        .table-responsive {{
            width: 100%;
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.875rem;
        }}

        thead {{
            background: rgba(16, 21, 36, 0.8);
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            padding: 1rem 1.25rem;
            font-weight: 600;
            color: var(--text-secondary);
            font-family: 'Outfit', sans-serif;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            cursor: pointer;
            user-select: none;
        }}

        th:hover {{
            color: var(--text-primary);
        }}

        th i {{
            margin-left: 0.25rem;
            font-size: 0.7rem;
            opacity: 0.5;
        }}

        tbody tr {{
            border-bottom: 1px solid var(--border-color);
            transition: var(--transition-smooth);
            cursor: pointer;
        }}

        tbody tr:last-child {{
            border-bottom: none;
        }}

        tbody tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}

        td {{
            padding: 1rem 1.25rem;
            vertical-align: middle;
        }}

        .col-species {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .sp-name-row {{
            font-weight: 600;
            color: var(--text-primary);
            font-style: italic;
        }}

        .badges-wrapper {{
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
            align-items: center;
        }}

        /* Badges styling */
        .badge-best {{
            background: rgba(59, 130, 246, 0.1);
            color: var(--accent-blue);
            border: 1px solid rgba(59, 130, 246, 0.2);
            padding: 0.1rem 0.4rem;
            font-size: 0.65rem;
            border-radius: 4px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .badge-prob {{
            background: rgba(239, 68, 68, 0.1);
            color: var(--accent-red);
            border: 1px solid rgba(239, 68, 68, 0.2);
            padding: 0.1rem 0.4rem;
            font-size: 0.65rem;
            border-radius: 4px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .sp-count-badge {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.1rem 0.35rem;
            font-size: 0.7rem;
            color: var(--text-secondary);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            display: inline-block;
        }}

        tbody tr:hover .sp-count-badge {{
            background: rgba(0, 242, 254, 0.15);
            color: var(--accent-cyan);
            border-color: rgba(0, 242, 254, 0.3);
        }}

        .col-common {{
            color: var(--text-secondary);
        }}

        .badge-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .badge-chromosome {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .badge-scaffold {{
            background: rgba(59, 130, 246, 0.1);
            color: var(--accent-blue);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }}

        .badge-contig {{
            background: rgba(249, 115, 22, 0.1);
            color: var(--accent-orange);
            border: 1px solid rgba(249, 115, 22, 0.2);
        }}

        .badge-genome {{
            background: rgba(139, 92, 246, 0.1);
            color: var(--accent-purple);
            border: 1px solid rgba(139, 92, 246, 0.2);
        }}

        .col-n50 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
        }}

        .col-genes {{
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
            color: var(--accent-cyan);
        }}

        .pagination-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 0.5rem;
        }}

        .pagination-info {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .pagination-buttons {{
            display: flex;
            gap: 0.5rem;
        }}

        .btn-pagination {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.5rem 0.85rem;
            color: var(--text-primary);
            font-size: 0.8rem;
            cursor: pointer;
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }}

        .btn-pagination:hover:not(:disabled) {{
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--text-muted);
        }}

        .btn-pagination:disabled {{
            opacity: 0.35;
            cursor: not-allowed;
        }}

        /* Modal styling */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(5, 7, 11, 0.85);
            backdrop-filter: blur(8px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: var(--transition-smooth);
        }}

        .modal-overlay.active {{
            opacity: 1;
            pointer-events: auto;
        }}

        .modal-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            width: 100%;
            max-width: 750px;
            overflow-y: auto;
            max-height: 90vh;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
            transform: scale(0.9) translateY(20px);
            transition: var(--transition-smooth);
            position: relative;
        }}

        .modal-overlay.active .modal-card {{
            transform: scale(1) translateY(0);
        }}

        .modal-header {{
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.01);
        }}

        .modal-title-group h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            font-style: italic;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .modal-title-group p {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .btn-close-modal {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            cursor: pointer;
            transition: var(--transition-smooth);
        }}

        .btn-close-modal:hover {{
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border-color: rgba(239, 68, 68, 0.2);
        }}

        .modal-body {{
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.75rem;
        }}

        .details-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }}

        .detail-item {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .detail-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .detail-value {{
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-primary);
        }}

        .detail-value.italic {{
            font-style: italic;
        }}

        /* Problematic alert box */
        .prob-alert-box {{
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            color: var(--accent-red);
            font-size: 0.85rem;
        }}

        .prob-alert-title {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .prob-alert-reasons {{
            padding-left: 1.25rem;
            line-height: 1.5;
            color: var(--text-primary);
        }}

        .projection-stats-section {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
        }}

        .projection-header {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--accent-cyan);
        }}

        .projection-bars-container {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .bar-group {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }}

        .bar-label-group {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
        }}

        .bar-label-name {{
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .bar-label-val {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }}

        .bar-bg {{
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            width: 100%;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease-out;
        }}

        .bar-intact {{
            background: linear-gradient(to right, var(--accent-cyan), var(--accent-blue));
        }}

        .bar-mutations {{
            background: linear-gradient(to right, var(--accent-orange), var(--accent-pink));
        }}

        .bar-missing {{
            background: linear-gradient(to right, var(--text-muted), var(--text-secondary));
        }}

        .no-projections-badge {{
            text-align: center;
            padding: 1.5rem;
            border: 1px dashed var(--border-color);
            border-radius: 12px;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        /* Sibling assemblies section */
        .sibling-assemblies-section {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
        }}

        .sibling-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            margin-top: 0.75rem;
        }}

        .sibling-table th {{
            padding: 0.5rem;
            font-size: 0.7rem;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .sibling-table td {{
            padding: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .sibling-table tr:last-child td {{
            border-bottom: none;
        }}

        .sibling-table tr.active-sibling {{
            background: rgba(0, 242, 254, 0.06);
        }}

        .sibling-table tr.clickable-sibling {{
            transition: var(--transition-smooth);
        }}

        .sibling-table tr.clickable-sibling:hover {{
            background: rgba(255, 255, 255, 0.04);
            cursor: pointer;
        }}

        /* Responsive adjustments */
        @media (max-width: 1024px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
            .ticker-item:nth-child(n+3) {{
                display: none;
            }}
        }}

        @media (max-width: 768px) {{
            header {{
                flex-direction: column;
                gap: 1rem;
                align-items: flex-start;
            }}
            .header-stats-ticker {{
                width: 100%;
                justify-content: space-between;
            }}
            .ticker-item {{
                padding-right: 0.75rem;
                gap: 0.5rem;
            }}
            main {{
                padding: 1rem;
            }}
            .details-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="brand-container">
            <div class="brand-logo">
                <i class="fa-solid fa-dna"></i>
            </div>
            <div class="brand-title-group">
                <h1>TOGA Alignment Hub</h1>
                <p>Genomic Assemblies & Projections</p>
            </div>
        </div>
        <div class="header-stats-ticker" id="tickerContainer">
            <!-- Dynamic ticker items loaded in JS -->
        </div>
    </header>

    <main>
        <!-- KPI Cards Grid -->
        <div class="kpi-container" id="kpiContainer">
            <!-- Loaded by JS -->
        </div>

        <!-- Charts Grid -->
        <div class="charts-row">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-title-group">
                        <i class="fa-solid fa-chart-pie"></i>
                        <span class="chart-title">Taxonomic Order Distribution</span>
                    </div>
                </div>
                <div class="chart-canvas-container">
                    <canvas id="orderChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-title-group">
                        <i class="fa-solid fa-calendar-alt"></i>
                        <span class="chart-title">Assemblies Submitted Over Time</span>
                    </div>
                </div>
                <div class="chart-canvas-container">
                    <canvas id="timeChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Main Data section -->
        <div class="data-section">
            <div class="filter-toolbar">
                <div class="search-wrapper">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    <input type="text" id="searchInput" placeholder="Search species or accession..." class="search-input">
                </div>
                <div class="filter-controls">
                    <select id="qualityFilter" class="select-filter">
                        <option value="">All Genomes</option>
                        <option value="best">Representatives Only</option>
                        <option value="prob">Problematic Genomes</option>
                        <option value="ok">Representatives & Quality OK</option>
                    </select>
                    <select id="orderFilter" class="select-filter">
                        <option value="">All Orders</option>
                    </select>
                    <select id="statusFilter" class="select-filter">
                        <option value="">All Statuses</option>
                    </select>
                </div>
            </div>

            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 32%" id="th-species">Species <i class="fa-solid fa-sort"></i></th>
                            <th style="width: 20%" id="th-common">Common Name <i class="fa-solid fa-sort"></i></th>
                            <th style="width: 12%" id="th-order">Order <i class="fa-solid fa-sort"></i></th>
                            <th style="width: 10%" id="th-status">Status <i class="fa-solid fa-sort"></i></th>
                            <th style="width: 13%" id="th-c_n50">Contig N50 <i class="fa-solid fa-sort"></i></th>
                            <th style="width: 13%" id="th-s_n50">Scaffold N50 <i class="fa-solid fa-sort"></i></th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        <!-- Loaded dynamically -->
                    </tbody>
                </table>
            </div>

            <div class="pagination-container">
                <div class="pagination-info" id="paginationInfo">
                    Showing 0 to 0 of 0 entries
                </div>
                <div class="pagination-buttons">
                    <button id="btnPrev" class="btn-pagination" disabled>
                        <i class="fa-solid fa-chevron-left"></i> Previous
                    </button>
                    <button id="btnNext" class="btn-pagination" disabled>
                        Next <i class="fa-solid fa-chevron-right"></i>
                    </button>
                </div>
            </div>
        </div>
    </main>

    <!-- Details Modal -->
    <div class="modal-overlay" id="modalOverlay">
        <div class="modal-card">
            <div class="modal-header">
                <div class="modal-title-group">
                    <h3 id="modalSpecies">Homo sapiens</h3>
                    <p id="modalCommon">Human</p>
                </div>
                <button class="btn-close-modal" id="btnCloseModal">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
            <div class="modal-body">
                <!-- Potentially problematic alert box -->
                <div class="prob-alert-box" id="modalProbBox" style="display: none;">
                    <div class="prob-alert-title">
                        <i class="fa-solid fa-circle-exclamation"></i>
                        <span>Potentially Problematic Genome</span>
                    </div>
                    <ul class="prob-alert-reasons" id="modalProbReasons">
                        <!-- Loaded dynamically -->
                    </ul>
                </div>

                <div class="details-grid">
                    <div class="detail-item">
                        <span class="detail-label">NCBI Accession</span>
                        <span class="detail-value" id="modalAccession">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Assembly Name</span>
                        <span class="detail-value" id="modalDirName">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Order & Family</span>
                        <span class="detail-value" id="modalTaxon">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Assembly Status</span>
                        <span class="detail-value" id="modalStatus">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Contig N50 (bp)</span>
                        <span class="detail-value col-n50" id="modalContigN50">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Scaffold N50 (bp)</span>
                        <span class="detail-value col-n50" id="modalScaffoldN50">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Submission Date</span>
                        <span class="detail-value" id="modalDate">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Submitter Organization</span>
                        <span class="detail-value" id="modalSubmitter">-</span>
                    </div>
                </div>

                <div class="projection-stats-section" id="modalProjectionsSection">
                    <div class="projection-header">
                        <i class="fa-solid fa-chart-column"></i>
                        <span>TOGA Ancestral Gene Projections</span>
                    </div>
                    <div class="projection-bars-container" id="modalProjectionsContent">
                        <!-- Custom progress bars loaded dynamically -->
                    </div>
                </div>

                <!-- Sibling Assemblies Section -->
                <div class="sibling-assemblies-section" id="modalSiblingsSection" style="display: none;">
                    <div class="projection-header" style="color: var(--accent-blue);">
                        <i class="fa-solid fa-code-compare"></i>
                        <span>Assemblies for this Species</span>
                    </div>
                    <div id="modalSiblingsContent">
                        <!-- Sibling assemblies comparison table loaded dynamically -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Data Injection & Dashboard Engine -->
    <script>
        const rawData = {records_json};
        
        // State variables
        let filteredData = [...rawData];
        let currentPage = 1;
        const rowsPerPage = 15;
        let sortColumn = 'sp';
        let sortDirection = 'asc'; // 'asc' or 'desc'
        
        // Chart handles
        let orderChartInstance = null;
        let timeChartInstance = null;

        // Formatter functions
        function formatBP(value) {{
            if (value === null || value === undefined) return '-';
            return Number(value).toLocaleString();
        }}

        function formatPercent(val, total) {{
            return ((val / total) * 100).toFixed(1) + '%';
        }}

        // Initialize App
        document.addEventListener('DOMContentLoaded', () => {{
            setupFilters();
            renderKPIs();
            renderTicker();
            updateDashboard();
            setupEvents();
        }});

        function setupFilters() {{
            const orders = new Set();
            const statuses = new Set();
            
            rawData.forEach(item => {{
                if (item.order) orders.add(item.order);
                if (item.status) statuses.add(item.status);
            }});

            const orderFilter = document.getElementById('orderFilter');
            [...orders].sort().forEach(ord => {{
                const opt = document.createElement('option');
                opt.value = ord;
                opt.textContent = ord;
                orderFilter.appendChild(opt);
            }});

            const statusFilter = document.getElementById('statusFilter');
            [...statuses].sort().forEach(st => {{
                const opt = document.createElement('option');
                opt.value = st;
                opt.textContent = st;
                statusFilter.appendChild(opt);
            }});
        }}

        function renderKPIs() {{
            const total = rawData.length;
            const uniqueSpecies = new Set(rawData.map(d => d.sp)).size;
            
            // Chromosome status count
            const chromCount = rawData.filter(d => d.status.toLowerCase() === 'chromosome').length;
            const chromPct = formatPercent(chromCount, total);
            
            // Valid projections
            const projections = rawData.filter(d => d.intact !== null && d.intact !== undefined);
            const projCount = projections.length;
            const projPct = formatPercent(projCount, total);

            const container = document.getElementById('kpiContainer');
            container.innerHTML = `
                <div class="kpi-card" style="--accent-gradient: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)); --accent-color: var(--accent-cyan)">
                    <div class="kpi-content">
                        <span class="kpi-label">Total Assemblies</span>
                        <span class="kpi-value">${{total.toLocaleString()}}</span>
                        <span class="kpi-subtext">Across ${{uniqueSpecies.toLocaleString()}} distinct species</span>
                    </div>
                    <div class="kpi-icon-wrapper">
                        <i class="fa-solid fa-folder-tree"></i>
                    </div>
                </div>
                <div class="kpi-card" style="--accent-gradient: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); --accent-color: var(--accent-blue)">
                    <div class="kpi-content">
                        <span class="kpi-label">Chromosome Level</span>
                        <span class="kpi-value">${{chromPct}}</span>
                        <span class="kpi-subtext">${{chromCount.toLocaleString()}} high-fidelity assemblies</span>
                    </div>
                    <div class="kpi-icon-wrapper">
                        <i class="fa-solid fa-microscope"></i>
                    </div>
                </div>
                <div class="kpi-card" style="--accent-gradient: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)); --accent-color: var(--accent-purple)">
                    <div class="kpi-content">
                        <span class="kpi-label">TOGA Projected</span>
                        <span class="kpi-value">${{projPct}}</span>
                        <span class="kpi-subtext">${{projCount.toLocaleString()}} assemblies annotated</span>
                    </div>
                    <div class="kpi-icon-wrapper">
                        <i class="fa-solid fa-chart-line"></i>
                    </div>
                </div>
            `;
        }}

        function renderTicker() {{
            // Clean up N50 metrics for averages
            const contigN50s = rawData.map(d => d.c_n50).filter(v => v !== null);
            const scaffoldN50s = rawData.map(d => d.s_n50).filter(v => v !== null);
            
            // Calculate medians
            contigN50s.sort((a,b) => a-b);
            scaffoldN50s.sort((a,b) => a-b);
            
            const medianContig = contigN50s.length > 0 ? contigN50s[Math.floor(contigN50s.length / 2)] : 0;
            const medianScaffold = scaffoldN50s.length > 0 ? scaffoldN50s[Math.floor(scaffoldN50s.length / 2)] : 0;

            const container = document.getElementById('tickerContainer');
            container.innerHTML = `
                <div class="ticker-item">
                    <p class="ticker-label">Median Contig N50</p>
                    <p class="ticker-val">${{formatBP(medianContig)}} bp</p>
                </div>
                <div class="ticker-item">
                    <p class="ticker-label">Median Scaffold N50</p>
                    <p class="ticker-val">${{formatBP(medianScaffold)}} bp</p>
                </div>
            `;
        }}

        function updateDashboard() {{
            filterAndSortData();
            renderTable();
            renderCharts();
        }}

        function filterAndSortData() {{
            const searchVal = document.getElementById('searchInput').value.toLowerCase().trim();
            const orderVal = document.getElementById('orderFilter').value;
            const statusVal = document.getElementById('statusFilter').value;
            const qualityVal = document.getElementById('qualityFilter').value;

            filteredData = rawData.filter(item => {{
                const matchesSearch = !searchVal || 
                    item.sp.toLowerCase().includes(searchVal) || 
                    item.common.toLowerCase().includes(searchVal) ||
                    item.acc.toLowerCase().includes(searchVal);
                
                const matchesOrder = !orderVal || item.order === orderVal;
                const matchesStatus = !statusVal || item.status === statusVal;
                
                let matchesQuality = true;
                if (qualityVal === 'best') matchesQuality = item.best;
                else if (qualityVal === 'prob') matchesQuality = item.prob;
                else if (qualityVal === 'ok') matchesQuality = item.best && !item.prob;

                return matchesSearch && matchesOrder && matchesStatus && matchesQuality;
            }});

            // Apply Sort
            filteredData.sort((a, b) => {{
                let valA = a[sortColumn];
                let valB = b[sortColumn];

                // Handle nulls in sort
                if (valA === null || valA === undefined) return sortDirection === 'asc' ? 1 : -1;
                if (valB === null || valB === undefined) return sortDirection === 'asc' ? -1 : 1;

                if (typeof valA === 'string') {{
                    return sortDirection === 'asc' 
                        ? valA.localeCompare(valB) 
                        : valB.localeCompare(valA);
                }} else {{
                    return sortDirection === 'asc' ? valA - valB : valB - valA;
                }}
            }});
        }}

        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            const total = filteredData.length;
            const totalPages = Math.ceil(total / rowsPerPage) || 1;
            
            if (currentPage > totalPages) currentPage = totalPages;
            
            const startIdx = (currentPage - 1) * rowsPerPage;
            const endIdx = Math.min(startIdx + rowsPerPage, total);
            const pageData = filteredData.slice(startIdx, endIdx);

            if (pageData.length === 0) {{
                tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 3rem; color: var(--text-muted);">No assemblies found matching filters.</td></tr>`;
                document.getElementById('paginationInfo').textContent = 'Showing 0 to 0 of 0 entries';
                document.getElementById('btnPrev').disabled = true;
                document.getElementById('btnNext').disabled = true;
                return;
            }}

            pageData.forEach(item => {{
                const tr = document.createElement('tr');
                tr.addEventListener('click', () => openModal(item));

                let statusBadgeClass = 'badge-status';
                const s = item.status.toLowerCase();
                if (s.includes('chromosome')) statusBadgeClass += ' badge-chromosome';
                else if (s.includes('scaffold')) statusBadgeClass += ' badge-scaffold';
                else if (s.includes('contig')) statusBadgeClass += ' badge-contig';
                else statusBadgeClass += ' badge-genome';

                const spBadge = item.sp_count > 1 
                    ? `<span class="sp-count-badge" title="${{item.sp_count}} assemblies in dataset">${{item.sp_count}}</span>` 
                    : '';

                let badgesHtml = '';
                if (item.best) badgesHtml += ` <span class="badge-best" title="Best representative for this species"><i class="fa-solid fa-star"></i> Representative</span>`;
                if (item.prob) badgesHtml += ` <span class="badge-prob" title="Problematic: ${{item.reason}}"><i class="fa-solid fa-triangle-exclamation"></i> Flagged</span>`;

                tr.innerHTML = `
                    <td class="col-species">
                        <span class="sp-name-row">${{item.sp}}</span>
                        <div class="badges-wrapper" style="margin-top: 0.2rem;">
                            ${{badgesHtml}}
                            ${{spBadge}}
                        </div>
                    </td>
                    <td class="col-common">${{item.common || '<span style="opacity: 0.3">-</span>'}}</td>
                    <td>${{item.order}}</td>
                    <td><span class="${{statusBadgeClass}}">${{item.status}}</span></td>
                    <td class="col-n50">${{formatBP(item.c_n50)}}</td>
                    <td class="col-n50">${{formatBP(item.s_n50)}}</td>
                `;
                tbody.appendChild(tr);
            }});

            // Update pagination text
            document.getElementById('paginationInfo').textContent = `Showing ${{startIdx + 1}} to ${{endIdx}} of ${{total.toLocaleString()}} entries`;
            document.getElementById('btnPrev').disabled = currentPage === 1;
            document.getElementById('btnNext').disabled = currentPage === totalPages;
        }}

        function renderCharts() {{
            // Chart 1: Taxonomic Order Distribution (Doughnut)
            const orderCounts = {{}};
            filteredData.forEach(d => {{
                if (d.order) {{
                    orderCounts[d.order] = (orderCounts[d.order] || 0) + 1;
                }}
            }});

            const sortedOrders = Object.entries(orderCounts)
                .sort((a, b) => b[1] - a[1]);

            const topCount = 6;
            const chartLabels = [];
            const chartData = [];
            let otherSum = 0;

            sortedOrders.forEach((item, idx) => {{
                if (idx < topCount) {{
                    chartLabels.push(item[0]);
                    chartData.push(item[1]);
                }} else {{
                    otherSum += item[1];
                }}
            }});

            if (otherSum > 0) {{
                chartLabels.push('Other');
                chartData.push(otherSum);
            }}

            const orderCtx = document.getElementById('orderChart').getContext('2d');
            if (orderChartInstance) {{
                orderChartInstance.destroy();
            }}

            orderChartInstance = new Chart(orderCtx, {{
                type: 'doughnut',
                data: {{
                    labels: chartLabels,
                    datasets: [{{
                        data: chartData,
                        backgroundColor: [
                            '#00f2fe',
                            '#3b82f6',
                            '#8b5cf6',
                            '#ec4899',
                            '#10b981',
                            '#f97316',
                            '#64748b'
                        ],
                        borderWidth: 1,
                        borderColor: '#101524'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'right',
                            labels: {{
                                color: '#94a3b8',
                                font: {{
                                    family: 'Plus Jakarta Sans',
                                    size: 11
                                }},
                                boxWidth: 12,
                                padding: 15
                            }}
                        }}
                    }},
                    cutout: '65%'
                }}
            }});

            // Chart 2: Assemblies Submitted Over Time (Area Chart)
            const yearCounts = {{}};
            filteredData.forEach(d => {{
                if (d.date) {{
                    const yr = new Date(d.date).getFullYear();
                    if (!isNaN(yr)) {{
                        yearCounts[yr] = (yearCounts[yr] || 0) + 1;
                    }}
                }}
            }});

            const sortedYears = Object.entries(yearCounts).sort((a,b) => a[0] - b[0]);
            const timeLabels = sortedYears.map(x => x[0]);
            const timeData = sortedYears.map(x => x[1]);

            const timeCtx = document.getElementById('timeChart').getContext('2d');
            if (timeChartInstance) {{
                timeChartInstance.destroy();
            }}

            timeChartInstance = new Chart(timeCtx, {{
                type: 'line',
                data: {{
                    labels: timeLabels,
                    datasets: [{{
                        label: 'Submitted Assemblies',
                        data: timeData,
                        fill: true,
                        backgroundColor: 'rgba(0, 242, 254, 0.1)',
                        borderColor: '#00f2fe',
                        borderWidth: 2,
                        pointBackgroundColor: '#8b5cf6',
                        pointBorderColor: '#080b11',
                        pointHoverRadius: 6,
                        tension: 0.3
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
                                color: 'rgba(255, 255, 255, 0.04)'
                            }},
                            ticks: {{
                                color: '#64748b',
                                font: {{
                                    family: 'Plus Jakarta Sans'
                                }}
                            }}
                        }},
                        y: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.04)'
                            }},
                            ticks: {{
                                color: '#64748b',
                                font: {{
                                    family: 'Plus Jakarta Sans'
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}

        function setupEvents() {{
            // Search Input
            document.getElementById('searchInput').addEventListener('input', () => {{
                currentPage = 1;
                updateDashboard();
            }});

            // Filters
            document.getElementById('orderFilter').addEventListener('change', () => {{
                currentPage = 1;
                updateDashboard();
            }});

            document.getElementById('statusFilter').addEventListener('change', () => {{
                currentPage = 1;
                updateDashboard();
            }});

            document.getElementById('qualityFilter').addEventListener('change', () => {{
                currentPage = 1;
                updateDashboard();
            }});

            // Pagination Buttons
            document.getElementById('btnPrev').addEventListener('click', () => {{
                if (currentPage > 1) {{
                    currentPage--;
                    renderTable();
                }}
            }});

            document.getElementById('btnNext').addEventListener('click', () => {{
                const total = filteredData.length;
                const totalPages = Math.ceil(total / rowsPerPage);
                if (currentPage < totalPages) {{
                    currentPage++;
                    renderTable();
                }}
            }});

            // Sorting Column Click Listeners
            const sortMapping = {{
                'th-species': 'sp',
                'th-common': 'common',
                'th-order': 'order',
                'th-status': 'status',
                'th-c_n50': 'c_n50',
                'th-s_n50': 's_n50'
            }};

            Object.entries(sortMapping).forEach(([thId, colKey]) => {{
                document.getElementById(thId).addEventListener('click', () => {{
                    if (sortColumn === colKey) {{
                        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        sortColumn = colKey;
                        sortDirection = 'asc';
                    }}

                    // Reset header sort icons
                    Object.keys(sortMapping).forEach(id => {{
                        const th = document.getElementById(id);
                        const icon = th.querySelector('i');
                        if (id === thId) {{
                            icon.className = sortDirection === 'asc' 
                                ? 'fa-solid fa-sort-up' 
                                : 'fa-solid fa-sort-down';
                            icon.style.opacity = 1;
                        }} else {{
                            icon.className = 'fa-solid fa-sort';
                            icon.style.opacity = 0.5;
                        }}
                    }});

                    updateDashboard();
                }});
            }});

            // Modal Close Events
            document.getElementById('btnCloseModal').addEventListener('click', closeModal);
            document.getElementById('modalOverlay').addEventListener('click', (e) => {{
                if (e.target.id === 'modalOverlay') closeModal();
            }});

            // Press Escape to close modal
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') closeModal();
            }});
        }}

        function openModalByAccession(acc) {{
            const item = rawData.find(d => d.acc === acc);
            if (item) openModal(item);
        }}

        function openModal(item) {{
            // Render modal header badges
            let badgesHtml = '';
            if (item.best) badgesHtml += ` <span class="badge-best" style="font-size: 0.75rem;"><i class="fa-solid fa-star"></i> Representative</span>`;
            if (item.prob) badgesHtml += ` <span class="badge-prob" style="font-size: 0.75rem;"><i class="fa-solid fa-triangle-exclamation"></i> Flagged</span>`;
            
            document.getElementById('modalSpecies').innerHTML = `${{item.sp}} ${{badgesHtml}}`;
            document.getElementById('modalCommon').textContent = item.common || 'Common Name Unreported';
            document.getElementById('modalAccession').textContent = item.acc || 'Unknown';
            document.getElementById('modalDirName').textContent = item.dir || 'Unknown';
            document.getElementById('modalTaxon').textContent = `${{item.order}} / ${{item.family}}`;
            document.getElementById('modalStatus').textContent = item.status;
            document.getElementById('modalContigN50').textContent = formatBP(item.c_n50);
            document.getElementById('modalScaffoldN50').textContent = formatBP(item.s_n50);
            document.getElementById('modalDate').textContent = item.date || 'Unknown';
            document.getElementById('modalSubmitter').textContent = item.org || 'Unknown';

            // Problematic Warning Box
            const probBox = document.getElementById('modalProbBox');
            const probReasons = document.getElementById('modalProbReasons');
            if (item.prob) {{
                probBox.style.display = 'block';
                probReasons.innerHTML = item.reason.split('|')
                    .map(r => `<li>${{r.trim()}}</li>`)
                    .join('');
            }} else {{
                probBox.style.display = 'none';
            }}

            const projSection = document.getElementById('modalProjectionsSection');
            const projContent = document.getElementById('modalProjectionsContent');

            if (item.intact !== null && item.intact !== undefined) {{
                projSection.style.display = 'block';
                
                const intact = item.intact || 0;
                const mut = item.mut || 0;
                const miss = item.miss || 0;
                const totalGenes = intact + mut + miss;
                
                const intactPct = totalGenes > 0 ? (intact / totalGenes * 100).toFixed(1) : 0;
                const mutPct = totalGenes > 0 ? (mut / totalGenes * 100).toFixed(1) : 0;
                const missPct = totalGenes > 0 ? (miss / totalGenes * 100).toFixed(1) : 0;

                projContent.innerHTML = `
                    <div class="bar-group">
                        <div class="bar-label-group">
                            <span class="bar-label-name">Intact ORFs</span>
                            <span class="bar-label-val">${{intact.toLocaleString()}} (${{intactPct}}%)</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-intact" style="width: ${{intactPct}}%"></div>
                        </div>
                    </div>
                    <div class="bar-group">
                        <div class="bar-label-group">
                            <span class="bar-label-name">Inactivating Mutations</span>
                            <span class="bar-label-val">${{mut.toLocaleString()}} (${{mutPct}}%)</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-mutations" style="width: ${{mutPct}}%"></div>
                        </div>
                    </div>
                    <div class="bar-group">
                        <div class="bar-label-group">
                            <span class="bar-label-name">Missing Sequences</span>
                            <span class="bar-label-val">${{miss.toLocaleString()}} (${{missPct}}%)</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-missing" style="width: ${{missPct}}%"></div>
                        </div>
                    </div>
                `;
            }} else {{
                projSection.style.display = 'block';
                projContent.innerHTML = `
                    <div class="no-projections-badge">
                        <i class="fa-solid fa-triangle-exclamation" style="margin-bottom: 0.5rem; display: block; font-size: 1.25rem; color: var(--text-muted)"></i>
                        No TOGA projections data available for this genome assembly.
                    </div>
                `;
            }}

            // Sibling assemblies comparison table
            const siblingSection = document.getElementById('modalSiblingsSection');
            const siblingContent = document.getElementById('modalSiblingsContent');
            
            const siblings = rawData.filter(d => d.sp === item.sp);
            if (siblings.length > 1) {{
                siblingSection.style.display = 'block';
                let tbodyHtml = '';
                siblings.forEach(sib => {{
                    const isActive = sib.acc === item.acc;
                    const rowClass = isActive ? 'active-sibling' : 'clickable-sibling';
                    const clickAttr = isActive ? '' : `onclick="event.stopPropagation(); openModalByAccession('${{sib.acc}}')"`;
                    
                    let sBadge = 'badge-status';
                    const s = sib.status.toLowerCase();
                    if (s.includes('chromosome')) sBadge += ' badge-chromosome';
                    else if (s.includes('scaffold')) sBadge += ' badge-scaffold';
                    else if (s.includes('contig')) sBadge += ' badge-contig';
                    else sBadge += ' badge-genome';

                    let sibBadges = '';
                    if (sib.best) sibBadges += ` <span class="badge-best" style="font-size: 0.55rem; padding: 0.05rem 0.25rem;"><i class="fa-solid fa-star" style="font-size: 0.5rem;"></i> Rep</span>`;
                    if (sib.prob) sibBadges += ` <span class="badge-prob" style="font-size: 0.55rem; padding: 0.05rem 0.25rem;"><i class="fa-solid fa-triangle-exclamation" style="font-size: 0.5rem;"></i> Flagged</span>`;

                    tbodyHtml += `
                        <tr class="${{rowClass}}" ${{clickAttr}} style="height: 38px;">
                            <td style="font-weight: 500; padding-left: 0.75rem; color: ${{isActive ? 'var(--accent-cyan)' : 'var(--text-primary)'}}">
                                ${{sib.acc}}${{sibBadges}}
                            </td>
                            <td><span class="${{sBadge}}" style="font-size: 0.65rem; padding: 0.15rem 0.45rem;">${{sib.status}}</span></td>
                            <td class="col-n50">${{formatBP(sib.s_n50)}}</td>
                            <td class="col-genes" style="padding-right: 0.75rem; text-align: right;">${{sib.intact ? sib.intact.toLocaleString() : '-'}}</td>
                        </tr>
                    `;
                }});
                
                siblingContent.innerHTML = `
                    <table class="sibling-table">
                        <thead>
                            <tr>
                                <th style="text-align: left; padding-left: 0.75rem;">Accession</th>
                                <th style="text-align: left;">Status</th>
                                <th style="text-align: left;">Scaffold N50</th>
                                <th style="text-align: right; padding-right: 0.75rem;">Intact ORFs</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{tbodyHtml}}
                        </tbody>
                    </table>
                `;
            }} else {{
                siblingSection.style.display = 'none';
            }}

            document.getElementById('modalOverlay').classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('modalOverlay').classList.remove('active');
        }}
    </script>
</body>
</html>
"""
    
    print(f"Writing self-contained dashboard file to: {output_html_path}")
    with open(output_html_path, 'w') as f:
        f.write(html_template)
        
    print("Dashboard generation completed successfully!")

if __name__ == "__main__":
    main()
