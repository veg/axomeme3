#!/usr/bin/env python3
"""
summarize_metadata.py
---------------------
Analyzes assemblies_and_species.tsv and provides summary statistics about:
1. Overall counts (assemblies, unique species)
2. Taxonomic lineage distribution (e.g. Orders, Families, Genera)
3. Assembly status (Chromosome, Scaffold, Contig, etc.) and assembly submitters
4. Assembly quality metrics (contig/scaffold N50 distributions)
5. Ancestral gene projection status (intact ORFs, inactivating mutations, missing sequences)
6. Historical trends (assembly submission dates)

Saves the summary report in markdown format to docs/genome_metadata_summary.md.
"""

import os
import pandas as pd
import numpy as np

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    try:
        f = float(val)
        if f == -1.0:
            return np.nan
        return f
    except ValueError:
        s = str(val).strip().upper()
        if s in ('NULL', 'NAN', ''):
            return np.nan
        return np.nan

def analyze_tsv(tsv_path, output_md_path):
    print(f"Reading TSV file from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    total_records = len(df)
    unique_species = df['Species'].nunique()
    unique_taxids = df['Species Taxonomy ID'].nunique()
    
    # 1. Clean numerical columns
    df['contig N50 (bp)'] = df['contig N50 (bp)'].apply(clean_numeric)
    df['scaffold N50 (bp)'] = df['scaffold N50 (bp)'].apply(clean_numeric)
    df['No. ancestral genes with intact ORF'] = df['No. ancestral genes with intact ORF'].apply(clean_numeric)
    df['No. ancestral genes with inactivating mutations'] = df['No. ancestral genes with inactivating mutations'].apply(clean_numeric)
    df['No. ancestral genes with missing sequences'] = df['No. ancestral genes with missing sequences'].apply(clean_numeric)
    
    # 2. Taxonomy Analysis
    # Extract lineage levels
    # Let's see the taxonomic lineage format: e.g. "Aves; Neognathae; Galloanserae; Anseriformes; Anatidae; Anatinae; Anas"
    # We can parse them into lists and count unique entries at different levels.
    def parse_lineage(x):
        if pd.isna(x):
            return []
        return [item.strip() for item in str(x).split(';') if item.strip()]
        
    lineages = df['Taxonomic Lineage'].apply(parse_lineage)
    
    # Let's find common ranks. Since these are all birds (Aves), let's extract:
    # Class (usually index 0, e.g. Aves)
    # Order (usually index 3, e.g. Anseriformes or Galliformes)
    # Family (usually index 4, e.g. Anatidae or Phasianidae)
    # Genus (usually the last or second to last element before species)
    
    orders = []
    families = []
    genera = []
    for lin in lineages:
        if len(lin) > 3:
            orders.append(lin[3])
        else:
            orders.append('Unknown')
            
        if len(lin) > 4:
            families.append(lin[4])
        else:
            families.append('Unknown')
            
        if len(lin) > 0:
            genera.append(lin[-1]) # Usually the genus is the last element in the lineage
        else:
            genera.append('Unknown')
            
    df['Taxon_Order'] = orders
    df['Taxon_Family'] = families
    df['Taxon_Genus'] = genera
    
    # 3. Assembly Status
    assembly_status_counts = df['NCBI assembly status'].value_counts(dropna=False)
    
    # 4. Assembly Submitter / Source
    submitter_counts = df['Assembly submitter organization'].value_counts().head(10)
    source_counts = df['Source of assembly'].value_counts(dropna=False)
    
    # 5. Quality Metrics Stats
    n50_stats = df[['contig N50 (bp)', 'scaffold N50 (bp)']].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    
    # 6. Ancestral Gene Projections
    gene_cols = [
        'No. ancestral genes with intact ORF',
        'No. ancestral genes with inactivating mutations',
        'No. ancestral genes with missing sequences'
    ]
    # Rows that actually have projection data (i.e. not NaN)
    valid_projections = df[gene_cols].dropna(how='all')
    num_valid_projections = len(valid_projections)
    gene_stats = df[gene_cols].describe()
    
    # 7. Date trends
    df['Assembly submission date'] = pd.to_datetime(df['Assembly submission date'], errors='coerce')
    df['Year'] = df['Assembly submission date'].dt.year
    year_counts = df['Year'].value_counts().sort_index()

    # Generate Markdown Report
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    
    with open(output_md_path, 'w') as f:
        f.write("# Genome Assemblies and Species Metadata Summary\n\n")
        f.write("This document summarizes the metadata for genomes and species available in `assemblies_and_species.tsv`.\n\n")
        
        f.write("## 1. Overall Summary\n")
        f.write(f"- **Total Records (rows):** {total_records}\n")
        f.write(f"- **Unique Species:** {unique_species}\n")
        f.write(f"- **Unique NCBI Accessions:** {df['NCBI accession'].nunique()}\n")
        f.write(f"- **Unique NCBI Assembly Names:** {df['NCBI assembly name'].nunique()}\n\n")
        
        f.write("## 2. Taxonomic Distribution\n")
        f.write("Top Orders, Families, and Genera in the dataset:\n\n")
        
        f.write("### Top Orders\n")
        order_counts = df['Taxon_Order'].value_counts()
        f.write("| Order | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for idx, val in order_counts.items():
            f.write(f"| {idx} | {val} | {val/total_records*100:.2f}% |\n")
        f.write("\n")
        
        f.write("### Top 10 Families\n")
        family_counts = df['Taxon_Family'].value_counts().head(10)
        f.write("| Family | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for idx, val in family_counts.items():
            f.write(f"| {idx} | {val} | {val/total_records*100:.2f}% |\n")
        f.write("\n")

        f.write("### Top 15 Genera\n")
        genus_counts = df['Taxon_Genus'].value_counts().head(15)
        f.write("| Genus | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for idx, val in genus_counts.items():
            f.write(f"| {idx} | {val} | {val/total_records*100:.2f}% |\n")
        f.write("\n")

        f.write("## 3. Assembly Level and Status\n")
        f.write("### NCBI Assembly Status\n")
        f.write("| Status | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for idx, val in assembly_status_counts.items():
            f.write(f"| {idx} | {val} | {val/total_records*100:.2f}% |\n")
        f.write("\n")

        f.write("### Source of Assembly\n")
        f.write("| Source | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for idx, val in source_counts.items():
            f.write(f"| {idx} | {val} | {val/total_records*100:.2f}% |\n")
        f.write("\n")
        
        f.write("### Top 10 Assembly Submitter Organizations\n")
        f.write("| Submitter Organization | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for idx, val in submitter_counts.items():
            f.write(f"| {idx} | {val} | {val/total_records*100:.2f}% |\n")
        f.write("\n")

        f.write("## 4. Assembly Quality Metrics\n")
        f.write("Summary statistics for Contig N50 and Scaffold N50 (in base pairs, excluding missing/NULL values):\n\n")
        f.write(f"- **Contig N50 (non-null entries):** {df['contig N50 (bp)'].notna().sum()}\n")
        f.write(f"- **Scaffold N50 (non-null entries):** {df['scaffold N50 (bp)'].notna().sum()}\n\n")
        
        f.write("| Metric | Contig N50 (bp) | Scaffold N50 (bp) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Min | {n50_stats.loc['min', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['min', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| 10% | {n50_stats.loc['10%', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['10%', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| 25% | {n50_stats.loc['25%', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['25%', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| 50% (Median) | {n50_stats.loc['50%', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['50%', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| Mean | {n50_stats.loc['mean', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['mean', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| 75% | {n50_stats.loc['75%', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['75%', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| 90% | {n50_stats.loc['90%', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['90%', 'scaffold N50 (bp)']:,.0f} |\n")
        f.write(f"| Max | {n50_stats.loc['max', 'contig N50 (bp)']:,.0f} | {n50_stats.loc['max', 'scaffold N50 (bp)']:,.0f} |\n\n")

        f.write("## 5. TOGA / Ancestral Gene Projections\n")
        f.write(f"Number of assemblies with valid gene projection data: **{num_valid_projections}** / {total_records} ({num_valid_projections/total_records*100:.2f}%)\n\n")
        
        if num_valid_projections > 0:
            f.write("| Projection Metric | Mean | Std Dev | Min | Median | Max |\n")
            f.write("|---|---|---|---|---|---|\n")
            for col in gene_cols:
                short_name = col.replace("No. ancestral genes with ", "")
                stats_col = df[col].describe()
                f.write(f"| {short_name} | {stats_col['mean']:.1f} | {stats_col['std']:.1f} | {stats_col['min']:.0f} | {stats_col['50%']:.0f} | {stats_col['max']:.0f} |\n")
            f.write("\n")
        else:
            f.write("No valid gene projection data found in the current sheet (all are -1 or NULL).\n\n")

        f.write("## 6. Submission Date Trends\n")
        f.write("Number of assemblies submitted by year:\n\n")
        f.write("| Year | Count | Cumulative |\n")
        f.write("|---|---|---|\n")
        cum = 0
        for yr, cnt in year_counts.items():
            if pd.isna(yr):
                continue
            cum += cnt
            f.write(f"| {int(yr)} | {cnt} | {cum} |\n")
        f.write("\n")
        
    print(f"Report written to {output_md_path}")

if __name__ == "__main__":
    import sys
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    tsv_file = os.path.join(base_dir, "assemblies_and_species.tsv")
    out_md = os.path.join(base_dir, "docs", "genome_metadata_summary.md")
    
    analyze_tsv(tsv_file, out_md)
