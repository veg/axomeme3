#!/usr/bin/env python3
"""
select_best_genomes.py
---------------------
Selects the "best" representative genome assembly per species from assemblies_and_species.tsv.
Creates:
1. species_to_assembly.tsv - Map from Species to its best assembly details.
2. assembly_to_species.tsv - Map from all NCBI Accessions / Directory Names to Species (and reverse).
3. problematic_genomes.tsv - List of flagged low-quality or potentially problematic genomes.

Saves these files to the root workspace and under the docs/ directory.
Logs progress and results in docs/genome_selection_report.md.
"""

import os
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
        return None

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    tsv_path = os.path.join(base_dir, "assemblies_and_species.tsv")
    
    print(f"Loading metadata from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # 1. Clean numeric fields
    df['contig_N50_clean'] = df['contig N50 (bp)'].apply(clean_numeric)
    df['scaffold_N50_clean'] = df['scaffold N50 (bp)'].apply(clean_numeric)
    df['intact_clean'] = df['No. ancestral genes with intact ORF'].apply(clean_numeric)
    df['mut_clean'] = df['No. ancestral genes with inactivating mutations'].apply(clean_numeric)
    df['miss_clean'] = df['No. ancestral genes with missing sequences'].apply(clean_numeric)
    
    # 2. Assign Assembly Status Score
    # Chromosome/Complete Genome = 3, Scaffold = 2, Contig = 1, nan/other = 0
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

    # 3. Identify problematic genomes
    # We flag genomes based on multiple quality criteria
    problematic_flags = []
    
    for idx, row in df.iterrows():
        reasons = []
        status = str(row['NCBI assembly status']).strip() if pd.notna(row['NCBI assembly status']) else 'Unknown'
        c_n50 = row['contig_N50_clean']
        s_n50 = row['scaffold_N50_clean']
        intact = row['intact_clean']
        miss = row['miss_clean']
        mut = row['mut_clean']
        
        # Criteria 1: Contig status
        if status.lower() == 'contig':
            reasons.append("Assembly status is Contig")
            
        # Criteria 2: Extremely low Scaffold N50 (< 100 KB)
        if s_n50 is not None and s_n50 < 100000:
            reasons.append("Low Scaffold N50 (< 100 KB)")
            
        # Criteria 3: Extremely low Contig N50 (< 10 KB)
        if c_n50 is not None and c_n50 < 10000:
            reasons.append("Low Contig N50 (< 10 KB)")
            
        # Criteria 4: Poor TOGA projections
        if intact is not None:
            if intact < 8000:
                reasons.append("Low intact ORFs (< 8,000)")
            if miss is not None and miss > 2000:
                reasons.append("High missing sequences (> 2,000)")
            if mut is not None and mut > 2000:
                reasons.append("High inactivating mutations (> 2,000)")
                
        if reasons:
            problematic_flags.append({
                'Directory Name': row['Directory Name'],
                'Species': row['Species'],
                'NCBI accession': row['NCBI accession'],
                'NCBI assembly status': row['NCBI assembly status'],
                'contig N50 (bp)': row['contig N50 (bp)'],
                'scaffold N50 (bp)': row['scaffold N50 (bp)'],
                'No. ancestral genes with intact ORF': row['No. ancestral genes with intact ORF'],
                'No. ancestral genes with inactivating mutations': row['No. ancestral genes with inactivating mutations'],
                'No. ancestral genes with missing sequences': row['No. ancestral genes with missing sequences'],
                'Flag Reason': " | ".join(reasons)
            })
            
    prob_df = pd.DataFrame(problematic_flags)
    
    # 4. Select the "best" genome per species
    # We sort by:
    #   1. Status Score (descending)
    #   2. Scaffold N50 (descending)
    #   3. Contig N50 (descending)
    #   4. Intact ORFs (descending)
    #   5. Submission Date (descending)
    # Then we group by Species and take the first record.
    
    # Fill NAs for sorting purposes
    df['sort_s_n50'] = df['scaffold_N50_clean'].fillna(0)
    df['sort_c_n50'] = df['contig_N50_clean'].fillna(0)
    df['sort_intact'] = df['intact_clean'].fillna(0)
    df['sort_date'] = df['submission_date_dt'].fillna(pd.Timestamp('1900-01-01'))
    
    sorted_df = df.sort_values(
        by=['status_score', 'sort_s_n50', 'sort_c_n50', 'sort_intact', 'sort_date'],
        ascending=[False, False, False, False, False]
    )
    
    best_df = sorted_df.groupby('Species').first().reset_index()
    
    # 5. Create mapping files
    # Map 1: Species to Assembly Map
    species_to_assembly = best_df[[
        'Species', 'Common name', 'Directory Name', 'NCBI accession', 
        'NCBI assembly status', 'contig_N50_clean', 'scaffold_N50_clean', 'intact_clean'
    ]].copy()
    species_to_assembly.columns = [
        'Species', 'Common_Name', 'Directory_Name', 'NCBI_Accession', 
        'Assembly_Status', 'Contig_N50', 'Scaffold_N50', 'Intact_ORFs'
    ]
    
    # Map 2: Assembly to Species Map (containing all assemblies in the dataset)
    assembly_to_species = df[[
        'NCBI accession', 'Directory Name', 'Species', 'Common name'
    ]].copy()
    assembly_to_species.columns = [
        'NCBI_Accession', 'Directory_Name', 'Species', 'Common_Name'
    ]
    
    # 6. Save outputs
    # Save TSV files to workspace root
    species_to_assembly.to_csv(os.path.join(base_dir, "species_to_assembly.tsv"), sep='\t', index=False)
    assembly_to_species.to_csv(os.path.join(base_dir, "assembly_to_species.tsv"), sep='\t', index=False)
    prob_df.to_csv(os.path.join(base_dir, "problematic_genomes.tsv"), sep='\t', index=False)
    
    # Save TSV files to docs/ directory
    os.makedirs(os.path.join(base_dir, "docs"), exist_ok=True)
    species_to_assembly.to_csv(os.path.join(base_dir, "docs", "species_to_assembly.tsv"), sep='\t', index=False)
    assembly_to_species.to_csv(os.path.join(base_dir, "docs", "assembly_to_species.tsv"), sep='\t', index=False)
    prob_df.to_csv(os.path.join(base_dir, "docs", "problematic_genomes.tsv"), sep='\t', index=False)
    
    # 7. Generate markdown selection report
    report_path = os.path.join(base_dir, "docs", "genome_selection_report.md")
    
    total_genomes = len(df)
    unique_species = len(best_df)
    problematic_count = len(prob_df)
    
    # Group problematic reasons
    with open(report_path, 'w') as f:
        f.write("# Genome Assembly Selection and Quality Report\n\n")
        f.write("This report details the results of selecting the 'best' genome assembly per species and identifying potentially problematic assemblies.\n\n")
        
        f.write("## 1. Summary Statistics\n")
        f.write(f"- **Total assemblies examined:** {total_genomes}\n")
        f.write(f"- **Unique species represented:** {unique_species}\n")
        f.write(f"- **Assemblies selected as 'Best' representation:** {unique_species} (1 per species)\n")
        f.write(f"- **Potentially problematic genomes flagged:** {problematic_count} ({problematic_count/total_genomes*100:.2f}% of total)\n\n")
        
        f.write("## 2. Selection Hierarchy for 'Best' Genome\n")
        f.write("When a species had multiple assemblies, the best assembly was selected using the following hierarchical criteria:\n")
        f.write("1. **Assembly Level (Status):** `Chromosome` / `Complete Genome` > `Scaffold` > `Contig` > `Unknown`.\n")
        f.write("2. **Scaffold N50:** Higher value (base pairs) is preferred.\n")
        f.write("3. **Contig N50:** Higher value (base pairs) is preferred.\n")
        f.write("4. **TOGA Projections:** Higher count of intact ORFs (if available).\n")
        f.write("5. **Submission Date:** Newer submission dates are used as final tiebreakers.\n\n")
        
        f.write("## 3. Problematic Genome Criteria and Flags\n")
        f.write("Genomes were flagged as **potentially problematic** if they met any of the following quality thresholds:\n")
        f.write("- **Assembly Status:** Assembly status is explicitly `Contig` (indicating lack of scaffolding).\n")
        f.write("- **Scaffold N50:** Scaffold N50 is less than **100,000 bp** (100 KB).\n")
        f.write("- **Contig N50:** Contig N50 is less than **10,000 bp** (10 KB).\n")
        f.write("- **TOGA Projection Quality (if available):**\n")
        f.write("  - Intact ORFs < **8,000** (highly incomplete gene set).\n")
        f.write("  - Missing sequence count > **2,000** (significant coding region gaps).\n")
        f.write("  - Inactivating mutation count > **2,000** (potential frameshift/indels in coding sequences).\n\n")
        
        f.write("### Summary of Flagged Reasons\n")
        # Breakdown of reasons
        reasons_list = []
        for r in prob_df['Flag Reason']:
            if pd.notna(r):
                reasons_list.extend([x.strip() for x in r.split('|') if x.strip()])
        
        reason_counts = pd.Series(reasons_list).value_counts()
        
        f.write("| Flagged Reason | Number of Genomes | Percentage of Total Flagged |\n")
        f.write("|---|---|---|\n")
        for reason, cnt in reason_counts.items():
            f.write(f"| {reason} | {cnt} | {cnt/problematic_count*100:.2f}% |\n")
        f.write("\n")
        
        f.write("### Top 15 Flagged Problematic Genomes\n")
        f.write("| Directory Name | Species | Status | Scaffold N50 | Intact ORFs | Primary Reason |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in prob_df.head(15).iterrows():
            f.write(f"| {row['Directory Name']} | {row['Species']} | {row['NCBI assembly status']} | {row['scaffold N50 (bp)']} | {row['No. ancestral genes with intact ORF']} | {row['Flag Reason']} |\n")
        f.write("\n")
        
        f.write("## 4. Mapping Files Outputs\n")
        f.write("The following mapping files have been generated:\n")
        f.write("1. **Species to Best Assembly Map:** `species_to_assembly.tsv` (also saved in `docs/`)\n")
        f.write("2. **Assembly to Species Map (both directions):** `assembly_to_species.tsv` (also saved in `docs/`)\n")
        f.write("3. **Problematic Genomes List:** `problematic_genomes.tsv` (also saved in `docs/`)\n")

    print("Genome selection and classification complete!")

if __name__ == "__main__":
    main()
