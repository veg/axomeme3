#!/usr/bin/env python3
"""
analyze_extreme_substitutions.py
--------------------------------
Queries the MEME database to find how many significant sites (p <= 0.05)
are supported by multi-nucleotide (2-nt or 3-nt) substitutions.
"""

import sqlite3
import os

def count_diffs(c1, c2):
    if len(c1) != 3 or len(c2) != 3:
        return 0
    return sum(1 for a, b in zip(c1, c2) if a != b)

def main():
    db_path = "/Users/sergei/Projects/TOGA_MEME/meme_results.db"
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}!")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Get significant sites
    cursor.execute("""
        SELECT gene_name, site_index, p_value 
        FROM site_results 
        WHERE is_significant = 1
    """)
    sig_sites = cursor.fetchall()
    sig_site_keys = {(r[0], r[1]): r[2] for r in sig_sites}
    print(f"Total significant sites (p <= 0.05) in DB: {len(sig_site_keys)}")
    
    # 2. Get substitutions
    cursor.execute("""
        SELECT gene_name, site_index, branch_name, ancestral_codon, derived_codon 
        FROM site_substitutions
    """)
    substitutions = cursor.fetchall()
    
    # Analyze substitutions
    total_substs = 0
    sig_site_substs = []
    
    snt_count = 0
    dnt_count = 0
    tnt_count = 0
    
    sig_snt_count = 0
    sig_dnt_count = 0
    sig_tnt_count = 0
    
    # Dictionary to collect diff types per significant site: (gene, site) -> list of diffs
    sig_site_diffs = {}
    
    for row in substitutions:
        gene, site, branch, anc, der = row
        diffs = count_diffs(anc, der)
        if diffs == 0:
            continue
            
        total_substs += 1
        if diffs == 1:
            snt_count += 1
        elif diffs == 2:
            dnt_count += 1
        elif diffs == 3:
            tnt_count += 1
            
        # Check if it maps to a significant site
        if (gene, site) in sig_site_keys:
            sig_site_substs.append(row)
            if (gene, site) not in sig_site_diffs:
                sig_site_diffs[(gene, site)] = []
            sig_site_diffs[(gene, site)].append(diffs)
            
            if diffs == 1:
                sig_snt_count += 1
            elif diffs == 2:
                sig_dnt_count += 1
            elif diffs == 3:
                sig_tnt_count += 1

    print("\n--- Substitution Mappings globally (all sites) ---")
    print(f"Total mapped substitutions: {total_substs}")
    if total_substs > 0:
        print(f"  Single nucleotide changes (SNT): {snt_count} ({snt_count/total_substs*100:.2f}%)")
        print(f"  Double nucleotide changes (DNT): {dnt_count} ({dnt_count/total_substs*100:.2f}%)")
        print(f"  Triple nucleotide changes (TNT): {tnt_count} ({tnt_count/total_substs*100:.2f}%)")
    
    print("\n--- Substitution Mappings at Significant Sites (p <= 0.05) ---")
    sig_total_substs = sig_snt_count + sig_dnt_count + sig_tnt_count
    print(f"Total substitutions at sig sites: {sig_total_substs}")
    if sig_total_substs > 0:
        print(f"  SNT: {sig_snt_count} ({sig_snt_count/sig_total_substs*100:.2f}%)")
        print(f"  DNT: {sig_dnt_count} ({sig_dnt_count/sig_total_substs*100:.2f}%)")
        print(f"  TNT: {sig_tnt_count} ({sig_tnt_count/sig_total_substs*100:.2f}%)")
        
    # Classify significant sites by supporting substitutions
    sites_with_dnt = 0
    sites_with_tnt = 0
    sites_only_dnt_or_tnt = 0
    sites_only_tnt = 0
    
    for (gene, site), diff_list in sig_site_diffs.items():
        has_snt = 1 in diff_list
        has_dnt = 2 in diff_list
        has_tnt = 3 in diff_list
        
        if has_dnt:
            sites_with_dnt += 1
        if has_tnt:
            sites_with_tnt += 1
        if (has_dnt or has_tnt) and not has_snt:
            sites_only_dnt_or_tnt += 1
        if has_tnt and not has_snt and not has_dnt:
            sites_only_tnt += 1
            
    print("\n--- Significant Sites Classification ---")
    num_sig_sites_with_substs = len(sig_site_diffs)
    print(f"Total significant sites with mapped substitutions: {num_sig_sites_with_substs} / {len(sig_site_keys)}")
    if num_sig_sites_with_substs > 0:
        print(f"  Sites containing at least one DNT (2-nt change): {sites_with_dnt} ({sites_with_dnt/num_sig_sites_with_substs*100:.2f}%)")
        print(f"  Sites containing at least one TNT (3-nt change): {sites_with_tnt} ({sites_with_tnt/num_sig_sites_with_substs*100:.2f}%)")
        print(f"  Sites supported ONLY by DNT/TNT (no 1-nt changes): {sites_only_dnt_or_tnt} ({sites_only_dnt_or_tnt/num_sig_sites_with_substs*100:.2f}%)")
        print(f"  Sites supported ONLY by TNT (no 1-nt or 2-nt changes): {sites_only_tnt} ({sites_only_tnt/num_sig_sites_with_substs*100:.2f}%)")
    
    conn.close()

if __name__ == '__main__':
    main()
