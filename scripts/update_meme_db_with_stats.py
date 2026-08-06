#!/usr/bin/env python3
import json
import sqlite3
import os

DB_PATH = "meme_results.db"
STATS_PATH = "gene_stats.json"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        return
        
    if not os.path.exists(STATS_PATH):
        print(f"Error: Stats file {STATS_PATH} not found!")
        return
        
    print(f"Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Add columns if they do not exist
    cursor.execute("PRAGMA table_info(gene_results)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "global_dnds" not in columns:
        print("Adding global_dnds column to gene_results...")
        cursor.execute("ALTER TABLE gene_results ADD COLUMN global_dnds REAL")
        
    if "num_branch_outliers" not in columns:
        print("Adding num_branch_outliers column to gene_results...")
        cursor.execute("ALTER TABLE gene_results ADD COLUMN num_branch_outliers INTEGER DEFAULT 0")
        
    if "outlier_branches" not in columns:
        print("Adding outlier_branches column to gene_results...")
        cursor.execute("ALTER TABLE gene_results ADD COLUMN outlier_branches TEXT")
        
    if "outlier_threshold" not in columns:
        print("Adding outlier_threshold column to gene_results...")
        cursor.execute("ALTER TABLE gene_results ADD COLUMN outlier_threshold REAL")
        
    if "branch_lengths" not in columns:
        print("Adding branch_lengths column to gene_results...")
        cursor.execute("ALTER TABLE gene_results ADD COLUMN branch_lengths TEXT")
        
    conn.commit()
    
    # 2. Update database values
    print("Updating gene results in database...")
    cursor.execute("SELECT gene_name FROM gene_results")
    db_genes = [r[0] for r in cursor.fetchall()]
    
    updated_count = 0
    for gene in db_genes:
        g_stats = stats.get(gene)
        if not g_stats:
            for k, v in stats.items():
                if k.upper() == gene.upper():
                    g_stats = v
                    break
                    
        if g_stats:
            global_dnds = g_stats.get("global_dnds")
            num_outliers = g_stats.get("num_outliers", 0)
            outliers_list = g_stats.get("outlier_branches", [])
            outliers_str = ", ".join(outliers_list) if outliers_list else None
            threshold = g_stats.get("outlier_threshold")
            
            lengths = g_stats.get("branch_lengths", [])
            lengths_str = ",".join(map(str, lengths)) if lengths else None
            
            cursor.execute("""
                UPDATE gene_results
                SET global_dnds = ?, num_branch_outliers = ?, outlier_branches = ?,
                    outlier_threshold = ?, branch_lengths = ?
                WHERE gene_name = ?
            """, (global_dnds, num_outliers, outliers_str, threshold, lengths_str, gene))
            updated_count += 1
            
    conn.commit()
    conn.close()
    
    print(f"Successfully updated {updated_count} genes with global dN/dS, branch outlier stats, thresholds, and branch lengths!")

if __name__ == "__main__":
    main()
