#!/usr/bin/env python3
import sqlite3
import os

DB_PATH = "meme_results.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        return
        
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get total sites count
    cursor.execute("SELECT COUNT(*) FROM site_results")
    total_sites = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM site_results WHERE is_significant=1")
    old_sig_count = cursor.fetchone()[0]
    print(f"Total sites: {total_sites}")
    print(f"Significant sites at current threshold: {old_sig_count}")
    
    # 1. Update is_significant in site_results based on q_value <= 0.10
    print("Updating is_significant column based on q_value <= 0.10...")
    cursor.execute("UPDATE site_results SET is_significant = CASE WHEN q_value <= 0.10 THEN 1 ELSE 0 END")
    
    # 2. Reset classification, filter reason, and selection type for non-significant sites (q_value > 0.10)
    print("Resetting classification and filter_reason for non-significant sites...")
    cursor.execute("UPDATE site_results SET classification = 'NONE', filter_reason = NULL, selection_type = NULL WHERE is_significant = 0")
    
    # 3. Update num_selected_sites in gene_results
    print("Updating num_selected_sites in gene_results table...")
    cursor.execute("""
    UPDATE gene_results SET num_selected_sites = (
        SELECT COUNT(*) FROM site_results 
        WHERE site_results.gene_name = gene_results.gene_name AND site_results.is_significant = 1
    )
    """)
    
    conn.commit()
    
    # Verify new counts
    cursor.execute("SELECT COUNT(*) FROM site_results WHERE is_significant=1")
    new_sig_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT classification, COUNT(*) FROM site_results WHERE is_significant=1 GROUP BY classification")
    class_counts = cursor.fetchall()
    
    print(f"\nSignificance update completed successfully!")
    print(f"Significant sites at new threshold (q <= 0.10): {new_sig_count} (out of {total_sites})")
    print("Breakdown by support category:")
    for cls, count in class_counts:
        print(f"  {cls}: {count}")
        
    conn.close()

if __name__ == "__main__":
    main()
