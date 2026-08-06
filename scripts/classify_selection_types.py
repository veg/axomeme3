#!/usr/bin/env python3
import sqlite3
import os
import sys

DB_PATH = "meme_results.db"

def main():
    print("Classifying selection types for GOLD and SILVER sites...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Add selection_type column to site_results if it doesn't exist
    try:
        c.execute("ALTER TABLE site_results ADD COLUMN selection_type TEXT")
        conn.commit()
        print("Added column 'selection_type' to site_results table.")
    except sqlite3.OperationalError:
        # Already exists
        print("Column 'selection_type' already exists.")
        
    # 2. Fetch all significant GOLD/SILVER sites
    c.execute("""
        SELECT gene_name, site_index 
        FROM site_results 
        WHERE is_significant = 1 AND classification IN ('GOLD', 'SILVER')
    """)
    sig_sites = c.fetchall()
    print(f"Found {len(sig_sites)} GOLD/SILVER sites to classify.")
    
    # 3. For each site, fetch all non-synonymous substitutions
    updates = []
    category_counts = {
        "LINEAGE_SPECIFIC": 0,
        "PERVASIVE": 0,
        "CLADE_RESTRICTED": 0,
        "DEEP_ANCESTRAL": 0,
        "MIXED": 0
    }
    
    for idx, (gene, site) in enumerate(sig_sites):
        c.execute("""
            SELECT mn.is_leaf, mn.descendants
            FROM site_substitutions s
            JOIN branch_mappings bm ON s.gene_name = bm.gene_name AND s.branch_name = bm.branch_name
            JOIN master_nodes mn ON bm.master_node_id = mn.node_id
            WHERE s.gene_name = ? AND s.site_index = ?
              AND s.is_synonymous = 0
        """, (gene, site))
        
        subs = c.fetchall()
        if not subs:
            # No non-synonymous substitutions recorded, classify as MIXED or OTHER
            sel_type = "MIXED"
            updates.append((sel_type, gene, site))
            category_counts[sel_type] += 1
            continue
            
        total_subs = len(subs)
        leaves = [s for s in subs if s[0] == 1]
        internals = [s for s in subs if s[0] == 0]
        
        num_leaves = len(leaves)
        num_internals = len(internals)
        
        # Parse sizes of clades (leaf count for each branch)
        internal_sizes = [len(s[1].split(',')) for s in internals]
        
        # Classification Logic:
        if total_subs <= 3 and num_leaves == total_subs:
            # Very few substitutions, all on individual terminal species
            sel_type = "LINEAGE_SPECIFIC"
        elif total_subs >= 7 and (num_leaves / total_subs) >= 0.40:
            # Many substitutions, heavily scattered across terminal species
            sel_type = "PERVASIVE"
        elif num_internals >= 1 and all(size <= 15 for size in internal_sizes) and (num_leaves == 0 or all(1 <= 15 for _ in range(num_leaves))):
            # All substitutions are restricted to small clades or leaves (<= 15 leaves)
            sel_type = "CLADE_RESTRICTED"
        elif num_internals >= 1 and sum(1 for size in internal_sizes if size > 15) / total_subs >= 0.50:
            # Most substitutions occur on deep ancestral nodes
            sel_type = "DEEP_ANCESTRAL"
        else:
            # Doesn't fit cleanly
            sel_type = "MIXED"
            
        updates.append((sel_type, gene, site))
        category_counts[sel_type] += 1
        
    # 4. Perform batch update in database
    c.executemany("""
        UPDATE site_results 
        SET selection_type = ? 
        WHERE gene_name = ? AND site_index = ?
    """, updates)
    
    # Create index for selection_type
    c.execute("CREATE INDEX IF NOT EXISTS idx_site_sel_type ON site_results(selection_type)")
    
    conn.commit()
    conn.close()
    
    print("\nClassification summary:")
    for cat, count in category_counts.items():
        pct = (count / len(sig_sites)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    print("\nDatabase updated successfully.")

if __name__ == "__main__":
    main()
