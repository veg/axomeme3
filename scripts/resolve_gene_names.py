#!/usr/bin/env python3
"""
resolve_gene_names.py
---------------------
1. Alters the alignments table in alignments_stats.db to add curation/mapping columns:
   - resolved_gene_name TEXT
   - biotype TEXT
   - is_pseudogene INTEGER
   - is_sus_gene INTEGER
   - sus_reasons TEXT
2. Fixes the row for the non-standard filename ENST00000285879.5#MAGEC1_guided_rerun.bes.fas.codonified.fa.
3. Identifies all ENSG gene names in the database and queries Ensembl REST API using 
   POST request to resolve them to HUGO/HGNC symbols, biotypes, and descriptions in bulk.
4. Identifies "sus" genes based on quantitative alignment properties:
   - Reference sequence (hg38) contains frameshifts, premature stops, or low coverage.
   - High proportion of species sequences in the alignment are suspect (> 50%).
   - Ensembl biotype indicates it is a pseudogene.
"""

import os
import json
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime

DB_PATH = "/Users/sergei/Dropbox/TOGA2026/alignments_stats.db"

def main():
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Alter table to add columns if they do not exist
    cursor.execute("PRAGMA table_info(alignments)")
    columns = [col[1] for col in cursor.fetchall()]
    
    new_cols = {
        'resolved_gene_name': 'TEXT',
        'biotype': 'TEXT',
        'is_pseudogene': 'INTEGER DEFAULT 0',
        'is_sus_gene': 'INTEGER DEFAULT 0',
        'sus_reasons': 'TEXT'
    }
    
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            print(f"Adding column {col_name} ({col_type}) to alignments table...")
            cursor.execute(f"ALTER TABLE alignments ADD COLUMN {col_name} {col_type}")
            conn.commit()
            
    # Initialize resolved_gene_name = gene_name
    print("Initializing resolved_gene_name values...")
    cursor.execute("UPDATE alignments SET resolved_gene_name = gene_name WHERE resolved_gene_name IS NULL")
    conn.commit()
    
    # 2. Fix the MAGEC1 guided rerun filename row
    # The original parser set transcript_id = "Unknown" and gene_name = "ENST00000285879"
    # because of the multiple dots. Let's fix this in the database.
    print("Fixing MAGEC1 guided rerun record...")
    cursor.execute("""
        UPDATE alignments
        SET transcript_id = 'ENST00000285879.5',
            gene_name = 'MAGEC1',
            resolved_gene_name = 'MAGEC1',
            biotype = 'protein_coding'
        WHERE file_name = 'ENST00000285879.5#MAGEC1_guided_rerun.bes.fas.codonified.fa'
    """)
    conn.commit()
    
    # 3. Find and resolve ENSG IDs
    cursor.execute("SELECT DISTINCT gene_name FROM alignments WHERE gene_name LIKE 'ENSG%'")
    ensg_ids = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(ensg_ids)} unique Ensembl Gene IDs to resolve.")
    
    ensg_mappings = {}
    if ensg_ids:
        # Ensembl REST API lookup POST
        url = "https://rest.ensembl.org/lookup/id"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        body = json.dumps({"ids": ensg_ids})
        req = urllib.request.Request(url, data=body.encode('utf-8'), headers=headers, method="POST")
        
        try:
            print("Querying Ensembl REST API for resolved gene names and biotypes...")
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                for ensg_id, data in result.items():
                    if data:
                        display_name = data.get("display_name", ensg_id)
                        biotype = data.get("biotype", "Unknown")
                        description = data.get("description", "")
                        ensg_mappings[ensg_id] = {
                            'resolved_name': display_name,
                            'biotype': biotype,
                            'description': description
                        }
        except Exception as e:
            print(f"Error calling Ensembl API: {e}")
            
    # Update the resolved names and biotypes in the database
    print("Updating database with resolved Ensembl mappings...")
    for ensg_id, mapped in ensg_mappings.items():
        is_pseudo = 1 if 'pseudogene' in mapped['biotype'].lower() else 0
        cursor.execute("""
            UPDATE alignments
            SET resolved_gene_name = ?,
                biotype = ?,
                is_pseudogene = ?
            WHERE gene_name = ?
        """, (mapped['resolved_name'], mapped['biotype'], is_pseudo, ensg_id))
    conn.commit()
    
    # 4. Identify "sus" genes based on alignment metrics
    print("Evaluating alignments for quality and suspicious status...")
    cursor.execute("SELECT id, file_name, num_sequences, num_sus_sequences, gene_name, biotype FROM alignments")
    alignments = cursor.fetchall()
    
    # Identify alignments where the reference (hg38) sequence is suspect
    cursor.execute("""
        SELECT alignment_id, sus_reasons 
        FROM sequences 
        WHERE sequence_name = 'hg38' AND is_sus = 1
    """)
    hg38_sus = {row[0]: row[1] for row in cursor.fetchall()}
    
    updates = []
    sus_count = 0
    pseudo_count = 0
    
    for al_id, f_name, num_seqs, num_sus_seqs, g_name, biotype in alignments:
        reasons = []
        is_pseudo = 0
        
        # Check biotype if it is already set as pseudogene
        if biotype and 'pseudogene' in biotype.lower():
            is_pseudo = 1
            reasons.append(f"Ensembl biotype: {biotype}")
            
        # Check if reference sequence is suspect
        if al_id in hg38_sus:
            reasons.append(f"Human reference (hg38) is suspect: {hg38_sus[al_id]}")
            
        # Check if more than 50% of sequences in the alignment are suspect
        sus_pct = (num_sus_seqs / num_seqs) * 100 if num_seqs > 0 else 0
        if sus_pct > 50:
            reasons.append(f"High species sus rate: {sus_pct:.1f}% ({num_sus_seqs}/{num_seqs} sequences)")
            
        is_sus_gene = 1 if reasons else 0
        if is_sus_gene:
            sus_count += 1
            if is_pseudo == 0 and ("pseudogene" in g_name.lower() or (biotype and "pseudogene" in biotype.lower())):
                is_pseudo = 1
            
            if is_pseudo:
                pseudo_count += 1
                
            reasons_str = " | ".join(reasons)
        else:
            reasons_str = None
            
        updates.append((is_pseudo, is_sus_gene, reasons_str, al_id))
        
    print(f"Updating {len(updates)} alignments with sus/pseudogene status...")
    cursor.executemany("""
        UPDATE alignments
        SET is_pseudogene = ?,
            is_sus_gene = ?,
            sus_reasons = ?
        WHERE id = ?
    """, updates)
    conn.commit()
    
    print(f"Curation complete! Found {sus_count} suspicious genes, including {pseudo_count} identified as pseudogenes.")
    conn.close()

if __name__ == "__main__":
    main()
