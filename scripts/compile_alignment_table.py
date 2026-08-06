#!/usr/bin/env python3
import os
import re
import gzip
import sqlite3
import json
import multiprocessing as mp

def process_gene(row):
    gene_name, file_name, transcript_id, orig_len, orig_seqs, is_outlier, is_pseudogene, msa_dir = row
    
    status = "Kept"
    retained_seqs = 0
    retained_codons = 0
    
    if is_outlier:
        status = "Filtered (Outlier)"
    elif is_pseudogene:
        status = "Filtered (Pseudogene)"
    else:
        curated_name = f"{gene_name}.gz"
        file_path = os.path.join(msa_dir, curated_name)
        
        if os.path.exists(file_path):
            try:
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline()
                    if first_line.startswith("#NEXUS"):
                        ntax = 0
                        nchar = 0
                        for line in f:
                            m_ntax = re.search(r'ntax\s*=\s*(\d+)', line, re.I)
                            m_nchar = re.search(r'nchar\s*=\s*(\d+)', line, re.I)
                            if m_ntax:
                                ntax = int(m_ntax.group(1))
                            if m_nchar:
                                nchar = int(m_nchar.group(1))
                            if ntax > 0 and nchar > 0:
                                break
                        retained_seqs = ntax
                        retained_codons = nchar // 3
                    else:
                        seq_count = 0
                        first_seq_len = 0
                        current_seq = []
                        
                        def process_fasta_line(line):
                            nonlocal seq_count, first_seq_len, current_seq
                            line = line.strip()
                            if line.startswith('>'):
                                seq_count += 1
                                if seq_count == 2:
                                    first_seq_len = len("".join(current_seq))
                                current_seq = []
                            elif seq_count == 1:
                                current_seq.append(line)
                                
                        process_fasta_line(first_line)
                        for line in f:
                            process_fasta_line(line)
                        if seq_count == 1:
                            first_seq_len = len("".join(current_seq))
                            
                        retained_seqs = seq_count
                        retained_codons = first_seq_len // 3
                        
                if retained_seqs < 3:
                    status = "Filtered (Too few sequences)"
            except Exception as e:
                status = f"Error: {e}"
        else:
            status = "Filtered (Too few sequences)"
            
    return {
        'gene': gene_name,
        'transcript': transcript_id,
        'status': status,
        'orig_seqs': orig_seqs,
        'retained_seqs': retained_seqs,
        'orig_codons': orig_len // 3,
        'retained_codons': retained_codons
    }

def main():
    db_path = "/Users/sergei/Projects/TOGA_MEME/alignments_stats.db"
    msa_dir = "/Users/sergei/Projects/TOGA_MEME/msa"
    output_path = "/Users/sergei/Projects/TOGA_MEME/scratch/alignment_gene_table.json"
    
    if not os.path.exists(db_path):
        # Local fallback if run on local host
        db_path = "alignments_stats.db"
        msa_dir = "/Users/sergei/Projects/TOGA_MEME/msa" # Keep remote/local mapping
        if not os.path.exists(msa_dir):
            msa_dir = "msa"
            
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT gene_name, file_name, transcript_id, alignment_length, num_sequences, 
               is_outlier, is_pseudogene
        FROM alignments
    """)
    rows = c.fetchall()
    conn.close()
    
    print(f"Loaded {len(rows)} alignments from database. Processing details...")
    
    # Prepare task arguments
    tasks = []
    for r in rows:
        tasks.append((r[0], r[1], r[2], r[3], r[4], bool(r[5]), bool(r[6]), msa_dir))
        
    # Process in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_gene, tasks)
        
    # Write output to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
        
    print(f"Compiled stats for {len(results)} genes and saved to: {output_path}")

if __name__ == "__main__":
    main()
