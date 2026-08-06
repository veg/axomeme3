#!/usr/bin/env python3
"""
build_alignment_db.py
---------------------
Parses all 19,953 codonified FASTA alignments in individualAlis/ and builds a SQLite
database (alignments_stats.db) to track sequence quality, stop codons, resolved length,
and identify potentially problematic/pseudogene alignments.
Uses multiprocessing and chunked SQLite insertions for maximum performance.
"""

import os
import re
import sqlite3
import multiprocessing as mp
from datetime import datetime

# Define stop codons (standard genetic code)
STOP_CODONS = {'TAA', 'TAG', 'TGA'}

def clean_numeric(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0

def parse_fasta_file(file_path):
    """Parses a single FASTA file and returns its raw sequences."""
    sequences = []
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
                        sequences.append((current_name, "".join(current_seq)))
                    current_name = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_name:
                sequences.append((current_name, "".join(current_seq)))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return sequences

def analyze_single_file(args):
    """Worker function to analyze a single alignment file in alignment space."""
    file_path, file_name = args
    
    # Parse transcript ID and gene name from filename
    match = re.match(r'([^#]+)#([^.]+)\.codonified\.fa', file_name)
    if match:
        transcript_id, gene_name = match.group(1), match.group(2)
    else:
        # Fallback for non-standard filenames (like MAGEC1 guided rerun)
        if "#" in file_name:
            parts = file_name.split("#")
            transcript_id = parts[0]
            gene_name = parts[1].split(".")[0]
        else:
            transcript_id, gene_name = "Unknown", file_name.split('.')[0]

    raw_seqs = parse_fasta_file(file_path)
    if not raw_seqs:
        return None
        
    alignment_length = len(raw_seqs[0][1])
    
    # Find reference sequence (hg38 is the reference template)
    ref_seq = None
    for name, seq in raw_seqs:
        if name == 'hg38':
            ref_seq = seq
            break
    if ref_seq is None:
        ref_seq = raw_seqs[0][1] # fallback to first sequence
        
    seqs_stats = []
    num_sus = 0
    
    # Helper to find gap run lengths
    def get_non_triple_gap_runs(s):
        # Find contiguous runs of dashes
        import re
        runs = re.findall(r'-+', s)
        return [len(r) for r in runs if len(r) % 3 != 0]

    for seq_name, seq_content in raw_seqs:
        seq_upper = seq_content.upper()
        
        # Calculate base counts in alignment space
        a_cnt = seq_upper.count('A')
        c_cnt = seq_upper.count('C')
        g_cnt = seq_upper.count('G')
        t_cnt = seq_upper.count('T')
        resolved_len = a_cnt + c_cnt + g_cnt + t_cnt
        n_cnt = seq_upper.count('N')
        gap_cnt = seq_upper.count('-')
        
        # 1. Frameshift detection in alignment space
        # Check for non-triple gap runs in sequence (deletions) and in reference (insertions)
        query_gap_runs = get_non_triple_gap_runs(seq_upper)
        ref_gap_runs = get_non_triple_gap_runs(ref_seq.upper()) if seq_name != 'hg38' else []
        
        has_frameshift = 1 if (query_gap_runs or ref_gap_runs) else 0
        
        # 2. Premature stop codon detection in alignment space
        # Split sequence into triplets (codons) based on alignment columns
        codons = [seq_upper[i:i+3] for i in range(0, len(seq_upper), 3)]
        stops = []
        for idx, codon in enumerate(codons):
            # A stop codon must have no gaps, no Ns, and be in STOP_CODONS
            if '-' not in codon and 'N' not in codon and codon in STOP_CODONS:
                stops.append(idx)
                
        # Find the last non-gap codon (codon index containing at least one non-gap base)
        last_codon_idx = -1
        for idx in range(len(codons) - 1, -1, -1):
            if any(c != '-' for c in codons[idx]):
                last_codon_idx = idx
                break
                
        has_premature_stop = 1 if any(idx < last_codon_idx for idx in stops) else 0
        stop_codon_count = len(stops)
        
        # Curation warnings
        reasons = []
        if has_premature_stop:
            reasons.append("Premature stop codon")
        if resolved_len == 0:
            reasons.append("Empty sequence (all gaps/Ns)")
        elif (resolved_len / alignment_length) < 0.3:
            reasons.append(f"Low coverage ({resolved_len / alignment_length * 100:.1f}%)")
            
        is_sus = 1 if reasons else 0
        if is_sus:
            num_sus += 1
            
        seqs_stats.append({
            'name': seq_name,
            'resolved_length': resolved_len,
            'gap_count': gap_cnt,
            'n_count': n_cnt,
            'stop_codon_count': stop_codon_count,
            'has_premature_stop': has_premature_stop,
            'is_sus': is_sus,
            'sus_reasons': " | ".join(reasons)
        })
        
    return {
        'file_name': file_name,
        'transcript_id': transcript_id,
        'gene_name': gene_name,
        'alignment_length': alignment_length,
        'num_sequences': len(raw_seqs),
        'num_sus_sequences': num_sus,
        'sequences': seqs_stats
    }

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    alis_dir = os.path.join(base_dir, "individualAlis")
    db_path = os.path.join(base_dir, "alignments_stats.db")
    
    # Remove existing db if running fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"Connecting to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT UNIQUE,
        transcript_id TEXT,
        gene_name TEXT,
        alignment_length INTEGER,
        num_sequences INTEGER,
        num_sus_sequences INTEGER
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sequences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alignment_id INTEGER,
        sequence_name TEXT,
        resolved_length INTEGER,
        gap_count INTEGER,
        n_count INTEGER,
        stop_codon_count INTEGER,
        has_premature_stop INTEGER,
        is_sus INTEGER,
        sus_reasons TEXT,
        FOREIGN KEY(alignment_id) REFERENCES alignments(id)
    )""")
    conn.commit()
    
    # Scan files
    print(f"Scanning files in {alis_dir}...")
    all_files = [f for f in os.listdir(alis_dir) if f.endswith('.fa')]
    all_files.sort()
    total_files = len(all_files)
    print(f"Found {total_files} codonified alignments to process.")
    
    # Process in chunks of 500 files to manage memory and maximize write performance
    chunk_size = 500
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    # Multiprocessing pool
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} parallel workers.")
    pool = mp.Pool(num_workers)
    
    start_time = datetime.now()
    processed_count = 0
    
    for chunk_idx, chunk in enumerate(file_chunks):
        chunk_start = datetime.now()
        args_list = []
        for f_name in chunk:
            args_list.append((os.path.join(alis_dir, f_name), f_name))
            
        # Run parallel analysis
        results = pool.map(analyze_single_file, args_list)
        results = [r for r in results if r is not None]
        
        # Insert results inside a single transaction
        cursor.execute("BEGIN TRANSACTION")
        for r in results:
            cursor.execute("""
            INSERT INTO alignments (file_name, transcript_id, gene_name, alignment_length, num_sequences, num_sus_sequences)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (r['file_name'], r['transcript_id'], r['gene_name'], r['alignment_length'], r['num_sequences'], r['num_sus_sequences']))
            
            alignment_id = cursor.lastrowid
            
            # Prepare sequence bulk inserts
            seq_tuples = []
            for s in r['sequences']:
                seq_tuples.append((
                    alignment_id,
                    s['name'],
                    s['resolved_length'],
                    s['gap_count'],
                    s['n_count'],
                    s['stop_codon_count'],
                    s['has_premature_stop'],
                    s['is_sus'],
                    s['sus_reasons']
                ))
                
            cursor.executemany("""
            INSERT INTO sequences (alignment_id, sequence_name, resolved_length, gap_count, n_count, stop_codon_count, has_premature_stop, is_sus, sus_reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, seq_tuples)
            
        conn.commit()
        processed_count += len(results)
        
        chunk_end = datetime.now()
        chunk_duration = (chunk_end - chunk_start).total_seconds()
        pct_done = (processed_count / total_files) * 100
        print(f"Processed batch {chunk_idx+1}/{len(file_chunks)}: {processed_count} files done ({pct_done:.1f}%). Batch time: {chunk_duration:.2f}s")
        
    # Build indexes to optimize dashboard queries
    print("Building indices for alignments and sequences tables...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alignments_gene ON alignments(gene_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alignments_transcript ON alignments(transcript_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequences_alignment_id ON sequences(alignment_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequences_name ON sequences(sequence_name)")
    conn.commit()
    
    pool.close()
    pool.join()
    conn.close()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Database generation finished! Total time: {duration/60:.2f} minutes.")

if __name__ == "__main__":
    main()
