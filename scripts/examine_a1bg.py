#!/usr/bin/env python3
"""
examine_a1bg.py
----------------
Analyzes the A1BG codonified FASTA alignment to determine why so many sequences
are out of frame relative to the reference sequence (hg38).
It scans the alignment column-by-column, counts gaps vs bases, and identifies
indel regions that disrupt the triplet codon structure.
"""

import os

ALIGNMENT_PATH = "/Users/sergei/Dropbox/TOGA2026/individualAlis/ENST00000263100.8#A1BG.codonified.fa"

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name:
            sequences[current_name] = "".join(current_seq)
    return sequences

def main():
    if not os.path.exists(ALIGNMENT_PATH):
        print(f"Error: Alignment file not found at {ALIGNMENT_PATH}")
        return
        
    seqs = read_fasta(ALIGNMENT_PATH)
    ref_name = 'hg38'
    if ref_name not in seqs:
        print(f"Error: Reference sequence {ref_name} not found in alignment.")
        return
        
    ref_seq = seqs[ref_name]
    alignment_len = len(ref_seq)
    
    other_names = [name for name in seqs.keys() if name != ref_name]
    num_others = len(other_names)
    
    print(f"Alignment length: {alignment_len} bp")
    print(f"Reference sequence (hg38) gap-stripped length: {len(ref_seq.replace('-', ''))} bp")
    print(f"Number of other species: {num_others}")
    
    # Analyze column by column
    # We want to identify contiguous blocks of gaps in hg38 or in other species
    indels = []
    in_indel = False
    indel_start = 0
    indel_type = None # 'hg38_gap' (insertion in others) or 'others_gap' (deletion in others)
    
    for i in range(alignment_len):
        ref_char = ref_seq[i]
        
        # Check gap status in others
        other_chars = [seqs[name][i] for name in other_names]
        other_gap_pct = other_chars.count('-') / num_others
        
        # Determine if this column is an indel column
        is_ref_gap = (ref_char == '-')
        is_others_mostly_gap = (other_gap_pct > 0.8)
        
        col_type = None
        if is_ref_gap and not is_others_mostly_gap:
            col_type = 'hg38_gap' # Insertion in most other species
        elif not is_ref_gap and is_others_mostly_gap:
            col_type = 'others_gap' # Deletion in most other species
            
        if col_type:
            if not in_indel:
                in_indel = True
                indel_start = i
                indel_type = col_type
            elif col_type != indel_type:
                # End current indel, start new one
                indels.append((indel_start, i - 1, indel_type, i - indel_start))
                indel_start = i
                indel_type = col_type
        else:
            if in_indel:
                indels.append((indel_start, i - 1, indel_type, i - indel_start))
                in_indel = False
                
    if in_indel:
        indels.append((indel_start, alignment_len - 1, indel_type, alignment_len - indel_start))
        
    print("\n--- Identified Indel Regions (where hg38 and >80% of other species disagree on gap status) ---")
    for start, end, itype, length in indels:
        ref_segment = ref_seq[start:end+1]
        print(f"Pos {start+1}-{end+1} (Length: {length} bp): Type: {itype}")
        print(f"  hg38 segment: {ref_segment}")
        
        # Sample some other species segments
        sample_chars = []
        for name in other_names[:5]:
            sample_chars.append(f"{name}: {seqs[name][start:end+1]}")
        print("  Samples from other species:")
        for s in sample_chars:
            print(f"    {s}")
            
    # Let's count how many species are in frame and inspect if they share a specific deletion/insertion
    print("\n--- Modulo 3 analysis of sequence lengths ---")
    mod_counts = {0: 0, 1: 0, 2: 0}
    mod_species = {0: [], 1: [], 2: []}
    for name, seq in seqs.items():
        s_len = len(seq.replace('-', ''))
        mod = s_len % 3
        mod_counts[mod] += 1
        if len(mod_species[mod]) < 5:
            mod_species[mod].append((name, s_len))
            
    for m, count in mod_counts.items():
        print(f"Modulo {m}: {count} species")
        for name, s_len in mod_species[m]:
            print(f"  - {name}: {s_len} bp")

if __name__ == "__main__":
    main()
