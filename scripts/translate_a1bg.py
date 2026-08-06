#!/usr/bin/env python3
"""
translate_a1bg.py
------------------
Translates the gap-stripped hg38 and HLcapSib1 sequences from the A1BG alignment
to see the effect of the frame shift on the resulting protein translation.
"""

import os

ALIGNMENT_PATH = "/Users/sergei/Dropbox/TOGA2026/individualAlis/ENST00000263100.8#A1BG.codonified.fa"

# Standard genetic code dictionary
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGA':'*',
}

def translate(seq):
    seq = seq.upper().replace('N', 'X')
    pep = []
    for i in range(0, len(seq) - len(seq)%3, 3):
        codon = seq[i:i+3]
        if 'X' in codon:
            pep.append('X')
        else:
            pep.append(CODON_TABLE.get(codon, '?'))
    return "".join(pep)

def main():
    with open(ALIGNMENT_PATH, 'r') as f:
        seqs = {}
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name:
                    seqs[current_name] = "".join(current_seq)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name:
            seqs[current_name] = "".join(current_seq)
            
    for name in ['hg38', 'HLcapSib1']:
        raw_seq = seqs[name].replace('-', '')
        pep = translate(raw_seq)
        
        print(f"\n=== Translation for {name} ===")
        print(f"Nucleotide length (gap-stripped): {len(raw_seq)} bp")
        print(f"Protein length: {len(pep)} aa")
        
        # Find stop codons
        stops = [i for i, aa in enumerate(pep) if aa == '*']
        print(f"Stop codon positions: {stops}")
        
        if stops:
            first_stop = stops[0]
            print(f"First stop codon at position {first_stop} (amino acid index)")
            print(f"Spliced sequence around first stop: {pep[max(0, first_stop-10):first_stop+15]}")
        else:
            print("No stop codons found!")

if __name__ == "__main__":
    main()
