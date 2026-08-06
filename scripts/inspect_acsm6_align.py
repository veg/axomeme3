#!/usr/bin/env python3
import gzip
import os

def parse_nexus_gz(filepath):
    taxlabels = []
    sequences = []
    in_taxlabels = False
    in_matrix = False
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
                
            # Parse TAXLABELS
            if line_strip.upper().startswith("TAXLABELS"):
                in_taxlabels = True
                content = line_strip[len("TAXLABELS"):].strip()
                tokens = content.replace("'", "").replace(";", "").split()
                taxlabels.extend(tokens)
                if line_strip.endswith(";"):
                    in_taxlabels = False
                continue
            
            if in_taxlabels:
                tokens = line_strip.replace("'", "").replace(";", "").split()
                taxlabels.extend(tokens)
                if line_strip.endswith(";"):
                    in_taxlabels = False
                continue
                
            # Parse MATRIX
            if line_strip.upper().startswith("MATRIX"):
                in_matrix = True
                continue
                
            if in_matrix:
                if line_strip == ";":
                    in_matrix = False
                    continue
                if line_strip.endswith(";"):
                    sequences.append(line_strip[:-1].strip())
                    in_matrix = False
                    continue
                # Sequence lines in matrix
                sequences.append(line_strip)
                
    mapped = {}
    for label, seq in zip(taxlabels, sequences):
        mapped[label] = seq
    return mapped

def main():
    align_path = "msa/ACSM6.gz"
    if not os.path.exists(align_path):
        print(f"Alignment not found at {align_path}!")
        return
        
    seqs = parse_nexus_gz(align_path)
    print(f"Total sequences in alignment: {len(seqs)}")
    
    first_seq = next(iter(seqs.values()))
    print(f"Sequence length (nucleotides): {len(first_seq)}")
    
    # Primate species in the alignment
    target_names = []
    for name in seqs.keys():
        name_lower = name.lower()
        if any(x in name_lower for x in ['ate', 'hom', 'mac', 'pan', 'gor', 'pon', 'lag', 'hg']):
            target_names.append(name)
            
    print(f"Primate species: {target_names}")
    
    # Selected sites are 463 and 499 (1-based indices)
    # We will print from codon 450 to 510
    start_codon = 450
    end_codon = 510
    
    start_nuc = (start_codon - 1) * 3
    end_nuc = end_codon * 3
    
    print(f"\n--- Alignment Window (Codons {start_codon} to {end_codon}) ---")
    print("Site 463 is marked with [ ] and Site 499 with { }")
    print("-" * 120)
    
    for name in sorted(target_names):
        seq_slice = seqs[name][start_nuc:end_nuc]
        codons = [seq_slice[i:i+3] for i in range(0, len(seq_slice), 3)]
        
        formatted_codons = []
        for idx, c in enumerate(codons):
            codon_num = start_codon + idx
            if codon_num == 463:
                formatted_codons.append(f"[{c}]")
            elif codon_num == 499:
                formatted_codons.append(f"{{{c}}}")
            else:
                formatted_codons.append(c)
                
        print(f"{name:<15}: {' '.join(formatted_codons)}")

if __name__ == "__main__":
    main()
