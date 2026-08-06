import os
import sqlite3
import gzip
import subprocess
import shutil

# List of 48 legacy species leaf names
LEGACY_SPECIES = [
    'hg', 'panTro', 'gorGor', 'ponAbe', 'rheMac', 'papAnu', 'calJacc', 'saiBolBol', 
    'micMur', 'otoGar', 'musMusc', 'ratNor', 'dipOrd', 'sciCar', 'cavPor', 'hetGla', 
    'oryCuni', 'ochPri', 'susScro', 'vicPacHua', 'turTru', 'orcOrc', 'bosTau', 'oviArie', 
    'capHirc', 'equCaba', 'cerSimCot', 'felCat', 'canFam', 'ailMel', 'musFur', 'odoRos', 
    'lepWed', 'pteVam', 'myoLuc', 'eriEur', 'sorAra', 'conCri', 'loxAfr', 'triManLat', 
    'chrAsi', 'echTel', 'eleEdw', 'oryAfeAfe', 'dasNov', 'choHof', 'proCap', 'tupChi'
]

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
                sequences.append(line_strip)
                
    mapped = {}
    for label, seq in zip(taxlabels, sequences):
        mapped[label] = seq
    return mapped

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    db_path = os.path.join(base_dir, "meme_results.db")
    
    # Destination directories
    legacy_msa_dir = os.path.join(base_dir, "legacy_msa")
    scratch_dir = os.path.join(base_dir, "scratch")
    os.makedirs(legacy_msa_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Select a balanced set of 100 genes:
    # 1. 30 high selection (num_selected_sites >= 20)
    # 2. 40 moderate selection (num_selected_sites between 5 and 19)
    # 3. 30 low/no selection (num_selected_sites < 5)
    
    print("Selecting 100 representative genes from the database...")
    
    cursor.execute("""
        SELECT gene_name FROM gene_results 
        WHERE num_selected_sites >= 20 
        ORDER BY num_selected_sites DESC LIMIT 30
    """)
    high_genes = [r[0] for r in cursor.fetchall()]
    
    cursor.execute("""
        SELECT gene_name FROM gene_results 
        WHERE num_selected_sites >= 5 AND num_selected_sites < 20 
        ORDER BY num_selected_sites DESC LIMIT 40
    """)
    med_genes = [r[0] for r in cursor.fetchall()]
    
    cursor.execute("""
        SELECT gene_name FROM gene_results 
        WHERE num_selected_sites < 5 
        ORDER BY num_selected_sites ASC LIMIT 30
    """)
    low_genes = [r[0] for r in cursor.fetchall()]
    
    target_genes = high_genes + med_genes + low_genes
    print(f"Selected {len(target_genes)} genes: {len(high_genes)} high selection, {len(med_genes)} medium, {len(low_genes)} low.")
    
    conn.close()
    
    # Save the gene list to run MEME on
    gene_list_path = os.path.join(base_dir, "legacy_gene_list.txt")
    with open(gene_list_path, "w") as f:
        for g in target_genes:
            f.write(f"{g}\n")
    print(f"Gene list written to {gene_list_path}")
    
    legacy_set = set(LEGACY_SPECIES)
    tree_path = os.path.join(base_dir, "pruned_species_tree.nhx")
    
    success_count = 0
    for idx, gene in enumerate(target_genes):
        src_path = os.path.join(base_dir, "msa", f"{gene}.gz")
        out_gz_path = os.path.join(legacy_msa_dir, f"{gene}.gz")
        
        if not os.path.exists(src_path):
            print(f"[{idx+1}/100] Warning: original alignment for {gene} not found at {src_path}")
            continue
            
        # Parse original alignment
        full_seqs = parse_nexus_gz(src_path)
        
        # Filter to legacy species
        sub_seqs = {}
        for label, seq in full_seqs.items():
            clean_label = label.split('{')[0]
            if clean_label in legacy_set:
                sub_seqs[clean_label] = seq
                
        if len(sub_seqs) < 10:
            print(f"[{idx+1}/100] Skipping {gene}: too few legacy species present ({len(sub_seqs)})")
            continue
            
        # Prune all-gap codons
        seq_len = len(next(iter(sub_seqs.values())))
        keep_codon_indices = []
        num_codons = seq_len // 3
        for c in range(num_codons):
            codon_seqs = [sub_seqs[sp][3*c : 3*c+3] for sp in sub_seqs]
            is_all_gaps = all(all(char in ['-', '?'] for char in codon) for codon in codon_seqs)
            if not is_all_gaps:
                keep_codon_indices.append(c)
                
        # Reconstruct trimmed sequences
        trimmed_seqs = {}
        for sp in sub_seqs:
            trimmed_seq = "".join(sub_seqs[sp][3*c : 3*c+3] for c in keep_codon_indices)
            trimmed_seqs[sp] = trimmed_seq
            
        # Write temporary FASTA file
        tmp_fa_path = os.path.join(scratch_dir, f"{gene}_legacy_tmp.fa")
        with open(tmp_fa_path, "w") as f:
            for sp, seq in trimmed_seqs.items():
                f.write(f">{sp}\n{seq}\n")
                
        # Run HyPhy trim-label-tree
        cmd = [
            "hyphy", "trim-label-tree",
            "--tree", tree_path,
            "--msa", tmp_fa_path,
            "--regexp", ".",
            "--output", out_gz_path,
            "ENV=GZIP_COMPRESSION_LEVEL=9;"
        ]
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.exists(tmp_fa_path):
            os.remove(tmp_fa_path)
            
        if res.returncode == 0 and os.path.exists(out_gz_path) and os.path.getsize(out_gz_path) > 0:
            success_count += 1
            if success_count % 10 == 0 or success_count == 1:
                print(f"Processed {success_count} alignments successfully (latest: {gene}, sequences={len(trimmed_seqs)}, sites={len(keep_codon_indices)})")
        else:
            print(f"[{idx+1}/100] Error processing {gene}: returncode={res.returncode}, stderr={res.stderr}")
            
    print(f"\nCompleted! Generated {success_count} legacy alignments in {legacy_msa_dir}")

if __name__ == "__main__":
    main()
