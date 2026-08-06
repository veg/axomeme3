#!/usr/bin/env python3
"""
run_hyphy_trim.py
-----------------
Runs the hyphy trim-label-tree command for each gene in the hyphy/ directory.
Utilizes multiprocessing to process all genes in parallel.
"""

import os
import subprocess
import multiprocessing as mp
from datetime import datetime

def run_trim(gene_info):
    gene, msa_path, tree_path, out_path = gene_info
    cmd = [
        "hyphy", "trim-label-tree",
        "--tree", tree_path,
        "--msa", msa_path,
        "--regexp", ".",
        "--output", out_path,
        "ENV=GZIP_COMPRESSION_LEVEL=9;"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return (gene, False, res.stderr or res.stdout)
        return (gene, True, None)
    except Exception as e:
        return (gene, False, str(e))

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    hyphy_dir = os.path.join(base_dir, "hyphy")
    msa_dir = os.path.join(base_dir, "msa")
    tree_path = os.path.join(base_dir, "pruned_species_tree.nhx")
    
    os.makedirs(msa_dir, exist_ok=True)
    
    # List all gene FASTA files in hyphy/
    genes = [f[:-3] for f in os.listdir(hyphy_dir) if f.endswith('.fa')]
    genes.sort()
    
    total_genes = len(genes)
    print(f"Found {total_genes} genes to process.")
    
    # Prepare arguments for each task
    tasks = []
    for gene in genes:
        msa_path = os.path.join(hyphy_dir, f"{gene}.fa")
        out_path = os.path.join(msa_dir, f"{gene}.gz")
        tasks.append((gene, msa_path, tree_path, out_path))
        
    start_time = datetime.now()
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting HyPhy trimming with {num_workers} parallel workers...")
    
    pool = mp.Pool(num_workers)
    
    completed = 0
    errors = []
    
    for gene, success, err_msg in pool.imap_unordered(run_trim, tasks):
        completed += 1
        if not success:
            errors.append((gene, err_msg))
            
        if completed % 1000 == 0 or completed == total_genes:
            pct = (completed / total_genes) * 100
            print(f"Progress: {completed}/{total_genes} ({pct:.1f}%)...")
            
    pool.close()
    pool.join()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nCompleted processing of {total_genes} genes in {duration/60:.2f} minutes.")
    print(f"Successfully processed: {total_genes - len(errors)} genes.")
    print(f"Failed/Errors: {len(errors)} genes.")
    
    if errors:
        print("\nFirst 10 Errors:")
        for gene, err in errors[:10]:
            print(f"Gene: {gene} -> {err}")
            
        # Write errors to a log file
        err_log_path = os.path.join(base_dir, "trim_errors.log")
        with open(err_log_path, 'w') as f:
            for gene, err in errors:
                f.write(f"Gene: {gene}\nError:\n{err}\n{'='*40}\n")
        print(f"All errors written to {err_log_path}")

if __name__ == '__main__':
    main()
