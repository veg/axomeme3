#!/usr/bin/env python3
"""
partition_jobs.py
-----------------
Partitions the remaining aBSREL workload between silverback and magilla.
Excludes already completed genes (121) from the active database.
"""

import os
import glob
import subprocess

def get_completed_genes():
    # Query silverback for completed genes in the database
    cmd = ["ssh", "silverback", "sqlite3 /home/sergei/Projects/TOGA/absrel_results.db 'select gene_name from gene_results'"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        completed = {line.strip() for line in res.stdout.split("\n") if line.strip()}
        return completed
    except subprocess.CalledProcessError as e:
        print(f"Error querying completed genes: {e.stderr}")
        return set()

def main():
    # 1. Get all genes from the local msa directory
    msa_files = sorted(glob.glob("msa/*.gz"))
    all_genes = [os.path.basename(f)[:-3] for f in msa_files]
    print(f"Total genes in dataset: {len(all_genes)}")
    
    # 2. Get completed genes from silverback DB
    completed = get_completed_genes()
    print(f"Completed genes in DB: {len(completed)}")
    
    # 3. Filter remaining genes
    remaining = [g for g in all_genes if g not in completed]
    print(f"Remaining genes to process: {len(remaining)}")
    
    # 4. Partition 50/50
    # Even indices to silverback, odd indices to magilla
    silverback_list = remaining[::2]
    magilla_list = remaining[1::2]
    
    print(f"Partition size for silverback: {len(silverback_list)}")
    print(f"Partition size for magilla: {len(magilla_list)}")
    
    # 5. Write to files
    with open("gene_list_silverback.txt", "w") as f:
        for g in silverback_list:
            f.write(g + "\n")
            
    with open("gene_list_magilla.txt", "w") as f:
        for g in magilla_list:
            f.write(g + "\n")
            
    print("Partition files written locally: gene_list_silverback.txt, gene_list_magilla.txt")

if __name__ == "__main__":
    main()
