#!/usr/bin/env python3
"""
run_meme_loop.py
----------------
Loops through all reconciled alignments in Projects/TOGA_MEME/msa/,
running MEME using HYPHYMPI on 24 cores sequentially, one gene at a time.
Supports resuming from completed runs.
"""

import os
import glob
import subprocess
import time
import sys
import json
import argparse

BASE_DIR = "/Users/sergei/Projects/TOGA_MEME"
MSA_DIR = os.path.join(BASE_DIR, "msa")
OUT_DIR = os.path.join(BASE_DIR, "meme_results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

MPIRUN_PATH = "/opt/homebrew/bin/mpirun"
HYPHYMPI_PATH = "/usr/local/bin/hyphympi"
NUM_CORES = 24

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulo", type=int, default=None, help="Only process files where index % divisor == modulo")
    parser.add_argument("--divisor", type=int, default=2, help="Divisor for modulo filtering")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Find all gzipped NEXUS files
    alignment_paths = sorted(glob.glob(os.path.join(MSA_DIR, "*.gz")))
    if not alignment_paths:
        print(f"No alignment files found in {MSA_DIR}!")
        sys.exit(1)

    if args.modulo is not None:
        print(f"Modulo filter active: only processing index % {args.divisor} == {args.modulo}")
        # Filter list beforehand so progress indices [idx+1/total] are correct for this partition
        filtered_paths = [(orig_idx, p) for orig_idx, p in enumerate(alignment_paths) if orig_idx % args.divisor == args.modulo]
        print(f"Partition size: {len(filtered_paths)} / {len(alignment_paths)} alignments.")
    else:
        filtered_paths = [(orig_idx, p) for orig_idx, p in enumerate(alignment_paths)]
        print(f"Found {len(alignment_paths)} alignment files to process.")

    # Loop through each alignment
    for current_idx, (orig_idx, path) in enumerate(filtered_paths):
        filename = os.path.basename(path)
        gene_name = filename[:-3] if filename.endswith(".gz") else filename
        
        output_file = os.path.join(OUT_DIR, f"{gene_name}.MEME.json.gz")
        log_file = os.path.join(LOGS_DIR, f"{gene_name}.log")

        # Skip if already completed and non-empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                try:
                    import gzip
                    with gzip.open(output_file, "rt") as f:
                        json.load(f)
                except Exception:
                    with open(output_file, "r") as f:
                        json.load(f)
                continue # Valid JSON (compressed or plain), skip!
            except Exception:
                print(f"Output for {gene_name} is corrupted, re-running.")

        print(f"[{current_idx+1}/{len(filtered_paths)}] (Original Align #{orig_idx+1}) Processing {gene_name}...")
        start_time = time.time()

        # Build mpirun command
        cmd = [
            MPIRUN_PATH,
            "-np", str(NUM_CORES),
            HYPHYMPI_PATH,
            "meme",
            "--alignment", path,
            "--output", output_file,
            'ENV="GZIP_OUTPUT=1;"'
        ]

        try:
            with open(log_file, "w") as log_fh:
                # Run the process
                result = subprocess.run(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )
            
            elapsed = time.time() - start_time
            if result.returncode == 0:
                print(f"[{current_idx+1}/{len(filtered_paths)}] (Original Align #{orig_idx+1}) Finished {gene_name} in {elapsed:.1f}s (Status: Success)")
            else:
                print(f"[{current_idx+1}/{len(filtered_paths)}] (Original Align #{orig_idx+1}) Failed {gene_name} in {elapsed:.1f}s (Status: Error, Return Code {result.returncode})")
                
        except KeyboardInterrupt:
            print("\nLoop interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"[{current_idx+1}/{len(filtered_paths)}] (Original Align #{orig_idx+1}) Error running {gene_name}: {e}")

if __name__ == "__main__":
    main()
