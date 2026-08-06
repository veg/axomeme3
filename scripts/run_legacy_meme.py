import os
import subprocess
import time
import sys
import json
import glob
import multiprocessing as mp

def run_single_gene(args):
    path, out_dir, logs_dir = args
    filename = os.path.basename(path)
    gene_name = filename[:-3] if filename.endswith(".gz") else filename
    
    output_file = os.path.join(out_dir, f"{gene_name}.MEME.json.gz")
    log_file = os.path.join(logs_dir, f"{gene_name}.log")
    
    # Check if already completed and valid
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            import gzip
            with gzip.open(output_file, "rt") as f:
                json.load(f)
            return gene_name, True, 0.0, "Skipped (already completed)"
        except Exception:
            pass
            
    start_time = time.time()
    
    # Run hyphy meme, specifying CPU=2 threads to avoid over-subscription
    cmd = [
        "hyphy", "meme",
        "--alignment", path,
        "--output", output_file,
        "CPU=2",
        'ENV="GZIP_OUTPUT=1;"'
    ]
    
    try:
        with open(log_file, "w") as log_fh:
            result = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
        elapsed = time.time() - start_time
        if result.returncode == 0:
            return gene_name, True, elapsed, "Success"
        else:
            return gene_name, False, elapsed, f"Error (code {result.returncode})"
    except Exception as e:
        return gene_name, False, 0.0, f"Exception: {str(e)}"

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    msa_dir = os.path.join(base_dir, "legacy_msa")
    out_dir = os.path.join(base_dir, "legacy_meme_results")
    logs_dir = os.path.join(base_dir, "legacy_logs")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    alignment_paths = sorted(glob.glob(os.path.join(msa_dir, "*.gz")))
    if not alignment_paths:
        print(f"No legacy alignment files found in {msa_dir}!")
        sys.exit(1)
        
    print(f"Found {len(alignment_paths)} legacy alignments to process.")
    
    # Prepare arguments for multiprocessing pool
    tasks = [(p, out_dir, logs_dir) for p in alignment_paths]
    
    # We will run 6 workers in parallel, using 12 threads total
    num_workers = 6
    print(f"Starting parallel legacy MEME run with {num_workers} workers...")
    
    start_time = time.time()
    pool = mp.Pool(num_workers)
    
    completed = 0
    successes = 0
    failures = []
    
    for gene_name, success, elapsed, msg in pool.imap_unordered(run_single_gene, tasks):
        completed += 1
        if success:
            successes += 1
            if elapsed > 0.0:
                print(f"[{completed}/{len(tasks)}] Finished {gene_name} in {elapsed:.1f}s (Status: {msg})")
        else:
            failures.append((gene_name, msg))
            print(f"[{completed}/{len(tasks)}] FAILED {gene_name} (Status: {msg})")
            
    pool.close()
    pool.join()
    
    total_elapsed = time.time() - start_time
    print(f"\nLegacy MEME run finished!")
    print(f"- Total time: {total_elapsed/60:.2f} minutes")
    print(f"- Successes: {successes} / {len(tasks)}")
    print(f"- Failures: {len(failures)}")
    if failures:
        print("Failures list:")
        for name, err in failures:
            print(f"  {name}: {err}")

if __name__ == "__main__":
    main()
