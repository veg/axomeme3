#!/usr/bin/env python3
import os
import sys
import subprocess


LOCAL_DIR = os.path.join (os.getcwd(), "meme_results")
LOCAL_DB = os.path.join (os.getcwd(), "meme_results.db")

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"  Error: {res.stderr.strip()}")
        return False, res.stdout, res.stderr
    return True, res.stdout, res.stderr

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # 1. Sync from computational nodes/clusters to local
    hosts = [
        #("m3.local", "/Users/sergei/Projects/TOGA_MEME/meme_results/"),
        #("magilla", "/home/sergei/Projects/TOGA/meme_results/"),
        ("silverback", "/home/sergei/Projects/TOGA/meme_results/")
    ]
    
    print("=== Step 1: Syncing raw results from computational nodes to local ===")
    for host, remote_path in hosts:
        print(f"Syncing from {host}:{remote_path}...")
        cmd = [
            "rsync", "-av", 
            "--include=*/", 
            "--include=*.json.gz", 
            "--exclude=*", 
            f"{host}:{remote_path}", 
            LOCAL_DIR
        ]
        success, out, err = run_cmd(cmd)
        if not success:
            print(f"  Warning: Sync from {host} failed, continuing...")
            
    # 2. Compile database locally
    print("\n=== Step 2: Compiling database locally ===")
    cmd = ["python3", "scripts/compile_meme_results.py"]
    success, out, err = run_cmd(cmd)
    if not success:
        print("Error: Failed to compile database locally")
        sys.exit(1)
    print("Compilation output:")
    print(out)
    
    # 3. Run curation filters locally
    print("\n=== Step 3: Running curation filters locally ===")
    cmd = ["python3", "scripts/update_meme_db_filter.py"]
    success, out, err = run_cmd(cmd)
    if not success:
        print("Error: Failed to run curation filters locally")
        sys.exit(1)
    print("Filter output:")
    print(out)
    
    # 4. Run threshold update locally to enforce p <= 0.01
    print("\n=== Step 4: Running local database threshold update (p <= 0.01) ===")
    cmd = ["python3", "scripts/update_db_threshold.py"]
    success, out, err = run_cmd(cmd)
    if not success:
        print("Error: Failed to run database threshold update")
        sys.exit(1)
    print(out)
    
    # 5. Run branch mapping precomputations locally
    print("\n=== Step 5: Running local branch mapping precomputations ===")
    cmd = ["python3", "scripts/precompute_branch_mapping.py"]
    success, out, err = run_cmd(cmd)
    if not success:
        print("Error: Failed to run branch mapping precomputations")
        sys.exit(1)
    print(out)
    
    # 6. Run local selection type classification
    print("\n=== Step 6: Running local selection type classification ===")
    cmd = ["python3", "scripts/classify_selection_types.py"]
    success, out, err = run_cmd(cmd)
    if not success:
        print("Error: Failed to run selection type classification")
        sys.exit(1)
    print(out)
    
    print("\n=== UPDATE COMPLETE ===")

if __name__ == "__main__":
    main()
