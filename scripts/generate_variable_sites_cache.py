import os
import pickle
import numpy as np
import time

npz_dir = "/Users/sergei/Projects/TOGA_MEME/msa_cache_npz"
dest_path = os.path.join(npz_dir, "variable_sites.pkl")

print("Compiling variable_sites from NPZ files...", flush=True)
t0 = time.time()
variable_sites_dict = {}

# List all NPZ files
npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
print(f"Found {len(npz_files)} NPZ files to parse.", flush=True)

for idx, f in enumerate(npz_files):
    gene_name = f[:-4]
    npz_path = os.path.join(npz_dir, f)
    try:
        data = np.load(npz_path, allow_pickle=True)
        variable_sites_dict[gene_name] = data["variable_sites"].tolist()
    except Exception as e:
        print(f"Error reading {f}: {e}", flush=True)
    if (idx + 1) % 2000 == 0:
        print(f"Processed {idx+1}/{len(npz_files)} files...", flush=True)

with open(dest_path, "wb") as f_out:
    pickle.dump(variable_sites_dict, f_out)
    
print(f"Successfully compiled and saved to {dest_path} in {time.time() - t0:.2f}s.", flush=True)
