import gzip
import pickle
import os
import numpy as np
import time

src_path = "/Users/sergei/Projects/TOGA_MEME/msa_cache_compact.pkl.gz"
dest_dir = "/Users/sergei/Projects/TOGA_MEME/msa_cache_npz"
os.makedirs(dest_dir, exist_ok=True)

print("Loading compact cache...", flush=True)
t0 = time.time()
with gzip.open(src_path, "rb") as f:
    cache = pickle.load(f)
print(f"Loaded cache in {time.time() - t0:.2f}s.", flush=True)

print("Saving individual .npz files...", flush=True)
t1 = time.time()
for idx, (gene_name, data) in enumerate(cache.items()):
    gene_path = os.path.join(dest_dir, f"{gene_name}.npz")
    # Use np.savez instead of np.savez_compressed for 10x+ speedup
    np.savez(
        gene_path,
        codon_ids_matrix=data["codon_ids_matrix"],
        aa_ids_matrix=data["aa_ids_matrix"],
        species_names=np.array(data["species_names"]),
        dist_arr=data["dist_arr"],
        mds_coords=data["mds_coords"],
        variable_sites=np.array(data["variable_sites"]),
        selected_indices=np.array(data["selected_indices"]),
        valid_seqs=data["valid_seqs"]
    )
    if (idx + 1) % 1000 == 0:
        print(f"Saved {idx+1}/{len(cache)} files...", flush=True)
print(f"Completed in {time.time() - t1:.2f}s.", flush=True)
