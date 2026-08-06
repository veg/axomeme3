import gzip
import pickle
import time
import os
import numpy as np

# standard dictionaries
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

GENETIC_CODE = {
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
    'TGC':'C', 'TGT':'C', 'TGG':'W',
}

AA_LIST = "ACDEFGHIKLMNPQRSTVWY*-?"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

CODON_TO_AA_IDX = {}
for c in codons_list:
    aa = GENETIC_CODE.get(c, '?')
    CODON_TO_AA_IDX[c] = AA_TO_IDX.get(aa, 22)

# Create lookup tables
CODON_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 65
AA_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 22

grid = np.arange(256, dtype=np.int32)
b0 = grid[:, None, None] * 65536
b1 = grid[None, :, None] * 256
b2 = grid[None, None, :]
flat_indices = (b0 + b1 + b2).ravel()
b0_val = (flat_indices // 65536)
b1_val = (flat_indices // 256) % 256
b2_val = flat_indices % 256

gap_mask = (b0_val == 45) | (b1_val == 45) | (b2_val == 45)
CODON_LOOKUP[gap_mask] = 64
AA_LOOKUP[gap_mask] = 21

unknown_mask = (b0_val == 78) | (b1_val == 78) | (b2_val == 78) | \
               (b0_val == 110) | (b1_val == 110) | (b2_val == 110) | \
               (b0_val == 63) | (b1_val == 63) | (b2_val == 63)
unknown_mask = unknown_mask & (~gap_mask)
CODON_LOOKUP[unknown_mask] = 65
AA_LOOKUP[unknown_mask] = 22

for codon, c_idx in CODON_TO_IDX.items():
    if len(codon) == 3 and '-' not in codon and '?' not in codon:
        for c_str in [codon.upper(), codon.lower()]:
            b_encoded = c_str.encode('ascii')
            idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
            CODON_LOOKUP[idx] = c_idx
            aa = GENETIC_CODE.get(codon.upper(), '?')
            AA_LOOKUP[idx] = AA_TO_IDX.get(aa, 22)


def select_diverse_species_matrix(dist_matrix_np, k, keep_idx=0):
    n = dist_matrix_np.shape[0]
    if n <= k:
        return list(range(n))
    selected = [keep_idx]
    selected_set = {keep_idx}
    min_dists = dist_matrix_np[keep_idx].copy()
    for _ in range(k - 1):
        farthest_val = -1.0
        farthest_idx = -1
        for i in range(n):
            if i not in selected_set:
                if min_dists[i] > farthest_val:
                    farthest_val = min_dists[i]
                    farthest_idx = i
        if farthest_idx == -1:
            break
        selected.append(farthest_idx)
        selected_set.add(farthest_idx)
        min_dists = np.minimum(min_dists, dist_matrix_np[farthest_idx])
    return sorted(selected)


def compute_mds_coordinates(dist_matrix_np, n_components=4):
    N = dist_matrix_np.shape[0]
    if N <= n_components:
        return np.zeros((N, n_components), dtype=np.float32)
    D2 = dist_matrix_np ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (H @ D2 @ H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    coords = np.zeros((N, n_components), dtype=np.float32)
    for i in range(n_components):
        val = evals[i]
        if val > 0:
            coords[:, i] = evecs[:, i] * np.sqrt(val)
    return coords


def compact_cache(src_path, dest_path, max_species=256):
    print(f"Loading source cache from {src_path}...")
    t0 = time.time()
    with gzip.open(src_path, "rb") as f:
        src_cache = pickle.load(f)
    print(f"Loaded cache for {len(src_cache):,} genes in {time.time() - t0:.2f}s.")
    
    compacted = {}
    
    t_start = time.time()
    for idx, (gene_name, data) in enumerate(src_cache.items()):
        if "seq_dict" not in data or "species_names" not in data or "dist_arr" not in data:
            continue
            
        species_names = data["species_names"]
        seq_dict = data["seq_dict"]
        dist_arr = data["dist_arr"]
        
        # 1. Coordinate / MDS calculation
        mds_coords = data.get("mds_coords")
        if mds_coords is None:
            mds_coords = compute_mds_coordinates(dist_arr, n_components=4)
            
        # 2. Precompute diversity indices
        n_spec = len(species_names)
        if n_spec > max_species:
            selected_indices = select_diverse_species_matrix(dist_arr, max_species, keep_idx=0)
        else:
            selected_indices = list(range(n_spec))
            
        # 3. Vectorized tokenization of sequences
        any_seq = next(iter(seq_dict.values())) if seq_dict else ""
        seq_len_codons = len(any_seq) // 3
        
        codon_ids_list = []
        aa_ids_list = []
        valid_seqs = np.zeros(n_spec, dtype=bool)
        
        for i, spec in enumerate(species_names):
            seq = seq_dict.get(spec, "")
            if seq:
                valid_seqs[i] = True
                seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
                if len(seq_bytes) == seq_len_codons * 3:
                    codons_bytes = seq_bytes.reshape(seq_len_codons, 3)
                    indices = codons_bytes[:, 0].astype(np.int32) * 65536 + codons_bytes[:, 1].astype(np.int32) * 256 + codons_bytes[:, 2].astype(np.int32)
                    c_ids = CODON_LOOKUP[indices]
                    a_ids = AA_LOOKUP[indices]
                else:
                    c_ids = np.ones(seq_len_codons, dtype=np.int8) * 65
                    a_ids = np.ones(seq_len_codons, dtype=np.int8) * 22
            else:
                c_ids = np.ones(seq_len_codons, dtype=np.int8) * 65
                a_ids = np.ones(seq_len_codons, dtype=np.int8) * 22
            codon_ids_list.append(c_ids)
            aa_ids_list.append(a_ids)
            
        codon_ids_matrix = np.stack(codon_ids_list, axis=0)
        aa_ids_matrix = np.stack(aa_ids_list, axis=0)
        
        compacted[gene_name] = {
            "codon_ids_matrix": codon_ids_matrix,
            "aa_ids_matrix": aa_ids_matrix,
            "species_names": species_names,
            "dist_arr": dist_arr,
            "mds_coords": mds_coords,
            "variable_sites": data.get("variable_sites"),
            "selected_indices": selected_indices,
            "valid_seqs": valid_seqs
        }
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{len(src_cache)} genes...")
            
    print(f"Finished processing in {time.time() - t_start:.2f}s. Saving compacted cache...")
    
    t_save = time.time()
    with gzip.open(dest_path, "wb") as f:
        pickle.dump(compacted, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved compacted cache to {dest_path} in {time.time() - t_save:.2f}s.")


if __name__ == "__main__":
    src = "/Users/sergei/Projects/TOGA_MEME/msa_cache.pkl.gz"
    dest = "/Users/sergei/Projects/TOGA_MEME/msa_cache_compact.pkl.gz"
    compact_cache(src, dest)
