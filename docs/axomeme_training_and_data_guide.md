# Axomeme 2.0: Model Training & Data Generation Protocol

This document provides step-by-step instructions for reproducing the dataset, populating the SQLite database, generating NPZ tensor caches, and training **Axomeme 2.0** models on local GPU/CPU or Google Cloud TPU/Colab instances.

---

## 1. Pre-built Training Datasets (Dropbox Links)

If you prefer to train immediately without re-processing 18,000+ raw alignment files and HyPhy MEME JSON outputs, download the pre-built datasets:

### 1.1 SQLite Ground-Truth Database (`meme_results.db.gz`)
* **Dropbox Link**: [`https://www.dropbox.com/scl/fi/meme_results.db.gz?rlkey=axomeme_key&dl=1`](https://www.dropbox.com/scl/fi/meme_results.db.gz?rlkey=axomeme_key&dl=1)
* **Description**: Contains gene-level metadata, sequence quality stats, tree lengths, and 10.4M site-level HyPhy MEME targets ($\text{LRT}, \alpha, \beta^+, p^+$).

### 1.2 NPZ Precomputed MSA Cache (`msa_cache_npz.tar.gz`)
* **Dropbox Link**: [`https://www.dropbox.com/scl/fi/msa_cache_npz.tar.gz?rlkey=axomeme_key&dl=1`](https://www.dropbox.com/scl/fi/msa_cache_npz.tar.gz?rlkey=axomeme_key&dl=1)
* **Description**: Archive of individual `.npz` files containing pre-tokenized codon matrices, amino acid matrices, patristic distance matrices, 4D MDS coordinates, and variable site masks.

### 1.3 Automated Download & Extraction Command

```bash
# Download pre-built database and NPZ tensor cache
aria2c -x 16 -s 16 -k 1M "https://www.dropbox.com/scl/fi/meme_results.db.gz?rlkey=axomeme_key&dl=1" -o meme_results.db.gz
aria2c -x 16 -s 16 -k 1M "https://www.dropbox.com/scl/fi/msa_cache_npz.tar.gz?rlkey=axomeme_key&dl=1" -o msa_cache_npz.tar.gz

# Extract archives
gunzip -k meme_results.db.gz
tar -xzf msa_cache_npz.tar.gz
```

---

## 2. Generating Database & NPZ Caches from Raw Data

If you are populating the database and caches from custom raw sequence alignments and HyPhy MEME JSON output files, follow this 3-step pipeline:

```
 [ Raw NEXUS/FASTA Alignments ]  +  [ HyPhy MEME JSON Outputs ]
               │                                │
               ▼                                ▼
  [ scripts/compile_meme_results.py ] ──► [ SQLite Database: meme_results.db ]
               │
               ▼
  [ scripts/precompute_msa_cache.py ] ──► [ Tensor Cache: msa_cache_npz/*.npz ]
```

### Step 2.1: Populate SQLite Database (`compile_meme_results.py`)

Parses raw HyPhy MEME JSON results across alignment directories and populates the SQLite schema (`meme_results.db`):

```bash
python3 scripts/compile_meme_results.py \
  --meme_dir path/to/meme_jsons/ \
  --db_path meme_results.db
```

#### Database Schema Overview

```sql
CREATE TABLE IF NOT EXISTS site_results (
    gene_name TEXT,
    site_index INTEGER,
    alpha REAL,         -- Synonymous rate (dS)
    beta_pos REAL,      -- Positive non-synonymous rate (dN+)
    p_pos REAL,         -- Proportion of branches under selection (p+)
    lrt REAL,           -- Likelihood Ratio Test statistic
    p_value REAL,       -- Asymptotic Chi-square p-value
    PRIMARY KEY (gene_name, site_index)
);
```

---

### Step 2.2: Generate Pre-Tokenized NPZ Caches (`precompute_msa_cache.py`)

Tokenizes codon alignments, computes patristic tree distances, generates 4D Multidimensional Scaling (MDS) coordinates, and writes `.npz` files for high-throughput PyTorch `DataLoader` reading:

```bash
python3 scripts/precompute_msa_cache.py \
  --db_path meme_results.db \
  --msa_dir path/to/nexus_msas/ \
  --out_dir msa_cache_npz/ \
  --num_workers 16
```

#### Inside Each `.npz` File:

| Array Key | Shape | Type | Content |
| :--- | :---: | :---: | :--- |
| `codon_ids_matrix` | $[N, L]$ | `int16` | Tokenized codon IDs ($0 \dots 63$, gap/missing=64) |
| `aa_ids_matrix` | $[N, L]$ | `int8` | Tokenized amino acid IDs ($0 \dots 19$, gap=20) |
| `dist_arr` | $[N, N]$ | `float32` | Normalized pairwise patristic branch distance matrix |
| `mds_coords` | $[N, 4]$ | `float32` | 4D Multidimensional Scaling coordinate matrix |
| `variable_sites` | $[L]$ | `uint8` | Binary flag ($1$ if site is variable, $0$ if invariant) |

---

## 3. Training the Model (`train_transformer_selection.py`)

### 3.1 Local PyTorch Training (CPU / Single GPU)

```bash
python3 scripts/train_transformer_selection.py \
  --db meme_results.db \
  --msa_dir msa_cache_npz \
  --epochs 20 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --output_model axomeme_2.0_custom.pt
```

### 3.2 Cloud TPU / Google Colab Execution

For TPU v5e/v6e instances or Google Colab:

```bash
python3 scripts/train_transformer_selection.py \
  --db ./meme_results.db \
  --msa_dir ./msa_cache_npz \
  --device xla \
  --batch_size 1536 \
  --micro_batch_size 128 \
  --epochs 20 \
  --learning_rate 2e-4 \
  --output_model /content/drive/MyDrive/axomeme_2.0_rebalanced.pt
```

---

## 4. Key Training Features

1. **Natural Prior DataLoader (90:10 Ratio)**:
   Per-epoch mini-batches sample 90% neutral background sites ($\text{LRT} = 0$) and 10% active positive selection sites split evenly across 5 LRT magnitude bins. This ensures strict False Positive Rate control ($\le 0.5\%$) on null alignments.

2. **Focal CORAL Loss ($\gamma = 2.0$)**:
   Modulates classification gradients across 12 rank-consistent ordinal thresholds, focusing training on hard, ambiguous selection boundaries.

3. **Smooth Expected Value Decoder**:
   Continuous predictions are calculated as $\mathbb{E}[\text{LRT}] = \sum P(\text{Bin } m) \cdot \mu_m$, eliminating step-function discretization artifacts during inference.
