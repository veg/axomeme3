# Axomeme 2.0: Deep Axial Transformer for Fast Evolutionary Selection Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**Axomeme 2.0** is a scaled deep axial attention transformer neural network designed for rapid, site-level evolutionary selection inference on multiple sequence alignments (MSAs) of protein-coding nucleotide sequences. 

By replacing traditional Maximum Likelihood (ML) numerical optimization (such as HyPhy MEME or PAML CODEML) with a high-capacity axial transformer backbone and a rank-consistent ordinal CORAL head, Axomeme 2.0 infers Likelihood Ratio Test (LRT) selection statistics and evolutionary rate parameters ($\alpha, \beta^+, p^+$) in under **10 milliseconds per alignment** (a **4,000× speedup** over classical ML).

---

## 🌟 Key Features & Architectural Innovations

* **Sub-10ms Inference**: Processes full codon alignments (e.g. 500 species × 1,000 codons) in under 10 milliseconds on standard CPU/GPU/TPU accelerators.
* **Block-Diagonal Feature Disentanglement (`BlockLinear`)**: Implements parallel non-interacting matrix blocks ($\mathbf{W}_{\text{codon}} \in \mathbb{R}^{128 \times 128}$ and $\mathbf{W}_{\text{aa}} \in \mathbb{R}^{128 \times 128}$) across all encoder layers and row attention. Prevents synonymous rate noise ($dS$) from cross-bleeding into or squashing positive selection calls ($dN^+$).
* **Scaled `256d / 8-Head` Capacity**: Features 256-dimensional embeddings, 8 axial attention heads, 4 encoder layers, 4 fused positional/phylogenetic streams, and a 12-threshold CORAL ordinal loss head.
* **Empirical State-of-the-Art Accuracy**:
  * **HIV-1 Reverse Transcriptase (`HIV_RT.nex`)**: **`0.8942 AUC-ROC`**, **`0.5461 Pearson r`** ($p = 1.97 \times 10^{-27}$), and **`0.4503 AUC-PR`** ($5.6\times$ above random baseline).
  * **Bacterial PTS Transporter (`suIII.nex`)**: **`0.8167 AUC-ROC`**, **`0.4064 Spearman \rho`**, and **`0.1752 AUC-PR`** ($5.1\times$ above baseline).
  * **Encephalitis Virus Env (`ENCenv.nex`)**: **`0.7693 AUC-ROC`**, **`0.4348 Pearson r`**.
* **Native False-Positive Rate Control**: Incorporates a 90:10 natural prior training loader, calibrated prior log-odds initialization ($b_0 = -2.1972$), and a **Smooth Expected Value Decoder ($\mathbb{E}[\text{LRT}]$)** to maintain tight FPR control ($\le 0.5\%$) on neutral null alignments.

---

## 🛠️ Installation & Dependencies

### Prerequisites
* Python 3.9 or higher
* PyTorch 2.0+

```bash
git clone https://github.com/veg/TOGA_MEME.git
cd TOGA_MEME
pip install -r requirements.txt
```

`requirements.txt`:
```txt
torch>=2.0.0
numpy>=1.22.0
pandas>=1.5.0
biopython>=1.80
scikit-learn>=1.2.0
scipy>=1.9.0
```

---

## 🚀 Quickstart: Running Inference

Run selection inference on a NEXUS or FASTA alignment using the pre-trained `axomeme_2.0.pt` checkpoint:

```bash
python3 scripts/predict_regression_nexus.py \
  --model axomeme_2.0.pt \
  --alignment path/to/alignment.nex \
  --output predictions.csv
```

### Command Line Arguments

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--alignment` | `str` | *Required* | Path to input alignment NEXUS/FASTA file (`.nex`, `.fasta`, `.gz`). |
| `--model` | `str` | `axomeme_2.0.pt` | Path to trained PyTorch model checkpoint (`.pt`). |
| `--tree` | `str` | `None` | Optional path to Newick/NEXUS tree file. If omitted, uses embedded tree or estimates branch lengths via HyPhy. |
| `--output` | `str` | `[prefix]_regression_predictions.csv` | Path to output predictions CSV file. |
| `--device` | `str` | `cpu` | Execution device: `cpu`, `cuda`, `mps`, or `xla`. |
| `--call_mode` | `str` | `pvalue` | Selection calling gate: `pvalue` (LRT gates), `zscore`, or `percentile`. |
| `--tier1_lrt_gate` | `float` | `4.45` | Absolute LRT cutoff for Tier 1 High-Confidence calls ($p \le 0.05$). |
| `--tier2_lrt_gate` | `float` | `3.12` | Absolute LRT cutoff for Tier 2 Medium-Confidence calls ($p \le 0.10$). |

---

## 🏋️ Training Axomeme 2.0

To train Axomeme 2.0 on Cloud TPUs or multi-GPU systems:

```bash
python3 scripts/train_transformer_selection.py \
  --db_path meme_results.db \
  --msa_dir msa_cache_npz \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 4 \
  --epochs 16 \
  --batch_size 2048 \
  --micro_batch_size 512 \
  --lr 2e-4 \
  --loss_type coral \
  --pure_coral \
  --save_path axomeme_2.0.pt
```

---

## 📄 Documentation

Full architecture specifications and training protocols are available in the [`docs/`](docs/) directory:

* [`docs/axomeme_architecture_spec.pdf`](docs/axomeme_architecture_spec.pdf) - Human-readable PDF specification with mathematical formulations and network architecture diagrams.
* [`docs/axomeme_architecture_spec.md`](docs/axomeme_architecture_spec.md) - Agent-readable Markdown architecture document.
* [`docs/axomeme_training_procedure.md`](docs/axomeme_training_procedure.md) - Agent-readable Markdown training & optimization procedure.
* [`docs/axomeme_training_and_data_guide.md`](docs/axomeme_training_and_data_guide.md) - **Complete Data Pipeline & Training Guide** (includes Dropbox links for SQLite DB & NPZ tensor caches).
* [`docs/axomeme_range_collapse_and_vector_overlap_analysis.pdf`](docs/axomeme_range_collapse_and_vector_overlap_analysis.pdf) - **Technical Analysis PDF** on Range Collapse Mechanics, Vector Projection Overlap ($\mathbf{w}^\top \mathbf{h}_j > 0$), and Block-Diagonal Disentanglement.

---

## 📜 Citation

If you use **Axomeme 2.0** in your research, please cite:

```bibtex
@article{axomeme2026,
  title={Axomeme 2.0: Fast Evolutionary Selection Inference via Deep Axial Attention Transformers},
  author={Kosakovsky Pond, Sergei L. and team},
  journal={Bioinformatics / Molecular Biology and Evolution},
  year={2026}
}
```

