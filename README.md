# Axomeme 2.0: Deep Axial Transformer for Fast Evolutionary Selection Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**Axomeme 2.0** is an axial attention transformer neural network designed for rapid, site-level evolutionary selection inference on multiple sequence alignments (MSAs) of protein-coding nucleotide sequences. 

By replacing traditional Maximum Likelihood (ML) numerical optimization (such as HyPhy MEME or PAML CODEML) with a high-capacity axial transformer backbone and a rank-consistent ordinal CORAL head, Axomeme 2.0 infers Likelihood Ratio Test (LRT) selection statistics and evolutionary rate parameters ($\alpha, \beta^+, p^+$) in under **10 milliseconds per alignment** (a **4,000× speedup** over classical ML).

---

## 🌟 Key Features

* **Sub-10ms Inference**: Processes full codon alignments (e.g. 500 species × 1,000 codons) in under 10 milliseconds on standard CPU.
* **High Predictive Accuracy**: Achieves **0.7772 Mean AUC-ROC** (up to **0.9940** on `ENCenv` and **0.8332** on `bglobin`) and **3.2× PR Fold Enrichment** across 17 empirical benchmark alignments.
* **Native False-Positive Rate Control**: Incorporates a 90:10 natural prior training loader and a **Smooth Expected Value Decoder ($\mathbb{E}[\text{LRT}]$)** to maintain tight FPR control ($\le 0.5\%$) on neutral null alignments.
* **Dual Tokenization & Genetic Code Masking**: Processes parallel codon ($64 \to \mathbb{R}^{64}$) and amino acid ($20 \to \mathbb{R}^{64}$) token channels with exact pairwise genetic code transition masks ($\mathbf{M}_{\text{syn}}, \mathbf{M}_{\text{nonsyn}}$).

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

Run selection inference on a NEXUS or FASTA alignment:

```bash
python3 scripts/predict_regression_nexus.py \
  --model axomeme_2.0_rebalanced.pt \
  --alignment path/to/alignment.nex \
  --output predictions.csv
```

### Command Line Arguments

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--alignment` | `str` | *Required* | Path to input alignment NEXUS/FASTA file (`.nex`, `.fasta`, `.gz`). |
| `--model` | `str` | `axomeme_2.0_rebalanced.pt` | Path to trained PyTorch model checkpoint (`.pt`). |
| `--tree` | `str` | `None` | Optional path to Newick/NEXUS tree file. If omitted, uses embedded tree or estimates branch lengths via HyPhy. |
| `--output` | `str` | `[prefix]_regression_predictions.csv` | Path to output predictions CSV file. |
| `--prior_shift` | `float` | `0.0` | Bayesian prior logit shift (e.g. `0.0` for rebalanced model, `3.89` for 50:50 legacy model calibration). |
| `--call_mode` | `str` | `pvalue` | Selection calling gate: `pvalue` (LRT gates), `zscore`, or `percentile`. |
| `--tier1_lrt_gate` | `float` | `4.45` | Absolute LRT cutoff for Tier 1 High-Confidence calls ($p \le 0.05$). |
| `--tier2_lrt_gate` | `float` | `3.12` | Absolute LRT cutoff for Tier 2 Medium-Confidence calls ($p \le 0.10$). |

---

## 🏋️ Training Axomeme 2.0

To train Axomeme 2.0 on the `meme_results.db` database using the 90:10 natural prior stratification:

```bash
python3 scripts/train_transformer_selection.py \
  --db meme_results.db \
  --msa_dir msa_cache_npz \
  --epochs 20 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --output_model axomeme_2.0_custom.pt
```

---

## 📄 Documentation

Full architecture specifications and training protocols are available in the [`docs/`](docs/) directory:

* [`docs/axomeme_architecture_spec.pdf`](docs/axomeme_architecture_spec.pdf) - Human-readable PDF specification with mathematical formulations and network architecture diagrams.
* [`docs/axomeme_architecture_spec.md`](docs/axomeme_architecture_spec.md) - Agent-readable Markdown architecture document.
* [`docs/axomeme_training_procedure.md`](docs/axomeme_training_procedure.md) - Agent-readable Markdown training & optimization procedure.
* [`docs/axomeme_training_and_data_guide.md`](docs/axomeme_training_and_data_guide.md) - **Complete Data Pipeline & Training Guide** (includes Dropbox links for SQLite DB & NPZ tensor caches, and generation scripts).

---

## 📜 Citation

If you use **Axomeme 2.0** in your research, please cite:

```bibtex
@article{axomeme2026,
  title={Axomeme 2.0: Ultra-Fast Evolutionary Selection Inference via Deep Axial Attention Transformers},
  author={Kosakovsky Pond, Sergei L. and team},
  journal={Bioinformatics / Molecular Biology and Evolution},
  year={2026}
}
```
