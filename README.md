# AxoMeme 3 — Selection Predictor Dashboard

AxoMeme 3 is a client-side, web-based surrogate predictor designed to scan codon sites for evolutionary selection strength directly in the browser. 

It provides an interactive, visual interface for analyzing Multiple Sequence Alignments (MSAs) and phylogenetic trees using pre-trained neural networks.

## 🎯 Goal of the Model

The primary goal of the pre-trained **Phylogenetic Axial Transformer** model is to **predict site-specific episodic diversifying selection strength** (specifically the Likelihood Ratio Test, or **LRT**, statistic calculated by the MEME model in HyPhy) directly from biological sequence data. 

* **Traditional Methods**: Fitting codon substitution models at each site using Maximum Likelihood numerical optimization (e.g. in HyPhy) is computationally intensive and can take hours or days.
* **AxoMeme 3**: Performs the selection scan in **seconds** by running a lightweight neural network (surrogate predictor) locally on the client CPU/GPU, eliminating the expensive optimization phase.

---

## 🛠️ Architecture & Core Features

* **PhyloAxialTransformer**: Alternates attention across rows (species) and columns (codon sites) with a custom learnable **Phylogenetic Bias** to penalize attention weights based on tree-distance (patristic distance).
* **Interactive Manhattan Plot**: Built using HTML5 Canvas, visualizing selection strength across codon positions, showing Tier 1 (High) and Tier 2 (Medium) confidence sites.
* **Entropy Overlays**: Real-time semi-transparent curves representing codon and amino-acid level Shannon entropy (in bits) displayed against a secondary Y-axis.
* **Site-Specific Trees**: Clicking a site (on the Manhattan plot or the Codon Sites table) opens a popup modal with the site's tree, featuring leaf labels with codon/AA states, and **Fitch Parsimony** branch highlighting to indicate substitutions.

---

## 📦 Project Structure

```
axomeme3/
├── index.html               # Single-page web application (HTML, CSS, JS)
├── MEME_transformer.onnx    # Pre-trained ONNX model for client-side execution
├── README.md                # Project documentation
└── .gitignore               # Excludes large compiled HyPhy WASM files
```

---

## ⚙️ Dependencies

AxoMeme 3 relies on several CDNs and external libraries, running entirely client-side:
1. **ONNX Runtime Web** (`ort.min.js`) — For running model inference on the CPU inside web workers.
2. **D3.js** & **Phylotree.js** (v2) — For rendering the phylogenetic trees.
3. **FontAwesome** — For interface iconography.
4. **HyPhy WebAssembly** — For client-side branch length optimization.

> [!IMPORTANT]
> **HyPhy WASM Assets**: The large compiled WebAssembly assets (`hyphy.js`, `hyphy.wasm`, and `hyphy.data`) are **not** committed to this repository. They can be obtained by either:
> 1. Downloading the pre-compiled `hyphy-wasm` artifacts from the latest successful run in [GitHub Actions](https://github.com/veg/hyphy/actions) of the primary [veg/hyphy](https://github.com/veg/hyphy) repository, or
> 2. Building them from source following the WebAssembly build instructions in the main repository.
> 
> Once obtained, place `hyphy.js`, `hyphy.wasm`, and `hyphy.data` in the project root directory during deployment.

---

## 🚀 Deployment Instructions

Because the application is completely static, it can be deployed to any static web hosting platform (e.g., GitHub Pages, Netlify, Apache, Nginx).

### Local Execution
1. Clone this repository (and ensure you place `hyphy.js`, `hyphy.wasm`, and `hyphy.data` in the cloned directory).
2. Run a simple HTTP server in the root directory:
   ```bash
   python3 -m http.server 8666
   ```
3. Open your browser and navigate to `http://localhost:8666`.

### Production Deployment
Sync the static files (excluding git configuration and backups) to your web root directory. For example, to push to `silverback`:
```bash
rsync -avz --exclude=".git" --exclude="*.bak" ./ silverback:/archive/sb-data/shares/web/web/axomeme3/
```
Ensure the directory permissions allow the web server to serve the `.wasm` and `.onnx` files with the correct MIME types.

---

## 🐍 Python Command-Line Inference & Training

You can also run inference (selection scans) and train models directly from the command line using Python.

### Prerequisites
Install the required Python packages:
```bash
pip install torch pandas numpy biopython
```

### Running Inference
The repository contains the pre-trained PyTorch model checkpoint `MEME_transformer_joint.pt` and the inference driver script `predict_regression_nexus.py`.

To run inference on an alignment (and an optional tree):
```bash
python3 predict_regression_nexus.py \
    --alignment path/to/alignment.phy \
    --model MEME_transformer_joint.pt \
    --max_species 256 \
    --output predictions.csv
```

**Options:**
* `--alignment`: Path to the Multiple Sequence Alignment file (supports Gzipped, NEXUS, PHYLIP, or FASTA formats; autodetected).
* `--tree`: (Optional) Path to a separate Newick tree file. If omitted, the script will automatically check for an embedded NEXUS tree in the alignment file, or estimate branch lengths using HyPhy if none are present.
* `--reference_seq`: (Optional) The name of the sequence to use as reference. If omitted, the script automatically picks a standard reference (e.g. human/hg38) or defaults to the first sequence.
* `--max_species`: (Default: `256`) The maximum number of species to include. If the alignment has more species, the exact greedy Phylogenetic Diversity (Faith's PD) maximization algorithm will prune the tree down to this threshold to capture the deepest splits and broadest taxonomic span.
* `--output`: Path to write the output site predictions as a CSV file containing predicted selection probabilities for each codon position.

### Training the Model
The repository also includes `train_transformer_selection.py` which contains the PyTorch dataset loaders, the model architecture `PhyloAxialTransformer`, and the training loop code.
