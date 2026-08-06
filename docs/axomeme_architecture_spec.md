# Axomeme 2.0: Deep Axial Transformer Architecture Specification

## Executive Overview

**Axomeme 2.0** is an axial attention transformer neural network designed for rapid, site-level evolutionary selection inference on multiple sequence alignments (MSAs) of protein-coding nucleotide sequences. By replacing traditional Maximum Likelihood (ML) numerical optimization (such as HyPhy MEME or PAML CODEML) with a high-capacity axial transformer backbone and a rank-consistent ordinal loss head, Axomeme 2.0 infers Likelihood Ratio Test (LRT) selection statistics and evolutionary parameters in under **10 milliseconds per alignment** (a **4,000× speedup** over classical ML).

---

## 1. High-Level System Architecture

```
                                  [ Input MSA Alignment ]
                                 (N Species x L Codons)
                                            │
               ┌────────────────────────────┴───────────────────────────┐
               ▼                                                        ▼
    [ Codon Token Embedding ]                               [ Amino Acid Token Embedding ]
     (64 Codons -> R^64)                                     (20 AAs -> R^64)
               │                                                        │
               └────────────────────────────┬───────────────────────────┘
                                            ▼
                              [ Elementwise Sum + Projection ]
                                    (Dimension d=128)
                                            │
                                            ├────────────────────────────────┐
                                            ▼                                ▼
                              [ 1D Positional Embedding ]      [ 4D MDS Tree Embedding ]
                              (Codon Window Pos -> R^128)      (Tree Coordinates -> R^128)
                                            │                                │
                                            └────────────────┬───────────────┘
                                                             ▼
                                             [ Input Representation Matrix X_0 ]
                                                (Batch x N x L x d=128)
                                                             │
                                                             ▼
                                             ┌───────────────────────────────┐
                                             │  Axial Transformer Backbone   │
                                             │     (4 Interleaved Blocks)    │
                                             │                               │
                                             │  1. Column-Attention (Codons) │
                                             │  2. Phylo-Row-Attention (Taxa)│
                                             └───────────────┬───────────────┘
                                                             ▼
                                               [ Site Representation X_4 ]
                                                  (Batch x N x d=128)
                                                             │
                                                             ▼
                                           [ Multi-Stream Species Pooling ]
                                         (Mean + Max + Softmax Attn + Diff)
                                                             │
                                                             ▼
                                            [ Stream Fusion Projection ]
                                                  (Dimension d=128)
                                                             │
                                                             ▼
                                          [ Rank-Consistent CORAL Head ]
                                              (12 Threshold Cutoffs)
                                                             │
                                                             ▼
                                           [ Smooth Expected Value Decoder ]
                                                E[LRT] = ∑ P(Bin m) * μ_m
```

---

## 2. Input Representations & Tokenization

Given an input alignment matrix of $N$ species (taxa) and $L$ codon sites:

### 2.1 Dual Codon-AA Tokenization
* **Codon Alphabet ($|\mathcal{C}| = 64$)**: Sense codons mapped to indices $0 \dots 63$. Stop codons, gaps (`---`), and unknown nucleotides (`NNN`) are assigned index $64$ (padded / missing token).
* **Amino Acid Alphabet ($|\mathcal{A}| = 20$)**: Standard amino acids mapped to indices $0 \dots 19$. Unknowns / gaps are assigned index $20$.

### 2.2 Embedding Layers
* **Codon Embedding $\mathbf{E}_{\text{codon}} \in \mathbb{R}^{65 \times 64}$**: Learns codon-specific dense vector features.
* **Amino Acid Embedding $\mathbf{E}_{\text{aa}} \in \mathbb{R}^{21 \times 64}$**: Learns amino acid biochemical features.
* Combined Token Vector:
  $$\mathbf{x}_{i,j}^0 = \text{Concat}\left(\mathbf{E}_{\text{codon}}[c_{i,j}], \mathbf{E}_{\text{aa}}[a_{i,j}]\right) + \mathbf{P}_{\text{pos}}[j] + \mathbf{W}_{\text{mds}} \mathbf{m}_i$$
  where $\mathbf{P}_{\text{pos}} \in \mathbb{R}^{L \times 128}$ is a 1D positional embedding and $\mathbf{m}_i \in \mathbb{R}^4$ is the Multidimensional Scaling (MDS) embedding of species $i$ derived from the phylogenetic tree distance matrix.

---

## 3. Axial Transformer Backbone

Axomeme 2.0 uses **4 stacked axial attention blocks**. Each block factorizes self-attention into separate sequence (column) and phylogenetic (row) operations to achieve $O(N \cdot L)$ computational complexity instead of $O(N^2 L^2)$:

### 3.1 Column Attention (Codon Context Layer)
Operates independently along codon positions for each species:
$$\mathbf{H}_{\text{col}} = \text{MultiHeadAttention}\left(\mathbf{X}, \mathbf{X}, \mathbf{X}\right) \in \mathbb{R}^{(B \cdot N) \times L \times 128}$$
Captures local sliding-window context (e.g. neighboring codon correlations, reading frames).

### 3.2 Phylogenetic Row Attention (Taxa Context Layer)
Operates along species for each codon site:
$$\mathbf{H}_{\text{row}} = \text{PhyloRowAttention}\left(\mathbf{X}, \mathbf{X}, \mathbf{X}, \mathbf{D}_{\text{norm}}, \mathbf{M}_{\text{syn}}, \mathbf{M}_{\text{nonsyn}}\right)$$
* **Relative Tree Distance Normalization**: Distance matrix $\mathbf{D}$ is normalized by the alignment-specific mean branch distance $\bar{D}$:
  $$\mathbf{D}_{\text{norm}} = \frac{\mathbf{D}}{\bar{D} + 10^{-4}}$$
* **Synonymous / Non-synonymous Masking**: Pre-computed pairwise genetic code masks ($\mathbf{M}_{\text{syn}}, \mathbf{M}_{\text{nonsyn}}$) inject exact genetic code structure directly into attention bias matrices.

---

## 4. Multi-Stream Species Representation Pooling

To aggregate features across $N$ species for a given codon site $j$ into a single fixed-size site representation $\mathbf{h}_j \in \mathbb{R}^{128}$:

$$\mathbf{h}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_{i,j}, \quad \mathbf{h}_{\text{max}} = \max_{i=1}^N (\mathbf{x}_{i,j})$$
$$\mathbf{h}_{\text{attn}} = \sum_{i=1}^N \text{Softmax}\left(\mathbf{w}_a^\top \mathbf{x}_{i,j}\right) \mathbf{x}_{i,j}, \quad \mathbf{h}_{\text{diff}} = \text{ReLU}\left(\mathbf{h}_{\text{max}} - \mathbf{h}_{\text{mean}}\right)$$

The fused representation is projected back to $d=128$:
$$\mathbf{h}_j = \text{Linear}\left(\text{Concat}\left(\mathbf{h}_{\text{mean}}, \mathbf{h}_{\text{max}}, \mathbf{h}_{\text{attn}}, \mathbf{h}_{\text{diff}}\right)\right) \in \mathbb{R}^{128}$$

---

## 5. Rank-Consistent CORAL Ordinal Head

Axomeme 2.0 frames LRT selection prediction as **Consistent Ordinal Regression (CORAL)** over 12 threshold boundaries:

$$\mathcal{E} = [0.0, 0.2738, 0.7500, 1.2500, 1.8272, 2.4500, 3.1248, 4.4537, 5.7987, 7.5909, 12.1310, 16.6963, 21.2737, 100.0]$$

### 5.1 Monotonic Threshold Parameterization
To guarantee strict rank consistency ($b_0 > b_1 > b_2 > \dots > b_{11}$):
$$b_0 = \text{Unconstrained Base Bias}$$
$$b_k = b_0 - \sum_{j=1}^k \text{Softplus}(\theta_j), \quad \theta_j \in \mathbb{R}$$

Logits for threshold head $k$:
$$z_k = \mathbf{w}^\top \mathbf{h}_j + b_k, \quad p_k = \sigma(z_k) = P(\text{LRT} > E_{k+1})$$

### 5.2 Smooth Expected Value Decoder ($\mathbb{E}[\text{LRT}]$)
Continuous LRT predictions are decoded via smooth discrete density integration over bin midpoints $\mu_m$:
$$P(\text{Bin } 0) = 1.0 - p_0, \quad P(\text{Bin } m) = p_{m-1} - p_m, \quad P(\text{Bin } 12) = p_{11}$$
$$\hat{y}_{\text{LRT}} = \mathbb{E}[\text{LRT}] = \sum_{m=0}^{12} P(\text{Bin } m) \cdot \mu_m$$

This expected value formulation provides **100% continuous, differentiable predictions** without discrete cliff-edge step jumps.

---

## 7. Multi-Dataset Benchmark Performance Summary

Evaluated across 17 empirical benchmark alignments from `hyphy/tests/data`:

| Dataset Alignment | Taxa ($N$) | Codons ($L$) | True Positives ($p \le 0.05$) | **AUC-ROC** | **AUC-PR** | Baseline PR | Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ENCenv.nex`** | 23 | 500 | 1 | **`0.9940`** | **`0.2500`** | `0.0020` | `0.1729` |
| **`flavNS5.nex`** | 18 | 342 | 2 | **`0.9706`** | **`0.1422`** | `0.0058` | `0.2521` |
| **`adh.nex`** | 23 | 254 | 8 | **`0.8740`** | **`0.3416`** | `0.0315` | `0.2959` |
| **`bglobin.nex`** | 17 | 144 | 11 | **`0.8332`** | **`0.3396`** | `0.0764` | `0.3751` |
| **`Bacterial_PTS_suIII.nex`** | 16 | 639 | 9 | **`0.8208`** | **`0.0601`** | `0.0141` | `0.3078` |
| **`vwf.nex`** | 62 | 392 | 9 | **`0.7949`** | **`0.1177`** | `0.0230` | `0.3124` |
| **`lysin.nex`** | 25 | 134 | 26 | **`0.7753`** | **`0.4892`** | `0.1940` | **`0.5574`** |
| **`HepatitisD.nex`** | 33 | 196 | 17 | **`0.7650`** | **`0.2211`** | `0.0867` | `0.3462` |
| **`IAV-human-H1N1-HA.nex`** | 466 | 589 | 24 | **`0.7564`** | **`0.1111`** | `0.0407` | `0.3695` |
| **`rbp3.nex`** | 54 | 412 | 3 | **`0.7563`** | **`0.0336`** | `0.0073` | `0.1335` |
| **`SARS-CoV-2-spike.nex`** | 180 | 1284 | 7 | **`0.7528`** | **`0.0838`** | `0.0055` | `0.2661` |
| **`HIV_RT.nex`** | 476 | 335 | 22 | **`0.7409`** | **`0.2141`** | `0.0657` | `0.2150` |
| **`rbcL.nex`** | 483 | 466 | 83 | **`0.7030`** | **`0.3549`** | `0.1781` | `0.3729` |
| **`HIVvif.nex`** | 29 | 192 | 8 | **`0.6929`** | **`0.1447`** | `0.0417` | `0.4689` |
| **`InfluenzaA.nex`** | 349 | 329 | 17 | **`0.6046`** | **`0.1414`** | `0.0517` | `0.2913` |
| **`camelid.nex`** | 212 | 96 | 29 | **`0.5996`** | **`0.4704`** | `0.3021` | `0.2023` |
| **Overall Mean (16 Datasets)** | — | — | — | **`0.7772`** | **`0.2197`** | `0.0688` | **`0.3029`** |
