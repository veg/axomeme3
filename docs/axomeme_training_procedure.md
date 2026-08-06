# Axomeme 2.0: Training Procedure & Optimization Protocol

## Executive Overview

This document details the exact training methodology, loss functions, sampling protocol, dataset preparation, and optimization hyperparameters for **Axomeme 2.0**.

---

## 1. Database & Training Corpus

The training database `meme_results.db` was constructed by executing HyPhy MEME across 18,240 empirical gene alignments (including 999 simulated null genes):

* **Total Alignments**: 18,240 coding MSAs
* **Total Codon Sites**: 10,405,375 sites
* **Average Sequence Count ($N$)**: 626.7 species (Median: 682, Max: 735)
* **Ground-Truth Target Values**:
  * $y_{\text{LRT}}$: Maximum Likelihood Ratio Test statistic
  * $y_{\alpha}$: Synonymous rate $\alpha = dS$
  * $y_{\beta}^+$: Positive non-synonymous rate $\beta^+ = dN^+$
  * $y_{p}^+$: Proportion of branches under positive selection $p^+$

---

## 2. Natural Prior DataLoader Sampling Protocol

To prevent positive selection prior inflation ($+3.89$ logit bias), training mini-batches are constructed using **Natural Empirical Prior Stratification**:

### 2.1 Mini-Batch Composition (90% Neutral / 10% Active)
For an epoch target of $M = 262,144$ codon sites:

1. **Neutral Partition (90% = 235,929 sites)**:
   Randomly sampled from sites where $y_{\text{LRT}} = 0.0$ (background neutral evolution).
2. **Active Partition (10% = 26,215 sites)**:
   Sampled evenly across 5 critical LRT threshold bins:
   * **Bin 1 ($0.27 < \text{LRT} \le 2.0$)**: $2.0\%$ of batch (5,243 sites)
   * **Bin 2 ($2.0 < \text{LRT} \le 5.14$)**: $2.0\%$ of batch (5,243 sites)
   * **Bin 3 ($5.14 < \text{LRT} \le 10.0$)**: $2.0\%$ of batch (5,243 sites)
   * **Bin 4 ($10.0 < \text{LRT} \le 20.0$)**: $2.0\%$ of batch (5,243 sites)
   * **Bin 5 ($\text{LRT} > 20.0$)**: $2.0\%$ of batch (5,243 sites)

---

## 3. Loss Functions & Multi-Task Objective

The total loss $\mathcal{L}_{\text{total}}$ is a multi-task combination of Focal CORAL Ordinal Loss and Auxiliary Rate Regression Losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CORAL}} + 0.25 \cdot \mathcal{L}_{\alpha} + 0.25 \cdot \mathcal{L}_{\beta^+} + 0.25 \cdot \mathcal{L}_{p^+}$$

### 3.1 Focal CORAL Ordinal Loss ($\mathcal{L}_{\text{CORAL}}$)
For a target bin $C_i \in \{0, \dots, 12\}$, ordinal binary target indicators are defined as $y_{i,k} = 1$ if $k < C_i$ else $0$:

$$\mathcal{L}_{\text{CORAL}} = -\frac{1}{B} \sum_{i=1}^B \sum_{k=0}^{11} \left[ y_{i,k} (1 - p_{i,k})^\gamma \log(p_{i,k}) + (1 - y_{i,k}) p_{i,k}^\gamma \log(1 - p_{i,k}) \right]$$

where $\gamma = 2.0$ is the Focal modulating factor that focuses gradients on hard, misclassified threshold boundaries.

### 3.2 Auxiliary Rate Huber Losses
Auxiliary heads predict log-transformed parameters $\log(1 + dS)$, $\log(1 + dN^+)$, and $p^+$ using Smooth L1 (Huber) loss with $\delta = 1.0$:

$$\mathcal{L}_{\alpha} = \text{Huber}\left(\hat{y}_{\alpha}, \log(1 + \alpha_i)\right)$$

---

## 4. Hyperparameters & Optimization Setup

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Weight decay = $10^{-4}$ |
| **Learning Rate** | $3 \times 10^{-4}$ | Cosine Annealing schedule with warm-up |
| **Batch Size** | 256 | Micro-batch size = 64 (4 gradient accumulation steps) |
| **Embedding Dimension ($d$)** | 128 | Hidden dimension size |
| **Transformer Blocks** | 4 | Interleaved Column and Phylo-Row Attention blocks |
| **Attention Heads** | 4 | Multi-head self-attention heads |
| **Max Species ($N$)** | 256 / 512 | Dynamic species subsampling cap |
| **Dropout** | 0.10 | Applied to hidden projections during training |
| **Gradient Clipping** | 1.0 | Max gradient norm clipping |
| **Total Epochs** | 20 | Early stopping monitored on validation Spearman $\rho$ |

---

## 5. Evaluation Protocol

Validation performance is monitored after every epoch on an independent split of 1,000 empirical gene alignments:

1. **Validation Spearman Rank Correlation ($\rho_{\text{val}}$)**: Primary ranking performance metric over LRT targets.
2. **Validation LRT Mean Squared Error ($\text{MSE}_{\text{val}}$)**: Measures continuous prediction error.
3. **Null Alignment False Positive Rate (FPR)**: Evaluated on `null_BCL6.replicate.1.nex` (target FPR $\le 1.0\%$).
