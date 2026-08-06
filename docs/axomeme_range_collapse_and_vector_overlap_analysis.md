# Mathematical Mechanics of Range Collapse, Vector Overlap, and Ultra-Episodic Selection in Axomeme 2.0

## Executive Abstract

This document details the mathematical and structural mechanisms behind three core phenomena in deep evolutionary selection transformers:

1. **Epoch 1 Range Collapse**: Under 90:10 natural prior training when un-converged neural rate head predictions create an all-site physical penalty feedback loop.
2. **Vector Projection Overlap ($\mathbf{w}^\top \mathbf{h}_j > 0$)**: Where high synonymous rate neutral sites ($dS \gg 0$) produce non-zero difference vectors ($\mathbf{h}_{\text{diff}} \gg \mathbf{0}$) that project positively along selection weight vector $\mathbf{w}$.
3. **Ultra-Episodic Single-Branch Selection ($p^+ \le 2\%$)**: Where mean species pooling over hundreds of wild-type taxa suppresses isolated single-branch mutational bursts.

---

## 1. Root Cause of Epoch 1 Range Collapse

### 1.1 The Un-converged Rate Prediction Feedback Loop
Under a 90:10 natural prior stratification, 90% of all mini-batch training targets are $0.0$. The Focal-CORAL Binary Cross-Entropy (BCE) loss applies a strong downward gradient to ordinal threshold logits ($z_k \to -\infty$).

Before our optimization fix, the biological physical penalty was formulated using the model's own predictions:

$$\mathcal{L}_{\text{physics\_legacy}} = \text{ReLU}\left(\hat{y}_{\text{LRT}}\right) \cdot \text{ReLU}\left(\hat{\alpha}_{\text{pred}} - \hat{\beta}^+_{\text{pred}}\right)$$

At Epoch 1, the auxiliary rate heads $\hat{\alpha}_{\text{pred}}$ and $\hat{\beta}^+_{\text{pred}}$ start near initialization defaults ($\hat{\alpha} \approx 1.31, \hat{\beta}^+ \approx 0.50$). Consequently:

$$\text{ReLU}(\hat{\alpha}_{\text{pred}} - \hat{\beta}^+_{\text{pred}}) > 0$$

evaluated to a positive penalty across **100% of training sites** (including true positive selection sites where ground-truth $dN^+ > dS$).

This created a catastrophic feedback loop: un-converged auxiliary rate predictions penalized non-zero LRT predictions ($\hat{y}_{\text{LRT}} > 0$) across every single site in the database. Combined with the 90% neutral prior, all 12 CORAL logits $z_k$ collapsed to $-\infty$, restricting predicted raw LRTs to $\le 2.12$.

### 1.2 The Ground-Truth Rate Target Fix
We resolved this feedback loop by anchoring the physical rate violation penalty to verified ground-truth targets from `meme_results.db`:

$$m_{\text{physics}} = \mathbb{I}\left(y_{\beta_{\text{pos}}, \text{true}} \le y_{\alpha, \text{true}}\right)$$

$$\mathcal{L}_{\text{physics}} = \frac{1}{B} \sum_{i=1}^B \left[ m_i \cdot (\hat{y}_{\text{LRT}, i})^2 + m_{\text{physics}, i} \cdot (\hat{y}_{\text{LRT}, i})^2 \right]$$

Because $m_{\text{physics}}$ is a constant derived directly from the database, it never fires on true positive selection sites where $dN^+ > dS$, eliminating the feedback loop and preserving dynamic range from Epoch 1 onward.

---

## 2. Vector Projection Overlap ($\mathbf{w}^\top \mathbf{h}_j > 0$)

### 2.1 Embedding Geometry & Feature Superposition
Every codon position $j$ for species $i$ is embedded as a dual concatenation:

$$\mathbf{x}_{i,j}^0 = \text{Concat}\left(\mathbf{E}_{\text{codon}}[c_{i,j}], \mathbf{E}_{\text{aa}}[a_{i,j}]\right) \in \mathbb{R}^{128}$$

Consider three site types:

1. **Invariant Site** (All species `TTA` / Leucine): All taxa vectors $\mathbf{x}_{i,j}$ are identical.
2. **High Synonymous Rate Site** ($dS = 10.0, dN = 0$, all species Leucine, but using 6 synonymous codons `TTA`, `TTG`, `CTT`, `CTC`, `CTA`, `CTG`): Amino acid tokens are identical, but codon tokens vary significantly across taxa.
3. **Positive Selection Site** ($dN > dS$, e.g. `L`, `V`, `I`, `M`, `R`, `E`): Both amino acid tokens and codon tokens vary across taxa.

### 2.2 Species Difference Stream ($\mathbf{h}_{\text{diff}}$)
The species pooling layer computes a difference representation:

$$\mathbf{h}_{\text{diff}} = \text{ReLU}\left(\mathbf{h}_{\text{max}} - \mathbf{h}_{\text{mean}}\right)$$

* On an invariant site, $\mathbf{h}_{\text{max}} \approx \mathbf{h}_{\text{mean}}$, so $\mathbf{h}_{\text{diff}} \approx \mathbf{0}$.
* On a positive selection site ($dN > dS$), taxa representations diverge, producing $\mathbf{h}_{\text{diff}} \gg \mathbf{0}$.
* However, on a high synonymous rate site ($dS \gg 0$), the 6 synonymous codons also cause taxa representations to diverge in codon embedding space ($\mathbf{E}_{\text{codon}}$), producing a large non-zero vector $\mathbf{h}_{\text{diff}} \gg \mathbf{0}$.

Because the linear weight vector $\mathbf{w}$ in `RankConsistentCoralHead` assigns positive weights to variance in $\mathbf{h}_{\text{diff}}$, high synonymous rate neutral sites project positively along $\mathbf{w}$:

$$\mathbf{w}^\top \mathbf{h}_j > 0 \implies z_0 > 0 \implies \text{False Positive Call}$$

Enforcing $\mathcal{L}_{\text{physics}}$ penalizes non-zero predictions when $dN \le dS$, forcing $\mathbf{w}$ to orthogonalize against pure synonymous codon variance.

---

## 3. Ultra-Episodic Selection ($p^+ \le 2\%$)

### 3.1 The Species Pooling Dilemma on Large Trees
In HyPhy MEME, episodic selection is modeled via a two-component mixture $\beta \sim (1 - p^+) \delta_{\beta^1} + p^+ \delta_{\beta^+}$. When $p^+ \le 0.02$ on a tree of $N = 476$ taxa, only 1 to 5 species undergo a non-synonymous substitution, while 471 species carry the identical wild-type amino acid.

When pooling representations across 476 species:

$$\mathbf{h}_{\text{mean}} = \frac{471}{476} \mathbf{u}_{\text{wt}} + \frac{5}{476} \mathbf{u}_{\text{variant}} \approx 0.989 \mathbf{u}_{\text{wt}}$$

The overwhelming background of 471 wild-type species dominates mean and attention pooling ($\mathbf{h}_{\text{mean}}, \mathbf{h}_{\text{attn}}$), projecting a site vector $\mathbf{h}_j$ that appears 99% identical to a conserved neutral site.

### 3.2 Architectural Solution: Top-$K$ Extreme Difference Pooling
We resolved this by introducing a Top-$K$ Extreme Difference Pooling stream ($K = 8$):

$$d_i = \|\mathbf{x}_{i,j} - \mathbf{h}_{\text{mean}}\|_2, \quad \mathbf{h}_{\text{top\_k}} = \frac{1}{K} \sum_{k \in \text{TopK}} \mathbf{x}_{k,j}$$

$$\mathbf{h}_j = \text{Linear}\left(\text{Concat}\left(\mathbf{h}_{\text{mean}}, \mathbf{h}_{\text{max}}, \mathbf{h}_{\text{attn}}, \mathbf{h}_{\text{diff}}, \mathbf{h}_{\text{top\_k}}\right)\right)$$

By extracting the mean of the top $K=8$ most deviant species representations, the pooling head isolates single-branch mutational spikes regardless of overall tree size $N$.
