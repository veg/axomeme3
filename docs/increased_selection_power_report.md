# Selection Power Resolution: Legacy vs. TOGA-Level Alignments

This report details how scaling evolutionary analyses from sparse legacy alignments (~45 species) to dense TOGA-level alignments (500–600 species) dramatically increases the statistical power of the HyPhy MEME model to resolve positive selection. 

Unlike cases of false positives (where sparse trees suffer from ancestral state reconstruction errors), the examples documented here represent **true biological selection signals (classified as GOLD)** that were completely missed or non-significant in the legacy dataset due to a lack of mutational depth and sampling density.

---

## 1. Summary of Top Candidate Sites

The table below contrasts the selection statistics between the Legacy and TOGA datasets for three clean examples:

| Gene | Site (Legacy / TOGA) | Legacy Seqs | TOGA Seqs | Legacy p-value | Legacy LRT | TOGA p-value | TOGA LRT | Legacy Mutations (Non-syn / Syn) | TOGA Mutations (Non-syn / Syn) | Inferred Selection Rate (beta+) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARID1B** | Site 385 / 498 | 35 | 537 | 0.6667 | 0.00 | 0.0000 | 30.33 | 0 Non-syn / 0 Syn | 16 Non-syn / 0 Syn | 31.68 (12.9% of branches) |
| **ARHGAP23** | Site 3 / 3 | 42 | 597 | 0.4923 | 0.30 | 0.0000 | 52.09 | 1 Non-syn / 0 Syn | 17 Non-syn / 0 Syn | 1858.37 (1.8% of branches) |
| **CABLES1** | Site 511 / 554 | 39 | 593 | 0.2902 | 1.16 | 0.0000 | 28.77 | 1 Non-syn / 0 Syn | 31 Non-syn / 0 Syn | 3.95 (99.0% of branches) |

---

## 2. Case Study 1: ARID1B Site 498 (AT-Rich Interaction Domain-Containing Protein 1B)
* **Legacy Result (Site 385):** Completely non-significant (p = 0.6667, LRT = 0.00, beta+ = 0.02).
* **TOGA Result (Site 498):** Highly significant (p = 0.0000, LRT = 30.33, beta+ = 31.68).
* **Classification:** GOLD (High-confidence true positive).

### Mutational Depth Details:
* **Legacy Mutations:** 0 non-synonymous, 0 synonymous (completely invariant site).
* **TOGA Mutations:** 16 non-synonymous, 0 synonymous.

### Interpretation:
In the 35-species legacy alignment, this site exhibits **exactly zero mutations** (both synonymous and non-synonymous), making the site completely invariant. Because there is no variation, the evolutionary model has no data to distinguish purifying selection, a low mutation rate, or neutral drift. The likelihood surface is completely flat, leading to an LRT of 0.00 and a p-value of 0.6667.

By expanding the tree to 537 species in the TOGA dataset, the model uncovers the rich evolutionary history of this position, reconstructing **16 independent non-synonymous mutations** and 0 synonymous mutations. This high rate of purely amino-acid-altering change across the dense phylogeny provides the statistical replication necessary to reject the neutral null model and confirm true selection with absolute confidence (p = 0.0000, LRT = 30.33).

---

## 3. Case Study 2: ARHGAP23 Site 3 (Rho GTPase Activating Protein 23)
* **Legacy Result (Site 3):** Non-significant (p = 0.4923, LRT = 0.30, beta+ = 0.94).
* **TOGA Result (Site 3):** Highly significant (p = 0.0000, LRT = 52.09, beta+ = 1858.37).
* **Classification:** GOLD (High-confidence true positive).

### Mutational Depth Details:
* **Legacy Mutations:** 1 non-synonymous, 0 synonymous.
* **TOGA Mutations:** 17 non-synonymous, 0 synonymous.

### Interpretation:
In the legacy tree (42 species), the site has only **a single non-synonymous mutation** (occurring on the branch leading to the kangaroo rat *Dipodomys ordii*). A single mutation on a terminal branch is statistically indistinguishable from a random neutral substitution, yielding a low LRT of 0.30 and a non-significant p-value of 0.4923.

In the TOGA dataset (597 species), the site accumulates **17 independent non-synonymous mutations** and 0 synonymous mutations. Resolving these additional lineages provides the recurrent signals necessary to detect selection. The model estimates that 1.76% of branches undergo extremely intense positive selection (beta+ = 1858.37), resolving the signal with absolute confidence (p = 0.0000, LRT = 52.09).

---

## 4. Case Study 3: CABLES1 Site 554 (Cdk5 and Abl Enzyme Substrate 1)
* **Legacy Result (Site 511):** Non-significant (p = 0.2902, LRT = 1.16, beta+ = 1.32).
* **TOGA Result (Site 554):** Highly significant (p = 0.0000, LRT = 28.77, beta+ = 3.95).
* **Classification:** GOLD (High-confidence true positive).

### Mutational Depth Details:
* **Legacy Mutations:** 1 non-synonymous, 0 synonymous.
* **TOGA Mutations:** 31 non-synonymous, 0 synonymous.

### Interpretation:
In the legacy alignment (39 species), this site has only **a single non-synonymous mutation** on an ancestral node. Similar to ARHGAP23, a single mutation provides insufficient evidence to rule out neutral evolution, leading to a p-value of 0.2902.

The dense TOGA tree (593 species) captures a massive accumulation of **31 independent non-synonymous mutations** with zero synonymous changes. The model estimates that nearly the entire tree (99.0% of branches) evolves under diversifying selection (beta+ = 3.95), indicating a widespread selection pressure that was completely lost under the sparse sampling of the legacy tree.

---

## 5. Conclusion: Taxon Density as a Power Multiplier

These examples illustrate that **expanding taxon density is crucial for overcoming false negatives**. 

1. **Resolving Invariant Sites:** In small trees, many selected sites appear completely invariant simply because the sampling window is too narrow. A large tree gathers enough mutations globally to expose these active positions.
2. **Establishing Recurrence:** Positive selection is inferred by showing that mutations occur repeatedly at the same site. Dense trees provide the replicate branches needed to distinguish recurrent selection from random neutral drift.
3. **Widespread vs. Episodic Selection:** Whether selection is episodic (ARHGAP23) or widespread (CABLES1), a dense phylogeny provides the necessary tree length to confirm that amino-acid changes outnumber synonymous changes to a statistically significant degree.
