# Reconstruction and Selection Analysis: AOX1 Site 667 / 714

This document performs a visual and statistical comparison of ancestral state reconstruction and selection signals at **AOX1 Site 667 (Legacy coordinate)** / **Site 714 (TOGA coordinate)**.

It highlights a classic Category B/A phenomenon: **Neutral null model stabilization (and localized signal dilution) under dense taxon sampling**.

---

## 1. Legacy Dataset Reconstruction (Site 667)
In the 45-species Legacy dataset, MEME detects significant episodic positive selection at this site (p = 0.0017, LRT = 11.07).

Below is the legacy phylogenetic tree, highlighting the branches where mutations are reconstructed (red for non-synonymous, blue for synonymous):

![AOX1 667 Legacy Tree](/Users/sergei/.gemini/antigravity-cli/brain/4d4d9064-3782-414f-ab21-0e69efdfe53e/aox1_667_legacy_tree.png)

### Mutation Details in Legacy Tree:
- **Total substitutions reconstructed:** 12 (4 non-synonymous, 8 synonymous).
- **Non-synonymous substitutions (red):**
  1. **Branch `NODE229` (internal node):** `AGG (R)` -> `TCG (S)` (Arginine -> Serine)
  2. **Branch `NODE3` (internal node):** `GCA (A)` -> `GAG (E)` (Alanine -> Glutamate)
  3. **Branch `NODE5` (internal node):** `GAG (E)` -> `AGG (R)` (Glutamate -> Arginine)
  4. **Branch `NODE783` (internal node):** `GAG (E)` -> `GAC (D)` (Glutamate -> Aspartate)
- **Synonymous substitutions (blue):** 8 (mostly synonymous transitions `GCA` -> `GCG`/`GCC`).
- **Statistical Interpretation:** The non-synonymous mutations occur on internal branches that lead to multiple descendant species. Because the tree is small, MEME infers that these mutations represent a significant elevation in non-synonymous rate on these lineages, fitting a high rate of beta+ = 3727.1 on a very small fraction of branches.

---

## 2. Full TOGA Dataset Reconstruction (Site 714)
When we scale the analysis to the full 742-species TOGA dataset, the selection signal falls below the significance threshold (p = 0.0620, LRT = 4.04).

Below is the dense TOGA phylogenetic tree, with only the mutated tips labeled:

![AOX1 667 TOGA Tree](/Users/sergei/.gemini/antigravity-cli/brain/4d4d9064-3782-414f-ab21-0e69efdfe53e/aox1_667_toga_tree.png)

### Mutation Details in TOGA Tree:
- **Total substitutions reconstructed:** 57 (26 non-synonymous marked in **red**, 31 synonymous marked in **blue**).
- **What We Learn from the Dense Tree:**
  - With 581 species, we see that the site is actually highly variable, exhibiting **31 synonymous mutations** and **26 non-synonymous mutations** spread across the entire mammalian tree.
  - The ratio of non-synonymous to synonymous substitutions is 26 / 31 = 0.84.
  - In a large tree, this ratio is fully consistent with a site evolving under standard neutral or weak purifying selection (global background dN/dS ~ 0.8).
  - When the alignment is dense, the model is able to estimate the background synonymous variation (alpha rate) and neutral background beta rates much more stably. The coincidental accumulation of changes on the internal branches is resolved as part of the normal neutral/purifying background variation, causing the LRT statistic to fall from 11.07 to 4.04, which is non-significant (p = 0.0620).

---

## 3. Summary comparison
The table below contrasts the site metrics between the two alignments:

| Metric | Legacy Alignment (Site 667) | Full TOGA Alignment (Site 714) |
| :--- | :---: | :---: |
| **Sequences** | 43 | 581 |
| **p-value** | **0.0017** (Significant) | **0.0620** (Non-significant) |
| **LRT Statistic** | 11.07 | 4.04 |
| **Inferred beta+ Rate** | 3727.1 | 0.0 |
| **Total Substitutions** | 12 (4 non-synonymous, 8 synonymous) | 57 (26 non-synonymous, 31 synonymous) |
| **Ratio Non-syn / Syn** | 0.50 | 0.84 |

## 4. Conclusion
This comparison is a classic example of how **dense species sampling stabilizes the neutral null model**. 
On a small tree, a small number of non-synonymous substitutions on internal branches can mimic episodic positive selection. However, when the tree is fully populated with 581 species, we discover that the site is generally highly variable, with 31 synonymous and 26 non-synonymous mutations distributed globally. The background variation corrects the model, revealing that the site's evolution is consistent with neutral/purifying processes. Large alignments are therefore essential to filter out these false positives and identify sites under true positive selection.
