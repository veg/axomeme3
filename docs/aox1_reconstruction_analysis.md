# Reconstruction and Selection Analysis: AOX1 Site 697 / 744

This document performs a visual and statistical comparison of ancestral state reconstruction and selection signals at **AOX1 Site 697 (Legacy coordinate)** / **Site 744 (TOGA coordinate)**. 

It highlights a classic Category A error: a **False Positive driven by small-tree ancestral reconstruction error (phylogenetic under-sampling)**.

---

## 1. Legacy Dataset Reconstruction (Site 697)
In the 45-species Legacy dataset, MEME detects strong episodic positive selection at this site (p = 0.0002, LRT = 15.13).

Below is the legacy phylogenetic tree, highlighting the branches where mutations are reconstructed:

![AOX1 Legacy Tree](/Users/sergei/.gemini/antigravity-cli/brain/4d4d9064-3782-414f-ab21-0e69efdfe53e/aox1_legacy_tree.png)

### Mutation Details in Legacy Tree:
- **Total substitutions reconstructed:** 2 (both non-synonymous, marked in **red**).
  1. **Branch `HETGLA` (Naked mole-rat):** `AAG (K)` -> `CGC (R)` (Lysine -> Arginine, a **2-nucleotide change**).
  2. **Branch `NODE828` (Internal ancestor):** `AAG (K)` -> `AAC (N)` (Lysine -> Asparagine).
- **Synonymous substitutions:** 0.
- **Statistical Interpretation:** The presence of 2 non-synonymous mutations and 0 synonymous mutations on a small tree causes MEME to infer that the site is highly constrained against neutral change but undergoes intense episodic selection on these branches, fitting a high selection rate of beta+ = 117.8.

---

## 2. Full TOGA Dataset Reconstruction (Site 744)
When we scale the analysis to the full 742-species TOGA dataset, the selection signal disappears completely (p = 0.6667, LRT = 0.00).

Below is the dense TOGA phylogenetic tree, with only the mutated tips labeled for readability:

![AOX1 TOGA Tree](/Users/sergei/.gemini/antigravity-cli/brain/4d4d9064-3782-414f-ab21-0e69efdfe53e/aox1_toga_tree.png)

### Mutation Details in TOGA Tree:
- **Total substitutions reconstructed:** 15 (10 non-synonymous marked in **red**, 5 synonymous marked in **blue**).
- **The Key Correction on `HETGLA`:**
  - In the full tree, the branch leading to `HETGLA` (Naked mole-rat) is reconstructed as a **synonymous transition** (`CGG (R)` -> `CGC (R)`).
  - This is because the full TOGA tree contains closely related mole-rats (such as `fukDam` and `batSui`). The presence of these sister taxa allows the model to correctly resolve the ancestral state as `CGG (R)`.
  - In the legacy alignment, because no closely related mole-rats were present, the model had to reconstruct the ancestral state over a much wider phylogenetic distance, misidentifying it as `AAG (K)`. This caused the simple 1-nucleotide synonymous transition (`CGG` -> `CGC`) to be misclassified as a 2-nucleotide non-synonymous mutation (`AAG` -> `CGC`).
- **Statistical Interpretation:** The addition of 5 synonymous mutations across other Eutherian clades establishes a neutral background rate. Combined with the corrected branch mutation on `HETGLA`, the substitution pattern is fully consistent with neutral expectation (dN/dS ~ 1.0).

---

## 3. Summary comparison
The table below contrasts the site metrics between the two alignments:

| Metric | Legacy Alignment (Site 697) | Full TOGA Alignment (Site 744) |
| :--- | :---: | :---: |
| **Sequences** | 43 | 581 |
| **p-value** | **0.0002** (Significant) | **0.6667** (Non-significant) |
| **LRT Statistic** | 15.13 | 0.00 |
| **Inferred beta+ Rate** | 117.8 | 0.0 |
| **Reconstructed Mutations** | 2 non-synonymous, 0 synonymous | 10 non-synonymous, 5 synonymous |
| **`HETGLA` Mutation Type** | **Non-synonymous** (`AAG (K)` -> `CGC (R)`) | **Synonymous** (`CGG (R)` -> `CGC (R)`) |

## 4. Conclusion
This sanity check confirms that **taxon density is a vital shield against false positives**. Without dense representation of sister lineages, wide-branch ancestral reconstructions are prone to error. A single misreconstructed codon state can easily create a false multi-nucleotide selection signal on a small tree, which is immediately corrected when the dense TOGA dataset is utilized.
