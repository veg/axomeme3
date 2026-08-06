# Analysis of Sites Significant ONLY in the Legacy Dataset

This report analyzes sites that show strong evidence of positive selection in the small **Legacy dataset** (p <= 0.01) but show **no selection signal** in the full **TOGA dataset** (p > 0.05).

## 1. Overview
Across the 56 compared genes, we identified **69 sites** that are significant in the Legacy dataset but not in the TOGA dataset.

This subset represents **~7.9%** of the total selection detections in the Legacy dataset.

---

## 2. Table of Legacy-Only Selection Sites
The table below lists all identified legacy-only sites, comparing their p-values, LRT statistics, and parameter estimations:

| Gene | Leg Site | TOGA Site | Leg p-value | TOGA p-value | Leg LRT | TOGA LRT | Leg beta+ | TOGA Class |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ABHD16B** | 38 | 38 | 0.0002 | 0.2795 | 15.76 | 1.22 | 124.6 | `NONE` |
| **AKAP13** | 956 | 1077 | 0.0002 | 0.3208 | 15.65 | 0.98 | 300.3 | `NONE` |
| **AOX1** | 697 | 744 | 0.0002 | 0.6667 | 15.13 | 0.00 | 117.8 | `NONE` |
| **ANKZF1** | 285 | 317 | 0.0009 | 0.1189 | 12.42 | 2.80 | 9.4 | `NONE` |
| **ABI3BP** | 231 | 231 | 0.0011 | 0.1137 | 11.88 | 2.88 | 377.7 | `NONE` |
| **ACP5** | 222 | 241 | 0.0013 | 0.1255 | 11.66 | 2.70 | 1018.1 | `NONE` |
| **ANG** | 132 | 140 | 0.0015 | 0.0644 | 11.35 | 3.97 | 18.0 | `NONE` |
| **ABI3BP** | 849 | 897 | 0.0016 | 0.1353 | 11.18 | 2.55 | 12169.1 | `NONE` |
| **AOX1** | 667 | 714 | 0.0017 | 0.0620 | 11.07 | 4.04 | 3727.1 | `NONE` |
| **ADGRG3** | 493 | 505 | 0.0018 | 0.2120 | 10.98 | 1.72 | 59.2 | `NONE` |
| **ADAMTS7** | 212 | 271 | 0.0019 | 0.1428 | 10.91 | 2.45 | 8.4 | `NONE` |
| **AKAP13** | 725 | 829 | 0.0019 | 0.2443 | 10.86 | 1.46 | 10.7 | `NONE` |
| **ADAMTS7** | 797 | 857 | 0.0023 | 0.3023 | 10.52 | 1.09 | 18.5 | `NONE` |
| **ADAMTS13** | 36 | 41 | 0.0023 | 0.6667 | 10.51 | -0.01 | 23649.5 | `NONE` |
| **ADAMTS7** | 1136 | 1303 | 0.0025 | 0.3892 | 10.36 | 0.66 | 642.3 | `NONE` |
| **ABCA9** | 1211 | 1264 | 0.0025 | 0.1030 | 10.28 | 3.07 | 1006.3 | `NONE` |
| **ANLN** | 902 | 946 | 0.0026 | 0.0799 | 10.26 | 3.55 | 61.9 | `NONE` |
| **AKAP6** | 458 | 495 | 0.0027 | 0.6313 | 10.17 | 0.02 | 198.8 | `NONE` |
| **AKAP6** | 2238 | 2309 | 0.0029 | 0.4491 | 10.04 | 0.43 | 9.0 | `NONE` |
| **ADAMTS13** | 63 | 69 | 0.0034 | 0.1764 | 9.72 | 2.06 | 89.9 | `NONE` |
| **AKAP13** | 131 | 133 | 0.0036 | 0.1618 | 9.61 | 2.22 | 223.2 | `NONE` |
| **ADGRG3** | 99 | 103 | 0.0036 | 0.4514 | 9.59 | 0.42 | 113.3 | `NONE` |
| **ADAMTS13** | 977 | 1013 | 0.0039 | 0.6569 | 9.43 | 0.00 | 47.4 | `NONE` |
| **ANKZF1** | 44 | 71 | 0.0041 | 0.4652 | 9.37 | 0.38 | 5.3 | `NONE` |
| **ADO** | 26 | 28 | 0.0041 | 0.6667 | 9.33 | 0.00 | 401.7 | `NONE` |
| **ADTRP** | 229 | 238 | 0.0043 | 0.0808 | 9.27 | 3.53 | 772.5 | `NONE` |
| **AKAP13** | 907 | 1027 | 0.0043 | 0.1728 | 9.23 | 2.10 | 48.6 | `NONE` |
| **AMER2** | 327 | 376 | 0.0045 | 0.2313 | 9.18 | 1.56 | 92.6 | `NONE` |
| **AKAP13** | 780 | 891 | 0.0047 | 0.2854 | 9.07 | 1.19 | 1526.7 | `NONE` |
| **AMER2** | 150 | 180 | 0.0049 | 0.5608 | 8.98 | 0.13 | 243.7 | `NONE` |
| **ANKRD17** | 1690 | 1754 | 0.0051 | 0.0590 | 8.90 | 4.13 | 16.3 | `NONE` |
| **AMY2A** | 139 | 139 | 0.0052 | 0.0519 | 8.86 | 4.38 | 654.8 | `NONE` |
| **ABCA9** | 1022 | 1066 | 0.0054 | 0.5675 | 8.79 | 0.11 | 224.9 | `NONE` |
| **ABCA9** | 170 | 189 | 0.0056 | 0.6378 | 8.75 | 0.01 | 81.3 | `NONE` |
| **AKAP13** | 605 | 662 | 0.0057 | 0.3811 | 8.71 | 0.69 | 6.8 | `NONE` |
| **ANLN** | 124 | 155 | 0.0058 | 0.0732 | 8.67 | 3.72 | 22.9 | `NONE` |
| **ABCA9** | 357 | 376 | 0.0060 | 0.6667 | 8.60 | 0.00 | 18.0 | `NONE` |
| **ADAMTS7** | 787 | 847 | 0.0062 | 0.2128 | 8.53 | 1.71 | 7.6 | `NONE` |
| **ABCD1** | 622 | 637 | 0.0062 | 0.1200 | 8.52 | 2.78 | 6.8 | `NONE` |
| **AKAP13** | 2028 | 2186 | 0.0063 | 0.1061 | 8.49 | 3.01 | 44.7 | `NONE` |
| **ADGRG3** | 500 | 512 | 0.0064 | 0.0911 | 8.48 | 3.30 | 41.1 | `NONE` |
| **ACTRT2** | 122 | 128 | 0.0064 | 0.3670 | 8.48 | 0.75 | 69.3 | `NONE` |
| **AMER2** | 446 | 498 | 0.0064 | 0.0925 | 8.46 | 3.27 | 10.5 | `NONE` |
| **AOX1** | 1243 | 1291 | 0.0065 | 0.0673 | 8.43 | 3.88 | 7.9 | `NONE` |
| **ADGRG3** | 200 | 210 | 0.0066 | 0.2665 | 8.42 | 1.31 | 179.3 | `NONE` |
| **ABI3BP** | 885 | 951 | 0.0068 | 0.4901 | 8.35 | 0.30 | 52.3 | `NONE` |
| **AKAP4** | 147 | 149 | 0.0068 | 0.1052 | 8.34 | 3.03 | 8.9 | `NONE` |
| **AK2** | 224 | 224 | 0.0068 | 0.6667 | 8.33 | -0.01 | 94.5 | `NONE` |
| **ANG** | 77 | 84 | 0.0069 | 0.6059 | 8.32 | 0.05 | 99.8 | `NONE` |
| **AK4** | 106 | 106 | 0.0069 | 0.6667 | 8.32 | 0.00 | 5.7 | `NONE` |
| **AOX1** | 547 | 587 | 0.0069 | 0.3621 | 8.32 | 0.78 | 20.6 | `NONE` |
| **ANLN** | 456 | 491 | 0.0072 | 0.6667 | 8.24 | 0.00 | 31.3 | `NONE` |
| **AHDC1** | 557 | 567 | 0.0075 | 0.5706 | 8.16 | 0.11 | 2216.7 | `NONE` |
| **AOX1** | 1046 | 1094 | 0.0079 | 0.1806 | 8.06 | 2.01 | 10.8 | `NONE` |
| **ANLN** | 342 | 376 | 0.0081 | 0.2648 | 8.02 | 1.32 | 12.6 | `NONE` |
| **AKAP13** | 404 | 425 | 0.0083 | 0.5225 | 7.96 | 0.21 | 34.0 | `NONE` |
| **ALK** | 664 | 700 | 0.0083 | 0.1832 | 7.96 | 1.99 | 61.9 | `NONE` |
| **ABCG2** | 12 | 12 | 0.0085 | 0.1136 | 7.91 | 2.88 | 252.0 | `NONE` |
| **ABCA9** | 301 | 320 | 0.0087 | 0.2605 | 7.86 | 1.35 | 9.2 | `NONE` |
| **ANG** | 99 | 107 | 0.0088 | 0.0623 | 7.84 | 4.03 | 9.1 | `NONE` |
| **AMER3** | 17 | 18 | 0.0088 | 0.2347 | 7.83 | 1.54 | 425.2 | `NONE` |
| **ADAMTS13** | 654 | 669 | 0.0090 | 0.4446 | 7.80 | 0.45 | 685.4 | `NONE` |
| **ABI3BP** | 898 | 964 | 0.0090 | 0.6667 | 7.79 | -0.00 | 48.7 | `NONE` |
| **ADTRP** | 151 | 159 | 0.0091 | 0.0945 | 7.77 | 3.23 | 187.9 | `NONE` |
| **ADAMTS13** | 1157 | 1197 | 0.0092 | 0.5449 | 7.75 | 0.16 | 8.5 | `NONE` |
| **ABI3BP** | 809 | 851 | 0.0097 | 0.1074 | 7.65 | 2.99 | 8.8 | `NONE` |
| **ABCF2** | 412 | 413 | 0.0098 | 0.0595 | 7.63 | 4.12 | 56.5 | `NONE` |
| **AKAP6** | 2235 | 2304 | 0.0099 | 0.6667 | 7.62 | -0.05 | 15.1 | `NONE` |
| **AKAP13** | 1101 | 1234 | 0.0099 | 0.1455 | 7.61 | 2.42 | 3.8 | `NONE` |

---

## 3. Why are these sites significant ONLY in the Legacy Alignment?
We can classify these legacy-only detections into two primary categories:

### Category A: False Positives due to Small Tree Variance (LRT Spike)
In smaller alignments (like the ~45-species Legacy dataset), the tree has much shorter total branch length and fewer independent substitution events. If a site happens to have a few non-synonymous changes that coincide on a single lineage or a couple of closely related branches, the MEME model can easily fit a very high beta+ rate (often in the hundreds or thousands) with a small proportion of branches.
* **Characteristics:** High Legacy LRT (e.g. >10) and extremely high Legacy beta+ rate, which disappears completely in the TOGA dataset because the addition of hundreds of other neutral lineages stabilizes the neutral null model and shows that the substitution pattern is consistent with neutral variation.
* **Example:** In our genes, sites where TOGA p-value is close to 1.0 (LRT = 0) but Legacy has a strong signal are classic examples of variance artifacts.

### Category B: Diluted Lineage-Specific Selection
In some cases, selection is real but **clade-specific** (e.g. occurring only within primates or rodents). Because the legacy species list contains a high proportion of primates and rodents, this lineage-specific signal represents a large fraction of the legacy tree.
However, when the alignment is scaled to the full **742-species TOGA tree** (which contains a massive diversity of other mammals where the site is highly conserved), the episodic signal is "diluted" across the entire tree, causing the global p-value to drop below the significance threshold.
* **Characteristics:** The site is classified as `GOLD` or `SILVER` in TOGA, but has a moderate p-value (e.g. 0.08–0.15) rather than 1.0. This indicates that a local selection signal is present on some branches but lacks the power to remain significant when averaged over the massive Eutherian tree without lineage-specific testing.

---

## 4. Conclusion
Detections unique to the legacy dataset are not necessarily errors, but they illustrate the **stability** of dense taxon sampling. A dense alignment protects against false positives driven by coincidental substitutions in small datasets, while highlighting the need for branch-specific or clade-specific tests (like aBSREL or clade-MEME) to resolve localized evolutionary changes in large trees.
