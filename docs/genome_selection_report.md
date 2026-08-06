# Genome Assembly Selection and Quality Report

This report details the results of selecting the 'best' genome assembly per species and identifying potentially problematic assemblies.

## 1. Summary Statistics
- **Total assemblies examined:** 1971
- **Unique species represented:** 1685
- **Assemblies selected as 'Best' representation:** 1685 (1 per species)
- **Potentially problematic genomes flagged:** 294 (14.92% of total)

## 2. Selection Hierarchy for 'Best' Genome
When a species had multiple assemblies, the best assembly was selected using the following hierarchical criteria:
1. **Assembly Level (Status):** `Chromosome` / `Complete Genome` > `Scaffold` > `Contig` > `Unknown`.
2. **Scaffold N50:** Higher value (base pairs) is preferred.
3. **Contig N50:** Higher value (base pairs) is preferred.
4. **TOGA Projections:** Higher count of intact ORFs (if available).
5. **Submission Date:** Newer submission dates are used as final tiebreakers.

## 3. Problematic Genome Criteria and Flags
Genomes were flagged as **potentially problematic** if they met any of the following quality thresholds:
- **Assembly Status:** Assembly status is explicitly `Contig` (indicating lack of scaffolding).
- **Scaffold N50:** Scaffold N50 is less than **100,000 bp** (100 KB).
- **Contig N50:** Contig N50 is less than **10,000 bp** (10 KB).
- **TOGA Projection Quality (if available):**
  - Intact ORFs < **8,000** (highly incomplete gene set).
  - Missing sequence count > **2,000** (significant coding region gaps).
  - Inactivating mutation count > **2,000** (potential frameshift/indels in coding sequences).

### Summary of Flagged Reasons
| Flagged Reason | Number of Genomes | Percentage of Total Flagged |
|---|---|---|
| Low Scaffold N50 (< 100 KB) | 150 | 51.02% |
| Assembly status is Contig | 80 | 27.21% |
| High missing sequences (> 2,000) | 55 | 18.71% |
| High inactivating mutations (> 2,000) | 54 | 18.37% |
| Low Contig N50 (< 10 KB) | 30 | 10.20% |
| Low intact ORFs (< 8,000) | 13 | 4.42% |

### Top 15 Flagged Problematic Genomes
| Directory Name | Species | Status | Scaffold N50 | Intact ORFs | Primary Reason |
|---|---|---|---|---|---|
| Aix_sponsa__Wood_duck__HLaixSpon1__GCA_051175885.1 | Aix sponsa | Contig | 5242973 | nan | Assembly status is Contig |
| Anas_bahamensis__white-cheeked_pintail__HLanaBaha1__GCA_038381715.1 | Anas bahamensis | Scaffold | 76845102 | nan | Low Contig N50 (< 10 KB) |
| Anas_capensis__Cape_teal__HLanaCape1__GCA_032357685.1 | Anas capensis | Scaffold | 64715642 | nan | Low Contig N50 (< 10 KB) |
| Anas_melleri__Mellers_duck__HLanaMell1__GCA_052075425.1 | Anas melleri | Scaffold | 29552448 | nan | Low Contig N50 (< 10 KB) |
| Mergellus_albellus__Smew__HLmerAlbe1__GCA_051254355.1 | Mergellus albellus | Scaffold | 38391571 | nan | Low Contig N50 (< 10 KB) |
| Branta_hutchinsii__Cackling_goose__HLbraHutc1__GCA_032270845.1 | Branta hutchinsii | Scaffold | 83206430 | nan | Low Contig N50 (< 10 KB) |
| Cygnus_atratus__black_swan__HLcygAtr1__GCA_013377495.1 | Cygnus atratus | Contig | 12814145 | -1.0 | Assembly status is Contig |
| Aythya_nyroca__Ferruginous_duck__HLaytNyro1__GCA_052075125.1 | Aythya nyroca | Scaffold | 76982613 | nan | Low Contig N50 (< 10 KB) |
| Bambusicola_thoracicus__Chinese_bamboo-partridge__HLbamTho1__GCA_002909625.1 | Bambusicola thoracicus | Scaffold | 13160 | nan | Low Scaffold N50 (< 100 KB) |
| Gavia_stellata__red-throated_loon__gavSte1__GCF_000690875.1 | Gavia stellata | Scaffold | 45523 | -1.0 | Low Scaffold N50 (< 100 KB) |
| Butorides_virescens__green_heron__HLbutVir1__GCA_017310255.1 | Butorides virescens | Contig | 222988 | -1.0 | Assembly status is Contig |
| Gorsachius_goisagi__Japanese_night-heron__HLgorGoi1__GCA_018403395.1 | Gorsachius goisagi | Scaffold | 73010 | -1.0 | Low Scaffold N50 (< 100 KB) |
| Pelecanus_erythrorhynchos__American_white_pelican__HLpelEryt1__GCA_051176275.1 | Pelecanus erythrorhynchos | Contig | 5571114 | nan | Assembly status is Contig |
| Pelecanus_crispus__Dalmatian_pelican__pelCri1__GCF_000687375.1 | Pelecanus crispus | Scaffold | 43364 | -1.0 | Low Scaffold N50 (< 100 KB) |
| Scopus_umbretta__Hamerkop__HLscoUmbr1__GCA_013400535.1 | Scopus umbretta | Scaffold | 32357 | nan | Low Scaffold N50 (< 100 KB) |

## 4. Mapping Files Outputs
The following mapping files have been generated:
1. **Species to Best Assembly Map:** `species_to_assembly.tsv` (also saved in `docs/`)
2. **Assembly to Species Map (both directions):** `assembly_to_species.tsv` (also saved in `docs/`)
3. **Problematic Genomes List:** `problematic_genomes.tsv` (also saved in `docs/`)
