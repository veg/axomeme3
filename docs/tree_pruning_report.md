# Species Tree Pruning and Taxonomic Mapping Report

This report summarizes the process of pruning the species tree to select single representative genomes and establishing a clean taxonomic mapping.

## 1. Summary Statistics
- **Initial leaf nodes in tree:** 882
- **Leaves pruned:** 140
- **Final representative leaves remaining:** 742
- **Non-monophyletic species detected:** 6

## 2. Non-Monophyletic Species Report
The following species were found to be **non-monophyletic** (paraphyletic or polyphyletic) in the original tree. This occurs when assemblies for the same species do not form an exclusive clade, indicating either taxonomic nesting (e.g. wild ancestor vs. domestic descendant) or branch placement inconsistencies.

| Non-Monophyletic Species | Leaves in tree | Nesting Clade Contaminants (Other Species inside MRCA) |
|---|---|---|
| **Capra hircus** | HLcapHirc3, HLcapHir2 | HLcapAeg1 (Capra aegagrus) |
| **Ovis aries** | HLoviArie7, HLoviAri6 | HLoviOri1 (Ovis orientalis) |
| **Cervus albirostris** | HLprzAlb2, HLprzAlb1 | HLcerHanYar1 (Cervus hanglu yarkandensis), HLcerCan1 (Cervus canadensis), HLcerNip1 (Cervus nippon), HLcerEla2 (Cervus elaphus) |
| **Giraffa tippelskirchi** | HLgirTip2, HLgirTip1 | HLgirCam2 (Giraffa camelopardalis) |
| **Balaenoptera ricei** | HLbalRic1, HLbalRic2 | HLbalEde1 (Balaenoptera edeni) |
| **Artibeus jamaicensis** | HLartJam3, HLartJam2 | HLartLit1A (Artibeus lituratus) |

## 3. Trimmed Taxon Name Collisions
Renaming terminal leaves to their pure `abcXyz` taxon names could cause duplicate leaf names in the tree. To prevent this, unique suffixes were appended in the final tree, though the mapping remains many-to-one for the `Trimmed_Taxon_Name` column.

| Trimmed Code | Occurrences | Species Involved | Unique Tree Names |
|---|---|---|---|
| `odoHem` | 2 | Odocoileus hemionus hemionus, Odocoileus hemionus | odoHem, odoHem_2 |
| `equQua` | 2 | Equus quagga quagga, Equus quagga | equQua, equQua_2 |
| `gorGor` | 2 | Gorilla gorilla gorilla, Gorilla gorilla | gorGor, gorGor_2 |

## 4. Mapping File Locations
The outputs have been generated and saved at:
1. **Pruned species tree:** `pruned_species_tree.nhx` (also in `docs/`)
2. **Species to Taxon map:** `species_to_taxon_map.tsv` (also in `docs/`)
