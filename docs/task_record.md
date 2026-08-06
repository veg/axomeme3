# Project Record: TOGA Genomic Alignments Analysis

This file maintains a detailed chronological log of all tasks, scripts, and findings for the TOGA genomic alignments project.

---

## Task 1: Examine and Summarize Genome Metadata

### Objectives
- Read and examine the schema and format of `assemblies_and_species.tsv`.
- Summarize the genome metadata, including species counts, taxonomy distributions, assembly quality metrics, and TOGA ancestral gene projections.
- Save documented scripts in the `scripts/` directory and summaries in the `docs/` directory.

### Actions Taken
1. **Initial Inspection:** Viewed the headers and first 100 rows of `assemblies_and_species.tsv` to verify column names and formatting.
2. **Script Creation:** Created [summarize_metadata.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/summarize_metadata.py) under the `scripts/` directory. The script:
   - Cleans numeric columns (converting strings like `NULL` or `-1` to `NaN`).
   - Parses the semicolon-separated `Taxonomic Lineage` column to extract order, family, and genus level information.
   - Summarizes unique counts, taxonomic distributions, assembly status, submitter organizations, Contig/Scaffold N50 percentiles, and submission date trends.
   - Calculates summary statistics (mean, median, range) for ancestral gene projection columns (intact ORF, inactivating mutations, missing sequences).
3. **Report Generation:** Executed the script and generated the markdown summary [genome_metadata_summary.md](file:///Users/sergei/Dropbox/TOGA2026/docs/genome_metadata_summary.md) under the `docs/` directory.

---

## Task 2: Create HTML Dashboard

### Objectives
- Build an interactive, visually stunning HTML dashboard to explore the genome assembly metadata and TOGA projections.
- Enable searching, sorting, and filtering of all 1,971 records.
- Provide charts for taxonomy distribution and submission timelines.
- Ensure the dashboard is self-contained and opens locally (via `file://`) without CORS restrictions.

### Actions Taken
1. **Script Creation:** Created [generate_dashboard.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/generate_dashboard.py) in the `scripts/` directory. The script reads the TSV data, cleans the numeric and date columns, formats it into a compact JSON payload, and injects it into a premium HTML/CSS/JS template.
2. **Dashboard Generation:** Executed the script to output a fully self-contained [dashboard.html](file:///Users/sergei/Dropbox/TOGA2026/dashboard.html) at the root of the workspace.

---

## Task 3: Add Assembly Counts per Species and Sibling Comparison Explorer

### Objectives
- Integrate assembly count calculations per species to see how many assemblies are available for any given species.
- Visually indicate in the table if a species has multiple assemblies.
- Provide an interactive sibling assembly explorer in the detail modal to compare quality and projection metrics between different assemblies of the same species.

### Actions Taken
1. **Script Upgrades:** Modified [generate_dashboard.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/generate_dashboard.py) to:
   - Group the dataset by species and calculate the assembly count for each species record.
   - Inject the count (`sp_count`) into each item in the JSON payload.
2. **Dashboard UI Improvements:**
   - **Species badges:** Table rows now display a subtle count badge (e.g. `2`, `5`) next to the species name if that species has multiple assemblies in the dataset.
   - **Sibling comparison table:** Added a section to the details modal titled "Assemblies for this Species". If a species has more than one assembly, it renders a sub-table comparing the accession, assembly status, scaffold N50, and intact ORF gene projections of all sibling assemblies.
   - **Cross-navigation:** Sibling table entries are fully clickable, enabling the user to switch the modal detail view to any sibling assembly with one click.
3. **Re-compilation:** Executed the upgraded dashboard generator script to output the updated [dashboard.html](file:///Users/sergei/Dropbox/TOGA2026/dashboard.html).

---

## Task 4: Select Representative Genomes and Identify Problematic Assemblies

### Objectives
- Establish a rigorous criteria to select the single "best" representative genome assembly for each of the 1,685 species.
- Map assemblies to species (both directions) and output clean mapping files.
- Establish empirical thresholds to flag potentially problematic (low-quality or high-error) assemblies.

### Actions Taken
1. **Script Creation:** Created [select_best_genomes.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/select_best_genomes.py) in the `scripts/` directory.
2. **Best Assembly Selection Logic:** Sorted assemblies within each species using:
   - **Assembly Status Tier:** `Chromosome`/`Complete Genome` (Tier 3) > `Scaffold` (Tier 2) > `Contig` (Tier 1).
   - **Scaffold N50:** Higher is preferred.
   - **Contig N50:** Higher is preferred.
   - **Intact ORFs:** Higher projected intact genes is preferred (if TOGA data is available).
   - **Recency:** Newer submission dates are used as tiebreakers.
3. **Problematic Assembly Criteria:** Flagged 294 assemblies (14.92%) based on the following:
   - Assembly status is Contig.
   - Scaffold N50 < 100 KB (extremely low contiguity).
   - Contig N50 < 10 KB (highly fragmented).
   - If TOGA data is available: Intact ORFs < 8,000, missing sequence count > 2,000, or inactivating mutation count > 2,000.
4. **Outputs Generated:**
   - **Best representatives map:** [species_to_assembly.tsv](file:///Users/sergei/Dropbox/TOGA2026/species_to_assembly.tsv) (also saved in `docs/`).
   - **Bidirectional map:** [assembly_to_species.tsv](file:///Users/sergei/Dropbox/TOGA2026/assembly_to_species.tsv) (also saved in `docs/`).
   - **Problematic list:** [problematic_genomes.tsv](file:///Users/sergei/Dropbox/TOGA2026/problematic_genomes.tsv) (also saved in `docs/`).
   - **Report:** Detailed markdown report generated at [genome_selection_report.md](file:///Users/sergei/Dropbox/TOGA2026/docs/genome_selection_report.md).

---

## Task 5: Prune Species Tree and Rename Taxa to abcXyz Format

### Objectives
- Examine leaf names in `species_tree.nhx` and resolve how they map to species.
- Group the tree leaves by species and identify non-monophyletic species.
- Select the single "best" leaf node for each species based on quality metrics and prune all non-representative leaves.
- Trim leaf names to their core `abcXyz` species code and resolve collisions.
- Output mapping tables and the pruned tree.

### Actions Taken
1. **Script Creation:** Created [prune_and_map_tree.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/prune_and_map_tree.py) in the `scripts/` directory.
2. **Monophyly Validation:** Identified 6 species that were non-monophyletic in the tree:
   - `Capra hircus` (paraphyletic, nested wild `Capra aegagrus`)
   - `Ovis aries` (paraphyletic, nested mouflon `Ovis orientalis`)
   - `Cervus albirostris` (paraphyletic, nested other deer species)
   - `Giraffa tippelskirchi` (paraphyletic, nested `Giraffa camelopardalis`)
   - `Balaenoptera ricei` (paraphyletic, nested `Balaenoptera edeni`)
   - `Artibeus jamaicensis` (paraphyletic, nested `Artibeus lituratus`)
3. **Pruning Implementation:** Pruned 140 leaves, reducing the species tree from 882 terminal leaves down to exactly 742 representative leaf nodes (one per represented species).
4. **Renaming & Collision Resolution:**
   - Leaves were renamed to their `abcXyz` species codes (removing `HL` and trailing version digits).
   - Flagged and resolved 3 name collisions between species/subspecies pairs (*Odocoileus hemionus*, *Equus quagga*, *Gorilla gorilla*) by adding unique numeric suffixes (e.g. `odoHem_2`) in the pruned tree to maintain uniqueness.
5. **Outputs Generated:**
   - **Pruned and renamed tree:** [pruned_species_tree.nhx](file:///Users/sergei/Dropbox/TOGA2026/pruned_species_tree.nhx) (also in `docs/`).
   - **Species-to-Taxon Mapping:** [species_to_taxon_map.tsv](file:///Users/sergei/Dropbox/TOGA2026/species_to_taxon_map.tsv) (also in `docs/`).
   - **Detailed report:** [tree_pruning_report.md](file:///Users/sergei/Dropbox/TOGA2026/docs/tree_pruning_report.md).

---

## Task 6: Prepare Individual Alignments and Curation Dashboard

### Objectives
- Inspect and analyze all 19,953 codonified FASTA alignments in `individualAlis/` in parallel.
- Build a SQLite database (`alignments_stats.db`) containing sequence quality metrics (stop codons, resolved length, gaps, and Ns) for all sequences.
- Map filenames to resolved gene names, resolving Ensembl Gene IDs (ENSG) to standard gene symbols (HGNC).
- Identify pseudogenes and suspicious ("sus") genes using quantitative sequence-level and metadata criteria.
- Design and serve an interactive dashboard showing global summary statistics, graphs, filters, and alignments detail view.

### Actions Taken
1. **Parallel SQLite Compiler:** Created [build_alignment_db.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/build_alignment_db.py) which scans all 19,953 alignment files in parallel, parses sequence properties, detects in-frame stops/frameshifts/low coverage, and writes results in chunked batches within single transaction blocks. Compiled exactly 19,953 alignments and 14,997,766 sequences into `/Users/sergei/Dropbox/TOGA2026/alignments_stats.db` (964 MB database) in 1.87 minutes. Added database indexes for instant query performance.
2. **Gene Name Resolution & Curation Script:** Created [resolve_gene_names.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/resolve_gene_names.py) to:
   - Alter the alignments table to store resolved names, biotypes, pseudogene and sus flags.
   - Fix the non-standard filename row for `MAGEC1` (which failed standard regex due to multiple dots in its name).
   - Use Ensembl's POST REST API in bulk to resolve all 127 Ensembl Gene IDs (ENSG) to standard gene symbols, descriptions, and biotypes.
   - Flag "sus" genes quantitatively in alignment space (excluding frameshift indels): An alignment is sus if its reference sequence `hg38` is suspect (contains premature stops or resolved coverage < 30%) or if >50% of the species sequences in the alignment are suspect (due to premature stops or low coverage/gaps/Ns).
   - Identified 846 suspicious alignments and 5 Ensembl-resolved pseudogenes.
3. **Interactive Curation Dashboard:** Created [alignments_dashboard.html](file:///Users/sergei/Dropbox/TOGA2026/docs/alignments_dashboard.html) utilizing a premium dark glassmorphic UI, metadata cards, Chart.js diagnostic graphs, paginated query tables with search/sorting/filtering, and a slide-out drawer displaying details of all species alignments (highlighting human reference `hg38` at the top).
4. **Lightweight Dashboard Server:** Created [serve_alignments_dashboard.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/serve_alignments_dashboard.py) which pre-calculates and caches global sequence-level stats on startup to ensure instant dashboard load times. The server exposes the paginated catalog and alignment-specific APIs, running on port 8000.

---

## Task 7: Alignment Size Histograms and Robust Outlier Detection

### Objectives
- Add visual histograms of alignment sizes (in codons and sequence counts) to the curation dashboard.
- Implement robust outlier detection methods to identify abnormally sized genes/alignments and sequence coverage issues.
- Expose statistical metrics and allow filtering by outliers in the REST API and the catalog table.

### Actions Taken
1. **Robust Outlier Detection Implementation:**
   - **Codon Lengths (Log10 IQR):** Since gene lengths are highly right-skewed, we applied a $\log_{10}$ transformation before calculating the Interquartile Range (IQR) bounds. This yielded a lower outlier bound of **67.5 codons** (short outliers) and an upper bound of **3,200.1 codons** (long outliers), flagging exactly 370 alignments.
   - **Sequence Counts (Log10 Z-score):** To detect genes with unusually low species coverage, we calculated the mean and standard deviation of sequence counts in $\log_{10}$ space. A Z-score threshold of $< -2.0$ was applied to flag alignments with **fewer than 130.7 sequences** as outliers (flagging 1,166 alignments).
   - In total, **1,519 distinct alignments** (7.6% of the dataset) were identified as outliers.
2. **Schema Migration & DB Update:** Updated [serve_alignments_dashboard.py](file:///Users/sergei/Dropbox/TOGA2026/scripts/serve_alignments_dashboard.py) to automatically migrate the SQLite database by adding `is_outlier` and `outlier_reasons` columns to the `alignments` table. The outlier status is evaluated and written to the database in a single transaction block on server startup.
3. **Binned Histograms calculation:** Added dynamic log-scale binning for codon lengths and linear binning for sequence counts.
4. **Dashboard Frontend Upgrades:**
   - **Outlier Metrics:** Added a fifth summary card "Size Outliers" showing the outlier count and percentage (7.6%).
   - **Histogram Distributions:** Added a new row containing two responsive area charts (using Chart.js and Chart.js Annotation Plugin) showing the continuous distribution curves for codon lengths and sequence counts, with red dashed vertical lines marking the exact statistical outlier thresholds.
   - **Curation Table & Filter:** Added an "Outliers" button in the filter toolbar and implemented automatic warning badges for outlier rows (e.g. `Outlier: Short`, `Outlier: Long`, `Outlier: Few Seqs`) linked to the specific reason in the details drawer.


