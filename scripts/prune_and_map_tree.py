#!/usr/bin/env python3
"""
prune_and_map_tree.py
---------------------
Prunes species_tree.nhx so that each species is represented by a single "best" genome leaf.
Checks for monophyly of species with multiple leaves and reports non-monophyletic clades.
Establishes a species-to-taxon mapping trimmed to the abcXyz format.
Saves the pruned tree and mapping tables.
"""

import os
import re
import pandas as pd
from Bio import Phylo

def clean_numeric(val):
    if pd.isna(val):
        return None
    try:
        f = float(val)
        if f == -1.0:
            return None
        return int(f) if f.is_integer() else f
    except ValueError:
        return None

def trim_leaf_name(name):
    # Remove HL prefix if present
    if name.startswith('HL'):
        name = name[2:]
    # Remove trailing version digits and uppercase letters (e.g. 1, 2A, 10, 38)
    name = re.sub(r'\d+[A-Za-z]*$', '', name)
    return name

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    tsv_path = os.path.join(base_dir, "assemblies_and_species.tsv")
    tree_path = os.path.join(base_dir, "species_tree.nhx")
    
    print(f"Loading metadata: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Pre-clean columns for quality ranking
    df['contig_N50_clean'] = df['contig N50 (bp)'].apply(clean_numeric)
    df['scaffold_N50_clean'] = df['scaffold N50 (bp)'].apply(clean_numeric)
    df['intact_clean'] = df['No. ancestral genes with intact ORF'].apply(clean_numeric)
    df['submission_date_dt'] = pd.to_datetime(df['Assembly submission date'], errors='coerce')
    
    # Assign assembly tier score
    def get_status_score(status):
        if pd.isna(status):
            return 0
        s = str(status).strip().lower()
        if 'chromosome' in s or 'complete' in s:
            return 3
        elif 'scaffold' in s:
            return 2
        elif 'contig' in s:
            return 1
        return 0
    df['status_score'] = df['NCBI assembly status'].apply(get_status_score)
    
    # Create dict mapping Assembly Name to its row values for quick access
    assembly_meta = {}
    for _, row in df.iterrows():
        assembly_meta[row['Assembly name']] = {
            'Species': row['Species'],
            'Common_Name': row['Common name'],
            'status_score': row['status_score'],
            'scaffold_N50': row['scaffold_N50_clean'] or 0,
            'contig_N50': row['contig_N50_clean'] or 0,
            'intact': row['intact_clean'] or 0,
            'date': row['submission_date_dt'] or pd.Timestamp('1900-01-01'),
            'status': row['NCBI assembly status']
        }
        
    print(f"Loading tree: {tree_path}")
    tree = Phylo.read(tree_path, 'newick')
    initial_leaves = [node.name for node in tree.get_terminals()]
    print(f"Initial leaves in tree: {len(initial_leaves)}")
    
    # Group tree leaves by species
    species_to_leaves = {}
    for leaf in initial_leaves:
        meta = assembly_meta.get(leaf)
        if meta:
            sp = meta['Species']
            species_to_leaves.setdefault(sp, []).append(leaf)
        else:
            print(f"Warning: Leaf label '{leaf}' not found in metadata TSV!")
            
    # Check monophyly for species with multiple leaves
    non_monophyletic_species = {}
    
    for sp, leaves_list in species_to_leaves.items():
        if len(leaves_list) > 1:
            mrca = tree.common_ancestor(leaves_list)
            mrca_terminals = {node.name for node in mrca.get_terminals()}
            # If terminals under MRCA contain leaves not in our species leaf list
            others = mrca_terminals - set(leaves_list)
            if others:
                other_details = []
                for o in others:
                    o_sp = assembly_meta.get(o, {}).get('Species', 'Unknown')
                    other_details.append(f"{o} ({o_sp})")
                non_monophyletic_species[sp] = {
                    'leaves': leaves_list,
                    'nested': other_details
                }
                
    # Select the single "best" representative leaf for each species
    prune_targets = []
    best_leaves = {} # species -> best leaf name
    
    for sp, leaves_list in species_to_leaves.items():
        if len(leaves_list) == 1:
            best_leaves[sp] = leaves_list[0]
        else:
            # Sort leaves of the same species in the tree using our quality criteria
            def sort_key(leaf_name):
                m = assembly_meta.get(leaf_name)
                if not m:
                    return (0, 0, 0, 0, pd.Timestamp('1900-01-01'))
                return (m['status_score'], m['scaffold_N50'], m['contig_N50'], m['intact'], m['date'])
                
            sorted_leaves = sorted(leaves_list, key=sort_key, reverse=True)
            best_leaf = sorted_leaves[0]
            best_leaves[sp] = best_leaf
            
            # Prune all others
            for leaf in sorted_leaves[1:]:
                prune_targets.append(leaf)
                
    # Execute pruning
    print(f"Pruning {len(prune_targets)} non-representative leaves...")
    for target in prune_targets:
        tree.prune(target)
        
    final_leaves_nodes = tree.get_terminals()
    print(f"Final leaves in tree after pruning: {len(final_leaves_nodes)}")
    
    # Establish mapping between species and trimmed taxon name
    # Trim logic: strip HL prefix, strip version suffixes
    species_to_taxon = []
    taxon_to_species = {}
    
    # We must check for collisions in the trimmed taxon names
    # Collisions will be resolved in the tree leaf names to keep tree leaves unique,
    # but the mapping will register the clean trimmed names.
    taxon_occurrences = {}
    
    for node in final_leaves_nodes:
        orig_name = node.name
        meta = assembly_meta.get(orig_name, {})
        sp = meta.get('Species', 'Unknown')
        common = meta.get('Common_Name', '')
        
        trimmed = trim_leaf_name(orig_name)
        
        # Add to unique name registry for tree leaves
        taxon_occurrences[trimmed] = taxon_occurrences.get(trimmed, 0) + 1
        unique_leaf_name = trimmed
        if taxon_occurrences[trimmed] > 1:
            unique_leaf_name = f"{trimmed}_{taxon_occurrences[trimmed]}"
            
        node.name = unique_leaf_name
        
        species_to_taxon.append({
            'Species': sp,
            'Common_Name': common,
            'Original_Assembly_Name': orig_name,
            'Trimmed_Taxon_Name': trimmed,
            'Tree_Leaf_Name': unique_leaf_name
        })
        
    # Write species_to_taxon mapping to TSV
    map_df = pd.DataFrame(species_to_taxon)
    map_df.to_csv(os.path.join(base_dir, "species_to_taxon_map.tsv"), sep='\t', index=False)
    map_df.to_csv(os.path.join(base_dir, "docs", "species_to_taxon_map.tsv"), sep='\t', index=False)
    print("Mapping file saved to species_to_taxon_map.tsv")
    
    # Write pruned tree to file
    out_tree_path = os.path.join(base_dir, "pruned_species_tree.nhx")
    Phylo.write(tree, out_tree_path, 'newick')
    Phylo.write(tree, os.path.join(base_dir, "docs", "pruned_species_tree.nhx"), 'newick')
    print(f"Pruned tree saved to {out_tree_path}")
    
    # Write markdown report
    report_path = os.path.join(base_dir, "docs", "tree_pruning_report.md")
    with open(report_path, 'w') as f:
        f.write("# Species Tree Pruning and Taxonomic Mapping Report\n\n")
        f.write("This report summarizes the process of pruning the species tree to select single representative genomes and establishing a clean taxonomic mapping.\n\n")
        
        f.write("## 1. Summary Statistics\n")
        f.write(f"- **Initial leaf nodes in tree:** {len(initial_leaves)}\n")
        f.write(f"- **Leaves pruned:** {len(prune_targets)}\n")
        f.write(f"- **Final representative leaves remaining:** {len(final_leaves_nodes)}\n")
        f.write(f"- **Non-monophyletic species detected:** {len(non_monophyletic_species)}\n\n")
        
        f.write("## 2. Non-Monophyletic Species Report\n")
        f.write("The following species were found to be **non-monophyletic** (paraphyletic or polyphyletic) in the original tree. ")
        f.write("This occurs when assemblies for the same species do not form an exclusive clade, indicating either taxonomic nesting (e.g. wild ancestor vs. domestic descendant) or branch placement inconsistencies.\n\n")
        
        if non_monophyletic_species:
            f.write("| Non-Monophyletic Species | Leaves in tree | Nesting Clade Contaminants (Other Species inside MRCA) |\n")
            f.write("|---|---|---|\n")
            for sp, details in non_monophyletic_species.items():
                nested_str = ", ".join(details['nested'])
                leaves_str = ", ".join(details['leaves'])
                f.write(f"| **{sp}** | {leaves_str} | {nested_str} |\n")
            f.write("\n")
        else:
            f.write("No non-monophyletic species found.\n\n")
            
        f.write("## 3. Trimmed Taxon Name Collisions\n")
        f.write("Renaming terminal leaves to their pure `abcXyz` taxon names could cause duplicate leaf names in the tree. ")
        f.write("To prevent this, unique suffixes were appended in the final tree, though the mapping remains many-to-one for the `Trimmed_Taxon_Name` column.\n\n")
        
        collisions = {k: v for k, v in taxon_occurrences.items() if v > 1}
        if collisions:
            f.write("| Trimmed Code | Occurrences | Species Involved | Unique Tree Names |\n")
            f.write("|---|---|---|---|\n")
            for code, count in collisions.items():
                sps = map_df[map_df['Trimmed_Taxon_Name'] == code]['Species'].tolist()
                leaves_in_tree = map_df[map_df['Trimmed_Taxon_Name'] == code]['Tree_Leaf_Name'].tolist()
                f.write(f"| `{code}` | {count} | {', '.join(sps)} | {', '.join(leaves_in_tree)} |\n")
            f.write("\n")
        else:
            f.write("No trimmed taxon name collisions detected.\n\n")
            
        f.write("## 4. Mapping File Locations\n")
        f.write("The outputs have been generated and saved at:\n")
        f.write("1. **Pruned species tree:** `pruned_species_tree.nhx` (also in `docs/`)\n")
        f.write("2. **Species to Taxon map:** `species_to_taxon_map.tsv` (also in `docs/`)\n")

    print("Tree pruning and report generation complete!")

if __name__ == "__main__":
    main()
