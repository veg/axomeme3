#!/usr/bin/env python3
import gzip
import os
import sqlite3
import re

def parse_nexus_gz(filepath):
    taxlabels = []
    sequences = []
    tree_str = None
    in_taxlabels = False
    in_matrix = False
    in_trees = False
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
                
            if line_strip.upper().startswith("TAXLABELS"):
                in_taxlabels = True
                content = line_strip[len("TAXLABELS"):].strip()
                tokens = content.replace("'", "").replace(";", "").split()
                taxlabels.extend(tokens)
                if line_strip.endswith(";"):
                    in_taxlabels = False
                continue
            
            if in_taxlabels:
                tokens = line_strip.replace("'", "").replace(";", "").split()
                taxlabels.extend(tokens)
                if line_strip.endswith(";"):
                    in_taxlabels = False
                continue
                
            if line_strip.upper().startswith("MATRIX"):
                in_matrix = True
                continue
                
            if in_matrix:
                if line_strip == ";":
                    in_matrix = False
                    continue
                if line_strip.endswith(";"):
                    sequences.append(line_strip[:-1].strip())
                    in_matrix = False
                    continue
                sequences.append(line_strip)
                continue
                
            if line_strip.upper().startswith("BEGIN TREES"):
                in_trees = True
                continue
                
            if in_trees:
                if line_strip.upper().startswith("TREE "):
                    parts = line_strip.split('=', 1)
                    if len(parts) > 1:
                        tree_str = parts[1].strip()
                if line_strip == "END;":
                    in_trees = False
                    continue
                    
    mapped = {}
    for label, seq in zip(taxlabels, sequences):
        mapped[label] = seq
    return mapped, tree_str

def parse_newick(newick_str):
    if not newick_str:
        return {}
    cleaned = re.sub(r'\{[^}]*\}', '', newick_str)
    cleaned = re.sub(r':[0-9.e-]+', '', cleaned)
    
    parents = {}
    tokens = []
    i = 0
    while i < len(cleaned):
        c = cleaned[i]
        if c in '(),;':
            tokens.append(c)
            i += 1
        else:
            name = []
            while i < len(cleaned) and cleaned[i] not in '(),;':
                name.append(cleaned[i])
                i += 1
            tokens.append(''.join(name).strip())
            
    group_stack = []
    current_group = []
    anon_counter = 0
    
    token_idx = 0
    while token_idx < len(tokens):
        t = tokens[token_idx]
        if t == '(':
            group_stack.append(current_group)
            current_group = []
            token_idx += 1
        elif t == ')':
            token_idx += 1
            parent_name = ""
            if token_idx < len(tokens) and tokens[token_idx] not in '(),;':
                parent_name = tokens[token_idx]
                token_idx += 1
            else:
                anon_counter += 1
                parent_name = f"ANON_{anon_counter}"
            
            for child in current_group:
                parents[child] = parent_name
                
            parent_group = group_stack.pop()
            parent_group.append(parent_name)
            current_group = parent_group
        elif t in ',;':
            token_idx += 1
        else:
            current_group.append(t)
            token_idx += 1
            
    return parents

def get_descendant_leaves(node, children_map, leaves_set):
    if node not in children_map:
        return [node]
    desc_leaves = []
    for child in children_map[node]:
        desc_leaves.extend(get_descendant_leaves(child, children_map, leaves_set))
    return desc_leaves

def get_codon_hamming_dist(c1, c2):
    if len(c1) != 3 or len(c2) != 3:
        return 0
    return sum(1 for a, b in zip(c1, c2) if a != b)

def analyze_site(seqs, tree_parents, tree_children, leaves_set, site_idx, branch, substitutions):
    site_start = (site_idx - 1) * 3
    win_start = max(0, (site_idx - 11) * 3)
    win_end = (site_idx + 10) * 3
    
    # Substitutions can be empty if no specific change is reconstructed at this node
    is_mns = False
    if substitutions:
        # Check if any change on this branch was MNS
        for b, anc_codon, der_codon in substitutions:
            if b.upper() == branch.upper():
                dist = get_codon_hamming_dist(anc_codon, der_codon)
                if dist >= 2:
                    is_mns = True
                    break
                    
    carrying_leaves = []
    is_leaf = branch in leaves_set or any(l.upper() == branch.upper() for l in leaves_set)
    
    if is_leaf:
        match_leaf = [l for l in leaves_set if l.upper() == branch.upper()][0]
        carrying_leaves = [match_leaf]
    else:
        match_node = [n for n in tree_children.keys() if n.upper() == branch.upper()]
        if match_node:
            carrying_leaves = get_descendant_leaves(match_node[0], tree_children, leaves_set)
            
    if not carrying_leaves:
        return 'LIKELY_ERROR', f"No carrying leaves found for branch {branch}"
        
    total_leaves = len(carrying_leaves)
    valid_leaves_count = 0
    local_gap_densities = []
    short_fragment_failures = 0
    flanking_clean_for_mns = True
    
    for leaf in carrying_leaves:
        seq = seqs.get(leaf, "")
        if not seq:
            continue
        codon = seq[site_start:site_start+3].upper() if len(seq) > site_start + 2 else "---"
        if '-' in codon or 'N' in codon or len(codon) < 3:
            continue
        
        valid_leaves_count += 1
        subseq = seq[win_start:win_end]
        gaps = subseq.count('-') + subseq.lower().count('n')
        density = gaps / len(subseq) if subseq else 1.0
        local_gap_densities.append(density)
        
        left = site_start
        while left > 0 and seq[left-1] != '-' and seq[left-1].upper() != 'N':
            left -= 1
        right = site_start + 3
        while right < len(seq) and seq[right] != '-' and seq[right].upper() != 'N':
            right += 1
        block_len = right - left
        if block_len < 30:
            short_fragment_failures += 1
            
        if is_mns:
            fl_start = max(0, (site_idx - 6) * 3)
            fl_end = (site_idx + 5) * 3
            fl_seq = seq[fl_start:fl_end]
            fl_gaps = fl_seq.count('-') + fl_seq.lower().count('n')
            if fl_gaps > 0:
                flanking_clean_for_mns = False
                
    if total_leaves > 1:
        valid_frac = valid_leaves_count / total_leaves
        mean_density = sum(local_gap_densities) / len(local_gap_densities) if local_gap_densities else 1.0
        if valid_frac < 0.5:
            return 'LIKELY_ERROR', f"Low descendant representation ({valid_frac:.1f})"
        if mean_density > 0.3:
            return 'LIKELY_ERROR', f"High descendant gap density ({mean_density:.1f})"
    else:
        density = local_gap_densities[0] if local_gap_densities else 1.0
        if density > 0.3:
            return 'LIKELY_ERROR', f"High gap density ({density:.1f})"
            
    if short_fragment_failures > 0:
        return 'LIKELY_ERROR', "Carrying sequence is an isolated short fragment"
        
    if is_mns:
        if not flanking_clean_for_mns:
            return 'LIKELY_ERROR', "MNS with gap/N in flanking sequence"
        return 'SILVER', "Valid multi-nucleotide substitution (high flanking homology)"
        
    # Check if this site was a serine transition (synonymous Serine islands)
    # We query the substitutions table to check codon changes on this branch
    is_serine_island = False
    if substitutions:
        for b, anc, der in substitutions:
            if b.upper() == branch.upper():
                tcn = ('TCA', 'TCC', 'TCG', 'TCT')
                agy = ('AGC', 'AGT')
                if (anc in tcn and der in agy) or (anc in agy and der in tcn):
                    is_serine_island = True
                    break
                    
    if is_serine_island:
        return 'SILVER', "Valid Serine island transition"
        
    return 'GOLD', "Valid positive selection site"

def main():
    db_path = "absrel_results.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        return
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if classification column exists in selected_sites
    c.execute("PRAGMA table_info(selected_sites)")
    cols = [row[1] for row in c.fetchall()]
    if 'classification' not in cols:
        print("Adding classification column to selected_sites table...")
        c.execute("ALTER TABLE selected_sites ADD COLUMN classification TEXT DEFAULT 'UNCLASSIFIED'")
        conn.commit()
    if 'filter_reason' not in cols:
        print("Adding filter_reason column to selected_sites table...")
        c.execute("ALTER TABLE selected_sites ADD COLUMN filter_reason TEXT")
        conn.commit()
        
    # Get list of all selected sites
    c.execute("select gene_name, branch_name, site_index from selected_sites")
    all_rows = c.fetchall()
    print(f"Updating classification for {len(all_rows)} selection events...")
    
    # Group by gene
    genes_to_records = {}
    for gene, branch, site in all_rows:
        genes_to_records.setdefault(gene, []).append((branch, site))
        
    updated_count = 0
    
    for gene, records in sorted(genes_to_records.items()):
        # Spatial runs check: group records by branch
        branch_runs = {}
        for branch, site in records:
            branch_runs.setdefault(branch, []).append(site)
            
        flagged_cliques = set() # (branch, site)
        for branch, sites in branch_runs.items():
            sites_sorted = sorted(sites)
            n = len(sites_sorted)
            # Find runs of length >= 3 where spacing between adjacent sites is <= 5 codons
            i = 0
            while i < n:
                run = [i]
                j = i + 1
                while j < n and (sites_sorted[j] - sites_sorted[j-1]) <= 5:
                    run.append(j)
                    j += 1
                if len(run) >= 3:
                    for idx in run:
                        flagged_cliques.add((branch, sites_sorted[idx]))
                i = j
                
        align_path = f"msa/{gene}.gz"
        if not os.path.exists(align_path):
            for branch, site in records:
                c.execute("update selected_sites set classification='LIKELY_ERROR', filter_reason='Missing alignment file' where gene_name=? and branch_name=? and site_index=?", (gene, branch, site))
            conn.commit()
            continue
            
        try:
            seqs, tree_str = parse_nexus_gz(align_path)
            parents = parse_newick(tree_str)
            children = {}
            for child, parent in parents.items():
                children.setdefault(parent, []).append(child)
            leaves_set = set(seqs.keys())
        except Exception as e:
            for branch, site in records:
                c.execute("update selected_sites set classification='LIKELY_ERROR', filter_reason='Error parsing alignment file' where gene_name=? and branch_name=? and site_index=?", (gene, branch, site))
            conn.commit()
            continue
            
        for branch, site in records:
            # Query site substitutions on this specific branch if any
            c.execute("""
                select branch_name, ancestral_codon, derived_codon 
                from site_substitutions 
                where gene_name=? and site_index=? and branch_name=?
            """, (gene, site, branch))
            subst = c.fetchall()
            
            # Analyze alignment quality and substitutions
            status, reason = analyze_site(seqs, parents, children, leaves_set, site, branch, subst)
            
            # Override if part of a spatial cluster on this branch
            if (branch, site) in flagged_cliques:
                status = 'LIKELY_ERROR'
                reason = f"Spatially contiguous run of selected sites on branch; {reason}"
                
            c.execute("update selected_sites set classification=?, filter_reason=? where gene_name=? and branch_name=? and site_index=?", (status, reason, gene, branch, site))
            
        conn.commit()
        updated_count += len(records)
        if updated_count % 1000 == 0 or updated_count == len(all_rows):
            print(f"  Updated {updated_count} / {len(all_rows)} sites...")
            
    print("Database updated successfully!")
    conn.close()

if __name__ == "__main__":
    main()
