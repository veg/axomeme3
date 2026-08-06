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

def analyze_site(seqs, tree_parents, tree_children, leaves_set, site_idx, substitutions):
    site_start = (site_idx - 1) * 3
    win_start = max(0, (site_idx - 11) * 3)
    win_end = (site_idx + 10) * 3
    
    has_valid_single = False
    has_valid_mns = False
    failure_reasons = []
    
    for branch, anc_codon, der_codon in substitutions:
        dist = get_codon_hamming_dist(anc_codon, der_codon)
        is_mns = dist >= 2
        
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
            continue
            
        total_leaves = len(carrying_leaves)
        valid_leaves_count = 0
        local_gap_densities = []
        short_fragment_failures = 0
        flanking_clean_for_mns = True
        
        for leaf in carrying_leaves:
            seq = seqs.get(leaf, "")
            if not seq:
                continue
            codon = seq[site_start:site_start+3].upper()
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
                failure_reasons.append(f"Branch {branch}: low descendant representation ({valid_frac:.1f})")
                continue
            if mean_density > 0.3:
                failure_reasons.append(f"Branch {branch}: high descendant gap density ({mean_density:.1f})")
                continue
        else:
            density = local_gap_densities[0] if local_gap_densities else 1.0
            if density > 0.3:
                failure_reasons.append(f"Branch {branch}: high gap density ({density:.1f})")
                continue
                
        if short_fragment_failures > 0:
            failure_reasons.append(f"Branch {branch}: carrying sequence is an isolated short fragment")
            continue
            
        if is_mns:
            if not flanking_clean_for_mns:
                failure_reasons.append(f"Branch {branch}: MNS with gap/N in flanking sequence")
                continue
            has_valid_mns = True
        else:
            has_valid_single = True
            
    if has_valid_single:
        return 'GOLD', "Valid single nucleotide substitution"
    elif has_valid_mns:
        return 'SILVER', "Valid multi-nucleotide substitution (high flanking homology)"
    else:
        reason_str = "; ".join(failure_reasons) if failure_reasons else "No non-synonymous substitutions carried"
        return 'LIKELY_ERROR', reason_str

def main():
    db_path = "meme_results.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        return
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if classification column exists in site_results
    c.execute("PRAGMA table_info(site_results)")
    cols = [row[1] for row in c.fetchall()]
    if 'classification' not in cols:
        print("Adding classification column to site_results table...")
        c.execute("ALTER TABLE site_results ADD COLUMN classification TEXT DEFAULT 'UNCLASSIFIED'")
        conn.commit()
    if 'filter_reason' not in cols:
        print("Adding filter_reason column to site_results table...")
        c.execute("ALTER TABLE site_results ADD COLUMN filter_reason TEXT")
        conn.commit()
        
    # Get list of all significant sites (incremental by default, use --force to reclassify all)
    import sys
    force = '--force' in sys.argv
    if force:
        c.execute("select gene_name, site_index from site_results where is_significant=1")
    else:
        c.execute("select gene_name, site_index from site_results where is_significant=1 and (classification='UNCLASSIFIED' or classification IS NULL or classification='NONE')")
    sig_rows = c.fetchall()
    print(f"Updating classification for {len(sig_rows)} significant sites...")
    
    genes_to_sites = {}
    for gene, site in sig_rows:
        genes_to_sites.setdefault(gene, []).append(site)
        
    updated_count = 0
    
    for gene, sites in sorted(genes_to_sites.items()):
        # Identify spatially contiguous runs: Near-consecutive L>=5 (spacing<=3)
        sites_sorted = sorted(sites)
        n = len(sites_sorted)
        flagged_indices = set()
        
        i = 0
        while i < n:
            run = [i]
            j = i + 1
            while j < n and (sites_sorted[j] - sites_sorted[j-1]) <= 3:
                run.append(j)
                j += 1
            
            if len(run) >= 5:
                for idx in run:
                    flagged_indices.add(sites_sorted[idx])
            i = j

        align_path = f"msa/{gene}.gz"
        if not os.path.exists(align_path):
            # If missing alignment, classify as LIKELY_ERROR due to missing alignment
            for site in sites:
                c.execute("update site_results set classification='LIKELY_ERROR', filter_reason='Missing alignment file' where gene_name=? and site_index=?", (gene, site))
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
            for site in sites:
                c.execute("update site_results set classification='LIKELY_ERROR', filter_reason='Error parsing alignment file' where gene_name=? and site_index=?", (gene, site))
            conn.commit()
            continue
            
        for site in sites:
            c.execute("""
                select branch_name, ancestral_codon, derived_codon 
                from site_substitutions 
                where gene_name=? and site_index=? 
                and (
                    is_synonymous=0 
                    or (
                        (ancestral_codon in ('TCA','TCC','TCG','TCT') and derived_codon in ('AGC','AGT'))
                        or 
                        (ancestral_codon in ('AGC','AGT') and derived_codon in ('TCA','TCC','TCG','TCT'))
                    )
                )
            """, (gene, site))
            subst = c.fetchall()
            
            if not subst:
                status = 'LIKELY_ERROR'
                reason = 'No non-synonymous substitutions recorded'
            else:
                status, reason = analyze_site(seqs, parents, children, leaves_set, site, subst)
                
            # Override to LIKELY_ERROR if part of a spatial run, unless the site is individually GOLD/SILVER
            if site in flagged_indices:
                if status not in ('GOLD', 'SILVER'):
                    status = 'LIKELY_ERROR'
                    reason = f"Spatially contiguous run of selected sites; {reason}"
                else:
                    reason = f"Spatially contiguous run of selected sites (high-quality site preserved); {reason}"
                
            c.execute("update site_results set classification=?, filter_reason=? where gene_name=? and site_index=?", (status, reason, gene, site))
            
        conn.commit()
        updated_count += len(sites)
        if updated_count % 1000 == 0 or updated_count == len(sig_rows):
            print(f"  Updated {updated_count} / {len(sig_rows)} sites...")
            
    # Set all non-significant sites to NONE if not already done
    c.execute("update site_results set classification='NONE' where is_significant=0 and (classification != 'NONE' or classification IS NULL)")
    conn.commit()
    
    print("Database updated successfully!")
    conn.close()

if __name__ == "__main__":
    main()
