#!/usr/bin/env python3
import gzip
import os
import sqlite3
import re
import json

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
    # returns (status, reason)
    # Status can be: 'GOLD', 'SILVER', 'BRONZE'
    
    # site_idx is 1-based.
    # Nucleotide indices:
    site_start = (site_idx - 1) * 3
    
    # Local window index range: ±10 codons
    win_start = max(0, (site_idx - 11) * 3)
    win_end = (site_idx + 10) * 3 # python slice is exclusive
    
    # Analyze each substitution branch
    has_valid_single = False
    has_valid_mns = False
    
    failure_reasons = []
    
    for branch, anc_codon, der_codon in substitutions:
        dist = get_codon_hamming_dist(anc_codon, der_codon)
        is_mns = dist >= 2
        
        # Identify the carrying taxa
        carrying_leaves = []
        is_leaf = branch in leaves_set or any(l.upper() == branch.upper() for l in leaves_set)
        
        if is_leaf:
            # exact match
            match_leaf = [l for l in leaves_set if l.upper() == branch.upper()][0]
            carrying_leaves = [match_leaf]
        else:
            # internal node, get all descendant leaves
            match_node = [n for n in tree_children.keys() if n.upper() == branch.upper()]
            if match_node:
                carrying_leaves = get_descendant_leaves(match_node[0], tree_children, leaves_set)
            else:
                carrying_leaves = []
                
        if not carrying_leaves:
            # Can't find carrying leaves, skip branch
            continue
            
        # Check local window quality on carrying leaves
        total_leaves = len(carrying_leaves)
        valid_leaves_count = 0
        local_gap_densities = []
        short_fragment_failures = 0
        flanking_clean_for_mns = True
        
        for leaf in carrying_leaves:
            seq = seqs.get(leaf, "")
            if not seq:
                continue
                
            # Check if codon is valid (not gaps/Ns)
            codon = seq[site_start:site_start+3].upper()
            if '-' in codon or 'N' in codon or len(codon) < 3:
                continue
            
            valid_leaves_count += 1
            
            # Local gap/N density
            subseq = seq[win_start:win_end]
            gaps = subseq.count('-')
            ns = subseq.lower().count('n')
            density = (gaps + ns) / len(subseq) if subseq else 1.0
            local_gap_densities.append(density)
            
            # Fragment length check (contiguous non-gap block around the site)
            # Scan left
            left = site_start
            while left > 0 and seq[left-1] != '-' and seq[left-1].upper() != 'N':
                left -= 1
            # Scan right
            right = site_start + 3
            while right < len(seq) and seq[right] != '-' and seq[right].upper() != 'N':
                right += 1
            block_len = right - left
            if block_len < 30: # less than 10 codons
                short_fragment_failures += 1
                
            # MNS Flanking window check (±5 codons, i.e., 30 bp)
            if is_mns:
                fl_start = max(0, (site_idx - 6) * 3)
                fl_end = (site_idx + 5) * 3
                fl_seq = seq[fl_start:fl_end]
                fl_gaps = fl_seq.count('-') + fl_seq.lower().count('n')
                if fl_gaps > 0:
                    flanking_clean_for_mns = False
                    
        # Apply filters to this branch
        if total_leaves > 1:
            # Internal node filters
            valid_frac = valid_leaves_count / total_leaves
            mean_density = sum(local_gap_densities) / len(local_gap_densities) if local_gap_densities else 1.0
            
            if valid_frac < 0.5:
                failure_reasons.append(f"Branch {branch}: low descendant representation ({valid_frac:.1f})")
                continue
            if mean_density > 0.3:
                failure_reasons.append(f"Branch {branch}: high descendant gap density ({mean_density:.1f})")
                continue
        else:
            # Leaf node filters
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
            
    # Classify the site based on the branch results
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
    
    # Get list of all significant sites
    c.execute("select gene_name, site_index, p_value from site_results where is_significant=1")
    sig_rows = c.fetchall()
    print(f"Total significant sites to evaluate: {len(sig_rows)}")
    
    # Group by gene
    genes_to_sites = {}
    for gene, site, pval in sig_rows:
        genes_to_sites.setdefault(gene, []).append((site, pval))
        
    gold_count = 0
    silver_count = 0
    bronze_count = 0
    skipped_count = 0
    
    processed_count = 0
    
    # For reporting
    reasons_count = {}
    
    for gene, sites in sorted(genes_to_sites.items()):
        # Load alignment
        align_path = f"msa/{gene}.gz"
        if not os.path.exists(align_path):
            skipped_count += len(sites)
            continue
            
        try:
            seqs, tree_str = parse_nexus_gz(align_path)
            parents = parse_newick(tree_str)
            
            # Build children dict
            children = {}
            for child, parent in parents.items():
                children.setdefault(parent, []).append(child)
                
            leaves_set = set(seqs.keys())
        except Exception as e:
            print(f"Error parsing alignment for {gene}: {e}")
            skipped_count += len(sites)
            continue
            
        for site, pval in sites:
            # Get substitutions for this site
            c.execute("""
                select branch_name, ancestral_codon, derived_codon 
                from site_substitutions 
                where gene_name=? and site_index=? and is_synonymous=0
            """, (gene, site))
            subst = c.fetchall()
            
            if not subst:
                # No non-synonymous substitutions recorded
                bronze_count += 1
                reasons_count["No non-synonymous substitutions recorded"] = reasons_count.get("No non-synonymous substitutions recorded", 0) + 1
                continue
                
            status, reason = analyze_site(seqs, parents, children, leaves_set, site, subst)
            
            if status == 'GOLD':
                gold_count += 1
            elif status == 'SILVER':
                silver_count += 1
            else:
                bronze_count += 1
                # Truncate reason to keep it clean
                trunc_reason = reason.split("; ")[0] if "; " in reason else reason
                reasons_count[trunc_reason] = reasons_count.get(trunc_reason, 0) + 1
                
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count} sites...")
                
    total_valid = gold_count + silver_count
    total_evaluated = gold_count + silver_count + bronze_count
    
    print("\n================ FILTERING RESULTS ================")
    print(f"Total significant sites evaluated: {total_evaluated}")
    print(f"  GOLD (Robust Single Nucleotide): {gold_count} ({gold_count/total_evaluated*100:.1f}%)")
    print(f"  SILVER (Robust MNS):            {silver_count} ({silver_count/total_evaluated*100:.1f}%)")
    print(f"  BRONZE (Artefactual/Flagged):   {bronze_count} ({bronze_count/total_evaluated*100:.1f}%)")
    print(f"Skipped (missing alignment):      {skipped_count}")
    
    print("\nTop reasons for flagging (BRONZE):")
    for reason, count in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {reason}: {count} ({count/bronze_count*100:.1f}%)")
        
    conn.close()

if __name__ == "__main__":
    main()
