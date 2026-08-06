#!/usr/bin/env python3
import os
import sys
import re
import gzip
import sqlite3

DB_PATH = "meme_results.db"
MASTER_TREE_PATH = "docs/pruned_species_tree.nhx"
TAXON_MAP_PATH = "docs/species_to_taxon_map.tsv"
MSA_DIR = "msa"

class Node:
    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None

def parse_newick_to_tree(newick_str):
    root = Node()
    current = root
    i = 0
    while i < len(newick_str):
        c = newick_str[i]
        if c == '(':
            child = Node()
            child.parent = current
            current.children.append(child)
            current = child
            i += 1
        elif c == ',':
            parent = current.parent
            child = Node()
            child.parent = parent
            parent.children.append(child)
            current = child
            i += 1
        elif c == ')':
            current = current.parent
            i += 1
        elif c == ';' or c == ' ':
            i += 1
        else:
            match = re.match(r'[^(),;:]+', newick_str[i:])
            if match:
                name = match.group(0).strip()
                name = re.sub(r'\{[^}]*\}', '', name) # remove annotations
                current.name = name
                i += len(match.group(0))
            if i < len(newick_str) and newick_str[i] == ':':
                i += 1
                match_len = re.match(r'[0-9.eE+-]+', newick_str[i:])
                if match_len:
                    i += len(match_len.group(0))
    return root

def assign_internal_names(node, counter=[1]):
    if node.children:
        node.name = f"M_NODE_{counter[0]}"
        counter[0] += 1
    for child in node.children:
        assign_internal_names(child, counter)

def build_parent_map(node, parent_map):
    for child in node.children:
        parent_map[child.name] = node.name
        build_parent_map(child, parent_map)

def get_descendants_map(node, desc_map):
    if not node.children:
        desc_map[node.name] = {node.name}
        return {node.name}
    leaves = set()
    for child in node.children:
        leaves.update(get_descendants_map(child, desc_map))
    desc_map[node.name] = leaves
    return leaves

def find_mrca_two(u, v, parent_map):
    u_ancestors = set()
    curr = u
    while curr:
        u_ancestors.add(curr)
        curr = parent_map.get(curr)
    curr = v
    while curr:
        if curr in u_ancestors:
            return curr
        curr = parent_map.get(curr)
    return None

def find_mrca_set(L, parent_map):
    if not L:
        return None
    leaves = list(L)
    curr_mrca = leaves[0]
    for leaf in leaves[1:]:
        curr_mrca = find_mrca_two(curr_mrca, leaf, parent_map)
        if not curr_mrca:
            break
    return curr_mrca

def parse_nexus_tree(filepath):
    tree_str = None
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line_strip = line.strip()
            if line_strip.upper().startswith("TREE "):
                parts = line_strip.split('=', 1)
                if len(parts) > 1:
                    tree_str = parts[1].strip()
                    break
    return tree_str

def main():
    print("Pre-calculating branch mappings to master tree...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found!")
        sys.exit(1)
    if not os.path.exists(MASTER_TREE_PATH):
        print(f"Error: Master tree {MASTER_TREE_PATH} not found!")
        sys.exit(1)
        
    # 1. Load Species Metadata
    species_meta = {}
    if os.path.exists(TAXON_MAP_PATH):
        with open(TAXON_MAP_PATH, 'r') as f:
            header = f.readline().strip().split('\t')
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    leaf_upper = parts[4].strip().upper()
                    species_meta[leaf_upper] = {
                        "species": parts[0].strip(),
                        "common": parts[1].strip()
                    }
        print(f"Loaded metadata for {len(species_meta)} species.")
    else:
        print("Warning: species_to_taxon_map.tsv not found.")

    # 2. Parse and build Master Tree
    with open(MASTER_TREE_PATH, 'r') as f:
        master_newick = f.read().strip()
    master_root = parse_newick_to_tree(master_newick)
    assign_internal_names(master_root)
    
    parent_map = {}
    build_parent_map(master_root, parent_map)
    
    master_desc = {}
    get_descendants_map(master_root, master_desc)
    
    master_leaves_upper = {k.upper(): k for k in master_desc.keys() if not k.startswith('M_NODE_')}
    print(f"Master tree has {len(master_desc)} nodes ({len(master_leaves_upper)} leaves).")

    # 3. Open DB Connection and Setup Tables
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("DROP TABLE IF EXISTS master_nodes")
    c.execute("""
    CREATE TABLE master_nodes (
        node_id TEXT PRIMARY KEY,
        is_leaf INTEGER,
        node_label TEXT,
        descendants TEXT
    )""")
    
    c.execute("DROP TABLE IF EXISTS branch_mappings")
    c.execute("""
    CREATE TABLE branch_mappings (
        gene_name TEXT,
        branch_name TEXT,
        master_node_id TEXT,
        PRIMARY KEY (gene_name, branch_name)
    )""")
    
    # 4. Insert Master Tree Node Definitions
    master_node_rows = []
    for node_id, desc_leaves in master_desc.items():
        is_leaf = 0 if node_id.startswith('M_NODE_') else 1
        
        # Build a nice descriptive label
        if is_leaf:
            meta = species_meta.get(node_id.upper())
            if meta:
                node_label = f"{meta['common']} ({meta['species']})"
            else:
                node_label = node_id
        else:
            # Internal node: sort leaves and summarize
            common_names = []
            for leaf in desc_leaves:
                meta = species_meta.get(leaf.upper())
                if meta:
                    common_names.append(meta['common'])
                else:
                    common_names.append(leaf)
            common_names.sort()
            
            if len(common_names) <= 5:
                node_label = "Ancestor of: " + ", ".join(common_names)
            else:
                node_label = "Ancestor of: " + ", ".join(common_names[:5]) + f" and {len(common_names) - 5} others"
                
        descendants_str = ",".join(sorted(list(desc_leaves)))
        master_node_rows.append((node_id, is_leaf, node_label, descendants_str))
        
    c.executemany("INSERT INTO master_nodes VALUES (?,?,?,?)", master_node_rows)
    conn.commit()
    print(f"Inserted {len(master_node_rows)} master node definitions.")
    
    # 5. Fetch all genes
    c.execute("SELECT gene_name FROM gene_results")
    genes = [r[0] for r in c.fetchall()]
    print(f"Mapping branches for {len(genes)} genes...")
    
    mappings_to_insert = []
    mapped_count = 0
    skipped_count = 0
    
    for idx, gene in enumerate(genes):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(genes)} genes...")
            
        gene_path = os.path.join(MSA_DIR, f"{gene}.gz")
        if not os.path.exists(gene_path):
            skipped_count += 1
            continue
            
        tree_str = parse_nexus_tree(gene_path)
        if not tree_str:
            skipped_count += 1
            continue
            
        gene_root = parse_newick_to_tree(tree_str)
        gene_desc = {}
        get_descendants_map(gene_root, gene_desc)
        
        for g_node, g_leaves in gene_desc.items():
            if not g_node:
                continue
            # Find the corresponding leaves in the master tree
            m_leaves_focal = set()
            for l in g_leaves:
                l_upper = l.upper()
                if l_upper in master_leaves_upper:
                    m_leaves_focal.add(master_leaves_upper[l_upper])
            
            if not m_leaves_focal:
                continue
                
            # Find MRCA in master tree
            master_node_id = find_mrca_set(m_leaves_focal, parent_map)
            if master_node_id:
                # Save mapping with branch name in uppercase for exact DB join compatibility
                mappings_to_insert.append((gene, g_node.upper(), master_node_id))
                mapped_count += 1
                
    # Insert mappings into DB
    c.executemany("INSERT OR REPLACE INTO branch_mappings VALUES (?,?,?)", mappings_to_insert)
    
    # Create indexes for performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_bm_lookup ON branch_mappings(gene_name, branch_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_bm_master ON branch_mappings(master_node_id)")
    
    conn.commit()
    conn.close()
    
    print(f"Finished mapping. Mapped {mapped_count} branches. Skipped {skipped_count} genes due to missing alignments/trees.")

if __name__ == "__main__":
    main()
