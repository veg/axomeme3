import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import time
import re
import gzip
import math
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# Standard genetic code dictionary
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGA':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

# Codon vocabulary setup (64 codons + gap + unknown)
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

def get_codon_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 64
    if len(codon) != 3 or 'N' in codon:
        return 65
    return CODON_TO_IDX.get(codon, 65)

# --- Lightweight Newick Parser and Distance Calculator ---
class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

def parse_newick(newick_str):
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str.strip())  # remove annotations
    newick_str = re.sub(r'\[.*?\]', '', newick_str)            # remove comments
    
    tokens = []
    i = 0
    while i < len(newick_str):
        c = newick_str[i]
        if c in '(),;':
            tokens.append(c)
            i += 1
        elif c == ':':
            i += 1
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;':
                i += 1
            tokens.append(('length', float(newick_str[start:i])))
        else:
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;:':
                i += 1
            tokens.append(('name', newick_str[start:i].strip()))
            
    root = TreeNode()
    current = root
    for t in tokens:
        if t == '(':
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ',':
            current = current.parent
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ')':
            current = current.parent
        elif isinstance(t, tuple) and t[0] == 'name':
            current.name = t[1]
        elif isinstance(t, tuple) and t[0] == 'length':
            current.length = t[1]
            
    return root

def get_path_to_root(node):
    path = []
    curr = node
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path

def get_leaves(node):
    if not node.children:
        return [node]
    leaves = []
    for c in node.children:
        leaves.extend(get_leaves(c))
    return leaves

def get_patristic_distances(root):
    leaves = get_leaves(root)
    leaf_paths = {}
    leaf_len_to_root = {}
    for leaf in leaves:
        if not leaf.name:
            continue
        path = get_path_to_root(leaf)
        leaf_paths[leaf.name] = path
        length = 0.0
        for node in path[:-1]:
            length += node.length
        leaf_len_to_root[leaf.name] = length
        
    names = list(leaf_len_to_root.keys())
    n = len(names)
    dist_matrix = {}
    for name in names:
        dist_matrix[name] = {name: 0.0}
        
    for i in range(n):
        name1 = names[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, n):
            name2 = names[j]
            path2 = leaf_paths[name2]
            lca = None
            for node in path2:
                if node in set1:
                    lca = node
                    break
            
            if lca is not None:
                lca_path = get_path_to_root(lca)
                lca_len = sum(node.length for node in lca_path[:-1])
                dist = leaf_len_to_root[name1] + leaf_len_to_root[name2] - 2 * lca_len
            else:
                dist = leaf_len_to_root[name1] + leaf_len_to_root[name2]
                
            dist_matrix[name1][name2] = dist
            dist_matrix[name2][name1] = dist
            
    return names, dist_matrix

# --- PyTorch Dataset Loader ---
class MSADataset(Dataset):
    def __init__(self, db_path, msa_dir, sites_list, window_size=5, max_species=100):
        """
        sites_list: list of tuples (gene_name, site_index_1based, label)
        """
        self.db_path = db_path
        self.msa_dir = msa_dir
        self.sites = sites_list
        self.window_size = window_size
        self.max_species = max_species
        self.alignment_cache = {}
        
    def __len__(self):
        return len(self.sites)
        
    def _load_msa(self, gene_name):
        if gene_name in self.alignment_cache:
            return self.alignment_cache[gene_name]
            
        align_path = os.path.join(self.msa_dir, f"{gene_name}.gz")
        if not os.path.exists(align_path):
            return None
            
        taxlabels = []
        sequences = []
        tree_str = None
        in_taxlabels = False
        in_matrix = False
        in_trees = False
        
        try:
            with gzip.open(align_path, 'rt') as f:
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
        except Exception as e:
            print(f"Error loading MSA for {gene_name}: {e}")
            return None
            
        seq_dict = {}
        for label, seq in zip(taxlabels, sequences):
            seq_dict[label] = seq
            
        dist_matrix = {}
        species_names = []
        if tree_str:
            try:
                tree_root = parse_newick(tree_str)
                species_names, dist_matrix = get_patristic_distances(tree_root)
            except Exception as e:
                pass
                
        if not species_names:
            species_names = list(seq_dict.keys())
            dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}
            
        # Precompute the distance matrix as a PyTorch tensor to bypass slow Python nested loops
        n_spec = len(species_names)
        dist_arr = np.zeros((n_spec, n_spec), dtype=np.float32)
        for i, spec1 in enumerate(species_names):
            for j, spec2 in enumerate(species_names):
                dist_arr[i, j] = dist_matrix.get(spec1, {}).get(spec2, 0.0)
        dist_tensor_cached = torch.from_numpy(dist_arr)
            
        self.alignment_cache[gene_name] = (seq_dict, species_names, dist_tensor_cached)
        num_cached = len(self.alignment_cache)
        if num_cached % 50 == 0:
            print(f"    [DataLoader Cache] Loaded and parsed {num_cached} unique gene alignments/trees...", flush=True)
        return self.alignment_cache[gene_name]
        
    def __getitem__(self, idx):
        gene_name, site_idx, label = self.sites[idx]
        
        data = self._load_msa(gene_name)
        if data is None:
            dummy_msa = torch.zeros(self.max_species, self.window_size, dtype=torch.long)
            dummy_dist = torch.zeros(self.max_species, self.max_species)
            return dummy_msa, dummy_dist, torch.tensor(label, dtype=torch.float)
            
        seq_dict, species_names, dist_tensor_cached = data
        selected_species = species_names[:self.max_species]
        
        msa_tokens = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 65
        dist_tensor = torch.zeros(self.max_species, self.max_species)
        
        half_win = self.window_size // 2
        
        for s_idx, spec in enumerate(selected_species):
            seq = seq_dict.get(spec, "")
            seq_len_codons = len(seq) // 3
            
            for w_idx in range(self.window_size):
                codon_pos_1based = site_idx - half_win + w_idx
                if 1 <= codon_pos_1based <= seq_len_codons:
                    nuc_idx = (codon_pos_1based - 1) * 3
                    codon = seq[nuc_idx:nuc_idx+3]
                    msa_tokens[s_idx, w_idx] = get_codon_token(codon)
                    
        # Optimized tensor slicing to bypass slow python loop
        n_spec_actual = len(selected_species)
        if n_spec_actual > 0:
            dist_tensor[:n_spec_actual, :n_spec_actual] = dist_tensor_cached[:n_spec_actual, :n_spec_actual]
                
        return msa_tokens, dist_tensor, torch.tensor(label, dtype=torch.float)

# --- Custom Row Attention with Learnable Phylogenetic Bias ---
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.phylo_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix):
        batch_size, num_species, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = dist_matrix.unsqueeze(1)
        scores = scores - torch.exp(self.phylo_scale) * bias
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        return self.out_proj(out)

# --- Custom Multihead Attention Pooling to bypass Apple MPS bugs ---
# --- Custom Multihead Attention to bypass Apple MPS bugs ---
class StableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        
    def forward(self, query, key, value):
        # query shape: (batch_size, q_seq_len, embed_dim)
        # key shape: (batch_size, k_seq_len, embed_dim)
        # value shape: (batch_size, k_seq_len, embed_dim)
        batch_size, q_seq_len, _ = query.shape
        k_seq_len = key.shape[1]
        
        w_q, w_k, w_v = torch.chunk(self.in_proj_weight, 3, dim=0)
        b_q, b_k, b_v = torch.chunk(self.in_proj_bias, 3, dim=0)
        
        # Ensure chunks are contiguous to prevent Apple Silicon MPS offset/view buffer bugs
        w_q, w_k, w_v = w_q.contiguous(), w_k.contiguous(), w_v.contiguous()
        b_q, b_k, b_v = b_q.contiguous(), b_k.contiguous(), b_v.contiguous()
        
        q_proj = F.linear(query, w_q, b_q)
        k_proj = F.linear(key, w_k, b_k)
        v_proj = F.linear(value, w_v, b_v)
        
        q_h = q_proj.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)
        return self.out_proj(out)

# --- Custom Stable Transformer Encoder Layer ---
class StableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = StableAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        # Self-attention pooling
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_out))
        
        # Feed-forward network (matching relu activation default of original encoder)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src

# --- Nano-scale Axial Transformer Architecture ---
class PhyloAxialTransformer(nn.Module):
    def __init__(self, num_tokens=66, embed_dim=64, num_heads=4, num_layers=2, window_size=5, max_species=100, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_species = max_species
        
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        
        self.col_layers = nn.ModuleList([
            StableTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2*embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pool_attn = StableAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, msa, dist_matrix):
        batch_size, num_species, window_size = msa.shape
        
        x = self.embedding(msa)
        x = x + self.pos_embedding.unsqueeze(1)
        
        for i in range(len(self.col_layers)):
            col_in = x.reshape(batch_size * num_species, window_size, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x = col_out.reshape(batch_size, num_species, window_size, self.embed_dim)
            
            row_in = x.transpose(1, 2).contiguous().view(batch_size * window_size, num_species, self.embed_dim)
            dist_dup = dist_matrix.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species)
            
            row_out = self.row_layers[i](row_in, dist_dup)
            row_out = self.row_norms[i](row_in + row_out)
            x = row_out.reshape(batch_size, window_size, num_species, self.embed_dim).transpose(1, 2)
            
        central_idx = window_size // 2
        site_repr = x[:, :, central_idx, :]
        
        q = self.pool_query.expand(batch_size, -1, -1).contiguous()
        pooled_repr = self.pool_attn(q, site_repr, site_repr)
        pooled_repr = pooled_repr.squeeze(1)
        
        logits = self.mlp(pooled_repr)
        return logits.squeeze(1)

# --- Full Training & Validation Routine ---
def train_full_model(db_path="meme_results.db", msa_dir="msa", epochs=5, batch_size=128, lr=1e-3, subsample_limit=None):
    print("=" * 80)
    print("🚀 STEP 1: INITIALIZING HARDWARE ACCELERATION")
    print("=" * 80)
    # Determine local hardware acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" -> Detected Apple Silicon GPU! Utilizing MPS (Metal Performance Shaders) backend.")
        print("    This offloads heavy matrix calculations from the CPU to the GPU cores.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(" -> Detected NVIDIA GPU! Utilizing CUDA backend for hardware acceleration.")
    else:
        device = torch.device("cpu")
        print(" -> GPU acceleration not found. Training will run on the CPU (this will be slow).")
        
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found!")
        return
        
    print("\n" + "=" * 80)
    print("🚀 STEP 2: EXTRACTING AND SPLITTING DATASET (PREVENTING DATA LEAKAGE)")
    print("=" * 80)
    print(" -> Querying SQLite database for all labeled site-level results...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT gene_name, site_index, is_significant 
        FROM site_results 
        WHERE is_significant IS NOT NULL
    """)
    raw_sites = c.fetchall()
    conn.close()
    print(f" -> Successfully fetched {len(raw_sites):,} sites from the database.")
    
    # Shuffle and split by gene to avoid homology data leakage
    # Concept: If we put sites from the same gene in both train and validation sets,
    # the model might just memorize the specific gene structure (cheat) rather than
    # learning general evolutionary rules. Grouping by gene prevents this.
    all_genes = list(set(row[0] for row in raw_sites))
    random.seed(42)
    random.shuffle(all_genes)
    
    split_idx = int(len(all_genes) * 0.8)
    train_genes = set(all_genes[:split_idx])
    val_genes = set(all_genes[split_idx:])
    
    train_sites = [row for row in raw_sites if row[0] in train_genes]
    val_sites = [row for row in raw_sites if row[0] in val_genes]
    
    if subsample_limit:
        print(f" -> NOTE: Subsampling dataset to {subsample_limit} sites for demonstration/testing.")
        train_sites = random.sample(train_sites, min(len(train_sites), subsample_limit))
        val_sites = random.sample(val_sites, min(len(val_sites), subsample_limit // 4))
        
    print(f" -> Split Stats:")
    print(f"    - Total unique genes: {len(all_genes)}")
    print(f"    - Training: {len(train_genes)} genes ({len(train_sites):,} sites)")
    print(f"    - Validation: {len(val_genes)} genes ({len(val_sites):,} sites)")
    
    # Class balance adjustment
    # Concept: Positive selection sites are rare (~1.5% of the data).
    # If the model simply predicts 0 (no selection) for every site, it would get 98.5% accuracy
    # but be completely useless. We use a class weight to tell the model's loss function
    # to penalize errors on positive sites significantly more.
    train_labels = [row[2] for row in train_sites]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight_val = num_neg / max(1.0, num_pos)
    print(f" -> Class Balance:")
    print(f"    - Positive Sites (under selection): {num_pos:,}")
    print(f"    - Negative Sites (conserved/neutral): {num_neg:,}")
    print(f"    - Calculated pos_weight coefficient: {pos_weight_val:.2f}")
    print("      (Errors on positive sites will be penalized close to 65x more than negative ones)")
    
    print("\n" + "=" * 80)
    print("🚀 STEP 3: CREATING PYTORCH DATA LOADERS")
    print("=" * 80)
    # The DataLoader slices the data into batches and handles loading from disk in the background.
    train_dataset = MSADataset(db_path, msa_dir, train_sites, window_size=5)
    val_dataset = MSADataset(db_path, msa_dir, val_sites, window_size=5)
    
    drop_last_train = len(train_dataset) >= batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f" -> Training Batches: {len(train_loader)} (Batch Size: {batch_size})")
    print(f" -> Validation Batches: {len(val_loader)} (Batch Size: {batch_size})")
    
    print("\n" + "=" * 80)
    print("🚀 STEP 4: INSTANTIATING MODEL, OPTIMIZER, AND LOSS FUNCTION")
    print("=" * 80)
    model = PhyloAxialTransformer(embed_dim=64, num_heads=4, num_layers=2, window_size=5).to(device)
    
    # Optimizer (AdamW): Handles adjusting the model's weights based on calculated gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # Loss Function (weighted BCE): Measures how far the model's output is from the true label
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    
    best_pr_auc = 0.0
    
    print("\n" + "=" * 80)
    print("🚀 STEP 5: TRAINING LOOP STARTING")
    print("=" * 80)
    for epoch in range(1, epochs + 1):
        print(f"\n--- 🌟 STARTING EPOCH {epoch}/{epochs} ---")
        model.train()
        train_loss = 0.0
        
        start_epoch_time = time.time() if 'time' in sys.modules else None
        
        for batch_idx, (msa, dist, labels) in enumerate(train_loader):
            # Move data to GPU acceleration device (MPS/CUDA)
            msa = msa.to(device)
            dist = dist.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Feed alignments and tree distances through the transformer layers
            logits = model(msa, dist)
            # Calculate loss (error)
            loss = criterion(logits, labels)
            # Backward pass: Calculate gradients (backpropagation of errors)
            loss.backward()
            # Optimizer step: Tweak the network weights slightly to reduce loss
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                # Log progress
                phylo_scales = torch.exp(model.row_layers[0].phylo_scale).detach().cpu().squeeze().tolist()
                scales_str = ", ".join(f"{s:.3f}" for s in (phylo_scales if isinstance(phylo_scales, list) else [phylo_scales]))
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Tree Scales: [{scales_str}]")
                
        avg_train_loss = train_loss / max(1, len(train_loader))
        
        print(f"\n -> Epoch {epoch} training complete. Running validation evaluation...")
        
        # Validation cycle: Evaluate model on unseen validation genes
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad(): # Disable gradient calculations to save memory and speed up
            for msa, dist, labels in val_loader:
                msa = msa.to(device)
                dist = dist.to(device)
                
                logits = model(msa, dist)
                probs = torch.sigmoid(logits)
                
                val_preds.extend(probs.cpu().tolist())
                val_targets.extend(labels.tolist())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Calculate standard ML metrics
        # Concept: ROC AUC measures ranking quality (is a selected site ranked higher than a non-selected site).
        # PR AUC (Average Precision) measures precision-recall balance. For heavily imbalanced datasets,
        # PR AUC is the most critical metric. We want this as high as possible.
        if len(np.unique(val_targets)) > 1:
            roc_auc = roc_auc_score(val_targets, val_preds)
            pr_auc = average_precision_score(val_targets, val_preds)
        else:
            roc_auc, pr_auc = 0.5, 0.0
            
        print(f"\n📈 Epoch {epoch} Metrics Summary:")
        print(f"  - Average Training Loss: {avg_train_loss:.4f} (lower is better)")
        print(f"  - Validation ROC AUC: {roc_auc:.4f} (ranges 0.5 to 1.0, random guess is 0.5)")
        print(f"  - Validation PR AUC (Average Precision): {pr_auc:.4f} (random baseline is {num_pos / (num_pos + num_neg):.4f})")
        
        # Save best model checkpoint
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            checkpoint_path = "selection_transformer_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pr_auc': best_pr_auc,
            }, checkpoint_path)
            print(f"  💾 SUCCESS: Saved new best model checkpoint to '{checkpoint_path}' (PR AUC improved to {best_pr_auc:.4f})")
        else:
            print(f"  ℹ️ Checkpoint not saved (Validation PR AUC {pr_auc:.4f} did not exceed best of {best_pr_auc:.4f})")
            
    print("\n" + "=" * 80)
    print("🎉 MODEL TRAINING COMPLETE!")
    print(f"   Best Validation PR AUC: {best_pr_auc:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    import time
    # Set subsample_limit=None to run on the full 1.82 million sites dataset!
    # batch_size=128 is optimal for the M3 Ultra's GPU core count and memory bandwidth.
    train_full_model(epochs=5, batch_size=128, subsample_limit=None)
