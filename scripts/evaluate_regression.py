#!/usr/bin/env python3
import os
import sys
import random
import math
import sqlite3
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Ensure the scripts folder is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_regression import PhyloAxialTransformer, MSADataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default="/Users/sergei/Dropbox/TOGA2026/meme_results.db", help="Path to SQLite database")
    parser.add_argument("--msa_dir", default="/Users/sergei/Dropbox/TOGA2026/msa", help="Path to MSA directory")
    parser.add_argument("--model_path", default="/Users/sergei/Documents/MEME_transformer_joint.pt", help="Path to model checkpoint")
    parser.add_argument("--report_path", default="/Users/sergei/.gemini/antigravity-cli/brain/4d4d9064-3782-414f-ab21-0e69efdfe53e/transformer_regression_validation_report.md", help="Path to validation report")
    args = parser.parse_args()
    
    db_path = args.db_path
    msa_dir = args.msa_dir
    model_path = args.model_path
    artifact_path = args.report_path
    
    print("Connecting to database...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT gene_name, site_index, lrt, p_value, is_significant 
        FROM site_results 
        WHERE lrt IS NOT NULL
    """)
    raw_sites = c.fetchall()
    conn.close()
    
    print(f"Fetched {len(raw_sites):,} sites from database.")
    
    # Recreate the exact gene split used during training (seed 42)
    all_genes = sorted(list(set(row[0] for row in raw_sites)))
    random.seed(42)
    random.shuffle(all_genes)
    
    split_idx = int(len(all_genes) * 0.8)
    train_genes = set(all_genes[:split_idx])
    val_genes = set(all_genes[split_idx:])
    
    print(f"Split details: {len(train_genes)} train genes, {len(val_genes)} validation genes.")
    
    # Filter validation sites
    val_sites = [row for row in raw_sites if row[0] in val_genes]
    
    # Find validation genes that have at least one significant site (p_value <= 0.01)
    val_genes_with_sig = {}
    for gene_name, site_index, lrt, p_val, is_sig in val_sites:
        if gene_name not in val_genes_with_sig:
            val_genes_with_sig[gene_name] = []
        val_genes_with_sig[gene_name].append((site_index, lrt, p_val, is_sig))
        
    eligible_val_genes = sorted([g for g, sites in val_genes_with_sig.items() if any(s[3] == 1 for s in sites)])
    print(f"Found {len(eligible_val_genes)} validation genes with at least one significant site (p <= 0.01).")
    
    # Pick 4 random validation genes to inspect in detail
    random.seed(12345)
    selected_genes = random.sample(eligible_val_genes, min(4, len(eligible_val_genes)))
    print(f"Selected validation genes for detailed evaluation: {selected_genes}")
    
    # Load model checkpoint
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Running inference on device: {device}")
    
    print(f"Loading transformer model from {model_path}...")
    model = PhyloAxialTransformer(embed_dim=128, num_heads=8, num_layers=4, window_size=1, max_species=256).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    report_data = []
    
    # Run evaluation on each selected gene
    for gene in selected_genes:
        print(f"\nEvaluating gene: {gene}...")
        gene_sites = [row for row in val_sites if row[0] == gene]
        # Sort by site index
        gene_sites.sort(key=lambda x: x[1])
        
        # Instantiate dataset/dataloader for this single gene
        # sites are formatted as (gene_name, site_index, label_log_lrt_plus_1)
        dataset_sites = []
        for row in gene_sites:
            gene_name, site_idx, lrt, p_val, is_sig = row
            y = math.log(max(0.0, lrt) + 1.0)
            dataset_sites.append((gene_name, site_idx, y))
            
        dataset = MSADataset(db_path, msa_dir, dataset_sites, window_size=1, max_species=256, subsample_species=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        
        preds_log_lrt = []
        with torch.no_grad():
            for codon_tokens, aa_tokens, dist, mds_coords, padding_mask, _ in loader:
                codon_tokens = codon_tokens.to(device)
                aa_tokens = aa_tokens.to(device)
                dist = dist.to(device)
                mds_coords = mds_coords.to(device)
                padding_mask = padding_mask.to(device)
                logits = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
                preds_log_lrt.extend(logits.cpu().numpy())
                
        # Build DataFrame of results
        results = []
        for row, pred_log_lrt in zip(gene_sites, preds_log_lrt):
            gene_name, site_idx, lrt, p_val, is_sig = row
            pred_log_val = float(pred_log_lrt)
            pred_lrt_val = max(0.0, math.exp(pred_log_val) - 1.0)
            results.append({
                "site_idx": site_idx,
                "true_lrt": lrt,
                "p_value": p_val,
                "is_sig": is_sig,
                "pred_log_lrt": pred_log_val,
                "pred_lrt": pred_lrt_val
            })
        df_res = pd.DataFrame(results)
        
        # Calculate metrics
        y_true = df_res["true_lrt"].values
        y_pred = df_res["pred_lrt"].values
        
        # Pearson and Spearman correlations
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            p_r, _ = pearsonr(y_true, y_pred)
            s_rho, _ = spearmanr(y_true, y_pred)
        else:
            p_r, s_rho = np.nan, np.nan
            
        print(f"  Pearson correlation R: {p_r:.4f}" if not np.isnan(p_r) else "  Pearson R: N/A")
        print(f"  Spearman correlation Rho: {s_rho:.4f}" if not np.isnan(s_rho) else "  Spearman Rho: N/A")
        
        # Find top 10 sites with highest predicted LRT
        top_preds = df_res.sort_values(by="pred_lrt", ascending=False).head(10)
        # Find all actual significant sites
        sig_sites = df_res[df_res["is_sig"] == 1].sort_values(by="p_value")
        
        report_data.append({
            "gene": gene,
            "pearson_r": p_r,
            "spearman_rho": s_rho,
            "df_res": df_res,
            "top_preds": top_preds,
            "sig_sites": sig_sites
        })
        
    # Generate a beautiful Markdown report
    print(f"\nWriting detailed evaluation report to {artifact_path}...")
    
    with open(artifact_path, "w") as f:
        f.write("# Continuous Regression Transformer Model Validation Report\n\n")
        f.write("This report evaluates the trained continuous selection **PhyloAxialTransformer** on selected validation genes. ")
        f.write("These genes were completely excluded from the training dataset (preventing any data leakage) ")
        f.write("and are evaluated against the true selection signals calculated by **MEME**.\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write("| Gene Name | Total Sites | Significant Sites (p <= 0.01) | Pearson R | Spearman Rho |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for g_data in report_data:
            gene = g_data["gene"]
            df_res = g_data["df_res"]
            total_sites = len(df_res)
            sig_count = len(df_res[df_res["is_sig"] == 1])
            r_str = f"{g_data['pearson_r']:.4f}" if not np.isnan(g_data['pearson_r']) else "N/A"
            rho_str = f"{g_data['spearman_rho']:.4f}" if not np.isnan(g_data['spearman_rho']) else "N/A"
            f.write(f"| {gene} | {total_sites:,} | {sig_count} | {r_str} | {rho_str} |\n")
            
        f.write("\n## Detailed Site Comparisons\n\n")
        for g_data in report_data:
            gene = g_data["gene"]
            f.write(f"### Gene: {gene}\n\n")
            
            f.write("#### Significant Sites (MEME p <= 0.01) & Transformer Predictions:\n\n")
            f.write("| Site Index | MEME p-value | True LRT | Predicted LRT | Predicted log(LRT+1) |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for _, row in g_data["sig_sites"].iterrows():
                f.write(f"| {int(row['site_idx'])} | {row['p_value']:.4e} | {row['true_lrt']:.2f} | **{row['pred_lrt']:.2f}** | {row['pred_log_lrt']:.4f} |\n")
                
            f.write("\n#### Top 10 Sites Predicted by Transformer:\n\n")
            f.write("| Site Index | MEME p-value | True LRT | Predicted LRT | Predicted log(LRT+1) | True Label (Sig) |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for _, row in g_data["top_preds"].iterrows():
                sig_marker = f"**{int(row['is_sig'])}**" if row['is_sig'] == 1 else f"{int(row['is_sig'])}"
                f.write(f"| {int(row['site_idx'])} | {row['p_value']:.4e} | {row['true_lrt']:.2f} | **{row['pred_lrt']:.2f}** | {row['pred_log_lrt']:.4f} | {sig_marker} |\n")
            f.write("\n---\n\n")
            
    print("Report written successfully.")

if __name__ == "__main__":
    main()
