import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

def main():
    base_dir = "/Users/sergei/Dropbox/TOGA2026"
    genes_csv = os.path.join(base_dir, "legacy_vs_toga_genes.csv")
    sites_csv = os.path.join(base_dir, "legacy_vs_toga_sites.csv")
    
    if not os.path.exists(genes_csv) or not os.path.exists(sites_csv):
        print("Data files not found! Please run compare_legacy_vs_toga.py first.")
        return
        
    df_genes = pd.read_csv(genes_csv)
    df_sites = pd.read_csv(sites_csv)
    
    # Set premium styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Panel 1: Scatter Plot of Number of Sites (x = TOGA, y = legacy)
    x = df_genes['toga_sig_01'].values
    y = df_genes['legacy_sig_01'].values
    
    # Run linear regression: y = m * x + c
    if len(x) > 1:
        m, c = np.polyfit(x, y, 1)
        # Calculate R^2
        y_pred = m * x + c
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        correlation = np.corrcoef(x, y)[0, 1]
    else:
        m, c = 0.0, 0.0
        r_squared = 0.0
        correlation = 0.0
        
    # Plot scatter points for genes
    ax1.scatter(x, y, alpha=0.7, color='#2B6CB0', edgecolor='none', s=60, label='Genes')
    
    # Plot regression line
    x_line = np.linspace(0, max(x) if len(x) > 0 else 100, 100)
    y_line = m * x_line + c
    ax1.plot(x_line, y_line, color='#E53E3E', linestyle='-', linewidth=2, 
             label=f'Regression line:\ny = {m:.3f}x + {c:.2f}\nR² = {r_squared:.3f}')
             
    # Plot reference line y = x (representing equal sensitivity)
    max_val = max(x.max(), y.max()) if len(x) > 0 else 100
    ax1.plot([0, max_val], [0, max_val], linestyle='--', color='#A0AEC0', alpha=0.8, label='y = x (equal sensitivity)')
    
    # Annotate top genes
    top_genes = df_genes.sort_values(by='toga_sig_01', ascending=False).head(6)
    for idx, row in top_genes.iterrows():
        ax1.annotate(row['gene'], (row['toga_sig_01'], row['legacy_sig_01']),
                     textcoords="offset points", xytext=(5, -5), ha='left', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E0", lw=0.5, alpha=0.8))
                     
    ax1.set_xlabel('Selected Sites in TOGA MSA (p <= 0.01)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Selected Sites in Legacy MSA (p <= 0.01)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('A: Selected Sites Comparison & Regression', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10, loc='upper left')
    
    # Panel 2: Scatter Plot of LRT statistics
    # Filter sites that are significant in at least one run
    df_sites_sig = df_sites[(df_sites['toga_sig_01'] == 1) | (df_sites['legacy_sig_01'] == 1)]
    
    ax2.scatter(df_sites_sig['toga_lrt'], df_sites_sig['legacy_lrt'], alpha=0.6, color='#5A67D8', edgecolor='none', s=40)
    
    # Reference line y = x
    max_val_lrt = max(df_sites_sig['toga_lrt'].max(), df_sites_sig['legacy_lrt'].max()) if len(df_sites_sig) > 0 else 50
    ax2.plot([0, max_val_lrt], [0, max_val_lrt], linestyle='--', color='#718096', alpha=0.8, label='y = x (equal support)')
    
    ax2.set_xlabel('TOGA LRT (Likelihood Ratio Test) Statistic', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Legacy LRT Statistic', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('B: Site-level Statistical Support (LRT)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    
    # Clean up layout
    plt.tight_layout()
    
    # Save the figure
    out_img_path = os.path.join(base_dir, "docs", "legacy_vs_toga_comparison.png")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {out_img_path}")
    
    # Copy to artifact directory
    conv_id = "4d4d9064-3782-414f-ab21-0e69efdfe53e"
    artifact_img_path = f"/Users/sergei/.gemini/antigravity-cli/brain/{conv_id}/legacy_vs_toga_comparison.png"
    shutil.copy(out_img_path, artifact_img_path)
    print(f"Comparison plot copied to artifact directory at {artifact_img_path}")

if __name__ == "__main__":
    main()
