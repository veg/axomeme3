#!/usr/bin/env python3
"""
mine_meme_physical_coupling.py
------------------------------
Mines /Users/sergei/Projects/TOGA_MEME/meme_results.db (table 'site_results')
to empirically test how sensible the physical coupling constraint is:
  Excess = max(0, beta_pos - alpha) * (1.0 - p_neg)
  vs MEME LRT
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import scipy.stats as stats

DB_PATH = "/Users/sergei/Projects/TOGA_MEME/meme_results.db"

def run():
    if not os.path.exists(DB_PATH):
        print(f"[!] DB file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Inspect site_results columns
    cursor.execute("PRAGMA table_info(site_results);")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"[*] 'site_results' columns: {cols}\n")

    df = pd.read_sql_query("SELECT * FROM site_results", conn)
    conn.close()

    print(f"[*] Total site records in 'site_results': {len(df):,}")

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'lrt' in cl: col_map['lrt'] = col
        elif 'alpha' in cl: col_map['alpha'] = col
        elif 'beta_neg' in cl or 'beta_1' in cl or 'beta1' in cl: col_map['beta_neg'] = col
        elif 'beta_pos' in cl or 'beta_2' in cl or 'beta2' in cl: col_map['beta_pos'] = col
        elif 'p_neg' in cl or 'p_1' in cl or 'q_neg' in cl or 'prop_neg' in cl or 'p_pos' in cl: col_map['p_pos'] = col
        elif 'pval' in cl: col_map['pval'] = col

    print(f"[*] Mapped Columns: {col_map}")

    lrt_col = col_map.get('lrt')
    alpha_col = col_map.get('alpha')
    beta_pos_col = col_map.get('beta_pos')

    if not all([lrt_col, alpha_col, beta_pos_col]):
        print(f"[!] Could not map all required columns. Columns: {cols}")
        return

    # Check which proportion column exists
    p_col = None
    for possible_p in ['p_neg', 'p_pos', 'p_1', 'p_2', 'q_neg', 'q_pos', 'prop_pos', 'prop_neg']:
        if possible_p in df.columns:
            p_col = possible_p
            break

    if p_col is None:
        print(f"[!] Proportion column not found in {cols}")
        return

    sub_df = df[[lrt_col, alpha_col, beta_pos_col, p_col]].dropna().copy()
    sub_df.columns = ['lrt', 'alpha', 'beta_pos', 'p_val_or_prop']

    sub_df['lrt'] = pd.to_numeric(sub_df['lrt'], errors='coerce')
    sub_df['alpha'] = pd.to_numeric(sub_df['alpha'], errors='coerce')
    sub_df['beta_pos'] = pd.to_numeric(sub_df['beta_pos'], errors='coerce')
    sub_df['p_val_or_prop'] = pd.to_numeric(sub_df['p_val_or_prop'], errors='coerce')

    sub_df = sub_df.dropna()

    # Calculate Physical Excess:
    # If p_col is p_neg (proportion under conserved rate), prop_pos = (1 - p_neg)
    # If p_col is p_pos (proportion under positive selection), prop_pos = p_pos
    if 'neg' in p_col:
        sub_df['prop_pos'] = np.clip(1.0 - sub_df['p_val_or_prop'], 0.0, 1.0)
    else:
        sub_df['prop_pos'] = np.clip(sub_df['p_val_or_prop'], 0.0, 1.0)

    sub_df['beta_diff'] = np.maximum(0.0, sub_df['beta_pos'] - sub_df['alpha'])
    sub_df['excess'] = sub_df['beta_diff'] * sub_df['prop_pos']

    sub_df['lrt_log'] = np.log1p(np.maximum(0.0, sub_df['lrt']))
    sub_df['excess_log'] = np.log1p(sub_df['excess'])

    print(f"[*] Valid parsed rows for analysis: {len(sub_df):,}")

    corr_spearman, p_spearman = stats.spearmanr(sub_df['excess'], sub_df['lrt'])
    corr_pearson, p_pearson = stats.pearsonr(sub_df['excess_log'], sub_df['lrt_log'])

    print("\n" + "="*80)
    print("📊 EMPIRICAL PHYSICAL COUPLING CORRELATION IN MEME DATABASE")
    print("="*80)
    print(f"  - Spearman Rank Correlation [Excess vs LRT]:           {corr_spearman:.4f} (p = {p_spearman:.4e})")
    print(f"  - Pearson Correlation [Log(Excess+1) vs Log(LRT+1)]:   {corr_pearson:.4f} (p = {p_pearson:.4e})")

    pos_sub = sub_df[sub_df['lrt'] > 0].copy()
    if not pos_sub.empty:
        pos_spearman, _ = stats.spearmanr(pos_sub['excess'], pos_sub['lrt'])
        pos_pearson, _ = stats.pearsonr(pos_sub['excess_log'], pos_sub['lrt_log'])
        print(f"\n  - Positive Selection Subset (LRT > 0, N={len(pos_sub):,}):")
        print(f"    * Spearman Rank Correlation:                         {pos_spearman:.4f}")
        print(f"    * Pearson Log-Log Correlation:                       {pos_pearson:.4f}")

    slope, intercept, r_val, p_val, std_err = stats.linregress(sub_df['excess_log'], sub_df['lrt_log'])
    r2 = r_val ** 2

    print(f"\n  - Fitted Functional Relationship:")
    print(f"    Log(LRT + 1) = {slope:.4f} * Log(Excess + 1) + {intercept:.4f}")
    print(f"    R^2 (Coefficient of Determination):                  {r2:.4f}")

if __name__ == "__main__":
    run()
