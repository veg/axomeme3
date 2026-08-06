import os
import glob
import json
import gzip
import multiprocessing as mp

def parse_one(path):
    filename = os.path.basename(path)
    gene_name = filename.split('.')[0]
    try:
        try:
            with gzip.open(path, 'rt') as f:
                data = json.load(f)
        except Exception:
            with open(path, 'r') as f:
                data = json.load(f)
                
        # 1. Global dN/dS
        fits = data.get('fits', {})
        global_fit = fits.get('Global MG94xREV', {})
        rate_dists = global_fit.get('Rate Distributions', {})
        omega_key = 'non-synonymous/synonymous rate ratio for *test*'
        omega_list = rate_dists.get(omega_key, [])
        global_dnds = omega_list[0][0] if omega_list else None
        
        # 2. Branch length outliers
        branch_data = data.get('branch attributes', {}).get('0', {})
        lengths = [v.get('Global MG94xREV') for v in branch_data.values() if v.get('Global MG94xREV') is not None]
        
        outliers = []
        num_outliers = 0
        threshold = 0.0
        lengths_sorted = []
        
        if len(lengths) >= 4:
            lengths_sorted = sorted(lengths)
            n = len(lengths_sorted)
            q1 = lengths_sorted[int(0.25 * n)]
            q3 = lengths_sorted[int(0.75 * n)]
            iqr = q3 - q1
            threshold = max(q3 + 20.0 * iqr, 0.4) # more stringent threshold, at least 0.4
            
            for k, v in branch_data.items():
                l = v.get('Global MG94xREV')
                if l is not None and l > threshold:
                    outliers.append(v.get('original name', k))
            num_outliers = len(outliers)
            
        return gene_name, {
            'global_dnds': global_dnds,
            'num_outliers': num_outliers,
            'outlier_branches': outliers,
            'outlier_threshold': threshold,
            'branch_lengths': lengths_sorted
        }
    except Exception as e:
        return gene_name, None

def main():
    results_dir = "/Users/sergei/Projects/TOGA_MEME/meme_results"
    if not os.path.exists(results_dir):
        results_dir = "/Users/sergei/Dropbox/TOGA2026/scratch/meme_results"
    pattern = os.path.join(results_dir, "*.MEME.json.gz")
    files = glob.glob(pattern)
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(parse_one, files)
        
    stats = {}
    for gene_name, res in results:
        if res is not None:
            stats[gene_name] = res
            
    print(json.dumps(stats))

if __name__ == '__main__':
    main()
