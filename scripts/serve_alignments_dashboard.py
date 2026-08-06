#!/usr/bin/env python3
"""
serve_alignments_dashboard.py
-----------------------------
A zero-dependency Python HTTP server to serve the TOGA Alignments Curation Dashboard.
Exposes a lightweight REST API and serves the front-end HTML.
Caches expensive global aggregates on startup to ensure instant dashboard load times.
Integrates robust statistical outlier detection for sizes (codons & sequences) and binned histograms.
"""

import os
import sys
import json
import sqlite3
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

DB_PATH = "/Users/sergei/Dropbox/TOGA2026/alignments_stats.db"
HTML_PATH = "/Users/sergei/Dropbox/TOGA2026/docs/alignments_dashboard.html"

# In-memory cache for expensive global stats
stats_cache = {}

def load_global_stats():
    """Calculates global aggregates, outlier detection, and binned histograms on startup."""
    print("Pre-calculating global statistics for instant dashboard load...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}!", file=sys.stderr)
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1. Update schema to ensure outlier columns exist in alignments table
        cursor.execute("PRAGMA table_info(alignments)")
        cols = [row[1] for row in cursor.fetchall()]
        if 'is_outlier' not in cols:
            print("Adding is_outlier column to alignments table...")
            cursor.execute("ALTER TABLE alignments ADD COLUMN is_outlier INTEGER DEFAULT 0")
            conn.commit()
        if 'outlier_reasons' not in cols:
            print("Adding outlier_reasons column to alignments table...")
            cursor.execute("ALTER TABLE alignments ADD COLUMN outlier_reasons TEXT")
            conn.commit()

        # 2. Total Alignments
        cursor.execute("SELECT count(*) FROM alignments")
        total_alignments = cursor.fetchone()[0]
        
        # 3. Total Sequences
        cursor.execute("SELECT sum(num_sequences) FROM alignments")
        total_sequences = cursor.fetchone()[0] or 0
        
        # 4. Total Suspect Genes
        cursor.execute("SELECT count(*) FROM alignments WHERE is_sus_gene = 1")
        total_sus_genes = cursor.fetchone()[0]
        
        # 5. Total Pseudogenes
        cursor.execute("SELECT count(*) FROM alignments WHERE is_pseudogene = 1")
        total_pseudogenes = cursor.fetchone()[0]
        
        # 6. Reasons Breakdown
        print("Parsing sequence curation reject reasons...")
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN sus_reasons LIKE '%Premature stop%' THEN 1 ELSE 0 END) as premature_stop_count,
                SUM(CASE WHEN sus_reasons LIKE '%Low coverage%' THEN 1 ELSE 0 END) as low_coverage_count,
                SUM(CASE WHEN sus_reasons LIKE '%Empty sequence%' THEN 1 ELSE 0 END) as empty_sequence_count
            FROM sequences
            WHERE is_sus = 1
        """)
        reasons_row = cursor.fetchone()
        reasons_breakdown = {
            'Premature Stops': reasons_row[0] or 0,
            'Low Coverage (<30%)': reasons_row[1] or 0,
            'Empty Sequences': reasons_row[2] or 0
        }
        
        # 7. Robust Outlier Detection
        print("Running robust outlier detection (log10-IQR & log10-Z-score)...")
        cursor.execute("SELECT id, alignment_length, num_sequences FROM alignments")
        db_alignments = cursor.fetchall()
        
        lengths = [r[1] for r in db_alignments]
        codons = [l / 3 for l in lengths]
        seqs = [r[2] for r in db_alignments]
        
        import math
        
        # Codon bounds via Log10 IQR (robust to heavy right tail)
        log_codons = sorted([math.log10(c) for c in codons if c > 0])
        n_c = len(log_codons)
        if n_c > 0:
            q1_log = log_codons[int(0.25 * n_c)]
            q3_log = log_codons[int(0.75 * n_c)]
            iqr_log = q3_log - q1_log
            codon_lower_bound = 10 ** (q1_log - 1.5 * iqr_log)
            codon_upper_bound = 10 ** (q3_log + 1.5 * iqr_log)
        else:
            codon_lower_bound, codon_upper_bound = 10, 3000
            
        # Sequence count bounds via Log10 Z-score (robust left-tail detection)
        log_seqs = [math.log10(s) for s in seqs if s > 0]
        n_s = len(log_seqs)
        if n_s > 0:
            mean_log_seq = sum(log_seqs) / n_s
            var_log_seq = sum((x - mean_log_seq)**2 for x in log_seqs) / n_s
            std_log_seq = math.sqrt(var_log_seq) if var_log_seq > 0 else 1.0
            seq_lower_bound = 10 ** (mean_log_seq - 2.0 * std_log_seq)
        else:
            seq_lower_bound = 100
            
        # Identify outliers and compile database updates
        updates = []
        outliers_count = 0
        for al_id, al_len, num_seq in db_alignments:
            c_len = al_len / 3
            reasons = []
            if c_len < codon_lower_bound:
                reasons.append(f"Short ({c_len:.0f} codons)")
            elif c_len > codon_upper_bound:
                reasons.append(f"Long ({c_len:.0f} codons)")
                
            if num_seq < seq_lower_bound:
                reasons.append(f"Few seqs ({num_seq})")
                
            is_outlier = 1 if reasons else 0
            reasons_str = " | ".join(reasons) if is_outlier else None
            
            if is_outlier:
                outliers_count += 1
                
            updates.append((is_outlier, reasons_str, al_id))
            
        cursor.execute("BEGIN TRANSACTION")
        cursor.executemany("""
            UPDATE alignments
            SET is_outlier = ?,
                outlier_reasons = ?
            WHERE id = ?
        """, updates)
        conn.commit()
        
        # 8. Build Binned Histograms
        def get_hist(data, bins_count=15, is_log=False):
            if not data:
                return {'counts': [], 'bin_edges': [], 'bin_centers': []}
            if is_log:
                data = [math.log10(x) for x in data if x > 0]
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                max_val += 1.0
                
            width = (max_val - min_val) / bins_count
            counts = [0] * bins_count
            edges = [min_val + i * width for i in range(bins_count + 1)]
            
            for x in data:
                idx = int((x - min_val) / width)
                if idx >= bins_count:
                    idx = bins_count - 1
                if idx >= 0:
                    counts[idx] += 1
                    
            if is_log:
                edges = [10**x for x in edges]
                
            bin_centers = []
            for i in range(bins_count):
                if is_log:
                    center = math.sqrt(edges[i] * edges[i+1]) # geometric mean for log display
                else:
                    center = (edges[i] + edges[i+1]) / 2
                bin_centers.append(round(center, 1))
                
            return {
                'counts': counts,
                'bin_edges': [round(e, 1) for e in edges],
                'bin_centers': bin_centers
            }
            
        codon_hist = get_hist(codons, bins_count=15, is_log=True)
        seq_hist = get_hist(seqs, bins_count=15, is_log=False)
        
        # 9. Top Suspect Candidates
        cursor.execute("""
            SELECT id, resolved_gene_name, transcript_id, num_sus_sequences, num_sequences, biotype, 
                   (CAST(num_sus_sequences AS REAL)/num_sequences)*100 as sus_pct 
            FROM alignments 
            WHERE num_sequences > 5 
            ORDER BY sus_pct DESC, num_sus_sequences DESC 
            LIMIT 10
        """)
        top_sus_candidates = []
        for row in cursor.fetchall():
            top_sus_candidates.append({
                'id': row[0],
                'resolved_gene_name': row[1],
                'transcript_id': row[2],
                'num_sus_sequences': row[3],
                'num_sequences': row[4],
                'biotype': row[5],
                'sus_pct': row[6]
            })
            
        stats_cache.update({
            'total_alignments': total_alignments,
            'total_sequences': total_sequences,
            'total_sus_genes': total_sus_genes,
            'total_pseudogenes': total_pseudogenes,
            'reasons_breakdown': reasons_breakdown,
            'top_sus_candidates': top_sus_candidates,
            'outliers_count': outliers_count,
            'codon_stats': {
                'lower_bound': round(codon_lower_bound, 1),
                'upper_bound': round(codon_upper_bound, 1)
            },
            'seq_stats': {
                'lower_bound': round(seq_lower_bound, 1)
            },
            'codon_histogram': codon_hist,
            'seq_histogram': seq_hist
        })
        print(f"Pre-calculation complete! Outliers detected: {outliers_count}. Statistics cached.")
    except Exception as e:
        print(f"Error calculating stats: {e}", file=sys.stderr)
    finally:
        conn.close()

class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress standard request logging and keep shell output clean
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        # Route: Main Page
        if path in ('/', '/index.html'):
            self.serve_html_file()
            
        # Route: API Stats
        elif path == '/api/stats':
            self.send_json_response(stats_cache)
            
        # Route: API Alignments (Paginated catalog)
        elif path == '/api/alignments':
            self.handle_api_alignments(query_params)
            
        # Route: API Alignment Details
        elif path.startswith('/api/alignment/'):
            alignment_id_str = path.split('/')[-1]
            try:
                alignment_id = int(alignment_id_str)
                self.handle_api_alignment_detail(alignment_id)
            except ValueError:
                self.send_error(400, "Invalid Alignment ID")
                
        # Not Found
        else:
            self.send_error(404, "File Not Found")

    def serve_html_file(self):
        try:
            with open(HTML_PATH, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading dashboard file: {e}")

    def send_json_response(self, data):
        try:
            json_bytes = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(json_bytes))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json_bytes)
        except Exception as e:
            self.send_error(500, f"JSON Encoding Error: {e}")

    def handle_api_alignments(self, query_params):
        page = int(query_params.get('page', [1])[0])
        limit = int(query_params.get('limit', [15])[0])
        search = query_params.get('search', [''])[0].strip()
        filt = query_params.get('filter', ['all'])[0]
        sort_by = query_params.get('sort_by', ['resolved_gene_name'])[0]
        sort_dir = query_params.get('sort_dir', ['asc'])[0]
        
        offset = (page - 1) * limit
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build query filters
        where_clauses = []
        params = []
        
        if search:
            where_clauses.append("(resolved_gene_name LIKE ? OR transcript_id LIKE ? OR file_name LIKE ?)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param])
            
        if filt == 'sus':
            where_clauses.append("is_sus_gene = 1")
        elif filt == 'pseudogene':
            where_clauses.append("is_pseudogene = 1")
        elif filt == 'clean':
            where_clauses.append("is_sus_gene = 0")
        elif filt == 'outlier':
            where_clauses.append("is_outlier = 1")
            
        where_str = " AND ".join(where_clauses)
        if where_str:
            where_str = "WHERE " + where_str
            
        # Sorting validation
        allowed_sorts = {
            'resolved_gene_name': 'resolved_gene_name',
            'transcript_id': 'transcript_id',
            'alignment_length': 'alignment_length',
            'num_sequences': 'num_sequences',
            'num_sus_sequences': '(CAST(num_sus_sequences AS REAL)/num_sequences)'
        }
        sort_col = allowed_sorts.get(sort_by, 'resolved_gene_name')
        sort_order = 'DESC' if sort_dir.lower() == 'desc' else 'ASC'
        
        try:
            # Get total matching count
            count_query = f"SELECT count(*) FROM alignments {where_str}"
            cursor.execute(count_query, params)
            total_matches = cursor.fetchone()[0]
            
            # Fetch matching alignments
            query = f"""
                SELECT id, file_name, transcript_id, gene_name, resolved_gene_name, 
                       alignment_length, num_sequences, num_sus_sequences, 
                       is_pseudogene, is_sus_gene, sus_reasons, is_outlier, outlier_reasons 
                FROM alignments 
                {where_str} 
                ORDER BY {sort_col} {sort_order} 
                LIMIT ? OFFSET ?
            """
            cursor.execute(query, params + [limit, offset])
            
            alignments = []
            for row in cursor.fetchall():
                alignments.append({
                    'id': row[0],
                    'file_name': row[1],
                    'transcript_id': row[2],
                    'gene_name': row[3],
                    'resolved_gene_name': row[4],
                    'alignment_length': row[5],
                    'num_sequences': row[6],
                    'num_sus_sequences': row[7],
                    'is_pseudogene': bool(row[8]),
                    'is_sus_gene': bool(row[9]),
                    'sus_reasons': row[10],
                    'is_outlier': bool(row[11]),
                    'outlier_reasons': row[12]
                })
                
            response = {
                'total': total_matches,
                'page': page,
                'limit': limit,
                'alignments': alignments
            }
            self.send_json_response(response)
        except Exception as e:
            self.send_error(500, f"Database query error: {e}")
        finally:
            conn.close()

    def handle_api_alignment_detail(self, alignment_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Fetch alignment metadata
            cursor.execute("""
                SELECT id, file_name, transcript_id, gene_name, resolved_gene_name, 
                       alignment_length, num_sequences, num_sus_sequences, 
                       biotype, is_pseudogene, is_sus_gene, sus_reasons, is_outlier, outlier_reasons 
                FROM alignments 
                WHERE id = ?
            """, (alignment_id,))
            row = cursor.fetchone()
            if not row:
                self.send_error(404, "Alignment not found")
                return
                
            alignment_data = {
                'id': row[0],
                'file_name': row[1],
                'transcript_id': row[2],
                'gene_name': row[3],
                'resolved_gene_name': row[4],
                'alignment_length': row[5],
                'num_sequences': row[6],
                'num_sus_sequences': row[7],
                'biotype': row[8],
                'is_pseudogene': bool(row[9]),
                'is_sus_gene': bool(row[10]),
                'sus_reasons': row[11],
                'is_outlier': bool(row[12]),
                'outlier_reasons': row[13]
            }
            
            # Fetch sequence details (Human reference hg38 goes first, then the rest sorted alphabetically)
            cursor.execute("""
                SELECT sequence_name, resolved_length, gap_count, n_count, 
                       stop_codon_count, has_premature_stop, is_sus, sus_reasons 
                FROM sequences 
                WHERE alignment_id = ? 
                ORDER BY (sequence_name = 'hg38') DESC, sequence_name ASC
            """, (alignment_id,))
            
            sequences = []
            for s_row in cursor.fetchall():
                sequences.append({
                    'sequence_name': s_row[0],
                    'resolved_length': s_row[1],
                    'gap_count': s_row[2],
                    'n_count': s_row[3],
                    'stop_codon_count': s_row[4],
                    'has_premature_stop': bool(s_row[5]),
                    'is_sus': bool(s_row[6]),
                    'sus_reasons': s_row[7]
                })
                
            response = {
                'alignment': alignment_data,
                'sequences': sequences
            }
            self.send_json_response(response)
        except Exception as e:
            self.send_error(500, f"Database query error: {e}")
        finally:
            conn.close()

def find_free_port(start_port=8000, max_port=8050):
    """Finds an available port to bind to."""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except socket.error:
                continue
    raise RuntimeError(f"Could not find any free port in range {start_port}-{max_port}")

def main():
    load_global_stats()
    
    port = find_free_port()
    server_address = ('localhost', port)
    
    try:
        httpd = HTTPServer(server_address, DashboardRequestHandler)
        print("\n" + "="*60)
        print(f"TOGA Alignments Dashboard Server running at: http://localhost:{port}/")
        print("Press Ctrl+C to stop the server.")
        print("="*60 + "\n")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
