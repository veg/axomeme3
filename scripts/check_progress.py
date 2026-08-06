#!/usr/bin/env python3
"""
check_progress.py
-----------------
Checks the execution progress of both the aBSREL SLURM job array on silverback + magilla
and the sequential MEME loop on m2.local + m3.local. Uses Base64-encoded single-SSH calls
to avoid connection rate-limiting and resets.
"""

import subprocess
import json
import os
import base64
from datetime import datetime, timedelta

TOTAL_GENES = 18275

def run_ssh_python(host, script_content):
    # Base64 encode the script to prevent any quoting/escaping issues over SSH
    encoded = base64.b64encode(script_content.encode('utf-8')).decode('utf-8')
    ssh_command = f"python3 -c \"import base64; exec(base64.b64decode('{encoded}').decode('utf-8'))\""
    
    cmd = ["ssh", host, ssh_command]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command on {host}: {e.stderr}")
        return ""

def format_duration(seconds):
    if seconds <= 0:
        return "N/A"
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 and not parts:
        parts.append(f"{secs}s")
    return " ".join(parts) if parts else "0s"

def check_absrel():
    # 0. Sync results from magilla to local, then local to silverback
    os.makedirs("/tmp/absrel_magilla", exist_ok=True)
    subprocess.run(["rsync", "-avz", "magilla:/home/sergei/Projects/TOGA/absrel/", "/tmp/absrel_magilla/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["rsync", "-avz", "/tmp/absrel_magilla/", "silverback:/home/sergei/Projects/TOGA/absrel/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Script to run on silverback
    sb_script = """
import os, glob, sqlite3, subprocess, json

# Run compile first
subprocess.run(["python3", "/home/sergei/Projects/TOGA/compile_absrel_results.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

out_dir = "/home/sergei/Projects/TOGA/absrel"
completed_cnt = len([f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) > 0])
running_cnt = len([f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) == 0])

# Get active SLURM tasks
res = subprocess.run(["squeue", "-u", "sergei", "-h", "-t", "running"], stdout=subprocess.PIPE, text=True)
squeue_cnt = len([line for line in res.stdout.split("\\n") if line.strip()])

db_path = "/home/sergei/Projects/TOGA/absrel_results.db"
db_count, avg_runtime = 0, 0.0
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("select count(*), avg(runtime_sec) from gene_results")
        db_count, avg_runtime = c.fetchone()
        avg_runtime = avg_runtime or 0.0
        conn.close()
    except Exception:
        pass

files = [f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) > 0]
times = sorted([os.path.getmtime(f) for f in files])

print(json.dumps({
    "completed": completed_cnt,
    "running": running_cnt,
    "squeue": squeue_cnt,
    "db_count": db_count,
    "avg_runtime": avg_runtime,
    "times": times
}))
"""

    # Script to run on magilla
    mg_script = """
import os, glob, subprocess, json
out_dir = "/home/sergei/Projects/TOGA/absrel"
running_cnt = len([f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) == 0])
res = subprocess.run(["squeue", "-u", "sergei", "-h", "-t", "running"], stdout=subprocess.PIPE, text=True)
squeue_cnt = len([line for line in res.stdout.split("\\n") if line.strip()])
print(json.dumps({
    "running": running_cnt,
    "squeue": squeue_cnt
}))
"""

    # Execute on silverback
    sb_res_raw = run_ssh_python("silverback", sb_script)
    sb_data = json.loads(sb_res_raw) if sb_res_raw else {"completed": 0, "running": 0, "squeue": 0, "db_count": 0, "avg_runtime": 0.0, "times": []}
    
    # Execute on magilla
    mg_res_raw = run_ssh_python("magilla", mg_script)
    mg_data = json.loads(mg_res_raw) if mg_res_raw else {"running": 0, "squeue": 0}
    
    # Combine results
    completed_cnt = sb_data["completed"]
    running_cnt = sb_data["running"]  # sb_data["running"] already includes the sync'd magilla running files
    total_active_tasks = sb_data["squeue"] + mg_data["squeue"]
    
    # Calculate rates from silverback times
    times = sb_data["times"]
    rates = {"overall": 0.0, "stabilized": 0.0}
    if len(times) >= 2:
        span_sec = times[-1] - times[0]
        rates['overall'] = len(times) / (span_sec / 3600.0) if span_sec > 0 else 0.0
        
        now = max(times)
        last_10h = [t for t in times if now - t <= 10 * 3600]
        if len(last_10h) >= 2:
            span_10 = last_10h[-1] - last_10h[0]
            rates['stabilized'] = len(last_10h) / (span_10 / 3600.0) if span_10 > 0 else 0.0
        else:
            rates['stabilized'] = rates['overall']
            
    return {
        'completed': completed_cnt,
        'running': running_cnt,
        'active_tasks': total_active_tasks,
        'db_count': sb_data["db_count"],
        'avg_runtime': sb_data["avg_runtime"],
        'rate_overall': rates['overall'],
        'rate_stabilized': rates['stabilized']
    }

def check_meme():
    # Script to run on m2.local (which syncs from m3, compiles, and counts)
    m2_script = """
import os, glob, sqlite3, subprocess, json

# Sync first
subprocess.run(["rsync", "-avz", "m3.local:/Users/sergei/Projects/TOGA_MEME/meme_results/", "/Users/sergei/Projects/TOGA_MEME/meme_results/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["python3", "/Users/sergei/Projects/TOGA_MEME/compile_meme_results.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

out_dir = "/Users/sergei/Projects/TOGA_MEME/meme_results"
completed_cnt = len([f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) > 0])
running_cnt = len([f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) == 0])

db_path = "/Users/sergei/Projects/TOGA_MEME/meme_results.db"
db_count, avg_runtime = 0, 0.0
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("select count(*), avg(runtime_sec) from gene_results")
        db_count, avg_runtime = c.fetchone()
        avg_runtime = avg_runtime or 0.0
        conn.close()
    except Exception:
        pass

files = [f for f in glob.glob(out_dir + "/*.json.gz") if os.path.getsize(f) > 0]
times = sorted([os.path.getmtime(f) for f in files])

print(json.dumps({
    "completed": completed_cnt,
    "running": running_cnt,
    "db_count": db_count,
    "avg_runtime": avg_runtime,
    "times": times
}))
"""

    m2_res_raw = run_ssh_python("m2.local", m2_script)
    m2_data = json.loads(m2_res_raw) if m2_res_raw else {"completed": 0, "running": 0, "db_count": 0, "avg_runtime": 0.0, "times": []}
    
    # Calculate rates
    times = m2_data["times"]
    rates = {"overall": 0.0, "stabilized": 0.0}
    if len(times) >= 2:
        span_sec = times[-1] - times[0]
        rates['overall'] = len(times) / (span_sec / 3600.0) if span_sec > 0 else 0.0
        
        now = max(times)
        last_10h = [t for t in times if now - t <= 10 * 3600]
        if len(last_10h) >= 2:
            span_10 = last_10h[-1] - last_10h[0]
            rates['stabilized'] = len(last_10h) / (span_10 / 3600.0) if span_10 > 0 else 0.0
        else:
            rates['stabilized'] = rates['overall']
            
    return {
        'completed': m2_data["completed"],
        'running': m2_data["running"],
        'db_count': m2_data["db_count"],
        'avg_runtime': m2_data["avg_runtime"],
        'rate_overall': rates['overall'],
        'rate_stabilized': rates['stabilized']
    }

def print_report():
    print("Collecting progress metrics from hosts...")
    absrel = check_absrel()
    meme = check_meme()
    
    print("\n" + "="*80)
    print("                         JOB PROGRESS & RUNTIME PROJECTIONS")
    print("="*80)
    print(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Genes in Dataset: {TOTAL_GENES}\n")
    
    for name, data, is_parallel, concurrency, concurrency_limit in [
        ("aBSREL (on silverback + magilla)", absrel, True, "375 tasks total (250 on sb at 4 cores, 125 on magilla at 8 cores)", 375.0),
        ("MEME (on m2.local + m3.local)", meme, False, "2 sequential tasks (24 cores each)", 2.0)
    ]:
        completed = data['completed']
        running = data['running']
        remaining = TOTAL_GENES - completed
        
        pct_completed = (completed / TOTAL_GENES) * 100
        pct_remaining = (remaining / TOTAL_GENES) * 100
        
        avg_rt = data['avg_runtime']
        rate_overall = data['rate_overall']
        rate_stabilized = data['rate_stabilized']
        
        # Projections
        # Overall
        if rate_overall > 0:
            proj_overall_sec = (remaining / rate_overall) * 3600
            eta_overall = datetime.now() + timedelta(seconds=proj_overall_sec)
            eta_overall_str = eta_overall.strftime('%Y-%m-%d %H:%M')
        else:
            proj_overall_sec = 0
            eta_overall_str = "N/A"
            
        # Stabilized
        if rate_stabilized > 0:
            proj_stabilized_sec = (remaining / rate_stabilized) * 3600
            eta_stabilized = datetime.now() + timedelta(seconds=proj_stabilized_sec)
            eta_stabilized_str = eta_stabilized.strftime('%Y-%m-%d %H:%M')
        else:
            proj_stabilized_sec = 0
            eta_stabilized_str = "N/A"
            
        # Theoretical based on average runtime & concurrency
        if avg_rt > 0:
            theo_rate_sec = concurrency_limit / avg_rt
            theo_rate_hour = theo_rate_sec * 3600
            proj_theo_sec = remaining / theo_rate_sec
            eta_theo = datetime.now() + timedelta(seconds=proj_theo_sec)
            eta_theo_str = eta_theo.strftime('%Y-%m-%d %H:%M')
        else:
            theo_rate_hour = 0.0
            proj_theo_sec = 0
            eta_theo_str = "N/A"
            
        print(f"### {name}")
        print(f"  Configuration:            {concurrency}")
        print(f"  Completed:                {completed}/{TOTAL_GENES} ({pct_completed:.2f}%)")
        print(f"  Currently Running:        {running}")
        print(f"  Remaining to Complete:    {remaining}/{TOTAL_GENES} ({pct_remaining:.2f}%)")
        print(f"  Average Gene Runtime:     {format_duration(avg_rt)} ({avg_rt:.1f} sec)")
        print(f"  Throughput Rates:")
        print(f"    - Overall (from start): {rate_overall:.2f} genes/hour")
        print(f"    - Stabilized (last 10h):{rate_stabilized:.2f} genes/hour")
        print(f"    - Theoretical (limit):  {theo_rate_hour:.2f} genes/hour")
        print(f"  Projected Remaining Wall-Clock Time:")
        print(f"    - Based on Overall Rate:   {format_duration(proj_overall_sec)} (ETA: {eta_overall_str})")
        print(f"    - Based on Stabilized Rate:{format_duration(proj_stabilized_sec)} (ETA: {eta_stabilized_str})")
        print(f"    - Based on Theoretical Rate:{format_duration(proj_theo_sec)} (ETA: {eta_theo_str})")
        print("-" * 80)

if __name__ == "__main__":
    print_report()
