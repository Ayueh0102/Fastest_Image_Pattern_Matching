"""Tables of the Nsight runs in profile/ (not formal timings): per-frame kernel time, then Nsight Compute key metrics.
python profile_tables.py profile_dir"""
import csv, glob, os, re, sys
P = sys.argv[1]
for path in sorted(glob.glob(f"{P}/**/*.cuda_gpu_kern_sum.csv", recursive=True)):
    name = os.path.basename(path).split(".")[0]
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l for l in f if not l.startswith(("Generating", "Processing"))]
    for r in csv.DictReader(lines):
        n = r["Name"]
        short = re.sub(r"^void ", "", n).replace("<unnamed>::", "").replace("(int)", "").split("(const")[0].split("(float")[0].split("(unsigned")[0].strip()
        rows.append((short, int(r["Instances"]), float(r["Avg (ns)"]) / 1e3, float(r["Total Time (ns)"]) / 1e6))
    if not rows:
        print(f"== {name}: no data (see the disclosure)")
        continue
    frames = next(i for s, i, a, t in rows if s.startswith("packEntries"))
    total = sum(t for *_, t in rows)
    print(f"== {name}: {frames} frames, kernels per frame {total / frames:.3f} ms")
    for s, i, a, t in rows:
        print(f"   {s:32s} {i / frames:7.2f} x {a:10.1f} us = {t / frames:9.3f} ms")
    for rep, key in (("cuda_gpu_mem_time_sum", "Operation"),):
        with open(path.replace("cuda_gpu_kern_sum", rep), encoding="utf-8", errors="replace") as f:
            lines = [l for l in f if not l.startswith(("Generating", "Processing"))]
        for r in csv.DictReader(lines):
            print(f"   mem {r[key]:30s} {int(r['Count']) / frames:7.2f} x {float(r['Avg (ns)']) / 1e3:8.1f} us")
def grab(text, label):
    m = re.search(rf"^\s*{re.escape(label)}\s+(?:\S+\s+)?([\d,\.]+)\s*$", text, re.M)
    return m.group(1) if m else "?"
print("== Nsight Compute")
for path in sorted(glob.glob(f"{P}/ncu-*.txt")):
    if path.endswith(".stdout.txt"): continue
    t = open(path, encoding="utf-8", errors="replace").read()
    print(os.path.basename(path)[4:-4])
    print("  ", {k: grab(t, k) for k in ("Duration", "Elapsed Cycles", "Compute (SM) Throughput", "L1/TEX Cache Throughput", "Memory Throughput",
          "Executed Ipc Active", "Issue Slots Busy", "Executed Instructions", "Achieved Occupancy", "Theoretical Occupancy", "Dynamic Shared Memory Per Block",
          "Threads", "Static Shared Memory Per Block", "Grid Size", "Avg. Divergent Branches")})
