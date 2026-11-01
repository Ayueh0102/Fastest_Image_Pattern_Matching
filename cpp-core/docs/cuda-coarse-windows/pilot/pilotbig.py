"""Pilot: coarse NCC modes on larger top-level templates (smaller pyramid depth). Not formal."""
import json, statistics as st, sys
sys.path.insert(0, r"C:\r-cuda\gpu4\wt\cpp-core\tools")
from pathlib import Path
from benchmark_frame_backend import run_arm
bench = r"C:\r-cuda\gpu4\build\experiments\fpm_frame_bench.exe"
pgm = Path(r"C:\r51\bundle\pgm")
cases = [("Src1.bmp", "Dst1.bmp", 5, .8, 180, .0, 4096), ("Src5-0.bmp", "Dst5.bmp", 1, .8, 180, .0, 1024),
         ("Src5-0.bmp", "Dst5.bmp", 1, .8, 180, .0, 6000), ("Src7.bmp", "Dst7.bmp", 5, .8, 180, .0, 4096),
         ("Src6.jpg", "Dst6.bmp", 15, .8, 180, .0, 10000), ("Src3.bmp", "Dst3.bmp", 38, .8, 0, .0, 4096)]
arms = {"c3f4": (2, 1, 256, 3, 3, 4), "c5f4": (2, 1, 256, 5, 3, 4), "c6f4": (2, 1, 256, 6, 3, 4)}
out = []
for case in cases:
    runs = {}
    for name, arm in arms.items():
        r = [run_arm(bench, arm, "exhaustive", 5, 300.0, case, pgm) for _ in range(3)]
        for x in r:
            if "error" in x: print("ERROR", case, name, x["error"])
        runs[name] = r
    same = all(x["matches"] == runs["c3f4"][0]["matches"] for rr in runs.values() for x in rr)
    row = {n: st.median(x["steady_ms"] for x in rr) for n, rr in runs.items()}
    k = runs["c6f4"][0]["kernel"]
    print(case[0], "min_area", case[6], "same", same, " ".join(f"{n} {v:.3f}" for n, v in row.items()),
          f"c5/c3 {row['c5f4']/row['c3f4']:.3f} c6/c3 {row['c6f4']/row['c3f4']:.3f} windows {k['coarse_windows']}", flush=True)
    out.append({"case": case, "steady": row, "same": same, "kernel": k})
json.dump(out, open(r"C:\r-cuda\gpu4\pilotbig.json", "w"), indent=1)
