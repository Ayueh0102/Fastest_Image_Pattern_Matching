"""Per-case tables of the report from timing.json (rounds 1-7, medians): steady ms per arm, c3 kernel metrics,
prepare/match/coarse wait/fine wall of m3, t4 and u7, peak RSS and device bytes.

python derived_tables.py timing.json
"""
import json, sys, statistics as st
r = json.load(open(sys.argv[1]))
recs = [x for x in r["records"] if x["round"] >= 1 and "error" not in x]
arms = ["cpu", "m3", "c2", "c3", "f2", "best", "u7", "t4", "fp32"]
modes = ["exhaustive", "fast", "subpixel"]
cases = []
for x in r["records"]:
    if x["case"] not in cases: cases.append(x["case"])
def med(arm, mode, case, f):
    v = [f(x) for x in recs if x["arm"] == arm and x["mode"] == mode and x["case"] == case]
    return st.median(v)
out = {}
for mode in modes:
    print(f"== {mode} steady ms (median rounds 1-7)")
    print("case".ljust(10) + "".join(a.rjust(9) for a in arms) + "  best/m3")
    for c in cases:
        row = {a: med(a, mode, c, lambda x: x["steady_ms"]) for a in arms}
        out.setdefault(mode, {})[c] = row
        print(c.ljust(10) + "".join(f"{row[a]:9.3f}" for a in arms) + f"  {row['best']/row['m3']:.3f}")
print("== c3 kernel metrics (round 1, exhaustive/fast/subpixel): windows, fp64 scored, fraction, fp64 tile px")
for c in cases:
    s = []
    for mode in modes:
        x = [x for x in recs if x["arm"] == "c3" and x["mode"] == mode and x["case"] == c][0]["kernel"]
        s.append(f"{x['coarse_windows']:>9} {x['coarse_fp64_scores']:>8} {x['coarse_fp64_scores']/max(1,x['coarse_windows']):.3f} {x['coarse_fp64_pixels']:>6}")
    print(c.ljust(10) + " | ".join(s))
tot = {}
for mode in modes:
    w = sum([x for x in recs if x["arm"] == "c3" and x["mode"] == mode and x["case"] == c][0]["kernel"]["coarse_windows"] for c in cases)
    f = sum([x for x in recs if x["arm"] == "c3" and x["mode"] == mode and x["case"] == c][0]["kernel"]["coarse_fp64_scores"] for c in cases)
    p = sum([x for x in recs if x["arm"] == "c3" and x["mode"] == mode and x["case"] == c][0]["kernel"]["coarse_fp64_pixels"] for c in cases)
    print(mode, "windows", w, "fp64 scored", f, f"{f/w:.4f}", "fp64 tile px", p)
# variability of kernel metrics across rounds (should be constant)
var = set()
for a in ["c2", "c3", "best"]:
    for mode in modes:
        for c in cases:
            ks = {json.dumps(x["kernel"], sort_keys=True) for x in recs if x["arm"] == a and x["mode"] == mode and x["case"] == c}
            if len(ks) != 1: var.add((a, mode, c))
print("kernel metrics vary across rounds:", sorted(var))
print("== prepare / match / coarse wait (ms), m3 vs t4 vs u7, exhaustive")
for c in cases:
    def g(a, f): return med(a, "exhaustive", c, f)
    pm = lambda x: st.median(x["prepare_ms"]); mm = lambda x: st.median(x["match_ms"])
    cw = lambda x: x["coarse"]["wait_ms"]; fw = lambda x: x["fine"]["wall_ms"]
    print(c.ljust(10) + " ".join(f"{a}:{g(a,pm):6.3f}/{g(a,mm):6.3f}/{g(a,cw):6.3f}/{g(a,fw):6.3f}" for a in ["m3", "t4", "u7"]))
print("== peak RSS MiB and gpu_bytes (max over cases), retained fine bytes")
for a in arms:
    rss = max(x["peak_rss_mib"] for x in recs if x["arm"] == a)
    gb = max(x["coarse"]["gpu_bytes"] for x in recs if x["arm"] == a)
    rb = max(x["fine"]["retained_gpu_bytes"] for x in recs if x["arm"] == a)
    rssm = st.median([x["peak_rss_mib"] for x in recs if x["arm"] == a])
    print(a.ljust(6), f"rss max {rss:8.2f} median {rssm:8.2f} gpu_bytes max {gb} retained {rb}")
print("== cold geomean vs m3 is in summary; transfer copy threads")
for a in ["m3", "t4", "best"]:
    print(a, {x["kernel"]["copy_threads"] for x in recs if x["arm"] == a}, {x["transfer"]["mode"] for x in recs if x["arm"] == a})
