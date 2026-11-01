"""Geometric means of the development pilots (not formal timings): python pilot_summary.py raw_dir_of_pilot-raw.tar.gz"""
import json, math, statistics as st, sys
S = sys.argv[1]
def tab(path, ref, arms, modes):
    r = json.load(open(path))
    recs = [x for x in r["records"] if x["round"] > 0 and "error" not in x] or r["records"]
    cases = []
    for x in recs:
        if x["case"] not in cases: cases.append(x["case"])
    out = [f"{path.split('/')[-1]} mismatches {len(r['mismatches'])} failures {len(r['failures'])} cases {len(cases)} rounds {sorted({x['round'] for x in r['records']})}"]
    for mode in modes:
        have = [x for x in recs if x["mode"] == mode]
        if not have: continue
        med = {(a, c): st.median(x["steady_ms"] for x in have if x["arm"] == a and x["case"] == c) for a in [ref, *arms] for c in cases}
        s = []
        for a in arms:
            g = math.exp(sum(math.log(med[a, c] / med[ref, c]) for c in cases) / len(cases))
            s.append(f"{a} {g:.3f}/{sum(med[a, c] for c in cases):.2f}ms")
        out.append(f"  {mode:10s} {ref} {sum(med[ref, c] for c in cases):.2f}ms | " + " | ".join(s))
    return "\n".join(out)
def arms_of(path):
    return sorted({x["arm"] for x in json.load(open(path))["records"]})
print("== warps per block"); 
for w in (8, 4, 2, 1):
    p = f"{S}/warps-w{w}.json"; a = arms_of(p); print(a); print(tab(p, "best" if "best" in a else a[0], [x for x in a if x not in ("best", "cpu")], ("exhaustive", "fast", "subpixel")))
print("== fine column tiles")
for t in (0, 256, 128, 64):
    p = f"{S}/tiles-t{t}.json"; a = arms_of(p); print(tab(p, "best", [x for x in a if x != "best"], ("exhaustive", "subpixel")))
print("== pilot3 (row-level screen fixed, warps)")
for w in (2, 4, 8):
    p = f"{S}/pilot3-w{w}.json"; a = arms_of(p); print(tab(p, "best" if "best" in a else a[0], [x for x in a if x != "best"], ("exhaustive", "fast", "subpixel")))
print("== fine ROI width threshold")
for w in (0, 64, 128, 256, 512):
    p = f"{S}/width-w{w}.json"; a = arms_of(p); print(tab(p, "best", [x for x in a if x != "best"], ("subpixel",)))
    r = json.load(open(p)); recs = [x for x in r["records"] if x["round"] > 0]
    print("   ", {c: round(st.median(x["steady_ms"] for x in recs if x["arm"] == "f4" and x["case"] == c) / st.median(x["steady_ms"] for x in recs if x["arm"] == "best" and x["case"] == c), 3) for c in ("Src1", "Src2", "Src4", "Src6", "Src7", "Src8")})
