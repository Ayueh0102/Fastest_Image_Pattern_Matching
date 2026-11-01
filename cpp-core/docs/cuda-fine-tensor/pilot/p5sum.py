"""Pilot 5 (not formal): m (coarse 6, fine 5) against k6 (coarse 6, fine 4) per ROI width threshold and shared memory split."""
import json, math, statistics as st, sys
S = sys.argv[1]
def report(name):
    r = json.load(open(f"{S}/{name}.json"))
    recs = [x for x in r["records"] if x["round"] > 0 and "error" not in x]
    cases = []
    for x in recs:
        if x["case"] not in cases: cases.append(x["case"])
    print(name, "mismatches", len(r["mismatches"]), "failures", len(r["failures"]))
    for mode in ("exhaustive", "subpixel"):
        med = {(a, c): st.median(x["steady_ms"] for x in recs if x["arm"] == a and x["case"] == c and x["mode"] == mode) for a in ("k6", "m") for c in cases}
        g = math.exp(sum(math.log(med["m", c] / med["k6", c]) for c in cases) / len(cases))
        print(f"  {mode:10s} k6 {sum(med['k6', c] for c in cases):7.2f} m {sum(med['m', c] for c in cases):7.2f} m/k6 geo {g:.3f} |",
              " ".join(f"{c}:{med['m', c] / med['k6', c]:.3f}" for c in cases))
    return r
for n in ("w0", "w64", "w128", "w256", "w512", "d3-m16", "d3-m8", "d4-m16", "d4-m8"):
    report(n)
