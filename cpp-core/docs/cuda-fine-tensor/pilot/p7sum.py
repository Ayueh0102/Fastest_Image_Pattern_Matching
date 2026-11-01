"""Pilot 7 (not formal): fine mode 5 (m) against mode 4 (k6) per threads per block."""
import json, math, statistics as st, sys
S = sys.argv[1]
for n in (256, 320, 384, 448):
    r = json.load(open(f"{S}/n{n}.json"))
    recs = [x for x in r["records"] if x["round"] > 0 and "error" not in x]
    cases = []
    for x in recs:
        if x["case"] not in cases: cases.append(x["case"])
    print(n, "mismatches", len(r["mismatches"]), "failures", len(r["failures"]))
    for mode in ("exhaustive", "fast", "subpixel"):
        med = {(a, c): st.median(x["steady_ms"] for x in recs if x["arm"] == a and x["case"] == c and x["mode"] == mode) for a in ("k6", "m") for c in cases}
        g = math.exp(sum(math.log(med["m", c] / med["k6", c]) for c in cases) / len(cases))
        print(f"  {mode:10s} k6 {sum(med['k6', c] for c in cases):7.2f} m {sum(med['m', c] for c in cases):7.2f} geo {g:.3f} |",
              " ".join(f"{c}:{med['m', c] / med['k6', c]:.3f}" for c in cases))
