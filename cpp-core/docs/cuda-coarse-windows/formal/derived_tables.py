"""Per-case tables of the report (rounds 1-7, medians): steady ms per arm in both sets, kernel metrics of modes 3, 5, 6,
coarse wait and fine wall, peak RSS and device bytes, cold first calls.

python derived_tables.py results_dir
"""
import json, sys, statistics as st
D = sys.argv[1]
def load(n):
    r = json.load(open(f"{D}/{n}.json"))
    return r, [x for x in r["records"] if x["round"] >= 1 and "error" not in x]
def table(name, arms, modes):
    r, recs = load(name)
    cases = []
    for x in r["records"]:
        if x["case"] not in cases: cases.append(x["case"])
    def med(arm, mode, case, f):
        return st.median(f(x) for x in recs if x["arm"] == arm and x["mode"] == mode and x["case"] == case)
    for mode in modes:
        print(f"== {name} {mode} steady ms (median rounds 1-7)")
        print("case".ljust(10) + "".join(a.rjust(10) for a in arms))
        for c in cases:
            print(c.ljust(10) + "".join(f"{med(a, mode, c, lambda x: x['steady_ms']):10.3f}" for a in arms))
        print("total".ljust(10) + "".join(f"{sum(med(a, mode, c, lambda x: x['steady_ms']) for c in cases):10.3f}" for a in arms))
    for mode in modes:
        print(f"== {name} {mode} cold first call ms (median rounds 1-7)")
        print("case".ljust(10) + "".join(a.rjust(10) for a in arms))
        for c in cases:
            print(c.ljust(10) + "".join(f"{med(a, mode, c, lambda x: x['first_ms']):10.3f}" for a in arms))
    print(f"== {name} prepare / match / coarse wait / fine wall ms, exhaustive")
    pm = lambda x: st.median(x["prepare_ms"]); mm = lambda x: st.median(x["match_ms"])
    cw = lambda x: x["coarse"]["wait_ms"]; fw = lambda x: x["fine"]["wall_ms"]
    gpu = [a for a in arms if a != "cpu"]
    for c in cases:
        print(c.ljust(10) + " ".join(f"{a}:{med(a, modes[0], c, pm):7.3f}/{med(a, modes[0], c, mm):7.3f}/{med(a, modes[0], c, cw):7.3f}/{med(a, modes[0], c, fw):7.3f}" for a in gpu if a in ("b2", "k5", "k6")))
    print(f"== {name} kernel metrics (round 1): windows, fp64 scored, fraction, fp64 tile px")
    for a in gpu:
        for mode in modes:
            k = [x["kernel"] for x in recs if x["arm"] == a and x["mode"] == mode and x["round"] == 1]
            w = sum(x.get("coarse_windows", 0) for x in k); f = sum(x.get("coarse_fp64_scores", 0) for x in k); p = sum(x.get("coarse_fp64_pixels", 0) for x in k)
            print(a.ljust(4), mode.ljust(11), "windows", w, "fp64 scored", f, f"{f / max(1, w):.4f}", "fp64 tile px", p)
    var = set()
    for a in gpu:
        for mode in modes:
            for c in cases:
                ks = {json.dumps(x["kernel"], sort_keys=True) for x in recs if x["arm"] == a and x["mode"] == mode and x["case"] == c}
                if len(ks) != 1: var.add((a, mode, c))
    print("kernel metrics vary across rounds:", sorted(var))
    print(f"== {name} peak RSS MiB and device bytes (over cases)")
    for a in arms:
        s = [x for x in recs if x["arm"] == a]
        print(a.ljust(4), f"rss max {max(x['peak_rss_mib'] for x in s):8.2f} median {st.median(x['peak_rss_mib'] for x in s):8.2f}",
              "coarse gpu_bytes max", max(x["coarse"]["gpu_bytes"] for x in s), "fine retained max", max(x["fine"]["retained_gpu_bytes"] for x in s))
    print(f"== {name} steady spread: largest (max-min)/median over rounds, per arm")
    for a in arms:
        worst = (0, None)
        for mode in modes:
            for c in cases:
                v = [x["steady_ms"] for x in recs if x["arm"] == a and x["mode"] == mode and x["case"] == c]
                worst = max(worst, ((max(v) - min(v)) / st.median(v), f"{c}/{mode}"), key=lambda t: t[0])
        print(a.ljust(4), f"{worst[0]:.3f}", worst[1])
table("timing-standard", ["cpu", "b2", "c5", "c6", "f4", "k5", "k6"], ["exhaustive", "fast", "subpixel"])
table("timing-shallow", ["b2", "c5", "k5", "k6"], ["exhaustive", "subpixel"])
