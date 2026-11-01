"""Independent recheck of the identity gate: exact comparison of the `matches` lists of every record of the exact arms
(f32 and old32 are inexact and left out), no use of the benchmark driver's own comparison. python identity_check.py results_dir output.json"""
import json, sys, collections
D, OUT = sys.argv[1], sys.argv[2]
INEXACT = {"f32", "old32"}
files = {n: json.load(open(f"{D}/{n}.json")) for n in ("timing-standard", "identity-standard", "timing-shallow", "identity-shallow")}
out = {"files": {}, "cross": {}}
ref = {"standard": {}, "shallow": {}}
for name, data in files.items():
    group = collections.defaultdict(list)
    errors = [r for r in data["records"] if "error" in r]
    exact = [r for r in data["records"] if "error" not in r and r["arm"] not in INEXACT]
    for r in exact:
        group[(r["round"], r["case"], r["mode"])].append(r)
    diffs = []
    for key, runs in group.items():
        first = json.dumps(runs[0]["matches"], sort_keys=True)
        for r in runs[1:]:
            if json.dumps(r["matches"], sort_keys=True) != first:
                diffs.append([*key, runs[0]["arm"], r["arm"]])
    across = []
    by_case = collections.defaultdict(set)
    for r in exact:
        by_case[(r["case"], r["mode"])].add(json.dumps(r["matches"], sort_keys=True))
    across = [list(k) for k, v in by_case.items() if len(v) != 1]
    set_name = "shallow" if "shallow" in name else "standard"
    for key, values in by_case.items():
        ref[set_name].setdefault(key, set()).update(values)
    out["files"][name] = {"records": len(data["records"]), "errors": len(errors), "groups": len(group),
                          "arms": sorted({r["arm"] for r in data["records"]}), "differences_within_group": diffs,
                          "keys_with_more_than_one_result_over_all_rounds": across,
                          "driver_mismatches": len(data["mismatches"]), "driver_failures": len(data["failures"]),
                          "exact_records": len(exact), "matches_total": sum(r["count"] for r in exact)}
out["cross"] = {s: {"keys": len(v), "keys_with_more_than_one_result_across_files": [list(k) for k, x in v.items() if len(x) != 1]}
                for s, v in ref.items()}
cpu = [r for r in files["identity-shallow"]["records"] if r["arm"] == "cpu" and r["round"] == 0]
both = collections.defaultdict(set)
for r in files["timing-standard"]["records"]:
    if r["arm"] in INEXACT: both[(r["case"], r["mode"], r["arm"])].add(json.dumps(r["matches"], sort_keys=True))
out["f32_equals_old32"] = {"cases": len(both) // 2, "different": [list(k[:2]) for k in both if k[2] == "f32" and both[k] != both[(k[0], k[1], "old32")]],
                           "stable_over_rounds": all(len(v) == 1 for v in both.values())}
out["shallow_cpu_ms"] = {f"{r['case']}/{r['mode']}": r["first_ms"] for r in cpu}
out["pass"] = all(not f["differences_within_group"] and not f["keys_with_more_than_one_result_over_all_rounds"] and f["errors"] == 0
                  and f["driver_mismatches"] == 0 and f["driver_failures"] == 0 for f in out["files"].values()) \
    and all(not c["keys_with_more_than_one_result_across_files"] for c in out["cross"].values())
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: {a: b for a, b in v.items() if a != "arms"} for k, v in out["files"].items()}, indent=0))
print(out["cross"], "pass", out["pass"])
