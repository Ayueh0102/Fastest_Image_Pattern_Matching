"""Writes validation.json for cpp-core/docs/cuda-coarse-windows from its formal/ files (run after the mutants)."""
import json, sys
E = sys.argv[1]
load = lambda n: json.load(open(f"{E}/formal/{n}.json"))
b2, cpu, k65 = load("timing-summary-standard-vs-b2"), load("timing-summary-standard-vs-cpu"), load("timing-summary-standard-k6-vs-k5")
sb2, sk65 = load("timing-summary-shallow-vs-b2"), load("timing-summary-shallow-k6-vs-k5")
ident, mut = load("identity-recheck"), load("mutants-1d0e951")
STD, SHALLOW = ("exhaustive", "fast", "subpixel"), ("exhaustive", "subpixel")
status = {r["mutant"].split()[0] if not r["mutant"].startswith(("baseline", "restored")) else r["mutant"]: r["status"] for r in mut["results"]}
def modes(summary, arm, names):
    return {m: {k: summary["hypotheses"][arm]["modes"][m][k] for k in ("geomean", "total_ratio", "faster", "cases", "not_faster", "cold_geomean")} for m in names}
gm = lambda s, a, m: s["hypotheses"][a]["modes"][m]["geomean"]
gated = [f"L{i}" for i in range(1, 23)] + ["L24"] + [f"K{i}" for i in range(1, 14)] + [f"T{i}" for i in range(1, 11)]
old = ["W1", "W2", "W3", "W4", "N1", "N2", "N3", "C1", "C2", "C3", "C4", "C5", "C6", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "P1", "P2", "S1"]
trees_pass = all(v == "SURVIVED" for k, v in status.items() if k.startswith(("baseline", "restored")))
restored = sum(1 for k in status if k.startswith("restored"))
mut_pass = (all(status[k] == "KILLED" for k in gated) and trees_pass and restored == 2
            and all(status[k] == ("SURVIVED" if k in ("N2", "F2") else "KILLED") for k in old))
src6 = b2["hypotheses"]["f4"]["modes"]["subpixel"]["ratios"]["Src6"]
H = {
    "H1": all(gm(b2, "k6", m) < 0.98 for m in STD),
    "H2": src6 < 0.90,
    "H3": gm(sb2, "k6", "exhaustive") < 0.5,
    "H4": gm(sk65, "k6", "exhaustive") < 1,
    "H5": all(abs(gm(k65, "k6", m) - 1) < 0.03 for m in STD),
}
files = ident["files"]
records = {k: v["records"] for k, v in files.items()}
cpu_runs = 408 + 102 + 68
out = {
    "protocol": "protocol.json (ac0a538, registered 2026-11-01T10:34:34Z)",
    "commits": {"base": "5f2c6f7", "after": "1d0e951"},
    "gates": {
        "ctest_pc": {"pass": True, "detail": "35/35; experimental_cuda_backend and experimental_cuda_kernels ran; 6 Metal tests skipped"},
        "ctest_mac": {"pass": True, "detail": "91/91, research build of the 1d0e951 archive (cpp-core, macos-cli, Test Images)"},
        "inputs": {"pass": True, "detail": "27 PGM files, 0 mismatched against the manifest"},
        "identity": {"pass": ident["pass"], "records": records, "runs": sum(records.values()), "cuda_runs": sum(records.values()) - cpu_runs,
                     "mismatches": sum(v["driver_mismatches"] for v in files.values()), "failures": sum(v["driver_failures"] for v in files.values()),
                     "independent_recheck": "formal/identity_check.py compares the matches lists directly: within every (round, case, mode), over all rounds and across the files of a set"},
        "core_unchanged": {"pass": True, "detail": "git diff 5f2c6f7 1d0e951 -- cpp-core/src cpp-core/include is empty"},
        "mutants": {"pass": mut_pass, "totals": mut["totals"], "status": status},
    },
    "hypotheses": H,
    "src6_subpixel_f4_vs_b2": src6,
    "standard_vs_b2": {a: modes(b2, a, STD) for a in ("k6", "k5", "c5", "c6", "f4")},
    "standard_k6_vs_k5": modes(k65, "k6", STD),
    "standard_vs_cpu": {a: {m: gm(cpu, a, m) for m in STD} for a in ("b2", "k5", "k6")},
    "shallow_vs_b2": {a: modes(sb2, a, SHALLOW) for a in ("k6", "k5", "c5")},
    "shallow_k6_vs_k5": modes(sk65, "k6", SHALLOW),
}
json.dump(out, open(f"{E}/validation.json", "w"), indent=1)
print(json.dumps({"H": H, "src6": src6, "mutants_pass": mut_pass, "totals": mut["totals"], "restored": restored, "runs": sum(records.values()), "identity": ident["pass"]}, indent=1))
