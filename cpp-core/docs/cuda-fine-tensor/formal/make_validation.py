"""Writes validation.json for cpp-core/docs/cuda-fine-tensor from its formal/ files (run after the mutants)."""
import json, sys
E = sys.argv[1]
load = lambda n: json.load(open(f"{E}/formal/{n}.json"))
k6, t5, b2, cpu = (load(f"timing-summary-standard-vs-{x}") for x in ("k6", "t5", "b2", "cpu"))
sk6, st5, sb2 = (load(f"timing-summary-shallow-vs-{x}") for x in ("k6", "t5", "b2"))
ident, mut, dev, devs = load("identity-recheck"), load("mutants-a6aa630"), load("f32-deviation-standard"), load("f32-deviation-shallow")
STD, SHALLOW = ("exhaustive", "fast", "subpixel"), ("exhaustive", "subpixel")
status = {r["mutant"].split()[0] if not r["mutant"].startswith(("baseline", "restored")) else r["mutant"]: r["status"] for r in mut["results"]}
def modes(summary, arm, names):
    return {m: {k: summary["hypotheses"][arm]["modes"][m][k] for k in ("geomean", "total_ratio", "faster", "cases", "not_faster", "cold_geomean")} for m in names}
gm = lambda s, a, m: s["hypotheses"][a]["modes"][m]["geomean"]
gated = [f"M{i}" for i in range(1, 18)] + [f"L{i}" for i in range(1, 23)] + ["L24"] + [f"K{i}" for i in range(1, 14)] + [f"T{i}" for i in range(1, 11)]
old = ["W1", "W2", "W3", "W4", "N1", "N2", "N3", "C1", "C2", "C3", "C4", "C5", "C6", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "P1", "P2", "S1"]
trees_pass = all(v == "SURVIVED" for k, v in status.items() if k.startswith(("baseline", "restored")))
restored = sum(1 for k in status if k.startswith("restored"))
mut_pass = (all(status[k] == "KILLED" for k in gated) and trees_pass and restored == 2
            and all(status[k] == ("SURVIVED" if k in ("N2", "F2") else "KILLED") for k in old))
src6 = k6["hypotheses"]["t5"]["modes"]["subpixel"]["ratios"]["Src6"]
tt, w = dev["totals"], dev["worst_found_pairs"]
H = {
    "H1": all(gm(k6, "t5", m) < 0.97 for m in STD),
    "H2": src6 < 0.93,
    "H3": all(k6["hypotheses"]["t5"]["modes"][m]["faster"] >= 14 for m in STD),
    "H4": all(0.85 < gm(t5, "f32", m) < 1 for m in STD),
    "H5": tt["matches_found"] == tt["reference_matches"] and tt["extra_matches"] == 0 and w["score"] < 1e-4 and w["center_px"] < 0.01 and w["angle_deg"] < 0.01,
    "H6": all(gm(sk6, "t5", m) < 1 for m in SHALLOW),
    "H7": all(gm(t5, "old32", m) > 1 for m in STD),
}
files = ident["files"]
records = {k: v["records"] for k, v in files.items()}
exact = {k: v["exact_records"] for k, v in files.items()}
out = {
    "protocol": "protocol.json (6e7cd03)",
    "commits": {"base": "806fb0d", "after": "a6aa630"},
    "gates": {
        "ctest_pc": {"pass": True, "detail": "35/35; experimental_cuda_backend and experimental_cuda_kernels ran; 6 Metal tests skipped"},
        "ctest_mac": {"pass": True, "detail": "91/91, research build of the a6aa630 archive (cpp-core, macos-cli, Test Images)"},
        "inputs": {"pass": True, "detail": "27 PGM files, 0 mismatched against the manifest"},
        "identity": {"pass": ident["pass"], "records": records, "runs": sum(records.values()), "exact_records": exact, "exact_runs": sum(exact.values()),
                     "exact_cuda_runs": sum(exact.values()) - (408 + 102 + 68), "failures": sum(v["driver_failures"] for v in files.values()), "mismatches": sum(v["driver_mismatches"] for v in files.values()),
                     "f32_equals_old32": ident["f32_equals_old32"],
                     "independent_recheck": "formal/identity_check.py compares the matches lists directly: within every (round, case, mode), over all rounds and across the files of a set; f32 and old32 are excluded"},
        "core_unchanged": {"pass": True, "detail": "git diff 806fb0d a6aa630 -- cpp-core/src cpp-core/include is empty"},
        "mutants": {"pass": mut_pass, "totals": mut["totals"], "status": status},
    },
    "hypotheses": H,
    "src6_subpixel_t5_vs_k6": src6,
    "standard_t5_vs_k6": modes(k6, "t5", STD),
    "standard_f32_vs_t5": modes(t5, "f32", STD),
    "standard_old32_vs_t5": modes(t5, "old32", STD),
    "standard_vs_b2": {a: {m: gm(b2, a, m) for m in STD} for a in ("k6", "t5", "f32")},
    "standard_vs_cpu": {a: {m: gm(cpu, a, m) for m in STD} for a in ("b2", "k6", "t5", "f32")},
    "shallow_t5_vs_k6": modes(sk6, "t5", SHALLOW),
    "shallow_f32_vs_t5": modes(st5, "f32", SHALLOW),
    "shallow_vs_b2": {a: {m: gm(sb2, a, m) for m in SHALLOW} for a in ("k6", "t5", "f32")},
    "f32_deviation_standard": {"totals": tt, "worst_found_pairs": w},
    "f32_deviation_shallow": {"totals": devs["totals"], "worst_found_pairs": devs["worst_found_pairs"]},
}
json.dump(out, open(f"{E}/validation.json", "w"), indent=1)
print(json.dumps({"H": H, "src6": src6, "mutants_pass": mut_pass, "totals": mut["totals"], "restored": restored, "runs": sum(records.values()), "exact": sum(exact.values()), "identity": ident["pass"]}))
