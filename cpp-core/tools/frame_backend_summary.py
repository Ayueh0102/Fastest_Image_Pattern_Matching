"""Summarize benchmark_frame_backend.py output and evaluate the CUDA protocol's H1/H2 (stdlib only).

python cpp-core/tools/frame_backend_summary.py timing.json --reference cpu --arm cuda --output summary.json
Rounds after round 0 only. Per arm, mode and case: the median over rounds of the steady per-frame
time (prepare + match), of the cold first call, of the prepare and match parts, of process peak RSS
and of the backend diagnostics. H1: geometric mean of arm/reference below --h1 (0.5) in every mode;
H2: at least --h2 (15) cases faster in every mode.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timing", type=Path)
    parser.add_argument("--reference", default="cpu")
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--h1", type=float, default=0.5)
    parser.add_argument("--h2", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.timing.read_text())
    records = [r for r in report["records"] if r.get("round", 0) > 0 and "error" not in r]
    arms = [args.reference, *args.arm]
    summary = {"source": args.timing.name, "rounds_used": sorted({r["round"] for r in records}),
               "mismatches": len(report["mismatches"]), "failures": len(report["failures"]), "arms": {}}
    for arm in arms:
        for mode in report["modes"]:
            cases = {}
            for case in sorted({r["case"] for r in records}, key=lambda c: [r["case"] for r in records].index(c)):
                runs = [r for r in records if r["arm"] == arm and r["mode"] == mode and r["case"] == case]
                if not runs:
                    continue
                cases[case] = {
                    "rounds": len(runs),
                    "steady_ms": med(r["steady_ms"] for r in runs),
                    "steady_ms_min": min(r["steady_ms"] for r in runs),
                    "steady_ms_max": max(r["steady_ms"] for r in runs),
                    "cold_ms": med(r["first_ms"] for r in runs),
                    "prepare_ms": med(statistics.median(r["prepare_ms"]) for r in runs),
                    "match_ms": med(statistics.median(r["match_ms"]) for r in runs),
                    "peak_rss_mib": med(r["peak_rss_mib"] for r in runs),
                    "coarse_device_ms": med(r["coarse"]["device_ms"] for r in runs),
                    "coarse_syncs": med(r["coarse"]["syncs"] for r in runs),
                    "fine_device_ms": med(r["fine"]["device_ms"] for r in runs),
                    "fine_calls": med(r["fine"]["calls"] for r in runs),
                    "fine_fp64_pixels": med(r["fine"].get("repair_pixels") for r in runs),
                    "fine_roi_pixels": med(r["fine"].get("roi_pixels") for r in runs),
                    "fine_fp64_maps": med(r["fine"].get("repair_all_maps") for r in runs),
                    "gpu_bytes": med(r["coarse"]["gpu_bytes"] + r["fine"]["retained_gpu_bytes"] for r in runs),
                    "matches": runs[0]["count"],
                }
            summary["arms"].setdefault(arm, {})[mode] = {
                "steady_total_ms": sum(c["steady_ms"] for c in cases.values()),
                "cold_total_ms": sum(c["cold_ms"] for c in cases.values()),
                "cases": cases}
    hypotheses = {}
    for arm in args.arm:
        h1 = h2 = True
        per_mode = {}
        for mode in report["modes"]:
            base, other = summary["arms"][args.reference][mode]["cases"], summary["arms"][arm][mode]["cases"]
            common = [c for c in base if c in other]
            ratios = {c: other[c]["steady_ms"] / base[c]["steady_ms"] for c in common}
            geomean = math.exp(sum(math.log(v) for v in ratios.values()) / len(ratios))
            faster = [c for c, v in ratios.items() if v < 1]
            per_mode[mode] = {"geomean": geomean, "total_ratio": sum(other[c]["steady_ms"] for c in common) /
                              sum(base[c]["steady_ms"] for c in common), "faster": len(faster), "cases": len(common),
                              "not_faster": [c for c in common if c not in faster],
                              "cold_geomean": math.exp(sum(math.log(other[c]["cold_ms"] / base[c]["cold_ms"]) for c in common) / len(common)),
                              "ratios": ratios}
            h1 &= geomean < args.h1
            h2 &= len(faster) >= args.h2
        hypotheses[arm] = {"H1": h1, "H2": h2, "modes": per_mode}
    summary["hypotheses"] = {"thresholds": {"H1_geomean_below": args.h1, "H2_cases_faster_at_least": args.h2}, **hypotheses}
    args.output.write_text(json.dumps(summary, indent=1))
    for arm, h in hypotheses.items():
        print(arm, "H1", h["H1"], "H2", h["H2"])
        for mode, m in h["modes"].items():
            print(f"  {mode:10s} geomean {m['geomean']:.4f} total {m['total_ratio']:.4f} faster {m['faster']}/{m['cases']} "
                  f"not faster {m['not_faster']} cold geomean {m['cold_geomean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
