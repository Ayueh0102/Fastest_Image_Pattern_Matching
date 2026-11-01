"""Interleaved frame-API timing and identity across backends (stdlib only).

python cpp-core/tools/benchmark_frame_backend.py --bench build/x/experiments/fpm_frame_bench \
    --pgm-dir fixtures-pgm --arm cpu=0,0,0,0 --arm cuda=2,1,256,0 --rounds 8 --repeat 5 --output out.json

Each arm is BACKEND,PYRAMID,FINE_BATCH,NCC_MODE[,TRANSFER[,FINE_MODE[,THREADS]]] for fpm_frame_bench. Every (round, case, mode,
arm) is a fresh process; the arm order alternates between rounds. Per process, the first call is
reported as the cold time; after --warmup-ms of untimed calls, the median of --repeat timed calls
is the steady per-frame time (prepare + match; image decoding and frame/pattern creation are
excluded). Round 0 is a warm-up and is kept in the raw records but left out of the summary. Matches are compared bit for bit with the
first arm in every run; any difference or failed run gives a nonzero exit status, except that differences of an arm named by
--inexact (an approximate ablation) are recorded separately as inexact_differences. On Windows,
start this driver with `start /affinity MASK` so every child inherits the CPU set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_cases import CASES, pgm_path

MODES = ("exhaustive", "fast", "subpixel")


def run_arm(bench: Path, arm: tuple[int, ...], mode: str, repeat: int, warmup_ms: float, case,
            pgm_dir: Path) -> dict:
    source, templ, targets, score, tolerance, overlap, area = case
    cmd = [str(bench), *map(str, arm[:4]), mode, str(repeat), repr(warmup_ms), str(pgm_path(pgm_dir, source)),
           str(pgm_path(pgm_dir, templ)),
           str(targets), repr(score), repr(float(tolerance)), repr(overlap), str(area), *map(str, arm[4:])]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if done.returncode:
        return {"error": done.stderr.strip() or f"exit {done.returncode}", "cmd": cmd}
    record = json.loads(done.stdout)
    record["first_ms"] = record["first_prepare_ms"] + record["first_match_ms"]
    record["steady_ms"] = statistics.median(p + m for p, m in zip(record["prepare_ms"], record["match_ms"]))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--pgm-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=BACKEND,PYRAMID,FINE_BATCH,NCC_MODE[,TRANSFER[,FINE_MODE[,THREADS]]]")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=500)
    parser.add_argument("--cases", default="", help="comma-separated source stems; default all 17")
    parser.add_argument("--min-area-scale", type=int, default=1, help="multiply each case's minimum top-level area (shallower pyramids)")
    parser.add_argument("--inexact", action="append", default=[], help="arm whose outputs may differ (ablation)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for item in args.arm:
        name, values = item.split("=", 1)
        arms[name] = tuple(int(v) for v in values.split(","))
    names = list(arms)
    modes = args.modes.split(",")
    cases = [(*c[:6], c[6] * args.min_area_scale) for c in CASES if not args.cases or Path(c[0]).stem in args.cases.split(",")]
    records, mismatches, failures, inexact = [], [], [], []
    started = time.time()
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else names[::-1]
        for case in cases:
            for mode in modes:
                by_arm = {}
                for name in order:
                    record = run_arm(args.bench, arms[name], mode, args.repeat, args.warmup_ms, case, args.pgm_dir)
                    record.update(round=round_index, case=Path(case[0]).stem, mode=mode, arm=name)
                    records.append(record)
                    if "error" in record:
                        failures.append(record)
                    else:
                        by_arm[name] = record
                reference = by_arm.get(names[0])
                for name, record in by_arm.items():
                    if reference is not None and record["matches"] != reference["matches"]:
                        entry = {"round": round_index, "case": record["case"], "mode": mode, "arm": name}
                        (inexact if name in args.inexact else mismatches).append(entry)
        print(f"round {round_index} done, {time.time() - started:.0f}s, mismatches {len(mismatches)}, failures {len(failures)}",
              flush=True)
    summary = {}
    for name in names:
        for mode in modes:
            per_case = {}
            for case in cases:
                stem = Path(case[0]).stem
                runs = [r for r in records if r.get("arm") == name and r.get("mode") == mode and r.get("case") == stem
                        and r.get("round", 0) > 0 and "error" not in r]
                if runs:
                    per_case[stem] = {"steady_ms": statistics.median(r["steady_ms"] for r in runs),
                                      "first_ms": statistics.median(r["first_ms"] for r in runs),
                                      "runs": len(runs)}
            summary.setdefault(name, {})[mode] = {
                "cases": per_case,
                "steady_total_ms": sum(v["steady_ms"] for v in per_case.values()),
                "first_total_ms": sum(v["first_ms"] for v in per_case.values()),
            }
    ratios = {}
    for name in names[1:]:
        for mode in modes:
            base, other = summary[names[0]][mode]["cases"], summary[name][mode]["cases"]
            common = [c for c in base if c in other]
            if common:
                logs = [math.log(other[c]["steady_ms"] / base[c]["steady_ms"]) for c in common]
                ratios[f"{name}/{names[0]}/{mode}"] = {
                    "steady_geomean": math.exp(sum(logs) / len(logs)),
                    "steady_total": sum(other[c]["steady_ms"] for c in common) / sum(base[c]["steady_ms"] for c in common),
                    "cases_faster": sum(1 for v in logs if v < 0), "cases": len(common)}
    report = {
        "tool": "benchmark_frame_backend.py", "arms": arms, "modes": modes, "rounds": args.rounds,
        "repeat": args.repeat, "warmup_ms": args.warmup_ms, "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "bench": str(args.bench), "summary": summary, "ratios": ratios,
        "mismatches": mismatches, "inexact_arms": args.inexact, "inexact_differences": inexact, "failures": failures,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({"ratios": ratios, "mismatches": len(mismatches), "failures": len(failures)}, indent=1))
    return 1 if mismatches or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
