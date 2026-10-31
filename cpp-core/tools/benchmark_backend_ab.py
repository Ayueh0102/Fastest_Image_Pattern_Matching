"""Interleaved A/B timing of two fpm_cli builds on the compare_versions fixtures.

python-port/venv/bin/python cpp-core/tools/benchmark_backend_ab.py \
    --build before=build/a --build after=build/b --output ab.json
Each case runs both builds as fresh processes for --rounds rounds. The order
alternates (A,B then B,A) and round 0 is a discarded warm-up. match_ms is
fpm_cli's own clock: it excludes image decoding and template learning. Both
builds must print the same matches in every run, otherwise this is an error.
All raw times, load averages (where the OS has them) and binary hashes are kept
in the JSON. --pgm-dir passes export_fixture_pgm.py copies instead of `Test Images`,
for pure-core builds without OpenCV codecs; it needs only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpp-core/tools"))
from fixture_cases import CASES, pgm_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="append", required=True, help="NAME=build directory; give two")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8, help="including one warm-up round")
    parser.add_argument("--pgm-dir", type=Path, help="fixture PGMs from export_fixture_pgm.py")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra fpm_cli options")
    args = parser.parse_args()
    builds = [item.partition("=") for item in args.build]
    if len(builds) != 2 or any(not sep or not name for name, sep, _ in builds) or builds[0][0] == builds[1][0]:
        parser.error("give exactly two --build NAME=dir with distinct names")
    if args.rounds < 2:
        parser.error("rounds must include a warm-up and at least one measured round")
    binaries = {name: next((p.resolve() for p in (Path(path) / "fpm_cli", Path(path) / "fpm_cli.exe") if p.exists()),
                           Path(path) / "fpm_cli") for name, _, path in builds}
    load_average = getattr(os, "getloadavg", lambda: None)
    names = list(binaries)
    report = {"platform": platform.platform(), "machine": platform.machine(), "rounds": args.rounds,
              "warmup_rounds": 1, "extra_options": args.extra,
              "inputs": f"PGM copies in {args.pgm_dir}" if args.pgm_dir else "Test Images",
              "timing": "fpm_cli match_ms; excludes decoding and learning; fresh process per run",
              "binaries": {n: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for n, p in binaries.items()},
              "load_average_start": load_average(), "cases": []}
    for src, tmpl, count, score, angle, overlap, area in CASES:
        images = [pgm_path(args.pgm_dir, n) if args.pgm_dir else ROOT / "Test Images" / n for n in (src, tmpl)]
        options = ["--src", str(images[0]), "--tmpl", str(images[1]),
                   "--target-num", str(count), "--score", str(score), "--angle", str(angle),
                   "--overlap", str(overlap), "--min-area", str(area), *args.extra]
        times = {n: [] for n in names}
        backends, matches = {}, {}
        for round_index in range(args.rounds):
            matches = {}
            for name in (names if round_index % 2 == 0 else names[::-1]):
                result = json.loads(subprocess.check_output([str(binaries[name]), *options], text=True, timeout=180))
                backends[name] = result["backend"]
                matches[name] = result["matches"]
                if round_index:
                    times[name].append(result["match_ms"])
            if matches[names[0]] != matches[names[1]]:
                raise RuntimeError(f"{src}: builds returned different matches in round {round_index}")
        medians = {n: statistics.median(times[n]) for n in names}
        entry = {"source": src, "template": tmpl, "count": len(matches[names[0]]), "backend": backends,
                 "times_ms": times, "median_ms": medians,
                 "median_ratio": medians[names[1]] / medians[names[0]]}
        report["cases"].append(entry)
        print(src, {n: round(m, 3) for n, m in medians.items()}, "ratio", round(entry["median_ratio"], 3), flush=True)
    report["load_average_end"] = load_average()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
