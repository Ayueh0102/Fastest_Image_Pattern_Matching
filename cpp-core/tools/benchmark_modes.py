"""Compare modes on the same Src7 input/configuration, serially.

Run a Release build first. Times exclude image decoding and template learning;
each mode gets one warm-up, then three measured runs by default. A count alone
does not certify equal poses/precision, so every result is retained in JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "build/portable")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp-core/docs/mode-benchmark.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    source, template = [ROOT / "Test Images" / name for name in ("Src7.bmp", "Dst7.bmp")]
    options = ["--src", str(source), "--tmpl", str(template), "--target-num", "70",
               "--score", ".5", "--angle", "180", "--overlap", ".2", "--min-area", "256"]
    modes = [("legacy_cpp", "fpm_legacy_reference", []), ("portable_scalar", "fpm_cli", ["--no-simd"]),
             ("portable_simd", "fpm_cli", []), ("portable_fast", "fpm_cli", ["--fast"]),
             ("portable_subpixel", "fpm_cli", ["--subpixel"])]
    paths = list((ROOT / "cpp-core/src").glob("*")) + list((ROOT / "cpp-core/include/fpm").glob("*"))
    paths += [ROOT / "macos-cli/match_core.cpp", ROOT / "macos-cli/match_core.h", Path(__file__).resolve()]
    report = {"machine": platform.machine(), "platform": platform.platform(),
              "config": dict(source=source.name, template=template.name, target_num=70, score=.5,
                             angle=180, overlap=.2, min_area=256),
              "timing": "match_ms excludes image decoding and learning; one warm-up then measured runs",
              "repeats": args.repeats,
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, template)},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
              "modes": []}
    for name, binary, extra in modes:
        runs = []
        for repeat in range(args.repeats + 1):
            result = json.loads(subprocess.check_output(
                [str(args.build.resolve() / binary), *options, *extra], text=True, timeout=180))
            if repeat:
                runs.append(result["match_ms"])
        entry = dict(mode=name, backend=result.get("backend", "legacy build"), times_ms=runs,
                     median_match_ms=statistics.median(runs), matches=result["matches"])
        report["modes"].append(entry)
        print(name, entry["backend"], round(entry["median_match_ms"], 3), "ms; count", len(entry["matches"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
