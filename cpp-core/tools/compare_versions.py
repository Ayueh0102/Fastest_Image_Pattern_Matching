"""Reproduce the version audit; requires the existing Python port dependencies.

python-port/venv/bin/python cpp-core/tools/compare_versions.py --build build/portable
All inputs are repository fixtures. Missing input files and subprocess failures
are errors, never silently skipped. The original MFC binary is NOT run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python-port/src"))
import cv2
import numpy as np
from fpm_py.pyramid import pyramid_match, _find_topk_suppressed
from fpm_py.nms import nms, nms_rotated

CASES = [
    ("Src1.bmp", "Dst1.bmp", 5, .8, 180, .0, 256),
    ("Src2.bmp", "Dst2.bmp", 5, .8, 180, .0, 256),
    ("Src3.bmp", "Dst3.bmp", 38, .8, 0, .0, 256),
    ("Src4.bmp", "Dst4.bmp", 5, .8, 180, .0, 256),
    ("Src6.jpg", "Dst6.bmp", 15, .8, 180, .0, 256),
    ("Src7.bmp", "Dst7.bmp", 5, .8, 180, .0, 256),
    ("Src8.bmp", "Dst8.bmp", 5, .8, 180, .8, 256),
    ("Src9.bmp", "Dst9.bmp", 100, .5, 0, .5, 1024),
    ("Src10.bmp", "Dst10.jpg", 100, .8, 0, .0, 256),
] + [(f"Src5-{angle}.bmp", "Dst5.bmp", 1, .8, 180, .0, 256)
     for angle in (0, 45, 90, 135, 180, 225, 270, 315)]


def correspondence(reference, actual):
    available = set(range(len(actual)))
    errors = []
    for old in reference:
        if not available:
            break
        def distance(i):
            r = actual[i]
            return math.hypot(r["pos_x"] - old["pos_x"], r["pos_y"] - old["pos_y"])
        i = min(available, key=distance)
        new = actual[i]
        errors.append((distance(i), abs((new["angle"] - old["angle"] + 180) % 360 - 180),
                       abs(new["score"] - old["score"])))
        available.remove(i)
    maxima = [max((e[j] for e in errors), default=0) for j in range(3)]
    return dict(reference_count=len(reference), actual_count=len(actual),
                max_position_error_px=maxima[0], max_angle_error_deg=maxima[1],
                max_score_error=maxima[2],
                within_tolerance=len(reference) == len(actual) and maxima[0] <= 2 and maxima[1] <= 1 and maxima[2] <= .03)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "cpp-core/docs/validation.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    report = {"audit_date": "2026-10-08", "historical_root": "432442a", "first_cpp": "f521243",
              "local_baseline": "3546282", "upstream_baseline": "d8edabb",
              "platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "opencv_python": cv2.__version__,
              "repeats_after_one_warmup": args.repeats,
              "tolerances": {"position_px": 2, "angle_degrees": 1, "score": .03},
              "note": "Runtime reference is macOS CMatchCore, not the Windows MFC executable. Timings exclude image loading; Python includes learning.",
              "cases": []}
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT / "cpp-core/src", ROOT / "cpp-core/include", ROOT / "macos-cli")
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix in (".cpp", ".h", ".hpp") and "build" not in p.parts
    }
    for src_name, tmpl_name, count, score, angle, overlap, area in CASES:
        src_path, tmpl_path = ROOT / "Test Images" / src_name, ROOT / "Test Images" / tmpl_name
        src, tmpl = cv2.imread(str(src_path), 0), cv2.imread(str(tmpl_path), 0)
        if src is None or tmpl is None:
            raise RuntimeError(f"missing fixture: {src_path}, {tmpl_path}")
        options = ["--src", str(src_path), "--tmpl", str(tmpl_path), "--target-num", str(count),
                   "--score", str(score), "--angle", str(angle), "--overlap", str(overlap), "--min-area", str(area)]
        entry = {"source": src_name, "template": tmpl_name,
                 "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (src_path, tmpl_path)},
                 "config": dict(target_num=count, score=score, tolerance_angle=angle, max_overlap=overlap,
                                min_reduced_area=area, subpixel=False)}
        for name, binary in (("legacy_cpp", "fpm_legacy_reference"), ("portable_cpp", "fpm_cli")):
            runs = []
            for repeat in range(args.repeats + 1):
                output = subprocess.run([str(args.build.resolve() / binary), *options], capture_output=True, text=True, check=True, timeout=120)
                result = json.loads(output.stdout)
                if repeat:
                    runs.append(result["match_ms"])
            entry[name] = {"median_match_ms": statistics.median(runs), "matches": result["matches"]}
        runs = []
        for repeat in range(args.repeats + 1):
            start = time.perf_counter()
            results = pyramid_match(src, tmpl, **entry["config"])
            if repeat:
                runs.append((time.perf_counter() - start) * 1000)
        entry["python"] = {"median_learn_and_match_ms": statistics.median(runs), "matches": results}
        entry["portable_vs_legacy"] = correspondence(entry["legacy_cpp"]["matches"], entry["portable_cpp"]["matches"])
        entry["python_vs_legacy"] = correspondence(entry["legacy_cpp"]["matches"], results)
        report["cases"].append(entry)
        print(src_name, "counts (legacy/new/python):", *(len(entry[k]["matches"]) for k in ("legacy_cpp", "portable_cpp", "python")), entry["portable_vs_legacy"], flush=True)
    response = np.zeros((30, 100), np.float32)
    response[:20, :20] = .99
    response[25, 90] = .90
    report["python_counterexamples"] = {
        "broad_peak_expected_count": 2,
        "broad_peak_actual": _find_topk_suppressed(response, 2, 25, 25, .8),
        "axis_nms": nms([(0, 0, 10, 10), (4, 0, 10, 10)], [.99, .95], .5),
        "rotated_nms": nms_rotated([((5, 5), (10, 10), 0), ((9, 5), (10, 10), 0)], [.99, .95], .5),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    failures = [c["source"] for c in report["cases"] if not c["portable_vs_legacy"]["within_tolerance"]]
    print("Saved:", args.output, "cases outside comparison tolerance:", failures)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
