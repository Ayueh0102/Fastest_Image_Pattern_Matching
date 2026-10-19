"""Reproduce Src7's 1/3/6 detections and measure rotated-box overlap.

Run with python-port/venv/bin/python; OpenCV is used only for independent polygon
intersection measurements. Both matchers run as compiled C++ executables.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "build/portable")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp-core/docs/rotor-validation.json")
    args = parser.parse_args()
    report = {"source": "Src7.bmp", "template": "Dst7.bmp", "target_num": 70,
              "min_reduced_area": 256, "subpixel": False,
              "reference": "Local macOS CMatchCore; Windows MFC executable not run",
              "input_sha256": {}, "source_sha256": {}, "cases": []}
    for name in (report["source"], report["template"]):
        report["input_sha256"][name] = hashlib.sha256((ROOT / "Test Images" / name).read_bytes()).hexdigest()
    paths = list((ROOT / "cpp-core/src").glob("*.cpp")) + list((ROOT / "cpp-core/include/fpm").glob("*"))
    paths += [ROOT / "macos-cli/match_core.cpp", ROOT / "macos-cli/match_core.h", Path(__file__).resolve()]
    for path in sorted(paths):
        report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = [(8, 0, .5, 1), (180, 0, .5, 3), (180, .05, .5, 3),
             (180, .1, .5, 6), (180, .1, .8, 3)]
    six = None
    for angle, overlap, score, expected in cases:
        case = dict(angle=angle, overlap=overlap, score=score, expected_count=expected)
        options = ["--src", str(ROOT / "Test Images" / report["source"]),
                   "--tmpl", str(ROOT / "Test Images" / report["template"]),
                   "--target-num", "70", "--angle", str(angle), "--overlap", str(overlap),
                   "--score", str(score), "--min-area", "256"]
        for name in ("fpm_cli", "fpm_legacy_reference"):
            run = subprocess.run([str(args.build.resolve() / name), *options],
                                 capture_output=True, text=True, check=True, timeout=120)
            case[name] = json.loads(run.stdout)["matches"]
            if len(case[name]) != expected:
                raise RuntimeError(f"{name}: {case} has unexpected count")
        if expected == 6:
            six = case["fpm_cli"]
        report["cases"].append(case)
        print(f"angle={angle}, overlap={overlap}, score={score}: both C++ versions found {expected}")
    pairs = []
    for i in range(3, 6):
        for j in range(3):
            a, b = (np.array(six[k]["corners"], dtype=np.float32) for k in (i, j))
            intersection, _ = cv2.intersectConvexConvex(a, b)
            ratio = float(intersection) / (762 * 521)
            if ratio > 0:
                pairs.append(dict(lower_score_index=i, higher_score_index=j, overlap_ratio=ratio))
    report["cross_group_overlaps"] = pairs
    if len(pairs) != 6 or not all(.08 < p["overlap_ratio"] <= .1 for p in pairs):
        raise RuntimeError("Unexpected rotor overlap geometry")
    report["explanation"] = "Six poses pass score=0.5. Each lower-score pose overlaps higher-score poses; maxOverlap=0 or 0.05 rejects it. At score=0.8 only the three high-score poses qualify."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
