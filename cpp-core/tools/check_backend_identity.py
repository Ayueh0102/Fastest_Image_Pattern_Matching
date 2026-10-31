"""Check that core builds and backends return bit-identical matches.

python-port/venv/bin/python cpp-core/tools/check_backend_identity.py \
    --lib before=build/a/libfpm_c.dylib --lib after=build/b/libfpm_c.dylib --output out.json
Each C ABI library runs the 17 compare_versions.py fixture cases in four modes
(exhaustive, pruned, FastMode, subpixel) with its SIMD and scalar backends.
Images are decoded like fpm_cli (OpenCV color, then grayFromRgb), or read from
the PGM copies of export_fixture_pgm.py with --pgm-dir, which needs only the
standard library (fpm_c.dll on Windows). Every double
is kept as a binary64 hex string and compared with the first library's scalar
output; any difference, or a failed call, gives a nonzero exit status.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpp-core/examples"))
sys.path.insert(0, str(ROOT / "cpp-core/tools"))
from fixture_cases import CASES, pgm_path, read_pgm
from python_ctypes import Image, Library

MODES = {"exhaustive": {}, "pruned": {"prune": True}, "fast": {"stop_layer": 1}, "subpixel": {"subpixel": 1}}


IMAGES: dict = {}


def gray(path: Path):
    import cv2
    import numpy as np
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise RuntimeError(f"missing fixture: {path}")
    b, g, r = (bgr[..., i].astype(np.uint32) for i in range(3))
    return np.ascontiguousarray(((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8))


def load(name: str, pgm_dir: Path | None) -> Image:
    """A gray8 view of a fixture; the pixel buffer is kept alive in IMAGES."""
    if name not in IMAGES:
        if pgm_dir:
            width, height, pixels = read_pgm(pgm_path(pgm_dir, name))
            buffer = (C.c_uint8 * len(pixels)).from_buffer_copy(pixels)
            IMAGES[name] = buffer, Image(C.cast(buffer, C.POINTER(C.c_uint8)), width, height, width)
        else:
            pixels = gray(ROOT / "Test Images" / name)
            IMAGES[name] = pixels, Image(pixels.ctypes.data_as(C.POINTER(C.c_uint8)), pixels.shape[1],
                                         pixels.shape[0], pixels.strides[0])
    return IMAGES[name][1]


def run(library: Library, case, mode: str, scalar: bool, pgm_dir: Path | None = None) -> list:
    src_name, tmpl_name, count, score, angle, overlap, area = case
    source, templ = load(src_name, pgm_dir), load(tmpl_name, pgm_dir)
    config = library.config()
    config.max_targets, config.min_score, config.tolerance_angle = count, score, angle
    config.max_overlap, config.scalar = overlap, int(scalar)
    options = dict(MODES[mode])
    prune = options.pop("prune", False)
    for key, value in options.items():
        setattr(config, key, value)
    with library.learn(templ, area) as pattern:
        matches = pattern.match(source, config, prune=prune)
    return [[float(m[k]).hex() for k in ("score", "angle", "pos_x", "pos_y")] +
            [float(v).hex() for corner in m["corners"] for v in corner] for m in matches]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", action="append", required=True, help="NAME=path/to/libfpm_c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pgm-dir", type=Path, help="read export_fixture_pgm.py copies instead of decoding with OpenCV")
    args = parser.parse_args()
    libraries = []
    for item in args.lib:
        name, sep, path = item.partition("=")
        if not sep or not name or any(name == n for n, _, _ in libraries):
            parser.error(f"--lib needs a unique NAME=path: {item}")
        path = Path(path).resolve()
        libraries.append((name, path, Library(str(path))))
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "libraries": {n: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for n, p, _ in libraries},
              "inputs": f"PGM copies in {args.pgm_dir}" if args.pgm_dir else "Test Images via OpenCV",
              "reference": f"{libraries[0][0]}/scalar", "modes": list(MODES), "cases": []}
    differences = 0
    for case in CASES:
        for mode in MODES:
            outputs = {f"{n}/{b}": run(lib, case, mode, b == "scalar", args.pgm_dir)
                       for n, _, lib in libraries for b in ("simd", "scalar")}
            reference = outputs[report["reference"]]
            entry = {"source": case[0], "template": case[1], "mode": mode, "count": len(reference),
                     "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in outputs.items()},
                     "different": sorted(k for k, v in outputs.items() if v != reference),
                     "reference_matches": reference}
            differences += len(entry["different"])
            report["cases"].append(entry)
            print(case[0], mode, "count", len(reference), "different:", entry["different"] or "none", flush=True)
    report["runs"] = len(report["cases"]) * 2 * len(libraries)
    report["different_runs"] = differences
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved:", args.output, "runs:", report["runs"], "different:", differences)
    return int(differences > 0)


if __name__ == "__main__":
    sys.exit(main())
