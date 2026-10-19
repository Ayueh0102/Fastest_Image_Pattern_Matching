"""Independent full-resolution search around all six Src7 locations.

Unlike the production matcher, this reference uses no pyramid, candidate
truncation, intermediate score threshold, or overlap suppression. It evaluates
all angles on a half-degree grid, then a 0.05-degree grid around the best angle.
Translation is exhaustively searched on each rotated pixel grid within +/-96
pixels of each supplied center. This is a bounded diagnostic, not a proof of the
continuous global optimum. Requires the existing Python OpenCV environment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sample_window(source, center, angle, width, height, radius):
    # Independent destination-to-source affine transform, using output CCW
    # angles and the public width/2, height/2 geometric-center convention.
    radians = math.radians(angle)
    c, s = math.cos(radians), math.sin(radians)
    rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    origin = np.asarray(center) - rotation @ np.array([width / 2 + radius, height / 2 + radius])
    matrix = np.column_stack((rotation, origin))
    window = cv2.warpAffine(source, matrix, (width + 2 * radius, height + 2 * radius),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return window, rotation


def independent_score(source, template, center, angle, radius):
    h, w = template.shape
    window, rotation = sample_window(source, center, angle, w, h, radius)
    response = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, pos = cv2.minMaxLoc(response)
    actual_center = np.asarray(center) + rotation @ (np.asarray(pos) - radius)
    return {"score": score, "angle": (float(angle) + 180) % 360 - 180,
            "center": actual_center.tolist(), "offset": list(pos)}


def render_alignment(source, template, report, output):
    """Inspection contact sheet; each patch is unrotated at its measured pose."""
    h, w = template.shape
    canvas = np.full((680, 1600, 3), 245, np.uint8)
    for row, indices in enumerate(((None, 0, 1, 2), (None, 3, 4, 5))):
        for col, index in enumerate(indices):
            if index is None:
                patch, label = template, "Original template"
            else:
                pose = report["independent_search"][index]["best"]
                patch, _ = sample_window(source, pose["center"], pose["angle"], w, h, 0)
                label = f"#{index} {pose['angle']:.2f} deg / {pose['score']:.4f}"
            tile = cv2.cvtColor(cv2.resize(patch, (381, 261)), cv2.COLOR_GRAY2BGR)
            y, x = row * 340, col * 400
            cv2.putText(canvas, label, (x + 9, y + 33), cv2.FONT_HERSHEY_SIMPLEX,
                        .58, (30, 30, 30), 1, cv2.LINE_AA)
            canvas[y + 55:y + 316, x + 9:x + 390] = tile
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Cannot save {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "build/portable")
    parser.add_argument("--output", type=Path, default=ROOT / "cpp-core/docs/rotation-search-validation.json")
    parser.add_argument("--radius", type=int, default=96)
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("radius must be positive")
    cv2.setNumThreads(1)
    source_path, template_path = [ROOT / "Test Images" / name for name in ("Src7.bmp", "Dst7.bmp")]
    source, template = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (source_path, template_path)]
    if source is None or template is None:
        raise RuntimeError("missing fixture")
    config = ["--src", str(source_path), "--tmpl", str(template_path), "--angle", "180",
              "--target-num", "70", "--score", ".5", "--overlap", ".2"]
    report = {"source": source_path.name, "template": template_path.name,
              "reference": "OpenCV warpAffine + TM_CCOEFF_NORMED at original resolution; no pyramid/NMS",
              "opencv": cv2.__version__, "translation_radius": args.radius,
              "angle_step_degrees": .5, "fine_angle_step_degrees": .05,
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_path, template_path)},
              "source_sha256": {}, "production_variants": [], "independent_search": []}
    paths = list((ROOT / "cpp-core/src").glob("*.cpp")) + list((ROOT / "cpp-core/include/fpm").glob("*"))
    paths += [Path(__file__).resolve()]
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    for area, subpixel in ((256, False), (1024, False), (4096, False), (256, True)):
        options = [str(args.build.resolve() / "fpm_cli"), *config, "--min-area", str(area)]
        if subpixel:
            options.append("--subpixel")
        result = json.loads(subprocess.check_output(options, text=True, timeout=120))
        report["production_variants"].append(dict(min_area=area, subpixel=subpixel, result=result))
        print("production", area, subpixel, [round(m["score"], 6) for m in result["matches"]], flush=True)
    matches = report["production_variants"][0]["result"]["matches"]
    if len(matches) != 6:
        raise RuntimeError("expected six seed locations")
    for index, match in enumerate(matches):
        started = time.perf_counter()
        center = [match["pos_x"], match["pos_y"]]
        coarse = [independent_score(source, template, center, angle * .5, args.radius) for angle in range(-360, 360)]
        best = max(coarse, key=lambda item: item["score"])
        fine = [independent_score(source, template, center, best["angle"] + delta * .05, args.radius)
                for delta in range(-20, 21)]
        best = max([best, *fine], key=lambda item: item["score"])
        # Recenter the translation sampling lattice for a second fine pass;
        # this can improve fractional translations without a pyramid or a fit.
        fine2 = [independent_score(source, template, best["center"], best["angle"] + delta * .01, 3)
                 for delta in range(-5, 6)]
        best = max([best, *fine2], key=lambda item: item["score"])
        entry = {"index": index, "seed": match, "best": best,
                 "coarse_angle_maxima": [dict(angle=m["angle"], score=m["score"]) for m in coarse],
                 "seconds": time.perf_counter() - started}
        report["independent_search"].append(entry)
        # Save after each location so a long diagnostic is resumable by inspection.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("independent", index, best, "seconds", round(entry["seconds"], 2), flush=True)
    print("Saved", args.output)
    render_alignment(source, template, report, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
