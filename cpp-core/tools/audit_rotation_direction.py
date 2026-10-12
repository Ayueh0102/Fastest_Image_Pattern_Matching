"""Compare rotating source patches with rotating the template at six known poses.

This is a local diagnostic, not a detector or a speed benchmark. Both directions
use the same template interior (four border pixels excluded for all interpolation
kernels). A rotated-template mask excludes its empty bounding-box corners.
Discrete pixel supports differ between directions; their NCC scores need not be
identical. Requires NumPy/OpenCV from the existing Python environment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MARGIN = 4


def matrix_at(center, angle, width, height):
    radians = math.radians(angle)
    c, s = math.cos(radians), math.sin(radians)
    rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    return np.column_stack((rotation, np.asarray(center) - rotation @ [width / 2, height / 2]))


class PoseComparison:
    def __init__(self, source, template, center):
        self.source, self.template = source, template
        self.height, self.width = template.shape
        self.mask = np.zeros(template.shape, np.uint8)
        self.mask[MARGIN:-MARGIN, MARGIN:-MARGIN] = 255
        # Covers every orientation and the complete +/-3-pixel translation box.
        radius = int(math.ceil(math.hypot(self.width, self.height) / 2)) + 8
        x, y = int(math.floor(center[0])) - radius, int(math.floor(center[1])) - radius
        size = 2 * radius + 1
        if x < 0 or y < 0 or x + size > source.shape[1] or y + size > source.shape[0]:
            raise ValueError("diagnostic crop would exceed the source image")
        self.origin = np.array([x, y])
        self.crop = source[y:y + size, x:x + size]
        self.crop_float = self.crop.astype(np.float32)
        self.size = (size, size)

    def score(self, center, angle, interpolation, direction):
        if direction == "source":
            matrix = matrix_at(center, angle, self.width, self.height)
            patch = cv2.warpAffine(self.source, matrix, (self.width, self.height),
                                   flags=interpolation | cv2.WARP_INVERSE_MAP)
            a = patch[MARGIN:-MARGIN, MARGIN:-MARGIN]
            b = self.template[MARGIN:-MARGIN, MARGIN:-MARGIN]
            return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
        matrix = matrix_at(np.asarray(center) - self.origin, angle, self.width, self.height)
        patch = cv2.warpAffine(self.template, matrix, self.size, flags=interpolation)
        mask = cv2.warpAffine(self.mask, matrix, self.size, flags=cv2.INTER_NEAREST)
        mean_a, std_a = cv2.meanStdDev(self.crop, mask=mask)
        mean_b, std_b = cv2.meanStdDev(patch, mask=mask)
        cross = cv2.mean(cv2.multiply(self.crop_float, patch.astype(np.float32)), mask=mask)[0]
        return float((cross - mean_a.item() * mean_b.item()) / (std_a.item() * std_b.item()))


def refine(evaluator, seed, interpolation, direction):
    """Coordinate search bounded to +/-3 px and +/-1 degree from the seed.

    Progressively tests joint x/y grids and angle grids, stopping at 0.1 px and
    0.01 degree. This does not establish a continuous or global optimum.
    """
    seed_center = np.array(seed["center"])
    seed_angle = seed["angle"]
    best = dict(center=seed_center.tolist(), angle=seed_angle,
                score=evaluator.score(seed_center, seed_angle, interpolation, direction))
    evaluations = 1
    for step_xy, step_angle in ((1.0, .2), (.5, .05), (.1, .01)):
        for _ in range(3):
            previous = best["score"]
            origin, angle = np.array(best["center"]), best["angle"]
            for dy in (-step_xy, 0, step_xy):
                for dx in (-step_xy, 0, step_xy):
                    center = origin + [dx, dy]
                    if np.max(np.abs(center - seed_center)) > 3 + 1e-9:
                        continue
                    score = evaluator.score(center, angle, interpolation, direction)
                    evaluations += 1
                    if score > best["score"]:
                        best = dict(center=center.tolist(), angle=angle, score=score)
            center, angle = best["center"], best["angle"]
            for delta in range(-5, 6):
                candidate = angle + delta * step_angle
                if abs(candidate - seed_angle) > 1 + 1e-9:
                    continue
                score = evaluator.score(center, candidate, interpolation, direction)
                evaluations += 1
                if score > best["score"]:
                    best = dict(center=list(center), angle=candidate, score=score)
            if best["score"] <= previous + 1e-10:
                break
    best["evaluations"] = evaluations
    return best


def sanity_check():
    # Exact 0/90/180-degree permutations must preserve the asymmetric test
    # pattern and score in BOTH directions; verifies angle/center conventions.
    rng = np.random.default_rng(715)
    template = rng.integers(0, 256, (39, 52), dtype=np.uint8)
    for angle, center in ((0, [70., 69.5]), (90, [70.5, 70.]), (180, [70., 70.5])):
        matrix = matrix_at(center, angle, template.shape[1], template.shape[0])
        source = cv2.warpAffine(template, matrix, (144, 144), flags=cv2.INTER_NEAREST)
        comparison = PoseComparison(source, template, center)
        for direction in ("source", "template"):
            score = comparison.score(center, angle, cv2.INTER_LINEAR, direction)
            if score < .99999:
                raise AssertionError((angle, direction, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "cpp-core/docs/rotation-direction-validation.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sanity_check()
    source_path, template_path = [ROOT / "Test Images" / name for name in ("Src7.bmp", "Dst7.bmp")]
    seed_path = ROOT / "cpp-core/docs/rotation-search-validation.json"
    source, template = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in (source_path, template_path)]
    if source is None or template is None:
        raise RuntimeError("missing fixture")
    seeds = json.loads(seed_path.read_text())["independent_search"]
    paths = [source_path, template_path, seed_path, Path(__file__).resolve()]
    report = {"opencv": cv2.__version__, "excluded_template_border_pixels": MARGIN,
              "method": "masked NCC; template mask rotated with nearest neighbor; no padded corners included",
              "search": "local coordinate search +/-3 source pixels and +/-1 degree; final steps 0.1 px / 0.01 degree",
              "limitations": "different discrete pixel supports; local search, not global optimum; not a performance comparison",
              "sanity_checks": "exact asymmetric template at 0, 90, 180 degrees scores >0.99999 in both directions",
              "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "locations": []}
    for location in seeds:
        started = time.perf_counter()
        seed = location["best"]
        comparison = PoseComparison(source, template, seed["center"])
        entry = dict(index=location["index"], seed=seed, variants=[])
        for name, interpolation in (("linear", cv2.INTER_LINEAR), ("cubic", cv2.INTER_CUBIC), ("lanczos4", cv2.INTER_LANCZOS4)):
            for direction in ("source", "template"):
                fixed = comparison.score(seed["center"], seed["angle"], interpolation, direction)
                best = refine(comparison, seed, interpolation, direction)
                entry["variants"].append(dict(interpolation=name, rotated=direction, fixed_pose_score=fixed, best=best))
                print(location["index"], name, direction, "fixed", round(fixed, 6), "refined", round(best["score"], 6), flush=True)
        entry["seconds"] = time.perf_counter() - started
        report["locations"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
