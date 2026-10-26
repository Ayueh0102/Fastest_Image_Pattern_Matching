"""Round 50: OpenCV matchTemplate (TM_CCOEFF_NORMED) with brute-force rotation and scale.

The traditional baseline of the literature comparison, run with python-port/venv.
For every scale and every angle on a fixed grid the template is rotated about its
centre onto a canvas that holds the whole rotated rectangle, and a mask of the
rotated rectangle restricts the normalised correlation to template pixels.
Peaks are local maxima of each score map at or above --score; all peaks across
maps are then suppressed like the fpm NMS (rotated-rectangle intersection divided
by the smaller rectangle area, compared to --overlap) and the best --limit are kept.

This is a reference implementation, not an optimised one; its timings answer
"how expensive is the textbook approach", nothing more. Output follows the
literature_score record format.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import time

import cv2
import numpy as np


def rotated_template(templ, angle, scale):
    h, w = templ.shape
    sw, sh = w * scale, h * scale
    r = math.radians(angle)
    cw = int(math.ceil(abs(sw * math.cos(r)) + abs(sh * math.sin(r)))) + 2
    ch = int(math.ceil(abs(sw * math.sin(r)) + abs(sh * math.cos(r)))) + 2
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    m[0, 2] += (cw - 1) / 2 - (w - 1) / 2
    m[1, 2] += (ch - 1) / 2 - (h - 1) / 2
    image = cv2.warpAffine(templ, m, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Mask: pixels whose bilinear footprint lies fully inside the template.
    ones = np.full((h, w), 255, np.uint8)
    mask = cv2.warpAffine(ones, m, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return image, (mask == 255).astype(np.uint8)


def peaks(score_map, threshold, per_map):
    m = np.nan_to_num(score_map, nan=-1., posinf=-1., neginf=-1.)
    local = cv2.dilate(m, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((m >= threshold) & (m >= local))
    order = np.argsort(-m[ys, xs])[:per_map]
    return [(float(m[ys[k], xs[k]]), int(xs[k]), int(ys[k])) for k in order]


def overlap(a, b):
    ra = ((a['cx'], a['cy']), (a['width'], a['height']), -a['angle_deg'])
    rb = ((b['cx'], b['cy']), (b['width'], b['height']), -b['angle_deg'])
    kind, points = cv2.rotatedRectangleIntersection(ra, rb)
    if kind == cv2.INTERSECT_NONE or points is None:
        return 0.
    area = cv2.contourArea(cv2.convexHull(points))
    return area / min(a['width'] * a['height'], b['width'] * b['height'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--angle-step', type=float, default=1.)
    p.add_argument('--angle-min', type=float, default=-180.)
    p.add_argument('--angle-max', type=float, default=180.)
    p.add_argument('--scales', type=float, nargs='+', default=[1.])
    p.add_argument('--score', type=float, default=.8)
    p.add_argument('--overlap', type=float, default=.2)
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--per-map', type=int, default=0, help='peaks kept per score map; 0 means --limit')
    p.add_argument('--max-maps', type=int, default=0, help='smoke only: stop after this many maps')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)

    source = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    templ = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    th, tw = templ.shape
    n = int(round((args.angle_max - args.angle_min) / args.angle_step))
    angles = [args.angle_min + k * args.angle_step for k in range(n)]
    per_map = args.per_map or args.limit
    t0 = time.perf_counter()
    candidates, maps, map_ms = [], 0, []
    for scale in args.scales:
        for angle in angles:
            if args.max_maps and maps >= args.max_maps:
                break
            t = time.perf_counter()
            image, mask = rotated_template(templ, angle, scale)
            if image.shape[0] > source.shape[0] or image.shape[1] > source.shape[1]:
                continue
            score_map = cv2.matchTemplate(source, image, cv2.TM_CCOEFF_NORMED, mask=mask)
            ch, cw = image.shape
            for s, x, y in peaks(score_map, args.score, per_map):
                candidates.append(dict(kind='pose', cx=x + (cw - 1) / 2, cy=y + (ch - 1) / 2, angle_deg=angle,
                                       width=tw * scale, height=th * scale, scale=scale, score=s))
            maps += 1
            map_ms.append((time.perf_counter() - t) * 1e3)
    search_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    kept = []
    for c in sorted(candidates, key=lambda c: -c['score']):
        if len(kept) >= args.limit:
            break
        if all(overlap(c, k) <= args.overlap for k in kept):
            kept.append(c)
    nms_ms = (time.perf_counter() - t1) * 1e3
    planned = len(angles) * len(args.scales)
    out = dict(method='opencv_matchtemplate', device='cpu', opencv=cv2.__version__, numpy=np.__version__,
               threads=cv2.getNumThreads(), image_shape=list(source.shape), template_shape=[th, tw],
               angle_step=args.angle_step, scales=args.scales, maps=maps, planned_maps=planned,
               candidates=len(candidates),
               timing_ms=dict(search=search_ms, nms=nms_ms, per_map_median=float(np.median(map_ms)) if map_ms else None,
                              extrapolated_full_search=float(np.median(map_ms)) * planned if map_ms else None),
               memory=dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               detections=kept, count=len(kept))
    text = json.dumps(out, indent=1)
    if args.output:
        args.output.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
