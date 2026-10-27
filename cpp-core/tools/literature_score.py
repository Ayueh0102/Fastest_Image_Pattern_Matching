"""Round 50: one independent scorer for every method in the literature comparison.

Standard library only, so the same file scores outputs from every environment.
Ground truth never reaches a method; methods write detections, this module reads
both. Three result kinds share one record format (see ``normalise``):

* pose  - centre, angle (degrees, counter-clockwise on screen, y down), object
          width/height in pixels. Methods that report a scale give it together
          with their reference size, so a 36x36-relative TDCM scale and a
          template-relative fpm scale become the same physical width/height.
* box   - axis-aligned x0, y0, x1, y1 in pixels. Poses convert to their
          enclosing box for box-only comparisons.
* count - the number of kept detections (or a method's own count).

Primary metrics, fixed in the round-50 protocol:

* pose-capable methods: strict pose matches (centre <= 2 px, angle <= 1 deg modulo
  the declared symmetry period, width and height ratio in [0.98, 1.02]) by
  maximum-cardinality bipartite matching, as many_object_accuracy.evaluate does;
* box-only comparisons: IoU >= 0.5, matched greedily in descending score as in
  COCO, precision, recall and COCO 101-point AP50 pooled over images.

Descriptive only: a 20 px association radius (localisation recall and the
errors of those pairs), TDCM's paper metrics (rotated-rectangle IoU in its own
angle convention, Succ. = IoU > 0.1). Errors of matched pairs are reported apart
from the counts, never folded into them.
"""
import math
import sys

STRICT = dict(position_px=2., angle_deg=1., relative_size=.02)
ASSOCIATION_RADIUS_PX = 20.


def angle_error(a, b, period=360.):
    """Smallest absolute difference of two angles in degrees, modulo period."""
    d = (a - b) % period
    return min(d, period - d)


def pose_corners(cx, cy, width, height, angle_deg):
    """Corners of a rotated rectangle; angle is counter-clockwise on a y-down screen."""
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    out = []
    for dx, dy in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
        # Counter-clockwise on screen with y down is clockwise in maths axes: y' flips sign of the sine.
        out.append((cx + dx * c + dy * s, cy - dx * s + dy * c))
    return out


def enclosing_box(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def normalise(det):
    """Return a detection with pose fields (if any) and an axis-aligned box."""
    kind = det['kind']
    out = dict(score=float(det.get('score', 0.)))
    if kind == 'pose':
        if 'width' in det:
            width, height = float(det['width']), float(det['height'])
        else:
            ref_w, ref_h = det['scale_reference']
            width = float(det['scale_x']) * ref_w
            height = float(det.get('scale_y', det['scale_x'])) * ref_h
        out.update(cx=float(det['cx']), cy=float(det['cy']), angle=float(det['angle_deg']), width=width, height=height)
        corners = det.get('corners') or pose_corners(out['cx'], out['cy'], width, height, out['angle'])
        out['box'] = enclosing_box(corners)
        out['valid'] = all(math.isfinite(v) for v in (out['cx'], out['cy'], out['angle'], width, height)) and width > 0 and height > 0
    elif kind == 'box':
        out['box'] = (float(det['x0']), float(det['y0']), float(det['x1']), float(det['y1']))
        out['cx'] = (out['box'][0] + out['box'][2]) / 2
        out['cy'] = (out['box'][1] + out['box'][3]) / 2
        out['valid'] = all(math.isfinite(v) for v in out['box'])
    else:
        raise ValueError('Unknown detection kind: ' + str(kind))
    return out


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def _polygon_area(points):
    return abs(sum(points[i][0] * points[i - 1][1] - points[i - 1][0] * points[i][1] for i in range(len(points)))) / 2


def _clip(subject, clipper):
    """Sutherland-Hodgman clipping of a convex polygon by a convex polygon (both counter-clockwise in maths axes)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def cross_point(p, q, a, b):
        x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    output = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i - 1], clipper[i]
        points, output = output, []
        for j in range(len(points)):
            p, q = points[j - 1], points[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    output.append(cross_point(p, q, a, b))
                output.append(q)
            elif inside(p, a, b):
                output.append(cross_point(p, q, a, b))
        if not output:
            return []
    return output


def _counter_clockwise(points):
    signed = sum(points[i - 1][0] * points[i][1] - points[i][0] * points[i - 1][1] for i in range(len(points)))
    return points if signed > 0 else points[::-1]


def rotated_iou(corners_a, corners_b):
    """IoU of two convex quadrilaterals given by their corners."""
    a, b = _counter_clockwise(list(corners_a)), _counter_clockwise(list(corners_b))
    inter = _clip(a, b)
    area_i = _polygon_area(inter) if len(inter) >= 3 else 0.
    union = _polygon_area(a) + _polygon_area(b) - area_i
    return area_i / union if union > 0 else 0.


def tdcm_paper_iou(gt, pred, reference=(36, 36)):
    """TDCM's getIOU: rectangles (cx, cy, sx*tw, sy*th) rotated by its rotated_rect convention.

    Its rotated_rect turns corners by +angle in y-down image coordinates (clockwise on
    screen), and getIOU passes -r, so each rectangle turns counter-clockwise by r: our
    pose_corners at +angle. (Until the round-50 merge acceptance this used -angle and drew
    both rectangles mirrored.) gt and pred are dicts with cx, cy, scale_x, scale_y, angle_deg.
    """
    tw, th = reference
    corners = [pose_corners(d['cx'], d['cy'], d['scale_x'] * tw, d['scale_y'] * th, d['angle_deg']) for d in (gt, pred)]
    return rotated_iou(*corners)


def _maximum_matching(edges, n_left):
    """Kuhn augmenting paths; edges[i] lists right indices in a fixed order. Returns right index -> left index."""
    match_right = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_right or augment(match_right[j], seen):
                match_right[j] = i
                return True
        return False

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * n_left + 100))
    try:
        for i in range(n_left):
            augment(i, set())
    finally:
        sys.setrecursionlimit(limit)
    return match_right


def _neighbours(gt, preds, radius):
    """For each truth, prediction indices whose centre lies within radius (grid buckets, deterministic order)."""
    cell = max(radius, 1.)
    grid = {}
    for j, p in enumerate(preds):
        if p['valid']:
            grid.setdefault((math.floor(p['cx'] / cell), math.floor(p['cy'] / cell)), []).append(j)
    out = []
    for g in gt:
        gx, gy = math.floor(g['cx'] / cell), math.floor(g['cy'] / cell)
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if math.hypot(preds[j]['cx'] - g['cx'], preds[j]['cy'] - g['cy']) <= radius:
                        near.append(j)
        out.append(sorted(near))
    return out


def _strict_edge(g, p, period):
    return (math.hypot(p['cx'] - g['cx'], p['cy'] - g['cy']) <= STRICT['position_px']
            and angle_error(p['angle'], g['angle'], period) <= STRICT['angle_deg']
            and abs(p['width'] / g['width'] - 1) <= STRICT['relative_size'] + 1e-12
            and abs(p['height'] / g['height'] - 1) <= STRICT['relative_size'] + 1e-12)


def match_poses(gt, preds, period=360.):
    """Strict pose pairs (truth index, prediction index) by maximum-cardinality matching."""
    near = _neighbours(gt, preds, STRICT['position_px'])
    edges = [[j for j in near[i] if _strict_edge(gt[i], preds[j], period)] for i in range(len(gt))]
    return sorted((i, j) for j, i in _maximum_matching(edges, len(gt)).items())


def match_centres(gt, preds, radius=ASSOCIATION_RADIUS_PX):
    """Descriptive localisation pairs within radius, maximum-cardinality matching."""
    edges = _neighbours(gt, preds, radius)
    return sorted((i, j) for j, i in _maximum_matching(edges, len(gt)).items())


def match_boxes(gt_boxes, preds, threshold=.5):
    """COCO-style: predictions in descending score take the unmatched truth of highest IoU >= threshold."""
    used, flags = set(), []
    centres = [((g[0] + g[2]) / 2, (g[1] + g[3]) / 2, max(g[2] - g[0], g[3] - g[1])) for g in gt_boxes]
    reach = max((c[2] for c in centres), default=0.)
    for i in sorted(range(len(preds)), key=lambda k: -preds[k]['score']):
        p, best = preds[i], None
        if p['valid']:
            span = max(p['box'][2] - p['box'][0], p['box'][3] - p['box'][1])
            for j, g in enumerate(gt_boxes):
                if j in used or abs(centres[j][0] - p['cx']) > reach + span or abs(centres[j][1] - p['cy']) > reach + span:
                    continue
                v = iou(p['box'], g)
                if v >= threshold and (best is None or v > best[0]):
                    best = (v, j)
        if best is not None:
            used.add(best[1])
        flags.append((p['score'], best is not None, None if best is None else best[1], i))
    return flags


def average_precision(ranked_flags, total_gt):
    """COCO-style 101-point interpolated AP from (score, is_tp) pairs pooled over images."""
    if total_gt == 0:
        return None
    flags = sorted(ranked_flags, key=lambda f: -f[0])
    tp = fp = 0
    precision, recall = [], []
    for _, hit in flags:
        tp += hit
        fp += not hit
        precision.append(tp / (tp + fp))
        recall.append(tp / total_gt)
    for k in range(len(precision) - 2, -1, -1):
        precision[k] = max(precision[k], precision[k + 1])
    total, r = 0., 0
    for t in range(101):
        level = t / 100
        while r < len(recall) and recall[r] < level:
            r += 1
        total += precision[r] if r < len(recall) else 0.
    return total / 101


def pose_errors(gt, preds, pairs, period=360.):
    rows = []
    for i, j in pairs:
        g, p = gt[i], preds[j]
        rows.append(dict(position_px=math.hypot(p['cx'] - g['cx'], p['cy'] - g['cy']),
                         angle_deg=angle_error(p['angle'], g['angle'], period),
                         width_rel=p['width'] / g['width'] - 1, height_rel=p['height'] / g['height'] - 1))
    return rows


def summary(values):
    values = sorted(values)
    if not values:
        return dict(n=0)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return dict(n=len(values), mean=sum(values) / len(values), median=median,
                p90=values[min(len(values) - 1, int(math.ceil(.9 * len(values))) - 1)], max=values[-1])


def error_summary(rows):
    return {k: summary([abs(r[k]) for r in rows]) for k in ('position_px', 'angle_deg', 'width_rel', 'height_rel')}


def score_pose_image(gt, preds, period=360.):
    """Score one image for a pose-capable method; gt and preds are raw records."""
    g = [normalise(x) for x in gt]
    p = [normalise(x) for x in preds]
    pairs = match_poses(g, p, period)
    loose = match_centres(g, p)
    return dict(gt=len(g), predictions=len(p), invalid_predictions=sum(not x['valid'] for x in p),
                strict_tp=len(pairs), strict_fp=len(p) - len(pairs), strict_fn=len(g) - len(pairs),
                strict_errors=pose_errors(g, p, pairs, period),
                association_tp=len(loose), association_errors=pose_errors(g, p, loose, period))


def score_box_images(images, threshold=.5):
    """images: list of (gt_boxes, raw preds). Returns pooled precision, recall, AP50 and centre errors."""
    pooled, total_gt, tp, npred, centre = [], 0, 0, 0, []
    for gt_boxes, preds in images:
        p = [normalise(x) for x in preds]
        flags = match_boxes(gt_boxes, p, threshold)
        total_gt += len(gt_boxes)
        npred += len(p)
        for score, hit, j, i in flags:
            pooled.append((score, hit))
            if hit:
                tp += 1
                g = gt_boxes[j]
                centre.append(math.hypot(p[i]['cx'] - (g[0] + g[2]) / 2, p[i]['cy'] - (g[1] + g[3]) / 2))
    return dict(gt=total_gt, predictions=npred, tp=tp, precision=tp / npred if npred else None,
                recall=tp / total_gt if total_gt else None, ap50=average_precision(pooled, total_gt),
                centre_error_px=summary(centre))


def count_errors(pairs):
    """pairs: list of (gt_count, predicted_count). MAE and RMSE over images."""
    if not pairs:
        return dict(n=0)
    diffs = [p - g for g, p in pairs]
    return dict(n=len(diffs), mae=sum(abs(d) for d in diffs) / len(diffs),
                rmse=math.sqrt(sum(d * d for d in diffs) / len(diffs)),
                empty_scene_false_positives=sum(p for g, p in pairs if g == 0))
