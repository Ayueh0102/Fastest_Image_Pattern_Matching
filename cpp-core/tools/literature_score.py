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

Strict pose matching: centre error <= tol_px, angle error <= tol_deg (modulo the
declared symmetry period) and relative width and height error <= tol_scale.
Greedy one-to-one matching in descending score, nearest centre first, so a method
cannot score twice on one object. Box matching uses IoU >= 0.5 the same way, and
AP50 is the COCO 101-point interpolated precision over a pooled ranking. Errors of
matched pairs are reported separately from the counts, never folded into them.
"""
import math


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
    elif kind == 'box':
        out['box'] = (float(det['x0']), float(det['y0']), float(det['x1']), float(det['y1']))
        out['cx'] = (out['box'][0] + out['box'][2]) / 2
        out['cy'] = (out['box'][1] + out['box'][3]) / 2
    else:
        raise ValueError('Unknown detection kind: ' + str(kind))
    return out


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def _ranked(preds):
    return sorted(range(len(preds)), key=lambda i: -preds[i]['score'])


def match_poses(gt, preds, tol_px=2., tol_deg=1., tol_scale=.02, period=360.):
    """Strict pose matching. gt and preds are normalised pose records."""
    used, pairs = set(), []
    for i in _ranked(preds):
        p, best = preds[i], None
        for j, g in enumerate(gt):
            if j in used:
                continue
            d = math.hypot(p['cx'] - g['cx'], p['cy'] - g['cy'])
            if d > tol_px or angle_error(p['angle'], g['angle'], period) > tol_deg:
                continue
            if abs(p['width'] / g['width'] - 1) > tol_scale or abs(p['height'] / g['height'] - 1) > tol_scale:
                continue
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None:
            used.add(best[1])
            pairs.append((i, best[1]))
    return pairs


def match_centres(gt, preds, radius):
    """Loose localisation: one-to-one within radius pixels, used to split failure causes."""
    used, pairs = set(), []
    for i in _ranked(preds):
        p, best = preds[i], None
        for j, g in enumerate(gt):
            if j in used:
                continue
            d = math.hypot(p['cx'] - g['cx'], p['cy'] - g['cy'])
            if d <= radius and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            used.add(best[1])
            pairs.append((i, best[1]))
    return pairs


def match_boxes(gt_boxes, preds, threshold=.5):
    used, flags = set(), []
    for i in _ranked(preds):
        best = None
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(preds[i]['box'], g)
            if v >= threshold and (best is None or v > best[0]):
                best = (v, j)
        if best is not None:
            used.add(best[1])
        flags.append((preds[i]['score'], best is not None, None if best is None else best[1], i))
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
        p, g = preds[i], gt[j]
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
    return dict(n=len(values), mean=sum(values) / len(values), median=median, max=values[-1])


def score_pose_image(gt, preds, tol_px=2., tol_deg=1., tol_scale=.02, period=360., loose_radius=None):
    """Score one image for a pose-capable method; gt and preds are raw records."""
    g = [normalise(x) for x in gt]
    p = [normalise(x) for x in preds]
    pairs = match_poses(g, p, tol_px, tol_deg, tol_scale, period)
    out = dict(gt=len(g), predictions=len(p), strict_tp=len(pairs), strict_fp=len(p) - len(pairs),
               strict_fn=len(g) - len(pairs), errors=pose_errors(g, p, pairs, period))
    if loose_radius is not None:
        loose = match_centres(g, p, loose_radius)
        out.update(loose_tp=len(loose), loose_errors=pose_errors(g, p, loose, period))
    return out


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
