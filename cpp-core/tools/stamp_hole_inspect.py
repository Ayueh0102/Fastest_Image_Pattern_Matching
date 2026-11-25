"""Stamp-sheet pilot: native FPM holes, whole curved lines, and paired grid offsets.

Geometry defaults describe the two 4504 x 4096 stem_hole sample photographs;
this is an offline research tool, not a camera-calibration or production gate.
Dependencies: NumPy, SciPy, OpenCV, and the built fpm_c shared library.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cpp-core/examples'))
import python_ctypes as fpm


@dataclass
class Geometry:
    rows: int = 5
    columns: int = 10
    horizontal_intervals: int = 16
    vertical_intervals: int = 20
    margin_intervals: int = 8


@dataclass
class Detection:
    ids: list
    prediction: np.ndarray
    observations: np.ndarray
    accepted: np.ndarray
    candidates: np.ndarray
    stages_ms: dict


def hole_candidates(image, fast=True):
    lut = np.clip(np.arange(256) * (80 / 127 + 1) - 80, 0, 255).astype('uint8')
    gray = cv2.cvtColor(cv2.LUT(image, lut), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.inRange(gray, 0, 128), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        # An enclosing radius < 8 requires both integer bounding dimensions <= 16.
        # This exact rejection avoids expensive circles around huge background contours.
        if fast and max(cv2.boundingRect(contour)[2:]) > 16:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not 3 < radius < 8:
            continue
        area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
        if perimeter > 0 and 4 * np.pi * area / perimeter**2 > .45:
            points.append([x, y, radius, area])
    points = np.asarray(points, dtype=float).reshape(-1, 4)
    if len(points) < 30:
        raise ValueError('Too few hole candidates for this stamp-sheet geometry')
    return points


def initial_lines(points, geometry):
    xy = points[:, :2]
    pairs = cKDTree(xy).query_pairs(29, output_type='ndarray')
    dt = xy[pairs[:, 1]] - xy[pairs[:, 0]]
    result = []
    for axis, count in [(0, geometry.rows + 1), (1, geometry.columns + 1)]:
        u, v = xy[:, axis], xy[:, 1 - axis]
        du, dv = dt[:, axis], dt[:, 1 - axis]
        select = (abs(du) > 17) & (abs(dv) < 5)
        if not select.any():
            raise ValueError('No near-axis hole pitch found')
        slope = np.median(dv[select] / du[select])
        intercept = v - slope * (u - 2000)
        hist = np.bincount(np.maximum(0, intercept.round().astype(int)), minlength=4600)
        hist = gaussian_filter1d(hist.astype(float), 13)
        peaks, _ = find_peaks(hist, distance=160)
        if len(peaks) < count:
            raise ValueError('Not enough separated grid lines')
        peaks = sorted(peaks[np.argsort(hist[peaks])[-count:]])
        lines = []
        for peak in peaks:
            prediction = peak + slope * (u - 2000)
            for bound in [40, 15, 7, 4, 3]:
                use = abs(v - prediction) < bound
                if use.sum() < 12:
                    raise ValueError('Insufficient support for a whole grid line')
                curve = np.polynomial.Polynomial.fit(u[use], v[use], 3).convert()
                prediction = curve(u)
            lines.append(curve)
        result.append(lines)
    return result


def predicted_slots(lines, geometry):
    horizontal, vertical = lines
    crossings = []
    for h in horizontal:
        row = []
        for v in vertical:
            x = v(1800)
            for _ in range(4):
                x = v(h(x))
            row.append([x, h(x)])
        crossings.append(row)
    cross = np.array(crossings)
    slots = {}
    hp, vp = geometry.horizontal_intervals, geometry.vertical_intervals
    for row, curve in enumerate(horizontal):
        for k in range(geometry.columns * hp + 1):
            x = np.interp(k, np.arange(geometry.columns + 1) * hp, cross[row, :, 0])
            slots[('H', row, k)] = [x, curve(x)]
    for col, curve in enumerate(vertical):
        for k in range(-geometry.margin_intervals, geometry.rows * vp + 1):
            # Each intersection is detected once, then shared by both line measurements.
            if k >= 0 and k % vp == 0:
                continue
            y = (np.interp(k, np.arange(geometry.rows + 1) * vp, cross[:, col, 1])
                 if k >= 0 else cross[0, col, 1] + k * (cross[1, col, 1] - cross[0, col, 1]) / vp)
            slots[('V', col, k)] = [curve(y), y]
    return list(slots), np.array(list(slots.values()))


def image_view(image):
    return fpm.Image(image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                     image.shape[1], image.shape[0], image.strides[0])


def measure(image, ids, prediction, candidates, pattern, config, geometry):
    green = np.ascontiguousarray(image[:, :, 1])
    out = np.full((len(ids), 5), np.nan)
    height, width = green.shape
    for i, ((axis, _, k), (xx, yy)) in enumerate(zip(ids, prediction)):
        x, y = round(xx), round(yy)
        rx, ry = (10, 18) if axis == 'H' else (18, 11)
        if axis == 'H' and k % geometry.horizontal_intervals == 0:
            rx, ry = 10, 11
        left, top = x - rx - 9, y - ry - 9
        right, bottom = x + rx + 10, y + ry + 10
        if left < 0 or top < 0 or right > width or bottom > height:
            continue
        hits = pattern.match(image_view(green[top:bottom, left:right]), config)
        if hits:
            hit = hits[0]
            out[i, :3] = [round(left + hit['pos_x'] - .5),
                          round(top + hit['pos_y'] - .5), hit['score']]
    # Batch the independent checks; no change to native matching or pixel values.
    valid = np.flatnonzero(np.isfinite(out[:, 0]))
    centers = out[valid, :2].astype(int)
    yy, xx = np.mgrid[-9:10, -9:10]
    radius = np.hypot(xx, yy)
    ring, core = (radius >= 7) & (radius <= 9), radius <= 3.5
    def samples(mask):
        return green[centers[:, 1, None] + yy[mask], centers[:, 0, None] + xx[mask]]
    out[valid, 3] = np.median(samples(ring), axis=1) - np.median(samples(core), axis=1)
    out[valid, 4] = cKDTree(candidates[:, :2]).query(centers)[0]
    return out


def accepted_centers(observations, prediction):
    ok = (observations[:, 2] >= .80) & (observations[:, 3] >= 24)
    indices = np.flatnonzero(ok)
    for a, b in cKDTree(observations[indices, :2]).query_pairs(6):
        closest = sorted([indices[a], indices[b]],
                         key=lambda i: np.linalg.norm(observations[i, :2] - prediction[i]))
        ok[closest[1]] = False
    return ok


def detect(image, pattern, config, geometry, fast=True):
    t0 = perf_counter()
    candidates = hole_candidates(image, fast)
    t1 = perf_counter()
    ids, prediction = predicted_slots(initial_lines(candidates, geometry), geometry)
    t2 = perf_counter()
    observations = measure(image, ids, prediction, candidates, pattern, config, geometry)
    accepted = accepted_centers(observations, prediction)
    t3 = perf_counter()
    return Detection(ids, prediction, observations, accepted, candidates,
                     dict(candidates=(t1-t0)*1000, grid=(t2-t1)*1000,
                          native_match_and_checks=(t3-t2)*1000, total=(t3-t0)*1000))


def line_members(ids, geometry):
    index = {s: i for i, s in enumerate(ids)}
    members = {}
    hp, vp = geometry.horizontal_intervals, geometry.vertical_intervals
    for row in range(geometry.rows + 1):
        members[f'H{row+1}'] = [(k, index[('H', row, k)])
                                for k in range(geometry.columns * hp + 1)]
    for col in range(geometry.columns + 1):
        members[f'V{col+1}'] = [
            (k, index[('H', k // vp, col * hp)] if k >= 0 and k % vp == 0
             else index[('V', col, k)])
            for k in range(-geometry.margin_intervals, geometry.rows * vp + 1)]
    return members


def robust_fit(design, target, iterations=8):
    """Fixed low-order model; Tukey weighting prevents isolated defects pulling it."""
    weights = np.ones(len(design))
    for _ in range(iterations):
        root = np.sqrt(weights)
        coeff = np.linalg.lstsq(design * root[:, None], target * root[:, None], rcond=None)[0]
        residual = target - design @ coeff
        # Floor avoids overfitting quantized native centers; cap rejects defects >= 3 px.
        scale = max(.5, float(1.4826 * np.median(abs(residual - np.median(residual, axis=0)))))
        cutoff = min(3., 4.685 * scale)
        radius = np.linalg.norm(residual, axis=1) / cutoff
        weights = np.where(radius < 1, (1 - radius**2)**2, 0.)
        if np.count_nonzero(weights) < design.shape[1] + 4:
            raise ValueError('Insufficient inliers for smooth geometry model')
    return coeff


def fit_whole_curve(points, axis):
    u, v = points[:, axis], points[:, 1-axis]
    center, scale = float(np.mean(u)), float(np.ptp(u) / 2)
    if scale < 1:
        raise ValueError('Degenerate line')
    z = (u - center) / scale
    coeff = robust_fit(np.polynomial.polynomial.polyvander(z, 3), v[:, None])[:, 0]
    return np.polynomial.Polynomial(coeff)(np.polynomial.Polynomial([-center/scale, 1/scale]))


def distance_to_curve(points, curve, axis):
    """Signed closest normal distance; H positive down, V positive right."""
    u, v = points[:, axis], points[:, 1-axis]
    t = u.copy()
    first, second = curve.deriv(), curve.deriv(2)
    for _ in range(5):
        f, slope = curve(t), first(t)
        t -= ((t-u) + (f-v)*slope) / (1+slope*slope+(f-v)*second(t))
    foot = np.empty_like(points)
    foot[:, axis], foot[:, 1-axis] = t, curve(t)
    normal = np.empty_like(points)
    slope = first(t)
    normal[:, axis] = -slope / np.sqrt(1+slope*slope)
    normal[:, 1-axis] = 1 / np.sqrt(1+slope*slope)
    signed = np.sum((points-foot)*normal, axis=1)
    return signed, foot, normal


def warp_design(points):
    # Normalization only improves conditioning; it is not lens calibration.
    x, y = (points[:, 0]-2252)/2252, (points[:, 1]-2048)/2048
    return np.column_stack([x**a * y**b for degree in range(4)
                            for a in range(degree+1) for b in [degree-a]])


def geometry_support(detection):
    # Border/text matches can pass the loose detector. Require an independent
    # circular contour, or a strong template match, before using a point as geometry.
    obs = detection.observations
    return detection.accepted & ((obs[:, 4] <= 3) | (obs[:, 2] >= .90))


def analyze_lines(bad, good, geometry):
    """Separate per-hole deviation from whole-line motion relative to the good sheet.

    The registration for a line excludes EVERY point belonging to that line,
    including intersections, so a shifted full line cannot explain itself away.
    Smooth global changes remain empirical, not physically identified lens errors.
    """
    if bad.ids != good.ids:
        raise ValueError('Different grid identities')
    members = line_members(bad.ids, geometry)
    bxy, gxy = bad.observations[:, :2], good.observations[:, :2]
    supported_bad, supported_good = geometry_support(bad), geometry_support(good)
    paired = supported_bad & supported_good
    design = warp_design(gxy)
    summaries, records, curves = [], [], {}
    for name, slots in members.items():
        axis = 0 if name[0] == 'H' else 1
        keys, indices = np.array(slots).T
        okb, okg = supported_bad[indices], supported_good[indices]
        # A lone corner beyond a long obscured edge has no supported local baseline.
        # Do not report extrapolated curve distances there as measured defects.
        okb &= np.convolve(okb.astype(int), np.ones(13, int), mode='same') - okb >= 4
        okg &= np.convolve(okg.astype(int), np.ones(13, int), mode='same') - okg >= 4
        if min(okb.sum(), okg.sum()) < 20:
            raise ValueError(f'Insufficient accepted holes on {name}')
        bc = fit_whole_curve(bxy[indices[okb]], axis)
        gc = fit_whole_curve(gxy[indices[okg]], axis)
        rb, feet, normal = distance_to_curve(bxy[indices], bc, axis)
        rg, _, _ = distance_to_curve(gxy[indices], gc, axis)
        train = paired.copy()
        train[indices] = False
        if train.sum() < 100:
            raise ValueError('Too few independent points for line-excluded registration')
        coeff = robust_fit(design[train], bxy[train])
        mapped_good = design[indices] @ coeff
        # Incomplete bottom borders can bias the provisional pitch indices.
        # Match visible centers mutually after registration, within half a pitch.
        # This avoids reporting a one-hole index slip as a 25 px physical defect.
        bi, gi = np.flatnonzero(okb), np.flatnonzero(okg)
        distance, forward = cKDTree(mapped_good[gi]).query(bxy[indices[bi]])
        backward = cKDTree(bxy[indices[bi]]).query(mapped_good[gi])[1]
        pitch = float(np.median(np.diff(np.sort(bxy[indices[bi],axis]))))
        matched = np.full(len(indices), -1, dtype=int)
        mutual = (backward[forward] == np.arange(len(bi))) & (distance < .45*pitch)
        matched[bi[mutual]] = gi[forward[mutual]]
        valid = matched >= 0
        difference = np.full_like(mapped_good, np.nan)
        difference[valid] = bxy[indices[valid]] - mapped_good[matched[valid]]
        paired_normal = np.sum(difference * normal, axis=1)
        tangent = np.empty_like(normal)
        tangent[:, axis], tangent[:, 1-axis] = normal[:, 1-axis], -normal[:, axis]
        paired_along = np.sum(difference*tangent, axis=1)
        intervals = geometry.horizontal_intervals if axis == 0 else geometry.vertical_intervals
        nominal_count = geometry.columns if axis == 0 else geometry.rows
        segments = []
        for s in range(nominal_count):
            take = valid & (keys >= s*intervals) & (keys < (s+1)*intervals)
            if take.sum() >= 4:
                segments.append(dict(segment=s+1, count=int(take.sum()),
                                     median_normal_px=float(np.median(paired_normal[take])),
                                     median_along_px=float(np.median(paired_along[take]))))
        max_at = np.flatnonzero(okb)[np.argmax(abs(rb[okb]))]
        straight = np.polynomial.Polynomial.fit(bxy[indices[okb], axis], bxy[indices[okb], 1-axis], 1)
        summaries.append(dict(line=name, accepted_bad=int(okb.sum()), accepted_good=int(okg.sum()),
                              paired=int(valid.sum()), reassigned_pitch_indices=int(np.sum(valid & (matched != np.arange(len(indices))))),
                              nominal_sites=len(indices), median_reference_normal_px=float(np.median(paired_normal[valid])),
                              p95_abs_reference_normal_px=float(np.percentile(abs(paired_normal[valid]),95)),
                              max_abs_hole_to_curve_px=float(abs(rb[max_at])),
                              maximum_site_index=int(keys[max_at]),
                              good_p95_abs_hole_to_curve_px=float(np.percentile(abs(rg[okg]),95)),
                              curve_vs_straight_max_px=float(np.max(abs(bc(bxy[indices[okb],axis])-straight(bxy[indices[okb],axis])))),
                              segments=segments))
        curves[name] = dict(bad_coeff=bc.coef.tolist(), good_coeff=gc.coef.tolist(),
                            reference_to_bad_coeff=coeff.tolist(),
                            range=[float(np.nanmin(bxy[indices[okb],axis])),float(np.nanmax(bxy[indices[okb],axis]))])
        for n, i in enumerate(indices):
            records.append(dict(line=name, pitch_index=int(keys[n]), site_index=int(i),
                                accepted_bad=bool(okb[n]), accepted_good=bool(okg[n]),
                                reference_pitch_index=int(keys[matched[n]]) if valid[n] else None,
                                reference_pair_reliable=bool(valid[n]),
                                bad_x_px=float(bxy[i,0]),bad_y_px=float(bxy[i,1]),
                                curve_x_px=float(feet[n,0]),curve_y_px=float(feet[n,1]),
                                bad_normal_to_curve_px=float(rb[n]) if okb[n] else None,
                                good_normal_to_curve_px=float(rg[n]) if okg[n] else None,
                                matched_good_normal_to_curve_px=float(rg[matched[n]]) if valid[n] else None,
                                registered_reference_normal_px=float(paired_normal[n]) if valid[n] else None,
                                registered_reference_along_px=float(paired_along[n]) if valid[n] else None))
    return summaries, records, curves


def compare_pilot(detections, directory):
    checks = []
    for j, detection in enumerate(detections, 1):
        old = np.load(directory / f'obs{j}.npy')
        pts = np.load(directory / f'grid{j}.npz')['pts']
        checks.append(dict(image=j, candidates_identical=bool(np.array_equal(pts,detection.candidates)),
                           observations_identical=bool(np.allclose(old,detection.observations,equal_nan=True,atol=1e-8,rtol=0))))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bad', type=Path, required=True)
    parser.add_argument('--good', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True, help='19x19 green-channel single-hole crop')
    parser.add_argument('--library', type=Path, default=ROOT/'build/scene-workbench/libfpm_c.dylib')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--legacy-contours', action='store_true', help='Slow enclosing-circle-first comparison')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')
    args.out.mkdir(parents=True, exist_ok=True)
    template = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    if template is None or template.shape != (19,19):
        parser.error('template must be a readable 19x19 PNG')
    lib = fpm.Library(str(args.library.resolve()))
    config = lib.config()
    config.min_score, config.max_targets = .40, 1
    geometry = Geometry()
    runs = []
    with lib.learn(image_view(template), 1024) as pattern:
        for repeat in range(args.repeats + 1):
            start = perf_counter()
            detections, decode_ms = [], []
            for path in [args.bad,args.good]:
                t0 = perf_counter()
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Cannot read {path}')
                if image.shape[:2] != (4096,4504):
                    raise ValueError('This pilot requires the original 4504x4096 photographs')
                decode_ms.append((perf_counter()-t0)*1000)
                detections.append(detect(image,pattern,config,geometry,not args.legacy_contours))
            t0 = perf_counter()
            summaries, records, curves = analyze_lines(*detections,geometry)
            analysis_ms = (perf_counter()-t0)*1000
            total = (perf_counter()-start)*1000
            runs.append(dict(repeat=repeat, warmup=repeat==0, decode_ms=decode_ms,
                             detection_ms=[d.stages_ms for d in detections],
                             line_analysis_ms=analysis_ms,pair_total_ms=total))
            print(f'run {repeat}: {total:.1f} ms, accepted {[int(d.accepted.sum()) for d in detections]}',flush=True)
    checks = compare_pilot(detections,args.baseline_dir) if args.baseline_dir else []
    if checks and not all(x['candidates_identical'] and x['observations_identical'] for x in checks):
        raise AssertionError(f'Changed baseline observations: {checks}')
    for i,d in enumerate(detections,1):
        np.savez(args.out/f'detection{i}.npz',ids=np.asarray(d.ids),prediction=d.prediction,
                 observations=d.observations,accepted=d.accepted,candidates=d.candidates)
    with (args.out/'per-hole-lines.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    result = dict(input_images=[str(args.bad.resolve()),str(args.good.resolve())],
                  image_size=[4504,4096], units='original-resolution pixels; not millimeters',
                  model='Robust full-line cubic; degree-3 paired warp fitted with target line excluded',
                  caveat='Empirical smooth-shape compensation cannot distinguish physical sheet warp, lens distortion, and broad defects. No camera calibration or individual ground truth.',
                  accepted_counts=[int(d.accepted.sum()) for d in detections],
                  geometry_supported_counts=[int(geometry_support(d).sum()) for d in detections],
                  geometry=vars(geometry), baseline_checks=checks, lines=summaries,curves=curves,
                  benchmark=dict(runs=runs, excludes=['Python imports','camera acquisition','report files'],
                                 measured_repeats=args.repeats,warmup_excluded=1,
                                 pair_median_ms=float(np.median([r['pair_total_ms'] for r in runs[1:]])),
                                 in_memory_pair_median_ms=float(np.median([sum(d['total'] for d in r['detection_ms'])+r['line_analysis_ms'] for r in runs[1:]]))))
    (args.out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'benchmark':{k:v for k,v in result['benchmark'].items() if k!='runs'},'checks':checks},indent=2))


if __name__ == '__main__':
    main()
