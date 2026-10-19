"""Bounded CPU-only NCC slices for retained small-template scale failures.

This is an exploratory objective diagnostic, not a search benchmark or optimizer.
For each truth, hold either its x/y/angle or its retained prediction's x/y/angle
fixed and score 41 scales spanning truth * [0.95, 1.05], plus the exact truth and
estimated scales. An independent float64 bilinear/two-pass NCC checks every point.
No generated input, acceptance threshold, or matching implementation is changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from many_object_accuracy import evaluate
from pose_experiment import Scorer, library_path

ROOT = Path(__file__).resolve().parents[2]
ORACLE_TOLERANCE = 5e-5
FACTORS = np.linspace(.95, 1.05, 41)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed_pose(pose, scale, shape):
    height, width = shape
    return np.asarray([pose['pos_x'] - width / 2., height / 2. - pose['pos_y'],
                       pose['angle'], math.log(scale)], dtype=np.float32)


def oracle(source, template, poses):
    """Independent float64 geometry, zero-extended bilinear and two-pass NCC.

    Public float32 pose inputs are shared with C++; packing, coordinate math,
    sampling and centering all use float64 here. Small numerical differences
    are expected because production packs geometry and samples in float32.
    """
    sh, sw = source.shape
    th, tw = template.shape
    y, x = np.indices(template.shape, dtype=np.float64)
    u, v = x.ravel() - tw / 2., th / 2. - y.ravel()
    templ = template.astype(np.float64).ravel()
    templ -= templ.mean()
    template_energy = float(templ @ templ)
    image = source.astype(np.float64)
    results = []
    for tx, ty, degrees, log_scale in np.asarray(poses, dtype=np.float64):
        angle, scale = degrees * math.pi / 180., math.exp(log_scale)
        cs, sn = scale * math.cos(angle), scale * math.sin(angle)
        sx = sw / 2. + tx + cs * u - sn * v
        sy = sh / 2. - ty - sn * u - cs * v
        ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        fx, fy = sx - ix, sy - iy
        patch = np.zeros(len(u), dtype=np.float64)
        for oy in (0, 1):
            for ox in (0, 1):
                xx, yy = ix + ox, iy + oy
                valid = (xx >= 0) & (xx < sw) & (yy >= 0) & (yy < sh)
                weights = (fx if ox else 1. - fx) * (fy if oy else 1. - fy)
                patch[valid] += image[yy[valid], xx[valid]] * weights[valid]
        patch -= patch.mean()
        energy = float(patch @ patch)
        if energy / len(u) <= 1e-12 or template_energy / len(u) <= 1e-12:
            results.append(-2.)
        else:
            results.append(float(np.clip((patch @ templ) / math.sqrt(energy * template_energy), -1., 1.)))
    return np.asarray(results)


def oracle_control():
    source = np.random.default_rng(20261016).random((32, 40), dtype=np.float32)
    template = source[9:14, 11:18].copy()
    exact = packed_pose(dict(pos_x=14.5, pos_y=11.5, angle=0.), 1., source.shape)
    shifted = exact.copy()
    shifted[0] += 1.
    values = oracle(source, template, [exact, shifted])
    if abs(values[0] - 1.) > 1e-12 or values[1] >= .99:
        raise ValueError('Independent oracle coordinate control failed')
    return dict(exact_copy_score=float(values[0]), shifted_score=float(values[1]), passed=True)


def diagnose(cases_path, raw_path, build, name):
    cases = json.loads(cases_path.read_text())
    selected = [case for case in cases if case['name'] == name]
    if len(selected) != 1:
        raise ValueError('Expected one named case')
    case = selected[0]
    raw = json.loads(raw_path.read_text())
    first = raw['first']
    accuracy = evaluate(first['matches'], case['truth'])
    if accuracy['localization']['fn'] or accuracy['localization']['fp'] or not case['truth']:
        raise ValueError('This diagnostic requires complete unique localized predictions')
    if len(case['truth']) > 32:
        raise ValueError('Diagnostic is bounded to 32 truths')
    images = [cv2.imread(case[key], cv2.IMREAD_GRAYSCALE) for key in ('source', 'template')]
    if any(image is None for image in images):
        raise ValueError('Cannot read grayscale images')
    if images[1].size > 16384 or images[0].size > 4 * 1024 * 1024:
        raise ValueError('Diagnostic image bound exceeded')
    source, template = [image.astype(np.float32) / np.float32(255.) for image in images]
    controls = oracle_control()
    associations = {pair['truth_index']: pair['prediction_index']
                    for pair in accuracy['localization']['pairs']}
    poses, slices = [], []
    for truth_index, truth in enumerate(case['truth']):
        prediction_index = associations[truth_index]
        prediction = first['matches'][prediction_index]
        scales = [float(truth['scale'] * factor) for factor in FACTORS]
        scales.extend([truth['scale'], prediction['scale']])
        for anchor_name, anchor in (('truth_xy_angle', truth), ('predicted_xy_angle', prediction)):
            offset = len(poses)
            poses.extend(packed_pose(anchor, scale, source.shape) for scale in scales)
            slices.append(dict(truth_index=truth_index, prediction_index=prediction_index,
                               anchor=anchor_name, anchor_pose={key: anchor[key] for key in ('pos_x', 'pos_y', 'angle')},
                               truth_scale=truth['scale'], estimated_scale=prediction['scale'],
                               retained_estimated_ncc=prediction['score'], offset=offset, scales=scales))
    poses = np.asarray(poses, dtype=np.float32)
    scorer = Scorer(build, source, template, backend='cpu', threads=1)
    try:
        scores, metrics = scorer.score(poses)
        device, memory = scorer.device, scorer.memory()
    finally:
        scorer.close()
    if metrics['gpu_buffer_bytes'] or metrics['gpu_allocated_bytes'] or metrics['device_ms'] != -1:
        raise ValueError('Unexpected GPU activity in CPU-only diagnostic')
    independent = oracle(source, template, poses)
    difference = np.abs(scores.astype(np.float64) - independent)
    for item in slices:
        offset, scales = item.pop('offset'), item.pop('scales')
        measured = scores[offset:offset + 43].astype(np.float64)
        checked = independent[offset:offset + 43]
        best, oracle_best = int(np.argmax(measured[:41])), int(np.argmax(checked[:41]))
        item.update(grid=[dict(scale=scales[i], relative_to_truth=scales[i] / item['truth_scale'],
                               ncc=float(measured[i]), float64_ncc=float(checked[i])) for i in range(41)],
                    truth_scale_ncc=float(measured[41]), estimated_scale_ncc=float(measured[42]),
                    truth_scale_float64_ncc=float(checked[41]), estimated_scale_float64_ncc=float(checked[42]),
                    grid_best_index=best, grid_best_scale=scales[best], grid_best_ncc=float(measured[best]),
                    grid_best_signed_scale_error=scales[best] / item['truth_scale'] - 1.,
                    grid_best_minus_truth_ncc=float(measured[best] - measured[41]),
                    estimated_minus_truth_ncc=float(measured[42] - measured[41]),
                    grid_best_on_boundary=best in (0, 40),
                    float64_grid_best_index=oracle_best,
                    float64_grid_best_scale=scales[oracle_best],
                    max_abs_oracle_difference=float(difference[offset:offset + 43].max()))
    by_anchor = {}
    for anchor in ('truth_xy_angle', 'predicted_xy_angle'):
        selected = [item for item in slices if item['anchor'] == anchor]
        by_anchor[anchor] = dict(
            count=len(selected),
            best_below_truth_count=sum(item['grid_best_signed_scale_error'] < 0 for item in selected),
            grid_best_gt_gain_range=[min(item['grid_best_minus_truth_ncc'] for item in selected), max(item['grid_best_minus_truth_ncc'] for item in selected)],
            estimated_gt_gain_range=[min(item['estimated_minus_truth_ncc'] for item in selected), max(item['estimated_minus_truth_ncc'] for item in selected)],
            best_scale_error_range=[min(item['grid_best_signed_scale_error'] for item in selected), max(item['grid_best_signed_scale_error'] for item in selected)],
            independent_best_index_agrees=all(item['grid_best_index'] == item['float64_grid_best_index'] for item in selected),
            boundary_optima=sum(item['grid_best_on_boundary'] for item in selected))
    predicted = [item for item in slices if item['anchor'] == 'predicted_xy_angle']
    retained_error = max(abs(item['retained_estimated_ncc'] - item['estimated_scale_ncc']) for item in predicted)
    files = [cases_path, raw_path, Path(case['source']), Path(case['template']), Path(case['metadata']),
             Path(__file__), ROOT / 'cpp-core/tools/pose_experiment.py', ROOT / 'cpp-core/tools/many_object_accuracy.py',
             ROOT / 'cpp-core/tools/many_object_scenes.py', ROOT / 'cpp-core/experiments/pose_ncc.cpp',
             ROOT / 'cpp-core/experiments/pose_internal.hpp', library_path(build)]
    return dict(version=1, case=name, scope='Exploratory CPU-only fixed-pose score slices, excluded from formal speed comparisons.',
                backend=dict(requested='cpu', device=device, threads=1, memory=memory),
                score_count=len(poses), grid_factors=FACTORS.tolist(), grid_points=41,
                extra_points_per_slice=['truth_scale', 'estimated_scale'],
                normalization='Saved gray8 / float32(255), same normalization as native Session.',
                coordinates='tx=x-width/2, ty=height/2-y, angle unchanged in degrees, log_scale=log(scale); inputs narrowed to float32.',
                independent_oracle='Float64 bilinear zero extension, geometric transforms and two-pass centered NCC, using the same public float32 pose inputs; no production sampling helper.',
                oracle_tolerance=ORACLE_TOLERANCE, max_abs_oracle_difference=float(difference.max()),
                max_retained_prediction_score_difference=retained_error,
                verification_passed=bool(difference.max() <= ORACLE_TOLERANCE and retained_error <= ORACLE_TOLERANCE),
                oracle_coordinate_control=controls, accuracy=accuracy, summaries=by_anchor, slices=slices,
                source_sha256={str(path.resolve()): sha(path) for path in files},
                limitations=['One saved small-template synthetic scene with 12 poses; no inference about all cameras/templates.',
                             '41-point bounded one-dimensional slices do not establish a global optimum or full optimizer convergence.',
                             'Positive score gain away from truth supports a sampling/objective bias in this saved input; does not prove aliasing or a unique root cause.',
                             'No renderer/filter/coordinate convention was changed or tuned; no new search, refinement, GPU dispatch, or acceptance threshold change.',
                             'Output is not a formal time or memory benchmark; CPU execution and oracle analysis are outside measured GPU workers.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'build/resident-multiscale-inputs/cases.json')
    parser.add_argument('--raw', type=Path, default=ROOT / 'build/resident-multiscale-pilot/new_small12-mode0.result.json')
    parser.add_argument('--build', type=Path, default=ROOT / 'build/run-diagnostics')
    parser.add_argument('--case', default='new_small12')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/stage33-scale-bias.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    result = diagnose(args.cases, args.raw, args.build, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('score_count', 'verification_passed', 'max_abs_oracle_difference', 'summaries')}, ensure_ascii=False))
    if not result['verification_passed']:
        raise SystemExit('CPU/oracle cross-check failed; diagnostic retained')


if __name__ == '__main__':
    main()
