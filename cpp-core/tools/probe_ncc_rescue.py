"""Truth-free, bounded multistart NCC rescue exploration on the frozen CPU ABI.

This probe deliberately keeps the old library intact. Since that ABI exposes
only symmetric bounds about a supplied start, retries use conservative inner
boxes contained in the ORIGINAL seed's box. A production implementation can
retain the entire original box using explicit anchors; this probe never expands
the search region by recentering it. Truth is used only after all timed matching.
Timings are exploratory single-process timings, not the release comparison.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from benchmark_hybrid_refinement import CONFIG, pose_errors
from benchmark_pose_stress import ROOT, cases, digest, fixture, peak_rss, reference_ncc
from pose_experiment import Scorer, library_path


def halton(index, base):
    value, factor = 0., 1.
    while index:
        factor /= base
        value += factor * (index % base)
        index //= base
    return value


def bank_offsets(name):
    axes = np.concatenate((np.eye(4), -np.eye(4)))
    if name == 'axis8_half':
        return axes * [.25, .25, .5, .005]
    if name == 'axis8':
        return axes * [.5, .5, 1., .01]
    if name == 'axis8_wide':
        return axes * [1., 1., 2., .02]
    if name == 'axis8_broad':
        return axes * [2., 2., 4., .04]
    if name == 'axis8_far':
        return axes * [3., 3., 6., .06]
    if name in ('halton8', 'halton16_broad'):
        count = 8 if name == 'halton8' else 16
        radii = [1., 1., 2., .02] if name == 'halton8' else [3., 3., 6., .06]
        return np.asarray([[2*halton(i, base)-1 for base in (2, 3, 5, 7)]
                           for i in range(1, count+1)]) * radii
    raise ValueError(name)


def bounds(seed, config):
    limits = np.asarray([config['max_translation'], config['max_translation'],
                         config['max_angle_degrees'] if config['optimize_angle'] else 0.,
                         config['max_log_scale'] if config['optimize_scale'] else 0.])
    global_limit = np.asarray([1e6, 1e6, 36000., 3.])
    return np.maximum(seed-limits, -global_limit), np.minimum(seed+limits, global_limit)


def inward_starts(starts, lower, upper):
    # FP32 outputs must lie within the exact original FP64 interval.
    lo, hi = lower.astype(np.float32), upper.astype(np.float32)
    lo = np.where(lo.astype(np.float64) < lower, np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi.astype(np.float64) > upper, np.nextafter(hi, np.float32(-np.inf)), hi)
    return np.clip(starts, lo, hi).astype(np.float32)


def conservative_config(starts, lower, upper, config, iterations):
    distances = np.maximum(0., np.minimum(starts.astype(np.float64)-lower,
                                          upper-starts.astype(np.float64)))
    radii = distances.min(axis=0)
    # Translation uses one radius for both x/y in the frozen public ABI.
    return dict(config, max_iterations=iterations,
                max_translation=float(min(radii[0], radii[1])),
                max_angle_degrees=float(radii[2]), max_log_scale=float(radii[3]))


def rescue(scorer, seeds, baseline, config, bank='axis8', short_iterations=8,
           max_parents=4, min_score=.1, trigger_score=.8):
    """No image truth, case identity or expected answer enters this policy."""
    poses, scores, stats, _ = baseline
    poses, scores = poses.copy(), scores.copy()
    eligible = np.flatnonzero((scores >= min_score) & (scores < trigger_score))
    # The best unresolved matches get the bounded rescue budget first. Stable
    # input order breaks score ties; this ranking uses no expected geometry.
    eligible = sorted(eligible.tolist(), key=lambda i: (-float(scores[i]), i))
    selected = eligible[:max_parents]
    offset_bank = bank_offsets(bank)
    if not config['optimize_angle']:
        offset_bank[:, 2] = 0
    if not config['optimize_scale']:
        offset_bank[:, 3] = 0
    retries = []
    started = time.perf_counter()
    for parent in selected:
        original = seeds[parent].astype(np.float64)
        lower, upper = bounds(original, config)
        starts = inward_starts(original+offset_bank, lower, upper)
        # Bounds clipping/frozen coordinates may duplicate starts. Keep stable
        # first occurrence and omit the already-refined original seed.
        _, indices = np.unique(starts, axis=0, return_index=True)
        starts = starts[np.sort(indices)]
        starts = starts[np.any(starts != seeds[parent], axis=1)]
        if not len(starts):
            continue
        short_config = conservative_config(starts, lower, upper, config, short_iterations)
        short = scorer.refine(starts, **short_config)
        best = int(np.argmax(short[1]))
        candidate_pose, candidate_score = short[0][best], float(short[1][best])
        # If all short trials are worse than the known endpoint, further work
        # is not justified by the current evidence. This intentionally trades
        # completeness for a bounded retry cost.
        full = None
        if candidate_score > float(scores[parent]):
            full_config = conservative_config(candidate_pose[None], lower, upper,
                                              config, config['max_iterations'])
            full = scorer.refine(candidate_pose[None], **full_config)
            if float(full[1][0]) > candidate_score:
                candidate_pose, candidate_score = full[0][0], float(full[1][0])
        accepted = candidate_score > float(scores[parent])
        if accepted:
            poses[parent], scores[parent] = candidate_pose, candidate_score
        retries.append(dict(parent=parent, starts=len(starts), short_scores=short[1].tolist(),
                            short_wall_ms=short[3]['wall_ms'], accepted=accepted,
                            continuation_wall_ms=full[3]['wall_ms'] if full else 0.,
                            short_evaluations=sum(s['evaluations'] for s in short[2]),
                            continuation_evaluations=full[2][0]['evaluations'] if full else 0))
    elapsed = (time.perf_counter()-started)*1000
    return poses, scores, dict(eligible=eligible, selected=selected, retries=retries,
                              extra_wall_ms=elapsed,
                              extra_evaluations=sum(r['short_evaluations']+r['continuation_evaluations']
                                                    for r in retries))


def specifications(fresh_count, fresh_seed=620261014):
    for _, case in cases(SimpleNamespace(suite='accuracy', profile='large')):
        yield dict(case, split='shared', truth_accuracy=True)
    # Independent texture/pose RNG and template dimensions. The declared local
    # error prior includes harder seeds; no policy parameter reads these values.
    rng = np.random.default_rng(fresh_seed)
    for index in range(fresh_count):
        size = (80, 113, 160)[index % 3]
        truth = rng.uniform([-16, -16, -175, math.log(.8)], [16, 16, 175, math.log(1.2)])
        errors = rng.uniform([-6, -6, -9, -.14], [6, 6, 9, .14], (8, 4))
        yield dict(name=f'fresh_{index:02}', split='fresh', size=size, channels=1 if index%2 == 0 else 3,
                   source_shape=[384, 448], seed=1900101+(fresh_seed-620261014)*17+index*997, truth=truth.tolist(),
                   initial_poses=(truth+errors).tolist(), truth_accuracy=True,
                   appearance='gain_bias_noise' if index%3 == 1 else 'clean')
    truth = [2.125, -1.375, -43.25, math.log(1.13)]
    seeds = (np.asarray(truth)+[[1, -1, 2, -.025], [-1.5, 2, -3, .04]]).tolist()
    for kind in ('constant', 'low_contrast', 'negative', 'outside', 'frozen', 'all_frozen'):
        yield dict(name=f'regression_{kind}', split='regression', kind=kind, size=65,
                   channels=3, source_shape=[192, 224], seed=82171, truth=truth,
                   initial_poses=seeds, truth_accuracy=kind == 'low_contrast')


def prepare(case):
    source, template = fixture(case)
    seeds = np.asarray(case['initial_poses'], np.float32)
    config = dict(CONFIG)
    kind = case.get('kind')
    if kind == 'constant':
        source.fill(.5)
        template.fill(.5)
    elif kind == 'low_contrast':
        source = source*.0002+.5
        template = template*.0002+.5
    elif kind == 'negative':
        rng = np.random.default_rng(918271)
        source = rng.uniform(.2, .6, source.shape).astype(np.float32)
    elif kind == 'outside':
        seeds[:] = [1e5, 1e5, 0, 0]
    elif kind == 'frozen':
        config.update(optimize_angle=False, optimize_scale=False)
    elif kind == 'all_frozen':
        config.update(max_translation=0., max_angle_degrees=0., max_log_scale=0.)
    return source, template, seeds, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/resident-baseline')
    parser.add_argument('--output', type=Path, default=ROOT/'build/ncc-rescue/probe.json')
    parser.add_argument('--fresh-count', type=int, default=24)
    parser.add_argument('--fresh-seed', type=int, default=620261014)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--banks', nargs='+', default=['axis8_half', 'axis8', 'axis8_wide', 'halton8'])
    args = parser.parse_args()
    binary = library_path(args.build)
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    report = dict(protocol=dict(policy_truth_free=True, original_seed_box_never_expanded=True,
                  retry_bounds='conservative symmetric inner boxes; frozen CPU ABI limitation',
                  short_iterations=8, max_parents=4, min_score=.1, trigger_score=.8,
                  timing='exploratory; one process; no accuracy/oracle work in matching intervals',
                  fresh_seed=args.fresh_seed,
                  holdout='shared suite is calibration evidence; fresh independent RNG tests generalization'),
                  binary=dict(path=str(binary), sha256=binary_hash), runs=[])
    for case in specifications(args.fresh_count, args.fresh_seed):
        source, template, seeds, config = prepare(case)
        with Scorer(args.build, source, template, 'cpu', args.threads) as scorer:
            start = time.perf_counter()
            baseline = scorer.refine(seeds, **config)
            base_ms = (time.perf_counter()-start)*1000
            variants = {bank: rescue(scorer, seeds, baseline, config, bank) for bank in args.banks}
            owned, rss = scorer.memory(), peak_rss()
            # Truth and oracle comparisons begin only after all matching work.
            row = dict(case=case, configuration=config, baseline_wall_ms=base_ms,
                       baseline_scores=baseline[1].tolist(), baseline_poses=baseline[0].tolist(),
                       memory=dict(owned=owned, process_peak_bytes=rss), variants={},
                       input_sha256={name:digest(value) for name,value in
                                     [('source',source),('template',template),('seeds',seeds)]})
            row['baseline_accuracy'] = pose_errors(baseline[0], case['truth'], baseline[1]) if case['truth_accuracy'] else None
            for bank, (poses, scores, audit) in variants.items():
                inside = all(np.all((p >= bounds(s.astype(np.float64), config)[0]) &
                                    (p <= bounds(s.astype(np.float64), config)[1])) for p,s in zip(poses,seeds))
                oracle = [reference_ncc(source, template, pose) for pose in poses]
                row['variants'][bank] = dict(audit, poses=poses.tolist(), scores=scores.tolist(),
                    accuracy=pose_errors(poses,case['truth'],scores) if case['truth_accuracy'] else None,
                    original_bounds_passed=inside, score_monotonic=bool(np.all(scores >= baseline[1])),
                    oracle_max_error=float(np.max(np.abs(scores-oracle))))
            report['runs'].append(row)
        print(case['name'], 'base', round(base_ms,3), 'ms', 'rescued',
              {k:sum(r['accepted'] for r in v[2]['retries']) for k,v in variants.items()}, flush=True)
    report['summary'] = {}
    for bank in args.banks:
        groups = {}
        for split in ('shared','fresh','regression'):
            rows = [r for r in report['runs'] if r['case']['split'] == split]
            accuracy = [r for r in rows if r['baseline_accuracy'] is not None]
            groups[split] = dict(total=sum(len(r['baseline_accuracy']) for r in accuracy),
                baseline_precise=sum(e['precise_pass'] for r in accuracy for e in r['baseline_accuracy']),
                rescued_precise=sum(e['precise_pass'] for r in accuracy for e in r['variants'][bank]['accuracy']),
                baseline_wall_sum_ms=sum(r['baseline_wall_ms'] for r in rows),
                extra_wall_sum_ms=sum(r['variants'][bank]['extra_wall_ms'] for r in rows),
                retried_parents=sum(len(r['variants'][bank]['retries']) for r in rows),
                bounds_passed=all(r['variants'][bank]['original_bounds_passed'] for r in rows),
                monotonic=all(r['variants'][bank]['score_monotonic'] for r in rows))
        report['summary'][bank] = groups
    if hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
        raise RuntimeError('Frozen baseline changed during exploratory probing')
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__ == '__main__':
    main()
