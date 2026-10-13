"""Bounded, truth-free NCC rescue versus frozen CPU and resident-Metal baselines.

Fresh worker processes run serially. Truth is only consulted after timing and
RSS measurement. Supplied-seed, source-size, workload and full-detection suites
are deliberately separate: an 8K refinement does not claim an 8K global search.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from benchmark_hybrid_refinement import CONFIG, pose_errors
from benchmark_pose_stress import ROOT, SCORE_TOLERANCE, digest, field, fixture, metadata, peak_rss, reference_ncc, summary
from benchmark_resident_refinement import freeze_baseline, local_cases as resident_cases, prepare_local, sha
from pose_experiment import Scorer, library_path


METHODS = ('old_cpu', 'old_resident_verified', 'rescue_cpu', 'rescue_resident_verified')
SPLITS = ('original', 'heldout', 'challenge', 'source_size', 'workload', 'full')
DEFAULT_RESCUE = dict(trigger_min_score=.1, trigger_score=.8, max_rescued_seeds=4,
    translation_radius=3., angle_radius_degrees=6., log_scale_radius=.06,
    short_iterations=8, max_retry_sample_iterations=512*1024*1024)
EXPECTED_BASELINE_COMMIT = '958ab796ba1161a48a6e0511d0d4c6d687b5d735'


def input_digest(array):
    # Hash the contiguous buffer directly. tobytes() would transiently allocate
    # another 398 MB for an 8K RGB image and contaminate the measured peak RSS.
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def verify_baseline_provenance(manifest, frozen_hashes):
    """Require both copied libraries to match the explicitly frozen old build."""
    if manifest is None:
        return dict(verified=False, reason='No baseline manifest supplied; exploratory results cannot pass formal acceptance')
    if manifest.get('commit') != EXPECTED_BASELINE_COMMIT:
        raise ValueError('Baseline manifest does not identify the required 958ab79 commit')
    recorded = manifest.get('sha256', {})
    matches = {}
    for path, actual in frozen_hashes.items():
        name = Path(path).name
        expected = {key: value for key, value in recorded.items()
                    if PurePosixPath(key.replace('\\', '/')).name == name}
        if not expected or any(value != actual for value in expected.values()):
            raise ValueError('Frozen baseline library hash does not match manifest: '+name)
        matches[name] = dict(sha256=actual, matched_manifest_entries=expected)
    if len(matches) != 2:
        raise ValueError('Expected separately verified pose and core baseline libraries')
    return dict(verified=True, commit=EXPECTED_BASELINE_COMMIT, libraries=matches)


def cases(args):
    selected = SPLITS if args.suite == 'all' else (args.suite,)
    if 'original' in selected:
        for old in resident_cases(SimpleNamespace(suite='all', profile=args.profile)):
            yield dict(old, split='original')
    if 'heldout' in selected:
        # Frozen before any rescue-policy selection. These textures, truths and
        # offsets are not used by the policy-research script or production API.
        rng = np.random.default_rng(2026101307)
        count = 4 if args.profile == 'smoke' else 24
        for index in range(count):
            truth = rng.uniform([-18, -18, -175, math.log(.76)], [18, 18, 175, math.log(1.24)])
            errors = rng.uniform([-5, -5, -8, -.12], [5, 5, 8, .12], (4, 4))
            yield dict(name=f'independent_{index:02d}', scope='local', split='heldout', kind='heldout',
                size=(96, 160, 224)[index%3], channels=1 if index%2 == 0 else 3,
                source_shape=[640, 704], seed=790001+211*index, truth=truth.tolist(),
                initial_poses=(truth+errors).tolist(), truth_accuracy=True,
                appearance='gain_bias_noise' if index%3 == 2 else 'clean')
    if 'challenge' in selected:
        for channels in (1, 3):
            truth = [0., 0., 0., 0.]
            base = dict(scope='local', split='challenge', size=96, channels=channels,
                source_shape=[256, 320], seed=822701+channels, truth=truth,
                initial_poses=[[1., -1., 2., -.03], [4., -4., 7., -.10]], truth_accuracy=False)
            yield dict(base, name=f'negative_c{channels}', kind='negative', expected_no_detection=True)
            # A repetitive appearance admits a high-score wrong position. Report
            # this as an identifiability challenge, never quietly as precision
            # success. A low-score-triggered rescue need not touch it.
            yield dict(base, name=f'ambiguous_periodic_c{channels}', kind='ambiguous',
                initial_poses=[[8., 0., 0., 0.]], diagnostic_truth=True,
                config=dict(optimize_angle=False, optimize_scale=False))
            yield dict(base, name=f'bounded_frozen_c{channels}', kind='negative',
                expected_no_detection=True, config=dict(max_translation=0.,
                    max_angle_degrees=0., max_log_scale=0.))
    if 'source_size' in selected:
        # Same template and poses; only source extent changes. Setup still scans
        # and copies the full image, while local iterations sample its support.
        truth = np.asarray([13.125, -9.375, 41.2, math.log(.91)])
        offsets = np.random.default_rng(4917).uniform([-3, -3, -5, -.07], [3, 3, 5, .07], (32, 4))
        for height, width in ((2160, 3840), (4320, 7680)):
            for channels in (1, 3):
                yield dict(name=f'source_{width}x{height}_c{channels}_t256_n32', scope='local',
                    split='source_size', kind='large', size=256, channels=channels,
                    source_shape=[height, width], seed=69371+channels, truth=truth.tolist(),
                    initial_poses=(truth+offsets).tolist(), truth_accuracy=True)
    if 'workload' in selected:
        truth = np.asarray([13.125, -9.375, 41.2, math.log(.91)])
        sizes = ((128, 256), (128, 1024), (256, 256), (256, 1024), (768, 256))
        if args.profile == 'smoke':
            sizes = sizes[:1]
        for size, count in sizes:
            for channels in (1, 3):
                offsets = np.random.default_rng(25113+size).uniform(
                    [-3, -3, -5, -.07], [3, 3, 5, .07], (count, 4))
                yield dict(name=f'workload_t{size}_c{channels}_n{count}', scope='local',
                    split='workload', kind='large', size=size, channels=channels,
                    source_shape=[2160, 3840], seed=61373+size+channels, truth=truth.tolist(),
                    initial_poses=(truth+offsets).tolist(), truth_accuracy=True)


def prepare_case(case, destination):
    if case['kind'] not in ('negative', 'ambiguous'):
        return prepare_local(case, destination)
    source, template = fixture(case)
    if case['kind'] == 'negative':
        rng = np.random.default_rng(case['seed']+197)
        source = rng.uniform(.1, .9, source.shape).astype(np.float32)
        template = rng.uniform(.1, .9, template.shape).astype(np.float32)
    else:
        for array in (source, template):
            height, width = array.shape[:2]
            yy, xx = np.indices((height, width), dtype=np.float64)
            xx, yy = xx-width/2, height/2-yy
            periodic = .4+.11*np.sin(2*math.pi*xx/8)+.08*np.cos(2*math.pi*yy/13)
            unique = .004*np.exp(-((xx-9)**2+(yy+7)**2)/130)
            value = (periodic+unique).astype(np.float32)
            array[:] = value if array.ndim == 2 else value[:, :, None]
    seeds = np.asarray(case['initial_poses'], np.float32)
    destination.mkdir()
    arrays = dict(source=source, template=template, seeds=seeds)
    for name, array in arrays.items():
        np.save(destination/(name+'.npy'), array, allow_pickle=False)
    return dict(input_sha256={name: digest(value) for name, value in arrays.items()},
                shapes={name: list(value.shape) for name, value in arrays.items()})


def invoke(scorer, seeds, method, config, rescue_config):
    if method == 'old_cpu':
        return scorer.refine(seeds, **config)
    if method == 'old_resident_verified':
        return scorer.refine_resident(seeds, cpu_verify=True, **config)
    return scorer.refine_rescue(seeds, resident=method == 'rescue_resident_verified',
                                **config, **rescue_config)


def checked_outputs(source, template, seeds, records, baseline, config, case, rescue_config):
    """Independent checks after the timing/memory snapshot; no truth feedback."""
    threshold = case.get('score', .8)
    limits = np.asarray([config['max_translation']]*2+[
        config['max_angle_degrees'] if config['optimize_angle'] else 0.,
        config['max_log_scale'] if config['optimize_scale'] else 0.])
    lower = np.maximum(seeds.astype(np.float64)-limits, [-1e6, -1e6, -36000., -3.])
    upper = np.minimum(seeds.astype(np.float64)+limits, [1e6, 1e6, 36000., 3.])
    output = []
    with Scorer(baseline, source, template, 'cpu', 8) as oracle:
        initial_scores, _ = oracle.score(seeds)
        for record in records:
            value = record['result']
            poses, scores = np.asarray(value['poses'], np.float32), np.asarray(value['scores'], np.float32)
            cpu_scores, _ = oracle.score(poses)
            accuracy = pose_errors(poses, case['truth'], cpu_scores) if case.get('truth_accuracy') or case.get('diagnostic_truth') else None
            exact_cpu_scores = bool(np.array_equal(scores, cpu_scores))
            bounds = bool(np.all((poses >= lower) & (poses <= upper)))
            frozen = all(np.array_equal(poses[:, axis], seeds[:, axis]) for axis in range(4) if limits[axis] == 0)
            monotonic = bool(np.all(cpu_scores >= initial_scores))
            invalid = all(scores[i] == -2 and np.array_equal(poses[i], seeds[i]) for i in case.get('expected_invalid', []))
            finite = bool(np.isfinite(poses).all() and np.isfinite(scores).all())
            no_detection = bool(np.all(cpu_scores < threshold)) if case.get('expected_no_detection') else None
            rescue = value['metrics'].get('rescue')
            iteration_cap = config['max_iterations']+(16*rescue_config['short_iterations']+config['max_iterations'] if rescue else 0)
            stats_valid = all(0 <= s['iterations'] <= iteration_cap and s['evaluations'] >= 1 and 0 <= s['status'] <= 3 for s in value['stats'])
            bounded = None
            if rescue is not None:
                reserved_per_parent = int(template.size)*(16*rescue_config['short_iterations']+config['max_iterations'])
                bounded = (0 <= rescue['accepted_seeds'] <= rescue['selected_seeds'] <=
                    min(rescue_config['max_rescued_seeds'], rescue['eligible_seeds']) and
                    0 <= rescue['retry_starts'] <= 16*rescue['selected_seeds'] and
                    0 <= rescue['continuations'] <= rescue['selected_seeds'] and
                    rescue['reserved_retry_sample_iterations'] == rescue['selected_seeds']*reserved_per_parent and
                    rescue['reserved_retry_sample_iterations'] <= rescue_config['max_retry_sample_iterations'] and
                    rescue['command_buffers'] == rescue['cpu_waits'] and
                    rescue['auxiliary_buffer_bytes'] <= 16*1024*1024)
            row = dict(cpu_rechecked_scores=cpu_scores.tolist(), exact_cpu_scores=exact_cpu_scores,
                score_max_error=float(np.max(np.abs(scores-cpu_scores))), finite=finite,
                original_seed_bounds_passed=bounds, frozen_passed=frozen, monotonic=monotonic,
                invalid_passed=invalid, stats_valid=stats_valid, bounded_retry_work_passed=bounded,
                expected_no_detection_passed=no_detection,
                detection_threshold=threshold, threshold_disagreements=int(np.sum((scores >= threshold) != (cpu_scores >= threshold))),
                accepted_count=int(np.sum(cpu_scores >= threshold)), accuracy=accuracy)
            row['passed'] = all((exact_cpu_scores, bounds, frozen, monotonic, invalid, finite,
                                 stats_valid, bounded is not False, no_detection is not False, row['threshold_disagreements'] == 0))
            output.append(row)
    last = records[-1]['result']
    poses, scores = np.asarray(last['poses']), np.asarray(last['scores'])
    indices = range(len(seeds)) if len(seeds) <= 16 else sorted({0, len(seeds)//2, len(seeds)-1})
    independent = [dict(index=i, score=float(scores[i]), oracle=reference_ncc(source, template, poses[i])) for i in indices]
    oracle_error = max(abs(row['score']-row['oracle']) for row in independent)
    accuracy = output[-1]['accuracy']
    return dict(calls=output, initial_scores=initial_scores.tolist(), oracle=independent,
        oracle_max_error=oracle_error, seed_count=len(seeds),
        precise_pass_count=sum(r['precise_pass'] for r in accuracy) if accuracy is not None else None,
        accuracy_is_diagnostic=bool(case.get('diagnostic_truth')),
        passed=all(r['passed'] for r in output) and oracle_error <= SCORE_TOLERANCE)


def worker(job):
    case, method = job['case'], job['method']
    full = case['scope'] == 'full'
    rescue_config = dict(DEFAULT_RESCUE, **job.get('rescue_config', {}))
    if full:
        import cv2
        import real_image_experiments as real
        cv2.setNumThreads(1)
        original_scorer = real.Scorer
        if method.startswith('rescue'):
            class RescueScorer(Scorer):
                def refine(self, seeds, **config):
                    return invoke(self, seeds, method, config, rescue_config)

                def refine_resident(self, seeds, cpu_verify=True, **config):
                    return invoke(self, seeds, method, config, rescue_config)
            real.Scorer = RescueScorer
        stage_method = 'continuous_direct_cpu' if method.endswith('_cpu') else 'resident_verified_direct_metal'
        before = peak_rss()
        start = time.perf_counter()
        try:
            search = real.Search(dict(job, method=stage_method, audit_refinement=True, prune=False))
        finally:
            real.Scorer = original_scorer
        constructor_ms = (time.perf_counter()-start)*1000
        scorer, call, close = search.scorer, search.run, search.close
        source, template = search.source_float, search.template.astype(np.float32)/255
        setup_ms = search.setup_ms
        initial_hashes = {name: sha(case[name]) for name in ('source', 'template')}
    else:
        source, template, seeds = [np.load(Path(job['fixture'])/(name+'.npy'), allow_pickle=False)
                                   for name in ('source', 'template', 'seeds')]
        before = peak_rss()
        start = time.perf_counter()
        scorer = Scorer(job['build'], source, template, 'cpu' if method.endswith('_cpu') else 'metal', job['threads'])
        constructor_ms = (time.perf_counter()-start)*1000
        setup_ms, close = scorer.setup_ms, scorer.close
        config = dict(CONFIG, **case.get('config', {}))
        call = lambda: invoke(scorer, seeds, method, config, rescue_config)
        initial_hashes = {name: input_digest(array) for name, array in zip(('source', 'template', 'seeds'), (source, template, seeds))}
    try:
        initial_memory = scorer.memory()
        memories = [initial_memory]

        def one():
            start = time.perf_counter()
            value = call()
            elapsed = (time.perf_counter()-start)*1000
            memories.append(scorer.memory())
            if full:
                audit = search.last_refinement
                if audit is None:
                    raise RuntimeError('Full scene yielded no refinement candidates')
                original, poses, scores, stats, metrics = audit
                value = dict(value, initial_poses=original.tolist(), poses=poses.tolist(), scores=scores.tolist(), stats=stats, metrics=metrics)
            else:
                poses, scores, stats, metrics = value
                value = dict(poses=poses.tolist(), scores=scores.tolist(), stats=stats, metrics=metrics)
            return dict(wall_ms=elapsed, result=value)

        first = one()
        warm_start, warmups = time.perf_counter(), 0
        while warmups < 1 or (time.perf_counter()-warm_start)*1000 < job['warmup_ms']:
            call()
            warmups += 1
        measured = [one() for _ in range(job['repeats'])]
        process_peak = peak_rss()
        records = [first, *measured]
        if full:
            config = dict(CONFIG, optimize_scale=search.allow_scale, optimize_angle=search.rotation)
            seeds = np.asarray(first['result']['initial_poses'], np.float32)
            identical_seeds = all(record['result']['initial_poses'] == first['result']['initial_poses'] for record in measured)
        else:
            identical_seeds = True
        checked = checked_outputs(source, template, seeds, records, job['reference_build'], config, case, rescue_config)
        metrics = [record['result']['metrics'] for record in measured]
        result = dict(device=scorer.device, input_sha256=initial_hashes, setup_ms=setup_ms,
            constructor_wall_ms=constructor_ms, cold_constructor_and_first_ms=constructor_ms+first['wall_ms'],
            first=first, measured=measured, warmups=warmups,
            algorithm_wall=summary([record['wall_ms'] for record in measured]),
            c_api_wall=summary([metric['wall_ms'] for metric in metrics]),
            gpu_command=summary([metric['device_ms'] for metric in metrics]) if all(metric['device_ms'] >= 0 for metric in metrics) else None,
            memory=dict(process_peak_before_setup_bytes=before, process_peak_bytes=process_peak,
                max_observed_owned_buffers={key: max(memory[key] for memory in memories) for key in initial_memory}),
            initial_memory=initial_memory, validation=checked, identical_repeat_seeds=identical_seeds,
            passed=checked['passed'] and identical_seeds)
        if full:
            result['truth_accuracy'] = [real.truth_accuracy(record['result']['matches'], case['truth']) for record in records] if case.get('truth') else None
        return result
    finally:
        close()


def comparisons(report):
    output = []
    for name in sorted({row['job']['case']['name'] for row in report['runs']}):
        group = {row['job']['method']: row for row in report['runs'] if row['job']['case']['name'] == name}
        for method, row in group.items():
            reference = 'old_cpu' if method.endswith('_cpu') else 'old_resident_verified'
            if reference not in group:
                continue
            actual, baseline = row['result'], group[reference]['result']
            if actual.get('status') == 'unavailable' or baseline.get('status') == 'unavailable':
                continue
            old_checks, checks = baseline['validation']['calls'], actual['validation']['calls']
            old_records = [baseline['first'], *baseline['measured']]
            records = [actual['first'], *actual['measured']]
            same_seeds = all(a['result'].get('initial_poses') == b['result'].get('initial_poses')
                             for a, b in zip(records, old_records))
            item = dict(case=name, split=row['job']['case']['split'], method=method, baseline_method=reference,
                identical_inputs=actual['input_sha256'] == baseline['input_sha256'], identical_coarse_seeds=same_seeds,
                algorithm_slowdown=actual['algorithm_wall']['median_ms']/baseline['algorithm_wall']['median_ms'],
                refinement_slowdown=actual['c_api_wall']['median_ms']/baseline['c_api_wall']['median_ms'],
                process_peak_delta_bytes=actual['memory']['process_peak_bytes']-baseline['memory']['process_peak_bytes'],
                retained_buffer_delta_bytes={key: value-baseline['memory']['max_observed_owned_buffers'][key]
                    for key, value in actual['memory']['max_observed_owned_buffers'].items()},
                precise_pass_count=actual['validation']['precise_pass_count'],
                baseline_precise_pass_count=baseline['validation']['precise_pass_count'],
                seed_count=actual['validation']['seed_count'])
            if method.startswith('rescue'):
                diagnostics = [record['result']['metrics'].get('rescue') for record in records]
                item['diagnostics_present'] = all(value is not None for value in diagnostics)
                item['retry_parent_counts'] = [value['selected_seeds'] for value in diagnostics if value is not None]
                item['no_retries'] = item['diagnostics_present'] and not any(item['retry_parent_counts'])
                old_ms, new_ms = baseline['algorithm_wall']['median_ms'], actual['algorithm_wall']['median_ms']
                if row['job']['case']['scope'] == 'full':
                    limit = max(old_ms*.1, 2.)
                    item['full_pipeline_overhead_passed'] = new_ms-old_ms <= limit
                    item['overhead_limit_ms'] = limit
                elif item['no_retries']:
                    limit = max(old_ms*.1, .5)
                    item['easy_case_overhead_passed'] = new_ms-old_ms <= limit
                    item['overhead_limit_ms'] = limit
            if same_seeds:
                item['worst_cpu_score_loss_vs_baseline'] = max(float(np.max(np.asarray(a['cpu_rechecked_scores'])-
                    np.asarray(b['cpu_rechecked_scores']))) for a, b in zip(old_checks, checks))
                item['threshold_acceptance_changes'] = sum(int(np.sum(
                    (np.asarray(a['cpu_rechecked_scores']) >= a['detection_threshold']) !=
                    (np.asarray(b['cpu_rechecked_scores']) >= b['detection_threshold']))) for a, b in zip(old_checks, checks))
            if checks[-1]['accuracy'] is not None:
                item['precise_regressions'] = sum(x['precise_pass'] and not y['precise_pass']
                    for old_call, call in zip(old_checks, checks) for x, y in zip(old_call['accuracy'], call['accuracy']))
                item['precise_improvements'] = sum(not x['precise_pass'] and y['precise_pass']
                    for old_call, call in zip(old_checks, checks) for x, y in zip(old_call['accuracy'], call['accuracy']))
            if row['job']['case']['scope'] == 'full':
                from real_image_experiments import compare
                item['detection_agreement'] = compare(old_records[-1]['result']['matches'], records[-1]['result']['matches'])
                if not row['job']['case'].get('truth'):
                    from audit_resident_refinement import match_detections
                    item['unlabeled_detection_agreement'] = []
                    for index, (old_record, record) in enumerate(zip(old_records, records)):
                        agreement = match_detections(old_record['result']['matches'], record['result']['matches'])
                        agreement['reference_kind'] = 'frozen same-backend detection; agreement, not calibrated truth'
                        agreement['call'] = 'first' if index == 0 else 'measured_'+str(index-1)
                        item['unlabeled_detection_agreement'].append(agreement)
                    item['unlabeled_detection_agreement_passed'] = (len(old_records) == len(records) and
                        all(value['passed'] for value in item['unlabeled_detection_agreement']))
                if actual['truth_accuracy']:
                    item['full_truth'] = actual['truth_accuracy'][-1]
                    item['full_false_positive_increases'] = sum(max(0, a['pose_fp']-b['pose_fp'])
                        for a, b in zip(actual['truth_accuracy'], baseline['truth_accuracy']))
            output.append(item)
    return output


def finish_report(report):
    report['comparisons'] = comparisons(report)
    runs = [row for row in report['runs'] if row['result'].get('status') != 'unavailable']
    rescue = [row for row in runs if row['job']['method'].startswith('rescue')]
    local_truth = [row for row in rescue if row['job']['case'].get('truth_accuracy')]
    full_truth = [row for row in rescue if row['job']['case']['scope'] == 'full' and row['result']['truth_accuracy'] is not None]
    report['complete'] = len(report['runs']) == report['expected_runs']
    report['unavailable_runs'] = len(report['runs'])-len(runs)
    report['correctness_passed'] = bool(runs) and all(row['result']['passed'] for row in runs)
    report['rescue_local_accuracy_passed'] = all(all(p['precise_pass'] for p in call['accuracy'])
        for row in local_truth for call in row['result']['validation']['calls']) if local_truth else None
    report['rescue_full_accuracy_passed'] = all(call['pose_fn'] == 0 and call['pose_fp'] == 0
        for row in full_truth for call in row['result']['truth_accuracy']) if full_truth else None
    report['cross_method_nonregression'] = bool(report['comparisons']) and len(report['comparisons']) == len(runs) and all(
        row['identical_inputs'] and row['identical_coarse_seeds'] and row.get('precise_regressions', 0) == 0
        and row.get('full_false_positive_increases', 0) == 0 and row.get('worst_cpu_score_loss_vs_baseline', 0) <= 0
        and row.get('diagnostics_present', True) and row.get('unlabeled_detection_agreement_passed', True)
        for row in report['comparisons'])
    failure = [row for row in rescue if row['job']['case']['name'] == 'heldout_16']
    report['common_failure_recovered'] = all(call['accuracy'][3]['precise_pass'] for row in failure
        for call in row['result']['validation']['calls']) if failure else None
    easy = [row['easy_case_overhead_passed'] for row in report['comparisons'] if 'easy_case_overhead_passed' in row]
    full = [row['full_pipeline_overhead_passed'] for row in report['comparisons'] if 'full_pipeline_overhead_passed' in row]
    report['easy_case_overhead_passed'] = all(easy) if easy else None
    report['full_pipeline_overhead_passed'] = all(full) if full else None
    report['source_hashes_unchanged'] = all(sha(ROOT/path) == value for path, value in report['source_sha256'].items())
    report['binary_hashes_unchanged'] = all(sha(path) == value for path, value in report['binary_hashes'].items())
    # Ambiguous repeated textures are explicitly diagnostic. A pass here means
    # these declared identifiable fixtures pass; it is not global NCC proof.
    report['accuracy_acceptance_passed'] = bool(rescue) and report['complete'] and report['unavailable_runs'] == 0 and all(
        report[key] for key in ('correctness_passed', 'cross_method_nonregression', 'source_hashes_unchanged', 'binary_hashes_unchanged')) \
        and report['baseline_verification']['verified'] \
        and report['rescue_local_accuracy_passed'] is not False and report['rescue_full_accuracy_passed'] is not False
    report['acceptance_passed'] = report['accuracy_acceptance_passed'] and all(report[key] is not False for key in (
        'common_failure_recovered', 'easy_case_overhead_passed', 'full_pipeline_overhead_passed'))
    report['completed_at'] = datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--baseline-manifest', type=Path, help='Required for formal acceptance; verifies both old libraries and commit')
    parser.add_argument('--rescue-config', type=Path, help='Frozen policy JSON; configuration is passed identically to CPU and Metal')
    parser.add_argument('--output', type=Path, default=ROOT/'build/ncc-rescue/report.json')
    parser.add_argument('--suite', choices=(*SPLITS, 'all'), default='original')
    parser.add_argument('--profile', choices=('smoke', 'standard', 'large'), default='large')
    parser.add_argument('--case', action='append', help='Select case names containing any supplied substring')
    parser.add_argument('--methods', choices=METHODS, nargs='+', default=list(METHODS))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-ms', type=float, default=1000.)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            result = worker(job)
        except RuntimeError as error:
            if not job['method'].endswith('_cpu') and str(error) in (
                'No Metal GPU available', 'Metal backend was not compiled on this platform'):
                result = dict(status='unavailable', reason=str(error))
            else:
                raise
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if args.baseline is None:
        parser.error('--baseline must identify the separately frozen old 958ab79 build')
    if args.repeats < 1 or args.repeats%2 != 1 or args.threads < 1 or args.warmup_ms < 0:
        parser.error('Use odd positive repeats, threads>=1 and warmup-ms>=0')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assets = args.output.parent/(args.output.stem+'-artifacts-'+str(time.time_ns()))
    assets.mkdir()
    frozen = assets/'baseline-frozen'
    binary_hashes = freeze_baseline(args.baseline.resolve(), frozen)
    baseline_manifest = json.loads(args.baseline_manifest.read_text()) if args.baseline_manifest else None
    baseline_verification = verify_baseline_provenance(baseline_manifest, binary_hashes)
    binary_hashes.update({str(path.resolve()): sha(path) for path in (
        library_path(args.build), args.build/('libfpm_c.dylib' if sys.platform == 'darwin' else 'libfpm_c.so'))})
    selected = list(cases(args))
    if args.suite in ('all', 'full'):
        from real_image_experiments import controlled_scene, held_out_scene
        for name, generate in (('controlled_scale', controlled_scene), ('held_out_scale', held_out_scene)):
            path = assets/(name+'.png')
            selected.append(dict(name=name, scope='full', split='full', source=str(path),
                template=str(ROOT/'Test Images/Dst3.bmp'), truth=generate(path), angle=180,
                limit=70, proposal_score=.5, score=.8, overlap=.2, scales=[.75, 1., 1.25]))
        selected.append(dict(name='src7', scope='full', split='full', source=str(ROOT/'Test Images/Src7.bmp'),
            template=str(ROOT/'Test Images/Dst7.bmp'), angle=180, limit=70, proposal_score=.5,
            score=.5, overlap=.2, scales=[1.]))
    selected = [case for case in selected if not args.case or any(term in case['name'] for term in args.case)]
    if not selected:
        parser.error('No selected cases')
    policy = dict(DEFAULT_RESCUE, **(json.loads(args.rescue_config.read_text()) if args.rescue_config else {}))
    info = metadata(args)
    files = [Path(__file__).resolve(), ROOT/'cpp-core/tools/benchmark_resident_refinement.py',
        ROOT/'cpp-core/tools/benchmark_hybrid_refinement.py', ROOT/'cpp-core/tools/real_image_experiments.py',
        ROOT/'cpp-core/tools/audit_resident_refinement.py',
        ROOT/'cpp-core/examples/python_ctypes.py', ROOT/'cpp-core/tests/pose_rescue_test.cpp',
        ROOT/'cpp-core/docs/ncc-rescue/protocol.json']
    files += list((ROOT/'cpp-core/src').glob('*'))
    info['source_sha256'].update({str(path.relative_to(ROOT)): sha(path) for path in files if path.is_file()})
    report = dict(**info, binary_hashes=binary_hashes, baseline_origin=str(args.baseline.resolve()),
        baseline_provenance=baseline_manifest, baseline_verification=baseline_verification,
        baseline_manifest_sha256=sha(args.baseline_manifest) if args.baseline_manifest else None,
        policy=policy, policy_sha256=sha(args.rescue_config) if args.rescue_config else None,
        protocol=dict(serial=True, fresh_worker_per_variant=True, shuffled_order_seed=64197,
            repeats=args.repeats, warmup_ms=args.warmup_ms, cpu_threads=args.threads, config=CONFIG,
            timing='Outer call includes rescue work and final CPU verification, excludes setup/loading/output serialization/validation. GPU command time overlaps wall.',
            cold='Constructor+first includes no cache purge; source_size isolates full-image setup from supplied-seed warm computation.',
            memory='Peak RSS captured before validation. Owned buffer snapshots are retained capacities, not allocation-event peaks; GPU shared memory overlaps RSS.',
            local_accuracy='Identifiable fixtures: CPU NCC>=.8 and <=.5px/.5degree/1% scale; original thresholds unchanged.',
            full_accuracy='Live unchanged CPU coarse search; 6/12 truth objects <=2px/1degree/2%; Src7 compares detections, has no calibrated truth.',
            challenges='Random negatives require no scores>=.8; periodic high-score wrong peaks report truth diagnostically and limit claims, not policy training.',
            independent_split='24 new textures/truths/offsets with RNG2026101307, frozen recipe before policy testing. No truth used by rescue.',
            verification='All first/repeated outputs checked against frozen CPU, original seed bounds and monotonicity. Float64 oracle allN<=16 or first/middle/last.',
            numerical_tolerance=SCORE_TOLERANCE), fixtures={}, runs=[])
    jobs = []
    for case in selected:
        directory = assets/case['name']
        if case['scope'] == 'local':
            report['fixtures'][case['name']] = prepare_case(case, directory)
        for method in dict.fromkeys(args.methods):
            jobs.append(dict(case=case, method=method, fixture=str(directory), threads=args.threads,
                build=str(frozen if method.startswith('old_') else args.build.resolve()), reference_build=str(frozen),
                repeats=args.repeats, warmup_ms=args.warmup_ms, rescue_config=policy))
    np.random.default_rng(64197).shuffle(jobs)
    report['expected_runs'] = len(jobs)
    for index, job in enumerate(jobs):
        prefix = assets/(job['case']['name']+'-'+job['method'])
        request, response = Path(str(prefix)+'.request.json'), Path(str(prefix)+'.response.json')
        request.write_text(json.dumps(job))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request), '--output', str(response)],
            check=True, timeout=args.timeout)
        report['runs'].append(json.loads(response.read_text()))
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(f"{index+1}/{len(jobs)} {job['case']['name']} {job['method']}", flush=True)
    finish_report(report)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['acceptance_passed']:
        raise SystemExit('Acceptance failed/incomplete: all evidence retained; inspect accuracy and overhead gates')


if __name__ == '__main__':
    main()
