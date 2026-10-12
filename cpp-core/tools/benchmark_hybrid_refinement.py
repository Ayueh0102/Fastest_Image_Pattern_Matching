"""CPU versus Metal-assisted local NCC refinement, with independent fixtures.

This measures supplied local seeds, not whole-image detection. Fresh workers run
serially. Input generation, input loading, context setup, and reference checks
are excluded from algorithm timings. No opaque GPU time is called CPU time.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from benchmark_pose_stress import (ROOT, SCORE_TOLERANCE, cases as heldout_cases,
                                   digest, field, fixture, metadata, peak_rss,
                                   reference_ncc, summary, warm_and_measure)
from pose_experiment import Scorer


CONFIG = dict(max_iterations=30, max_translation=12., max_angle_degrees=15.,
              max_log_scale=math.log(1.3), optimize_scale=True, optimize_angle=True)


def specifications(args):
    if args.suite in ('all', 'accuracy'):
        options = SimpleNamespace(suite='accuracy', profile=args.profile)
        for _, case in heldout_cases(options):
            yield dict(case, kind='heldout', truth_accuracy=True)
    if args.suite in ('all', 'large'):
        configurations = [(256, 3, 32), (768, 1, 4), (768, 3, 32)]
        if args.profile == 'large':
            configurations += [(768, 1, 128), (1024, 3, 32)]
        for size, channels, count in configurations:
            rng = np.random.default_rng(3500+size+channels+count)
            truth = np.asarray([13.125, -9.375, 41.2, math.log(.91)])
            errors = rng.uniform([-5, -5, -8, -.1], [5, 5, 8, .1], (count, 4))
            # At 1024 pixels, the truth's rotated extent is already 1315 px:
            # a 1280-row source clips the object and changes the NCC optimum.
            # Keep complete-support performance cases separate from the explicit
            # boundary regression; larger sources also contain the seed range.
            source_shape = [1280, 1536] if size <= 768 else [2*size, 2*size]
            yield dict(name=f'large_{size}_c{channels}_n{count}', kind='large',
                       size=size, channels=channels, source_shape=source_shape,
                       seed=74291+size+channels, truth=truth.tolist(),
                       initial_poses=(truth+errors).tolist(), truth_accuracy=True)
    if args.suite in ('all', 'regression'):
        truth = np.asarray([3.25, -2.625, 26.35, math.log(.94)])
        poses = (truth + [[1.2, -1.3, 2.5, -.03], [-2.1, 1.7, -4, .055]]).tolist()
        common = dict(size=65, channels=3, source_shape=[160, 192], seed=22719,
                      truth=truth.tolist(), initial_poses=poses, truth_accuracy=False)
        yield dict(common, name='regression_constant', kind='constant', expected_invalid=[0, 1])
        yield dict(common, name='regression_low_contrast', kind='low_contrast')
        yield dict(common, name='regression_partial_tile_65x67', kind='partial_tile',
                   template_shape=[67, 65], truth_accuracy=True)
        boundary = dict(common, source_shape=[96, 112], truth=[40.25, 25.625, 0, 0],
                        initial_poses=[[39, 24, 0, 0], [1e5, 1e5, 0, 0]])
        yield dict(boundary, name='regression_boundary_outside', kind='boundary', expected_invalid=[1])
        yield dict(common, name='regression_frozen_angle_scale', kind='frozen',
                   config=dict(optimize_angle=False, optimize_scale=False))
        rng = np.random.default_rng(88517)
        # 576 tiles * 144 bytes * 256 poses would exceed 20 MiB without
        # chunking. One iteration exercises the shared workspace cap while
        # keeping this a practical regression test rather than a long run.
        offsets = rng.uniform([-1, -1, -1, -.01], [1, 1, 1, .01], (256, 4))
        yield dict(common, name='regression_workspace_cap', kind='workspace_cap',
                   size=768, channels=1, source_shape=[1024, 1280],
                   initial_poses=(truth+offsets).tolist(), config=dict(max_iterations=1))


def prepare(case, destination):
    source, template = fixture(case)
    if case['kind'] == 'large':
        # A speed benchmark with geometric truth must actually contain that
        # support. Keep clipped/occluded inputs in separately labeled regressions.
        theta, scale = np.deg2rad(case['truth'][2]), math.exp(case['truth'][3])
        height, width = template.shape[:2]
        radius_x = scale*(abs(math.cos(theta))*width+abs(math.sin(theta))*height)/2
        radius_y = scale*(abs(math.sin(theta))*width+abs(math.cos(theta))*height)/2
        center_x = source.shape[1]/2+case['truth'][0]
        center_y = source.shape[0]/2-case['truth'][1]
        if (center_x-radius_x < 1 or center_x+radius_x >= source.shape[1]-1 or
                center_y-radius_y < 1 or center_y+radius_y >= source.shape[0]-1):
            raise ValueError('Large performance fixture clips its known template support')
    if case['kind'] == 'constant':
        source.fill(.5)
        template.fill(.5)
    elif case['kind'] == 'low_contrast':
        source = source*.0002 + .5
        template = template*.0002 + .5
    elif case['kind'] == 'partial_tile':
        height, width = case['template_shape']
        x = np.arange(width)[None, :] - width/2
        y = height/2 - np.arange(height)[:, None]
        template = field(x, y, case['size'], case['channels'], case['seed'])
    seeds = np.asarray(case['initial_poses'], np.float32)
    destination.mkdir()
    arrays = dict(source=source, template=template, seeds=seeds)
    for name, array in arrays.items():
        np.save(destination/(name+'.npy'), array, allow_pickle=False)
    return dict(input_sha256={name: digest(array) for name, array in arrays.items()},
                shapes={name: list(array.shape) for name, array in arrays.items()})


def pose_errors(poses, truth, scores):
    truth = np.asarray(truth)
    output = []
    for pose, score in zip(poses, scores):
        position = float(np.linalg.norm(pose[:2]-truth[:2]))
        angle = abs(float((pose[2]-truth[2]+180)%360-180))
        scale = abs(float(np.exp(pose[3]-truth[3])-1))
        output.append(dict(position_px=position, angle_deg=angle, relative_scale=scale,
                           precise_pass=bool(score >= .8 and position <= .5 and angle <= .5 and scale <= .01),
                           existing_gate_pass=bool(score >= .8 and position <= 2 and angle <= 1 and scale <= .02)))
    return output


def worker(job):
    directory = Path(job['fixture_directory'])
    source, template, seeds = [np.load(directory/(name+'.npy'), allow_pickle=False)
                               for name in ('source', 'template', 'seeds')]
    case = job['case']
    config = dict(CONFIG, **case.get('config', {}))
    before = peak_rss()
    with Scorer(job['build'], source, template, job['backend'], job['threads']) as scorer:
        initial_memory = scorer.memory()
        if job['method'] != 'cpu' and not hasattr(scorer.lib, 'fpm_pose_refine_accelerated'):
            return dict(status='unavailable', reason='Accelerated refinement library symbol is unavailable')
        def call():
            if job['method'] == 'cpu':
                return scorer.refine(seeds, **config)
            return scorer.refine_accelerated(seeds, cpu_polish_iterations=job['polish'], **config)
        records, times, warm = warm_and_measure(call, job)
        poses, scores, stats, _ = records[-1]
        metrics = [record[3] for record in records]
        memory = dict(process_peak_before_setup_bytes=before, process_peak_bytes=peak_rss(),
                      **{key: max(row[key] for row in metrics) for key in initial_memory})
        # All correctness checks start after algorithm and memory measurements.
        # CPU re-evaluation independently verifies every reported GPU result.
        with Scorer(job['build'], source, template, 'cpu', job['threads']) as reference:
            initial_scores, _ = reference.score(seeds)
            rechecked, _ = reference.score(poses)
        oracle_indices = [0] if case['kind'] in ('large', 'workspace_cap') else range(len(seeds))
        oracle = [dict(index=i, actual=float(scores[i]),
                       reference=reference_ncc(source, template, poses[i])) for i in oracle_indices]
        oracle_error = max(abs(row['actual']-row['reference']) for row in oracle)
        delta = np.abs(poses-seeds)
        limits = np.asarray([config['max_translation'], config['max_translation'],
                             config['max_angle_degrees'] if config['optimize_angle'] else 0,
                             config['max_log_scale'] if config['optimize_scale'] else 0])
        bounds_ok = bool(np.all(delta <= limits+2e-5))
        frozen_ok = all(np.array_equal(poses[:, index], seeds[:, index])
                        for index in (2, 3) if limits[index] == 0)
        invalid_ok = all(float(scores[i]) == -2 and np.array_equal(poses[i], seeds[i])
                         for i in case.get('expected_invalid', []))
        monotonic = bool(np.all(rechecked >= initial_scores-SCORE_TOLERANCE))
        recheck_error = float(np.max(np.abs(scores-rechecked)))
        auxiliary = memory['gpu_buffer_bytes']-initial_memory['gpu_buffer_bytes']
        # One shared workspace: 16-byte pose, 144-byte derivative statistics per
        # tile, 144-byte merged result. Each section starts on a 256-byte boundary.
        # Source/template residency is excluded from this documented budget.
        tiles = math.ceil(template.shape[0]*template.shape[1]/1024)
        one_pose_bytes = math.ceil((256+tiles*144)/256)*256+144
        auxiliary_limit = max(job['aux_budget_bytes'], one_pose_bytes)
        budget_ok = auxiliary <= auxiliary_limit
        errors = pose_errors(poses, case['truth'], rechecked) if case.get('truth_accuracy') else None
        gpu_times = [m['device_ms'] for m in metrics]
        return dict(device=scorer.device, setup_ms=scorer.setup_ms, **warm,
                    algorithm_wall=summary(times), c_api_wall=summary([m['wall_ms'] for m in metrics]),
                    gpu_command=summary(gpu_times) if all(t >= 0 for t in gpu_times) else None,
                    cpu_solve_ms=None, cpu_polish_ms=None,
                    stage_timing_note='Current ABI exposes total wall and summed GPU commands only; CPU solve/polish are not individually measured.',
                    seeds_per_second=len(seeds)/(np.median(times)/1000), poses=poses.tolist(),
                    scores=scores.tolist(), initial_scores=initial_scores.tolist(), cpu_rechecked_scores=rechecked.tolist(),
                    stats=stats, memory=memory, initial_memory=initial_memory,
                    retained_auxiliary_buffer_bytes=auxiliary, auxiliary_budget_bytes=auxiliary_limit,
                    auxiliary_budget_passed=budget_ok, finite=bool(np.all(np.isfinite(poses)) and np.all(np.isfinite(scores))),
                    bound_checks_passed=bounds_ok, frozen_coordinates_preserved=frozen_ok,
                    invalid_cases_passed=invalid_ok, monotonic=monotonic,
                    cpu_score_recheck_max_error=recheck_error, oracle=oracle, oracle_max_error=oracle_error,
                    accuracy=errors, precise_pass_count=sum(e['precise_pass'] for e in errors) if errors else None,
                    existing_gate_pass_count=sum(e['existing_gate_pass'] for e in errors) if errors else None,
                    seed_count=len(seeds), input_sha256={name: digest(array) for name, array in
                        [('source', source), ('template', template), ('seeds', seeds)]},
                    passed=bool(bounds_ok and frozen_ok and invalid_ok and monotonic and budget_ok and
                                recheck_error <= SCORE_TOLERANCE and oracle_error <= SCORE_TOLERANCE and
                                np.all(np.isfinite(poses)) and np.all(np.isfinite(scores))))


def comparisons(report):
    results = []
    available = [r for r in report['runs'] if r['result'].get('status') != 'unavailable']
    for name in sorted({r['job']['case']['name'] for r in available}):
        selected = [r for r in available if r['job']['case']['name'] == name]
        baseline = next(r['result'] for r in selected if r['job']['method'] == 'cpu')
        ref_poses, ref_scores = np.asarray(baseline['poses']), np.asarray(baseline['cpu_rechecked_scores'])
        for row in selected:
            actual = row['result']
            poses, scores = np.asarray(actual['poses']), np.asarray(actual['cpu_rechecked_scores'])
            position = np.linalg.norm(poses[:, :2]-ref_poses[:, :2], axis=1)
            angle = np.abs((poses[:, 2]-ref_poses[:, 2]+180)%360-180)
            scale = np.abs(np.exp(poses[:, 3]-ref_poses[:, 3])-1)
            same_inputs = actual['input_sha256'] == baseline['input_sha256']
            item = dict(case=name, method=row['job']['method'], backend=row['job']['backend'], polish=row['job']['polish'],
                        identical_inputs=same_inputs, speedup_over_cpu8=baseline['algorithm_wall']['median_ms']/actual['algorithm_wall']['median_ms'],
                        max_position_difference_px=float(position.max()), max_angle_difference_deg=float(angle.max()),
                        max_relative_scale_difference=float(scale.max()), max_absolute_score_difference=float(np.max(np.abs(scores-ref_scores))),
                        worst_score_loss_vs_cpu=float(np.max(ref_scores-scores)),
                        changed_acceptance_count=int(np.sum((scores >= .8) != (ref_scores >= .8))),
                        precise_pass_count=actual['precise_pass_count'], existing_gate_pass_count=actual['existing_gate_pass_count'])
            item['cpu_fallback_exact'] = (bool(np.array_equal(poses, ref_poses) and
                np.array_equal(scores, ref_scores) and actual['stats'] == baseline['stats'])
                if row['job']['method'] == 'cpu_fallback' else None)
            # Different optimizers can select different local maxima. Report
            # pose differences without declaring numerical identity or masking
            # lower NCC / true-pose accuracy behind a parity tolerance.
            item['true_accuracy_regression_count'] = (sum(a['existing_gate_pass'] and not b['existing_gate_pass']
                for a, b in zip(baseline['accuracy'], actual['accuracy'])) if baseline['accuracy'] else None)
            results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT/'build/hybrid-refinement/report.json')
    parser.add_argument('--suite', choices=('all', 'accuracy', 'large', 'regression'), default='all')
    parser.add_argument('--profile', choices=('smoke', 'standard', 'large'), default='standard')
    parser.add_argument('--case', action='append', help='Select names containing any supplied text; may repeat')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--warmup-ms', type=float, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--polish', type=int, nargs='+', default=[0, 2, 4])
    parser.add_argument('--aux-budget-mib', type=float, default=16)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--no-metal', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            result = worker(job)
        except RuntimeError as error:
            if job['backend'] == 'metal' and str(error) in ('No Metal GPU available', 'Metal backend was not compiled on this platform'):
                result = dict(status='unavailable', reason=str(error))
            else:
                raise
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if min(args.repeats, args.warmups, args.threads) < 1 or args.warmup_ms < 0 or args.aux_budget_mib <= 0:
        parser.error('repeats, warmups, threads >=1; warmup-ms >=0; aux-budget-mib >0')
    if any(p < 0 or p > 30 for p in args.polish):
        parser.error('CPU polish must be 0..30 iterations')
    selected = [c for c in specifications(args) if not args.case or any(text in c['name'] for text in args.case)]
    if not selected:
        parser.error('No cases selected')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = metadata(args)
    info['source_sha256'][str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(**info, protocol=dict(
        scope='Local refinement with supplied seeds, not full-image detection or recall.',
        serial=True, fresh_process_per_case_variant=True, fixture_generation='Independent analytic held-out fixtures, cached once per case.',
        repeats=args.repeats, warmups=args.warmups, warmup_ms=args.warmup_ms,
        timing='Algorithm wall includes Python binding, CPU/GPU submission, wait, solves, verification inside C ABI, and requested CPU polish. Excludes input creation/loading/setup and post-run checks.',
        stages='Summed GPU command duration is measured; CPU solve/polish individual timings unavailable. Wall minus GPU is not CPU computation time.',
        memory='Process peak includes Python/NumPy/input loading/setup/algorithm; captured before post-run verification. Metal shares system RAM; do not add GPU allocations to RSS.',
        precision='p95 is an empirical percentile, not a confidence bound. Independent methods may choose different local maxima.',
        accuracy='Accept NCC>=0.8; precise <=0.5px,0.5deg,1%; existing gate <=2px,1deg,2%. Truth is only used for local fixture/seed generation and post-evaluation.',
        variants='CPU portable LM; Metal Jacobian/NCC sufficient statistics + CPU small solve; optional bounded CPU polish. CPU accelerated API fallback tested on regressions.',
        configuration=CONFIG, cuda='Not implemented or measured.'), fixtures={}, runs=[])
    with tempfile.TemporaryDirectory(prefix='hybrid-refinement-') as temporary:
        root = Path(temporary)
        tasks = []
        for case in selected:
            destination = root/case['name']
            report['fixtures'][case['name']] = prepare(case, destination)
            variants = [('cpu', 'cpu', 0)]
            if not args.no_metal:
                variants += [('accelerated', 'metal', polish) for polish in sorted(set(args.polish))]
            if case['kind'] not in ('heldout', 'large'):
                variants.append(('cpu_fallback', 'cpu', 2))
            for method, backend, polish in variants:
                tasks.append(dict(case=case, fixture_directory=str(destination), method=method, backend=backend,
                                  threads=args.threads, polish=polish, build=str(args.build.resolve()),
                                  repeats=args.repeats, warmups=args.warmups, warmup_ms=args.warmup_ms,
                                  aux_budget_bytes=int(args.aux_budget_mib*1024*1024)))
        np.random.default_rng(81793).shuffle(tasks)
        for i, job in enumerate(tasks):
            request, response = root/'request.json', root/'response.json'
            request.write_text(json.dumps(job))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request),
                            '--output', str(response)], check=True, timeout=args.timeout)
            row = json.loads(response.read_text())
            row['job'].pop('fixture_directory')
            report['runs'].append(row)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            result = row['result']
            value = f"{result['algorithm_wall']['median_ms']:.3f} ms" if 'algorithm_wall' in result else result['reason']
            print(f"{i+1}/{len(tasks)} {job['case']['name']} {job['method']} {job['backend']} polish={job['polish']}: {value}", flush=True)
    report['comparisons'] = comparisons(report)
    available = [r for r in report['runs'] if r['result'].get('status') != 'unavailable']
    report['correctness_passed'] = all(r['result']['passed'] for r in available) and all(
        r['identical_inputs'] and r['cpu_fallback_exact'] is not False for r in report['comparisons'])
    report['local_accuracy_nonregression_passed'] = all(
        not r['true_accuracy_regression_count'] for r in report['comparisons'])
    report['solver_score_nonregression_within_1e_minus_4'] = all(
        r['worst_score_loss_vs_cpu'] <= 1e-4 for r in report['comparisons'])
    report['accuracy_summary'] = []
    for method, backend, polish in sorted({(r['job']['method'], r['job']['backend'], r['job']['polish']) for r in available}):
        chosen = [r['result'] for r in available if r['job']['case']['kind'] == 'heldout' and
                  (r['job']['method'], r['job']['backend'], r['job']['polish']) == (method, backend, polish)]
        if chosen:
            report['accuracy_summary'].append(dict(method=method, backend=backend, polish=polish,
                seeds=sum(r['seed_count'] for r in chosen), precise_pass=sum(r['precise_pass_count'] for r in chosen),
                existing_gate_pass=sum(r['existing_gate_pass_count'] for r in chosen)))
    report['completed_at'] = datetime.now().astimezone().isoformat()
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output, flush=True)
    if not report['correctness_passed']:
        raise SystemExit('Correctness checks failed; inspect the report')


if __name__ == '__main__':
    main()
