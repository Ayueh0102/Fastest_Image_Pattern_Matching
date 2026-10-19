"""Serial CPU/Metal crossover calibration and held-out automatic routing checks.

Inputs are known local registration seeds, not full-image detection results.
Every method/case has a fresh worker; generation and reference checks are outside
timing. Cold figures include scorer construction and its first refinement, but
do not purge the operating system's driver/shader caches.
"""
import argparse
import ctypes as C
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from benchmark_hybrid_refinement import CONFIG, pose_errors, prepare
from benchmark_pose_stress import (ROOT, SCORE_TOLERANCE, digest, metadata,
                                   peak_rss, reference_ncc, summary, warm_and_measure)
from pose_experiment import AutoConfig, Scorer, library_path


def specifications(suite):
    for split, sizes, counts, seed_base in (
            ('calibration', (96, 192, 384, 768), (1, 8, 32), 610001),
            ('heldout', (160, 256, 512), (4, 16), 810013)):
        if suite not in ('all', split):
            continue
        for size in sizes:
            for channels in (1, 3):
                texture_seed = seed_base + size*7 + channels*97
                rng = np.random.default_rng(texture_seed)
                truth = np.asarray([7.125, -5.375, 41.2, math.log(.91)])
                if split == 'heldout':
                    truth = rng.uniform([-9, -9, -175, math.log(.85)],
                                        [9, 9, 175, math.log(1.15)])
                offsets = rng.uniform([-4, -4, -7, -.09], [4, 4, 7, .09], (max(counts), 4))
                for count in counts:
                    # Counts within a texture share identical images and a seed
                    # prefix, so the batch dimension is not confounded by texture.
                    source_size = max(512, 2*size)
                    yield dict(name=f'{split}_s{size:04d}_c{channels}_n{count:03d}',
                               split=split, kind='large', size=size, channels=channels,
                               source_shape=[source_size, source_size], seed=texture_seed,
                               truth=truth.tolist(), initial_poses=(truth+offsets[:count]).tolist(),
                               truth_accuracy=True)
    if suite in ('all', 'edge'):
        for channels in (1, 3):
            truth = np.asarray([7.125, -5.375, 41.2, math.log(.91)])
            rng = np.random.default_rng(94017+channels)
            offsets = rng.uniform([-2, -2, -3, -.04], [2, 2, 3, .04], (16, 4))
            common = dict(split='edge', kind='large', size=384, channels=channels,
                          source_shape=[768, 768], seed=94017+channels, truth=truth.tolist(), truth_accuracy=True)
            yield dict(common, name=f'edge_constant_c{channels}', kind='constant', truth_accuracy=False,
                       initial_poses=(truth+offsets).tolist(), expected_invalid=list(range(16)))
            yield dict(common, name=f'edge_near_perfect_c{channels}',
                       initial_poses=(truth+offsets*.0001).tolist())
            frozen = offsets.copy()
            frozen[:, 2:] = 0
            yield dict(common, name=f'edge_frozen_c{channels}', initial_poses=(truth+frozen).tolist(),
                       config=dict(optimize_angle=False, optimize_scale=False))
            yield dict(common, name=f'edge_one_iteration_c{channels}', initial_poses=(truth+offsets).tolist(),
                       config=dict(max_iterations=1))


def call_refiner(scorer, method, seeds, routing, config):
    if method == 'cpu':
        return scorer.refine(seeds, **config)
    if method == 'metal':
        return scorer.refine_accelerated(seeds, cpu_polish_iterations=0, **config)
    return scorer.refine_auto(seeds, **routing, **config)


def worker(job):
    if job['method'] == 'auto' and not hasattr(Scorer, 'refine_auto'):
        return dict(status='unavailable', reason='Scorer.refine_auto is not available')
    source, template, seeds = [np.load(Path(job['fixture_directory'])/(name+'.npy'), allow_pickle=False)
                               for name in ('source', 'template', 'seeds')]
    before = peak_rss()
    config = dict(CONFIG, **job['case'].get('config', {}))
    started = time.perf_counter()
    with Scorer(job['build'], source, template, job['method'], job['threads']) as scorer:
        constructor_ms = (time.perf_counter()-started)*1000
        creation_device = scorer.device
        initial_memory = scorer.memory()
        defaults = None
        if hasattr(scorer.lib, 'fpm_pose_auto_default_config'):
            policy = AutoConfig()
            scorer.lib.fpm_pose_auto_default_config(C.byref(policy))
            defaults = {key: getattr(policy, key) for key, _ in AutoConfig._fields_}
        effective = dict(defaults, **job['routing']) if defaults is not None else None
        first = None

        def call():
            nonlocal first
            record = call_refiner(scorer, job['method'], seeds, job['routing'], config)
            if first is None:
                first = record
            return record

        records, times, warm = warm_and_measure(call, job)
        poses, scores, stats, final_metrics = records[-1]
        metrics = [r[3] for r in records]
        after_memory = scorer.memory()
        # Freeze memory measurements before extra oracle contexts or arrays.
        memory = dict(process_peak_before_setup_bytes=before, process_peak_bytes=peak_rss(),
                      **{key: max([after_memory[key]]+[row[key] for row in metrics])
                         for key in after_memory})
        first_route = first[3].get('route')
        measured_routes = [row.get('route') for row in metrics]
        selected = (final_metrics['route']['selected_backend'] if job['method'] == 'auto'
                    else int(job['method'] == 'metal'))
        forced = None
        with Scorer(job['build'], source, template, 'cpu', job['threads']) as reference:
            initial_scores, _ = reference.score(seeds)
            checked_scores, _ = reference.score(poses)
            if job['method'] == 'auto' and selected == 0:
                forced = reference.refine(seeds, **config)
        if job['method'] == 'auto' and selected == 1:
            with Scorer(job['build'], source, template, 'metal', job['threads']) as reference:
                forced = reference.refine_accelerated(seeds, cpu_polish_iterations=effective['cpu_polish_iterations'], **config)
        exact_selected = (bool(np.array_equal(poses, forced[0]) and np.array_equal(scores, forced[1]) and
                               stats == forced[2]) if forced is not None else None)
        oracle_indices = sorted(set((0, len(seeds)-1)))
        oracle = [dict(index=i, actual=float(scores[i]),
                       reference=reference_ncc(source, template, poses[i])) for i in oracle_indices]
        oracle_error = max(abs(row['actual']-row['reference']) for row in oracle)
        score_error = float(np.max(np.abs(scores-checked_scores)))
        monotonic = bool(np.all(checked_scores >= initial_scores-SCORE_TOLERANCE))
        limits = np.asarray([config['max_translation'], config['max_translation'],
                             config['max_angle_degrees'] if config['optimize_angle'] else 0,
                             config['max_log_scale'] if config['optimize_scale'] else 0])
        bounds_ok = bool(np.all(np.abs(poses-seeds) <= limits+2e-5))
        frozen_ok = all(np.array_equal(poses[:, index], seeds[:, index]) for index in (2, 3) if limits[index] == 0)
        invalid_ok = all(scores[i] == -2 and np.array_equal(poses[i], seeds[i])
                         for i in job['case'].get('expected_invalid', []))
        finite = bool(np.all(np.isfinite(poses)) and np.all(np.isfinite(scores)))
        errors = pose_errors(poses, job['case']['truth'], checked_scores) if job['case']['truth_accuracy'] else None
        stable_route = (all(route is not None and route['selected_backend'] == selected
                            for route in [first_route]+measured_routes) if job['method'] == 'auto' else True)
        gpu_times = [m['device_ms'] for m in metrics]
        return dict(device_at_creation=creation_device, selected_backend=selected,
                    loaded_policy_defaults=defaults, effective_auto_policy=effective,
                    constructor_wall_ms=constructor_ms, setup_ms=scorer.setup_ms, **warm,
                    cold_constructor_and_first_call_ms=constructor_ms+warm['first_call_wall_ms'],
                    first_metrics=first[3], first_route=first_route, measured_routes=measured_routes,
                    algorithm_wall=summary(times), c_api_wall=summary([m['wall_ms'] for m in metrics]),
                    gpu_command=summary(gpu_times) if all(value >= 0 for value in gpu_times) else None,
                    poses=poses.tolist(), scores=scores.tolist(), stats=stats,
                    initial_scores=initial_scores.tolist(), cpu_rechecked_scores=checked_scores.tolist(),
                    memory=memory, initial_memory=initial_memory,
                    input_sha256={name: digest(array) for name, array in
                                  [('source', source), ('template', template), ('seeds', seeds)]},
                    oracle=oracle, oracle_max_error=oracle_error, cpu_score_recheck_max_error=score_error,
                    monotonic=monotonic, finite=finite, bounds_passed=bounds_ok,
                    frozen_coordinates_preserved=frozen_ok, invalid_cases_passed=invalid_ok,
                    stable_selected_backend=stable_route, exact_selected_forced_result=exact_selected,
                    accuracy=errors, precise_pass_count=sum(e['precise_pass'] for e in errors) if errors else None,
                    existing_gate_pass_count=sum(e['existing_gate_pass'] for e in errors) if errors else None, seed_count=len(seeds),
                    passed=bool(finite and monotonic and bounds_ok and frozen_ok and invalid_ok and stable_route and
                                exact_selected is not False and score_error <= SCORE_TOLERANCE and
                                oracle_error <= SCORE_TOLERANCE))


def compare(report):
    output = []
    available = [r for r in report['runs'] if r['result'].get('status') != 'unavailable']
    for name in sorted({r['job']['case']['name'] for r in available}):
        rows = {r['job']['method']: r for r in available if r['job']['case']['name'] == name}
        case = next(iter(rows.values()))['job']['case']
        hashes = [r['result']['input_sha256'] for r in rows.values()]
        same_inputs = all(value == hashes[0] for value in hashes)
        item = dict(case=name, split=case['split'], template_size=case['size'], channels=case['channels'],
                    count=len(case['initial_poses']), pose_samples=case['size']**2*case['channels'],
                    batch_samples=case['size']**2*case['channels']*len(case['initial_poses']),
                    identical_inputs=same_inputs)
        if 'cpu' in rows and 'metal' in rows:
            cpu, metal = rows['cpu']['result'], rows['metal']['result']
            cpu_ms, metal_ms = cpu['algorithm_wall']['median_ms'], metal['algorithm_wall']['median_ms']
            item.update(cpu_ms=cpu_ms, metal_ms=metal_ms, cpu_over_metal=cpu_ms/metal_ms,
                        fastest_measured_backend='cpu' if cpu_ms <= metal_ms else 'metal',
                        cpu_cold_ms=cpu['cold_constructor_and_first_call_ms'],
                        metal_cold_ms=metal['cold_constructor_and_first_call_ms'])
            cpu_pose, metal_pose = np.asarray(cpu['poses']), np.asarray(metal['poses'])
            cpu_score, metal_score = np.asarray(cpu['cpu_rechecked_scores']), np.asarray(metal['cpu_rechecked_scores'])
            item.update(max_position_difference_px=float(np.linalg.norm(cpu_pose[:, :2]-metal_pose[:, :2], axis=1).max()),
                        max_angle_difference_deg=float(np.abs((cpu_pose[:, 2]-metal_pose[:, 2]+180)%360-180).max()),
                        max_relative_scale_difference=float(np.abs(np.exp(cpu_pose[:, 3]-metal_pose[:, 3])-1).max()),
                        max_absolute_score_difference=float(np.abs(cpu_score-metal_score).max()),
                        max_metal_score_loss=float((cpu_score-metal_score).max()),
                        true_accuracy_regression_count=(sum(a['precise_pass'] and not b['precise_pass']
                            for a, b in zip(cpu['accuracy'], metal['accuracy'])) if cpu['accuracy'] else None))
        if 'auto' in rows:
            auto = rows['auto']['result']
            chosen = 'metal' if auto['selected_backend'] else 'cpu'
            item.update(auto_ms=auto['algorithm_wall']['median_ms'], auto_selected=chosen,
                        auto_cold_ms=auto['cold_constructor_and_first_call_ms'],
                        first_route=auto['first_route'], exact_selected_forced_result=auto['exact_selected_forced_result'])
            if chosen in rows and (chosen == 'cpu' or auto['effective_auto_policy']['cpu_polish_iterations'] == 0):
                forced = rows[chosen]['result']
                item['auto_exact_cross_worker_result'] = bool(
                    np.array_equal(auto['poses'], forced['poses']) and np.array_equal(auto['scores'], forced['scores']) and
                    auto['stats'] == forced['stats'])
                item['auto_over_selected_forced_warm_ratio'] = item['auto_ms']/forced['algorithm_wall']['median_ms']
            if 'cpu_ms' in item:
                item['auto_over_fastest_measured_warm_ratio'] = item['auto_ms']/min(item['cpu_ms'], item['metal_ms'])
        output.append(item)
    return output


def provenance(args):
    info = metadata(args)
    for path in (Path(__file__).resolve(), ROOT/'cpp-core/tools/benchmark_hybrid_refinement.py'):
        info['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT/'build/pose-routing/report.json')
    parser.add_argument('--suite', choices=('calibration', 'heldout', 'edge', 'all'), default='calibration')
    parser.add_argument('--case', action='append', help='Select names containing any supplied text; may repeat')
    parser.add_argument('--methods', nargs='+', choices=('cpu', 'metal', 'auto'), default=['cpu', 'metal'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--warmup-ms', type=float, default=200)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--min-pose-samples', type=int, help='Override C++ default only when explicitly supplied')
    parser.add_argument('--min-batch-samples', type=int, help='Override C++ default only when explicitly supplied')
    parser.add_argument('--min-iterations', type=int, help='Override C++ default only when explicitly supplied')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            result = worker(job)
        except RuntimeError as error:
            if job['method'] == 'metal' and str(error) in ('No Metal GPU available', 'Metal backend was not compiled on this platform'):
                result = dict(status='unavailable', reason=str(error))
            else:
                raise
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if min(args.repeats, args.warmups, args.threads) < 1 or args.warmup_ms < 0:
        parser.error('repeats, warmups, threads >=1; warmup-ms >=0')
    if any(value is not None and not 0 <= value < 2**64 for value in (args.min_pose_samples, args.min_batch_samples)):
        parser.error('Routing sample thresholds must fit uint64')
    if args.min_iterations is not None and not 1 <= args.min_iterations <= 200:
        parser.error('min-iterations must be 1..200')
    chosen_cases = [c for c in specifications(args.suite)
                    if not args.case or any(text in c['name'] for text in args.case)]
    if not chosen_cases:
        parser.error('No cases selected')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    routing = {key: value for key, value in dict(min_pose_samples=args.min_pose_samples,
        min_batch_samples=args.min_batch_samples, min_iterations=args.min_iterations).items() if value is not None}
    report = dict(**provenance(args), suite=args.suite, routing=routing, protocol=dict(
        serial=True, fresh_process_per_case_method=True, shuffled_order_seed=29413,
        scope='Local refinement at supplied seeds; not full-image detection or recall.',
        fixtures='Continuous analytic fields; distinct RGB channels; complete true-pose support; identical cached inputs across methods. Same texture and seed prefix across counts.',
        calibration='24 cases: sizes96/192/384/768, channels1/3, counts1/8/32.',
        heldout='12 cases: sizes160/256/512, channels1/3, counts4/16; independent texture seeds and truth poses.',
        edge='8 cases: gray/RGB near-perfect, constant, frozen angle/scale, one iteration. Constant cases have no geometric accuracy; an iteration-limited solve is not expected to meet every final accuracy gate.',
        repeats=args.repeats, warmups=args.warmups, warmup_ms=args.warmup_ms, configuration=CONFIG,
        algorithm='CPU portable refinement versus Metal statistics plus CPU solve/verification, forced Metal polish0. Auto loads C++ policy defaults, optionally overriding explicitly supplied CLI fields.',
        warm_timing='Outer Python call includes routing, C ABI, dispatch/wait, solve and CPU final verification; excludes fixture/loading, constructor and post-run validation.',
        cold_timing='Constructor wall plus first refinement wall, including lazy Metal setup. Fresh process; driver/shader caches are not purged. Input loading and Python imports excluded.',
        memory='Peak RSS captured before extra reference contexts, includes Python/input loading/setup/algorithm. Owned buffers reported separately; Metal shared RAM overlaps RSS.',
        verification='CPU rechecks every score, independent float64 oracle checks first/last poses. Auto must exactly match selected forced backend including pose, score, stats; checks are outside timing.',
        accuracy='NCC>=0.8 and <=0.5px/0.5deg/1% relative scale. Local true accuracy is separate from numerical correctness.',
        comparison='Fastest measured backend and ratios are empirical per-case medians, not guarantees or confidence intervals.',
        cuda='Not implemented or measured.'), fixtures={}, runs=[])
    with tempfile.TemporaryDirectory(prefix='pose-routing-') as temporary:
        root = Path(temporary)
        tasks = []
        for case in chosen_cases:
            destination = root/case['name']
            report['fixtures'][case['name']] = prepare(case, destination)
            for method in dict.fromkeys(args.methods):
                tasks.append(dict(case=case, method=method, routing=routing, threads=args.threads,
                                  fixture_directory=str(destination), build=str(args.build.resolve()),
                                  repeats=args.repeats, warmups=args.warmups, warmup_ms=args.warmup_ms))
        np.random.default_rng(29413).shuffle(tasks)
        for index, job in enumerate(tasks):
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
            print(f"{index+1}/{len(tasks)} {job['case']['name']} {job['method']}: {value}", flush=True)
    report['comparisons'] = compare(report)
    available = [r for r in report['runs'] if r['result'].get('status') != 'unavailable']
    report['unavailable_runs'] = len(report['runs'])-len(available)
    report['correctness_passed'] = bool(available) and all(r['result']['passed'] for r in available) and all(
        c['identical_inputs'] and c.get('auto_exact_cross_worker_result') is not False for c in report['comparisons'])
    report['accuracy_passed'] = bool(available) and all(r['result']['precise_pass_count'] == r['result']['seed_count']
        for r in available if r['result']['accuracy'] is not None)
    report['loaded_policy_defaults'] = [json.loads(value) for value in sorted({json.dumps(
        r['result']['loaded_policy_defaults'], sort_keys=True) for r in available})]
    report['source_hashes_unchanged_during_run'] = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value
                                                    for name, value in report['source_sha256'].items())
    report['binary_hash_unchanged_during_run'] = hashlib.sha256(library_path(args.build).read_bytes()).hexdigest() == report['binary_sha256']
    report['complete'] = True
    report['completed_at'] = datetime.now().astimezone().isoformat()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output, flush=True)
    if not all(report[key] for key in ('correctness_passed', 'source_hashes_unchanged_during_run', 'binary_hash_unchanged_during_run')):
        raise SystemExit('Correctness or run provenance failed; inspect the report')


if __name__ == '__main__':
    main()
