"""Serial CPU/hybrid/resident NCC comparisons against a frozen baseline binary.

Local supplied-seed accuracy and full-image detection are separate suites. No
truth pose enters the full-image search. All CPU/oracle validation happens after
timing and process-RSS collection. GPU-only and CPU-verified results are distinct.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from benchmark_hybrid_refinement import CONFIG, pose_errors, prepare, specifications as hybrid_cases
from benchmark_pose_stress import ROOT, SCORE_TOLERANCE, digest, metadata, peak_rss, reference_ncc, summary
from pose_experiment import Scorer, library_path


METHODS = ('legacy_cpu', 'legacy_hybrid', 'resident', 'resident_verified')
FULL_METHODS = dict(zip(METHODS, ('continuous_direct_cpu', 'accelerated_direct_metal',
                                'resident_direct_metal', 'resident_verified_direct_metal')))
UNAVAILABLE = ('No Metal GPU available', 'Metal backend was not compiled on this platform')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_cases(args):
    if args.suite not in ('all', 'full'):
        suites = [args.suite]
    else:
        suites = ['accuracy', 'regression', 'large'] if args.suite == 'all' else []
    for suite in suites:
        for case in hybrid_cases(SimpleNamespace(suite=suite, profile=args.profile)):
            yield dict(case, scope='local', split=suite)
    if 'regression' in suites:
        for channels in (1, 3):
            truth = [2.125, -1.375, -43.25, math.log(1.13)]
            common = dict(scope='local', split='regression', size=65, channels=channels,
                          source_shape=[192, 224], seed=80431+channels, truth=truth,
                          initial_poses=(np.asarray(truth)+[[1, -1, 2, -.025], [-1.5, 2, -3, .04]]).tolist(),
                          truth_accuracy=False)
            for kind in ('constant_source', 'constant_template', 'rank_deficient'):
                case = dict(common, kind=kind, name=f'regression_{kind}_c{channels}')
                if kind.startswith('constant_'):
                    case['expected_invalid'] = [0, 1]
                yield case
            yield dict(common, name=f'regression_all_frozen_c{channels}', kind='frozen',
                       config=dict(max_translation=0., max_angle_degrees=0., max_log_scale=0.))


def prepare_local(case, destination):
    info = prepare(case, destination)
    kind = case['kind']
    if kind not in ('constant_source', 'constant_template', 'rank_deficient'):
        return info
    source = np.load(destination/'source.npy', allow_pickle=False)
    template = np.load(destination/'template.npy', allow_pickle=False)
    if kind == 'constant_source':
        source.fill(.5)
    elif kind == 'constant_template':
        template.fill(.5)
    else:
        # Nonconstant NCC, but a one-direction texture cannot identify every
        # geometric degree of freedom. Test finite bounded behavior, not truth.
        for array in (source, template):
            row = .4+.1*np.sin(np.arange(array.shape[1], dtype=np.float32)*.17)
            array[:] = row[None, :] if array.ndim == 2 else row[None, :, None]
    for name, array in (('source', source), ('template', template)):
        np.save(destination/(name+'.npy'), array, allow_pickle=False)
        info['input_sha256'][name] = digest(array)
    return info


def freeze_baseline(build, destination):
    """Copy, never rebuild or overwrite, the explicitly supplied old libraries."""
    destination.mkdir()
    (destination/'experiments').mkdir()
    files = [library_path(build), build/('libfpm_c.dylib' if sys.platform == 'darwin' else 'libfpm_c.so')]
    output = {}
    for source in files:
        target = destination/'experiments'/source.name if source == files[0] else destination/source.name
        shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise RuntimeError('Frozen baseline copy hash mismatch')
        output[str(target)] = sha(target)
    return output


def validate(source, template, seeds, outputs, baseline, config, case, verified):
    """Recheck every returned score from first and measured calls, untimed."""
    initial = np.asarray(seeds, np.float32)
    rows = []
    with Scorer(baseline, source, template, 'cpu', 8) as oracle:
        initial_scores, _ = oracle.score(initial)
        for poses, scores, stats in outputs:
            poses, scores = np.asarray(poses, np.float32), np.asarray(scores, np.float32)
            rescored, _ = oracle.score(poses)
            limits = np.asarray([config['max_translation']]*2 + [
                config['max_angle_degrees'] if config['optimize_angle'] else 0.,
                config['max_log_scale'] if config['optimize_scale'] else 0.])
            bounds = bool(np.all(np.abs(poses.astype(np.float64)-initial) <= limits))
            frozen = all(np.array_equal(poses[:, i], initial[:, i]) for i in range(4) if limits[i] == 0)
            invalid = all(scores[i] == -2 and np.array_equal(poses[i], initial[i])
                          for i in case.get('expected_invalid', []))
            score_error = float(np.max(np.abs(scores-rescored)))
            exact = bool(np.array_equal(scores, rescored))
            monotonic = bool(np.all(rescored >= initial_scores-SCORE_TOLERANCE))
            stable = np.abs(rescored-.8) > SCORE_TOLERANCE
            acceptance_changes = int(np.sum(((scores >= .8) != (rescored >= .8)) & stable))
            finite = bool(np.isfinite(poses).all() and np.isfinite(scores).all())
            iterations = all(0 <= s['iterations'] <= config['max_iterations'] and
                             s['evaluations'] >= 1 and 0 <= s['status'] <= 3 for s in stats)
            accuracy = pose_errors(poses, case['truth'], rescored) if case.get('truth_accuracy') else None
            rows.append(dict(cpu_rechecked_scores=rescored.tolist(), score_max_error=score_error,
                             accuracy=accuracy,
                             exact_cpu_scores=exact, bounds_passed=bounds, frozen_passed=frozen,
                             invalid_passed=invalid, finite=finite, stats_passed=iterations,
                             monotonic_within_tolerance=monotonic,
                             score_regression_count=int(np.sum(rescored < initial_scores-SCORE_TOLERANCE)),
                             threshold_disagreements=acceptance_changes,
                             passed=bool(bounds and frozen and invalid and finite and iterations and
                                         score_error <= SCORE_TOLERANCE and acceptance_changes == 0 and
                                         (not verified or (exact and np.all(rescored >= initial_scores))))))
    poses, scores, _ = outputs[-1]
    indices = list(range(len(initial))) if len(initial) <= 16 else sorted({0, len(initial)//2, len(initial)-1})
    independent = [dict(index=i, actual=float(scores[i]),
                        reference=reference_ncc(source, template, poses[i])) for i in indices]
    error = max(abs(r['actual']-r['reference']) for r in independent)
    true_error = rows[-1]['accuracy']
    return dict(calls=rows, initial_scores=initial_scores.tolist(), oracle=independent,
                oracle_max_error=error, accuracy=true_error,
                precise_pass_count=sum(r['precise_pass'] for r in true_error) if true_error else None,
                seed_count=len(initial), passed=all(r['passed'] for r in rows) and error <= SCORE_TOLERANCE)


def worker(job):
    case, method = job['case'], job['method']
    full = case['scope'] == 'full'
    before = peak_rss()
    if full:
        import cv2
        from real_image_experiments import Search
        cv2.setNumThreads(1)
        # Search.setup_ms excludes decoding; outer constructor wall includes it.
        start = time.perf_counter()
        search = Search(dict(job, method=FULL_METHODS[method], cpu_polish_iterations=0,
                             audit_refinement=True, prune=False))
        constructor_ms = (time.perf_counter()-start)*1000
        scorer, call, close = search.scorer, search.run, search.close
        source, template = search.source_float, search.template.astype(np.float32)/255
        inputs = {name: sha(case[name]) for name in ('source', 'template')}
        setup_ms = search.setup_ms
    else:
        arrays = [np.load(Path(job['fixture'])/(name+'.npy'), allow_pickle=False)
                  for name in ('source', 'template', 'seeds')]
        source, template, seeds = arrays
        before = peak_rss()
        start = time.perf_counter()
        scorer = Scorer(job['build'], source, template, 'cpu' if method == 'legacy_cpu' else 'metal', job['threads'])
        constructor_ms = (time.perf_counter()-start)*1000
        setup_ms, close = scorer.setup_ms, scorer.close
        config = dict(CONFIG, **case.get('config', {}))
        if method == 'legacy_cpu':
            call = lambda: scorer.refine(seeds, **config)
        elif method == 'legacy_hybrid':
            call = lambda: scorer.refine_accelerated(seeds, cpu_polish_iterations=0, **config)
        else:
            call = lambda: scorer.refine_resident(seeds, cpu_verify=method == 'resident_verified', **config)
        inputs = {name: digest(array) for name, array in zip(('source', 'template', 'seeds'), arrays)}
    try:
        initial_memory = scorer.memory()
        memory_snapshots = [initial_memory]
        def one():
            start = time.perf_counter()
            value = call()
            elapsed = (time.perf_counter()-start)*1000
            # Array conversion and diagnostics extraction are outside timing.
            memory_snapshots.append(scorer.memory())
            if full:
                audit = search.last_refinement
                if audit is None:
                    raise RuntimeError('Full-image fixture produced no candidates')
                initial, poses, scores, stats, metrics = audit
                value = dict(value, initial_poses=initial.tolist(), poses=poses.tolist(),
                             scores=scores.tolist(), stats=stats, metrics=metrics)
            else:
                poses, scores, stats, metrics = value
                value = dict(poses=poses.tolist(), scores=scores.tolist(), stats=stats, metrics=metrics)
            return dict(wall_ms=elapsed, result=value)
        first = one()
        start = time.perf_counter()
        warmups = 0
        while warmups < 1 or (time.perf_counter()-start)*1000 < job['warmup_ms']:
            call()
            warmups += 1
        measured = [one() for _ in range(job['repeats'])]
        process_peak = peak_rss()
        memory = {key: max(row[key] for row in memory_snapshots) for key in initial_memory}
        records = [first, *measured]
        metrics = [r['result']['metrics'] for r in measured]
        if full:
            config = dict(CONFIG, optimize_scale=search.allow_scale, optimize_angle=search.rotation)
            seeds = np.asarray(first['result']['initial_poses'], np.float32)
            identical_seeds = all(r['result']['initial_poses'] == first['result']['initial_poses'] for r in measured)
        else:
            identical_seeds = True
        checked = validate(source, template, seeds,
                           [(r['result']['poses'], r['result']['scores'], r['result']['stats']) for r in records],
                           job['reference_build'], config, case, method != 'resident')
        resident = [r['result']['metrics'].get('resident') for r in records]
        diagnostics = None
        if method.startswith('resident'):
            diagnostics = dict(present=all(r is not None for r in resident))
            if diagnostics['present']:
                diagnostics.update(commands_equal_waits=all(r['command_buffers'] == r['cpu_waits'] for r in resident),
                    verify_flag_matches=all(bool(r['cpu_verified']) == (method == 'resident_verified') for r in resident),
                    pure_gpu_has_no_cpu_verification=all(r['cpu_verify_ms'] == 0 for r in resident) if method == 'resident' else True,
                    retained_auxiliary_bytes=max(r['auxiliary_buffer_bytes'] for r in resident),
                    auxiliary_budget_bytes=job['aux_budget_bytes'],
                    auxiliary_budget_passed=all(r['auxiliary_buffer_bytes'] <= job['aux_budget_bytes'] for r in resident))
        result = dict(device=scorer.device, setup_ms=setup_ms, constructor_wall_ms=constructor_ms,
                      cold_constructor_and_first_ms=constructor_ms+first['wall_ms'],
                      cold_setup_and_first_ms=setup_ms+first['wall_ms'], first=first, measured=measured,
                      warmups=warmups, algorithm_wall=summary([r['wall_ms'] for r in measured]),
                      c_api_wall=summary([r['wall_ms'] for r in metrics]),
                      gpu_command=summary([r['device_ms'] for r in metrics]) if all(r['device_ms'] >= 0 for r in metrics) else None,
                      representative_run=sorted(range(len(measured)), key=lambda i: measured[i]['wall_ms'])[len(measured)//2],
                      memory=dict(process_peak_before_setup_bytes=before, process_peak_bytes=process_peak,
                                  max_observed_owned_buffers=memory), initial_memory=initial_memory,
                      resident_diagnostics=diagnostics, input_sha256=inputs, validation=checked,
                      identical_repeat_seeds=identical_seeds,
                      passed=checked['passed'] and identical_seeds and (diagnostics is None or all(
                          diagnostics.get(key, False) for key in ('present', 'commands_equal_waits', 'verify_flag_matches',
                                                                'pure_gpu_has_no_cpu_verification', 'auxiliary_budget_passed'))))
        if full:
            from real_image_experiments import truth_accuracy
            result['accuracy'] = [truth_accuracy(r['result']['matches'], case['truth']) for r in records] if case.get('truth') else None
        return result
    finally:
        close()


def compare(report):
    output = []
    for name in sorted({r['job']['case']['name'] for r in report['runs']}):
        group = [r for r in report['runs'] if r['job']['case']['name'] == name and r['result'].get('status') != 'unavailable']
        baseline = next((r for r in group if r['job']['method'] == 'legacy_cpu'), None)
        if baseline is None:
            continue
        base = baseline['result']
        for row in group:
            r = row['result']
            last, ref = r['measured'][-1]['result'], base['measured'][-1]['result']
            a, b = np.asarray(last['poses']), np.asarray(ref['poses'])
            av, bv = r['validation'], base['validation']
            score_a, score_b = np.asarray(av['calls'][-1]['cpu_rechecked_scores']), np.asarray(bv['calls'][-1]['cpu_rechecked_scores'])
            records = [r['first'], *r['measured']]
            reference_records = [base['first'], *base['measured']]
            same_seeds = len(records) == len(reference_records) and all(
                actual['result'].get('initial_poses') == expected['result'].get('initial_poses')
                for actual, expected in zip(records, reference_records))
            item = dict(case=name, method=row['job']['method'], identical_inputs=r['input_sha256'] == base['input_sha256'],
                        identical_coarse_seeds=same_seeds, cpu_speedup=base['algorithm_wall']['median_ms']/r['algorithm_wall']['median_ms'],
                        precise_pass_count=av['precise_pass_count'], seed_count=av['seed_count'])
            if a.shape == b.shape and same_seeds:
                item.update(max_position_difference_px=float(np.linalg.norm(a[:, :2]-b[:, :2], axis=1).max()),
                    max_angle_difference_deg=float(np.abs((a[:, 2]-b[:, 2]+180)%360-180).max()),
                    max_relative_scale_difference=float(np.abs(np.exp(a[:, 3]-b[:, 3])-1).max()),
                    max_score_difference=float(np.abs(score_a-score_b).max()),
                    worst_score_loss_vs_cpu=float((score_b-score_a).max()))
            if av['accuracy'] is not None:
                item['precise_accuracy_regressions'] = sum(x['precise_pass'] and not y['precise_pass']
                    for cpu_call, actual_call in zip(bv['calls'], av['calls'])
                    for x, y in zip(cpu_call['accuracy'], actual_call['accuracy']))
            if row['job']['case']['scope'] == 'full':
                from real_image_experiments import compare as compare_matches
                item['detection_agreement'] = compare_matches(ref['matches'], last['matches'])
                item['truth_accuracy'] = r['accuracy'][-1] if r['accuracy'] else None
                if r['accuracy']:
                    item['full_accuracy_regressions'] = sum(x['pose_pass'] and not y['pose_pass']
                        for cpu_call, actual_call in zip(base['accuracy'], r['accuracy'])
                        for x, y in zip(cpu_call['details'], actual_call['details']))
                    item['full_false_positive_increases'] = sum(max(0, actual_call['pose_fp']-cpu_call['pose_fp'])
                        for cpu_call, actual_call in zip(base['accuracy'], r['accuracy']))
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--baseline', type=Path, help='Previously frozen old build; never rebuilt by this tool')
    parser.add_argument('--baseline-manifest', type=Path, help='Optional provenance of the old binary; current sources are not attributed to it')
    parser.add_argument('--output', type=Path, default=ROOT/'build/resident-refinement/report.json')
    parser.add_argument('--suite', choices=('accuracy', 'regression', 'large', 'full', 'all'), default='accuracy')
    parser.add_argument('--profile', choices=('smoke', 'standard', 'large'), default='standard')
    parser.add_argument('--case', action='append')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-ms', type=float, default=1000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--aux-budget-mib', type=float, default=16)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            result = worker(job)
        except RuntimeError as error:
            if job['method'] != 'legacy_cpu' and str(error) in UNAVAILABLE:
                result = dict(status='unavailable', reason=str(error))
            else:
                raise
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if args.baseline is None:
        parser.error('--baseline must identify the old frozen build')
    if args.repeats < 1 or args.repeats % 2 != 1 or args.threads < 1 or args.warmup_ms < 0 or args.aux_budget_mib <= 0:
        parser.error('Use positive odd repeats, threads>=1, warmup-ms>=0, aux-budget-mib>0')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assets = args.output.parent/(args.output.stem+'-artifacts-'+str(time.time_ns()))
    assets.mkdir()
    frozen = assets/'baseline-frozen'
    binary_hashes = freeze_baseline(args.baseline.resolve(), frozen)
    for path in (library_path(args.build), args.build/('libfpm_c.dylib' if sys.platform == 'darwin' else 'libfpm_c.so')):
        binary_hashes[str(path.resolve())] = sha(path)
    cases = list(local_cases(args))
    if args.suite in ('all', 'full'):
        from real_image_experiments import controlled_scene, held_out_scene
        for name, generate in (('controlled_scale', controlled_scene), ('held_out_scale', held_out_scene)):
            image = assets/(name+'.png')
            truth = generate(image)
            cases.append(dict(name=name, scope='full', split='full', source=str(image),
                template=str(ROOT/'Test Images/Dst3.bmp'), truth=truth, angle=180,
                limit=70, proposal_score=.5, score=.8, overlap=.2, scales=[.75, 1., 1.25]))
        cases.append(dict(name='src7', scope='full', split='full', source=str(ROOT/'Test Images/Src7.bmp'),
            template=str(ROOT/'Test Images/Dst7.bmp'), angle=180, limit=70, proposal_score=.5,
            score=.5, overlap=.2, scales=[1.]))
    cases = [c for c in cases if not args.case or any(s in c['name'] for s in args.case)]
    if not cases:
        parser.error('No selected cases')
    info = metadata(args)
    files = [Path(__file__).resolve(), ROOT/'cpp-core/tools/benchmark_hybrid_refinement.py',
             ROOT/'cpp-core/tools/real_image_experiments.py', ROOT/'cpp-core/examples/python_ctypes.py']
    files += list((ROOT/'cpp-core/src').glob('*'))
    files += [ROOT/'cpp-core/CMakeLists.txt', ROOT/'cpp-core/tests/pose_resident_test.cpp']
    info['source_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()})
    report = dict(**info, binary_hashes=binary_hashes, baseline_origin=str(args.baseline.resolve()),
        baseline_provenance=json.loads(args.baseline_manifest.read_text()) if args.baseline_manifest else None,
        baseline_provenance_note='Frozen binary hashes are authoritative. Current source hashes describe the candidate/harness only.',
        protocol=dict(serial=True, fresh_worker_per_variant=True, shuffled_order_seed=76317,
            repeats=args.repeats, warmup_ms=args.warmup_ms, cpu_threads=args.threads, cpu_polish_iterations=0,
            local='Supplied perturbed seeds; reused independent analytic held-out fixture suite, not full-image detection.',
            full='Live CPU coarse search each run; truth used only after detection; original Src7 has no calibrated pose truth.',
            timing='Outer call excludes loading/setup/serialization/post-validation. C API wall includes requested final verification. GPU time overlaps waits.',
            cold='Local constructor excludes input loading; full constructor includes decode, setup_ms excludes decode. Driver/shader caches not purged.',
            memory='RSS sampled before independent checks. Owned buffer figures are maximum observed snapshots, not allocation-event peaks; shared GPU memory overlaps RSS.',
            accuracy='Local NCC>=.8 and <=.5px/.5deg/1%; full controlled <=2px/1deg/2%. Degenerate/clipped/frozen/one-iteration cases do not claim identifiable true poses.',
            verification='Every first/measured output rescored on frozen CPU; float64 oracle checks all <=16-pose batches or first/middle/last. Pure GPU may differ and lacks CPU monotonic guarantee.',
            numerical_tolerance=SCORE_TOLERANCE, configuration=CONFIG), fixtures={}, runs=[])
    jobs = []
    for case in cases:
        fixture = assets/case['name']
        if case['scope'] == 'local':
            report['fixtures'][case['name']] = prepare_local(case, fixture)
        for method in dict.fromkeys(args.methods):
            jobs.append(dict(case=case, method=method, fixture=str(fixture), threads=args.threads,
                build=str(frozen if method.startswith('legacy') else args.build.resolve()), reference_build=str(frozen),
                repeats=args.repeats, warmup_ms=args.warmup_ms, aux_budget_bytes=int(args.aux_budget_mib*2**20)))
    np.random.default_rng(76317).shuffle(jobs)
    for index, job in enumerate(jobs):
        prefix = assets/(job['case']['name']+'-'+job['method'])
        request, response = Path(str(prefix)+'.request.json'), Path(str(prefix)+'.response.json')
        request.write_text(json.dumps(job))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request), '--output', str(response)],
                       check=True, timeout=args.timeout)
        row = json.loads(response.read_text())
        report['runs'].append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(f"{index+1}/{len(jobs)} {job['case']['name']} {job['method']}", flush=True)
    report['comparisons'] = compare(report)
    available = [r for r in report['runs'] if r['result'].get('status') != 'unavailable']
    report['unavailable_runs'] = len(report['runs'])-len(available)
    local_truth = [r for r in available if r['result']['validation']['accuracy'] is not None]
    full_truth = [r for r in available if r['job']['case']['scope'] == 'full' and r['result']['accuracy'] is not None]
    report['selection_counts'] = dict(cases=len(cases), methods=len(set(args.methods)), expected_runs=len(jobs),
        completed_runs=len(report['runs']), available_runs=len(available), unavailable_runs=report['unavailable_runs'],
        compared_runs=len(report['comparisons']), local_truth_runs=len(local_truth), full_truth_runs=len(full_truth),
        local_truth_pose_checks=sum(len(call['accuracy']) for r in local_truth for call in r['result']['validation']['calls']),
        full_truth_object_checks=sum(len(call['details']) for r in full_truth for call in r['result']['accuracy']))
    report['correctness_passed'] = bool(available) and all(r['result']['passed'] for r in available)
    # None explicitly means not exercised, never an empty all() claiming success.
    report['local_accuracy_passed'] = all(all(a['precise_pass'] for a in call['accuracy'])
        for r in local_truth for call in r['result']['validation']['calls']) if local_truth else None
    report['full_accuracy_passed'] = all(a['pose_fn'] == 0 and a['pose_fp'] == 0
        for r in full_truth for a in r['result']['accuracy']) if full_truth else None
    report['cross_method_passed'] = bool(report['comparisons']) and len(report['comparisons']) == len(available) and all(
        r['identical_inputs'] and r['identical_coarse_seeds'] and r.get('precise_accuracy_regressions', 0) == 0
        and r.get('full_accuracy_regressions', 0) == 0 and r.get('full_false_positive_increases', 0) == 0
        for r in report['comparisons'])
    report['source_hashes_unchanged'] = all(sha(ROOT/p) == value for p, value in info['source_sha256'].items())
    report['binary_hashes_unchanged'] = all(sha(p) == value for p, value in binary_hashes.items())
    report['complete'] = True
    report['acceptance_passed'] = (report['unavailable_runs'] == 0 and len(available) == len(jobs)
        and all(report[key] for key in ('correctness_passed', 'cross_method_passed', 'source_hashes_unchanged',
                                       'binary_hashes_unchanged'))
        and report['local_accuracy_passed'] is not False and report['full_accuracy_passed'] is not False)
    report['completed_at'] = datetime.now().astimezone().isoformat()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['acceptance_passed']:
        raise SystemExit('Acceptance failed or incomplete: inspect accuracy, CPU comparisons, availability and provenance')


if __name__ == '__main__':
    main()
