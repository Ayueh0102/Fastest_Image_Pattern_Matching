"""Ablate CPU coarse peak selection and GPU SIMD NCC against frozen 010c898.

All routes reuse the same shared source frame, four-angle coarse batches and
512-map fine batches. New frame preparation is inside every timed call. Each
case/method/block gets a fresh process and runs serially. The current build's
resident rescue is shared across all routes, including the frozen baseline.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

import benchmark_shared_fine as shared
import benchmark_pipeline_stages as previous
import coarse_experiment as coarse
import real_image_experiments as real
from benchmark_coarse_ncc import core_library_path, file_hash
from benchmark_pose_experiments import peak_rss
from pose_experiment import library_path as pose_library_path
from rescue_policy import rescue_policy_for_threshold
from summarize_pipeline_stages import load_worker, paired_regression


BASELINE_COMMIT = '010c8982890c4abbbaedb1968b6b1e158c554578'
ROUTES = {'previous_shared': None, 'current_control': (0, 0),
          'peak_only': (1, 0), 'simd_only': (0, 1), 'combined': (1, 1)}
BASE_METHODS = tuple(ROUTES)
METHODS = BASE_METHODS + tuple(name+'_profile' for name in BASE_METHODS[1:])


def verify_baseline(build):
    build = Path(build).resolve()
    path = build/'baseline-provenance.json'
    manifest = json.loads(path.read_text())
    if manifest.get('commit') != BASELINE_COMMIT:
        raise ValueError('Expected frozen 010c898 core/coarse/pose baseline')
    hashes = {}
    for library in (core_library_path(build), coarse.library_path(build), pose_library_path(build)):
        library = Path(library).resolve()
        actual = file_hash(library)
        if manifest.get('files', {}).get(library.relative_to(build).as_posix()) != actual:
            raise ValueError('Frozen baseline differs from manifest: ' + str(library))
        hashes[str(library)] = actual
    return dict(verified=True, commit=BASELINE_COMMIT, manifest=str(path),
                manifest_sha256=file_hash(path), library_hashes=hashes)


class FrameAdapter(shared.FrameAdapter):
    def __init__(self, job, patterns):
        base = job['method'].removesuffix('_profile')
        build = job['baseline'] if base == 'previous_shared' else job['build']
        super().__init__(build, True, patterns, fine_batch_size=job.get('fine_batch_size', 512))
        route = ROUTES[base]
        try:
            if route is not None:
                self.frame.set_coarse_optimization(peak_mode=route[0], ncc_mode=route[1])
                actual = self.frame.coarse_optimization
                if any(actual[key] != value for key, value in zip(('peak_mode', 'ncc_mode'), route)):
                    raise ValueError('coarse optimization setter did not retain the requested route')
            self.config['coarse_optimization'] = dict(peak_mode=route[0] if route else 0,
                ncc_mode=route[1] if route else 0, frozen_api=route is None)
        except Exception:
            self.close()
            raise


class Search(shared.Search):
    """Reuse existing run/refine/NMS/coarse methods with explicit dependencies.

    Patterns and source frame select the requested frozen/current library.
    Rescue always selects job['build']; no process-global function replacement.
    """
    def __init__(self, job):
        self.job = dict(job, audit_refinement=True, prune=False)
        case = job['case']
        self.source = cv2.imread(case['source'], cv2.IMREAD_GRAYSCALE)
        self.template = cv2.imread(case['template'], cv2.IMREAD_GRAYSCALE)
        if self.source is None or self.template is None:
            raise ValueError('missing source/template')
        self.shape = self.template.shape
        self.method = 'resident_verified_direct_metal'
        self.automatic = self.accelerated = False
        self.hybrid = self.continuous = self.resident = case['workflow'] == 'full_rescue'
        self.scales = case['scales']
        self.allow_scale = len(self.scales) > 1 and self.hybrid
        self.rotation = case['angle'] != 0
        self.source_float, self.scorer, self.frame = None, None, None
        self.patterns, self.fourier_cache = [], {}
        self.policy = rescue_policy_for_threshold(case['score']) if self.hybrid else None
        self.prepare_calls, self.last_prepare_wall_ms = 0, 0.
        base = job['method'].removesuffix('_profile')
        pattern_job = dict(job, build=job['baseline'] if base == 'previous_shared' else job['build'],
            method='shared_metal_fine'+('_profile' if job['method'].endswith('_profile') else ''))
        start = time.perf_counter()
        try:
            self.library = real.Library(str(core_library_path(job['baseline'])))
            for scale in self.scales:
                scaled = real.scaled_template(self.template, scale)
                self.patterns.append((scale, shared.PatternAdapter(pattern_job, self.library, real.image_view(scaled))))
            self.config = self.library.config()
            self.config.max_targets = case['limit']
            self.config.min_score = case.get('proposal_score', case['score'])
            self.config.max_overlap = case['overlap']
            self.config.tolerance_angle = case['angle']
            self.config.stop_layer = case['stop_layer']
            self.config.subpixel = case['subpixel']
            self.config.invert_source = case.get('invert_source', 0)
            self.frame = FrameAdapter(job, [pattern.pattern for _, pattern in self.patterns])
            if self.hybrid:
                self.source_float = self.source.astype(np.float32)/255
                self.scorer = shared.RescueScorer(job['build'], self.source_float,
                    self.template.astype(np.float32)/255, self.policy)
            self.setup_ms = (time.perf_counter()-start)*1000
        except Exception:
            self.close()
            raise


profile_checks = shared.profile_checks
fine_checks = shared.fine_checks
make_cases = shared.make_cases


def worker(job):
    cv2.setNumThreads(1)
    start = time.perf_counter()
    search = Search(job)
    construct_ms = (time.perf_counter()-start)*1000
    expected_calls = 0

    def memory():
        return dict(process_peak_rss_bytes=peak_rss(),
            pose_owned_buffers=search.scorer.memory() if search.scorer else None,
            shared_frame=search.frame.snapshot() if search.frame else None)

    def run():
        nonlocal expected_calls
        before = search.prepare_calls
        start = time.perf_counter()
        value = search.run()
        elapsed = (time.perf_counter()-start)*1000
        expected_calls += 1
        errors = []
        if search.frame and (search.prepare_calls-before != 1 or search.prepare_calls != expected_calls):
            errors.append('shared frame was not refreshed exactly once inside each pipeline call')
        record = dict(value, call_wall_ms=elapsed, total_wall_ms=value['algorithm_ms'],
            proposal_wall_ms=value['cpu_search_ms'], frame_prepare_wall_ms=search.last_prepare_wall_ms,
            prepare_calls=search.prepare_calls, proposals=search.last_proposals)
        if search.last_refinement is not None:
            initial, poses, scores, _, metrics = search.last_refinement
            record.update(initial_poses=initial.tolist(), refined_poses=poses.tolist(),
                          refined_scores=scores.tolist(), refinement_metrics=metrics)
        record['patterns'] = [p.snapshot(scale) for scale, p in search.patterns]
        record['frame'] = search.frame.snapshot() if search.frame else None
        if record['frame']:
            metrics = record['frame']['metrics']
            if metrics['generation'] != expected_calls:
                errors.append('frame API generation did not advance for every pipeline call')
            if metrics['models_matched'] != len(search.patterns):
                errors.append('frame model counter differs from the complete scale bank')
        record['profile_errors'] = (errors + profile_checks(record['patterns'], search.frame is not None)
                                    + fine_checks(record['patterns'], record['frame']))
        if job['method'].endswith('_profile') and any(p['profile'] is None for p in record['patterns']):
            record['profile_errors'].append('requested profile missing')
        if record['frame_prepare_wall_ms'] > record['proposal_wall_ms']+1e-6:
            record['profile_errors'].append('frame preparation not enclosed by proposal timer')
        record['output_sha256'] = previous.output_digest(record)
        return record

    try:
        setup_memory = memory()
        first = run()
        first_memory = memory()
        warmups, start = [], time.perf_counter()
        while time.perf_counter()-start < job['warmup_seconds']:
            record = run()
            warmups.append({key: record[key] for key in ('output_sha256', 'profile_errors',
                'total_wall_ms', 'proposal_wall_ms', 'call_wall_ms', 'frame_prepare_wall_ms', 'prepare_calls')})
            warmups[-1]['compact_warmup'] = True
            del record
        runs = [run() for _ in range(job['repeats'])]
        final_memory = memory()
        for record in [first, *runs]:
            if job['case'].get('truth'):
                record['truth_accuracy'] = real.truth_accuracy(record['matches'], job['case']['truth'])
        return dict(case=job['case']['name'], method=job['method'], block=job['block'],
            construct_ms=construct_ms, algorithm_setup_ms=search.setup_ms,
            cold_construct_plus_first_ms=construct_ms+first['total_wall_ms'],
            first=first, warmups=warmups, runs=runs, effective_rescue_policy=search.policy,
            source_shape=list(search.source.shape), template_shape=list(search.template.shape),
            memory=dict(after_construct=setup_memory, after_first=first_memory, after_measurements=final_memory))
    finally:
        search.close()


def summarize_case(case, workers, reference):
    row = previous.summarize_case(case, workers, reference)
    reference_work = work_counts(reference['first'])
    for method, data in row['methods'].items():
        subset = [worker for worker in workers if worker['method'] == method]
        measured = [record for worker in subset for record in worker['runs']]
        data['frame_prepare_wall_ms'] = previous.distribution(
            [record['frame_prepare_wall_ms'] for record in measured])
        data['final_owned_memory'] = [worker['memory']['after_measurements'] for worker in subset]
        data['frame_prepare_metrics'] = [record['frame'] for record in measured]
        data['per_call_fine_metrics'] = [
            [pattern['fine_metrics'] for pattern in record['patterns']] for record in measured]
        data['per_model_nominal_output_buffers'] = data.pop('nominal_output_buffers')
        data['memory_scope'] = 'source pyramid/frame backend retained capacity counted once; per-model counters are diagnostics, not independent allocations'
        data['work_counts_match_reference'] = all(work_counts(record) == reference_work
            for worker in subset for record in [worker['first'], *worker['runs']])
    row['work_counts_match_reference'] = all(data['work_counts_match_reference'] for data in row['methods'].values())
    row['profile_overhead_percent_by_block'] = {}
    for base in BASE_METHODS[1:]:
        if base not in row['methods'] or base+'_profile' not in row['methods']:
            continue
        plain = {b['block']: b for b in row['methods'][base]['blocks']}
        profiled = {b['block']: b for b in row['methods'][base+'_profile']['blocks']}
        row['profile_overhead_percent_by_block'][base] = [dict(block=b,
            total_percent=100*(profiled[b]['total_ms']/plain[b]['total_ms']-1),
            proposal_percent=100*(profiled[b]['proposal_ms']/plain[b]['proposal_ms']-1))
            for b in sorted(plain.keys() & profiled.keys())]
    row['performance'] = {name: {baseline: paired_regression(row['methods'], name, baseline)
        for baseline in ('previous_shared', 'current_control') if baseline != name}
        for name in BASE_METHODS[1:] if name in row['methods']}
    return row


def work_counts(record):
    # Available with profiling disabled on both old and new ABIs. A faster
    # method must perform the same ordered per-model coarse/fine map workload.
    return [dict(scale=item['scale'], coarse_maps=item['metrics']['score_maps'],
                 fine_maps=item['fine_metrics']['maps']) for item in record['patterns']]


def compact_summary(report, report_path, worker_root=None):
    hashes, workers = {}, {}
    for entry in report['execution_order']:
        logical, digest, worker_result = load_worker(entry, report_path, worker_root)
        if logical in hashes:
            raise ValueError('Duplicate archived worker: ' + logical)
        hashes[logical] = digest
        workers.setdefault(entry['case'], []).append(worker_result)
    rows = []
    for entry in report['cases']:
        case_workers = workers[entry['case']['name']]
        reference = next(worker for worker in case_workers if worker['method'] == 'previous_shared')
        measured = [worker for worker in case_workers if worker['block'] >= 0]
        rows.append(summarize_case(entry['case'], measured, reference))
    return dict(report=Path(report_path).name, report_sha256=file_hash(report_path),
        raw_worker_sha256=hashes, complete=report.get('complete'),
        correctness_and_integrity_passed=report.get('passed'), design=report['design'],
        timing='frame prepare counted once within every total/proposal call; shared profile stages exclude separate frame prepare; device/wait/copy diagnostics are nested, never additive',
        memory='one frame host/coarse/fine retained allocation, never sum across models; RSS overlaps retained allocations; high-water includes construction/first/warmup',
        performance_policy='per matching order block warm total median <= reference + max(10% of reference, 2ms); accuracy passing is not a speed claim',
        cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(real.ROOT/'build/fourier-gpu'))
    parser.add_argument('--baseline', default=str(real.ROOT/'build/coarse-next-baseline'))
    parser.add_argument('--cases')
    parser.add_argument('--methods', default=','.join(BASE_METHODS))
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--fine-batch-size', type=int, default=512,
                        help='Fine GPU map batch cap 1..1024 (all routes use the same cap)')
    parser.add_argument('--output', default=str(real.ROOT/'build/coarse-next/report.json'))
    parser.add_argument('--worker')
    parser.add_argument('--summarize', metavar='REPORT', help='Only read saved results; no matching')
    parser.add_argument('--worker-root', help='Authoritative archived workers root for --summarize')
    parser.add_argument('--pilot', action='store_true', help='One order block, one measurement, no warmup; never treat as formal speed evidence')
    args = parser.parse_args()
    if args.pilot:
        args.blocks, args.repeats, args.warmup_seconds = 1, 1, 0.
    if args.worker:
        print(json.dumps(worker(json.loads(Path(args.worker).read_text())), allow_nan=False))
        return
    output = Path(args.output).resolve()
    if args.summarize:
        report_path = Path(args.summarize).resolve()
        if output == report_path:
            parser.error('summary must not overwrite raw report')
        result = compact_summary(json.loads(report_path.read_text()), report_path, args.worker_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        return
    methods = args.methods.split(',')
    if len(methods) != len(set(methods)) or set(methods)-set(METHODS):
        parser.error('choose unique methods from ' + ','.join(METHODS))
    if args.blocks < 1 or args.repeats < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error('positive blocks/repeats and finite nonnegative warmup required')
    if not 1 <= args.fine_batch_size <= 1024:
        parser.error('fine-batch-size must be in 1..1024')
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs, jobs = output.parent/'inputs', output.parent/(output.stem+'-jobs')
    inputs.mkdir(exist_ok=True); jobs.mkdir(exist_ok=True)
    baseline = verify_baseline(args.baseline)
    cases = make_cases(args, inputs)
    source_hashes = previous.source_manifest()
    for path in (Path(__file__).resolve(), Path(shared.__file__).resolve(),
                 Path(__file__).with_name('summarize_pipeline_stages.py').resolve(),
                 real.ROOT/'cpp-core/docs/coarse-next/protocol.json'):
        source_hashes[str(path)] = file_hash(path)
    library_hashes = dict(baseline['library_hashes'])
    for path in (core_library_path(args.build), coarse.library_path(args.build), pose_library_path(args.build)):
        library_hashes[str(path)] = file_hash(path)
    input_hashes = {case[key]: file_hash(case[key]) for case in cases for key in ('source', 'template')}
    report = dict(created=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        design=dict(methods=methods, blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup_seconds,
                    pyramid_mode=1, batch_angles=4, fine_batch_size=args.fine_batch_size,
                    shared_prepare_every_call=True, pilot=args.pilot,
                    routes={name: dict(peak_mode=mode[0], ncc_mode=mode[1]) if mode else 'frozen API'
                            for name, mode in ROUTES.items()}),
        baseline_provenance=baseline, source_hashes=source_hashes, library_hashes=library_hashes,
        input_hashes=input_hashes, execution_order=[], cases=[], complete=False,
        scope='online full source search; prepare is timed once per frame every call; shared source reuse is across scales only, never across changing frames; same current resident rescue for real scenes; 8K/small strict cases are core-only',
        accuracy='ordered proposals plus final matches/refined poses/scores exact vs frozen 010c898 shared frame; coarse/fine map counts unchanged; GT evaluated after time/RSS; no truth supplied to algorithms; warmups retain only digest/time/errors',
        timing='forward/reverse fresh-process order blocks; each method first, warmup, measured; setup excludes from total, prepare included in total; profile on/off independent processes, not simultaneous')

    def run_worker(case, method, block, reference=False):
        job = dict(case=case, method=method, block=block, build=str(Path(args.build).resolve()),
                   baseline=str(Path(args.baseline).resolve()), repeats=1 if reference else args.repeats,
                   warmup_seconds=0 if reference else args.warmup_seconds,
                   fine_batch_size=args.fine_batch_size)
        label = f'{case["name"]}-{method}-b{block}'
        path = jobs/(label+'.json')
        path.write_text(json.dumps(job, indent=2))
        print('Running '+label, flush=True)
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                               capture_output=True, text=True)
        (jobs/(label+'.stderr.txt')).write_text(child.stderr)
        if child.returncode:
            raise RuntimeError(f'{label} failed ({child.returncode}): {child.stderr[-4000:]}')
        result = json.loads(child.stdout)
        result_path = jobs/(label+'.result.json')
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        report['execution_order'].append(dict(case=case['name'], method=method, block=block,
            reference_only=reference, job=str(path), job_sha256=file_hash(path),
            result=str(result_path), result_sha256=file_hash(result_path)))
        return result

    for case in cases:
        results, reference = [], None
        if 'previous_shared' not in methods:
            reference = run_worker(case, 'previous_shared', -1, True)
        for block in range(args.blocks):
            for method in methods if block % 2 == 0 else methods[::-1]:
                result = run_worker(case, method, block)
                results.append(result)
                if method == 'previous_shared' and reference is None:
                    reference = result
        row = summarize_case(case, results, reference)
        report['cases'].append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'{case["name"]}: exact={row["exact_all_outputs"]}, GT={row["truth_passed"]}, profile={row["profile_valid"]}', flush=True)
    all_hashes = {**source_hashes, **library_hashes, **input_hashes, baseline['manifest']: baseline['manifest_sha256']}
    report['sources_unchanged'] = all(file_hash(path) == digest for path, digest in all_hashes.items())
    report['complete'] = True
    report['passed'] = report['sources_unchanged'] and all(row['exact_all_outputs'] and row['profile_valid']
        and row['work_counts_match_reference'] and row['truth_passed'] is not False for row in report['cases'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
