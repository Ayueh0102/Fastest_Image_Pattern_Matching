"""Profile the existing CPU/Metal search before choosing another GPU stage.

Workers run serially in fresh processes. Reversing method order between blocks
reduces fixed-order drift; it does not remove thermal/frequency uncertainty.
Real-image cases run the same resident rescue after proposal search. Large
canvases test the complete core matcher without allocating a full-image pose
scorer. Truth is evaluated only after timed calls and memory snapshots.
"""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2

import real_image_experiments as real
from benchmark_coarse_ncc import cases_for, core_library_path, file_hash
from benchmark_pose_experiments import peak_rss
from coarse_experiment import CoarseMetrics, CoarsePattern, library_path as coarse_library_path
from pose_experiment import Scorer, library_path as pose_library_path
from python_ctypes import Config, Image, Result
from rescue_policy import rescue_policy_for_threshold


BASELINE_COMMIT = '91842dd8a94d7278a221bd177ee964dc98d0db46'
DEFAULT_METHODS = ('frozen_cpu', 'frozen_metal', 'cpu', 'cpu_profile', 'metal', 'metal_profile')
ACCELERATED_METHODS = ('pyramid_direct', 'pyramid_separable', 'batched', 'combined')
METHODS = DEFAULT_METHODS + tuple(name for base in ACCELERATED_METHODS for name in (base, base+'_profile'))
EXCLUSIVE_STAGES = ('source_copy_invert_ms', 'pyramid_ms', 'coarse_rotation_ms',
    'coarse_ncc_ms', 'peak_selection_ms', 'fine_rotation_ms', 'fine_ncc_ms',
    'fine_control_ms', 'nms_ms', 'unclassified_ms')


def verify_baseline(build):
    build = Path(build).resolve()
    manifest_path = build / 'baseline-provenance.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('commit') != BASELINE_COMMIT:
        raise ValueError('Expected the frozen 91842dd baseline manifest')
    required = (core_library_path(build), coarse_library_path(build), pose_library_path(build))
    hashes = {}
    for path in required:
        path = Path(path).resolve()
        relative = path.relative_to(build).as_posix()
        actual = file_hash(path)
        if manifest.get('files', {}).get(relative) != actual:
            raise ValueError('Frozen baseline library differs from manifest: ' + relative)
        hashes[str(path)] = actual
    return dict(verified=True, commit=BASELINE_COMMIT, manifest=str(manifest_path),
                manifest_sha256=file_hash(manifest_path), library_hashes=hashes)


class FrozenMetalPattern:
    """Bind only the ABI available in 91842dd, without newer profile symbols."""
    def __init__(self, build, image, min_area):
        self.dll = C.CDLL(str(coarse_library_path(build)))
        self.handle, self.metrics = C.c_void_p(), CoarseMetrics()
        self.error = C.create_string_buffer(1024)
        self.dll.fpm_coarse_create.argtypes = [Image, C.c_int, C.POINTER(C.c_void_p),
            C.POINTER(CoarseMetrics), C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_create.restype = C.c_int
        self.dll.fpm_coarse_destroy.argtypes = [C.c_void_p]
        self.dll.fpm_coarse_destroy.restype = None
        self.dll.fpm_coarse_match.argtypes = [C.c_void_p, Image, C.POINTER(Config),
            C.POINTER(Result), C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(CoarseMetrics),
            C.POINTER(C.c_char), C.c_size_t]
        self.dll.fpm_coarse_match.restype = C.c_int
        self.check(self.dll.fpm_coarse_create(image, min_area, C.byref(self.handle),
            C.byref(self.metrics), self.error, len(self.error)))
        self.setup_metrics = self.last_metrics

    def check(self, status):
        if status:
            raise RuntimeError(self.error.value.decode('utf-8', errors='replace'))

    @property
    def last_metrics(self):
        return {key: getattr(self.metrics, key) for key, _ in self.metrics._fields_}

    def match(self, image, config, *, prune=False):
        del prune
        results, count = (Result * config.max_targets)(), C.c_size_t()
        self.check(self.dll.fpm_coarse_match(self.handle, image, C.byref(config), results,
            len(results), C.byref(count), C.byref(self.metrics), self.error, len(self.error)))
        return [dict(index=i, score=r.score, angle=r.angle, pos_x=r.center.x, pos_y=r.center.y,
                     corners=[[p.x, p.y] for p in r.corners]) for i, r in enumerate(results[:count.value])]

    def close(self):
        if self.handle:
            self.dll.fpm_coarse_destroy(self.handle)
            self.handle = C.c_void_p()


class PatternAdapter:
    def __init__(self, job, library, image, min_area):
        self.method = job['method']
        if self.method == 'frozen_cpu':
            self.pattern = library.learn(image, min_area)
        elif self.method == 'frozen_metal':
            self.pattern = FrozenMetalPattern(job['baseline'], image, min_area)
        else:
            base = self.method.removesuffix('_profile')
            pyramid_mode = (1 if base == 'pyramid_direct' else 2 if base == 'pyramid_separable'
                            else job['combined_pyramid_mode'] if base == 'combined' else 0)
            batch_angles = 4 if base in ('batched', 'combined') else 1
            self.pattern = CoarsePattern(job['build'], image, min_area,
                backend='cpu' if self.method.startswith('cpu') else 'metal',
                pyramid_mode=pyramid_mode, batch_angles=batch_angles)

    def match(self, image, config, *, prune=False):
        if self.method.startswith('frozen_'):
            return self.pattern.match(image, config, prune=prune)
        return self.pattern.match(image, config, prune=prune, profile=self.method.endswith('_profile'))

    def snapshot(self, scale, setup=False):
        return dict(scale=scale,
            metrics=getattr(self.pattern, 'setup_metrics' if setup else 'last_metrics', None),
            profile=None if setup else getattr(self.pattern, 'last_profile', None),
            execution_config=getattr(self.pattern, 'execution_config', None),
            execution_metrics=getattr(self.pattern, 'last_execution_metrics', None))

    def close(self):
        self.pattern.close()


def profile_checks(patterns):
    errors = []
    for item in patterns:
        profile = item['profile']
        if profile is None:
            continue
        if profile['version'] != 1:
            errors.append('unexpected profile version')
        names = (*EXCLUSIVE_STAGES, 'core_wall_ms', 'api_wall_ms', 'api_residual_ms')
        if any(not math.isfinite(profile[key]) or profile[key] < -1e-6 for key in names):
            errors.append('nonfinite/negative profile time')
        residual = sum(profile[key] for key in EXCLUSIVE_STAGES) - profile['core_wall_ms']
        if abs(residual) > max(1e-5, profile['core_wall_ms']*1e-9):
            errors.append('exclusive stage sum differs from core wall')
        residual = profile['core_wall_ms'] + profile['api_residual_ms'] - profile['api_wall_ms']
        if abs(residual) > max(1e-5, profile['api_wall_ms']*1e-9):
            errors.append('API residual does not reconcile')
        if profile['backend'] == 1 and item['metrics']['score_maps'] != profile['coarse_maps']:
            errors.append('Metal score map count differs from core count')
        execution = item.get('execution_metrics')
        config = item.get('execution_config')
        if execution and config:
            if any(execution[key] != config[key] for key in ('pyramid_mode', 'batch_angles')):
                errors.append('execution metrics differ from configured route')
            if execution['max_batch_maps'] > config['batch_angles']:
                errors.append('actual angle batch exceeds configured cap')
            if config['pyramid_mode'] and profile['pyramid_levels'] > 1 and execution['pyramid_command_buffers'] < 1:
                errors.append('GPU pyramid requested but no command executed')
    return errors


def output_payload(record):
    return {key: record[key] for key in ('matches', 'proposals', 'initial_poses', 'refined_poses',
                                        'refined_scores') if key in record}


def output_digest(record):
    return hashlib.sha256(json.dumps(output_payload(record), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def same_output(record, expected, expected_digest):
    return (record['output_sha256'] == expected_digest if record.get('compact_warmup')
            else output_payload(record) == expected)


def truth_pass(records):
    # Compact warmups must match the full-call output digest. The global exact
    # gate then implies they have the same GT outcome without retaining output.
    return all(record['truth_accuracy']['pose_fp'] == 0 and record['truth_accuracy']['pose_fn'] == 0
               for record in records if not record.get('compact_warmup'))


def create_real_search(job):
    saved_library, saved_scorer = real.Library, real.Scorer
    policy = rescue_policy_for_threshold(job['case']['score'])

    class RescueScorer(Scorer):
        def refine_resident(self, seeds, cpu_verify=True, **config):
            return self.refine_rescue(seeds, resident=True, **policy, **config)

    class SearchLibrary:
        def __init__(self, _):
            self.library = saved_library(str(core_library_path(job['baseline'])))

        def config(self):
            return self.library.config()

        def learn(self, image, min_area=256):
            return PatternAdapter(job, self.library, image, min_area)

    real.Library, real.Scorer = SearchLibrary, RescueScorer
    try:
        search = real.Search(dict(job, method='resident_verified_direct_metal',
                                  audit_refinement=True, prune=False))
    finally:
        real.Library, real.Scorer = saved_library, saved_scorer
    original_coarse = search.coarse

    def capture_proposals():
        search.profile_proposals = original_coarse()
        return search.profile_proposals
    search.coarse = capture_proposals
    return search, policy


def worker(job):
    cv2.setNumThreads(1)
    # A separate process for each backend/order block isolates allocator and
    # process-peak RSS history. Every real variant uses the same current scorer.
    library = real.Library(str(core_library_path(job['baseline'])))
    case = job['case']
    setup_start = time.perf_counter()
    search, policy, patterns = None, None, []
    try:
        if case['workflow'] == 'full_rescue':
            search, policy = create_real_search(job)
            patterns = search.patterns
            source, template = search.source, search.template
            algorithm_setup_ms = search.setup_ms
        else:
            source = cv2.imread(case['source'], cv2.IMREAD_GRAYSCALE)
            template = cv2.imread(case['template'], cv2.IMREAD_GRAYSCALE)
            if source is None or template is None:
                raise ValueError('missing source/template')
            construct_start = time.perf_counter()
            for scale in case['scales']:
                scaled = real.scaled_template(template, scale)
                patterns.append((scale, PatternAdapter(job, library, real.image_view(scaled), 256)))
            algorithm_setup_ms = (time.perf_counter()-construct_start)*1000
            config = library.config()
            config.max_targets, config.min_score = case['limit'], case['score']
            config.max_overlap, config.tolerance_angle = case['overlap'], case['angle']
            config.stop_layer, config.subpixel = 1, 0
        construct_ms = (time.perf_counter()-setup_start)*1000
        setup_patterns = [p.snapshot(scale, setup=True) for scale, p in patterns]
        setup_memory = dict(process_peak_rss_bytes=peak_rss(),
                           pose_owned_buffers=search.scorer.memory() if search else None)

        def run():
            start = time.perf_counter()
            if search:
                value = search.run()
                wall_ms = (time.perf_counter()-start)*1000
                record = dict(value, call_wall_ms=wall_ms, proposal_wall_ms=value['cpu_search_ms'],
                              total_wall_ms=value['algorithm_ms'], proposals=search.profile_proposals)
                if search.last_refinement is not None:
                    initial, poses, scores, _, metrics = search.last_refinement
                    record.update(initial_poses=initial.tolist(), refined_poses=poses.tolist(),
                                  refined_scores=scores.tolist(), refinement_metrics=metrics)
            else:
                matches = []
                for scale, pattern in patterns:
                    rows = pattern.match(real.image_view(source), config)
                    for row in rows:
                        row['scale'] = scale
                    matches.extend(rows)
                wall_ms = (time.perf_counter()-start)*1000
                record = dict(matches=matches, call_wall_ms=wall_ms, proposal_wall_ms=wall_ms,
                              total_wall_ms=wall_ms)
            # Snapshot/serialization/hash work is outside all recorded times.
            record['patterns'] = [p.snapshot(scale) for scale, p in patterns]
            record['profile_errors'] = profile_checks(record['patterns'])
            if job['method'].endswith('_profile') and any(p['profile'] is None for p in record['patterns']):
                record['profile_errors'].append('profile API returned no profile')
            record['output_sha256'] = output_digest(record)
            return record

        first = run()
        first_memory = dict(process_peak_rss_bytes=peak_rss(),
                            pose_owned_buffers=search.scorer.memory() if search else None)
        warmups, warm_start = [], time.perf_counter()
        while time.perf_counter()-warm_start < job['warmup_seconds']:
            record = run()
            warmups.append(dict(compact_warmup=True, output_sha256=record['output_sha256'],
                profile_errors=record['profile_errors'], total_wall_ms=record['total_wall_ms'],
                proposal_wall_ms=record['proposal_wall_ms'], call_wall_ms=record['call_wall_ms']))
            del record
        runs = [run() for _ in range(job['repeats'])]
        measured_memory = dict(process_peak_rss_bytes=peak_rss(),
                               pose_owned_buffers=search.scorer.memory() if search else None)
        # Check every call, including warmups, only after memory/time capture.
        for record in [first, *runs]:
            if case.get('truth'):
                record['truth_accuracy'] = real.truth_accuracy(record['matches'], case['truth'])
        return dict(method=job['method'], block=job['block'], case=case['name'],
            construct_ms=construct_ms, algorithm_setup_ms=algorithm_setup_ms,
            construct_includes_image_io=True, cold_construct_plus_first_ms=construct_ms+first['total_wall_ms'],
            setup_patterns=setup_patterns, effective_rescue_policy=policy, first=first,
            warmups=warmups, runs=runs, memory=dict(after_construct=setup_memory,
                after_first=first_memory, after_measurements=measured_memory),
            source_shape=list(source.shape), template_shape=list(template.shape))
    finally:
        if search:
            search.close()
        else:
            for _, pattern in patterns:
                pattern.close()


def make_cases(args, inputs):
    selected = set(args.cases.split(',')) if args.cases else None
    suite = args.suite
    if selected:
        real_selected = bool(selected & {'src7', 'controlled6', 'heldout12'})
        large_selected = any(name.startswith('8k_') for name in selected)
        suite = 'all' if real_selected and large_selected else 'real' if real_selected else 'large'
    cases = []
    for original in cases_for(SimpleNamespace(suite=suite), inputs):
        if original['name'].startswith('8k_'):
            cases.append(dict(original, workflow='core_only'))
            if original['name'] == '8k_t256':
                cases.append(dict(original, name='8k_t256_angle180', angle=180,
                    workflow='core_only', workload='same six upright exact copies; full 360-degree search stress'))
        else:
            cases.append(dict(original, workflow='full_rescue', proposal_score=original['score'],
                              score=.5 if original['name'] == 'src7' else .8))
    if selected:
        available = {case['name'] for case in cases}
        if selected-available:
            raise ValueError('Unknown cases: ' + ','.join(sorted(selected-available)))
        cases = [case for case in cases if case['name'] in selected]
    return cases


def calls(result):
    return [result['first'], *result['warmups'], *result['runs']]


def distribution(values):
    return dict(count=len(values), median=statistics.median(values), min=min(values), max=max(values),
                values=values)


def summarize_case(case, results, reference):
    expected = output_payload(reference['first'])
    expected_digest = output_digest(reference['first'])
    reference_exact = all(same_output(record, expected, expected_digest) for record in calls(reference))
    rows, exact, profiles_ok = {}, reference_exact, True
    truth_ok = truth_pass(calls(reference)) if case.get('truth') else None
    for method in dict.fromkeys(result['method'] for result in results):
        workers = [result for result in results if result['method'] == method]
        records = [record for result in workers for record in calls(result)]
        measured = [record for result in workers for record in result['runs']]
        same = all(same_output(record, expected, expected_digest) for record in records)
        profile_pass = all(not record['profile_errors'] for record in records)
        accurate = truth_pass(records) and same if case.get('truth') else None
        exact &= same
        profiles_ok &= profile_pass
        if truth_ok is not None:
            truth_ok &= accurate
        totals = {key: distribution([record[key] for record in measured]) for key in
                  ('proposal_wall_ms', 'total_wall_ms', 'call_wall_ms')}
        stage_times = {}
        work_counters, output_buffers = {}, {}
        if method.endswith('_profile'):
            for key in (*EXCLUSIVE_STAGES, 'core_wall_ms', 'api_wall_ms', 'api_residual_ms'):
                stage_times[key] = distribution([sum(p['profile'][key] for p in record['patterns'])
                                                for record in measured])
            fields = measured[0]['patterns'][0]['profile']
            for key in fields:
                if key.endswith('_bytes') or key.endswith('_ms') or key in ('version', 'struct_bytes', 'backend', 'reserved'):
                    continue
                work_counters[key] = distribution([sum(p['profile'][key] for p in record['patterns'])
                                                   for record in measured])
            for key in fields:
                if not key.endswith('_bytes') or key == 'struct_bytes':
                    continue
                # Scales execute serially; per-output/layer peaks are maxima,
                # while allocation traffic fields ending in total are summed.
                reduction = sum if '_total_' in key or key == 'pyramid_generated_bytes' else max
                output_buffers[key] = distribution([reduction(p['profile'][key] for p in record['patterns'])
                                                    for record in measured])
            height, width = workers[0]['source_shape']
            output_buffers['cpu_pyramid_horizontal_scratch_peak_proxy_bytes'] = (
                height*((width+1)//2)*C.sizeof(C.c_int)
                if any(p['profile']['pyramid_levels'] > 1 and
                       (p.get('execution_config') or {}).get('pyramid_mode', 0) == 0
                       for r in measured for p in r['patterns']) else 0)
        rows[method] = dict(exact_all_calls=same, profile_valid=profile_pass,
            truth_passed=accurate, measured=totals, stages=stage_times,
            work_counters=work_counters, nominal_output_buffers=output_buffers,
            execution_config=[p.get('execution_config') for p in measured[0]['patterns']],
            first_total_ms=[w['first']['total_wall_ms'] for w in workers],
            construct_ms=[w['construct_ms'] for w in workers],
            peak_rss_bytes=[w['memory']['after_measurements']['process_peak_rss_bytes'] for w in workers],
            blocks=[dict(block=w['block'], total_ms=statistics.median(r['total_wall_ms'] for r in w['runs']),
                         proposal_ms=statistics.median(r['proposal_wall_ms'] for r in w['runs'])) for w in workers])
    overhead = {}
    for backend in ('cpu', 'metal', *ACCELERATED_METHODS):
        if backend not in rows or backend+'_profile' not in rows:
            continue
        plain = {r['block']: r for r in rows[backend]['blocks']}
        profiled = {r['block']: r for r in rows[backend+'_profile']['blocks']}
        overhead[backend] = [dict(block=b,
            proposal_percent=100*(profiled[b]['proposal_ms']/plain[b]['proposal_ms']-1),
            total_percent=100*(profiled[b]['total_ms']/plain[b]['total_ms']-1))
            for b in plain.keys() & profiled.keys()]
    return dict(case=case, reference_all_calls_exact=reference_exact, exact_all_outputs=exact, profile_valid=profiles_ok,
                truth_passed=truth_ok, methods=rows, profile_overhead_percent_by_block=overhead)


def source_manifest():
    paths = set()
    for directory in ('src', 'include', 'experiments'):
        for path in (real.ROOT/'cpp-core'/directory).rglob('*'):
            if path.is_file() and path.suffix in ('.cpp', '.hpp', '.h', '.mm', '.metal', '.in', '.txt'):
                paths.add(path)
    for name in ('benchmark_pipeline_stages.py', 'benchmark_coarse_ncc.py', 'real_image_experiments.py',
                 'coarse_experiment.py', 'pose_experiment.py', 'rescue_policy.py',
                 'research_ncc_fourier.py', 'benchmark_pose_experiments.py'):
        paths.add(Path(__file__).with_name(name))
    paths.add(real.ROOT/'cpp-core/examples/python_ctypes.py')
    paths.add(real.ROOT/'cpp-core/CMakeLists.txt')
    return {str(path.resolve()): file_hash(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(real.ROOT/'build/fourier-gpu'))
    parser.add_argument('--baseline', default=str(real.ROOT/'build/pipeline-baseline'))
    parser.add_argument('--suite', choices=('real', 'large', 'all'), default='all')
    parser.add_argument('--cases', help='Comma-separated case names; overrides suite')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--combined-pyramid', choices=('direct', 'separable'), default='separable')
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--output', default=str(real.ROOT/'build/pipeline-stages/report.json'))
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(Path(args.worker).read_text())), allow_nan=False))
        return
    methods = args.methods.split(',')
    if not methods or len(set(methods)) != len(methods) or set(methods)-set(METHODS):
        parser.error('methods must be a unique comma-separated subset of ' + ','.join(METHODS))
    if args.repeats < 1 or args.blocks < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error('positive repeats/blocks and finite nonnegative warmup are required')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = output.parent/'inputs'
    inputs.mkdir(exist_ok=True)
    jobs = output.parent/(output.stem+'-jobs')
    jobs.mkdir(exist_ok=True)
    baseline = verify_baseline(args.baseline)
    cases = make_cases(args, inputs)
    source_hashes = source_manifest()
    library_hashes = dict(baseline['library_hashes'])
    for path in (core_library_path(args.build), coarse_library_path(args.build), pose_library_path(args.build)):
        library_hashes[str(path)] = file_hash(path)
    input_hashes = {case[key]: file_hash(case[key]) for case in cases for key in ('source', 'template')}
    report = dict(created=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        protocol='serial fresh-process workers; forward/reverse method order in alternating blocks; each worker constructs, runs first, warms, then repeats; no cross-method concurrent load',
        timing='core exclusive stages sum to core_wall; API residual completes API wall; coarse metrics upload/wait/device/normalization are nested subintervals, never added to those stages; Python proposal/total wall encloses API calls; construct includes image I/O while algorithm_setup excludes image decoding',
        accuracy='first/measured outputs including ordered proposals/refined arrays must exactly equal frozen CPU; warmup output SHA must equal the same reference and only digest/time/error records are retained to avoid method-speed-dependent RSS growth. GT is tested after timing/RSS; matching warmup digests imply the same GT outcome. Src7 has agreement only; six/twelve scenes have rotated/scaled objects; 8K has six upright exact copies, with an additional broad-angle search case',
        memory='RSS is process high water including Python, images and driver; owned-buffer metrics overlap RSS. Profile byte counts are nominal pixel/score-map outputs, not heap/RSS. Total allocation traffic is not simultaneous peak. Sum of per-scale maxima is not global peak. Pose device_current_allocated_bytes includes other Metal allocations.',
        profiling='opt-in clocks can perturb short sections; unprofiled variants measure overhead; sparse pilot methods may omit overhead, so do not present pilot stage timings as zero-overhead production timings',
        design=dict(methods=methods, blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup_seconds,
                    combined_pyramid_mode=1 if args.combined_pyramid == 'direct' else 2),
        baseline_provenance=baseline, source_hashes=source_hashes, library_hashes=library_hashes,
        input_hashes=input_hashes, execution_order=[], cases=[], complete=False)

    def run_worker(case, method, block, reference_only=False):
        job = dict(build=str(Path(args.build).resolve()), baseline=str(Path(args.baseline).resolve()),
                   case=case, method=method, block=block,
                   combined_pyramid_mode=1 if args.combined_pyramid == 'direct' else 2,
                   repeats=1 if reference_only else args.repeats,
                   warmup_seconds=0 if reference_only else args.warmup_seconds)
        label = f'{case["name"]}-{method}-b{block}'
        path = jobs/(label+'.json')
        path.write_text(json.dumps(job, indent=2))
        print('Running '+label+(' (frozen reference only)' if reference_only else ''), flush=True)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                                   capture_output=True, text=True)
        (jobs/(label+'.stderr.txt')).write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'Worker {label} failed ({completed.returncode}): {completed.stderr[-4000:]}')
        result = json.loads(completed.stdout)
        (jobs/(label+'.result.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
        report['execution_order'].append(dict(case=case['name'], method=method, block=block,
            reference_only=reference_only, job=str(path), job_sha256=file_hash(path),
            result=str(jobs/(label+'.result.json')), result_sha256=file_hash(jobs/(label+'.result.json'))))
        return result

    for case in cases:
        results, reference = [], None
        if 'frozen_cpu' not in methods:
            reference = run_worker(case, 'frozen_cpu', -1, True)
        for block in range(args.blocks):
            order = methods if block % 2 == 0 else methods[::-1]
            for method in order:
                result = run_worker(case, method, block)
                results.append(result)
                if method == 'frozen_cpu' and reference is None:
                    reference = result
        row = summarize_case(case, results, reference)
        row['reference_output_sha256'] = reference['first']['output_sha256']
        report['cases'].append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'{case["name"]}: exact={row["exact_all_outputs"]}, truth={row["truth_passed"]}, '
              f'profile valid={row["profile_valid"]}', flush=True)
    all_hashes = {**source_hashes, **library_hashes, **input_hashes,
                  baseline['manifest']: baseline['manifest_sha256']}
    report['sources_unchanged'] = all(file_hash(path) == digest for path, digest in all_hashes.items())
    report['complete'] = True
    report['passed'] = report['sources_unchanged'] and all(row['exact_all_outputs'] and row['profile_valid']
                           and row['truth_passed'] is not False for row in report['cases'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
