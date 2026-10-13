"""Shared source pyramid and batched Metal fine search versus frozen 2c28ffb.

Source-frame preparation is repeated INSIDE every timed full-pipeline call,
including warmups. All methods search online and share the same local rescue
policy. No module-global matcher/scorer monkeypatching is used. Workers run
serially in fresh processes, with method order reversed in the second block.
"""
import argparse
import copy
from datetime import datetime, timezone
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
import numpy as np

import benchmark_pipeline_stages as previous
import coarse_experiment as coarse
import real_image_experiments as real
from benchmark_coarse_ncc import core_library_path, file_hash
from benchmark_pose_experiments import peak_rss
from pose_experiment import Scorer, library_path as pose_library_path
from rescue_policy import rescue_policy_for_threshold
from summarize_pipeline_stages import load_worker, paired_regression


BASELINE_COMMIT = '2c28ffbde86d3ef9840cb00c2d15268b5cf72624'
BASE_METHODS = ('frozen_cpu', 'previous_combined', 'current_unshared', 'shared_cpu_fine', 'shared_metal_fine')
METHODS = BASE_METHODS + tuple(name+'_profile' for name in BASE_METHODS[2:])


def verify_baseline(build):
    build = Path(build).resolve()
    path = build/'baseline-provenance.json'
    manifest = json.loads(path.read_text())
    if manifest.get('commit') != BASELINE_COMMIT:
        raise ValueError('Expected frozen 2c28ffb core/coarse/pose baseline')
    hashes = {}
    for library in (core_library_path(build), coarse.library_path(build), pose_library_path(build)):
        library = Path(library).resolve()
        actual = file_hash(library)
        if manifest.get('files', {}).get(library.relative_to(build).as_posix()) != actual:
            raise ValueError('Frozen baseline differs from manifest: ' + str(library))
        hashes[str(library)] = actual
    return dict(verified=True, commit=BASELINE_COMMIT, manifest=str(path),
                manifest_sha256=file_hash(path), library_hashes=hashes)


class FrameAdapter:
    """The sole integration point for the new explicit frame API.

    There is intentionally no CPU fallback: an unavailable new API must fail
    the requested experiment. Preparation must happen once per run, not once
    per unchanged NumPy address or once per scale.
    """
    def __init__(self, build, fine_metal, patterns, fine_batch_size=128):
        self.patterns, self.generation = patterns, 0
        self.config = dict(backend='metal', pyramid_mode=1, batch_angles=4,
                           fine_batch_size=fine_batch_size if fine_metal else 0)
        self.frame = coarse.SourceFrame(build, **self.config)

    def prepare(self, source, invert=False):
        self.frame.prepare(source, self.patterns, invert_source=invert)
        self.generation += 1

    def match(self, pattern, config, profile=False):
        return self.frame.match(pattern, config, profile=profile)

    def match_snapshot(self):
        # SourceFrame.match assigns fresh dictionaries for each call. Retain
        # those references now, and copy/serialize only after the outer timer.
        return (self.frame.last_metrics, self.frame.last_profile,
                self.frame.last_execution_metrics, self.frame.last_fine_metrics)

    def snapshot(self):
        return dict(config=dict(self.config), metrics=copy.deepcopy(self.frame.frame_metrics),
                    prepare_execution_metrics=copy.deepcopy(self.frame.prepare_execution_metrics))

    def close(self):
        self.frame.close()


class PatternAdapter:
    def __init__(self, job, library, image):
        self.method = job['method']
        self.frame_snapshot = None
        base = self.method.removesuffix('_profile')
        self.pattern = (library.learn(image, 256) if base == 'frozen_cpu' else
            coarse.CoarsePattern(job['baseline'] if base == 'previous_combined' else job['build'],
                image, 256, backend='cpu' if base.startswith('shared_') else 'metal',
                pyramid_mode=0 if base.startswith('shared_') else 1,
                batch_angles=1 if base.startswith('shared_') else 4))

    def match(self, source, config, frame):
        if frame is not None:
            rows = frame.match(self.pattern, config, profile=self.method.endswith('_profile'))
            # Preserve already-returned snapshots before the next scale replaces
            # the frame attributes. No metrics getter or deepcopy is timed.
            self.frame_snapshot = frame.match_snapshot()
            return rows
        if self.method == 'frozen_cpu':
            return self.pattern.match(source, config, prune=False)
        return self.pattern.match(source, config, prune=False, profile=self.method.endswith('_profile'))

    def snapshot(self, scale, setup=False):
        if not setup and self.frame_snapshot is not None:
            metrics, profile, execution, fine = self.frame_snapshot
            return copy.deepcopy(dict(scale=scale, metrics=metrics, profile=profile,
                execution_config=dict(version=1, struct_bytes=32, pyramid_mode=1,
                                      batch_angles=4, reserved0=0, reserved1=0),
                execution_metrics=execution, fine_metrics=fine))
        return dict(scale=scale, metrics=getattr(self.pattern, 'setup_metrics' if setup else 'last_metrics', None),
            profile=None if setup else getattr(self.pattern, 'last_profile', None),
            execution_config=getattr(self.pattern, 'execution_config', None),
            execution_metrics=getattr(self.pattern, 'last_execution_metrics', None),
            fine_metrics=getattr(self.pattern, 'last_fine_metrics', None))

    def close(self):
        self.pattern.close()


class RescueScorer(Scorer):
    def __init__(self, build, source, template, policy):
        super().__init__(build, source, template, 'metal', 8)
        self.policy = dict(policy)

    def refine_resident(self, seeds, cpu_verify=True, **config):
        return self.refine_rescue(seeds, resident=True, **self.policy, **config)


class Search(real.Search):
    """Reuse unchanged refinement/NMS with explicitly constructed dependencies."""
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
        start = time.perf_counter()
        try:
            self.library = real.Library(str(core_library_path(job['baseline'])))
            for scale in self.scales:
                # Keep this local owner alive through the synchronous learn.
                scaled = real.scaled_template(self.template, scale)
                self.patterns.append((scale, PatternAdapter(job, self.library, real.image_view(scaled))))
            self.config = self.library.config()
            self.config.max_targets = case['limit']
            self.config.min_score = case.get('proposal_score', case['score'])
            self.config.max_overlap = case['overlap']
            self.config.tolerance_angle = case['angle']
            self.config.stop_layer = case['stop_layer']
            self.config.subpixel = case['subpixel']
            self.config.invert_source = case.get('invert_source', 0)
            if job['method'].startswith('shared_'):
                self.frame = FrameAdapter(job['build'], job['method'].startswith('shared_metal_fine'),
                                          [pattern.pattern for _, pattern in self.patterns],
                                          fine_batch_size=job.get('fine_batch_size', 128))
            if self.hybrid:
                self.source_float = self.source.astype(np.float32)/255
                self.scorer = RescueScorer(job['build'], self.source_float,
                    self.template.astype(np.float32)/255, self.policy)
            self.setup_ms = (time.perf_counter()-start)*1000
        except Exception:
            self.close()
            raise

    def coarse(self):
        # real.Search.run starts its full and proposal timers before this call.
        start = time.perf_counter()
        if self.frame:
            self.frame.prepare(real.image_view(self.source), invert=bool(self.config.invert_source))
            self.prepare_calls += 1
        self.last_prepare_wall_ms = (time.perf_counter()-start)*1000 if self.frame else 0.
        candidates = []
        source = real.image_view(self.source)
        for scale, pattern in self.patterns:
            for match in pattern.match(source, self.config, self.frame):
                match['scale'] = scale
                match['seed_scale'] = scale
                match['corners'] = real.corners([match['pos_x'], match['pos_y']], match['angle'], scale, self.shape)
                candidates.append(match)
        self.last_proposals = candidates
        return candidates

    def close(self):
        for _, pattern in self.patterns:
            pattern.close()
        self.patterns = []
        if self.scorer:
            self.scorer.close()
            self.scorer = None
        if self.frame:
            self.frame.close()
            self.frame = None


def profile_checks(patterns, shared):
    # Per-pattern shared matches correctly report no pyramid command: the
    # separate frame prepare owns it. Keep exclusive-stage/API checks, without
    # reusing the old per-pattern-pyramid requirement for a shared frame.
    items = [dict(item, execution_metrics=None) if shared else item for item in patterns]
    errors = previous.profile_checks(items)
    if shared:
        for item in patterns:
            execution, config = item.get('execution_metrics'), item.get('execution_config')
            if execution is None or config is None:
                errors.append('shared model execution metrics missing')
                continue
            if any(execution[key] != config[key] for key in ('pyramid_mode', 'batch_angles')):
                errors.append('shared execution metrics differ from configured route')
            if execution['max_batch_maps'] > config['batch_angles']:
                errors.append('shared angle batch exceeds configured cap')
            if execution['pyramid_command_buffers'] != 0:
                errors.append('shared match rebuilt its source pyramid')
    return errors


def fine_checks(patterns, frame):
    if frame is None:
        return []
    errors, config = [], frame['config']
    for item in patterns:
        fine = item.get('fine_metrics')
        if fine is None or fine.get('version') != 1:
            errors.append('shared fine metrics missing or unexpected version')
            continue
        if fine['peak_batch_maps'] > config['fine_batch_size']:
            errors.append('fine map batch exceeds configured cap')
        if fine['repair_pixels'] > fine['roi_pixels']:
            errors.append('fine repaired pixels exceed sampled pixels')
        if not config['fine_batch_size'] and (fine['maps'] or fine['command_buffers']):
            errors.append('CPU fine route unexpectedly dispatched fine GPU work')
        if config['fine_batch_size'] and item.get('profile') and fine['maps'] != item['profile']['fine_maps']:
            errors.append('fine GPU map count differs from core count')
        if fine['maps'] and fine['command_buffers'] < 2:
            errors.append('fine GPU maps have no warp and NCC commands')
    return errors


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


def make_cases(args, directory):
    selected = set(args.cases.split(',')) if args.cases else None
    need_real = not selected or bool(selected & {'src7', 'controlled6', 'heldout12'})
    need_large = not selected or any(name.startswith('8k_') for name in selected)
    cases = []
    if need_real or need_large:
        suite = 'all' if need_real and need_large else 'real' if need_real else 'large'
        for case in previous.make_cases(SimpleNamespace(suite=suite, cases=None), directory):
            cases.append(dict(case, stop_layer=1, subpixel=0))
            if case['name'] in ('8k_t256', '8k_t256_angle180'):
                cases.append(dict(case, name=case['name']+'_3scales', scales=[.75, 1., 1.25],
                                  stop_layer=1, subpixel=0))
    if not selected or any(name.startswith('small_fullres') for name in selected):
        template = np.random.default_rng(872913).integers(0, 256, (64, 96), dtype=np.uint8)
        source = np.full((512, 640), 127, dtype=np.uint8)
        truths = []
        for x, y in ((128, 128), (480, 384)):
            source[y-32:y+32, x-48:x+48] = template
            truths.append(dict(pos_x=x, pos_y=y, angle=0., scale=1.))
        src, tmp = directory/'small_fullres.png', directory/'small_template.png'
        cv2.imwrite(str(src), source); cv2.imwrite(str(tmp), template)
        for scales in ([1.], [.75, 1., 1.25]):
            cases.append(dict(name='small_fullres'+('_3scales' if len(scales) > 1 else ''),
                source=str(src), template=str(tmp), scales=scales, angle=180, limit=8,
                score=.7, overlap=0., truth=truths, workflow='core_only', stop_layer=0, subpixel=1))
    if selected:
        missing = selected-{case['name'] for case in cases}
        if missing:
            raise ValueError('Unknown cases: ' + ','.join(sorted(missing)))
        cases = [case for case in cases if case['name'] in selected]
    return cases


def summarize_case(case, workers, reference):
    row = previous.summarize_case(case, workers, reference)
    for method, data in row['methods'].items():
        subset = [worker for worker in workers if worker['method'] == method]
        data['frame_prepare_wall_ms'] = previous.distribution(
            [record['frame_prepare_wall_ms'] for worker in subset for record in worker['runs']])
        data['final_owned_memory'] = [worker['memory']['after_measurements'] for worker in subset]
        measured = [record for worker in subset for record in worker['runs']]
        # Diagnostics are separate from exclusive stage totals. Shared retained
        # capacities are read once from the frame, never summed per pattern.
        data['frame_prepare_metrics'] = [record['frame'] for record in measured if record['frame']]
        data['per_call_fine_metrics'] = [
            [pattern.get('fine_metrics') for pattern in record['patterns']] for record in measured]
        if method.startswith('shared_'):
            data['per_model_nominal_output_buffers'] = data.pop('nominal_output_buffers')
            data['memory_scope'] = 'source pyramid owned once in frame; per-model nominal counters are diagnostics and must not be summed as shared allocations'
    row['profile_overhead_percent_by_block'] = {}
    for base in BASE_METHODS[2:]:
        if base not in row['methods'] or base+'_profile' not in row['methods']:
            continue
        plain = {b['block']: b for b in row['methods'][base]['blocks']}
        profiled = {b['block']: b for b in row['methods'][base+'_profile']['blocks']}
        row['profile_overhead_percent_by_block'][base] = [dict(block=b,
            total_percent=100*(profiled[b]['total_ms']/plain[b]['total_ms']-1),
            proposal_percent=100*(profiled[b]['proposal_ms']/plain[b]['proposal_ms']-1))
            for b in sorted(plain.keys() & profiled.keys())]
    row['performance'] = {name: {baseline: paired_regression(row['methods'], name, baseline)
        for baseline in ('frozen_cpu', 'previous_combined', 'current_unshared', 'shared_cpu_fine') if baseline != name}
        for name in BASE_METHODS[2:] if name in row['methods']}
    return row


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
        reference = next(worker for worker in case_workers if worker['method'] == 'frozen_cpu')
        measured = [worker for worker in case_workers if worker['block'] >= 0]
        rows.append(summarize_case(entry['case'], measured, reference))
    return dict(report=Path(report_path).name, report_sha256=file_hash(report_path),
        raw_worker_sha256=hashes, complete=report.get('complete'),
        correctness_and_integrity_passed=report.get('passed'), design=report['design'],
        timing='shared frame prepare is counted once within every proposal/total call; profile stages exclude separately recorded frame preparation, so add preparation at most once; GPU device/wait/copy are nested',
        memory='shared frame owned buffers counted once, never summed per scale; RSS overlaps retained buffers; all high-water values include first call and warmups',
        performance_policy='per matching order block warm total median <= reference + max(10% of reference, 2ms); report each reference separately; accuracy pass is not a speed claim',
        cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(real.ROOT/'build/fourier-gpu'))
    parser.add_argument('--baseline', default=str(real.ROOT/'build/shared-fine-baseline'))
    parser.add_argument('--cases')
    parser.add_argument('--methods', default=','.join(BASE_METHODS))
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--fine-batch-size', type=int, default=128,
                        help='Fine GPU map batch cap 1..1024 (shared CPU fine remains disabled)')
    parser.add_argument('--output', default=str(real.ROOT/'build/shared-fine/report.json'))
    parser.add_argument('--worker')
    parser.add_argument('--summarize', metavar='REPORT', help='Only read saved results; no matching')
    parser.add_argument('--worker-root', help='Authoritative archived workers root for --summarize')
    args = parser.parse_args()
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
    for path in (Path(__file__).resolve(), Path(__file__).with_name('summarize_pipeline_stages.py').resolve()):
        source_hashes[str(path)] = file_hash(path)
    library_hashes = dict(baseline['library_hashes'])
    for path in (core_library_path(args.build), coarse.library_path(args.build), pose_library_path(args.build)):
        library_hashes[str(path)] = file_hash(path)
    input_hashes = {case[key]: file_hash(case[key]) for case in cases for key in ('source', 'template')}
    report = dict(created=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        design=dict(methods=methods, blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup_seconds,
                    pyramid_mode=1, batch_angles=4, fine_batch_size=args.fine_batch_size,
                    shared_prepare_every_call=True),
        baseline_provenance=baseline, source_hashes=source_hashes, library_hashes=library_hashes,
        input_hashes=input_hashes, execution_order=[], cases=[], complete=False,
        scope='online full source search; prepare is timed once per frame every call; shared source reuse is across scales only, never across changing frames; same current resident rescue for real scenes; 8K/small strict cases are core-only',
        accuracy='ordered proposals plus final matches/refined poses/scores exact vs frozen CPU; GT evaluated after time/RSS; no truth supplied to algorithms; warmups retain only digest/time/errors',
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
        if 'frozen_cpu' not in methods:
            reference = run_worker(case, 'frozen_cpu', -1, True)
        for block in range(args.blocks):
            for method in methods if block % 2 == 0 else methods[::-1]:
                result = run_worker(case, method, block)
                results.append(result)
                if method == 'frozen_cpu' and reference is None:
                    reference = result
        row = summarize_case(case, results, reference)
        report['cases'].append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'{case["name"]}: exact={row["exact_all_outputs"]}, GT={row["truth_passed"]}, profile={row["profile_valid"]}', flush=True)
    all_hashes = {**source_hashes, **library_hashes, **input_hashes, baseline['manifest']: baseline['manifest_sha256']}
    report['sources_unchanged'] = all(file_hash(path) == digest for path, digest in all_hashes.items())
    report['complete'] = True
    report['passed'] = report['sources_unchanged'] and all(row['exact_all_outputs'] and row['profile_valid']
        and row['truth_passed'] is not False for row in report['cases'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
