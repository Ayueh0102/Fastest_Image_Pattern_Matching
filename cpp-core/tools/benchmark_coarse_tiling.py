"""Ablate Metal coarse NCC spatial tiling against frozen d60ec04.

All routes use the CPU peak tree, synchronous coarse submission, shared source
pyramid and fine batch 512. GT is held by the parent process. Workers are fresh,
serial, and retain complete ordered outputs separately from performance gates.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import benchmark_coarse_overlap as overlap
import benchmark_coarse_next as coarse
import benchmark_many_objects as many
from benchmark_coarse_ncc import core_library_path, file_hash
from pose_experiment import library_path as pose_library_path
from summarize_pipeline_stages import load_worker, paired_regression
from many_object_accuracy import evaluate

BASELINE_COMMIT = 'd60ec04c0d78ce10df6e021da7f0969f37a12d95'
ROUTES = {'previous_peak': 0, 'current_peak': 0, 'tile_direct': 2, 'tile_shared': 3}
METHODS = tuple(ROUTES) + tuple(name+'_profile' for name in ROUTES if name != 'previous_peak')
DEFAULT_CASES = 'controlled6,heldout12,8k_t256_angle180_3scales,small_fullres,many128_pose,many512_pose'


class FrameAdapter(overlap.FrameAdapter):
    def __init__(self, job, patterns):
        super().__init__(job, patterns)
        try:
            mode = job['_ncc_mode']
            self.frame.set_coarse_optimization(peak_mode=1, ncc_mode=mode)
            actual = self.frame.coarse_optimization
            if actual != dict(peak_mode=1, ncc_mode=mode):
                raise ValueError('Coarse optimization route did not match request')
            self.config['coarse_optimization'] = dict(actual, frozen_api=job['_frozen'])
        except Exception:
            self.close()
            raise


class Search(coarse.Search):
    def __init__(self, job):
        base = job['method'].removesuffix('_profile')
        frozen = base == 'previous_peak'
        mapped = 'previous_shared' if frozen else 'peak_only'
        if job['method'].endswith('_profile'):
            mapped += '_profile'
        mapped_job = dict(job, method=mapped, _overlap=False,
                          _ncc_mode=ROUTES[base], _frozen=frozen)
        super().__init__(mapped_job, frame_adapter=FrameAdapter,
                         pattern_adapter=overlap.PatternAdapter)


def worker(job):
    # Defend the worker boundary even when --worker is invoked directly.
    if job['case'] != many.worker_case(job['case']):
        raise ValueError('Worker case must contain only search inputs/settings; GT and scene metadata stay in parent')
    return coarse.worker(job, search_factory=Search)


def verify_baseline(build):
    build = Path(build).resolve()
    path = build/'provenance.json'
    manifest = json.loads(path.read_text())
    if manifest.get('commit') != BASELINE_COMMIT:
        raise ValueError('Expected frozen d60ec04 core/coarse/pose baseline')
    hashes = {}
    for library in (core_library_path(build), coarse.coarse.library_path(build), pose_library_path(build)):
        library = Path(library).resolve()
        relative = library.relative_to(build).as_posix()
        actual = file_hash(library)
        if manifest.get('files', {}).get(relative) != actual:
            raise ValueError('Frozen library differs from manifest: '+relative)
        hashes[str(library)] = actual
    return dict(commit=BASELINE_COMMIT, verified=True, manifest=str(path),
                manifest_sha256=file_hash(path), library_hashes=hashes)


def route_errors(record, method):
    expected_mode = ROUTES[method.removesuffix('_profile')]
    config = record['frame']['config']
    errors = []
    if (config.get('coarse_overlap') is not False or config['coarse_optimization']['peak_mode'] != 1
            or config['coarse_optimization']['ncc_mode'] != expected_mode):
        errors.append('Wrong coarse route or overlap enabled')
    for pattern in record['patterns']:
        metrics, fine, profile = pattern['metrics'], pattern['fine_metrics'], pattern['profile']
        asynchronous = pattern.get('overlap_metrics')
        if asynchronous is None or any(asynchronous[key] != 0 for key in (
                'enabled', 'submitted_batches', 'completed_batches', 'submitted_maps',
                'fallback_maps', 'drained_batches', 'max_pending')):
            errors.append('Synchronous route dispatched overlap work or diagnostics missing')
        if asynchronous is not None and asynchronous['retained_gpu_bytes'] > metrics['gpu_bytes']:
            errors.append('Overlap retained capacity excluded from coarse memory')
        if profile and (profile['coarse_windows'] != metrics['windows']
                        or profile['fine_maps'] != fine['maps']
                        or profile['fine_windows'] != 49*fine['maps']):
            errors.append('Profile workload differs from backend counters')
    return errors


def summarize_case(case, workers):
    # No truth is supplied to a search, refinement, or parameter selection call.
    if case.get('truth'):
        for item in workers:
            for record in [item['first'], *item['runs']]:
                record['truth_accuracy'] = coarse.real.truth_accuracy(record['matches'], case['truth'])
    reference = next(item for item in workers if item['method'] == 'previous_peak')
    row = coarse.summarize_case(case, workers, reference)
    row['performance'] = {name: {ref: paired_regression(row['methods'], name, ref)
        for ref in ('previous_peak', 'current_peak') if ref != name}
        for name in ROUTES if name != 'previous_peak' and name in row['methods']}
    row.update(outer_stages={}, independent_accuracy={}, route_errors={}, coarse_diagnostics={})
    reference_windows = [p['metrics']['windows'] for p in reference['first']['patterns']]
    row['window_counts_match_reference'] = True
    for method in dict.fromkeys(item['method'] for item in workers):
        group = [item for item in workers if item['method'] == method]
        records = [record for item in group for record in [item['first'], *item['runs']]]
        measured = [record for item in group for record in item['runs']]
        row['route_errors'][method] = sorted({error for record in records for error in route_errors(record, method)})
        row['window_counts_match_reference'] &= all(
            [p['metrics']['windows'] for p in record['patterns']] == reference_windows for record in records)
        row['outer_stages'][method] = {key: coarse.previous.distribution([record[key] for record in measured])
            for key in ('frame_prepare_wall_ms', 'proposal_wall_ms', 'refine_ms', 'nms_ms', 'seed_count')}
        row['coarse_diagnostics'][method] = {key: coarse.previous.distribution([
            sum(p['metrics'][key] for p in record['patterns']) for record in measured])
            for key in ('device_ms', 'wait_ms', 'upload_ms', 'normalization_ms', 'ncc_ms',
                        'score_maps', 'windows', 'command_buffers', 'integer32_maps', 'integer64_maps')}
        if case.get('truth'):
            assessments = [evaluate(record['matches'], case['truth']) for record in records]
            first = assessments[0]
            row['independent_accuracy'][method] = dict(first=first,
                all_calls_same_counts=all(all(a['strict'][key] == first['strict'][key]
                    for key in ('tp', 'fp', 'fn')) for a in assessments),
                all_calls_strict_pass=all(a['strict']['fp'] == a['strict']['fn'] == 0 for a in assessments))
    row['route_valid'] = not any(row['route_errors'].values())
    row['profile_overhead_percent_by_block'] = {}
    for name in ROUTES:
        if name not in row['methods'] or name+'_profile' not in row['methods']:
            continue
        plain = {b['block']: b for b in row['methods'][name]['blocks']}
        profiled = {b['block']: b for b in row['methods'][name+'_profile']['blocks']}
        row['profile_overhead_percent_by_block'][name] = [dict(block=block,
            total_percent=100*(profiled[block]['total_ms']/plain[block]['total_ms']-1),
            proposal_percent=100*(profiled[block]['proposal_ms']/plain[block]['proposal_ms']-1))
            for block in sorted(plain.keys() & profiled.keys())]
    return row


def replay(report, report_path, worker_root=None):
    groups, worker_hashes, job_hashes, identities = {}, {}, {}, set()
    for entry in report['execution_order']:
        logical, digest, result = load_worker(entry, report_path, worker_root)
        job_entry = dict(result=entry['job'], result_sha256=entry['job_sha256'])
        job_logical, job_digest, job = load_worker(job_entry, report_path, worker_root)
        identity = (entry['case'], entry['method'], entry['block'])
        if logical in worker_hashes or job_logical in job_hashes or identity in identities:
            raise ValueError('Duplicate archived worker/job identity')
        if identity != (result['case'], result['method'], result['block']) or identity != (
                job['case']['name'], job['method'], job['block']):
            raise ValueError('Worker/job identity differs from execution entry')
        if job['case'] != many.worker_case(job['case']):
            raise ValueError('Archived worker job contains GT or scene metadata')
        identities.add(identity)
        worker_hashes[logical], job_hashes[job_logical] = digest, job_digest
        groups.setdefault(entry['case'], []).append(result)
    return dict(report=Path(report_path).name, report_sha256=file_hash(report_path),
        raw_worker_sha256=worker_hashes, raw_job_sha256=job_hashes,
        complete=report.get('complete'), integrity_passed=report.get('integrity_passed'),
        detection_all_passed=report.get('detection_all_passed'), design=report['design'],
        cases=[summarize_case(item['case'], groups[item['case']['name']]) for item in report['cases']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(coarse.real.ROOT/'build/coarse-tiling'))
    parser.add_argument('--baseline', default=str(coarse.real.ROOT/'build/coarse-tiling-baseline'))
    parser.add_argument('--cases', default=DEFAULT_CASES)
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=coarse.real.ROOT/'build/coarse-tiling-results/formal.json')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--worker-root', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker.read_text())), allow_nan=False))
        return
    output = args.output.resolve()
    if args.summarize:
        if output == args.summarize.resolve():
            parser.error('Do not overwrite raw report')
        many.save(output, replay(json.loads(args.summarize.read_text()), args.summarize, args.worker_root))
        return
    methods, names = args.methods.split(','), args.cases.split(',')
    if set(methods)-set(METHODS) or len(methods) != len(set(methods)) or 'previous_peak' not in methods:
        parser.error('Unique supported methods including previous_peak required')
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        parser.error('Unique nonempty case names required')
    if args.pilot:
        args.blocks, args.repeats, args.warmup_seconds = 1, 1, 0.
    if args.blocks < 1 or args.repeats < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error('Positive repeat/block and nonnegative finite warmup required')
    inputs, jobs = output.parent/'inputs', output.parent/(output.stem+'-jobs')
    inputs.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(exist_ok=True)
    regular, crowded = [n for n in names if not n.startswith('many')], [n for n in names if n.startswith('many')]
    cases = coarse.make_cases(SimpleNamespace(cases=','.join(regular), suite='all'), inputs) if regular else []
    if crowded:
        cases += many.make_cases(inputs, crowded)
    cases = sorted(cases, key=lambda case: names.index(case['name']))
    baseline = verify_baseline(args.baseline)
    sources = coarse.previous.source_manifest()
    for name in ('benchmark_coarse_tiling.py', 'benchmark_coarse_overlap.py', 'benchmark_many_objects.py',
                 'many_object_scenes.py', 'many_object_accuracy.py', 'benchmark_coarse_next.py',
                 'benchmark_shared_fine.py', 'summarize_pipeline_stages.py',
                 'coarse_tile_probe.cpp', 'benchmark_tile_probe.py'):
        path = Path(__file__).with_name(name).resolve()
        sources[str(path)] = file_hash(path)
    protocol = coarse.real.ROOT/'cpp-core/docs/coarse-tiling/protocol.json'
    sources[str(protocol)] = file_hash(protocol)
    libraries = dict(baseline['library_hashes'])
    for path in (core_library_path(args.build), coarse.coarse.library_path(args.build), pose_library_path(args.build)):
        libraries[str(Path(path).resolve())] = file_hash(path)
    input_hashes = {case[key]: file_hash(case[key]) for case in cases
                    for key in ('source', 'template', 'metadata') if key in case}
    report = dict(created=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        baseline_provenance=baseline, source_hashes=sources, library_hashes=libraries, input_hashes=input_hashes,
        execution_order=[], cases=[], complete=False,
        design=dict(methods=methods, blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup_seconds,
            pilot=args.pilot, pyramid_mode=1, batch_angles=4, fine_batch_size=512, peak_mode=1, overlap=False,
            routes=ROUTES, timing='Fresh serial workers, forward/reverse blocks; frame prepare inside every call; construction/first separate; profile workers separate',
            accuracy='GT stays in parent; ordered proposals/matches/poses/scores and coarse/fine map/window counts exact; independent strict 2px/1degree/2% scale gates',
            memory='Process peak RSS includes Python/images/driver and overlaps frame/pose retained allocations; frame capacity counted once across scales',
            diagnostics='Coarse GPU device/wait/upload/normalization timings are nested in pipeline timing, not additive; device duration is not isolated pure kernel instruction time'))
    many.save(output, report)
    for case in cases:
        workers = []
        for block in range(args.blocks):
            for method in methods if block % 2 == 0 else methods[::-1]:
                job = dict(case=many.worker_case(case), method=method, block=block,
                    build=str(Path(args.build).resolve()), baseline=str(Path(args.baseline).resolve()),
                    repeats=args.repeats, warmup_seconds=args.warmup_seconds, fine_batch_size=512)
                label = f'{case["name"]}-{method}-b{block}'
                job_path = jobs/(label+'.json')
                many.save(job_path, job)
                print('Running '+label, flush=True)
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(job_path)],
                                       capture_output=True, text=True)
                (jobs/(label+'.stderr.txt')).write_text(child.stderr)
                if child.returncode:
                    raise RuntimeError(f'{label}: {child.stderr[-4000:]}')
                result = json.loads(child.stdout)
                result_path = jobs/(label+'.result.json')
                many.save(result_path, result)
                workers.append(result)
                report['execution_order'].append(dict(case=case['name'], method=method, block=block,
                    job=str(job_path), job_sha256=file_hash(job_path),
                    result=str(result_path), result_sha256=file_hash(result_path)))
                many.save(output, report)
        row = summarize_case(case, workers)
        report['cases'].append(row)
        many.save(output, report)
        print(f'{case["name"]}: exact={row["exact_all_outputs"]} GT={row["truth_passed"]} route={row["route_valid"]}', flush=True)
    hashes = {**sources, **libraries, **input_hashes, baseline['manifest']: baseline['manifest_sha256']}
    report['sources_unchanged'] = all(file_hash(path) == digest for path, digest in hashes.items())
    report['complete'] = True
    report['integrity_passed'] = report['sources_unchanged'] and all(
        row['exact_all_outputs'] and row['profile_valid'] and row['route_valid']
        and row['work_counts_match_reference'] and row['window_counts_match_reference']
        and all(item['all_calls_same_counts'] for item in row['independent_accuracy'].values()) for row in report['cases'])
    report['detection_all_passed'] = all(item['all_calls_strict_pass']
        for row in report['cases'] for item in row['independent_accuracy'].values())
    many.save(output, report)
    if not report['integrity_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
