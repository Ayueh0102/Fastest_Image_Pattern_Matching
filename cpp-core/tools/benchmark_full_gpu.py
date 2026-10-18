"""Fresh-process comparison of hybrid, resident-coarse, and complete Metal paths.

Truth remains in this parent process. A GPU-residency claim requires one command
buffer, one final wait, no intermediate readback, and no CPU algorithm steps.
Those counters are an implementation contract, not hardware profiler evidence.
Changed ordering or FP32 scores are recorded; independent pose accuracy decides
correctness. Failed accuracy/performance cases are retained rather than hidden.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from types import SimpleNamespace

import benchmark_native_pipeline as native
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = 'controlled6,heldout12,small_fullres,8k_t256_angle180_3scales,many128_pose,many512_pose'
METHODS = {
    'hybrid_best': dict(full_gpu=False, coarse_peak_mode=1),
    'coarse_resident': dict(full_gpu=False, coarse_peak_mode=2),
    'full_gpu': dict(full_gpu=True, coarse_peak_mode=1),
}
LIGHT_CASES = {'boundary_one', 'empty_scene', 'constant_source'}
RESIDENCY_COUNTS = {'command_buffers': 1, 'cpu_waits': 1,
                    'intermediate_readbacks': 0, 'cpu_algorithm_steps': 0}
GPU_FIELDS = (*RESIDENCY_COUNTS, 'gpu_bytes', 'upload_bytes', 'readback_bytes',
              'encode_ms', 'wait_ms', 'device_ms')
STAGE_FIELDS = ('total_ms', 'prepare_ms', 'proposal_ms', 'refine_ms', 'nms_ms')
POSE_FIELDS = ('pos_x', 'pos_y', 'angle', 'scale', 'score')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def job(case, method, repeats, warmup):
    result = native.native_job(case, 'native_tile', repeats, warmup)
    result.update(METHODS[method])
    result.update(fine_ncc_mode=2, fine_fused=True, fine_angle_reuse=True,
                  pose_resident_mode=0, pose_input_mode=1,
                  pose_storage_mode=1, refinement_backend=1)
    return result


def selected_names(value, allowed, label):
    names = value.split(',')
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(label+' must be nonempty and unique')
    if allowed is not None and set(names)-set(allowed):
        raise ValueError('Unknown '+label+': '+','.join(sorted(set(names)-set(allowed))))
    return names


def make_cases(names, directory):
    regular = [name for name in names if not name.startswith('many') and name not in LIGHT_CASES]
    crowded = [name for name in names if name.startswith('many')]
    cases = native.old.coarse.make_cases(SimpleNamespace(cases=','.join(regular), suite='all'), directory) if regular else []
    if crowded:
        cases += native.many.make_cases(directory, crowded)
    if set(names) & LIGHT_CASES:
        import cv2
        import numpy as np
        rng = np.random.default_rng(370019)
        template = rng.integers(0, 256, (48, 64), dtype=np.uint8)
        template_path = directory/'edge_template.png'
        if not cv2.imwrite(str(template_path), template):
            raise RuntimeError('Cannot write lightweight template')
        for name in names:
            if name not in LIGHT_CASES:
                continue
            source = np.full((192, 256), 127, dtype=np.uint8)
            truth = []
            if name == 'boundary_one':
                source[:48, :64] = template
                truth = [dict(pos_x=32., pos_y=24., angle=0., scale=1.)]
            elif name == 'empty_scene':
                source = rng.integers(0, 256, source.shape, dtype=np.uint8)
            source_path = directory/(name+'.png')
            if not cv2.imwrite(str(source_path), source):
                raise RuntimeError('Cannot write lightweight source')
            cases.append(dict(name=name, source=str(source_path), template=str(template_path),
                scales=[1.], angle=0, limit=8, score=.9, overlap=0., truth=truth,
                workflow='core_only', stop_layer=0, subpixel=0))
    if {case['name'] for case in cases} != set(names):
        raise ValueError('Requested cases do not match generated cases')
    return sorted(cases, key=lambda case: names.index(case['name']))


def finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid '+label)
    return value


def worker_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Worker path outside experiment directory')
    return path


def stage_timings_available(record):
    # Early pilot records did not carry this flag. Preserve their explicitly
    # documented legacy interpretation rather than rewriting archived workers.
    available = record.get('stage_timings_available', True)
    if not isinstance(available, bool):
        raise ValueError('Invalid stage_timings_available boolean')
    return available


def device_timestamp_available(metrics):
    # Legacy records support only finite nonnegative timestamps. A null value
    # always requires the new explicit unavailable flag; absence is not proof
    # that a missing timestamp is an intentional measurement limitation.
    if 'device_timestamp_available' not in metrics:
        finite_nonnegative(metrics.get('device_ms'), 'legacy full_gpu.device_ms')
        return True
    available = metrics['device_timestamp_available']
    if not isinstance(available, bool):
        raise ValueError('Invalid full_gpu.device_timestamp_available boolean')
    if available:
        finite_nonnegative(metrics.get('device_ms'), 'full_gpu.device_ms')
    elif 'device_ms' not in metrics or metrics['device_ms'] is not None:
        raise ValueError('Unavailable full_gpu.device_ms must explicitly be null')
    return available


def validate_record(record, full_gpu):
    stage_available = stage_timings_available(record)
    for key in STAGE_FIELDS:
        value = record.get(key)
        if key == 'total_ms' or stage_available or value is not None:
            finite_nonnegative(value, key)
    if record['total_ms'] <= 0:
        raise ValueError('Nonpositive total_ms')
    for name in ('matches', 'proposals'):
        if not isinstance(record[name], list):
            raise ValueError('Invalid '+name)
        for pose in record[name]:
            for key in POSE_FIELDS:
                value = pose[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Invalid '+name+' '+key)
            if pose['scale'] <= 0 or not -1.00001 <= pose['score'] <= 1.00001:
                raise ValueError('Invalid output scale/score')
    metrics = record.get('full_gpu', {})
    if full_gpu:
        if not isinstance(metrics.get('enabled'), bool):
            raise ValueError('Missing full_gpu.enabled boolean')
        for key in GPU_FIELDS:
            if key != 'device_ms':
                finite_nonnegative(metrics[key], 'full_gpu.'+key)
        device_timestamp_available(metrics)
        for key in (*RESIDENCY_COUNTS, 'gpu_bytes', 'upload_bytes', 'readback_bytes'):
            if not isinstance(metrics[key], int):
                raise ValueError('Noninteger full_gpu.'+key)
        return (metrics['enabled'] and all(metrics[key] == value for key, value in RESIDENCY_COUNTS.items())
                and metrics['gpu_bytes'] > 0)
    return not metrics.get('enabled', False)


def distribution(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)


def summarize(manifest, root):
    if not manifest.get('complete'):
        raise ValueError('Incomplete experiment')
    if manifest.get('unchanged') is not True:
        raise ValueError('Source, binaries, or inputs changed during measurement')
    before = {key: manifest[key] for key in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(before.values()) or manifest.get('after_sha256') != before:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    methods = manifest['methods']
    if not isinstance(methods, list) or len(set(methods)) != len(methods) or not {'hybrid_best', 'full_gpu'} <= set(methods):
        raise ValueError('Unique hybrid_best and full_gpu methods are required')
    if set(methods)-set(METHODS):
        raise ValueError('Unknown experiment method')
    case_names = [case['name'] for case in manifest['cases']]
    if not case_names or len(set(case_names)) != len(case_names):
        raise ValueError('Empty or duplicate cases')
    if manifest['blocks'] < 1 or manifest['repeats'] < 1:
        raise ValueError('Empty measurement blocks')
    expected = {(case, method, block) for case in case_names for method in methods for block in range(manifest['blocks'])}
    entries = {}
    for entry in manifest['executions']:
        key = (entry['case'], entry['method'], entry['block'])
        if key not in expected or key in entries:
            raise ValueError('Unexpected or duplicate worker '+str(key))
        paths = {}
        for name in ('job', 'result'):
            path = worker_path(root, entry[name])
            if sha(path) != entry[name+'_sha256']:
                raise ValueError('Changed '+name+': '+entry[name])
            paths[name] = path
        work_job = json.loads(paths['job'].read_text())
        for name, value in METHODS[entry['method']].items():
            if type(work_job.get(name)) is not type(value) or work_job[name] != value:
                raise ValueError('Worker route does not match method')
        if any(name in work_job for name in ('truth', 'truths', 'ground_truth')):
            raise ValueError('Ground truth leaked into worker')
        case = next(case for case in manifest['cases'] if case['name'] == entry['case'])
        if work_job != job(case, entry['method'], manifest['repeats'], manifest['warmup_seconds']):
            raise ValueError('Worker job does not match shared search settings')
        worker = json.loads(paths['result'].read_text())
        if len(worker['runs']) != manifest['repeats']:
            raise ValueError('Missing or extra measured repeats')
        finite_nonnegative(worker['construct_ms'], 'construct_ms')
        finite_nonnegative(worker['final_memory']['peak_rss_bytes'], 'peak_rss_bytes')
        if not isinstance(worker['exact_repeat'], bool):
            raise ValueError('Invalid exact_repeat flag')
        entries[key] = worker
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    rows = []
    for case in manifest['cases']:
        reference = entries[(case['name'], 'hybrid_best', 0)]['first']
        method_rows = {}
        for method in methods:
            blocks = [(block, entries[(case['name'], method, block)]) for block in range(manifest['blocks'])]
            calls = [record for _, worker in blocks for record in [worker['first'], *worker['runs']]]
            runs = [record for _, worker in blocks for record in worker['runs']]
            route_valid = [validate_record(record, method == 'full_gpu') for record in calls]
            accuracy = [evaluate(record['matches'], case['truth'])['strict'] for record in calls]
            control_accuracy = [evaluate(record['matches'], reference['matches'])['strict'] for record in calls]
            score_deltas = [abs(record['matches'][pair['prediction_index']]['score']-
                                reference['matches'][pair['truth_index']]['score'])
                            for record, accuracy_row in zip(calls, control_accuracy) for pair in accuracy_row['pairs']]
            scores = [match['score'] for record in calls for match in record['matches']]
            gpu_rows = [record.get('full_gpu', {}) for record in calls]
            stage_flags = [stage_timings_available(record) for record in calls]
            measured_stages_available = all(stage_timings_available(record) for record in runs)
            gpu_metrics = None
            timestamp_availability = None
            if method == 'full_gpu':
                timestamp_flags = [device_timestamp_available(row) for row in gpu_rows]
                gpu_metrics = {key: distribution([row[key] for row in gpu_rows])
                               for key in GPU_FIELDS if key != 'device_ms'}
                timestamps = [row['device_ms'] for row, available in zip(gpu_rows, timestamp_flags) if available]
                gpu_metrics['device_ms'] = distribution(timestamps) if timestamps else None
                timestamp_availability = dict(available_calls=sum(timestamp_flags), total_calls=len(calls),
                    legacy_assumed_available_calls=sum('device_timestamp_available' not in row for row in gpu_rows))
            method_rows[method] = dict(
                stages={key: distribution([record[key] for record in runs])
                        if key == 'total_ms' or measured_stages_available else None for key in STAGE_FIELDS},
                stage_timing_availability=dict(available_calls=sum(stage_flags), total_calls=len(calls),
                    measured_stages_complete=measured_stages_available,
                    legacy_assumed_available_calls=sum('stage_timings_available' not in record for record in calls)),
                blocks=[dict(block=block, median_ms=statistics.median(record['total_ms'] for record in worker['runs']))
                        for block, worker in blocks],
                first_ms=[worker['first']['total_ms'] for _, worker in blocks],
                construct_ms=[worker['construct_ms'] for _, worker in blocks],
                peak_rss_bytes=max(worker['final_memory']['peak_rss_bytes'] for _, worker in blocks),
                retained_gpu_bytes=max((record['full_gpu']['gpu_bytes'] if method == 'full_gpu' else
                    record['frame_coarse_gpu_bytes']+record['frame_fine_gpu_bytes']) for record in calls),
                retained_gpu_scope=('full_gpu.gpu_bytes: retained named Metal buffer lengths, including shared input/output; not physical VRAM' if method == 'full_gpu' else
                                    'frame_coarse_gpu_bytes + frame_fine_gpu_bytes; excludes other owners'),
                residency_metrics=gpu_metrics,
                device_timestamp_availability=timestamp_availability,
                route_valid=all(route_valid),
                full_pipeline_residency_verified=(all(route_valid) if method == 'full_gpu' else False),
                accuracy_counts=sorted({(row['tp'], row['fp'], row['fn']) for row in accuracy}),
                all_gt_pass=all(row['fp'] == row['fn'] == 0 for row in accuracy),
                max_errors={key: max((row['max_errors'][key] for row in accuracy if row['max_errors'][key] is not None), default=None)
                            for key in ('position_px', 'angle_deg', 'relative_scale')},
                scores=distribution(scores) if scores else None,
                max_matched_score_delta_from_control=max(score_deltas, default=None),
                strict_pose_equivalence_to_control=all(row['fp'] == row['fn'] == 0 for row in control_accuracy),
                ordered_matches_equal_control=all(record['matches'] == reference['matches'] for record in calls),
                ordered_proposals_equal_control=all(record['proposals'] == reference['proposals'] for record in calls),
                repeat_exact=all(worker['exact_repeat'] for _, worker in blocks) and
                    all(record['matches'] == calls[0]['matches'] and record['proposals'] == calls[0]['proposals'] for record in calls),
                proposal_counts=sorted({len(record['proposals']) for record in calls}),
                work=[record.get('work', []) for record in runs])
        baseline = method_rows['hybrid_best']
        comparisons = {}
        for method in methods:
            if method == 'hybrid_best':
                continue
            measured = method_rows[method]
            pairs = [dict(block=left['block'], delta_percent=100*(right['median_ms']/left['median_ms']-1),
                          within_regression_budget=right['median_ms'] <= left['median_ms']+max(2., .1*left['median_ms']))
                     for left, right in zip(baseline['blocks'], measured['blocks'])]
            memory_ok = measured['peak_rss_bytes'] <= baseline['peak_rss_bytes']+max(8*1024**2, .1*baseline['peak_rss_bytes'])
            comparisons[method] = dict(delta_percent=100*(measured['stages']['total_ms']['median']/baseline['stages']['total_ms']['median']-1),
                paired=pairs, memory_within_budget=memory_ok,
                candidate_for_adoption=baseline['all_gt_pass'] and baseline['route_valid'] and measured['all_gt_pass'] and
                    measured['route_valid'] and memory_ok and all(pair['within_regression_budget'] for pair in pairs))
        rows.append(dict(name=case['name'], truth_count=len(case['truth']), methods=method_rows, comparisons=comparisons))
    return dict(cases=rows,
        all_full_gpu_gt_pass=all(row['methods']['full_gpu']['all_gt_pass'] for row in rows),
        all_full_gpu_resident=all(row['methods']['full_gpu']['full_pipeline_residency_verified'] for row in rows),
        all_full_gpu_adoption_gates_pass=all(row['comparisons']['full_gpu']['candidate_for_adoption'] for row in rows),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        memory_note='Process peak RSS and retained Metal buffer allocation lengths are different scopes; the latter is not physical VRAM, and unified-memory values must not be added.',
        transfer_note='upload_bytes counts source/template/level/range memcpy payloads and excludes setBytes parameter uploads; readback_bytes is the logical status/output payload. Neither is measured memory-bus traffic.',
        timing_note='Unavailable split stages are null and total_ms remains measured. device_ms includes only available timestamps; availability counts include first and measured calls. Legacy records without availability flags require finite nonnegative device_ms and assume split stages available.',
        parity_note='Exact ordering is diagnostic only. Independent one-to-one ground-truth accuracy is the correctness gate.',
        residency_note='Source-audited implementation counters, not a GPU hardware trace; upload and final readback remain necessary.')


def fingerprints(binary, cases):
    sources = [path for folder in ('src', 'include', 'experiments', 'tools', 'tests')
               for path in (ROOT/'cpp-core'/folder).rglob('*') if path.is_file() and
               (path.suffix in ('.cpp', '.hpp', '.h', '.mm', '.metal', '.py', '.in') or path.name == 'CMakeLists.txt')]
    sources += [ROOT/'cpp-core/CMakeLists.txt']
    binaries = [binary, *binary.parent.glob('*.dylib')]
    return dict(source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sorted(set(sources))},
                binary_sha256={str(path): sha(path) for path in sorted(set(binaries))},
                input_sha256={case[key]: sha(case[key]) for case in cases for key in ('source', 'template')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT/'build/stage37-full-gpu/formal/manifest.json')
    parser.add_argument('--cases', default=DEFAULT_CASES)
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=float, default=1.)
    parser.add_argument('--timeout', type=float, default=1200.)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        save(args.output, summarize(json.loads(args.summarize.read_text()), args.summarize.resolve().parent))
        return
    if args.blocks < 1 or args.repeats < 1 or not 0 <= args.warmup <= 60 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('positive blocks/repeats/timeout and warmup in 0..60 required')
    names = selected_names(args.cases, None, 'cases')
    methods = selected_names(args.methods, METHODS, 'methods')
    if not {'hybrid_best', 'full_gpu'} <= set(methods):
        parser.error('hybrid_best and full_gpu methods are required')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('inputs', 'jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    inputs = root/'inputs'
    inputs.mkdir(parents=True)
    cases = make_cases(names, inputs)
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    before = fingerprints(binary, cases)
    manifest = dict(version=1, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup,
        timeout_seconds=args.timeout, cases=cases, methods=methods, method_options={name: METHODS[name] for name in methods},
        executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        for case in cases:
            for block in range(args.blocks):
                for method in methods[::1 if block % 2 == 0 else -1]:
                    label = f"{case['name']}-{method}-b{block}"
                    job_path, result_path = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                    save(job_path, job(case, method, args.repeats, args.warmup))
                    print('Running '+label, flush=True)
                    try:
                        child = subprocess.run([str(binary), str(job_path)], capture_output=True, text=True, timeout=args.timeout)
                    except subprocess.TimeoutExpired as error:
                        job_path.with_suffix('.stdout.txt').write_text(error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or '')
                        job_path.with_suffix('.stderr.txt').write_text(error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or '')
                        raise
                    job_path.with_suffix('.stdout.txt').write_text(child.stdout)
                    job_path.with_suffix('.stderr.txt').write_text(child.stderr)
                    if child.returncode:
                        raise RuntimeError(label+': '+child.stderr[-4000:])
                    save(result_path, json.loads(child.stdout))
                    manifest['executions'].append(dict(case=case['name'], method=method, block=block,
                        job=str(job_path.relative_to(root)), job_sha256=sha(job_path),
                        result=str(result_path.relative_to(root)), result_sha256=sha(result_path)))
                    save(args.output, manifest)
        after = fingerprints(binary, cases)
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                        after_sha256=after, unchanged=before == after)
        save(args.output, manifest)
        manifest['summary'] = summarize(manifest, root)
        save(args.output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(args.output, manifest)
        raise


if __name__ == '__main__':
    main()
