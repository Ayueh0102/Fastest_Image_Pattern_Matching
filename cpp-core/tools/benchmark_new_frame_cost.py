"""Stage 41: per-new-frame cost of the hybrid path versus full-GPU fine4.

Round-40 formal calls repeat one unchanged source. The hybrid Session reuses
the source pose context its first call built, while the complete GPU path
rebuilds all source-dependent state on every call, so those timings omit a
cost the hybrid pays whenever the source changes. Here every timed call goes
through Session::updateSource: the runner's source_frames sequence mode
("reuse") lists one archived fixture and repeats it for 1+warmup+measured
cycles. Each call after the first copies a freshly decoded frame, releases the
hybrid pose context and rebuilds it on the next refinement. Round-40 static
jobs run in the same fresh-process blocks as a same-build control.

Sequence mode requires warmup_seconds=0. Call 0 constructs the Session and is
cold; the next warmup_calls (planned from round-40 medians) are warm-up. Both
are excluded from timing distributions but checked against independent truth.
The primary metric is frame_wall_ms: source update plus run, excluding image
decode, comparison and JSON. Truth stays in this parent, every worker and
failed gate is retained, and RSS and named Metal bytes are never added.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import benchmark_gpu_ncc_closure as stage40
from closure_lifecycle import valid_route

previous = stage40.previous
ROOT = stage40.ROOT
DOCS = ROOT/'cpp-core/docs'
HYBRID, GPU = 'hybrid_best', 'full_gpu_simd'
METHODS = {name: stage40.METHODS[name] for name in (HYBRID, GPU)}
MODES = ('frames', 'static')
ARMS = {f'{method}_{mode}': (method, mode) for mode in MODES for method in METHODS}
ARM_OPTIONS = {arm: dict(method=method, mode=mode, **METHODS[method]) for arm, (method, mode) in ARMS.items()}
DEFAULT_CASES = 'many128_pose,many512_pose,controlled6,8k_t256_angle180_3scales'
STATIC_WARMUP_SECONDS = 1.
# Planned warm-up per sequence process at the faster round-40 formal median;
# the observed warm-up wall time is gated at the round-40 one second.
WARMUP_TARGET_MS = 1500.
WARMUP_MINIMUM_MS = 1000.
SEQUENCE_TIMES = ('total_ms', 'frame_wall_ms', 'source_update_ms', 'construct_ms', 'release_ms',
                  'source_decode_ms', 'pose_create_ms', 'elapsed_ms')
SEQUENCE_COUNTS = ('source_generation', 'source_copy_bytes', 'retained_template_models', 'source_width', 'source_height')
# Builds, test drivers and other benchmark runners must not share the machine while timing.
BUSY = re.compile(r'^(cmake|ctest|make|gmake|ninja|clang|clang\+\+|c\+\+|cc|ld|swift-frontend|xcodebuild|metal|metallib|fpm_\w+)$')
sha, save, distribution, integer, evaluate = stage40.sha, stage40.save, stage40.distribution, stage40.integer, stage40.evaluate
finite = previous.finite_nonnegative


def warmup_calls(summary, names):
    """Warm-up calls per case: WARMUP_TARGET_MS at the faster round-40 formal median."""
    cases = {case['name']: case for case in summary['cases']}
    missing = set(names)-cases.keys()
    if missing:
        raise ValueError('Round-40 summary lacks cases: '+','.join(sorted(missing)))
    return {name: max(1, math.ceil(WARMUP_TARGET_MS/min(cases[name]['methods'][method]['total_ms']['median'] for method in METHODS)))
            for name in names}


def job(case, arm, warmup_calls, measured):
    method, mode = ARMS[arm]
    if mode == 'static':
        return stage40.job(case, method, measured, STATIC_WARMUP_SECONDS)
    # One frame repeated for 1+warmup+measured cycles; the runner compares every
    # cycle with call 0 and puts each later call through updateSource.
    result = stage40.job(case, method, 1+warmup_calls+measured, 0.)
    result.pop('source')
    result.update(source_frames=[case['source']], sequence_mode='reuse')
    return result


def outputs(record):
    return record['matches'], record['proposals']


def call_route(record, method, refine):
    """Residency/refinement route and whether the requested fine mode ran."""
    gpu = method == GPU
    route = previous.validate_record(record, gpu)
    if not gpu:
        return route and valid_route(record) and record['refinement_candidates'] == (len(record['proposals']) if refine else 0), True
    metrics = record['full_gpu']
    if type(metrics.get('peak_mode')) is not int:
        raise ValueError('Invalid peak_mode')
    stage40.fine_work_record(record, False)
    route &= (metrics['peak_mode'] == METHODS[GPU]['full_gpu_peak_mode'] and
              metrics['fine_mode'] == METHODS[GPU]['full_gpu_fine_mode'])
    return route, metrics['fine_effective_mode'] == METHODS[GPU]['full_gpu_fine_mode']


def sequence_calls(case, method, worker, warmup, measured):
    """Validate one reuse-sequence worker; return its rows and sequence-route flag."""
    calls = 1+warmup+measured
    if worker.get('sequence_mode') != 'reuse' or worker.get('source_frames') != [case['source']]:
        raise ValueError('Sequence worker mode or frame list differs from the job')
    if integer(worker.get('cycles'), 'cycles') != calls or integer(worker.get('calls'), 'calls') != calls:
        raise ValueError('Sequence worker call count differs from the job')
    rows = worker.get('runs')
    if not isinstance(rows, list) or len(rows) != calls:
        raise ValueError('Missing or extra sequence calls')
    for flag in ('exact_sequence_repeat', 'drained'):
        if type(worker.get(flag)) is not bool:
            raise ValueError('Missing '+flag+' boolean')
    integer(worker['final_memory']['peak_rss_bytes'], 'peak_rss_bytes')
    gpu, refine = method == GPU, case['workflow'] == 'full_rescue'
    route, shape = worker['drained'], None
    for call, row in enumerate(rows):
        if (row.get('call'), row.get('source_index'), row.get('sequence_cycle')) != (call, 0, call):
            raise ValueError('Wrong sequence call order')
        if row.get('source_path') != case['source'] or type(row.get('reference_cycle')) is not bool or row['reference_cycle'] != (call == 0):
            raise ValueError('Wrong frame identity or reference cycle')
        for field in SEQUENCE_TIMES:
            finite(row.get(field), field)
        for field in SEQUENCE_COUNTS:
            integer(row.get(field), field)
        for field in ('pose_invalidated', 'source_dimensions_changed'):
            if type(row.get(field)) is not bool:
                raise ValueError('Missing '+field+' boolean')
        if row['frame_wall_ms']+1e-6 < row['total_ms']+row['source_update_ms']+row['construct_ms']+row['release_ms']:
            raise ValueError('Frame wall time is shorter than its declared parts')
        # elapsed_ms is stamped after each record; decode precedes the next frame.
        start = rows[call-1]['elapsed_ms'] if call else 0.
        if row['elapsed_ms']+1e-6 < start+row['frame_wall_ms']+row['source_decode_ms']:
            raise ValueError('Elapsed time is shorter than decode plus frame wall time')
        shape = shape or (row['source_width'], row['source_height'])
        if (row['source_width'], row['source_height']) != shape:
            raise ValueError('Frame dimensions changed within a repeated frame')
        if call:
            route &= (row['construct_ms'] == 0 and row['release_ms'] == 0 and row['source_generation'] == call and
                      row['source_copy_bytes'] == shape[0]*shape[1] and row['retained_template_models'] == len(case['scales']) and
                      not row['source_dimensions_changed'] and row['pose_invalidated'] == (refine and not gpu))
        else:
            route &= (row['construct_ms'] > 0 and row['source_update_ms'] == 0 and row['source_generation'] == 0 and
                      row['source_copy_bytes'] == 0 and row['retained_template_models'] == 0 and
                      not row['pose_invalidated'] and not row['source_dimensions_changed'])
        # Every refining hybrid call rebuilds its pose context; GPU and core-only never create one.
        route &= row['pose_create_ms'] > 0 if refine and not gpu else row['pose_create_ms'] == 0
    return rows, bool(route)


def static_calls(case, method, worker, measured):
    """Validate one round-40 style worker; the hybrid must reuse its first-call pose context."""
    runs = worker.get('runs')
    if not isinstance(runs, list) or len(runs) != measured:
        raise ValueError('Missing or extra measured repeats')
    finite(worker.get('construct_ms'), 'construct_ms')
    integer(worker['final_memory']['peak_rss_bytes'], 'peak_rss_bytes')
    if type(worker.get('exact_repeat')) is not bool:
        raise ValueError('Missing exact_repeat boolean')
    warmups = worker.get('warmups_ms')
    if not isinstance(warmups, list) or not warmups:
        raise ValueError('Static control ran no warm-up call')
    for value in warmups:
        finite(value, 'warmups_ms')
    gpu, refine = method == GPU, case['workflow'] == 'full_rescue'
    route = True
    for index, record in enumerate([worker['first'], *runs]):
        finite(record.get('pose_create_ms'), 'pose_create_ms')
        integer(record.get('source_generation'), 'source_generation')
        route &= record['source_generation'] == 0 and (
            record['pose_create_ms'] > 0 if index == 0 and refine and not gpu else record['pose_create_ms'] == 0)
    return [worker['first'], *runs], route


def accuracy_summary(records, truth):
    accuracy = [evaluate(record['matches'], truth)['strict'] for record in records]
    return dict(all_gt_pass=all(row['fp'] == row['fn'] == 0 for row in accuracy),
                accuracy_counts=sorted({(row['tp'], row['fp'], row['fn']) for row in accuracy}),
                max_errors={name: max((row['max_errors'][name] for row in accuracy if row['max_errors'][name] is not None), default=None)
                            for name in ('position_px', 'angle_deg', 'relative_scale')})


def memory_summary(records, method):
    gpu = method == GPU
    return dict(named_gpu_bytes=sorted({record['full_gpu']['gpu_bytes'] for record in records}) if gpu else None,
                frame_gpu_bytes=None if gpu else sorted({record['frame_coarse_gpu_bytes']+record['frame_fine_gpu_bytes'] for record in records}),
                pose_gpu_bytes=None if gpu else sorted({record['pose_gpu_bytes'] for record in records}),
                pose_host_bytes=None if gpu else sorted({record['pose_host_bytes'] for record in records}),
                gpu_scope=('full_gpu.gpu_bytes: all named full-GPU Metal buffers' if gpu else
                           'frame coarse+fine buffers; pose context buffers listed separately; other owners excluded'))


def frames_arm(case, method, workers, warmup, measured, gate):
    sequences = [(block, worker, *sequence_calls(case, method, worker, warmup, measured)) for block, worker in workers]
    calls = [row for _, _, rows, _ in sequences for row in rows]
    timed = [row for _, _, rows, _ in sequences for row in rows[1+warmup:]]
    refine = case['workflow'] == 'full_rescue'
    routes = [call_route(row, method, refine) for row in calls]
    blocks = []
    for block, worker, rows, _ in sequences:
        timed_rows, warm_rows = rows[1+warmup:], rows[1:1+warmup]
        warmup_wall = rows[warmup]['elapsed_ms']-rows[0]['elapsed_ms']
        if warmup_wall < WARMUP_MINIMUM_MS:
            gate('warmup_shorter_than_1s', block)
        blocks.append(dict(block=block, frame_wall_median_ms=statistics.median(row['frame_wall_ms'] for row in timed_rows),
            total_median_ms=statistics.median(row['total_ms'] for row in timed_rows),
            warmup_calls=warmup, warmup_wall_ms=warmup_wall, warmup_frame_wall_ms=sum(row['frame_wall_ms'] for row in warm_rows),
            cold_frame_wall_ms=rows[0]['frame_wall_ms'], cold_total_ms=rows[0]['total_ms'], cold_construct_ms=rows[0]['construct_ms'],
            cold_pose_create_ms=rows[0]['pose_create_ms'], peak_rss_bytes=worker['final_memory']['peak_rss_bytes'],
            after_release_current_rss_bytes=worker.get('after_release_memory', {}).get('current_rss_bytes')))
    row = dict(method=method, mode='frames', **{name: distribution([call[name] for call in timed]) for name in
               ('frame_wall_ms', 'total_ms', 'source_update_ms', 'pose_create_ms', 'source_decode_ms')},
        blocks=blocks, cold_pose_create_ms=[rows[0]['pose_create_ms'] for _, _, rows, _ in sequences],
        warmup_pose_create_ms=distribution([call['pose_create_ms'] for _, _, rows, _ in sequences for call in rows[1:1+warmup]]),
        peak_rss_bytes=max(block['peak_rss_bytes'] for block in blocks),
        measured_current_rss_bytes=distribution([call['memory']['current_rss_bytes'] for call in timed]),
        **memory_summary(calls, method), **accuracy_summary(calls, case['truth']),
        route_valid=all(route for route, _ in routes), requested_algorithm_executed=all(executed for _, executed in routes),
        sequence_route_valid=all(route for _, _, _, route in sequences),
        repeat_exact=all(worker['exact_sequence_repeat'] for _, worker, _, _ in sequences) and all(outputs(call) == outputs(calls[0]) for call in calls),
        cold_calls=len(sequences), warmup_calls=len(sequences)*warmup, measured_calls=len(timed), gt_evaluated_calls=len(calls))
    return row, calls


def static_arm(case, method, workers, measured):
    checked = [(block, worker, *static_calls(case, method, worker, measured)) for block, worker in workers]
    calls = [record for _, _, records, _ in checked for record in records]
    timed = [record for _, worker, _, _ in checked for record in worker['runs']]
    routes = [call_route(record, method, case['workflow'] == 'full_rescue') for record in calls]
    blocks = [dict(block=block, total_median_ms=statistics.median(record['total_ms'] for record in worker['runs']),
                   first_total_ms=worker['first']['total_ms'], first_pose_create_ms=worker['first']['pose_create_ms'],
                   warmup_calls=len(worker['warmups_ms']), warmup_algorithm_ms=sum(worker['warmups_ms']),
                   construct_ms=worker['construct_ms'], peak_rss_bytes=worker['final_memory']['peak_rss_bytes'])
              for block, worker, _, _ in checked]
    row = dict(method=method, mode='static', total_ms=distribution([record['total_ms'] for record in timed]), blocks=blocks,
        first_pose_create_ms=[block['first_pose_create_ms'] for block in blocks],
        measured_pose_create_ms=distribution([record['pose_create_ms'] for record in timed]),
        peak_rss_bytes=max(block['peak_rss_bytes'] for block in blocks),
        **memory_summary(calls, method), **accuracy_summary(calls, case['truth']),
        route_valid=all(route for route, _ in routes), requested_algorithm_executed=all(executed for _, executed in routes),
        static_route_valid=all(route for _, _, _, route in checked),
        repeat_exact=all(worker['exact_repeat'] for _, worker, _, _ in checked) and all(outputs(record) == outputs(calls[0]) for record in calls),
        cold_calls=len(checked), warmup_calls=sum(block['warmup_calls'] for block in blocks), measured_calls=len(timed),
        gt_evaluated_calls=len(calls))
    return row, calls


def compare(hybrid, gpu, metric, block_field):
    """GPU against hybrid; a direction needs every paired block to agree with the pooled medians."""
    left, right = hybrid[metric]['median'], gpu[metric]['median']
    paired = [dict(block=a['block'], hybrid_ms=a[block_field], gpu_ms=b[block_field], delta_ms=b[block_field]-a[block_field],
                   delta_percent=100*(b[block_field]/a[block_field]-1),
                   within_regression_budget=b[block_field] <= a[block_field]+max(2., .1*a[block_field]))
              for a, b in zip(hybrid['blocks'], gpu['blocks'])]
    signs = {(row['gpu_ms'] > row['hybrid_ms'])-(row['gpu_ms'] < row['hybrid_ms']) for row in paired}
    pooled = (right > left)-(right < left)
    consistent = pooled != 0 and signs == {pooled}
    return dict(metric=metric, hybrid_median_ms=left, gpu_median_ms=right, delta_ms=right-left, delta_percent=100*(right/left-1),
                paired=paired, gpu_faster_blocks=sum(row['gpu_ms'] < row['hybrid_ms'] for row in paired), blocks=len(paired),
                consistent_direction=consistent, faster=(GPU if pooled < 0 else HYBRID) if consistent else None)


def host_entry(state):
    if not isinstance(state, dict) or not isinstance(state.get('competing_processes'), list):
        raise ValueError('Missing host state')
    for process in state['competing_processes']:
        if not isinstance(process, dict) or not isinstance(process.get('command'), str):
            raise ValueError('Invalid competing process record')
    return state['competing_processes']


def summarize(manifest, root):
    if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
        raise ValueError('Experiment must be complete with unchanged fingerprints')
    fingerprints = {name: manifest.get(name) for name in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(fingerprints.values()) or manifest.get('after_sha256') != fingerprints:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    phase = manifest.get('phase')
    if phase not in ('pilot', 'formal'):
        raise ValueError('Phase must be pilot or formal')
    arms = manifest.get('arms')
    if not isinstance(arms, list) or len(arms) != len(set(arms)) or set(arms) != ARMS.keys():
        raise ValueError('Arms must be exactly '+','.join(ARMS))
    if json.dumps(manifest.get('arm_options'), sort_keys=True) != json.dumps(ARM_OPTIONS, sort_keys=True):
        raise ValueError('Manifest arm_options differ from the declared arms')
    blocks, measured = integer(manifest.get('blocks'), 'blocks'), integer(manifest.get('measured_calls'), 'measured_calls')
    if blocks < 1 or measured < 1:
        raise ValueError('Empty blocks or measured calls')
    if manifest.get('static_warmup_seconds') != STATIC_WARMUP_SECONDS or type(manifest['static_warmup_seconds']) is not float:
        raise ValueError('Static control warm-up differs from the round-40 one second')
    if phase == 'formal' and (blocks < 3 or measured < 2):
        raise ValueError('Formal run requires at least 3 blocks and 2 measured calls per process')
    cases = {case['name']: case for case in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Empty or duplicate cases')
    warmups = manifest.get('warmup_calls')
    if not isinstance(warmups, dict) or warmups.keys() != cases.keys() or any(integer(value, 'warmup_calls') < 1 for value in warmups.values()):
        raise ValueError('Every case needs a positive integer warm-up call count')
    expected = {(name, arm, block) for name in cases for arm in arms for block in range(blocks)}
    entries, failures = {}, []
    for entry in manifest['executions']:
        integer(entry['block'], 'execution.block')
        key = (entry['case'], entry['arm'], entry['block'])
        if key not in expected or key in entries:
            raise ValueError('Unexpected or duplicate worker '+str(key))
        paths = {name: previous.worker_path(root, entry[name]) for name in ('job', 'result')}
        for name, path in paths.items():
            if sha(path) != entry[name+'_sha256']:
                raise ValueError('Changed '+name+': '+entry[name])
        # Serialized equality keeps bool/int mode aliases and leaked truth out.
        expected_job = job(cases[entry['case']], entry['arm'], warmups[entry['case']], measured)
        if json.dumps(json.loads(paths['job'].read_text()), sort_keys=True) != json.dumps(expected_job, sort_keys=True):
            raise ValueError('Worker job differs from the declared protocol')
        if host_entry(entry.get('host')):
            failures.append(dict(case=entry['case'], arm=entry['arm'], method=ARMS[entry['arm']][0], block=entry['block'], gate='host_busy'))
        entries[key] = json.loads(paths['result'].read_text())
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    rows = []
    for name, case in cases.items():
        arm_rows, calls = {}, {}
        for arm in ARMS:
            method, mode = ARMS[arm]
            def gate(gate_name, block=None, arm=arm, method=method):
                failures.append(dict(case=name, arm=arm, method=method, gate=gate_name, **({} if block is None else dict(block=block))))
            workers = [(block, entries[(name, arm, block)]) for block in range(blocks)]
            if mode == 'frames':
                arm_rows[arm], calls[arm] = frames_arm(case, method, workers, warmups[name], measured, gate)
            else:
                arm_rows[arm], calls[arm] = static_arm(case, method, workers, measured)
        for arm, (method, mode) in ARMS.items():
            row = arm_rows[arm]
            if mode == 'frames':
                # Same pixels and options: output must equal the same method's static control.
                reference = outputs(calls[f'{method}_static'][0])
                row['same_method_output_parity'] = all(outputs(call) == reference for call in calls[arm])
            checks = [('independent_ground_truth', row['all_gt_pass']), ('route_and_residency', row['route_valid']),
                      ('requested_algorithm_not_executed', row['requested_algorithm_executed']), ('exact_repeat', row['repeat_exact'])]
            checks += ([('sequence_route', row['sequence_route_valid']), ('same_method_output_parity', row['same_method_output_parity'])]
                       if mode == 'frames' else [('static_route', row['static_route_valid'])])
            failures.extend(dict(case=name, arm=arm, method=method, gate=gate_name) for gate_name, passed in checks if not passed)
        frames = {method: arm_rows[f'{method}_frames'] for method in METHODS}
        static = {method: arm_rows[f'{method}_static'] for method in METHODS}
        comparisons = dict(frames_frame_wall=compare(frames[HYBRID], frames[GPU], 'frame_wall_ms', 'frame_wall_median_ms'),
                           frames_run_only=compare(frames[HYBRID], frames[GPU], 'total_ms', 'total_median_ms'),
                           static_run_only=compare(static[HYBRID], static[GPU], 'total_ms', 'total_median_ms'))
        primary = comparisons['frames_frame_wall']
        for pair in primary['paired']:
            if not pair['within_regression_budget']:
                failures.append(dict(case=name, arm=f'{GPU}_frames', method=GPU, baseline=f'{HYBRID}_frames', gate='time_budget', block=pair['block']))
        hybrid_rss = frames[HYBRID]['peak_rss_bytes']
        if frames[GPU]['peak_rss_bytes'] > hybrid_rss+max(8*1024**2, .1*hybrid_rss):
            failures.append(dict(case=name, arm=f'{GPU}_frames', method=GPU, baseline=f'{HYBRID}_frames', gate='rss_budget'))
        reuse = {method: dict(run_ms=frames[method]['total_ms']['median']-static[method]['total_ms']['median'],
                              run_percent=100*(frames[method]['total_ms']['median']/static[method]['total_ms']['median']-1),
                              frame_wall_minus_static_run_ms=frames[method]['frame_wall_ms']['median']-static[method]['total_ms']['median'],
                              frame_wall_vs_static_run_percent=100*(frames[method]['frame_wall_ms']['median']/static[method]['total_ms']['median']-1))
                 for method in METHODS}
        first_pose = statistics.median(static[HYBRID]['first_pose_create_ms'])
        estimate = static[HYBRID]['total_ms']['median']+first_pose
        rows.append(dict(name=name, truth_count=len(case['truth']), refine=case['workflow'] == 'full_rescue',
            warmup_calls=warmups[name], arms=arm_rows, comparisons=comparisons, reuse_effect=reuse,
            same_build_estimate=dict(hybrid_static_median_ms=static[HYBRID]['total_ms']['median'],
                first_call_pose_create_median_ms=first_pose, estimate_ms=estimate,
                measured_run_ms=frames[HYBRID]['total_ms']['median'], measured_frame_wall_ms=frames[HYBRID]['frame_wall_ms']['median'],
                measured_run_minus_estimate_ms=frames[HYBRID]['total_ms']['median']-estimate,
                measured_frame_wall_minus_estimate_ms=frames[HYBRID]['frame_wall_ms']['median']-estimate,
                measured_pose_create_median_ms=frames[HYBRID]['pose_create_ms']['median'],
                gpu_static_vs_estimate_percent=100*(static[GPU]['total_ms']['median']/estimate-1),
                note='Round-40 method applied to this build: static hybrid median plus its first-call pose_create_ms median.'),
            decision=dict(formal_decision_eligible=phase == 'formal', metric='frame_wall_ms',
                faster_per_frame=primary['faster'], consistent_direction=primary['consistent_direction'],
                gpu_vs_hybrid_percent=primary['delta_percent'], run_only_faster=comparisons['frames_run_only']['faster'],
                run_only_gpu_vs_hybrid_percent=comparisons['frames_run_only']['delta_percent'],
                static_faster=comparisons['static_run_only']['faster'],
                static_gpu_vs_hybrid_percent=comparisons['static_run_only']['delta_percent'])))
    arm_list = [row['arms'][arm] for row in rows for arm in ARMS]
    return dict(phase=phase, formal_decision_eligible=phase == 'formal', cases=rows, failed_gates=failures,
        all_accuracy_pass=all(row['all_gt_pass'] for row in arm_list),
        all_routes_pass=all(row['route_valid'] and row['requested_algorithm_executed'] and
                            row.get('sequence_route_valid', row.get('static_route_valid')) for row in arm_list),
        all_repeat_exact=all(row['repeat_exact'] for row in arm_list),
        counts=dict(fresh_processes=len(entries), cold_calls=sum(row['cold_calls'] for row in arm_list),
            gt_checked_warmup_calls=sum(row['warmup_calls'] for row in arm_list if row['mode'] == 'frames'),
            native_only_warmup_calls=sum(row['warmup_calls'] for row in arm_list if row['mode'] == 'static'),
            measured_calls=sum(row['measured_calls'] for row in arm_list), gt_evaluated_calls=sum(row['gt_evaluated_calls'] for row in arm_list)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note=('Primary frame_wall_ms covers Session::updateSource (copy, pose release) plus run; it excludes decode, comparison and JSON. '
                     'total_ms is run only, the round-40 scope. Cold call 0 (Session construction) and warm-up calls are excluded from distributions. '
                     'A faster method is named only when every paired block agrees with the pooled medians.'),
        warmup_note=('Sequence mode forbids warmup_seconds; warmup_calls per case were planned for 1.5 s at the faster round-40 median, '
                     'and every process must show at least 1 s between the ends of call 0 and the last warm-up call.'),
        memory_note=('Peak process RSS, named Metal buffer lengths and pose-context buffers are separate scopes and are never added. '
                     'Sequence RSS includes per-call decoding and the runner retaining every output row.'),
        parity_note='Independent strict GT is evaluated on every cold, warm-up and measured call; exact same-method parity with the static control is a separate check.',
        frame_note='One archived fixture repeated: each updateSource copies a freshly decoded identical frame, so data-dependent work is unchanged; not a changing camera stream.')


def estimate_check(summary, archived_rows):
    """Compare each case's measured per-frame direction with the archive-only round-40 estimate."""
    archived = {row['case']: row for row in archived_rows}
    result = {}
    for case in summary['cases']:
        row = archived.get(case['name'])
        if row is None or GPU not in row['methods']:
            result[case['name']] = None
            continue
        estimate = row['methods'][GPU]['vs_hybrid_with_pose_create_percent']
        primary = case['comparisons']['frames_frame_wall']
        result[case['name']] = dict(archived_estimate_percent=estimate,
            archived_static_percent=row['methods'][GPU]['vs_hybrid_percent'],
            archived_hybrid_with_pose_create_ms=row['hybrid_with_pose_create_ms'], archived_gpu_median_ms=row['methods'][GPU]['median_ms'],
            archived_first_call_pose_create_median_ms=row['first_call_pose_create_median_ms'],
            measured_percent=primary['delta_percent'], measured_run_only_percent=case['comparisons']['frames_run_only']['delta_percent'],
            measured_minus_estimate_points=primary['delta_percent']-estimate,
            measured_pose_create_median_ms=case['arms'][f'{HYBRID}_frames']['pose_create_ms']['median'],
            consistent_direction=primary['consistent_direction'],
            direction_held=((primary['delta_percent'] < 0) == (estimate < 0)) if primary['consistent_direction'] else None,
            note='Estimate is round-40 archive arithmetic from another build; only its direction is judged, the difference is reported.')
    return result


def host_state(samples=1):
    """Load average, GPU device utilization and competing build/benchmark process names."""
    utilization = []
    for index in range(samples):
        if index:
            time.sleep(.5)
        listing = subprocess.run(['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'], capture_output=True, text=True).stdout
        utilization += [int(value) for value in re.findall(r'"Device Utilization %"=(\d+)', listing)]
    competing = []
    for line in subprocess.run(['ps', '-Ao', 'pid=,pcpu=,comm=,args='], capture_output=True, text=True).stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) < 3 or int(fields[0]) == os.getpid():
            continue
        command = Path(fields[2]).name
        script = re.search(r'(benchmark_\w+|soak_\w+|profile_\w+)\.py', fields[3] if len(fields) > 3 else '')
        if BUSY.match(command) or (command.startswith('python') and script):
            competing.append(dict(command=script.group(0) if script and command.startswith('python') else command, pcpu=float(fields[1])))
    return dict(utc=datetime.now(timezone.utc).isoformat(), load_average=list(os.getloadavg()),
                gpu_device_utilization_percent=utilization, competing_processes=competing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/stage41-release')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures-manifest', type=Path, default=ROOT/'build/stage37-full-gpu/formal/manifest.json')
    parser.add_argument('--round40-summary', type=Path, default=DOCS/'stage40-ncc-closure/formal-summary.json')
    parser.add_argument('--cases', default=DEFAULT_CASES)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--measured', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--pilot', action='store_true', help='Screen only; never per-frame decision evidence')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--round40-raw', type=Path, help='With --summarize: compare with the archive-only round-40 estimate')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        summary = summarize(manifest, args.summarize.resolve().parent)
        if args.round40_raw:
            import audit_full_gpu_disclosure as audit
            summary['round40_estimate'] = estimate_check(summary, audit.pose_context(args.round40_raw, 'formal'))
            summary['round40_raw_sha256'] = sha(args.round40_raw)
        save(args.output, summary)
        return
    if args.round40_raw:
        parser.error('--round40-raw is available only with --summarize')
    if args.blocks < 1 or args.measured < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Positive blocks, measured calls and timeout required')
    if not args.pilot and (args.blocks < 3 or args.measured < 2):
        parser.error('Formal requires at least 3 blocks and 2 measured calls per process; use --pilot to screen')
    names = previous.selected_names(args.cases, None, 'cases')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    cases = stage40.fixture_cases(args.fixtures_manifest, names)
    warmups = warmup_calls(json.loads(args.round40_summary.read_text()), names)
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    before = previous.fingerprints(binary, cases)
    host = host_state(samples=5)
    if not args.pilot and host['competing_processes']:
        raise RuntimeError('Competing build or benchmark processes: '+json.dumps(host['competing_processes']))
    manifest = dict(version=1, phase='pilot' if args.pilot else 'formal', platform=platform.platform(),
        started_utc=datetime.now(timezone.utc).isoformat(), blocks=args.blocks, measured_calls=args.measured,
        static_warmup_seconds=STATIC_WARMUP_SECONDS, warmup_calls=warmups,
        warmup_rule=f'max(1, ceil({WARMUP_TARGET_MS:g} ms / faster round-40 formal median)); observed gate {WARMUP_MINIMUM_MS:g} ms',
        round40_summary=str(args.round40_summary.resolve()), round40_summary_sha256=sha(args.round40_summary),
        timeout_seconds=args.timeout, cases=cases, arms=list(ARMS), arm_options=ARM_OPTIONS,
        fixtures_manifest=str(args.fixtures_manifest.resolve()), fixtures_manifest_sha256=sha(args.fixtures_manifest),
        frame_choice='source_frames lists the archived fixture once and repeats it for 1+warmup+measured cycles',
        host=dict(before=host), executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        for case in cases:
            for block in range(args.blocks):
                for arm in list(ARMS)[::1 if block % 2 == 0 else -1]:
                    label = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                    save(jp, job(case, arm, warmups[case['name']], args.measured))
                    state = host_state()
                    print('Running '+label, flush=True)
                    try:
                        child = subprocess.run([str(binary), str(jp)], capture_output=True, text=True, timeout=args.timeout)
                    except subprocess.TimeoutExpired as error:
                        jp.with_suffix('.stdout.txt').write_text(error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or '')
                        jp.with_suffix('.stderr.txt').write_text(error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or '')
                        raise
                    jp.with_suffix('.stdout.txt').write_text(child.stdout)
                    jp.with_suffix('.stderr.txt').write_text(child.stderr)
                    if child.returncode:
                        raise RuntimeError(label+': '+child.stderr[-4000:])
                    save(rp, json.loads(child.stdout))
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block,
                        job=str(jp.relative_to(root)), job_sha256=sha(jp), result=str(rp.relative_to(root)), result_sha256=sha(rp), host=state))
                    save(args.output, manifest)
        after = previous.fingerprints(binary, cases)
        manifest['host']['after'] = host_state(samples=5)
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), after_sha256=after, unchanged=before == after)
        save(args.output, manifest)
        manifest['summary'] = summarize(manifest, root)
        save(args.output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(args.output, manifest)
        raise


if __name__ == '__main__':
    main()
