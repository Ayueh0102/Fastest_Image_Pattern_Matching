"""Stage 48: where the round-41 per-new-frame overheads come from.

Round 41 measured, per new frame (every call through Session::updateSource with
a freshly decoded frame), more hybrid time than the pose-context rebuild explains
(search +6/+30 ms at 128/512 objects, refinement +1/+7 ms) and a full-GPU fine4
device-time rise on the 8K scene. Every new-frame call there was also preceded
by the runner decoding the frame (65-131 ms with the GPU idle), while the static
control ran back to back. This round separates the two with the runner's
sequence options: every arm is a source_frames sequence of one archived fixture
in static mode (source and pose context kept from call 0) or reuse mode
(updateSource on every later call), with no gap, an idle sleep matched to the
round-41 decode time, or the decode itself between calls.

Phases: pilot and formal (five conditions per method; round-41 hybrid_best and
full_gpu_simd on three cases, plus the current full_gpu_parallel recommendation
on 8K), dose (static sleeps of 0-300 ms), profile (full-GPU timestamps, kept
apart from timing). Call 0 constructs the Session and the next warm-up calls
(round-41 counts) are excluded from timing but checked against independent
truth, as is every measured call. Hypothesis verdicts follow fixed rules on
effects whose direction every paired block agrees with. Every worker and failed
gate is retained; RSS and named Metal bytes are never added.
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

import benchmark_new_frame_cost as stage41
import benchmark_parallel_peaks as stage42
from closure_lifecycle import valid_route

stage40, previous = stage41.stage40, stage41.previous
ROOT, DOCS = stage41.ROOT, stage41.DOCS
HYBRID, G1, G2 = 'hybrid_best', 'full_gpu_simd', 'full_gpu_parallel'
METHODS = {HYBRID: stage40.METHODS[HYBRID], G1: stage40.METHODS[G1], G2: stage42.METHODS[G2]}
GPU_METHODS = {G1, G2}
CONDITIONS = {'static_none': ('static', 'none'), 'static_sleep': ('static', 'sleep'), 'static_decode': ('static', 'decode'),
              'reuse_none': ('reuse', 'none'), 'reuse_decode': ('reuse', 'decode')}
C128, C512, C8K = 'many128_pose', 'many512_pose', '8k_t256_angle180_3scales'
CASE_METHODS = {C128: (HYBRID, G1), C512: (HYBRID, G1), C8K: (HYBRID, G1, G2)}
PHASE_CASES = dict(pilot=[C128, C512, C8K], formal=[C128, C512, C8K], dose=[C512, C8K], profile=[C8K])
DOSE_METHODS = {C512: (HYBRID,), C8K: (G1, G2)}
DOSE_SLEEP_MS = (10, 30, 100, 300)
PROFILE_METHODS = {C8K: (G1,)}
PHASE_MINIMUMS = dict(pilot=(1, 1), formal=(3, 2), dose=(3, 2), profile=(1, 1))
PHASE_DEFAULTS = dict(pilot=(1, 3), formal=(6, 3), dose=(3, 3), profile=(2, 3))
WARMUP_CALLS = {name: stage42.FRAME_WARMUP_CALLS[name] for name in (C128, C512, C8K)}
# Round-41 hybrid per-frame decode medians, rounded: the sleep matches the gap it replaces.
SLEEP_MS = {C128: 65, C512: 130, C8K: 131}
WARMUP_MINIMUM_MS = stage41.WARMUP_MINIMUM_MS
SLEEP_TOLERANCE_MS = 50.  # longer oversleep is a gap-contract failure
NO_GAP_MAXIMUM_MS = 1.
# 1-minute load average before a worker; above it the worker is a host_load_high failure.
HOST_LOAD_LIMIT = 4.
# Round-41 detection missed framework Python (ucomm "Python") and Blender renders.
BUSY_TOOLS = re.compile(r'^(cmake|ctest|make|gmake|ninja|clang|clang\+\+|c\+\+|cc|ld|swift-frontend|xcodebuild|metal|metallib|blender|fpm_\w+)$', re.I)
BUSY_SCRIPTS = re.compile(r'\b(?:benchmark|testbed|soak|profile|diagnose|challenge)_\w+\.py\b')
BUSY_CPU_PERCENT = 100.
CONTRASTS = dict(round41=('reuse_decode', 'static_none'), update=('reuse_none', 'static_none'),
                 decode_gap=('static_decode', 'static_none'), sleep_gap=('static_sleep', 'static_none'),
                 gap_on_update=('reuse_decode', 'reuse_none'))
HYBRID_METRICS = ('total_ms', 'frame_wall_ms', 'prepare_ms', 'proposal_ms', 'coarse_device_ms', 'coarse_wait_ms', 'fine_wait_ms',
                  'fine_ncc_ms', 'cpu_normalization_ms', 'refine_ms', 'pose_create_ms', 'refine_minus_pose_ms',
                  'refinement_device_ms', 'source_update_ms')
GPU_METRICS = ('total_ms', 'frame_wall_ms', 'device_ms', 'wait_ms', 'upload_ms', 'encode_ms', 'pose_create_ms', 'source_update_ms')
SEQUENCE_TIMES = (*stage41.SEQUENCE_TIMES, 'gap_ms')
sha, save, distribution, integer, evaluate = stage40.sha, stage40.save, stage40.distribution, stage40.integer, stage40.evaluate
finite = previous.finite_nonnegative


def spec(method, mode, gap, sleep_ms=0):
    return dict(method=method, mode=mode, gap=gap, sleep_ms=sleep_ms if gap == 'sleep' else 0)


def condition_of(item):
    return f"{item['mode']}_{item['gap']}"


def arm_name(item, dose=False):
    return f"{item['method']}__{condition_of(item)}{item['sleep_ms'] if dose and item['gap'] == 'sleep' else ''}"


def parse_arm(name):
    """Spec of an arm whose name carries its sleep length (dose arms and gapless/decode arms)."""
    method, _, condition = name.partition('__')
    match = re.fullmatch(r'(static|reuse)_(none|decode|sleep)(\d*)', condition)
    if method not in METHODS or not match or (match[2] == 'sleep') != bool(match[3]):
        raise ValueError('Arm name does not carry a complete spec: '+name)
    return spec(method, match[1], match[2], int(match[3]) if match[3] else 0)


def phase_arms(phase, case):
    """(name, spec) per arm in run order: conditions outer, methods inner."""
    if case not in PHASE_CASES[phase]:
        raise ValueError(case+' is not a '+phase+' case')
    if phase == 'dose':
        items = [spec(method, 'static', 'sleep', ms) if ms else spec(method, 'static', 'none')
                 for ms in (0, *DOSE_SLEEP_MS) for method in DOSE_METHODS[case]]
        return [(arm_name(item, True), item) for item in items]
    methods = PROFILE_METHODS[case] if phase == 'profile' else CASE_METHODS[case]
    return [(arm_name(item), item) for item in (spec(method, *CONDITIONS[name], SLEEP_MS[case]) for name in CONDITIONS for method in methods)]


def job(case, item, warmup_calls, measured, profile=False):
    # The round-41 frames job (reuse, decode gap) plus this arm's sequence keys.
    result = stage41.job(case, f"{HYBRID if item['method'] == HYBRID else G1}_frames", warmup_calls, measured)
    result.update(METHODS[item['method']])
    result['full_gpu_profiling'] = profile
    if item['mode'] != 'reuse':
        result['sequence_mode'] = item['mode']
    if item['gap'] != 'decode':
        result['sequence_gap'] = item['gap']
    if item['gap'] == 'sleep':
        result['sequence_gap_sleep_ms'] = float(item['sleep_ms'])
    return result


def sleep_from_round41(summary):
    """Rounded round-41 hybrid frames decode medians for this round's cases."""
    cases = {row['name']: row for row in summary['cases']}
    return {name: round(cases[name]['arms']['hybrid_best_frames']['source_decode_ms']['median']) for name in SLEEP_MS}


def sequence_calls(case, item, worker, warmup, measured):
    """Validate one sequence worker against its arm; return rows, route and gap-contract flags."""
    calls = 1+warmup+measured
    sleep = float(item['sleep_ms']) if item['gap'] == 'sleep' else None
    if (worker.get('sequence_mode'), worker.get('sequence_gap'), worker.get('sequence_gap_sleep_ms')) != (item['mode'], item['gap'], sleep):
        raise ValueError('Sequence worker mode, gap or sleep differs from its arm')
    if worker.get('source_frames') != [case['source']]:
        raise ValueError('Sequence worker frame list differs from the job')
    if integer(worker.get('cycles'), 'cycles') != calls or integer(worker.get('calls'), 'calls') != calls:
        raise ValueError('Sequence worker call count differs from the job')
    rows = worker.get('runs')
    if not isinstance(rows, list) or len(rows) != calls:
        raise ValueError('Missing or extra sequence calls')
    for flag in ('exact_sequence_repeat', 'drained'):
        if type(worker.get(flag)) is not bool:
            raise ValueError('Missing '+flag+' boolean')
    integer(worker['final_memory']['peak_rss_bytes'], 'peak_rss_bytes')
    gpu, refine, reuse = item['method'] in GPU_METHODS, case['workflow'] == 'full_rescue', item['mode'] == 'reuse'
    route, gap_ok, shape = worker['drained'], True, None
    for call, row in enumerate(rows):
        if (row.get('call'), row.get('source_index'), row.get('sequence_cycle')) != (call, 0, call):
            raise ValueError('Wrong sequence call order')
        if row.get('source_path') != case['source'] or type(row.get('reference_cycle')) is not bool or row['reference_cycle'] != (call == 0):
            raise ValueError('Wrong frame identity or reference cycle')
        if row.get('sequence_gap') != item['gap']:
            raise ValueError('Row gap differs from its arm')
        for field in SEQUENCE_TIMES:
            finite(row.get(field), field)
        for field in stage41.SEQUENCE_COUNTS:
            integer(row.get(field), field)
        for field in ('pose_invalidated', 'source_dimensions_changed'):
            if type(row.get(field)) is not bool:
                raise ValueError('Missing '+field+' boolean')
        if row['frame_wall_ms']+1e-6 < row['total_ms']+row['source_update_ms']+row['construct_ms']+row['release_ms']:
            raise ValueError('Frame wall time is shorter than its declared parts')
        # elapsed_ms is stamped after each record; the gap precedes the frame.
        start = rows[call-1]['elapsed_ms'] if call else 0.
        if row['elapsed_ms']+1e-6 < start+row['gap_ms']+row['frame_wall_ms']:
            raise ValueError('Elapsed time is shorter than gap plus frame wall time')
        shape = shape or (row['source_width'], row['source_height'])
        if (row['source_width'], row['source_height']) != shape:
            raise ValueError('Frame dimensions changed within a repeated frame')
        if call:
            route &= row['construct_ms'] == 0 and row['release_ms'] == 0 and not row['source_dimensions_changed']
            route &= (row['source_generation'] == (call if reuse else 0) and row['source_copy_bytes'] == (shape[0]*shape[1] if reuse else 0) and
                      row['retained_template_models'] == (len(case['scales']) if reuse else 0) and
                      row['pose_invalidated'] == (reuse and refine and not gpu) and (reuse or row['source_update_ms'] == 0))
            if item['gap'] == 'decode':
                gap_ok &= row['source_decode_ms'] > 0 and row['gap_ms'] >= row['source_decode_ms']
            elif item['gap'] == 'sleep':
                gap_ok &= row['source_decode_ms'] == 0 and sleep <= row['gap_ms'] <= sleep+SLEEP_TOLERANCE_MS
            else:
                gap_ok &= row['source_decode_ms'] == 0 and row['gap_ms'] <= NO_GAP_MAXIMUM_MS
        else:
            route &= (row['construct_ms'] > 0 and row['source_update_ms'] == 0 and row['source_generation'] == 0 and
                      row['source_copy_bytes'] == 0 and row['retained_template_models'] == 0 and
                      not row['pose_invalidated'] and not row['source_dimensions_changed'])
            gap_ok &= row['gap_ms'] == 0 and row['source_decode_ms'] == 0
        # Refining hybrid calls build a pose context at call 0 and after every updateSource.
        creates = refine and not gpu and (call == 0 or reuse)
        route &= row['pose_create_ms'] > 0 if creates else row['pose_create_ms'] == 0
    return rows, bool(route), bool(gap_ok)


def call_route(record, method, refine, profile):
    """Residency/refinement route, whether fine mode 4 ran, and the validated profile record."""
    gpu = method in GPU_METHODS
    route = previous.validate_record(record, gpu)
    if not gpu:
        return route and valid_route(record) and record['refinement_candidates'] == (len(record['proposals']) if refine else 0), True, None
    metrics = record['full_gpu']
    if type(metrics.get('peak_mode')) is not int:
        raise ValueError('Invalid peak_mode')
    stage40.fine_work_record(record, profile)
    profiled = stage40.profile_record(record, profile)
    route &= (metrics['peak_mode'] == METHODS[method]['full_gpu_peak_mode'] and
              metrics['fine_mode'] == METHODS[method]['full_gpu_fine_mode'])
    return route, metrics['fine_effective_mode'] == METHODS[method]['full_gpu_fine_mode'], profiled


def metrics_of(record, method):
    """Per-call values; full-GPU device_ms is None when the device timestamp is unavailable."""
    values = {name: record[name] for name in ('total_ms', 'frame_wall_ms', 'source_update_ms', 'pose_create_ms', 'gap_ms', 'source_decode_ms')}
    if method in GPU_METHODS:
        gpu = record['full_gpu']
        values.update(device_ms=gpu['device_ms'] if previous.device_timestamp_available(gpu) else None,
                      **{name: gpu[name] for name in ('wait_ms', 'upload_ms', 'encode_ms')})
        return values
    work = record['work']
    for name in ('prepare_ms', 'proposal_ms', 'refine_ms', 'refinement_device_ms'):
        values[name] = finite(record[name], name)
    values['refine_minus_pose_ms'] = record['refine_ms']-record['pose_create_ms']
    for name, field in (('coarse_device_ms', 'device_ms'), ('coarse_wait_ms', 'wait_ms'), ('fine_wait_ms', 'fine_wait_ms'),
                        ('fine_ncc_ms', 'fine_ncc_ms'), ('cpu_normalization_ms', 'normalization_ms')):
        values[name] = sum(finite(model[field], 'work.'+field) for model in work)
    return values


def competing_processes(names, args, own):
    """Build/benchmark tools, repository experiment scripts and any other process using a full core.

    names maps pid to (pcpu, ucomm) and args maps pid to the argument line; own pids are skipped.
    """
    found = []
    for pid, (pcpu, name) in sorted(names.items()):
        if pid in own:
            continue
        script = BUSY_SCRIPTS.search(args.get(pid, '')) if name.lower().startswith('python') else None
        reason = 'tool' if BUSY_TOOLS.match(name) else 'script' if script else 'cpu' if pcpu >= BUSY_CPU_PERCENT else None
        if reason:
            found.append(dict(command=script.group(0) if reason == 'script' else name, pcpu=pcpu, reason=reason))
    return found


def host_state(samples=1):
    """Load average, GPU device utilization samples and competing processes (names only, no arguments)."""
    utilization = []
    for index in range(samples):
        if index:
            time.sleep(.5)
        listing = subprocess.run(['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'], capture_output=True, text=True).stdout
        utilization += [int(value) for value in re.findall(r'"Device Utilization %"=(\d+)', listing)]
    names, args = {}, {}
    for line in subprocess.run(['ps', '-Ao', 'pid=,pcpu=,ucomm='], capture_output=True, text=True).stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3:
            names[int(fields[0])] = (float(fields[1]), fields[2].strip())
    for line in subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout.splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2:
            args[int(fields[0])] = fields[1]
    return dict(utc=datetime.now(timezone.utc).isoformat(), load_average=list(os.getloadavg()),
                gpu_device_utilization_percent=utilization, competing_processes=competing_processes(names, args, {os.getpid()}))


def accuracy_summary(records, truth):
    accuracy = [evaluate(record['matches'], truth)['strict'] for record in records]
    return dict(all_gt_pass=all(row['fp'] == row['fn'] == 0 for row in accuracy),
                accuracy_counts=sorted({(row['tp'], row['fp'], row['fn']) for row in accuracy}),
                max_errors={name: max((row['max_errors'][name] for row in accuracy if row['max_errors'][name] is not None), default=None)
                            for name in ('position_px', 'angle_deg', 'relative_scale')})


def arm_summary(case, item, workers, warmup, profile, gate):
    checked = [(block, worker, *sequence_calls(case, item, worker, warmup, len(worker['runs'])-1-warmup)) for block, worker in workers]
    calls = [row for _, _, rows, _, _ in checked for row in rows]
    refine = case['workflow'] == 'full_rescue'
    routes = [call_route(row, item['method'], refine, profile) for row in calls]
    names = GPU_METRICS if item['method'] in GPU_METHODS else HYBRID_METRICS
    blocks, timed = [], []
    for block, worker, rows, route, gap_ok in checked:
        warmup_wall = rows[warmup]['elapsed_ms']-rows[0]['elapsed_ms']
        for gate_name, failed in (('warmup_shorter_than_1s', warmup_wall < WARMUP_MINIMUM_MS), ('sequence_route', not route),
                                  ('gap_contract', not gap_ok)):
            if failed:
                gate(gate_name, block)
        values = [metrics_of(row, item['method']) for row in rows[1+warmup:]]
        timed += values
        blocks.append(dict(block=block, warmup_calls=warmup, warmup_wall_ms=warmup_wall,
                           medians={name: statistics.median(row[name] for row in values) if all(row[name] is not None for row in values) else None
                                    for name in names},
                           peak_rss_bytes=worker['final_memory']['peak_rss_bytes']))
    if any(row[name] is None for row in timed for name in names):
        gate('device_timestamp_unavailable')
    profiles = [result[2] for result, row in zip(routes, calls) if result[2] is not None]
    # Stage times keep the protocol's metric scope: measured calls only, not cold or warm-up calls.
    measured = [index > warmup for _, _, rows, _, _ in checked for index in range(len(rows))]
    timed_profiles = [result[2] for result, use in zip(routes, measured) if use and result[2] is not None]
    reference = calls[0]
    row = dict(spec=item, blocks=blocks,
               metrics={name: distribution([value[name] for value in timed if value[name] is not None]) if any(value[name] is not None for value in timed) else None
                        for name in names},
               gap_ms=distribution([value['gap_ms'] for value in timed]), source_decode_ms=distribution([value['source_decode_ms'] for value in timed]),
               peak_rss_bytes=max(block['peak_rss_bytes'] for block in blocks),
               measured_current_rss_bytes=distribution([call['memory']['current_rss_bytes'] for _, _, rows, _, _ in checked for call in rows[1+warmup:]]),
               named_gpu_bytes=sorted({call['full_gpu']['gpu_bytes'] for call in calls}) if item['method'] in GPU_METHODS else None,
               frame_gpu_bytes=None if item['method'] in GPU_METHODS else sorted({call['frame_coarse_gpu_bytes']+call['frame_fine_gpu_bytes'] for call in calls}),
               pose_gpu_bytes=None if item['method'] in GPU_METHODS else sorted({call['pose_gpu_bytes'] for call in calls}),
               **accuracy_summary(calls, case['truth']),
               route_valid=all(result[0] for result in routes), requested_algorithm_executed=all(result[1] for result in routes),
               sequence_route_valid=all(route for _, _, _, route, _ in checked), gap_contract_valid=all(gap_ok for _, _, _, _, gap_ok in checked),
               repeat_exact=all(worker['exact_sequence_repeat'] for _, worker, _, _, _ in checked) and
                   all((call['matches'], call['proposals']) == (reference['matches'], reference['proposals']) for call in calls),
               cold_calls=len(checked), warmup_calls=len(checked)*warmup, measured_calls=len(timed), gt_evaluated_calls=len(calls))
    if profile:
        stages = sorted({name for record in timed_profiles for name in record['stages']})
        row['profile_stages'] = {name: distribution([record['stages'][name]['gpu_ms'] for record in timed_profiles if name in record['stages']]) for name in stages}
        row['profile_available_calls'] = len(profiles)
    return row, calls


def effect(arm, baseline, metric):
    """Arm minus baseline on pooled medians; a direction needs every paired block to agree."""
    if arm['metrics'][metric] is None or baseline['metrics'][metric] is None:
        return dict(delta_ms=None, direction=None, paired=[])
    left, right = baseline['metrics'][metric]['median'], arm['metrics'][metric]['median']
    paired = [dict(block=a['block'], delta_ms=b['medians'][metric]-a['medians'][metric])
              for a, b in zip(baseline['blocks'], arm['blocks']) if a['medians'][metric] is not None and b['medians'][metric] is not None]
    delta = right-left
    pooled = (delta > 0)-(delta < 0)
    signs = {(row['delta_ms'] > 0)-(row['delta_ms'] < 0) for row in paired}
    complete = len(paired) == len(arm['blocks'])
    return dict(baseline_ms=left, arm_ms=right, delta_ms=delta, delta_percent=100*delta/left if left else None, paired=paired,
                direction=('up' if pooled > 0 else 'down') if pooled and complete and signs == {pooled} else None)


def hypotheses(rows):
    """Pre-registered verdict rules on the formal (or pilot) effects."""
    cases = {row['name']: row for row in rows}
    def direction(case, method, contrast, metric):
        effects = cases.get(case, {}).get('effects', {}).get(method)
        return effects[contrast][metric]['direction'] if effects else None
    def up(*args):
        return direction(*args) == 'up'
    reproduced = dict(hybrid_128_total=up(C128, HYBRID, 'round41', 'total_ms'), hybrid_512_total=up(C512, HYBRID, 'round41', 'total_ms'),
                      full_gpu_8k_device=up(C8K, G1, 'round41', 'device_ms'))
    sleep = dict(hybrid_512_coarse=up(C512, HYBRID, 'sleep_gap', 'coarse_device_ms'), full_gpu_8k_device=up(C8K, G1, 'sleep_gap', 'device_ms'))
    decode = dict(hybrid_512_coarse=up(C512, HYBRID, 'decode_gap', 'coarse_device_ms'), full_gpu_8k_device=up(C8K, G1, 'decode_gap', 'device_ms'))
    h1 = ('supported' if all(sleep.values()) else 'decode_only' if all(decode.values()) else
          'rejected' if not any([*sleep.values(), *decode.values()]) else 'mixed')
    def update_only(metric):
        update = up(C512, HYBRID, 'update', metric)
        gaps = up(C512, HYBRID, 'decode_gap', metric) or up(C512, HYBRID, 'sleep_gap', metric)
        return dict(update=update, gap=gaps), 'rejected' if not update else 'confounded' if gaps else 'supported'
    h2_inputs, h2 = update_only('refinement_device_ms')
    h3_inputs, h3 = update_only('prepare_ms')
    def attribution(method):
        update = up(C8K, method, 'update', 'device_ms')
        gap = up(C8K, method, 'sleep_gap', 'device_ms') or up(C8K, method, 'decode_gap', 'device_ms')
        return dict(verdict='both' if update and gap else 'update' if update else 'gap' if gap else 'neither',
                    inputs=dict(update=update, gap=gap))
    return dict(
        round41_reproduced=dict(reproduced=all(reproduced.values()), inputs=reproduced,
            rule='reuse_decode minus static_none is up in every block for hybrid total_ms at 128 and 512 and full_gpu_simd device_ms at 8K'),
        H1_gap_slows_first_gpu_stage=dict(verdict=h1, inputs=dict(sleep=sleep, decode=decode),
            rule='supported: sleep gap raises hybrid 512 coarse device time and 8K full_gpu_simd device time; decode_only: only the decode gap does; rejected: no gap raises either'),
        H2_new_pose_buffers_slow_refinement=dict(verdict=h2, inputs=h2_inputs,
            rule='512 hybrid refinement_device_ms: update up and no gap up = supported; update not up = rejected; both = confounded'),
        H3_new_source_slows_frame_prepare=dict(verdict=h3, inputs=h3_inputs,
            rule='512 hybrid prepare_ms: update up and no gap up = supported; update not up = rejected; both = confounded'),
        H4_8k_full_gpu_attribution=dict(**attribution(G1), rule='8K full_gpu_simd device_ms raised by update, by a gap, both or neither'),
        S1_8k_peak2_attribution=dict(**attribution(G2), rule='Same as H4 for the current full_gpu_parallel recommendation (supplementary)'))


def dose_summary(case, arm_rows):
    result = {}
    for method in DOSE_METHODS[case]:
        metric = 'device_ms' if method in GPU_METHODS else 'coarse_device_ms'
        names = [arm_name(spec(method, 'static', 'sleep', ms) if ms else spec(method, 'static', 'none'), True) for ms in (0, *DOSE_SLEEP_MS)]
        values = [arm_rows[name]['metrics'][metric] for name in names]
        result[method] = {'levels_ms': [0, *DOSE_SLEEP_MS], 'metric': metric, metric: values,
                          'total_ms': [arm_rows[name]['metrics']['total_ms'] for name in names],
                          'monotone_nondecreasing': all(a['median'] <= b['median'] for a, b in zip(values, values[1:])) if all(values) else None,
                          'top_vs_none': effect(arm_rows[names[-1]], arm_rows[names[0]], metric)}
    return result


def summarize(manifest, root):
    if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
        raise ValueError('Experiment must be complete with unchanged fingerprints')
    fingerprints = {name: manifest.get(name) for name in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(fingerprints.values()) or manifest.get('after_sha256') != fingerprints:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    phase = manifest.get('phase')
    if phase not in PHASE_CASES:
        raise ValueError('Unknown phase')
    blocks, measured = integer(manifest.get('blocks'), 'blocks'), integer(manifest.get('measured_calls'), 'measured_calls')
    if blocks < PHASE_MINIMUMS[phase][0] or measured < PHASE_MINIMUMS[phase][1]:
        raise ValueError(phase+' needs at least %d blocks and %d measured calls per process' % PHASE_MINIMUMS[phase])
    profile = manifest.get('profiling')
    if type(profile) is not bool or profile != (phase == 'profile'):
        raise ValueError('Profiling must be on exactly in the profile phase')
    cases = {case['name']: case for case in manifest['cases']}
    if len(cases) != len(manifest['cases']) or set(cases) != set(PHASE_CASES[phase]):
        raise ValueError('Cases differ from the phase')
    if manifest.get('warmup_calls') != {name: WARMUP_CALLS[name] for name in cases} or manifest.get('sleep_ms') != {name: SLEEP_MS[name] for name in cases}:
        raise ValueError('Warm-up or sleep lengths differ from the protocol')
    arms = {name: phase_arms(phase, name) for name in cases}
    if manifest.get('arms') != {name: [arm for arm, _ in rows] for name, rows in arms.items()} or json.dumps(
            manifest.get('arm_specs'), sort_keys=True) != json.dumps({name: dict(rows) for name, rows in arms.items()}, sort_keys=True):
        raise ValueError('Arms differ from the phase')
    expected = {(name, arm, block) for name in cases for arm, _ in arms[name] for block in range(blocks)}
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
        item = dict(arms[entry['case']])[entry['arm']]
        expected_job = job(cases[entry['case']], item, WARMUP_CALLS[entry['case']], measured, profile)
        # Serialized equality keeps bool/int aliases and leaked truth out.
        if json.dumps(json.loads(paths['job'].read_text()), sort_keys=True) != json.dumps(expected_job, sort_keys=True):
            raise ValueError('Worker job differs from the declared protocol')
        if stage41.host_entry(entry.get('host')):
            failures.append(dict(case=entry['case'], arm=entry['arm'], block=entry['block'], gate='host_busy'))
        load = entry['host'].get('load_average')
        if not isinstance(load, list) or len(load) != 3:
            raise ValueError('Missing host load average')
        if finite(load[0], 'load_average') > HOST_LOAD_LIMIT:
            failures.append(dict(case=entry['case'], arm=entry['arm'], block=entry['block'], gate='host_load_high'))
        entries[key] = json.loads(paths['result'].read_text())
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    rows = []
    for name, case in cases.items():
        arm_rows, calls = {}, {}
        for arm, item in arms[name]:
            def gate(gate_name, block=None, arm=arm):
                failures.append(dict(case=name, arm=arm, gate=gate_name, **({} if block is None else dict(block=block))))
            if len(entries[(name, arm, 0)]['runs']) != 1+WARMUP_CALLS[name]+measured:
                raise ValueError('Sequence length differs from the protocol')
            arm_rows[arm], calls[arm] = arm_summary(case, item, [(block, entries[(name, arm, block)]) for block in range(blocks)],
                                                     WARMUP_CALLS[name], profile, gate)
            row = arm_rows[arm]
            for gate_name, passed in (('independent_ground_truth', row['all_gt_pass']), ('route_and_residency', row['route_valid']),
                                      ('requested_algorithm_not_executed', row['requested_algorithm_executed']), ('exact_repeat', row['repeat_exact'])):
                if not passed:
                    gate(gate_name)
        # Same pixels and options: every arm of a method must give one ordered output,
        # and peak 2 must equal peak 1 (round 42).
        outputs = {}
        for arm, item in arms[name]:
            outputs.setdefault(item['method'], []).extend((call['matches'], call['proposals']) for call in calls[arm])
        for method, values in outputs.items():
            if any(value != values[0] for value in values):
                failures.append(dict(case=name, method=method, gate='cross_arm_output_parity'))
        if G1 in outputs and G2 in outputs and any(value != outputs[G1][0] for value in outputs[G2]):
            failures.append(dict(case=name, method=G2, gate='peak2_equals_peak1'))
        summary_row = dict(name=name, truth_count=len(case['truth']), refine=case['workflow'] == 'full_rescue',
                           warmup_calls=WARMUP_CALLS[name], sleep_ms=SLEEP_MS[name], arms=arm_rows)
        if phase == 'dose':
            summary_row['dose'] = dose_summary(name, arm_rows)
        else:
            by_condition = {}
            for arm, item in arms[name]:
                by_condition.setdefault(item['method'], {})[condition_of(item)] = arm_rows[arm]
            summary_row['effects'] = {method: {contrast: {metric: effect(rows_[a], rows_[b], metric)
                                                          for metric in (GPU_METRICS if method in GPU_METHODS else HYBRID_METRICS)}
                                               for contrast, (a, b) in CONTRASTS.items()}
                                      for method, rows_ in by_condition.items()}
            summary_row['attribution'] = {method: dict(
                round41_ms=values['round41']['total_ms']['delta_ms'], update_ms=values['update']['total_ms']['delta_ms'],
                decode_gap_ms=values['decode_gap']['total_ms']['delta_ms'],
                sum_update_and_decode_gap_ms=values['update']['total_ms']['delta_ms']+values['decode_gap']['total_ms']['delta_ms'],
                residual_ms=values['round41']['total_ms']['delta_ms']-values['update']['total_ms']['delta_ms']-values['decode_gap']['total_ms']['delta_ms'],
                pose_create_ms=by_condition[method]['reuse_none']['metrics']['pose_create_ms']['median'])
                for method, values in summary_row['effects'].items()}
        rows.append(summary_row)
    arm_list = [row['arms'][arm] for row in rows for arm in row['arms']]
    return dict(phase=phase, formal_decision_eligible=phase == 'formal', profiling=profile, cases=rows, failed_gates=failures,
        hypotheses=hypotheses(rows) if phase in ('pilot', 'formal') else None,
        all_accuracy_pass=all(row['all_gt_pass'] for row in arm_list),
        all_routes_pass=all(row['route_valid'] and row['requested_algorithm_executed'] and row['sequence_route_valid'] for row in arm_list),
        all_repeat_exact=all(row['repeat_exact'] for row in arm_list),
        counts=dict(fresh_processes=len(entries), cold_calls=sum(row['cold_calls'] for row in arm_list),
                    warmup_calls=sum(row['warmup_calls'] for row in arm_list), measured_calls=sum(row['measured_calls'] for row in arm_list),
                    gt_evaluated_calls=sum(row['gt_evaluated_calls'] for row in arm_list)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note=('Every arm is a source_frames sequence: call 0 constructs the Session, the next round-41 warm-up calls are excluded, '
                     'the rest are measured. total_ms is run only; frame_wall_ms adds updateSource; gap_ms (decode or sleep) precedes the call '
                     'and is outside both. A direction needs every paired block to agree with the pooled medians.'),
        memory_note='Peak process RSS, named full-GPU Metal bytes, hybrid frame buffers and pose-context buffers are separate scopes and are never added.',
        parity_note='Independent strict GT on every cold, warm-up and measured call; exact ordered outputs within and across arms of a method, and peak 2 against peak 1, are separate checks.',
        frame_note='One archived fixture repeated; reuse arms copy a decoded frame with identical pixels, so data-dependent work is unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=list(PHASE_CASES))
    parser.add_argument('--build', type=Path, default=ROOT/'build/stage48-release')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures-manifest', type=Path, default=ROOT/'build/stage37-full-gpu/formal/manifest.json')
    parser.add_argument('--blocks', type=int)
    parser.add_argument('--measured', type=int)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        save(args.output, summarize(json.loads(args.summarize.read_text()), args.summarize.resolve().parent))
        return
    if not args.phase:
        parser.error('--phase is required unless --summarize')
    blocks, measured = (args.blocks or PHASE_DEFAULTS[args.phase][0]), (args.measured or PHASE_DEFAULTS[args.phase][1])
    if blocks < PHASE_MINIMUMS[args.phase][0] or measured < PHASE_MINIMUMS[args.phase][1] or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Blocks or measured calls below the phase minimum, or invalid timeout')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    names = PHASE_CASES[args.phase]
    cases = stage40.fixture_cases(args.fixtures_manifest, names)
    arms = {name: phase_arms(args.phase, name) for name in names}
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    before = previous.fingerprints(binary, cases)
    host = host_state(samples=5)
    if host['competing_processes'] or host['load_average'][0] > HOST_LOAD_LIMIT:
        raise RuntimeError('Host not quiet: '+json.dumps(dict(load=host['load_average'], competing=host['competing_processes'])))
    profile = args.phase == 'profile'
    manifest = dict(version=1, phase=args.phase, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        blocks=blocks, measured_calls=measured, profiling=profile, warmup_calls={name: WARMUP_CALLS[name] for name in names},
        sleep_ms={name: SLEEP_MS[name] for name in names}, timeout_seconds=args.timeout, cases=cases,
        arms={name: [arm for arm, _ in rows] for name, rows in arms.items()}, arm_specs={name: dict(rows) for name, rows in arms.items()},
        method_options=METHODS, fixtures_manifest=str(args.fixtures_manifest.resolve()), fixtures_manifest_sha256=sha(args.fixtures_manifest),
        host=dict(before=host), executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        for case in cases:
            for block in range(blocks):
                for arm, item in arms[case['name']][::1 if block % 2 == 0 else -1]:
                    label = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                    save(jp, job(case, item, WARMUP_CALLS[case['name']], measured, profile))
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
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block, job=str(jp.relative_to(root)),
                        job_sha256=sha(jp), result=str(rp.relative_to(root)), result_sha256=sha(rp), host=state))
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
