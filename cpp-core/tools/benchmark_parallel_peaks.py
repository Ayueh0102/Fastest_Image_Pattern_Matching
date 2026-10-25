"""Stage 42: exact parallel-round coarse peak selection (full_gpu_peak_mode=2).

Round 40's profile put 28-34% of full-GPU time in coarse peak selection: one
shared score map serializes every angle/scale map, and each map runs up to
limit dependent pick/suppress steps in one threadgroup. Peak mode 2 selects a
batch of maps per dispatch, each in its own ring slot, with exact parallel
rounds. Outputs must be byte-identical to peak mode 1.

Arms, one fresh process per case, arm and block, alternating order per block:
  static  round-40 job: first call, 1 s warm-up, measured repeats on one source
          hybrid_best, full_gpu_simd (peak 1, fine 4: the baseline),
          full_gpu_parallel_b1 (peak 2, one map per dispatch: ablation),
          full_gpu_parallel (peak 2, eight maps per dispatch: candidate)
  frames  round-41 sequence job: every call goes through Session::updateSource
          hybrid_best, full_gpu_parallel
Static compares the two full-GPU peak modes on the round-40 scope, where both
rebuild every source-dependent state on each call. The hybrid reuses its pose
context on static calls, so it is compared per new frame (round 41).

Phases: pilot (1 block, screening only), formal, profile (encoder timestamps,
diagnostic only, never gated on time), holdout (round-40 holdout fixtures,
exactness and GT only, timing reported). Truth stays in this parent; every
worker and failed gate is retained; RSS and named Metal bytes are never added.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

import benchmark_gpu_ncc_closure as stage40
import benchmark_new_frame_cost as stage41
from closure_lifecycle import valid_route

previous = stage40.previous
ROOT = stage40.ROOT
DOCS = ROOT/'cpp-core/docs'
HYBRID, BASELINE, ABLATION, CANDIDATE = 'hybrid_best', 'full_gpu_simd', 'full_gpu_parallel_b1', 'full_gpu_parallel'
METHODS = {
    HYBRID: stage40.METHODS[HYBRID],
    BASELINE: stage40.METHODS[BASELINE],
    ABLATION: dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=2, full_gpu_fine_mode=4, full_gpu_peak_batch=1),
    CANDIDATE: dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=2, full_gpu_fine_mode=4, full_gpu_peak_batch=8),
}
GPU_METHODS = {name for name, options in METHODS.items() if options['full_gpu']}
PARALLEL = (ABLATION, CANDIDATE)
ARMS = {f'{method}_static': (method, 'static') for method in METHODS}
ARMS.update({f'{method}_frames': (method, 'frames') for method in (HYBRID, CANDIDATE)})
PHASE_ARMS = dict(pilot=list(ARMS), formal=list(ARMS),
                  profile=[f'{method}_static' for method in (BASELINE, *PARALLEL)],
                  holdout=[f'{method}_static' for method in (BASELINE, *PARALLEL)])
DEFAULT_CASES = 'many128_pose,many512_pose,controlled6,8k_t256_angle180_3scales'
HOLDOUT_CASES = 'heldout12,small_fullres,boundary_one,empty_scene,constant_source,many128_holdout,many512_holdout'
PRIMARY_CASES = {'many128_pose', 'many512_pose'}
PRIMARY_TARGET_PERCENT = -10.
STATIC_WARMUP_SECONDS = stage41.STATIC_WARMUP_SECONDS
# Round-41 per-case warm-up calls for frames processes (planned from round-40 medians).
FRAME_WARMUP_CALLS = {'many128_pose': 2, 'many512_pose': 1, 'controlled6': 26, '8k_t256_angle180_3scales': 23}
sha, save, distribution, integer, evaluate = stage40.sha, stage40.save, stage40.distribution, stage40.integer, stage40.evaluate
finite = previous.finite_nonnegative


def phase_settings(phase):
    """Blocks, measured calls, static warm-up seconds and profiling for a phase."""
    return dict(pilot=(1, 2, STATIC_WARMUP_SECONDS, False), formal=(6, 3, STATIC_WARMUP_SECONDS, False),
                profile=(1, 2, STATIC_WARMUP_SECONDS, True), holdout=(2, 2, STATIC_WARMUP_SECONDS, False))[phase]


def job(case, arm, measured, warmup_seconds, profile, warmup_calls):
    method, mode = ARMS[arm]
    gpu = method in GPU_METHODS
    if mode == 'static':
        result = stage40.job(case, BASELINE if gpu else HYBRID, measured, warmup_seconds, profile)
    else:
        result = stage41.job(case, f'{BASELINE if gpu else HYBRID}_frames', warmup_calls, measured)
        result['full_gpu_profiling'] = profile
    result.update(METHODS[method])
    return result


def call_route(record, method, refine, profile):
    """Residency/refinement route, and whether the requested peak and fine kernels ran."""
    gpu = method in GPU_METHODS
    route = previous.validate_record(record, gpu)
    if not gpu:
        return route and valid_route(record) and record['refinement_candidates'] == (len(record['proposals']) if refine else 0), True
    metrics = record['full_gpu']
    options = METHODS[method]
    for field in ('peak_mode', 'peak_batch_maps', 'coarse_score_bytes', 'peak_scratch_bytes'):
        integer(metrics.get(field), field)
    stage40.fine_work_record(record, profile)
    stage40.profile_record(record, profile)
    batch = options.get('full_gpu_peak_batch', 1)
    route &= (metrics['peak_mode'] == options['full_gpu_peak_mode'] and metrics['fine_mode'] == options['full_gpu_fine_mode'] and
              1 <= metrics['peak_batch_maps'] <= batch and metrics['coarse_score_bytes'] <= metrics['gpu_bytes'] and
              metrics['peak_scratch_bytes'] <= metrics['gpu_bytes'])
    return route, metrics['fine_effective_mode'] == options['full_gpu_fine_mode']


def outputs(record):
    return record['matches'], record['proposals']


def accuracy_summary(records, truth):
    accuracy = [evaluate(record['matches'], truth)['strict'] for record in records]
    return dict(all_gt_pass=all(row['fp'] == row['fn'] == 0 for row in accuracy),
                accuracy_counts=sorted({(row['tp'], row['fp'], row['fn']) for row in accuracy}),
                max_errors={name: max((row['max_errors'][name] for row in accuracy if row['max_errors'][name] is not None), default=None)
                            for name in ('position_px', 'angle_deg', 'relative_scale')})


def memory_summary(records, method):
    if method not in GPU_METHODS:
        return dict(frame_gpu_bytes=sorted({record['frame_coarse_gpu_bytes']+record['frame_fine_gpu_bytes'] for record in records}),
                    pose_gpu_bytes=sorted({record['pose_gpu_bytes'] for record in records}),
                    gpu_scope='frame coarse+fine buffers; pose context buffers listed separately; other owners excluded')
    metrics = [record['full_gpu'] for record in records]
    return dict(named_gpu_bytes=sorted({row['gpu_bytes'] for row in metrics}),
                coarse_score_bytes=sorted({row['coarse_score_bytes'] for row in metrics}),
                peak_scratch_bytes=sorted({row['peak_scratch_bytes'] for row in metrics}),
                peak_batch_maps=sorted({row['peak_batch_maps'] for row in metrics}),
                gpu_scope='full_gpu.gpu_bytes: all named full-GPU Metal buffers; coarse_score_bytes and peak_scratch_bytes are parts of it')


def common(case, method, calls, routes):
    return dict(**memory_summary(calls, method), **accuracy_summary(calls, case['truth']),
                route_valid=all(route for route, _ in routes), requested_algorithm_executed=all(executed for _, executed in routes),
                repeat_exact=all(outputs(call) == outputs(calls[0]) for call in calls),
                proposal_counts=sorted({len(call['proposals']) for call in calls}), gt_evaluated_calls=len(calls))


def static_arm(case, method, workers, measured, profile):
    proxy = stage41.GPU if method in GPU_METHODS else stage41.HYBRID
    checked = [(block, worker, *stage41.static_calls(case, proxy, worker, measured)) for block, worker in workers]
    calls = [record for _, _, records, _ in checked for record in records]
    timed = [record for _, worker, _, _ in checked for record in worker['runs']]
    refine = case['workflow'] == 'full_rescue'
    routes = [call_route(record, method, refine, profile) for record in calls]
    blocks = [dict(block=block, total_median_ms=statistics.median(record['total_ms'] for record in worker['runs']),
                   first_total_ms=worker['first']['total_ms'], warmup_calls=len(worker['warmups_ms']),
                   construct_ms=worker['construct_ms'], peak_rss_bytes=worker['final_memory']['peak_rss_bytes'])
              for block, worker, _, _ in checked]
    row = dict(method=method, mode='static', total_ms=distribution([record['total_ms'] for record in timed]), blocks=blocks,
               peak_rss_bytes=max(block['peak_rss_bytes'] for block in blocks),
               static_route_valid=all(route for _, _, _, route in checked), cold_calls=len(checked),
               warmup_calls=sum(block['warmup_calls'] for block in blocks), measured_calls=len(timed),
               **common(case, method, calls, routes))
    row['repeat_exact'] &= all(worker['exact_repeat'] for _, worker, _, _ in checked)
    if method in GPU_METHODS:
        devices = [record['full_gpu']['device_ms'] for record in timed if previous.device_timestamp_available(record['full_gpu'])]
        row['device_ms'] = distribution(devices) if devices else None
    if profile and method in GPU_METHODS:
        available = [record for record in (stage40.profile_record(record, True) for record in timed) if record is not None]
        names = sorted({stage for record in available for stage in record['stages']})
        remainders = [record['device_remainder_ms'] for record in available if record['device_remainder_ms'] is not None]
        row['profile'] = dict(measured_available_calls=len(available), measured_calls=len(timed),
            stages={stage: distribution([record['stages'][stage]['gpu_ms'] for record in available if stage in record['stages']]) for stage in names},
            stage_sum_ms=distribution([record['stage_sum_ms'] for record in available]) if available else None,
            device_remainder_ms=distribution(remainders) if remainders else None,
            note='Encoder-boundary timestamps; stages of overlapping encoders can double count and the signed remainder is kept.')
    return row, calls


def frames_arm(case, method, workers, warmup, measured, gate):
    proxy = stage41.GPU if method in GPU_METHODS else stage41.HYBRID
    sequences = [(block, worker, *stage41.sequence_calls(case, proxy, worker, warmup, measured)) for block, worker in workers]
    calls = [row for _, _, rows, _ in sequences for row in rows]
    timed = [row for _, _, rows, _ in sequences for row in rows[1+warmup:]]
    routes = [call_route(row, method, case['workflow'] == 'full_rescue', False) for row in calls]
    blocks = []
    for block, worker, rows, _ in sequences:
        timed_rows = rows[1+warmup:]
        warmup_wall = rows[warmup]['elapsed_ms']-rows[0]['elapsed_ms']
        if warmup_wall < stage41.WARMUP_MINIMUM_MS:
            gate('warmup_shorter_than_1s', block)
        blocks.append(dict(block=block, frame_wall_median_ms=statistics.median(row['frame_wall_ms'] for row in timed_rows),
                           total_median_ms=statistics.median(row['total_ms'] for row in timed_rows),
                           warmup_calls=warmup, warmup_wall_ms=warmup_wall, cold_total_ms=rows[0]['total_ms'],
                           peak_rss_bytes=worker['final_memory']['peak_rss_bytes']))
    row = dict(method=method, mode='frames', **{name: distribution([call[name] for call in timed]) for name in
               ('frame_wall_ms', 'total_ms', 'source_update_ms', 'pose_create_ms', 'source_decode_ms')},
               blocks=blocks, peak_rss_bytes=max(block['peak_rss_bytes'] for block in blocks),
               sequence_route_valid=all(route for _, _, _, route in sequences), cold_calls=len(sequences),
               warmup_calls=len(sequences)*warmup, measured_calls=len(timed), **common(case, method, calls, routes))
    row['repeat_exact'] &= all(worker['exact_sequence_repeat'] for _, worker, _, _ in sequences)
    return row, calls


def compare(baseline, candidate, metric, block_field, names):
    """Candidate against baseline; a direction needs every paired block to agree with the pooled medians."""
    left, right = baseline[metric]['median'], candidate[metric]['median']
    paired = [dict(block=a['block'], baseline_ms=a[block_field], candidate_ms=b[block_field],
                   delta_ms=b[block_field]-a[block_field], delta_percent=100*(b[block_field]/a[block_field]-1),
                   within_regression_budget=b[block_field] <= a[block_field]+max(2., .1*a[block_field]))
              for a, b in zip(baseline['blocks'], candidate['blocks'])]
    signs = {(row['candidate_ms'] > row['baseline_ms'])-(row['candidate_ms'] < row['baseline_ms']) for row in paired}
    pooled = (right > left)-(right < left)
    consistent = pooled != 0 and signs == {pooled}
    return dict(baseline=names[0], candidate=names[1], metric=metric, baseline_median_ms=left, candidate_median_ms=right,
                delta_ms=right-left, delta_percent=100*(right/left-1), paired=paired,
                candidate_faster_blocks=sum(row['candidate_ms'] < row['baseline_ms'] for row in paired), blocks=len(paired),
                consistent_direction=consistent, faster=(names[1] if pooled < 0 else names[0]) if consistent else None)


def summarize(manifest, root):
    if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
        raise ValueError('Experiment must be complete with unchanged fingerprints')
    fingerprints = {name: manifest.get(name) for name in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(fingerprints.values()) or manifest.get('after_sha256') != fingerprints:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    phase = manifest.get('phase')
    if phase not in PHASE_ARMS:
        raise ValueError('Unknown phase')
    arms = manifest.get('arms')
    if arms != PHASE_ARMS[phase]:
        raise ValueError('Arms differ from the declared phase arms')
    if json.dumps(manifest.get('method_options'), sort_keys=True) != json.dumps(METHODS, sort_keys=True):
        raise ValueError('Manifest method_options differ from the declared methods')
    blocks, measured, warmup_seconds, profile = phase_settings(phase)
    if (manifest.get('blocks'), manifest.get('measured_calls'), manifest.get('static_warmup_seconds'), manifest.get('profiling')) != \
            (blocks, measured, warmup_seconds, profile):
        raise ValueError('Blocks, measured calls, warm-up or profiling differ from the phase protocol')
    cases = {case['name']: case for case in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Empty or duplicate cases')
    framed = any(ARMS[arm][1] == 'frames' for arm in arms)
    warmups = manifest.get('frame_warmup_calls')
    if framed and (not isinstance(warmups, dict) or set(cases) - warmups.keys() or
                   any(warmups[name] != FRAME_WARMUP_CALLS.get(name) for name in cases)):
        raise ValueError('Frames arms need the round-41 warm-up call counts')
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
        expected_job = job(cases[entry['case']], entry['arm'], measured, warmup_seconds, profile,
                           warmups[entry['case']] if framed else None)
        if json.dumps(json.loads(paths['job'].read_text()), sort_keys=True) != json.dumps(expected_job, sort_keys=True):
            raise ValueError('Worker job differs from the declared protocol')
        if stage41.host_entry(entry.get('host')):
            failures.append(dict(case=entry['case'], arm=entry['arm'], block=entry['block'], gate='host_busy'))
        entries[key] = json.loads(paths['result'].read_text())
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    rows = []
    for name, case in cases.items():
        arm_rows, calls = {}, {}
        for arm in arms:
            method, mode = ARMS[arm]
            def gate(gate_name, block=None, arm=arm):
                failures.append(dict(case=name, arm=arm, gate=gate_name, **({} if block is None else dict(block=block))))
            workers = [(block, entries[(name, arm, block)]) for block in range(blocks)]
            if mode == 'frames':
                arm_rows[arm], calls[arm] = frames_arm(case, method, workers, warmups[name], measured, gate)
            else:
                arm_rows[arm], calls[arm] = static_arm(case, method, workers, measured, profile)
        reference = outputs(calls[f'{BASELINE}_static'][0])
        for arm in arms:
            method, mode = ARMS[arm]
            row = arm_rows[arm]
            checks = [('independent_ground_truth', row['all_gt_pass']), ('route_and_residency', row['route_valid']),
                      ('requested_algorithm_not_executed', row['requested_algorithm_executed']), ('exact_repeat', row['repeat_exact']),
                      ('sequence_route', row['sequence_route_valid']) if mode == 'frames' else ('static_route', row['static_route_valid'])]
            if mode == 'frames':
                # Same pixels and options: outputs must equal the same method's static control.
                own = outputs(calls[f'{method}_static'][0])
                row['same_method_output_parity'] = all(outputs(call) == own for call in calls[arm])
                checks.append(('same_method_output_parity', row['same_method_output_parity']))
            if method in PARALLEL:
                # Separate from GT: complete ordered outputs of every call equal peak mode 1.
                row['ordered_matches_equal_mode1'] = all(call['matches'] == reference[0] for call in calls[arm])
                row['ordered_proposals_equal_mode1'] = all(call['proposals'] == reference[1] for call in calls[arm])
                checks += [('exact_ordered_matches_vs_mode1', row['ordered_matches_equal_mode1']),
                           ('exact_ordered_proposals_vs_mode1', row['ordered_proposals_equal_mode1'])]
            failures.extend(dict(case=name, arm=arm, gate=gate_name) for gate_name, passed in checks if not passed)
        comparisons = {}
        if phase in ('pilot', 'formal', 'holdout'):
            static = {method: arm_rows[f'{method}_static'] for method in METHODS if f'{method}_static' in arm_rows}
            for method in PARALLEL:
                result = compare(static[BASELINE], static[method], 'total_ms', 'total_median_ms', (BASELINE, method))
                comparisons[f'{method}_vs_mode1_static'] = result
                if phase == 'holdout':
                    continue
                for pair in result['paired']:
                    if not pair['within_regression_budget']:
                        failures.append(dict(case=name, arm=f'{method}_static', baseline=f'{BASELINE}_static', gate='time_budget', block=pair['block']))
                rss = static[BASELINE]['peak_rss_bytes']
                if static[method]['peak_rss_bytes'] > rss+max(8*1024**2, .1*rss):
                    failures.append(dict(case=name, arm=f'{method}_static', baseline=f'{BASELINE}_static', gate='rss_budget'))
                if method == CANDIDATE and name in PRIMARY_CASES:
                    result['primary_target_percent'] = PRIMARY_TARGET_PERCENT
                    result['primary_target_met'] = result['consistent_direction'] and result['delta_percent'] <= PRIMARY_TARGET_PERCENT
                    if not result['primary_target_met']:
                        failures.append(dict(case=name, arm=f'{method}_static', baseline=f'{BASELINE}_static', gate='primary_10_percent_improvement'))
            comparisons['batch_vs_b1_static'] = compare(static[ABLATION], static[CANDIDATE], 'total_ms', 'total_median_ms', (ABLATION, CANDIDATE))
        if f'{HYBRID}_frames' in arm_rows:
            frames, static = arm_rows[f'{HYBRID}_frames'], arm_rows[f'{HYBRID}_static']
            mine, mine_static = arm_rows[f'{CANDIDATE}_frames'], arm_rows[f'{CANDIDATE}_static']
            comparisons['vs_hybrid_frames_frame_wall'] = compare(frames, mine, 'frame_wall_ms', 'frame_wall_median_ms', (HYBRID, CANDIDATE))
            comparisons['vs_hybrid_frames_run_only'] = compare(frames, mine, 'total_ms', 'total_median_ms', (HYBRID, CANDIDATE))
            comparisons['vs_hybrid_static_run_only'] = compare(static, mine_static, 'total_ms', 'total_median_ms', (HYBRID, CANDIDATE))
            comparisons['mode1_vs_hybrid_static_run_only'] = compare(static, arm_rows[f'{BASELINE}_static'], 'total_ms', 'total_median_ms', (HYBRID, BASELINE))
            for pair in comparisons['vs_hybrid_frames_frame_wall']['paired']:
                if not pair['within_regression_budget']:
                    failures.append(dict(case=name, arm=f'{CANDIDATE}_frames', baseline=f'{HYBRID}_frames', gate='time_budget', block=pair['block']))
            if mine['peak_rss_bytes'] > frames['peak_rss_bytes']+max(8*1024**2, .1*frames['peak_rss_bytes']):
                failures.append(dict(case=name, arm=f'{CANDIDATE}_frames', baseline=f'{HYBRID}_frames', gate='rss_budget'))
        rows.append(dict(name=name, truth_count=len(case['truth']), refine=case['workflow'] == 'full_rescue',
                         frame_warmup_calls=warmups.get(name) if framed else None, arms=arm_rows, comparisons=comparisons))
    arm_list = [row['arms'][arm] for row in rows for arm in arms]
    exact = [row['arms'][arm] for row in rows for arm in arms if ARMS[arm][0] in PARALLEL]
    primary = [row['comparisons'].get(f'{CANDIDATE}_vs_mode1_static', {}).get('primary_target_met') for row in rows if row['name'] in PRIMARY_CASES]
    return dict(phase=phase, formal_decision_eligible=phase == 'formal', profiling=profile, cases=rows, failed_gates=failures,
        all_accuracy_pass=all(row['all_gt_pass'] for row in arm_list),
        all_routes_pass=all(row['route_valid'] and row['requested_algorithm_executed'] and
                            row.get('sequence_route_valid', row.get('static_route_valid')) for row in arm_list),
        all_repeat_exact=all(row['repeat_exact'] for row in arm_list),
        all_parallel_exact_mode1=all(row['ordered_matches_equal_mode1'] and row['ordered_proposals_equal_mode1'] for row in exact),
        primary_target_met=all(primary) if phase == 'formal' and len(primary) == len(PRIMARY_CASES) else None,
        counts=dict(fresh_processes=len(entries), cold_calls=sum(row['cold_calls'] for row in arm_list),
            gt_checked_warmup_calls=sum(row['warmup_calls'] for row in arm_list if row['mode'] == 'frames'),
            native_only_warmup_calls=sum(row['warmup_calls'] for row in arm_list if row['mode'] == 'static'),
            measured_calls=sum(row['measured_calls'] for row in arm_list), gt_evaluated_calls=sum(row['gt_evaluated_calls'] for row in arm_list)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note=('Static total_ms is the round-40 scope (run only, after a first call and a 1 s warm-up). Frames frame_wall_ms is the round-41 scope: '
                     'Session::updateSource plus run, excluding decode, comparison and JSON. A faster method is named only when every paired block '
                     'agrees with the pooled medians. Profile runs carry encoder timestamps and never enter time gates.'),
        memory_note=('Peak process RSS, named Metal buffer lengths and hybrid pose-context buffers are separate scopes and are never added. '
                     'coarse_score_bytes (score map or ring) and peak_scratch_bytes are parts of the named Metal total.'),
        parity_note=('Independent strict GT is evaluated on every first, warm-up (frames) and measured call. Exact ordered matches AND proposals '
                     'of every peak-mode-2 call against the peak-mode-1 static first call are a separate check.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/stage42-release')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=list(PHASE_ARMS), default='formal')
    parser.add_argument('--fixtures-manifest', type=Path)
    parser.add_argument('--cases')
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        save(args.output, summarize(manifest, args.summarize.resolve().parent))
        return
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Positive timeout required')
    fixtures = args.fixtures_manifest or (DOCS/'stage40-ncc-closure/holdout-fixtures.json' if args.phase == 'holdout'
                                          else Path('/Users/Ayueh/Fastest_Image_Pattern_Matching/build/stage37-full-gpu/formal/manifest.json'))
    names = previous.selected_names(args.cases or (HOLDOUT_CASES if args.phase == 'holdout' else DEFAULT_CASES), None, 'cases')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    cases = stage40.fixture_cases(fixtures, names)
    arms = PHASE_ARMS[args.phase]
    blocks, measured, warmup_seconds, profile = phase_settings(args.phase)
    framed = any(ARMS[arm][1] == 'frames' for arm in arms)
    if framed and set(names) - FRAME_WARMUP_CALLS.keys():
        parser.error('Frames arms are defined for the four round-41 cases only')
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    before = previous.fingerprints(binary, cases)
    host = stage41.host_state(samples=5)
    if args.phase != 'pilot' and host['competing_processes']:
        raise RuntimeError('Competing build or benchmark processes: '+json.dumps(host['competing_processes']))
    manifest = dict(version=1, phase=args.phase, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        blocks=blocks, measured_calls=measured, static_warmup_seconds=warmup_seconds, profiling=profile,
        frame_warmup_calls={name: FRAME_WARMUP_CALLS[name] for name in names} if framed else None,
        timeout_seconds=args.timeout, cases=cases, arms=arms, method_options=METHODS,
        fixtures_manifest=str(fixtures.resolve()), fixtures_manifest_sha256=sha(fixtures),
        host=dict(before=host), executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        for case in cases:
            for block in range(blocks):
                for arm in arms[::1 if block % 2 == 0 else -1]:
                    label = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                    save(jp, job(case, arm, measured, warmup_seconds, profile, FRAME_WARMUP_CALLS.get(case['name']) if framed else None))
                    state = stage41.host_state()
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
        manifest['host']['after'] = stage41.host_state(samples=5)
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
