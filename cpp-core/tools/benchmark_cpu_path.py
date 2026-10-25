"""Stage 45: the full CPU research path next to the hybrid and full-GPU peak 2, per new frame.

Round 44 ranked the hybrid and full-GPU combinations; the CPU research path
(backend 0, refinement_backend 0) was last compared in round 34, and only as a
CPU/Metal refinement swap behind a Metal front end. This round measures it on
one build next to the hybrid and full-GPU peak 2 / fine 4, each with candidate
dedup off and k3. There is no algorithm or default change.

CPU job: the hybrid_best job with backend 0 and refinement_backend 0, the
Metal-only options the pipeline rejects on the CPU switched off (ncc_mode 0,
fine_ncc_mode 0, fine_fused and fine_angle_reuse false), and pose_storage_mode
0 (shared pose pages require Metal). cpu_threads stays the runner default 8,
the M4 Pro's performance-core count. The CPU NCC calls the core dot product
with Backend::Auto, which is NEON+DOTPROD on this build.

Arms: frames only (round-41 sequence: every call goes through
Session::updateSource) x {cpu_full, hybrid_best, full_gpu_parallel} x {off,
k3}, one fresh process per case, arm and block, order reversed on odd blocks.
Cross-method comparisons use frame_wall_ms. Hybrid and full-GPU jobs equal the
round-44 frames jobs and every call must equal the round-43 archive; CPU calls
must show no Metal work at all. Independent GT is checked on every call.

Phases: gt (round-43 GT case variants with a hybrid path, CPU x {off, k3}; GT
and route only, no timing), pilot (1 block, screening), formal (6 blocks).
Truth stays in this parent; every worker and failed gate is retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics

import benchmark_candidate_dedup as stage43
import benchmark_dedup_peak2 as stage44
import benchmark_new_frame_cost as stage41
import benchmark_parallel_peaks as stage42

stage40, previous = stage44.stage40, stage44.previous
ROOT = stage44.ROOT
DOCS = ROOT/'cpp-core/docs'
REFERENCE = DOCS/'stage43-candidate-dedup'
CPU, HYBRID, GPU = 'cpu_full', stage44.HYBRID, stage44.GPU
CPU_OPTIONS = dict(backend=0, refinement_backend=0, ncc_mode=0, fine_ncc_mode=0, fine_fused=False, fine_angle_reuse=False,
                   overlap_enabled=False, pose_storage_mode=0)
METHODS = {CPU: dict(stage44.METHODS[HYBRID], **CPU_OPTIONS), HYBRID: stage44.METHODS[HYBRID], GPU: stage44.METHODS[GPU]}
PATHS = {CPU: 'hybrid', HYBRID: 'hybrid', GPU: 'full_gpu'}  # round-43 path names (dedup counter contract, archive keys)
DEDUP = stage44.DEDUP
ARMS = {f'{method}_frames_{dedup}': (method, dedup) for method in METHODS for dedup in DEDUP}
PHASE_SETTINGS = dict(pilot=(1, 2), formal=(6, 3))  # blocks, measured calls
FORMAL_CASES = stage43.FORMAL_CASES
FRAME_WARMUP_CALLS = stage44.FRAME_WARMUP_CALLS  # round-44 counts for every arm
RSS_GATE = stage43.RSS_GATE
FIXTURES = stage44.FIXTURES
JOB_EXEMPT = stage44.JOB_EXEMPT+tuple(CPU_OPTIONS)+('source', 'template')
COMPARISONS = [(f'{CPU}_frames_off', f'{CPU}_frames_k3'), (f'{HYBRID}_frames_off', f'{HYBRID}_frames_k3'),
               (f'{GPU}_frames_off', f'{GPU}_frames_k3'), (f'{CPU}_frames_off', f'{HYBRID}_frames_off'),
               (f'{CPU}_frames_k3', f'{HYBRID}_frames_k3'), (f'{CPU}_frames_off', f'{GPU}_frames_k3'),
               (f'{CPU}_frames_k3', f'{GPU}_frames_k3'), (f'{HYBRID}_frames_k3', f'{GPU}_frames_k3')]
sha, save, distribution, integer = stage40.sha, stage40.save, stage40.distribution, stage40.integer


def job(case, arm, measured, warmup_calls):
    method, dedup = ARMS[arm]
    result = stage44.job(case, f'{HYBRID if method == CPU else method}_frames_{dedup}', measured, stage42.STATIC_WARMUP_SECONDS,
                         False, warmup_calls)
    if method == CPU:
        result.update(CPU_OPTIONS)
    return result


def gt_job(case, dedup):
    result = stage43.make_job(case, 'hybrid', dedup, 1, 0.)
    result.update(CPU_OPTIONS)
    return result


def comparable(job_dict):
    return json.dumps({key: value for key, value in job_dict.items() if key not in JOB_EXEMPT}, sort_keys=True)


def cpu_route(record, refine):
    """No Metal work anywhere: frame, fine, pose and full-GPU counters are zero and refinement ran on the CPU."""
    metrics = record['full_gpu']
    active = record['refinement_candidates'] > 0
    return (previous.validate_record(record, False) and not metrics['enabled'] and metrics['command_buffers'] == 0 and
            metrics['gpu_bytes'] == 0 and record['frame_coarse_gpu_bytes'] == 0 and record['frame_fine_gpu_bytes'] == 0 and
            record['pose_gpu_bytes'] == 0 and record['refinement_dispatches'] == 0 and record['refinement_command_buffers'] == 0 and
            record['configured_refinement_backend'] == 0 and record['selected_refinement_backend'] == 0 and
            record['pose_storage_mode'] == 0 and record['pose_input_mode'] == 1 and record['refinement_mode'] == 0 and
            record['refinement_candidates'] == (len(record['proposals']) if refine else 0) and
            (record['pose_host_bytes'] > 0 if active else record['pose_host_bytes'] == 0) and
            all(row['coarse_commands'] == 0 and row['fine_commands'] == 0 and row['coarse_gpu_bytes'] == 0 for row in record['work']))


def call_route(record, method, refine):
    if method == CPU:
        return cpu_route(record, refine), True
    route, executed = stage42.call_route(record, method, refine, False)
    return route and (method != GPU or stage44.peak_route(record)), executed


def dedup_checked(case, method, dedup, calls):
    scales = len(case.get('scales') or case['native_search']['scales'])
    derived = [stage43.record_check(record, PATHS[method], dedup, scales) for record in calls]
    first = (derived[0]['input'], derived[0]['kept'])
    return dict(dedup_input=first[0], dedup_kept=first[1],
                dedup_counts_constant=all((row['input'], row['kept']) == first for row in derived))


def frames_arm(case, method, workers, warmup, measured, gate):
    """stage42.frames_arm with a CPU-aware route; the CPU rebuilds its pose context per frame like the hybrid."""
    proxy = stage41.GPU if method == GPU else stage41.HYBRID
    sequences = [(block, worker, *stage41.sequence_calls(case, proxy, worker, warmup, measured)) for block, worker in workers]
    calls = [row for _, _, rows, _ in sequences for row in rows]
    timed = [row for _, _, rows, _ in sequences for row in rows[1+warmup:]]
    routes = [call_route(row, method, case['workflow'] == 'full_rescue') for row in calls]
    blocks = []
    for block, worker, rows, _ in sequences:
        timed_rows = rows[1+warmup:]
        warmup_wall = rows[warmup]['elapsed_ms']-rows[0]['elapsed_ms']
        if warmup_wall < stage41.WARMUP_MINIMUM_MS:
            gate('warmup_shorter_than_1s', block)
        blocks.append(dict(block=block, frame_wall_median_ms=statistics.median(row['frame_wall_ms'] for row in timed_rows),
                           total_median_ms=statistics.median(row['total_ms'] for row in timed_rows), warmup_calls=warmup,
                           warmup_wall_ms=warmup_wall, cold_total_ms=rows[0]['total_ms'], peak_rss_bytes=worker['final_memory']['peak_rss_bytes']))
    row = dict(method=method, mode='frames', **{name: distribution([call[name] for call in timed]) for name in
               ('frame_wall_ms', 'total_ms', 'source_update_ms', 'pose_create_ms', 'source_decode_ms')},
               blocks=blocks, peak_rss_bytes=max(block['peak_rss_bytes'] for block in blocks),
               sequence_route_valid=all(route for _, _, _, route in sequences), cold_calls=len(sequences),
               warmup_calls=len(sequences)*warmup, measured_calls=len(timed), **stage42.common(case, method, calls, routes))
    row['repeat_exact'] &= all(worker['exact_sequence_repeat'] for _, worker, _, _ in sequences)
    if method != GPU:
        row['stages_ms'] = {key: distribution([call[key] for call in timed]) for key in ('proposal_ms', 'refine_ms', 'nms_ms')}
    return row, calls


def check_manifest(manifest, reference, phases):
    if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
        raise ValueError('Experiment must be complete with unchanged fingerprints')
    fingerprints = {name: manifest.get(name) for name in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(fingerprints.values()) or manifest.get('after_sha256') != fingerprints:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    if manifest.get('phase') not in phases:
        raise ValueError('Unknown phase')
    if json.dumps(manifest.get('method_options'), sort_keys=True) != json.dumps(METHODS, sort_keys=True) or \
            json.dumps(manifest.get('dedup_options'), sort_keys=True) != json.dumps(DEDUP, sort_keys=True):
        raise ValueError('Manifest options differ from the declared methods or dedup arms')
    if manifest.get('reference_sha256') != reference['sha256']:
        raise ValueError('Manifest names a different round-43 reference archive')
    cases = {case['name']: case for case in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Empty or duplicate cases')
    return cases


def summarize(manifest, root, reference):
    """Pilot or formal manifest; reference holds the round-43 formal block-0 results."""
    cases = check_manifest(manifest, reference, PHASE_SETTINGS)
    phase, arms = manifest['phase'], manifest.get('arms')
    if arms != list(ARMS):
        raise ValueError('Arms differ from the declared arms')
    blocks, measured = PHASE_SETTINGS[phase]
    if (manifest.get('blocks'), manifest.get('measured_calls')) != (blocks, measured):
        raise ValueError('Blocks or measured calls differ from the phase protocol')
    warmups = manifest.get('frame_warmup_calls')
    if not isinstance(warmups, dict) or any(warmups.get(name) != FRAME_WARMUP_CALLS.get(name) for name in cases):
        raise ValueError('Frames arms need the stage-44 warm-up call counts')
    expected = {(name, arm, block) for name in cases for arm in arms for block in range(blocks)}
    entries, failures = stage44.validated_entries(manifest, root, expected,
                                                  lambda name, arm, block: job(cases[name], arm, measured, warmups[name]))
    rows = []
    for name, case in cases.items():
        arm_rows, calls = {}, {}
        for arm in arms:
            method, dedup = ARMS[arm]
            def gate(gate_name, block=None, arm=arm):
                failures.append(dict(case=name, arm=arm, gate=gate_name, **({} if block is None else dict(block=block))))
            row, arm_calls = frames_arm(case, method, [(block, entries[(name, arm, block)]) for block in range(blocks)],
                                        warmups[name], measured, gate)
            row.update(dedup=dedup, **dedup_checked(case, method, dedup, arm_calls))
            row['cross_block_identity'] = all(stage42.outputs(call) == stage42.outputs(arm_calls[0]) for call in arm_calls)
            arm_rows[arm], calls[arm] = row, arm_calls
        for arm in arms:
            method, dedup = ARMS[arm]
            row = arm_rows[arm]
            checks = [('independent_ground_truth', row['all_gt_pass']), ('route', row['route_valid']),
                      ('requested_algorithm_not_executed', row['requested_algorithm_executed']), ('exact_repeat', row['repeat_exact']),
                      ('sequence_route', row['sequence_route_valid']), ('dedup_counts_constant', row['dedup_counts_constant']),
                      ('cross_block_identity', row['cross_block_identity'])]
            archived = reference['results'].get((name, PATHS[method], dedup))
            if method == CPU:
                # Not expected and not a gate: CPU binary64 coarse scores against the Metal hybrid archive.
                row['outputs_equal_round43_hybrid'] = archived is not None and \
                    stage42.outputs(calls[arm][0]) == stage42.outputs(archived['first'])
            else:
                row['exact_vs_round43_archive'] = archived is not None and \
                    all(stage42.outputs(call) == stage42.outputs(archived['first']) for call in calls[arm])
                checks.append(('exact_vs_round43_archive', row['exact_vs_round43_archive']))
            failures.extend(dict(case=name, arm=arm, gate=gate_name) for gate_name, passed in checks if not passed)
        comparisons = {}
        for left, right in COMPARISONS:
            comparisons[f'{right}_vs_{left}'] = result = stage42.compare(arm_rows[left], arm_rows[right], 'frame_wall_ms',
                                                                        'frame_wall_median_ms', (left, right))
            if ARMS[left][0] == ARMS[right][0]:  # k3 against off on one method: memory budget
                rss = arm_rows[left]['peak_rss_bytes']
                if arm_rows[right]['peak_rss_bytes'] > rss+max(RSS_GATE[0], RSS_GATE[1]*rss):
                    failures.append(dict(case=name, arm=right, baseline=left, gate='rss_budget'))
        order = sorted(arms, key=lambda arm: arm_rows[arm]['frame_wall_ms']['median'])
        versus = [stage42.compare(arm_rows[other], arm_rows[order[0]], 'frame_wall_ms', 'frame_wall_median_ms', (other, order[0]))
                  for other in order[1:]]
        # A fastest arm is named only when it beats every other arm in every paired block.
        comparisons['frames_ranking'] = dict(order=order, median_ms={arm: arm_rows[arm]['frame_wall_ms']['median'] for arm in order},
            fastest=order[0] if all(result['faster'] == order[0] for result in versus) else None,
            versus=[dict(other=result['baseline'], delta_percent=result['delta_percent'],
                         faster_blocks=result['candidate_faster_blocks'], consistent=result['consistent_direction']) for result in versus])
        rows.append(dict(name=name, truth_count=len(case['truth']), refine=case['workflow'] == 'full_rescue',
                         frame_warmup_calls=warmups[name], arms=arm_rows, comparisons=comparisons))
    arm_list = [row['arms'][arm] for row in rows for arm in arms]
    return dict(phase=phase, formal_decision_eligible=phase == 'formal', reference=reference['archive'], reference_sha256=reference['sha256'],
        cases=rows, failed_gates=failures,
        all_accuracy_pass=all(row['all_gt_pass'] for row in arm_list),
        all_routes_pass=all(row['route_valid'] and row['requested_algorithm_executed'] and row['sequence_route_valid'] for row in arm_list),
        all_repeat_exact=all(row['repeat_exact'] and row['cross_block_identity'] for row in arm_list),
        all_exact_round43=all(row['exact_vs_round43_archive'] for row in arm_list if 'exact_vs_round43_archive' in row),
        cpu_outputs_equal_round43_hybrid={row['name']: {dedup: row['arms'][f'{CPU}_frames_{dedup}']['outputs_equal_round43_hybrid']
                                                        for dedup in DEDUP} for row in rows},
        frames_fastest={row['name']: row['comparisons']['frames_ranking']['fastest'] for row in rows},
        counts=dict(fresh_processes=len(entries), cold_calls=sum(row['cold_calls'] for row in arm_list),
            measured_calls=sum(row['measured_calls'] for row in arm_list), gt_evaluated_calls=sum(row['gt_evaluated_calls'] for row in arm_list)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note=('frame_wall_ms is the round-41 scope (Session::updateSource plus run, excluding decode, comparison and JSON). '
                     'A faster arm is named only when every paired block agrees with the pooled medians. Same-build comparisons only; '
                     'never multiplied with earlier rounds.'),
        memory_note='Peak process RSS, hybrid frame/pose buffers and named full-GPU Metal buffers are separate scopes and are never added.')


def input_digests(path):
    """Round-43 input hashes keyed by the absolute path its jobs used (names repeat across rounds, e.g. empty_scene.png)."""
    return json.loads(Path(path).read_text())['inputs']


def summarize_gt(manifest, root, reference, inputs43):
    """GT phase: CPU x {off, k3} on the round-43 hybrid case variants; reference is the round-43 GT archive."""
    cases = check_manifest(manifest, reference, ('gt',))
    if manifest.get('arms') != list(DEDUP):
        raise ValueError('GT arms differ from the declared dedup arms')
    expected = {(name, dedup, 0) for name in cases for dedup in DEDUP}
    cells, failures = {}, []
    for entry in manifest['executions']:
        key = (entry['case'], entry['arm'], entry['block'])
        if key not in expected or key in cells:
            raise ValueError('Unexpected or duplicate GT worker '+str(key))
        if entry['returncode']:
            failures.append(dict(gate='worker', case=key[0], arm=key[1], detail=entry.get('stderr_tail', '')[-400:]))
            cells[key] = None
            continue
        name, dedup = key[0], key[1]
        job_path = previous.worker_path(root, entry['job'])
        mine = json.loads(job_path.read_text())
        if json.dumps(mine, sort_keys=True) != json.dumps(gt_job(cases[name], dedup), sort_keys=True):
            raise ValueError('GT worker job differs from the declared protocol')
        worker = stage43.load_worker(root, entry)
        calls = stage43.worker_calls(worker)
        refine = mine.get('refine', True)
        cell = stage43.gt_cell(cases[name], 'hybrid', dedup, worker)
        cell['cpu_route'] = all(cpu_route(call, refine) for call in calls)
        ref = (name, 'hybrid', dedup)
        old = reference['jobs'].get(ref)
        # Round-44 lesson: inputs are compared by content, not by path.
        cell['job_equals_round43'] = old is not None and comparable(mine) == comparable(old) and \
            all(manifest['input_sha256'][mine[k]] == inputs43.get(old[k]) for k in ('source', 'template'))
        archived = reference['results'].get(ref)
        cell['outputs_equal_round43_hybrid'] = archived is not None and stage42.outputs(calls[0]) == stage42.outputs(archived['first'])
        if archived is not None:
            hybrid = stage40.evaluate(archived['first']['matches'], cases[name]['truth'])['strict']
            cell['round43_hybrid_strict'] = dict(tp=hybrid['tp'], fp=hybrid['fp'], fn=hybrid['fn'])
        for gate in ('cpu_route', 'job_equals_round43'):
            if not cell[gate]:
                failures.append(dict(gate=gate, case=name, arm=dedup))
        if not cell['exact_repeat'] or not cell['same_counts']:
            failures.append(dict(gate='repeat', case=name, arm=dedup))
        cells[key] = cell
    if set(cells) != expected:
        raise ValueError('GT workers do not cover every case and dedup arm exactly')
    rows, total = [], dict(cases=0, regressions=[], improvements=[], tp_delta=0, fp_delta=0, fn_delta=0,
                           matches_identical=0, proposals_identical=0, input=0, kept=0)
    versus_hybrid = dict(cases=0, fewer_tp=[], more_fp=[], more_tp=[], fewer_fp=[], outputs_identical=0)
    for name, case in cases.items():
        off, k3 = cells[(name, 'off', 0)], cells[(name, 'k3', 0)]
        row = dict(case=name, family=case['family'], role=case['role'], truth=len(case['truth']),
                   arms={dedup: {k: v for k, v in cell.items() if not k.startswith('_')} for dedup, cell in (('off', off), ('k3', k3)) if cell})
        if off and k3:
            row['k3_versus_off'] = comparison = stage43.compare_cell(k3, off)
            total['cases'] += 1
            for field in ('tp', 'fp', 'fn'):
                total[field+'_delta'] += comparison['delta'][field]
            total['matches_identical'] += comparison['matches_identical']
            total['proposals_identical'] += comparison['proposals_identical']
            total['input'] += k3['dedup_input']; total['kept'] += k3['dedup_kept']
            if comparison['regression']:
                total['regressions'].append(dict(case=name, role=case['role'], **comparison['delta']))
                failures.append(dict(gate='gt_regression', case=name, arm='k3', role=case['role'], **comparison['delta']))
            if comparison['improvement']:
                total['improvements'].append(dict(case=name, role=case['role'], **comparison['delta']))
        for dedup, cell in (('off', off), ('k3', k3)):
            if not cell or 'round43_hybrid_strict' not in cell:
                continue
            mine, hybrid = cell['strict'][0], cell['round43_hybrid_strict']
            versus_hybrid['cases'] += 1
            versus_hybrid['outputs_identical'] += cell['outputs_equal_round43_hybrid']
            for field, worse, better in (('tp', 'fewer_tp', 'more_tp'), ('fp', 'more_fp', 'fewer_fp')):
                delta = mine[field]-hybrid[field]
                if (field == 'tp' and delta < 0) or (field == 'fp' and delta > 0):
                    versus_hybrid[worse].append(dict(case=name, arm=dedup, delta=delta))
                elif delta:
                    versus_hybrid[better].append(dict(case=name, arm=dedup, delta=delta))
        rows.append(row)
    total['kept_fraction'] = total['kept']/total['input'] if total['input'] else None
    live = [cell for cell in cells.values() if cell]
    return dict(phase='gt', reference=reference['archive'], reference_sha256=reference['sha256'], cases=rows, aggregate=total,
        cpu_versus_round43_hybrid=versus_hybrid, failures=failures,
        gt_neutral=not total['regressions'] and total['cases'] == len(cases),
        all_cpu_routes=len(live) == len(cells) and all(cell['cpu_route'] for cell in live),
        all_jobs_equal_round43=len(live) == len(cells) and all(cell['job_equals_round43'] for cell in live),
        rule=('GT-neutral: no case variant with fewer strict TP or more strict FP for CPU k3 than CPU off (first call), known failures '
              'included. CPU against the round-43 Metal hybrid is reported, not gated.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=('gt', *PHASE_SETTINGS), default='formal')
    parser.add_argument('--build', type=Path, default=stage43.DATA_ROOT/'build/stage45-release')
    parser.add_argument('--data-root', type=Path, default=stage43.DATA_ROOT, help='Checkout holding the frozen per-round build/ fixtures')
    parser.add_argument('--protocol', type=Path, default=DOCS/'stage45-cpu-path/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1, help='Parallel GT workers (never for timing phases)')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        root = args.summarize.resolve().parent
        if manifest.get('phase') == 'gt':
            summary = summarize_gt(manifest, root, stage44.reference_archive(REFERENCE/'raw-results-gt.tar.gz', 'gt'),
                                   input_digests(REFERENCE/'input-sha256.json'))
        else:
            summary = summarize(manifest, root, stage44.reference_archive(REFERENCE/'raw-results-timing.tar.gz', 'formal', '-b0'))
        save(args.output, dict(summary, manifest_sha256=sha(args.summarize)))
        return
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Positive timeout required')
    if args.phase != 'gt' and args.jobs != 1:
        parser.error('Timing phases run one worker at a time')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    if args.phase == 'gt':
        cases = [case for case in stage43.all_cases(args.data_root, root/'inputs') if 'hybrid' in case['paths']]
        arms, blocks, measured, warmups = list(DEDUP), 1, 1, None
        plan = [(f"{case['name']}-{CPU}-{dedup}", case, dedup, 0, gt_job(case, dedup)) for case in cases for dedup in DEDUP]
        reference = stage44.reference_archive(REFERENCE/'raw-results-gt.tar.gz', 'gt')
    else:
        cases = stage40.fixture_cases(FIXTURES, list(FORMAL_CASES))
        arms = list(ARMS)
        blocks, measured = PHASE_SETTINGS[args.phase]
        warmups = {name: FRAME_WARMUP_CALLS[name] for name in FORMAL_CASES}
        plan = [(f"{case['name']}-{arm}-b{block}", case, arm, block, job(case, arm, measured, warmups[case['name']]))
                for case in cases for block in range(blocks) for arm in arms[::1 if block % 2 == 0 else -1]]
        reference = stage44.reference_archive(REFERENCE/'raw-results-timing.tar.gz', 'formal', '-b0')
    before = previous.fingerprints(binary, cases)
    host = stage41.host_state(samples=5)
    if args.phase == 'formal' and host['competing_processes']:
        raise RuntimeError('Competing build or benchmark processes: '+json.dumps(host['competing_processes']))
    manifest = dict(version=1, stage=45, phase=args.phase, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(args.protocol), blocks=blocks, measured_calls=measured, frame_warmup_calls=warmups,
        timeout_seconds=args.timeout, cases=cases, arms=arms, method_options=METHODS, dedup_options=DEDUP,
        reference_archive=reference['archive'], reference_sha256=reference['sha256'],
        fixtures_manifest=None if args.phase == 'gt' else str(FIXTURES), host=dict(before=host), executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        def one(item):
            label, case, arm, block, job_dict = item
            return dict(case=case['name'], arm=arm, block=block, **stage44.run_worker(binary, root, label, job_dict, args.timeout))
        if args.phase == 'gt':
            with ThreadPoolExecutor(args.jobs) as pool:
                for result in pool.map(one, plan):
                    manifest['executions'].append(result); save(args.output, manifest)
        else:
            for item in plan:
                result = one(item)
                manifest['executions'].append(result); save(args.output, manifest)
                if result['returncode']:
                    raise RuntimeError(item[0]+': '+result.get('stderr_tail', ''))
        manifest['host']['after'] = stage41.host_state(samples=5)
        after = previous.fingerprints(binary, cases)
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), after_sha256=after, unchanged=before == after)
        save(args.output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(args.output, manifest)
        raise


if __name__ == '__main__':
    main()
