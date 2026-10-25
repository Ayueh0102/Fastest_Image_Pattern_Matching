"""Stage 44: candidate dedup (round 43) on exact parallel peak selection (round 42), measured together.

Round 43 measured candidate_dedup_top_k=3 on the full-GPU path with peak mode 1,
before round 42 was merged; round 42 then made peak mode 2 (batch 8) the
full-GPU research recommendation. Both are on main (847e785) and this round
measures them together on one build. There is no algorithm change: every arm is
an existing option, and k3 stays semantics-changing and off by default.

Arms, one fresh process per case, arm and block, order reversed on odd blocks:
  static  round-40 job: first call, 1 s warm-up, measured repeats on one source
  frames  round-41 sequence job: every call goes through Session::updateSource
  methods hybrid_best, full_gpu_parallel (peak 2, batch 8, fine 4)
  dedup   off (top_k 0) and k3 (top_k 3, radius .5, angle_steps 3)
Static jobs must equal the round-43 formal jobs apart from the peak keys and
the repeat count, and every call's ordered outputs must equal the round-43
archive of the same path and dedup arm (peak 2 equals peak 1 exactly by round
42). Frames outputs must equal the same arm's static output. Independent GT is a
separate check on every call.

Phases: gt (round-43 GT case variants that have a full-GPU path, peak 2 x
{off, k3}; GT and archive identity, no timing), pilot (1 block, screening),
formal (6 blocks), profile (encoder timestamps on full-GPU static, diagnostic
only). Truth stays in this parent; every worker and failed gate is retained;
RSS and named Metal bytes are never added.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import tarfile

import benchmark_candidate_dedup as stage43
import benchmark_gpu_ncc_closure as stage40
import benchmark_new_frame_cost as stage41
import benchmark_parallel_peaks as stage42

previous = stage40.previous
ROOT = stage40.ROOT
DOCS = ROOT/'cpp-core/docs'
REFERENCE = DOCS/'stage43-candidate-dedup'
HYBRID, GPU = stage42.HYBRID, stage42.CANDIDATE
METHODS = {HYBRID: stage42.METHODS[HYBRID], GPU: stage42.METHODS[GPU]}
PATHS = {HYBRID: 'hybrid', GPU: 'full_gpu'}  # round-43 path names
DEDUP = {name: stage43.ARMS[name] for name in ('off', 'k3')}
ARMS = {f'{method}_{mode}_{dedup}': (method, mode, dedup) for mode in ('static', 'frames') for method in METHODS for dedup in DEDUP}
PHASE_ARMS = dict(pilot=list(ARMS), formal=list(ARMS), profile=[f'{GPU}_static_{dedup}' for dedup in DEDUP])
# Keys allowed to differ from the round-43 jobs: the peak mode under test and the per-phase repeat count.
JOB_EXEMPT = ('full_gpu_peak_mode', 'full_gpu_peak_batch', 'repeats')
FORMAL_CASES = stage43.FORMAL_CASES
PRIMARY_CASES = set(stage43.PRIMARY_TIMING_CASES)
PROFILE_CASES = stage43.PROFILE_CASES
SPEED_TARGET = stage43.SPEED_TARGET  # percent: full-GPU k3 against off, static, primary cases
# Round-41 counts were planned for about 1 s at round-40 speed. Peak 2 with k3 runs many128 frames in
# 387 ms instead of 856 ms (implementation smoke), so every case and arm gets three times as many;
# the >= 1 s warm-up wall gate still applies to every worker.
FRAME_WARMUP_CALLS = {name: 3*calls for name, calls in stage42.FRAME_WARMUP_CALLS.items()}
RSS_GATE = stage43.RSS_GATE
FIXTURES = stage43.DATA_ROOT/'build/stage37-full-gpu/formal/manifest.json'
sha, save, distribution, integer = stage40.sha, stage40.save, stage40.distribution, stage40.integer


def dedup_options(dedup):
    options = DEDUP[dedup]
    return dict(candidate_dedup_top_k=options['top_k'], candidate_dedup_radius=options['radius'],
                candidate_dedup_angle_steps=options['angle_steps'])


def job(case, arm, measured, warmup_seconds, profile, warmup_calls):
    method, mode, dedup = ARMS[arm]
    result = stage42.job(case, f'{method}_{mode}', measured, warmup_seconds, profile, warmup_calls)
    result.update(dedup_options(dedup))
    return result


def gt_job(case, dedup):
    result = stage43.make_job(case, 'full_gpu', dedup, 1, 0.)
    result.update(METHODS[GPU])
    return result


def comparable(job_dict):
    return json.dumps({key: value for key, value in job_dict.items() if key not in JOB_EXEMPT}, sort_keys=True)


def reference_archive(path, phase, suffix=''):
    """Round-43 archived jobs and results of one phase, keyed by (case, path, dedup arm), hash-checked."""
    path = Path(path)
    expected = json.loads((path.parent/'archive.json').read_text())['files'][path.name]['sha256']
    if sha(path) != expected:
        raise ValueError('Round-43 archive differs from its archive.json hash: '+path.name)
    found = dict(jobs={}, results={})
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) != 4 or parts[1] != phase or parts[2] not in found or parts[3].startswith('._'):
                continue
            if not parts[3].endswith(suffix+'.json') or parts[3].endswith(('.stdout.json', '.stderr.json')):
                continue
            case, path_name, dedup = parts[3][:-len(suffix+'.json')].rsplit('-', 2)
            found[parts[2]][(case, path_name, dedup)] = json.load(tar.extractfile(member))
    return dict(sha256=expected, archive=path.name, **found)


def dedup_checked(case, method, dedup, calls):
    """Round-43 counter and residency contract on every call (raises on a broken contract)."""
    scales = len(case.get('scales') or case['native_search']['scales'])
    derived = [stage43.record_check(record, PATHS[method], dedup, scales) for record in calls]
    first = (derived[0]['input'], derived[0]['kept'])
    return dict(dedup_input=first[0], dedup_kept=first[1],
                dedup_counts_constant=all((row['input'], row['kept']) == first for row in derived))


def peak_route(record):
    metrics = record['full_gpu']
    return metrics['peak_mode'] == METHODS[GPU]['full_gpu_peak_mode'] and \
        1 <= metrics['peak_batch_maps'] <= METHODS[GPU]['full_gpu_peak_batch']


def without_times(value):
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items() if not key.endswith('_ms')}
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value


def validated_entries(manifest, root, expected, expected_job):
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
        if json.dumps(json.loads(paths['job'].read_text()), sort_keys=True) != json.dumps(expected_job(*key), sort_keys=True):
            raise ValueError('Worker job differs from the declared protocol')
        if stage41.host_entry(entry.get('host')):
            failures.append(dict(case=entry['case'], arm=entry['arm'], block=entry['block'], gate='host_busy'))
        entries[key] = json.loads(paths['result'].read_text())
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    return entries, failures


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
    """Pilot, formal or profile manifest; reference holds the round-43 formal block-0 jobs and results."""
    cases = check_manifest(manifest, reference, PHASE_ARMS)
    phase, arms = manifest['phase'], manifest.get('arms')
    if arms != PHASE_ARMS[phase]:
        raise ValueError('Arms differ from the declared phase arms')
    blocks, measured, warmup_seconds, profile = stage42.phase_settings(phase)
    if (manifest.get('blocks'), manifest.get('measured_calls'), manifest.get('static_warmup_seconds'), manifest.get('profiling')) != \
            (blocks, measured, warmup_seconds, profile):
        raise ValueError('Blocks, measured calls, warm-up or profiling differ from the phase protocol')
    framed = any(ARMS[arm][1] == 'frames' for arm in arms)
    warmups = manifest.get('frame_warmup_calls')
    if framed and (not isinstance(warmups, dict) or set(cases)-warmups.keys() or
                   any(warmups[name] != FRAME_WARMUP_CALLS.get(name) for name in cases)):
        raise ValueError('Frames arms need the stage-44 warm-up call counts')
    expected = {(name, arm, block) for name in cases for arm in arms for block in range(blocks)}
    entries, failures = validated_entries(manifest, root, expected, lambda name, arm, block: job(
        cases[name], arm, measured, warmup_seconds, profile, warmups[name] if framed else None))
    rows = []
    for name, case in cases.items():
        arm_rows, calls = {}, {}
        for arm in arms:
            method, mode, dedup = ARMS[arm]
            def gate(gate_name, block=None, arm=arm):
                failures.append(dict(case=name, arm=arm, gate=gate_name, **({} if block is None else dict(block=block))))
            workers = [(block, entries[(name, arm, block)]) for block in range(blocks)]
            if mode == 'frames':
                row, arm_calls = stage42.frames_arm(case, method, workers, warmups[name], measured, gate)
            else:
                row, arm_calls = stage42.static_arm(case, method, workers, measured, profile)
            row.update(dedup=dedup, **dedup_checked(case, method, dedup, arm_calls))
            arm_rows[arm], calls[arm] = row, arm_calls
        for arm in arms:
            method, mode, dedup = ARMS[arm]
            row, key = arm_rows[arm], (name, PATHS[method], dedup)
            checks = [('independent_ground_truth', row['all_gt_pass']), ('route_and_residency', row['route_valid']),
                      ('requested_algorithm_not_executed', row['requested_algorithm_executed']), ('exact_repeat', row['repeat_exact']),
                      ('sequence_route', row['sequence_route_valid']) if mode == 'frames' else ('static_route', row['static_route_valid']),
                      ('dedup_counts_constant', row['dedup_counts_constant'])]
            if method == GPU:
                row['peak_mode2_route'] = all(peak_route(call) for call in calls[arm])
                checks.append(('peak_mode2_route', row['peak_mode2_route']))
            if mode == 'frames':
                # Same pixels and options: outputs must equal the same arm's static control.
                own = stage42.outputs(calls[f'{method}_static_{dedup}'][0])
                row['same_arm_output_parity'] = all(stage42.outputs(call) == own for call in calls[arm])
                checks.append(('same_arm_output_parity', row['same_arm_output_parity']))
            elif phase != 'profile':
                row['job_equals_round43'] = key in reference['jobs'] and \
                    comparable(job(case, arm, measured, warmup_seconds, profile, None)) == comparable(reference['jobs'][key])
                checks.append(('job_equals_round43', row['job_equals_round43']))
            # Separate from GT: every call's ordered outputs equal the round-43 archive of this path and arm.
            archived = reference['results'].get(key)
            row['exact_vs_round43_archive'] = archived is not None and \
                all(stage42.outputs(call) == stage42.outputs(archived['first']) for call in calls[arm])
            checks.append(('exact_vs_round43_archive', row['exact_vs_round43_archive']))
            failures.extend(dict(case=name, arm=arm, gate=gate_name) for gate_name, passed in checks if not passed)
        comparisons = {}
        for method in METHODS:
            for mode, metric, field in (('static', 'total_ms', 'total_median_ms'), ('frames', 'frame_wall_ms', 'frame_wall_median_ms')):
                off, k3 = f'{method}_{mode}_off', f'{method}_{mode}_k3'
                if off not in arm_rows:
                    continue
                result = stage42.compare(arm_rows[off], arm_rows[k3], metric, field, (off, k3))
                comparisons[f'{k3}_vs_off'] = result
                if phase == 'profile':
                    continue
                rss = arm_rows[off]['peak_rss_bytes']
                if arm_rows[k3]['peak_rss_bytes'] > rss+max(RSS_GATE[0], RSS_GATE[1]*rss):
                    failures.append(dict(case=name, arm=k3, baseline=off, gate='rss_budget'))
                if mode != 'static':
                    continue
                if name in PRIMARY_CASES:
                    if method == GPU:
                        result['primary_target_percent'] = SPEED_TARGET
                        result['primary_target_met'] = result['consistent_direction'] and result['delta_percent'] <= SPEED_TARGET
                        if not result['primary_target_met']:
                            failures.append(dict(case=name, arm=k3, baseline=off, gate='primary_10_percent_improvement'))
                else:
                    for pair in result['paired']:
                        if not pair['within_regression_budget']:
                            failures.append(dict(case=name, arm=k3, baseline=off, gate='time_budget', block=pair['block']))
        frames = {arm: arm_rows[arm] for arm in arms if ARMS[arm][1] == 'frames'}
        if frames:
            pairs = [(f'{HYBRID}_frames_k3', f'{GPU}_frames_k3'), (f'{HYBRID}_frames_off', f'{GPU}_frames_k3'),
                     (f'{HYBRID}_frames_off', f'{GPU}_frames_off')]
            for left, right in pairs:
                comparisons[f'{right}_vs_{left}'] = stage42.compare(frames[left], frames[right], 'frame_wall_ms', 'frame_wall_median_ms', (left, right))
            order = sorted(frames, key=lambda arm: frames[arm]['frame_wall_ms']['median'])
            versus = [stage42.compare(frames[other], frames[order[0]], 'frame_wall_ms', 'frame_wall_median_ms', (other, order[0]))
                      for other in order[1:]]
            # A fastest arm is named only when it beats every other arm in every paired block.
            comparisons['frames_ranking'] = dict(order=order, median_ms={arm: frames[arm]['frame_wall_ms']['median'] for arm in order},
                fastest=order[0] if all(result['faster'] == order[0] for result in versus) else None,
                versus=[dict(other=result['baseline'], delta_percent=result['delta_percent'],
                             faster_blocks=result['candidate_faster_blocks'], consistent=result['consistent_direction']) for result in versus])
        rows.append(dict(name=name, truth_count=len(case['truth']), refine=case['workflow'] == 'full_rescue',
                         frame_warmup_calls=warmups.get(name) if framed else None, arms=arm_rows, comparisons=comparisons))
    arm_list = [row['arms'][arm] for row in rows for arm in arms]
    primary = [row['comparisons'].get(f'{GPU}_static_k3_vs_off', {}).get('primary_target_met') for row in rows if row['name'] in PRIMARY_CASES]
    return dict(phase=phase, formal_decision_eligible=phase == 'formal', profiling=profile, reference=reference['archive'],
        reference_sha256=reference['sha256'], cases=rows, failed_gates=failures,
        all_accuracy_pass=all(row['all_gt_pass'] for row in arm_list),
        all_routes_pass=all(row['route_valid'] and row['requested_algorithm_executed'] and row.get('peak_mode2_route', True) and
                            row.get('sequence_route_valid', row.get('static_route_valid')) for row in arm_list),
        all_repeat_exact=all(row['repeat_exact'] for row in arm_list),
        all_exact_round43=all(row['exact_vs_round43_archive'] for row in arm_list),
        primary_target_met=all(primary) if phase == 'formal' and len(primary) == len(PRIMARY_CASES) else None,
        frames_fastest={row['name']: row['comparisons']['frames_ranking']['fastest'] for row in rows if 'frames_ranking' in row['comparisons']},
        counts=dict(fresh_processes=len(entries), cold_calls=sum(row['cold_calls'] for row in arm_list),
            measured_calls=sum(row['measured_calls'] for row in arm_list), gt_evaluated_calls=sum(row['gt_evaluated_calls'] for row in arm_list)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note=('Static total_ms is the round-40 scope (run only, after a first call and a 1 s warm-up); frames frame_wall_ms is the '
                     'round-41 scope (Session::updateSource plus run). A faster arm is named only when every paired block agrees with the '
                     'pooled medians. Speedups are same-build comparisons and are never multiplied with earlier rounds.'),
        memory_note='Peak process RSS, named Metal buffer lengths and hybrid pose-context buffers are separate scopes and are never added.')


def summarize_gt(manifest, root, reference):
    """GT phase: peak 2 x {off, k3} on the round-43 full-GPU case variants; reference is the round-43 GT archive."""
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
        job_path = previous.worker_path(root, entry['job'])
        if json.dumps(json.loads(job_path.read_text()), sort_keys=True) != json.dumps(gt_job(cases[key[0]], key[1]), sort_keys=True):
            raise ValueError('GT worker job differs from the declared protocol')
        worker = stage43.load_worker(root, entry)
        calls = stage43.worker_calls(worker)
        name, dedup = key[0], key[1]
        cell = stage43.gt_cell(cases[name], 'full_gpu', dedup, worker)
        ref = (name, 'full_gpu', dedup)
        cell['peak_mode2_route'] = all(peak_route(call) for call in calls)
        cell['job_equals_round43'] = ref in reference['jobs'] and comparable(gt_job(cases[name], dedup)) == comparable(reference['jobs'][ref])
        archived = reference['results'].get(ref)
        cell['exact_vs_round43_archive'] = archived is not None and all(
            stage42.outputs(call) == stage42.outputs(archived['first']) and
            without_times(call['candidate_dedup']) == without_times(archived['first']['candidate_dedup']) for call in calls)
        for gate in ('peak_mode2_route', 'job_equals_round43', 'exact_vs_round43_archive'):
            if not cell[gate]:
                failures.append(dict(gate=gate, case=name, arm=dedup))
        if not cell['exact_repeat'] or not cell['same_counts']:
            failures.append(dict(gate='repeat', case=name, arm=dedup))
        cells[key] = cell
    if set(cells) != expected:
        raise ValueError('GT workers do not cover every case and dedup arm exactly')
    rows, total = [], dict(cases=0, regressions=[], improvements=[], tp_delta=0, fp_delta=0, fn_delta=0,
                           matches_identical=0, proposals_identical=0, input=0, kept=0)
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
        rows.append(row)
    total['kept_fraction'] = total['kept']/total['input'] if total['input'] else None
    live = [cell for cell in cells.values() if cell]
    return dict(phase='gt', reference=reference['archive'], reference_sha256=reference['sha256'], cases=rows, aggregate=total,
        failures=failures, gt_neutral=not total['regressions'] and total['cases'] == len(cases),
        all_exact_round43=len(live) == len(cells) and all(cell['exact_vs_round43_archive'] for cell in live),
        all_jobs_equal_round43=len(live) == len(cells) and all(cell['job_equals_round43'] for cell in live),
        rule=('GT-neutral: no case variant with fewer strict TP or more strict FP for k3 than off (first call), known failures '
              'included. Output identity with the round-43 peak-1 archive is a separate check.'))


def run_worker(binary, root, label, job_dict, timeout):
    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
    save(jp, job_dict)
    state = stage41.host_state()
    print('Running '+label, flush=True)
    try:
        child = subprocess.run([str(binary), str(jp)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        jp.with_suffix('.stdout.txt').write_text(error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or '')
        jp.with_suffix('.stderr.txt').write_text(error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or '')
        raise
    jp.with_suffix('.stdout.txt').write_text(child.stdout)
    jp.with_suffix('.stderr.txt').write_text(child.stderr)
    entry = dict(job=str(jp.relative_to(root)), job_sha256=sha(jp), returncode=child.returncode, host=state)
    if child.returncode == 0:
        save(rp, json.loads(child.stdout))
        entry.update(result=str(rp.relative_to(root)), result_sha256=sha(rp))
    else:
        entry['stderr_tail'] = child.stderr[-4000:]
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=('gt', *PHASE_ARMS), default='formal')
    parser.add_argument('--build', type=Path, default=stage43.DATA_ROOT/'build/stage44-release')
    parser.add_argument('--data-root', type=Path, default=stage43.DATA_ROOT, help='Checkout holding the frozen per-round build/ fixtures')
    parser.add_argument('--protocol', type=Path, default=DOCS/'stage44-dedup-peak2/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1, help='Parallel GT workers (never for timing phases)')
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        root = args.summarize.resolve().parent
        if manifest.get('phase') == 'gt':
            summary = summarize_gt(manifest, root, reference_archive(REFERENCE/'raw-results-gt.tar.gz', 'gt'))
        else:
            summary = summarize(manifest, root, reference_archive(REFERENCE/'raw-results-timing.tar.gz', 'formal', '-b0'))
        save(args.output, dict(summary, manifest_sha256=sha(args.summarize)))
        return
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Positive timeout required')
    if args.phase != 'gt' and args.jobs != 1:
        parser.error('Timing and profile phases run one worker at a time')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    if args.phase == 'gt':
        cases = [case for case in stage43.all_cases(args.data_root, root/'inputs') if 'full_gpu' in case['paths']]
        arms, blocks, measured, warmup_seconds, profile, warmups = list(DEDUP), 1, 1, 0., False, None
        plan = [(f"{case['name']}-full_gpu-{dedup}", case, dedup, 0, gt_job(case, dedup)) for case in cases for dedup in DEDUP]
        reference = reference_archive(REFERENCE/'raw-results-gt.tar.gz', 'gt')
    else:
        names = list(PROFILE_CASES if args.phase == 'profile' else FORMAL_CASES)
        cases = stage40.fixture_cases(FIXTURES, names)
        arms = PHASE_ARMS[args.phase]
        blocks, measured, warmup_seconds, profile = stage42.phase_settings(args.phase)
        framed = any(ARMS[arm][1] == 'frames' for arm in arms)
        warmups = {name: FRAME_WARMUP_CALLS[name] for name in names} if framed else None
        plan = [(f"{case['name']}-{arm}-b{block}", case, arm, block,
                 job(case, arm, measured, warmup_seconds, profile, warmups[case['name']] if framed else None))
                for case in cases for block in range(blocks) for arm in arms[::1 if block % 2 == 0 else -1]]
        reference = reference_archive(REFERENCE/'raw-results-timing.tar.gz', 'formal', '-b0')
    before = previous.fingerprints(binary, cases)
    host = stage41.host_state(samples=5)
    if args.phase in ('formal', 'profile') and host['competing_processes']:
        raise RuntimeError('Competing build or benchmark processes: '+json.dumps(host['competing_processes']))
    manifest = dict(version=1, stage=44, phase=args.phase, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(args.protocol), blocks=blocks, measured_calls=measured, static_warmup_seconds=warmup_seconds,
        profiling=profile, frame_warmup_calls=warmups, timeout_seconds=args.timeout, cases=cases, arms=arms,
        method_options=METHODS, dedup_options=DEDUP, reference_archive=reference['archive'], reference_sha256=reference['sha256'],
        fixtures_manifest=None if args.phase == 'gt' else str(FIXTURES), host=dict(before=host), executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        def one(item):
            label, case, arm, block, job_dict = item
            return dict(case=case['name'], arm=arm, block=block, **run_worker(binary, root, label, job_dict, args.timeout))
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
