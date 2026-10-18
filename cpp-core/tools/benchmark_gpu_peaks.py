"""Stage 38: exact tiled GPU peak selection and separately instrumented timings.

Uses the frozen stage-37 cases and independent truth, never sends truth to the
worker, and preserves every result. Profile runs are diagnostic: their timing is
never pooled into unprofiled speed comparisons. Memory scopes must not be added.
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

import benchmark_full_gpu as previous
from many_object_accuracy import evaluate

ROOT = previous.ROOT
DEFAULT_CASES = 'many128_pose,many512_pose,controlled6,8k_t256_angle180_3scales'
PROFILE_CASES = 'many128_pose,many512_pose'
METHODS = {
    'hybrid_best': dict(full_gpu=False, coarse_peak_mode=1, full_gpu_peak_mode=0),
    'full_gpu': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=0),
    'full_gpu_tiled': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=1),
}
GPU_METHODS = {'full_gpu', 'full_gpu_tiled'}
PRIMARY_CASES = {'many128_pose', 'many512_pose'}
sha, save, distribution = previous.sha, previous.save, previous.distribution


def job(case, method, repeats, warmup, profile=False):
    result = previous.job(case, 'full_gpu' if method in GPU_METHODS else 'hybrid_best', repeats, warmup)
    result.update(METHODS[method])
    result['full_gpu_profiling'] = profile
    return result


def fixture_cases(path, names):
    archived = json.loads(path.read_text())
    if archived.get('complete') is not True or archived.get('unchanged') is not True:
        raise ValueError('Fixtures require a completed unchanged stage-37 manifest')
    cases_by_name = {case['name']: case for case in archived['cases']}
    if set(names)-cases_by_name.keys():
        raise ValueError('Unknown archived cases: '+','.join(sorted(set(names)-cases_by_name.keys())))
    cases = [copy.deepcopy(cases_by_name[name]) for name in names]
    for case in cases:
        for key in ('source', 'template'):
            if sha(case[key]) != archived['input_sha256'].get(case[key]):
                raise ValueError('Fixture input differs from stage-37 archive: '+case[key])
    return cases


def integer(value, name):
    previous.finite_nonnegative(value, name)
    if type(value) is not int:
        raise ValueError('Noninteger '+name)
    return value


def profile_record(record, requested):
    """Validate timestamp evidence while preserving unsupported measurements."""
    metrics = record['full_gpu']
    if type(metrics.get('profile_requested')) is not bool or metrics['profile_requested'] != requested:
        raise ValueError('Mismatched profile_requested')
    if type(metrics.get('profile_available')) is not bool:
        raise ValueError('Missing profile_available boolean')
    if not isinstance(metrics.get('profile_status'), str) or not metrics['profile_status']:
        raise ValueError('Missing profile_status')
    for name in ('profile_samples', 'profile_counter_bytes', 'profile_resolve_bytes', 'peak_scratch_bytes'):
        integer(metrics.get(name), name)
    if metrics['peak_scratch_bytes'] > metrics['gpu_bytes']:
        raise ValueError('Named peak scratch allocation exceeds total named Metal bytes')
    previous.finite_nonnegative(metrics.get('profile_host_ms'), 'profile_host_ms')
    stages = metrics.get('stages')
    if not isinstance(stages, list):
        raise ValueError('Missing stages array')
    available = metrics['profile_available']
    if not available:
        if stages:
            raise ValueError('Unavailable profile has stage measurements')
        # Older implementations can initialize this numeric member to zero. It
        # conveys no measurement and is represented as null in the summary.
        if (isinstance(metrics.get('profile_gpu_ms'), bool) or
                metrics.get('profile_gpu_ms') not in (None, 0, 0.)):
            raise ValueError('Unavailable profile has a nonzero stage duration')
        return None
    if not requested or not stages or metrics['profile_samples'] < 2:
        raise ValueError('Available profile lacks a request/stages/samples')
    names = []
    encoders = 0
    for stage in stages:
        if not isinstance(stage.get('name'), str) or not stage['name']:
            raise ValueError('Invalid profile stage name')
        names.append(stage['name'])
        previous.finite_nonnegative(stage.get('gpu_ms'), 'stage.gpu_ms')
        if integer(stage.get('encoders'), 'stage.encoders') == 0:
            raise ValueError('Empty profile encoder group')
        encoders += stage['encoders']
    if len(set(names)) != len(names):
        raise ValueError('Duplicate profile stages')
    total = previous.finite_nonnegative(metrics.get('profile_gpu_ms'), 'profile_gpu_ms')
    stage_sum = sum(stage['gpu_ms'] for stage in stages)
    if not math.isclose(stage_sum, total, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError('Profile stage sum differs from profile_gpu_ms')
    # Two timestamp samples delimit each encoder interval. Sample count reports
    # used samples, not the capacity of the Metal counter allocation.
    if metrics['profile_samples'] != encoders*2:
        raise ValueError('Timestamp sample count differs from encoder coverage')
    device_available = previous.device_timestamp_available(metrics)
    remainder = metrics['device_ms']-stage_sum if device_available else None
    # Retain signed remainder: distinct timestamp instruments can disagree.
    return dict(stages={stage['name']: dict(gpu_ms=stage['gpu_ms'], encoders=stage['encoders']) for stage in stages},
                stage_sum_ms=stage_sum, device_remainder_ms=remainder)


def summarize(manifest, root):
    if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
        raise ValueError('Experiment must be complete with unchanged fingerprints')
    fingerprints = {name: manifest.get(name) for name in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(fingerprints.values()) or manifest.get('after_sha256') != fingerprints:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    requested = manifest.get('profiling')
    if type(requested) is not bool:
        raise ValueError('Missing profiling boolean')
    methods = manifest['methods']
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or set(methods)-METHODS.keys():
        raise ValueError('Empty, duplicate, or unknown methods')
    if requested and not set(methods) <= GPU_METHODS:
        raise ValueError('Profile runs require full-GPU methods')
    if not requested and not GPU_METHODS <= set(methods):
        raise ValueError('Formal runs require original and tiled GPU methods')
    for name in ('blocks', 'repeats'):
        if integer(manifest[name], name) < 1:
            raise ValueError('Empty '+name)
    previous.finite_nonnegative(manifest['warmup_seconds'], 'warmup_seconds')
    cases = {case['name']: case for case in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Empty or duplicate cases')
    expected = {(name, method, block) for name in cases for method in methods for block in range(manifest['blocks'])}
    entries = {}
    for entry in manifest['executions']:
        key = (entry['case'], entry['method'], entry['block'])
        if key not in expected or key in entries:
            raise ValueError('Unexpected or duplicate worker '+str(key))
        paths = {name: previous.worker_path(root, entry[name]) for name in ('job', 'result')}
        for name, path in paths.items():
            if sha(path) != entry[name+'_sha256']:
                raise ValueError('Changed '+name+': '+entry[name])
        work_job = json.loads(paths['job'].read_text())
        expected_job = job(cases[entry['case']], entry['method'], manifest['repeats'], manifest['warmup_seconds'], requested)
        # Python equality treats bool as int; serialized equality preserves the
        # worker's strict boolean/mode contracts and every shared setting.
        if json.dumps(work_job, sort_keys=True) != json.dumps(expected_job, sort_keys=True):
            raise ValueError('Worker job differs from declared shared search settings')
        worker = json.loads(paths['result'].read_text())
        if len(worker['runs']) != manifest['repeats']:
            raise ValueError('Missing or extra measured repeats')
        previous.finite_nonnegative(worker['construct_ms'], 'construct_ms')
        integer(worker['final_memory']['peak_rss_bytes'], 'peak_rss_bytes')
        if type(worker.get('exact_repeat')) is not bool:
            raise ValueError('Missing exact_repeat boolean')
        entries[key] = worker
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    failures, rows = [], []
    for case_name, case in cases.items():
        reference_method = 'full_gpu' if 'full_gpu' in methods else methods[0]
        reference = entries[(case_name, reference_method, 0)]['first']
        method_rows = {}
        for method in methods:
            blocks = [(block, entries[(case_name, method, block)]) for block in range(manifest['blocks'])]
            calls = [record for _, worker in blocks for record in [worker['first'], *worker['runs']]]
            runs = [record for _, worker in blocks for record in worker['runs']]
            gpu = method in GPU_METHODS
            routes = [previous.validate_record(record, gpu) for record in calls]
            profiles = []
            if gpu:
                for record in calls:
                    if type(record['full_gpu'].get('peak_mode')) is not int:
                        raise ValueError('Invalid peak_mode')
                    routes.append(record['full_gpu']['peak_mode'] == METHODS[method]['full_gpu_peak_mode'])
                    profiles.append(profile_record(record, requested))
            accuracy = [evaluate(record['matches'], case['truth'])['strict'] for record in calls]
            gt_pass = all(row['fp'] == row['fn'] == 0 for row in accuracy)
            exact_matches = all(record['matches'] == reference['matches'] for record in calls)
            exact_proposals = all(record['proposals'] == reference['proposals'] for record in calls)
            repeat_exact = all(worker['exact_repeat'] for _, worker in blocks) and all(
                record['matches'] == calls[0]['matches'] and record['proposals'] == calls[0]['proposals'] for record in calls)
            measured_metrics = [record['full_gpu'] for record in runs] if gpu else []
            timing_flags = [previous.device_timestamp_available(record) for record in measured_metrics]
            gpu_fields = (*previous.GPU_FIELDS, 'peak_scratch_bytes', 'profile_counter_bytes', 'profile_resolve_bytes', 'profile_host_ms')
            gpu_metrics = {name: distribution([record[name] for record in measured_metrics]) for name in gpu_fields if name != 'device_ms'} if gpu else None
            if gpu:
                values = [record['device_ms'] for record, available in zip(measured_metrics, timing_flags) if available]
                gpu_metrics['device_ms'] = distribution(values) if values else None
            measured_profile = [profile_record(record, requested) for record in runs] if gpu else []
            available_profile = [row for row in measured_profile if row is not None]
            stage_names = sorted({name for row in available_profile for name in row['stages']})
            profiling = None
            if gpu:
                profiling = dict(requested=requested, available_calls=sum(row is not None for row in profiles),
                    total_calls=len(calls), measured_available_calls=len(available_profile), measured_calls=len(runs),
                    status_counts={status: sum(record['full_gpu']['profile_status'] == status for record in calls)
                                   for status in sorted({record['full_gpu']['profile_status'] for record in calls})},
                    sample_counts=sorted({record['full_gpu']['profile_samples'] for record in calls}),
                    stages={name: dict(gpu_ms=distribution([row['stages'][name]['gpu_ms'] for row in available_profile if name in row['stages']]),
                                       encoders=sorted({row['stages'][name]['encoders'] for row in available_profile if name in row['stages']}),
                                       measured_available_calls=sum(name in row['stages'] for row in available_profile)) for name in stage_names},
                    stage_sum_ms=distribution([row['stage_sum_ms'] for row in available_profile]) if available_profile else None,
                    device_remainder_ms=distribution([row['device_remainder_ms'] for row in available_profile if row['device_remainder_ms'] is not None])
                        if any(row['device_remainder_ms'] is not None for row in available_profile) else None)
            method_rows[method] = dict(total_ms=distribution([row['total_ms'] for row in runs]),
                blocks=[dict(block=block, median_ms=statistics.median(row['total_ms'] for row in worker['runs'])) for block, worker in blocks],
                first_ms=[worker['first']['total_ms'] for _, worker in blocks], construct_ms=[worker['construct_ms'] for _, worker in blocks],
                peak_rss_bytes=max(worker['final_memory']['peak_rss_bytes'] for _, worker in blocks),
                retained_gpu_bytes=max(record['full_gpu']['gpu_bytes'] if gpu else record['frame_coarse_gpu_bytes']+record['frame_fine_gpu_bytes'] for record in calls),
                retained_gpu_scope='all named full-GPU Metal buffers' if gpu else 'frame coarse + fine buffers only; other owners excluded',
                measured_gpu_metrics=gpu_metrics, profiling=profiling,
                device_timestamp_available_measured_calls=sum(timing_flags) if gpu else None,
                all_gt_pass=gt_pass, accuracy_counts=sorted({(row['tp'], row['fp'], row['fn']) for row in accuracy}),
                max_errors={name: max((row['max_errors'][name] for row in accuracy if row['max_errors'][name] is not None), default=None)
                            for name in ('position_px', 'angle_deg', 'relative_scale')},
                route_valid=all(routes), full_pipeline_residency_verified=all(routes) if gpu else False,
                ordered_matches_equal_original_gpu=exact_matches, ordered_proposals_equal_original_gpu=exact_proposals,
                repeat_exact=repeat_exact, proposal_counts=sorted({len(row['proposals']) for row in calls}),
                first_calls=len(blocks), measured_calls=len(runs), gt_evaluated_calls=len(calls))
            for name, passed in [('independent_ground_truth', gt_pass), ('route_and_residency', all(routes)), ('exact_repeat', repeat_exact)]:
                if not passed:
                    failures.append(dict(case=case_name, method=method, gate=name))
            if method == 'full_gpu_tiled' and 'full_gpu' in methods:
                for name, passed in [('exact_ordered_matches', exact_matches), ('exact_ordered_proposals', exact_proposals)]:
                    if not passed:
                        failures.append(dict(case=case_name, method=method, gate=name))
        comparisons = {}
        if not requested:
            measured = method_rows['full_gpu_tiled']
            for baseline_name in ('full_gpu', 'hybrid_best'):
                if baseline_name not in method_rows:
                    continue
                baseline = method_rows[baseline_name]
                paired = []
                for left, right in zip(baseline['blocks'], measured['blocks']):
                    passed = right['median_ms'] <= left['median_ms']+max(2., .1*left['median_ms'])
                    paired.append(dict(block=left['block'], delta_percent=100*(right['median_ms']/left['median_ms']-1), within_regression_budget=passed))
                    if not passed:
                        failures.append(dict(case=case_name, method='full_gpu_tiled', baseline=baseline_name, gate='time_budget', block=left['block']))
                memory_ok = measured['peak_rss_bytes'] <= baseline['peak_rss_bytes']+max(8*1024**2, .1*baseline['peak_rss_bytes'])
                if not memory_ok:
                    failures.append(dict(case=case_name, method='full_gpu_tiled', baseline=baseline_name, gate='rss_budget'))
                semantic_pass = all(row['all_gt_pass'] and row['route_valid'] and row['repeat_exact'] for row in (baseline, measured)) and measured['ordered_matches_equal_original_gpu'] and measured['ordered_proposals_equal_original_gpu']
                delta = 100*(measured['total_ms']['median']/baseline['total_ms']['median']-1)
                target = delta <= -20. if baseline_name == 'full_gpu' and case_name in PRIMARY_CASES else None
                if target is False:
                    failures.append(dict(case=case_name, method='full_gpu_tiled', baseline=baseline_name, gate='primary_20_percent_improvement'))
                comparisons[baseline_name] = dict(delta_percent=delta, primary_20_percent_improvement=target,
                    paired=paired, memory_within_budget=memory_ok,
                    candidate_for_adoption=semantic_pass and memory_ok and all(row['within_regression_budget'] for row in paired))
        rows.append(dict(name=case_name, truth_count=len(case['truth']), methods=method_rows, tiled_comparisons=comparisons))
    return dict(profiling=requested, cases=rows, failed_gates=failures,
        all_accuracy_pass=all(row['methods'][method]['all_gt_pass'] for row in rows for method in methods),
        all_gpu_resident=all(row['methods'][method]['full_pipeline_residency_verified'] for row in rows for method in methods if method in GPU_METHODS),
        all_tiled_exact_original=all(row['methods']['full_gpu_tiled']['ordered_matches_equal_original_gpu'] and row['methods']['full_gpu_tiled']['ordered_proposals_equal_original_gpu'] for row in rows) if GPU_METHODS <= set(methods) else None,
        all_tiled_vs_original_gates_pass=all(row['tiled_comparisons']['full_gpu']['candidate_for_adoption'] for row in rows) if not requested else None,
        all_primary_20_percent_target_pass=all(row['tiled_comparisons']['full_gpu']['primary_20_percent_improvement'] and row['tiled_comparisons']['full_gpu']['candidate_for_adoption']
            for row in rows if row['name'] in PRIMARY_CASES) if not requested and PRIMARY_CASES <= cases.keys() else None,
        all_tiled_vs_hybrid_gates_pass=all(row['tiled_comparisons']['hybrid_best']['candidate_for_adoption'] for row in rows) if not requested and 'hybrid_best' in methods else None,
        counts=dict(fresh_processes=len(entries), first_calls=len(entries), measured_calls=len(entries)*manifest['repeats'], gt_evaluated_calls=len(entries)*(manifest['repeats']+1)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note='First calls are separate. Distributions use measured repeats only. Profile diagnostics never enter unprofiled speed gates; unavailable stages and remainder are null. Signed device remainder includes unsampled work, gaps, and timestamp-instrument differences.',
        memory_note='Peak process RSS and retained named Metal buffer lengths differ in scope and must not be added; GPU buffer lengths are not physical VRAM. Counter payload capacity is logical and opaque to Metal allocation accounting. Resolved timestamp bytes are transient CPU NSData payload, excluded from gpu_bytes; peak scratch bytes are already included in full-GPU gpu_bytes.',
        parity_note='Independent one-to-one GT accuracy plus exact ordered matches AND proposals against the original full-GPU route are required for tiled adoption.',
        residency_note='Audited implementation counters, not a hardware trace: one command buffer, one final wait, no intermediate readback or image-dependent CPU algorithm.')


def profile_overhead(profile_manifest, profile_summary, baseline_manifest, baseline_summary):
    if not profile_manifest['profiling'] or baseline_manifest['profiling']:
        raise ValueError('Overhead requires profiled current and unprofiled reference manifests')
    same = all(profile_manifest[name] == baseline_manifest[name] for name in ('source_sha256', 'binary_sha256'))
    baseline = {row['name']: row for row in baseline_summary['cases']}
    rows = []
    for row in profile_summary['cases']:
        if row['name'] not in baseline:
            continue
        source_case = next(case for case in profile_manifest['cases'] if case['name'] == row['name'])
        target_case = next(case for case in baseline_manifest['cases'] if case['name'] == row['name'])
        case_same = source_case == target_case and all(profile_manifest['input_sha256'].get(source_case[key]) == baseline_manifest['input_sha256'].get(target_case[key]) for key in ('source', 'template'))
        for method, measured in row['methods'].items():
            if method not in baseline[row['name']]['methods']:
                continue
            ref = baseline[row['name']]['methods'][method]
            comparable = same and case_same
            device, ref_device = measured['measured_gpu_metrics']['device_ms'], ref['measured_gpu_metrics']['device_ms']
            rows.append(dict(case=row['name'], method=method, comparable=comparable,
                total_delta_percent=100*(measured['total_ms']['median']/ref['total_ms']['median']-1) if comparable else None,
                device_delta_percent=100*(device['median']/ref_device['median']-1) if comparable and device and ref_device and ref_device['median'] > 0 else None,
                note='Separate fresh processes, diagnostic overhead includes ordinary run variance; source/binary/input/search settings must match.'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures-manifest', type=Path, default=ROOT/'build/stage37-full-gpu/formal/manifest.json')
    parser.add_argument('--cases')
    parser.add_argument('--methods')
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=float, default=1.)
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--unprofiled-manifest', type=Path, help='Separate formal run used only to report profiler overhead')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        summary = summarize(manifest, args.summarize.resolve().parent)
        if args.unprofiled_manifest:
            baseline = json.loads(args.unprofiled_manifest.read_text())
            summary['profile_overhead'] = profile_overhead(manifest, summary, baseline, summarize(baseline, args.unprofiled_manifest.resolve().parent))
            summary['unprofiled_manifest_sha256'] = sha(args.unprofiled_manifest)
        save(args.output, summary)
        return
    if args.unprofiled_manifest:
        parser.error('--unprofiled-manifest is available only with --summarize')
    if args.blocks < 1 or args.repeats < 1 or not 0 <= args.warmup <= 60 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Positive blocks/repeats/timeout and warmup in 0..60 required')
    names = previous.selected_names(args.cases or (PROFILE_CASES if args.profile else DEFAULT_CASES), None, 'cases')
    methods = previous.selected_names(args.methods or ('full_gpu,full_gpu_tiled' if args.profile else ','.join(METHODS)), METHODS, 'methods')
    if args.profile and not set(methods) <= GPU_METHODS or not args.profile and not GPU_METHODS <= set(methods):
        parser.error('Profile requires only GPU methods; formal requires both original and tiled GPU methods')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    cases = fixture_cases(args.fixtures_manifest, names)
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    before = previous.fingerprints(binary, cases)
    manifest = dict(version=1, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        profiling=args.profile, blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup,
        timeout_seconds=args.timeout, cases=cases, methods=methods, method_options={name: METHODS[name] for name in methods},
        fixtures_manifest=str(args.fixtures_manifest.resolve()), fixtures_manifest_sha256=sha(args.fixtures_manifest),
        executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        for case in cases:
            for block in range(args.blocks):
                for method in methods[::1 if block % 2 == 0 else -1]:
                    label = f"{case['name']}-{method}-b{block}"
                    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                    save(jp, job(case, method, args.repeats, args.warmup, args.profile))
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
                    manifest['executions'].append(dict(case=case['name'], method=method, block=block,
                        job=str(jp.relative_to(root)), job_sha256=sha(jp), result=str(rp.relative_to(root)), result_sha256=sha(rp)))
                    save(args.output, manifest)
        after = previous.fingerprints(binary, cases)
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
