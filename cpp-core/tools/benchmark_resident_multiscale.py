"""Stage33: compose three existing Metal resident solvers with the multiscale frontend.

Serial fresh processes, fixed forward/reverse blocks and no warmup. Ground truth
and scene construction metadata never reach the native worker. Exact frontend
work, bounded one-to-one final agreement, independent accuracy and performance
are separate gates. Common errors remain errors. Replay needs only hashed local
protocol/jobs/results and works after moving the evidence directory.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess

from benchmark_fine_fused import (CAPACITIES, CLOCKS, COUNTERS, FUSION_COUNTS,
    FUSION_CLOCKS, numeric_tree, route as fused_route, source_files)
from benchmark_fine_strip import sha, save, dist, max_present, checked
from benchmark_gray_loader import valid_memory, ensure_empty_output, BINARY_NAMES
from benchmark_native_pipeline import native_job
from benchmark_pose_simd_pipeline import final_comparison, PAIR_TOLERANCE
from benchmark_run_diagnostics import (count, finite_ms, png_shape,
                                      calls, stage_clocks, pair_metrics)
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['resident0', 'fused256', 'simd128']
CASE_NAMES = ['controlled6', 'many128_pose', 'many512_pose', 'many512_pose_fullres_subpixel',
              'new_texture12', 'new_small12', 'degraded12', 'empty_scene']
REFINEMENT_COUNTS = ('refinement_dispatches', 'refinement_command_buffers', 'refinement_cpu_waits',
                     'refinement_auxiliary_bytes', 'refinement_threads', 'refinement_simd_width',
                     'refinement_threadgroup_bytes', 'rescued_seeds')
REFINEMENT_CLOCKS = ('refinement_setup_ms', 'refinement_encode_ms', 'refinement_wait_ms',
                     'refinement_verify_ms', 'refinement_device_ms', 'pose_create_ms')
IMAGE_BYTES = ('pose_host_image_bytes', 'pose_gpu_image_bytes', 'pose_shared_image_bytes',
               'pose_unique_image_bytes')
POSE_BYTES = (*IMAGE_BYTES, 'pose_host_bytes', 'pose_gpu_bytes', 'pose_conversion_temporary_bytes')
WORK_COUNTS = (*COUNTERS, 'fine_ncc_mode', 'fine_row_bytes', 'fine_row_dispatches',
               'async_submitted', 'async_completed', 'async_maps', 'async_fallback_maps', 'async_max_pending')
WORK_CLOCKS = ('async_copy_ms', 'async_device_ms', 'async_wait_ms', 'copy_ms', 'device_ms',
               'fine_ncc_ms', 'fine_normalization_ms', 'fine_repair_ms', 'fine_rotation_ms',
               'fine_stats_device_ms', 'fine_wait_ms', 'normalization_ms', 'wait_ms')
OBSERVATION_COUNTS = ('calls', 'requests', 'within_call_duplicates', 'cross_call_duplicates',
    'untracked_call_keys', 'tracked_unique_keys', 'history_limit_keys', 'history_key_payload_bytes',
    'history_bucket_count', 'peak_call_key_capacity_bytes', 'warp_batches', 'zero_repair_batches',
    'guard_words', 'nonzero_guard_words')
SEARCH_COUNTS = (*OBSERVATION_COUNTS, 'profiling_enabled', 'angle_reuse_enabled',
                 'logical_requests', 'reused_maps', 'cache_peak_bytes')


def job_for(case, method):
    if method not in METHODS:
        raise ValueError('Unknown resident method')
    job = native_job(case, 'native_tile', 1, 0.)
    job.update(fine_ncc_mode=2, fine_profiling=False, fine_angle_reuse=True, fine_fused=True,
               pose_resident_mode=METHODS.index(method), pose_input_mode=1, pose_storage_mode=1,
               gray_load_mode='direct_gray8', run_resource_diagnostics=False)
    return job



def image_shape(path):
    with Path(path).open('rb') as stream:
        head = stream.read(54)
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        return png_shape(path)
    if len(head) >= 26 and head[:2] == b'BM' and struct.unpack('<I', head[14:18])[0] >= 40:
        width, height = struct.unpack('<ii', head[18:26])
        if width > 0 and height != 0:
            return [width, abs(height)]
    raise ValueError('Unsupported or invalid benchmark image: ' + str(path))


def loader_valid(raw, reference):
    try:
        load = raw['load']
        if (load['mode'] != 'direct_gray8' or not finite_ms(load['decode_ms']) or
                not valid_memory(raw['after_load_memory']) or
                strip_clocks(load) != strip_clocks(reference['load'])):
            return False
        return all(type(load[k]['direct_gray8']) is bool and count(load[k]['rgb_temporary_bytes']) and
                   ((load[k]['rgb_temporary_bytes'] == 0) == load[k]['direct_gray8'])
                   for k in ('source', 'template'))
    except (KeyError, TypeError, ValueError):
        return False


def baseline(build, protocol):
    recorded = json.loads(Path(protocol).read_text())['binary_sha256']
    verified = {}
    for name in BINARY_NAMES:
        matches = [digest for path, digest in recorded.items()
                   if path.endswith('/run-diagnostics/experiments/' + name)]
        if len(matches) != 1 or sha(Path(build) / 'experiments' / name) != matches[0]:
            raise ValueError('Frozen stage32 binary differs: ' + name)
        verified[name] = matches[0]
    return verified


def worker_order(cases):
    return [(case['name'], method, block) for case in cases for block in range(2)
            for method in (METHODS if block == 0 else METHODS[::-1])]


def strip_clocks(value):
    if isinstance(value, dict):
        return {k: strip_clocks(v) for k, v in value.items() if not k.endswith('_ms')}
    if isinstance(value, list):
        return [strip_clocks(v) for v in value]
    return value


def frontend_state(run):
    return strip_clocks({k: run[k] for k in ('generation', 'models_matched', *CAPACITIES,
                                            'work', 'fine_search', 'fine_fusion')})


def valid_clock(value, device=False):
    return finite_ms(value) or (device and value == -1 and type(value) is not bool)


def frontend_valid(run, reference, case):
    try:
        if (frontend_state(run) != frontend_state(reference) or
                any(not count(run[k]) for k in CAPACITIES) or
                run['models_matched'] != len(case['scales']) or
                len(run['work']) != len(case['scales']) or
                len(run['fine_search']) != len(run['work']) or
                len(run['fine_fusion']) != len(run['work'])):
            return False
        for work, search, fusion in zip(run['work'], run['fine_search'], run['fine_fusion']):
            if (any(not count(work.get(k)) for k in WORK_COUNTS) or
                    any(not count(search.get(k)) for k in SEARCH_COUNTS) or
                    any(not count(fusion.get(k)) for k in FUSION_COUNTS)):
                return False
            for values, keys in ((work, WORK_CLOCKS), (search, CLOCKS), (fusion, FUSION_CLOCKS)):
                if any(not valid_clock(values.get(k), k.endswith('device_ms')) for k in keys):
                    return False
            if (work['fine_ncc_mode'] != 2 or work['fine_row_bytes'] or work['fine_row_dispatches'] or
                    any(work[k] for k in WORK_COUNTS if k.startswith('async_')) or
                    search['profiling_enabled'] != 0 or search['angle_reuse_enabled'] != 1 or
                    any(search[k] for k in OBSERVATION_COUNTS) or
                    any(search[k] for k in CLOCKS if k != 'warp_device_ms') or search['warp_device_ms'] != -1 or
                    search['logical_requests'] != work['fine_maps'] + search['reused_maps'] or
                    (search['reused_maps'] > 0 and search['cache_peak_bytes'] == 0) or
                    fusion['enabled'] != 1 or
                    fusion['fused_batches'] + fusion['fallback_batches'] != work['fine_stats_dispatches'] or
                    work['fine_commands'] != work['fine_stats_dispatches'] + fusion['fallback_batches'] or
                    work['fine_stats_device_ms'] != -1):
                return False
        if any(w['fine_maps'] for w in run['work']):
            return fused_route(run, reference, 'fused')
        # Empty or threshold-rejected searches can legitimately have no fine
        # dispatch. They still require complete zero work and sentinel fields.
        return all(not w['fine_stats_dispatches'] and not w['fine_commands'] and
                   f['fused_device_ms'] == -1 and not f['fused_batches'] and not f['fallback_batches']
                   for w, f in zip(run['work'], run['fine_fusion']))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def pose_capacities_valid(run, case, first, page_bytes):
    try:
        if run['pose_input_mode'] != 1 or run['pose_storage_mode'] != 1:
            return False
        if any(not count(run.get(k)) for k in POSE_BYTES):
            return False
        active = bool(run['proposals'])
        if not active:
            return all(run[k] == 0 for k in POSE_BYTES) and run['pose_create_ms'] == 0
        sizes = [4 * math.prod(case[k]) for k in ('source_shape', 'template_shape')]
        image = sum((size + page_bytes - 1) // page_bytes * page_bytes for size in sizes)
        if any(run[k] != image for k in IMAGE_BYTES) or run['pose_conversion_temporary_bytes'] != 0:
            return False
        if not finite_ms(run['pose_create_ms']) or (run['pose_create_ms'] > 0) is not first:
            return False
        host, gpu, shared = run['pose_host_image_bytes'], run['pose_gpu_image_bytes'], run['pose_shared_image_bytes']
        if run['pose_unique_image_bytes'] != host + gpu - shared:
            return False
        # Host solver state and retained device workspace can differ by solver.
        # The reported auxiliary bytes are a high-water mark across rescue calls.
        return (run['pose_host_bytes'] >= host and run['pose_gpu_bytes'] >= gpu and
                0 < run['pose_gpu_bytes'] - gpu <= run['refinement_auxiliary_bytes'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def route(run, reference, method, case, first, page_bytes):
    try:
        if (method not in METHODS or not numeric_tree(run) or 'resources' in run or
                not frontend_valid(run, reference, case) or
                not pose_capacities_valid(run, case, first, page_bytes) or
                run['refinement_mode'] != METHODS.index(method) or
                any(not count(run.get(k)) for k in REFINEMENT_COUNTS) or
                any(not valid_clock(run.get(k), k == 'refinement_device_ms') for k in REFINEMENT_CLOCKS)):
            return False
        active = bool(run['proposals'])
        if active is not case['expect_refine']:
            return False
        dispatch, commands, waits = (run[k] for k in REFINEMENT_COUNTS[:3])
        info = [run[k] for k in ('refinement_threads', 'refinement_simd_width', 'refinement_threadgroup_bytes')]
        if not active:
            return (run['selected_refinement_backend'] == 0 and
                    all(run[k] == 0 for k in REFINEMENT_COUNTS) and
                    all(run[k] == 0 for k in REFINEMENT_CLOCKS if k != 'refinement_device_ms') and
                    run['refinement_device_ms'] in (-1, 0))
        if run['selected_refinement_backend'] != 1 or not (dispatch >= commands == waits > 0):
            return False
        if method == 'resident0':
            return info == [0, 0, 0]
        return (dispatch == commands == waits and commands >= math.ceil(len(run['proposals']) / 512) and
                info[0] == (256 if method == 'fused256' else 128) and info[1] == 32 and
                0 < info[2] <= 32768 and info[2] % 16 == 0)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_cases(cases):
    if not isinstance(cases, list) or [c.get('name') for c in cases] != CASE_NAMES:
        raise ValueError('Expected the predeclared eight cases in order')
    for case in cases:
        if (case.get('validity') not in ('normal', 'challenge', 'negative') or
                case.get('origin') not in ('existing', 'new') or
                type(case.get('expect_refine')) is not bool or case.get('workflow') != 'full_rescue' or
                not isinstance(case.get('truth'), list)):
            raise ValueError('Invalid case metadata')
        if case['validity'] == 'negative' and (case['truth'] or case['expect_refine']):
            raise ValueError('Negative empty scene must have no truth/refinement')
        for key in ('source_shape', 'template_shape'):
            if (not isinstance(case.get(key), list) or len(case[key]) != 2 or
                    any(not count(v) or v == 0 for v in case[key])):
                raise ValueError('Invalid image dimensions')
        if math.prod(case['template_shape']) > 16384:
            raise ValueError('Fused resident supports templates <=16384 pixels')
        evaluate([], case['truth'])


def summarize(report, base):
    base = Path(base)
    protocol = checked(base / 'protocol.json', report['protocol_sha256'])
    dynamic = ('protocol_sha256', 'execution_order', 'complete', 'unchanged', 'post_sha256')
    if ({k: v for k, v in report.items() if k not in dynamic} !=
            {k: v for k, v in protocol.items() if k not in dynamic}):
        raise ValueError('Formal settings differ from predeclared protocol')
    cases = report['cases']
    validate_cases(cases)
    if report['methods'] != METHODS or report['blocks'] != 2 or report['repeats'] != 1 or report['warm_seconds'] != 0:
        raise ValueError('Unsupported experiment protocol')
    page = report['memory_page_bytes']
    if not count(page) or page < 4096 or int(page) & (int(page) - 1):
        raise ValueError('Invalid host page size')
    for group in ('source_sha256', 'binary_sha256', 'input_sha256'):
        if not report[group] or report['post_sha256'].get(group) != report[group]:
            raise ValueError('Source, binary or input hashes changed')
    if not report['unchanged']:
        raise ValueError('Source, binary or input state changed')
    prior = checked(base / 'baseline-protocol.json', report['baseline_protocol_sha256'])
    for name in BINARY_NAMES:
        expected = [v for p, v in prior['binary_sha256'].items() if p.endswith('/run-diagnostics/experiments/' + name)]
        actual = [v for p, v in report['binary_sha256'].items() if p.endswith('/experiments/' + name)]
        if len(expected) != 1 or actual != expected or report['frozen_binary_verified'].get(name) != expected[0]:
            raise ValueError('Frozen binary provenance mismatch')
    order = worker_order(cases)
    if not report['complete'] or report['expected_workers'] != len(order) or len(report['execution_order']) != len(order):
        raise ValueError('Incomplete experiment')
    groups = {}
    for entry, key in zip(report['execution_order'], order):
        if (entry['case'], entry['method'], entry['block']) != key:
            raise ValueError('Missing, duplicate or reordered worker')
        case = next(c for c in cases if c['name'] == entry['case'])
        if checked(base / entry['job'], entry['job_sha256']) != job_for(case, entry['method']):
            raise ValueError('Worker settings differ from protocol')
        if entry['returncode'] != 0 or sha(base / entry['stderr']) != entry['stderr_sha256']:
            raise ValueError('Worker failed or stderr hash differs')
        raw = checked(base / entry['result'], entry['result_sha256'])
        if len(raw['runs']) != 1 or raw['warmups_ms'] != []:
            raise ValueError('Expected first plus one repeat without warmup')
        if (not numeric_tree(raw) or not valid_memory(raw['final_memory']) or not finite_ms(raw['construct_ms']) or
                any(not finite_ms(r['total_ms']) or r['total_ms'] <= 0 or not valid_memory(r['memory']) for _, r in calls(raw))):
            raise ValueError('Invalid timing or memory')
        groups[key] = raw
    summaries, performance_failures, accuracy_failures = [], [], []
    for case in cases:
        name, methods = case['name'], {}
        base_first = groups[name, 'resident0', 0]['first']
        for method in METHODS:
            raws = [groups[name, method, b] for b in range(2)]
            records = [r for raw in raws for _, r in calls(raw)]
            paired = [final_comparison(r['matches'], base_first['matches']) for r in records]
            accuracies = [evaluate(r['matches'], case['truth']) for r in records]
            routed, blocks = True, []
            for block, raw in enumerate(raws):
                reference = groups[name, 'resident0', block]
                routed &= loader_valid(raw, reference)
                for index, ((label, run), (_, ref)) in enumerate(zip(calls(raw), calls(reference))):
                    routed &= route(run, ref, method, case, index == 0, page)
                    routed &= run['generation'] == index + 1
                blocks.append(dict(block=block, first_ms=raw['first']['total_ms'], repeat_ms=raw['runs'][0]['total_ms'],
                    rss_bytes=raw['final_memory']['peak_rss_bytes'],
                    calls=[dict(phase=label, stages=stage_clocks(run), memory=run['memory'],
                        capacities={k: v for k, v in run.items() if k.endswith('_bytes')},
                        refinement={k: run.get(k) for k in (*REFINEMENT_COUNTS, 'selected_refinement_backend',
                            'refinement_mode', 'pose_input_mode', 'pose_storage_mode')}) for label, run in calls(raw)]))
            strict = [a['strict'] for a in accuracies]
            accuracy = dict(truth_count=len(case['truth']), max_fp=max(a['fp'] for a in strict),
                max_fn=max(a['fn'] for a in strict),
                max_errors={k: max_present(a['max_errors'][k] for a in strict) for k in strict[0]['max_errors']},
                calls=[dict(block=i // 2, phase='first' if i % 2 == 0 else 'repeat1', **a) for i, a in enumerate(accuracies)])
            accuracy['passed'] = accuracy['max_fp'] == accuracy['max_fn'] == 0
            if not accuracy['passed']:
                accuracy_failures.append(dict(case=name, method=method, validity=case['validity'],
                    origin=case['origin'], max_fp=accuracy['max_fp'], max_fn=accuracy['max_fn']))
            # Own repeats are required to be exact, but cross-solver floating
            # point differences use the complete pairing criterion separately.
            own_exact = all(raw['exact_repeat'] is True and raw['runs'][0]['matches'] == raw['first']['matches'] and
                            raw['runs'][0]['proposals'] == raw['first']['proposals'] for raw in raws)
            methods[method] = dict(route=bool(routed), own_repeat_exact=own_exact,
                proposals_exact=all(r['proposals'] == base_first['proposals'] for r in records),
                frontend_exact=all(frontend_state(run) == frontend_state(ref)
                    for raw in raws for (_, run), (_, ref) in zip(calls(raw), calls(groups[name, 'resident0', 0]))),
                final_exact=all(r['matches'] == base_first['matches'] for r in records),
                final_paired=all(p['valid'] for p in paired),
                pair_max_errors={k: max_present(p['max_errors'][k] for p in paired) for k in PAIR_TOLERANCE},
                pair_comparisons=[dict(block=i // 2, phase='first' if i % 2 == 0 else 'repeat1', **p) for i, p in enumerate(paired)],
                first_ms=dist([raw['first']['total_ms'] for raw in raws]),
                repeat_ms=dist([raw['runs'][0]['total_ms'] for raw in raws]),
                construct_ms=[raw['construct_ms'] for raw in raws],
                rss_bytes=[raw['final_memory']['peak_rss_bytes'] for raw in raws], blocks=blocks, accuracy=accuracy)
        comparisons = {}
        for method, reference in [('fused256', 'resident0'), ('simd128', 'resident0'), ('simd128', 'fused256')]:
            key, pairs = method + '_vs_' + reference, []
            for block in range(2):
                pair = pair_metrics(groups[name, method, block], groups[name, reference, block], block)
                pairs.append(pair)
                failed = [p['phase'] for p in pair['calls'] if not p['time_tolerance_pass']]
                if failed or not pair['rss_tolerance_pass']:
                    performance_failures.append(dict(case=name, comparison=key, block=block,
                        failed_time_phases=failed, rss_failed=not pair['rss_tolerance_pass']))
            comparisons[key] = pairs
        summaries.append(dict(name=name, validity=case['validity'], origin=case['origin'],
                              methods=methods, comparisons=comparisons))
    methods = [m for c in summaries for m in c['methods'].values()]
    integrity = all(m['route'] and m['own_repeat_exact'] and m['proposals_exact'] and m['frontend_exact'] for m in methods)
    agreement = all(m['final_paired'] for m in methods)
    return dict(integrity_passed=integrity, paired_final_passed=agreement,
        accuracy_passed=not accuracy_failures,
        normal_accuracy_passed=not any(f['validity'] == 'normal' for f in accuracy_failures),
        correctness_passed=integrity and agreement and not accuracy_failures,
        all_performance_tolerances_passed=not performance_failures,
        accuracy_failures=accuracy_failures, performance_failures=performance_failures,
        unchanged=report['unchanged'], workers=len(order), algorithm_calls=2 * len(order),
        first_calls=len(order), repeat_calls=len(order), warmup_calls=0,
        timing_scope='Two independent processes per case/method; first and repeat nested in each. Root and child clocks overlap and must not be added. GPU wait is not a transfer-time measurement.',
        scope='Existing grayscale resident modes composed with the existing multiscale frontend. Shared synthetic source variants are not independent real-camera datasets. No default changed.',
        rescue_observation_limit='Native runner reports selected rescue seeds but not retry-start/continuation counts; no exact rescue dispatch-count attribution.',
        cases=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-json', type=Path, default=ROOT / 'build/resident-multiscale-inputs/cases.json')
    parser.add_argument('--build', type=Path, default=ROOT / 'build/run-diagnostics')
    parser.add_argument('--baseline-protocol', type=Path, default=ROOT / 'cpp-core/docs/run-diagnostics/reproduction-protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.summarize:
        if args.output.exists():
            raise ValueError('Refusing to overwrite replay output')
        summary = summarize(json.loads(args.summarize.read_text()), args.summarize.parent)
        save(args.output, summary)
        if not summary['correctness_passed']:
            raise SystemExit(1)
        return
    dest = args.output.resolve()
    ensure_empty_output(dest)
    cases = json.loads(args.cases_json.read_text())
    for case in cases:
        case.update(source_shape=image_shape(case['source']), template_shape=image_shape(case['template']))
    validate_cases(cases)
    verified = baseline(args.build, args.baseline_protocol)
    binaries = [args.build / 'experiments' / name for name in BINARY_NAMES]
    inputs = [Path(case[key]) for case in cases for key in ('source', 'template', 'metadata') if key in case]
    inputs.append(args.cases_json)
    report = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(), working_tree_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        frozen_binary_verified=verified, baseline_protocol_sha256=sha(args.baseline_protocol),
        methods=METHODS, cases=cases, blocks=2, repeats=1, warm_seconds=0., memory_page_bytes=os.sysconf('SC_PAGE_SIZE'),
        expected_workers=len(worker_order(cases)),
        rules=dict(order='Serial fresh processes: methods forward in block0, reversed in block1; no warmup',
            frontend='Ordered proposals, phase-matched nonclock work, fine/coarse frame capacities exact',
            final='Own repeats exact; mode0 versus both alternatives complete one-to-one pose/score assignment with independently evaluated truth',
            pair_tolerance=PAIR_TOLERANCE,
            accuracy='Keep all common FP/FN, including normal/new/challenge/negative cases; no early stop on accuracy failures',
            route='Metal backend; resident0 kernel info zero; fused256/simd128 dispatch=commands=waits>=ceil(proposals/512); empty scene zero execution; no all-zero positive case accepted',
            memory='u8/shared image pages individually rounded; unique=host+gpu-shared; no temporary. Auxiliary/solver workspace may differ. RSS and capacities overlap.',
            timing='Algorithm includes prepare, excludes decode/model construction/serialization. Root and child clocks overlap; wait/device not transfer measurements.',
            tolerance='Per paired block first/repeat time max(10%,2ms), RSS max(10%,16MiB); retain every failure',
            scope='Existing solver composition, grayscale templates <=16384 pixels; no new solver, automatic fallback or default change'),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files()},
        binary_sha256={str(p.resolve()): sha(p) for p in binaries},
        input_sha256={str(p.resolve()): sha(p) for p in inputs}, execution_order=[], complete=False)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.baseline_protocol, dest / 'baseline-protocol.json')
    save(dest / 'protocol.json', report)
    report['protocol_sha256'] = sha(dest / 'protocol.json')
    for name, method, block in worker_order(cases):
        case = next(c for c in cases if c['name'] == name)
        label = f'{name}-{method}-b{block}'
        job = dest / 'jobs' / (label + '.json')
        save(job, job_for(case, method))
        print('Running ' + label, flush=True)
        result = subprocess.run([str((args.build / 'experiments/fpm_native_pipeline_runner').resolve()), str(job)],
                                capture_output=True, text=True)
        rawpath, errpath = job.with_suffix('.result.json'), job.with_suffix('.stderr.txt')
        rawpath.write_text(result.stdout)
        errpath.write_text(result.stderr)
        report['execution_order'].append(dict(case=name, method=method, block=block,
            job=str(job.relative_to(dest)), job_sha256=sha(job), result=str(rawpath.relative_to(dest)), result_sha256=sha(rawpath),
            stderr=str(errpath.relative_to(dest)), stderr_sha256=sha(errpath), returncode=result.returncode))
        save(dest / 'formal.json', report)
        if result.returncode:
            raise RuntimeError(label + ' failed: ' + result.stderr[-3000:])
        json.loads(result.stdout)
    report['complete'] = True
    report['post_sha256'] = dict(source_sha256={k: sha(ROOT / k) for k in report['source_sha256']},
        **{group: {k: sha(k) for k in report[group]} for group in ('binary_sha256', 'input_sha256')})
    report['unchanged'] = all(report['post_sha256'][group] == report[group] for group in report['post_sha256'])
    save(dest / 'formal.json', report)
    summary = summarize(report, dest)
    save(dest / 'summary.json', summary)
    if not summary['correctness_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
