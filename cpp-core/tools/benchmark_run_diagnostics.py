"""Stage32: diagnose the 512-object timing outlier, then test pose initialization.

Each phase is an independent, predeclared serial fresh-process experiment. The
reproduction phase must be interpreted before separately starting pose_memory.
Probe arms measure resource-counter overhead and are not speedup evidence.
Replay uses only protocol and hashed jobs/results, including after relocation.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess

from benchmark_fine_fused import (numeric_tree, route as fused_route, source_files,
                                  sum_gpu_clock)
from benchmark_fine_strip import sha, save, dist, max_present, checked
from benchmark_gray_loader import valid_memory, ensure_empty_output, BINARY_NAMES
from benchmark_native_pipeline import native_job
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
FROZEN_COMMIT = 'c3c59ee05a957a94b5f4e528fbbba8d50308fc23'
PHASES = {
    'reproduction': dict(methods=['frozen_legacy', 'frozen_direct', 'current_legacy',
                                 'current_direct', 'probe_legacy', 'probe_direct'],
                         cases=['many512_pose'], repeats=3),
    'pose_memory': dict(methods=['current', 'u8', 'shared', 'combined'],
                        cases=['many512_pose', 'many512_pose_fullres_subpixel'], repeats=1),
}
POSE_IMAGE_BYTES = ('pose_host_image_bytes', 'pose_gpu_image_bytes',
                    'pose_shared_image_bytes', 'pose_unique_image_bytes')
POSE_VARIABLE = (*POSE_IMAGE_BYTES, 'pose_host_bytes', 'pose_gpu_bytes',
                 'pose_conversion_temporary_bytes', 'pose_storage_mode', 'pose_input_mode')
PROCESS_MS = ('process_user_ms', 'process_system_ms')
PROCESS_COUNTS = ('minor_faults', 'major_faults', 'voluntary_switches', 'involuntary_switches')
HOST_COUNTS = ('host_busy_ticks', 'host_idle_ticks')


def count(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value >= 0 and int(value) == value)


def finite_ms(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value >= 0)


def job_for(case, method, phase):
    settings = PHASES[phase]
    if method not in settings['methods']:
        raise ValueError('Unknown method for phase')
    job = native_job(case, 'native_tile', settings['repeats'], 0.)
    job.update(fine_ncc_mode=2, fine_profiling=False, fine_angle_reuse=True,
               fine_fused=True, pose_resident_mode=0)
    if phase == 'reproduction':
        job['gray_load_mode'] = 'direct_gray8' if method.endswith('_direct') else 'legacy_rgb'
        # Frozen stage31 does not implement this option; do not let an ignored
        # option falsely suggest that it was enabled there.
        if not method.startswith('frozen_'):
            job['run_resource_diagnostics'] = method.startswith('probe_')
    else:
        job.update(gray_load_mode='direct_gray8', run_resource_diagnostics=False,
                   pose_input_mode=int(method in ('u8', 'combined')),
                   pose_storage_mode=int(method in ('shared', 'combined')))
    return job


def baseline(build, protocol):
    recorded = json.loads(protocol.read_text())['binary_sha256']
    verified = {}
    for name in BINARY_NAMES:
        matches = [digest for path, digest in recorded.items()
                   if path.endswith('/gray-loader/experiments/' + name)]
        if len(matches) != 1 or sha(build / 'experiments' / name) != matches[0]:
            raise ValueError('Frozen stage31 binary differs: ' + name)
        verified[name] = matches[0]
    return verified


def png_shape(path):
    with Path(path).open('rb') as stream:
        head = stream.read(24)
    if len(head) != 24 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        raise ValueError('Expected benchmark PNG input: ' + str(path))
    width, height = struct.unpack('>II', head[16:24])
    if width == 0 or height == 0:
        raise ValueError('Invalid input dimensions')
    return [width, height]


def nonclock_state(run, pose_memory=False):
    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()
                    if not key.endswith('_ms')}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value
    excluded = ('matches', 'proposals', 'memory', 'resources')
    if pose_memory:
        excluded += POSE_VARIABLE
    return strip({key: value for key, value in run.items() if key not in excluded})


def resources_valid(run, enabled):
    if not enabled:
        return 'resources' not in run
    value = run.get('resources')
    if not isinstance(value, dict):
        return False
    if type(value.get('available')) is not bool or type(value.get('host_load_available')) is not bool:
        return False
    if not finite_ms(value.get('wall_ms')) or value['wall_ms'] <= 0:
        return False
    for names, available in ((PROCESS_MS, value['available']),
                             (PROCESS_COUNTS, value['available']),
                             (HOST_COUNTS, value['host_load_available'])):
        for key in names:
            valid = finite_ms(value.get(key)) if key in PROCESS_MS else count(value.get(key))
            if (not valid if available else value.get(key) != -1):
                return False
    return True


def loader_valid(raw, direct):
    try:
        load = raw['load']
        if (load['mode'] != ('direct_gray8' if direct else 'legacy_rgb') or
                not finite_ms(load['decode_ms']) or not valid_memory(raw['after_load_memory'])):
            return False
        return all(load[key]['direct_gray8'] is direct and
                   count(load[key]['rgb_temporary_bytes']) and
                   ((load[key]['rgb_temporary_bytes'] == 0) == direct)
                   for key in ('source', 'template'))
    except (KeyError, TypeError, ValueError):
        return False


def pose_capacities_valid(run, reference, method, case, first, page_bytes):
    """Verify image bytes, page rounding and unchanged non-image workspaces."""
    try:
        shared = method in ('shared', 'combined')
        direct = method in ('u8', 'combined')
        sizes = [4 * math.prod(case[key]) for key in ('source_shape', 'template_shape')]
        plain = sum(sizes)
        images = sum(((size + page_bytes - 1) // page_bytes) * page_bytes for size in sizes) if shared else plain
        expected = dict(pose_storage_mode=int(shared), pose_input_mode=int(direct),
                        pose_host_image_bytes=images, pose_gpu_image_bytes=images,
                        pose_shared_image_bytes=images if shared else 0,
                        pose_unique_image_bytes=images if shared else 2 * images,
                        pose_conversion_temporary_bytes=plain if first and not direct else 0)
        if any(not count(run.get(key)) or run[key] != value for key, value in expected.items()):
            return False
        if not finite_ms(run.get('pose_create_ms')) or (run['pose_create_ms'] > 0) is not first:
            return False
        if run.get('selected_refinement_backend') != 1 or run.get('refinement_mode') != 0:
            return False
        for key, image in (('pose_host_bytes', 'pose_host_image_bytes'),
                           ('pose_gpu_bytes', 'pose_gpu_image_bytes')):
            if (not count(run.get(key)) or run[key] < run[image] or
                    run[key] - run[image] != reference[key] - reference[image]):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def route(run, reference, method, phase, case, first, page_bytes):
    try:
        different_pose = phase == 'pose_memory'
        valid = (fused_route(run, reference, 'fused') and
                 nonclock_state(run, different_pose) == nonclock_state(reference, different_pose) and
                 all(count(value) for key, value in run.items() if key.endswith('_bytes')) and
                 run['selected_refinement_backend'] == 1 and run['refinement_mode'] == 0 and
                 resources_valid(run, phase == 'reproduction' and method.startswith('probe_')))
        if different_pose:
            valid &= pose_capacities_valid(run, reference, method, case, first, page_bytes)
        else:
            valid &= run['pose_input_mode'] == run['pose_storage_mode'] == 0
        return bool(valid)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def worker_order(cases, phase):
    methods = PHASES[phase]['methods']
    return [(case['name'], method, block) for case in cases for block in range(2)
            for method in (methods if block == 0 else methods[::-1])]


def calls(raw):
    return [('first', raw['first']), *[('repeat' + str(i + 1), run)
                                      for i, run in enumerate(raw['runs'])]]


def stage_clocks(run):
    """Keep nested clocks in separate families; never sum parents and children.

    Models within each family run sequentially. Any unavailable GPU timestamp
    makes that family aggregate unavailable instead of creating a negative sum.
    Raw per-model clocks are retained alongside these convenient aggregates.
    """
    result = dict(root={key: value for key, value in run.items() if key.endswith('_ms')})
    for family in ('work', 'fine_search', 'fine_fusion'):
        models = [{key: value for key, value in entry.items() if key.endswith('_ms')}
                  for entry in run[family]]
        keys = set.intersection(*(set(model) for model in models)) if models else set()
        result[family] = dict(models=models, sums={key: (sum_gpu_clock if key.endswith('device_ms') else sum)
                                                     (model[key] for model in models)
                                                  for key in sorted(keys)})
    return result


def pair_metrics(candidate, reference, block):
    result = dict(block=block, diagnostic_only=False,
                  rss_delta_bytes=candidate['final_memory']['peak_rss_bytes'] - reference['final_memory']['peak_rss_bytes'])
    a, b = reference['final_memory']['peak_rss_bytes'], candidate['final_memory']['peak_rss_bytes']
    result['rss_tolerance_pass'] = b <= a + max(.1 * a, 16 * 1024 ** 2)
    result['calls'] = []
    for (phase, new), (_, old) in zip(calls(candidate), calls(reference)):
        a, b = old['total_ms'], new['total_ms']
        result['calls'].append(dict(phase=phase, reference_ms=a, candidate_ms=b,
                                   time_delta_percent=100 * (b / a - 1),
                                   time_tolerance_pass=b <= a + max(.1 * a, 2.)))
    a = statistics.median(run['total_ms'] for run in reference['runs'])
    b = statistics.median(run['total_ms'] for run in candidate['runs'])
    result['repeat_median'] = dict(reference_ms=a, candidate_ms=b,
                                 time_delta_percent=100 * (b / a - 1),
                                 time_tolerance_pass=b <= a + max(.1 * a, 2.))
    result['decode_delta_ms'] = (candidate['load']['decode_ms'] - reference['load']['decode_ms']
                                 if candidate.get('load') and reference.get('load') else None)
    return result


def comparisons_for(phase):
    if phase == 'pose_memory':
        return [(method, 'current') for method in PHASES[phase]['methods'][1:]]
    return [(method, 'frozen_' + method.split('_')[1]) for method in PHASES[phase]['methods'][2:]] + [
        ('probe_legacy', 'current_legacy'), ('probe_direct', 'current_direct')]


def summarize(report, base):
    base = Path(base)
    protocol = checked(base / 'protocol.json', report['protocol_sha256'])
    dynamic = ('protocol_sha256', 'execution_order', 'complete', 'unchanged')
    if ({k: v for k, v in report.items() if k not in dynamic} !=
            {k: v for k, v in protocol.items() if k not in dynamic}):
        raise ValueError('Formal settings differ from predeclared protocol')
    phase = report['phase']
    if phase not in PHASES:
        raise ValueError('Unsupported experiment phase')
    settings = PHASES[phase]
    cases = report['cases']
    if ([case['name'] for case in cases] != settings['cases'] or
            report['methods'] != settings['methods'] or report['blocks'] != 2 or
            report['repeats'] != settings['repeats'] or report['warm_seconds'] != 0.):
        raise ValueError('Unsupported experiment protocol')
    page_bytes = report['memory_page_bytes']
    if not count(page_bytes) or page_bytes < 4096 or int(page_bytes) & (int(page_bytes) - 1):
        raise ValueError('Invalid host page size')
    for case in cases:
        if any(not isinstance(case.get(key), list) or len(case[key]) != 2 or
               any(not count(v) or v == 0 for v in case[key])
               for key in ('source_shape', 'template_shape')):
            raise ValueError('Missing/invalid input dimensions')
    order = worker_order(cases, phase)
    if (not report['complete'] or report['expected_workers'] != len(order) or
            len(report['execution_order']) != len(order)):
        raise ValueError('Incomplete experiment')
    groups = {}
    for entry, key in zip(report['execution_order'], order):
        if (entry['case'], entry['method'], entry['block']) != key:
            raise ValueError('Missing, duplicate or reordered worker')
        case = next(case for case in cases if case['name'] == entry['case'])
        if checked(base / entry['job'], entry['job_sha256']) != job_for(case, entry['method'], phase):
            raise ValueError('Worker settings differ from protocol')
        if entry['returncode'] != 0:
            raise ValueError('Worker failed')
        if sha(base / entry['stderr']) != entry['stderr_sha256']:
            raise ValueError('Worker stderr hash mismatch')
        raw = checked(base / entry['result'], entry['result_sha256'])
        if len(raw['runs']) != settings['repeats'] or raw['warmups_ms'] != []:
            raise ValueError('Expected exact first/repeat count and no warmup')
        if (not numeric_tree(raw) or not valid_memory(raw['final_memory']) or
                not finite_ms(raw['construct_ms']) or
                any(not finite_ms(run['total_ms']) or run['total_ms'] <= 0 or
                    not valid_memory(run['memory']) for _, run in calls(raw))):
            raise ValueError('Invalid timing or process memory')
        groups[key] = raw
    summaries, failures = [], []
    for case in cases:
        name, methods = case['name'], {}
        reference = groups[name, settings['methods'][0], 0]['first']
        for method in settings['methods']:
            raws = [groups[name, method, block] for block in range(2)]
            all_calls = [run for raw in raws for _, run in calls(raw)]
            exact = all(raw['exact_repeat'] is True for raw in raws) and all(
                run[key] == reference[key] for run in all_calls for key in ('matches', 'proposals'))
            routed = True
            blocks = []
            for block, raw in enumerate(raws):
                ref_method = ('frozen_' + method.split('_')[1]) if phase == 'reproduction' else 'current'
                ref = groups[name, ref_method, block]
                routed &= loader_valid(raw, phase == 'pose_memory' or method.endswith('_direct'))
                for index, ((label, run), (_, old)) in enumerate(zip(calls(raw), calls(ref))):
                    routed &= route(run, old, method, phase, case, index == 0, page_bytes)
                    routed &= run['generation'] == index + 1
                blocks.append(dict(block=block, first_ms=raw['first']['total_ms'],
                                   repeat_ms=dist([run['total_ms'] for run in raw['runs']]),
                                   rss_bytes=raw['final_memory']['peak_rss_bytes'],
                                   calls=[dict(phase=label, stages=stage_clocks(run),
                                               resources=run.get('resources'), memory=run['memory'],
                                               capacities={k: v for k, v in run.items() if k.endswith('_bytes')},
                                               pose_storage_mode=run['pose_storage_mode'], pose_input_mode=run['pose_input_mode'])
                                          for label, run in calls(raw)]))
            hits = sum(d['reused_maps'] for run in all_calls for d in run['fine_search'])
            routed &= (hits > 0) if case.get('expect_reuse', False) else (hits == 0)
            routed &= any(f['fused_batches'] > 0 for run in all_calls for f in run['fine_fusion'])
            accuracies = [evaluate(run['matches'], case['truth'])['strict'] for run in all_calls]
            methods[method] = dict(exact=exact, route=bool(routed), diagnostic_only=method.startswith('probe_'),
                first_ms=dist([raw['first']['total_ms'] for raw in raws]),
                repeat_ms=dist([run['total_ms'] for raw in raws for run in raw['runs']]),
                construct_ms=[raw['construct_ms'] for raw in raws],
                rss_bytes=[raw['final_memory']['peak_rss_bytes'] for raw in raws],
                blocks=blocks, load=[raw.get('load') for raw in raws],
                after_load_memory=[raw.get('after_load_memory') for raw in raws],
                accuracy=dict(truth_count=len(case['truth']), max_fp=max(a['fp'] for a in accuracies),
                              max_fn=max(a['fn'] for a in accuracies),
                              max_errors={k: max_present(a['max_errors'][k] for a in accuracies)
                                          for k in accuracies[0]['max_errors']}))
        comparisons = {}
        for method, ref in comparisons_for(phase):
            key = method + '_vs_' + ref
            pairs = []
            for block in range(2):
                pair = pair_metrics(groups[name, method, block], groups[name, ref, block], block)
                pair['diagnostic_only'] = method.startswith('probe_')
                pairs.append(pair)
                failed_calls = [p['phase'] for p in pair['calls'] if not p['time_tolerance_pass']]
                if not pair['repeat_median']['time_tolerance_pass']:
                    failed_calls.append('repeat_median')
                if failed_calls or not pair['rss_tolerance_pass']:
                    failures.append(dict(case=name, comparison=key, block=block,
                                         diagnostic_only=pair['diagnostic_only'], failed_time_phases=failed_calls,
                                         rss_failed=not pair['rss_tolerance_pass']))
            comparisons[key] = pairs
        summaries.append(dict(name=name, methods=methods, comparisons=comparisons))
    correct = all(method['exact'] and method['route'] and
                  method['accuracy']['max_fp'] == method['accuracy']['max_fn'] == 0
                  for case in summaries for method in case['methods'].values())
    return dict(phase=phase, correctness_passed=correct,
                all_performance_tolerances_passed=not failures,
                uninstrumented_performance_tolerances_passed=not any(not f['diagnostic_only'] for f in failures),
                performance_failures=failures, unchanged=report['unchanged'],
                workers=len(order), algorithm_calls=(1 + settings['repeats']) * len(order),
                first_calls=len(order), repeat_calls=settings['repeats'] * len(order), warmup_calls=0,
                timing_scope='First and repeats are nested within two independent processes per method/case; zero warmup. Probe arms are diagnostic only. Root and child stage clocks overlap and must not be added.',
                cases=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/run-diagnostics')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'build/gray-loader')
    parser.add_argument('--previous', type=Path, default=ROOT / 'cpp-core/docs/gray-loader/protocol.json')
    parser.add_argument('--baseline-protocol', type=Path, default=ROOT / 'cpp-core/docs/gray-loader/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.summarize:
        if args.output.exists():
            raise ValueError('Refusing to overwrite existing replay output')
        report = json.loads(args.summarize.read_text())
        if args.phase is not None and args.phase != report['phase']:
            raise ValueError('Requested phase differs from replay')
        summary = summarize(report, args.summarize.parent)
        save(args.output, summary)
        if not summary['correctness_passed'] or not summary['unchanged']:
            raise SystemExit(1)
        return
    if args.phase is None:
        parser.error('--phase is required when starting measurements')
    dest = args.output.resolve()
    ensure_empty_output(dest)
    settings = PHASES[args.phase]
    previous = json.loads(args.previous.read_text())['cases']
    if len({case['name'] for case in previous}) != len(previous):
        raise ValueError('Duplicate previous cases')
    cases = []
    for name in settings['cases']:
        case = dict(next(case for case in previous if case['name'] == name), benchmark_repeats=settings['repeats'])
        case.update(source_shape=png_shape(case['source']), template_shape=png_shape(case['template']))
        cases.append(case)
    verified = baseline(args.baseline, args.baseline_protocol)
    builds = (args.build, args.baseline) if args.phase == 'reproduction' else (args.build,)
    binaries = [build / 'experiments' / name for build in builds for name in BINARY_NAMES]
    report = dict(version=1, phase=args.phase, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(), baseline_commit=FROZEN_COMMIT,
        working_tree_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        frozen_binary_verified=verified, baseline_protocol_sha256=sha(args.baseline_protocol),
        methods=settings['methods'], cases=cases, blocks=2, repeats=settings['repeats'], warm_seconds=0.,
        memory_page_bytes=os.sysconf('SC_PAGE_SIZE'), expected_workers=len(worker_order(cases, args.phase)),
        rules=dict(order='Two forward/reverse blocks; serial fresh processes; reproduction and pose_memory are separately launched phases',
            exact='Ordered proposals/matches and phase-corresponding non-clock state; resources excluded. Pose-memory changes only input/storage metadata and capacity formulas; other workspace/work is exact.',
            timing='No warmup; first and repeats nested per worker, not independent samples. Algorithm includes prepare, excludes load/model construction/serialization. Nested clocks and GPU wait/device overlap; never sum them.',
            probes='Only probe arms enable process/host diagnostics; compare overhead with same-loader uninstrumented current arms; not speedup evidence',
            memory='Whole-process peak RSS is separate from image capacities; shared unique=host+gpu-shared; shared source/template float pages individually rounded; uint8 input avoids only first-call float temporary',
            tolerance='Per block/phase time max(10%,2ms), repeat median also reported; RSS max(10%,16MiB); retain failed pairs',
            scope='Two settings reuse the same synthetic 512-object scene; no new independent camera evidence; no solver/kernel/default change'),
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_files()},
        binary_sha256={str(path.resolve()): sha(path) for path in binaries},
        input_sha256={case[key]: sha(case[key]) for case in cases for key in ('source', 'template')},
        previous_sha256=sha(args.previous), execution_order=[], complete=False)
    save(dest / 'protocol.json', report)
    report['protocol_sha256'] = sha(dest / 'protocol.json')
    for name, method, block in worker_order(cases, args.phase):
        case = next(case for case in cases if case['name'] == name)
        label = f'{name}-{method}-b{block}'
        jobpath = dest / 'jobs' / (label + '.json')
        save(jobpath, job_for(case, method, args.phase))
        print('Running ' + label, flush=True)
        build = args.baseline if method.startswith('frozen_') else args.build
        result = subprocess.run([str((build / 'experiments/fpm_native_pipeline_runner').resolve()), str(jobpath)],
                                capture_output=True, text=True)
        rawpath = jobpath.with_suffix('.result.json')
        rawpath.write_text(result.stdout)
        errpath = jobpath.with_suffix('.stderr.txt')
        errpath.write_text(result.stderr)
        report['execution_order'].append(dict(case=name, method=method, block=block,
            job=str(jobpath.relative_to(dest)), job_sha256=sha(jobpath),
            result=str(rawpath.relative_to(dest)), result_sha256=sha(rawpath),
            stderr=str(errpath.relative_to(dest)), stderr_sha256=sha(errpath), returncode=result.returncode))
        save(dest / 'formal.json', report)
        if result.returncode:
            raise RuntimeError(label + ' failed: ' + result.stderr[-3000:])
        json.loads(result.stdout)
    report['complete'] = True
    report['unchanged'] = (all(sha(ROOT / key) == digest for key, digest in report['source_sha256'].items()) and
        all(sha(key) == digest for group in ('binary_sha256', 'input_sha256') for key, digest in report[group].items()))
    save(dest / 'formal.json', report)
    summary = summarize(report, dest)
    save(dest / 'summary.json', summary)
    if not summary['correctness_passed'] or not report['unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
