"""Stage30 exact warp/NCC command fusion, serial fresh-process comparison.

The predeclared protocol freezes source/input/binary hashes before work starts.
Ground truth stays in this parent; replay needs only the movable hashed jobs and
raw results. Performance failures remain evidence, independent of correctness.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

from benchmark_fine_strip import sha, save, dist, max_present, checked
from benchmark_fine_search import CLOCKS
from benchmark_native_pipeline import native_job
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
FROZEN_COMMIT = '66d362e8a45f829a915cd09e697e598cc22280bc'
METHODS = ['frozen', 'current', 'fused']
COUNTERS = ('maps', 'windows', 'fine_maps', 'fine_stats_dispatches', 'fine_commands',
            'fine_uploads', 'fine_upload_bytes', 'fine_roi_pixels',
            'fine_repair_pixels', 'fine_peak_batch')
# The physical repair counter remains flagged visits in both implementations;
# changed_pixels separately counts provisional samples needing correction.
SAME_WORK = tuple(k for k in COUNTERS if k != 'fine_commands')
CAPACITIES = ('frame_host_bytes', 'frame_coarse_gpu_bytes', 'frame_fine_gpu_bytes')
FUSION_COUNTS = ('enabled', 'fused_batches', 'fallback_batches', 'dense_maps',
                 'changed_pixels', 'corrected_windows')
FUSION_CLOCKS = ('correction_ms', 'fused_wait_ms', 'fused_device_ms')
REPEATS = 3
WARM_SECONDS = 1.


def job_for(case, method):
    if method not in METHODS:
        raise ValueError('Unknown method')
    job = native_job(case, 'native_tile', REPEATS, WARM_SECONDS)
    job.update(fine_ncc_mode=2, fine_profiling=False, fine_angle_reuse=True)
    if method != 'frozen':
        job['fine_fused'] = method == 'fused'
    return job


def baseline(build, protocol):
    recorded = json.loads(protocol.read_text())['binary_sha256']
    verified = {}
    for name in ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib',
                 'libfpm_pose_experiment.dylib'):
        candidates = [v for k, v in recorded.items()
                      if k.endswith('/fine-search/experiments/' + name)]
        if len(candidates) != 1 or sha(build / 'experiments' / name) != candidates[0]:
            raise ValueError('Frozen stage29 binary differs: ' + name)
        verified[name] = candidates[0]
    return verified


def numeric_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(numeric_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(numeric_tree(v) for v in value)
    return True


def count(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 and int(value) == value


def sum_gpu_clock(values):
    """A combined per-run GPU clock is unavailable if any model lacks it."""
    values = list(values)
    return -1 if any(value < 0 for value in values) else sum(values)


def route(run, ref, method):
    """Check physical command routing separately from ordered output equality."""
    try:
        if method not in METHODS or not numeric_tree(run):
            return False
        if run['total_ms'] <= 0 or any(run[k] < 0 for k in CAPACITIES):
            return False
        work, expected = run['work'], ref['work']
        if len(work) != len(expected) or run['models_matched'] != ref['models_matched']:
            return False
        diagnostics = run['fine_search']
        if len(diagnostics) != len(work) or not any(w['fine_maps'] for w in work):
            return False
        if method != 'frozen' and len(run['fine_fusion']) != len(work):
            return False
        if method != 'fused' and any(run[k] != ref[k] for k in CAPACITIES):
            return False
        if method == 'fused' and any(run[k] != ref[k] for k in CAPACITIES[:2]):
            return False
        for index, (w, r, d, rd) in enumerate(zip(work, expected, diagnostics, ref['fine_search'])):
            if any(not count(w[k]) for k in COUNTERS):
                return False
            if any(w[k] != r[k] for k in SAME_WORK):
                return False
            if w['fine_ncc_mode'] != 2 or w['fine_row_bytes'] or w['fine_row_dispatches']:
                return False
            if d['profiling_enabled'] != 0 or d['angle_reuse_enabled'] != 1:
                return False
            if any(not count(v) for k, v in d.items() if k not in CLOCKS):
                return False
            if any(d[k] for k in CLOCKS if k != 'warp_device_ms') or d['warp_device_ms'] != -1:
                return False
            # Observation profiling is disabled in every formal arm.
            if any(d[k] for k in ('calls', 'requests', 'within_call_duplicates', 'cross_call_duplicates',
                                 'tracked_unique_keys', 'untracked_call_keys', 'history_limit_keys',
                                 'history_key_payload_bytes', 'history_bucket_count',
                                 'peak_call_key_capacity_bytes', 'warp_batches', 'zero_repair_batches',
                                 'guard_words', 'nonzero_guard_words')):
                return False
            if d['logical_requests'] != w['fine_maps'] + d['reused_maps']:
                return False
            if any(d[k] != rd[k] for k in ('logical_requests', 'reused_maps', 'cache_peak_bytes')):
                return False
            if d['reused_maps'] and not d['cache_peak_bytes']:
                return False
            if method != 'fused':
                if w['fine_commands'] != 2 * w['fine_stats_dispatches'] or w['fine_stats_device_ms'] < 0:
                    return False
                if w['fine_repair_pixels'] != r['fine_repair_pixels']:
                    return False
            if method == 'frozen':
                continue
            fusion = run['fine_fusion'][index]
            if any(not count(fusion[k]) for k in FUSION_COUNTS):
                return False
            if fusion['correction_ms'] < 0 or fusion['fused_wait_ms'] < 0:
                return False
            if fusion['enabled'] != int(method == 'fused'):
                return False
            if method == 'current':
                if any(fusion[k] for k in (*FUSION_COUNTS, 'correction_ms', 'fused_wait_ms')):
                    return False
                if fusion['fused_device_ms'] != -1:
                    return False
                continue
            if fusion['fused_batches'] + fusion['fallback_batches'] != w['fine_stats_dispatches']:
                return False
            if w['fine_commands'] != w['fine_stats_dispatches'] + fusion['fallback_batches']:
                return False
            if w['fine_stats_device_ms'] != -1 or fusion['dense_maps'] > w['fine_maps']:
                return False
            if fusion['changed_pixels'] > w['fine_repair_pixels']:
                return False
            if fusion['corrected_windows'] > 49 * fusion['changed_pixels']:
                return False
            if fusion['fused_wait_ms'] > w['fine_wait_ms'] + 1e-7:
                return False
            if not fusion['fallback_batches'] and abs(fusion['fused_wait_ms'] - w['fine_wait_ms']) > 1e-7:
                return False
            if fusion['dense_maps'] and not fusion['changed_pixels']:
                return False
            if fusion['fused_batches']:
                if fusion['fused_device_ms'] <= 0:
                    return False
            elif fusion['fused_device_ms'] != -1:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def summarize(report, base):
    protocol = checked(base / 'protocol.json', report['protocol_sha256'])
    excluded = ('protocol_sha256', 'execution_order', 'complete', 'unchanged')
    if {k: v for k, v in report.items() if k not in excluded} != {k: v for k, v in protocol.items() if k not in excluded}:
        raise ValueError('Formal settings differ from predeclared protocol')
    if not report['cases'] or len({c['name'] for c in report['cases']}) != len(report['cases']):
        raise ValueError('Duplicate case names')
    if report['methods'] != METHODS or report['blocks'] != 2 or report['repeats'] != REPEATS or report['warm_seconds'] != WARM_SECONDS:
        raise ValueError('Unsupported experiment protocol')
    order = [(c['name'], m, b) for c in report['cases'] for b in range(2)
             for m in (METHODS if b == 0 else METHODS[::-1])]
    if not report['complete'] or report['expected_workers'] != len(order) or len(report['execution_order']) != len(order):
        raise ValueError('Incomplete experiment')
    groups = {}
    for entry, key in zip(report['execution_order'], order):
        if (entry['case'], entry['method'], entry['block']) != key:
            raise ValueError('Missing, duplicate or reordered worker')
        case = next(c for c in report['cases'] if c['name'] == entry['case'])
        if checked(base / entry['job'], entry['job_sha256']) != job_for(case, entry['method']):
            raise ValueError('Worker settings differ from protocol')
        if entry['returncode'] != 0:
            raise ValueError('Worker failed')
        raw = checked(base / entry['result'], entry['result_sha256'])
        if len(raw['runs']) != REPEATS:
            raise ValueError('Incorrect sample count')
        if (not numeric_tree(raw) or not count(raw['final_memory']['peak_rss_bytes']) or
                raw['final_memory']['peak_rss_bytes'] == 0 or raw['construct_ms'] < 0):
            raise ValueError('Invalid timing or process memory')
        groups[key] = raw
    cases = []
    for case in report['cases']:
        name, methods = case['name'], {}
        reference = groups[name, 'frozen', 0]['first']
        for method in METHODS:
            raws = [groups[name, method, block] for block in range(2)]
            calls = [run for raw in raws for run in [raw['first'], *raw['runs']]]
            runs = [run for raw in raws for run in raw['runs']]
            exact = all(raw['exact_repeat'] for raw in raws) and all(
                run[key] == reference[key] for run in calls for key in ('matches', 'proposals'))
            routed = all(route(run, ref, method)
                         for block, raw in enumerate(raws)
                         for run, ref in zip([raw['first'], *raw['runs']],
                                             [groups[name, 'frozen', block]['first'], *groups[name, 'frozen', block]['runs']]))
            hits = sum(d['reused_maps'] for run in runs for d in run['fine_search'])
            routed &= (hits > 0) if case.get('expect_reuse', False) else (hits == 0)
            if method == 'fused':
                routed &= any(d['fused_batches'] > 0 for run in runs for d in run['fine_fusion'])
            accuracy = [evaluate(run['matches'], case['truth'])['strict'] for run in calls]
            data = dict(exact=exact, route=bool(routed), time_ms=dist([r['total_ms'] for r in runs]),
                        first_ms=[raw['first']['total_ms'] for raw in raws],
                        setup_ms=[raw['construct_ms'] for raw in raws],
                        blocks=[dict(block=b, median_ms=statistics.median(r['total_ms'] for r in raw['runs'])) for b, raw in enumerate(raws)],
                        rss_bytes=[raw['final_memory']['peak_rss_bytes'] for raw in raws],
                        capacities=[{k: raw['runs'][-1][k] for k in CAPACITIES} for raw in raws],
                        accuracy=dict(truth_count=len(case['truth']), max_fp=max(a['fp'] for a in accuracy),
                                      max_fn=max(a['fn'] for a in accuracy),
                                      max_errors={k: max_present(a['max_errors'][k] for a in accuracy) for k in accuracy[0]['max_errors']}),
                        stages={k: dist([(sum_gpu_clock if k == 'fine_stats_device_ms' else sum)(w[k] for w in r['work']) for r in runs])
                                for k in ('fine_stats_device_ms', 'fine_rotation_ms', 'fine_repair_ms',
                                          'fine_ncc_ms', 'fine_normalization_ms', 'fine_wait_ms', *COUNTERS[2:])},
                        reuse={k: dist([sum(d[k] for d in r['fine_search']) for r in runs])
                               for k in ('logical_requests', 'reused_maps', 'cache_peak_bytes')},
                        max_model_cache_bytes=max(d['cache_peak_bytes'] for r in runs for d in r['fine_search']))
            if method != 'frozen':
                data['fusion'] = {k: dist([(sum_gpu_clock if k == 'fused_device_ms' else sum)(d[k] for d in r['fine_fusion']) for r in runs])
                                  for k in (*FUSION_COUNTS[1:], *FUSION_CLOCKS)}
            methods[method] = data
        comparisons = {}
        for method, ref in [('current', 'frozen'), ('fused', 'current'), ('fused', 'frozen')]:
            a, b = methods[ref], methods[method]
            comparisons[method + '_vs_' + ref] = [dict(
                block=i, time_delta_percent=100 * (b['blocks'][i]['median_ms'] / a['blocks'][i]['median_ms'] - 1),
                time_tolerance_pass=b['blocks'][i]['median_ms'] <= a['blocks'][i]['median_ms'] + max(.1 * a['blocks'][i]['median_ms'], 2.),
                rss_delta_bytes=b['rss_bytes'][i] - a['rss_bytes'][i],
                rss_tolerance_pass=b['rss_bytes'][i] <= a['rss_bytes'][i] + max(.1 * a['rss_bytes'][i], 16 * 1024 ** 2))
                for i in range(2)]
        cases.append(dict(name=name, methods=methods, comparisons=comparisons))
    correct = all(m['exact'] and m['route'] and m['accuracy']['max_fp'] == m['accuracy']['max_fn'] == 0
                  for case in cases for m in case['methods'].values())
    performance = all(pair['time_tolerance_pass'] and pair['rss_tolerance_pass']
                      for case in cases for pairs in case['comparisons'].values() for pair in pairs)
    return dict(correctness_passed=correct, all_performance_tolerances_passed=performance,
                unchanged=report['unchanged'], workers=len(order), formal_calls=len(order) * REPEATS, cases=cases)


def source_files():
    suffixes = ('.c', '.cpp', '.h', '.hpp', '.mm', '.metal', '.py', '.txt', '.in', '.json',
                '.swift', '.m', '.plist', '.storyboard', '.xib', '.xcconfig', '.pbxproj')
    sources = []
    for folder in ('src', 'include', 'experiments', 'tests', 'tools', 'examples'):
        sources += [f for f in (ROOT / 'cpp-core' / folder).rglob('*') if f.is_file() and f.suffix in suffixes]
    sources.append(ROOT / 'cpp-core/CMakeLists.txt')
    # Source-only tracked GUI inputs avoid hashing build products or user files.
    tracked = subprocess.check_output(['git', 'ls-files', 'macos-cli', 'MatchTool', 'python-port/src'], cwd=ROOT, text=True)
    sources += [ROOT / name for name in tracked.splitlines() if Path(name).suffix in suffixes]
    return sorted(set(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/fine-fused')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'build/fine-search')
    parser.add_argument('--previous', type=Path, default=ROOT / 'cpp-core/docs/fine-angle-reuse/protocol.json')
    parser.add_argument('--baseline-protocol', type=Path, default=ROOT / 'cpp-core/docs/fine-angle-reuse/protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/fine-fused-results')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--case-names', help='Optional comma-separated pilot subset; recorded before measurements')
    args = parser.parse_args()
    if args.summarize:
        summary = summarize(json.loads(args.summarize.read_text()), args.summarize.parent)
        save(args.output, summary)
        if not summary['correctness_passed'] or not summary['unchanged']:
            raise SystemExit(1)
        return
    dest = args.output.resolve()
    if (dest / 'protocol.json').exists():
        raise ValueError('Refusing to overwrite existing experiment')
    cases = json.loads(args.previous.read_text())['cases']
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case names')
    if args.case_names:
        selected = args.case_names.split(',')
        if len(set(selected)) != len(selected) or set(selected) - {c['name'] for c in cases}:
            raise ValueError('Unknown or duplicate requested case')
        cases = [c for c in cases if c['name'] in selected]
    cases = [dict(c, benchmark_repeats=REPEATS) for c in cases]
    if not cases:
        raise ValueError('Empty experiment')
    verified = baseline(args.baseline, args.baseline_protocol)
    binaries = [b / 'experiments' / name for b in (args.build, args.baseline)
                for name in ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib', 'libfpm_pose_experiment.dylib')]
    report = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(),
                  baseline_commit=FROZEN_COMMIT,
                  working_tree_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  frozen_binary_verified=verified, baseline_protocol_sha256=sha(args.baseline_protocol),
                  methods=METHODS, cases=cases, blocks=2, repeats=REPEATS, warm_seconds=WARM_SECONDS,
                  expected_workers=len(METHODS) * 2 * len(cases),
                  rules=dict(order='frozen/current/fused then reverse, serial fresh processes',
                             exact='Ordered full proposals/matches and identical logical/physical fine-map workload; exact statistics correction',
                             timing='Warm total includes frame preparation; excludes decode/model learning/serialization/GT. Host wait includes queue and execution; not transfer time.',
                             memory='Process high-water RSS is separate from explicit buffer capacities; no allocator/compression cause inferred from RSS alone.',
                             tolerance='time max(10%,2ms); RSS max(10%,16MiB); every failed pair retained',
                             decision='opt-in only; no automatic default change',
                             data='Nine settings of six reused scenes, not nine independent camera datasets'),
                  source_sha256={str(f.relative_to(ROOT)): sha(f) for f in source_files()},
                  binary_sha256={str(f.resolve()): sha(f) for f in binaries},
                  input_sha256={c[k]: sha(c[k]) for c in cases for k in ('source', 'template')},
                  previous_sha256=sha(args.previous), execution_order=[], complete=False)
    save(dest / 'protocol.json', report)
    report['protocol_sha256'] = sha(dest / 'protocol.json')
    for case in cases:
        for block in range(2):
            for method in (METHODS if block == 0 else METHODS[::-1]):
                label = f'{case["name"]}-{method}-b{block}'
                jobpath = dest / 'jobs' / (label + '.json')
                save(jobpath, job_for(case, method))
                print('Running ' + label, flush=True)
                build = args.baseline if method == 'frozen' else args.build
                result = subprocess.run([str((build / 'experiments/fpm_native_pipeline_runner').resolve()), str(jobpath)], capture_output=True, text=True)
                rawpath = jobpath.with_suffix('.result.json')
                rawpath.write_text(result.stdout)
                jobpath.with_suffix('.stderr.txt').write_text(result.stderr)
                report['execution_order'].append(dict(case=case['name'], method=method, block=block,
                    job=str(jobpath.relative_to(dest)), job_sha256=sha(jobpath),
                    result=str(rawpath.relative_to(dest)), result_sha256=sha(rawpath), returncode=result.returncode))
                save(dest / 'formal.json', report)
                if result.returncode:
                    raise RuntimeError(label + ' failed: ' + result.stderr[-3000:])
                json.loads(result.stdout)
    report['complete'] = True
    report['unchanged'] = (all(sha(ROOT / k) == v for k, v in report['source_sha256'].items()) and
                           all(sha(k) == v for group in ('binary_sha256', 'input_sha256') for k, v in report[group].items()))
    save(dest / 'formal.json', report)
    summary = summarize(report, dest)
    save(dest / 'summary.json', summary)
    if not summary['correctness_passed'] or not report['unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
