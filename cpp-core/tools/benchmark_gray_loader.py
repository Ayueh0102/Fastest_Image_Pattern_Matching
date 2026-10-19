"""Stage31 grayscale loading: serial fresh-process first/repeat comparison.

All arms use stage30 fused fine search. No warmup is performed: decode, model
construction, the first algorithm call and one repeat are reported separately.
Ground truth stays in this parent. Hash-checked jobs/raw results support replay
after moving the result directory; performance failures remain in the summary.
"""
import argparse
import datetime
import json
from pathlib import Path
import platform
import subprocess

from benchmark_fine_fused import (numeric_tree, count, route as fused_route,
                                  source_files)
from benchmark_fine_strip import sha, save, dist, max_present, checked
from benchmark_native_pipeline import native_job
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
FROZEN_COMMIT = 'e392a018fd4f857b8c8c09737ef71173fff31a48'
METHODS = ['frozen', 'current_legacy', 'current_direct']
CASE_NAMES = ['small_fullres', 'many512_pose', 'many512_pose_fullres_subpixel']
BINARY_NAMES = ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib',
                'libfpm_pose_experiment.dylib')
REPEATS = 1
WARM_SECONDS = 0.


def job_for(case, method):
    if method not in METHODS:
        raise ValueError('Unknown method')
    job = native_job(case, 'native_tile', REPEATS, WARM_SECONDS)
    job.update(fine_ncc_mode=2, fine_profiling=False, fine_angle_reuse=True,
               fine_fused=True)
    if method != 'frozen':
        job['gray_load_mode'] = ('direct_gray8' if method == 'current_direct'
                                 else 'legacy_rgb')
    return job


def baseline(build, protocol):
    """Verify the frozen stage30 artifacts, rather than trusting their name."""
    recorded = json.loads(protocol.read_text())['binary_sha256']
    verified = {}
    for name in BINARY_NAMES:
        matches = [digest for path, digest in recorded.items()
                   if path.endswith('/fine-fused/experiments/' + name)]
        if len(matches) != 1 or sha(build / 'experiments' / name) != matches[0]:
            raise ValueError('Frozen stage30 binary differs: ' + name)
        verified[name] = matches[0]
    return verified


def nonclock_state(run):
    """Include every work/pose/capacity field, comparing first/repeat separately."""
    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()
                    if not key.endswith('_ms')}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value
    return strip({key: value for key, value in run.items()
                  if key not in ('matches', 'proposals', 'memory')})


def route(run, reference):
    try:
        return (fused_route(run, reference, 'fused') and
                nonclock_state(run) == nonclock_state(reference) and
                all(count(value) for key, value in run.items()
                    if key.endswith('_bytes')))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def valid_memory(memory):
    return (isinstance(memory, dict) and
            all(count(memory.get(key)) and memory[key] > 0
                for key in ('current_rss_bytes', 'peak_rss_bytes')))


def loader_route(raw, method):
    if method == 'frozen':
        # Stage30 has neither field. Do not manufacture measurements for it.
        return 'load' not in raw and 'after_load_memory' not in raw
    try:
        load = raw['load']
        direct = method == 'current_direct'
        if (load['mode'] != ('direct_gray8' if direct else 'legacy_rgb') or
                load['decode_ms'] < 0 or not valid_memory(raw['after_load_memory'])):
            return False
        # These three benchmark settings use Grayscale8 PNG inputs. General
        # RGB/fallback behavior is covered by the loader's separate unit tests.
        for name in ('source', 'template'):
            info = load[name]
            if info['direct_gray8'] is not direct or not count(info['rgb_temporary_bytes']):
                return False
            if (info['rgb_temporary_bytes'] == 0) != direct:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def worker_order(cases):
    return [(case['name'], method, block) for case in cases for block in range(2)
            for method in (METHODS if block == 0 else METHODS[::-1])]


def summarize(report, base):
    base = Path(base)
    protocol = checked(base / 'protocol.json', report['protocol_sha256'])
    dynamic = ('protocol_sha256', 'execution_order', 'complete', 'unchanged')
    if ({k: v for k, v in report.items() if k not in dynamic} !=
            {k: v for k, v in protocol.items() if k not in dynamic}):
        raise ValueError('Formal settings differ from predeclared protocol')
    cases = report['cases']
    if not cases or len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate case names')
    if (report['methods'] != METHODS or report['blocks'] != 2 or
            report['repeats'] != REPEATS or report['warm_seconds'] != WARM_SECONDS):
        raise ValueError('Unsupported experiment protocol')
    order = worker_order(cases)
    if (not report['complete'] or report['expected_workers'] != len(order) or
            len(report['execution_order']) != len(order)):
        raise ValueError('Incomplete experiment')
    groups = {}
    for entry, key in zip(report['execution_order'], order):
        if (entry['case'], entry['method'], entry['block']) != key:
            raise ValueError('Missing, duplicate or reordered worker')
        case = next(c for c in cases if c['name'] == entry['case'])
        if checked(base / entry['job'], entry['job_sha256']) != job_for(case, entry['method']):
            raise ValueError('Worker settings differ from protocol')
        if entry['returncode'] != 0:
            raise ValueError('Worker failed')
        raw = checked(base / entry['result'], entry['result_sha256'])
        if len(raw['runs']) != REPEATS or raw['warmups_ms'] != []:
            raise ValueError('Expected exactly first plus one repeat, with no warmup')
        if (not numeric_tree(raw) or not valid_memory(raw['final_memory']) or
                raw['construct_ms'] < 0 or
                any(run['total_ms'] <= 0 or not valid_memory(run['memory'])
                    for run in [raw['first'], *raw['runs']])):
            raise ValueError('Invalid timing or process memory')
        groups[key] = raw
    summaries = []
    for case in cases:
        name, methods = case['name'], {}
        reference = groups[name, 'frozen', 0]['first']
        for method in METHODS:
            raws = [groups[name, method, block] for block in range(2)]
            calls = [run for raw in raws for run in [raw['first'], *raw['runs']]]
            exact = all(raw['exact_repeat'] is True for raw in raws) and all(
                run[key] == reference[key] for run in calls for key in ('matches', 'proposals'))
            routed = all(route(run, ref)
                         for block, raw in enumerate(raws)
                         for run, ref in zip([raw['first'], *raw['runs']],
                                             [groups[name, 'frozen', block]['first'],
                                              *groups[name, 'frozen', block]['runs']]))
            routed &= all(loader_route(raw, method) for raw in raws)
            hits = sum(d['reused_maps'] for run in calls for d in run['fine_search'])
            routed &= (hits > 0) if case.get('expect_reuse', False) else (hits == 0)
            routed &= any(f['fused_batches'] > 0 for run in calls for f in run['fine_fusion'])
            accuracies = [evaluate(run['matches'], case['truth'])['strict'] for run in calls]
            capacities = [dict(block=block, **{
                phase: {k: v for k, v in run.items() if k.endswith('_bytes')}
                for phase, run in [('first', raw['first']), ('repeat', raw['runs'][0])]})
                for block, raw in enumerate(raws)]
            data = dict(exact=exact, route=bool(routed),
                        first_ms=dist([raw['first']['total_ms'] for raw in raws]),
                        repeat_ms=dist([raw['runs'][0]['total_ms'] for raw in raws]),
                        construct_ms=[raw['construct_ms'] for raw in raws],
                        rss_bytes=[raw['final_memory']['peak_rss_bytes'] for raw in raws],
                        capacities=capacities,
                        blocks=[dict(block=b, first_ms=raw['first']['total_ms'],
                                     repeat_ms=raw['runs'][0]['total_ms'])
                                for b, raw in enumerate(raws)],
                        accuracy=dict(truth_count=len(case['truth']),
                                      max_fp=max(a['fp'] for a in accuracies),
                                      max_fn=max(a['fn'] for a in accuracies),
                                      max_errors={k: max_present(a['max_errors'][k] for a in accuracies)
                                                  for k in accuracies[0]['max_errors']}),
                        load=None, after_load_memory=None)
            if method != 'frozen':
                data['load'] = [raw.get('load') for raw in raws]
                data['after_load_memory'] = [raw.get('after_load_memory') for raw in raws]
            methods[method] = data
        comparisons = {}
        for method, ref in [('current_legacy', 'frozen'), ('current_direct', 'current_legacy'),
                            ('current_direct', 'frozen')]:
            candidate, prior = methods[method], methods[ref]
            pairs = []
            for block in range(2):
                pair = dict(block=block, rss_delta_bytes=candidate['rss_bytes'][block] - prior['rss_bytes'][block],
                            rss_tolerance_pass=candidate['rss_bytes'][block] <= prior['rss_bytes'][block] +
                            max(.1 * prior['rss_bytes'][block], 16 * 1024 ** 2))
                for phase in ('first', 'repeat'):
                    a, b = prior['blocks'][block][phase + '_ms'], candidate['blocks'][block][phase + '_ms']
                    pair[phase + '_time_delta_percent'] = 100 * (b / a - 1)
                    pair[phase + '_time_tolerance_pass'] = b <= a + max(.1 * a, 2.)
                if ref == 'current_legacy':
                    a, b = prior['load'][block], candidate['load'][block]
                    am, bm = prior['after_load_memory'][block], candidate['after_load_memory'][block]
                    # Missing load records already fail correctness; keep a
                    # reportable negative control instead of throwing here.
                    pair['decode_delta_ms'] = b['decode_ms'] - a['decode_ms'] if a and b else None
                    pair['after_load_current_rss_delta_bytes'] = (
                        bm['current_rss_bytes'] - am['current_rss_bytes'] if am and bm else None)
                    pair['after_load_peak_rss_delta_bytes'] = (
                        bm['peak_rss_bytes'] - am['peak_rss_bytes'] if am and bm else None)
                pairs.append(pair)
            comparisons[method + '_vs_' + ref] = pairs
        summaries.append(dict(name=name, methods=methods, comparisons=comparisons))
    correct = all(m['exact'] and m['route'] and m['accuracy']['max_fp'] == m['accuracy']['max_fn'] == 0
                  for case in summaries for m in case['methods'].values())
    performance = all(pair[key] for case in summaries for pairs in case['comparisons'].values()
                      for pair in pairs for key in ('first_time_tolerance_pass',
                                                   'repeat_time_tolerance_pass', 'rss_tolerance_pass'))
    return dict(correctness_passed=correct, all_performance_tolerances_passed=performance,
                unchanged=report['unchanged'], workers=len(order), algorithm_calls=2 * len(order),
                first_calls=len(order), repeat_calls=len(order), warmup_calls=0,
                timing_scope='Fresh-process first plus one repeat; no warmup; not stage30 warm steady-state timing',
                cases=summaries)


def ensure_empty_output(dest):
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('Refusing to overwrite existing experiment: ' + str(dest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/gray-loader')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'build/fine-fused')
    parser.add_argument('--previous', type=Path, default=ROOT / 'cpp-core/docs/fine-fused/protocol.json')
    parser.add_argument('--baseline-protocol', type=Path, default=ROOT / 'cpp-core/docs/fine-fused/protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/gray-loader-results')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--case-names', default=','.join(CASE_NAMES), help='Predeclared pilot subset or the three formal settings')
    args = parser.parse_args()
    if args.summarize:
        if args.output.exists():
            raise ValueError('Refusing to overwrite existing replay output')
        summary = summarize(json.loads(args.summarize.read_text()), args.summarize.parent)
        save(args.output, summary)
        if not summary['correctness_passed'] or not summary['unchanged']:
            raise SystemExit(1)
        return
    dest = args.output.resolve()
    ensure_empty_output(dest)
    previous = json.loads(args.previous.read_text())['cases']
    selected = args.case_names.split(',')
    if (len({c['name'] for c in previous}) != len(previous) or
            not selected or len(set(selected)) != len(selected) or set(selected) - set(CASE_NAMES)):
        raise ValueError('Unknown or duplicate requested case')
    cases = [dict(next(c for c in previous if c['name'] == name), benchmark_repeats=REPEATS)
             for name in selected]
    verified = baseline(args.baseline, args.baseline_protocol)
    binaries = [build / 'experiments' / name for build in (args.build, args.baseline) for name in BINARY_NAMES]
    report = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), baseline_commit=FROZEN_COMMIT,
                  working_tree_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  frozen_binary_verified=verified, baseline_protocol_sha256=sha(args.baseline_protocol),
                  methods=METHODS, cases=cases, blocks=2, repeats=REPEATS, warm_seconds=WARM_SECONDS,
                  expected_workers=6 * len(cases),
                  rules=dict(order='frozen/current_legacy/current_direct then reverse; serial fresh processes',
                             exact='Ordered proposals/matches, all non-clock fine/work counters and frame/pose capacities, corresponding first/repeat phases',
                             timing='First and one repeat without warmup; algorithm includes prepare, excludes decode/model construction/serialization/GT; separate decode and construction metrics',
                             memory='Whole-process peak RSS; after-load current/peak RSS is available only for current arms; no invented frozen decode data',
                             tolerance='Per block and phase time max(10%,2ms); peak RSS max(10%,16MiB); retain every failed pair',
                             data='Three reused settings from two scenes; the 512 scene is reused at two accuracy settings; not independent camera data'),
                  source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_files()},
                  binary_sha256={str(path.resolve()): sha(path) for path in binaries},
                  input_sha256={case[key]: sha(case[key]) for case in cases for key in ('source', 'template')},
                  previous_sha256=sha(args.previous), execution_order=[], complete=False)
    save(dest / 'protocol.json', report)
    report['protocol_sha256'] = sha(dest / 'protocol.json')
    for name, method, block in worker_order(cases):
        case = next(c for c in cases if c['name'] == name)
        label = f'{name}-{method}-b{block}'
        jobpath = dest / 'jobs' / (label + '.json')
        save(jobpath, job_for(case, method))
        print('Running ' + label, flush=True)
        build = args.baseline if method == 'frozen' else args.build
        result = subprocess.run([str((build / 'experiments/fpm_native_pipeline_runner').resolve()), str(jobpath)],
                                capture_output=True, text=True)
        rawpath = jobpath.with_suffix('.result.json')
        rawpath.write_text(result.stdout)
        jobpath.with_suffix('.stderr.txt').write_text(result.stderr)
        report['execution_order'].append(dict(case=name, method=method, block=block,
            job=str(jobpath.relative_to(dest)), job_sha256=sha(jobpath),
            result=str(rawpath.relative_to(dest)), result_sha256=sha(rawpath), returncode=result.returncode))
        save(dest / 'formal.json', report)
        if result.returncode:
            raise RuntimeError(label + ' failed: ' + result.stderr[-3000:])
        json.loads(result.stdout)
    report['complete'] = True
    report['unchanged'] = (all(sha(ROOT / key) == digest for key, digest in report['source_sha256'].items()) and
                           all(sha(key) == digest for group in ('binary_sha256', 'input_sha256')
                               for key, digest in report[group].items()))
    save(dest / 'formal.json', report)
    summary = summarize(report, dest)
    save(dest / 'summary.json', summary)
    if not summary['correctness_passed'] or not report['unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
