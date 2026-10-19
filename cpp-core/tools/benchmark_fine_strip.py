"""Stage 28 SIMD strip load reuse, serial fresh-process ablation; truth never enters native workers.

The manifest is written before measurements. Replaying the summary uses only
relative raw-result paths and verifies their hashes, so archives are movable.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from benchmark_native_pipeline import native_job
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
MICRO = [
    ('tiny', 3, 3, 1, 512), ('small128', 19, 17, 128, 512),
    ('medium1', 64, 48, 1, 512), ('medium128', 64, 48, 128, 512),
    ('medium512', 64, 48, 512, 512), ('large128', 128, 128, 128, 512),
    ('uint32_edge', 257, 257, 17, 512), ('uint64_edge', 257, 258, 17, 512),
    ('tail513', 19, 17, 513, 512), ('narrow128', 1, 257, 128, 512),
    ('narrow_split291', 1, 1024, 291, 512), ('chunk_tail128', 63, 47, 128, 512),
]

MODES = {'frozen': 0, 'original': 0, 'strip32': 2, 'strip16': 3}
MICRO_METHODS = ['original', 'strip32', 'strip16']
FULL_METHODS = ['frozen', *MICRO_METHODS]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def dist(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)

def max_present(values):
    """No true positives must remain a reportable result, not an exception."""
    present = [value for value in values if value is not None]
    return max(present) if present else None

def checked(path, digest):
    if sha(path) != digest:
        raise ValueError('Hash mismatch: ' + str(path))
    return json.loads(path.read_text())

def full_job(case, method):
    job = native_job(case, 'native_tile', 5, 1.)
    if method != 'frozen':
        job['fine_ncc_mode'] = MODES[method]
    return job

def verify_baseline(build, protocol):
    recorded = json.loads(protocol.read_text())['binary_sha256']
    verified = {}
    for name in ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib', 'libfpm_pose_experiment.dylib'):
        candidates = [digest for path, digest in recorded.items() if path.endswith('/fine-reuse/experiments/'+name)]
        path = build/'experiments'/name
        if len(candidates) != 1 or sha(path) != candidates[0]:
            raise ValueError('Frozen stage27 binary differs: '+str(path))
        verified[name] = candidates[0]
    return verified

def counters(raw, suite):
    """Compare cold/warm phases separately: uploads legitimately change."""
    def one(r):
        if suite == 'micro':
            return [r[k] for k in ('maps','command_buffers','roi_pixels','repair_pixels','stats_dispatches','peak_batch','gpu_bytes','statistics_bytes')]
        work = [[w[k] for k in ('maps','windows','fine_maps','fine_stats_dispatches','fine_commands','fine_uploads','fine_upload_bytes','fine_roi_pixels','fine_repair_pixels','fine_peak_batch')] for w in r['work']]
        return [work, *[r[k] for k in ('frame_host_bytes','frame_coarse_gpu_bytes','frame_fine_gpu_bytes')]]
    return [one(raw['first']), *[one(r) for r in raw['runs']]]

def summarize(report, base):
    if not report['complete'] or len(report['execution_order']) != report['expected_workers']:
        raise ValueError('Incomplete experiment')
    grouped = {}
    seen = set()
    for e in report['execution_order']:
        job = checked(base / e['job'], e['job_sha256'])
        if e['returncode'] != 0:
            raise ValueError('Worker failed')
        if 'truth' in job:
            raise ValueError('Truth leaked to worker')
        raw = checked(base / e['result'], e['result_sha256'])
        key = (e['suite'], e['case'], e['method'], e['block'])
        if key in seen:
            raise ValueError('Duplicate worker')
        seen.add(key)
        if e['suite'] == 'micro':
            spec = next(c for c in report['micro_cases'] if c['name'] == e['case'])
            expected_job = dict(arguments=[MODES[e['method']],spec['tw'],spec['th'],spec['count'],spec['batch'],10,1000])
            if raw['mode'] != MODES[e['method']] or raw['template'] != [spec['tw'],spec['th']] or raw['count'] != spec['count'] or raw['batch'] != spec['batch']:
                raise ValueError('Probe route or dimensions differ')
        elif e['suite'] == 'full':
            spec = next(c for c in report['cases'] if c['name'] == e['case'])
            expected_job = full_job(spec, e['method'])
        else:
            raise ValueError('Unknown suite')
        if job != expected_job:
            raise ValueError('Worker settings differ from protocol')
        grouped.setdefault((e['suite'], e['case'], e['method']), []).append((e['block'], raw))
    cases = []
    for suite, specs in [('micro', report['micro_cases']), ('full', report['cases'])]:
        methods = MICRO_METHODS if suite == 'micro' else FULL_METHODS
        for spec in specs:
            name = spec['name']
            ref = grouped[(suite, name, methods[0])][0][1]
            results = {}
            for method in methods:
                blocks = sorted(grouped[(suite, name, method)])
                if sorted(b for b, _ in blocks) != [0, 1]:
                    raise ValueError('Missing/duplicate block')
                runs = [r for _, raw in blocks for r in raw['runs']]
                calls = [r for _, raw in blocks for r in [raw['first'], *raw['runs']]]
                expected = 10 if suite == 'micro' else 5
                if any(len(raw['runs']) != expected for _, raw in blocks):
                    raise ValueError('Wrong sample count')
                time_key = 'wall_ms' if suite == 'micro' else 'total_ms'
                m = dict(time_ms=dist([r[time_key] for r in runs]),
                         first_ms=[raw['first'][time_key] for _, raw in blocks],
                         setup_ms=[raw['setup_ms' if suite == 'micro' else 'construct_ms'] for _, raw in blocks],
                         blocks=[dict(block=b, median_ms=statistics.median(r[time_key] for r in raw['runs'])) for b, raw in blocks])
                reference_blocks = dict(grouped[(suite, name, methods[0])])
                m['counters_capacities_equal'] = all(counters(raw, suite) == counters(reference_blocks[b], suite) for b, raw in blocks)
                if suite == 'micro':
                    all_calls = calls
                    m['exact'] = all(raw['cpu_reference_bitwise_equal'] and raw['warmups_exact'] and raw['maps_fnv1a64'] == ref['maps_fnv1a64'] for _, raw in blocks) and all(r['exact_to_first'] for r in all_calls)
                    m['route'] = all(r['maps'] == spec['count'] and r['row_dispatches'] == 0 and r['row_bytes'] == 0 and r['command_buffers'] == 2*r['stats_dispatches'] and r['stats_device_ms'] >= 0 for r in all_calls)
                    m['rss_bytes'] = [raw['peak_rss_bytes_before_reference'] for _, raw in blocks]
                    m['stages'] = {k: dist([r[k] for r in runs]) for k in ('stats_device_ms', 'rotation_ms', 'repair_ms', 'ncc_ms', 'normalization_ms', 'wait_ms')}
                    m['capacities'] = [{k: raw['runs'][-1][k] for k in ('gpu_bytes', 'statistics_bytes', 'row_bytes')} for _, raw in blocks]
                    m['cpu_reference_windows'] = spec['count'] * 49 * len(blocks)
                else:
                    reference = ref['first']
                    m['exact'] = all(raw['exact_repeat'] for _, raw in blocks) and all(r['matches'] == reference['matches'] and r['proposals'] == reference['proposals'] for r in calls)
                    def workload(r):
                        return [[w[k] for k in ('maps', 'windows', 'fine_maps')] for w in r['work']]
                    m['work_equal'] = all(workload(r) == workload(reference) for r in calls)
                    m['route'] = all(r['models_matched'] == len(spec['scales']) for r in calls)
                    # Frozen stage27 already exposes the same counters.
                    m['route'] &= all(all(w['fine_ncc_mode'] == MODES[method] and
                        w['fine_row_dispatches'] == 0 and w['fine_row_bytes'] == 0 and w['fine_stats_device_ms'] >= 0 and
                        w['fine_commands'] == 2*w['fine_stats_dispatches'] for w in r['work']) for r in calls)
                    m['route'] &= all(any(w['fine_maps'] > 0 for w in r['work']) for r in calls)
                    m['stages'] = {k: dist([sum(w[k] for w in r['work']) for r in runs]) for k in (
                        'fine_stats_device_ms', 'fine_rotation_ms', 'fine_repair_ms', 'fine_ncc_ms', 'fine_normalization_ms', 'fine_wait_ms', 'fine_commands', 'fine_upload_bytes', 'fine_roi_pixels')}
                    m['rss_bytes'] = [raw['final_memory']['peak_rss_bytes'] for _, raw in blocks]
                    m['capacities'] = [{k: raw['runs'][-1][k] for k in ('frame_host_bytes', 'frame_coarse_gpu_bytes', 'frame_fine_gpu_bytes')} for _, raw in blocks]
                    accuracies = [evaluate(r['matches'], spec['truth'])['strict'] for r in calls]
                    m['accuracy'] = dict(truth_count=len(spec['truth']), max_fp=max(a['fp'] for a in accuracies), max_fn=max(a['fn'] for a in accuracies),
                        max_errors={k:max_present(a['max_errors'][k] for a in accuracies) for k in accuracies[0]['max_errors']})
                results[method] = m
            comparisons = {}
            for method, reference in [('strip32', 'original'), ('strip16', 'original'), ('strip16', 'strip32')] + ([(m, 'frozen') for m in MICRO_METHODS] if suite == 'full' else []):
                a, b = results[reference], results[method]
                comparisons[method+'_vs_'+reference] = [dict(block=i,
                    time_delta_percent=100*(b['blocks'][i]['median_ms']/a['blocks'][i]['median_ms']-1),
                    time_tolerance_pass=b['blocks'][i]['median_ms'] <= a['blocks'][i]['median_ms'] + max(.1*a['blocks'][i]['median_ms'], 2. if suite == 'full' else .1),
                    rss_delta_bytes=b['rss_bytes'][i]-a['rss_bytes'][i],
                    rss_tolerance_pass=b['rss_bytes'][i] <= a['rss_bytes'][i]+max(.1*a['rss_bytes'][i], 16*1024**2)) for i in range(2)]
            cases.append(dict(suite=suite, name=name, methods=results, comparisons=comparisons))
    correct = all(m['exact'] and m['route'] and m['counters_capacities_equal'] and m.get('work_equal', True) and m.get('accuracy', {}).get('max_fp', 0) == m.get('accuracy', {}).get('max_fn', 0) == 0 for c in cases for m in c['methods'].values())
    return dict(correctness_passed=correct, unchanged=report['unchanged'], workers=len(report['execution_order']),
                formal_calls=sum(len(raw['runs']) for blocks in grouped.values() for _, raw in blocks), cases=cases)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT/'build/fine-strip')
    p.add_argument('--baseline', type=Path, default=ROOT/'build/fine-reuse')
    p.add_argument('--previous', type=Path, default=ROOT/'build/gui-integration-results/formal.json')
    p.add_argument('--output', type=Path, default=ROOT/'build/fine-strip-results')
    p.add_argument('--summarize', type=Path)
    p.add_argument('--baseline-protocol', type=Path, default=ROOT/'cpp-core/docs/fine-reuse/protocol.json')
    p.add_argument('--cases', help='Comma-separated full case names; omit for all')
    p.add_argument('--skip-micro', action='store_true', help='Full-only confirmation run')
    a = p.parse_args()
    if a.summarize:
        summary = summarize(json.loads(a.summarize.read_text()), a.summarize.parent)
        save(a.output, summary)
        if not summary['correctness_passed'] or not summary['unchanged']:
            raise SystemExit(1)
        return
    dest = a.output.resolve()
    if (dest/'protocol.json').exists():
        raise ValueError('Refusing to overwrite an existing experiment')
    cases = json.loads(a.previous.read_text())['cases']
    micro = [dict(name=n, tw=w, th=h, count=c, batch=b) for n,w,h,c,b in MICRO]
    if a.cases:
        names = a.cases.split(',')
        if len(set(names)) != len(names) or set(names)-{c['name'] for c in cases}:
            raise ValueError('Unknown or duplicate full case')
        cases = [next(c for c in cases if c['name'] == name) for name in names]
    if a.skip_micro:
        micro = []
    frozen = verify_baseline(a.baseline, a.baseline_protocol)
    sources = [ROOT/'cpp-core/CMakeLists.txt', Path(__file__).resolve()]
    for folder in ('src', 'include', 'experiments', 'tests', 'tools'):
        sources += [f for f in (ROOT/'cpp-core'/folder).rglob('*') if f.is_file() and f.suffix in ('.cpp','.h','.hpp','.mm','.metal','.py','.txt')]
    binaries = [a.build/'experiments/fpm_fine_reuse_probe'] + [b/'experiments'/n for b in (a.build,a.baseline) for n in ('fpm_native_pipeline_runner','libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib')]
    report = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), platform=platform.platform(),
        baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_binary_verified=frozen, baseline_protocol_sha256=sha(a.baseline_protocol),
        cases=cases, micro_cases=micro, blocks=2, micro_repeats=10, full_repeats=5, warm_seconds=1,
        expected_workers=6*len(micro)+8*len(cases),
        rules=dict(order='original/strip32/strip16 then reverse; full frozen/original/strip32/strip16 then reverse; all serial fresh processes',
            exact='CPU map bits plus ordered native proposals/results; strict GT outside workers',
            speed_tolerance='full max(10%,2ms); isolated max(10%,0.1ms); retain failures and raw deltas',
            rss_tolerance='max(10%,16MiB); separate RSS from overlapping retained buffer capacities',
            decision='opt-in only; no automatic routing/global promotion based on this finite suite',
            timing='first/setup separate; full includes frame preparation; excludes decode, model learning, serialization and GT',
            data='Reused stage14 synthetic cases, not new held-out or real-camera evidence'),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(sources))},
        binary_sha256={str(f.resolve()):sha(f) for f in binaries},
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template')},
        previous_sha256=sha(a.previous), execution_order=[], complete=False)
    save(dest/'protocol.json', report)
    def run(suite, case, method, block, job, cmd):
        label=f'{suite}-{case["name"]}-{method}-b{block}'
        jobpath=dest/'jobs'/(label+'.json')
        save(jobpath,job)
        print('Running '+label,flush=True)
        result=subprocess.run(cmd(jobpath),capture_output=True,text=True)
        rawpath=jobpath.with_suffix('.result.json')
        rawpath.write_text(result.stdout)
        jobpath.with_suffix('.stderr.txt').write_text(result.stderr)
        report['execution_order'].append(dict(suite=suite,case=case['name'],method=method,block=block,
            job=str(jobpath.relative_to(dest)),job_sha256=sha(jobpath),result=str(rawpath.relative_to(dest)),result_sha256=sha(rawpath),returncode=result.returncode))
        save(dest/'formal.json',report)
        if result.returncode:
            raise RuntimeError(label+' failed: '+result.stderr[-3000:])
        json.loads(result.stdout)
    for c in micro:
        for block in range(2):
            for method in (MICRO_METHODS if block == 0 else MICRO_METHODS[::-1]):
                args=[MODES[method],c['tw'],c['th'],c['count'],c['batch'],10,1000]
                run('micro',c,method,block,dict(arguments=args),lambda _,args=args:[str((a.build/'experiments/fpm_fine_reuse_probe').resolve()),*map(str,args)])
    for c in cases:
        for block in range(2):
            for method in (FULL_METHODS if block == 0 else FULL_METHODS[::-1]):
                build=a.baseline if method=='frozen' else a.build
                run('full',c,method,block,full_job(c,method),lambda j,build=build:[str((build/'experiments/fpm_native_pipeline_runner').resolve()),str(j)])
    report['complete']=True
    report['unchanged']=all(sha(ROOT/k)==v for k,v in report['source_sha256'].items()) and all(sha(k)==v for group in ('binary_sha256','input_sha256') for k,v in report[group].items())
    save(dest/'formal.json',report)
    summary=summarize(report,dest)
    save(dest/'summary.json',summary)
    if not summary['correctness_passed'] or not report['unchanged']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
