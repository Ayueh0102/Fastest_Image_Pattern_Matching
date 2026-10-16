"""Stage34 balanced same-frontend CPU/Metal refinement and latency distribution.

Predeclared independent fresh processes; first/repeat are correlated observations.
Replay verifies job/raw/stderr hashes and does not need the original machine.
Common GT failures are retained separately from implementation agreement.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

from benchmark_native_pipeline import sha, save
from benchmark_resident_multiscale import job_for, frontend_state, frontend_valid
from benchmark_fine_fused import source_files
from benchmark_pose_simd_pipeline import final_comparison
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
CASES = ['controlled6', 'many128_pose', 'many512_pose', 'new_small12']
METHODS = ['cpu', 'metal']


def distribution(values):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Expected finite nonnegative observations')
    ordered = sorted(values)
    def quantile(p):
        x = (len(ordered) - 1) * p
        i = int(x)
        return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (x - i)
    return dict(n=len(values), median=statistics.median(values), p95=quantile(.95),
                min=min(values), max=max(values), values=values,
                note='P95 is descriptive linear interpolation; small-sample tail estimate')


def paired_change(base, candidate):
    if len(base) != len(candidate) or not base or any(v <= 0 for v in base):
        raise ValueError('Need complete positive paired baseline')
    deltas = [100 * (b / a - 1) for a, b in zip(base, candidate)]
    rng = random.Random(2026101605)
    boot = sorted(statistics.median(rng.choices(deltas, k=len(deltas))) for _ in range(4000))
    return dict(candidate_minus_baseline_percent=deltas, median_percent=statistics.median(deltas),
                percentile_bootstrap_95=[boot[99], boot[3899]], independent_pairs=len(base),
                note='Resample process pairs, not first/repeat or individual detected objects')


def order(cases, blocks):
    if blocks < 2 or blocks % 2:
        raise ValueError('Need an even number of at least two blocks')
    return [(case, block, method) for block in range(blocks)
            for case in (cases if block % 2 == 0 else cases[::-1])
            for method in (METHODS if block % 2 == 0 else METHODS[::-1])]


def job(case, method):
    result = job_for(case, 'resident0')
    result.update(refinement_backend=int(method == 'metal'), cpu_threads=8,
                  pose_storage_mode=int(method == 'metal'), run_resource_diagnostics=True)
    return result


def route(row, method):
    active = bool(row['proposals'])
    cpu = method == 'cpu'
    return (active and row['refinement_mode'] == 0 and row['pose_input_mode'] == 1
            and row['pose_storage_mode'] == int(not cpu)
            and (not active or row['selected_refinement_backend'] == int(not cpu))
            and ((row['pose_gpu_bytes'] == 0 and row['refinement_dispatches'] == 0)
                 if cpu else (not active or row['pose_gpu_bytes'] > 0 and row['refinement_dispatches'] > 0)))


def summarize(report, base):
    protocol = report['protocol']
    if not report.get('complete') or sha(base/'protocol.json') != report['protocol_sha256']:
        raise ValueError('Incomplete run or altered protocol')
    if json.loads((base/'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Embedded protocol differs')
    expected = order([c['name'] for c in protocol['cases']], protocol['blocks'])
    if ([c['name'] for c in protocol['cases']] != CASES or
            protocol.get('cpu_threads') != 8 or protocol.get('first_and_one_repeat') is not True or
            protocol.get('order') != [list(x) for x in expected] and protocol.get('order') != expected or
            protocol.get('time_gate') != 'max(10%,2ms)' or protocol.get('rss_gate') != 'max(10%,16MiB)'):
        raise ValueError('Protocol semantics differ from predeclared study')
    if report.get('post_hashes') != protocol['hashes']:
        raise ValueError('Source, binary or input provenance changed')
    entries = report['entries']
    if [(e['case'], e['block'], e['method']) for e in entries] != expected:
        raise ValueError('Missing, duplicate or reordered workers')
    workers = {}
    for entry in entries:
        for field in ('job', 'raw', 'stderr'):
            if sha(base / entry[field]) != entry[field + '_sha256']:
                raise ValueError('Tampered ' + entry[field])
        raw = json.loads((base / entry['raw']).read_text())
        case = next(c for c in protocol['cases'] if c['name'] == entry['case'])
        if json.loads((base/entry['job']).read_text()) != job(case,entry['method']):
            raise ValueError('Worker job differs from protocol')
        if entry['returncode'] or len(raw['runs']) != 1 or raw.get('warmups_ms') != []:
            raise ValueError('Worker failed or unexpected repeats')
        workers[(entry['case'], entry['block'], entry['method'])] = raw
    cases = []
    for case in protocol['cases']:
        methods, pairs = {}, []
        for method in METHODS:
            rows = [workers[(case['name'], b, method)] for b in range(protocol['blocks'])]
            calls = [r for w in rows for r in [w['first'], *w['runs']]]
            accuracies = [evaluate(r['matches'], case['truth'])['strict'] for r in calls]
            methods[method] = dict(
                phases={phase: {key: distribution([(w['first'] if phase == 'first' else w['runs'][0])[key] for w in rows])
                            for key in ('total_ms', 'proposal_ms', 'refine_ms', 'prepare_ms', 'nms_ms')}
                        for phase in ('first', 'repeat')},
                peak_rss_bytes=distribution([w['final_memory']['peak_rss_bytes'] for w in rows]),
                repeat_resources={key: distribution([w['runs'][0]['resources'][key] for w in rows])
                    for key in ('wall_ms','process_user_ms','process_system_ms','minor_faults','major_faults','voluntary_switches','involuntary_switches')},
                route_valid=all(route(r, method) for r in calls),
                frontend_valid=all(frontend_valid(w[phase] if phase == 'first' else w['runs'][0],
                                   workers[(case['name'], 0, 'cpu')][phase] if phase == 'first' else workers[(case['name'], 0, 'cpu')]['runs'][0], case)
                                   and (w[phase] if phase == 'first' else w['runs'][0])['proposals'] ==
                                       (workers[(case['name'], 0, 'cpu')][phase] if phase == 'first' else workers[(case['name'], 0, 'cpu')]['runs'][0])['proposals']
                                   for w in rows for phase in ('first', 'repeat')),
                # Small templates can stop at the top level and legitimately
                # perform only coarse GPU maps. Do not require nonexistent fine work.
                frontend_gpu_exercised=all(r['frame_coarse_gpu_bytes'] > 0 and
                    any(work['maps'] > 0 and work['device_ms'] >= 0 for work in r['work']) and
                    all(not work['fine_maps'] or work['fine_stats_dispatches'] > 0 for work in r['work']) for r in calls),
                repeat_exact=all(w['exact_repeat'] for w in rows),
                strict_tp=min(a['tp'] for a in accuracies), strict_fp=max(a['fp'] for a in accuracies),
                strict_fn=max(a['fn'] for a in accuracies), gt_all=all(a['fp'] == a['fn'] == 0 for a in accuracies))
        for block in range(protocol['blocks']):
            a, b = [workers[(case['name'], block, method)] for method in METHODS]
            for phase, cpu, metal in [('first', a['first'], b['first']), ('repeat', a['runs'][0], b['runs'][0])]:
                pairs.append(dict(block=block, phase=phase,
                    frontend_exact=frontend_state(cpu) == frontend_state(metal) and cpu['proposals'] == metal['proposals'],
                    final_pair=final_comparison(metal['matches'], cpu['matches']),
                    total_time_gate=metal['total_ms'] <= cpu['total_ms'] + max(2., cpu['total_ms'] * .1),
                    rss_gate=b['final_memory']['peak_rss_bytes'] <= a['final_memory']['peak_rss_bytes'] + max(16*2**20, a['final_memory']['peak_rss_bytes']*.1)))
        changes = {phase: {key: paired_change(methods['cpu']['phases'][phase][key]['values'], methods['metal']['phases'][phase][key]['values'])
                    for key in ('total_ms', 'refine_ms')} for phase in ('first', 'repeat')}
        cases.append(dict(name=case['name'], methods=methods, pairs=pairs, metal_vs_cpu=changes))
    result = dict(complete=True, workers=len(entries), algorithm_calls=2*len(entries), cases=cases,
        integrity_passed=bool(report.get('unchanged')) and all(m['route_valid'] and m['repeat_exact'] and m['frontend_valid'] and m['frontend_gpu_exercised'] for c in cases for m in c['methods'].values())
            and all(p['frontend_exact'] for c in cases for p in c['pairs']),
        paired_final_passed=all(p['final_pair']['valid'] for c in cases for p in c['pairs']),
        accuracy_passed=all(m['gt_all'] for c in cases for m in c['methods'].values()),
        limitations=['Four fixed scenes; repeats do not add independent accuracy scenes.',
            'CPU and Metal use their supported image storage modes; not a kernel-only microbenchmark.',
            'The recorded process pairs characterize this machine/run; not all machines or a reliable extreme-tail estimate.'])
    result.update(correctness_passed=result['integrity_passed'] and result['paired_final_passed'] and result['accuracy_passed'],
                  all_time_tolerances_passed=all(p['total_time_gate'] for c in cases for p in c['pairs']),
                  all_rss_tolerances_passed=all(p['rss_gate'] for c in cases for p in c['pairs']),
                  accuracy_failures=[dict(case=c['name'], method=m, tp=v['strict_tp'], fp=v['strict_fp'], fn=v['strict_fn'])
                                     for c in cases for m,v in c['methods'].items() if not v['gt_all']],
                  performance_failures=[dict(case=c['name'], block=p['block'], phase=p['phase'], time_failed=not p['total_time_gate'], rss_failed=not p['rss_gate'])
                                        for c in cases for p in c['pairs'] if not p['total_time_gate'] or not p['rss_gate']])
    result['all_performance_tolerances_passed'] = result['all_time_tolerances_passed'] and result['all_rss_tolerances_passed']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT/'build/stage34-closure')
    p.add_argument('--cases-json', type=Path, default=ROOT/'build/resident-multiscale-inputs/cases.json')
    p.add_argument('--blocks', type=int, default=12)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summarize', type=Path)
    args = p.parse_args()
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        result = summarize(report, args.summarize.parent)
        save(args.output, result)
        return 0 if result['correctness_passed'] else 1
    dest = args.output.resolve()
    if dest.exists():
        raise ValueError('Refusing to overwrite evidence')
    dest.mkdir(parents=True)
    all_cases = json.loads(args.cases_json.read_text())
    cases = [next(c for c in all_cases if c['name'] == name) for name in CASES]
    sources = source_files()
    binaries = [args.build/'experiments'/n for n in ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib', 'libfpm_pose_experiment.dylib')]
    frozen = {str(f.resolve()): sha(f) for f in [*sources, *binaries, args.cases_json, *[Path(c[k]) for c in cases for k in ('source', 'template')]]}
    report = dict(protocol=dict(cases=cases, blocks=args.blocks, first_and_one_repeat=True,
        cpu_threads=8, order=order(CASES,args.blocks), hashes=frozen,
        time_gate='max(10%,2ms)', rss_gate='max(10%,16MiB)',
        storage_note='CPU private image; Metal shared image; all u8 input; same proposals/budget'), entries=[], complete=False)
    save(dest/'protocol.json', report['protocol'])
    report['protocol_sha256'] = sha(dest/'protocol.json')
    runner = args.build/'experiments/fpm_native_pipeline_runner'
    for name, block, method in order(CASES, args.blocks):
        case = next(c for c in cases if c['name'] == name)
        stem = f'{name}-{method}-{block}'
        jp = dest/'jobs'/(stem+'.json'); rp = jp.with_suffix('.raw.json'); ep = jp.with_suffix('.stderr.txt')
        save(jp, job(case, method))
        print(stem, flush=True)
        child = subprocess.run([str(runner), str(jp)], capture_output=True, text=True)
        rp.write_text(child.stdout); ep.write_text(child.stderr)
        entry = dict(case=name, block=block, method=method, returncode=child.returncode)
        for field, path in [('job',jp), ('raw',rp), ('stderr',ep)]:
            entry[field] = str(path.relative_to(dest)); entry[field+'_sha256'] = sha(path)
        report['entries'].append(entry); save(dest/'formal.json',report)
        if child.returncode:
            raise RuntimeError(stem+': '+child.stderr)
    post_hashes = {k: sha(k) for k in frozen}
    report.update(complete=True, unchanged=post_hashes == frozen, post_hashes=post_hashes)
    save(dest/'formal.json', report)
    result = summarize(report, dest); save(dest/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'},indent=2))
    return 0 if result['correctness_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
