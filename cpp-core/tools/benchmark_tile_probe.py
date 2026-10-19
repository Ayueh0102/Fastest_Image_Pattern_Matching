"""Isolated coarse NCC tile-kernel probe, separate from full search benchmarks.

Each child measures one fixed source/template pair. Full CPU bitwise comparison
runs after RSS sampling; correctness checks/digests are outside timed intervals.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

CASES = {
    'tiny3': (64, 64, 3, 3),
    'coarse13x17': (512, 512, 13, 17),
    'template32': (256, 256, 32, 32),
    'template64': (256, 256, 64, 64),
    'template128': (256, 256, 128, 128),
    'uint32_edge': (273, 271, 257, 257),
    'uint64_edge': (273, 272, 257, 258),
    'chunked3': (1537, 1001, 3, 3),
    'narrow3x5': (20, 1024, 3, 5),
}
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def distribution(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)


def summarize(report, worker_root=None):
    cases = {}
    for entry in report['execution_order']:
        path = Path(entry['result']) if worker_root is None else Path(worker_root)/entry['relative_result']
        if sha(path) != entry['sha256']:
            raise ValueError('Worker hash mismatch: '+str(path))
        worker = json.loads(path.read_text())
        cases.setdefault(entry['case'], {}).setdefault(str(entry['mode']), []).append((entry['block'], worker))
    result = {}
    for name, methods in cases.items():
        reference = methods['0'][0][1]
        row = {}
        for mode, blocks in methods.items():
            runs = [r for _, w in blocks for r in w['runs']]
            calls = [r for _, w in blocks for r in [w['first'], *w['warmups'], *w['runs']]]
            exact = all(w['cpu_reference_bitwise_equal'] and
                        w['map_fnv1a64_native_binary64'] == reference['map_fnv1a64_native_binary64']
                        for _, w in blocks) and all(r['exact_to_first'] for r in calls)
            stable = all(all(r[k] == w['first'][k] for k in ('gpu_bytes', 'workspace_bytes'))
                         for _, w in blocks for r in [*w['warmups'], *w['runs']])
            work = all(all(r[k] == reference['first'][k] for k in ('windows', 'command_buffers',
                        'integer32_maps', 'integer64_maps')) for r in calls)
            row[mode] = dict(exact=exact, stable_capacity=stable, identical_work=work,
                times={k: distribution([r[k] for r in runs]) for k in
                       ('wall_ms', 'device_ms', 'upload_ms', 'wait_ms', 'normalization_ms')},
                rss_bytes=[w['peak_rss_bytes_before_cpu_reference'] for _, w in blocks],
                setup_ms=[w['setup_ms'] for _, w in blocks], first=[w['first'] for _, w in blocks],
                blocks=[dict(block=b, wall_ms=statistics.median(r['wall_ms'] for r in w['runs']),
                             device_ms=statistics.median(r['device_ms'] for r in w['runs'])) for b, w in blocks],
                map_digest=reference['map_fnv1a64_native_binary64'])
        result[name] = row
    return dict(complete=report['complete'], design=report['design'], cases=result,
                exact=all(m['exact'] and m['stable_capacity'] and m['identical_work']
                          for row in result.values() for m in row.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT/'build/coarse-tiling')
    p.add_argument('--cases', default=','.join(CASES))
    p.add_argument('--output', type=Path, default=ROOT/'build/coarse-tiling-probe/formal.json')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--summarize', type=Path); p.add_argument('--worker-root', type=Path)
    args = p.parse_args(); out = args.output.resolve()
    if args.summarize:
        if out == args.summarize.resolve(): p.error('Do not overwrite raw report')
        save(out, summarize(json.loads(args.summarize.read_text()), args.worker_root)); return
    names = args.cases.split(',')
    if set(names)-set(CASES) or len(names)!=len(set(names)): p.error('Invalid or duplicate cases')
    build = args.build.resolve(); probe = build/'experiments/fpm_coarse_tile_probe'
    jobs = out.parent/(out.stem+'-workers'); jobs.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT/'cpp-core/tools/coarse_tile_probe.cpp', Path(__file__).resolve(),
                    ROOT/'cpp-core/docs/coarse-tiling/protocol.json']
    for directory in ('src', 'include', 'experiments'):
        source_files += [f for f in (ROOT/'cpp-core'/directory).rglob('*') if f.is_file()]
    source_files += [ROOT/'cpp-core/CMakeLists.txt']
    hashes = {str(f): sha(f) for f in source_files}
    binary_files = (probe, build/'experiments/libfpm_coarse_experiment.dylib')
    binaries = {str(f): sha(f) for f in binary_files}
    report = dict(created=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        design=dict(cases={n: CASES[n] for n in names}, modes=[0,2,3], blocks=1 if args.pilot else 2,
                    repeats=2 if args.pilot else 10, warm_ms=0 if args.pilot else 1000, pilot=args.pilot,
                    rss='Includes source/template/retained first score map/Metal buffers; sampled before CPU reference.',
                    exact='Every call full bitwise vs first; first full bitwise vs CPU scalar. Digest is FNV1a64, not a cryptographic hash.'),
        source_hashes=hashes, binary_hashes=binaries, execution_order=[], complete=False)
    save(out, report)
    for name in names:
        for block in range(report['design']['blocks']):
            for mode in [0,2,3] if block%2==0 else [3,2,0]:
                command = [str(probe), str(mode), *map(str, CASES[name]), str(report['design']['repeats']),
                           str(report['design']['warm_ms'])]
                label=f'{name}-mode{mode}-b{block}'; print('Running '+label, flush=True)
                child = subprocess.run(command, capture_output=True, text=True)
                (jobs/(label+'.stderr.txt')).write_text(child.stderr)
                if child.returncode: raise RuntimeError(label+': '+child.stderr)
                worker = json.loads(child.stdout); path = jobs/(label+'.json'); save(path, worker)
                report['execution_order'].append(dict(case=name, mode=mode, block=block, command=command,
                    result=str(path), relative_result=str(path.relative_to(out.parent)), sha256=sha(path)))
                save(out, report)
    report['sources_unchanged'] = all(sha(f)==h for f,h in {**hashes,**binaries}.items())
    report['complete'] = True; report['summary'] = summarize(report)
    save(out, report)
    if not report['summary']['exact'] or not report['sources_unchanged']: raise SystemExit(1)


if __name__ == '__main__': main()
