"""Isolated Metal timing diagnostics; never changes the normal experiment build.

Applies a timing-only patch to a copied Metal host source. Serial fresh workers
compare the normal and instrumented library, including exact numerical parity.
GPU intervals overlap commit/wait intervals: they must not be added together.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

from benchmark_hybrid_refinement import CONFIG, pose_errors, prepare
from benchmark_pose_routing import specifications
from benchmark_pose_stress import ROOT, digest, peak_rss
from pose_experiment import Scorer, library_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_build(destination, baseline):
    source = destination/'source'
    build = destination/'build'
    source.mkdir(parents=True, exist_ok=True)
    for name in ('pose_metal.mm', 'pose_ncc.metal', 'metal_source.hpp.in'):
        shutil.copy2(ROOT/'cpp-core/experiments'/name, source/name)
    patch = Path(__file__).with_name('metal_cost_instrumentation.patch')
    subprocess.run(['patch', '--batch', '-p1', '-i', str(patch.resolve())], cwd=source, check=True)
    cmake = '''cmake_minimum_required(VERSION 3.20)
project(FpmMetalCostProfile LANGUAGES CXX OBJCXX)
find_package(Threads REQUIRED)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pose_ncc.metal" FPM_METAL_SOURCE)
configure_file(metal_source.hpp.in metal_source.hpp @ONLY)
add_library(fpm_pose_experiment SHARED "${FPM_SOURCE_ROOT}/pose_ncc.cpp" pose_metal.mm)
target_compile_features(fpm_pose_experiment PRIVATE cxx_std_17)
target_compile_definitions(fpm_pose_experiment PRIVATE FPM_EXPERIMENT_BUILD FPM_EXPERIMENT_METAL)
target_include_directories(fpm_pose_experiment PRIVATE "${FPM_SOURCE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(fpm_pose_experiment PRIVATE -Wall -Wextra -Wpedantic)
set_source_files_properties(pose_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(fpm_pose_experiment PRIVATE Threads::Threads "-framework Metal" "-framework Foundation")
set_target_properties(fpm_pose_experiment PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/experiments")
'''
    (source/'CMakeLists.txt').write_text(cmake)
    subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_OSX_ARCHITECTURES=arm64',
                    '-DFPM_SOURCE_ROOT='+str(ROOT/'cpp-core/experiments')], check=True)
    subprocess.run(['cmake', '--build', str(build), '-j', '4'], check=True)
    shutil.copy2(baseline/'libfpm_c.dylib', build/'libfpm_c.dylib')
    return build


def mark(phase, index, event):
    os.write(2, ('FPM_MARK '+json.dumps(dict(phase=phase, index=index, event=event))+'\n').encode())


def worker(job):
    if job['kind'] == 'local':
        arrays = [np.load(Path(job['fixture'])/(name+'.npy'), allow_pickle=False)
                  for name in ('source', 'template', 'seeds')]
        source, template, seeds = arrays
        constructor_begin = time.perf_counter()
        scorer = Scorer(job['build'], source, template, 'metal', 8)
        constructor_ms = (time.perf_counter()-constructor_begin)*1000
        close = scorer.close
        config = dict(CONFIG, **job['case'].get('config', {}))

        def call():
            poses, scores, stats, metrics = scorer.refine_accelerated(seeds, cpu_polish_iterations=0, **config)
            return dict(poses=poses.tolist(), scores=scores.tolist(), stats=stats, metrics=metrics)

        setup = dict(setup_ms=scorer.setup_ms, device=scorer.device,
                     input_sha256={name: digest(a) for name, a in zip(('source', 'template', 'seeds'), arrays)})
    else:
        import cv2
        from real_image_experiments import Search
        cv2.setNumThreads(1)
        constructor_begin = time.perf_counter()
        search = Search(dict(job, method='accelerated_direct_metal', cpu_polish_iterations=0, prune=False))
        constructor_ms = (time.perf_counter()-constructor_begin)*1000
        scorer = search.scorer
        close = search.close
        call = search.run
        setup = dict(setup_ms=search.setup_ms, device=scorer.device,
                     input_sha256={name: sha(job['case'][name]) for name in ('source', 'template')})
    try:
        initial_memory = scorer.memory()
        records = []

        def one(phase, index):
            mark(phase, index, 'begin')
            start = time.perf_counter()
            value = call()
            wall = (time.perf_counter()-start)*1000
            mark(phase, index, 'end')
            return dict(phase=phase, index=index, wall_ms=wall, result=value)

        records.append(one('first', 0))
        start = time.perf_counter()
        warmups = 0
        while warmups < 1 or (time.perf_counter()-start)*1000 < job['warmup_ms']:
            one('warmup', warmups)
            warmups += 1
        records.extend(one('measured', i) for i in range(job['repeats']))
        result = dict(**setup, constructor_wall_ms=constructor_ms, records=records, warmups=warmups,
                      memory=scorer.memory(), initial_memory=initial_memory, peak_rss_bytes=peak_rss())
        if job['kind'] == 'local':
            last = records[-1]['result']
            result['accuracy'] = pose_errors(last['poses'], job['case']['truth'], last['scores'])
        return result
    finally:
        close()


def parse_trace(path):
    setup, events, active = [], {}, None
    for line in path.read_text().splitlines():
        if line.startswith('FPM_MARK '):
            value = json.loads(line[len('FPM_MARK '):])
            active = (value['phase'], value['index']) if value['event'] == 'begin' else None
            if active:
                events.setdefault(active, [])
        elif line.startswith('FPM_PROFILE '):
            value = json.loads(line[len('FPM_PROFILE '):])
            if value['status'] != 'ok':
                raise RuntimeError('Failed instrumentation record')
            if value['operation'] == 'setup':
                setup.append(value)
            else:
                if active is None:
                    raise RuntimeError('Unattributed backend profile event')
                events[active].append(value)
    return setup, events


def summarize(entry):
    rows = [r for r in entry['result']['records'] if r['phase'] == 'measured']
    # Pick a complete median-wall run, so additive stage times stay consistent.
    representative = sorted(rows, key=lambda r: r['wall_ms'])[len(rows)//2]
    totals = {}
    for event in representative['profile']:
        for key, value in event.items():
            if type(value) in (int, float) and not key.endswith('_capacity'):
                totals[key] = totals.get(key, 0)+value
    return dict(wall_median_ms=statistics.median(r['wall_ms'] for r in rows),
                wall_samples_ms=[r['wall_ms'] for r in rows], representative_index=representative['index'],
                representative_profile_totals=totals,
                representative_api_metrics=representative['result'].get('metrics'),
                representative_full_stages={k: v for k, v in representative['result'].items()
                                            if k.endswith('_ms') or k == 'seed_count'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--directory', type=Path, default=ROOT/'build/metal-cost-profile')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmup-ms', type=float, default=1000)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        args.output.write_text(json.dumps(worker(json.loads(args.worker.read_text())), indent=2)+'\n')
        return
    if args.repeats < 1 or args.repeats % 2 != 1 or args.warmup_ms < 0:
        parser.error('repeats must be positive and odd, warmup-ms nonnegative')
    args.directory = args.directory.resolve()
    args.baseline = args.baseline.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    build = prepare_build(args.directory, args.baseline)
    files = list((ROOT/'cpp-core/experiments').glob('*'))
    files += [Path(__file__).resolve(), Path(__file__).with_name('metal_cost_instrumentation.patch')]
    files += [ROOT/'cpp-core/tools'/name for name in ('pose_experiment.py', 'benchmark_pose_routing.py',
              'benchmark_hybrid_refinement.py', 'benchmark_pose_stress.py', 'real_image_experiments.py')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()}
    binary = {str(p.relative_to(ROOT)): sha(p) for p in (library_path(args.baseline), library_path(build),
               args.baseline/'libfpm_c.dylib', build/'libfpm_c.dylib')}
    jobs = []
    names = {'heldout_s0160_c1_n004', 'heldout_s0512_c3_n016', 'calibration_s0768_c1_n032'}
    cases = [case for case in specifications('all') if case['name'] in names]
    chunked = dict(next(c for c in cases if c['size'] == 768))
    offsets = np.random.default_rng(31329).uniform([-2, -2, -3, -.03], [2, 2, 3, .03], (256, 4))
    chunked.update(name='chunked_768_c1_n256_one_iteration', initial_poses=(np.asarray(chunked['truth'])+offsets).tolist(),
                   config=dict(max_iterations=1))
    cases.append(chunked)
    for case in cases:
        fixture = args.directory/('fixture-'+case['name'])
        if fixture.exists():
            # Use a fresh subdirectory without deleting prior experiments.
            fixture = args.directory/('fixture-'+case['name']+'-'+str(time.time_ns()))
        prepare(case, fixture)
        jobs.append(dict(kind='local', case=case, fixture=str(fixture)))
    from real_image_experiments import held_out_scene
    heldout = args.directory/'held-out-real-texture.png'
    truth = held_out_scene(heldout)
    jobs += [dict(kind='full', case=dict(name='src7', source=str(ROOT/'Test Images/Src7.bmp'),
                  template=str(ROOT/'Test Images/Dst7.bmp'), angle=180, limit=70, proposal_score=.5,
                  score=.5, overlap=.2, scales=[1.])),
             dict(kind='full', case=dict(name='held_out_scale', source=str(heldout),
                  template=str(ROOT/'Test Images/Dst3.bmp'), angle=180, limit=70, proposal_score=.5,
                  score=.8, overlap=.2, scales=[.75, 1., 1.25], truth=truth))]
    jobs = [dict(job, variant=variant, build=str(path), repeats=args.repeats, warmup_ms=args.warmup_ms)
            for job in jobs for variant, path in [('baseline', args.baseline), ('profile', build)]]
    np.random.default_rng(19347).shuffle(jobs)
    report = dict(created_at=datetime.now().astimezone().isoformat(), platform=platform.platform(),
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_sha256=hashes, binary_sha256=binary,
                  protocol=dict(serial=True, repeats=args.repeats, warmup_ms=args.warmup_ms,
                    instrumented='host clocks and stderr JSON; numerical kernels unchanged',
                    timing='commit_wait includes GPU; readback_assemble includes CPU matrix assembly; wall residual is not transfer',
                    cold='context construction and first call, no OS/driver/shader cache purge',
                    constructor='full-image constructor_wall_ms includes image decode; setup_ms excludes decode',
                    modes='forced Metal, CPU polish=0; one iteration chunk case is a batching probe, not a convergence test'),
                  runs=[])
    output = args.output or args.directory/'report.json'
    for i, job in enumerate(jobs):
        stem = f"{job['case']['name']}-{job['variant']}"
        request, response, trace = [args.directory/(stem+ext) for ext in ('.request.json', '.response.json', '.trace.txt')]
        request.write_text(json.dumps(job))
        with trace.open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request), '--output', str(response)],
                           stderr=log, check=True, timeout=300)
        result = json.loads(response.read_text())
        setups, events = parse_trace(trace)
        for row in result['records']:
            row['profile'] = events.get((row['phase'], row['index']), [])
        entry = dict(job=job, result=result, metal_setup=setups, trace_sha256=sha(trace))
        entry['summary'] = summarize(entry)
        report['runs'].append(entry)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(f"{i+1}/{len(jobs)} {stem}: {entry['summary']['wall_median_ms']:.3f} ms", flush=True)
    report['parity'] = []
    for name in sorted({j['case']['name'] for j in jobs}):
        pair = {r['job']['variant']: r for r in report['runs'] if r['job']['case']['name'] == name}
        a, b = pair['baseline']['result'], pair['profile']['result']
        keys = ('poses', 'scores', 'stats') if pair['baseline']['job']['kind'] == 'local' else ('matches', 'solver_stats')
        same = a['input_sha256'] == b['input_sha256'] and len(a['records']) == len(b['records']) and all(
            all(left['result'][key] == right['result'][key] for key in keys)
            for left, right in zip(a['records'], b['records']))
        report['parity'].append(dict(case=name, exact_outputs=same,
            instrumented_over_baseline_wall=pair['profile']['summary']['wall_median_ms']/pair['baseline']['summary']['wall_median_ms']))
    report['unchanged_during_measurement'] = all(sha(ROOT/p) == value for p, value in {**hashes, **binary}.items())
    output.write_text(json.dumps(report, indent=2)+'\n')
    assert report['unchanged_during_measurement'] and all(r['exact_outputs'] for r in report['parity'])
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
