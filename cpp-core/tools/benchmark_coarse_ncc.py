"""Fresh-process frozen-CPU versus Metal coarse NCC full-search comparison.

Every call rebuilds source pyramid and finds candidates online. Metal only
replaces dense top-level NCC statistics; all other search semantics are shared.
This benchmark measures candidate search, before optional continuous refinement.
Truth scenes label inputs; bit-exact CPU output agreement is the acceptance gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

from coarse_experiment import CoarsePattern, library_path as coarse_library_path
from real_image_experiments import (ROOT, Library, image_view, scaled_template,
    controlled_scene, held_out_scene, truth_accuracy)
from benchmark_pose_experiments import peak_rss

BASELINE_COMMIT = '958ab796ba1161a48a6e0511d0d4c6d687b5d735'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_library_path(build):
    name = 'libfpm_c.dylib' if sys.platform == 'darwin' else 'fpm_c.dll' if sys.platform == 'win32' else 'libfpm_c.so'
    for path in (Path(build)/name, Path(build)/'Release'/name):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f'CPU core library missing: {build}')


def baseline_provenance(build):
    path = Path(build)/'baseline-provenance.json'
    if not path.exists():
        return dict(verified=False, reason='missing frozen baseline provenance manifest')
    manifest = json.loads(path.read_text())
    actual = file_hash(core_library_path(build))
    expected = [h for p, h in manifest.get('sha256', {}).items() if Path(p).name == core_library_path(build).name]
    verified = manifest.get('commit') == BASELINE_COMMIT and bool(expected) and all(h == actual for h in expected)
    if not verified:
        raise ValueError('Frozen CPU baseline does not match 958ab79 provenance manifest')
    return dict(verified=True, commit=BASELINE_COMMIT, sha256=actual, manifest=str(path))


def worker(job):
    cv2.setNumThreads(1)
    source = cv2.imread(job['case']['source'], cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(job['case']['template'], cv2.IMREAD_GRAYSCALE)
    if source is None or template is None:
        raise ValueError('source/template missing')
    library = Library(str(core_library_path(job['baseline'])))
    config = library.config()
    case = job['case']
    config.max_targets = case['limit']; config.min_score = case['score']
    config.max_overlap = case['overlap']; config.tolerance_angle = case['angle']
    config.stop_layer = 1; config.subpixel = 0
    start = time.perf_counter()
    patterns = []
    for scale in case['scales']:
        scaled = scaled_template(template, scale)
        pattern = (CoarsePattern(job['build'], image_view(scaled), 256) if job['method'] == 'metal'
                   else library.learn(image_view(scaled), 256))
        patterns.append((scale, pattern))
    setup_ms = (time.perf_counter() - start) * 1000
    def run():
        start = time.perf_counter()
        matches, metrics = [], []
        for scale, pattern in patterns:
            rows = pattern.match(image_view(source), config)
            for row in rows:
                row['scale'] = scale
                matches.append(row)
            if job['method'] == 'metal':
                metrics.append(pattern.last_metrics)
        return dict(wall_ms=(time.perf_counter() - start) * 1000, matches=matches, stages=metrics)
    try:
        first = run()
        warm_start = time.perf_counter(); warmups = 0
        while time.perf_counter() - warm_start < job['warmup_seconds']:
            run(); warmups += 1
        runs = [run() for _ in range(job['repeats'])]
        # Capture RSS before independent truth checking. These large synthetic
        # fixtures have six planted objects and are final-detection tests;
        # the real multi-scale scenes here intentionally measure proposals.
        process_peak = peak_rss()
        accuracy = ([truth_accuracy(record['matches'], case['truth']) for record in [first, *runs]]
                    if case.get('final_truth_gate') else None)
        return dict(method=job['method'], case=case['name'], setup_ms=setup_ms, first=first, runs=runs,
            warmups=warmups, median_ms=statistics.median(r['wall_ms'] for r in runs),
            process_peak_rss_bytes=process_peak, truth_accuracy=accuracy,
            source_shape=list(source.shape), template_shape=list(template.shape),
            library_hash=file_hash(coarse_library_path(job['build']) if job['method'] == 'metal'
                                   else core_library_path(job['baseline'])))
    finally:
        for _, pattern in patterns:
            pattern.close()


def cases_for(args, folder):
    cases = []
    if args.suite in ('real', 'all'):
        controlled, heldout = folder / 'controlled.png', folder / 'heldout.png'
        truth6, truth12 = controlled_scene(controlled), held_out_scene(heldout)
        cases.extend([
            dict(name='src7', source=str(ROOT / 'Test Images/Src7.bmp'), template=str(ROOT / 'Test Images/Dst7.bmp'),
                 scales=[1.], angle=180, limit=70, score=.5, overlap=.2),
            dict(name='controlled6', source=str(controlled), template=str(ROOT / 'Test Images/Dst3.bmp'),
                 scales=[.75, 1., 1.25], angle=180, limit=70, score=.5, overlap=.2, truth=truth6),
            dict(name='heldout12', source=str(heldout), template=str(ROOT / 'Test Images/Dst3.bmp'),
                 scales=[.75, 1., 1.25], angle=180, limit=70, score=.5, overlap=.2, truth=truth12)])
    if args.suite in ('large', 'all'):
        for size in (256, 512):
            source = np.full((8192, 8192), 127, dtype=np.uint8)
            template = np.random.default_rng(7341 + size).integers(0, 256, (size, size), dtype=np.uint8)
            truth = []
            for y, x in ((512, 512), (512, 3840), (512, 7040), (6144, 512), (6144, 3840), (6144, 7040)):
                source[y:y+size, x:x+size] = template
                truth.append(dict(pos_x=x+size/2, pos_y=y+size/2, angle=0, scale=1))
            src, tmp = folder/f'8k_t{size}.png', folder/f'template{size}.png'
            cv2.imwrite(str(src), source); cv2.imwrite(str(tmp), template)
            cases.append(dict(name=f'8k_t{size}', source=str(src), template=str(tmp), scales=[1.],
                              angle=0, limit=12, score=.7, overlap=0., truth=truth, final_truth_gate=True))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(ROOT/'build/fourier-gpu'))
    parser.add_argument('--baseline', default=str(ROOT/'build/rescue-baseline'))
    parser.add_argument('--suite', choices=('real', 'large', 'all'), default='all')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--output', default=str(ROOT/'build/coarse-ncc/report.json'))
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(Path(args.worker).read_text())))); return
    if args.repeats < 1 or args.warmup_seconds < 0:
        parser.error('repeats must be positive and warmup nonnegative')
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    baseline = baseline_provenance(args.baseline)
    inputs = output.parent/'inputs'; inputs.mkdir(exist_ok=True)
    source_paths = [ROOT/'cpp-core'/x for x in (
        'src/ncc.cpp', 'src/internal.hpp', 'src/matcher.cpp', 'experiments/coarse_ncc.cpp',
        'experiments/coarse_metal.mm', 'experiments/coarse_ncc.metal', 'experiments/coarse_ncc.h',
        'tools/coarse_experiment.py', 'tools/benchmark_coarse_ncc.py')]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    libraries = {str(core_library_path(args.baseline)): file_hash(core_library_path(args.baseline)),
                 str(coarse_library_path(args.build)): file_hash(coarse_library_path(args.build))}
    rows, input_hashes = [], {}
    for case in cases_for(args, inputs):
        for name in ('source', 'template'):
            input_hashes[case[name]] = file_hash(case[name])
        methods = {}
        for method in ('cpu', 'metal'):
            job = dict(build=args.build, baseline=args.baseline, case=case, method=method,
                       repeats=args.repeats, warmup_seconds=args.warmup_seconds)
            job_path = output.parent/f'job-{case["name"]}-{method}.json'
            job_path.write_text(json.dumps(job))
            result = subprocess.run([sys.executable, __file__, '--worker', str(job_path)],
                                    capture_output=True, text=True, check=True)
            methods[method] = json.loads(result.stdout)
        reference = methods['cpu']['first']['matches']
        same = all(run['matches'] == reference for data in methods.values() for run in [data['first'], *data['runs']])
        truth_pass = (all(call['pose_fn'] == 0 and call['pose_fp'] == 0 for data in methods.values()
                          for call in data['truth_accuracy']) if case.get('final_truth_gate') else None)
        rows.append(dict(case=case, exact_all_outputs=same, truth_passed=truth_pass, methods=methods,
                         speedup=methods['cpu']['median_ms']/methods['metal']['median_ms']))
        print(f'{case["name"]}: exact={same}, truth={truth_pass}, CPU={methods["cpu"]["median_ms"]:.3f} ms, '
              f'Metal={methods["metal"]["median_ms"]:.3f} ms', flush=True)
    unchanged = (all(file_hash(ROOT/p) == h for p, h in hashes.items())
                 and all(file_hash(p) == h for p, h in {**libraries, **input_hashes}.items()))
    report = dict(protocol='fresh processes, frozen pre-change CPU versus opt-in dense Metal integer NCC; stop_layer=1; full source search each run',
                  scope='CPU pyramid/rotation/normalization/selection/fine search; GPU top-layer window statistics only',
                  accuracy='all cases require bit-exact CPU output agreement; 8K six-object final detections additionally require <=2px/1degree/2% scale with zero FP/FN for first and every measured call; real six/twelve scenes here test proposals, final accuracy is measured separately with rescue; truth never enters search',
                  memory='RSS includes Python, images and driver; shared Metal buffers overlap RSS; workspace cap excludes source/template/CPU scoremap',
                  platform=platform.platform(), source_hashes=hashes, library_hashes=libraries,
                  input_hashes=input_hashes, baseline_provenance=baseline,
                  sources_unchanged=unchanged, cases=rows,
                  passed=baseline['verified'] and unchanged and all(row['exact_all_outputs'] and row['truth_passed'] is not False for row in rows))
    output.write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
