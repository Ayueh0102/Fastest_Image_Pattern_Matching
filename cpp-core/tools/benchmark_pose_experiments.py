"""Reproducible CPU/Metal microbenchmarks and ROI registration ablations.

Every variant/case runs in a fresh process, serially. Timings exclude image I/O,
synthetic generation, and setup; those boundaries are explicit in the report.
The FFT proposal uses NumPy on CPU; the fused warp/NCC has full C++/Metal source.
This is NOT a whole-image multi-instance detector benchmark.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from pose_experiment import Scorer, library_path
from research_ncc_fourier import (channels, log_polar_magnitude, phase_shift,
                                  similarity_scene, transform_template)

ROOT = Path(__file__).resolve().parents[2]
OFFSETS = np.array(list(itertools.product((-1, 0, 1), repeat=4)), np.float32)


def peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value*1024)
    except ImportError:
        return None


def summary(values):
    return dict(samples_ms=values, median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values))


def micro_data(case):
    if case.get("fixture"):
        import cv2  # decoding only
        source = cv2.imread(str(ROOT / "Test Images/Src7.bmp"), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        template = cv2.imread(str(ROOT / "Test Images/Dst7.bmp"), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        baseline = json.loads((ROOT / "cpp-core/docs/mode-benchmark.json").read_text())
        matches = next(m['matches'] for m in baseline['modes'] if m['mode'] == 'portable_simd')
        poses = []
        for m in matches:
            center = np.array([m['pos_x']-source.shape[1]/2, source.shape[0]/2-m['pos_y'], m['angle'], 0])
            poses.extend(center+np.array([dx, dy, angle, 0]) for dx, dy, angle in itertools.product((-1, 0, 1), repeat=3))
        return source, template, np.array(poses, np.float32)
    size, rgb, count = case['size'], case['rgb'], case['count']
    source = similarity_scene(512, .91, 31.25, (5.5, -8.25), rgb=rgb).astype(np.float32)
    template = similarity_scene(size, rgb=rgb).astype(np.float32)
    rng = np.random.default_rng(20261012+size+count)
    truth = np.array([5.5, -8.25, 31.25, np.log(.91)], np.float32)
    poses = truth+rng.uniform([-4, -4, -5, -.05], [4, 4, 5, .05], (count, 4)).astype(np.float32)
    poses[0] = truth
    return source, template, poses


def micro_worker(job):
    source, template, poses = micro_data(job['case'])
    before = peak_rss()
    start = time.perf_counter()
    with Scorer(job['build'], source, template, job['backend'], job['threads']) as scorer:
        setup = (time.perf_counter()-start)*1000
        scores, first = scorer.score(poses)
        for _ in range(job['warmups']-1):
            scorer.score(poses)
        warmed = time.perf_counter()
        extra_warmups = 0
        while time.perf_counter()-warmed < job['warmup_ms']/1000:
            scorer.score(poses)
            extra_warmups += 1
        records = [scorer.score(poses) for _ in range(job['repeats'])]
        scores = records[-1][0]
        metrics = [item[1] for item in records]
        return dict(device=scorer.device, setup_ms=setup, first_call=first, extra_warmups=extra_warmups,
                    wall=summary([m['wall_ms'] for m in metrics]),
                    gpu_command=summary([m['device_ms'] for m in metrics]) if job['backend'] == 'metal' else None,
                    memory=dict(process_peak_before_setup_bytes=before, process_peak_bytes=peak_rss(),
                                **{k: max(m[k] for m in metrics) for k in ('host_buffer_bytes', 'gpu_buffer_bytes', 'gpu_allocated_bytes')}),
                    shape=dict(source=list(source.shape), template=list(template.shape), poses=len(poses)),
                    scores=scores.tolist(), best_index=int(np.argmax(scores)),
                    truth_pose_included=not bool(job['case'].get('fixture')))


def registration_data(case):
    rgb = case.get('rgb', False)
    template = similarity_scene(case.get('template_size', 256), rgb=rgb).astype(np.float32)
    truth = case.get('truth', dict(scale=.8, angle=31, translation=[8, -6]))
    source = similarity_scene(256, truth['scale'], truth['angle'], truth['translation'], rgb=rgb)
    truths = [truth]
    rng = np.random.default_rng(1813)
    if case.get('stress') == 'background':
        source = source + .4*rng.random(source.shape)
    elif case.get('stress') == 'occlusion':
        source[:, 136:] = 0
    elif case.get('stress') == 'two_objects':
        truth = dict(scale=.8, angle=31, translation=[-8, 8])
        other = dict(scale=1.1, angle=-47, translation=[8, -8])
        source = (similarity_scene(256, truth['scale'], truth['angle'], truth['translation']) +
                  similarity_scene(256, other['scale'], other['angle'], other['translation']))
        truths = [truth, other]
    elif case.get('stress') == 'negative':
        source = rng.random(source.shape)
        truths = []
    return source.astype(np.float32), template, truths


class FourierProposal:
    def __init__(self, template, shape):
        self.template = np.zeros(tuple(shape[:2])+template.shape[2:], np.float64)
        y, x = (shape[0]-template.shape[0])//2, (shape[1]-template.shape[1])//2
        self.template[y:y+template.shape[0], x:x+template.shape[1]] = template
        self.reference, self.du, self.dphi = log_polar_magnitude(self.template)

    def propose(self, source):
        moving, _, _ = log_polar_magnitude(source)
        delta_phi, delta_u = phase_shift(self.reference, moving, pad_radial=True)
        alpha = -delta_u*self.du
        if not math.isfinite(alpha) or abs(alpha) > 3:
            return np.zeros((1, 4), np.float32)
        base = np.rad2deg(delta_phi*self.dphi)
        poses = []
        for angle in (base, base+180):
            transformed = transform_template(self.template, np.exp(alpha), angle)
            dy, dx = phase_shift(transformed, source)
            poses.append([dx, -dy, (angle+180)%360-180, alpha])
        return np.array(poses, np.float32)


class Registration:
    """ROI prior: translation +-12 px, any rotation, scale .75..1.25.

    Grid baseline uses the same final float NCC and 4-D local search. It is a
    search-work baseline, not an assertion that brute force is the fastest CPU.
    Fourier candidates outside the prior are discarded. Refinement is clamped.
    """
    lower = np.array([-12, -12, -np.inf, np.log(.75)])
    upper = np.array([12, 12, np.inf, np.log(1.25)])

    def __init__(self, job, source, template):
        self.method = job['method']
        self.source = source
        self.full = Scorer(job['build'], source, template, job['backend'], job['threads'])
        self.coarse = None
        self.fourier = None
        if self.method in ('grid', 'fourier_guarded'):
            # Gaussian scenes are smooth. This decimation is only a synthetic
            # experiment, NOT a general antialiased production pyramid.
            self.coarse = Scorer(job['build'], source[::2, ::2], template[::2, ::2], job['backend'], job['threads'])
            self.grid = np.array(list(itertools.product(np.arange(-12, 13, 4), np.arange(-12, 13, 4),
                                np.arange(-180, 180, 15), np.linspace(np.log(.75), np.log(1.25), 5))), np.float32)
        if self.method != 'grid':
            self.fourier = FourierProposal(template, source.shape)
        self.metrics = {}

    def close(self):
        self.full.close()
        if self.coarse:
            self.coarse.close()

    def evaluate(self, poses, coarse=False):
        scorer = self.coarse if coarse else self.full
        scores, metrics = scorer.score(poses)
        self.pose_count += len(poses)
        self.ncc_wall += metrics['wall_ms']
        if metrics['device_ms'] >= 0:
            self.gpu_ms += metrics['device_ms']
        self.metrics['coarse' if coarse else 'full'] = metrics
        index = int(np.argmax(scores))
        return poses[index].copy(), float(scores[index])

    def grid_proposal(self):
        poses = self.grid.copy()
        poses[:, :2] /= 2
        pose, score = self.evaluate(poses, coarse=True)
        pose[:2] *= 2
        return pose, score

    def run(self):
        self.pose_count, self.ncc_wall, self.gpu_ms = 0, 0., 0.
        begin = time.perf_counter()
        fallback, invalid = False, False
        proposals = []
        if self.method == 'grid':
            pose, score = self.grid_proposal()
        else:
            poses = self.fourier.propose(self.source)
            proposals = poses.tolist()
            valid = np.all((poses >= self.lower) & (poses <= self.upper), axis=1)
            invalid = not bool(np.any(valid))
            if invalid:
                pose, score = np.zeros(4, np.float32), -2.
            else:
                pose, score = self.evaluate(poses[valid])
            # Fixed policy, not selected after inspecting an individual result.
            if self.method == 'fourier_guarded' and (invalid or score < .85):
                pose, score = self.grid_proposal()
                fallback = True
        proposal_ms = (time.perf_counter()-begin)*1000
        start = time.perf_counter()
        if not invalid or fallback or self.method == 'grid':
            steps = np.array([2., 2., 7.5, (np.log(1.25)-np.log(.75))/8], np.float32)
            for _ in range(7):
                poses = np.clip(pose+OFFSETS*steps, self.lower, self.upper).astype(np.float32)
                pose, score = self.evaluate(poses)
                steps *= .5
        refine_ms = (time.perf_counter()-start)*1000
        algorithm_ms = (time.perf_counter()-begin)*1000
        memories = [self.full.memory()]+([self.coarse.memory()] if self.coarse else [])
        return dict(pose=pose.tolist(), score=score, accepted=score >= .8,
                    fourier_proposals=proposals,
                    fallback=fallback, proposal_outside_prior=invalid, evaluated_poses=self.pose_count,
                    algorithm_ms=algorithm_ms, proposal_ms=proposal_ms, refine_ms=refine_ms,
                    ncc_wall_ms=self.ncc_wall, gpu_command_ms=self.gpu_ms,
                    memory=dict(host_buffer_bytes=sum(m['host_buffer_bytes'] for m in memories),
                                gpu_buffer_bytes=sum(m['gpu_buffer_bytes'] for m in memories),
                                gpu_allocated_bytes=max((m['gpu_allocated_bytes'] for m in memories), default=0)))


def accuracy(result, truths):
    tx, ty, angle, alpha = result['pose']
    errors = []
    for truth in truths:
        errors.append(dict(position_px=float(np.linalg.norm(np.array([tx, ty])-truth['translation'])),
                           angle_deg=abs((angle-truth['angle']+180)%360-180),
                           relative_scale=abs(np.exp(alpha)/truth['scale']-1)))
    matched = result['accepted'] and any(e['position_px'] <= 1 and e['angle_deg'] <= 1 and
                                       e['relative_scale'] <= .01 for e in errors)
    tp = int(matched)
    return dict(errors_to_each_truth=errors, tp=tp, fp=int(result['accepted'])-tp, fn=len(truths)-tp,
                truth_count=len(truths))


def registration_worker(job):
    source, template, truths = registration_data(job['case'])
    before = peak_rss()
    start = time.perf_counter()
    matcher = Registration(job, source, template)
    setup = (time.perf_counter()-start)*1000
    try:
        first = matcher.run()
        for _ in range(job['warmups']-1):
            matcher.run()
        warmed = time.perf_counter()
        extra_warmups = 0
        while time.perf_counter()-warmed < job['warmup_ms']/1000:
            matcher.run()
            extra_warmups += 1
        runs = [matcher.run() for _ in range(job['repeats'])]
        result = dict(device=matcher.full.device, setup_ms=setup, first_call=first, extra_warmups=extra_warmups, runs=runs,
                      algorithm=summary([r['algorithm_ms'] for r in runs]),
                      proposal=summary([r['proposal_ms'] for r in runs]),
                      refine=summary([r['refine_ms'] for r in runs]),
                      memory=dict(process_peak_before_setup_bytes=before, process_peak_bytes=peak_rss(),
                                  **{k: max(r['memory'][k] for r in runs) for k in runs[0]['memory']}),
                      shapes=dict(source=list(source.shape), template=list(template.shape)),
                      truths=truths, accuracy=accuracy(runs[-1], truths))
        return result
    finally:
        matcher.close()


def jobs(args):
    backends = [('cpu', 1), ('cpu', args.threads)] + ([] if args.no_metal else [('metal', 1)])
    tasks = []
    if args.suite in ('all', 'micro'):
        cases = [dict(name=f'{size}_{"rgb" if rgb else "gray"}_{count}', size=size, rgb=rgb, count=count)
                 for size, rgb, count in [(64, False, 1), (64, False, 49), (64, False, 512),
                                          (128, False, 49), (128, False, 512), (128, False, 4096),
                                          (256, False, 512), (128, True, 512)]]
        if not args.no_fixture:
            cases.append(dict(name='src7_local_162', fixture=True))
        for case in cases:
            for backend, threads in backends:
                tasks.append(dict(suite='micro', case=case, backend=backend, threads=threads))
    if args.suite in ('all', 'registration'):
        cases = []
        for rgb in (False, True):
            for i, (scale, angle, translation) in enumerate(((1, 0, [7, -9]), (.8, 31, [8, -6]),
                                                            (1.2, -47, [-5, 7]), (1.1, 143, [4, 3]))):
                cases.append(dict(name=f'clean_{"rgb" if rgb else "gray"}_{i}', rgb=rgb,
                                  truth=dict(scale=scale, angle=angle, translation=translation)))
        cases += [dict(name=f'stress_{stress}', stress=stress) for stress in ('background', 'occlusion', 'two_objects', 'negative')]
        cases += [dict(name=f'cropped_{"rgb" if rgb else "gray"}', template_size=128, rgb=rgb)
                  for rgb in (False, True)]
        for case in cases:
            for method in ('grid', 'fourier', 'fourier_guarded'):
                for backend, threads in backends[1:]:
                    tasks.append(dict(suite='registration', case=case, method=method, backend=backend, threads=threads))
    # Fixed random order reduces a systematic CPU-first / GPU-last thermal bias.
    np.random.default_rng(2718).shuffle(tasks)
    for task in tasks:
        task.update(build=str(args.build.resolve()), repeats=args.repeats, warmups=args.warmups, warmup_ms=args.warmup_ms)
    return tasks


def source_hashes():
    files = list((ROOT / 'cpp-core/experiments').glob('*'))
    files += [Path(__file__).resolve(), ROOT / 'cpp-core/tools/pose_experiment.py',
              ROOT / 'cpp-core/tools/research_ncc_fourier.py', ROOT / 'cpp-core/tools/validate_pose_experiment.py',
              ROOT / 'cpp-core/CMakeLists.txt', ROOT / 'cpp-core/tests/pose_experiment_test.cpp']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=ROOT / 'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'cpp-core/docs/pose-experiment-benchmark.json')
    parser.add_argument('--suite', choices=('micro', 'registration', 'all'), default='all')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--warmup-ms', type=float, default=200,
                        help='Additional per-variant warm-up time to reduce short-dispatch GPU clock transients')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--no-metal', action='store_true')
    parser.add_argument('--no-fixture', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        result = micro_worker(job) if job['suite'] == 'micro' else registration_worker(job)
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if args.repeats < 1 or args.warmups < 1 or args.threads < 2 or args.warmup_ms < 0:
        parser.error('repeats/warmups >= 1; threads >= 2; warmup-ms >= 0')
    tasks = jobs(args)
    report = dict(date='2026-10-12', platform=platform.platform(), machine=platform.machine(),
                  logical_cpus=os.cpu_count(), numpy=np.__version__, source_sha256=source_hashes(),
                  cpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                      if sys.platform == 'darwin' else platform.processor(),
                  physical_memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
                      if sys.platform == 'darwin' else None,
                  binary_sha256=hashlib.sha256(library_path(args.build).read_bytes()).hexdigest(),
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                  input_sha256={name: hashlib.sha256((ROOT / 'Test Images' / name).read_bytes()).hexdigest()
                                for name in ('Src7.bmp', 'Dst7.bmp')} if not args.no_fixture else {},
                  protocol=dict(fresh_process_per_variant=True, serial=True, seeded_shuffled_order=True,
                                repeats=args.repeats, warmups=args.warmups, additional_warmup_ms=args.warmup_ms,
                                timer='steady/perf_counter wall; Metal GPUStartTime..GPUEndTime after completion',
                                timing_excludes='I/O, fixture/synthetic generation, handle setup and cached template preparation',
                                algorithm_wall_includes='FFT proposal, pose generation, C ABI, packing, growth, GPU submission/wait/readback, selection/refinement',
                                memory='process RSS high-water includes Python/NumPy/driver/setup; explicit owned buffer bytes are separate; Metal allocations share RAM and MUST NOT be added to RSS',
                                scope='micro: fixed pose batches; registration: supplied single ROI, NOT full-image detection',
                                accuracy='accept NCC >= .8, TP requires <=1px, <=1deg, <=1% relative scale; unmatched accepted output is FP',
                                fft='NumPy CPU FFT external backend; no GPU FFT implementation',
                                cuda='not implemented or benchmarked; no CUDA device on this host'), runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pose-experiment-', dir=ROOT / 'build') as temporary:
        path = Path(temporary)
        for i, job in enumerate(tasks):
            (path / 'request.json').write_text(json.dumps(job))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path / 'request.json'),
                            '--output', str(path / 'result.json')], check=True, timeout=300)
            result = json.loads((path / 'result.json').read_text())
            report['runs'].append(result)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            data = result['result']
            timing = data.get('algorithm', data.get('wall'))['median_ms']
            print(f"{i+1}/{len(tasks)} {job['case']['name']} {job.get('method', '')} {job['backend']}{job['threads']}: {timing:.3f} ms", flush=True)
    # Identical workloads: compare every CPU/GPU score, not just winning index.
    consistency = []
    for name in sorted({r['job']['case']['name'] for r in report['runs'] if r['job']['suite'] == 'micro'}):
        variants = [r for r in report['runs'] if r['job']['suite'] == 'micro' and r['job']['case']['name'] == name]
        baseline = next(r['result'] for r in variants if r['job']['backend'] == 'cpu' and r['job']['threads'] == 1)
        for variant in variants:
            maximum = float(np.max(np.abs(np.array(variant['result']['scores'])-baseline['scores'])))
            assert maximum < 3e-5, (name, maximum)
            assert variant['result']['best_index'] == baseline['best_index'], name
            consistency.append(dict(case=name, backend=variant['job']['backend'], threads=variant['job']['threads'], max_error=maximum))
    report['micro_consistency'] = consistency
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output)


if __name__ == '__main__':
    main()
