"""Large resident pose batches and held-out LOCAL refinement accuracy.

This is not a full-image detector benchmark. Candidate arrays and perturbed
registration seeds are supplied explicitly. Generation, setup, and independent
verification are outside the timed interval. Workers run serially in fresh
processes; score vectors are retained as compressed artifacts for audit.
"""
import argparse
from datetime import datetime
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

ROOT = Path(__file__).resolve().parents[2]
OFFSETS = np.asarray(list(itertools.product((-1, 0, 1), repeat=4)), np.float32)
SCORE_TOLERANCE = 3e-5


def peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except ImportError:
        return None


def summary(values):
    return dict(samples_ms=values, median_ms=statistics.median(values),
                p95_ms=float(np.percentile(values, 95)), min_ms=min(values), max_ms=max(values))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def field(x, y, size, channels, seed):
    """Continuous texture; not a discrete template warped with the tested code."""
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(channels):
        value = np.zeros(np.broadcast_shapes(x.shape, y.shape), np.float64)
        for _ in range(12):
            direction, phase = rng.uniform(-math.pi, math.pi, 2)
            frequency = rng.uniform(1.3, 7.0) * 2 * math.pi / size
            value += np.sin(frequency * (math.cos(direction) * x + math.sin(direction) * y) + phase)
        for _ in range(4):
            cx, cy = rng.uniform(-.35, .35, 2) * size
            radius = rng.uniform(.05, .16) * size
            value += 2 * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*radius**2))
        output.append(.4 + value * .025)
    return np.stack(output, axis=-1).astype(np.float32)


def fixture(case):
    size, channels = case['size'], case['channels']
    height, width = case['source_shape']
    truth = np.asarray(case['truth'], np.float64)
    theta, scale = np.deg2rad(truth[2]), np.exp(truth[3])
    x = np.arange(size, dtype=np.float64)[None, :] - size/2
    y = size/2 - np.arange(size, dtype=np.float64)[:, None]
    template = field(x, y, size, channels, case['seed'])
    source = np.empty((height, width, channels), np.float32)
    x = np.arange(width, dtype=np.float64)[None, :] - width/2 - truth[0]
    # Row blocks bound temporary generation memory, including 4K RGB fixtures.
    for first in range(0, height, 128):
        y = height/2 - np.arange(first, min(height, first+128), dtype=np.float64)[:, None] - truth[1]
        u = (math.cos(theta)*x + math.sin(theta)*y) / scale
        v = (-math.sin(theta)*x + math.cos(theta)*y) / scale
        source[first:first+128] = field(u, v, size, channels, case['seed'])
    if case.get('appearance') == 'gain_bias_noise':
        source *= .8
        source += .11
        rng = np.random.default_rng(case['seed'] + 29)
        for first in range(0, height, 128):
            block = source[first:first+128]
            block += rng.normal(0, .0005, block.shape).astype(np.float32)
    if channels == 1:
        source, template = source[:, :, 0], template[:, :, 0]
    return np.ascontiguousarray(source), np.ascontiguousarray(template)


def candidate_poses(case):
    rng = np.random.default_rng(case['seed'] + 101)
    count = case['count']
    height, width = case['source_shape']
    poses = rng.uniform([-width/2, -height/2, -180, math.log(.7)],
                        [width/2, height/2, 180, math.log(1.4)], (count, 4)).astype(np.float32)
    near = max(1, count//5)
    poses[:near] = np.asarray(case['truth']) + rng.uniform(
        [-2, -2, -2, -.025], [2, 2, 2, .025], (near, 4))
    poses[0] = case['truth']
    if count >= 8:
        poses[-3:] = [[width/2, height/2, 19, 0], [1e5, 1e5, 0, 0], [-1e5, 0, 0, 0]]
    return poses


def reference_ncc(source, template, pose):
    """Independent float64 NCC oracle at a fixed pose, including zero extension."""
    if source.ndim == 2:
        source, template = source[:, :, None], template[:, :, None]
    height, width, _ = source.shape
    th, tw, _ = template.shape
    yy, xx = np.indices((th, tw), dtype=np.float64)
    u, v = xx - tw/2, th/2 - yy
    theta, scale = np.deg2rad(float(pose[2])), np.exp(float(pose[3]))
    x = width/2 + float(pose[0]) + scale*(np.cos(theta)*u-np.sin(theta)*v)
    y = height/2 - float(pose[1]) - scale*(np.sin(theta)*u+np.cos(theta)*v)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x-ix, y-iy
    values = np.zeros(template.shape, np.float64)
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        valid = (ix+dx >= 0) & (ix+dx < width) & (iy+dy >= 0) & (iy+dy < height)
        values[valid] += source[iy[valid]+dy, ix[valid]+dx] * weight[valid, None]
    left, right = values.reshape(-1, source.shape[2]), template.astype(np.float64).reshape(-1, source.shape[2])
    left -= left.mean(axis=0)
    right -= right.mean(axis=0)
    a, b = np.sum(left*left), np.sum(right*right)
    if min(a, b)/(th*tw) <= 1e-12:
        return -2.
    return float(np.sum(left*right)/np.sqrt(a*b))


def warm_and_measure(call, job):
    start = time.perf_counter()
    first = call()
    first_wall = (time.perf_counter()-start)*1000
    for _ in range(job['warmups']-1):
        call()
    warm_start = time.perf_counter()
    extra = 0
    while time.perf_counter()-warm_start < job['warmup_ms']/1000:
        call()
        extra += 1
    records, times = [], []
    for _ in range(job['repeats']):
        start = time.perf_counter()
        records.append(call())
        times.append((time.perf_counter()-start)*1000)
    return records, times, dict(first_call_wall_ms=first_wall, extra_warmups=extra)


def batch_worker(job):
    case = job['case']
    source, template = fixture(case)
    poses = candidate_poses(case)
    before = peak_rss()
    with Scorer(job['build'], source, template, job['backend'], job['threads']) as scorer:
        initial_memory = scorer.memory()
        records, times, warm = warm_and_measure(lambda: scorer.score(poses), job)
        scores = records[-1][0]
        metrics = [record[1] for record in records]
        # Capture high-water before the independent oracle / split validation.
        memory = dict(process_peak_before_setup_bytes=before, process_peak_bytes=peak_rss(),
                      **{key: max(m[key] for m in metrics) for key in initial_memory})
        auxiliary_bytes = memory['gpu_buffer_bytes']-initial_memory['gpu_buffer_bytes']
        per_pose_auxiliary = 64*math.ceil(case['size']**2/1024)+20
        auxiliary_limit = max(16*1024*1024, per_pose_auxiliary)
        bounded = auxiliary_bytes <= auxiliary_limit
        split_error = None
        if job['split_count']:
            split = np.concatenate([scorer.score(poses[i:i+job['split_count']])[0]
                                    for i in range(0, len(poses), job['split_count'])])
            split_error = float(np.max(np.abs(scores-split)))
        indices = sorted(set([0, min(1, len(poses)-1), len(poses)//2, len(poses)-1,
                              max(0, len(poses)-3)]))
        oracle = [dict(index=i, reference=reference_ncc(source, template, poses[i]), actual=float(scores[i]))
                  for i in indices]
        max_oracle_error = max(abs(v['reference']-v['actual']) for v in oracle)
        destination = Path(job['score_file'])
        np.savez_compressed(destination, poses=poses, scores=scores)
        return dict(device=scorer.device, setup_ms=scorer.setup_ms, **warm,
                    call_wall=summary(times), c_api_wall=summary([m['wall_ms'] for m in metrics]),
                    gpu_command=summary([m['device_ms'] for m in metrics]) if job['backend'] == 'metal' else None,
                    candidates_per_second=len(poses)/(statistics.median(times)/1000),
                    sampled_channel_values_per_second=len(poses)*template.size/(statistics.median(times)/1000),
                    memory=memory, initial_memory=initial_memory,
                    retained_auxiliary_buffer_bytes=auxiliary_bytes,
                    auxiliary_budget_bytes=auxiliary_limit, auxiliary_budget_passed=bounded,
                    naive_metal_auxiliary_bytes=len(poses)*per_pose_auxiliary,
                    score_file=destination.name, score_file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    input_sha256=dict(source=digest(source), template=digest(template), poses=digest(poses)),
                    accepted_count=int(np.sum(scores >= .8)), best_index=int(np.argmax(scores)),
                    split_max_score_error=split_error, oracle=oracle, oracle_max_score_error=max_oracle_error,
                    passed=bounded and max_oracle_error <= SCORE_TOLERANCE and
                           (split_error is None or split_error <= SCORE_TOLERANCE))


def grid_refine(scorer, seeds):
    # Historical real-image experiment: its limited angular travel is retained
    # deliberately as the measured baseline, rather than silently improved here.
    poses = seeds.copy()
    steps = np.asarray([2, 2, 1, math.log(1.12)], np.float32)
    evaluations = 0
    for _ in range(6):
        candidates = (poses[:, None, :] + OFFSETS[None, :, :]*steps).reshape(-1, 4)
        scores, _ = scorer.score(candidates)
        scores = scores.reshape(len(poses), len(OFFSETS))
        indices = np.argmax(scores, axis=1)
        poses = candidates.reshape(len(poses), len(OFFSETS), 4)[np.arange(len(poses)), indices]
        best_scores = scores[np.arange(len(poses)), indices]
        evaluations += len(candidates)
        steps *= .5
    return poses, best_scores, [dict(evaluations=evaluations//len(poses)) for _ in poses], {}


def accuracy_worker(job):
    source, template = fixture(job['case'])
    seeds = np.asarray(job['case']['initial_poses'], np.float32)
    truth = np.asarray(job['case']['truth'], np.float64)
    before = peak_rss()
    with Scorer(job['build'], source, template, job['backend'], job['threads']) as scorer:
        if job['method'] == 'lm' and not hasattr(scorer, 'refine'):
            return dict(status='unavailable', reason='Scorer.refine binding is unavailable')
        def call():
            if job['method'] == 'grid':
                return grid_refine(scorer, seeds)
            return scorer.refine(seeds, max_iterations=40, max_translation=12,
                                 max_angle_degrees=15, max_log_scale=math.log(1.3))
        records, times, warm = warm_and_measure(call, job)
        poses, scores, stats, metrics = records[-1]
        final_peak = peak_rss()
        initial_scores, _ = scorer.score(seeds)
        rechecked, _ = scorer.score(poses)
        errors = []
        for pose, score, initial in zip(poses, scores, initial_scores):
            position = float(np.linalg.norm(pose[:2]-truth[:2]))
            angle = abs(float((pose[2]-truth[2]+180)%360-180))
            scale = abs(float(np.exp(pose[3]-truth[3])-1))
            errors.append(dict(position_px=position, angle_deg=angle, relative_scale=scale,
                               score=float(score), initial_score=float(initial),
                               precise_pass=bool(score >= .8 and position <= .5 and angle <= .5 and scale <= .01),
                               existing_gate_pass=bool(score >= .8 and position <= 2 and angle <= 1 and scale <= .02)))
        return dict(device=scorer.device, solver_device='CPU' if job['method'] == 'lm' else scorer.device,
                    setup_ms=scorer.setup_ms, **warm, call_wall=summary(times),
                    initial_poses=seeds.tolist(), poses=poses.tolist(), scores=scores.tolist(), stats=stats,
                    metrics=metrics, errors=errors, precise_pass_count=sum(e['precise_pass'] for e in errors),
                    existing_gate_pass_count=sum(e['existing_gate_pass'] for e in errors),
                    seed_count=len(seeds), monotonic=bool(np.all(scores >= initial_scores-SCORE_TOLERANCE)),
                    score_recheck_error=float(np.max(np.abs(scores-rechecked))),
                    memory=dict(process_peak_before_setup_bytes=before, process_peak_bytes=final_peak,
                                **scorer.memory()),
                    input_sha256=dict(source=digest(source), template=digest(template), seeds=digest(seeds)),
                    passed=bool(np.all(scores >= initial_scores-SCORE_TOLERANCE) and
                                np.max(np.abs(scores-rechecked)) <= SCORE_TOLERANCE))


def cases(args):
    if args.suite in ('all', 'batch'):
        specs = [(64, 1, 1, [768, 1024]), (64, 3, 64, [768, 1024]),
                 (256, 1, 8192, [1080, 1920])]
        if args.profile in ('standard', 'large'):
            specs += [(128, 1, 1024, [2160, 3840]), (128, 3, 1024, [2160, 3840]),
                      (256, 3, 8192, [2160, 3840]), (768, 1, 1024, [2160, 3840])]
        if args.profile == 'large':
            specs += [(256, 3, 65536, [2160, 3840]), (1024, 3, 2048, [2160, 3840])]
        for size, channels, count, shape in specs:
            yield 'batch', dict(name=f'{size}_c{channels}_n{count}', size=size, channels=channels,
                                count=count, source_shape=shape, seed=90210+size+channels,
                                truth=[17.25, -11.625, 37.35, math.log(.93)])
    if args.suite in ('all', 'accuracy'):
        rng = np.random.default_rng(8675309)
        count = {'smoke': 4, 'standard': 12, 'large': 24}[args.profile]
        for i in range(count):
            truth = rng.uniform([-18, -18, -175, math.log(.76)], [18, 18, 175, math.log(1.24)])
            # These seed-error distributions are a declared local test prior,
            # not a detection result, and remain identical for every method.
            errors = np.asarray([[.65, -.7, .6, .01], [1.3, -1.7, 2.5, -.035], [-3.2, 2.4, -5.5, .08],
                                 [5.0, -4.0, 8.0, -.12]], np.float64)
            errors *= rng.choice([-1, 1], size=(1, 4))
            yield 'accuracy', dict(name=f'heldout_{i:02d}', size=96 if i%3 else 128,
                                   channels=1 if i%2 == 0 else 3, source_shape=[384, 448],
                                   seed=49001+i*131, truth=truth.tolist(),
                                   initial_poses=(truth+errors).tolist(),
                                   appearance='gain_bias_noise' if i%3 == 2 else 'clean')


def metadata(args):
    files = list((ROOT/'cpp-core/experiments').glob('*'))
    files += [Path(__file__).resolve(), ROOT/'cpp-core/tools/pose_experiment.py']
    result = dict(created_at=datetime.now().astimezone().isoformat(), platform=platform.platform(),
                  machine=platform.machine(), logical_cpus=os.cpu_count(), numpy=np.__version__, python=sys.version,
                  binary_sha256=hashlib.sha256(library_path(args.build).read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files if p.is_file()},
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip())
    if sys.platform == 'darwin':
        result['cpu'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        result['physical_memory_bytes'] = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    return result


def consistency(report, artifacts):
    checks = []
    batches = [r for r in report['runs'] if r['job']['suite'] == 'batch' and r['result'].get('status') != 'unavailable']
    for name in sorted({r['job']['case']['name'] for r in batches}):
        variants = [r for r in batches if r['job']['case']['name'] == name]
        baseline = next(r for r in variants if r['job']['backend'] == 'cpu' and r['job']['threads'] == 1)
        with np.load(artifacts/baseline['result']['score_file']) as saved:
            reference, poses = saved['scores'], saved['poses']
        for variant in variants:
            with np.load(artifacts/variant['result']['score_file']) as saved:
                actual = saved['scores']
                same_poses = np.array_equal(poses, saved['poses'])
            maximum = float(np.max(np.abs(actual-reference)))
            same_inputs = variant['result']['input_sha256'] == baseline['result']['input_sha256']
            stable = np.abs(reference-.8) > SCORE_TOLERANCE
            disagreements = int(np.sum(((actual >= .8) != (reference >= .8)) & stable))
            loss = float(reference.max()-reference[np.argmax(actual)])
            checks.append(dict(case=name, backend=variant['job']['backend'], threads=variant['job']['threads'],
                               compared_scores=len(actual), max_score_error=maximum, identical_poses=same_poses,
                               identical_inputs=same_inputs,
                               threshold_disagreements_away_from_boundary=disagreements,
                               threshold_boundary_count=int(np.sum(~stable)), tie_aware_best_score_loss=loss,
                               passed=bool(same_inputs and same_poses and maximum <= SCORE_TOLERANCE and
                                           disagreements == 0 and loss <= SCORE_TOLERANCE)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT/'build/pose-stress/report.json')
    parser.add_argument('--suite', choices=('all', 'batch', 'accuracy'), default='all')
    parser.add_argument('--profile', choices=('smoke', 'standard', 'large'), default='smoke')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--warmup-ms', type=float, default=100)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--no-metal', action='store_true')
    parser.add_argument('--split-count', type=int, default=257, help='Untimed split-call parity chunk; 0 disables')
    parser.add_argument('--timeout', type=float, default=900, help='Per-worker timeout in seconds')
    parser.add_argument('--case', help='Run only cases whose names contain this text')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            result = batch_worker(job) if job['suite'] == 'batch' else accuracy_worker(job)
        except RuntimeError as error:
            if job['backend'] == 'metal' and str(error) in ('No Metal GPU available', 'Metal backend was not compiled on this platform'):
                result = dict(status='unavailable', reason=str(error))
            else:
                raise
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if min(args.repeats, args.warmups) < 1 or args.threads < 2 or min(args.warmup_ms, args.split_count) < 0:
        parser.error('repeats/warmups >= 1; threads >= 2; warmup-ms/split-count >= 0')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.output.parent/(args.output.stem+'-artifacts')
    artifacts.mkdir(exist_ok=True)
    tasks = []
    for suite, case in cases(args):
        if args.case and args.case not in case['name']:
            continue
        variants = [('cpu', 1), ('cpu', args.threads)]
        if not args.no_metal:
            variants.append(('metal', 1))
        for backend, threads in variants:
            for method in (('grid', 'lm') if suite == 'accuracy' else ('score',)):
                # LM currently executes on CPU even for a Metal handle. Avoid
                # relabeling that as GPU refinement or duplicating the workload.
                if method == 'lm' and backend == 'metal':
                    continue
                tasks.append(dict(suite=suite, case=case, method=method, backend=backend, threads=threads,
                                  build=str(args.build.resolve()), repeats=args.repeats, warmups=args.warmups,
                                  warmup_ms=args.warmup_ms, split_count=args.split_count,
                                  score_file=str(artifacts/f'{case["name"]}-{backend}{threads}.npz')))
    if not tasks:
        parser.error('No cases selected')
    np.random.default_rng(64051).shuffle(tasks)
    report = dict(**metadata(args), profile=args.profile, protocol=dict(
        serial=True, fresh_process_per_variant=True, shuffled_order_seed=64051,
        repeats=args.repeats, warmups=args.warmups, warmup_ms=args.warmup_ms,
        scope='Supplied fixed pose batches and local pose refinement; NOT full-image detection or recall.',
        synthetic='Independent continuous harmonic/Gaussian field; RGB channels differ; no reuse of scorer resampling.',
        batch_candidates='20% near truth, remaining random image positions/angles/scales; includes truth and outside-image cases.',
        local_seeds='Truth-perturbed seeds from declared error prior; test convergence basin only; no detector claims.',
        timing='Outer call wall includes Python/C ABI input conversion, NCC/refinement, submission/wait/output; excludes fixture/setup/verification.',
        memory='Fresh-process peak RSS includes Python, NumPy, fixture generation and setup. Metal allocations share RAM; do not add them to RSS. Reported retained buffers do not measure allocator transients.',
        correctness='All score vectors compared to CPU1, threshold 0.8, numerical tolerance 3e-5; best index is tie-aware. Independent float64 oracle checks selected poses.',
        accuracy='NCC>=0.8 plus precise <=0.5px/0.5deg/1%; existing gate <=2px/1deg/2%. Local success is separate from correctness.',
        precision='p95 is the empirical percentile of reported samples, not a confidence interval.',
        cuda='Not implemented or measured by this harness.'), score_artifact_directory=artifacts.name, runs=[])
    with tempfile.TemporaryDirectory(prefix='pose-stress-') as directory:
        temporary = Path(directory)
        for index, job in enumerate(tasks):
            request, response = temporary/'request.json', temporary/'result.json'
            request.write_text(json.dumps(job))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request),
                            '--output', str(response)], check=True, timeout=args.timeout)
            entry = json.loads(response.read_text())
            report['runs'].append(entry)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            result = entry['result']
            timing = f"{result['call_wall']['median_ms']:.3f} ms" if 'call_wall' in result else result['reason']
            print(f"{index+1}/{len(tasks)} {job['case']['name']} {job['method']} {job['backend']}{job['threads']}: {timing}", flush=True)
    report['consistency'] = consistency(report, artifacts)
    available = [r for r in report['runs'] if r['result'].get('status') != 'unavailable']
    report['correctness_passed'] = all(r['result']['passed'] for r in available) and all(c['passed'] for c in report['consistency'])
    report['local_accuracy'] = []
    for method, backend, threads in sorted({(r['job']['method'], r['job']['backend'], r['job']['threads'])
                                           for r in available if r['job']['suite'] == 'accuracy'}):
        selected = [r['result'] for r in available if (r['job']['method'], r['job']['backend'], r['job']['threads']) == (method, backend, threads)]
        report['local_accuracy'].append(dict(method=method, backend=backend, threads=threads,
            seeds=sum(r['seed_count'] for r in selected), precise_pass=sum(r['precise_pass_count'] for r in selected),
            existing_gate_pass=sum(r['existing_gate_pass_count'] for r in selected)))
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output, flush=True)
    if not report['correctness_passed']:
        raise SystemExit('Score/refinement consistency failed; see report')


if __name__ == '__main__':
    main()
