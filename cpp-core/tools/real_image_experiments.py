"""Run full-image CPU search versus CPU-proposal / CPU-or-Metal refinement.

All candidates are found online from the input image. No saved detections or
ground-truth positions are supplied to the search. Truth is used ONLY after
search on the controlled real-texture scene. Original photos have no calibrated
pose labels, so their comparison is agreement, not an accuracy percentage.
"""
import argparse
import ctypes as C
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from benchmark_pose_experiments import FourierProposal, peak_rss, summary
from pose_experiment import Scorer, library_path
from research_ncc_fourier import bilinear

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cpp-core/examples'))
from python_ctypes import Image, Library

METHODS = {
    'cpu_fixed': 'Current C++ CPU / fixed scale',
    'cpu_fast': 'Current C++ CPU / FastMode',
    'cpu_grid': 'Current C++ CPU / 3-scale bank',
    'hybrid_cpu': 'CPU coarse + CPU 8-thread refinement',
    'hybrid_metal': 'CPU coarse + Metal refinement',
    'fourier_metal': 'CPU coarse + Fourier proposals + Metal',
    'continuous_cpu': 'CPU coarse + CPU proposals + continuous CPU LM',
    'continuous_metal': 'CPU coarse + Metal proposals + continuous CPU LM',
    'continuous_direct_cpu': 'CPU coarse + continuous CPU LM (no extra proposal bank)',
}


def image_view(image):
    assert image.dtype == np.uint8 and image.ndim == 2 and image.flags.c_contiguous
    return Image(image.ctypes.data_as(C.POINTER(C.c_uint8)), image.shape[1], image.shape[0], image.strides[0])


def angle_distance(a, b):
    return abs((a-b+180) % 360-180)


def corners(center, angle, scale, shape):
    h, w = shape
    theta = math.radians(angle)
    matrix = scale*np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    return (np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]) @ matrix.T + center).tolist()


def cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def area(points):
    return abs(sum(cross(points[0], points[i], points[i+1]) for i in range(1, len(points)-1)))/2 if len(points) >= 3 else 0.


def overlap(a, b):
    """Convex clipping, generalized denominator = smaller rectangle area."""
    p, q = a['corners'], b['corners']
    if (max(x[0] for x in p) <= min(x[0] for x in q) or max(x[0] for x in q) <= min(x[0] for x in p)
            or max(x[1] for x in p) <= min(x[1] for x in q) or max(x[1] for x in q) <= min(x[1] for x in p)):
        return 0.
    denominator = min(area(p), area(q))
    if denominator <= 0:
        return 0.
    for edge in range(4):
        if not p:
            break
        v, w = q[edge], q[(edge+1) % 4]
        output = []
        previous = p[-1]
        previous_side = cross(v, w, previous)
        for current in p:
            side = cross(v, w, current)
            if (side >= 0) != (previous_side >= 0):
                t = previous_side/(previous_side-side)
                output.append([previous[j]+t*(current[j]-previous[j]) for j in range(2)])
            if side >= 0:
                output.append(current)
            previous, previous_side = current, side
        p = output
    return min(1., max(0., area(p)/denominator))


def suppress(matches, limit, maximum_overlap, threshold):
    kept = []
    for match in sorted(matches, key=lambda m: (-m['score'], m['pos_y'], m['pos_x'], m['angle'])):
        if match['score'] < threshold:
            continue
        if all(overlap(previous, match) <= maximum_overlap+1e-10 for previous in kept):
            kept.append(match)
            if len(kept) >= limit:
                break
    return kept


def scaled_template(template, scale):
    if scale == 1:
        return template.copy()
    h, w = template.shape
    nh, nw = round(h*scale), round(w*scale)
    yy, xx = np.indices((nh, nw), dtype=float)
    # Explicit center convention shared with the scorer. No OpenCV matching or
    # geometric transform is used by the search. This small bank is bilinear,
    # not a general antialiased multi-scale pyramid.
    sampled = bilinear(template, w/2+(xx-nw/2)/scale, h/2+(yy-nh/2)/scale)
    return np.ascontiguousarray(np.clip(np.floor(sampled+.5), 0, 255), np.uint8)


def controlled_scene(path):
    """Independent OpenCV warp of real photographed capacitor texture."""
    template = cv2.imread(str(ROOT/'Test Images/Dst3.bmp'), cv2.IMREAD_GRAYSCALE)
    canvas = np.full((740, 1100), 150., np.float32)
    truths = []
    values = [((180, 180), -35, .75), ((550, 180), 12, .9), ((920, 180), 83, 1.),
              ((180, 550), 143, 1.1), ((550, 550), -102, 1.25), ((920, 550), 46, .85)]
    for index, (center, angle, scale) in enumerate(values):
        theta = np.deg2rad(angle)
        linear = scale*np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        matrix = np.column_stack((linear, np.array(center)-linear @ [template.shape[1]/2, template.shape[0]/2]))
        patch = cv2.warpAffine(template.astype(np.float32), matrix, (1100, 740), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ones(template.shape, np.float32), matrix, (1100, 740), flags=cv2.INTER_LINEAR)
        canvas = canvas*(1-mask)+patch
        truths.append(dict(index=index, pos_x=center[0], pos_y=center[1], angle=angle, scale=scale,
                           corners=corners(center, angle, scale, template.shape)))
    canvas += np.random.default_rng(20261012).normal(0, 1, canvas.shape)
    cv2.imwrite(str(path), np.clip(np.rint(canvas), 0, 255).astype(np.uint8))
    return truths


def held_out_scene(path):
    """Predetermined held-out full-search scene; poses never enter Search."""
    template = cv2.imread(str(ROOT/'Test Images/Dst3.bmp'), cv2.IMREAD_GRAYSCALE)
    canvas = np.full((1200, 1800), 150., np.float32)
    rng = np.random.default_rng(721106)
    truths = []
    for row in range(3):
        for column in range(4):
            center = np.array([225.+450*column, 200.+400*row]) + rng.uniform(-35, 35, 2)
            angle, scale = float(rng.uniform(-180, 180)), float(np.exp(rng.uniform(math.log(.72), math.log(1.28))))
            theta = math.radians(angle)
            linear = scale*np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
            matrix = np.column_stack((linear, center-linear @ [template.shape[1]/2, template.shape[0]/2]))
            gain, bias = rng.uniform(.85, 1.1), rng.uniform(-12, 12)
            patch = cv2.warpAffine(template.astype(np.float32)*gain+bias, matrix, (1800, 1200), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(np.ones(template.shape, np.float32), matrix, (1800, 1200), flags=cv2.INTER_LINEAR)
            canvas = canvas*(1-mask)+patch
            truths.append(dict(index=len(truths), pos_x=float(center[0]), pos_y=float(center[1]),
                               angle=angle, scale=scale, corners=corners(center, angle, scale, template.shape)))
    canvas += rng.normal(0, 2, canvas.shape)
    cv2.imwrite(str(path), np.clip(np.rint(canvas), 0, 255).astype(np.uint8))
    return truths


class Search:
    def __init__(self, job):
        self.job = job
        case = job['case']
        self.source = cv2.imread(case['source'], cv2.IMREAD_GRAYSCALE)
        self.template = cv2.imread(case['template'], cv2.IMREAD_GRAYSCALE)
        if self.source is None or self.template is None:
            raise RuntimeError('Missing image')
        self.shape = self.template.shape
        self.source_float = None
        self.method = job['method']
        self.continuous = self.method.startswith('continuous')
        self.hybrid = self.method.startswith('hybrid') or self.method == 'fourier_metal' or self.continuous
        self.scales = [1.] if self.method == 'cpu_fixed' else case['scales']
        self.allow_scale = len(self.scales) > 1 and self.hybrid
        self.rotation = case['angle'] != 0
        self.scorer = None
        self.patterns = []
        start = time.perf_counter()
        path = Path(job['build']) / ('libfpm_c.dylib' if sys.platform == 'darwin' else 'libfpm_c.so')
        self.library = Library(str(path))
        for scale in self.scales:
            scaled = scaled_template(self.template, scale)
            # fpm_image is a borrowed C pointer; keep the NumPy owner alive
            # through the synchronous learn call, which copies the pixels.
            self.patterns.append((scale, self.library.learn(image_view(scaled), 256)))
        self.config = self.library.config()
        self.config.max_targets = case['limit']
        self.config.min_score = case['proposal_score']
        self.config.max_overlap = case['overlap']
        self.config.tolerance_angle = case['angle']
        self.config.stop_layer = 1 if self.hybrid or self.method == 'cpu_fast' else 0
        self.config.subpixel = 0
        if self.hybrid:
            self.source_float = self.source.astype(np.float32)/255
            self.scorer = Scorer(job['build'], self.source_float, self.template.astype(np.float32)/255,
                                 'cpu' if self.method.endswith('_cpu') else 'metal', 8)
        self.fourier_cache = {}
        if self.method == 'fourier_metal' and self.rotation:
            for scale in self.scales:
                extent = int(math.ceil(math.hypot(*self.shape)*scale*1.2))
                h, w = self.shape
                # Build an upright template for the SAME source-pixel scale as
                # its proposed ROI. Then Fourier estimates residual scale.
                yy, xx = np.indices((256, 256), dtype=np.float64)
                target = bilinear(self.template.astype(np.float64)/255,
                                  w/2+(xx-128)*extent/256/scale,
                                  h/2+(yy-128)*extent/256/scale)
                self.fourier_cache[scale] = (extent, FourierProposal(target, target.shape))
        self.setup_ms = (time.perf_counter()-start)*1000

    def close(self):
        for _, pattern in self.patterns:
            pattern.close()
        if self.scorer:
            self.scorer.close()

    def coarse(self):
        candidates = []
        for scale, pattern in self.patterns:
            for match in pattern.match(image_view(self.source), self.config, prune=self.job.get('prune', False)):
                match['scale'] = scale
                match['seed_scale'] = scale
                match['corners'] = corners([match['pos_x'], match['pos_y']], match['angle'], scale, self.shape)
                candidates.append(match)
        # Keep proposals from different scale banks until AFTER refinement.
        return candidates

    def score(self, poses):
        values, metrics = self.scorer.score(poses)
        self.ncc_ms += metrics['wall_ms']
        self.gpu_ms += max(0, metrics['device_ms'])
        self.evaluations += len(poses)
        return values

    def fourier(self, seeds, poses):
        proposals, owners = [], []
        for i, seed in enumerate(seeds):
            extent, proposer = self.fourier_cache[seed['seed_scale']]
            yy, xx = np.indices((256, 256), dtype=np.float64)
            crop = bilinear(self.source_float,
                            seed['pos_x']+(xx-128)*extent/256,
                            seed['pos_y']+(yy-128)*extent/256)
            for local in proposer.propose(crop):
                candidate = np.array([poses[i, 0]+local[0]*extent/256,
                                      poses[i, 1]+local[1]*extent/256,
                                      local[2], poses[i, 3]+local[3]], np.float32)
                if not self.allow_scale:
                    candidate[3] = 0
                if (np.linalg.norm(candidate[:2]-poses[i, :2]) <= 6 and
                    angle_distance(float(candidate[2]), float(poses[i, 2])) <= 15 and
                    abs(float(candidate[3]-poses[i, 3])) <= math.log(1.3)):
                    proposals.append(candidate); owners.append(i)
        best = self.score(poses)
        adopted = set()
        if proposals:
            scores = self.score(np.array(proposals, np.float32))
            for candidate, owner, score in zip(proposals, owners, scores):
                if score > best[owner]:
                    poses[owner] = candidate; best[owner] = score; adopted.add(owner)
        return poses, len(proposals), len(adopted)

    def refine(self, seeds):
        sh, sw = self.source.shape
        poses = np.array([[m['pos_x']-sw/2, sh/2-m['pos_y'], m['angle'], math.log(m['scale'])] for m in seeds], np.float32)
        origins = poses.copy()
        fft_ms, fft_valid, fft_adopted = 0., 0, 0
        if self.fourier_cache:
            start = time.perf_counter()
            poses, fft_valid, fft_adopted = self.fourier(seeds, poses)
            fft_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        solver_stats = []
        if self.continuous:
            # A broad, inexpensive orientation/scale proposal bank increases
            # the optimizer's basin. Keep original seeds too; truth is absent.
            initial = poses
            if self.method != 'continuous_direct_cpu':
                offsets = np.array(list(itertools.product((0,), (0,),
                    (-6., -3., 0., 3., 6.) if self.rotation else (0.,),
                    (-math.log(1.1), 0., math.log(1.1)) if self.allow_scale else (0.,))), np.float32)
                candidates = poses[:, None, :] + offsets[None, :, :]
                values = self.score(candidates.reshape(-1, 4)).reshape(candidates.shape[:2])
                best = candidates[np.arange(len(poses)), np.argmax(values, axis=1)]
                different = np.any(best != poses, axis=1)
                initial = np.concatenate((poses, best[different]))
            poses, final_scores, solver_stats, metrics = self.scorer.refine(initial,
                optimize_scale=self.allow_scale, optimize_angle=self.rotation)
            self.ncc_ms += metrics['wall_ms']
            self.evaluations += sum(s['evaluations'] for s in solver_stats)
        else:
            poses, final_scores = self.refine_grid(poses, origins)
        output = []
        for pose, score in zip(poses, final_scores):
            center = [float(pose[0]+sw/2), float(sh/2-pose[1])]
            angle, scale = float((pose[2]+180) % 360-180), math.exp(float(pose[3]))
            output.append(dict(pos_x=center[0], pos_y=center[1], angle=angle, scale=scale,
                               score=float(score), corners=corners(center, angle, scale, self.shape)))
        return output, dict(fft_ms=fft_ms, fft_valid_proposals=fft_valid, fft_adopted_seeds=fft_adopted,
                            solver_stats=solver_stats, refine_ms=(time.perf_counter()-start)*1000)

    def refine_grid(self, poses, origins):
        offsets = np.array(list(itertools.product((-1, 0, 1), (-1, 0, 1),
                           (-1, 0, 1) if self.rotation else (0,), (-1, 0, 1) if self.allow_scale else (0,))), np.float32)
        steps = np.array([2., 2., 1., math.log(1.12)], np.float32)
        for _ in range(6):
            candidates = poses[:, None, :]+offsets[None, :, :]*steps
            candidates[:, :, :2] = np.clip(candidates[:, :, :2], origins[:, None, :2]-6, origins[:, None, :2]+6)
            delta = (candidates[:, :, 2]-origins[:, None, 2]+180) % 360-180
            candidates[:, :, 2] = origins[:, None, 2]+np.clip(delta, -15, 15)
            if self.allow_scale:
                candidates[:, :, 3] = np.clip(candidates[:, :, 3], math.log(.65), math.log(1.4))
            else:
                candidates[:, :, 3] = 0
            scores = self.score(candidates.reshape(-1, 4)).reshape(candidates.shape[:2])
            indices = np.argmax(scores, axis=1)
            poses = candidates[np.arange(len(poses)), indices]
            final_scores = scores[np.arange(len(poses)), indices]
            steps *= .5
        return poses, final_scores

    def run(self):
        self.ncc_ms, self.gpu_ms, self.evaluations = 0., 0., 0
        start = time.perf_counter()
        seeds = self.coarse()
        search_ms = (time.perf_counter()-start)*1000
        stages = dict(fft_ms=0., fft_valid_proposals=0, fft_adopted_seeds=0, refine_ms=0.)
        matches = seeds
        if self.hybrid and seeds:
            matches, stages = self.refine(seeds)
        start_nms = time.perf_counter()
        matches = suppress(matches, self.job['case']['limit'], self.job['case']['overlap'], self.job['case']['score'])
        nms_ms = (time.perf_counter()-start_nms)*1000
        total = (time.perf_counter()-start)*1000
        for i, match in enumerate(matches):
            match['index'] = i
        return dict(matches=matches, algorithm_ms=total, cpu_search_ms=search_ms, nms_ms=nms_ms,
                    seed_count=len(seeds), ncc_wall_ms=self.ncc_ms, gpu_command_ms=self.gpu_ms,
                    pose_evaluations=self.evaluations, **stages)


def truth_accuracy(matches, truths):
    available = set(range(len(matches)))
    rows = []
    for truth in truths:
        if not available:
            rows.append(dict(truth=truth, localized=False, pose_pass=False))
            continue
        index = min(available, key=lambda j: math.hypot(matches[j]['pos_x']-truth['pos_x'], matches[j]['pos_y']-truth['pos_y']))
        match = matches[index]
        distance = math.hypot(match['pos_x']-truth['pos_x'], match['pos_y']-truth['pos_y'])
        localized = distance <= 20
        errors = dict(position_px=distance, angle_deg=angle_distance(match['angle'], truth['angle']),
                      relative_scale=abs(match['scale']/truth['scale']-1))
        passed = distance <= 2 and errors['angle_deg'] <= 1 and errors['relative_scale'] <= .02
        rows.append(dict(truth=truth, result_index=index if localized else None, localized=localized,
                         pose_pass=passed, errors=errors if localized else None))
        if localized:
            available.remove(index)
    tp = sum(x['pose_pass'] for x in rows)
    return dict(localized=sum(x['localized'] for x in rows), pose_tp=tp, pose_fp=len(matches)-tp,
                pose_fn=len(truths)-tp, details=rows)


def worker(job):
    cv2.setNumThreads(1)
    search = Search(job)
    try:
        first = search.run()
        warmed = time.perf_counter()
        warmups = 1
        while time.perf_counter()-warmed < .2:
            search.run(); warmups += 1
        runs = [search.run() for _ in range(job['repeats'])]
        return dict(label=METHODS[job['method']], setup_ms=search.setup_ms, first_call=first,
                    warmups=warmups, algorithm=summary([r['algorithm_ms'] for r in runs]), runs=runs,
                    memory=dict(process_peak_bytes=peak_rss(),
                                pose_scorer_buffers=search.scorer.memory() if search.scorer else None),
                    image_shapes=dict(source=list(search.source.shape), template=list(search.template.shape)),
                    accuracy=truth_accuracy(runs[-1]['matches'], job['case']['truth']) if job['case'].get('truth') else None)
    finally:
        search.close()


def compare(reference, matches):
    available = set(range(len(matches)))
    pairs = []
    for r in reference:
        if not available:
            break
        index = min(available, key=lambda j: math.hypot(matches[j]['pos_x']-r['pos_x'], matches[j]['pos_y']-r['pos_y']))
        m = matches[index]
        distance = math.hypot(m['pos_x']-r['pos_x'], m['pos_y']-r['pos_y'])
        if distance > 20:
            continue
        available.remove(index)
        pairs.append(dict(reference_index=r['index'], result_index=m['index'], position_px=distance,
                          angle_deg=angle_distance(m['angle'], r['angle']), score_delta=m['score']-r['score']))
    return dict(reference_count=len(reference), result_count=len(matches), paired_within_20px=len(pairs),
                pairs=pairs, note='Agreement with current CPU, NOT independent ground-truth accuracy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT/'cpp-core/docs/real-image-experiments.json')
    parser.add_argument('--assets', type=Path, default=ROOT/'cpp-core/docs/real-image-experiments')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--held-out', action='store_true')
    parser.add_argument('--prune', action='store_true')
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        args.output.write_text(json.dumps(dict(job=job, result=worker(job)), indent=2)+'\n')
        return
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.assets.mkdir(parents=True, exist_ok=True)
    controlled = args.assets/'controlled-real-texture.png'
    truth = controlled_scene(controlled)
    cases = [dict(name='src7', source=str(ROOT/'Test Images/Src7.bmp'), template=str(ROOT/'Test Images/Dst7.bmp'),
                  angle=180, limit=70, proposal_score=.5, score=.5, overlap=.2, scales=[1.]),
             dict(name='src3', source=str(ROOT/'Test Images/Src3.bmp'), template=str(ROOT/'Test Images/Dst3.bmp'),
                  angle=0, limit=38, proposal_score=.8, score=.8, overlap=0., scales=[1.]),
             dict(name='src5_45', source=str(ROOT/'Test Images/Src5-45.bmp'), template=str(ROOT/'Test Images/Dst5.bmp'),
                  angle=180, limit=1, proposal_score=.8, score=.8, overlap=0., scales=[1.]),
             dict(name='controlled_scale', source=str(controlled), template=str(ROOT/'Test Images/Dst3.bmp'),
                  angle=180, limit=70, proposal_score=.5, score=.8, overlap=.2, scales=[.75, 1., 1.25], truth=truth)]
    if args.held_out:
        held_out = args.assets/'held-out-real-texture.png'
        cases.append(dict(name='held_out_scale', source=str(held_out), template=str(ROOT/'Test Images/Dst3.bmp'),
                          angle=180, limit=70, proposal_score=.5, score=.8, overlap=.2,
                          scales=[.75, 1., 1.25], truth=held_out_scene(held_out)))
    if args.cases:
        missing = set(args.cases)-{c['name'] for c in cases}
        if missing:
            parser.error(f'Unknown cases: {sorted(missing)}')
        cases = [c for c in cases if c['name'] in args.cases]
    tasks = []
    for case in cases:
        methods = ['cpu_fixed', 'cpu_fast', 'hybrid_cpu', 'hybrid_metal']
        if case['angle']:
            methods.append('fourier_metal')
        if case.get('truth'):
            methods.append('cpu_grid')
        methods = args.methods or methods
        tasks.extend(dict(case=case, method=method, build=str(args.build.resolve()), repeats=args.repeats,
                          prune=args.prune) for method in methods)
    np.random.default_rng(27).shuffle(tasks)
    files = [Path(__file__).resolve(), ROOT/'cpp-core/tools/pose_experiment.py',
             ROOT/'cpp-core/tools/benchmark_pose_experiments.py', ROOT/'cpp-core/tools/research_ncc_fourier.py',
             ROOT/'cpp-core/examples/python_ctypes.py']
    files += list((ROOT/'cpp-core/experiments').glob('*'))+list((ROOT/'cpp-core/src').glob('*'))
    report = dict(created_at=datetime.now().astimezone().isoformat(), platform=platform.platform(),
                  numpy=np.__version__, opencv=cv2.__version__,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
                  binary_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (library_path(args.build), args.build/'libfpm_c.dylib')},
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                {Path(c[k]) for c in cases for k in ('source', 'template')}},
                  protocol=dict(algorithm='full source search every run; cached template/scorer initialization excluded and separately reported',
                                candidates='current C++ pyramid/NCC generates all seeds; no saved detections or truth positions',
                                scale_bank=[.75, 1., 1.25], refinement='6 local grid rounds; float sampling; fixed scale on original images',
                                continuous='15 angle/scale proposals (5 fixed-scale or 1 translation-only), preserve original + best, C++ CPU analytic LM; Metal only scores proposals',
                                continuous_direct='Ablation: identical CPU coarse seeds straight to LM without the additional proposal bank',
                                coarse_pruning=args.prune,
                                fourier='local ROI proposal after CPU coarse search; CPU NumPy FFT; retain seed when proposal invalid or scores worse',
                                raster='OpenCV used for image decoding and independent controlled-scene generation, not matching',
                                truth='only controlled scene: <=2px, <=1deg, <=2% scale; original photos only reference agreement',
                                memory='fresh Python process peak RSS including decoding/setup; scorer buffers separate and overlap shared RAM',
                                repeats=args.repeats, warmup='one run plus >=200ms', serial=True), results=[])
    with tempfile.TemporaryDirectory(dir=ROOT/'build', prefix='real-image-') as temporary:
        path = Path(temporary)
        for i, job in enumerate(tasks):
            (path/'request.json').write_text(json.dumps(job))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path/'request.json'),
                            '--output', str(path/'result.json')], check=True, timeout=300)
            entry = json.loads((path/'result.json').read_text())
            report['results'].append(entry)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            r = entry['result']
            print(f"{i+1}/{len(tasks)} {job['case']['name']} {job['method']}: "
                  f"{r['algorithm']['median_ms']:.2f} ms; {len(r['runs'][-1]['matches'])} matches; "
                  f"pose TP={r['accuracy']['pose_tp'] if r['accuracy'] else 'unlabeled'}", flush=True)
    for entry in report['results']:
        baseline = next((r for r in report['results'] if r['job']['case']['name'] == entry['job']['case']['name'] and r['job']['method'] == 'cpu_fixed'), None)
        if baseline:
            entry['result']['reference_agreement'] = compare(baseline['result']['runs'][-1]['matches'], entry['result']['runs'][-1]['matches'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output)


if __name__ == '__main__':
    main()
