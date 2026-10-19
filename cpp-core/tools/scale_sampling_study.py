"""CPU-only fixed-geometry scale objective ablation; never an end-to-end search.

The grid and methods are frozen before measurement. Truth x/y/angle define the
one-dimensional diagnostic slice, not a deployable refinement. Fresh workers
report process RSS separately from timed scoring; all negative results persist.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('inverse_raw', 'inverse_interior2', 'inverse_blur06', 'forward_pixel', 'forward_area4')
TEMPLATES = ('Dst4.bmp', 'Dst10.jpg', 'Dst3.bmp')
SEEDS = (2026101701, 2026101702)
FACTORS = np.linspace(.95, 1.05, 41)
SCALE_LIMIT = .02


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bilinear(image, x, y):
    """Float64 zero-extended sampler independent of production and OpenCV."""
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    out = np.zeros(x.shape, dtype=np.float64)
    h, w = image.shape
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = ix + ox, iy + oy
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        weight = (fx if ox else 1 - fx) * (fy if oy else 1 - fy)
        out[ok] += image[yy[ok], xx[ok]] * weight[ok]
    return out


def ncc(a, b):
    if len(a) < 8 or len(a) != len(b):
        return -2.
    a, b = a - a.mean(), b - b.mean()
    aa, bb = float(np.sum(a*a)), float(np.sum(b*b))
    if aa <= 1e-12 or bb <= 1e-12:
        return -2.
    return float(np.clip(np.sum(a*b) / math.sqrt(aa*bb), -1., 1.))


def linear(angle, scale):
    r = math.radians(angle)
    return scale*np.array([[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]])


def source_to_template(x, y, pose, shape, scale):
    h, w = shape
    r = math.radians(pose['angle'])
    dx, dy = (x - pose['pos_x']) / scale, (y - pose['pos_y']) / scale
    return math.cos(r)*dx - math.sin(r)*dy + w/2., math.sin(r)*dx + math.cos(r)*dy + h/2.


def score(source, template, pose, scale, method):
    """Candidate geometry alone defines forward support; never truth support."""
    if method not in METHODS or scale <= 0:
        raise ValueError('Unknown method or nonpositive scale')
    h, w = template.shape
    if method.startswith('inverse'):
        y, x = np.indices(template.shape, dtype=np.float64)
        u, v = x - w/2., y - h/2.
        mat = linear(pose['angle'], scale)
        sx, sy = pose['pos_x'] + mat[0, 0]*u + mat[0, 1]*v, pose['pos_y'] + mat[1, 0]*u + mat[1, 1]*v
        if method == 'inverse_blur06':
            # Both filters represent an added sigma=.6 source-pixel blur.
            source = cv2.GaussianBlur(source, (0, 0), .6, borderType=cv2.BORDER_REFLECT_101)
            template = cv2.GaussianBlur(template, (0, 0), .6/scale, borderType=cv2.BORDER_REFLECT_101)
        values = bilinear(source, sx, sy)
        if method == 'inverse_interior2':
            if min(h, w) < 7:
                raise ValueError('Template too small for two-pixel interior')
            values, template = values[2:-2, 2:-2], template[2:-2, 2:-2]
        return ncc(values.ravel(), template.ravel())
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]) @ linear(pose['angle'], scale).T
    lo, hi = np.floor(corners.min(axis=0) + [pose['pos_x'], pose['pos_y']]).astype(int), np.ceil(corners.max(axis=0) + [pose['pos_x'], pose['pos_y']]).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi + 1, [source.shape[1], source.shape[0]])
    if np.any(hi <= lo):
        return -2.
    y, x = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float64)
    offsets = (0.,) if method == 'forward_pixel' else (-.375, -.125, .125, .375)
    expected = np.zeros(x.shape, dtype=np.float64)
    valid = np.ones(x.shape, dtype=bool)
    for dy in offsets:
        for dx in offsets:
            tx, ty = source_to_template(x + dx, y + dy, pose, template.shape, scale)
            # One template-pixel support margin, including every area sample.
            valid &= (tx >= 1.) & (tx <= w - 2.) & (ty >= 1.) & (ty <= h - 2.)
            expected += bilinear(template, tx, ty) / len(offsets)**2
    return ncc(source[lo[1]:hi[1], lo[0]:hi[0]][valid], expected[valid])


def render(template, pose, shape, renderer, noise_seed, noise_std=2.):
    """OpenCV legacy warp or 4x4 area integration using OpenCV inverse map.

    Score-side geometry/sampling is independent float64 code. Area scenes are
    model controls, not claims to reproduce a real camera's optical response.
    """
    if renderer not in ('legacy_linear', 'area4'):
        raise ValueError('Unknown renderer')
    h, w = template.shape
    matrix = cv2.getRotationMatrix2D((w/2., h/2.), pose['angle'], pose['scale'])
    matrix[:, 2] += [pose['pos_x'] - w/2., pose['pos_y'] - h/2.]
    alpha = np.ones(template.shape, np.float32)
    values = template.astype(np.float32)*pose['gain'] + pose['bias']
    if renderer == 'legacy_linear':
        patch = cv2.warpAffine(values, matrix, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(alpha, matrix, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
        result = patch + 150.*(1. - mask)
    else:
        inverse = cv2.invertAffineTransform(matrix)
        y, x = np.indices(shape, dtype=np.float32)
        result = np.zeros(shape, np.float32)
        for dy in (-.375, -.125, .125, .375):
            for dx in (-.375, -.125, .125, .375):
                xx, yy = x + dx, y + dy
                tx = (inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2]).astype(np.float32)
                ty = (inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2]).astype(np.float32)
                patch = cv2.remap(values, tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                mask = cv2.remap(alpha, tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                result += (patch + 150.*(1. - mask))/16.
    if noise_std:
        result += np.random.default_rng(noise_seed).standard_normal(shape, dtype=np.float32)*noise_std
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def make_inputs(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for filename in TEMPLATES:
        template = cv2.imread(str(ROOT/'Test Images'/filename), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError('Missing fixture '+filename)
        template_path = output/(Path(filename).stem+'_template.png')
        cv2.imwrite(str(template_path), template)
        side = max(96, int(math.ceil(math.hypot(*template.shape)*1.3)) + 32)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            for index in range(6):
                pose = dict(pos_x=side/2.+float(rng.uniform(-3, 3)), pos_y=side/2.+float(rng.uniform(-3, 3)),
                            angle=float(rng.uniform(-180, 180)), scale=float(np.exp(rng.uniform(math.log(.72), math.log(1.28)))),
                            gain=float(rng.uniform(.85, 1.1)), bias=float(rng.uniform(-12, 12)))
                for renderer in ('legacy_linear', 'area4'):
                    name = f'{Path(filename).stem}-{seed}-{index}-{renderer}'
                    source_path = output/(name+'.png')
                    image = render(template, pose, (side, side), renderer, seed + index + 17)
                    cv2.imwrite(str(source_path), image)
                    cases.append(dict(name=name, fixture=filename, group='held_out', renderer=renderer, seed=seed,
                                      source=str(source_path), template=str(template_path), pose=pose,
                                      source_sha256=digest(source_path), template_sha256=digest(template_path)))
    # Retained stage33 failures are diagnostic development cases, never held out.
    old = json.loads((ROOT/'build/resident-multiscale-inputs/cases.json').read_text())
    old = next(case for case in old if case['name'] == 'new_small12')
    for index, pose in enumerate(old['truth']):
        cases.append(dict(name=f'stage33-small-{index}', fixture='Dst4.bmp', group='retained_stage33', renderer='legacy_linear', seed=2026101602,
                          source=old['source'], template=old['template'], pose=pose,
                          source_sha256=digest(old['source']), template_sha256=digest(old['template'])))
    manifest = dict(version=1, scope='Fixed truth x/y/angle, bounded scale-only objective diagnostics; no search/optimizer.',
                    methods=list(METHODS), scale_factors=FACTORS.tolist(), strict_scale_relative_error_limit=SCALE_LIMIT,
                    templates=list(TEMPLATES), held_out_seeds=list(SEEDS), points_per_curve=41,
                    held_out_unique_poses=36, retained_unique_poses=12, renderer_count=2,
                    source_code_sha256=digest(__file__), opencv_version=cv2.__version__, numpy_version=np.__version__,
                    cases=cases)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output/'manifest.json'


def worker(manifest_path, method, output):
    cv2.setNumThreads(1)
    manifest = json.loads(manifest_path.read_text())
    if digest(__file__) != manifest['source_code_sha256']:
        raise ValueError('Study source changed after frozen input generation')
    start = time.perf_counter()
    records = []
    for case in manifest['cases']:
        for key in ('source', 'template'):
            if digest(case[key]) != case[key+'_sha256']:
                raise ValueError('Input changed')
        source, template = [cv2.imread(case[key], cv2.IMREAD_GRAYSCALE).astype(np.float64)/255. for key in ('source', 'template')]
        samples = []
        # Two repeats measure scoring only; first retained separately, no warmup.
        for repeat in range(2):
            t0 = time.perf_counter()
            values = [score(source, template, case['pose'], case['pose']['scale']*float(factor), method) for factor in manifest['scale_factors']]
            samples.append(dict(ms=(time.perf_counter()-t0)*1000., scores=values))
        if samples[0]['scores'] != samples[1]['scores']:
            raise ValueError('Nondeterministic scoring')
        best = int(np.argmax(values))
        error = float(manifest['scale_factors'][best] - 1.)
        records.append(dict(name=case['name'], group=case['group'], fixture=case['fixture'], renderer=case['renderer'], seed=case['seed'],
                            best_index=best, signed_scale_error=error, scale_gate_passed=abs(error) <= SCALE_LIMIT + 1e-12,
                            best_on_boundary=best in (0, 40), best_ncc=values[best], truth_ncc=values[20],
                            curve=values, first_ms=samples[0]['ms'], repeat_ms=samples[1]['ms'],
                            input_float64_array_bytes=int(source.nbytes+template.nbytes)))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == 'darwin' else rss*1024
    result = dict(method=method, records=records, score_count=len(records)*41*2,
                  process_peak_rss_bytes=int(rss_bytes), worker_wall_ms=(time.perf_counter()-start)*1000.,
                  memory_scope='Fresh Python worker RSS high-water includes Python/NumPy/OpenCV/input loading and scoring, excludes parent and other methods; not native/GPU memory.',
                  timing_scope='CPU float64 Python/NumPy reference objective; includes per-candidate maps/filters and allocations, excludes image loading, hashing, evaluation, and JSON I/O; not production native throughput.',
                  native_or_gpu_called=False, source_code_sha256=digest(__file__), manifest_sha256=digest(manifest_path))
    output.write_text(json.dumps(result, indent=2)+'\n')


def summarize(manifest, paths):
    results = [json.loads(path.read_text()) for path in paths]
    rows = []
    for group in ('held_out', 'retained_stage33'):
        renderers = ('legacy_linear', 'area4') if group == 'held_out' else ('legacy_linear',)
        for renderer in renderers:
            for fixture in (TEMPLATES if group == 'held_out' else ('Dst4.bmp',)):
                for result in results:
                    selected = [x for x in result['records'] if x['group']==group and x['renderer']==renderer and x['fixture']==fixture]
                    rows.append(dict(group=group, renderer=renderer, fixture=fixture, method=result['method'], n=len(selected),
                                     strict_scale_pass_count=sum(x['scale_gate_passed'] for x in selected), boundary_count=sum(x['best_on_boundary'] for x in selected),
                                     mean_signed_error=float(np.mean([x['signed_scale_error'] for x in selected])),
                                     max_abs_error=max(abs(x['signed_scale_error']) for x in selected),
                                     median_curve_ms=float(np.median([x['repeat_ms'] for x in selected])),
                                     process_peak_rss_mib=result['process_peak_rss_bytes']/2**20))
    return dict(version=1, manifest_sha256=digest(manifest), rows=rows,
                worker_results_sha256={p.name:digest(p) for p in paths},
                measured_source_sha256=results[0]['source_code_sha256'],
                limitations=['Truth x/y/angle supplied: these are objective slices, not detection recall or end-to-end improvement.',
                             'Two synthetic renderer models, 3 photographed texture fixtures, 2 held-out seeds; not independent real capture ground truth.',
                             'Forward model matched to a renderer is a positive model control; cross-renderer results carry transfer evidence only within these models.',
                             '41 scale samples with .25% spacing over ±5%; boundary maxima are unresolved, not globally optimal.',
                             'Reference Python CPU times and process RSS cannot predict C++ or Metal costs; no default/kernel changed.',
                             'Two deterministic score repeats give timing descriptives, not confidence intervals or tail latency.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=METHODS)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite evidence')
    if args.worker:
        worker(args.manifest, args.worker, args.output)
        return
    args.output.mkdir(parents=True)
    manifest = make_inputs(args.output/'inputs')
    paths = []
    for method in METHODS:
        path = args.output/(method+'.json')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', method, '--manifest', str(manifest), '--output', str(path)], check=True, env=env)
        paths.append(path)
        print('Completed '+method, flush=True)
    result = summarize(manifest, paths)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['rows'], indent=2))


if __name__ == '__main__':
    main()
