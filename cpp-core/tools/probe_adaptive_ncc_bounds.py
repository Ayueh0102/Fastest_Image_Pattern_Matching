"""Mathematical screening probe, NOT a matcher or CPU/Metal benchmark.

Extend the existing four-rectangle orthogonal-projection NCC upper bound with
template-only adaptive rectangle splits. The split criterion is reduction of
template squared error; it is not the 2025 segmented-NCC split-and-merge method.
NumPy implements all projection/splitting math here. OpenCV only decodes files.
No proposed bound is installed in the production matcher by this experiment.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGES = (4, 8, 16, 32, 64)
THRESHOLDS = (.5, .7, .8, .9, .95)
# Diagnostic margin only: a production floating-point proof is a separate gate.
MARGIN = 1e-9


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integral(a):
    return np.pad(np.cumsum(np.cumsum(a.astype(np.float64), axis=0), axis=1),
                  ((1, 0), (1, 0)))


def rect_sum(s, r):
    x0, y0, x1, y1 = r
    return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]


def area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def grid(shape, n):
    h, w = shape
    xs = np.linspace(0, w, min(n, w) + 1, dtype=int)
    ys = np.linspace(0, h, min(n, h) + 1, dtype=int)
    return [(int(xs[x]), int(ys[y]), int(xs[x+1]), int(ys[y+1]))
            for y in range(len(ys)-1) for x in range(len(xs)-1)]


def best_split(s, r):
    x0, y0, x1, y1 = r
    total = rect_sum(s, r)
    best = None
    for axis, cuts in ((0, range(x0+1, x1)), (1, range(y0+1, y1))):
        for cut in cuts:
            a = (x0, y0, cut, y1) if axis == 0 else (x0, y0, x1, cut)
            b = (cut, y0, x1, y1) if axis == 0 else (x0, cut, x1, y1)
            na, nb = area(a), area(b)
            sa = rect_sum(s, a)
            gain = na * nb / (na + nb) * (sa/na - (total-sa)/nb)**2
            if best is None or gain > best[0]:
                best = (float(gain), a, b)
    return best


def partitions(t):
    s = integral(t)
    active = {i: r for i, r in enumerate(grid(t.shape, 2))}
    heap = []
    next_id = len(active)

    def enqueue(i, r):
        split = best_split(s, r)
        if split is not None:
            gain, a, b = split
            heapq.heappush(heap, (-gain, i, a, b))

    for i, r in active.items():
        enqueue(i, r)
    snapshots = {}
    while True:
        if len(active) in STAGES:
            snapshots[str(len(active))] = list(active.values())
        if len(active) >= max(STAGES) or not heap:
            break
        _, i, a, b = heapq.heappop(heap)
        del active[i]
        for r in (a, b):
            active[next_id] = r
            enqueue(next_id, r)
            next_id += 1
    snapshots['uniform64'] = grid(t.shape, 8)
    return snapshots


def projection(t, x, rectangles):
    # Global centering is essential: per-block NCC is a different score.
    mt, mx = float(np.mean(t)), float(np.mean(x))
    tc = t.astype(np.float64) - mt
    xc = x.astype(np.float64) - mx
    et = float(np.sum(tc*tc))
    ex = float(np.sum(xc*xc))
    if et == 0 or ex == 0:
        return None  # Production constant-image conventions must stay separate.
    st, sx = integral(t), integral(x)
    qt, qx = integral(t.astype(float)**2), integral(x.astype(float)**2)
    dot = pt = px = local_residual_bound = 0.
    for r in rectangles:
        n = area(r)
        a = rect_sum(st, r) - n*mt
        b = rect_sum(sx, r) - n*mx
        dot += a*b/n
        pt += a*a/n
        px += b*b/n
        local_t = max(0., rect_sum(qt, r)-rect_sum(st, r)**2/n)
        local_x = max(0., rect_sum(qx, r)-rect_sum(sx, r)**2/n)
        local_residual_bound += np.sqrt(local_t*local_x)
    denom = np.sqrt(et*ex)
    upper = (dot + np.sqrt(max(0., et-pt)*max(0., ex-px)))/denom
    approximate = float(dot/np.sqrt(pt*ex)) if pt > 1e-16*et else None
    return dict(exact=float(np.sum(tc*xc)/denom), upper=float(upper),
                block_upper=float((dot+local_residual_bound)/denom),
                approximate=approximate, template_energy_fraction=float(pt/et))


def pyr_down(a):
    """Same integer kernel, reflect101, and final rounding as src/image.cpp."""
    a = a.astype(np.int64)
    weights = (1, 4, 6, 4, 1)
    padded = np.pad(a, ((0, 0), (2, 2)), mode='reflect')
    h = sum(k*padded[:, i:i+a.shape[1]:2] for i, k in enumerate(weights))
    padded = np.pad(h, ((2, 2), (0, 0)), mode='reflect')
    v = sum(k*padded[i:i+a.shape[0]:2, :] for i, k in enumerate(weights))
    return ((v+128)//256).astype(np.uint8)


def read_gray(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None or a.dtype != np.uint8:
        raise ValueError(f'Expected byte image: {path}')
    if a.ndim == 3:
        b, g, r = (a[:, :, i].astype(np.uint32) for i in range(3))
        a = ((77*r+150*g+29*b+128) >> 8).astype(np.uint8)
    return a


def windows(t, source, rng):
    h, w = t.shape
    for _ in range(64):
        y = int(rng.integers(source.shape[0]-h+1))
        x = int(rng.integers(source.shape[1]-w+1))
        yield dict(kind='source_window', x=x, y=y), source[y:y+h, x:x+w]
    yield dict(kind='exact_template'), t.copy()
    for dx, dy in ((1, 0), (0, 1), (2, 2), (4, 0), (0, 4), (8, 8)):
        yield dict(kind='cyclic_shift_control', dx=dx, dy=dy), np.roll(t, (dy, dx), (0, 1))
    for sigma in (2., 12., 40.):
        for repeat in range(3):
            x = np.clip(np.rint(t+rng.normal(0, sigma, t.shape)), 0, 255).astype(np.uint8)
            yield dict(kind='noise_control', sigma=sigma, repeat=repeat), x
    for gain, bias in ((.8, 10), (.5, 40), (1.2, -10)):
        yield dict(kind='photometric_control', gain=gain, bias=bias), np.clip(
            np.rint(t.astype(float)*gain+bias), 0, 255).astype(np.uint8)
    yield dict(kind='inverted_control'), 255-t
    yield dict(kind='constant_control'), np.full(t.shape, 127, dtype=np.uint8)
    for repeat in range(8):
        yield dict(kind='random_control', repeat=repeat), rng.integers(0, 256, t.shape, dtype=np.uint8)


def self_check():
    count = 0
    binary = [np.array([(i >> k) & 1 for k in range(4)], dtype=np.uint8).reshape(2, 2)*255
              for i in range(16)]
    for t in binary:
        for x in binary:
            for rects in (grid(t.shape, 1), grid(t.shape, 2)):
                p = projection(t, x, rects)
                if p is not None:
                    assert p['upper']+MARGIN >= p['exact']
                    assert p['block_upper']+MARGIN >= p['exact']
                    assert p['block_upper'] <= p['upper']+MARGIN
                    count += 1
    rng = np.random.default_rng(26001)
    for shape in ((3, 5), (8, 8), (13, 17), (32, 48)):
        for _ in range(8):
            t = rng.integers(0, 256, shape, dtype=np.uint8)
            maps = partitions(t)
            for x in (t, rng.integers(0, 256, shape, dtype=np.uint8)):
                previous = float('inf')
                previous_block = float('inf')
                for stage in STAGES:
                    if str(stage) not in maps:
                        continue
                    p = projection(t, x, maps[str(stage)])
                    assert p['upper']+MARGIN >= p['exact']
                    assert p['upper'] <= previous+MARGIN
                    assert p['block_upper']+MARGIN >= p['exact']
                    assert p['block_upper'] <= p['upper']+MARGIN
                    assert p['block_upper'] <= previous_block+MARGIN
                    previous = p['upper']
                    previous_block = p['block_upper']
                    count += 1
    # Verify the vectorized pyramid against the literal separable 2D definition,
    # including singleton and odd dimensions.
    def reflect(i, n):
        if n == 1:
            return 0
        while i < 0 or i >= n:
            i = -i if i < 0 else 2*n-i-2
        return i
    for shape in ((1, 1), (1, 7), (7, 1), (5, 8), (13, 17)):
        a = rng.integers(0, 256, shape, dtype=np.uint8)
        expected = np.zeros(((shape[0]+1)//2, (shape[1]+1)//2), dtype=np.uint8)
        k = (1, 4, 6, 4, 1)
        for y in range(expected.shape[0]):
            for x in range(expected.shape[1]):
                val = sum(k[j]*k[i]*int(a[reflect(2*y+j-2, shape[0]), reflect(2*x+i-2, shape[1])])
                          for j in range(5) for i in range(5))
                expected[y, x] = (val+128)//256
        assert np.array_equal(pyr_down(a), expected)
    return dict(bound_checks=count, pyramid_shapes=5, passed=True)


def run():
    checks = self_check()
    cases = []
    input_hashes = {}
    started = time.perf_counter()
    for index in range(1, 11):
        name = f'Dst{index}'
        tp = ROOT/'Test Images'/f'{name}.bmp'
        if not tp.exists():
            tp = tp.with_suffix('.jpg')
        sp = ROOT/'Test Images'/f'Src{index}.bmp'
        if index == 5:
            sp = ROOT/'Test Images'/'Src5-180.bmp'
        if not sp.exists():
            sp = sp.with_suffix('.jpg')
        t, source = read_gray(tp), read_gray(sp)
        for path in (tp, sp):
            input_hashes[str(path.relative_to(ROOT))] = sha(path.read_bytes())
        levels = [(0, t, source)]
        n, level = t.size, 0
        while n > 256:
            t, source = pyr_down(t), pyr_down(source)
            n //= 4
            level += 1
        if level:
            levels.append((level, t, source))
        for level, t, source in levels:
            started_build = time.perf_counter()
            maps = partitions(t)
            build_ms = 1000*(time.perf_counter()-started_build)
            rng = np.random.default_rng(26000+index*100+level)
            rows = []
            for descriptor, x in windows(t, source, rng):
                values = {key: projection(t, x, rects) for key, rects in maps.items()}
                rows.append(dict(**descriptor, pixel_sha256=sha(x.tobytes()), values=values))
            stages = {}
            for key, rects in maps.items():
                valid = [r['values'][key] for r in rows if r['values'][key] is not None]
                stages[key] = dict(rectangles=len(rects), energy_fraction=valid[0]['template_energy_fraction'],
                    bound_violations=sum(p['upper']+MARGIN < p['exact'] for p in valid),
                    block_bound_violations=sum(p['block_upper']+MARGIN < p['exact'] for p in valid),
                    thresholds={str(tau): dict(
                        above_threshold=sum(p['exact'] >= tau for p in valid),
                        below_threshold=sum(p['exact'] < tau for p in valid),
                        rejected=sum(p['upper']+MARGIN < tau for p in valid),
                        block_rejected=sum(p['block_upper']+MARGIN < tau for p in valid),
                        false_rejections=sum(p['exact'] >= tau and p['upper']+MARGIN < tau for p in valid),
                        block_false_rejections=sum(p['exact'] >= tau and p['block_upper']+MARGIN < tau for p in valid),
                        approximation_false_rejections=sum(p['exact'] >= tau and
                            p['approximate'] is not None and p['approximate'] < tau for p in valid))
                        for tau in sorted(set(THRESHOLDS + (.8*.9**level,)))})
            assert all(s['bound_violations'] == s['block_bound_violations'] == 0 for s in stages.values())
            cases.append(dict(template=name, level=level, shape=list(t.shape), windows=len(rows),
                degenerate_windows=sum(r['values']['4'] is None for r in rows),
                template_pixel_sha256=sha(t.tobytes()), source_pixel_sha256=sha(source.tobytes()),
                original_layer_threshold_at_min_score_point8=.8*.9**level,
                template_build_python_ms=build_ms, partitions=maps, stages=stages, observations=rows))
            print(f'{name} level={level} shape={t.shape}: complete', file=sys.stderr, flush=True)
    return dict(schema_version=1, purpose='mathematical feasibility only; not matcher performance or recall',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=sha(Path(__file__).read_bytes()), input_hashes=input_hashes,
        environment=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform()),
        protocol=dict(seed_rule='26000 + template_index*100 + pyramid_level',
            templates='Dst1..Dst10; BMP preferred, paired Src# BMP/JPG except Src5-180.bmp', levels='native and minReducedArea256 top',
            stages=STAGES, thresholds=THRESHOLDS, diagnostic_margin=MARGIN,
            sample_counts=dict(source_windows=64, synthetic_controls=29),
            timings='Python diagnostic overhead only, no native or GPU performance comparison'),
        checks=checks, elapsed_python_seconds=time.perf_counter()-started,
        process_peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        process_peak_rss_units='bytes' if sys.platform == 'darwin' else 'KiB', cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    else:
        if args.output is None:
            parser.error('--output is required')
        result = run()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
