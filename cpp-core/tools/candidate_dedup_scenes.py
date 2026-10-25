"""Round-43 negative-control scenes for coarse-candidate dedup.

Generated independently of the matcher; truth is the renderer's transform and
is never a search input.

sym_marker12: a template that is exactly 180-degree symmetric except for an
off-center zero-mean 4x4-block checkerboard marker. Each 8x8 cell of it averages to
zero, so the reduced coarse level is (nearly) symmetric and the true pose and
its 180-degree alias receive almost equal coarse scores; finer levels and the
full-resolution pose refinement resolve the marker. Clustering on position only
can let the alias erase the true pose before fine search. Scales are redrawn in
[0.95, 1.05] so the marker is not also a scale-mismatch test.

adjacent_grid16: 16 copies of Test Images/Dst3.bmp at scale 1 and +/-2 degrees in
a 4x4 grid whose centers are 1.04 template widths/heights apart. Every pair can
be output under maxOverlap 0.2, but a dedup radius above about one template
height lets a stronger neighbor suppress a weaker object.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from many_object_scenes import _corners, _linear, _roi, make_placements, render_scene, validate_placements

ROOT = Path(__file__).resolve().parents[2]
SYMMETRIC_SEED, SYMMETRIC_NOISE_SEED = 2026102401, 2026102402
GRID_SEED, GRID_NOISE_SEED = 2026102403, 2026102404
STD = dict(scales=[.75, 1., 1.25], angle=180, limit=70, proposal_score=.5, score=.8,
           overlap=.2, stop_layer=1, subpixel=0, workflow='full_rescue')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symmetric_template(rng, shape=(80, 112)):
    """Exactly 180-degree symmetric smooth texture plus a zero-mean marker."""
    noise = rng.uniform(0, 255, shape).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 4.)
    texture = 40+175*(texture-texture.min())/(texture.max()-texture.min())
    symmetric = np.rint((texture+texture[::-1, ::-1])/2)
    if not np.array_equal(symmetric, symmetric[::-1, ::-1]):
        raise AssertionError('template base is not 180-degree symmetric')
    # 32x32 marker of 4x4 blocks, off center and aligned to 8x8 cells: every
    # reduced-level cell receives equal positive and negative area, while the
    # 180-degree alias maps the marker onto unmarked texture.
    y0, x0 = 24, 16
    yy, xx = np.indices((32, 32))
    checker = np.where(((yy//4)+(xx//4)) % 2 == 0, 60., -60.)
    marked = symmetric.copy()
    marked[y0:y0+32, x0:x0+32] += checker
    if any(abs(checker[r:r+8, c:c+8].sum()) > 0 for r in range(0, 32, 8) for c in range(0, 32, 8)):
        raise AssertionError('marker is not zero-mean per reduced cell')
    return np.clip(marked, 0, 255).astype(np.uint8)


def near_unit_scale(placements, template_shape, rng, low=.95, high=1.05):
    """Keep generator positions and angles; redraw scale log-uniformly in [low, high]."""
    for item in placements:
        item['scale'] = float(np.exp(rng.uniform(math.log(low), math.log(high))))
        item['corners'] = _corners([item['pos_x'], item['pos_y']], item['angle'], item['scale'], template_shape)
        item['warp_roi'] = _roi(item['corners'])
    validate_placements(placements, (768, 1024), template_shape)
    return placements


def grid_placements(template_shape, rng, rows=4, columns=4, spacing=1.04):
    height, width = template_shape
    placements, origin = [], np.array([60.+spacing*width/2, 60.+spacing*height/2])
    for row in range(rows):
        for column in range(columns):
            center = origin+[column*spacing*width, row*spacing*height]+rng.uniform(-1., 1., 2)
            angle = float(rng.uniform(-2., 2.))
            corners = _corners(center, angle, 1., template_shape)
            placements.append(dict(index=len(placements), cell=[row, column], pos_x=float(center[0]),
                pos_y=float(center[1]), angle=angle, scale=1., gain=float(rng.uniform(.9, 1.05)),
                bias=float(rng.uniform(-8., 8.)), corners=corners, warp_roi=_roi(corners)))
    return placements


def render_touching(template, placements, shape, noise_seed, background=150., noise_std=2.):
    """Like render_scene, but neighboring warp ROIs may overlap; later objects win."""
    th, tw = template.shape
    canvas = np.full(shape, background, np.float32)
    alpha = np.ones(template.shape, np.float32)
    for item in placements:
        x0, y0, x1, y1 = item['warp_roi']
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError('object leaves the image')
        linear = _linear(item['angle'], item['scale'])
        translation = np.array([item['pos_x']-x0, item['pos_y']-y0])-linear @ [tw/2, th/2]
        matrix = np.column_stack((linear, translation))
        patch = cv2.warpAffine(template.astype(np.float32)*item['gain']+item['bias'], matrix, (x1-x0, y1-y0),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(alpha, matrix, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        region = canvas[y0:y1, x0:x1]
        region *= 1.-mask
        region += patch
    canvas += np.random.default_rng(noise_seed).standard_normal(shape, dtype=np.float32)*noise_std
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def overlap_ratio(a, b):
    """Rotated-rectangle intersection / smaller area, the research NMS rule."""
    polygon_a, polygon_b = np.asarray(a['corners'], np.float32), np.asarray(b['corners'], np.float32)
    area, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    return float(area)/min(cv2.contourArea(polygon_a), cv2.contourArea(polygon_b))


def make_cases(inputs):
    inputs = Path(inputs).resolve()
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    template = symmetric_template(np.random.default_rng(SYMMETRIC_SEED))
    # Angle aliasing is the subject: scales stay near the 1.0 model so the
    # high-frequency marker is not also a scale-mismatch test.
    truth = near_unit_scale(make_placements(template.shape, rows=3, columns=4, seed=SYMMETRIC_SEED),
                            template.shape, np.random.default_rng(SYMMETRIC_SEED+10))
    image = render_scene(template, truth, (768, 1024), noise_seed=SYMMETRIC_NOISE_SEED)
    cases.append(('sym_marker12', template, image, truth,
        dict(symmetry='180-degree except a zero-mean 4x4-block checkerboard marker (+/-60) at rows 24..55, columns 16..47')))
    grid_template = cv2.imread(str(ROOT/'Test Images/Dst3.bmp'), cv2.IMREAD_GRAYSCALE)
    if grid_template is None:
        raise ValueError('missing Test Images/Dst3.bmp')
    rng = np.random.default_rng(GRID_SEED)
    grid = grid_placements(grid_template.shape, rng)
    pairs = [(a['index'], b['index'], overlap_ratio(a, b)) for i, a in enumerate(grid) for b in grid[:i]]
    if max(ratio for *_, ratio in pairs) > .2:
        raise AssertionError('grid neighbors exceed maxOverlap and could not all be output')
    image = render_touching(grid_template, grid, (560, 720), GRID_NOISE_SEED)
    spacing = sorted(math.hypot(a['pos_x']-b['pos_x'], a['pos_y']-b['pos_y']) for i, a in enumerate(grid) for b in grid[:i])
    cases.append(('adjacent_grid16', grid_template, image, grid,
        dict(spacing='1.04 x template width/height', max_pair_overlap=max(r for *_, r in pairs),
             min_center_distance_px=spacing[0])))
    result = []
    for name, templ, scene, objects, notes in cases:
        source, template_path, metadata = inputs/(name+'.png'), inputs/(name+'-template.png'), inputs/(name+'.json')
        if not cv2.imwrite(str(source), scene) or not cv2.imwrite(str(template_path), templ):
            raise RuntimeError('cannot write control scene '+name)
        metadata.write_text(json.dumps(dict(version=1, name=name, generator='candidate_dedup_scenes.py',
            numpy_version=np.__version__, opencv_version=cv2.__version__, template_shape=list(templ.shape),
            shape=list(scene.shape), source_sha256=sha256(source), template_sha256=sha256(template_path),
            truth=objects, notes=notes), indent=2, allow_nan=False)+'\n')
        result.append(dict(name=name, source=str(source), template=str(template_path), metadata=str(metadata),
                           truth=objects, **STD))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = make_cases(args.output)
    print(json.dumps([dict(name=c['name'], objects=len(c['truth'])) for c in cases]))


if __name__ == '__main__':
    main()
