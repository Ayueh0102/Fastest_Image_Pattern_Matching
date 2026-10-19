"""Reproducible crowded scenes, generated independently of the matcher.

Each object uses an OpenCV affine warp of a photographed texture. Only its
small destination ROI is warped. Ground truth describes the transform used by
the renderer; it is for post-search evaluation and never provides proposals.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_SOURCE = ROOT/'Test Images/Dst3.bmp'
SEED = 20261014
NOISE_SEED = 2026101401
CASE_NAMES = ('many64_pose', 'many128_pose', 'many256_pose',
              'many128_upright', 'many512_pose')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _linear(angle, scale):
    theta = math.radians(angle)
    return scale*np.array([[math.cos(theta), math.sin(theta)],
                           [-math.sin(theta), math.cos(theta)]], np.float64)


def _corners(center, angle, scale, template_shape):
    height, width = template_shape
    local = np.array([[-width/2, -height/2], [width/2, -height/2],
                      [width/2, height/2], [-width/2, height/2]])
    return (local @ _linear(angle, scale).T + center).tolist()


def _roi(corners):
    points = np.asarray(corners)
    # Include the bilinear footprint outside the nominal template rectangle.
    return [math.floor(float(points[:, 0].min()))-2,
            math.floor(float(points[:, 1].min()))-2,
            math.ceil(float(points[:, 0].max()))+2,
            math.ceil(float(points[:, 1].max()))+2]


def make_placements(template_shape, *, rows=16, columns=16, cell_size=256,
                    seed=SEED, jitter=8., upright=False):
    """Create the complete bank before slicing, so counts share a prefix.

    The upright control has the same positions and photometric changes as the
    pose bank. A separate RNG stream is used for background noise by rendering.
    """
    if rows < 1 or columns < 1 or cell_size < 1 or jitter < 0:
        raise ValueError('positive grid dimensions and nonnegative jitter required')
    if len(template_shape) != 2 or min(template_shape) < 2:
        raise ValueError('a two-dimensional template at least 2x2 is required')
    rng = np.random.default_rng(seed)
    cells = rng.permutation(rows*columns)
    placements = []
    for index, cell in enumerate(cells):
        row, column = divmod(int(cell), columns)
        center = np.array([(column+.5)*cell_size, (row+.5)*cell_size])
        center += rng.uniform(-jitter, jitter, 2)
        angle = float(rng.uniform(-180., 180.))
        scale = float(np.exp(rng.uniform(math.log(.72), math.log(1.28))))
        gain, bias = float(rng.uniform(.85, 1.1)), float(rng.uniform(-12., 12.))
        if upright:
            angle, scale = 0., 1.
        corners = _corners(center, angle, scale, template_shape)
        placements.append(dict(index=index, cell=[row, column],
            pos_x=float(center[0]), pos_y=float(center[1]), angle=angle,
            scale=scale, gain=gain, bias=bias, corners=corners, warp_roi=_roi(corners)))
    validate_placements(placements, (rows*cell_size, columns*cell_size), template_shape)
    return placements


def validate_placements(placements, shape, template_shape):
    """Reject clipping, overlap, or inconsistent poses before writing inputs.

    Disjoint padded axis-aligned warp ROIs are stronger than disjoint oriented
    rectangles: no interpolation pixel from one object can touch another.
    """
    height, width = shape
    minimum_boundary = math.inf
    minimum_separation = math.inf
    for index, item in enumerate(placements):
        expected = _corners([item['pos_x'], item['pos_y']], item['angle'],
                            item['scale'], template_shape)
        if not np.allclose(expected, item['corners'], rtol=0., atol=1e-10):
            raise ValueError('ground-truth corners disagree with pose')
        if _roi(expected) != item['warp_roi']:
            raise ValueError('warp ROI disagrees with pose')
        x0, y0, x1, y1 = item['warp_roi']
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('object interpolation ROI leaves the image')
        minimum_boundary = min(minimum_boundary, x0, y0, width-x1, height-y1)
        for other in placements[:index]:
            a0, b0, a1, b1 = other['warp_roi']
            dx, dy = max(a0-x1, x0-a1, 0), max(b0-y1, y0-b1, 0)
            if x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1:
                raise ValueError('object interpolation ROIs overlap')
            minimum_separation = min(minimum_separation, math.hypot(dx, dy))
    return dict(object_count=len(placements), all_inside=True, no_overlap=True,
        pairwise_roi_checks=len(placements)*(len(placements)-1)//2,
        min_roi_boundary_px=minimum_boundary if placements else None,
        min_roi_separation_px=minimum_separation if len(placements) > 1 else None)


def render_scene(template, placements, shape, *, noise_seed=NOISE_SEED,
                 noise_std=2., background=150.):
    """Render only local affine ROIs; use bounded noise chunks after pasting.

    The center convention is (width/2, height/2), matching the existing scorer.
    Positive angle means counterclockwise in image coordinates. The output is
    reproducible for the recorded NumPy/OpenCV versions and saved parameters.
    """
    if template.ndim != 2 or template.dtype != np.uint8:
        raise ValueError('uint8 grayscale template required')
    if noise_std < 0:
        raise ValueError('noise standard deviation must be nonnegative')
    validate_placements(placements, shape, template.shape)
    height, width = shape
    th, tw = template.shape
    canvas = np.full(shape, background, np.float32)
    alpha = np.ones(template.shape, np.float32)
    source = template.astype(np.float32)
    for item in placements:
        x0, y0, x1, y1 = item['warp_roi']
        linear = _linear(item['angle'], item['scale'])
        translation = np.array([item['pos_x']-x0, item['pos_y']-y0])-linear @ [tw/2, th/2]
        matrix = np.column_stack((linear, translation))
        patch = cv2.warpAffine(source*item['gain']+item['bias'], matrix,
            (x1-x0, y1-y0), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(alpha, matrix, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        region = canvas[y0:y1, x0:x1]
        region *= 1.-mask
        region += patch
    rng = np.random.default_rng(noise_seed)
    output = np.empty(shape, np.uint8)
    for y0 in range(0, height, 256):
        region = canvas[y0:min(y0+256, height)]
        if noise_std:
            region += rng.standard_normal(region.shape, dtype=np.float32)*noise_std
        output[y0:y0+len(region)] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
    return output


def make_cases(inputs: Path, selected=None):
    """Write selected PNG/metadata inputs and return benchmark Search cases."""
    inputs = Path(inputs).resolve()
    inputs.mkdir(parents=True, exist_ok=True)
    if isinstance(selected, str):
        selected = selected.split(',')
    selected = set(CASE_NAMES if selected is None else selected)
    unknown = selected-set(CASE_NAMES)
    if unknown:
        raise ValueError('unknown many-object scenes: '+','.join(sorted(unknown)))
    template = cv2.imread(str(TEMPLATE_SOURCE), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError('missing template: '+str(TEMPLATE_SOURCE))
    template_path = inputs/'many_template.png'
    if not cv2.imwrite(str(template_path), template):
        raise RuntimeError('failed to write template PNG')
    banks, cases = {}, []
    for name in CASE_NAMES:
        if name not in selected:
            continue
        count = int(name.split('_')[0].removeprefix('many'))
        upright = name.endswith('_upright')
        columns = 32 if count == 512 else 16
        key = (columns, upright)
        if key not in banks:
            banks[key] = make_placements(template.shape, columns=columns, upright=upright)
        truth = banks[key][:count]
        shape = (4096, columns*256)
        image = render_scene(template, truth, shape)
        source_path, metadata_path = inputs/(name+'.png'), inputs/(name+'.json')
        if not cv2.imwrite(str(source_path), image):
            raise RuntimeError('failed to write scene PNG')
        metadata = dict(version=1, name=name, generator='many_object_scenes.py',
            renderer='OpenCV warpAffine INTER_LINEAR, constant zero border; local ROIs',
            numpy_version=np.__version__, opencv_version=cv2.__version__,
            template_source='Test Images/Dst3.bmp', template_source_sha256=sha256(TEMPLATE_SOURCE),
            template_file=template_path.name, template_png_sha256=sha256(template_path),
            template_shape=list(template.shape), source_file=source_path.name,
            source_png_sha256=sha256(source_path),
            source_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            shape=list(shape), object_count=count, seed=SEED, noise_seed=NOISE_SEED, noise_std=2.,
            background=150., rows=16, columns=columns, cell_size=256,
            center_jitter=8., angle_range=[-180., 180.] if not upright else [0., 0.],
            scale_range=[.72, 1.28] if not upright else [1., 1.],
            scale_distribution='log-uniform' if not upright else 'fixed',
            gain_range=[.85, 1.1], bias_range=[-12., 12.],
            prefix_group='4k_pose' if columns == 16 and not upright else name,
            validation=validate_placements(truth, shape, template.shape), truth=truth,
            limitations=['Repeated copies of one template, not distinct object classes.',
                         'Separated grid placements; no occlusion or touching objects.',
                         'Ground truth is for evaluation after search, never a search input.'])
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        cases.append(dict(name=name, source=str(source_path), template=str(template_path),
            scales=[1.] if upright else [.75, 1., 1.25], angle=0 if upright else 180,
            limit=768 if count == 512 else 384, proposal_score=.5, score=.8,
            overlap=.2, stop_layer=1, subpixel=0,
            workflow='core_only' if upright else 'full_rescue', truth=truth,
            metadata=str(metadata_path), object_count=count, scene_shape=list(shape)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default=None, help='comma-separated scene names')
    args = parser.parse_args()
    cases = make_cases(args.output, args.cases)
    index = args.output/'cases.json'
    index.write_text(json.dumps(cases, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cases=[case['name'] for case in cases], index=str(index.resolve()))))


if __name__ == '__main__':
    main()
