"""Predeclared identity/occlusion challenges; geometry never comes from matching.

Development and heldout use distinct photographed fixtures and RNG streams.
Both are frozen before observing either method. The heldout split is a locked
validation split, not a claim of externally collected camera data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from benchmark_fine_strip import save, sha
from many_object_scenes import _corners, _linear, _roi

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {'development': ('Dst3.bmp', 2026103401), 'heldout': ('Dst5.bmp', 2026103402)}
KINDS = ('occlusion', 'distractors', 'contact')


def placement(index, center, angle, scale, shape, identity='target', strip_fraction=0.):
    corners = _corners(center, angle, scale, shape)
    return dict(index=index, identity=identity, pos_x=float(center[0]), pos_y=float(center[1]),
                angle=float(angle), scale=float(scale), corners=corners, warp_roi=_roi(corners),
                gain=1., bias=0., strip_fraction=strip_fraction)


def altered_identity(template):
    """Only a declared local identity patch changes; global shape stays identical."""
    variant = template.copy()
    h, w = template.shape
    y0, y1, x0, x1 = h//4, h//2, w//4, w//2
    variant[y0:y1, x0:x1] = 255 - variant[y0:y1, x0:x1]
    return variant, [x0, y0, x1, y1]


def composite(template, items, shape, seed, noise_std=2.):
    """Independent affine painter with explicit depth order and visible support."""
    if template.ndim != 2 or template.dtype != np.uint8 or noise_std < 0:
        raise ValueError('uint8 gray template and nonnegative noise required')
    h, w = template.shape
    canvas = np.full(shape, 150., np.float32)
    visible, support = [], []
    variant, patch = altered_identity(template)
    for item in items:
        if item['identity'] not in ('target', 'altered_patch') or not 0 <= item['strip_fraction'] <= 1:
            raise ValueError('Unknown identity or invalid occlusion')
        if _corners([item['pos_x'], item['pos_y']], item['angle'], item['scale'], template.shape) != item['corners']:
            raise ValueError('Invalid corners')
        x0, y0, x1, y1 = item['warp_roi']
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError('Object leaves source image')
        matrix = np.column_stack((_linear(item['angle'], item['scale']),
            np.asarray([item['pos_x']-x0, item['pos_y']-y0]) - _linear(item['angle'], item['scale']) @ [w/2, h/2]))
        texture = template if item['identity'] == 'target' else variant
        local = texture.astype(np.float32)
        columns = int(round(w * item['strip_fraction']))
        local[:, :columns] = 150.
        alpha = cv2.warpAffine(np.ones(template.shape, np.float32), matrix, (x1-x0, y1-y0),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        content = np.ones(template.shape, np.float32)
        content[:, :columns] = 0
        content = cv2.warpAffine(content, matrix, (x1-x0, y1-y0),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pixels = cv2.warpAffine(local, matrix, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        canvas[y0:y1, x0:x1] *= 1-alpha
        canvas[y0:y1, x0:x1] += pixels
        # Each later opaque support occludes every earlier object's visible content.
        for prior in visible:
            prior[y0:y1, x0:x1] *= 1-alpha
        full = np.zeros(shape, np.float32)
        full[y0:y1, x0:x1] = content
        visible.append(full)
        support.append(float(alpha.sum()))
        item['strip_columns'] = columns
    fractions = [float(mask.sum()) / total for mask, total in zip(visible, support)]
    canvas += np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) * noise_std
    image = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
    return image, fractions, patch


def scenes():
    result = []
    for split, (fixture, seed) in SPLITS.items():
        template = cv2.imread(str(ROOT/'Test Images'/fixture), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError('Fixture unavailable: '+fixture)
        rng = np.random.default_rng(seed)
        for kind_index, kind in enumerate(KINDS):
            items = []
            if kind == 'contact':
                shape = (960, 960)
                # Three pairs: exact touching, modest and stronger overlap. Both
                # objects in a pair share rotation, making the contact explicit.
                for pair, overlap in enumerate((0., .12, .30)):
                    angle = float(rng.uniform(-28., 28.))
                    scale = float(rng.uniform(.9, 1.1))
                    axis = _linear(angle, scale) @ [template.shape[1]*(1-overlap), 0.]
                    midpoint = np.asarray([480., 160.+pair*320])
                    for side in (-.5, .5):
                        item = placement(len(items), midpoint+side*axis, angle, scale, template.shape)
                        item.update(pair=pair, nominal_overlap_fraction=overlap)
                        items.append(item)
            else:
                count = 6 if kind == 'occlusion' else 8
                shape = (640, 1280)
                fractions = (0., .1, .2, .3, .4, .5)
                for index in range(count):
                    center = np.asarray([(index%4+.5)*320, (index//4+.5)*320]) + rng.uniform(-8., 8., 2)
                    identity = 'altered_patch' if kind == 'distractors' and index%2 else 'target'
                    items.append(placement(index, center, rng.uniform(-180., 180.),
                        rng.uniform(.85, 1.15), template.shape, identity,
                        fractions[index] if kind == 'occlusion' else 0.))
            image, visible, patch = composite(template, items, shape, seed+kind_index*100)
            for item, fraction in zip(items, visible):
                item['visible_content_fraction'] = fraction
            result.append(dict(name=split+'_'+kind, split=split, kind=kind, seed=seed+kind_index*100,
                template_fixture=fixture, template=template, image=image, objects=items,
                truth=[dict(i) for i in items if i['identity']=='target'],
                distractors=[dict(i) for i in items if i['identity']!='target'], identity_patch=patch))
    return result


def generate(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite frozen inputs')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for scene in scenes():
        name = scene['name']
        source, template, metadata = [output/(name+s) for s in ('.png', '-template.png', '.json')]
        if not cv2.imwrite(str(source), scene['image']) or not cv2.imwrite(str(template), scene['template']):
            raise RuntimeError('Failed to save fixture')
        data = {k:v for k,v in scene.items() if k not in ('image','template')}
        data.update(source_sha256=sha(source), template_sha256=sha(template),
            original_fixture_sha256=sha(ROOT/'Test Images'/scene['template_fixture']),
            pixels_sha256=hashlib.sha256(scene['image'].tobytes()).hexdigest(),
            shape=list(scene['image'].shape), template_shape=list(scene['template'].shape),
            renderer='OpenCV bilinear affine; ordered opaque composition; noise sigma2 after composition',
            numpy_version=np.__version__, opencv_version=cv2.__version__)
        save(metadata, data)
        cases.append(dict(name=name, split=scene['split'], kind=scene['kind'], source=str(source),
            template=str(template), metadata=str(metadata), truth=scene['truth'], distractors=scene['distractors'],
            scales=[.75,1.,1.25], angle=180, limit=70, proposal_score=.5, score=.8, overlap=.2,
            stop_layer=1, subpixel=0, workflow='full_rescue'))
    save(output/'cases.json', cases)
    save(output/'generation.json', dict(version=1, cases_sha256=sha(output/'cases.json'),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'cpp-core/tools/many_object_scenes.py':sha(ROOT/'cpp-core/tools/many_object_scenes.py')},
        distinct_target_poses=sum(len(c['truth']) for c in cases), distinct_distractor_poses=sum(len(c['distractors']) for c in cases),
        limitations=['Synthetic transformed existing photographed templates, not new camera frames.',
          'Altered-patch negatives define exact identity discrimination; plain NCC may intentionally lack that specificity.',
          'Targets retain their GT identity under occlusion; no visibility-based removal from strict accuracy.',
          'Both development and heldout frozen before results; no adaptive heldout tuning.']))
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/stage34-challenge-inputs')
    args=parser.parse_args()
    generate(args.output)
    print(args.output/'cases.json')

if __name__=='__main__':
    main()
