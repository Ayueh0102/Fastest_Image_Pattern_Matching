"""Predeclared held-out synthetic checks for the multiscale resident ablation.

New placements use photographed templates and the independent OpenCV renderer,
not matcher outputs. The degraded case reuses the new_texture12 geometry: it is
an additional condition, not twelve more independent objects. No thresholds are
adapted to any result. Raw images/metadata must be frozen before search starts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from many_object_scenes import make_placements, render_scene, validate_placements
from benchmark_fine_strip import save, sha

ROOT = Path(__file__).resolve().parents[2]
OLD_CASES = ('controlled6', 'many128_pose', 'many512_pose', 'many512_pose_fullres_subpixel')
NEW_CASES = ('new_texture12', 'new_small12', 'degraded12', 'empty_scene')
SEEDS = (2026101601, 2026101602)


def occluded_scene(template, truth, shape, *, fractions, noise_seed=2026101603,
                   sigma=1., noise_std=8., background=150.):
    if len(fractions) != len(truth) or any(not 0 <= v <= 1 for v in fractions):
        raise ValueError('One valid occlusion fraction per object is required')
    if sigma < 0 or noise_std < 0:
        raise ValueError('Noise and blur must be nonnegative')
    canvas = render_scene(template, truth, shape, noise_std=0, background=background).astype(np.float32)
    height, width = template.shape
    strips = []
    for item, fraction in zip(truth, fractions):
        columns = int(round(width * fraction))
        strips.append(columns)
        if not columns:
            continue
        mask = np.zeros(template.shape, np.float32)
        mask[:, :columns] = 1.
        x0, y0, x1, y1 = item['warp_roi']
        matrix = cv2.getRotationMatrix2D((width / 2., height / 2.), item['angle'], item['scale'])
        matrix[:, 2] += [item['pos_x'] - x0 - width / 2., item['pos_y'] - y0 - height / 2.]
        projected = cv2.warpAffine(mask, matrix, (x1 - x0, y1 - y0),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        patch = canvas[y0:y1, x0:x1]
        patch *= 1. - projected
        patch += background * projected
    if sigma:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    if noise_std:
        canvas += np.random.default_rng(noise_seed).standard_normal(shape, dtype=np.float32) * noise_std
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8), strips


def new_scenes():
    templates = [cv2.imread(str(ROOT / 'Test Images' / name), cv2.IMREAD_GRAYSCALE)
                 for name in ('Dst3.bmp', 'Dst4.bmp')]
    if any(t is None for t in templates):
        raise ValueError('Photographed fixture template is missing')
    result = []
    for name, template, seed, cell in zip(NEW_CASES[:2], templates, SEEDS, (256, 128)):
        shape = (3 * cell, 4 * cell)
        truth = make_placements(template.shape, rows=3, columns=4, cell_size=cell, seed=seed)
        image = render_scene(template, truth, shape, noise_seed=seed + 100, noise_std=2.)
        result.append(dict(name=name, template=template, image=image, truth=truth, seed=seed,
                           noise_seed=seed + 100, noise_std=2., blur_sigma=0.,
                           validity='normal', geometry_group=name, template_fixture='Dst3.bmp' if cell == 256 else 'Dst4.bmp'))
    base = result[0]
    fractions = [0., .1, .25, .4] * 3
    image, strips = occluded_scene(base['template'], base['truth'], base['image'].shape, fractions=fractions)
    result.append(dict(name='degraded12', template=base['template'], image=image,
                       truth=copy.deepcopy(base['truth']), seed=SEEDS[0], noise_seed=2026101603,
                       noise_std=8., blur_sigma=1., nominal_occlusion_fractions=fractions,
                       occlusion_columns=strips, validity='challenge', geometry_group='new_texture12',
                       template_fixture='Dst3.bmp'))
    empty = render_scene(base['template'], [], base['image'].shape, noise_seed=2026101604, noise_std=12.)
    result.append(dict(name='empty_scene', template=base['template'], image=empty, truth=[],
                       seed=None, noise_seed=2026101604, noise_std=12., blur_sigma=0.,
                       validity='negative', geometry_group='empty_scene', template_fixture='Dst3.bmp'))
    return result


def make_cases(dest, previous):
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Refusing to overwrite generated evidence')
    dest.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(previous).read_text())['cases']
    cases = []
    for name in OLD_CASES:
        matches = [c for c in old if c['name'] == name]
        if len(matches) != 1:
            raise ValueError('Missing/duplicate prior case: ' + name)
        case = dict(matches[0], validity='normal', origin='existing', expect_refine=True)
        cases.append(case)
    for rendered in new_scenes():
        name, template, image, truth = (rendered[k] for k in ('name', 'template', 'image', 'truth'))
        source_path, template_path, metadata_path = (dest / (name + suffix) for suffix in ('.png', '-template.png', '.json'))
        if not cv2.imwrite(str(source_path), image) or not cv2.imwrite(str(template_path), template):
            raise RuntimeError('Could not write generated fixture')
        metadata = {k: v for k, v in rendered.items() if k not in ('image', 'template')}
        metadata.update(version=1, renderer='OpenCV affine bilinear, transformed local strip occlusion, then blur/noise',
                        numpy_version=np.__version__, opencv_version=cv2.__version__,
                        shape=list(image.shape), template_shape=list(template.shape),
                        template_fixture_sha256=sha(ROOT / 'Test Images' / rendered['template_fixture']),
                        source_sha256=sha(source_path), template_sha256=sha(template_path),
                        pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        validation=validate_placements(truth, image.shape, template.shape),
                        limitations=['Synthetic transformed instances, not newly photographed scenes.',
                                     'Separated objects; occlusion is a synthetic local strip only.',
                                     'Degraded12 shares the clean texture geometry; not independent poses.',
                                     'Empty noise background does not represent all hard distractors.'])
        save(metadata_path, metadata)
        cases.append(dict(name=name, source=str(source_path), template=str(template_path),
                          scales=[.75, 1., 1.25], angle=180, limit=70, proposal_score=.5,
                          score=.8, overlap=.2, stop_layer=1, subpixel=0, workflow='full_rescue',
                          truth=truth, metadata=str(metadata_path), validity=rendered['validity'],
                          origin='new', geometry_group=rendered['geometry_group'],
                          expect_refine=bool(truth), expect_reuse=False))
    # Cases describe search only; native_job must remove truth and generation metadata.
    save(dest / 'cases.json', cases)
    save(dest / 'generation.json', dict(previous_sha256=sha(previous),
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__),
                       'cpp-core/tools/many_object_scenes.py': sha(ROOT / 'cpp-core/tools/many_object_scenes.py')},
        new_independent_geometries=2, new_independent_truth_poses=24,
        additional_degraded_conditions=12, negative_scenes=1,
        case_index_sha256=sha(dest / 'cases.json')))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/resident-multiscale-inputs')
    parser.add_argument('--previous', type=Path, default=ROOT / 'cpp-core/docs/fine-fused/protocol.json')
    args = parser.parse_args()
    make_cases(args.output, args.previous)
    print(args.output / 'cases.json')


if __name__ == '__main__':
    main()
