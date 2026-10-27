"""Round 52: holdout scenes for the small-template match verification, from never-used seeds.

No existing generator is modified. literature_scenes.tiny_cases (the round-49/50 recipe:
7x7 rotated copies, clean and noise positives, empty and distractor negatives) makes four
conditions per template; this file adds a fifth, ``boundary``: the clean layout cropped
so the first row and column of copies cross the top and left image edges (every centre
stays at least a quarter of the template diagonal inside), with seeded noise. Truth comes
from the generator's analytic poses; ``edge`` marks a copy whose footprint leaves the crop.

Holdout: 24 procedural templates (seeds 52100-52123, longer side 12-24 px, the round-49
sigma cycle) and Dst1, Dst2, Dst4, Dst6 resized to a 16 and a 24 px longer side, 5 scenes
each (160 cases). Built only after the frozen protocol names this file's SHA-256.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

import literature_scenes as lsc
import stage49_scenes as s49

PROCEDURAL_SEEDS = range(52100, 52124)
TEST_IMAGES = ('Dst1.bmp', 'Dst2.bmp', 'Dst4.bmp', 'Dst6.bmp')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corners(t):
    a = math.radians(t['angle'])
    c, s = math.cos(a), math.sin(a)
    w, h = t['width'], t['height']
    return [(t['pos_x'] + c * u + s * v, t['pos_y'] - s * u + c * v) for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def boundary_case(name, pixels, seed, out, family, extra):
    """The clean layout of tiny_cases (same seed) cropped through its first row and column."""
    h, w = pixels.shape
    size, background, objects = s49.layout(w, h, seed)
    cell = size / 7
    offset = int(math.floor(cell / 2 - 4 - math.hypot(w, h) / 4))
    image = s49.render(pixels, size, background, objects)[offset:, offset:]
    side = size - offset
    image = s49.finish(image, seed + 6000)
    spath = out / 'scenes' / f'{name}-boundary.png'
    spath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(spath), image)
    truth = []
    for o in objects:
        t = dict(o, pos_x=o['pos_x'] - offset, pos_y=o['pos_y'] - offset, width=w * o['scale'], height=h * o['scale'])
        t['edge'] = any(x < 0 or y < 0 or x > side - 1 or y > side - 1 for x, y in corners(t))
        truth.append(t)
    return dict(name=f'{name}_boundary', group='tiny', family=family, condition='boundary', positive=True,
                template=str(out / 'templates' / f'{name}.png'), source=str(spath), template_size=[w, h],
                source_size=[side, side], background=background, layout_seed=seed, crop_offset=offset, truth=truth,
                expected_count=len(truth), edge_count=sum(t['edge'] for t in truth), **extra)


def build_holdout(out):
    cases = []
    for k, seed in enumerate(PROCEDURAL_SEEDS):
        side = lsc.TINY_SIDES[k % len(lsc.TINY_SIDES)]
        rng = np.random.default_rng(seed)
        w, h = side, int(rng.integers(max(8, side // 2), side + 1))
        pixels = s49.procedural(seed, s49.PROCEDURAL_SIGMAS[k % 8], (w, h))
        extra = dict(seed=seed, split='holdout')
        cases += lsc.tiny_cases(f'r52proc{seed}', pixels, seed + 800, out, 'procedural', extra)
        cases.append(boundary_case(f'r52proc{seed}', pixels, seed + 800, out, 'procedural', extra))
    for j, name in enumerate(TEST_IMAGES):
        for side in (16, 24):
            pixels = lsc.resized_test_image(name, side)
            extra = dict(file=name, side=side, split='holdout')
            stem = f'r52{Path(name).stem.lower()}_{side}'
            cases += lsc.tiny_cases(stem, pixels, 52400 + 10 * j + side, out, 'test_images', extra)
            cases.append(boundary_case(stem, pixels, 52400 + 10 * j + side, out, 'test_images', extra))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True, help='frozen protocol.json naming this file')
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Refusing to overwrite frozen scenes')
    frozen = json.loads(args.protocol.read_text())
    if frozen.get('generators', {}).get('round52_scenes.py') != sha(__file__):
        raise SystemExit('The frozen protocol does not name this generator')
    here = Path(__file__).parent
    provenance = dict(generator_sha256=sha(__file__), numpy=np.__version__, opencv=cv2.__version__, protocol=str(args.protocol),
                      protocol_sha256=sha(args.protocol),
                      dependencies={f: sha(here / f) for f in ('literature_scenes.py', 'stage49_scenes.py')})
    out.mkdir(parents=True, exist_ok=True)
    cases = build_holdout(out)
    for case in cases:
        case['input_sha256'] = {key: sha(case[key]) for key in ('source', 'template')}
    lsc.save_json(out / 'cases.json', cases)
    lsc.save_json(out / 'provenance.json', provenance)
    print(len(cases), 'cases', out / 'cases.json')


if __name__ == '__main__':
    main()
