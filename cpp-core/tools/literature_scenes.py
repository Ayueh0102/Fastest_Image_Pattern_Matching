"""Round 50: development and group-C scenes from the existing generators with never-used seeds.

No existing generator is modified; this wrapper calls their functions:

* stage49_scenes (procedural textures, 7x7 rotated-copy layout, round-35 renderer,
  noise / empty / distractor conditions) for tiny templates of 12-24 px;
* many_object_scenes (make_placements, render_scene) for crowded scenes;
* challenge_coverage_scenes (occlusion, altered-identity distractors, contact)
  with its split table replaced by round-50 seeds.

Splits:
  dev  - built before the protocol freeze, used only to fix TDCM-adapted's peak
         threshold; never scored as results. One Dst8 crowded scene (6x6 cells of
         384 px, scales 0.72-1.28, seed 50001) and four procedural tiny templates
         (seeds 50000-50003, sides 12/16/20/24 px) with clean and noise positives
         and empty and distractor negatives.
  c    - group C, built only after the frozen protocol names this file's SHA-256:
         24 tiny templates (16 procedural, seeds 50100-50115, sides 12-24 px; Dst5,
         Dst7, Dst8, Dst9 resized to a 16 and a 24 px longer side), the three
         challenge kinds for Dst3 (seed 2026105001) and Dst5 (seed 2026105002), and
         one 8192x8192 scene with 512 Dst3 copies (seed 20261050).
Truth comes from the generators' analytic poses, never from a matcher.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import challenge_coverage_scenes as challenge
import many_object_scenes as many
import stage49_scenes as s49

ROOT = Path(__file__).resolve().parents[2]
TINY_SIDES = (12, 14, 16, 18, 20, 22, 24)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def tiny_cases(name, pixels, seed, out, family, extra):
    """Round-49 recipe: clean and noise positives, empty and distractor negatives."""
    h, w = pixels.shape
    tpath = out / 'templates' / f'{name}.png'
    tpath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tpath), pixels)
    size, background, objects = s49.layout(w, h, seed)
    clean = s49.render(pixels, size, background, objects)
    distractor = s49.procedural(seed + 5000, 2.0, (w, h))
    _, _, decoys = s49.layout(w, h, seed + 1000)
    scenes = dict(clean=(s49.finish(clean), objects), noise=(s49.finish(clean, seed + 2000), objects),
                  empty=(s49.finish(np.full((size, size), background, np.float32), seed + 3000), []),
                  distractor=(s49.finish(s49.render(distractor, size, background, decoys), seed + 4000), []))
    cases = []
    for condition, (image, truth) in scenes.items():
        spath = out / 'scenes' / f'{name}-{condition}.png'
        spath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(spath), image)
        cases.append(dict(name=f'{name}_{condition}', group='tiny', family=family, condition=condition, positive=bool(truth),
                          template=str(tpath), source=str(spath), template_size=[w, h], source_size=[size, size],
                          background=background, layout_seed=seed, truth=[dict(t, width=w * t['scale'], height=h * t['scale'])
                                                                         for t in truth], expected_count=len(truth), **extra))
    return cases


def crowded_case(name, template, rows, columns, cell, count, seed, noise_seed, out, group):
    placements = many.make_placements(template.shape, rows=rows, columns=columns, cell_size=cell, seed=seed)[:count]
    shape = (rows * cell, columns * cell)
    image = many.render_scene(template, placements, shape, noise_seed=noise_seed)
    spath, tpath = out / 'scenes' / f'{name}.png', out / 'templates' / f'{name}-template.png'
    spath.parent.mkdir(parents=True, exist_ok=True)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(spath), image)
    cv2.imwrite(str(tpath), template)
    th, tw = template.shape
    truth = [dict(index=p['index'], pos_x=p['pos_x'], pos_y=p['pos_y'], angle=p['angle'], scale=p['scale'],
                  width=tw * p['scale'], height=th * p['scale'], corners=p['corners']) for p in placements]
    return dict(name=name, group=group, positive=True, template=str(tpath), source=str(spath), template_size=[tw, th],
                source_size=[shape[1], shape[0]], layout_seed=seed, noise_seed=noise_seed, truth=truth, expected_count=count,
                scale_range=[.72, 1.28])


def resized_test_image(name, side):
    image = cv2.imread(str(ROOT / 'Test Images' / name), cv2.IMREAD_UNCHANGED)
    image = s49.gray(image)
    h, w = image.shape
    k = side / max(h, w)
    return cv2.resize(image, (max(2, round(w * k)), max(2, round(h * k))), interpolation=cv2.INTER_AREA)


def build_dev(out):
    cases = [crowded_case('dev_many_dst8', s49.gray(cv2.imread(str(ROOT / 'Test Images/Dst8.bmp'), cv2.IMREAD_UNCHANGED)),
                          6, 6, 384, 36, 50001, 50002, out, 'dev')]
    for k, side in enumerate((12, 16, 20, 24)):
        seed = 50000 + k
        pixels = s49.procedural(seed, s49.PROCEDURAL_SIGMAS[2 * k], (side, side))
        cases += tiny_cases(f'dev_proc{seed}', pixels, seed + 600, out, 'procedural', dict(seed=seed, split='dev'))
    return cases


def build_c(out):
    cases = []
    for k in range(16):
        seed = 50100 + k
        side = TINY_SIDES[k % len(TINY_SIDES)]
        rng = np.random.default_rng(seed)
        w, h = side, int(rng.integers(max(8, side // 2), side + 1))
        pixels = s49.procedural(seed, s49.PROCEDURAL_SIGMAS[k % 8], (w, h))
        cases += tiny_cases(f'proc{seed}', pixels, seed + 700, out, 'procedural', dict(seed=seed, split='c'))
    for j, name in enumerate(('Dst5.bmp', 'Dst7.bmp', 'Dst8.bmp', 'Dst9.bmp')):
        for side in (16, 24):
            pixels = resized_test_image(name, side)
            cases += tiny_cases(f'{Path(name).stem.lower()}_{side}', pixels, 50200 + 10 * j + side, out, 'test_images',
                                dict(file=name, side=side, split='c'))
    challenge.SPLITS = {'r50dst3': ('Dst3.bmp', 2026105001), 'r50dst5': ('Dst5.bmp', 2026105002)}
    for case in challenge.generate(out / 'challenge'):
        meta = json.loads(Path(case['metadata']).read_text())
        th, tw = meta['template_shape']
        truth = [dict(t, width=tw * t['scale'], height=th * t['scale']) for t in case['truth']]
        cases.append(dict(name=case['name'], group='challenge', kind=case['kind'], positive=True, template=case['template'],
                          source=case['source'], template_size=[tw, th], truth=truth, expected_count=len(truth),
                          distractors=case['distractors'], scale_range=[.85, 1.15] if case['kind'] != 'contact' else [.9, 1.1]))
    dst3 = cv2.imread(str(ROOT / 'Test Images/Dst3.bmp'), cv2.IMREAD_GRAYSCALE)
    cases.append(crowded_case('many8k_512', dst3, 32, 32, 256, 512, 20261050, 2026105003, out, 'large'))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('dev', 'c'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, help='frozen protocol.json; required for the c split')
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Refusing to overwrite frozen scenes')
    provenance = dict(split=args.split, generator_sha256=sha(__file__), numpy=np.__version__, opencv=cv2.__version__,
                      dependencies={f: sha(Path(__file__).parent / f) for f in
                                    ('stage49_scenes.py', 'many_object_scenes.py', 'challenge_coverage_scenes.py')})
    if args.split == 'c':
        if args.protocol is None:
            raise SystemExit('The c split needs the frozen protocol')
        frozen = json.loads(args.protocol.read_text())
        if frozen.get('generators', {}).get('literature_scenes.py') != sha(__file__):
            raise SystemExit('The frozen protocol does not name this generator')
        provenance.update(protocol=str(args.protocol), protocol_sha256=sha(args.protocol))
    out.mkdir(parents=True, exist_ok=True)
    cases = build_dev(out) if args.split == 'dev' else build_c(out)
    for case in cases:
        case['input_sha256'] = {key: sha(case[key]) for key in ('source', 'template')}
    save_json(out / 'cases.json', cases)
    save_json(out / 'provenance.json', provenance)
    print(len(cases), 'cases', out / 'cases.json')


if __name__ == '__main__':
    main()
