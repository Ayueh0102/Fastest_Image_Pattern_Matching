"""Stage 49 small-template recall scenes: development and holdout splits.

Templates come from three families, split by source photo or seed so that no
holdout template is seen while the rule is tuned: FSC147 exemplar crops (all
three exemplar boxes of each photo, compressed to .65*x+45 like the round-35
controls), procedural blurred-noise textures (separate seed ranges), and repo
Test Images templates. Each template gets four gray scenes on one random flat
background: 49 rotated copies (clean), the same plus Gaussian noise (noise), an
empty noisy plate (empty) and 49 copies of an unrelated procedural texture
(distractor). The round-35 scenes of the same names are reused unchanged
(development and heldout clean/exposure; hashes checked). The renderer is the
round-35 OpenCV bilinear warp; truth never enters a search job.

The holdout split is refused unless --protocol names a protocol whose
scenes.generator_sha256 equals this file, so the generator is frozen first.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

FSC_IMAGES = dict(development=['2933', '1913', '2927', '825', '1924'],
                  holdout=['7417', '2931', '7222', '7263', '1921', '1944', '2928'])
PROCEDURAL_SEEDS = dict(development=range(49100, 49108), holdout=range(49200, 49208))
PROCEDURAL_SIGMAS = (1.0, 1.0, 1.6, 1.6, 2.5, 2.5, 4.0, 4.0)
DISTRACTOR_SEED = dict(development=49300, holdout=49400)
TEST_IMAGES = dict(development=['Dst10.jpg'], holdout=['Dst4.bmp', 'Dst3.bmp'])
STAGE35_CASES = dict(development=['development_clean', 'development_exposure'],
                     holdout=['heldout_clean', 'heldout_exposure'])
SCENE_SEED = dict(development=49500, holdout=49600)
OBJECTS, NOISE_SIGMA, BACKGROUND_RANGE = 49, 4.0, (60., 190.)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gray(bgr):
    """The runners' integer luma (77R+150G+29B+128)>>8; equal channels stay exact."""
    if bgr.ndim == 2:
        return bgr
    b, g, r = (bgr[:, :, k].astype(np.uint32) for k in range(3))
    return ((77*r+150*g+29*b+128) >> 8).astype(np.uint8)


def fsc_template(row, box_index):
    image = cv2.imread(row['image'])
    if image is None:
        raise ValueError('Cannot read FSC image '+row['image'])
    box = np.asarray(row['annotation']['box_examples_coordinates'][box_index])
    x0, y0 = np.floor(box.min(axis=0)).astype(int)
    x1, y1 = np.ceil(box.max(axis=0)).astype(int)
    patch = image[y0:y1, x0:x1].astype(float)
    # Round-35 control recipe (compressed intensities), then the runner luma.
    return gray(np.uint8(np.rint(patch*.65+45))), [int(x0), int(y0), int(x1), int(y1)]


def procedural(seed, sigma, size=None):
    rng = np.random.default_rng(seed)
    w, h = size if size else (int(rng.integers(24, 73)), int(rng.integers(24, 73)))
    pad = int(math.ceil(4*sigma))
    noise = rng.standard_normal((h+2*pad, w+2*pad))
    smooth = cv2.GaussianBlur(noise, (0, 0), sigma)[pad:pad+h, pad:pad+w]
    smooth = (smooth-smooth.mean())/smooth.std()
    return np.uint8(np.clip(np.rint(128+40*smooth), 0, 255))


def layout(w, h, seed):
    rng = np.random.default_rng(seed)
    cell = int(math.ceil(math.hypot(w, h)))+30
    background = float(rng.uniform(*BACKGROUND_RANGE))
    objects = []
    for i in range(OBJECTS):
        objects.append(dict(index=i, pos_x=(i % 7+.5)*cell+float(rng.uniform(-4, 4)),
                            pos_y=(i//7+.5)*cell+float(rng.uniform(-4, 4)),
                            angle=float(rng.uniform(-175, 175)), scale=1.))
    return cell*7, background, objects


def render(templ, size, background, objects):
    """Round-35 renderer: bilinear warp about (w/2, h/2) and alpha compositing."""
    h, w = templ.shape
    image = np.full((size, size), background, np.float32)
    for obj in objects:
        theta = math.radians(obj['angle'])
        matrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        affine = np.column_stack((matrix, np.array([obj['pos_x'], obj['pos_y']])-matrix@[w/2, h/2]))
        alpha = cv2.warpAffine(np.ones((h, w), np.float32), affine, (size, size), flags=cv2.INTER_LINEAR)
        pixels = cv2.warpAffine(templ.astype(np.float32), affine, (size, size), flags=cv2.INTER_LINEAR)
        image = image*(1-alpha)+pixels
    return image


def finish(image, noise_seed=None):
    if noise_seed is not None:
        image = image+np.random.default_rng(noise_seed).normal(0, NOISE_SIGMA, image.shape)
    return np.uint8(np.clip(np.rint(image), 0, 255))


def templates(split, fsc_manifest, repo):
    fsc = {row['id']: row for row in json.loads(fsc_manifest.read_text())['images']}
    out = []
    for identifier in FSC_IMAGES[split]:
        for box in range(3):
            pixels, crop = fsc_template(fsc[identifier], box)
            out.append(dict(name=f'fsc{identifier}b{box}', family='fsc', pixels=pixels, fsc_id=identifier,
                            box_index=box, crop_xyxy=crop, category=fsc[identifier]['category']))
    for k, seed in enumerate(PROCEDURAL_SEEDS[split]):
        sigma = PROCEDURAL_SIGMAS[k]
        out.append(dict(name=f'proc{seed}', family='procedural', pixels=procedural(seed, sigma), seed=seed, sigma=sigma))
    for name in TEST_IMAGES[split]:
        path = repo/'Test Images'/name
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        out.append(dict(name=Path(name).stem.lower(), family='test_images', pixels=gray(image), file=str(path), file_sha256=sha(path)))
    return out


def generate(split, fsc_manifest, stage35_inputs, output, repo, protocol=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite frozen scenes')
    provenance = dict(split=split, generator_sha256=sha(__file__), fsc_manifest=str(fsc_manifest),
                      fsc_manifest_sha256=sha(fsc_manifest), opencv=cv2.__version__, numpy=np.__version__,
                      stage35_inputs=str(stage35_inputs))
    if split == 'holdout':
        if protocol is None:
            raise ValueError('The holdout split needs the frozen protocol')
        frozen = json.loads(protocol.read_text())
        if frozen.get('scenes', {}).get('generator_sha256') != sha(__file__):
            raise ValueError('Protocol does not freeze this generator')
        provenance.update(protocol=str(protocol), protocol_sha256=sha(protocol))
    (output/'templates').mkdir(parents=True)
    (output/'scenes').mkdir()
    cases = []
    for index, item in enumerate(templates(split, fsc_manifest, repo)):
        pixels = item.pop('pixels')
        h, w = pixels.shape
        tpath = output/'templates'/(item['name']+'.png')
        cv2.imwrite(str(tpath), pixels)
        seed = SCENE_SEED[split]+index
        size, background, objects = layout(w, h, seed)
        clean = render(pixels, size, background, objects)
        distractor = procedural(DISTRACTOR_SEED[split]+index, 2.0, (w, h))
        _, _, decoys = layout(w, h, seed+1000)
        scenes = dict(clean=(finish(clean), objects), noise=(finish(clean, seed+2000), objects),
                      empty=(finish(np.full((size, size), background, np.float32), seed+3000), []),
                      distractor=(finish(render(distractor, size, background, decoys), seed+4000), []))
        for condition, (image, truth) in scenes.items():
            spath = output/'scenes'/f"{item['name']}-{condition}.png"
            cv2.imwrite(str(spath), image)
            cases.append(dict(name=f"{item['name']}_{condition}", split=split, template_name=item['name'],
                              condition=condition, positive=bool(truth), template=str(tpath), source=str(spath),
                              template_size=[w, h], source_size=[size, size], background=background,
                              layout_seed=seed, noise_sigma=0 if condition == 'clean' else NOISE_SIGMA,
                              truth=truth, expected_count=len(truth),
                              **{key: value for key, value in item.items() if key != 'name'}))
    stage35 = {case['name']: case for case in json.loads((stage35_inputs/'cases.json').read_text())}
    for name in STAGE35_CASES[split]:
        case = stage35[name]
        for key, digest in case['input_sha256'].items():
            if sha(case[key]) != digest:
                raise ValueError(f'Round-35 input changed: {name} {key}')
        templ = cv2.imread(case['template'])
        cases.append(dict(name='stage35_'+name, split=split, template_name='stage35_'+name.split('_')[0],
                          condition=case['condition'], positive=True, template=case['template'], source=case['source'],
                          template_size=[templ.shape[1], templ.shape[0]], truth=case['truth'], expected_count=49,
                          family='stage35', fsc_id=case['source_fsc_id'], crop_xyxy=case['template_crop_xyxy'],
                          layout_seed=case['seed'], round35_case=name))
    for case in cases:
        case['input_sha256'] = {key: sha(case[key]) for key in ('source', 'template')}
    save(output/'cases.json', cases)
    save(output/'provenance.json', provenance)
    print(output/'cases.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['development', 'holdout'], required=True)
    p.add_argument('--fsc-manifest', type=Path, required=True)
    p.add_argument('--stage35-inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path)
    a = p.parse_args()
    generate(a.split, a.fsc_manifest.resolve(), a.stage35_inputs.resolve(), a.output.resolve(),
             Path(__file__).resolve().parents[2], a.protocol.resolve() if a.protocol else None)
