"""Test-bed v1: procedural textures, synthetic RGB same-colour scenes, finalization.

Subcommands (run with the python-port venv from the repository root):
  textures  --output DIR                  RGB capsule textures (base colour plus CIEDE2000-
                                          controlled variants) and a 180-degree symmetric gray texture
  synthetic --textures DIR --output DIR   OpenCV RGB scenes under illumination conditions, with GT
  finalize  --rendered DIR --synthetic DIR --output DIR
                                          crop templates, validate GT, write manifest and example jobs

Ground truth lives only in *.truth.json files and is never written into a job.
Colour labels are defined once, on the textures under the neutral reference
illuminant. Illumination conditions change the light, never the object colour,
so a same-colour instance stays same-colour under every condition.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from many_object_scenes import _corners, _linear, _roi, validate_placements  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEXTURE_W, TEXTURE_H = 128, 80
COLOUR_FIELD = (3, 3, 70, 77)          # x0, y0, x1, y1 inside the texture, uniform target colour
FAMILIES = {'red': (196, 58, 48), 'teal': (40, 140, 150)}
TARGETS = (2., 4., 8., 16.)
LIGHTNESS_TARGETS = (4., 8.)
CELL, COLUMNS, ROWS, MARGIN_TOP = 256, 8, 4, 128
CONDITIONS = {
    'neutral': dict(kind='uniform', gain=[1., 1., 1.]),
    'exposure_dark': dict(kind='uniform', gain=[.6, .6, .6]),
    'exposure_bright': dict(kind='uniform', gain=[1.35, 1.35, 1.35]),
    'wb_warm': dict(kind='uniform', gain=[1.18, 1., .78]),
    'wb_cool': dict(kind='uniform', gain=[.82, 1., 1.22]),
    'shading': dict(kind='horizontal_gradient', left=.55, right=1.15),
    'spot_warm': dict(kind='gaussian_spot', centre=[.3, .55], sigma=.22, peak=[1.45, 1.05, .65]),
    'noise_jpeg': dict(kind='uniform', gain=[1., 1., 1.], noise_std=4., jpeg_quality=75),
}
SEEDS = dict(texture=2026102401, layout={'red': 2026102402, 'teal': 2026102403}, noise=2026102404, background=2026102405)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(f'cannot write {path}')


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f'cannot read {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- colour science
def srgb_to_linear(u):
    # Odd extension below zero keeps bisection over out-of-gamut trial colours finite.
    u = np.asarray(u, np.float64)
    a = np.abs(u)
    return np.sign(u) * np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(np.asarray(v, np.float64), 0, None)
    return np.where(v <= .0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - .055)


_RGB_TO_XYZ = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
_WHITE = np.array([.95047, 1., 1.08883])


def rgb8_to_lab(rgb):
    xyz = srgb_to_linear(np.asarray(rgb, np.float64) / 255.) @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    return 255. * linear_to_srgb(xyz @ np.linalg.inv(_RGB_TO_XYZ).T)


def ciede2000(lab1, lab2):
    """CIEDE2000 (Sharma, Wu, Dalal 2005), kL = kC = kH = 1."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    c_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = .5 * (1 - math.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.
    dl, dc = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dh = 0.
    else:
        dh = h2p - h1p
        dh = dh - 360 if dh > 180 else dh + 360 if dh < -180 else dh
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    l_bar, c_barp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar = (h1p + h2p) / 2
    else:
        h_bar = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - .17 * math.cos(math.radians(h_bar - 30)) + .24 * math.cos(math.radians(2 * h_bar))
         + .32 * math.cos(math.radians(3 * h_bar + 6)) - .20 * math.cos(math.radians(4 * h_bar - 63)))
    d_theta = 30 * math.exp(-((h_bar - 275) / 25) ** 2)
    rc = 2 * math.sqrt(c_barp ** 7 / (c_barp ** 7 + 25 ** 7))
    sl = 1 + .015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    sc, sh = 1 + .045 * c_barp, 1 + .015 * c_barp * t
    rt = -math.sin(math.radians(2 * d_theta)) * rc
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dH / sh) ** 2 + rt * (dc / sc) * (dH / sh))


def gray8(rgb):
    """fpm::grayFromRgb: (77R + 150G + 29B + 128) >> 8 on 8-bit sRGB."""
    rgb = np.asarray(rgb, np.int64)
    return (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2] + 128) >> 8


def _bisect(function, lo, hi, target, iterations=80):
    flo = function(lo) - target
    for _ in range(iterations):
        mid = (lo + hi) / 2
        fmid = function(mid) - target
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return (lo + hi) / 2


def _in_gamut(rgb):
    return bool(np.all(np.asarray(rgb) >= 0) and np.all(np.asarray(rgb) <= 255))


def colour_variant(base_rgb, direction, target):
    """Return an in-gamut 8-bit colour at CIEDE2000 ~target from base along direction."""
    base_lab = rgb8_to_lab(base_rgb)
    L, a, b = base_lab
    chroma, hue = math.hypot(a, b), math.atan2(b, a)
    if direction in ('hue+', 'hue-'):
        sign = 1 if direction == 'hue+' else -1
        make = lambda t: lab_to_rgb([L, chroma * math.cos(hue + sign * t), chroma * math.sin(hue + sign * t)])
        param = _bisect(lambda t: ciede2000(base_lab, rgb8_to_lab(make(t))), 0., math.pi / 2, target)
    elif direction == 'chroma-':
        make = lambda t: lab_to_rgb([L, (1 - t) * a, (1 - t) * b])
        param = _bisect(lambda t: ciede2000(base_lab, rgb8_to_lab(make(t))), 0., 1., target)
    elif direction == 'lightness+':
        make = lambda t: lab_to_rgb([L + t, a, b])
        param = _bisect(lambda t: ciede2000(base_lab, rgb8_to_lab(make(t))), 0., 40., target)
    elif direction in ('isogray0', 'isogray90'):
        # Move inside the plane 77R + 150G + 29B = const, so the repo's gray8 is unchanged.
        # The positive axis direction is tried first; the negative one if that leaves the gamut.
        n = np.array([77., 150., 29.]); n /= np.linalg.norm(n)
        u = np.cross(n, [0., 0., 1.]); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        axis = u if direction == 'isogray0' else v
        for sign in (1., -1.):
            d = sign * axis
            make = lambda t, d=d: np.asarray(base_rgb, np.float64) + t * d
            param = _bisect(lambda t: ciede2000(base_lab, rgb8_to_lab(make(t))), 0., 120., target)
            if _in_gamut(np.rint(make(param))):
                break
    else:
        raise ValueError(direction)
    rgb = np.rint(make(param)).astype(np.int64)
    if direction.startswith('isogray'):
        rgb = _repair_gray(rgb, int(gray8(base_rgb)))
    if not _in_gamut(rgb):
        raise ValueError(f'{direction} {target} leaves the sRGB gamut: {rgb.tolist()}')
    return rgb


def _repair_gray(rgb, gray):
    """Smallest integer nudge (green first) that restores the exact gray8 value."""
    for delta in sorted(((dr, dg, db) for dr in range(-2, 3) for dg in range(-2, 3) for db in range(-2, 3)),
                        key=lambda d: (abs(d[0]) + abs(d[1]) + abs(d[2]), abs(d[0]) + abs(d[2]))):
        candidate = np.clip(rgb + delta, 0, 255)
        if int(gray8(candidate)) == gray:
            return candidate
    raise ValueError('cannot restore gray8 value')


# ---------------------------------------------------------------- textures
def capsule_texture(colour, seed=SEEDS['texture']):
    """128x80 label: uniform colour field (left), printed marks on an off-white panel (right)."""
    rng = np.random.default_rng(seed)  # same seed for every colour: identical non-colour content
    image = np.empty((TEXTURE_H, TEXTURE_W, 3), np.uint8)
    image[:] = (38, 38, 40)
    x0, y0, x1, y1 = COLOUR_FIELD
    image[y0:y1, x0:x1] = colour
    image[3:77, 73:125] = (236, 234, 226)
    for row in range(5):
        x = 77
        while x < 120:
            width = int(rng.integers(2, 9))
            if rng.random() < .75:
                image[8 + row * 14:8 + row * 14 + int(rng.integers(5, 10)), x:min(x + width, 121)] = (52, 50, 56)
            x += width + int(rng.integers(1, 4))
    cv2.circle(image, (36, 40), 13, (236, 234, 226), 2, lineType=cv2.LINE_8)
    cv2.line(image, (28, 40), (44, 40), (236, 234, 226), 2, lineType=cv2.LINE_8)
    return image


def make_textures(output):
    output = Path(output)
    records = []
    for family, base in FAMILIES.items():
        base = np.array(base, np.int64)
        base_lab = rgb8_to_lab(base)
        variants = [('base', 0., base)]
        for direction in ('hue+', 'hue-', 'chroma-', 'isogray0', 'isogray90'):
            variants += [(direction, t, colour_variant(base, direction, t)) for t in TARGETS]
        variants += [('lightness+', t, colour_variant(base, 'lightness+', t)) for t in LIGHTNESS_TARGETS]
        base_texture = capsule_texture(base)
        for direction, target, rgb in variants:
            name = f'{family}_base' if direction == 'base' else f'{family}_{direction}_{target:g}'
            texture = capsule_texture(rgb)
            path = output / f'{name}.png'
            write_png(path, texture)
            records.append(dict(name=name, family=family, direction=direction, same_colour=direction == 'base',
                delta_e00_target=target, delta_e00_achieved=round(ciede2000(base_lab, rgb8_to_lab(rgb)), 4),
                field_rgb=[int(c) for c in rgb], field_gray8=int(gray8(rgb)), base_gray8=int(gray8(base)),
                gray_texture_identical_to_base=bool(np.array_equal(gray8(texture), gray8(base_texture))),
                file=path.name, sha256=sha256(path)))
    gray = cv2.imread(str(ROOT / 'Test Images/Dst3.bmp'), cv2.IMREAD_GRAYSCALE)
    symmetric = np.rint((gray.astype(np.float64) + np.rot90(gray, 2)) / 2).astype(np.uint8)
    write_png(output / 'gray_dst3.png', gray)
    write_png(output / 'gray_dst3_symmetric180.png', symmetric)
    save(output / 'textures.json', dict(texture_size=[TEXTURE_W, TEXTURE_H], colour_field_xyxy=list(COLOUR_FIELD),
        families={k: list(v) for k, v in FAMILIES.items()}, delta_e='CIEDE2000, sRGB D65, kL=kC=kH=1',
        gray8='fpm::grayFromRgb (77R+150G+29B+128)>>8', seed=SEEDS['texture'], textures=records,
        gray_textures=[dict(file='gray_dst3.png', source='Test Images/Dst3.bmp', sha256=sha256(output / 'gray_dst3.png')),
                       dict(file='gray_dst3_symmetric180.png', rule='round((T + rot180(T)) / 2)',
                            sha256=sha256(output / 'gray_dst3_symmetric180.png'))]))
    return records


# ---------------------------------------------------------------- synthetic scenes
def layout(family, records):
    """32 objects on an 8x4 grid below a reference-card strip: 10 base + every variant once."""
    rng = np.random.default_rng(SEEDS['layout'][family])
    names = [f'{family}_base'] * 10 + [r['name'] for r in records if r['family'] == family and r['direction'] != 'base']
    if len(names) != COLUMNS * ROWS:
        raise ValueError(f'{len(names)} instances do not fill the grid')
    names = [names[i] for i in rng.permutation(len(names))]
    placements = []
    for index, name in enumerate(names):
        row, column = divmod(index, COLUMNS)
        centre = np.array([(column + .5) * CELL, MARGIN_TOP + (row + .5) * CELL]) + rng.uniform(-12, 12, 2)
        angle = float(rng.uniform(-180., 180.))
        scale = float(np.exp(rng.uniform(math.log(.85), math.log(1.15))))
        corners = _corners(centre, angle, scale, (TEXTURE_H, TEXTURE_W))
        placements.append(dict(index=index, texture=name, pos_x=float(centre[0]), pos_y=float(centre[1]),
                               angle=angle, scale=scale, corners=corners, warp_roi=_roi(corners)))
    validate_placements(placements, scene_shape(), (TEXTURE_H, TEXTURE_W))
    return placements


def scene_shape():
    return (MARGIN_TOP + ROWS * CELL, COLUMNS * CELL)


REFERENCE_CARD = dict(xyxy=[24, 16, 312, 112], patches=[('white', .90), ('gray18', .18), ('black', .03)])


def illumination(condition, shape):
    spec = CONDITIONS[condition]
    height, width = shape
    if spec['kind'] == 'uniform':
        return np.broadcast_to(np.array(spec['gain'], np.float32), (height, width, 3))
    if spec['kind'] == 'horizontal_gradient':
        ramp = np.linspace(spec['left'], spec['right'], width, dtype=np.float32)
        return np.broadcast_to(ramp[None, :, None], (height, width, 3))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = spec['centre'][0] * width, spec['centre'][1] * height
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (spec['sigma'] * width) ** 2))
    peak = np.array(spec['peak'], np.float32)
    return 1 + g[..., None] * (peak - 1)


def render_rgb_scene(placements, textures, condition, *, noise_seed, background_seed):
    """Linear-light compositing, illumination in linear RGB, then 8-bit sRGB with sensor noise."""
    shape = scene_shape()
    height, width = shape
    rng = np.random.default_rng(background_seed)
    texture_noise = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), 9) * 18
    background = np.clip(118 + texture_noise, 0, 255)
    canvas = np.repeat(srgb_to_linear(background / 255.).astype(np.float32)[..., None], 3, 2)
    x0, y0, x1, y1 = REFERENCE_CARD['xyxy']
    step = (x1 - x0) // len(REFERENCE_CARD['patches'])
    for k, (_, value) in enumerate(REFERENCE_CARD['patches']):
        canvas[y0:y1, x0 + k * step:x0 + (k + 1) * step] = value
    for item in placements:
        texture = srgb_to_linear(textures[item['texture']] / 255.).astype(np.float32)
        rx0, ry0, rx1, ry1 = item['warp_roi']
        linear = _linear(item['angle'], item['scale'])
        translation = np.array([item['pos_x'] - rx0, item['pos_y'] - ry0]) - linear @ [TEXTURE_W / 2, TEXTURE_H / 2]
        matrix = np.column_stack((linear, translation))
        patch = cv2.warpAffine(texture, matrix, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ones((TEXTURE_H, TEXTURE_W), np.float32), matrix, (rx1 - rx0, ry1 - ry0),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        region = canvas[ry0:ry1, rx0:rx1]
        region *= 1 - mask[..., None]
        region += patch
    canvas *= illumination(condition, shape)
    spec = CONDITIONS[condition]
    srgb = linear_to_srgb(canvas) * 255.
    noise = np.random.default_rng(noise_seed).standard_normal(srgb.shape) * spec.get('noise_std', 1.5)
    image = np.clip(np.rint(srgb + noise), 0, 255).astype(np.uint8)
    if 'jpeg_quality' in spec:
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, spec['jpeg_quality']])
        image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


def make_synthetic(textures_dir, output):
    textures_dir, output = Path(textures_dir), Path(output)
    meta = json.loads((textures_dir / 'textures.json').read_text())
    records = {r['name']: r for r in meta['textures']}
    textures = {name: read_rgb(textures_dir / r['file']).astype(np.float64) for name, r in records.items()}
    scenes = []
    for family in FAMILIES:
        placements = layout(family, list(records.values()))
        reference = [dict(index=0, texture=f'{family}_base', pos_x=CELL * 1.5, pos_y=MARGIN_TOP + CELL * .5,
                          angle=0., scale=1., corners=_corners([CELL * 1.5, MARGIN_TOP + CELL * .5], 0., 1., (TEXTURE_H, TEXTURE_W)))]
        reference[0]['warp_roi'] = _roi(reference[0]['corners'])
        ref_image = render_rgb_scene(reference, textures, 'neutral', noise_seed=SEEDS['noise'] + 99, background_seed=SEEDS['background'])
        cx, cy = int(reference[0]['pos_x']), int(reference[0]['pos_y'])
        template = ref_image[cy - TEXTURE_H // 2:cy + TEXTURE_H // 2, cx - TEXTURE_W // 2:cx + TEXTURE_W // 2]
        write_png(output / f'syn_{family}_template.png', template)
        for k, condition in enumerate(CONDITIONS):
            image = render_rgb_scene(placements, textures, condition, noise_seed=SEEDS['noise'] + k,
                                     background_seed=SEEDS['background'])
            name = f'syn_{family}_{condition}'
            write_png(output / f'{name}.png', image)
            truth = []
            for item in placements:
                r = records[item['texture']]
                truth.append(dict(index=item['index'], pos_x=item['pos_x'], pos_y=item['pos_y'], angle=item['angle'],
                    scale=item['scale'], corners=item['corners'], texture=r['name'], same_colour=r['same_colour'],
                    direction=r['direction'], delta_e00_target=r['delta_e00_target'],
                    delta_e00_achieved=r['delta_e00_achieved'], gray_texture_identical_to_base=r['gray_texture_identical_to_base']))
            save(output / f'{name}.truth.json', dict(scene=name, family=family, condition=condition,
                illumination=CONDITIONS[condition], image=f'{name}.png', image_sha256=sha256(output / f'{name}.png'),
                template=f'syn_{family}_template.png', template_crop_xywh=[cx - TEXTURE_W // 2, cy - TEXTURE_H // 2, TEXTURE_W, TEXTURE_H],
                source_shape=list(scene_shape()), reference_card=dict(xyxy=REFERENCE_CARD['xyxy'],
                patches=[dict(name=n, linear_reflectance=v) for n, v in REFERENCE_CARD['patches']]),
                pose_convention='many_object_scenes._corners: x=cx+s(cos u+sin v), y=cy+s(-sin u+cos v); CCW positive in image coordinates',
                truth=truth))
            scenes.append(name)
    save(output / 'synthetic.json', dict(scenes=scenes, conditions=CONDITIONS, seeds=SEEDS,
        script_sha256=sha256(__file__), renderer='OpenCV warpAffine INTER_LINEAR in linear light, sRGB 8-bit output'))
    return scenes


# ---------------------------------------------------------------- finalize
def analytic_corners(item, width, height):
    return np.asarray(_corners([item['pos_x'], item['pos_y']], item['angle'], item['scale'], (height, width)))


def finalize(rendered, synthetic, output):
    rendered, synthetic, output = Path(rendered), Path(synthetic), Path(output)
    report = dict(blender=[], synthetic=[], jobs=[])
    if rendered.exists():
        for meta_path in sorted(rendered.glob('bl_*.json')):
            meta = json.loads(meta_path.read_text())
            width, height = meta['template_plane_pixels']
            worst = 0.
            if meta['projection'] == 'orthographic frontal':
                for item in meta['objects']:
                    worst = max(worst, float(np.max(np.abs(analytic_corners(item, width, height) - np.asarray(item['corners'])))))
                if worst > 1e-3:
                    raise ValueError(f'{meta_path.name}: projected corners disagree with analytic pose ({worst})')
            if meta.get('template_crop_xywh'):
                x, y, w, h = meta['template_crop_xywh']
                image = read_rgb(rendered / meta['image'])
                write_png(output / f"{meta['name']}_template.png", image[y:y + h, x:x + w])
            report['blender'].append(dict(scene=meta['name'], projection=meta['projection'], objects=len(meta['objects']),
                                          max_corner_error_px=worst if meta['projection'] == 'orthographic frontal' else None))
    for truth_path in sorted(synthetic.glob('syn_*.truth.json')):
        meta = json.loads(truth_path.read_text())
        for item in meta['truth']:
            if not np.allclose(analytic_corners(item, TEXTURE_W, TEXTURE_H), item['corners'], atol=1e-9):
                raise ValueError(f'{truth_path.name}: corner/pose mismatch')
        report['synthetic'].append(dict(scene=meta['scene'], objects=len(meta['truth']),
            same_colour=sum(t['same_colour'] for t in meta['truth'])))
    files = sorted(p for d in (rendered, synthetic, output) if d.exists() for p in d.rglob('*') if p.is_file() and p.name != 'manifest.json')
    save(output / 'manifest.json', dict(generator_sha256=sha256(__file__), report=report,
        files={str(p): sha256(p) for p in files}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('textures'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('synthetic'); p.add_argument('--textures', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('finalize'); p.add_argument('--rendered', type=Path, required=True)
    p.add_argument('--synthetic', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'textures':
        records = make_textures(args.output)
        print(json.dumps([dict(name=r['name'], dE=r['delta_e00_achieved'], gray_identical=r['gray_texture_identical_to_base']) for r in records]))
    elif args.command == 'synthetic':
        print(json.dumps(make_synthetic(args.textures, args.output)))
    else:
        print(json.dumps(finalize(args.rendered, args.synthetic, args.output), indent=1))


if __name__ == '__main__':
    main()
