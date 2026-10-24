"""Colour-normalization probe on test-bed v1: raw chroma vs reference card vs object-internal white.

Usage: python-port/venv/bin/python cpp-core/tools/testbed_colour_probe.py build/testbed-v1


Uses GT poses to align each object (oracle localization: localization was 100% in these scenes),
so it isolates the colour decision. Threshold per family = p95 of same-colour distance under the
neutral/white condition only, then applied unchanged to every other illumination.
"""
import json
from pathlib import Path
import sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import testbed_generate as tb  # noqa: E402
D = Path(sys.argv[1])
lin = lambda im: tb.srgb_to_linear(im / 255.)
base = {f: tb.capsule_texture(c) for f, c in tb.FAMILIES.items()}
def masks(tex, colour):
    field = np.all(tex == colour, -1); panel = np.all(tex == (236, 234, 226), -1)
    k = np.ones((5, 5), np.uint8)
    return cv2.erode(field.astype(np.uint8), k) > 0, cv2.erode(panel.astype(np.uint8), k) > 0
def unwarp(image_lin, item, w=tb.TEXTURE_W, h=tb.TEXTURE_H):
    linear = tb._linear(item['angle'], item['scale']); t = np.array([item['pos_x'], item['pos_y']]) - linear @ [w / 2, h / 2]
    M = np.column_stack((linear, t)); return cv2.warpAffine(image_lin.astype(np.float32), cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_LINEAR)
def features(patch, fm, pm, card=None):
    field, panel = patch[fm].mean(0), np.median(patch[pm], 0)
    chroma = lambda v: v / v.sum()
    raw = chroma(field); obj = chroma(field / panel)
    cardn = chroma(field / card) if card is not None else None
    return raw, obj, cardn
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); return float(((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean()))
rows = []
for tp in sorted((D / 'synthetic').glob('syn_*.truth.json')):
    meta = json.loads(tp.read_text()); fam = meta['family']; fm, pm = masks(base[fam], tb.FAMILIES[fam])
    img = lin(tb.read_rgb(D / 'synthetic' / meta['image']).astype(float)); tmpl = lin(tb.read_rgb(D / 'synthetic' / meta['template']).astype(float))
    x0, y0, x1, y1 = tb.REFERENCE_CARD['xyxy']; step = (x1 - x0) // 3; card = np.median(img[y0 + 8:y1 - 8, x0 + 8:x0 + step - 8].reshape(-1, 3), 0)
    ref_card = np.full(3, .9)  # the template was rendered under neutral light with the same card value
    t_raw, t_obj, t_card = features(tmpl, fm, pm, ref_card)
    for item in meta['truth']:
        raw, obj, cardn = features(unwarp(img, item), fm, pm, card)
        rows.append(dict(src='syn', family=fam, cond=meta['condition'], same=item['same_colour'], direction=item['direction'],
            raw=np.linalg.norm(raw - t_raw), obj=np.linalg.norm(obj - t_obj), card=np.linalg.norm(cardn - t_card)))
for scene, cond in (('bl_rgb_white', 'neutral'), ('bl_rgb_warm3200', 'warm3200'), ('bl_rgb_mixed', 'mixed')):
    meta = json.loads((D / 'blender' / f'{scene}.json').read_text()); fm, pm = masks(base['red'], tb.FAMILIES['red'])
    img = lin(tb.read_rgb(D / 'blender' / meta['image']).astype(float)); tmpl = lin(tb.read_rgb(D / 'final' / 'bl_rgb_reference_template.png').astype(float))
    t_raw, t_obj, _ = features(tmpl, fm, pm)
    for o in meta['objects']:
        item = dict(pos_x=o['projected_center'][0], pos_y=o['projected_center'][1], angle=o['angle'], scale=o['scale'])
        raw, obj, _ = features(unwarp(img, item), fm, pm)
        rows.append(dict(src='blender', family='red', cond=cond, same=o['same_colour'], direction=o['direction'],
            raw=np.linalg.norm(raw - t_raw), obj=np.linalg.norm(obj - t_obj), card=np.nan))
for src in ('syn', 'blender'):
    R = [r for r in rows if r['src'] == src]
    methods = ('raw', 'card', 'obj') if src == 'syn' else ('raw', 'obj')
    thr = {(m, f): np.percentile([r[m] for r in R if r['family'] == f and r['cond'] == 'neutral' and r['same']], 95) for m in methods for f in {r['family'] for r in R}}
    print(f'== {src}: per condition, same accepted / different rejected / AUC  (threshold = neutral p95 per family)')
    for cond in dict.fromkeys(r['cond'] for r in R):
        C = [r for r in R if r['cond'] == cond]; S = [r for r in C if r['same']]; N = [r for r in C if not r['same']]
        cells = []
        for m in methods:
            acc = np.mean([r[m] <= thr[(m, r['family'])] for r in S]); rej = np.mean([r[m] > thr[(m, r['family'])] for r in N])
            cells.append(f"{m}: {acc:4.2f}/{rej:4.2f}/{auc([-r[m] for r in S], [-r[m] for r in N]):.3f}")
        print(f'  {cond:16s} ' + ' | '.join(cells))
    iso = [r for r in R if r['direction'].startswith('isogray')]
    if iso: print('  isogray rejected (all conditions):', {m: round(float(np.mean([r[m] > thr[(m, r['family'])] for r in iso])), 3) for m in methods})
