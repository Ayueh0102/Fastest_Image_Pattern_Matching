"""Stage 47: a pre-registered margin threshold rule for chroma_vk, evaluated on a freshly generated holdout.

Subcommands (python-port venv, repository root):
  thresholds  the catch-anchored thresholds from the label artwork only (no scene, no GT)
  holdout     regenerate the synthetic test-bed (same label artwork, new layout/noise/background seeds) and the
              bl_rgb Blender scenes (new Cycles, order and pose seeds); the artwork must be byte-identical to v1
  masks       template annotation masks (the stage-46 rule; the files must equal stage 46's)
  run         one fresh runner process per scene x method: the round-46 calibration-lighting scenes of test-bed v1
              and every scene of the fresh holdout; methods that existed before round 47 also run on the base runner
  summarize   read GT and the pre-registered protocol; write the summary with every gate and failure

Thresholds never use a holdout scene: the primary rule comes from the label artwork, the reported secondary rules
from the v1 calibration-lighting scenes only. Jobs never contain GT. No timing is measured or reported.

  PY=python-port/venv/bin/python
  $PY cpp-core/tools/benchmark_chroma_margin.py holdout --v1 build/testbed-v1 --output build/stage47-holdout \\
      --protocol cpp-core/docs/stage47-chroma-margin/protocol.json --blender /Applications/Blender.app/Contents/MacOS/Blender
  $PY cpp-core/tools/benchmark_chroma_margin.py masks --output build/stage47-margin/masks
  $PY cpp-core/tools/benchmark_chroma_margin.py run --v1 build/testbed-v1 --holdout build/stage47-holdout \\
      --runner build/stage47-release/experiments/fpm_color_verify_runner \\
      --base-runner build/stage47-base/experiments/fpm_color_verify_runner --masks build/stage47-margin/masks \\
      --protocol cpp-core/docs/stage47-chroma-margin/protocol.json --output build/stage47-margin/run/manifest.json
  $PY cpp-core/tools/benchmark_chroma_margin.py summarize --manifest build/stage47-margin/run/manifest.json \\
      --protocol cpp-core/docs/stage47-chroma-margin/protocol.json --output build/stage47-margin/summary.json
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_illuminant_colour as stage46  # noqa: E402
import benchmark_new_frame_cost as stage41  # noqa: E402
from many_object_accuracy import evaluate  # noqa: E402
import testbed_generate as tb  # noqa: E402

STAGE = 47
SEARCH, LIMIT = stage46.SEARCH, stage46.LIMIT
SYNTHETIC_SCALES, BLENDER_SCALES = stage46.SYNTHETIC_SCALES, stage46.BLENDER_SCALES
BLENDER = (('bl_rgb_white', 'white'), ('bl_rgb_warm3200', 'warm3200'), ('bl_rgb_mixed', 'mixed'))
CALIBRATION = ('neutral', 'exposure_dark', 'wb_cool', 'shading', 'white')   # round 46's calibration lighting
HOLDOUT = ('wb_warm', 'exposure_bright', 'spot_warm', 'noise_jpeg', 'warm3200', 'mixed')  # round 46's holdout lighting
GATED_SYNTHETIC = ('wb_warm', 'spot_warm', 'noise_jpeg')
GATED_BLENDER = ('warm3200', 'mixed')
GATED_DIRECTIONS = ('hue+', 'hue-', 'chroma-', 'isogray0', 'isogray90')
ISOGRAY = ('isogray0', 'isogray90')
GATED_MIN_DELTA_E = 4.
GROUPS = ('syn_red', 'syn_teal', 'bl_red')
GROUP_FAMILY = dict(syn_red='red', syn_teal='teal', bl_red='red')
# Primary rule: tau = CATCH_FRACTION x the smallest noise-free region chroma distance of a must-catch change.
CATCH_DELTA_E, CATCH_FRACTION, CATCH_STEP_DEG = 4., .5, 1
FEASIBILITY_FRACTION = .5               # calibration same-colour residuals must stay within tau/2
RAYLEIGH_FALSE_REJECT = 1e-3             # secondary rule
PERCENTILE = 95.                          # round-46 reference rule
TARGETS = dict(same_acceptance=.90, different_rejection=.95, isogray_rejection=.95, blender_same_acceptance=1.,
               blender_different_rejection=1., calibration_rejection=.95)
MIN_DECIDED = dict(synthetic=5, blender=2)  # decided (non-abstained) same-colour objects a gated cell needs
SIGN_TEST_ALPHA = .05
GRADIENT_CONDITIONS = ('spot_warm', 'mixed')
VK = stage46.VK
METHODS = {
    'gray_ncc': dict(method='gray_ncc'),
    'rgb_ncc': dict(method='rgb_ncc'),
    'chroma': dict(method='chroma', **stage46.CHROMA),
    'chroma_masked': dict(method='chroma', mask='colour', **stage46.CHROMA),
    'vk_object': dict(method='chroma_vk', mask='colour', reference_mask='reference', reference_clip_policy='rank', **VK),
    'vk_object_none': dict(method='chroma_vk', mask='colour', reference_mask='reference', reference_clip_policy='none', **VK),
    'card_gray': dict(method='chroma_vk', mask='colour', illuminant_reference='source_rect', reference_rect='gray18',
                      reference_clip_policy='rank', **VK),
    'vk_linear': dict(method='chroma_vk', mask='colour', reference_mask='reference', reference_clip_policy='rank',
                      reference_model='linear', **VK),
}
NEW_METHODS = ('vk_linear',)                                     # does not exist on the base runner
EXISTING = tuple(m for m in METHODS if m not in NEW_METHODS)     # must equal the base runner bit for bit
SYNTHETIC_ONLY = ('card_gray',)
ARMS = {
    'gray_ncc': ('gray_ncc', 'ncc'),
    'rgb_ncc': ('rgb_ncc', 'ncc'),
    'chroma': ('chroma', 'mean_distance'),
    'chroma_masked': ('chroma_masked', 'mean_distance'),
    'masked_region': ('vk_object_none', 'uncorrected_region_distance'),
    'vk_pixel': ('vk_object', 'mean_distance'),
    'vk_region': ('vk_object', 'region_distance'),
    'vk_none_region': ('vk_object_none', 'region_distance'),
    'card_gray_region': ('card_gray', 'region_distance'),
    'vk_linear_region': ('vk_linear', 'region_distance'),
    'vk_linear_pixel': ('vk_linear', 'mean_distance'),
}
NCC_ARMS = ('gray_ncc', 'rgb_ncc')   # not chromaticity distances: only the p95 rule applies
PRIMARY, GRADIENT = 'vk_region', 'vk_linear_region'
RULES = ('catch', 'rayleigh', 'p95')
FORBIDDEN_JOB_KEYS = stage46.FORBIDDEN_JOB_KEYS
sha, save = stage46.sha, stage46.save


# ---------------------------------------------------------------- primary threshold (artwork only)
def _linear(rgb):
    return tb.srgb_to_linear(np.asarray(rgb, np.float64) / 255.)


def region_chroma(a, b):
    """The chroma_vk distance between two linear RGB vectors: chromaticity Euclidean distance / sqrt(2)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(np.sum((a / a.sum() - b / b.sum()) ** 2) * .5))


def catch_distance(base):
    """Smallest noise-free distance of a must-catch change of the artwork colour `base` (8-bit sRGB):
    (a) CIEDE2000 = 4 on the constant-L* circle (1 degree steps, in-gamut, unrounded) and (b) every gated
    artwork variant (hue+/-, chroma-, isogray0/90 at target DeltaE >= 4, as testbed_generate renders them)."""
    base = np.asarray(base, np.float64)
    base_lab = tb.rgb8_to_lab(base)
    lightness, a, b = base_lab
    circle = []
    for degrees in range(0, 360, CATCH_STEP_DEG):
        angle = math.radians(degrees)
        make = lambda t: [lightness, a + t * math.cos(angle), b + t * math.sin(angle)]
        t = tb._bisect(lambda t: tb.ciede2000(base_lab, make(t)), 0., 60., CATCH_DELTA_E)
        rgb = tb.lab_to_rgb(make(t))
        if np.all(rgb >= 0) and np.all(rgb <= 255):
            circle.append(dict(degrees=degrees, distance=region_chroma(_linear(base), _linear(rgb))))
    variants = []
    for direction in GATED_DIRECTIONS:
        for target in tb.TARGETS:
            if target >= GATED_MIN_DELTA_E:
                rgb = tb.colour_variant(base.astype(np.int64), direction, target)
                variants.append(dict(direction=direction, target=target, rgb=[int(c) for c in rgb],
                                     distance=region_chroma(_linear(base), _linear(rgb))))
    closest_circle = min(circle, key=lambda c: c['distance'])
    closest_variant = min(variants, key=lambda v: v['distance'])
    return dict(circle_in_gamut=len(circle), circle_min=closest_circle, variant_min=closest_variant,
                distance=min(closest_circle['distance'], closest_variant['distance']))


def catch_thresholds():
    families = {family: catch_distance(colour) for family, colour in tb.FAMILIES.items()}
    return {group: dict(family=GROUP_FAMILY[group], tau=CATCH_FRACTION * families[GROUP_FAMILY[group]]['distance'],
                        **families[GROUP_FAMILY[group]]) for group in GROUPS}


# ---------------------------------------------------------------- fresh holdout
def make_holdout(v1, output, seeds, blender):
    """Regenerate the synthetic scenes and the bl_rgb scenes with the protocol's holdout seeds."""
    v1, output = Path(v1), Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing holdout directory')
    synthetic = seeds['synthetic']
    tb.SEEDS.update(layout=dict(synthetic['layout']), noise=synthetic['noise'], background=synthetic['background'])
    if tb.SEEDS['texture'] != synthetic['texture']:
        raise ValueError('The label artwork seed must stay the test-bed v1 seed')
    tb.make_textures(output / 'textures')
    artwork = artwork_differences(v1 / 'textures', output / 'textures')
    if artwork:
        raise ValueError(f'Label artwork differs from test-bed v1: {artwork}')
    tb.make_synthetic(output / 'textures', output / 'synthetic')
    rendered = output / 'blender'
    script = Path(__file__).resolve().parent / 'testbed_blender_render.py'
    b = seeds['blender']
    command = [str(blender), '--background', '--factory-startup', '--python', str(script), '--', '--textures',
               str(output / 'textures'), '--output', str(rendered), '--only', ','.join(name for name, _ in BLENDER),
               '--seed', str(b['cycles']), '--rgb-order-seed', str(b['order']), '--rgb-layout-seed', str(b['layout'])]
    child = subprocess.run(command, capture_output=True, text=True)
    (output / 'blender-render.log').write_text(child.stdout + child.stderr)
    if child.returncode:
        raise RuntimeError(f'Blender exited {child.returncode}')
    report = tb.finalize(rendered, output / 'synthetic', output / 'final')
    save(output / 'holdout.json', dict(stage=STAGE, seeds=seeds, command=command, generator_sha256=sha(tb.__file__),
        blender_script_sha256=sha(script), artwork_identical_to_v1=True, finalize=report,
        note='templates are taken from test-bed v1 (the learned model); the regenerated syn_*_template.png are unused'))
    return report


def artwork_differences(v1_textures, fresh_textures):
    v1_textures, fresh_textures = Path(v1_textures), Path(fresh_textures)
    names = sorted({p.name for p in v1_textures.iterdir()} | {p.name for p in fresh_textures.iterdir()})
    return [name for name in names if not (v1_textures / name).exists() or not (fresh_textures / name).exists()
            or sha(v1_textures / name) != sha(fresh_textures / name)]


# ---------------------------------------------------------------- inputs and jobs
def scenes(v1, fresh):
    """v1 calibration-lighting scenes (threshold calibration) and every fresh scene; templates always from v1."""
    v1, fresh = Path(v1), Path(fresh)
    out = []
    for dataset, root in (('v1', v1), ('r47', fresh)):
        for family in tb.FAMILIES:
            for condition in tb.CONDITIONS:
                if dataset == 'v1' and condition not in CALIBRATION:
                    continue
                name = f'syn_{family}_{condition}'
                out.append(dict(key=f'{dataset}/{name}', dataset=dataset, name=name, source='synthetic', family=family,
                                group=f'syn_{family}', condition=condition, split=split(dataset, condition),
                                image=root / 'synthetic' / f'{name}.png', template=v1 / 'synthetic' / f'syn_{family}_template.png',
                                truth=root / 'synthetic' / f'{name}.truth.json', scales=SYNTHETIC_SCALES))
        for name, condition in BLENDER:
            if dataset == 'v1' and condition not in CALIBRATION:
                continue
            out.append(dict(key=f'{dataset}/{name}', dataset=dataset, name=name, source='blender', family='red', group='bl_red',
                            condition=condition, split=split(dataset, condition), image=root / 'blender' / f'{name}.png',
                            template=v1 / 'final' / 'bl_rgb_reference_template.png', truth=root / 'blender' / f'{name}.json',
                            scales=BLENDER_SCALES))
    return out


def split(dataset, condition):
    if dataset == 'v1':
        if condition not in CALIBRATION:
            raise ValueError(f'v1 {condition} is round-46 evaluation data and is not used')
        return 'calibration'
    if condition in HOLDOUT:
        return 'holdout'
    if condition in CALIBRATION:
        return 'fresh_calibration'   # calibration lighting on the fresh layout: reported, never calibrates anything
    raise ValueError(condition)


def methods_for(scene):
    return [m for m in METHODS if scene['source'] == 'synthetic' or m not in SYNTHETIC_ONLY]


def job(scene, method, masks):
    spec = dict(METHODS[method])
    out = dict(source=str(Path(scene['image']).resolve()), template=str(Path(scene['template']).resolve()), backend='cpu',
               threads=1, warmups=0, repeats=1, search=dict(SEARCH, scales=scene['scales'], limit=LIMIT))
    for key in ('mask', 'reference_mask'):
        if key in spec:
            spec[key] = str((Path(masks) / f"{scene['family']}_{spec[key]}.png").resolve())
    if 'reference_rect' in spec:
        spec['reference_rect'] = stage46.CARD_RECTS[spec['reference_rect']]
    out.update(spec)
    return out


def gt_problems(job_dict):
    return stage46.gt_problems(job_dict)


def input_files(scene_list, masks):
    files = set()
    for scene in scene_list:
        files.update((Path(scene['image']).resolve(), Path(scene['template']).resolve()))
    files.update(path.resolve() for path in Path(masks).glob('*.png'))
    return sorted(files)


def check_inputs(inputs, testbeds):
    """Every test-bed input equals its own test-bed manifest (keyed 'subdirectory/file')."""
    for path, digest in inputs.items():
        owners = [(root, hashes) for root, hashes in testbeds if Path(path).is_relative_to(root)]
        if len(owners) > 1:
            raise ValueError(f'{path} is inside two test-beds')
        if owners and owners[0][1].get('/'.join(Path(path).parts[-2:])) != digest:
            raise ValueError(f'{path} differs from its test-bed manifest')


# ---------------------------------------------------------------- run
def run(args):
    root = args.output.resolve().parent
    if args.output.exists() or any((root / name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    scene_list = [s for s in scenes(args.v1, args.holdout) if not args.scene or s['key'] in args.scene]
    if args.scene and len(scene_list) != len(set(args.scene)):
        raise ValueError('Unknown scene in --scene')
    testbeds = []
    for testbed in (args.v1, args.holdout):
        hashes, generator = stage46.testbed_hashes(testbed)
        testbeds.append((Path(testbed).resolve(), hashes))
    inputs = {str(path): sha(path) for path in input_files(scene_list, args.masks)}
    check_inputs(inputs, testbeds)
    plan = [(scene, method, kind) for scene in scene_list for method in methods_for(scene)
            for kind in (('candidate', 'base') if method in EXISTING else ('candidate',))]
    host = stage41.host_state(samples=3)
    manifest = dict(version=1, stage=STAGE, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(args.protocol), git=stage46.git_head(), runner=str(args.runner.resolve()), runner_sha256=sha(args.runner),
        base_runner=str(args.base_runner.resolve()), base_runner_sha256=sha(args.base_runner),
        v1=str(Path(args.v1).resolve()), holdout=str(Path(args.holdout).resolve()),
        holdout_json_sha256=sha(Path(args.holdout) / 'holdout.json'), input_sha256=inputs,
        masks=str(Path(args.masks).resolve()), masks_json_sha256=sha(Path(args.masks) / 'masks.json'), scene_filter=args.scene, methods=METHODS, arms=ARMS,
        search=SEARCH, limit=LIMIT,
        scenes=[dict(scene, image=str(Path(scene['image']).resolve()), template=str(Path(scene['template']).resolve()),
                     truth=str(Path(scene['truth']).resolve())) for scene in scene_list],
        host=dict(before=host), executions=[], complete=False)
    save(args.output, manifest)
    try:
        for scene, method, kind in plan:
            label = f"{scene['dataset']}-{scene['name']}-{method}" + ('-base' if kind == 'base' else '')
            print('Running ' + label, flush=True)
            entry = stage46.run_worker(args.runner if kind == 'candidate' else args.base_runner, root, label,
                                       job(scene, method, args.masks), args.timeout)
            manifest['executions'].append(dict(scene=scene['key'], method=method, runner_kind=kind, **entry))
            save(args.output, manifest)
        manifest['host']['after'] = stage41.host_state(samples=3)
        after = dict(runner=sha(args.runner), base_runner=sha(args.base_runner), inputs={p: sha(p) for p in inputs})
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                        unchanged=after == dict(runner=manifest['runner_sha256'], base_runner=manifest['base_runner_sha256'], inputs=inputs))
        save(args.output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(args.output, manifest)
        raise


# ---------------------------------------------------------------- evaluation (GT read only here)
def scene_rows(scene, meta, results):
    """Per-GT-object rows; results maps method -> runner output (candidate runner)."""
    methods = list(results)
    matches = results[methods[0]]['first']['search_matches']
    truth = stage46.truth_objects(scene, meta)
    strict = evaluate(matches, [dict(pos_x=t['pos_x'], pos_y=t['pos_y'], angle=t['angle'], scale=t['scale']) for t in truth])['strict']
    mapping = {d['truth_index']: d['prediction_index'] for d in strict['details']}
    rows = []
    for k, item in enumerate(truth):
        p = mapping.get(k)
        row = dict(scene=scene['key'], dataset=scene['dataset'], source=scene['source'], group=scene['group'],
                   family=scene['family'], condition=scene['condition'], split=scene['split'], index=item['index'],
                   same_colour=bool(item['same_colour']), direction=item['direction'],
                   delta_e00_target=float(item['delta_e00_target']), delta_e00_achieved=float(item['delta_e00_achieved']),
                   gray_identical=bool(item['gray_texture_identical_to_base']), found=p is not None, prediction=p,
                   values={}, diagnostics={})
        if p is not None:
            for arm, (method, statistic) in ARMS.items():
                if method in results:
                    row['values'][arm] = stage46.dissimilarity(results[method]['first']['scores'][p], statistic)
            for method, result in results.items():
                score = result['first']['scores'][p]
                if isinstance(score, dict):
                    row['diagnostics'][method] = {key: score[key] for key in ('status', 'valid_fraction', 'clipped_pixels',
                        'reference_fraction', 'reference_clip_fraction', 'gain', 'reference_slope_u', 'reference_slope_v')
                        if key in score}
        rows.append(row)
    matched = set(mapping.values()) - {None}
    false_positives = [dict(scene=scene['key'], group=scene['group'], prediction=k,
                            values={arm: stage46.dissimilarity(results[method]['first']['scores'][k], statistic)
                                    for arm, (method, statistic) in ARMS.items() if method in results})
                       for k in range(len(matches)) if k not in matched]
    return rows, dict(tp=strict['tp'], fp=strict['fp'], fn=strict['fn'], predictions=len(matches)), false_positives


def calibration_values(rows, arm, group):
    return [r['values'][arm] for r in rows if r['split'] == 'calibration' and r['group'] == group and r['same_colour']
            and r['found'] and r['values'].get(arm) is not None]


def thresholds(rows, rule, catch):
    """{(arm, group): tau or None}. catch: artwork-only tau per group; rayleigh and p95: calibration split only."""
    out = {}
    for arm in ARMS:
        for group in GROUPS:
            if not any(arm in r['values'] for r in rows if r['group'] == group and r['found']):
                continue
            values = calibration_values(rows, arm, group)
            if rule == 'catch':
                out[(arm, group)] = None if arm in NCC_ARMS else catch[group]
            elif rule == 'rayleigh':
                scale = math.sqrt(float(np.mean(np.square(values))) / 2) if values else None
                out[(arm, group)] = None if arm in NCC_ARMS or scale is None else \
                    scale * math.sqrt(-2 * math.log(RAYLEIGH_FALSE_REJECT))
            elif rule == 'p95':
                out[(arm, group)] = float(np.percentile(values, PERCENTILE)) if values else None
            else:
                raise ValueError(rule)
    return out


def accepted(row, arm, table):
    value, threshold = row['values'].get(arm), table.get((arm, row['group']))
    return value is not None and threshold is not None and value <= threshold


def cell(rows, arm, table):
    found = [r for r in rows if r['found'] and arm in r['values']]
    same = [r for r in found if r['same_colour']]
    different = [r for r in found if not r['same_colour']]
    decided = [r for r in same if r['values'][arm] is not None]
    same_accepted = sum(accepted(r, arm, table) for r in same)
    wrong = [r for r in different if accepted(r, arm, table)]
    rate = lambda numerator, n: numerator / n if n else None
    return dict(same=dict(n=len(same), decided=len(decided), abstained=len(same) - len(decided), accepted=same_accepted,
                          acceptance_decided=rate(same_accepted, len(decided)), acceptance_end_to_end=rate(same_accepted, len(same)),
                          max_decided=max((r['values'][arm] for r in decided), default=None)),
                different=dict(n=len(different), wrongly_accepted=len(wrong), abstained=sum(r['values'][arm] is None for r in different),
                               rejection=rate(len(different) - len(wrong), len(different)),
                               min_decided=min((r['values'][arm] for r in different if r['values'][arm] is not None), default=None),
                               wrong=[dict(scene=r['scene'], index=r['index'], direction=r['direction'],
                                           delta_e00_target=r['delta_e00_target'], value=r['values'][arm]) for r in wrong]),
                auc=stage46.auc([r['values'][arm] for r in same], [r['values'][arm] for r in different]),
                auc_valid=stage46.auc([r['values'][arm] for r in same if r['values'][arm] is not None],
                                      [r['values'][arm] for r in different if r['values'][arm] is not None]),
                not_found=sum(not r['found'] for r in rows))


def direction_table(rows, arm, table):
    out = {}
    for r in rows:
        if not r['found'] or arm not in r['values']:
            continue
        key = 'same' if r['same_colour'] else f"{r['direction']}@{r['delta_e00_target']:g}"
        entry = out.setdefault(key, dict(n=0, accepted=0, abstained=0))
        entry['n'] += 1
        entry['accepted'] += bool(accepted(r, arm, table))
        entry['abstained'] += r['values'][arm] is None
    for key, entry in out.items():
        entry['rate'] = entry['accepted'] / entry['n'] if key == 'same' else 1 - entry['accepted'] / entry['n']
        entry['rate_kind'] = 'acceptance' if key == 'same' else 'rejection'
    return dict(sorted(out.items()))


def gated(row):
    return not row['same_colour'] and row['direction'] in GATED_DIRECTIONS and row['delta_e00_target'] >= GATED_MIN_DELTA_E


def check(name, value, target, n, kind='ge', **where):
    passed = value is not None and (value >= target if kind == 'ge' else value <= target)
    return dict(name=name, value=value, target=target, kind=kind, n=n, passed=passed, **where)


def hypotheses(rows, table, arm):
    """Pre-registered checks F1, F2, H1a-c, H2a-b on one arm; failures are kept. Decided = not abstained."""
    checks = []
    found = [r for r in rows if r['found'] and arm in r['values']]
    for group in GROUPS:
        tau = table.get((arm, group))
        same = calibration_values(rows, arm, group)
        checks.append(check('F1 calibration same-colour within tau/2', max(same) if same else None,
                            None if tau is None else FEASIBILITY_FRACTION * tau, len(same), kind='le', group=group))
        different = [r for r in found if r['split'] == 'calibration' and r['group'] == group and gated(r)]
        checks.append(check('F2 calibration gated different-colour rejection',
                            sum(not accepted(r, arm, table) for r in different) / len(different) if different else None,
                            TARGETS['calibration_rejection'], len(different), group=group))
    holdout = [r for r in found if r['split'] == 'holdout']
    for conditions, source, same_target, different_target, prefix in (
            (GATED_SYNTHETIC, 'synthetic', TARGETS['same_acceptance'], TARGETS['different_rejection'], 'H1'),
            (GATED_BLENDER, 'blender', TARGETS['blender_same_acceptance'], TARGETS['blender_different_rejection'], 'H2')):
        for condition in conditions:
            subset = [r for r in holdout if r['condition'] == condition and r['source'] == source]
            decided = [r for r in subset if r['same_colour'] and r['values'][arm] is not None]
            different = [r for r in subset if gated(r)]
            value = sum(accepted(r, arm, table) for r in decided) / len(decided) if decided else None
            result = check(f'{prefix}a decided same-colour acceptance', value, same_target, len(decided), condition=condition)
            result['min_decided'] = MIN_DECIDED[source]
            result['same_found'] = sum(r['same_colour'] for r in subset)
            result['passed'] = result['passed'] and len(decided) >= MIN_DECIDED[source]
            checks.append(result)
            checks.append(check(f'{prefix}b gated different-colour rejection',
                                sum(not accepted(r, arm, table) for r in different) / len(different) if different else None,
                                different_target, len(different), condition=condition))
    iso = [r for r in holdout if r['condition'] in GATED_SYNTHETIC and r['source'] == 'synthetic' and gated(r)
           and r['direction'] in ISOGRAY]
    checks.append(check('H1c isogray rejection (pooled)', sum(not accepted(r, arm, table) for r in iso) / len(iso) if iso else None,
                        TARGETS['isogray_rejection'], len(iso), condition='+'.join(GATED_SYNTHETIC)))
    return checks


def end_to_end(rows, table, arm):
    """Round-46 H1a/H2a definition (abstention = not accepted): reported, not part of the verdict."""
    out = []
    holdout = [r for r in rows if r['found'] and arm in r['values'] and r['split'] == 'holdout']
    for conditions, source, target in ((GATED_SYNTHETIC, 'synthetic', TARGETS['same_acceptance']),
                                       (GATED_BLENDER, 'blender', TARGETS['blender_same_acceptance'])):
        for condition in conditions:
            same = [r for r in holdout if r['condition'] == condition and r['source'] == source and r['same_colour']]
            result = check('E same-colour acceptance (abstention = not accepted)',
                           sum(accepted(r, arm, table) for r in same) / len(same) if same else None, target, len(same),
                           condition=condition)
            result['abstained'] = sum(r['values'][arm] is None for r in same)
            out.append(result)
    return out


def sign_test(wins, losses):
    """One-sided exact binomial P(X >= wins | n = wins + losses, p = 1/2); ties are excluded beforehand."""
    n = wins + losses
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def paired(rows, conditions, better=GRADIENT, worse=PRIMARY):
    """Same-colour holdout objects decided by both arms: does `better` give the smaller residual?"""
    pairs = [(r['values'][better], r['values'][worse]) for r in rows if r['found'] and r['same_colour'] and r['split'] == 'holdout'
             and r['condition'] in conditions and r['values'].get(better) is not None and r['values'].get(worse) is not None]
    wins, losses = sum(a < b for a, b in pairs), sum(a > b for a, b in pairs)
    ratios = [a / b for a, b in pairs if b > 0]
    return dict(conditions=list(conditions), n=len(pairs), wins=wins, losses=losses, ties=len(pairs) - wins - losses,
                p_one_sided=sign_test(wins, losses), median_ratio=float(np.median(ratios)) if ratios else None)


def integrity(manifest, root, scene_list):
    """Workers, repeats, localization identity, base identity, GT-free jobs and input hashes."""
    failures = []
    by_key, unreadable = {}, set()
    for entry in manifest['executions']:
        key = (entry['scene'], entry['method'], entry['runner_kind'])
        if key in by_key:
            failures.append(dict(gate='duplicate_execution', scene=entry['scene'], method=entry['method']))
        by_key[key] = entry
        job_dict = json.loads((root / entry['job']).read_text())
        if sha(root / entry['job']) != entry['job_sha256']:
            failures.append(dict(gate='job_hash', scene=entry['scene'], method=entry['method']))
        for problem in gt_problems(job_dict):
            failures.append(dict(gate='gt_in_job', scene=entry['scene'], method=entry['method'], detail=problem))
        if entry['returncode']:
            failures.append(dict(gate='worker_failed', scene=entry['scene'], method=entry['method'], runner=entry['runner_kind']))
        elif sha(root / entry['result']) != entry['result_sha256']:
            failures.append(dict(gate='result_hash', scene=entry['scene'], method=entry['method']))
            unreadable.add(key)   # a result that is not the recorded one is never evaluated
    results = {}
    for scene in scene_list:
        key = scene['key']
        results[key] = {}
        for method in methods_for(scene):
            entry = by_key.get((key, method, 'candidate'))
            if entry is None:
                failures.append(dict(gate='missing_worker', scene=key, method=method))
                continue
            if entry['returncode'] or (key, method, 'candidate') in unreadable:
                continue
            result = json.loads((root / entry['result']).read_text())
            if not result['exact_repeat']:
                failures.append(dict(gate='exact_repeat', scene=key, method=method))
            results[key][method] = result
            if method in EXISTING:
                base_entry = by_key.get((key, method, 'base'))
                if base_entry is None or base_entry['returncode'] or (key, method, 'base') in unreadable:
                    failures.append(dict(gate='missing_base_worker', scene=key, method=method))
                    continue
                reference = json.loads((root / base_entry['result']).read_text())
                if reference['first']['scores'] != result['first']['scores'] or \
                        reference['first']['search_matches'] != result['first']['search_matches']:
                    failures.append(dict(gate='existing_method_changed', scene=key, method=method))
        matches = [result['first']['search_matches'] for result in results[key].values()]
        if any(m != matches[0] for m in matches[1:]):
            failures.append(dict(gate='localization_differs_between_methods', scene=key))
    for path, digest in manifest['input_sha256'].items():
        if not Path(path).exists() or sha(path) != digest:
            failures.append(dict(gate='input_changed', path=path))
    if not manifest.get('complete') or not manifest.get('unchanged'):
        failures.append(dict(gate='run_incomplete_or_binaries_changed'))
    return results, failures


def holdout_failures(manifest, protocol):
    """The fresh holdout: protocol seeds, never-used seeds, byte-identical artwork, unchanged holdout record."""
    failures = []
    holdout, v1 = Path(manifest['holdout']), Path(manifest['v1'])
    if sha(holdout / 'holdout.json') != manifest['holdout_json_sha256']:
        failures.append(dict(gate='holdout_record_changed'))
    record = json.loads((holdout / 'holdout.json').read_text())
    seeds = protocol['data']['holdout_seeds']
    if record['seeds'] != seeds:
        failures.append(dict(gate='holdout_seeds_differ_from_protocol'))
    synthetic = json.loads((holdout / 'synthetic' / 'synthetic.json').read_text())['seeds']
    if synthetic != dict(texture=seeds['synthetic']['texture'], layout=seeds['synthetic']['layout'],
                         noise=seeds['synthetic']['noise'], background=seeds['synthetic']['background']):
        failures.append(dict(gate='synthetic_seeds_differ_from_protocol'))
    blender = json.loads((holdout / 'blender' / 'blender_generation.json').read_text())
    if (blender['seed'], blender['rgb_order_seed'], blender['rgb_layout_seed']) != \
            (seeds['blender']['cycles'], seeds['blender']['order'], seeds['blender']['layout']):
        failures.append(dict(gate='blender_seeds_differ_from_protocol'))
    used = set(protocol['data']['used_seeds'])
    fresh = [seeds['synthetic']['noise'], seeds['synthetic']['background'], *seeds['synthetic']['layout'].values(),
             seeds['blender']['cycles'], seeds['blender']['order'], seeds['blender']['layout']]
    if used & set(fresh) or len(set(fresh)) != len(fresh):
        failures.append(dict(gate='holdout_seed_reused'))
    for name in artwork_differences(v1 / 'textures', holdout / 'textures'):
        failures.append(dict(gate='artwork_differs_from_v1', file=name))
    return failures


def catch_failures(protocol, catch):
    """The catch thresholds recomputed from the artwork must equal the protocol's frozen values."""
    frozen = {group: value['tau'] for group, value in protocol['thresholds']['catch_tau'].items()}
    if set(frozen) != set(GROUPS) or any(abs(catch[g]['tau'] - frozen[g]) > 1e-12 for g in GROUPS):
        return [dict(gate='catch_threshold_differs_from_protocol', computed={g: catch[g]['tau'] for g in GROUPS}, protocol=frozen)]
    return []


def mask_failures(protocol, masks):
    return [dict(gate='mask_differs_from_stage46', file=name) for name, digest in protocol['masks']['files'].items()
            if not (Path(masks) / name).exists() or sha(Path(masks) / name) != digest]


def verdict(integrity_passed, evaluation):
    return dict(recommended=integrity_passed and evaluation['primary_passed'],
                gradient_preferred=integrity_passed and evaluation['gradient_checks_passed']
                and evaluation['gradient_better_on_gradients'])


def evaluate_rows(rows, catch, false_positives=()):
    tables = {rule: thresholds(rows, rule, catch) for rule in RULES}
    conditions = lambda group: dict.fromkeys((r['split'], r['condition']) for r in rows if r['group'] == group)
    per_condition = {rule: {arm: {group: {f'{s}/{c}': cell([r for r in rows if r['group'] == group and r['split'] == s
                                                             and r['condition'] == c], arm, tables[rule])
                                          for s, c in conditions(group)} for group in GROUPS}
                            for arm in ARMS} for rule in RULES}
    per_split = {rule: {arm: {group: {s: cell([r for r in rows if r['group'] == group and r['split'] == s], arm, tables[rule])
                                      for s in ('calibration', 'holdout', 'fresh_calibration')} for group in GROUPS}
                        for arm in ARMS} for rule in RULES}
    directions = {arm: {group: {s: direction_table([r for r in rows if r['group'] == group and r['split'] == s], arm, tables['catch'])
                                for s in ('calibration', 'holdout', 'fresh_calibration')} for group in GROUPS}
                  for arm in ARMS if arm not in NCC_ARMS}
    checks = {rule: {arm: hypotheses(rows, tables[rule], arm) for arm in (PRIMARY, GRADIENT)} for rule in RULES}
    primary, gradient = checks['catch'][PRIMARY], checks['catch'][GRADIENT]
    comparison = dict(gated=paired(rows, GRADIENT_CONDITIONS),
                      per_condition={c: paired(rows, (c,)) for c in HOLDOUT})
    gradient_better = comparison['gated']['p_one_sided'] is not None and comparison['gated']['p_one_sided'] < SIGN_TEST_ALPHA
    fp_decisions = [dict(fp, accepted={arm: accepted(dict(values=fp['values'], group=fp['group']), arm, tables['catch'])
                                       for arm in fp['values']}) for fp in false_positives]
    return dict(thresholds={rule: [dict(arm=a, group=g, value=v) for (a, g), v in sorted(table.items())]
                            for rule, table in tables.items()},
                per_condition=per_condition, per_split=per_split, directions=directions, hypotheses=checks,
                end_to_end={arm: end_to_end(rows, tables['catch'], arm) for arm in (PRIMARY, GRADIENT)},
                gradient_comparison=comparison, false_positive_decisions=fp_decisions,
                primary_passed=all(c['passed'] for c in primary), gradient_checks_passed=all(c['passed'] for c in gradient),
                gradient_better_on_gradients=gradient_better)


def summarize(manifest, root, protocol_path):
    if sha(protocol_path) != manifest['protocol_sha256']:
        raise ValueError('Protocol differs from the one recorded by the run')
    protocol = json.loads(Path(protocol_path).read_text())
    scene_list = manifest['scenes']
    results, failures = integrity(manifest, root, scene_list)
    failures += holdout_failures(manifest, protocol)
    catch = catch_thresholds()
    failures += catch_failures(protocol, catch) + mask_failures(protocol, manifest['masks'])
    rows, localization, false_positives = [], {}, []
    for scene in scene_list:
        if len(results[scene['key']]) != len(methods_for(scene)):
            continue
        meta = json.loads(Path(scene['truth']).read_text())
        scene_rows_, counts, fps = scene_rows(scene, meta, results[scene['key']])
        rows += scene_rows_
        false_positives += fps
        counts['all_found_no_fp'] = counts['tp'] == len(scene_rows_) and counts['fp'] == 0 and counts['fn'] == 0
        localization[scene['key']] = counts
        if scene['source'] == 'synthetic' and meta.get('reference_card', {}).get('xyxy') != list(tb.REFERENCE_CARD['xyxy']):
            failures.append(dict(gate='reference_card_moved', scene=scene['key']))
    evaluation = evaluate_rows(rows, {g: catch[g]['tau'] for g in GROUPS}, false_positives)
    integrity_passed = not failures
    return dict(stage=STAGE, protocol_sha256=manifest['protocol_sha256'], runner_sha256=manifest['runner_sha256'],
                base_runner_sha256=manifest['base_runner_sha256'], git=manifest['git'], catch=catch,
                split=dict(calibration=CALIBRATION, holdout=HOLDOUT, gated_synthetic=GATED_SYNTHETIC, gated_blender=GATED_BLENDER),
                targets=TARGETS, min_decided=MIN_DECIDED, primary=PRIMARY, gradient=GRADIENT, arms=ARMS, localization=localization,
                localization_all_found=all(v['all_found_no_fp'] for v in localization.values()),
                integrity_failures=failures, integrity_passed=integrity_passed, rows=rows, **evaluation,
                **verdict(integrity_passed, evaluation), defaults='unchanged: chroma_vk and reference_model are opt-in runner options; GUI, core and research defaults untouched')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('thresholds')
    p = sub.add_parser('holdout')
    for name in ('--v1', '--output', '--protocol', '--blender'):
        p.add_argument(name, type=Path, required=True)
    p.add_argument('--seeds-key', default='holdout_seeds', help='protocol data key; smoke runs use smoke_seeds')
    p = sub.add_parser('masks')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    for name in ('--v1', '--holdout', '--runner', '--base-runner', '--masks', '--protocol', '--output'):
        p.add_argument(name, type=Path, required=True)
    p.add_argument('--timeout', type=float, default=900.)
    p.add_argument('--scene', action='append', help='Smoke runs only: restrict to these scene keys (formal runs use all)')
    p = sub.add_parser('summarize')
    for name in ('--manifest', '--protocol', '--output'):
        p.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'thresholds':
        print(json.dumps(catch_thresholds(), indent=1))
    elif args.command == 'holdout':
        protocol = json.loads(args.protocol.read_text())
        print(json.dumps(make_holdout(args.v1, args.output, protocol['data'][args.seeds_key], args.blender), indent=1))
    elif args.command == 'masks':
        print(json.dumps(stage46.make_masks(args.output), indent=1))
    elif args.command == 'run':
        run(args)
    else:
        if args.output.exists():
            raise ValueError('Refusing to overwrite a summary')
        manifest = json.loads(args.manifest.read_text())
        summary = summarize(manifest, args.manifest.resolve().parent, args.protocol)
        save(args.output, dict(summary, manifest_sha256=sha(args.manifest)))
        print(json.dumps(dict(integrity_passed=summary['integrity_passed'], primary_passed=summary['primary_passed'],
                              recommended=summary['recommended'], gradient_preferred=summary['gradient_preferred'],
                              failures=len(summary['integrity_failures']))))


if __name__ == '__main__':
    main()
