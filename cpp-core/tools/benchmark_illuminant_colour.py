"""Stage 46: illumination-normalized same-colour verification on test-bed v1.

Subcommands (python-port venv, repository root):
  masks      template annotation masks (colour field, neutral reference panel) from the test-bed artwork
  run        one fresh runner process per scene x method, plus the pre-change base runner for existing methods
  summarize  read GT and the pre-registered protocol; write the summary with every gate and failure

Localization is the existing grayscale Session search with the test-bed v1 settings, shared by every method;
jobs never contain GT. Thresholds are calibrated one-class (same-colour objects only) on the calibration
lighting and applied unchanged to held-out lighting. No timing is measured or reported.

  PY=python-port/venv/bin/python
  $PY cpp-core/tools/benchmark_illuminant_colour.py masks --output build/stage46-illuminant/masks
  $PY cpp-core/tools/benchmark_illuminant_colour.py run --testbed build/testbed-v1 \\
      --runner build/stage46-release/experiments/fpm_color_verify_runner \\
      --base-runner build/stage46-base/experiments/fpm_color_verify_runner \\
      --masks build/stage46-illuminant/masks --protocol cpp-core/docs/stage46-illuminant-chroma/protocol.json \\
      --output build/stage46-illuminant/run/manifest.json
  $PY cpp-core/tools/benchmark_illuminant_colour.py summarize --manifest build/stage46-illuminant/run/manifest.json \\
      --protocol cpp-core/docs/stage46-illuminant-chroma/protocol.json --output build/stage46-illuminant/summary.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_new_frame_cost as stage41  # noqa: E402
from many_object_accuracy import evaluate  # noqa: E402
import testbed_evaluate  # noqa: E402
import testbed_generate as tb  # noqa: E402

STAGE = 46
SEARCH = testbed_evaluate.SEARCH
LIMIT = 64
SYNTHETIC_SCALES, BLENDER_SCALES = [.85, 1., 1.15], [.9, 1., 1.1]
BLENDER = (('bl_rgb_white', 'white'), ('bl_rgb_warm3200', 'warm3200'), ('bl_rgb_mixed', 'mixed'))
CALIBRATION = ('neutral', 'exposure_dark', 'wb_cool', 'shading', 'white')
HOLDOUT = ('wb_warm', 'exposure_bright', 'spot_warm', 'noise_jpeg', 'warm3200', 'mixed')
GATED_SYNTHETIC = ('wb_warm', 'spot_warm', 'noise_jpeg')   # exposure_bright: predicted clipped reference, reported
GATED_BLENDER = ('warm3200', 'mixed')
GATED_DIRECTIONS = ('hue+', 'hue-', 'chroma-', 'isogray0', 'isogray90')  # lightness+ is not a chromaticity change
ISOGRAY = ('isogray0', 'isogray90')
GATED_MIN_DELTA_E = 4.
PERCENTILE = 95.
TARGETS = dict(same_acceptance=.90, different_rejection=.95, isogray_rejection=.95,
               blender_same_acceptance=1., blender_different_rejection=1.)
PANEL = (236, 234, 226)
ERODE = 5
_X0, _Y0, _X1, _Y1 = tb.REFERENCE_CARD['xyxy']
_STEP = (_X1 - _X0) // len(tb.REFERENCE_CARD['patches'])
CARD_RECTS = dict(white=[_X0 + 8, _Y0 + 8, _X0 + _STEP - 8, _Y1 - 8],
                  gray18=[_X0 + _STEP + 8, _Y0 + 8, _X0 + 2 * _STEP - 8, _Y1 - 8])
CHROMA = dict(chroma_distance_threshold=.04, min_linear_sum=.03, min_valid_fraction=.95)
VK = dict(CHROMA, saturation_level=255, max_reference_clip_fraction=.45, min_reference_fraction=.5)
METHODS = {
    'gray_ncc': dict(method='gray_ncc'),
    'rgb_ncc': dict(method='rgb_ncc'),
    'chroma': dict(method='chroma', **CHROMA),
    'chroma_masked': dict(method='chroma', mask='colour', **CHROMA),
    'vk_object': dict(method='chroma_vk', mask='colour', reference_mask='reference', reference_clip_policy='rank', **VK),
    'vk_object_drop': dict(method='chroma_vk', mask='colour', reference_mask='reference', reference_clip_policy='drop', **VK),
    'vk_object_none': dict(method='chroma_vk', mask='colour', reference_mask='reference', reference_clip_policy='none', **VK),
    'card_white': dict(method='chroma_vk', mask='colour', illuminant_reference='source_rect', reference_rect='white',
                       reference_clip_policy='rank', **VK),
    'card_gray': dict(method='chroma_vk', mask='colour', illuminant_reference='source_rect', reference_rect='gray18',
                      reference_clip_policy='rank', **VK),
}
EXISTING = ('gray_ncc', 'rgb_ncc', 'chroma', 'chroma_masked')  # also run on the pre-change runner: must be identical
SYNTHETIC_ONLY = ('card_white', 'card_gray')                     # the Blender scenes have no reference card
# Decision arms: (method process, statistic). Dissimilarity: larger means "more different"; None abstains.
ARMS = {
    'gray_ncc': ('gray_ncc', 'ncc'),
    'rgb_ncc': ('rgb_ncc', 'ncc'),
    'chroma': ('chroma', 'mean_distance'),
    'chroma_masked': ('chroma_masked', 'mean_distance'),
    'masked_region': ('vk_object_none', 'uncorrected_region_distance'),
    'vk_pixel': ('vk_object', 'mean_distance'),
    'vk_region': ('vk_object', 'region_distance'),
    'vk_drop_region': ('vk_object_drop', 'region_distance'),
    'vk_none_region': ('vk_object_none', 'region_distance'),
    'card_white_region': ('card_white', 'region_distance'),
    'card_gray_region': ('card_gray', 'region_distance'),
    'vk_uncorrected_region': ('vk_object', 'uncorrected_region_distance'),
}
PRIMARY = 'vk_region'
COMPARED = ('chroma_masked', 'masked_region')
FALLBACK = ('vk_region', 'vk_uncorrected_region')  # secondary policy: corrected, else uncorrected when the reference abstains
FORBIDDEN_JOB_KEYS = {'truth', 'ground_truth', 'labels', 'same_colour', 'direction', 'delta_e00_target',
                      'delta_e00_achieved', 'texture', 'gray_texture_identical_to_base', 'objects', 'corners'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


# ---------------------------------------------------------------- inputs and jobs
def make_masks(output):
    """Template annotations from the label artwork: never from a scene, never from GT."""
    output = Path(output)
    records = {}
    kernel = np.ones((ERODE, ERODE), np.uint8)
    for family, colour in tb.FAMILIES.items():
        texture = tb.capsule_texture(colour)
        for name, region in (('colour', np.all(texture == colour, -1)), ('reference', np.all(texture == PANEL, -1))):
            mask = (cv2.erode(region.astype(np.uint8), kernel) > 0).astype(np.uint8) * 255
            path = output / f'{family}_{name}.png'
            tb.write_png(path, mask)
            records[path.name] = dict(family=family, region=name, pixels=int((mask > 0).sum()), sha256=sha(path))
    save(output / 'masks.json', dict(rule=f'texture pixels equal to the family colour (colour) or the panel {list(PANEL)} '
        f'(reference) in testbed_generate.capsule_texture, eroded {ERODE}x{ERODE}', generator_sha256=sha(tb.__file__),
        texture_size=[tb.TEXTURE_W, tb.TEXTURE_H], masks=records))
    return records


def scenes(testbed):
    testbed = Path(testbed)
    out = []
    for family in tb.FAMILIES:
        for condition in tb.CONDITIONS:
            name = f'syn_{family}_{condition}'
            out.append(dict(name=name, source='synthetic', family=family, group=f'syn_{family}', condition=condition,
                            image=testbed / 'synthetic' / f'{name}.png', template=testbed / 'synthetic' / f'syn_{family}_template.png',
                            truth=testbed / 'synthetic' / f'{name}.truth.json', scales=SYNTHETIC_SCALES))
    for name, condition in BLENDER:
        out.append(dict(name=name, source='blender', family='red', group='bl_red', condition=condition,
                        image=testbed / 'blender' / f'{name}.png', template=testbed / 'final' / 'bl_rgb_reference_template.png',
                        truth=testbed / 'blender' / f'{name}.json', scales=BLENDER_SCALES))
    return out


def split(condition):
    if condition in CALIBRATION:
        return 'calibration'
    if condition in HOLDOUT:
        return 'holdout'
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
        spec['reference_rect'] = CARD_RECTS[spec['reference_rect']]
    out.update(spec)
    return out


def gt_problems(job_dict):
    """Forbidden label keys anywhere in a job, or a path to a GT file."""
    problems = []
    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FORBIDDEN_JOB_KEYS:
                    problems.append(f'{where}.{key}')
                walk(item, f'{where}.{key}')
        elif isinstance(value, list):
            for k, item in enumerate(value):
                walk(item, f'{where}[{k}]')
        elif isinstance(value, str) and (value.endswith('.truth.json') or (value.endswith('.json') and '/blender/' in value)):
            problems.append(f'{where}: {value}')
    walk(job_dict, 'job')
    return problems


def input_files(scene_list, masks):
    files = set()
    for scene in scene_list:
        files.update((Path(scene['image']).resolve(), Path(scene['template']).resolve()))
    files.update(path.resolve() for path in Path(masks).glob('*.png'))
    return sorted(files)


def testbed_hashes(testbed):
    """Test-bed manifest hashes keyed by 'subdirectory/file' (the generator writes flat blender/, synthetic/, final/)."""
    manifest = json.loads((Path(testbed) / 'final' / 'manifest.json').read_text())
    return {'/'.join(Path(key).parts[-2:]): value for key, value in manifest['files'].items()}, manifest['generator_sha256']


# ---------------------------------------------------------------- run
def run_worker(runner, root, label, job_dict, timeout):
    jp, rp = root / 'jobs' / (label + '.json'), root / 'results' / (label + '.json')
    save(jp, job_dict)
    child = subprocess.run([str(runner), str(jp)], capture_output=True, text=True, timeout=timeout)
    jp.with_suffix('.stderr.txt').write_text(child.stderr)
    entry = dict(job=str(jp.relative_to(root)), job_sha256=sha(jp), returncode=child.returncode)
    if child.returncode == 0:
        save(rp, json.loads(child.stdout))
        entry.update(result=str(rp.relative_to(root)), result_sha256=sha(rp))
    else:
        entry['stderr_tail'] = child.stderr[-4000:]
    return entry


def git_head():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
    return dict(head=head, clean=not dirty)


def run(args):
    root = args.output.resolve().parent
    if args.output.exists() or any((root / name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    scene_list = [s for s in scenes(args.testbed) if not args.scene or s['name'] in args.scene]
    if args.scene and len(scene_list) != len(set(args.scene)):
        raise ValueError('Unknown scene in --scene')
    expected, generator = testbed_hashes(args.testbed)
    testbed_root = Path(args.testbed).resolve()
    inputs = {str(path): sha(path) for path in input_files(scene_list, args.masks)}
    for path, digest in inputs.items():
        if Path(path).is_relative_to(testbed_root) and expected.get('/'.join(Path(path).parts[-2:])) != digest:
            raise ValueError(f'{path} differs from the test-bed manifest')
    plan = [(scene, method, kind) for scene in scene_list for method in methods_for(scene)
            for kind in (('candidate', 'base') if method in EXISTING else ('candidate',))]
    host = stage41.host_state(samples=3)
    manifest = dict(version=1, stage=STAGE, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(args.protocol), git=git_head(), runner=str(args.runner.resolve()), runner_sha256=sha(args.runner),
        base_runner=str(args.base_runner.resolve()), base_runner_sha256=sha(args.base_runner),
        testbed=str(testbed_root), testbed_generator_sha256=generator, input_sha256=inputs,
        masks_json_sha256=sha(Path(args.masks) / 'masks.json'), scene_filter=args.scene, methods=METHODS, arms=ARMS, search=SEARCH, limit=LIMIT,
        scenes=[dict(scene, image=str(Path(scene['image']).resolve()), template=str(Path(scene['template']).resolve()),
                     truth=str(Path(scene['truth']).resolve())) for scene in scene_list],
        host=dict(before=host), executions=[], complete=False)
    save(args.output, manifest)
    try:
        for scene, method, kind in plan:
            label = f"{scene['name']}-{method}" + ('-base' if kind == 'base' else '')
            print('Running ' + label, flush=True)
            entry = run_worker(args.runner if kind == 'candidate' else args.base_runner, root, label,
                               job(scene, method, args.masks), args.timeout)
            manifest['executions'].append(dict(scene=scene['name'], method=method, runner_kind=kind, **entry))
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
def dissimilarity(score, statistic):
    if statistic == 'ncc':
        return None if score <= -1.5 else 1. - float(score)  # -2: invalid NCC (constant template)
    value = score[statistic]
    return None if value < 0 else float(value)


def truth_objects(scene, meta):
    if scene['source'] == 'synthetic':
        return meta['truth']
    return [dict(o, pos_x=o['projected_center'][0], pos_y=o['projected_center'][1]) for o in meta['objects']]


def scene_rows(scene, meta, results):
    """Per-GT-object rows for one scene; results maps method -> runner output (candidate runner)."""
    methods = list(results)
    matches = results[methods[0]]['first']['search_matches']
    truth = truth_objects(scene, meta)
    strict = evaluate(matches, [dict(pos_x=t['pos_x'], pos_y=t['pos_y'], angle=t['angle'], scale=t['scale']) for t in truth])['strict']
    mapping = {d['truth_index']: d['prediction_index'] for d in strict['details']}
    rows = []
    for k, item in enumerate(truth):
        p = mapping.get(k)
        row = dict(scene=scene['name'], source=scene['source'], group=scene['group'], family=scene['family'],
                   condition=scene['condition'], split=split(scene['condition']), index=item['index'],
                   same_colour=bool(item['same_colour']), direction=item['direction'],
                   delta_e00_target=float(item['delta_e00_target']), delta_e00_achieved=float(item['delta_e00_achieved']),
                   gray_identical=bool(item['gray_texture_identical_to_base']), found=p is not None, prediction=p,
                   values={}, diagnostics={})
        if p is not None:
            for arm, (method, statistic) in ARMS.items():
                if method in results:
                    row['values'][arm] = dissimilarity(results[method]['first']['scores'][p], statistic)
            for method, result in results.items():
                score = result['first']['scores'][p]
                if isinstance(score, dict):
                    row['diagnostics'][method] = {key: score[key] for key in ('status', 'valid_fraction', 'clipped_pixels',
                        'reference_fraction', 'reference_clip_fraction', 'gain') if key in score}
        rows.append(row)
    matched = set(mapping.values()) - {None}
    false_positives = [dict(scene=scene['name'], prediction=k, values={arm: dissimilarity(results[method]['first']['scores'][k], statistic)
                            for arm, (method, statistic) in ARMS.items() if method in results})
                       for k in range(len(matches)) if k not in matched]
    return rows, dict(tp=strict['tp'], fp=strict['fp'], fn=strict['fn'], predictions=len(matches)), false_positives


def percentile_threshold(values, rule='p95'):
    if not values:
        return None
    return float(np.max(values)) if rule == 'max' else float(np.percentile(values, PERCENTILE))


def thresholds(rows, rule='p95'):
    """One-class thresholds per (arm, group) from calibration same-colour, found, non-abstained objects only."""
    out = {}
    for arm in ARMS:
        for group in sorted({r['group'] for r in rows}):
            values = [r['values'][arm] for r in rows if r['split'] == 'calibration' and r['group'] == group and r['same_colour']
                      and r['found'] and r['values'].get(arm) is not None]
            if any(arm in r['values'] for r in rows if r['group'] == group and r['found']):
                out[(arm, group)] = percentile_threshold(values, rule)
    return out


def accepted(row, arm, table, group=None):
    value, threshold = row['values'].get(arm), table.get((arm, group or row['group']))
    return value is not None and threshold is not None and value <= threshold


def fallback_accepted(row, table, group=None):
    corrected, uncorrected = FALLBACK
    if row['values'].get(corrected) is not None:
        return accepted(row, corrected, table, group)
    return accepted(row, uncorrected, table, group)


def auc(same, different):
    """P(different > same) + P(tie)/2 over all pairs; an abstention (None) ranks as +inf. None for an empty class."""
    if not same or not different:
        return None
    s = [math.inf if v is None else v for v in same]
    d = [math.inf if v is None else v for v in different]
    total = sum(1. if a > b else .5 if a == b else 0. for a in d for b in s)
    return total / (len(s) * len(d))


def cell(rows, arm, table, group=None, decide=None):
    decide = decide or (lambda r: accepted(r, arm, table, group))
    found = [r for r in rows if r['found'] and arm in r['values']]
    same = [r for r in found if r['same_colour']]
    different = [r for r in found if not r['same_colour']]
    same_accepted = sum(decide(r) for r in same)
    wrong = [r for r in different if decide(r)]
    rate = lambda numerator, n: numerator / n if n else None
    return dict(same=dict(n=len(same), accepted=same_accepted, abstained=sum(r['values'][arm] is None for r in same),
                          acceptance=rate(same_accepted, len(same))),
                different=dict(n=len(different), wrongly_accepted=len(wrong), abstained=sum(r['values'][arm] is None for r in different),
                               rejection=rate(len(different) - len(wrong), len(different)),
                               wrong=[dict(scene=r['scene'], index=r['index'], direction=r['direction'], delta_e00_target=r['delta_e00_target'],
                                           value=r['values'][arm]) for r in wrong]),
                auc=auc([r['values'][arm] for r in same], [r['values'][arm] for r in different]),
                auc_valid=auc([r['values'][arm] for r in same if r['values'][arm] is not None],
                              [r['values'][arm] for r in different if r['values'][arm] is not None]),
                not_found=sum(not r['found'] for r in rows))


def direction_table(rows, arm, table, group=None, decide=None):
    decide = decide or (lambda r: accepted(r, arm, table, group))
    out = {}
    for r in rows:
        if not r['found'] or arm not in r['values']:
            continue
        key = 'same' if r['same_colour'] else f"{r['direction']}@{r['delta_e00_target']:g}"
        entry = out.setdefault(key, dict(n=0, accepted=0, abstained=0))
        entry['n'] += 1
        entry['accepted'] += bool(decide(r))
        entry['abstained'] += r['values'][arm] is None
    for key, entry in out.items():
        entry['rate'] = entry['accepted'] / entry['n'] if key == 'same' else 1 - entry['accepted'] / entry['n']
        entry['rate_kind'] = 'acceptance' if key == 'same' else 'rejection'
    return dict(sorted(out.items()))


def gated(row):
    return not row['same_colour'] and row['direction'] in GATED_DIRECTIONS and row['delta_e00_target'] >= GATED_MIN_DELTA_E


def check(name, value, target, n, **where):
    return dict(name=name, value=value, target=target, n=n, passed=value is not None and value >= target, **where)


def hypotheses(rows, table):
    """Pre-registered H1-H3 on the primary arm; returns every check (failures are kept)."""
    arm, checks = PRIMARY, []
    found = [r for r in rows if r['found'] and arm in r['values']]
    for condition in GATED_SYNTHETIC:
        subset = [r for r in found if r['condition'] == condition]
        same = [r for r in subset if r['same_colour']]
        different = [r for r in subset if gated(r)]
        checks.append(check('H1a same-colour acceptance', sum(accepted(r, arm, table) for r in same) / len(same) if same else None,
                            TARGETS['same_acceptance'], len(same), condition=condition))
        checks.append(check('H1b gated different-colour rejection',
                            sum(not accepted(r, arm, table) for r in different) / len(different) if different else None,
                            TARGETS['different_rejection'], len(different), condition=condition))
    iso = [r for r in found if r['condition'] in GATED_SYNTHETIC and gated(r) and r['direction'] in ISOGRAY]
    checks.append(check('H1c isogray rejection (pooled)', sum(not accepted(r, arm, table) for r in iso) / len(iso) if iso else None,
                        TARGETS['isogray_rejection'], len(iso), condition='+'.join(GATED_SYNTHETIC)))
    for condition in GATED_BLENDER:
        subset = [r for r in found if r['condition'] == condition]
        same = [r for r in subset if r['same_colour']]
        different = [r for r in subset if gated(r)]
        checks.append(check('H2a Blender same-colour acceptance', sum(accepted(r, arm, table) for r in same) / len(same) if same else None,
                            TARGETS['blender_same_acceptance'], len(same), condition=condition))
        checks.append(check('H2b Blender gated different-colour rejection',
                            sum(not accepted(r, arm, table) for r in different) / len(different) if different else None,
                            TARGETS['blender_different_rejection'], len(different), condition=condition))
    cells = [(group, condition) for group in ('syn_red', 'syn_teal') for condition in GATED_SYNTHETIC] + \
            [('bl_red', condition) for condition in GATED_BLENDER]
    for group, condition in cells:
        subset = [r for r in rows if r['group'] == group and r['condition'] == condition]
        primary = cell(subset, arm, table)['auc']
        for other in COMPARED:
            baseline = cell(subset, other, table)['auc']
            checks.append(dict(name=f'H3 AUC {arm} >= {other}', group=group, condition=condition, value=primary, baseline=baseline,
                               n=len([r for r in subset if r['found']]),
                               passed=primary is not None and baseline is not None and primary >= baseline))
    return checks


def integrity(manifest, root, scene_list):
    """Workers, repeats, localization identity, base identity, GT-free jobs and input hashes."""
    failures = []
    by_key = {}
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
    results, base = {}, {}
    for scene in scene_list:
        name = scene['name']
        results[name] = {}
        for method in methods_for(scene):
            entry = by_key.get((name, method, 'candidate'))
            if entry is None:
                failures.append(dict(gate='missing_worker', scene=name, method=method))
                continue
            if entry['returncode']:
                continue
            result = json.loads((root / entry['result']).read_text())
            if not result['exact_repeat']:
                failures.append(dict(gate='exact_repeat', scene=name, method=method))
            results[name][method] = result
            if method in EXISTING:
                base_entry = by_key.get((name, method, 'base'))
                if base_entry is None or base_entry['returncode']:
                    failures.append(dict(gate='missing_base_worker', scene=name, method=method))
                    continue
                reference = json.loads((root / base_entry['result']).read_text())
                base.setdefault(name, {})[method] = reference
                if reference['first']['scores'] != result['first']['scores'] or \
                        reference['first']['search_matches'] != result['first']['search_matches']:
                    failures.append(dict(gate='existing_method_changed', scene=name, method=method))
        matches = [result['first']['search_matches'] for result in results[name].values()]
        if any(m != matches[0] for m in matches[1:]):
            failures.append(dict(gate='localization_differs_between_methods', scene=name))
    for path, digest in manifest['input_sha256'].items():
        if not Path(path).exists() or sha(path) != digest:
            failures.append(dict(gate='input_changed', path=path))
    if not manifest.get('complete') or not manifest.get('unchanged'):
        failures.append(dict(gate='run_incomplete_or_binaries_changed'))
    return results, failures


def evaluate_rows(rows, false_positives=()):
    table = thresholds(rows)
    table_max = thresholds(rows, 'max')
    per_condition = {arm: {group: {c: cell([r for r in rows if r['group'] == group and r['condition'] == c], arm, table)
                                   for c in dict.fromkeys(r['condition'] for r in rows if r['group'] == group)}
                           for group in dict.fromkeys(r['group'] for r in rows)} for arm in ARMS}
    per_split = {arm: {group: {s: cell([r for r in rows if r['group'] == group and r['split'] == s], arm, table)
                               for s in ('calibration', 'holdout')}
                       for group in dict.fromkeys(r['group'] for r in rows)} for arm in ARMS}
    directions = {arm: {group: {s: direction_table([r for r in rows if r['group'] == group and r['split'] == s], arm, table)
                                for s in ('calibration', 'holdout')}
                        for group in dict.fromkeys(r['group'] for r in rows)} for arm in ARMS}
    isogray = {arm: {c: direction_table([r for r in rows if r['condition'] == c and (r['direction'] in ISOGRAY or r['same_colour'])],
                                        arm, table) for c in dict.fromkeys(r['condition'] for r in rows)} for arm in ARMS}
    blender = [r for r in rows if r['group'] == 'bl_red']
    transfer = {arm: {c: cell([r for r in blender if r['condition'] == c], arm, table, group='syn_red')
                      for c in dict.fromkeys(r['condition'] for r in blender)} for arm in ARMS if (arm, 'syn_red') in table}
    fallback = {group: {c: cell([r for r in rows if r['group'] == group and r['condition'] == c], FALLBACK[0], table,
                                decide=lambda r: fallback_accepted(r, table))
                        for c in dict.fromkeys(r['condition'] for r in rows if r['group'] == group)}
                for group in dict.fromkeys(r['group'] for r in rows)}
    max_rule = {arm: {group: {s: cell([r for r in rows if r['group'] == group and r['split'] == s], arm, table_max)
                              for s in ('calibration', 'holdout')} for group in dict.fromkeys(r['group'] for r in rows)}
                for arm in ARMS}
    checks = hypotheses(rows, table)
    fp_decisions = [dict(fp, accepted={arm: fp['values'][arm] is not None and table.get((arm, fp['group'])) is not None
                                       and fp['values'][arm] <= table[(arm, fp['group'])] for arm in fp['values']})
                    for fp in false_positives]
    return dict(thresholds=[dict(arm=a, group=g, value=v, rule=f'p{PERCENTILE:g}') for (a, g), v in sorted(table.items())],
                thresholds_max=[dict(arm=a, group=g, value=v, rule='max') for (a, g), v in sorted(table_max.items())],
                per_condition=per_condition, per_split=per_split, directions=directions, isogray=isogray,
                blender_transfer_syn_red=transfer, fallback=fallback, max_rule=max_rule, false_positive_decisions=fp_decisions,
                hypotheses=checks, hypotheses_passed=all(c['passed'] for c in checks))


def summarize(manifest, root, protocol_path):
    scene_list = manifest['scenes']
    if sha(protocol_path) != manifest['protocol_sha256']:
        raise ValueError('Protocol differs from the one recorded by the run')
    results, failures = integrity(manifest, root, scene_list)
    rows, localization, false_positives = [], {}, []
    for scene in scene_list:
        if len(results[scene['name']]) != len(methods_for(scene)):
            continue
        meta = json.loads(Path(scene['truth']).read_text())
        scene_rows_, counts, fps = scene_rows(scene, meta, results[scene['name']])
        rows += scene_rows_
        false_positives += [dict(fp, group=scene['group']) for fp in fps]
        expected = len(scene_rows_)
        counts['matches_testbed_v1'] = counts['tp'] == expected and counts['fp'] == 0 and counts['fn'] == 0
        localization[scene['name']] = counts
        if scene['source'] == 'synthetic' and meta.get('reference_card', {}).get('xyxy') != list(tb.REFERENCE_CARD['xyxy']):
            failures.append(dict(gate='reference_card_moved', scene=scene['name']))
    evaluation = evaluate_rows(rows, false_positives)
    integrity_passed = not failures
    return dict(stage=STAGE, protocol_sha256=manifest['protocol_sha256'], runner_sha256=manifest['runner_sha256'],
                base_runner_sha256=manifest['base_runner_sha256'], git=manifest['git'],
                split=dict(calibration=CALIBRATION, holdout=HOLDOUT, gated_synthetic=GATED_SYNTHETIC, gated_blender=GATED_BLENDER),
                targets=TARGETS, primary=PRIMARY, arms=ARMS, localization=localization,
                localization_matches_testbed_v1=all(v['matches_testbed_v1'] for v in localization.values()),
                integrity_failures=failures, integrity_passed=integrity_passed, rows=rows, **evaluation,
                recommended=integrity_passed and evaluation['hypotheses_passed'],
                defaults='unchanged: chroma_vk is an opt-in runner method; GUI, core and research defaults are untouched')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('masks')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    for name in ('--testbed', '--runner', '--base-runner', '--masks', '--protocol', '--output'):
        p.add_argument(name, type=Path, required=True)
    p.add_argument('--timeout', type=float, default=900.)
    p.add_argument('--scene', action='append', help='Smoke runs only: restrict to these scenes (formal runs use all 19)')
    p = sub.add_parser('summarize')
    for name in ('--manifest', '--protocol', '--output'):
        p.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'masks':
        print(json.dumps(make_masks(args.output), indent=1))
    elif args.command == 'run':
        run(args)
    else:
        if args.output.exists():
            raise ValueError('Refusing to overwrite a summary')
        manifest = json.loads(args.manifest.read_text())
        summary = summarize(manifest, args.manifest.resolve().parent, args.protocol)
        save(args.output, dict(summary, manifest_sha256=sha(args.manifest)))
        print(json.dumps(dict(integrity_passed=summary['integrity_passed'], hypotheses_passed=summary['hypotheses_passed'],
                              recommended=summary['recommended'], failures=len(summary['integrity_failures']))))


if __name__ == '__main__':
    main()
