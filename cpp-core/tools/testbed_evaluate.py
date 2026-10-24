"""Baseline sanity evaluation of test-bed v1 with the existing color-verify runner.

Localization and colour decisions are reported separately. Every run uses one
process per scene and method; GT is read only here, after the runner returns.
This is a dataset sanity check, not a formal timing study: warm-ups and repeats
are minimal and no time is reported as evidence.

  python-port/venv/bin/python cpp-core/tools/testbed_evaluate.py --testbed build/testbed-v1 \
    --runner build/research/experiments/fpm_color_verify_runner --output build/testbed-v1/eval
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from many_object_accuracy import evaluate  # noqa: E402

SEARCH = dict(backend=1, refinement_backend=1, angle=180, score=.65, proposal_score=.5, min_area=256, cpu_threads=8,
              pose_resident_mode=0, pose_storage_mode=1, pose_input_mode=1, fine_ncc_mode=2, fine_fused=True, fine_angle_reuse=True)
CHROMA_THRESHOLD = .04  # the runner's documented example default; reported, not tuned


def run(runner, job, path):
    path.write_text(json.dumps(job, indent=2))
    child = subprocess.run([str(runner), str(path)], capture_output=True, text=True, timeout=1800)
    if child.returncode:
        raise RuntimeError(f'{path.name}: {child.stderr[-800:]}')
    return json.loads(child.stdout)['first']


def assignment(matches, truth):
    """Strict one-to-one assignment: truth index -> prediction index (or None)."""
    result = evaluate(matches, truth)['strict']
    return {d['truth_index']: d['prediction_index'] for d in result['details']}, result


def synthetic(testbed, runner, output):
    rows = []
    for truth_path in sorted((testbed / 'synthetic').glob('syn_*.truth.json')):
        meta = json.loads(truth_path.read_text())
        source, template = testbed / 'synthetic' / meta['image'], testbed / 'synthetic' / meta['template']
        per_method = {}
        for method in ('gray_ncc', 'rgb_ncc', 'chroma'):
            job = dict(source=str(source.resolve()), template=str(template.resolve()), method=method, backend='cpu', threads=1,
                       warmups=0, repeats=1, search=dict(SEARCH, scales=[.85, 1., 1.15], limit=64))
            if method == 'chroma':
                job.update(chroma_distance_threshold=CHROMA_THRESHOLD, min_linear_sum=.03, min_valid_fraction=.95)
            per_method[method] = run(runner, job, output / f"{meta['scene']}-{method}.job.json")
        matches = per_method['chroma']['search_matches']
        for method in per_method:  # one shared grayscale search: detections must be identical
            if per_method[method]['search_matches'] != matches:
                raise RuntimeError(f"{meta['scene']}: search differs between methods")
        mapping, strict = assignment(matches, meta['truth'])
        for t_index, item in enumerate(meta['truth']):
            p = mapping.get(t_index)
            rows.append(dict(scene=meta['scene'], family=meta['family'], condition=meta['condition'], index=item['index'],
                same_colour=item['same_colour'], direction=item['direction'], delta_e00=item['delta_e00_achieved'],
                gray_identical=item['gray_texture_identical_to_base'], found=p is not None,
                **{f'{m}_score': (per_method[m]['scores'][p] if p is not None else None) for m in per_method}))
        rows.append(dict(scene=meta['scene'], summary=True, tp=strict['tp'], fp=strict['fp'], fn=strict['fn']))
    return rows


def blender(testbed, runner, output):
    final, rendered = testbed / 'final', testbed / 'blender'
    plans = [('bl_gray_crowded', 'bl_gray_reference', [.8, 1., 1.25], 'strict'),
             ('bl_gray_gloss_shadow', 'bl_gray_reference', [1.], 'strict'),
             ('bl_gray_tilt', 'bl_gray_reference', [1.], 'centre'),
             ('bl_gray_perspective', 'bl_gray_reference', [.8, 1., 1.25], 'centre'),
             ('bl_sym_scene', 'bl_sym_reference', [1.], 'symmetric'),
             ('bl_rgb_white', 'bl_rgb_reference', [.9, 1., 1.1], 'colour'),
             ('bl_rgb_warm3200', 'bl_rgb_reference', [.9, 1., 1.1], 'colour'),
             ('bl_rgb_mixed', 'bl_rgb_reference', [.9, 1., 1.1], 'colour')]
    rows = []
    for scene, ref, scales, mode in plans:
        meta_path = rendered / f'{scene}.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        template = final / f'{ref}_template.png'
        methods = ('gray_ncc', 'rgb_ncc', 'chroma') if mode == 'colour' else ('geometry_only',)
        results = {}
        for method in methods:
            job = dict(source=str((rendered / meta['image']).resolve()), template=str(template.resolve()), method=method,
                       backend='cpu', threads=1, warmups=0, repeats=1, search=dict(SEARCH, scales=scales, limit=64))
            if method == 'chroma':
                job.update(chroma_distance_threshold=CHROMA_THRESHOLD, min_linear_sum=.03, min_valid_fraction=.95)
            results[method] = run(runner, job, output / f'{scene}-{method}.job.json')
        matches = results[methods[0]]['search_matches']
        truth = [dict(pos_x=o['projected_center'][0], pos_y=o['projected_center'][1], angle=o['angle'], scale=o['scale']) for o in meta['objects']]
        if mode in ('strict', 'colour'):
            mapping, strict = assignment(matches, truth)
            summary = dict(tp=strict['tp'], fp=strict['fp'], fn=strict['fn'])
        elif mode == 'symmetric':
            flipped = [dict(t, angle=((t['angle'] + 180 + 180) % 360) - 180) for t in truth]
            a, sa = assignment(matches, truth); b, sb = assignment(matches, flipped)
            mapping = {k: a.get(k) if a.get(k) is not None else b.get(k) for k in range(len(truth))}
            found = sum(v is not None for v in mapping.values())
            summary = dict(tp_mod180=found, fn_mod180=len(truth) - found, predictions=len(matches))
        else:  # centre: projected-centre distance only, outside the similarity model
            mapping = {}
            for k, t in enumerate(truth):
                d = [math.hypot(m['pos_x'] - t['pos_x'], m['pos_y'] - t['pos_y']) for m in matches]
                mapping[k] = int(np.argmin(d)) if d and min(d) <= 10 else None
            found = sum(v is not None for v in mapping.values())
            summary = dict(centre_within_10px=found, missed=len(truth) - found, predictions=len(matches))
        for k, o in enumerate(meta['objects']):
            p = mapping.get(k)
            row = dict(scene=scene, index=o['index'], tilt_deg=o.get('tilt_deg', 0.), found=p is not None,
                       score=matches[p]['score'] if p is not None else None)
            if mode == 'colour':
                row.update(same_colour=o['same_colour'], direction=o['direction'], delta_e00=o['delta_e00_achieved'],
                           gray_identical=o['gray_texture_identical_to_base'],
                           **{f'{m}_score': (results[m]['scores'][p] if p is not None else None) for m in methods})
            rows.append(row)
        rows.append(dict(scene=scene, summary=True, mode=mode, **summary))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--testbed', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(chroma_threshold=CHROMA_THRESHOLD, search=SEARCH,
                  synthetic=synthetic(args.testbed, args.runner, args.output),
                  blender=blender(args.testbed, args.runner, args.output))
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    print(json.dumps({k: len(v) for k, v in report.items() if isinstance(v, list)}))


if __name__ == '__main__':
    main()
