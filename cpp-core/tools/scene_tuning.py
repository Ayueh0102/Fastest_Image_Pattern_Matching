"""Tune the desktop matcher's search settings against independent pose annotations.

Generate deterministic controls, or supply a JSON list with name, split (dev/test),
source, template and truth [{pos_x,pos_y,angle,scale}]. Paths may be relative to the
manifest. The optional real-photo probes use category points and never select a preset.
"""
import argparse
import hashlib
import json
import random
import statistics
import subprocess
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from many_object_accuracy import evaluate
import stage49_scenes as scenes

ARMS = {
    'baseline': {},
    'short12': {'min_top_short_side': 12},
    'more_candidates': {'proposal_score': .40, 'limit': 200},
    'wide_candidates': {'proposal_score': .25, 'limit': 300},
    'short12_verified': {'min_top_short_side': 12, 'border_min_inside': .5, 'small_template_area': 576},
}
BASE = dict(angle=180, scales=[1.], min_area=256, limit=100, proposal_score=.5,
            score=.8, overlap=.2, stop_layer=1, repeats=3, warmup_seconds=0)
ROOT = Path(__file__).resolve().parents[2]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for k in range(12):
        split = 'dev' if k < 4 else 'test'
        seed = 511100 + k
        width, height = [(43, 30), (68, 25), (32, 41), (56, 57)][k % 4]
        template = scenes.procedural(seed, [1., 2., 3., 4.][k % 4], (width, height))
        tpath = out / f'template-{seed}.png'
        if not cv2.imwrite(str(tpath), template):
            raise OSError(tpath)
        size, _, objects = scenes.layout(width, height, seed+200)
        background = [24., 82., 174., 228.][(k//4+k) % 4]
        decoy = scenes.procedural(seed+500, 2., (width, height))
        for condition, pixels, truth in [('clean', template, objects), ('noise', template, objects),
                                          ('distractor', decoy, [])]:
            img = scenes.render(pixels, size, background, objects)
            img = scenes.finish(img, seed+900 if condition != 'clean' else None)
            name = f'{seed}-{condition}'
            spath = out / f'{name}.png'
            if not cv2.imwrite(str(spath), img):
                raise OSError(spath)
            cases.append(dict(name=name, split=split, source=spath.name, template=tpath.name,
                              truth=truth, condition=condition))
    save(out/'cases.json', cases)


def run_job(runner, job, jobpath):
    save(jobpath, job)
    proc = subprocess.run([str(runner), str(jobpath)], capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f'{jobpath.name}: {proc.stderr.strip()}')
    result = json.loads(proc.stdout)
    if not result['exact_repeat']:
        raise RuntimeError(f'Inconsistent repeated results: {jobpath}')
    return result


def summarize(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(1, 2*tp+fp+fn),
                sum_scene_ms=sum(r['median_ms'] for r in rows),
                median_scene_ms=statistics.median(r['median_ms'] for r in rows),
                p95_run_ms=float(np.percentile([ms for r in rows for ms in r['repeat_ms']], 95)))


def measure(args):
    cases = json.loads(args.manifest.read_text())
    if not all(any(c['split'] == s for c in cases) for s in ('dev','test')):
        raise ValueError('Provide dev and test scene groups')
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Scene names must be unique')
    if args.photo_manifest:
        for case in json.loads(args.photo_manifest.read_text()):
            if case.get('kind') == 'real_category_probe':
                photo = dict(case, split='photo')
                for key in ('source','template'):
                    photo[key] = str((args.photo_manifest.parent / case[key]).resolve())
                cases.append(photo)
    arm_names = list(ARMS)
    rows, used = [], {'dev': set(), 'test': set()}
    lo, hi, step = args.scale_range
    scales = [round(lo+i*step, 3) for i in range(ceil(round((hi-lo)/step, 9)))] + [hi]
    base = dict(BASE, angle=args.angle, scales=scales)
    rng = random.Random(5111)
    with TemporaryDirectory() as tmp:
        for case in cases:
            src, templ = [(args.manifest.parent / case[k]).resolve() for k in ('source','template')]
            split = case['split']
            if split not in ('dev','test','photo'):
                raise ValueError(f'Unknown split {split}')
            digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,templ)]
            if split != 'photo':
                used[split].add(('source',digests[0]))
                if args.split_by == 'template':
                    used[split].add(('template',digests[1]))
                if case.get('group'):
                    used[split].add(('group',case['group']))
                if used['dev'] & used['test']:
                    raise ValueError('A source, acquisition group or held-out template occurs in both splits')
            order = arm_names.copy()
            rng.shuffle(order)
            for arm in order:
                settings = dict(base, backend=args.backend)
                if split == 'photo':
                    settings.update(scales=case['search']['scales'], score=case['search']['score'])
                settings.update(ARMS[arm])
                job = dict(settings, source=str(src), template=str(templ))
                result = run_job(args.runner, job, Path(tmp)/'job.json')
                first = result['first']
                matches = first['matches']
                ms = [r['total_ms'] for r in result['runs']]
                record = dict(name=case['name'], split=split, arm=arm, median_ms=statistics.median(ms),
                              repeat_ms=ms, first_ms=first['total_ms'], decode_ms=result['load']['decode_ms'],
                              setup_ms=result['construct_ms'], proposals=len(first['proposals']), matches=len(matches),
                              source_sha256=digests[0], template_sha256=digests[1])
                if split == 'photo':
                    p = np.asarray(case['points'])
                    q = np.array([[m['pos_x'],m['pos_y']] for m in matches]).reshape(-1,2)
                    close = np.linalg.norm(p[:,None]-q[None,:],axis=2) <= case['point_tolerance_px']
                    record['point_hits'] = int((maximum_bipartite_matching(csr_matrix(close),perm_type='column')>=0).sum()) if len(q) else 0
                    record['category_points'] = len(p)
                else:
                    metric = evaluate(matches, case['truth'])['strict']
                    record.update({k: metric[k] for k in ('tp','fp','fn')})
                rows.append(record)
    summaries = {split: {arm: summarize([r for r in rows if r['split']==split and r['arm']==arm])
                         for arm in ARMS} for split in ('dev','test')}
    choice, matched = select(summaries['dev'], args.time_ratio, args.max_p95_ms)
    save(args.output, dict(schema='fpm.scene-tuning.v1', backend=args.backend, split_by=args.split_by, baseline=base, arms=ARMS,
                           time_ratio=args.time_ratio, max_p95_ms=args.max_p95_ms, selected_on_dev=choice,
                           closest_candidate_budget_on_dev=matched, summaries=summaries, rows=rows))
    print(json.dumps(dict(choice=choice, closest_candidate_budget=matched, summaries=summaries), indent=2))


def select(dev, time_ratio, max_p95_ms):
    budget = time_ratio * dev['baseline']['sum_scene_ms']
    feasible = [a for a in dev if dev[a]['sum_scene_ms'] <= budget and
                (max_p95_ms is None or dev[a]['p95_run_ms'] <= max_p95_ms)]
    choice = min(feasible, key=lambda a: (-dev[a]['f1'], dev[a]['sum_scene_ms'])) if feasible else None
    closest = min(('more_candidates','wide_candidates'),
                  key=lambda a: abs(dev[a]['sum_scene_ms']-dev['short12']['sum_scene_ms']))
    return choice, closest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    gen = subs.add_parser('generate')
    gen.add_argument('directory', type=Path)
    run = subs.add_parser('measure')
    run.add_argument('manifest', type=Path)
    run.add_argument('--runner', type=Path, required=True)
    run.add_argument('--backend', type=int, choices=(0,1,2), default=0)
    run.add_argument('--photo-manifest', type=Path)
    run.add_argument('--split-by',choices=('acquisition','template'),default='acquisition',
                     help='Independent source frames/groups; template also holds out reference artwork')
    run.add_argument('--angle',type=float,default=180,help='Symmetric search tolerance in degrees')
    run.add_argument('--scale-range',type=float,nargs=3,default=[1.,1.,.25],metavar=('MIN','MAX','STEP'))
    run.add_argument('--time-ratio', type=float, default=2.)
    run.add_argument('--max-p95-ms', type=float, help='Optional development limit on pooled repeated matcher runtimes')
    run.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    if opts.action == 'generate':
        generate(opts.directory)
    else:
        if not np.isfinite(opts.time_ratio) or opts.time_ratio <= 0:
            parser.error('--time-ratio must be finite and positive')
        lo, hi, step = opts.scale_range
        if not (0 <= opts.angle <= 180 and round(opts.angle,2)==opts.angle):
            parser.error('--angle must be 0..180 with at most two decimal places')
        if not (.01<=lo<=hi<=8 and .01<=step<=8 and (hi-lo)/step<=63+1e-9
                and all(round(n,3)==n for n in opts.scale_range)):
            parser.error('Scale range requires 1..64 positive desktop scales, up to three decimal places')
        if opts.max_p95_ms is not None and (not np.isfinite(opts.max_p95_ms) or opts.max_p95_ms<=0):
            parser.error('--max-p95-ms must be finite and positive')
        measure(opts)
