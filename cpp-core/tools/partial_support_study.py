"""Compare full-template search with quadrant consensus and a specified identity patch.

Uses the stage34 generated cases. Coordinates of the identity region are part of
the template design. Object locations and labels enter only the final evaluation.
"""
import argparse
import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from many_object_accuracy import evaluate
from many_object_scenes import _linear
from scene_tuning import run_job, save


def compatible(a, b):
    return (math.hypot(a['pos_x']-b['pos_x'], a['pos_y']-b['pos_y']) <= 4
            and abs(math.remainder(a['angle']-b['angle'], 360)) <= 2
            and abs(math.log(a['scale']/b['scale'])) <= math.log(1.03))


def full_center(match, offset, region):
    shift = _linear(match['angle'], match['scale']) @ offset
    return dict(match, pos_x=match['pos_x']-shift[0], pos_y=match['pos_y']-shift[1], region=region)


def combine(matches, votes):
    # Full-template results have priority. Add a new object only if at least
    # two different quadrants agree on a single pose.
    combined = list(matches)
    for p in sorted(votes, key=lambda v: -v['score']):
        if any(compatible(p, q) for q in combined):
            continue
        peers = {}
        for q in votes:
            if compatible(p, q) and q['score'] > peers.get(q['region'], {}).get('score', -1.):
                peers[q['region']] = q
        if len(peers) >= 2:
            group = list(peers.values())
            fused = dict(p)
            for key in ('pos_x', 'pos_y', 'scale'):
                fused[key] = np.mean([v[key] for v in group])
            fused['angle'] += np.mean([math.remainder(v['angle']-p['angle'], 360) for v in group])
            combined.append(fused)
    return combined


def identity_score(image, template, box, pose):
    # Leave one source-interpolation pixel around the marked patch.
    x0, y0, x1, y1 = box
    y, x = np.mgrid[y0+2:y1-2, x0+2:x1-2]
    h, w = template.shape
    xy = _linear(pose['angle'], pose['scale']) @ np.stack([x.ravel()-w/2, y.ravel()-h/2])
    sx = (xy[0]+pose['pos_x']).reshape(x.shape).astype(np.float32)
    sy = (xy[1]+pose['pos_y']).reshape(y.shape).astype(np.float32)
    if sx.min()<0 or sy.min()<0 or sx.max()>=image.shape[1]-1 or sy.max()>=image.shape[0]-1:
        return None
    observed = cv2.remap(image, sx, sy, cv2.INTER_LINEAR).astype(float)
    expected = template[y, x].astype(float)
    a, b = observed-observed.mean(), expected-expected.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a*b).sum()/norm) if norm else None


def measure(case, base, runner, backend, tmp):
    image_path, template_path = [base / case[k] for k in ('source', 'template')]
    image, templ = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (image_path, template_path)]
    h, w = templ.shape
    job = {k: case[k] for k in ('scales','angle','limit','proposal_score','score','overlap','stop_layer','subpixel')}
    job.update(source=str(image_path), backend=backend, repeats=1, warmup_seconds=0)
    def search(path):
        result = run_job(runner, dict(job, template=str(path)), tmp/'job.json')['first']
        return result['matches'], result['total_ms']
    matches, base_ms = search(template_path)
    votes = []
    patch_ms = []
    for i, (x0, x1) in enumerate(((0,w//2), (w//2,w))):
        for j, (y0, y1) in enumerate(((0,h//2), (h//2,h))):
            path = tmp/'quadrant.png'
            if not cv2.imwrite(str(path),templ[y0:y1,x0:x1]):
                raise OSError(path)
            found, ms = search(path)
            patch_ms.append(ms)
            offset = [(x0+x1-w)/2, (y0+y1-h)/2]
            votes.extend(full_center(v, offset, 2*i+j) for v in found)
    start = time.perf_counter()
    combined = combine(matches, votes)
    # This region is known on both printed template designs before search.
    box = (w//4,h//4,w//2,h//2)
    scored = [identity_score(image, templ, box, p) for p in combined]
    verified = [p for p, s in zip(combined, scored) if s is not None and s >= .5]
    verify_ms = (time.perf_counter()-start)*1000.
    metrics = {}
    for name, poses in [('full',matches), ('full_plus_votes',combined), ('with_identity',verified)]:
        result = evaluate(poses, case['truth'])['strict']
        metrics[name] = {k:result[k] for k in ('tp','fp','fn','unmatched_truth_indices')}
        metrics[name]['distractor_hits'] = sum(any(math.hypot(p['pos_x']-d['pos_x'],p['pos_y']-d['pos_y']) <= 20
                                                    for d in case['distractors']) for p in poses)
    return dict(name=case['name'], split=case['split'], kind=case['kind'], metrics=metrics,
                region_candidates=len(votes), identity_scores=scored, full_first_ms=base_ms,
                quadrant_first_ms=patch_ms, consensus_verification_ms=verify_ms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--runner',type=Path,required=True)
    parser.add_argument('--backend',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with TemporaryDirectory() as tmp:
        rows = [measure(c,args.manifest.resolve().parent,args.runner,args.backend,Path(tmp))
                for c in json.loads(args.manifest.read_text())]
    save(args.output,dict(backend=args.backend, identity_ncc=.5, min_region_votes=2, rows=rows))
