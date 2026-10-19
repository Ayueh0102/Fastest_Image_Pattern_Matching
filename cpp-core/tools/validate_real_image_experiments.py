"""Validate benchmark provenance, CPU/Metal agreement, and the full-search bridge."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

from real_image_experiments import ROOT, angle_distance, corners, overlap, suppress
from pose_experiment import Scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'cpp-core/docs/real-image-experiments.json')
    parser.add_argument('--output', type=Path, default=ROOT/'cpp-core/docs/real-image-validation.json')
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    entries = {(e['job']['case']['name'], e['job']['method']): e for e in report['results']}
    assert len(entries) == 20
    for relative, expected in report['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
    for path, expected in report['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    build = Path(report['results'][0]['job']['build'])
    for name, expected in report['binary_sha256'].items():
        path = build/name if name == 'libfpm_c.dylib' else build/'experiments'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

    def match(x, y, angle=0, scale=1, score=.9):
        return dict(pos_x=x, pos_y=y, angle=angle, scale=scale, score=score,
                    corners=corners([x,y], angle, scale, (10, 20)))

    a = match(0, 0)
    assert overlap(a, a) == 1
    assert overlap(a, match(100, 0)) == 0
    assert abs(overlap(a, match(10, 0))-.5) < 1e-12
    assert abs(overlap(a, match(0, 0, scale=.5))-1) < 1e-12
    for angle in [-179, -35, 89, 179]:
        rotated = match(0, 0, angle)
        assert abs(overlap(rotated, rotated)-1) < 1e-12
        assert abs(overlap(a, rotated)-overlap(rotated, a)) < 1e-12
    assert len(suppress([a, match(0, 0, score=.8), match(100, 0)], 10, .2, .5)) == 2
    assert angle_distance(179, -179) == 2
    output = dict(provenance='pass', geometry='pass', cli_agreement=[], cpu_metal_agreement=[], image_channels=[])

    for case_name in ['src7', 'src3', 'src5_45', 'controlled_scale']:
        cpu = entries[case_name, 'hybrid_cpu']['result']['runs'][-1]['matches']
        gpu = entries[case_name, 'hybrid_metal']['result']['runs'][-1]['matches']
        assert len(cpu) == len(gpu)
        remaining, differences, nearby_poses = set(range(len(gpu))), [], []
        case = entries[case_name, 'hybrid_cpu']['job']['case']
        for a in cpu:
            i = min(remaining, key=lambda j: math.hypot(a['pos_x']-gpu[j]['pos_x'], a['pos_y']-gpu[j]['pos_y']))
            b = gpu[i]; remaining.remove(i)
            d = dict(position_px=math.hypot(a['pos_x']-b['pos_x'], a['pos_y']-b['pos_y']),
                     angle_deg=angle_distance(a['angle'], b['angle']), scale=abs(a['scale']-b['scale']),
                     score=abs(a['score']-b['score']))
            # Last-round spacing is 2 / 2**5 = 1/16 pixel. Nearly tied
            # FP32 scores can choose adjacent cells; quantify the regret below.
            assert d['position_px'] <= .0625+1e-6 and d['angle_deg'] < .001 and d['scale'] < 1e-5 and d['score'] < 3e-5, d
            if d['position_px'] > .001:
                nearby_poses.extend([a, b])
            differences.append(d)
        gaps = []
        if nearby_poses:
            source = cv2.imread(case['source'], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            template = cv2.imread(case['template'], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            sh, sw = source.shape
            poses = np.array([[m['pos_x']-sw/2, sh/2-m['pos_y'], m['angle'], math.log(m['scale'])]
                              for m in nearby_poses], np.float32)
            with Scorer(build, source, template, 'cpu', 1) as scorer:
                scores, _ = scorer.score(poses)
            gaps = [abs(float(a)-float(b)) for a, b in scores.reshape(-1, 2)]
            assert max(gaps) < 3e-6, gaps
        output['cpu_metal_agreement'].append(dict(case=case_name, count=len(cpu),
            maxima={k: max(d[k] for d in differences) for k in differences[0]},
            adjacent_cell_score_gaps_using_same_cpu_scorer=gaps))
        for method in ['cpu_fixed', 'cpu_fast', 'hybrid_cpu', 'hybrid_metal']:
            runs = entries[case_name, method]['result']['runs']
            assert all(r['matches'] == runs[0]['matches'] for r in runs), (case_name, method, 'nondeterministic results')

    for case_name in ['src7', 'src3', 'src5_45']:
        case = entries[case_name, 'cpu_fixed']['job']['case']
        for name in ['source', 'template']:
            image = cv2.imread(case[name], cv2.IMREAD_UNCHANGED)
            distinct = image.ndim == 3 and bool(np.any(image[:,:,:3] != image[:,:,:1]))
            assert not distinct
            output['image_channels'].append(dict(path=case[name], shape=list(image.shape), distinct_rgb=distinct))
        for method in ['cpu_fixed', 'cpu_fast']:
            command = [str(build/'fpm_cli'), '--src', case['source'], '--tmpl', case['template'],
                       '--target-num', str(case['limit']), '--score', str(case['score']),
                       '--overlap', str(case['overlap']), '--angle', str(case['angle']), '--min-area', '256']
            if method == 'cpu_fast':
                command.append('--fast')
            cli = json.loads(subprocess.check_output(command, text=True))['matches']
            reference = entries[case_name, method]['result']['runs'][-1]['matches']
            assert len(cli) == len(reference)
            maximum = 0.
            for a, b in zip(cli, reference):
                for key in ['pos_x','pos_y','angle','score']:
                    maximum = max(maximum, abs(a[key]-b[key]))
                maximum = max(maximum, float(np.max(np.abs(np.array(a['corners'])-b['corners']))))
            assert maximum < 1e-7, (case_name, method, maximum)
            output['cli_agreement'].append(dict(case=case_name, method=method, count=len(cli), max_numeric_difference=maximum))
    output['benchmark_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    output['status'] = 'pass'
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
