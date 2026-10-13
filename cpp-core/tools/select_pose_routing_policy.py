"""Select a transparent threshold policy from calibration timings only.

The grid is deliberately coarse (powers of two). This writes evidence, not code;
freeze the selected policy before evaluating held-out textures and image sizes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def evaluate(rows, pose_threshold, batch_threshold):
    decisions = []
    for row in rows:
        metal = row['pose_samples'] >= pose_threshold and row['batch_samples'] >= batch_threshold
        selected_ms = row['metal_ms'] if metal else row['cpu_ms']
        decisions.append(dict(case=row['case'], selected_backend='metal' if metal else 'cpu',
            selected_ms=selected_ms, fastest_measured_ms=min(row['cpu_ms'], row['metal_ms']),
            selected_over_fastest=selected_ms/min(row['cpu_ms'], row['metal_ms']),
            selected_over_cpu=selected_ms/row['cpu_ms']))
    ratios = [r['selected_over_fastest'] for r in decisions]
    return dict(min_pose_samples=pose_threshold, min_batch_samples=batch_threshold,
        min_iterations=4, cpu_polish_iterations=0,
        geometric_mean_selected_over_fastest=math.exp(sum(map(math.log, ratios))/len(ratios)),
        worst_selected_over_fastest=max(ratios),
        selected_slower_than_cpu_by_10_percent=sum(r['selected_over_cpu'] > 1.1 for r in decisions),
        decisions=decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.calibration.read_text())
    if not (report['suite'] == 'calibration' and report.get('complete') and
            report['correctness_passed'] and report['accuracy_passed'] and
            report['source_hashes_unchanged_during_run'] and
            report['binary_hash_unchanged_during_run']):
        raise ValueError('A complete, verified calibration-only report is required')
    rows = report['comparisons']
    if len(rows) != 24 or any(r['split'] != 'calibration' or not r['identical_inputs'] for r in rows):
        raise ValueError('Expected all 24 paired calibration cases')
    if any(not math.isfinite(r.get(key, float('nan'))) or r[key] <= 0
           for r in rows for key in ('cpu_ms', 'metal_ms')):
        raise ValueError('Paired positive CPU and Metal timings are required')
    candidates = [evaluate(rows, s, b) for s in [0]+[2**k for k in range(12, 22)]
                  for b in [0]+[2**k for k in range(14, 26)]]
    candidates.sort(key=lambda c: (c['geometric_mean_selected_over_fastest'],
        c['worst_selected_over_fastest'], c['selected_slower_than_cpu_by_10_percent'],
        c['min_pose_samples'], c['min_batch_samples']))
    result = dict(calibration=args.calibration.name,
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        objective='Minimize geometric mean of selected median / minimum CPU-or-Metal median, equally weighting the 24 calibration cases.',
        tie_break='Worst ratio, number slower than CPU by >10%, then lower per-pose and batch thresholds.',
        grid=dict(min_pose_samples=[0]+[2**k for k in range(12, 22)],
                  min_batch_samples=[0]+[2**k for k in range(14, 26)]),
        scope='Warm M4 Pro / CPU8 / Metal polish0 local refinement. Actual iterations are unknown to routing. Iteration floor4 is retained, not fitted here. No held-out or full-image timings used.',
        original_policy=evaluate(rows, 65536, 1048576), selected_policy=candidates[0],
        candidates=[{k: v for k, v in c.items() if k != 'decisions'} for c in candidates])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in candidates[0].items() if k != 'decisions'}, indent=2))


if __name__ == '__main__':
    main()
