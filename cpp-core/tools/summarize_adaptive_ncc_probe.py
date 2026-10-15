"""Validate and summarize the exploratory projection probe without running it."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    d = json.loads(path.read_text())
    margin = d['protocol']['diagnostic_margin']
    if digest(ROOT/'cpp-core/tools/probe_adaptive_ncc_bounds.py') != d['source_sha256']:
        raise ValueError('Probe source changed after measurement')
    for name, expected in d['input_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Input changed: {name}')
    bound_pairs = 0
    for c in d['cases']:
        assert c['windows'] == len(c['observations']) == 93
        for row in c['observations']:
            last_global = last_block = float('inf')
            for stage in ('4', '8', '16', '32', '64', 'uniform64'):
                p = row['values'][stage]
                if p is None:
                    continue
                assert p['exact'] <= p['block_upper']+margin
                assert p['block_upper'] <= p['upper']+margin
                if stage != 'uniform64':
                    assert p['upper'] <= last_global+margin
                    assert p['block_upper'] <= last_block+margin
                    last_global, last_block = p['upper'], p['block_upper']
                bound_pairs += 1
        for stage, s in c['stages'].items():
            assert s['bound_violations'] == s['block_bound_violations'] == 0
            values = [r['values'][stage] for r in c['observations'] if r['values'][stage] is not None]
            for key, counts in s['thresholds'].items():
                tau = float(key)
                rebuilt = dict(above_threshold=sum(p['exact'] >= tau for p in values),
                    below_threshold=sum(p['exact'] < tau for p in values),
                    rejected=sum(p['upper']+margin < tau for p in values),
                    block_rejected=sum(p['block_upper']+margin < tau for p in values),
                    false_rejections=sum(p['exact'] >= tau and p['upper']+margin < tau for p in values),
                    block_false_rejections=sum(p['exact'] >= tau and p['block_upper']+margin < tau for p in values),
                    approximation_false_rejections=sum(p['exact'] >= tau and p['approximate'] is not None
                                                       and p['approximate'] < tau for p in values))
                assert counts == rebuilt
                assert counts['false_rejections'] == counts['block_false_rejections'] == 0
    groups = {}
    for name, cases in (('native', [c for c in d['cases'] if c['level'] == 0]),
                        ('pyramid_top', [c for c in d['cases'] if c['level'] > 0])):
        stages = {}
        for stage in ('4', '8', '16', '32', '64', 'uniform64'):
            counts = [c['stages'][stage]['thresholds'][str(c['original_layer_threshold_at_min_score_point8'])]
                      for c in cases]
            stages[stage] = {key: sum(x[key] for x in counts) for key in counts[0]}
        groups[name] = dict(cases=len(cases), windows=sum(c['windows'] for c in cases),
            degenerate=sum(c['degenerate_windows'] for c in cases), stages=stages)
    return dict(purpose=d['purpose'], probe_sha256=d['source_sha256'], raw_sha256=digest(path),
        summary_tool_sha256=digest(Path(__file__)), source_commit=d['source_commit'],
        checks=dict(probe_self_checks=d['checks'], inputs_unchanged=True, probe_source_unchanged=True,
            observed_bound_pairs=bound_pairs, bounds_and_nested_monotonicity_passed=True,
            summary_recomputed_from_observations=True, all_recorded_thresholds_zero_false_rejection=True),
        threshold_rule='0.8 at native, 0.8*0.9**level at pyramid top', groups=groups,
        diagnostic_cost=dict(elapsed_python_seconds=d['elapsed_python_seconds'],
            process_peak_rss_raw=d['process_peak_rss_raw'], process_peak_rss_units=d['process_peak_rss_units'],
            note='Includes decoding, repeated diagnostic arrays, observations; not native algorithm performance'),
        limitations=['No matcher, Metal, full rotation/scale search or detection GT benchmark',
            'Sampled source windows and constructed controls, not actual fine requests',
            'Floating point margin is diagnostic, not a production safety proof',
            'Four-block comparison excludes existing partial-dot pruning and its enable heuristics',
            'Initial development attempts stopped on missing Src5.bmp and Src6.bmp; corrected names used'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.input), indent=2, allow_nan=False)+'\n')
