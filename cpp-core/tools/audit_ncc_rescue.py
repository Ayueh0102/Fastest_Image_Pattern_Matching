"""Audit every saved NCC rescue call without running a solver or pixel oracle.

Accepts the formal benchmark JSON or its gzip archive. It preserves failed
performance gates and separates supplied-seed accuracy, calibrated full-image
detections and near-periodic diagnostic failures. No matching is performed.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'cpp-core/tools'))
from audit_resident_refinement import match_detections

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, required=True, help='Formal benchmark JSON or JSON.gz')
parser.add_argument('--output', type=Path, required=True, help='Compact audit JSON, distinct from the report')
args = parser.parse_args()
source, destination = args.report.resolve(), args.output.resolve()
if source == destination:
    parser.error('--output must not overwrite --report')
raw = source.read_bytes()
report = json.loads(gzip.decompress(raw) if source.suffix == '.gz' else raw)
errors = []
methods = ('old_cpu', 'old_resident_verified', 'rescue_cpu', 'rescue_resident_verified')
groups = defaultdict(dict)
for run in report['runs']:
    case, method = run['job']['case']['name'], run['job']['method']
    if method in groups[case]:
        errors.append(case+': duplicate method '+method)
    groups[case][method] = run

def require(condition, context, message):
    if not bool(condition):
        errors.append(context+': '+message)

def distribution(values):
    return dict(count=len(values), min=float(min(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(max(values))) if values else None

def records(run):
    result = run['result']
    return [result['first'], *result['measured']]

require(report.get('complete') and len(report['runs']) == report.get('expected_runs'), 'report', 'incomplete run matrix')
require(bool(groups) and report.get('unavailable_runs') == 0, 'report', 'empty/unavailable benchmark')
require(report.get('baseline_verification', {}).get('verified'), 'report', 'unverified baseline provenance')
require(report.get('source_hashes_unchanged') and report.get('binary_hashes_unchanged'), 'report', 'failed completion provenance snapshot')

def geometric(pose, score, truth):
    position = math.hypot(pose[0]-truth[0], pose[1]-truth[1])
    angle = abs((pose[2]-truth[2]+180)%360-180)
    scale = abs(math.exp(pose[3]-truth[3])-1)
    return dict(position_px=position, angle_deg=angle, relative_scale=scale,
                precise_pass=score >= .8 and position <= .5 and angle <= .5 and scale <= .01)

precision = defaultdict(lambda: dict(cases=0, unique_seed_count=0, passed_every_call=0,
                                     checked_seed_calls=0, passed_seed_calls=0, failures=[]))
no_retry = defaultdict(lambda: dict(warm=[], cold=[], cases=set(), worst=None))
common, negative, periodic, full = {}, [], [], []
checked_calls = checked_seed_calls = 0
max_score_delta = 0.
max_oracle_error = 0.
per_method_memory = defaultdict(list)

for case_name, group in groups.items():
    require(set(group) == set(methods), case_name, 'missing method or unexpected method')
    for method, run in group.items():
        case, result = run['job']['case'], run['result']
        ctx = case_name+'/'+method
        recs, checks = records(run), result['validation']['calls']
        expected_calls = report['protocol']['repeats']+1
        require(len(recs) == expected_calls and len(checks) == expected_calls, ctx, 'call count mismatch')
        n = result['validation']['seed_count']
        require(n > 0, ctx, 'empty seed batch')
        config = dict(report['protocol']['config'], **case.get('config', {}))
        if case['scope'] == 'full':
            config.update(optimize_scale=len(case['scales']) > 1, optimize_angle=case['angle'] > 0)
            original = np.asarray(recs[0]['result']['initial_poses'], np.float32)
        else:
            original = np.asarray(case['initial_poses'], np.float32)
            # The boundary fixtures contain explicitly supplied seeds; no input
            # files or pixels are needed to verify their persisted identity.
            expected = report['fixtures'][case_name]['input_sha256']
            require(result['input_sha256'] == expected, ctx, 'fixture input hash mismatch')
            require(hashlib.sha256(original.tobytes()).hexdigest() == expected['seeds'], ctx, 'seed bytes mismatch')
        require(original.shape == (n, 4), ctx, 'seed shape mismatch')
        limits = np.asarray([config['max_translation']]*2+[
            config['max_angle_degrees'] if config['optimize_angle'] else 0.,
            config['max_log_scale'] if config['optimize_scale'] else 0.])
        lower = np.maximum(original.astype(np.float64)-limits, [-1e6, -1e6, -36000., -3.])
        upper = np.minimum(original.astype(np.float64)+limits, [1e6, 1e6, 36000., 3.])
        truth_passes = []
        for call_index, (record, check) in enumerate(zip(recs, checks)):
            label = ctx+('/first' if call_index == 0 else '/repeat'+str(call_index-1))
            value = record['result']
            poses, scores = np.asarray(value['poses']), np.asarray(value['scores'])
            exact = np.asarray(check['cpu_rechecked_scores'])
            initial_scores = np.asarray(result['validation']['initial_scores'])
            require(poses.shape == (n, 4) and scores.shape == exact.shape == initial_scores.shape == (n,), label, 'shape/count mismatch')
            require(len(value['stats']) == n, label, 'stats count mismatch')
            require(np.isfinite(poses).all() and np.isfinite(scores).all(), label, 'nonfinite result')
            require(np.array_equal(scores, exact), label, 'reported/CPU score mismatch')
            require(np.all(exact >= initial_scores), label, 'original seed NCC regression')
            require(np.all((poses >= lower) & (poses <= upper)), label, 'original bounds violation')
            for axis in range(4):
                if limits[axis] == 0:
                    require(np.array_equal(poses[:, axis], original[:, axis]), label, 'frozen axis changed')
            for index in case.get('expected_invalid', []):
                require(scores[index] == -2 and np.array_equal(poses[index], original[index]), label, 'invalid fixture changed')
            if case.get('expected_no_detection'):
                require(np.all(exact < case.get('score', .8)), label, 'negative crossed acceptance threshold')
            if case['scope'] == 'full':
                require(np.array_equal(np.asarray(value['initial_poses'], np.float32), original), label, 'coarse seeds changed across calls')
            if case.get('truth_accuracy') or case.get('diagnostic_truth'):
                accuracy = [geometric(p, s, case['truth']) for p, s in zip(poses, exact)]
                require([e['precise_pass'] for e in accuracy] == [e['precise_pass'] for e in check['accuracy']], label, 'saved precision disagrees with geometry')
                truth_passes.append([e['precise_pass'] for e in accuracy])
                if case.get('truth_accuracy') and method.startswith('rescue'):
                    require(all(e['precise_pass'] for e in accuracy), label, 'required rescue precision failure')
            diagnostic = value['metrics'].get('rescue')
            iteration_cap = config['max_iterations']
            if method.startswith('rescue'):
                require(diagnostic is not None, label, 'missing rescue diagnostics')
                policy = report['policy']
                d = diagnostic
                samples = int(np.prod(report['fixtures'][case_name]['shapes']['template'])) if case['scope'] == 'local' else None
                if samples is not None:
                    reserved = samples*(16*policy['short_iterations']+config['max_iterations'])
                    require(d['reserved_retry_sample_iterations'] == reserved*d['selected_seeds'], label, 'incorrect work reservation')
                require(0 <= d['accepted_seeds'] <= d['selected_seeds'] <= min(d['eligible_seeds'], policy['max_rescued_seeds']), label, 'parent count violation')
                require(0 <= d['retry_starts'] <= 16*d['selected_seeds'] and 0 <= d['continuations'] <= d['selected_seeds'], label, 'restart cap violation')
                require(d['reserved_retry_sample_iterations'] <= policy['max_retry_sample_iterations'], label, 'work cap violation')
                require(d['command_buffers'] == d['cpu_waits'] and d['auxiliary_buffer_bytes'] <= 16*1024*1024, label, 'GPU wait/workspace violation')
                require(d['selected_backend'] == int(method == 'rescue_resident_verified'), label, 'silent backend fallback')
                iteration_cap += 16*policy['short_iterations']+config['max_iterations']
                if d['selected_seeds'] == 0:
                    overhead = value['metrics']['wall_ms']-d['baseline_ms']
                    entry = no_retry[method]
                    entry['cold' if call_index == 0 else 'warm'].append(overhead)
                    entry['cases'].add(case_name)
                    if entry['worst'] is None or overhead > entry['worst']['ms']:
                        entry['worst'] = dict(case=case_name, call=call_index, ms=overhead)
                    require(overhead >= -1e-8, label, 'negative same-call overhead')
            require(all(0 <= s['iterations'] <= iteration_cap and s['evaluations'] >= 1 and 0 <= s['status'] <= 3 for s in value['stats']), label, 'invalid solver stats')
            checked_calls += 1
            checked_seed_calls += n
            max_score_delta = max(max_score_delta, float(np.max(np.abs(scores-exact))))
        # Inspect all persisted independent-oracle comparisons without re-running
        # any pixel operation; currently the benchmark stores the final call.
        stored_oracle = result['validation']['oracle']
        observed_error = max(abs(o['score']-o['oracle']) for o in stored_oracle)
        max_oracle_error = max(max_oracle_error, observed_error)
        require(observed_error <= report['protocol']['numerical_tolerance'], ctx, 'saved float64 oracle failure')
        if case.get('truth_accuracy'):
            key = case['split']+'/'+method
            bucket = precision[key]
            bucket['cases'] += 1; bucket['unique_seed_count'] += n
            all_calls = np.asarray(truth_passes)
            stable = all_calls.all(axis=0)
            bucket['passed_every_call'] += int(stable.sum())
            bucket['checked_seed_calls'] += all_calls.size
            bucket['passed_seed_calls'] += int(all_calls.sum())
            bucket['failures'] += [dict(case=case_name, seed=int(index)) for index in np.flatnonzero(~stable)]
        if case_name == 'heldout_16':
            target = 3
            common[method] = dict(checked_calls=len(recs), precise_passes=sum(p[target] for p in truth_passes),
                initial_score=result['validation']['initial_scores'][target],
                final_score_range=[min(r['result']['scores'][target] for r in recs), max(r['result']['scores'][target] for r in recs)],
                worst_position_px=max(geometric(r['result']['poses'][target], r['result']['scores'][target], case['truth'])['position_px'] for r in recs),
                median_outer_ms=result['algorithm_wall']['median_ms'], median_api_ms=result['c_api_wall']['median_ms'])
        if case.get('expected_no_detection'):
            negative.append(dict(case=case_name, method=method, checked_seed_calls=n*len(recs),
                accepted_count=sum(sum(s >= case.get('score', .8) for s in r['result']['scores']) for r in recs),
                maximum_score=max(max(r['result']['scores']) for r in recs)))
        if case.get('diagnostic_truth'):
            periodic.append(dict(case=case_name, method=method, checked_seed_calls=n*len(recs),
                precise_pass_count=sum(sum(p) for p in truth_passes),
                score_range=[min(min(r['result']['scores']) for r in recs), max(max(r['result']['scores']) for r in recs)],
                position_error_px=max(geometric(p,s,case['truth'])['position_px'] for r in recs for p,s in zip(r['result']['poses'],r['result']['scores'])),
                selected_retry_parent_calls=sum(r['result']['metrics'].get('rescue', {}).get('selected_seeds', 0) for r in recs)))
        if case['scope'] == 'full':
            baseline_method = 'old_cpu' if method.endswith('_cpu') else 'old_resident_verified'
            baseline_records = records(group[baseline_method])
            agreements = [match_detections(case['truth'] if case.get('truth') else old['result']['matches'],
                            r['result']['matches'], calibrated=bool(case.get('truth')))
                          for r,old in zip(recs,baseline_records)]
            require(all(a['passed'] for a in agreements), ctx, 'full-image geometry/count failure')
            full.append(dict(case=case_name, method=method, calibrated_truth=bool(case.get('truth')),
                unique_truth_objects=len(case['truth']) if case.get('truth') else None,
                checked_calls=len(recs), passed_calls=sum(a['passed'] for a in agreements),
                detected_counts=[len(r['result']['matches']) for r in recs],
                median_outer_ms=result['algorithm_wall']['median_ms']))
        if method.startswith('rescue'):
            base_method = 'old_cpu' if method.endswith('_cpu') else 'old_resident_verified'
            baseline = group[base_method]['result']
            require(result['input_sha256'] == baseline['input_sha256'], ctx, 'different baseline input hashes')
            for index, (record, old) in enumerate(zip(recs,records(group[base_method]))):
                require(np.all(np.asarray(record['result']['scores']) >= np.asarray(old['result']['scores'])), ctx+'/'+str(index), 'same-backend NCC regression')
            delta = {k: v-baseline['memory']['max_observed_owned_buffers'][k]
                     for k,v in result['memory']['max_observed_owned_buffers'].items()}
            per_method_memory[method].append(dict(case=case_name, split=case['split'],
                retained_delta_bytes=delta, process_peak_delta_bytes=result['memory']['process_peak_bytes']-baseline['memory']['process_peak_bytes'],
                peak_rss_bytes=result['memory']['process_peak_bytes']))

memory = {}
for method, rows in per_method_memory.items():
    memory[method] = dict(cases=len(rows), retained_delta_bytes={key: dict(
        minimum=min(r['retained_delta_bytes'][key] for r in rows),
        maximum=max(r['retained_delta_bytes'][key] for r in rows),
        maximum_case_count=sum(r['retained_delta_bytes'][key] == max(x['retained_delta_bytes'][key] for x in rows) for r in rows),
        maximum_case_examples=[r['case'] for r in rows if r['retained_delta_bytes'][key] == max(x['retained_delta_bytes'][key] for x in rows)][:5])
        for key in ('host_buffer_bytes', 'gpu_buffer_bytes', 'gpu_allocated_bytes')},
        process_peak_delta_bytes=distribution([r['process_peak_delta_bytes'] for r in rows]),
        largest_positive_process_deltas=sorted(rows,key=lambda r:r['process_peak_delta_bytes'],reverse=True)[:5],
        largest_process_peak=max(rows,key=lambda r:r['peak_rss_bytes']))
speed_failures = [dict(case=c['case'], method=c['method'], algorithm_slowdown=c['algorithm_slowdown'],
    refinement_slowdown=c['refinement_slowdown'], no_retries=c.get('no_retries'),
    easy_gate=c.get('easy_case_overhead_passed'), full_gate=c.get('full_pipeline_overhead_passed'))
    for c in report['comparisons'] if c.get('easy_case_overhead_passed') is False or c.get('full_pipeline_overhead_passed') is False]
summary = dict(source_report=str(source), source_report_sha256=hashlib.sha256(raw).hexdigest(),
    audit_script='cpp-core/tools/audit_ncc_rescue.py', audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    audit_dependency_sha256={'cpp-core/tools/audit_resident_refinement.py':
        hashlib.sha256((ROOT/'cpp-core/tools/audit_resident_refinement.py').read_bytes()).hexdigest()},
    audit_scope='All first and measured saved calls; arithmetic, bounds, exact saved CPU-score and geometry checks only. No solver/GPU/pixel oracle executed.',
    report_snapshot=dict(baseline_verification=report['baseline_verification'],
        source_hashes_unchanged_at_completion=report['source_hashes_unchanged'], binary_hashes_unchanged_at_completion=report['binary_hashes_unchanged'],
        accuracy_acceptance_passed=report['accuracy_acceptance_passed'], acceptance_passed=report['acceptance_passed'],
        note='Snapshot statements belong to the formal run; current working-tree hashes can change afterwards.'),
    expected_methods=list(methods), unique_cases=len(groups), variant_runs=len(report['runs']),
    checked_calls=checked_calls, checked_seed_calls=checked_seed_calls,
    unique_truth_seed_definition='Each (fixture case, supplied seed index) counted once, excluding repeats and methods; source-size/workload are LOCAL supplied-seed fixtures, not global detections.',
    unique_truth_seeds=sum(v['unique_seed_count'] for k,v in precision.items() if k.endswith('/rescue_cpu')),
    precision=dict(precision), common_failure=common,
    max_saved_cpu_score_error=max_score_delta, max_saved_final_call_float64_oracle_error=max_oracle_error,
    independent_oracle_limitation='Float64 oracle samples are persisted for the last call only; all calls have exact frozen-CPU checks and all geometric truth gates were independently re-evaluated here.',
    negative_cases=negative, periodic_challenges=periodic,
    periodic_limitation='Near-periodic inputs still return an approximately 8-pixel wrong position with high NCC; the low-score trigger does not run. These diagnostic failures remain visible and are not proof of universal basin recovery.',
    full_image=full, unique_calibrated_full_truth_objects=sum(
        value['unique_truth_objects'] for value in full if value['method'] == 'rescue_cpu' and value['calibrated_truth']),
    no_retry_same_call_overhead_ms={k:dict(unique_cases=len(v['cases']), warm=distribution(v['warm']),
        first_call=distribution(v['cold']), worst=v['worst']) for k,v in no_retry.items()},
    no_retry_overhead_definition='C API wall_ms minus rescue.baseline_ms from the SAME call; excludes Python/full-search overhead. Does not override the independently measured performance gate failures.',
    memory=memory, memory_note='Retained host/GPU request bytes are owned capacities. RSS and Metal allocated bytes vary by process/driver; shared GPU memory overlaps RSS and must not be added.',
    failed_performance_gates=speed_failures, audit_errors=errors, audit_passed=not errors)
destination.parent.mkdir(parents=True,exist_ok=True)
if hashlib.sha256(source.read_bytes()).hexdigest() != summary['source_report_sha256']:
    raise RuntimeError('Source report changed during read-only auditing')
destination.write_text(json.dumps(summary,indent=2,default=lambda value:value.item())+'\n')
print(json.dumps(dict(output=str(destination), audit_passed=not errors, errors=errors[:20],
    unique_truth_seeds=summary['unique_truth_seeds'], checked_calls=checked_calls,
    no_retry_same_call_overhead_ms=summary['no_retry_same_call_overhead_ms'],
    speed_failures=len(speed_failures)),indent=2))
if errors:
    raise SystemExit('Saved result audit failed; all errors retained in output')
