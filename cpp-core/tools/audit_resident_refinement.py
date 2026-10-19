"""Audit saved resident benchmark results; never run a solver or timed workload.

Every first/measured call is checked. Raw GPU-only CPU-score regressions are
reported, including float-sized changes; verified methods must remain exactly
CPU-scored and monotonic. Original Src7 is agreement with the frozen CPU result,
not calibrated ground truth. Matching thresholds do not establish global search
completeness on an unlabeled photograph.
"""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path


SCORE_TOLERANCE = 3e-5
EXPECTED_METHODS = ('legacy_cpu', 'legacy_hybrid', 'resident', 'resident_verified')


def angle_distance(first, second):
    return abs((first-second+180.) % 360.-180.)


def local_truth(poses, scores, truth):
    errors = [dict(position_px=math.hypot(p[0]-truth[0], p[1]-truth[1]),
                   angle_deg=angle_distance(p[2], truth[2]),
                   relative_scale=abs(math.exp(p[3]-truth[3])-1.), score=score)
              for p, score in zip(poses, scores)]
    failures = [i for i, e in enumerate(errors) if e['score'] < .8 or e['position_px'] > .5
                or e['angle_deg'] > .5 or e['relative_scale'] > .01]
    return dict(checked=len(errors), passed=len(errors)-len(failures), failed_indices=failures,
                max_position_px=max((e['position_px'] for e in errors), default=None),
                max_angle_deg=max((e['angle_deg'] for e in errors), default=None),
                max_relative_scale=max((e['relative_scale'] for e in errors), default=None))


def geometry(reference, candidate):
    return dict(position_px=math.hypot(candidate['pos_x']-reference['pos_x'], candidate['pos_y']-reference['pos_y']),
                angle_deg=angle_distance(candidate['angle'], reference['angle']),
                relative_scale=abs(candidate['scale']/reference['scale']-1.))


def match_detections(reference, candidate, calibrated=False):
    """Deterministic globally nearest remaining pair, with unique endpoints.

    All pairs within 2 px are considered by distance (then stable indices).
    Angle/scale acceptance is assessed after positional pairing, so a wrong
    rotation cannot disappear merely by leaving an otherwise nearby match out.
    This is a declared greedy correspondence rule, not an optimal assignment.
    """
    possible = []
    for i, first in enumerate(reference):
        for j, second in enumerate(candidate):
            e = geometry(first, second)
            if e['position_px'] <= 2:
                possible.append((e['position_px'], i, j, e))
    used_reference, used_candidate, pairs = set(), set(), []
    for _, i, j, e in sorted(possible, key=lambda p: p[:3]):
        if i in used_reference or j in used_candidate:
            continue
        used_reference.add(i); used_candidate.add(j)
        item = dict(reference_index=i, candidate_index=j, **e,
                    geometry_passed=e['angle_deg'] <= 1 and e['relative_scale'] <= .02)
        if 'score' in reference[i] and 'score' in candidate[j]:
            item.update(reference_score=reference[i]['score'], candidate_score=candidate[j]['score'],
                        score_delta=candidate[j]['score']-reference[i]['score'])
        pairs.append(item)
    missing = sorted(set(range(len(reference)))-used_reference)
    extra = sorted(set(range(len(candidate)))-used_candidate)
    bad_geometry = [p['reference_index'] for p in pairs if not p['geometry_passed']]
    deltas = [p['score_delta'] for p in pairs if 'score_delta' in p]
    return dict(reference_kind='calibrated fixture truth' if calibrated else 'frozen CPU detection; no calibrated ground truth',
                reference_count=len(reference), candidate_count=len(candidate), paired_within_2px=len(pairs),
                missing_reference_indices=missing, extra_candidate_indices=extra,
                geometry_failure_indices=bad_geometry, pairs=pairs,
                score_loss_count=sum(d < 0 for d in deltas),
                largest_score_loss=max([0., *(-d for d in deltas)]),
                passed=not missing and not extra and not bad_geometry)


def records(result):
    return [result['first'], *result['measured']]


def audit(report):
    errors, runs, comparisons = [], [], []
    if not report.get('complete'):
        errors.append('Input benchmark is incomplete')
    for key in ('source_hashes_unchanged', 'binary_hashes_unchanged', 'acceptance_passed'):
        if report.get(key) is not True:
            errors.append('Benchmark did not pass '+key)
    tolerance = report.get('protocol', {}).get('numerical_tolerance', SCORE_TOLERANCE)
    if tolerance != SCORE_TOLERANCE:
        errors.append('Unexpected benchmark NCC tolerance')
    expected_calls = report.get('protocol', {}).get('repeats', 0)+1
    groups = {}
    available = []
    for row in report.get('runs', []):
        job, result = row['job'], row['result']
        case, method = job['case'], job['method']
        key = case['name']+'/'+method
        group = groups.setdefault(case['name'], {})
        if method in group:
            errors.append(key+': duplicate variant')
        group[method] = row
        if result.get('status') == 'unavailable':
            errors.append(key+': unavailable')
            continue
        available.append(row)
        if not result.get('passed'):
            errors.append(key+': benchmark numerical validation failed')
        recs, checked = records(result), result['validation']['calls']
        if len(recs) != expected_calls or len(checked) != len(recs):
            errors.append(key+': first/measured/validation call counts differ')
        initial_scores = result['validation']['initial_scores']
        count = result['validation']['seed_count']
        if count < 1 or len(initial_scores) != count:
            errors.append(key+': empty or inconsistent seed/initial-score count')
        threshold = case.get('score', .8)
        call_reports = []
        for index, (rec, validation) in enumerate(zip(recs, checked)):
            values = rec['result']
            scores, cpu_scores, poses = values['scores'], validation['cpu_rechecked_scores'], values['poses']
            name = 'first' if index == 0 else 'measured_'+str(index-1)
            if not (len(scores) == len(cpu_scores) == len(poses) == len(values['stats']) == count):
                errors.append(key+'/'+name+': score/pose/stat counts differ')
                continue
            if any(len(p) != 4 for p in poses) or any(not math.isfinite(x) for p in poses for x in p):
                errors.append(key+'/'+name+': nonfinite or malformed pose')
                continue
            if any(not math.isfinite(x) for x in [*scores, *cpu_scores, *initial_scores]):
                errors.append(key+'/'+name+': nonfinite score')
                continue
            deltas = [final-initial for final, initial in zip(cpu_scores, initial_scores)]
            raw = [i for i, delta in enumerate(deltas) if delta < 0]
            material = [i for i in raw if deltas[i] < -tolerance]
            threshold_changes = [i for i, (s, c) in enumerate(zip(scores, cpu_scores)) if (s >= threshold) != (c >= threshold)]
            stable_changes = [i for i in threshold_changes if abs(cpu_scores[i]-threshold) > tolerance]
            exact = scores == cpu_scores
            score_error = max(abs(s-c) for s, c in zip(scores, cpu_scores))
            verified = method != 'resident'
            if verified and (not exact or raw):
                errors.append(key+'/'+name+': verified score is not exactly CPU-scored and monotonic')
            if score_error > tolerance or stable_changes:
                errors.append(key+'/'+name+': material score/threshold disagreement')
            item = dict(call=name, pose_count=count, actual_score_threshold=threshold,
                        raw_cpu_score_regression_count=len(raw), raw_cpu_score_regression_indices=raw,
                        beyond_tolerance_cpu_score_regression_count=len(material),
                        maximum_cpu_score_loss=max([0., *(-d for d in deltas)]),
                        threshold_disagreements=len(threshold_changes),
                        threshold_disagreement_indices=threshold_changes,
                        threshold_disagreements_outside_tolerance=len(stable_changes),
                        exact_cpu_scores=exact, maximum_score_error=score_error,
                        strict_verified_monotonic_passed=(exact and not raw) if verified else None)
            if case.get('truth_accuracy'):
                item['local_truth'] = local_truth(poses, cpu_scores, case['truth'])
                if item['local_truth']['failed_indices']:
                    errors.append(key+'/'+name+': local truth precision failed')
            if case['scope'] == 'full':
                matches = values['matches']
                item['detection_count'] = len(matches)
                if any(m['score'] < threshold or not math.isfinite(m['score']) for m in matches):
                    errors.append(key+'/'+name+': final detection below declared score threshold')
                if case.get('truth'):
                    item['full_truth'] = match_detections(case['truth'], matches, calibrated=True)
                    if not item['full_truth']['passed']:
                        errors.append(key+'/'+name+': calibrated full-image truth failed')
            call_reports.append(item)
        runs.append(dict(case=case['name'], method=method, scope=case['scope'],
                         source_input_sha256=result['input_sha256'], calls=call_reports))
    for name, group in groups.items():
        if 'legacy_cpu' not in group or group['legacy_cpu']['result'].get('status') == 'unavailable':
            errors.append(name+': frozen CPU comparison missing')
            continue
        baseline = group['legacy_cpu']['result']
        for method, row in group.items():
            actual = row['result']
            if actual.get('status') == 'unavailable':
                continue
            same_inputs = actual['input_sha256'] == baseline['input_sha256']
            cpu_calls, actual_calls = records(baseline), records(actual)
            same_seeds = len(cpu_calls) == len(actual_calls) and all(
                first['result'].get('initial_poses') == second['result'].get('initial_poses')
                for first, second in zip(cpu_calls, actual_calls))
            if not same_inputs or not same_seeds:
                errors.append(name+'/'+method+': cross-method inputs or coarse seeds differ')
            item = dict(case=name, method=method, identical_input_hashes=same_inputs,
                        identical_coarse_seeds=same_seeds, local_truth_regressions=0, full_truth_regressions=0)
            candidate_audit = next(r for r in runs if r['case'] == name and r['method'] == method)
            cpu_audit = next(r for r in runs if r['case'] == name and r['method'] == 'legacy_cpu')
            for cpu_call, actual_call in zip(cpu_audit['calls'], candidate_audit['calls']):
                if 'local_truth' in cpu_call:
                    item['local_truth_regressions'] += len(set(actual_call['local_truth']['failed_indices'])
                        -set(cpu_call['local_truth']['failed_indices']))
                if 'full_truth' in cpu_call and cpu_call['full_truth']['passed'] and not actual_call['full_truth']['passed']:
                    item['full_truth_regressions'] += 1
            if item['local_truth_regressions'] or item['full_truth_regressions']:
                errors.append(name+'/'+method+': new strict truth failure versus frozen CPU')
            if name == 'src7':
                item['src7_agreement'] = []
                for index, (cpu_call, actual_call) in enumerate(zip(cpu_calls, actual_calls)):
                    agreement = match_detections(cpu_call['result']['matches'], actual_call['result']['matches'])
                    agreement['call'] = 'first' if index == 0 else 'measured_'+str(index-1)
                    item['src7_agreement'].append(agreement)
                    if not agreement['passed']:
                        errors.append(name+'/'+method+'/'+agreement['call']+': CPU detection agreement failed')
            comparisons.append(item)
    selection = report.get('selection_counts', {})
    if (selection.get('expected_runs') != len(report.get('runs', []))
            or selection.get('available_runs') != len(available) or not available):
        errors.append('Selected/completed/available run counts differ or are empty')
    if any(method not in EXPECTED_METHODS for group in groups.values() for method in group):
        errors.append('Unknown method')
    calls = [call for run in runs for call in run['calls']]
    if any(len(run['calls']) != expected_calls for run in runs):
        errors.append('Incomplete per-call audit')
    local_calls = [c['local_truth'] for c in calls if 'local_truth' in c]
    full_calls = [c['full_truth'] for c in calls if 'full_truth' in c]
    src7_calls = [call for c in comparisons for call in c.get('src7_agreement', [])]
    coverage_complete = (bool(available) and len(available) == len(report.get('runs', []))
        and selection.get('expected_runs') == len(available) and len(comparisons) == len(available)
        and all(len(run['calls']) == expected_calls for run in runs))
    numeric_passed = all(c['maximum_score_error'] <= tolerance
        and c['threshold_disagreements_outside_tolerance'] == 0
        and c['strict_verified_monotonic_passed'] is not False for c in calls)
    comparison_passed = all(c['identical_input_hashes'] and c['identical_coarse_seeds']
        and c['local_truth_regressions'] == 0 and c['full_truth_regressions'] == 0 for c in comparisons)
    candidate_compared = any(c['method'] != 'legacy_cpu' for c in comparisons)
    return dict(protocol=dict(saved_data_only=True, score_tolerance=tolerance,
        verified_methods=['legacy_cpu', 'legacy_hybrid', 'resident_verified'],
        gpu_only_monotonicity='Every raw CPU-score loss is reported; GPU-only monotonicity is not a solver guarantee or an acceptance gate.',
        threshold='Use case.score, default .8; disagreements more than 3e-5 from the CPU score threshold are material.',
        local_truth='NCC>=.8, position<=.5px, angle<=.5deg, relative scale<=1%.',
        full_geometry='Position<=2px, angle<=1deg, relative scale<=2%; deterministic globally nearest unique positional pairs.',
        src7='Same first/measured call compared with frozen CPU. This is agreement, NOT calibrated detection accuracy.',
        score_loss='Inter-method matched-detection score losses are reported, not independently gated; score/oracle and actual-threshold gates are checked separately.',
        coverage='Reuses saved CPU rechecks from every timed output; does not rerun NCC or recover information absent from the input report.'),
        gates=dict(original_benchmark_acceptance_passed=report.get('acceptance_passed'),
            complete_comparison_coverage=coverage_complete,
            absolute_local_accuracy_passed=all(not c['failed_indices'] for c in local_calls) if local_calls else None,
            absolute_full_accuracy_passed=all(c['passed'] for c in full_calls) if full_calls else None,
            verified_monotonic_and_exact_passed=all(c['strict_verified_monotonic_passed'] is not False for c in calls),
            material_score_and_threshold_agreement_passed=numeric_passed,
            src7_cpu_agreement_passed=all(c['passed'] for c in src7_calls) if src7_calls else None,
            observed_nonregression_passed=(coverage_complete and numeric_passed and comparison_passed
                and all(c['passed'] for c in src7_calls)) if candidate_compared else None),
        selection_counts=dict(cases=len(groups), runs=len(runs), calls=len(calls),
            local_truth_calls=sum('local_truth' in c for c in calls),
            local_truth_pose_checks=sum(c['local_truth']['checked'] for c in calls if 'local_truth' in c),
            full_truth_calls=sum('full_truth' in c for c in calls),
            full_truth_object_checks=sum(c['full_truth']['reference_count'] for c in calls if 'full_truth' in c),
            src7_comparison_calls=sum(len(c.get('src7_agreement', [])) for c in comparisons)),
        summary=dict(raw_cpu_score_regressions=sum(c['raw_cpu_score_regression_count'] for c in calls),
            maximum_cpu_score_loss=max((c['maximum_cpu_score_loss'] for c in calls), default=None),
            beyond_tolerance_cpu_score_regressions=sum(c['beyond_tolerance_cpu_score_regression_count'] for c in calls),
            threshold_disagreements=sum(c['threshold_disagreements'] for c in calls),
            threshold_disagreements_outside_tolerance=sum(c['threshold_disagreements_outside_tolerance'] for c in calls)),
        runs=runs, comparisons=comparisons, errors=errors, passed=not errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Saved .json or .json.gz benchmark report')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.output.resolve():
        parser.error('Supplemental output must not overwrite the input report')
    stored_contents = args.report.read_bytes()
    contents = None
    compressed = args.report.suffix == '.gz'
    try:
        contents = gzip.decompress(stored_contents) if compressed else stored_contents
        result = audit(json.loads(contents))
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, StopIteration, OSError, EOFError) as error:
        result = dict(passed=False, errors=['Malformed/incomplete saved report: '+str(error)])
    result.update(input_report=str(args.report.resolve()), input_report_sha256=hashlib.sha256(stored_contents).hexdigest(),
                  input_report_encoding='gzip' if compressed else 'identity',
                  input_report_decompressed_sha256=hashlib.sha256(contents).hexdigest() if contents is not None else None,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  created_at=datetime.now().astimezone().isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('passed', 'selection_counts', 'summary', 'errors') if k in result}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
