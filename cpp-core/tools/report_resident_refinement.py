"""Summarize saved resident-refinement benchmarks without executing workloads.

All selected cases must contain all four methods. Missing/unavailable methods,
unfinished reports and unverified provenance cannot pass. An unexercised truth
accuracy gate is null, never true. Artifacts are written even when gates fail.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


METHODS = ('legacy_cpu', 'legacy_hybrid', 'resident', 'resident_verified')
LABELS = ('CPU', 'CPU + Metal', 'Resident GPU', 'Resident + CPU verification')
COLORS = ('#737d8c', '#3978a8', '#de9344', '#369878')
ERRORS = ('position_px', 'angle_deg', 'relative_scale')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def maximum(values):
    values = [v for v in values if number(v)]
    return max(values) if values else None


def timing(result, key):
    value = result.get(key)
    return value.get('median_ms') if isinstance(value, dict) and number(value.get('median_ms')) else None


def truth_summary(result, scope):
    calls = result.get('validation', {}).get('calls', [])
    truth_calls = result.get('accuracy') if scope == 'full' else [r.get('accuracy') for r in calls]
    truth_calls = [r for r in (truth_calls or []) if r is not None]
    if not truth_calls:
        return dict(exercised=False, passed=None, pass_count=None, check_count=0, max_errors=None)
    rows = [r for call in truth_calls for r in (call.get('details', []) if scope == 'full' else call)]
    flag = 'pose_pass' if scope == 'full' else 'precise_pass'
    errors = [r.get('errors') or {} for r in rows] if scope == 'full' else rows
    passed = bool(rows) and all(r.get(flag) is True for r in rows)
    limits = (2., 1., .02) if scope == 'full' else (.5, .5, .01)
    passed = passed and all(all(number(r.get(key)) and 0 <= r[key] <= limit
                               for key, limit in zip(ERRORS, limits)) for r in errors)
    if scope == 'full':
        passed = passed and all(r.get('pose_fn') == 0 and r.get('pose_fp') == 0 for r in truth_calls)
    return dict(exercised=True, passed=passed, pass_count=sum(r.get(flag) is True for r in rows),
                check_count=len(rows), calls=len(truth_calls),
                max_errors={key: maximum(r.get(key) for r in errors) for key in ERRORS},
                missing_error_rows=sum(not all(number(r.get(k)) for k in ERRORS) for r in errors),
                max_false_positives=maximum(r.get('pose_fp') for r in truth_calls) if scope == 'full' else None)


def method_table(row):
    result, job = row.get('result', {}), row.get('job', {})
    if result.get('status') == 'unavailable':
        return dict(available=False, reason=result.get('reason'), correctness_passed=False)
    measured = result.get('measured', [])
    representative = result.get('representative_run')
    valid_index = isinstance(representative, int) and 0 <= representative < len(measured)
    record = measured[representative] if valid_index else {}
    metrics = record.get('result', {}).get('metrics', {})
    memory = result.get('memory', {})
    buffers = memory.get('max_observed_owned_buffers', {})
    validation = result.get('validation', {})
    calls = validation.get('calls', [])
    required = ('passed', 'bounds_passed', 'frozen_passed', 'invalid_passed', 'finite', 'stats_passed')
    correct = result.get('passed') is True and validation.get('passed') is True and bool(calls)
    correct = correct and all(all(call.get(key) is True for key in required) for call in calls)
    resident = metrics.get('resident')
    if job.get('method', '').startswith('resident'):
        diagnostics = result.get('resident_diagnostics') or {}
        correct = correct and all(diagnostics.get(key) is True for key in (
            'present', 'commands_equal_waits', 'verify_flag_matches',
            'pure_gpu_has_no_cpu_verification', 'auxiliary_budget_passed'))
        correct = correct and isinstance(resident, dict)
    return dict(available=True, device=result.get('device'), correctness_passed=correct,
                warm_median_ms=dict(outer=timing(result, 'algorithm_wall'), api=timing(result, 'c_api_wall'),
                                    gpu=timing(result, 'gpu_command')),
                measured_calls=len(measured), checked_calls=len(calls), warmups=result.get('warmups'),
                constructor_wall_ms=result.get('constructor_wall_ms'), first_outer_ms=result.get('first', {}).get('wall_ms'),
                retained_host_bytes=buffers.get('host_buffer_bytes'), retained_gpu_bytes=buffers.get('gpu_buffer_bytes'),
                retained_auxiliary_bytes=(result.get('resident_diagnostics') or {}).get('retained_auxiliary_bytes'),
                process_peak_rss_bytes=memory.get('process_peak_bytes'),
                strict_accuracy=truth_summary(result, job.get('case', {}).get('scope')),
                max_score_error=maximum([validation.get('oracle_max_error'), *[r.get('score_max_error') for r in calls]]),
                score_regression_count=sum(r.get('score_regression_count', 0) for r in calls),
                representative_run=representative if valid_index else None,
                representative=dict(outer_ms=record.get('wall_ms'), api_ms=metrics.get('wall_ms'),
                                    gpu_ms=metrics.get('device_ms'), resident=resident))


def hash_map(value):
    return isinstance(value, dict) and bool(value) and all(isinstance(v, str) and len(v) == 64 and
        all(c in '0123456789abcdef' for c in v.lower()) for v in value.values())


def summarize(path, report):
    issues, cases = [], []
    runs = report.get('runs', [])
    groups = {}
    for row in runs:
        case = row.get('job', {}).get('case', {})
        name = case.get('name')
        if not isinstance(name, str) or case.get('scope') not in ('local', 'full'):
            issues.append('Malformed run case metadata')
            continue
        groups.setdefault(name, []).append(row)
    counts = report.get('selection_counts', {})
    complete = report.get('complete') is True and bool(groups)
    complete = complete and all(counts.get(k) == v for k, v in (
        ('cases', len(groups)), ('methods', len(METHODS)), ('expected_runs', len(groups)*len(METHODS)),
        ('completed_runs', len(runs)), ('available_runs', len(runs)), ('unavailable_runs', 0)))
    comparisons = report.get('comparisons', [])
    complete = complete and counts.get('compared_runs') == len(comparisons)
    compared = Counter((c.get('case'), c.get('method')) for c in comparisons)
    cross = report.get('cross_method_passed') is True and len(comparisons) == len(runs)
    for c in comparisons:
        cross = cross and c.get('identical_inputs') is True and c.get('identical_coarse_seeds') is True
        cross = cross and all(c.get(k, 0) == 0 for k in (
            'precise_accuracy_regressions', 'full_accuracy_regressions', 'full_false_positive_increases'))
    for name, rows in groups.items():
        methods = Counter(r.get('job', {}).get('method') for r in rows)
        if methods != Counter(METHODS):
            complete = False
            issues.append(f'{name}: expected one result for every method, got {dict(methods)}')
        scope = rows[0]['job']['case']['scope']
        case = dict(report=str(path), case=name, scope=scope, methods={})
        needs_truth = bool(rows[0]['job']['case'].get('truth') if scope == 'full'
                           else rows[0]['job']['case'].get('truth_accuracy'))
        case['truth_required'] = needs_truth
        for method in METHODS:
            selected = [r for r in rows if r.get('job', {}).get('method') == method]
            table = method_table(selected[0]) if len(selected) == 1 else dict(
                available=False, reason='Missing or duplicate method', correctness_passed=False)
            case['methods'][method] = table
            if not table['available']:
                complete = False
                issues.append(f'{name}/{method}: {table.get("reason", "unavailable")}')
                continue
            expected = selected[0].get('job', {}).get('repeats')
            valid_samples = isinstance(expected, int) and expected >= 1
            valid_samples = valid_samples and table['measured_calls'] == expected and table['checked_calls'] == expected+1
            valid_samples = valid_samples and table['representative_run'] is not None
            valid_samples = valid_samples and all(number(table['warm_median_ms'][k]) for k in ('outer', 'api'))
            valid_samples = valid_samples and all(number(table[k]) and table[k] >= 0 for k in (
                'retained_host_bytes', 'retained_gpu_bytes', 'process_peak_rss_bytes'))
            if needs_truth:
                accuracy = table['strict_accuracy']
                valid_samples = valid_samples and accuracy['exercised'] and accuracy.get('calls') == expected+1
                truth_count = len(rows[0]['job']['case']['truth']) if scope == 'full' else selected[0].get('result', {}).get('validation', {}).get('seed_count')
                valid_samples = valid_samples and isinstance(truth_count, int) and truth_count > 0
                valid_samples = valid_samples and accuracy['check_count'] == truth_count*(expected+1)
                comparison = next((c for c in comparisons if (c.get('case'), c.get('method')) == (name, method)), {})
                needed = ('full_accuracy_regressions', 'full_false_positive_increases') if scope == 'full' else ('precise_accuracy_regressions',)
                cross = cross and all(k in comparison for k in needed)
            if not valid_samples:
                complete = False
                issues.append(f'{name}/{method}: missing/inconsistent measured calls, truth checks, memory or timings')
            cross = cross and compared[(name, method)] == 1
        cases.append(case)
    correct = complete and report.get('correctness_passed') is True and bool(cases) and cross
    correct = correct and all(t['correctness_passed'] for c in cases for t in c['methods'].values())
    provenance = all(report.get(k) is True for k in ('source_hashes_unchanged', 'binary_hashes_unchanged'))
    provenance = provenance and hash_map(report.get('source_sha256')) and hash_map(report.get('binary_hashes'))
    provenance = provenance and bool(report.get('git_head')) and bool(report.get('baseline_origin'))
    baseline = report.get('baseline_provenance')
    if baseline is not None:
        baseline_hashes = baseline.get('sha256', {}) if isinstance(baseline, dict) else {}
        frozen_hashes = {Path(p).name: h for p, h in report.get('binary_hashes', {}).items() if 'baseline-frozen' in Path(p).parts}
        provenance = provenance and hash_map(baseline_hashes) and all(
            frozen_hashes.get(Path(p).name) == h for p, h in baseline_hashes.items())
    accuracy = {}
    for scope in ('local', 'full'):
        exercised = [t['strict_accuracy'] for c in cases if c['scope'] == scope
                     for t in c['methods'].values() if t['available'] and t['strict_accuracy']['exercised']]
        accuracy[scope] = all(t['passed'] for t in exercised) if exercised else None
        if exercised and report.get(scope+'_accuracy_passed') is not True:
            accuracy[scope] = False
        if exercised and any(not t['available'] for c in cases if c['scope'] == scope for t in c['methods'].values()):
            accuracy[scope] = False
        required = [c for c in cases if c['scope'] == scope and c['truth_required']]
        if required and not all(t['available'] and t['strict_accuracy']['exercised']
                                for c in required for t in c['methods'].values()):
            accuracy[scope] = False
    if not complete:
        issues.append('Report is unfinished, unavailable, or does not cover all four methods for each selected case')
    if not correct:
        issues.append('Correctness or cross-method comparison gate failed/incomplete')
    if not provenance:
        issues.append('Source/binary provenance gate failed/incomplete')
    if any(v is False for v in accuracy.values()):
        issues.append('Strict truth accuracy gate failed')
    gates = dict(complete=complete, correctness=bool(correct), provenance=bool(provenance), accuracy=accuracy)
    return cases, dict(report=str(path), gates=gates, passed=bool(complete and correct and provenance and
        all(v is not False for v in accuracy.values())), issues=issues,
        baseline_source_manifest_present=baseline is not None,
        provenance_basis='Saved run-time hash checks; files are not rehashed against the possibly changed checkout.')


def selected_cases(cases, scope):
    chosen = [c for c in cases if c['scope'] == scope]
    if scope == 'local':
        performance = [c for c in chosen if c['case'].startswith('large_')]
        chosen = performance or chosen
    return chosen


def plot(cases, destination, scope):
    chosen = selected_cases(cases, scope)
    if not chosen:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = Counter(c['case'] for c in chosen)
    labels = [c['case'] if names[c['case']] == 1 else Path(c['report']).stem+'/'+c['case'] for c in chosen]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, len(chosen)*.68+1.7)), sharey=True)
    for ax, metric, title in zip(axes, ('outer', 'api', 'gpu'), ('Outer call', 'Refinement C API', 'GPU command interval')):
        for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            values = [c['methods'][method].get('warm_median_ms', {}).get(metric) for c in chosen]
            values = [v if number(v) and v >= 0 else float('nan') for v in values]
            ax.barh([i+(index-1.5)*.18 for i in range(len(chosen))], values, height=.16, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel('Warm median milliseconds')
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(chosen)), labels, fontsize=8)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=4, fontsize=8)
    fig.suptitle(('Large supplied-seed refinement' if scope == 'local' and all(c['case'].startswith('large_') for c in chosen)
                 else 'Supplied-seed local refinement' if scope == 'local' else 'Full-image search')+
                 ' — independent cases, no inferred crossover curve', fontsize=12)
    fig.text(.5, .01, 'Panels overlap in time; do not add them. Missing/unavailable GPU timing is omitted.\n'
             'Timing comparison does not establish accuracy; see validation.json for strict accuracy gates.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .86))
    path = destination/(scope+'-timings.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_memory(cases, destination):
    chosen = selected_cases(cases, 'local')+selected_cases(cases, 'full')
    if not chosen:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = Counter(c['case'] for c in chosen)
    labels = [c['case'] if names[c['case']] == 1 else Path(c['report']).stem+'/'+c['case'] for c in chosen]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, len(chosen)*.68+1.7)), sharey=True)
    columns = ('retained_host_bytes', 'retained_gpu_bytes', 'process_peak_rss_bytes')
    titles = ('Retained owned host buffers', 'Retained owned GPU buffers', 'Process peak RSS')
    for ax, metric, title in zip(axes, columns, titles):
        for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            values = [c['methods'][method].get(metric) for c in chosen]
            values = [v/2**20 if number(v) and v >= 0 else float('nan') for v in values]
            ax.barh([i+(index-1.5)*.18 for i in range(len(chosen))], values, height=.16, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel('MiB')
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(chosen)), labels, fontsize=8)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=4, fontsize=8)
    fig.suptitle('Memory observations — local performance and full-image cases', fontsize=12)
    fig.text(.5, .01, 'Separate measures, not additive: shared GPU buffers can overlap RSS. Owned-buffer snapshots exclude transient allocation peaks.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .86))
    path = destination/'memory.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, reports, hashes = [], [], {}
    def reject_constant(value):
        raise ValueError('Non-finite JSON number: '+value)
    for path in args.reports:
        path = path.resolve()
        try:
            if str(path) in hashes:
                raise ValueError('Duplicate input report')
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            report = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw, parse_constant=reject_constant)
            if not isinstance(report, dict):
                raise ValueError('Report root must be an object')
            rows, checked = summarize(path, report)
            cases.extend(rows)
            reports.append(checked)
        except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError) as error:
            reports.append(dict(report=str(path), passed=False, gates=dict(complete=False, correctness=False,
                provenance=False, accuracy=dict(local=None, full=None)), issues=[str(error)]))
    aggregate = {k: bool(reports) and all(r['gates'][k] for r in reports) for k in ('complete', 'correctness', 'provenance')}
    aggregate['accuracy'] = {}
    for scope in ('local', 'full'):
        exercised = [r['gates']['accuracy'][scope] for r in reports if r['gates']['accuracy'][scope] is not None]
        aggregate['accuracy'][scope] = all(exercised) if exercised else None
    validation = dict(created_at=datetime.now(timezone.utc).isoformat(), report_sha256=hashes,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), gates=aggregate,
        passed=bool(reports) and all(r['passed'] for r in reports), reports=reports,
        notes=['Only selected cases are covered; an absent suite is not claimed tested.',
               'Retained owned-buffer snapshots are not transient allocation peaks.',
               'Shared GPU buffers can overlap process RSS; do not add them.',
               'GPU command intervals overlap wait/API/outer wall and include device memory stalls.',
               'Main local charts select large_* cases when present; table.json retains every case.',
               'Resident representative fields belong to the median outer-wall run; medians are not additive.'])
    (args.output/'table.json').write_text(json.dumps(dict(cases=cases), indent=2, allow_nan=False)+'\n')
    (args.output/'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False)+'\n')
    for scope in ('local', 'full'):
        if path := plot(cases, args.output, scope):
            print(args.output/path)
    if path := plot_memory(cases, args.output):
        print(args.output/path)
    print('Validation: '+('PASS' if validation['passed'] else 'FAIL — inspect validation.json'))
    raise SystemExit(0 if validation['passed'] else 1)


if __name__ == '__main__':
    main()
