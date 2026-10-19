"""NCC rescue dispatch follow-up: actual detection threshold and paired timing.

The original formal report, benchmark, protocol and binaries remain unchanged.
Full-image jobs only lower the caller's rescue trigger to its actual detection
threshold. Paired diagnostics reuse saved local fixtures and compare old/new
baseline and rescue contexts within one process, with all checks untimed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmark_ncc_rescue import (DEFAULT_RESCUE, METHODS, checked_outputs, finish_report,
    input_digest, invoke, verify_baseline_provenance, worker as full_worker)
from benchmark_resident_refinement import freeze_baseline, sha
from benchmark_pose_stress import ROOT, metadata, peak_rss, summary
from benchmark_hybrid_refinement import CONFIG
from pose_experiment import Scorer, library_path
from rescue_policy import rescue_policy_for_threshold


def verify_formal(formal, build):
    """Check the authoritative frozen binary provenance before any rerun."""
    if not formal.get('complete') or not formal.get('source_hashes_unchanged') or not formal.get('binary_hashes_unchanged'):
        raise ValueError('Formal report must be complete and have valid original provenance')
    if not formal.get('baseline_verification', {}).get('verified'):
        raise ValueError('Formal report lacks a verified baseline')
    candidate = [library_path(build), Path(build)/('libfpm_c.dylib' if sys.platform == 'darwin' else 'libfpm_c.so')]
    checked = {}
    for path in candidate:
        actual = sha(path)
        expected = formal['binary_hashes'].get(str(path.resolve()))
        if expected is None or expected != actual:
            raise ValueError('Candidate binary no longer matches formal report: '+str(path))
        checked[str(path.resolve())] = actual
    return checked


def paired_worker(job):
    case = job['case']
    source, template, seeds = [np.load(Path(job['fixture'])/(name+'.npy'), allow_pickle=False)
                               for name in ('source', 'template', 'seeds')]
    inputs = {name: input_digest(array) for name, array in zip(('source', 'template', 'seeds'), (source, template, seeds))}
    if inputs != job['expected_input_sha256']:
        raise ValueError('Paired fixture does not match formal input hashes')
    config = dict(CONFIG, **case.get('config', {}))
    policy = dict(DEFAULT_RESCUE, **job['rescue_config'])
    backend = job['backend']
    baseline_method = 'old_cpu' if backend == 'cpu' else 'old_resident_verified'
    rescue_method = 'rescue_cpu' if backend == 'cpu' else 'rescue_resident_verified'
    before = peak_rss()
    scorers = {}
    try:
        for variant in ('old_baseline', 'new_baseline', 'new_rescue'):
            scorers[variant] = Scorer(job['reference_build'] if variant == 'old_baseline' else job['build'],
                source, template, backend, job['threads'])
        initial_memory = {name: scorer.memory() for name, scorer in scorers.items()}

        def call(variant):
            return invoke(scorers[variant], seeds, rescue_method if variant == 'new_rescue' else baseline_method, config, policy)

        def one(variant, **annotation):
            start = time.perf_counter()
            poses, scores, stats, metrics = call(variant)
            elapsed = (time.perf_counter()-start)*1000
            return dict(variant=variant, wall_ms=elapsed, **annotation,
                result=dict(poses=poses.tolist(), scores=scores.tolist(), stats=stats, metrics=metrics))

        first = [one(name, stage='first') for name in scorers]
        warmups = {}
        for name in scorers:
            start, count = time.perf_counter(), 0
            while count < 1 or (time.perf_counter()-start)*1000 < job['warmup_ms']:
                call(name)
                count += 1
            warmups[name] = count
        measured = []
        for round_index in range(job['rounds']):
            pairs = ('old_baseline', 'new_baseline') if round_index%2 == 0 else ('new_baseline', 'old_baseline')
            for baseline in pairs:
                sequence = (baseline, 'new_rescue', 'new_rescue', baseline)
                if round_index%2:
                    sequence = ('new_rescue', baseline, baseline, 'new_rescue')
                for order, variant in enumerate(sequence):
                    measured.append(one(variant, stage='measured', round=round_index, pair=baseline, order=order))
        measured_peak = peak_rss()
        retained = {name: scorer.memory() for name, scorer in scorers.items()}
        checks = {}
        for name in scorers:
            rows = [row for row in first+measured if row['variant'] == name]
            checks[name] = checked_outputs(source, template, seeds, rows, job['reference_build'], config, case, policy)
        all_rescue = [row for row in first+measured if row['variant'] == 'new_rescue']
        no_retries = all(row['result']['metrics']['rescue']['selected_seeds'] == 0 for row in all_rescue)
        variants = {}
        for name in scorers:
            rows = [row for row in measured if row['variant'] == name]
            variants[name] = dict(outer_wall=summary([row['wall_ms'] for row in rows]),
                c_api_wall=summary([row['result']['metrics']['wall_ms'] for row in rows]))
            if name == 'new_rescue':
                diagnostics = [row['result']['metrics']['rescue'] for row in rows]
                for key in ('baseline_ms', 'retry_ms', 'cpu_verify_ms'):
                    variants[name][key] = summary([value[key] for value in diagnostics])
                variants[name]['outside_nested_baseline_ms'] = summary([
                    row['result']['metrics']['wall_ms']-row['result']['metrics']['rescue']['baseline_ms'] for row in rows])
        pair_results = []
        for baseline in ('old_baseline', 'new_baseline'):
            rounds = []
            for index in range(job['rounds']):
                rows = [row for row in measured if row['round'] == index and row['pair'] == baseline]
                base = [row for row in rows if row['variant'] == baseline]
                rescue = [row for row in rows if row['variant'] == 'new_rescue']
                old_ms, new_ms = np.mean([row['wall_ms'] for row in base]), np.mean([row['wall_ms'] for row in rescue])
                rounds.append(dict(round=index, baseline_mean_ms=float(old_ms), rescue_mean_ms=float(new_ms),
                    difference_ms=float(new_ms-old_ms), ratio=float(new_ms/old_ms)))
            old_median = float(np.median([row['baseline_mean_ms'] for row in rounds]))
            difference = float(np.median([row['difference_ms'] for row in rounds]))
            limit = max(.5, old_median*.1)
            pair_results.append(dict(baseline=baseline, rounds=rounds, baseline_median_ms=old_median,
                paired_median_difference_ms=difference, paired_median_ratio=float(np.median([row['ratio'] for row in rounds])),
                overhead_limit_ms=limit, overhead_passed=difference <= limit))
        # No-retry output must agree exactly with the same binary's baseline.
        # Saved old/new geometry is also checked independently, not merely timed.
        reference = first[1]['result']
        same_new_baseline = all(row['result']['poses'] == reference['poses'] and row['result']['scores'] == reference['scores']
            for row in first+measured if row['variant'] in ('new_baseline', 'new_rescue'))
        truth_passed = all(point['precise_pass'] for value in checks.values() for check in value['calls']
            for point in check['accuracy']) if case.get('truth_accuracy') else None
        old_scores = np.asarray(checks['old_baseline']['calls'][0]['cpu_rechecked_scores'])
        frozen_score_nonregression = all(np.all(np.asarray(check['cpu_rechecked_scores']) >= old_scores)
            for name, value in checks.items() if name != 'old_baseline' for check in value['calls'])
        return dict(first=first, measured=measured, warmups=warmups, variants=variants, paired=pair_results,
            validation=checks, no_retries=no_retries, identical_new_baseline_outputs=same_new_baseline,
            truth_accuracy_passed=truth_passed, frozen_score_nonregression=bool(frozen_score_nonregression),
            input_sha256=inputs, memory=dict(process_peak_before_setup_bytes=before, process_peak_bytes=measured_peak,
                initial_owned_buffers=initial_memory, retained_owned_buffers=retained,
                note='Three contexts alive together; this RSS must not be compared to a one-context formal worker.'),
            correctness_passed=no_retries and same_new_baseline and truth_passed is not False and
                bool(frozen_score_nonregression) and all(value['passed'] for value in checks.values()),
            overhead_passed=all(value['overhead_passed'] for value in pair_results))
    finally:
        for scorer in scorers.values():
            scorer.close()


def run_jobs(jobs, assets, args, report):
    for index, job in enumerate(jobs):
        suffix = job.get('method', job.get('backend', ''))
        prefix = assets/(job['case']['name']+'-'+suffix)
        request, response = Path(str(prefix)+'.request.json'), Path(str(prefix)+'.response.json')
        request.write_text(json.dumps(job))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request), '--output', str(response)],
            check=True, timeout=args.timeout)
        report['runs'].append(json.loads(response.read_text()))
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(f"{index+1}/{len(jobs)} {job['kind']} {job['case']['name']} {suffix}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', type=Path, default=ROOT/'build/ncc-rescue/formal.json')
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--suite', choices=('full', 'paired', 'all'), default='full')
    parser.add_argument('--output', type=Path, default=ROOT/'build/ncc-rescue/dispatch.json')
    parser.add_argument('--case', action='append')
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--warmup-ms', type=float, default=1000.)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        result = full_worker(job) if job['kind'] == 'full' else paired_worker(job)
        args.output.write_text(json.dumps(dict(job=job, result=result), indent=2)+'\n')
        return
    if args.repeats < 1 or args.repeats%2 != 1 or args.rounds < 1 or args.rounds%2 != 1 or args.warmup_ms < 0:
        parser.error('Use positive odd repeats/rounds and warmup-ms>=0')
    formal = json.loads(args.formal.read_text())
    if args.output.resolve() == args.formal.resolve():
        parser.error('The formal report must not be overwritten')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_hashes = verify_formal(formal, args.build)
    assets = args.output.parent/(args.output.stem+'-artifacts-'+str(time.time_ns()))
    assets.mkdir()
    baseline_origin = Path(next(row['job']['build'] for row in formal['runs'] if row['job']['method'] == 'old_cpu'))
    frozen = assets/'baseline-frozen'
    old_hashes = freeze_baseline(baseline_origin, frozen)
    verification = verify_baseline_provenance(formal['baseline_provenance'], old_hashes)
    binary_hashes.update(old_hashes)
    info = metadata(args)
    for path in (Path(__file__).resolve(), ROOT/'cpp-core/tools/benchmark_ncc_rescue.py',
                 ROOT/'cpp-core/tools/benchmark_resident_refinement.py', ROOT/'cpp-core/tools/audit_resident_refinement.py',
                 ROOT/'cpp-core/tools/real_image_experiments.py', ROOT/'cpp-core/tools/benchmark_hybrid_refinement.py',
                 ROOT/'cpp-core/tools/rescue_policy.py',
                 ROOT/'cpp-core/docs/ncc-rescue/protocol-v2.json'):
        info['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    report = dict(**info, suite=args.suite, formal_report=str(args.formal.resolve()), formal_report_sha256=sha(args.formal),
        baseline_provenance=formal['baseline_provenance'], baseline_verification=verification, binary_hashes=binary_hashes,
        protocol=json.loads((ROOT/'cpp-core/docs/ncc-rescue/protocol-v2.json').read_text()), runs=[])
    jobs = []
    if args.suite in ('all', 'full'):
        for row in formal['runs']:
            job = row['job']
            if job['case']['scope'] != 'full':
                continue
            config = rescue_policy_for_threshold(job['case']['score'], job['rescue_config'])
            jobs.append(dict(job, kind='full', build=str(frozen if job['method'].startswith('old_') else args.build.resolve()),
                reference_build=str(frozen), rescue_config=config, repeats=args.repeats, warmup_ms=args.warmup_ms))
    if args.suite in ('all', 'paired'):
        failed = {(row['case'], row['method']) for row in formal['comparisons'] if row.get('easy_case_overhead_passed') is False and row['no_retries']}
        for row in formal['runs']:
            job = row['job']
            if (job['case']['name'], job['method']) not in failed:
                continue
            jobs.append(dict(job, kind='paired', backend='cpu' if job['method'].endswith('_cpu') else 'metal',
                build=str(args.build.resolve()), reference_build=str(frozen), rounds=args.rounds,
                warmup_ms=args.warmup_ms, expected_input_sha256=row['result']['input_sha256']))
    jobs = [job for job in jobs if not args.case or any(term in job['case']['name'] for term in args.case)]
    if not jobs:
        parser.error('No selected jobs')
    np.random.default_rng(74913).shuffle(jobs)
    report['expected_runs'] = len(jobs)
    run_jobs(jobs, assets, args, report)
    full_runs = [row for row in report['runs'] if row['job']['kind'] == 'full']
    paired_runs = [row for row in report['runs'] if row['job']['kind'] == 'paired']
    if full_runs:
        full = dict(report, runs=full_runs, expected_runs=len(full_runs))
        finish_report(full)
        report['full'] = {key: value for key, value in full.items() if key not in report or key in ('comparisons',)}
    report['paired_summary'] = [dict(case=row['job']['case']['name'], backend=row['job']['backend'],
        paired=row['result']['paired'], variants=row['result']['variants'], correctness_passed=row['result']['correctness_passed'],
        overhead_passed=row['result']['overhead_passed']) for row in paired_runs]
    report['source_hashes_unchanged'] = all(sha(ROOT/path) == value for path, value in info['source_sha256'].items())
    report['binary_hashes_unchanged'] = all(sha(path) == value for path, value in binary_hashes.items())
    report['formal_report_unchanged'] = sha(args.formal) == report['formal_report_sha256']
    report['complete'] = len(report['runs']) == len(jobs)
    report['correctness_passed'] = (not full_runs or report['full']['accuracy_acceptance_passed']) and all(
        row['result']['correctness_passed'] for row in paired_runs)
    report['overhead_passed'] = (not full_runs or report['full']['full_pipeline_overhead_passed']) and all(
        row['result']['overhead_passed'] for row in paired_runs)
    report['acceptance_passed'] = all(report[key] for key in ('complete', 'source_hashes_unchanged', 'binary_hashes_unchanged',
        'formal_report_unchanged', 'correctness_passed', 'overhead_passed'))
    report['completed_at'] = datetime.now().astimezone().isoformat()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['acceptance_passed']:
        raise SystemExit('Dispatch follow-up failed an accuracy/provenance/overhead gate; original and new evidence retained')


if __name__ == '__main__':
    main()
