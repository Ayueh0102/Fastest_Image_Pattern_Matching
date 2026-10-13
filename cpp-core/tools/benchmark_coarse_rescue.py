"""Compare frozen CPU / exact Metal coarse search with identical GPU rescue.

Full source detection runs online for every measurement. CPU and Metal coarse
variants use the same new resident-rescue scorer, thresholds, scale bank and
NMS. Truth is consulted only after timings and process peak RSS are captured.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

import real_image_experiments as real
from benchmark_coarse_ncc import cases_for, file_hash, core_library_path, baseline_provenance
from benchmark_ncc_rescue import DEFAULT_RESCUE
from benchmark_pose_experiments import peak_rss
from coarse_experiment import CoarsePattern, library_path as coarse_library_path
from pose_experiment import Scorer, library_path as pose_library_path


def worker(job):
    cv2.setNumThreads(1)
    saved_library, saved_scorer = real.Library, real.Scorer
    policy = dict(DEFAULT_RESCUE, trigger_score=min(DEFAULT_RESCUE['trigger_score'], job['case']['score']))
    if job.get('effective_rescue_policy') != policy:
        raise ValueError('Worker rescue policy differs from the recorded per-case detection threshold policy')
    class RescueScorer(Scorer):
        def refine_resident(self, seeds, cpu_verify=True, **config):
            return self.refine_rescue(seeds, resident=True, **policy, **config)

    class SearchLibrary:
        def __init__(self, _):
            self.library = saved_library(str(core_library_path(job['baseline'])))

        def config(self):
            return self.library.config()

        def learn(self, image, min_area=256):
            return (CoarsePattern(job['build'], image, min_area) if job['method'] == 'metal'
                    else self.library.learn(image, min_area))

    real.Library, real.Scorer = SearchLibrary, RescueScorer
    start = time.perf_counter()
    try:
        search = real.Search(dict(job, method='resident_verified_direct_metal', audit_refinement=True,
                                  prune=False))
    finally:
        real.Library, real.Scorer = saved_library, saved_scorer
    setup_ms = (time.perf_counter()-start)*1000
    try:
        def run():
            value = search.run()
            if search.last_refinement is None:
                raise RuntimeError('scene yielded no refinement candidates')
            initial, poses, scores, stats, metrics = search.last_refinement
            return dict(value, initial_poses=initial.tolist(), refined_poses=poses.tolist(),
                        refined_scores=scores.tolist(), refine_metrics=metrics,
                        coarse_metrics=[pattern.last_metrics for _, pattern in search.patterns]
                        if job['method'] == 'metal' else [])
        first = run()
        warm_start, warmups = time.perf_counter(), 0
        while time.perf_counter()-warm_start < job['warmup_seconds']:
            search.run(); warmups += 1
        runs = [run() for _ in range(job['repeats'])]
        rss = peak_rss()
        accuracy = ([real.truth_accuracy(r['matches'], job['case']['truth']) for r in [first, *runs]]
                    if job['case'].get('truth') else None)
        return dict(method=job['method'], effective_rescue_policy=policy,
                    setup_ms=setup_ms, first=first, runs=runs, warmups=warmups,
                    median_ms=statistics.median(r['algorithm_ms'] for r in runs),
                    median_coarse_ms=statistics.median(r['cpu_search_ms'] for r in runs),
                    median_refine_ms=statistics.median(r['refine_ms'] for r in runs),
                    process_peak_rss_bytes=rss, pose_buffers=search.scorer.memory(),
                    truth_accuracy=accuracy)
    finally:
        search.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(real.ROOT/'build/fourier-gpu'))
    parser.add_argument('--baseline', default=str(real.ROOT/'build/rescue-baseline'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--output', default=str(real.ROOT/'build/coarse-ncc/full-rescue.json'))
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(Path(args.worker).read_text())))); return
    if args.repeats < 1 or args.warmup_seconds < 0:
        parser.error('repeats must be positive and warmup nonnegative')
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    baseline = baseline_provenance(args.baseline)
    folder = output.parent/'full-inputs'; folder.mkdir(exist_ok=True)
    args.suite = 'real'
    library_paths = [core_library_path(args.baseline), coarse_library_path(args.build), pose_library_path(args.build)]
    hashes = {str(p): file_hash(p) for p in library_paths}
    source_paths = [Path(__file__), Path(__file__).with_name('coarse_experiment.py'),
                    Path(__file__).with_name('real_image_experiments.py'), Path(__file__).with_name('pose_experiment.py'),
                    Path(__file__).with_name('benchmark_ncc_rescue.py')]
    source_hashes = {str(p): file_hash(p) for p in source_paths}
    rows, input_hashes = [], {}
    for original in cases_for(args, folder):
        case = dict(original, proposal_score=original['score'], score=.5 if original['name'] == 'src7' else .8)
        policy = dict(DEFAULT_RESCUE, trigger_score=min(DEFAULT_RESCUE['trigger_score'], case['score']))
        for name in ('source', 'template'):
            input_hashes[case[name]] = file_hash(case[name])
        methods = {}
        for method in ('cpu', 'metal'):
            job = dict(build=args.build, baseline=args.baseline, case=case, method=method,
                       repeats=args.repeats, warmup_seconds=args.warmup_seconds,
                       effective_rescue_policy=policy)
            job_path = output.parent/f'full-job-{case["name"]}-{method}.json'
            job_path.write_text(json.dumps(job))
            completed = subprocess.run([sys.executable, __file__, '--worker', str(job_path)],
                                       capture_output=True, text=True, check=True)
            methods[method] = json.loads(completed.stdout)
        reference = methods['cpu']['first']
        records = [r for data in methods.values() for r in [data['first'], *data['runs']]]
        identical_seeds = all(r['initial_poses'] == reference['initial_poses'] for r in records)
        identical_final = all(r['matches'] == reference['matches'] for r in records)
        truth_pass = (all(a['pose_fn'] == 0 and a['pose_fp'] == 0 for d in methods.values()
                          for a in d['truth_accuracy']) if case.get('truth') else None)
        rows.append(dict(case=case, effective_rescue_policy=policy,
                         exact_seeds=identical_seeds, exact_final_matches=identical_final,
                         truth_passed=truth_pass, methods=methods,
                         speedup=methods['cpu']['median_ms']/methods['metal']['median_ms']))
        print(f'{case["name"]}: exact seeds={identical_seeds}, exact final={identical_final}, '
              f'truth={truth_pass}, CPU coarse + rescue={methods["cpu"]["median_ms"]:.3f} ms, '
              f'Metal coarse + rescue={methods["metal"]["median_ms"]:.3f} ms', flush=True)
    unchanged = all(file_hash(p) == h for p, h in {**hashes, **source_hashes, **input_hashes}.items())
    report = dict(protocol='fresh processes, frozen CPU or exact integer Metal coarse + same new GPU resident rescue',
        scope='full online source search, CPU pyramid/rotation/normalization/peaks/fine search/NMS; Metal dense coarse statistics and resident local refinement',
        truth_policy='truth evaluated only after timing/RSS; Src7 has no calibrated pose truth',
        rescue_policy='opt-in integration configuration: per-case trigger_score=min(DEFAULT_RESCUE.trigger_score, actual final detection min_score); C++ and API defaults unchanged; original default-policy slowdown remains in the separate rescue report',
        default_rescue_config=DEFAULT_RESCUE, library_hashes=hashes, source_hashes=source_hashes,
        input_hashes=input_hashes, baseline_provenance=baseline,
        sources_unchanged=unchanged, cases=rows,
        passed=baseline['verified'] and unchanged and all(r['exact_seeds'] and r['exact_final_matches'] and r['truth_passed'] is not False for r in rows))
    output.write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
