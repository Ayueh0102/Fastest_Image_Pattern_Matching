"""Reproducible >100-instance stress study using the existing frozen/current matcher.

No matcher policy is changed. Scene generation and independent GT evaluation
are outside worker timing and memory. Workers execute serially in fresh processes.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import benchmark_coarse_next as coarse
from benchmark_coarse_ncc import core_library_path, file_hash
from pose_experiment import library_path as pose_library_path
from summarize_pipeline_stages import load_worker, worker_logical_path
from many_object_scenes import make_cases
from many_object_accuracy import evaluate

METHODS = ('previous_shared', 'current_control', 'peak_only',
           'current_control_profile', 'peak_only_profile')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def diagnose(record, case):
    """Post-run clues for missing truth; no GT can affect matching or thresholds."""
    accuracy = evaluate(record['matches'], case['truth'])
    return dict(accuracy=accuracy, proposals=len(record['proposals']),
                final_matches=len(record['matches']), effective_limit=case['limit'],
                refinement_metrics=record.get('refinement_metrics'))


def worker_case(case):
    # Only search settings cross the worker boundary. No GT, scene count or ROI hints.
    fields = ('name', 'source', 'template', 'scales', 'angle', 'limit', 'proposal_score',
              'score', 'overlap', 'stop_layer', 'subpixel', 'workflow', 'invert_source')
    return {key:case[key] for key in fields if key in case}


def summarize_case(case, workers):
    # Legacy accuracy is retained as a comparison, evaluated in the parent only.
    for worker in workers:
        for record in [worker['first'], *worker['runs']]:
            record['truth_accuracy'] = coarse.real.truth_accuracy(record['matches'], case['truth'])
    reference = next(w for w in workers if w['method'] == 'previous_shared')
    summary = coarse.summarize_case(case, workers, reference)
    summary['independent_accuracy'] = {}
    summary['outer_stages'] = {}
    for method in dict.fromkeys(w['method'] for w in workers):
        group = [w for w in workers if w['method'] == method]
        records = [r for w in group for r in [w['first'], *w['runs']]]
        assessments = [evaluate(r['matches'], case['truth']) for r in records]
        summary['independent_accuracy'][method] = dict(first=assessments[0],
            all_calls_same_counts=all(a['strict']['tp'] == assessments[0]['strict']['tp'] and
                a['strict']['fp'] == assessments[0]['strict']['fp'] and
                a['strict']['fn'] == assessments[0]['strict']['fn'] for a in assessments),
            diagnosis=diagnose(group[0]['first'], case))
        measured = [r for w in group for r in w['runs']]
        summary['outer_stages'][method] = {key:coarse.previous.distribution([r[key] for r in measured])
            for key in ('frame_prepare_wall_ms','proposal_wall_ms','refine_ms','nms_ms','seed_count')}
    return summary


def replay(report, report_path, worker_root=None):
    groups, hashes = {}, {}
    for entry in report['execution_order']:
        logical, digest, worker = load_worker(entry, report_path, worker_root)
        if logical in hashes:
            raise ValueError('Duplicate worker: '+logical)
        hashes[logical] = digest
        if worker_root is not None:
            job = Path(worker_root)/str(worker_logical_path(entry['job']))
        else:
            job = Path(entry['job'])
        if file_hash(job) != entry['job_sha256']:
            raise ValueError('Job changed: '+str(job))
        groups.setdefault(entry['case'], []).append(worker)
    return dict(report=Path(report_path).name, report_sha256=file_hash(report_path),
        raw_worker_sha256=hashes, complete=report.get('complete'),
        integrity_passed=report.get('integrity_passed'), detection_all_passed=report.get('detection_all_passed'),
        design=report['design'], cases=[summarize_case(c['case'], groups[c['case']['name']]) for c in report['cases']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default=str(coarse.real.ROOT/'build/fourier-gpu'))
    parser.add_argument('--baseline', default=str(coarse.real.ROOT/'build/coarse-next-baseline'))
    parser.add_argument('--cases')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-seconds', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=coarse.real.ROOT/'build/many-objects/formal.json')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--worker-root', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(coarse.worker(json.loads(args.worker.read_text())), allow_nan=False))
        return
    output = args.output.resolve()
    if args.summarize:
        if output == args.summarize.resolve(): parser.error('Do not overwrite the raw report')
        save(output, replay(json.loads(args.summarize.read_text()), args.summarize, args.worker_root))
        return
    methods = args.methods.split(',')
    if set(methods)-set(METHODS) or len(methods)!=len(set(methods)) or 'previous_shared' not in methods:
        parser.error('Use unique listed methods including previous_shared')
    if args.pilot: args.blocks,args.repeats,args.warmup_seconds=1,1,0.
    if args.blocks<1 or args.repeats<1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds<0:
        parser.error('Positive blocks/repeats and nonnegative finite warmup required')
    inputs, jobs = output.parent/'inputs', output.parent/(output.stem+'-jobs')
    inputs.mkdir(parents=True, exist_ok=True); jobs.mkdir(parents=True, exist_ok=True)
    cases = make_cases(inputs, args.cases.split(',') if args.cases else None)
    baseline = coarse.verify_baseline(args.baseline)
    sources = coarse.previous.source_manifest()
    for name in ('benchmark_many_objects.py','many_object_scenes.py','many_object_accuracy.py',
                 'benchmark_coarse_next.py','benchmark_shared_fine.py','summarize_pipeline_stages.py'):
        path=Path(__file__).with_name(name).resolve(); sources[str(path)]=file_hash(path)
    protocol=coarse.real.ROOT/'cpp-core/docs/many-objects/protocol.json'
    sources[str(protocol)] = file_hash(protocol)
    libraries = dict(baseline['library_hashes'])
    for path in (core_library_path(args.build),coarse.coarse.library_path(args.build),pose_library_path(args.build)):
        libraries[str(Path(path).resolve())]=file_hash(path)
    inputs_hash = {case[key]:file_hash(case[key]) for case in cases for key in ('source','template','metadata')}
    report=dict(created=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        design=dict(methods=methods,blocks=args.blocks,repeats=args.repeats,warmup_seconds=args.warmup_seconds,
            pilot=args.pilot,pyramid_mode=1,batch_angles=4,fine_batch_size=512,
            scenario='Multiple instances of one real-texture template; no occlusion; independent OpenCV rendering',
            timing='Fresh serial processes; forward/reverse order; prepare included per call; setup/first separate; profile separate',
            accuracy='Maximum-cardinality one-to-one pose gates: 2px/1deg/2%scale; 20px localization separate; GT after timing/RSS',
            memory='Process peak RSS overlaps frame/pose allocations; do not add them',
            policies='4K square pose cases share max_targets=384; 8K512 uses768; rescue policy unchanged (at most4 seeds)'),
        baseline_provenance=baseline,source_hashes=sources,library_hashes=libraries,input_hashes=inputs_hash,
        execution_order=[],cases=[],complete=False)
    save(output,report)
    for case in cases:
        workers=[]
        for block in range(args.blocks):
            for method in methods if block%2==0 else methods[::-1]:
                job=dict(case=worker_case(case),method=method,block=block,build=str(Path(args.build).resolve()),
                    baseline=str(Path(args.baseline).resolve()),repeats=args.repeats,
                    warmup_seconds=args.warmup_seconds,fine_batch_size=512)
                label=f'{case["name"]}-{method}-b{block}'; job_path=jobs/(label+'.json')
                save(job_path,job); print('Running '+label,flush=True)
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(job_path)],
                                     capture_output=True,text=True)
                (jobs/(label+'.stderr.txt')).write_text(child.stderr)
                if child.returncode:raise RuntimeError(f'{label}: {child.stderr[-4000:]}')
                result=json.loads(child.stdout); result_path=jobs/(label+'.result.json'); save(result_path,result)
                workers.append(result)
                report['execution_order'].append(dict(case=case['name'],method=method,block=block,
                    job=str(job_path),job_sha256=file_hash(job_path),result=str(result_path),result_sha256=file_hash(result_path)))
                save(output,report)
        row=summarize_case(case,workers);report['cases'].append(row);save(output,report)
        acc=row['independent_accuracy']['peak_only' if 'peak_only' in methods else 'previous_shared']['first']['strict']
        print(f'{case["name"]}: exact={row["exact_all_outputs"]}, TP/FP/FN={acc["tp"]}/{acc["fp"]}/{acc["fn"]}',flush=True)
    hashes={**sources,**libraries,**inputs_hash,baseline['manifest']:baseline['manifest_sha256']}
    report['sources_unchanged']=all(file_hash(p)==h for p,h in hashes.items())
    report['complete']=True
    report['integrity_passed']=report['sources_unchanged'] and all(c['exact_all_outputs'] and c['profile_valid'] and
        c['work_counts_match_reference'] and all(a['all_calls_same_counts'] for a in c['independent_accuracy'].values()) for c in report['cases'])
    report['detection_all_passed']=all(a['first']['strict']['fp']==0 and a['first']['strict']['fn']==0
        for c in report['cases'] for a in c['independent_accuracy'].values())
    save(output,report)
    if not report['integrity_passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
