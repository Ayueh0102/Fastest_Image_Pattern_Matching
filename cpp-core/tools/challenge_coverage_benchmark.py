"""Stage34 bounded expanded-search challenge; preserves all shared failures.

Full pipeline results, timings, peak RSS and proposal coverage are separate.
The heldout split is frozen before development results; no GT reaches workers.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess

from benchmark_fine_strip import save, sha
from benchmark_native_pipeline import native_job
from many_object_accuracy import evaluate

ROOT=Path(__file__).resolve().parents[2]
METHODS={'baseline':dict(proposal_score=.5,limit=70), 'expanded':dict(proposal_score=.4,limit=140)}
BINARIES=('fpm_native_pipeline_runner','libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib')


def worker_job(case, method):
    if method not in METHODS:
        raise ValueError('Unknown method')
    job=native_job(dict(case,**METHODS[method]),'native_tile',1,0.)
    job.update(fine_ncc_mode=2,fine_profiling=False,fine_angle_reuse=True,fine_fused=True,
        pose_resident_mode=0,pose_input_mode=1,pose_storage_mode=1,refinement_backend=1,
        gray_load_mode='direct_gray8',run_resource_diagnostics=False)
    return job


def proposal_coverage(proposals, truth):
    details=[]
    for ti,t in enumerate(truth):
        candidates=[]
        for pi,p in enumerate(proposals):
            dx,dy=abs(p['pos_x']-t['pos_x']),abs(p['pos_y']-t['pos_y'])
            angle=abs((p['angle']-t['angle']+180)%360-180)
            logscale=abs(math.log(p['scale']/t['scale']))
            if dx<=12 and dy<=12 and angle<=15 and logscale<=math.log(1.3):
                candidates.append(pi)
        details.append(dict(truth_index=ti,covered=bool(candidates),proposal_indices=candidates))
    return dict(covered=sum(i['covered'] for i in details),truth_count=len(truth),details=details,
        bounds=dict(per_axis_translation_px=12,angle_deg=15,log_scale=math.log(1.3)),
        note='Necessary solver-bound coverage; not proof of attraction-basin convergence.')


def diagnose(run, case):
    accuracy=evaluate(run['matches'],case['truth'])
    coverage=proposal_coverage(run['proposals'],case['truth'])
    strict=accuracy['strict']; locations=accuracy['localization']
    failures=[]
    localized={p['truth_index']:p['prediction_index'] for p in locations['pairs']}
    for ti in strict['unmatched_truth_indices']:
        covered=coverage['details'][ti]['covered']
        pi=localized.get(ti)
        failures.append(dict(truth_index=ti,visible_content_fraction=case['truth'][ti]['visible_content_fraction'],
            proposal_covered=covered,localized_prediction=pi,
            reason=('missing_proposal_within_solver_bounds' if not covered else
                    'accepted_wrong_pose' if pi is not None else 'covered_but_no_localized_acceptance'),
            note='No per-candidate rejection trace; verification, optimization and NMS are not separately attributable.'))
    distractor_hits=[]
    for pi in strict['unmatched_prediction_indices']:
        p=run['matches'][pi]
        close=[di for di,d in enumerate(case['distractors']) if math.hypot(p['pos_x']-d['pos_x'],p['pos_y']-d['pos_y'])<=20]
        if close:
            distractor_hits.append(dict(prediction_index=pi,distractor_indices=close,score=p['score']))
    return dict(accuracy=accuracy,proposal_coverage=coverage,failures=failures,distractor_false_acceptances=distractor_hits)


def historic_periodic():
    path=ROOT/'cpp-core/docs/ncc-rescue/table.json'
    rows=json.loads(path.read_text())
    periodic=[r for r in rows if r['case'].startswith('ambiguous_periodic_')]
    return dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),rows=periodic,
        historical_only=True,retested_current_binary=False,
        geometric_failures_preserved=sum(r.get('precise_pass_count')==0 for r in periodic),
        note='Historical diagnostic passed flag was not geometric success. High-NCC 8px alias remains unresolved; wider full-image search is not proof of its repair.')


def prepare(cases_path, output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite protocol/results')
    cases=json.loads(Path(cases_path).read_text())
    expected={s+'_'+k for s in ('development','heldout') for k in ('occlusion','distractors','contact')}
    if len(cases)!=6 or {c['name'] for c in cases}!=expected:
        raise ValueError('Exactly six predeclared challenge cases required')
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cases=cases,methods=METHODS,blocks=2,repeats=1,warmup_seconds=0,
        order=[dict(case=c['name'],block=b,method=m) for c in cases for b in range(2)
               for m in (list(METHODS) if b==0 else list(METHODS)[::-1])],
        inputs_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template','metadata')},
        cases_sha256=sha(cases_path),source_sha256={str(ROOT/'cpp-core/tools'/f):sha(ROOT/'cpp-core/tools'/f)
          for f in ('challenge_coverage_scenes.py','challenge_coverage_benchmark.py','many_object_scenes.py',
                    'many_object_accuracy.py','benchmark_native_pipeline.py','benchmark_fine_strip.py')},
        accuracy_gate=dict(position_px=2,angle_deg=1,relative_scale=.02),
        performance_gate=dict(relative=.1,absolute_ms=2.,rss_relative=.1,rss_absolute_bytes=16*1024**2),
        remedy='Only proposal threshold .5 to .4 and per-scale cap70 to140; all final/refinement/overlap gates unchanged.',
        heldout_policy='Both splits and both methods frozen before observing any matcher result; no adaptive tuning.',
        memory_scope='Native peak RSS through final_memory snapshot; excludes later final JSON encoding/output peaks.',
        limitations=['Two fresh processes per case/method, not a confidence interval.',
          'No external camera GT. Synthetic identity negatives test stricter exact identity than generic object resemblance.',
          'Necessary proposal coverage does not identify optimizer basin or distinguish optimizer/verification/NMS failures.'])
    save(output/'protocol.json',protocol)
    save(output/'historical-periodic.json',historic_periodic())
    return protocol


def summarize(report, root):
    root=Path(root)
    protocol_path=root/'protocol.json'
    if sha(protocol_path)!=report['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    protocol=json.loads(protocol_path.read_text())
    if any(report.get(k)!=v for k,v in protocol.items()) or not report.get('complete'):
        raise ValueError('Incomplete or modified protocol binding')
    grouped={c['name']:{m:[] for m in METHODS} for c in report['cases']}
    valid=True
    expected={(o['case'],o['method'],o['block']) for o in report['order']}
    got=[]
    for entry in report['execution']:
        key=(entry['case'],entry['method'],entry['block']);got.append(key)
        for path_key,hash_key in (('job','job_sha256'),('result','result_sha256'),('stderr','stderr_sha256')):
            if sha(root/entry[path_key])!=entry[hash_key]:
                raise ValueError('Evidence hash mismatch: '+entry[path_key])
        case=next(c for c in report['cases'] if c['name']==entry['case'])
        if json.loads((root/entry['job']).read_text())!=worker_job(case,entry['method']):
            raise ValueError('Worker job differs from frozen policy')
        raw=json.loads((root/entry['result']).read_text())
        if len(raw['runs'])!=1 or raw['warmups_ms'] or not raw['exact_repeat']:
            raise ValueError('Unexpected repeat/warmup state')
        calls=[raw['first'],*raw['runs']]
        for run in calls:
            if any(not math.isfinite(run[k]) or run[k]<0 for k in ('total_ms','proposal_ms','refine_ms')):
                raise ValueError('Invalid measured time')
            if run['proposals'] and run['selected_refinement_backend']!=1:
                raise ValueError('Wrong refinement backend')
            if (run['pose_input_mode'],run['pose_storage_mode'],run['refinement_mode'])!=(1,1,0):
                raise ValueError('Wrong pose settings')
        if calls[0]['matches']!=calls[1]['matches'] or calls[0]['proposals']!=calls[1]['proposals']:
            raise ValueError('Repeat output differs')
        diagnostics=[diagnose(r,case) for r in calls]
        grouped[case['name']][entry['method']].append(dict(block=entry['block'],
            first_ms=calls[0]['total_ms'],repeat_ms=calls[1]['total_ms'],
            proposal_ms=calls[1]['proposal_ms'],refine_ms=calls[1]['refine_ms'],
            rss_bytes=raw['final_memory']['peak_rss_bytes'],construct_ms=raw['construct_ms'],
            decode_ms=raw['load']['decode_ms'],proposal_count=len(calls[1]['proposals']),
            match_count=len(calls[1]['matches']),diagnostics=diagnostics))
    if len(got)!=len(expected) or set(got)!=expected or len(set(got))!=len(got):
        raise ValueError('Missing or duplicate workers')
    results=[]
    for case in report['cases']:
        methods={}
        for method,blocks in grouped[case['name']].items():
            methods[method]=dict(blocks=blocks,median={k:statistics.median(b[k] for b in blocks)
               for k in ('first_ms','repeat_ms','proposal_ms','refine_ms','rss_bytes','construct_ms','decode_ms','proposal_count','match_count')},
               all_strict_passed=all(d['accuracy']['strict']['fp']==d['accuracy']['strict']['fn']==0 for b in blocks for d in b['diagnostics']))
        base={b['block']:b for b in methods['baseline']['blocks']}
        paired=[]
        for trial in methods['expanded']['blocks']:
            b=base[trial['block']]
            paired.append(dict(block=trial['block'],**{k:dict(baseline=b[k],expanded=trial[k],
                delta_percent=100*(trial[k]/b[k]-1),passed=trial[k]<=b[k]+max(b[k]*.1,16*1024**2 if k=='rss_bytes' else 2))
                for k in ('first_ms','repeat_ms','rss_bytes')}))
        results.append(dict(name=case['name'],split=case['split'],kind=case['kind'],methods=methods,paired=paired))
    return dict(complete=True,workers=len(got),algorithm_calls=len(got)*2,
        evidence_integrity_passed=True,unchanged=report['unchanged'],cases=results,
        all_strict_passed=all(m['all_strict_passed'] for c in results for m in c['methods'].values()),
        all_performance_passed=all(p[k]['passed'] for c in results for p in c['paired'] for k in ('first_ms','repeat_ms','rss_bytes')),
        performance_claim='Descriptive paired blocks; no significance or general hardware optimality claim.')


def run(protocol_path,build):
    protocol_path=Path(protocol_path).resolve();output=protocol_path.parent
    if (output/'formal.json').exists():
        raise ValueError('Refusing to overwrite previous execution')
    report=json.loads(protocol_path.read_text())
    report.update(protocol_sha256=sha(protocol_path),execution=[],complete=False,
        binary_sha256={str(Path(build).resolve()/'experiments'/n):sha(Path(build)/'experiments'/n) for n in BINARIES})
    for name in ('inputs_sha256','source_sha256','binary_sha256'):
        if not all(sha(k)==v for k,v in report[name].items()):
            raise ValueError('Frozen '+name+' changed before run')
    save(output/'formal.json',report)
    cases={c['name']:c for c in report['cases']}
    for entry in report['order']:
        label=f"{entry['case']}-{entry['method']}-b{entry['block']}"
        job=output/'jobs'/(label+'.json');result=job.with_suffix('.result.json');stderr=job.with_suffix('.stderr.txt')
        save(job,worker_job(cases[entry['case']],entry['method']))
        print('Running '+label,flush=True)
        child=subprocess.run([str(Path(build).resolve()/'experiments/fpm_native_pipeline_runner'),str(job)],capture_output=True,text=True)
        stderr.write_text(child.stderr)
        if child.returncode:
            raise RuntimeError(label+': '+child.stderr[-4000:])
        save(result,json.loads(child.stdout))
        report['execution'].append(dict(entry,job=str(job.relative_to(output)),job_sha256=sha(job),
            result=str(result.relative_to(output)),result_sha256=sha(result),stderr=str(stderr.relative_to(output)),stderr_sha256=sha(stderr)))
        save(output/'formal.json',report)
    report['complete']=True
    report['unchanged']=all(sha(k)==v for name in ('inputs_sha256','source_sha256','binary_sha256') for k,v in report[name].items())
    save(output/'formal.json',report)
    summary=summarize(report,output);save(output/'summary.json',summary)
    return 0 if summary['all_strict_passed'] and summary['all_performance_passed'] and summary['unchanged'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path,help='Case manifest; write frozen protocol without running GPU')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--build',type=Path,default=ROOT/'build/stage34-closure')
    parser.add_argument('--summarize',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        prepare(args.prepare,args.output);return
    if args.summarize:
        summary=summarize(json.loads(args.summarize.read_text()),args.summarize.parent)
        save(args.output,summary)
        raise SystemExit(0 if summary['all_strict_passed'] and summary['all_performance_passed'] and summary['unchanged'] else 1)
    if not args.protocol:
        parser.error('--prepare, --protocol or --summarize required')
    if args.output.resolve()!=args.protocol.resolve().parent:
        parser.error('--output must be the protocol directory for execution')
    raise SystemExit(run(args.protocol,args.build))

if __name__=='__main__':
    main()
