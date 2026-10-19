"""Stage34 bounded scale-support polish on actual full-search predictions.

Unlike scale_sampling_study, truth never seeds the optimizer. Preserve original
thresholds, all failed outcomes, first/repeat timing and fresh-process RSS.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from benchmark_native_pipeline import sha, save
from benchmark_resident_multiscale import job_for, frontend_state
from benchmark_fine_fused import source_files
from benchmark_closure_backends import distribution
from many_object_accuracy import evaluate

ROOT=Path(__file__).resolve().parents[2]
METHODS=('baseline','support')


def cases_for(manifest, old_cases):
    old=next(c for c in old_cases if c['name']=='new_small12')
    cases=[dict(old,split='development',group='retained12')]
    for c in manifest['cases']:
        if c['group']!='held_out':continue
        # All 12 small-template poses ×2 renderers; two poses ×2 renderers
        # per larger control template. Selection is fixed by input identifiers.
        if c['fixture']!='Dst4.bmp' and not any(f'-{manifest["held_out_seeds"][0]}-{i}-' in c['name'] for i in (0,1)):continue
        cases.append(dict(name=c['name'],source=c['source'],template=c['template'],truth=[c['pose']],
            scales=[.75,1.,1.25],angle=180,limit=8,proposal_score=.5,score=.8,overlap=.2,
            stop_layer=1,subpixel=0,workflow='full_rescue',split='heldout',group=c['fixture'],renderer=c['renderer']))
    if len(cases)!=33:raise ValueError('Expected frozen 24 small +8 control +1 retained scenes')
    return cases


def make_job(case,method):
    j=job_for(case,'resident0')
    j.update(refinement_backend=1,scale_support_refine=method=='support')
    return j


def support_route(row,method):
    fields=('scale_support_refine_ms','scale_support_eligible','scale_support_changed',
            'scale_support_boundary','scale_support_evaluations','scale_support_samples')
    if row.get('scale_support_enabled') is not (method=='support') or any(k not in row for k in fields):
        return False
    if method=='baseline':return all(row[k]==0 for k in fields)
    return (0<=row['scale_support_changed']<=row['scale_support_eligible'] and
            0<=row['scale_support_boundary']<=row['scale_support_eligible'] and
            0<=row['scale_support_refine_ms']<=row['refine_ms'] and
            row['scale_support_evaluations']>=row['scale_support_eligible'] and
            row['scale_support_samples']>=row['scale_support_evaluations'])


def summarize(report,base):
    p=report['protocol']
    if not report.get('complete') or sha(base/'protocol.json')!=report['protocol_sha256'] or json.loads((base/'protocol.json').read_text())!=p:raise ValueError('Incomplete or changed protocol')
    expected=[(c['name'],b,m) for c in p['cases'] for b in range(2) for m in (METHODS if b==0 else METHODS[::-1])]
    if [(e['case'],e['block'],e['method']) for e in report['entries']]!=expected:raise ValueError('Missing or unordered workers')
    workers={}
    for e in report['entries']:
        for k in ('job','raw','stderr'):
            if sha(base/e[k])!=e[k+'_sha256']:raise ValueError('Changed raw evidence')
        c=next(c for c in p['cases'] if c['name']==e['case'])
        if json.loads((base/e['job']).read_text())!=make_job(c,e['method']):raise ValueError('Changed worker job')
        w=json.loads((base/e['raw']).read_text())
        if e['returncode'] or len(w['runs'])!=1 or w['warmups_ms']:raise ValueError('Failed/invalid worker')
        workers[(c['name'],e['block'],e['method'])]=w
    results=[]
    for c in p['cases']:
        methods={};pairs=[]
        for m in METHODS:
            ws=[workers[(c['name'],b,m)] for b in range(2)]
            calls=[r for w in ws for r in [w['first'],*w['runs']]]
            acc=[evaluate(r['matches'],c['truth']) for r in calls]
            methods[m]=dict(first_ms=distribution([w['first']['total_ms'] for w in ws]),
                repeat_ms=distribution([w['runs'][0]['total_ms'] for w in ws]),
                refine_ms=distribution([w['runs'][0]['refine_ms'] for w in ws]),
                peak_rss_bytes=distribution([w['final_memory']['peak_rss_bytes'] for w in ws]),
                repeat_exact=all(w['exact_repeat'] for w in ws),
                support_route_passed=all(support_route(r,m) for r in calls),
                strict_tp=min(a['strict']['tp'] for a in acc),strict_fp=max(a['strict']['fp'] for a in acc),
                strict_fn=max(a['strict']['fn'] for a in acc),localization_tp=min(a['localization']['tp'] for a in acc),
                accuracy=acc[0],support_metrics={k:[r[k] for r in calls] for k in calls[0] if k.startswith('scale_support_')})
        for b in range(2):
            a,z=[workers[(c['name'],b,m)] for m in METHODS]
            for phase,r,q in [('first',a['first'],z['first']),('repeat',a['runs'][0],z['runs'][0])]:
                pairs.append(dict(block=b,phase=phase,frontend_exact=frontend_state(r)==frontend_state(q) and r['proposals']==q['proposals'],
                    time_gate=q['total_ms']<=r['total_ms']+max(2.,.1*r['total_ms']),
                    rss_gate=z['final_memory']['peak_rss_bytes']<=a['final_memory']['peak_rss_bytes']+max(16*2**20,.1*a['final_memory']['peak_rss_bytes'])))
        results.append(dict(name=c['name'],split=c['split'],group=c['group'],methods=methods,pairs=pairs))
    return dict(complete=True,workers=len(report['entries']),algorithm_calls=2*len(report['entries']),
        unchanged=report['unchanged'],cases=results,
        integrity_passed=report['unchanged'] and all(m['repeat_exact'] and m['support_route_passed'] for c in results for m in c['methods'].values()) and all(p['frontend_exact'] for c in results for p in c['pairs']),
        all_support_accuracy_passed=all(c['methods']['support']['strict_fp']==c['methods']['support']['strict_fn']==0 for c in results),
        all_time_tolerances_passed=all(p['time_gate'] for c in results for p in c['pairs']),
        all_rss_tolerances_passed=all(p['rss_gate'] for c in results for p in c['pairs']),
        limits=['Small-template opt-in CPU scale polish after Metal refinement; not a new GPU kernel.',
                'Held-out geometry was used in prior fixed-GT diagnostic, so this is independent end-to-end validation, not a blind final holdout.',
                'Two fresh processes per method/case; small timing changes are descriptive.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/stage34-closure')
    p.add_argument('--manifest',type=Path,default=ROOT/'build/stage34-scale/study/inputs/manifest.json')
    p.add_argument('--old-cases',type=Path,default=ROOT/'build/resident-multiscale-inputs/cases.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summarize',type=Path)
    a=p.parse_args()
    if a.summarize:
        result=summarize(json.loads(a.summarize.read_text()),a.summarize.parent);save(a.output,result);return
    dest=a.output.resolve()
    if dest.exists():raise ValueError('Refusing to overwrite')
    dest.mkdir(parents=True)
    cases=cases_for(json.loads(a.manifest.read_text()),json.loads(a.old_cases.read_text()))
    paths=[*source_files(),a.manifest,a.old_cases,*[a.build/'experiments'/n for n in ('fpm_native_pipeline_runner','libfpm_pose_experiment.dylib','libfpm_coarse_experiment.dylib')],*[Path(c[k]) for c in cases for k in ('source','template')]]
    frozen={str(p.resolve()):sha(p) for p in paths}
    protocol=dict(cases=cases,blocks=2,repeats=1,hashes=frozen,methods=METHODS,
        scale_gate=.02,position_gate_px=2.,angle_gate_deg=1.,time_gate='max(10%,2ms)',rss_gate='max(10%,16MiB)')
    # JSON-normalize before hashing and replay comparison.
    protocol=json.loads(json.dumps(protocol));save(dest/'protocol.json',protocol)
    report=dict(protocol=protocol,protocol_sha256=sha(dest/'protocol.json'),entries=[],complete=False)
    for c in cases:
        for b in range(2):
            for m in (METHODS if b==0 else METHODS[::-1]):
                label=f'{c["name"]}-{m}-{b}';jp=dest/'jobs'/(label+'.json');rp=jp.with_suffix('.raw.json');ep=jp.with_suffix('.stderr.txt')
                save(jp,make_job(c,m));print(label,flush=True)
                child=subprocess.run([str(a.build/'experiments/fpm_native_pipeline_runner'),str(jp)],capture_output=True,text=True)
                rp.write_text(child.stdout);ep.write_text(child.stderr)
                e=dict(case=c['name'],block=b,method=m,returncode=child.returncode)
                for k,path in [('job',jp),('raw',rp),('stderr',ep)]:e[k]=str(path.relative_to(dest));e[k+'_sha256']=sha(path)
                report['entries'].append(e);save(dest/'formal.json',report)
                if child.returncode:raise RuntimeError(child.stderr)
    report.update(complete=True,unchanged=all(sha(k)==v for k,v in frozen.items()))
    save(dest/'formal.json',report);summary=summarize(report,dest);save(dest/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
