"""Native GUI scheduler / Python parity, combined NCC+overlap, and bounded soak.

Ground truth stays in this parent. Serial fresh children; all timings retain
setup/first/warmup separately. Soak records stream from the native runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import benchmark_coarse_tiling as old
import benchmark_many_objects as many
from many_object_accuracy import evaluate

ROOT=Path(__file__).resolve().parents[2]
METHODS={'python_frozen':(0,False),'python_reference':(0,False),'native_original':(0,False),'native_tile':(2,False),
         'native_overlap':(0,True),'native_combined':(2,True)}
CASE_NAMES='controlled6,heldout12,8k_t256_angle180_3scales,small_fullres,many128_pose,many512_pose'
FIELDS=('pos_x','pos_y','angle','score','scale')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def dist(xs):return dict(median=statistics.median(xs),min=min(xs),max=max(xs),values=xs)
def outputs_equal(a,b):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if any(abs(x[k]-y[k])>1e-5 for k in FIELDS):return False
        if any(abs(u-v)>1e-5 for xx,yy in zip(x['corners'],y['corners']) for u,v in zip(xx,yy)):return False
    return True

def native_job(case,method,repeats,warm):
    j=many.worker_case(case)
    # Native runner only receives search inputs; prune unsupported descriptive keys.
    j={k:v for k,v in j.items() if k in ('source','template','scales','angle','limit','score','overlap','proposal_score','stop_layer','subpixel')}
    mode,overlap=METHODS[method]
    return dict(j,backend=1,min_area=256,ncc_mode=mode,overlap_enabled=overlap,
                refine=case['workflow']=='full_rescue',subpixel=bool(case['subpixel']),invert_source=bool(case.get('invert_source',0)),repeats=repeats,warmup_seconds=warm)

def normalize(raw,method):
    if not method.startswith('python_'):return raw
    def record(r):
        return dict(total_ms=r['total_wall_ms'],prepare_ms=r['frame_prepare_wall_ms'],proposal_ms=r['proposal_wall_ms'],refine_ms=r['refine_ms'],nms_ms=r['nms_ms'],
                    matches=r['matches'],proposals=r['proposals'],generation=r['frame']['metrics']['generation'],
                    models_matched=r['frame']['metrics']['models_matched'],
                    frame_host_bytes=r['frame']['metrics']['host_pyramid_bytes'],frame_coarse_gpu_bytes=r['frame']['metrics']['coarse_gpu_bytes'],frame_fine_gpu_bytes=r['frame']['metrics']['fine_gpu_bytes'],
                    work=[dict(maps=p['metrics']['score_maps'],windows=p['metrics']['windows'],fine_maps=p['fine_metrics']['maps']) for p in r['patterns']])
    return dict(construct_ms=raw['construct_ms'],first=record(raw['first']),runs=[record(r) for r in raw['runs']],
                final_memory=raw['memory']['after_measurements'],exact_repeat=not any(r['profile_errors'] for r in [raw['first'],*raw['runs']]))

def summarize(report,base):
    cases=[]
    for case in report['cases']:
        entries=[e for e in report['execution_order'] if e['case']==case['name']]
        grouped={}
        for e in entries:
            p=base/e['result'];j=base/e['job'];assert sha(p)==e['sha256'] and sha(j)==e['job_sha256']
            grouped.setdefault(e['method'],[]).append((e['block'],normalize(json.loads(p.read_text()),e['method'])))
        reference=grouped['python_reference'][0][1]['first'];native_reference=grouped['native_original'][0][1]['first'];methods={}
        for method,blocks in grouped.items():
            runs=[r for _,w in blocks for r in w['runs']];all_calls=[r for _,w in blocks for r in [w['first'],*w['runs']]]
            eq=all(outputs_equal(r['matches'],reference['matches']) and outputs_equal(r['proposals'],reference['proposals']) for r in all_calls)
            work=all([{k:w[k] for k in ('maps','windows','fine_maps')} for w in r['work']]==[{k:w[k] for k in ('maps','windows','fine_maps')} for w in reference['work']] for r in all_calls)
            accuracy=[evaluate(r['matches'],case['truth'])['strict'] for r in all_calls]
            route=all(all(w.get('async_submitted',0)==w.get('async_completed',0) for w in r['work']) and r['models_matched']==len(case['scales']) for r in all_calls)
            if method not in ('python_frozen','python_reference','native_original','native_tile'):
                route &= all(any(w['async_submitted']>0 for w in r['work']) for r in all_calls)
            rss=[w['final_memory'].get('peak_rss_bytes',w['final_memory'].get('process_peak_rss_bytes')) for _,w in blocks]
            methods[method]=dict(total_ms=dist([r['total_ms'] for r in runs]),rss_bytes=rss,blocks=[dict(block=b,total_ms=statistics.median(r['total_ms'] for r in w['runs'])) for b,w in blocks],
                ordered_parity_within_1e_minus5=eq,native_routes_exact=method.startswith('python_') or all(r['matches']==native_reference['matches'] and r['proposals']==native_reference['proposals'] for r in all_calls),work_equal=work,route_valid=route,repeat_exact=all(w['exact_repeat'] for _,w in blocks),
                all_strict_gt=all(a['fp']==a['fn']==0 for a in accuracy),truth_count=len(case['truth']),
                max_errors={k:max(a['max_errors'][k] for a in accuracy) for k in accuracy[0]['max_errors']},
                construct_ms=[w['construct_ms'] for _,w in blocks],first_ms=[w['first']['total_ms'] for _,w in blocks],
                stage_medians={k:statistics.median(r[k] for r in runs) for k in ('prepare_ms','proposal_ms','refine_ms','nms_ms')},
                frame_capacities=[{k:w['runs'][-1][k] for k in ('frame_host_bytes','frame_coarse_gpu_bytes','frame_fine_gpu_bytes')} for _,w in blocks])
        gates={}
        for name,m in methods.items():
            if name.startswith('python_'):continue
            gates[name]={}
            for ref in ('python_frozen','python_reference','native_original'):
                if ref==name:continue
                a={b['block']:b['total_ms'] for b in methods[ref]['blocks']}
                pairs=[dict(block=b['block'],delta_percent=100*(b['total_ms']/a[b['block']]-1),passed=b['total_ms']<=a[b['block']]+max(2.,a[b['block']]*.1)) for b in m['blocks']]
                gates[name][ref]=dict(passed=all(b['passed'] for b in pairs),blocks=pairs)
        cases.append(dict(name=case['name'],methods=methods,performance=gates))
    return dict(cases=cases,correctness_passed=all(m[k] for c in cases for m in c['methods'].values() for k in ('ordered_parity_within_1e_minus5','native_routes_exact','work_equal','route_valid','repeat_exact','all_strict_gt')))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/gui-integration');p.add_argument('--baseline',type=Path,default=ROOT/'build/coarse-tiling')
    p.add_argument('--output',type=Path,default=ROOT/'build/gui-integration-results/formal.json');p.add_argument('--cases',default=CASE_NAMES);p.add_argument('--pilot',action='store_true');p.add_argument('--worker',type=Path)
    p.add_argument('--summarize',type=Path);p.add_argument('--worker-root',type=Path);a=p.parse_args()
    if a.worker:
        print(json.dumps(old.worker(json.loads(a.worker.read_text())),allow_nan=False));return
    if a.summarize:
        save(a.output,summarize(json.loads(a.summarize.read_text()),a.worker_root or a.summarize.parent));return
    names=a.cases.split(',');dest=a.output.resolve().parent;inputs=dest/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    regular=[n for n in names if not n.startswith('many')];crowded=[n for n in names if n.startswith('many')]
    cases=old.coarse.make_cases(SimpleNamespace(cases=','.join(regular),suite='all'),inputs) if regular else []
    if crowded:cases+=many.make_cases(inputs,crowded)
    cases.sort(key=lambda c:names.index(c['name']))
    sources=[Path(__file__).resolve(),ROOT/'cpp-core/tools/native_pipeline_runner.cpp',ROOT/'cpp-core/docs/gui-integration/protocol.json']
    for folder in ('src','include','experiments'):sources +=[f for f in (ROOT/'cpp-core'/folder).rglob('*') if f.is_file()]
    sources += [ROOT/'macos-cli/mainwindow.cpp',ROOT/'macos-cli/mainwindow.h']
    sources += [Path(f) for f in old.coarse.previous.source_manifest()]
    sources += list((ROOT/'cpp-core/tools').glob('*.py'))
    binaries=[a.build/'experiments/fpm_native_pipeline_runner']+[b/'experiments'/n for b in (a.build,a.baseline) for n in ('libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib')]+[a.baseline/'libfpm_c.dylib']
    report=dict(version=1,pilot=a.pilot,cases=cases,methods=METHODS,blocks=1 if a.pilot else 2,repeats=1 if a.pilot else 5,warm_seconds=0 if a.pilot else 1.,
                source_sha256={str(f):sha(f) for f in sources},binary_sha256={str(f.resolve()):sha(f) for f in binaries},input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template')},execution_order=[],complete=False)
    save(a.output,report)
    for c in cases:
        for block in range(report['blocks']):
            for method in list(METHODS) if block%2==0 else list(METHODS)[::-1]:
                label=f"{c['name']}-{method}-b{block}";jobpath=dest/(a.output.stem+'-jobs')/(label+'.json');resultpath=jobpath.with_suffix('.result.json')
                if method.startswith('python_'):
                    j=dict(case=many.worker_case(c),method='current_peak',block=block,build=str((a.baseline if method=='python_frozen' else a.build).resolve()),baseline=str(a.baseline.resolve()),repeats=report['repeats'],warmup_seconds=report['warm_seconds'])
                    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(jobpath)]
                else:
                    j=native_job(c,method,report['repeats'],report['warm_seconds']);cmd=[str((a.build/'experiments/fpm_native_pipeline_runner').resolve()),str(jobpath)]
                save(jobpath,j);print('Running '+label,flush=True);child=subprocess.run(cmd,capture_output=True,text=True)
                jobpath.with_suffix('.stderr.txt').write_text(child.stderr)
                if child.returncode:raise RuntimeError(label+': '+child.stderr[-4000:])
                save(resultpath,json.loads(child.stdout))
                report['execution_order'].append(dict(case=c['name'],method=method,block=block,job=str(jobpath.relative_to(dest)),job_sha256=sha(jobpath),result=str(resultpath.relative_to(dest)),sha256=sha(resultpath)))
                save(a.output,report)
    report['complete']=True;report['unchanged']=all(sha(k)==v for group in ('source_sha256','binary_sha256','input_sha256') for k,v in report[group].items());report['summary']=summarize(report,dest);save(a.output,report)
    if not report['unchanged'] or not report['summary']['correctness_passed']:raise SystemExit(1)

if __name__=='__main__':main()
