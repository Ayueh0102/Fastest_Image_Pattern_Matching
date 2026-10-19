"""Stage29 bounded angle-map reuse and observational rotation diagnostics.

Serial fresh processes; protocol precedes all measurements. Worker jobs contain
only pixels/search options. GT and hash-checked movable replay stay in parent.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from benchmark_fine_strip import sha, save, dist, max_present, checked
from benchmark_native_pipeline import native_job
from many_object_accuracy import evaluate

ROOT=Path(__file__).resolve().parents[2]
METHODS=['frozen','current','profile','reuse']
COUNTERS=('maps','windows','fine_maps','fine_stats_dispatches','fine_commands','fine_uploads','fine_upload_bytes','fine_roi_pixels','fine_repair_pixels','fine_peak_batch')
CAPACITIES=('frame_host_bytes','frame_coarse_gpu_bytes','frame_fine_gpu_bytes')
CLOCKS=('observe_ms','plan_ms','descriptor_ms','warp_encode_ms','warp_wait_ms','warp_device_ms','repair_all_ms','guard_scan_ms','ncc_encode_ms','ncc_wait_ms')

def job_for(case,method):
    j=native_job(case,'native_tile',case.get('benchmark_repeats',5),1.)
    j['fine_ncc_mode']=2
    if method!='frozen': j.update(fine_profiling=method=='profile',fine_angle_reuse=method=='reuse')
    return j

def baseline(build,protocol):
    old=json.loads(protocol.read_text())['binary_sha256']; verified={}
    for n in ('fpm_native_pipeline_runner','libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib'):
        matches=[v for k,v in old.items() if k.endswith('/fine-strip/experiments/'+n)]
        if len(matches)!=1 or sha(build/'experiments'/n)!=matches[0]:raise ValueError('Frozen stage28 binary differs: '+n)
        verified[n]=matches[0]
    return verified

def route(run,ref,method):
    def numeric_tree(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,dict):return all(numeric_tree(v) for v in value.values())
        if isinstance(value,list):return all(numeric_tree(v) for v in value)
        return True
    if not numeric_tree(run):return False
    if any(run[k]<0 for k in ('total_ms',*CAPACITIES)):return False
    work=run['work']; expected=ref['work']
    if any(any(w[k]<0 or int(w[k])!=w[k] for k in COUNTERS) for w in work):return False
    if len(work)!=len(expected) or run['models_matched']!=ref['models_matched']: return False
    if any(w['fine_ncc_mode']!=2 or w['fine_row_bytes'] or w['fine_row_dispatches'] or w['fine_stats_device_ms']<0 or w['fine_commands']!=2*w['fine_stats_dispatches'] for w in work):return False
    if not any(w['fine_maps'] for w in work):return False
    if method!='reuse':
        if any([w[k] for k in COUNTERS]!=[r[k] for k in COUNTERS] for w,r in zip(work,expected)):return False
        if any(run[k]!=ref[k] for k in CAPACITIES):return False
    if method=='frozen':return True
    ds=run['fine_search']
    if len(ds)!=len(work):return False
    for d,w,r in zip(ds,work,expected):
        if any(d[k]<0 for k in CLOCKS if k!='warp_device_ms'):return False
        if any(v<0 or int(v)!=v for k,v in d.items() if k not in CLOCKS):return False
        if d['profiling_enabled']!=(method=='profile') or d['angle_reuse_enabled']!=(method=='reuse'):return False
        if method=='reuse':
            if d['logical_requests']!=r['fine_maps'] or d['reused_maps']+w['fine_maps']!=d['logical_requests']:return False
            if any(w[k]!=r[k] for k in ('maps','windows','fine_uploads','fine_upload_bytes')):return False
            if any(w[k]>r[k] for k in ('fine_maps','fine_commands','fine_stats_dispatches','fine_roi_pixels','fine_repair_pixels')):return False
            if d['reused_maps'] and not d['cache_peak_bytes']:return False
            if any(run[k]!=ref[k] for k in CAPACITIES[:2]) or run['frame_fine_gpu_bytes']>ref['frame_fine_gpu_bytes']:return False
            if d['requests'] or d['history_key_payload_bytes']:return False
        else:
            if d['logical_requests'] or d['reused_maps'] or d['cache_peak_bytes']:return False
        if method=='profile':
            if d['requests']!=w['fine_maps'] or d['requests']!=sum(d[k] for k in ('within_call_duplicates','cross_call_duplicates','tracked_unique_keys','untracked_call_keys')):return False
            if d['tracked_unique_keys']>65536 or d['history_limit_keys']!=65536:return False
            if d['warp_batches']!=w['fine_stats_dispatches'] or d['zero_repair_batches']>d['warp_batches']:return False
            if d['nonzero_guard_words']>d['guard_words'] or d['warp_device_ms']<0:return False
            if abs(d['warp_wait_ms']+d['ncc_wait_ms']-w['fine_wait_ms'])>1e-6:return False
        else:
            off=('calls','requests','within_call_duplicates','cross_call_duplicates','tracked_unique_keys','untracked_call_keys','history_limit_keys','history_key_payload_bytes','history_bucket_count','peak_call_key_capacity_bytes','warp_batches','zero_repair_batches','guard_words','nonzero_guard_words')
            if any(d[k] for k in off) or any(d[k] for k in CLOCKS if k!='warp_device_ms') or d['warp_device_ms']!=-1:return False
    return True

def summarize(report,base):
    protocol=checked(base/'protocol.json',report['protocol_sha256'])
    if {k:v for k,v in report.items() if k not in ('protocol_sha256','execution_order','complete','unchanged')} != {k:v for k,v in protocol.items() if k not in ('execution_order','complete')}:
        raise ValueError('Formal settings differ from predeclared protocol')
    if len({c['name'] for c in report['cases']})!=len(report['cases']):raise ValueError('Duplicate case names')
    order=[(c['name'],m,b) for c in report['cases'] for b in range(2) for m in (METHODS if b==0 else METHODS[::-1])]
    if not report['complete'] or report['expected_workers']!=len(order) or len(report['execution_order'])!=len(order):raise ValueError('Incomplete experiment')
    groups={}
    for e,key in zip(report['execution_order'],order):
        if (e['case'],e['method'],e['block'])!=key:raise ValueError('Missing, duplicate or reordered worker')
        c=next(c for c in report['cases'] if c['name']==e['case'])
        job=checked(base/e['job'],e['job_sha256'])
        if job!=job_for(c,e['method']):raise ValueError('Worker settings differ from protocol')
        if e['returncode']!=0:raise ValueError('Worker failed')
        raw=checked(base/e['result'],e['result_sha256'])
        if len(raw['runs'])!=c.get('benchmark_repeats',5):raise ValueError('Incorrect sample count')
        groups[key]=raw
    cases=[]
    for c in report['cases']:
        name=c['name']; methods={}
        for method in METHODS:
            raws=[groups[name,method,b] for b in range(2)]; calls=[r for raw in raws for r in [raw['first'],*raw['runs']]]; runs=[r for raw in raws for r in raw['runs']]
            ref=groups[name,'frozen',0]['first']
            exact=all(raw['exact_repeat'] for raw in raws) and all(r[k]==ref[k] for r in calls for k in ('matches','proposals'))
            routed=True
            for b,raw in enumerate(raws):
                baseline_raw=groups[name,'frozen',b]
                routed &= route(raw['first'],baseline_raw['first'],method)
                routed &= all(route(r,rr,method) for r,rr in zip(raw['runs'],baseline_raw['runs']))
            accuracy=[evaluate(r['matches'],c['truth'])['strict'] for r in calls]
            m=dict(exact=exact,route=bool(routed),time_ms=dist([r['total_ms'] for r in runs]),
                first_ms=[raw['first']['total_ms'] for raw in raws],setup_ms=[raw['construct_ms'] for raw in raws],
                blocks=[dict(block=b,median_ms=statistics.median(r['total_ms'] for r in raw['runs'])) for b,raw in enumerate(raws)],
                rss_bytes=[raw['final_memory']['peak_rss_bytes'] for raw in raws],
                capacities=[{k:raw['runs'][-1][k] for k in CAPACITIES} for raw in raws],
                accuracy=dict(truth_count=len(c['truth']),max_fp=max(a['fp'] for a in accuracy),max_fn=max(a['fn'] for a in accuracy),max_errors={k:max_present(a['max_errors'][k] for a in accuracy) for k in accuracy[0]['max_errors']}),
                stages={k:dist([sum(w[k] for w in r['work']) for r in runs]) for k in ('fine_stats_device_ms','fine_rotation_ms','fine_repair_ms','fine_ncc_ms','fine_normalization_ms','fine_wait_ms',*COUNTERS[2:])})
            if method=='reuse':
                hits=sum(d['reused_maps'] for r in runs for d in r['fine_search'])
                m['route'] &= (hits>0) if c.get('expect_reuse',False) else (hits==0)
            if method!='frozen':
                fields=('logical_requests','reused_maps','cache_peak_bytes','requests','within_call_duplicates','cross_call_duplicates','tracked_unique_keys','untracked_call_keys','history_key_payload_bytes','peak_call_key_capacity_bytes','warp_batches','zero_repair_batches','guard_words','nonzero_guard_words')
                m['diagnostics']={k:dist([sum(d[k] for d in r['fine_search']) for r in runs]) for k in fields}
                # Max simultaneous cache/history differs from a sum across sequential model runs.
                m['max_model_cache_bytes']=max(d['cache_peak_bytes'] for r in runs for d in r['fine_search'])
                m['max_model_history_payload_bytes']=max(d['history_key_payload_bytes'] for r in runs for d in r['fine_search'])
                if method=='profile':m['profile_clocks']={k:dist([sum(d[k] for d in r['fine_search']) for r in runs]) for k in CLOCKS}
            methods[method]=m
        comparisons={}
        for method,ref in [('current','frozen'),('profile','current'),('reuse','current'),('reuse','frozen')]:
            a,b=methods[ref],methods[method]
            comparisons[method+'_vs_'+ref]=[dict(block=i,time_delta_percent=100*(b['blocks'][i]['median_ms']/a['blocks'][i]['median_ms']-1),
                time_tolerance_pass=b['blocks'][i]['median_ms']<=a['blocks'][i]['median_ms']+max(.1*a['blocks'][i]['median_ms'],2.),
                rss_delta_bytes=b['rss_bytes'][i]-a['rss_bytes'][i],
                rss_tolerance_pass=b['rss_bytes'][i]<=a['rss_bytes'][i]+max(.1*a['rss_bytes'][i],16*1024**2)) for i in range(2)]
        cases.append(dict(name=name,methods=methods,comparisons=comparisons))
    correct=all(m['exact'] and m['route'] and m['accuracy']['max_fp']==m['accuracy']['max_fn']==0 for c in cases for m in c['methods'].values())
    return dict(correctness_passed=correct,unchanged=report['unchanged'],workers=len(order),formal_calls=sum(c.get('benchmark_repeats',5)*8 for c in report['cases']),cases=cases)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/fine-search');p.add_argument('--baseline',type=Path,default=ROOT/'build/fine-strip')
    p.add_argument('--previous',type=Path,default=ROOT/'build/gui-integration-results/formal.json')
    p.add_argument('--baseline-protocol',type=Path,default=ROOT/'cpp-core/docs/fine-strip/protocol.json')
    p.add_argument('--output',type=Path,default=ROOT/'build/fine-search-results');p.add_argument('--summarize',type=Path)
    a=p.parse_args()
    if a.summarize:
        s=summarize(json.loads(a.summarize.read_text()),a.summarize.parent);save(a.output,s)
        if not s['correctness_passed'] or not s['unchanged']:raise SystemExit(1)
        return
    dest=a.output.resolve()
    if (dest/'protocol.json').exists():raise ValueError('Refusing to overwrite existing experiment')
    cases=json.loads(a.previous.read_text())['cases']
    if len({c['name'] for c in cases})!=len(cases):raise ValueError('Duplicate case names')
    cases=[dict(c,expect_reuse=c['name']=='small_fullres',benchmark_repeats=5) for c in cases]
    for name in ('controlled6','many128_pose','many512_pose'):
        c=next(c for c in cases if c['name']==name)
        cases.append(dict(c,name=name+'_fullres_subpixel',stop_layer=0,subpixel=1,expect_reuse=True,benchmark_repeats=3))
    verified=baseline(a.baseline,a.baseline_protocol)
    sources=[]
    for folder in ('src','include','experiments','tests','tools','examples'):
        sources += [f for f in (ROOT/'cpp-core'/folder).rglob('*') if f.is_file() and f.suffix in ('.cpp','.h','.hpp','.mm','.metal','.py','.txt','.in','.json')]
    sources+=[ROOT/'cpp-core/CMakeLists.txt']
    binaries=[b/'experiments'/n for b in (a.build,a.baseline) for n in ('fpm_native_pipeline_runner','libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib')]
    report=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),platform=platform.platform(),baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_binary_verified=verified,baseline_protocol_sha256=sha(a.baseline_protocol),cases=cases,blocks=2,repeats='5 historical cases, 3 added fullres/subpixel variants',warm_seconds=1,expected_workers=8*len(cases),
        rules=dict(order='frozen/current/profile/reuse then reverse, serial fresh processes',exact='Ordered full proposals/matches; logical workload retained while physical fine maps may decrease',
            timing='Warm total includes frame prepare; excludes decode/model learning/serialization/GT. Profiling overhead compared independently.',
            tolerance='time max(10%,2ms); RSS max(10%,16MiB); all failures retained',decision='opt-in only; no automatic default change',data='Six reused stage14 scenes plus three changed search configurations on same images, not independent camera data'),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(sources))},binary_sha256={str(f.resolve()):sha(f) for f in binaries},
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template')},previous_sha256=sha(a.previous),execution_order=[],complete=False)
    save(dest/'protocol.json',report);report['protocol_sha256']=sha(dest/'protocol.json')
    for c in cases:
        for block in range(2):
            for method in (METHODS if block==0 else METHODS[::-1]):
                label=f'{c["name"]}-{method}-b{block}';jobpath=dest/'jobs'/(label+'.json');save(jobpath,job_for(c,method));print('Running '+label,flush=True)
                build=a.baseline if method=='frozen' else a.build
                r=subprocess.run([str((build/'experiments/fpm_native_pipeline_runner').resolve()),str(jobpath)],capture_output=True,text=True)
                rawpath=jobpath.with_suffix('.result.json');rawpath.write_text(r.stdout);jobpath.with_suffix('.stderr.txt').write_text(r.stderr)
                report['execution_order'].append(dict(case=c['name'],method=method,block=block,job=str(jobpath.relative_to(dest)),job_sha256=sha(jobpath),result=str(rawpath.relative_to(dest)),result_sha256=sha(rawpath),returncode=r.returncode));save(dest/'formal.json',report)
                if r.returncode:raise RuntimeError(label+' failed: '+r.stderr[-3000:])
                json.loads(r.stdout)
    report['complete']=True
    report['unchanged']=all(sha(ROOT/k)==v for k,v in report['source_sha256'].items()) and all(sha(k)==v for group in ('binary_sha256','input_sha256') for k,v in report[group].items())
    save(dest/'formal.json',report);summary=summarize(report,dest);save(dest/'summary.json',summary)
    if not summary['correctness_passed'] or not report['unchanged']:raise SystemExit(1)

if __name__=='__main__':main()
