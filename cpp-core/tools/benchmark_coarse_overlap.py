"""Compare frozen peak-tree mode with current synchronous/overlapped coarse work.

The same resident rescue build is used for every route. GT stays in the parent;
all workers run serially. Full output/order/map equality is checked separately
from independent crowded-scene accuracy and performance.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import benchmark_coarse_next as coarse
import benchmark_many_objects as many
from benchmark_coarse_ncc import core_library_path,file_hash
from pose_experiment import library_path as pose_library_path
from summarize_pipeline_stages import paired_regression,load_worker,worker_logical_path
from many_object_accuracy import evaluate

METHODS=('previous_peak','current_peak','overlap','current_peak_profile','overlap_profile')
BASELINE_COMMIT='409275452dc1d676f37830ae73c520ca4389d3cd'


class FrameAdapter(coarse.FrameAdapter):
    def __init__(self,job,patterns):
        super().__init__(job,patterns)
        try:
            if job['_overlap'] is None:
                self.frame.set_coarse_optimization(peak_mode=1,ncc_mode=0)
                self.config['coarse_optimization']=dict(peak_mode=1,ncc_mode=0,frozen_api=True)
            else:
                self.frame.set_coarse_overlap(enabled=job['_overlap'])
            self.config['coarse_overlap']=job['_overlap']
        except Exception:
            self.close();raise

    def snapshot(self):
        value=super().snapshot()
        value['overlap_metrics']=copy.deepcopy(getattr(self.frame,'last_overlap_metrics',None))
        return value


class PatternAdapter(coarse.shared.PatternAdapter):
    def match(self,source,config,frame):
        result=super().match(source,config,frame)
        # Dictionary is replaced on every call; retaining its reference is enough.
        self.overlap_snapshot=getattr(frame.frame,'last_overlap_metrics',None)
        return result

    def snapshot(self,scale,setup=False):
        value=super().snapshot(scale,setup)
        value['overlap_metrics']=copy.deepcopy(getattr(self,'overlap_snapshot',None))
        return value


class Search(coarse.Search):
    def __init__(self,job):
        base=job['method'].removesuffix('_profile')
        frozen=base=='previous_peak'
        mapped='previous_shared' if frozen else 'peak_only'
        if job['method'].endswith('_profile'):mapped+='_profile'
        mapped_job=dict(job,method=mapped,_overlap=None if frozen else base=='overlap')
        super().__init__(mapped_job,frame_adapter=FrameAdapter,pattern_adapter=PatternAdapter)


def verify_baseline(build):
    build=Path(build).resolve();path=build/'provenance.json';data=json.loads(path.read_text())
    if data['commit']!=BASELINE_COMMIT:raise ValueError('Expected frozen4092754 peak-tree baseline')
    hashes={}
    for name,digest in data['files'].items():
        target=build/name
        if file_hash(target)!=digest:raise ValueError('Frozen library changed: '+str(target))
        hashes[str(target)]=digest
    return dict(commit=data['commit'],manifest=str(path),manifest_sha256=file_hash(path),library_hashes=hashes)


def summarize_case(case,workers):
    for worker in workers:
        if case.get('truth'):
            for record in [worker['first'],*worker['runs']]:
                record['truth_accuracy']=coarse.real.truth_accuracy(record['matches'],case['truth'])
    reference=next(w for w in workers if w['method']=='previous_peak')
    row=coarse.summarize_case(case,workers,reference)
    row['performance']={name:{ref:paired_regression(row['methods'],name,ref)
        for ref in ('previous_peak','current_peak') if ref!=name}
        for name in ('current_peak','overlap') if name in row['methods']}
    row['outer_stages']={};row['accuracy']={};row['overlap_diagnostics']={}
    row['overlap_valid']=True
    row['profile_overhead_percent_by_block']={}
    for method in dict.fromkeys(w['method'] for w in workers):
        group=[w for w in workers if w['method']==method]
        records=[r for w in group for r in w['runs']]
        row['outer_stages'][method]={key:coarse.previous.distribution([r[key] for r in records])
            for key in ('frame_prepare_wall_ms','proposal_wall_ms','refine_ms','nms_ms','seed_count')}
        if case.get('truth'):
            row['accuracy'][method]=evaluate(group[0]['first']['matches'],case['truth'])
        row['overlap_diagnostics'][method]=[[p['overlap_metrics'] for p in r['patterns']] for r in records]
        if method!='previous_peak':
            expected_enabled=int(method.removesuffix('_profile')=='overlap')
            for record in [r for w in group for r in [w['first'],*w['runs']]]:
                for pattern in record['patterns']:
                    m=pattern['overlap_metrics']
                    valid=(m is not None and m['enabled']==expected_enabled and m['max_pending']<=2
                        and m['submitted_batches']==m['completed_batches'] and m['drained_batches']==0
                        and m['retained_gpu_bytes']<=pattern['metrics']['gpu_bytes'])
                    if expected_enabled:
                        valid &= m['submitted_maps']+m['fallback_maps']==pattern['metrics']['score_maps']
                    else:
                        valid &= m['submitted_maps']==m['fallback_maps']==0
                    row['overlap_valid'] &= bool(valid)
    for method in ('current_peak','overlap'):
        if method not in row['methods'] or method+'_profile' not in row['methods']:continue
        plain={b['block']:b for b in row['methods'][method]['blocks']}
        profiled={b['block']:b for b in row['methods'][method+'_profile']['blocks']}
        row['profile_overhead_percent_by_block'][method]=[
            dict(block=b,total_percent=100*(profiled[b]['total_ms']/plain[b]['total_ms']-1))
            for b in sorted(plain.keys()&profiled.keys())]
    return row


def replay(report,path,worker_root):
    groups={};hashes={}
    for entry in report['execution_order']:
        logical,digest,worker=load_worker(entry,path,worker_root)
        if logical in hashes:raise ValueError('Duplicate worker')
        hashes[logical]=digest
        job=Path(entry['job']) if worker_root is None else Path(worker_root)/str(worker_logical_path(entry['job']))
        if file_hash(job)!=entry['job_sha256']:raise ValueError('Job hash mismatch')
        groups.setdefault(entry['case'],[]).append(worker)
    return dict(report=Path(path).name,report_sha256=file_hash(path),raw_worker_sha256=hashes,
        complete=report['complete'],integrity_passed=report['integrity_passed'],design=report['design'],
        cases=[summarize_case(c['case'],groups[c['case']['name']]) for c in report['cases']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',default=str(coarse.real.ROOT/'build/coarse-overlap'))
    parser.add_argument('--baseline',default=str(coarse.real.ROOT/'build/many-objects-current'))
    parser.add_argument('--cases',default='controlled6,heldout12,8k_t256_angle180_3scales,small_fullres,many128_pose,many512_pose')
    parser.add_argument('--methods',default=','.join(METHODS))
    parser.add_argument('--blocks',type=int,default=2);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup-seconds',type=float,default=1.)
    parser.add_argument('--output',type=Path,default=coarse.real.ROOT/'build/coarse-overlap-results/formal.json')
    parser.add_argument('--worker',type=Path);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--summarize',type=Path);parser.add_argument('--worker-root',type=Path)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(coarse.worker(json.loads(args.worker.read_text()),search_factory=Search),allow_nan=False));return
    output=args.output.resolve()
    if args.summarize:
        if output==args.summarize.resolve():parser.error('Do not overwrite raw report')
        many.save(output,replay(json.loads(args.summarize.read_text()),args.summarize,args.worker_root));return
    methods=args.methods.split(',')
    if set(methods)-set(METHODS) or len(methods)!=len(set(methods)) or 'previous_peak' not in methods:
        parser.error('Unique supported methods including previous_peak required')
    if args.pilot:args.blocks,args.repeats,args.warmup_seconds=1,1,0.
    if args.blocks<1 or args.repeats<1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds<0:
        parser.error('Invalid repeat/block/warmup')
    inputs=output.parent/'inputs';jobs=output.parent/(output.stem+'-jobs');inputs.mkdir(parents=True,exist_ok=True);jobs.mkdir(exist_ok=True)
    names=args.cases.split(',');regular=[n for n in names if not n.startswith('many')];crowded=[n for n in names if n.startswith('many')]
    cases=coarse.make_cases(SimpleNamespace(cases=','.join(regular),suite='all'),inputs) if regular else []
    if crowded:cases+=many.make_cases(inputs,crowded)
    baseline=verify_baseline(args.baseline);sources=coarse.previous.source_manifest()
    for name in ('benchmark_coarse_overlap.py','benchmark_many_objects.py','many_object_scenes.py','many_object_accuracy.py',
                 'benchmark_coarse_next.py','benchmark_shared_fine.py','summarize_pipeline_stages.py'):
        path=Path(__file__).with_name(name).resolve();sources[str(path)]=file_hash(path)
    protocol=coarse.real.ROOT/'cpp-core/docs/coarse-overlap/protocol.json';sources[str(protocol)]=file_hash(protocol)
    libraries=dict(baseline['library_hashes'])
    for path in (core_library_path(args.build),coarse.coarse.library_path(args.build),pose_library_path(args.build)):
        libraries[str(Path(path).resolve())]=file_hash(path)
    input_hashes={case[k]:file_hash(case[k]) for case in cases for k in ('source','template','metadata') if k in case}
    report=dict(created=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),baseline_provenance=baseline,
        source_hashes=sources,library_hashes=libraries,input_hashes=input_hashes,execution_order=[],cases=[],complete=False,
        design=dict(methods=methods,blocks=args.blocks,repeats=args.repeats,warmup_seconds=args.warmup_seconds,pilot=args.pilot,
            pyramid_mode=1,batch_angles=4,fine_batch_size=512,peak_mode=1,ncc_mode=0,
            routes='Frozen4092754 peak tree; new current_peak overlap0; new overlap1; separate profile runs',
            timing='Fresh serial workers in forward/reverse blocks; prepare inside every call; current sync/overlap both collect new metrics',
            accuracy='Full ordered outputs/maps exact; GT kept in parent; independent 2px/1degree/2%scale gates',
            memory='RSS overlaps frame/pose buffers; async slot bytes must be included in coarse_gpu_bytes'))
    many.save(output,report)
    for case in cases:
        workers=[]
        for block in range(args.blocks):
            for method in methods if block%2==0 else methods[::-1]:
                job=dict(case=many.worker_case(case),method=method,block=block,build=str(Path(args.build).resolve()),
                    baseline=str(Path(args.baseline).resolve()),repeats=args.repeats,warmup_seconds=args.warmup_seconds,fine_batch_size=512)
                label=f'{case["name"]}-{method}-b{block}';job_path=jobs/(label+'.json');many.save(job_path,job)
                print('Running '+label,flush=True)
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(job_path)],capture_output=True,text=True)
                (jobs/(label+'.stderr.txt')).write_text(child.stderr)
                if child.returncode:raise RuntimeError(f'{label}: {child.stderr[-4000:]}')
                result=json.loads(child.stdout);result_path=jobs/(label+'.result.json');many.save(result_path,result);workers.append(result)
                report['execution_order'].append(dict(case=case['name'],method=method,block=block,job=str(job_path),
                    job_sha256=file_hash(job_path),result=str(result_path),result_sha256=file_hash(result_path)))
                many.save(output,report)
        row=summarize_case(case,workers);report['cases'].append(row);many.save(output,report)
        print(f'{case["name"]}: exact={row["exact_all_outputs"]} GT={row["truth_passed"]}',flush=True)
    hashes={**sources,**libraries,**input_hashes,baseline['manifest']:baseline['manifest_sha256']}
    report['sources_unchanged']=all(file_hash(p)==h for p,h in hashes.items());report['complete']=True
    report['integrity_passed']=report['sources_unchanged'] and all(c['exact_all_outputs'] and c['profile_valid'] and c['overlap_valid'] and c['work_counts_match_reference'] for c in report['cases'])
    report['detection_all_passed']=all(a['strict']['fp']==0 and a['strict']['fn']==0 for c in report['cases'] for a in c['accuracy'].values())
    many.save(output,report)
    if not report['integrity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
