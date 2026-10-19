"""Serial, frozen-control study of grayscale Metal NCC resident refinement fusion.

All routes use the identical split-submission Metal sparse frontend. Truth and
placement ROIs are evaluated only after timing and never reach native workers.
Frozen/current resident outputs must be exact; CPU and fused outputs require a
complete one-to-one pose/score comparison as well as independent GT accuracy.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

from benchmark_fixed_scale import ROOT, GIT, sha, save, distribution
from benchmark_sparse_orientation import BINARY_NAMES, proposal_accuracy
from benchmark_sparse_compact import COUNTS, verify_workers
from benchmark_sparse_backends import unchanged
from many_object_accuracy import evaluate, _maximum_matching

METHODS = ('frozen_metal', 'current_cpu', 'current_resident', 'current_fused')
BASELINE_COMMIT = '60a8936834da4565bc6da5447a9d4a5752d7da90'
GRADIENT_LOOKUP_BYTES = 511 * 511
GPU_COUNTS = ('command_buffers', 'cpu_waits', 'dispatches', 'peak_overflow_retries')
REFINEMENT_COUNTS = ('refinement_dispatches', 'refinement_command_buffers', 'refinement_cpu_waits', 'refinement_auxiliary_bytes')
REFINEMENT_TIMES = ('refinement_setup_ms', 'refinement_encode_ms', 'refinement_wait_ms', 'refinement_verify_ms', 'refinement_device_ms')
PAIR_TOLERANCE = dict(position_px=.25, angle_deg=.25, relative_scale=.002, score=1e-4)
GATES = ('all_workers_succeeded', 'exact_proposals_all', 'exact_resident_final_all',
         'fused_final_all', 'route_integrity_all', 'frontend_identical_all', 'normal_gt_all')


def proposal_payload(r):
    return dict(proposals=r['proposals'], counts={k:r['sparse'][k] for k in COUNTS})


def integer(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 and value==int(value)


def final_comparison(matches, reference, tolerance=PAIR_TOLERANCE):
    """Complete one-to-one assignment including pose and CPU-verified score.

    Candidate order may change with floating-point optimization. Matching by
    nearest position alone can hide duplicate outputs; maximum cardinality is
    used with all four bounds enforced on each accepted assignment edge.
    """
    adjacency=[]; errors={}
    for j,old in enumerate(reference):
        neighbors=[]
        for i,new in enumerate(matches):
            try:
                values=[float(p[k]) for p in (new,old) for k in ('pos_x','pos_y','angle','scale','score')]
                if not all(math.isfinite(v) for v in values) or new['scale']<=0 or old['scale']<=0:continue
                error=dict(position_px=math.hypot(new['pos_x']-old['pos_x'],new['pos_y']-old['pos_y']),
                    angle_deg=abs((new['angle']-old['angle']+180.)%360.-180.),
                    relative_scale=abs(new['scale']/old['scale']-1.),score=abs(new['score']-old['score']))
            except (KeyError,TypeError,ValueError,OverflowError):continue
            if all(error[k]<=tolerance[k] for k in tolerance):
                neighbors.append(i);errors[(j,i)]=error
        neighbors.sort(key=lambda i:(errors[(j,i)]['position_px'],errors[(j,i)]['angle_deg'],i))
        adjacency.append(neighbors)
    assignment=_maximum_matching(adjacency,len(matches))
    pairs=[dict(reference_index=j,match_index=i,**errors[(j,i)]) for j,i in enumerate(assignment) if i>=0]
    return dict(valid=len(matches)==len(reference)==len(pairs),reference_count=len(reference),match_count=len(matches),
        matched_count=len(pairs),tolerance=tolerance,pairs=pairs,
        max_errors={k:max((pair[k] for pair in pairs),default=None) for k in tolerance})


def route_valid(r,method):
    sparse=r['sparse'];active=sparse['feature_count']>0;has_poses=bool(r['proposals']);cpu=method=='current_cpu'
    frontend=(sparse['enabled'] and sparse['source_levels']==1 and sparse['response_mode']==1 and
        sparse.get('search_backend')==1 and sparse.get('gradient_mode')==1 and
        sparse.get('gradient_lookup_bytes')==(GRADIENT_LOOKUP_BYTES if active else 0) and
        sparse.get('metal_submission_mode')==0 and all(integer(sparse.get(k)) for k in GPU_COUNTS) and
        (not sparse['windows'] or (sparse.get('gpu_buffer_bytes',0)>0 and sparse['command_buffers']>0)) and
        r['frame_host_bytes']==0 and all(p['scale']==1 for key in ('matches','proposals') for p in r[key]))
    backend=(not has_poses or r['selected_refinement_backend']==(0 if cpu else 1)) and (r['pose_gpu_bytes']==0 if cpu else (not has_poses or r['pose_gpu_bytes']>0))
    if method=='frozen_metal':return frontend and backend
    metrics=(integer(r.get('refinement_mode')) and r.get('refinement_mode')==int(method=='current_fused') and all(integer(r.get(k)) for k in REFINEMENT_COUNTS) and
        all(isinstance(r.get(k),(int,float)) and not isinstance(r[k],bool) and math.isfinite(r[k]) and
            (r[k]>=0 or (k=='refinement_device_ms' and r[k]==-1)) for k in REFINEMENT_TIMES))
    if not metrics:return False
    d,c,w=[r[k] for k in REFINEMENT_COUNTS[:3]]
    execution=(all(r[k]==0 for k in REFINEMENT_COUNTS) if cpu else
        ((d==c==w and c>0) if method=='current_fused' else (d>c==w and c>0)) if has_poses else d==c==w==0)
    if cpu:execution=execution and all(r[k]==0 for k in REFINEMENT_TIMES[:-1]) and r['refinement_device_ms']==-1
    return frontend and backend and execution


def frontend_checks(workers):
    checks=[]
    for block in sorted({b for method,b in workers}):
        old=workers.get(('frozen_metal',block))
        for method in METHODS[1:]:
            current=workers.get((method,block))
            if current is None or old is None:
                checks.append(dict(method=method,block=block,valid=False,reason='missing worker'));continue
            calls=[current['first'],*current['runs']];refs=[old['first'],*old['runs']]
            if len(calls)!=len(refs):
                checks.append(dict(method=method,block=block,valid=False,reason='different call counts'));continue
            for i,(call,ref) in enumerate(zip(calls,refs)):
                s=call['sparse'];r=ref['sparse'];present=all(k in s and k in r for k in GPU_COUNTS)
                delta={k:r[k]-s[k] for k in GPU_COUNTS} if present else {}
                checks.append(dict(method=method,block=block,call='first' if i==0 else 'run'+str(i),
                    valid=present and all(v==0 for v in delta.values()),counter_reduction=delta))
    return checks


def summarize(report,base):
    protocol=report['protocol'];groups=[]
    for case in protocol['cases']:
        entries=[e for e in report['execution_order'] if e['case']==case['name']]
        ref=next((json.loads((base/e['result']).read_text())['first'] for e in entries if e['method']=='frozen_metal' and e['status']=='ok'),None)
        methods={};block_workers={}
        for method in METHODS:
            workers=[(e,json.loads((base/e['result']).read_text())) for e in entries if e['method']==method and e['status']=='ok']
            if not workers:methods[method]=dict(status='missing');continue
            block_workers.update({(method,e['block']):w for e,w in workers})
            calls=[r for _,w in workers for r in [w['first'],*w['runs']]];runs=[r for _,w in workers for r in w['runs']]
            acc=[evaluate(r['matches'],case['truth']) for r in calls] if case['truth'] is not None else []
            numeric=[k for k,v in runs[0]['sparse'].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
            comparisons=[final_comparison(r['matches'],ref['matches']) for r in calls] if ref is not None else []
            metrics=[k for k in (*REFINEMENT_COUNTS,*REFINEMENT_TIMES,'pose_host_bytes','pose_gpu_bytes') if all(k in r for r in runs)]
            methods[method]=dict(status='complete' if {e['block'] for e,_ in workers}==set(range(protocol['blocks'])) and len(workers)==protocol['blocks'] and all(len(w['runs'])==protocol['repeats'] for _,w in workers) else 'partial',
                total_ms=distribution([r['total_ms'] for r in runs]),construct_ms=[w['construct_ms'] for _,w in workers],first_ms=[w['first']['total_ms'] for _,w in workers],
                peak_rss_bytes=[w['final_memory']['peak_rss_bytes'] for _,w in workers],
                stage_medians={k:statistics.median(r[k] for r in runs) for k in ('prepare_ms','proposal_ms','refine_ms','nms_ms')},
                sparse_medians={k:statistics.median(r['sparse'][k] for r in runs) for k in numeric},
                refinement_medians={k:statistics.median(r[k] for r in runs) for k in metrics},
                exact_proposals=ref is not None and all(proposal_payload(r)==proposal_payload(ref) for r in calls),
                exact_final=ref is not None and all(r['matches']==ref['matches'] for r in calls),
                paired_final=bool(comparisons) and all(c['valid'] for c in comparisons),
                first_final_comparison=comparisons[0] if comparisons else None,
                pair_max_errors={k:max((c['max_errors'][k] for c in comparisons if c['max_errors'][k] is not None),default=None) for k in PAIR_TOLERANCE},
                route_valid=all(route_valid(r,method) for r in calls),own_repeat_exact=all(w['exact_repeat'] for _,w in workers),
                normal_gt_all=None if not acc else all(a['strict']['fp']==a['strict']['fn']==0 for a in acc),
                first_accuracy=acc[0] if acc else None,
                first_proposal_coverage=proposal_accuracy(calls[0]['proposals'],case['truth'],protocol['proposal_basin']) if acc else None,
                first_outputs=calls[0]['matches'],
                blocks=[dict(block=e['block'],median_ms=statistics.median(r['total_ms'] for r in w['runs']),refine_median_ms=statistics.median(r['refine_ms'] for r in w['runs']),peak_rss_bytes=w['final_memory']['peak_rss_bytes']) for e,w in workers])
        pairs=[]
        for method,reference,attribution in (('current_fused','current_resident','fused_vs_resident'),
                ('current_resident','frozen_metal','resident_vs_frozen'),('current_fused','current_cpu','fused_vs_cpu'),('current_fused','frozen_metal','fused_vs_frozen')):
            if methods[method]['status']!='complete' or methods[reference]['status']!='complete':continue
            for b in methods[method]['blocks']:
                old=next(x for x in methods[reference]['blocks'] if x['block']==b['block'])
                pairs.append(dict(method=method,reference=reference,attribution=attribution,block=b['block'],
                    time_ratio=b['median_ms']/old['median_ms'],speedup=old['median_ms']/b['median_ms'],
                    refinement_speedup=old['refine_median_ms']/b['refine_median_ms'] if b['refine_median_ms'] else None,
                    speed_gate=b['median_ms']<=old['median_ms']+max(2.,.1*old['median_ms']),
                    rss_delta_bytes=b['peak_rss_bytes']-old['peak_rss_bytes'],
                    memory_gate=b['peak_rss_bytes']<=old['peak_rss_bytes']+max(32*2**20,.1*old['peak_rss_bytes'])))
        checks=frontend_checks(block_workers)
        groups.append(dict(name=case['name'],validity=case['validity'],methods=methods,paired_comparison=pairs,
            frontend=checks,frontend_identical=bool(checks) and all(c['valid'] for c in checks)))
    allmethods=[m for c in groups for m in c['methods'].values()];expected=len(protocol['cases'])*len(METHODS)*protocol['blocks']
    return dict(cases=groups,workers=len(report['execution_order']),expected_workers=expected,
        measured_calls=sum(len(json.loads((base/e['result']).read_text())['runs']) for e in report['execution_order'] if e['status']=='ok'),
        all_workers_succeeded=len(report['execution_order'])==expected and all(e['status']=='ok' for e in report['execution_order']),
        exact_proposals_all=all(m.get('status')=='complete' and m['exact_proposals'] and m['own_repeat_exact'] for m in allmethods),
        exact_resident_final_all=all(c['methods'][k].get('status')=='complete' and c['methods'][k]['exact_final'] for c in groups for k in ('frozen_metal','current_resident')),
        paired_final_all=all(m.get('status')=='complete' and m['paired_final'] for m in allmethods),
        fused_final_all=all(c['methods']['current_fused'].get('status')=='complete' and c['methods']['current_fused']['paired_final'] for c in groups),
        cpu_final_comparison_all=all(c['methods']['current_cpu'].get('status')=='complete' and c['methods']['current_cpu']['paired_final'] for c in groups),
        route_integrity_all=all(m.get('status')=='complete' and m['route_valid'] for m in allmethods),
        frontend_identical_all=all(c['frontend_identical'] and all(m.get('status')=='complete' for m in c['methods'].values()) for c in groups),
        normal_gt_all=all(m.get('normal_gt_all') for c in groups if c['validity']=='fixed_scale_gt' for m in c['methods'].values()))


def prepare(args,path):
    prior=json.loads(args.prior.read_text());cases=json.loads(json.dumps(prior['cases']))
    if args.cases:
        names=args.cases.split(',')
        if len(names)!=len(set(names)) or set(names)-{c['name'] for c in cases}:raise ValueError('Invalid cases')
        cases=[c for c in cases if c['name'] in names]
    inputs=path.parent/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    for c in cases:
        c['held_out']=False
        for k in ('source','template','metadata'):
            old=Path(c[k]);target=inputs/old.name
            if sha(old)!=prior['input_sha256'][str(old)]:raise ValueError('Changed prior fixture')
            if not target.exists():shutil.copyfile(old,target)
            if sha(target)!=sha(old):raise ValueError('Changed copied fixture')
            c[k]=str(target.resolve())
    builds={m:str((args.baseline if m=='frozen_metal' else args.build).resolve()) for m in METHODS}
    binaries={str((Path(b)/'experiments'/n).resolve()):sha(Path(b)/'experiments'/n) for b in set(builds.values()) for n in BINARY_NAMES}
    for n in BINARY_NAMES:
        p=str((args.baseline/'experiments'/n).resolve())
        expected=str((Path(prior['builds']['split'])/'experiments'/n).resolve())
        if binaries[p]!=prior['binary_sha256'][expected]:raise ValueError('Frozen stage21 Metal binary changed')
    sources=[]
    for d in ('src','include','experiments','tools','tests'):
        sources += [p for p in (ROOT/'cpp-core'/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and
            (p.suffix in ('.cpp','.hpp','.h','.mm','.metal','.in','.py') or p.name=='CMakeLists.txt')]
    sources += [ROOT/'cpp-core/CMakeLists.txt',ROOT/'macos-cli/main.cpp',ROOT/'macos-cli/mainwindow.cpp',ROOT/'macos-cli/mainwindow.h']
    snapshot=path.parent/'source-snapshot'
    if snapshot.exists():raise ValueError('Existing source snapshot')
    for p in sources:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(args.prior,path.parent/'prior-protocol.json')
    archive=path.parent/'baseline-source.tar'
    with archive.open('wb') as dest:
        subprocess.run([GIT,'archive',BASELINE_COMMIT,'cpp-core/src','cpp-core/include','cpp-core/experiments','cpp-core/tests','cpp-core/tools','cpp-core/CMakeLists.txt','macos-cli/main.cpp','macos-cli/mainwindow.cpp','macos-cli/mainwindow.h'],cwd=ROOT,stdout=dest,check=True)
    protocol=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        implementation_base=subprocess.check_output([GIT,'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),baseline_commit=BASELINE_COMMIT,
        cases=cases,methods=list(METHODS),builds=builds,blocks=1 if args.pilot else 2,repeats=1 if args.pilot else 3,
        warmup_seconds=0. if args.pilot else 1.,timeout_seconds=180.,pilot=args.pilot,
        shared_settings=prior['shared_settings'],sparse_settings=prior['sparse_settings'],proposal_basin=prior['proposal_basin'],
        scope='Grayscale only. Frozen stage21 resident Metal refinement, current CPU refinement, current resident Metal refinement, current fused Metal refinement. Every route uses identical compact sparse Metal frontend with split submission mode 0, fixtures, candidate rules, rescue settings and CPU final verification. CPU candidate selection and rescue orchestration remain; this is not all-GPU search.',
        attribution='Paired per-block fused / current resident isolates refinement fusion. Current resident / frozen resident is the exact runtime regression control; fused / CPU is the alternative hybrid comparison. Do not mix old stage21 CPU-refinement results with these new frozen resident measurements.',
        timing='Serial fresh workers, forward/reverse route order. Source preparation, refinement, CPU verification and rescue included in total. Decode/JSON, construction and first call separate. Refinement GPU counters aggregate initial and rescue batches. Device interval is nested inside wait; do not add them. RSS is process peak; GPU capacities overlap RSS and must not be added.',
        gates=dict(exactness='Ordered proposals and integer sparse work exact vs frozen on all four routes; frozen/current resident final results exact. CPU/fused use complete one-to-one pose and score tolerance in addition to independent GT. CPU cross-solver tolerance is reported separately, not a fused-regression gate. All normal GT remains mandatory. Negative scale and unlabeled photo retained.',
            pair_tolerance=PAIR_TOLERANCE,frontend='GPU sparse command buffers, CPU waits, dispatches and overflow retries identical for every same-block first/measured call. Current CPU has zero refinement GPU counts; current fused has one dispatch per command buffer; current resident has multiple dispatches per command buffer.',
            speed='Each paired block versus its named reference <= max(10%,2ms) increase; report gate failures, not an unconditional win.',memory='Each paired block versus its named reference <= max(10%,32MiB) increase. Report failures; buffers and RSS do not add.'),
        source_sha256={str(p.resolve()):sha(p) for p in sources},binary_sha256=binaries,
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template','metadata')},
        snapshot_sha256={str(p.resolve()):sha(p) for p in [archive,path.parent/'prior-protocol.json',*[p for p in snapshot.rglob('*') if p.is_file()]]})
    save(path,protocol);return protocol


def run_worker(protocol,c,method,block,out):
    settings=dict(protocol['shared_settings']);settings.pop('optimize_scale')
    job=dict(settings,**protocol['sparse_settings'],source=c['source'],template=c['template'],angle=c['angle'],
        backend=0 if method=='current_cpu' else 1,proposal_method='sparse_orientation',sparse_response_mode=1,
        repeats=protocol['repeats'],warmup_seconds=protocol['warmup_seconds'])
    job['sparse_search_backend']=1
    job['sparse_cpu_gradient_mode']=0
    job['sparse_metal_submission_mode']=0
    if method!='frozen_metal':job['pose_resident_mode']=int(method=='current_fused')
    p=out.parent/'jobs'/f'{c["name"]}-{method}-block{block}.json';save(p,job)
    e=dict(case=c['name'],method=method,block=block,job=str(p.relative_to(out.parent)),job_sha256=sha(p))
    print('Running '+p.stem,flush=True)
    try:
        child=subprocess.run([str(Path(protocol['builds'][method])/'experiments/fpm_native_pipeline_runner'),str(p)],capture_output=True,text=True,timeout=protocol['timeout_seconds'])
        stdout,stderr=child.stdout,child.stderr;e.update(status='ok' if child.returncode==0 else 'process_error',returncode=child.returncode)
    except subprocess.TimeoutExpired as error:
        stdout,stderr=error.stdout or '',error.stderr or '';e['status']='timeout'
        if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
        if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
    for k,v in (('stdout',stdout),('stderr',stderr)):
        dest=p.with_suffix('.'+k+'.txt');dest.write_text(v);e[k]=str(dest.relative_to(out.parent));e[k+'_sha256']=sha(dest)
    if e['status']=='ok':
        try:
            raw=json.loads(stdout)
            if len(raw['runs'])!=protocol['repeats'] or not raw['exact_repeat']:raise ValueError('Missing runs or nonidentical repeats')
            dest=p.with_suffix('.result.json');save(dest,raw);e.update(result=str(dest.relative_to(out.parent)),result_sha256=sha(dest))
        except (ValueError,KeyError,TypeError) as error:e.update(status='invalid_output',error=str(error))
    return e


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/pose-fused')
    p.add_argument('--baseline',type=Path,default=ROOT/'build/sparse-metal-merge');p.add_argument('--prior',type=Path,default=ROOT/'cpp-core/docs/sparse-metal-merge/protocol.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases');p.add_argument('--pilot',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize',type=Path)
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Preserve previous evidence; select a new output path')
    if args.summarize:
        report=json.loads(args.summarize.read_text());verify_workers(report,args.summarize.parent);save(out,summarize(report,args.summarize.parent));return
    protocolpath=out.with_suffix('.protocol.json')
    if protocolpath.exists():
        protocol=json.loads(protocolpath.read_text())
        if protocol['builds']['current_resident']!=str(args.build.resolve()) or protocol['builds']['frozen_metal']!=str(args.baseline.resolve()) or protocol['pilot']!=args.pilot:raise ValueError('Prepared settings differ')
        if args.cases and set(args.cases.split(','))!={c['name'] for c in protocol['cases']}:raise ValueError('Prepared cases differ')
    else:protocol=prepare(args,protocolpath)
    if args.prepare_only:print(protocolpath);return
    if not unchanged(protocol):raise ValueError('Changed inputs/sources/binaries/snapshot')
    report=dict(protocol=protocol,protocol_sha256=sha(protocolpath),execution_order=[],complete=False);save(out,report)
    for c in protocol['cases']:
        for block in range(protocol['blocks']):
            for m in METHODS if block%2==0 else METHODS[::-1]:
                report['execution_order'].append(run_worker(protocol,c,m,block,out));save(out,report)
    report.update(complete=True,unchanged=unchanged(protocol));report['summary']=summarize(report,out.parent);save(out,report)
    if not report['unchanged'] or not all(report['summary'][k] for k in GATES):raise SystemExit(1)

if __name__=='__main__':main()
