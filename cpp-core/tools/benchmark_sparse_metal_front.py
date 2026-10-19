"""Controlled full-image sparse Metal frontend study, with a frozen CPU baseline.

No GT/count/ROI goes to the native worker. Exact proposals and integer sparse
work are required across all routes; final CPU/GPU refinement floats use GT.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tarfile

from benchmark_fixed_scale import ROOT, GIT, sha, save, distribution
from benchmark_sparse_orientation import BINARY_NAMES, proposal_accuracy
from benchmark_sparse_compact import COUNTS, verify_workers
from benchmark_sparse_backends import unchanged
from many_object_accuracy import evaluate

METHODS = ('frozen_cpu', 'cpu', 'metal_cpu', 'metal_metal')


def proposal_payload(r):
    return dict(proposals=r['proposals'], counts={k:r['sparse'][k] for k in COUNTS})


def route_valid(r, method):
    sparse=r['sparse'];metal=method.startswith('metal_');refine=int(method=='metal_metal')
    return (sparse['enabled'] and sparse['source_levels']==1 and sparse['response_mode']==1 and
        sparse.get('search_backend',0)==int(metal) and r['frame_host_bytes']==0 and
        (not r['proposals'] or r['selected_refinement_backend']==refine) and
        (refine!=0 or r['pose_gpu_bytes']==0) and
        (not sparse['windows'] or not metal or (sparse.get('gpu_buffer_bytes',0)>0 and sparse.get('command_buffers',0)>0)) and
        all(p['scale']==1 for key in ('matches','proposals') for p in r[key]))


def summarize(report, base):
    protocol=report['protocol'];groups=[]
    for case in protocol['cases']:
        entries=[e for e in report['execution_order'] if e['case']==case['name']]
        ref=next((json.loads((base/e['result']).read_text())['first'] for e in entries if e['method']=='frozen_cpu' and e['status']=='ok'),None)
        methods={}
        for method in METHODS:
            workers=[(e,json.loads((base/e['result']).read_text())) for e in entries if e['method']==method and e['status']=='ok']
            if not workers:methods[method]=dict(status='missing');continue
            calls=[r for _,w in workers for r in [w['first'],*w['runs']]];runs=[r for _,w in workers for r in w['runs']]
            acc=[evaluate(r['matches'],case['truth']) for r in calls] if case['truth'] is not None else []
            numeric=[k for k,v in runs[0]['sparse'].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
            methods[method]=dict(status='complete' if {e['block'] for e,_ in workers}==set(range(protocol['blocks'])) and len(workers)==protocol['blocks'] else 'partial',
                total_ms=distribution([r['total_ms'] for r in runs]),construct_ms=[w['construct_ms'] for _,w in workers],first_ms=[w['first']['total_ms'] for _,w in workers],
                peak_rss_bytes=[w['final_memory']['peak_rss_bytes'] for _,w in workers],
                stage_medians={k:statistics.median(r[k] for r in runs) for k in ('prepare_ms','proposal_ms','refine_ms','nms_ms')},
                sparse_medians={k:statistics.median(r['sparse'][k] for r in runs) for k in numeric},
                exact_proposals=ref is not None and all(proposal_payload(r)==proposal_payload(ref) for r in calls),
                exact_cpu_final=None if method=='metal_metal' else ref is not None and all(r['matches']==ref['matches'] for r in calls),
                route_valid=all(route_valid(r,method) for r in calls),own_repeat_exact=all(w['exact_repeat'] for _,w in workers),
                normal_gt_all=None if not acc else all(a['strict']['fp']==a['strict']['fn']==0 for a in acc),
                first_accuracy=acc[0] if acc else None,
                first_proposal_coverage=proposal_accuracy(calls[0]['proposals'],case['truth'],protocol['proposal_basin']) if acc else None,
                first_outputs=calls[0]['matches'],
                blocks=[dict(block=e['block'],median_ms=statistics.median(r['total_ms'] for r in w['runs']),peak_rss_bytes=w['final_memory']['peak_rss_bytes']) for e,w in workers])
        pairs=[]
        for method in ('metal_cpu','metal_metal'):
            if methods[method]['status']!='complete' or methods['cpu']['status']!='complete':continue
            for b in methods[method]['blocks']:
                old=next(x for x in methods['cpu']['blocks'] if x['block']==b['block'])
                pairs.append(dict(method=method,block=b['block'],time_ratio=b['median_ms']/old['median_ms'],
                    speed_gate=b['median_ms']<=old['median_ms']+max(2.,.1*old['median_ms']),
                    rss_delta_bytes=b['peak_rss_bytes']-old['peak_rss_bytes'],
                    memory_gate=b['peak_rss_bytes']<=old['peak_rss_bytes']+max(32*2**20,.1*old['peak_rss_bytes'])))
        groups.append(dict(name=case['name'],validity=case['validity'],methods=methods,paired_comparison=pairs))
    allmethods=[m for c in groups for m in c['methods'].values()];expected=len(protocol['cases'])*4*protocol['blocks']
    return dict(cases=groups,workers=len(report['execution_order']),expected_workers=expected,
        measured_calls=sum(len(json.loads((base/e['result']).read_text())['runs']) for e in report['execution_order'] if e['status']=='ok'),
        all_workers_succeeded=len(report['execution_order'])==expected and all(e['status']=='ok' for e in report['execution_order']),
        exact_proposals_all=all(m.get('status')=='complete' and m['exact_proposals'] and m['own_repeat_exact'] for m in allmethods),
        exact_cpu_final_all=all(m.get('status')=='complete' and m['exact_cpu_final'] for c in groups for name,m in c['methods'].items() if name!='metal_metal'),
        route_integrity_all=all(m.get('status')=='complete' and m['route_valid'] for m in allmethods),
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
    builds={m:str((args.baseline if m=='frozen_cpu' else args.build).resolve()) for m in METHODS}
    binaries={str((Path(b)/'experiments'/n).resolve()):sha(Path(b)/'experiments'/n) for b in set(builds.values()) for n in BINARY_NAMES}
    for n in BINARY_NAMES:
        p=str((args.baseline/'experiments'/n).resolve())
        if binaries[p]!=prior['binary_sha256'][p]:raise ValueError('Frozen binary changed')
    sources=[]
    for d in ('src','include','experiments','tools','tests'):
        sources += [p for p in (ROOT/'cpp-core'/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and
            (p.suffix in ('.cpp','.hpp','.h','.mm','.metal','.in','.py') or p.name=='CMakeLists.txt')]
    sources += [ROOT/'cpp-core/CMakeLists.txt',ROOT/'macos-cli/mainwindow.cpp',ROOT/'macos-cli/mainwindow.h']
    snapshot=path.parent/'source-snapshot'
    if snapshot.exists():raise ValueError('Existing source snapshot')
    for p in sources:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(args.prior,path.parent/'prior-protocol.json')
    archive=path.parent/'baseline-source.tar'
    with archive.open('wb') as dest:
        subprocess.run([GIT,'archive',prior['implementation_commit'],'cpp-core/src','cpp-core/include','cpp-core/experiments','cpp-core/tests','cpp-core/tools','cpp-core/CMakeLists.txt','macos-cli/mainwindow.cpp','macos-cli/mainwindow.h'],cwd=ROOT,stdout=dest,check=True)
    protocol=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        implementation_base=subprocess.check_output([GIT,'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),baseline_commit=prior['implementation_commit'],
        cases=cases,methods=list(METHODS),builds=builds,blocks=1 if args.pilot else 2,repeats=1 if args.pilot else 3,
        warmup_seconds=0. if args.pilot else 1.,timeout_seconds=180.,pilot=args.pilot,
        shared_settings=prior['shared_settings'],sparse_settings=prior['sparse_settings'],proposal_basin=prior['proposal_basin'],
        scope='Frozen CPU, current CPU, Metal frontend plus CPU refinement, Metal frontend plus Metal refinement. Compact mask throughout. GPU handles gradient/spread/scoring/local peaks; CPU deterministic selection remains. No learned model. Same templates and GT as stage18.',
        timing='Serial fresh workers, forward/reverse route order. Per-call source preparation included in proposal and total. Decode/JSON, construction and first call separate. GPU device interval nested inside wait; do not add them.',
        gates=dict(exactness='Ordered proposals and integer sparse work exact vs frozen CPU. Final outputs exact when refinement CPU; independently check GT for Metal refinement. Keep known negative/proposal failures.',
            speed='Per block vs current CPU <= max(10%,2ms) increase; not an unconditional win.',memory='Per block vs current CPU <= max(10%,32MiB) increase. Report failures; buffers and RSS do not add.'),
        source_sha256={str(p.resolve()):sha(p) for p in sources},binary_sha256=binaries,
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template','metadata')},
        snapshot_sha256={str(p.resolve()):sha(p) for p in [archive,path.parent/'prior-protocol.json',*[p for p in snapshot.rglob('*') if p.is_file()]]})
    save(path,protocol);return protocol


def run_worker(protocol,c,method,block,out):
    settings=dict(protocol['shared_settings']);settings.pop('optimize_scale')
    job=dict(settings,**protocol['sparse_settings'],source=c['source'],template=c['template'],angle=c['angle'],
        backend=int(method=='metal_metal'),proposal_method='sparse_orientation',sparse_response_mode=1,
        repeats=protocol['repeats'],warmup_seconds=protocol['warmup_seconds'])
    if method!='frozen_cpu':job['sparse_search_backend']=int(method.startswith('metal_'))
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/sparse-metal')
    p.add_argument('--baseline',type=Path,default=ROOT/'build/sparse-compact');p.add_argument('--prior',type=Path,default=ROOT/'cpp-core/docs/sparse-backends/protocol.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases');p.add_argument('--pilot',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize',type=Path)
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Preserve previous evidence; select a new output path')
    if args.summarize:
        report=json.loads(args.summarize.read_text());verify_workers(report,args.summarize.parent);save(out,summarize(report,args.summarize.parent));return
    protocolpath=out.with_suffix('.protocol.json')
    if protocolpath.exists():
        protocol=json.loads(protocolpath.read_text())
        if protocol['builds']['cpu']!=str(args.build.resolve()) or protocol['builds']['frozen_cpu']!=str(args.baseline.resolve()) or protocol['pilot']!=args.pilot:raise ValueError('Prepared settings differ')
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
    if not report['unchanged'] or not all(report['summary'][k] for k in ('all_workers_succeeded','exact_proposals_all','exact_cpu_final_all','route_integrity_all','normal_gt_all')):raise SystemExit(1)

if __name__=='__main__':main()
