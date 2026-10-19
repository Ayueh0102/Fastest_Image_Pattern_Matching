"""Compare all-CPU with CPU proposals + Metal refinement on frozen runtime.

Both eight-plane and compact representations are controlled separately. CPU
proposal workload is identical; CPU/Metal final pose values may differ, so raw
proposals/work must match exactly but final accuracy uses independent GT.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

from benchmark_fixed_scale import ROOT, GIT, sha, save, distribution
from benchmark_sparse_compact import run_worker, verify_workers, COUNTS
from benchmark_sparse_orientation import proposal_accuracy, BINARY_NAMES, unchanged as live_unchanged
from many_object_accuracy import evaluate


def unchanged(protocol):
    return live_unchanged(protocol) and all(Path(p).is_file() and sha(p)==h for p,h in protocol['snapshot_sha256'].items())


def prepare(args,path):
    prior=json.loads(args.prior.read_text());cases=json.loads(json.dumps(prior['cases']))
    if args.cases:
        names=args.cases.split(',')
        if len(set(names))!=len(names) or set(names)-{c['name'] for c in cases}:raise ValueError('Invalid case selection')
        cases=[c for c in cases if c['name'] in names]
    inputs=path.parent/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    for case in cases:
        case['backends']=[0,1];case['held_out']=False
        for key in ('source','template','metadata'):
            old=Path(case[key]);expected=prior['input_sha256'].get(str(old));target=inputs/old.name
            if expected is None or sha(old)!=expected:raise ValueError('Changed previous input '+str(old))
            if not target.exists():shutil.copyfile(old,target)
            if sha(target)!=expected:raise ValueError('Changed copied input '+str(target))
            case[key]=str(target.resolve())
    binaries={str((args.build/'experiments'/name).resolve()):sha(args.build/'experiments'/name) for name in BINARY_NAMES}
    for file,digest in binaries.items():
        if prior['binary_sha256'].get(file)!=digest:raise ValueError('Runtime differs from frozen compact study')
    sources=[]
    for directory in ('src','include','experiments','tests','tools'):
        sources += [p for p in (ROOT/'cpp-core'/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and
            (p.suffix in ('.cpp','.hpp','.h','.mm','.metal','.in','.py') or p.name=='CMakeLists.txt')]
    sources += [ROOT/'cpp-core/CMakeLists.txt',ROOT/'macos-cli/mainwindow.cpp',ROOT/'macos-cli/mainwindow.h']
    for file in sources:
        if file.parent.name in ('src','experiments') and str(file) in prior['source_sha256'] and sha(file)!=prior['source_sha256'][str(file)]:
            raise ValueError('Runtime source changed since frozen study '+str(file))
    snapshot=path.parent/'source-snapshot'
    if snapshot.exists():raise ValueError('Existing snapshot; choose new output directory')
    for source in sources:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(args.prior,path.parent/'prior-protocol.json')
    shared=dict(prior['shared_settings']);shared['cpu_threads']=8
    protocol=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        implementation_commit=subprocess.check_output([GIT,'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),runtime_unchanged=True,
        cases=cases,blocks=2,repeats=3,warmup_seconds=1.,timeout_seconds=120.,
        builds=dict(planes=str(args.build.resolve()),compact=str(args.build.resolve())),methods=['planes','compact'],backends=[0,1],
        shared_settings=shared,sparse_settings=prior['sparse_settings'],proposal_basin=prior['proposal_basin'],
        timing='Fresh serial processes. Two reverse-order blocks; three measured calls each after >=1s warmup. Session setup/first call separate. Each algorithm call includes preparation/proposals/refinement/NMS; decoding and JSON excluded.',
        scope='All CPU vs identical CPU proposals plus Metal resident refinement. Same source, template, angles, proposal budget, threshold and rescue policy. No GT or object count passed to worker. This is not a fully GPU-resident matcher.',
        accuracy='Raw ordered proposals and sparse work counts exactly equal across backend. Final CPU/GPU float poses may differ; both independently evaluated at2px/1deg/scale ratio[.98,1.02]. Own repeated outputs exact. Preserve negative/unlabelled cases.',
        memory='Peak process RSS and retained frame/pose capacities kept separately. CPU backend must report zero poseGPU bytes; shared GPU counters and RSS must not be added.',
        decision='Report each paired block and medians. A winner on these fixed templates is not a universal routing threshold or a candidate-count crossover curve.',
        source_sha256={str(p.resolve()):sha(p) for p in sources},binary_sha256=binaries,
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template','metadata')},
        snapshot_sha256={str(p.resolve()):sha(p) for p in [path.parent/'prior-protocol.json',*[p for p in snapshot.rglob('*') if p.is_file()]]})
    save(path,protocol);return protocol


def proposal_payload(record):
    return dict(proposals=record['proposals'],work={key:record['sparse'][key] for key in COUNTS})


def summarize(report,base):
    protocol=report['protocol'];groups=[]
    for case in protocol['cases']:
        for mode in protocol['methods']:
            entries=[e for e in report['execution_order'] if e['case']==case['name'] and e['method']==mode]
            refs=[json.loads((base/e['result']).read_text()) for e in entries if e['status']=='ok' and e['backend']==0]
            reference=proposal_payload(refs[0]['first']) if refs else None;methods={}
            for backend,name in ((0,'cpu'),(1,'metal')):
                workers=[(e,json.loads((base/e['result']).read_text())) for e in entries if e['status']=='ok' and e['backend']==backend]
                if not workers:methods[name]=dict(status='missing');continue
                runs=[r for _,w in workers for r in w['runs']];calls=[r for _,w in workers for r in [w['first'],*w['runs']]]
                accuracy=[evaluate(r['matches'],case['truth']) for r in calls] if case['truth'] is not None else []
                coverage=proposal_accuracy(calls[0]['proposals'],case['truth'],protocol['proposal_basin']) if case['truth'] is not None else None
                area=case['scene_shape'][0]*case['scene_shape'][1]
                route=all(r['sparse']['response_mode']==int(mode=='compact') and r['sparse']['source_bytes']==area*(1 if mode=='compact' else 8) and
                    r['frame_host_bytes']==0 and r['sparse']['source_levels']==1 and r['sparse']['enabled'] and
                    (not r['proposals'] or r['selected_refinement_backend']==backend) and (backend!=0 or r['pose_gpu_bytes']==0) for r in calls)
                methods[name]=dict(status='complete' if len(workers)==protocol['blocks'] else 'partial',
                    total_ms=distribution([r['total_ms'] for r in runs]),construct_ms=[w['construct_ms'] for _,w in workers],first_ms=[w['first']['total_ms'] for _,w in workers],
                    peak_rss_bytes=[w['final_memory']['peak_rss_bytes'] for _,w in workers],current_rss_bytes=[w['final_memory']['current_rss_bytes'] for _,w in workers],
                    stage_medians={key:statistics.median(r[key] for r in runs) for key in ('prepare_ms','proposal_ms','refine_ms','nms_ms')},
                    counts=calls[0]['sparse'],pose_host_bytes=statistics.median(r['pose_host_bytes'] for r in runs),pose_gpu_bytes=statistics.median(r['pose_gpu_bytes'] for r in runs),
                    rescued_seeds=[r['rescued_seeds'] for r in runs],route_valid=route,proposals_equal=reference is not None and all(proposal_payload(r)==reference for r in calls),
                    own_repeat_exact=all(w['exact_repeat'] for _,w in workers),fixed_scale=all(m['scale']==1 for r in calls for key in ('matches','proposals') for m in r[key]),
                    final_gt_all=None if not accuracy else all(a['strict']['fp']==a['strict']['fn']==0 for a in accuracy),
                    first_accuracy=accuracy[0] if accuracy else None,first_proposal_coverage=coverage,first_outputs=calls[0]['matches'],
                    blocks=[dict(block=e['block'],median_ms=statistics.median(r['total_ms'] for r in w['runs']),refine_ms=statistics.median(r['refine_ms'] for r in w['runs']),peak_rss_bytes=w['final_memory']['peak_rss_bytes']) for e,w in workers])
            pairs=[]
            if all(m['status']=='complete' for m in methods.values()):
                for c in methods['cpu']['blocks']:
                    g=next(b for b in methods['metal']['blocks'] if b['block']==c['block'])
                    pairs.append(dict(block=c['block'],metal_over_cpu_time=g['median_ms']/c['median_ms'],cpu_faster=c['median_ms']<g['median_ms'],cpu_minus_metal_rss_bytes=c['peak_rss_bytes']-g['peak_rss_bytes']))
            groups.append(dict(name=case['name'],representation=mode,validity=case['validity'],methods=methods,paired_comparison=pairs))
    allmethods=[m for g in groups for m in g['methods'].values()];expected=len(protocol['cases'])*len(protocol['methods'])*2*protocol['blocks']
    return dict(cases=groups,workers=len(report['execution_order']),expected_workers=expected,
        measured_calls=sum(len(json.loads((base/e['result']).read_text())['runs']) for e in report['execution_order'] if e['status']=='ok'),
        all_workers_succeeded=len(report['execution_order'])==expected and all(e['status']=='ok' for e in report['execution_order']),
        route_integrity_all=all(m.get('status')=='complete' and m.get('route_valid') and m.get('fixed_scale') and m.get('own_repeat_exact') for m in allmethods),
        proposal_equality_all=all(m.get('status')=='complete' and m.get('proposals_equal') for m in allmethods),
        normal_gt_all=all(m.get('final_gt_all') for g in groups if g['validity']=='fixed_scale_gt' for m in g['methods'].values()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/sparse-compact')
    p.add_argument('--prior',type=Path,default=ROOT/'cpp-core/docs/sparse-compact/protocol.json')
    p.add_argument('--output',type=Path,default=ROOT/'build/sparse-backends-study/formal/results.json');p.add_argument('--cases')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize',type=Path)
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Preserve existing output; choose new path')
    if args.summarize:
        report=json.loads(args.summarize.read_text());verify_workers(report,args.summarize.parent);save(out,summarize(report,args.summarize.parent));return
    protocolpath=out.with_suffix('.protocol.json')
    if protocolpath.exists():
        protocol=json.loads(protocolpath.read_text())
        if protocol['builds']['compact']!=str(args.build.resolve()) or (args.cases and set(args.cases.split(','))!={c['name'] for c in protocol['cases']}):raise ValueError('Prepared settings differ')
    else:protocol=prepare(args,protocolpath)
    if args.prepare_only:print(protocolpath);return
    if not unchanged(protocol):raise ValueError('Changed prepared sources, binaries, inputs or snapshot')
    report=dict(version=1,protocol=protocol,protocol_sha256=sha(protocolpath),execution_order=[],complete=False);save(out,report)
    routes=[('planes',0),('planes',1),('compact',0),('compact',1)]
    for case in protocol['cases']:
        for block in range(protocol['blocks']):
            for method,backend in routes if block%2==0 else routes[::-1]:
                report['execution_order'].append(run_worker(protocol,case,backend,method,block,out));save(out,report)
    report.update(complete=True,unchanged=unchanged(protocol));report['summary']=summarize(report,out.parent);save(out,report)
    if not report['unchanged'] or not all(report['summary'][key] for key in ('all_workers_succeeded','route_integrity_all','proposal_equality_all','normal_gt_all')):raise SystemExit(1)

if __name__=='__main__':main()
