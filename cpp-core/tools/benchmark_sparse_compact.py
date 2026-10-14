"""Exact sparse-response representation study; keep frozen/controlled references.

Three serial routes: frozen 70727a8 planes, current planes, current compact mask.
Independent GT is parent-only; complete candidate/final records are compared
exactly across methods and repeated calls. Source/input/binary hashes are frozen
before measurement. This study changes representation, not search coverage.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import gzip
import tempfile
import sys
import platform

import cv2
import numpy as np
from benchmark_fixed_scale import ROOT, GIT, sha, save, distribution, write_image
from benchmark_sparse_orientation import proposal_accuracy, BINARY_NAMES, unchanged as live_unchanged
from many_object_accuracy import evaluate
from many_object_scenes import _corners, _roi, render_scene, validate_placements

METHODS = ('frozen_planes', 'planes', 'compact')
COUNTS = ('feature_count', 'angle_count', 'candidate_count', 'windows', 'feature_evaluations', 'threshold_survivors')


def unchanged(protocol):
    return live_unchanged(protocol) and all(Path(p).is_file() and sha(p)==h
        for p,h in protocol.get('snapshot_sha256',{}).items())


def prepare(args, path):
    prior = json.loads(args.baseline_protocol.read_text())
    cases = json.loads(json.dumps(prior['cases']))
    inputs = path.parent/'inputs'; inputs.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case['held_out'] = False
        for key in ('source', 'template', 'metadata'):
            old = Path(case[key]); target = inputs/old.name
            expected = prior['input_sha256'].get(str(old))
            if expected is None or sha(old)!=expected: raise ValueError('Changed prior fixture '+str(old))
            if not target.exists(): shutil.copyfile(old, target)
            if sha(target)!=expected: raise ValueError('Changed copied fixture '+str(target))
            case[key] = str(target.resolve())
    if not args.pilot:
        template = cv2.imread(str(inputs/'src3_native_crop.png'), 0)
        truth = []
        for i, (x, y, angle) in enumerate([(512.35,512.65,-17.3),(2048.35,512.65,9.7),
                (3584.35,512.65,-3.4),(512.35,3584.65,14.6),(2048.35,3584.65,-11.8),(3584.35,3584.65,2.3)]):
            corners = _corners([x,y],angle,1.,template.shape)
            truth.append(dict(index=i,pos_x=x,pos_y=y,angle=angle,scale=1.,gain=1.,bias=0.,corners=corners,warp_roi=_roi(corners)))
        source = inputs/'large4096_6.png'; meta = inputs/'large4096_6.json'
        write_image(source, render_scene(template,truth,(4096,4096),noise_seed=2026101417,noise_std=2.,background=128.))
        save(meta,dict(name='large4096_6',truth=truth,source_shape=[4096,4096],template_shape=list(template.shape),
            renderer='OpenCV native-pixel independent warpAffine, INTER_LINEAR',noise_seed=2026101417,noise_std=2.,
            validation=validate_placements(truth,(4096,4096),template.shape),
            limitations=['Six separated copies, simple background, synthetic GT; not independent-camera accuracy.']))
        cases.append(dict(name='large4096_6',source=str(source.resolve()),template=str((inputs/'src3_native_crop.png').resolve()),
            metadata=str(meta.resolve()),scene_shape=[4096,4096],template_shape=list(template.shape),angle=20.,truth=truth,
            validity='fixed_scale_gt',backends=[1],held_out=True))
    if args.cases:
        selected = args.cases.split(',')
        if len(set(selected))!=len(selected) or set(selected)-{c['name'] for c in cases}: raise ValueError('Unknown/duplicate cases')
        cases=[c for c in cases if c['name'] in selected]
    builds = {'frozen_planes':str(args.baseline_build.resolve()),'planes':str(args.build.resolve()),'compact':str(args.build.resolve())}
    baseline_expected={Path(p).name:h for p,h in prior['binary_sha256'].items() if '/sparse-orientation/' in p}
    binaries={str((Path(build)/'experiments'/name).resolve()):sha(Path(build)/'experiments'/name)
        for build in builds.values() for name in BINARY_NAMES}
    for name in BINARY_NAMES:
        if sha(args.baseline_build/'experiments'/name)!=baseline_expected[name]: raise ValueError('Changed frozen baseline '+name)
    sources=[]
    for directory in ('src','include','experiments','tests','tools'):
        sources += [p for p in (ROOT/'cpp-core'/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and
            (p.suffix in ('.cpp','.hpp','.h','.mm','.metal','.in','.py') or p.name=='CMakeLists.txt')]
    sources += [ROOT/'cpp-core/CMakeLists.txt',ROOT/'macos-cli/mainwindow.cpp',ROOT/'macos-cli/mainwindow.h']
    snapshot=path.parent/'source-snapshot'
    if snapshot.exists(): raise ValueError('Choose a new study directory; snapshot exists')
    for source in sources:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    archive=path.parent/'baseline-source.tar'
    with archive.open('wb') as target:
        subprocess.run([GIT,'archive','70727a8','cpp-core/src','cpp-core/include','cpp-core/experiments','cpp-core/tests',
            'cpp-core/tools','cpp-core/CMakeLists.txt','macos-cli/mainwindow.cpp','macos-cli/mainwindow.h'],cwd=ROOT,stdout=target,check=True)
    # Confirm archived baseline runtime files match the measured prior snapshot.
    with tarfile.open(archive) as tar:
        import hashlib
        for member in tar.getmembers():
            key=str(ROOT/member.name)
            if member.isfile() and key in prior['source_sha256']:
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=prior['source_sha256'][key]:
                    raise ValueError('Baseline source mismatch '+member.name)
    shutil.copyfile(args.baseline_protocol,path.parent/'baseline-protocol.json')
    protocol=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        baseline_commit='70727a87bb91684d519614d47ec2560356c3d996',implementation_base=subprocess.check_output([GIT,'rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        pilot=args.pilot,blocks=1 if args.pilot else 2,repeats=1 if args.pilot else 3,warmup_seconds=0. if args.pilot else 1.,timeout_seconds=120.,
        cases=cases,builds=builds,methods=list(METHODS),shared_settings=prior['shared_settings'],sparse_settings=prior['sparse_settings'],
        proposal_basin=prior['proposal_basin'],
        timing='Fresh serial workers; forward/reverse block order, construction and first call separate; prepare is included in proposal and total, decoding/JSON excluded.',
        scope='Same full-image fixed-scale rotation search and candidate budget, no GT or count supplied to worker. Existing cases are reused without tuning; 4096 square added for representation scaling.',
        gates=dict(exactness='All raw ordered proposals, final matches and work counts exactly equal frozen reference on every first/measured call; exact own warm repeats.',
            performance='Compact per-block median <= reference + max(2ms,10%) for frozen and current planes independently.',
            memory='Compact peak RSS <= reference + max(32MiB,10%) for both references; source representation N vs8N measured separately.',
            accuracy='Normal GT final zeroFP/FN; keep inherited proposal-basin failures, scale+4% negative and unlabelled real image separate.'),
        memory='Shared immutable LUT2048bytes accounted separately from transient source (8N orN); working_bytes is allocation estimate, not RSS bound. Never add shared GPU capacities and RSS.',
        source_sha256={str(p.resolve()):sha(p) for p in sources},binary_sha256=binaries,
        input_sha256={c[k]:sha(c[k]) for c in cases for k in ('source','template','metadata')},
        snapshot_sha256={str(p.resolve()):sha(p) for p in [archive,path.parent/'baseline-protocol.json',
            *[p for p in snapshot.rglob('*') if p.is_file()]]},
        baseline_source_sha256=sha(archive),baseline_protocol_sha256=sha(args.baseline_protocol))
    save(path,protocol);return protocol


def run_worker(protocol,case,backend,method,block,output):
    settings=dict(protocol['shared_settings']);settings.pop('optimize_scale')
    job=dict(settings,**protocol['sparse_settings'],source=case['source'],template=case['template'],angle=case['angle'],
        backend=backend,proposal_method='sparse_orientation',sparse_response_mode=int(method=='compact'),
        repeats=protocol['repeats'],warmup_seconds=protocol['warmup_seconds'])
    name=f'{case["name"]}-b{backend}-{method}-block{block}'
    path=output.parent/'jobs'/(name+'.json');save(path,job)
    entry=dict(case=case['name'],backend=backend,method=method,block=block,job=str(path.relative_to(output.parent)),job_sha256=sha(path))
    print('Running '+name,flush=True)
    try:
        child=subprocess.run([str(Path(protocol['builds'][method])/'experiments/fpm_native_pipeline_runner'),str(path)],capture_output=True,text=True,timeout=protocol['timeout_seconds'])
        stdout,stderr=child.stdout,child.stderr;entry.update(status='ok' if child.returncode==0 else 'process_error',returncode=child.returncode)
    except subprocess.TimeoutExpired as error:
        stdout,stderr=error.stdout or '',error.stderr or ''
        if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
        if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
        entry.update(status='timeout')
    for key,value in (('stdout',stdout),('stderr',stderr)):
        dest=path.with_suffix('.'+key+'.txt');dest.write_text(value)
        entry[key]=str(dest.relative_to(output.parent));entry[key+'_sha256']=sha(dest)
    if entry['status']=='ok':
        try:
            raw=json.loads(stdout)
            if len(raw['runs'])!=protocol['repeats'] or not raw['exact_repeat']:raise ValueError('Missing measurements or repeat mismatch')
            dest=path.with_suffix('.result.json');save(dest,raw);entry.update(result=str(dest.relative_to(output.parent)),result_sha256=sha(dest))
        except (ValueError,KeyError,TypeError) as error:entry.update(status='invalid_output',error=str(error))
    return entry


def payload(record):
    return dict(matches=record['matches'],proposals=record['proposals'],counts={k:record['sparse'][k] for k in COUNTS})


def summarize(report,base):
    protocol=report['protocol'];groups=[]
    for case in protocol['cases']:
        for backend in case['backends']:
            entries=[e for e in report['execution_order'] if e['case']==case['name'] and e['backend']==backend]
            refs=[json.loads((base/e['result']).read_text()) for e in entries if e['status']=='ok' and e['method']=='frozen_planes']
            reference=payload(refs[0]['first']) if refs else None
            methods={}
            for method in METHODS:
                workers=[(e,json.loads((base/e['result']).read_text())) for e in entries if e['status']=='ok' and e['method']==method]
                if not workers:methods[method]=dict(status='missing');continue
                runs=[r for _,w in workers for r in w['runs']];allcalls=[r for _,w in workers for r in [w['first'],*w['runs']]]
                accuracy=[evaluate(r['matches'],case['truth']) for r in allcalls] if case['truth'] is not None else []
                proposals=[proposal_accuracy(r['proposals'],case['truth'],protocol['proposal_basin']) for r in allcalls] if case['truth'] is not None else []
                area=np.prod(case['scene_shape']).item();expected=area*(1 if method=='compact' else 8)
                route=all(r['sparse']['enabled'] and r['sparse']['source_levels']==1 and r['frame_host_bytes']==0 and
                    r['sparse']['source_bytes']==expected and (method=='frozen_planes' or
                    (r['sparse']['response_mode']==int(method=='compact') and r['sparse']['lookup_bytes']==2048)) for r in allcalls)
                methods[method]=dict(status='complete' if len(workers)==protocol['blocks'] else 'partial',
                    total_ms=distribution([r['total_ms'] for r in runs]),setup_ms=[w['construct_ms'] for _,w in workers],first_ms=[w['first']['total_ms'] for _,w in workers],
                    peak_rss_bytes=[w['final_memory']['peak_rss_bytes'] for _,w in workers],current_rss_bytes=[w['final_memory']['current_rss_bytes'] for _,w in workers],
                    stage_medians={k:statistics.median(r[k] for r in runs) for k in ('prepare_ms','proposal_ms','refine_ms','nms_ms')},
                    sparse_medians={k:statistics.median(r['sparse'][k] for r in runs) for k in runs[0]['sparse'] if isinstance(runs[0]['sparse'][k],(int,float)) and not isinstance(runs[0]['sparse'][k],bool)},
                    exact_frozen=reference is not None and all(payload(r)==reference for r in allcalls),own_repeat_exact=all(w['exact_repeat'] for _,w in workers),route_valid=route,
                    fixed_scale=all(m['scale']==1 for r in allcalls for key in ('matches','proposals') for m in r[key]),
                    backend_valid=all(r['selected_refinement_backend']==backend for r in allcalls if r['proposals']),
                    first_outputs=workers[0][1]['first']['matches'],first_accuracy=accuracy[0] if accuracy else None,
                    first_proposal_coverage=proposals[0] if proposals else None,
                    all_final_gt=None if not accuracy else all(a['strict']['fp']==a['strict']['fn']==0 for a in accuracy),
                    all_proposal_gt=None if not proposals else all(a['strict']['fn']==0 for a in proposals),
                    blocks=[dict(block=e['block'],median_ms=statistics.median(r['total_ms'] for r in w['runs']),peak_rss_bytes=w['final_memory']['peak_rss_bytes']) for e,w in workers])
            pairs=[]
            if all(m.get('status')=='complete' for m in methods.values()):
                for ref in ('frozen_planes','planes'):
                    for block in methods['compact']['blocks']:
                        old=next(b for b in methods[ref]['blocks'] if b['block']==block['block'])
                        pairs.append(dict(reference=ref,block=block['block'],time_ratio=block['median_ms']/old['median_ms'],
                            speed_gate=block['median_ms']<=old['median_ms']+max(2.,.1*old['median_ms']),
                            rss_delta_bytes=block['peak_rss_bytes']-old['peak_rss_bytes'],
                            memory_gate=block['peak_rss_bytes']<=old['peak_rss_bytes']+max(32*2**20,.1*old['peak_rss_bytes'])))
            groups.append(dict(name=case['name'],backend=backend,validity=case['validity'],methods=methods,paired_comparison=pairs))
    allmethods=[m for g in groups for m in g['methods'].values()]
    expected=sum(len(c['backends']) for c in protocol['cases'])*protocol['blocks']*len(METHODS)
    return dict(cases=groups,workers=len(report['execution_order']),expected_workers=expected,
        measured_calls=sum(len(json.loads((base/e['result']).read_text())['runs']) for e in report['execution_order'] if e['status']=='ok'),
        all_workers_succeeded=len(report['execution_order'])==expected and all(e['status']=='ok' for e in report['execution_order']),
        exact_frozen_all=all(m.get('status')=='complete' and m.get('exact_frozen') and m.get('own_repeat_exact') for m in allmethods),
        route_integrity_all=all(m.get('status')=='complete' and m.get('route_valid') and m.get('fixed_scale') and m.get('backend_valid') for m in allmethods),
        fixed_scale_gt_all=all(m.get('all_final_gt') for g in groups if g['validity']=='fixed_scale_gt' for m in g['methods'].values()),
        fixed_scale_proposal_coverage_all=all(m.get('all_proposal_gt') for g in groups if g['validity']=='fixed_scale_gt' for m in g['methods'].values()))


def verify_workers(report,base):
    for e in report['execution_order']:
        for key in ('job','result','stdout','stderr'):
            if key in e and sha(base/e[key])!=e[key+'_sha256']:raise ValueError('Changed worker artifact '+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/sparse-compact')
    p.add_argument('--baseline-build',type=Path,default=ROOT/'build/sparse-orientation')
    p.add_argument('--baseline-protocol',type=Path,default=ROOT/'cpp-core/docs/sparse-orientation/protocol.json')
    p.add_argument('--output',type=Path,default=ROOT/'build/sparse-compact-study/formal/results.json')
    p.add_argument('--cases');p.add_argument('--pilot',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summarize',type=Path)
    args=p.parse_args();output=args.output.resolve()
    if args.summarize:
        if output.exists():raise ValueError('Preserve existing output; choose a new summary path')
        report=json.loads(args.summarize.read_text());verify_workers(report,args.summarize.parent)
        save(output,summarize(report,args.summarize.parent));return
    if output.exists():raise ValueError('Preserve existing measurements; choose a new directory')
    path=output.with_suffix('.protocol.json')
    if path.exists():
        protocol=json.loads(path.read_text())
        if protocol['pilot']!=args.pilot or protocol['builds']['planes']!=str(args.build.resolve()) or protocol['builds']['frozen_planes']!=str(args.baseline_build.resolve()):raise ValueError('Prepared settings differ')
        if args.cases and set(args.cases.split(','))!={c['name'] for c in protocol['cases']}:raise ValueError('Prepared cases differ')
    else:protocol=prepare(args,path)
    if args.prepare_only:print(path);return
    if not unchanged(protocol):raise ValueError('Sources/inputs/binaries changed after preparation')
    report=dict(version=1,protocol=protocol,protocol_sha256=sha(path),execution_order=[],complete=False);save(output,report)
    for c in protocol['cases']:
        for backend in c['backends']:
            for block in range(protocol['blocks']):
                for method in METHODS if block%2==0 else METHODS[::-1]:
                    report['execution_order'].append(run_worker(protocol,c,backend,method,block,output));save(output,report)
    report['complete']=True;report['unchanged']=unchanged(protocol);report['summary']=summarize(report,output.parent);save(output,report)
    if not report['unchanged'] or not all(report['summary'][key] for key in
        ('all_workers_succeeded','exact_frozen_all','route_integrity_all','fixed_scale_gt_all')):raise SystemExit(1)

if __name__=='__main__':main()
