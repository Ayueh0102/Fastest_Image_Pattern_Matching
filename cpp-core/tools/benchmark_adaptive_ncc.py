"""Frozen, serial CPU study of conservative adaptive NCC screening.

Prepare writes the protocol before measurement. Each worker is a fresh process;
truth is used only here, after its timing/RSS snapshot. Every raw reply, including
failed gates, remains on disk. This is an original-pyramid experiment, not the
sparse fixed-scale pipeline or a Metal benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

import cv2
import numpy as np
from many_object_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = [*sorted((ROOT/'cpp-core/src').glob('*.*')),
                *sorted((ROOT/'cpp-core/include/fpm').glob('*.*')),
                ROOT/'cpp-core/CMakeLists.txt', ROOT/'cpp-core/experiments/CMakeLists.txt',
                ROOT/'cpp-core/experiments/adaptive_ncc.cpp', ROOT/'cpp-core/experiments/adaptive_ncc.hpp',
                ROOT/'cpp-core/tests/adaptive_ncc_test.cpp', ROOT/'cpp-core/tools/adaptive_ncc_runner.cpp',
                Path(__file__), ROOT/'cpp-core/tools/many_object_accuracy.py']
METHODS = ['full', 'legacy'] + [f'{kind}{cap}' for kind in ('global','block') for cap in (4,8,16,32,64)]
FULL_METHODS = ['full', 'legacy', 'global16', 'block16']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def prepare(out, binary):
    if (out/'protocol.json').exists():
        raise ValueError('Protocol exists: choose a new directory, do not overwrite evidence')
    inputs = out/'inputs'; inputs.mkdir(parents=True, exist_ok=True)
    cases=[]
    def local(path):
        dest=inputs/Path(path).name
        if dest.exists() and sha(dest)!=sha(path): raise ValueError('Input basename collision')
        if not dest.exists(): shutil.copyfile(path,dest)
        return str(dest.relative_to(out))
    def add(name, source, templ, **job):
        cases.append(dict(name=name,job=dict(source=local(source), template=local(templ), **job),truth=None))
    # Real native 7x7 response maps isolate large template dot products; these
    # fixed top-left crops are NOT the actual matcher's adaptive fine windows.
    for i in range(1,11):
        templ=ROOT/'Test Images'/f'Dst{i}.{ "jpg" if i==10 else "bmp"}'
        source=ROOT/'Test Images'/({5:'Src5-180.bmp',6:'Src6.jpg'}.get(i,f'Src{i}.bmp'))
        h,w=cv2.imread(str(templ),0).shape
        add(f'native49_dst{i}',source,templ,mode='map',level=0,source_crop=[0,0,w+6,h+6])
        if i in (3,4):
            # Pattern divides the original area by four for its layer budget.
            top=0
            area=h*w
            while area>256: area//=4;top+=1
            add(f'coarse_dst{i}',source,templ,mode='map',level=top)
    rng=np.random.default_rng(2026101601)
    for texture in ('structured','noise'):
        y,x=np.indices((256,256))
        t=(20+60*(x>=43)+90*(y>=181)+50*(x>=207)).astype(np.uint8) if texture=='structured' else rng.integers(0,256,(256,256),dtype=np.uint8)
        s=rng.integers(0,256,(384,384),dtype=np.uint8);s[61:317,53:309]=t
        tp=inputs/f'{texture}_template.png'; sp=inputs/f'{texture}_scene.png'
        if not cv2.imwrite(str(tp),t) or not cv2.imwrite(str(sp),s):raise ValueError('Image write failed')
        cases.append(dict(name=f'dense_{texture}',job=dict(source=str(sp.relative_to(out)),template=str(tp.relative_to(out)),mode='map',level=0),truth=None))
    # Reuse the exact prior synthetic inputs/independent labels, never new tuned
    # data. Ground truth/placement ROIs are stripped from worker jobs.
    prior=json.loads((ROOT/'cpp-core/docs/pose-u8/protocol.json').read_text())
    wanted={'crop_small360','crop_medium20','crop_lowcontrast20','thin_detail20','repo_src4','heldout_crowd128_20'}
    for c in prior['cases']:
        if c['name'] not in wanted:continue
        add('full_'+c['name'],c['source'],c['template'],mode='full',angle=c['angle'],subpixel=True,limit=2000)
        cases[-1]['truth']=c['truth']
        cases[-1]['source_protocol']='cpp-core/docs/pose-u8/protocol.json'
    if len([c for c in cases if c['job']['mode']=='full'])!=6:raise ValueError('Expected six full cases')
    snapshots={}
    for path in SOURCE_FILES:
        rel=path.relative_to(ROOT); dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);snapshots[str(rel)]=sha(path)
    protocol=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
        baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        binary=str(binary.resolve()),binary_sha256=sha(binary),source_sha256=snapshots,
        input_sha256={str(p.relative_to(out)):sha(p) for p in sorted(inputs.iterdir())},
        cases=cases,map_methods=METHODS,full_methods=FULL_METHODS,blocks=2,repeats=5,warmup_seconds=.2,
        timeout_seconds=240,order='forward then reverse methods, serial fresh processes',
        accuracy='Map false rejects=0, survivor bits exact, peak order exact; full ordered outputs exact. Independent GT strict<=2px,<=1deg,<=2% scale.',
        performance=dict(improvement='At least 10% faster than legacy in BOTH paired blocks for adoption in a workload; no universal adoption from selected wins.',
            tolerance_ms='new <= baseline + max(0.1*baseline,2)',rss_bytes='new <= baseline + max(0.1*baseline,32MiB)'),
        timing='Core NCC/map or entire original match. Excludes I/O, model construction, source map pyramid, hashes, stats, oracle. First call and construction recorded separately.',
        memory='Fresh-process peak RSS before oracle; includes image decoder and original full-image load even for map crops. Model retained/temporary bytes are nominal capacities, not allocator totals.',
        limitations=['Gray CPU/NEON only; fine path and GUI defaults unchanged.','No guarantee of all-image accuracy from agreement.','Synthetic cases are previously used, not unseen held-out data.','Native49 maps are microbenchmarks, not a timed fine-search integration.'])
    save(out/'protocol.json',protocol)


def run(out,binary=None):
    p=json.loads((out/'protocol.json').read_text());binary=Path(binary or p['binary'])
    if sha(binary)!=p['binary_sha256']:raise ValueError('Binary changed since protocol')
    for rel,digest in p['source_sha256'].items():
        if sha(out/'source'/rel)!=digest:raise ValueError('Snapshot changed')
    for rel,digest in p['input_sha256'].items():
        if sha(out/rel)!=digest:raise ValueError('Input changed')
    for c in p['cases']:
        methods=p['full_methods'] if c['job']['mode']=='full' else p['map_methods']
        for block in range(p['blocks']):
            for method in (methods if block==0 else list(reversed(methods))):
                stem=f"{c['name']}_b{block}_{method}"
                dest=out/'raw'/f'{stem}.json'
                if dest.exists():continue
                job=dict(score=.8,overlap=.2,angle=180,min_area=256,backend='auto',repeats=p['repeats'],warmup_seconds=p['warmup_seconds'])
                job.update(c['job'])
                job['method']=method.rstrip('0123456789');job['cap']=int(method[len(job['method']):] or 16)
                for field in ('source','template'):job[field]=str((out/job[field]).resolve())
                path=out/'jobs'/f'{stem}.json';save(path,job)
                try:
                    r=subprocess.run([str(binary),str(path)],capture_output=True,text=True,timeout=p['timeout_seconds'])
                    record=dict(case=c['name'],block=block,method=method,returncode=r.returncode,stderr=r.stderr)
                    try:record['result']=json.loads(r.stdout)
                    except json.JSONDecodeError:record['stdout']=r.stdout
                except subprocess.TimeoutExpired as error:
                    record=dict(case=c['name'],block=block,method=method,returncode=-1,error='timeout',timeout_seconds=p['timeout_seconds'])
                save(dest,record)
                print(stem,record['returncode'],flush=True)
    report(out)


def report(out):
    p=json.loads((out/'protocol.json').read_text());rows=[];pairs=[];all_valid=True;all_gt=True
    expected=0
    for c in p['cases']:
        methods=p['full_methods'] if c['job']['mode']=='full' else p['map_methods']
        by_method={}
        for method in methods:
            entries=[]
            for block in range(p['blocks']):
                expected+=1;path=out/'raw'/f"{c['name']}_b{block}_{method}.json"
                if not path.exists():raise ValueError('Missing worker '+str(path))
                entry=json.loads(path.read_text());r=entry.get('result',{})
                valid=entry['returncode']==0 and r.get('valid') is True and r.get('method')==method.rstrip('0123456789') and r.get('rss_includes_oracle') is False and r.get('timed_calls_collect_stats') is False and len(r.get('samples_ms',[]))==p['repeats']
                v=r.get('validation',{})
                valid=valid and v.get('false_rejects')==0 and v.get('survivor_bit_mismatches')==0 and v.get('peak_order_exact') is True and r.get('exact_repeat') is True and r.get('exact_final') is True
                all_valid &= valid
                if not valid:continue
                accuracy=evaluate(r['matches'],c['truth']) if c['truth'] is not None else None
                if accuracy:all_gt &= accuracy['strict']['fp']==0 and accuracy['strict']['fn']==0
                entries.append(dict(block=block,median_ms=statistics.median(r['samples_ms']),peak_rss_bytes=r['measured_memory']['peak_rss_bytes'],accuracy=accuracy,result=r))
            by_method[method]=entries
            if len(entries)!=p['blocks']:continue
            results=[e['result'] for e in entries]
            row=dict(case=c['name'],mode=c['job']['mode'],method=method,median_ms=statistics.median([t for r in results for t in r['samples_ms']]),
                peak_rss_mib=statistics.median([e['peak_rss_bytes']/2**20 for e in entries]),model_build_ms=statistics.median([r['model_build_ms'] for r in results]),
                model_retained_bytes=results[0]['model_retained_bytes'],model_build_workspace_bytes=results[0]['model_build_workspace_bytes'],
                first_ms=statistics.median([r['first_ms'] for r in results]),statistics=results[0]['statistics'],validation=results[0]['validation'],
                blocks=[{k:v for k,v in e.items() if k!='result'} for e in entries],profile_after_timing=results[0]['profile_after_timing'])
            rows.append(row)
        if c['job']['mode']=='full' and len(by_method['full'])==2:
            reference=by_method['full'][0]['result']['matches']
            for entries in by_method.values():
                all_valid &= all(e['result']['matches']==reference for e in entries)
        for method,entries in by_method.items():
            if method in ('full','legacy'):continue
            for e,old in zip(entries,by_method['legacy']):
                if e['block']!=old['block']:raise ValueError('Incomplete paired blocks')
                pairs.append(dict(case=c['name'],method=method,block=e['block'],ratio=e['median_ms']/old['median_ms'],
                    improves_10pct=e['median_ms']<=.9*old['median_ms'],
                    time_tolerance=e['median_ms']<=old['median_ms']+max(.1*old['median_ms'],2),
                    memory_tolerance=e['peak_rss_bytes']<=old['peak_rss_bytes']+max(.1*old['peak_rss_bytes'],32*2**20)))
    summary=dict(schema=1,workers=expected,formal_calls=expected*p['repeats'],all_valid=all_valid,all_synthetic_gt=all_gt,rows=rows,pairs=pairs,
        time_tolerance_failures=[v for v in pairs if not v['time_tolerance']],memory_tolerance_failures=[v for v in pairs if not v['memory_tolerance']])
    save(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('workers','formal_calls','all_valid','all_synthetic_gt')}),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','report']);a.add_argument('--out',type=Path,required=True);a.add_argument('--binary',type=Path)
    args=a.parse_args();out=args.out.resolve()
    if args.action=='prepare':prepare(out,args.binary or ROOT/'build/adaptive-ncc/experiments/fpm_adaptive_ncc_runner')
    elif args.action=='run':run(out,args.binary)
    else:report(out)

if __name__=='__main__':main()
