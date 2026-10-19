"""Stage34 new-frame ownership, fresh-vs-reused sessions, and latest 30min soak.

Generate synthetic shifted/padded/empty frames with independent GT bookkeeping.
No real-camera generalization claim. Formal runs are serial and source-hashed.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import cv2
import numpy as np

from benchmark_native_pipeline import save, sha
from benchmark_resident_multiscale import job_for, strip_clocks
from benchmark_fine_fused import source_files
from benchmark_closure_backends import distribution, paired_change
from many_object_accuracy import evaluate

ROOT=Path(__file__).resolve().parents[2]
BINARIES=('fpm_native_pipeline_runner','libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib')


def manifest(build, inputs):
    return {str(p.resolve()):sha(p) for p in [*source_files(), *[build/'experiments'/n for n in BINARIES], *inputs]}


def frames(cases, dest):
    case=next(c for c in cases if c['name']=='new_texture12')
    original=cv2.imread(case['source'],cv2.IMREAD_GRAYSCALE)
    h,w=original.shape
    specs=[(0,0,0,0,False),(5,3,32,32,False),(9,6,0,0,False),
           (12,15,512,256,False),(0,0,0,0,True),(0,0,0,0,False)]
    result=[]
    for index,(dx,dy,padw,padh,empty) in enumerate(specs):
        image=np.full((h+padh,w+padw),150,dtype=np.uint8)
        truth=[] if empty else copy.deepcopy(case['truth'])
        if not empty:
            # Original targets have generous margins, so translated source-edge
            # background may be cropped without changing any target support.
            image[dy:min(dy+h,image.shape[0]),dx:min(dx+w,image.shape[1])]=original[:min(h,image.shape[0]-dy),:min(w,image.shape[1]-dx)]
            for pose in truth:
                pose['pos_x']+=dx;pose['pos_y']+=dy
                if 'corners' in pose: pose['corners']=[[x+dx,y+dy] for x,y in pose['corners']]
                if 'warp_roi' in pose:
                    x0,y0,x1,y1=pose['warp_roi'];pose['warp_roi']=[x0+dx,y0+dy,x1+dx,y1+dy]
        path=dest/f'frame-{index}.png'
        if not cv2.imwrite(str(path),image): raise RuntimeError('Cannot save frame')
        result.append(dict(index=index,path=str(path),truth=truth,shape=list(image.shape),
                           translation=[dx,dy],empty=empty,sha256=sha(path)))
    return case,result


def checked(report,base):
    if not report.get('complete') or sha(base/'protocol.json')!=report['protocol_sha256']:
        raise ValueError('Incomplete run or altered protocol')
    if json.loads((base/'protocol.json').read_text())!=report['protocol']:
        raise ValueError('Protocol mismatch')
    if report.get('post_hashes')!=report['protocol']['hashes'] or not report.get('unchanged'):
        raise ValueError('Source, binary or input provenance changed')
    for e in report['entries']:
        for k in ('job','raw','stderr'):
            if sha(base/e[k])!=e[k+'_sha256']:raise ValueError('Changed worker evidence')
        if e['returncode']:raise ValueError('Worker did not succeed')


def valid_route(row):
    active=row['refinement_candidates']>0
    return (row['configured_refinement_backend']==1 and row['refinement_mode']==0
        and row['pose_input_mode']==1 and row['pose_storage_mode']==1
        and (row['selected_refinement_backend']==1 and row['pose_gpu_bytes']>0
             and row['refinement_dispatches']>0 and row['refinement_command_buffers']>0
             if active else row['pose_gpu_bytes']==0 and row['refinement_dispatches']==0))


def finite_fields(row, keys):
    if any(not isinstance(row[k], (int,float)) or not math.isfinite(row[k]) or row[k]<0 for k in keys):
        raise ValueError('Invalid nonnegative metric')


def execute(build,dest,label,job):
    jp=dest/(label+'.job.json');rp=dest/(label+'.raw.json');ep=dest/(label+'.stderr.txt')
    save(jp,job);print(label,flush=True)
    child=subprocess.run([str(build/'experiments/fpm_native_pipeline_runner'),str(jp)],capture_output=True,text=True)
    rp.write_text(child.stdout);ep.write_text(child.stderr)
    entry=dict(label=label,returncode=child.returncode)
    for k,p in [('job',jp),('raw',rp),('stderr',ep)]:entry[k]=p.name;entry[k+'_sha256']=sha(p)
    if child.returncode:raise RuntimeError(child.stderr)
    return entry


def sequence_summary(report,base):
    checked(report,base);p=report['protocol'];items=p['frames'];cycles=p['cycles'];blocks=p['blocks']
    if blocks<2 or blocks%2 or cycles<2 or not items or len(items)>64:
        raise ValueError('Need balanced blocks and repeated nonempty frame sequence')
    if p['job']!={**job_for(p['case'],'resident0'), 'refinement_backend':1,'run_resource_diagnostics':False}:
        raise ValueError('Sequence protocol settings changed')
    for i,f in enumerate(items):
        if f['index']!=i or p['hashes'].get(f['path'])!=f['sha256']:
            raise ValueError('Frame identity/hash mismatch')
    expected=[(b,m) for b in range(blocks) for m in (['fresh','reuse'] if b%2==0 else ['reuse','fresh'])]
    if [(e['block'],e['mode']) for e in report['entries']]!=expected:raise ValueError('Sequence coverage/order')
    workers={};accuracies=[];routes=True;repeats=True
    for e in report['entries']:
        raw=json.loads((base/e['raw']).read_text());job=json.loads((base/e['job']).read_text())
        if job!={**p['job'], 'source_frames':[f['path'] for f in items], 'sequence_mode':e['mode'], 'repeats':cycles}:raise ValueError('Unexpected sequence job')
        if raw['sequence_mode']!=e['mode'] or raw['source_frames']!=[f['path'] for f in items] or raw['cycles']!=cycles:
            raise ValueError('Sequence worker metadata differs')
        rows=raw['runs']
        if len(rows)!=len(items)*cycles or raw['calls']!=len(rows):raise ValueError('Missing source frames')
        for call,r in enumerate(rows):
            index=call%len(items);f=items[index]
            if (r['call'],r['source_index'],r['sequence_cycle'])!=(call,index,call//len(items)):raise ValueError('Wrong frame order')
            if [r['source_height'],r['source_width']]!=f['shape'] or r['source_path']!=f['path']:
                raise ValueError('Wrong frame identity or shape')
            finite_fields(r,('total_ms','frame_wall_ms','source_update_ms','construct_ms','release_ms','source_decode_ms','pose_create_ms'))
            if r['frame_wall_ms']+1e-6<r['total_ms']+r['source_update_ms']+r['construct_ms']+r['release_ms']:
                raise ValueError('Inconsistent frame timing scope')
            repeats &= (r['matches']==rows[index]['matches'] and r['proposals']==rows[index]['proposals']
                and r['exact_sequence_repeat'] and r['reference_cycle']==(call<len(items)))
            accuracies.append(evaluate(r['matches'],f['truth'])['strict'])
            reused=e['mode']=='reuse' and call>0
            routes &= (valid_route(r) and r['refinement_candidates']==len(r['proposals']) and r['source_generation']==(call if e['mode']=='reuse' else 0)
                and r['retained_template_models']==(len(p['case']['scales']) if reused else 0)
                and r['source_copy_bytes']==(int(np.prod(f['shape'])) if reused else 0)
                and r['pose_invalidated']==(reused and not items[(index-1)%len(items)]['empty'])
                and (r['pose_create_ms']>0 if r['refinement_candidates'] else r['pose_create_ms']==0))
            if reused:routes &= r['source_dimensions_changed']==(f['shape']!=items[(index-1)%len(items)]['shape'])
        routes &= raw['exact_sequence_repeat'] and raw['drained']
        workers[(e['block'],e['mode'])]=raw
    equal=True;work=True
    for b in range(blocks):
        for a,z in zip(workers[(b,'fresh')]['runs'],workers[(b,'reuse')]['runs']):
            equal &= a['matches']==z['matches'] and a['proposals']==z['proposals']
            work &= strip_clocks(a['work'])==strip_clocks(z['work'])
    methods={}
    for mode in ('fresh','reuse'):
        ws=[workers[(b,mode)] for b in range(blocks)]
        rows=[r for w in ws for r in w['runs']]
        methods[mode]=dict(frame_wall_ms=distribution([r['frame_wall_ms'] for r in rows]),
            process_median_frame_wall_ms=[statistics.median(r['frame_wall_ms'] for r in w['runs']) for w in ws],
            algorithm_ms=distribution([r['total_ms'] for r in rows]),
            update_ms=distribution([r['source_update_ms'] for r in rows]),
            construct_ms=distribution([r['construct_ms'] for r in rows]),
            source_decode_ms=distribution([r['source_decode_ms'] for r in rows]),
            peak_rss_bytes=distribution([w['final_memory']['peak_rss_bytes'] for w in ws]),
            final_release_ms=distribution([w['final_release_ms'] for w in ws]),
            after_release_rss_bytes=distribution([w['after_release_memory']['current_rss_bytes'] for w in ws]))
    changes=paired_change(methods['fresh']['process_median_frame_wall_ms'],methods['reuse']['process_median_frame_wall_ms'])
    result=dict(complete=True,workers=len(workers),calls=sum(len(w['runs']) for w in workers.values()),
        unchanged=report['unchanged'],routes_passed=bool(routes),repeat_exact=bool(repeats),fresh_reuse_ordered_exact=equal,logical_work_exact=work,
        accuracy_passed=all(a['fp']==a['fn']==0 for a in accuracies),
        max_fp=max(a['fp'] for a in accuracies),max_fn=max(a['fn'] for a in accuracies),
        methods=methods,reuse_vs_fresh=changes,
        limits=['Synthetic translated/padded frames, not real camera footage.',
                'Template/model bank and frame arenas retained; pose image context rebuilt each source update.',
                'Sequence reference results/JSON retained in both bounded arms; RSS includes harness output.',
                'Initial source/template decode is in load, first row decode=0; later rows decode separately.',
                'frame_wall excludes decode/serialization; no inference of end-to-end camera throughput.'])
    result['passed']=bool(routes and repeats and equal and work and result['accuracy_passed'] and report['unchanged'])
    return result


def soak_summary(report,base):
    checked(report,base);p=report['protocol']
    if len(report['entries'])!=1 or p['seconds']<=0 or p['seconds']>3600:raise ValueError('Invalid soak coverage/duration')
    e=report['entries'][0]
    if json.loads((base/e['job']).read_text())!=p['job']:raise ValueError('Unexpected soak job')
    expected={**job_for(p['case'],'resident0'), 'refinement_backend':1,'run_resource_diagnostics':False,
        'soak_seconds':p['seconds'],'stream_path':p['job']['stream_path'],'warmup_seconds':1.}
    if p['job']!=expected:raise ValueError('Soak protocol settings changed')
    if sha(base/'reference.json')!=p['reference_sha256']:raise ValueError('Changed same-binary reference')
    raw=json.loads((base/e['raw']).read_text());rows=[json.loads(s) for s in (base/'samples.jsonl').read_text().splitlines()]
    if sha(base/'samples.jsonl')!=report['stream_sha256']:raise ValueError('Changed stream')
    if len(rows)!=raw['calls'] or not rows:raise ValueError('Missing samples')
    for r in rows:
        finite_fields(r,('total_ms','elapsed_ms','recreate_ms'))
        finite_fields(r['memory'],('current_rss_bytes','peak_rss_bytes'))
    rss=[r['memory']['current_rss_bytes'] for r in rows];n=max(1,len(rows)//5)
    start=statistics.median(rss[:n]);end=statistics.median(rss[-n:]);allowed=max(32*2**20,.1*start)
    capkeys=('frame_host_bytes','frame_coarse_gpu_bytes','frame_fine_gpu_bytes','pose_host_bytes','pose_gpu_bytes')
    capacities=all([r[k] for k in capkeys]==[rows[0][k] for k in capkeys] for r in rows)
    work=all(strip_clocks(r['work'])==strip_clocks(rows[0]['work']) for r in rows)
    generation=all(r['call']==i and r['context_index']==i//100 and
        (r['generation']==1 and r['recreate_ms']>0 if i and i%100==0 else r['recreate_ms']==0) and
        (i%100==0 or r['generation']==rows[i-1]['generation']+1) and
        (not i or r['elapsed_ms']>=rows[i-1]['elapsed_ms']) for i,r in enumerate(rows))
    generation &= raw['contexts_recreated']==(len(rows)-1)//100 and raw['duration_ms']>=rows[-1]['elapsed_ms']
    reference=json.loads((base/'reference.json').read_text())['first']
    exact=raw['exact'] and all(r['exact'] for r in rows) and raw['first']['matches']==reference['matches'] and raw['first']['proposals']==reference['proposals']
    routes=all(valid_route(r) and r['source_generation']==0 and
        r['refinement_candidates']==len(reference['proposals']) for r in [raw['first'],*rows])
    gt=evaluate(raw['first']['matches'],p['case']['truth'])['strict']
    result=dict(duration_ms=raw['duration_ms'],calls=len(rows),contexts_recreated=raw['contexts_recreated'],
        requested_seconds=p['seconds'],all_outputs_exact=exact,async_drained=raw['drained'],generation_valid=generation,
        routes_passed=routes,logical_work_constant=work,retained_capacity_constant=capacities,
        strict_tp=gt['tp'],strict_fp=gt['fp'],strict_fn=gt['fn'],
        rss=dict(first_fifth_median=start,last_fifth_median=end,delta=end-start,allowed_growth=allowed,min=min(rss),max=max(rss)),
        time_ms=distribution([r['total_ms'] for r in rows]),
        unchanged=report['unchanged'],
        note='Latest configuration fixed image, streamed compact records, periodic Session recreation; no camera/general leak-freedom claim')
    result['passed']=bool(exact and routes and raw['drained'] and generation and work and capacities and end-start<=allowed and report['unchanged'] and gt['fp']==gt['fn']==0 and raw['duration_ms']>=p['seconds']*1000)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/stage34-closure')
    parser.add_argument('--cases-json',type=Path,default=ROOT/'build/resident-multiscale-inputs/cases.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['sequence','soak'],default='sequence')
    parser.add_argument('--seconds',type=float,default=1800.)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.summarize:
        report=json.loads(args.summarize.read_text());fn=sequence_summary if report['protocol']['mode']=='sequence' else soak_summary
        result=fn(report,args.summarize.parent);save(args.output,result);return 0 if result['passed'] else 1
    dest=args.output.resolve()
    if dest.exists():raise ValueError('Refusing to overwrite evidence')
    dest.mkdir(parents=True)
    cases=json.loads(args.cases_json.read_text())
    if args.mode=='sequence':
        case,items=frames(cases,dest);inputs=[Path(f['path']) for f in items]+[Path(case['template'])]
        protocol=dict(mode='sequence',case=case,frames=items,blocks=4,cycles=4,job=job_for(case,'resident0'))
        protocol['job'].update(refinement_backend=1,run_resource_diagnostics=False)
    else:
        if args.reference is None:raise ValueError('Soak needs formal same-binary reference')
        case=next(c for c in cases if c['name']=='many512_pose');items=[]
        inputs=[Path(case[k]) for k in ('source','template')]
        reference=dest/'reference.json';reference.write_bytes(args.reference.read_bytes());inputs.append(reference)
        protocol=dict(mode='soak',case=case,seconds=args.seconds,reference_sha256=sha(reference),job=job_for(case,'resident0'))
        protocol['job'].update(refinement_backend=1,run_resource_diagnostics=False,soak_seconds=args.seconds,
                               stream_path=str(dest/'samples.jsonl'),warmup_seconds=1.)
    protocol['hashes']=manifest(args.build,[*inputs,args.cases_json])
    save(dest/'protocol.json',protocol)
    report=dict(protocol=protocol,protocol_sha256=sha(dest/'protocol.json'),entries=[],complete=False)
    if args.mode=='sequence':
        for block in range(protocol['blocks']):
            for mode in (['fresh','reuse'] if block%2==0 else ['reuse','fresh']):
                j={**protocol['job'],'source_frames':[f['path'] for f in items],'sequence_mode':mode,'repeats':protocol['cycles']}
                e=execute(args.build,dest,f'{mode}-{block}',j);e.update(block=block,mode=mode)
                report['entries'].append(e);save(dest/'formal.json',report)
    else:
        report['entries'].append(execute(args.build,dest,'soak',protocol['job']))
        report['stream_sha256']=sha(dest/'samples.jsonl')
    post_hashes={k:sha(k) for k in protocol['hashes']}
    report.update(complete=True,unchanged=post_hashes==protocol['hashes'],post_hashes=post_hashes)
    save(dest/'formal.json',report)
    result=(sequence_summary if args.mode=='sequence' else soak_summary)(report,dest)
    save(dest/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('methods','time_ms')},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
