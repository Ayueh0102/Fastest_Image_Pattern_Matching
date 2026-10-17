"""Stage35 serial same-process search/color comparison and offline replay.

Four methods share gray geometry; geometry_original preserves the earlier
search budget as a separate accuracy/cost control. Calibration uses three
normal images before heldout execution. Missing calibration candidates remain
explicit abstentions; partial calibration is provisional and no valid image
means an unset threshold, not a successful normal classification.
Saved jobs never contain ground truth, evaluation masks or expected poses.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import cv2
import numpy as np

from benchmark_fine_fused import source_files
from many_object_accuracy import evaluate, _maximum_matching
from stage35_scenes import save, sha

ROOT=Path(__file__).resolve().parents[2]
SHARED_GEOMETRY_METHODS=['geometry','rgb_cpu','rgb_metal','chroma']
METHODS=[*SHARED_GEOMETRY_METHODS,'geometry_original']


def job(case, method):
    selected={'geometry':('geometry_only','cpu',1),'geometry_original':('geometry_only','cpu',1),
              'rgb_cpu':('rgb_ncc','cpu',8),
              'rgb_metal':('rgb_ncc','metal',1),'chroma':('chroma','cpu',1)}[method]
    search=copy.deepcopy(case['search'])
    if method=='geometry_original':
        search.update(min_area=256,proposal_score=.5,limit=100)
    out=dict(source=case['source'],template=case['template'],search=search,
             method=selected[0],backend=selected[1],threads=selected[2],
             warmups=0,repeats=1,chroma_distance_threshold=.04,
             min_linear_sum=.03,min_valid_fraction=.95)
    if method=='chroma' and case.get('mask'):
        out['mask']=case['mask']
    return out


def candidate_values(row,method):
    if len(row['scores'])!=len(row['search_matches']):
        raise ValueError('Score/detection length mismatch')
    if method=='chroma':
        return [s['changed_fraction'] if s['status']==0 else None for s in row['scores']]
    return [1-float(s) if math.isfinite(s) and -1<=s<=1.00001 else None for s in row['scores']]


def image_value(row,method):
    valid=[x for x in candidate_values(row,method) if x is not None]
    return max(valid) if valid else None


def category_coverage(matches,case):
    centers=np.array([[m['pos_x'],m['pos_y']] for m in matches]).reshape(-1,2)
    points=np.asarray(case['points'])
    distances=np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2)
    assignment=_maximum_matching([np.flatnonzero(d<=case['point_tolerance_px']).tolist() for d in distances],len(matches))
    covered=sum(i>=0 for i in assignment)
    return dict(category_points=len(points),detected_patches=len(matches),covered_points=covered,
                uncovered_points=len(points)-covered,unassigned_detections=len(matches)-covered,
                interpretation='Category coverage only; no query identity or color accuracy claim.')


def accuracy(row,method,case,threshold):
    if case['kind']=='real_category_probe':
        return category_coverage(row['search_matches'],case)
    values=candidate_values(row,method)
    flags=[threshold is not None and v is not None and v>threshold for v in values]
    if case['kind']=='semi_synthetic49':
        geometry=evaluate(row['search_matches'],case['truth'])
        pairs=geometry['strict']['pairs']
        mapping={p['prediction_index']:p['truth_index'] for p in pairs}
        positives={i for i,t in enumerate(case['truth']) if t['color_anomaly']}
        found={mapping[i] for i,flag in enumerate(flags) if flag and i in mapping and mapping[i] in positives}
        fp=sum(flag and (i not in mapping or mapping[i] not in positives) for i,flag in enumerate(flags))
        return dict(geometry=geometry,threshold=threshold,color_tp=len(found),color_fn=len(positives-found),
                    color_fp=int(fp),color_flagged=sum(flags),invalid_color_candidates=sum(v is None for v in values),
                    note='End-to-end color TP requires strict pose pairing; missing geometry stays a color FN.')
    value=image_value(row,method)
    uncalibrated=threshold is None
    actual=case['color_anomaly']; predicted=not uncalibrated and value is not None and value>threshold
    out=dict(image_value=value,threshold=threshold,actual_anomaly=actual,predicted_anomaly=predicted,
             uncalibrated=uncalibrated,abstained=uncalibrated or value is None,
             abstention_reason='uncalibrated' if uncalibrated else ('no_valid_candidates' if value is None else None),
             tp=int(actual and predicted),fn=int(actual and not predicted),
             fp=int(not actual and predicted),tn=int(not actual and not predicted and value is not None and not uncalibrated),
             flagged_candidates=sum(flags),candidate_count=len(flags),
             invalid_color_candidates=sum(v is None for v in values))
    if actual:
        mask=cv2.imread(case['mask_gt'],cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim!=2:
            raise ValueError('Invalid evaluation mask')
        ys,xs=np.where(mask==2)
        if not len(xs):
            raise ValueError('No discolor pixels')
        fractions=[]
        for match in row['search_matches']:
            # Used AFTER matching. Never supplies proposals or ROI pixels.
            poly=np.asarray(match['corners'],np.float32)
            inside=sum(cv2.pointPolygonTest(poly,(float(x),float(y)),False)>=0 for x,y in zip(xs,ys))
            fractions.append(inside/len(xs))
        out.update(discolor_pixels=len(xs),max_detection_discolor_coverage=max(fractions,default=0),
            localized_color_tp=bool(any(flag and coverage>=.5 for flag,coverage in zip(flags,fractions))),
            flagged_without_discolor_overlap=sum(flag and coverage==0 for flag,coverage in zip(flags,fractions)),
            localization_rule='A flagged detected polygon covers at least 50% of official id=2 pixels; not a predicted defect mask.')
    return out


def distribution(xs):
    values=list(map(float,xs))
    if not values or not all(math.isfinite(v) and v>=0 for v in values):
        raise ValueError('Invalid measurements')
    return dict(median=statistics.median(values),min=min(values),max=max(values),values=values)


def calibration_names(cases):
    return [c['name'] for c in cases if c['kind']=='visa_real' and c['split']=='pilot'
            and not c['color_anomaly'] and not c['template_reference']]


def sequence(names,blocks):
    return [(name,b,m) for b in range(blocks) for name in (names if b%2==0 else names[::-1])
            for m in (METHODS if b%2==0 else METHODS[::-1])]


def load_worker(base,entry):
    for field in ('job','raw','stderr'):
        if sha(base/entry[field])!=entry[field+'_sha256']:
            raise ValueError('Altered worker '+field)
    if entry['returncode']:
        raise ValueError('Failed worker')
    return json.loads((base/entry['raw']).read_text())


def calibrate(cases,entries,base):
    names=calibration_names(cases)
    if len(names)!=3:
        raise ValueError('Require exactly three independent normal calibration images')
    result=dict(rule='max of valid normal image maxima plus 1e-6; strict greater-than; fewer than three valid images provisional; none uncalibrated',
                names=names,methods={},uses_anomalous_labels=False,uses_heldout=False)
    for method in METHODS:
        values=[]
        for name in names:
            entry=next(e for e in entries if e['case']==name and e['block']==0 and e['method']==method)
            value=image_value(load_worker(base,entry)['first'],method)
            values.append(value)
        valid_images=[name for name,value in zip(names,values) if value is not None]
        abstained_images=[name for name,value in zip(names,values) if value is None]
        valid_values=[value for value in values if value is not None]
        result['methods'][method]=dict(normal_image_values=values,
            threshold=max(valid_values)+1e-6 if valid_values else None,
            valid_images=valid_images,abstained_images=abstained_images,
            valid_image_count=len(valid_images),abstained_image_count=len(abstained_images),
            coverage=len(valid_images)/len(names),provisional=len(valid_images)<len(names),
            uncalibrated=not valid_values)
    return result


def summarize(report,base,input_map=None):
    protocol=report['protocol']; cases=protocol['cases']; blocks=protocol['blocks']
    if report.get('post_hashes')!=protocol['hashes'] or report.get('unchanged') is not True:
        raise ValueError('Recorded sources, binaries or inputs changed')
    if blocks<2 or blocks%2 or protocol['methods']!=METHODS or protocol['first_and_one_repeat'] is not True:
        raise ValueError('Unexpected protocol semantics')
    if not report['complete'] or sha(base/'protocol.json')!=report['protocol_sha256']:
        raise ValueError('Incomplete or altered protocol')
    if json.loads((base/'protocol.json').read_text())!=protocol:
        raise ValueError('Protocol mismatch')
    names=calibration_names(cases)
    expected=sequence(names,blocks)+sequence([c['name'] for c in cases if c['name'] not in names],blocks)
    if [(e['case'],e['block'],e['method']) for e in report['entries']]!=expected:
        raise ValueError('Missing/reordered workers')
    calibration=calibrate(cases,report['entries'],base)
    if sha(base/'calibration.json')!=report['calibration_sha256'] or calibration!=json.loads((base/'calibration.json').read_text()):
        raise ValueError('Calibration differs from pre-heldout threshold file')
    if report['calibration_completed_workers']!=len(names)*blocks*len(METHODS):
        raise ValueError('Calibration timing was not recorded before evaluation')
    raw={}
    for entry in report['entries']:
        case=next(c for c in cases if c['name']==entry['case'])
        worker=load_worker(base,entry)
        if json.loads((base/entry['job']).read_text())!=job(case,entry['method']):
            raise ValueError('Unexpected worker job')
        expected_job=job(case,entry['method'])
        if any(worker[k]!=expected_job[k] for k in ('method','backend','threads')) or worker['search_config']!=expected_job['search']:
            raise ValueError('Worker backend/configuration differs')
        if not worker['integrated_search'] or len(worker['runs'])!=1 or worker['warmups_ms']:
            raise ValueError('Unexpected measurement scope')
        if worker['exact_repeat'] != all(worker['first'][k]==worker['runs'][0][k] for k in ('scores','search_matches')):
            raise ValueError('Incorrect repeat flag')
        raw[entry['case'],entry['block'],entry['method']]=worker
    result=[]
    for case in cases:
        evaluation_case=dict(case)
        if 'mask_gt' in case:
            mask=Path(input_map.get(case['mask_gt'],case['mask_gt']) if input_map else case['mask_gt'])
            if sha(mask)!=case['input_sha256']['mask_gt'] or protocol['hashes'][case['mask_gt']]!=sha(mask):
                raise ValueError('Evaluation mask differs from frozen source')
            evaluation_case['mask_gt']=str(mask)
        methods={};pairs=[]; outputs_same=True; rgb_error=0.
        for method in METHODS:
            rows=[raw[case['name'],b,method] for b in range(blocks)]
            cutoff=(None if case['kind']=='real_category_probe' else
                    calibration['methods'][method]['threshold'] if case['kind']=='visa_real' else
                    (.1 if method=='chroma' else .2))
            measurements={phase:{metric:distribution([(w['first'] if phase=='first' else w['runs'][0])[metric] for w in rows])
                for metric in ('pipeline_ms','search_ms','verification_ms')}
                for phase in ('first','repeat')}
            a=accuracy(rows[0]['first'],method,evaluation_case,cutoff)
            methods[method]=dict(timing=measurements,
                peak_rss_bytes=distribution([w['final_memory']['peak_rss_bytes'] for w in rows]),
                construct_ms=distribution([w['construct_ms'] for w in rows]),
                color_construct_ms=distribution([w['color_construct_ms'] for w in rows]),
                cold_ready_ms=distribution([w['decode_ms']+w['construct_ms']+w['first']['pipeline_ms'] for w in rows]),
                exact_repeat=all(w['exact_repeat'] for w in rows),accuracy=a,
                first_scores=rows[0]['first']['scores'],
                reference_matches=rows[0]['first']['search_matches'],
                cross_process_scores_exact=all(w['first']['scores']==rows[0]['first']['scores'] for w in rows))
            methods[method]['cross_process_geometry_exact']=all(w['first']['search_matches']==rows[0]['first']['search_matches'] for w in rows)
            for b,w in enumerate(rows):
                ref=raw[case['name'],b,'geometry']
                for phase in ('first','repeat'):
                    rr=ref['first'] if phase=='first' else ref['runs'][0]
                    wr=w['first'] if phase=='first' else w['runs'][0]
                    if method in SHARED_GEOMETRY_METHODS:
                        outputs_same &= wr['search_matches']==rr['search_matches']
                    if method!='geometry':
                        pairs.append(dict(block=b,phase=phase,method=method,reference_method='geometry',
                            shared_geometry=method in SHARED_GEOMETRY_METHODS,
                            pipeline_delta_percent=100*(wr['pipeline_ms']/rr['pipeline_ms']-1),
                            time_gate=wr['pipeline_ms']<=rr['pipeline_ms']+max(2.,.1*rr['pipeline_ms']),
                            rss_gate=w['final_memory']['peak_rss_bytes']<=ref['final_memory']['peak_rss_bytes']+
                                     max(16*2**20,.1*ref['final_memory']['peak_rss_bytes'])))
        for b in range(blocks):
            a=raw[case['name'],b,'rgb_cpu']['first']['scores']
            c=raw[case['name'],b,'rgb_metal']['first']['scores']
            rgb_error=max(rgb_error,max((abs(x-y) for x,y in zip(a,c)),default=0))
        result.append(dict(name=case['name'],kind=case['kind'],split=case['split'],methods=methods,pairs=pairs,
            geometry_exact_between_methods=outputs_same,geometry_comparison_methods=SHARED_GEOMETRY_METHODS,
            rgb_max_score_error=rgb_error,rgb_pair_within_3e_minus5=rgb_error<3e-5,
            reference_matches=raw[case['name'],0,'geometry']['first']['search_matches']))
    return dict(complete=True,workers=len(report['entries']),algorithm_calls=2*len(report['entries']),
        blocks=blocks,thresholds=calibration,cases=result,source_unchanged=report['unchanged'],
        integrity_passed=report['unchanged'] and all(c['geometry_exact_between_methods'] and c['rgb_pair_within_3e_minus5'] and
            all(m['exact_repeat'] and m['cross_process_scores_exact'] and m['cross_process_geometry_exact'] for m in c['methods'].values()) for c in result),
        performance_failures=[dict(case=c['name'],**p) for c in result for p in c['pairs'] if not p['time_gate'] or not p['rss_gate']],
        limitations=['Real FSC category coverage is not same-template identity accuracy.',
            'VisA heldout has only three normal and three mixed bubble/discolor images.',
            'Calibration with fewer than three valid normal images is provisional; none means abstention, not a normal true negative.',
            'geometry_original uses the earlier search budget; its cost/accuracy comparison is not a color-verification-only change.',
            'Four conditions reuse each semi-synthetic layout; repeated workers do not increase independent image count.',
            'All searches are gray; candidate color verification cannot recover missed geometry.',
            'Chroma uses a reference support mask; these are useful-method comparisons, not equal-kernel-work comparisons.',
            'Process peak RSS includes RGB decode, search, verifier, driver and allocator. Shared Metal bytes must not be added again.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path)
    p.add_argument('--build',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--summarize',type=Path)
    p.add_argument('--input-map',type=Path,help='Offline map from original evaluation-mask paths to paths relative to this map file')
    args=p.parse_args()
    if args.summarize:
        mapping={k:str(args.input_map.parent/v) for k,v in json.loads(args.input_map.read_text()).items()} if args.input_map else None
        summary=summarize(json.loads(args.summarize.read_text()),args.summarize.parent,mapping)
        save(args.output,summary); return
    if args.blocks<2 or args.blocks%2:
        raise ValueError('Need even number of independent paired blocks')
    dest=args.output.resolve()
    if dest.exists(): raise ValueError('Refusing to overwrite evidence')
    dest.mkdir(parents=True)
    cases=json.loads(args.cases.read_text())
    for case in cases:
        for key,digest in case['input_sha256'].items():
            if sha(case[key])!=digest:
                raise ValueError('Input changed since scene freeze: '+case['name']+'/'+key)
    runner=args.build.resolve()/'experiments/fpm_color_verify_runner'
    files=[*source_files(),runner,args.cases.resolve()]
    files.extend(p for p in (args.build.resolve()/'experiments').glob('*.dylib'))
    files.extend(Path(c[k]) for c in cases for k in ('source','template','mask','mask_gt') if k in c)
    hashes={str(f.resolve()):sha(f) for f in set(files)}
    report=dict(protocol=dict(version=2,platform=platform.platform(),cases=cases,blocks=args.blocks,
        methods=METHODS,first_and_one_repeat=True,serial_fresh_processes=True,hashes=hashes,
        threshold_rule='VisA max valid normal image maximum + 1e-6 before heldout; <3 valid provisional, none uncalibrated; controls NCC loss>.2 or chroma fraction>.1; category probes need no threshold',
        time_gate='max(10%,2ms)',rss_gate='max(10%,16MiB)',
        timing='Same-process gray search plus color verification. Decode and construction separate; no JSON, evaluation or image writing.'),
        entries=[],complete=False)
    save(dest/'protocol.json',report['protocol']);report['protocol_sha256']=sha(dest/'protocol.json')
    names=calibration_names(cases)
    phases=[sequence(names,args.blocks),sequence([c['name'] for c in cases if c['name'] not in names],args.blocks)]
    for phase_index,order in enumerate(phases):
        for name,block,method in order:
            case=next(c for c in cases if c['name']==name)
            stem=f'{name}-{method}-{block}'
            jp=dest/'workers'/f'{stem}.job.json';rp=jp.with_suffix('.raw.json');ep=jp.with_suffix('.stderr.txt')
            save(jp,job(case,method))
            child=subprocess.run([str(runner),str(jp)],capture_output=True,text=True,timeout=300)
            rp.write_text(child.stdout);ep.write_text(child.stderr)
            entry=dict(case=name,block=block,method=method,returncode=child.returncode)
            for key,path in [('job',jp),('raw',rp),('stderr',ep)]:
                entry[key]=str(path.relative_to(dest));entry[key+'_sha256']=sha(path)
            report['entries'].append(entry);save(dest/'formal.json',report)
            if child.returncode: raise RuntimeError(stem+': '+child.stderr)
            record=json.loads(child.stdout)
            print(stem,round(record['runs'][0]['pipeline_ms'],3),record['pose_count'],flush=True)
        if phase_index==0:
            save(dest/'calibration.json',calibrate(cases,report['entries'],dest))
            report['calibration_sha256']=sha(dest/'calibration.json')
            report['calibration_completed_workers']=len(report['entries'])
            save(dest/'formal.json',report)
    report['post_hashes']={f:sha(f) for f in hashes}
    report.update(complete=True,unchanged=hashes==report['post_hashes'])
    save(dest/'formal.json',report)
    summary=summarize(report,dest);save(dest/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','performance_failures')},indent=2))


if __name__=='__main__': main()
