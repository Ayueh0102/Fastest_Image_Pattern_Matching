"""Validate Blender camera GT independently and freeze cropped-template jobs.

No matcher is invoked. Reference and source RGB renders become grayscale using
one documented OpenCV conversion. Perspective/depth rotations are excluded.
"""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path

import cv2
import numpy as np

from benchmark_fine_strip import save,sha
from challenge_coverage_benchmark import METHODS

ROOT=Path(__file__).resolve().parents[2]
NAMES=('blender_neutral','blender_changed_light','blender_occlusion','blender_distractors')
TOLERANCE=1e-4


def analytic_corners(item,width=96,height=72):
    theta=math.radians(item['angle']);c,s=math.cos(theta),math.sin(theta)
    result=[]
    for x,y in ((-width/2,-height/2),(width/2,-height/2),(width/2,height/2),(-width/2,height/2)):
        result.append([item['pos_x']+item['scale']*(c*x+s*y),
                       item['pos_y']+item['scale']*(-s*x+c*y)])
    return np.asarray(result)


def check_geometry(metadata):
    if metadata['projection']!='orthographic frontal' or not metadata['perspective_out_of_model_excluded']:
        raise ValueError('Only orthographic frontal GT is supported')
    max_corner=max_center=0.
    for item in metadata['objects']:
        if item['identity'] not in ('target','altered_patch') or item['scale']<=0:
            raise ValueError('Invalid identity or scale')
        expected=analytic_corners(item)
        actual=np.asarray(item['corners'])
        error=float(np.max(np.abs(expected-actual)))
        center_error=float(np.max(np.abs(np.asarray(item['projected_center'])-[item['pos_x'],item['pos_y']])))
        if error>TOLERANCE or center_error>TOLERANCE:
            raise ValueError('Camera-projected geometry disagrees with independent analytic transform')
        if (expected<0).any() or (expected>512).any():
            raise ValueError('Clipped plane')
        max_corner=max(max_corner,error);max_center=max(max_center,center_error)
    return dict(corner_max_error_px=max_corner,center_max_error_px=max_center,tolerance_px=TOLERANCE,
        independent_formula='x=cx+s*(cos(theta)*u+sin(theta)*v); y=cy+s*(-sin(theta)*u+cos(theta)*v)')


def gray(path):
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(512,512):
        raise ValueError('Expected 512x512 RGB render')
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


def prepare(rendered,output,results):
    rendered,output,results=(Path(p).resolve() for p in (rendered,output,results))
    for path in (output,results):
        if path.exists() and any(path.iterdir()):
            raise ValueError('Refusing to overwrite frozen evidence: '+str(path))
    generation=json.loads((rendered/'generation.json').read_text())
    if generation['script_sha256']!=sha(ROOT/'cpp-core/tools/blender_challenge_render.py'):
        raise ValueError('Rendered generator source differs')
    checks={};metas={}
    for name in ('reference',*NAMES):
        data=json.loads((rendered/(name+'.json')).read_text())
        if sha(rendered/(name+'.blend'))!=data['blend_sha256'] or sha(rendered/(name+'-rgb.png'))!=data['rgb_sha256']:
            raise ValueError('Changed raw scene/render')
        checks[name]=check_geometry(data);metas[name]=data
    reference=metas['reference']['objects'][0]
    if reference['corners']!=[[208.,220.],[304.,220.],[304.,292.],[208.,292.]]:
        raise ValueError('Reference plane does not equal declared crop')
    output.mkdir(parents=True,exist_ok=True);results.mkdir(parents=True,exist_ok=True)
    template=gray(rendered/'reference-rgb.png')[220:292,208:304].copy()
    template_path=output/'reference-template.png'
    if not cv2.imwrite(str(template_path),template):raise RuntimeError('Template save failed')
    cases=[]
    for name in NAMES:
        source=output/(name+'.png');metadata_path=output/(name+'.json')
        image=gray(rendered/(name+'-rgb.png'))
        if not cv2.imwrite(str(source),image):raise RuntimeError('Source save failed')
        data=copy.deepcopy(metas[name])
        for item in data['objects']:
            # Exact planar support fraction under the rectangular occluder;
            # shadows and pixel filter support are photometric, not extra GTs.
            item['visible_content_fraction']=1-item['nominal_strip_occlusion_fraction']
        data['truth']=[o for o in data['objects'] if o['identity']=='target']
        data['distractors']=[o for o in data['objects'] if o['identity']!='target']
        data.update(geometry_check=checks[name],source_sha256=sha(source),template_sha256=sha(template_path),
            grayscale='OpenCV BGR2GRAY on saved 8-bit sRGB Standard render',
            template_definition='96x72 pixel crop of neutral reference plane, crop xywh208,220,96,72',
            split='development' if name=='blender_neutral' else 'heldout',
            geometry_group='development_6' if name=='blender_neutral' else 'heldout_6',
            lighting_scope='Physically shaded coplanar texture cards; no out-of-plane or perspective robustness claim')
        save(metadata_path,data)
        cases.append(dict(name=name,source=str(source),template=str(template_path),metadata=str(metadata_path),
            split=data['split'],kind=name.removeprefix('blender_'),truth=data['truth'],distractors=data['distractors'],
            scales=[.75,1.,1.25],angle=180,limit=70,proposal_score=.5,score=.8,overlap=.2,
            stop_layer=1,subpixel=0,workflow='full_rescue'))
    save(output/'cases.json',cases)
    assets={str(p):sha(p) for p in rendered.iterdir() if p.suffix in ('.json','.png','.blend')}
    # Include full render assets as frozen inputs, in addition to the worker's
    # smaller PNGs. The common runner hashes every one before/after execution.
    inputs={**assets,**{c[k]:sha(c[k]) for c in cases for k in ('source','template','metadata')}}
    source_paths=[ROOT/'cpp-core/tools'/name for name in (
        'blender_challenge_render.py','blender_challenge_prepare.py','challenge_coverage_benchmark.py',
        'many_object_accuracy.py','benchmark_native_pipeline.py','benchmark_fine_strip.py')]
    protocol=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cases=cases,methods=METHODS,blocks=2,repeats=1,warmup_seconds=0,
        order=[dict(case=c['name'],block=b,method=m) for c in cases for b in range(2)
               for m in (list(METHODS) if b==0 else list(METHODS)[::-1])],
        inputs_sha256=inputs,cases_sha256=sha(output/'cases.json'),source_sha256={str(p):sha(p) for p in source_paths},
        accuracy_gate=dict(position_px=2,angle_deg=1,relative_scale=.02),
        performance_gate=dict(relative=.1,absolute_ms=2.,rss_relative=.1,rss_absolute_bytes=16*1024**2),
        remedy='Only proposal threshold .5 to .4 and per-scale cap70 to140; all final/refinement/overlap gates unchanged.',
        geometry_validation=checks,independent_geometric_poses=12,
        heldout_policy='All renderer scene/geometry/lighting and both methods frozen before matcher observation.',
        memory_scope='Native peak RSS through final_memory snapshot; excludes later final JSON encoding/output peaks.',
        limitations=['Rendered physically shaded planar cards, not actual captured objects.',
            'Same heldout six-pose geometry reused by three conditions; not eighteen independent poses.',
            'Perspective, curvature and out-of-plane rotations excluded from this 2D model.',
            'Material identity-negative has a local inverted patch; generic resemblance may still produce high NCC.',
            'Two fresh processes per case/method; no confidence interval or significance claim.'])
    save(results/'protocol.json',protocol)
    save(results/'generation-validation.json',dict(passed=True,geometry=checks,render_assets=len(assets),
        reference_crop_pixel_sha256=__import__('hashlib').sha256(template.tobytes()).hexdigest(),
        reference_crop=[208,220,96,72],initial_projection_failure_retained='rendered-projection-check-rejected',
        max_corner_error_px=max(c['corner_max_error_px'] for c in checks.values()),
        observed_matching_before_freeze=False))
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rendered',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();prepare(args.rendered,args.output,args.results)
    print(args.results/'protocol.json')

if __name__=='__main__':main()
