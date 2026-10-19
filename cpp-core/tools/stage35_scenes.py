"""Freeze real RGB probes and explicitly semi-synthetic 49-object controls.

FSC centers are category annotations, never same-template identity ground truth.
No GT or masks from a test source enter the search jobs. The renderer is OpenCV;
the matcher and color verification are the separately implemented C++ kernels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear(rgb):
    x=rgb.astype(np.float64)/255.
    return np.where(x<=.04045, x/12.92, ((x+.055)/1.055)**2.4)


def srgb(values):
    x=np.clip(values,0,1)
    return np.uint8(np.clip(np.rint(255*np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)),0,255))


def crop(row, dest):
    image=cv2.imread(row['image'])
    box=np.asarray(row['annotation']['box_examples_coordinates'][0])
    # FSC exemplar quadrilaterals are x,y, already in resized image coordinates.
    x0,y0=np.floor(box.min(axis=0)).astype(int)
    x1,y1=np.ceil(box.max(axis=0)).astype(int)
    patch=image[y0:y1,x0:x1].copy()
    if min(patch.shape[:2])<8:
        raise ValueError('Template too small')
    path=dest/f"fsc-{row['id']}-template.png"
    cv2.imwrite(str(path),patch)
    return path,patch,[int(x0),int(y0),int(x1),int(y1)]


def search_settings(scales, score=.8):
    return dict(backend=1,refinement_backend=1,scales=scales,angle=180,
        limit=300,score=score,proposal_score=.25,overlap=.2,min_area=512,
        stop_layer=1,subpixel=False,cpu_threads=8,pose_resident_mode=0,
        pose_storage_mode=1,pose_input_mode=1,ncc_mode=2,
        fine_ncc_mode=2,fine_fused=True,fine_angle_reuse=True)


def visa_cases(manifest_path, output):
    manifest=json.loads(manifest_path.read_text())
    root=manifest_path.parent
    reference=manifest['template_reference']
    image=cv2.imread(str(root/reference['image']))
    x,y,w,h=reference['bbox_xywh']
    templ=image[y:y+h,x:x+w].copy()
    path=output/'visa-template.png'; mask=output/'visa-template-mask.png'
    cv2.imwrite(str(path),templ)
    # Foreground support derives ONLY from the normal reference crop. Neutral
    # background and white highlights are excluded before reading any test mask.
    values=templ.astype(np.int16)
    foreground=((values.max(2)-values.min(2)>25)&(values[:,:,1]>values[:,:,0]+10))
    foreground=cv2.erode(foreground.astype(np.uint8)*255,np.ones((5,5),np.uint8))
    foreground[:3]=0; foreground[-3:]=0; foreground[:,:3]=0; foreground[:,-3:]=0
    cv2.imwrite(str(mask),foreground)
    cases=[]
    for row in manifest['samples']:
        case=dict(name=row['sample_id'],kind='visa_real',split=row['split'],
            source=str(root/row['image']),template=str(path),mask=str(mask),
            search=search_settings([.85,1,1.15],.65),
            color_anomaly=row['label']!='normal',
            author_defect_labels=row['author_defect_labels'],
            template_reference=row['sample_id']==reference['sample_id'],
            labels='Official image labels and discolor pixels (id=2); no complete instance boxes/poses.',
            template_support='Reference-only saturated foreground, eroded 5x5; no source GT enters verification.')
        if row['mask']:
            case['mask_gt']=str(root/row['mask'])
        cases.append(case)
    return cases


def generate(fsc_manifest, output, visa_manifest=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite frozen cases')
    output.mkdir(parents=True)
    fsc=json.loads(fsc_manifest.read_text())
    byid={r['id']:r for r in fsc['images']}
    cases=[]
    for identifier in ['2933','7417','1913','1924']:
        row=byid[identifier]
        path,_,box=crop(row,output)
        cases.append(dict(name='fsc_'+identifier,kind='real_category_probe',split='descriptive',
            source=row['image'],template=str(path),template_crop_xyxy=box,
            search=search_settings([.75,1,1.25],.65),
            points=row['annotation']['points'],expected_count=row['official_point_count'],
            category=row['category'],point_tolerance_px=10,
            labels='Official category centers; same-template identity and color labels unavailable.'))
    for split,identifier,seed in [('development','2933',2026101701),('heldout','7417',2026101702)]:
        _,raw,box=crop(byid[identifier],output)
        # Explicit intensity compression permits an unclipped RGB offset control.
        templ=np.uint8(np.rint(raw.astype(float)*.65+45))
        h,w=templ.shape[:2]
        template=output/f'{split}-control-template.png'
        mask=output/f'{split}-control-mask.png'
        cv2.imwrite(str(template),templ)
        support=np.zeros((h,w),np.uint8); support[2:-2,2:-2]=255
        cv2.imwrite(str(mask),support)
        cell=int(math.ceil(math.hypot(w,h)))+30
        size=cell*7
        rng=np.random.default_rng(seed)
        objects=[]
        for i in range(49):
            x=(i%7+.5)*cell+float(rng.uniform(-4,4))
            y=(i//7+.5)*cell+float(rng.uniform(-4,4))
            angle=float(rng.uniform(-175,175))
            objects.append(dict(index=i,pos_x=x,pos_y=y,angle=angle,scale=1.))
        for condition in ['clean','one_offset','exposure','white_balance']:
            image=np.full((size,size,3),120,np.float32)
            for obj in objects:
                patch=templ.copy()
                if condition=='one_offset' and obj['index']==24:
                    # BGR delta: gray integer weights yield approximately no
                    # luminance change; per-channel centered NCC cancels offsets.
                    patch=np.uint8(patch.astype(np.int16)+[-2,-15,30])
                elif condition=='exposure':
                    gain=.8+.35*(obj['index']/48)
                    patch=srgb(linear(patch)*gain)
                theta=math.radians(obj['angle'])
                matrix=np.array([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
                affine=np.column_stack((matrix,np.array([obj['pos_x'],obj['pos_y']])-matrix@[w/2,h/2]))
                alpha=cv2.warpAffine(np.ones((h,w),np.float32),affine,(size,size),flags=cv2.INTER_LINEAR)
                pixels=cv2.warpAffine(patch.astype(np.float32),affine,(size,size),flags=cv2.INTER_LINEAR)
                image=image*(1-alpha[:,:,None])+pixels
            image=np.uint8(np.clip(np.rint(image),0,255))
            if condition=='white_balance':
                # Negative robustness control: the scene illumination/camera
                # changes, while physical object identities are still normal.
                image=srgb(linear(image)*[1.25,.94,.9])
            path=output/f'{split}-{condition}.png'
            cv2.imwrite(str(path),image)
            truth=[dict(o,color_anomaly=condition=='one_offset' and o['index']==24) for o in objects]
            cases.append(dict(name=split+'_'+condition,kind='semi_synthetic49',split=split,
                source=str(path),template=str(template),mask=str(mask),truth=truth,
                search=search_settings([1.]),condition=condition,seed=seed,
                expected_count=49,source_fsc_id=identifier,template_crop_xyxy=box,
                labels='Analytic 2D geometry; one prescribed color-offset target or zero identity anomalies.',
                renderer='OpenCV affine bilinear, uint8 rounding; photographed RGB crop compressed to .65*x+45.',
                independence='Four conditions reuse the same 49 poses per template; not independent poses.'))
    if visa_manifest is not None:
        cases.extend(visa_cases(visa_manifest,output))
    for case in cases:
        case['input_sha256']={k:sha(case[k]) for k in ('source','template','mask','mask_gt') if k in case}
    save(output/'cases.json',cases)
    save(output/'provenance.json',dict(fsc_manifest_sha256=sha(fsc_manifest),
        generator_sha256=sha(__file__),opencv=cv2.__version__,numpy=np.__version__,
        recipe_selection='Pilot-selected min_area512/proposal_score.25/limit300; original256/.5/100 retained as a fifth formal method. No heldout tuning.',
        excluded_before_matching=[dict(id='2928',reason='Visual QA: multiple official dots on one visible stamp; category count not reliable instance GT.')],
        photo_policy='FSC photographs and derived RGB controls stay local; fetch and rendering sources are preserved.'))
    print(output/'cases.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fsc-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--visa-manifest',type=Path)
    a=p.parse_args()
    generate(a.fsc_manifest.resolve(),a.output.resolve(),a.visa_manifest.resolve() if a.visa_manifest else None)
