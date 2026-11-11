"""Evaluate a darker printed neutral reference on the stage 47 synthetic holdout.

Use the archived searched poses, the original artwork thresholds and the same C++
colour verifier as the desktop. Truth is used only to score a completed decision.
"""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from many_object_accuracy import evaluate
from scene_tuning import run_job, save


def measure(args, tmp):
    thresholds = json.loads(args.thresholds.read_text())['thresholds']['catch']
    rows = []
    for family in ('red','teal'):
        template = args.scenes/f'syn_{family}_template.png'
        rgb = cv2.imread(str(template)).astype(np.int16)
        colour = args.masks/f'{family}_colour.png'
        primary = args.masks/f'{family}_reference.png'
        # Choose only dark, nearly neutral ink, away from region edges.
        dark = ((np.ptp(rgb,axis=2)<20) & (rgb.mean(2)>35) & (rgb.mean(2)<90)
                & (cv2.imread(str(colour),0)==0) & (cv2.imread(str(primary),0)==0))
        dark = cv2.erode(dark.astype(np.uint8)*255,np.ones((3,3),np.uint8))
        if not dark.any() or not cv2.imwrite(str(tmp/'dark.png'),dark):
            raise ValueError('No usable dark template region')
        for truth_path in sorted(args.scenes.glob(f'syn_{family}_*.truth.json')):
            metadata = json.loads(truth_path.read_text())
            name = metadata['scene']
            prior = json.loads((args.search/f'r47-{name}-vk_object.json').read_text())
            matches = prior['first']['search_matches']
            height,width = metadata['source_shape']
            poses = [[p['pos_x']-width/2,height/2-p['pos_y'],p['angle'],math.log(p['scale'])] for p in matches]
            job = dict(source=str((args.scenes/metadata['image']).resolve()),template=str(template.resolve()),
                       method='chroma_vk',backend='cpu',mask=str(colour.resolve()),poses=poses,warmups=0,repeats=1)
            def score(mask):
                return run_job(args.runner,dict(job,reference_mask=str(mask.resolve())),tmp/'job.json')['first']['scores']
            first, second = score(primary), score(tmp/'dark.png')
            assigned = evaluate(matches,metadata['truth'])['strict']
            if assigned['tp'] != len(matches) or assigned['fn']:
                raise ValueError('Archived poses no longer match scene')
            row = dict(scene=name, dark_template_pixels=int(np.count_nonzero(dark)))
            for label in ('primary','fallback'):
                tally = dict(total=len(matches),review=0,normal_accepted=0,normal_rejected=0,
                             must_catch_detected=0,must_catch_missed=0,secondary_used=0)
                for pair in assigned['pairs']:
                    i=pair['prediction_index']
                    truth=metadata['truth'][pair['truth_index']]
                    s=first[i]
                    if label=='fallback' and s['status']==3:
                        s=second[i]
                        tally['secondary_used']+=1
                    if s['status']!=0:
                        tally['review']+=1
                        continue
                    above=s['region_distance']>thresholds[f'syn_{family}']
                    if truth['same_colour']:
                        tally['normal_rejected' if above else 'normal_accepted']+=1
                    elif truth['delta_e00_target']>=4:
                        tally['must_catch_detected' if above else 'must_catch_missed']+=1
                row[label]=tally
            rows.append(row)
    return dict(thresholds=thresholds, rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('scenes','masks','search','thresholds','runner','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    with TemporaryDirectory() as tmp:
        save(args.output,measure(args,Path(tmp)))
