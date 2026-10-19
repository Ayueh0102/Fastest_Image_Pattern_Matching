"""Render saved many-object measurements and detection overlays; no matching."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_pipeline_stages import load_worker

LABELS = {'many64_pose':'64 / 4096 x 4096 / pose',
          'many128_pose':'128 / 4096 x 4096 / pose',
          'many256_pose':'256 / 4096 x 4096 / pose',
          'many128_upright':'128 / 4096 x 4096 / upright',
          'many512_pose':'512 / 8192 x 4096 / pose'}
METHODS = [('previous_shared','Previous shared pipeline','#7e8996'),
           ('current_control','Current build, original peaks','#b0bac6'),
           ('peak_only','Current build, peak tree','#398d98')]


def performance(report, output):
    cases=report['cases'];y=np.arange(len(cases))
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharey=True,gridspec_kw={'width_ratios':[1.3,1,.8]})
    for i,(method,label,color) in enumerate(METHODS):
        med=[c['methods'][method]['measured']['total_wall_ms']['median'] for c in cases]
        axes[0].barh(y+(i-1)*.24,med,.22,color=color,label=label)
        mem=[max(c['methods'][method]['peak_rss_bytes'])/2**20 for c in cases]
        axes[1].barh(y+(i-1)*.24,mem,.22,color=color)
    for i,c in enumerate(cases):
        a=c['independent_accuracy']['peak_only']['first']['strict']
        axes[2].barh(i,a['recall']*100,.5,color='#398d98')
        axes[2].text(2,i,f"{a['tp']}/{c['case']['object_count']}  FP {a['fp']}",va='center',color='white')
    axes[0].set_yticks(y,[LABELS[c['case']['name']] for c in cases]); axes[0].invert_yaxis()
    axes[0].set_xscale('log');axes[0].set_xlabel('Median total time (ms, log scale)')
    axes[1].set_xlabel('Process peak RSS (MiB)')
    axes[2].set_xlabel('Strict pose recall (%)');axes[2].set_xlim(0,105)
    for ax in axes:
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Many instances of one template: speed, memory and pose accuracy',fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.55,.045),frameon=False)
    fig.text(.55,.018,'Pose = random rotation + scale 0.72–1.28; no occlusion. 512 changes image size and search cap.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(output/'performance.png',dpi=150);plt.close(fig)


def overlay(case, record, accuracy, output, inputs=None):
    source=Path(case['source']) if inputs is None else inputs/Path(case['source']).name
    image=cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError('Missing scene: '+str(source))
    ratio=min(1.,2200/image.shape[1]); display=cv2.resize(image,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
    display=cv2.cvtColor(display,cv2.COLOR_GRAY2BGR)
    a=accuracy['strict']; paired={p['prediction_index'] for p in a['pairs']}
    for i,m in enumerate(record['matches']):
        color=(110,225,40) if i in paired else (0,145,255)
        points=np.rint(np.asarray(m['corners'])*ratio).astype(np.int32)
        cv2.polylines(display,[points],True,color,2,cv2.LINE_AA)
    for i in a['unmatched_truth_indices']:
        t=case['truth'][i];points=np.rint(np.asarray(t['corners'])*ratio).astype(np.int32)
        cv2.polylines(display,[points],True,(45,45,235),3,cv2.LINE_AA)
        cv2.putText(display,str(i),tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,255),1,cv2.LINE_AA)
    canvas=cv2.copyMakeBorder(display,72,0,0,0,cv2.BORDER_CONSTANT,value=(248,248,248))
    title=f"{case['object_count']} objects | TP {a['tp']} / FP {a['fp']} / FN {a['fn']} | 2px, 1deg, 2% scale"
    cv2.putText(canvas,title,(18,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(canvas,'Green: matched predictions   Red: unmatched truth   Orange: unmatched predictions',
                (18,57),cv2.FONT_HERSHEY_SIMPLEX,.52,(70,70,70),1,cv2.LINE_AA)
    cv2.imwrite(str(output/(case['name']+'-detections.png')),canvas)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--worker-root',type=Path);parser.add_argument('--inputs',type=Path)
    args=parser.parse_args();report=json.loads(args.report.read_text());args.output.mkdir(parents=True,exist_ok=True)
    performance(report,args.output)
    for case in report['cases']:
        if case['case']['name'] not in ('many128_pose','many256_pose','many512_pose'):continue
        entry=next(e for e in report['execution_order'] if e['case']==case['case']['name'] and e['method']=='peak_only' and e['block']==0)
        _,_,worker=load_worker(entry,args.report,args.worker_root)
        overlay(case['case'],worker['first'],case['independent_accuracy']['peak_only']['first'],args.output,args.inputs)


if __name__=='__main__':main()
