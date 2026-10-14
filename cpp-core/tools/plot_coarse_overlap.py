"""Plot saved serial/overlap measurements; no matching or GPU work."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_pipeline_stages import load_worker

LABELS={'controlled6':'6 rotated objects / 3 scales','heldout12':'12 rotated objects / 3 scales',
'8k_t256_angle180_3scales':'8K / full angle / 3 scales','small_fullres':'Small / full angle / subpixel',
'many128_pose':'128 rotated objects / 4096 x 4096','many512_pose':'512 rotated objects / 8192 x 4096'}
METHODS=[('previous_peak','Frozen previous peak tree','#7e8996'),('current_peak','New build, synchronous','#b0bac6'),('overlap','New build, CPU/GPU overlap','#398d98')]


def render(report,path,out,worker_root=None):
    fig,axes=plt.subplots(1,3,figsize=(17,7),sharey=True,gridspec_kw={'width_ratios':[1.1,1,1]})
    cases=report['cases'];y=np.arange(len(cases))
    for i,(m,label,color) in enumerate(METHODS):
        totals=[c['methods'][m]['measured']['total_wall_ms']['median'] for c in cases]
        mem=[max(c['methods'][m]['peak_rss_bytes'])/2**20 for c in cases]
        axes[0].barh(y+(i-1)*.22,totals,.20,color=color,label=label)
        axes[1].barh(y+(i-1)*.22,mem,.20,color=color)
    for i,(m,label,color) in enumerate(METHODS[1:]):
        med=[]
        for case in cases:
            values=[]
            for entry in report['execution_order']:
                if entry['case']==case['case']['name'] and entry['method']==m+'_profile':
                    _,_,worker=load_worker(entry,path,worker_root)
                    values += [sum(p['metrics']['wait_ms'] for p in r['patterns']) for r in worker['runs']]
            med.append(statistics.median(values))
        axes[2].barh(y+(i-.5)*.28,med,.25,color=color)
    axes[0].set_yticks(y,[LABELS[c['case']['name']] for c in cases]);axes[0].invert_yaxis()
    axes[0].set_xscale('log');axes[0].set_xlabel('Total median (ms, log scale)')
    axes[1].set_xlabel('Process peak RSS (MiB)')
    axes[2].set_xlabel('Coarse host wait (ms, profile runs)')
    for ax in axes:
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two-slot coarse scheduling: same results, less waiting, additional buffers',fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.56,.045),frameon=False)
    fig.text(.56,.018,'M4 Pro. Serial GPU queue; CPU work overlaps submitted GPU batches. Wait and device time are not additive.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94));out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('output',type=Path);p.add_argument('--worker-root',type=Path)
    a=p.parse_args();render(json.loads(a.report.read_text()),a.report,a.output,a.worker_root)
