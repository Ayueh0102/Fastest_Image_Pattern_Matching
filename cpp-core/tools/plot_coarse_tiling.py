"""Render saved NCC tiling results without running matching or GPU work."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'controlled6':'6 rotated objects / 3 scales', 'heldout12':'12 rotated objects / 3 scales',
          '8k_t256_angle180_3scales':'8K / full angle / 3 scales', 'small_fullres':'Small / full angle / subpixel',
          'many128_pose':'128 objects / 4096 x 4096', 'many512_pose':'512 objects / 8192 x 4096'}
METHODS = [('current_peak','Original NCC','#99a6b6'), ('tile_direct','4-output register strip','#268d85'),
           ('tile_shared','8x8 staged tile','#bd8854')]


def pipeline(report, out):
    cases = report['cases']; y=np.arange(len(cases))
    fig, axes = plt.subplots(1,3,figsize=(17,7),sharey=True)
    for i,(name,label,color) in enumerate(METHODS):
        values=[c['methods'][name]['measured']['total_wall_ms']['median'] for c in cases]
        device=[c['coarse_diagnostics'][name+'_profile']['device_ms']['median'] for c in cases]
        rss=[max(c['methods'][name]['peak_rss_bytes'])/2**20 for c in cases]
        for ax,v in zip(axes,(values,device,rss)):
            ax.barh(y+(i-1)*.23,v,.21,label=label,color=color)
    axes[0].set_yticks(y,[LABELS[c['case']['name']] for c in cases]);axes[0].invert_yaxis()
    axes[0].set_xscale('log');axes[0].set_xlabel('Whole pipeline median (ms, log)')
    axes[1].set_xlabel('Coarse GPU duration (ms, profile)')
    axes[2].set_xlabel('Process peak RSS (MiB)')
    for ax in axes:
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('NCC spatial reuse: same outputs and search workload',fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.56,.045),frameon=False)
    fig.text(.56,.018,'M4 Pro. 2 order blocks x 5 calls; separate profiling. All coarse overlap disabled.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(out,dpi=150);plt.close(fig)


def probe(report, out):
    cases=report['summary']['cases']; names=list(cases); y=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    for i,(mode,(_,label,color)) in enumerate(zip(('0','2','3'),METHODS)):
        for ax,key in zip(axes,('device_ms','wall_ms')):
            ratios=[cases[n]['0']['times'][key]['median']/cases[n][mode]['times'][key]['median'] for n in names]
            ax.barh(y+(i-1)*.23,ratios,.21,color=color,label=label)
    dims=report['design']['cases']
    axes[0].set_yticks(y,[f'{n}  (T={dims[n][2]}x{dims[n][3]})' for n in names]);axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(1,color='#333',linewidth=.8);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlabel('GPU duration speed ratio (original / candidate)')
    axes[1].set_xlabel('Single-map host wall speed ratio')
    fig.suptitle('Isolated NCC map probe: above 1 is faster; this is not full search',fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.57,.035),frameon=False)
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(out,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--probe',action='store_true');a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    (probe if a.probe else pipeline)(json.loads(a.report.read_text()),a.output)
