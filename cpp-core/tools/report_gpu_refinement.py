"""Render saved CPU/Metal refinement measurements without rerunning search."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def complete_support_view(directory):
    """Retain provenance while selecting the corrected 1024-pixel fixture."""
    paths = ['large-workloads-clipped.json', 'large-1024-complete-support.json']
    reports = [json.loads((directory/name).read_text()) for name in paths]
    rows, comparisons = [], []
    for index, (name, report) in enumerate(zip(paths, reports)):
        selected = [r for r in report['runs'] if (r['job']['case']['size'] == 1024) == bool(index)]
        names = {r['job']['case']['name'] for r in selected}
        rows.extend(dict(r, measurement_report=name) for r in selected)
        comparisons.extend(r for r in report['comparisons'] if r['case'] in names)
    result = dict(artifact_kind='Derived view of two sequential measurement sessions; not a new benchmark run.',
        provenance=[dict(path=name, sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest()) for name in paths],
        selection='Keep 256 and 768 cases from the first report; use only 2048x2048-source complete-support 1024 cases from the second. Original cropped 1024 failure remains in the first report.',
        protocol=reports[0]['protocol'], runs=rows, comparisons=comparisons,
        correctness_passed=all(r['result']['passed'] for r in rows),
        all_complete_support_seeds_precise=all(r['result']['precise_pass_count']==r['result']['seed_count'] for r in rows))
    if len(rows) != 15 or not result['correctness_passed'] or not result['all_complete_support_seeds_precise']:
        raise ValueError('Complete-support measurements are incomplete or fail verification')
    (directory/'large-workloads.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'cpp-core/docs/gpu-refinement')
    args = parser.parse_args()
    large = complete_support_view(args.data)
    full = json.loads((args.data/'full-image-no-polish.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': '#f6f8fb'})
    specs = {r['job']['case']['name']: r['job']['case'] for r in large['runs']}
    names = sorted(specs, key=lambda n: (specs[n]['size'], specs[n]['channels'], len(specs[n]['initial_poses'])))
    lookup = {(r['job']['case']['name'], r['job']['method'], r['job']['polish']): r['result'] for r in large['runs']}
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    for offset, method, polish, title, color in [(-.25, 'cpu', 0, 'CPU 8 threads', '#425a75'),
            (0, 'accelerated', 0, 'Metal statistics + CPU solve/verification', '#098d8d'),
            (.25, 'accelerated', 2, 'Same + up to 2 CPU polish iterations', '#97b7cd')]:
        values = [lookup[n, method, polish]['algorithm_wall']['median_ms'] for n in names]
        bars = ax.bar(x+offset, values, width=.24, color=color, label=title)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    ax.set_xticks(x, [f"{specs[n]['size']} x {specs[n]['size']} / {'RGB' if specs[n]['channels']==3 else 'gray'}\n{len(specs[n]['initial_poses'])} seeds" for n in names])
    ax.set_ylabel('Median refinement wall time (ms)')
    ax.set_title('Local pose refinement: identical inputs and seeds', loc='left', weight='bold', fontsize=15)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(p.get_height() for p in ax.patches)*1.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.data/'local-performance.png', dpi=145)
    plt.close(fig)

    cases = ['src7', 'src3', 'src5_45', 'controlled_scale', 'held_out_scale']
    lookup = {(r['job']['case']['name'],r['job']['method']):r['result'] for r in full['results']}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(cases))
    for offset, method, title, color in [(-.2,'continuous_direct_cpu','CPU coarse + CPU LM','#425a75'),
            (.2,'accelerated_direct_metal','CPU coarse + Metal LM + CPU solve/verification','#098d8d')]:
        bars = ax.bar(x+offset,[lookup[c,method]['algorithm']['median_ms'] for c in cases],width=.38,label=title,color=color)
        ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    ax.set_xticks(x,['4K rotor','Circuit board','Rotated part','6 known poses','12 known poses'])
    ax.set_ylabel('Median full-image algorithm time (ms)')
    ax.set_title('Full-image search: CPU and hybrid refinement',loc='left',weight='bold',fontsize=14)
    ax.legend(frameon=False,fontsize=9)
    ax.set_ylim(0,max(p.get_height() for p in ax.patches)*1.25)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.data/'full-image-performance.png',dpi=145)
    plt.close(fig)
    print('Rendered',args.data)


if __name__ == '__main__':
    main()
