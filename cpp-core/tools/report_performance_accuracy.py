"""Render recorded measurements; never run or tune matching while reporting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'cpp-core/docs/performance-accuracy')
    args = parser.parse_args()
    data = args.data
    full = json.loads((data/'full-image.json').read_text())
    stress = json.loads((data/'stress.json').read_text())
    pruning = json.loads((data/'ncc-pruning.json').read_text())
    ablation = json.loads((data/'ablation.json').read_text())
    get = lambda case, method: next(e for e in full['results']
        if e['job']['case']['name'] == case and e['job']['method'] == method)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': '#f6f8fb'})
    for case in ('controlled_scale', 'held_out_scale'):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.7))
        for ax, method, label in zip(axes, ('hybrid_metal', 'continuous_metal'),
                ('Previous GPU grid refinement', 'New continuous refinement')):
            entry = get(case, method)
            result, spec = entry['result'], entry['job']['case']
            source = Path(spec['source'])
            if not source.exists():
                source = data/source.name
            ax.imshow(plt.imread(source), cmap='gray')
            passes = {r['result_index']: r['pose_pass'] for r in result['accuracy']['details'] if r['localized']}
            for truth in spec['truth']:
                ax.add_patch(Polygon(truth['corners'], fill=False, edgecolor='#1ab6dd', linestyle='--', linewidth=1))
            for m in result['runs'][-1]['matches']:
                color = '#10a458' if passes.get(m['index']) else '#ec721d'
                ax.add_patch(Polygon(m['corners'], fill=False, edgecolor=color, linewidth=1.6))
                ax.text(m['pos_x'], m['pos_y'], f"{m['score']:.3f}", ha='center', va='center', color='white',
                        fontsize=8, bbox=dict(facecolor='#162a3b', alpha=.85, edgecolor='none', pad=2))
            ax.set_title(f"{label}\nPose pass {result['accuracy']['pose_tp']}/{len(spec['truth'])}",
                         loc='left', weight='bold', color='#18314a', fontsize=14)
            ax.axis('off')
        fig.suptitle('Full-image search / photographed texture with independent known poses', x=.035,
                     y=.99, ha='left', weight='bold', fontsize=16)
        fig.text(.035, .02, 'Dashed cyan: truth. Green: pose pass. Orange: pose fails. Labels: NCC. Gate: 2 px / 1 degree / 2% scale.', fontsize=10)
        fig.subplots_adjust(left=.025, right=.99, top=.90, bottom=.07, wspace=.06)
        fig.savefig(data/f'{case}.png', dpi=135)
        plt.close(fig)

    names = ['128_c3_n1024', '256_c3_n8192', '768_c1_n1024']
    batch = lambda name, backend, threads: next(e['result'] for e in stress['runs']
        if e['job']['case']['name'] == name and e['job']['backend'] == backend and e['job']['threads'] == threads)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(names))
    for offset, backend, threads, label, color in [(-.2, 'cpu', 8, 'CPU 8 threads', '#445a72'),
                                                 (.2, 'metal', 1, 'Metal', '#16848d')]:
        values = [batch(n, backend, threads)['call_wall']['median_ms'] for n in names]
        bars = axes[0].bar(x+offset, values, width=.38, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.1f', padding=3)
    axes[0].set_xticks(x, ['128 RGB\n1,024 poses', '256 RGB\n8,192 poses', '768 gray\n1,024 poses'])
    axes[0].set_ylabel('Median call wall time (ms)')
    axes[0].set_ylim(0, max(batch(n,'cpu',8)['call_wall']['median_ms'] for n in names)*1.2)
    axes[0].set_title('Fixed pose scoring on 4K sources', loc='left', weight='bold')
    axes[0].legend(frameon=False)
    memory = [batch(n,'metal',1) for n in names]
    for offset, key, label, color in [(-.2, 'naive_metal_auxiliary_bytes', 'Unbounded allocation formula', '#acb7c5'),
                                     (.2, 'retained_auxiliary_buffer_bytes', 'Measured retained workspace', '#16848d')]:
        bars = axes[1].bar(x+offset, [r[key]/2**20 for r in memory], width=.38, label=label, color=color)
        axes[1].bar_label(bars, fmt='%.2f', padding=3)
    axes[1].set_xticks(x, ['128 RGB\n1,024 poses', '256 RGB\n8,192 poses', '768 gray\n1,024 poses'])
    axes[1].set_ylabel('Auxiliary GPU buffers (MiB)')
    axes[1].set_ylim(0, 45)
    axes[1].set_title('Workspace cap; resident images excluded', loc='left', weight='bold')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(data/'batch-performance.png', dpi=150)
    plt.close(fig)
    summary = dict(full_image=[dict(case=e['job']['case']['name'], method=e['job']['method'],
        median_ms=e['result']['algorithm']['median_ms'], detections=len(e['result']['runs'][-1]['matches']),
        pose_pass=e['result']['accuracy']['pose_tp'] if e['result']['accuracy'] else None,
        peak_rss_mib=e['result']['memory']['process_peak_bytes']/2**20) for e in full['results']],
        ablation=[dict(case=e['job']['case']['name'], method=e['job']['method'],
            median_ms=e['result']['algorithm']['median_ms'], pose_pass=e['result']['accuracy']['pose_tp'])
            for e in ablation['results']], local_accuracy=stress['local_accuracy'],
        consistency_passed=stress['correctness_passed'],
        max_cpu_metal_score_difference=max(r['max_score_error'] for r in stress['consistency']),
        pruning_all_threshold_parity=all(r['threshold_parity'] for r in pruning['runs']))
    (data/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Rendered', data)


if __name__ == '__main__':
    main()
