"""Plot saved CPU/Metal crossover and held-out automatic routing measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'cpp-core/docs/pose-routing')
    args = parser.parse_args()
    calibration = json.loads((args.data/'calibration.json').read_text())
    heldout = json.loads((args.data/'heldout.json').read_text())
    full = json.loads((args.data/'full-image.json').read_text())
    auto_rows = [r for r in heldout['runs'] if r['job']['method'] == 'auto']
    policy = auto_rows[0]['result']['effective_auto_policy']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': '#f6f8fb'})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for row, channels in enumerate((1, 3)):
        for col, count in enumerate((1, 8, 32)):
            ax = axes[row, col]
            cases = sorted([c for c in calibration['comparisons']
                            if c['channels'] == channels and c['count'] == count],
                           key=lambda c: c['template_size'])
            for key, title, color in [('cpu_ms', 'CPU 8 threads', '#425a75'),
                                      ('metal_ms', 'Metal + CPU solve', '#098d8d')]:
                ax.plot([c['template_size'] for c in cases], [c[key] for c in cases],
                        marker='o', color=color, label=title, linewidth=2)
            cutoff = max((policy['min_pose_samples']/channels)**.5,
                         (policy['min_batch_samples']/channels/count)**.5)
            if 96 <= cutoff <= 768:
                ax.axvline(cutoff, color='#b48343', linestyle='--', alpha=.8,
                           label='Auto workload cutoff')
            ax.set_title(f"{'Gray' if channels == 1 else 'RGB'} / {count} {'seed' if count == 1 else 'seeds'}", loc='left')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xticks([96, 192, 384, 768], ['96', '192', '384', '768'])
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.18)
            if col == 0:
                ax.set_ylabel('Median local refinement (ms)')
            if row == 1:
                ax.set_xlabel('Square template side (pixels)')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('CPU / Metal crossover: measured sizes, fixed candidate counts',
                 x=.05, ha='left', weight='bold', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.data/'crossover.png', dpi=150)
    plt.close(fig)

    rows = sorted(heldout['comparisons'], key=lambda r: (r['template_size'], r['channels'], r['count']))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(14, 5))
    for delta, key, label, color in [(-.25, 'cpu_ms', 'CPU', '#425a75'),
            (0, 'metal_ms', 'Forced Metal', '#97b7cd'), (.25, 'auto_ms', 'Auto', '#098d8d')]:
        ax.bar(x+delta, [r[key] for r in rows], width=.24, color=color, label=label)
    ax.set_xticks(x, [f"{r['template_size']} / {'RGB' if r['channels']==3 else 'gray'}\n{r['count']} seeds\n{'GPU' if r['auto_selected']=='metal' else 'CPU'} route" for r in rows], fontsize=8)
    ax.set_ylabel('Median local refinement (ms)')
    ax.set_title('Held-out textures and sizes: frozen routing policy', loc='left', weight='bold', fontsize=15)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.data/'heldout.png', dpi=150)
    plt.close(fig)

    lookup = {(r['job']['case']['name'], r['job']['method']): r['result'] for r in full['results']}
    names = ['src7', 'src3', 'src5_45', 'controlled_scale', 'held_out_scale']
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    for delta, method, label, color in [(-.25, 'continuous_direct_cpu', 'CPU', '#425a75'),
            (0, 'accelerated_direct_metal', 'Forced Metal', '#97b7cd'),
            (.25, 'adaptive_direct_auto', 'Auto', '#098d8d')]:
        bars = ax.bar(x+delta, [lookup[n, method]['algorithm']['median_ms'] for n in names],
                      width=.24, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    ax.set_xticks(x, ['4K rotor', 'Circuit board', 'Rotated part', '6 known poses', '12 known poses'])
    ax.set_ylim(0, max(p.get_height() for p in ax.patches)*1.18)
    ax.set_ylabel('Median full-image search (ms)')
    ax.set_title('Full-image search with automatic CPU / Metal refinement', loc='left', weight='bold', fontsize=15)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.data/'full-image.png', dpi=150)
    plt.close(fig)
    print('Saved routing figures to', args.data)


if __name__ == '__main__':
    main()
