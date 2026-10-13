"""Render exported scientific figures from saved shared-fine results only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    'src7': 'Src7 / 1 scale', 'controlled6': '6 rotated objects / 3 scales',
    'heldout12': '12 held-out objects / 3 scales',
    '8k_t256': '8K / T256 / angle 0',
    '8k_t256_3scales': '8K / T256 / angle 0 / 3 scales',
    '8k_t256_angle180': '8K / T256 / angle ±180',
    '8k_t256_angle180_3scales': '8K / T256 / angle ±180 / 3 scales',
    '8k_t512': '8K / T512 / angle 0',
    'small_fullres': '640×512 / subpixel / 1 scale',
    'small_fullres_3scales': '640×512 / subpixel / 3 scales',
}


def render(report, output):
    methods = [('previous_combined', 'Previous Metal + CPU fine', '#8995a5'),
               ('shared_cpu_fine', 'Shared pyramid + CPU fine', '#4099a6'),
               ('shared_metal_fine', 'Shared pyramid + Metal fine', '#d88636')]
    cases = report['cases']
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True,
                             gridspec_kw={'width_ratios': [1.15, 1]})
    y = np.arange(len(cases))
    for i, (key, label, color) in enumerate(methods):
        latency = [c['methods'][key]['measured']['total_wall_ms']['median'] for c in cases]
        memory = [max(c['methods'][key]['peak_rss_bytes']) / 2**20 for c in cases]
        axes[0].barh(y+(i-1)*.24, latency, .22, color=color, label=label)
        axes[1].barh(y+(i-1)*.24, memory, .22, color=color)
    axes[0].set_yticks(y, [LABELS[c['case']['name']] for c in cases])
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Warm full-pipeline median (ms, log scale)')
    axes[1].set_xlabel('Process peak RSS (MiB, max of two blocks)')
    for axis in axes:
        axis.grid(axis='x', alpha=.18)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Shared source pyramid and batched Metal fine search', fontsize=16, x=.53)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               bbox_to_anchor=(.55, .035), frameon=False)
    fig.text(.55, .015, 'Apple M4 Pro · batch 512 · prepare included · exact outputs checked · RSS includes Python/driver',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .085, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)
