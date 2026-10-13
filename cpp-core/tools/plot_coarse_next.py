"""Plot saved coarse ablations; no image matching or GPU calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_shared_fine import LABELS


def render(report, output):
    methods = [('previous_shared', 'Previous shared pipeline', '#7e8996'),
               ('current_control', 'New build, both changes off', '#b0bac6'),
               ('peak_only', 'CPU peak tree only (recommended)', '#398d98'),
               ('simd_only', 'GPU SIMD/window only (experiment)', '#c4a272'),
               ('combined', 'Both changes (experiment)', '#b36d5f')]
    cases = report['cases']
    y = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True,
                             gridspec_kw={'width_ratios': [1.3, 1]})
    for i, (key, label, color) in enumerate(methods):
        values = [c['methods'][key]['measured']['total_wall_ms']['median'] for c in cases]
        axes[0].barh(y+(i-2)*.16, values, .15, color=color, label=label)
    for i, (key, label, color) in enumerate([methods[0], methods[2]]):
        values = [max(c['methods'][key]['peak_rss_bytes']) / 2**20 for c in cases]
        axes[1].barh(y+(i-.5)*.28, values, .25, color=color)
    axes[0].set_yticks(y, [LABELS[c['case']['name']] for c in cases])
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Warm full-pipeline median (ms, log scale)')
    axes[1].set_xlabel('Process peak RSS (MiB, max of two blocks)')
    for axis in axes:
        axis.grid(axis='x', alpha=.18); axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Exact coarse peak selection and GPU NCC ablation', fontsize=16, x=.54)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(.56, .025), frameon=False)
    fig.text(.55, .012, 'Apple M4 Pro · fine batch 512 · prepare included · exact outputs and map counts checked',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .105, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)
