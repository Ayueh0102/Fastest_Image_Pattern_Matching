"""Plot saved pipeline summaries; never load a matching library or run GPU work."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = report['cases']
    names = [case['name'] for case in cases]
    methods = [('frozen_cpu', 'Frozen CPU', '#80909d'),
               ('frozen_metal', 'Frozen Metal coarse', '#c59247'),
               ('combined', 'Metal pyramid + batched coarse', '#358976')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 2+.75*len(cases)), sharey=True)
    for offset, (method, label, color) in enumerate(methods):
        y = [i+(offset-1)*.24 for i in range(len(cases))]
        times = [case['methods'][method]['total_ms']['median'] for case in cases]
        memory = [max(case['methods'][method]['peak_rss_bytes'])/2**20 for case in cases]
        axes[0].barh(y, times, height=.22, label=label, color=color)
        axes[1].barh(y, memory, height=.22, label=label, color=color)
    axes[0].set_yticks(range(len(cases)), names)
    axes[0].invert_yaxis()
    axes[0].set_title('Whole workflow, warm median (ms)')
    axes[1].set_title('Largest process peak RSS across blocks (MiB)')
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.07))
    fig.suptitle('Gray8 proposal mode: stop_layer=1, subpixel=0; real scenes add resident rescue')
    fig.tight_layout()
    fig.savefig(args.output/'time-memory.png', dpi=160)
    plt.close(fig)

    groups = [('pyramid_ms', 'Pyramid', '#c59247'),
              ('coarse_rotation_ms', 'Coarse rotation', '#d8bb8f'),
              ('coarse_ncc_ms', 'Coarse NCC incl. wait', '#447895'),
              ('fine_rotation_ms', 'Fine rotation', '#8caebf'),
              ('fine_ncc_ms', 'Fine NCC', '#358976'),
              (None, 'Other CPU stages', '#a6aaa8')]
    fig, axes = plt.subplots(len(cases), 1, figsize=(12, 1.9*len(cases)+1), squeeze=False)
    axes = axes[:, 0]
    for ax, case in zip(axes, cases):
        labels = ['CPU', 'Old Metal path', 'Combined Metal']
        selected = ['cpu_profile', 'metal_profile', 'combined_profile']
        cumulative = [0., 0., 0.]
        for key, label, color in groups:
            values = []
            for method in selected:
                record = case['methods'][method]['representative_median_proposal_call']
                stages = record['exclusive_stages']
                value = (stages[key]['ms'] if key else record['core_wall_ms']-
                         sum(stages[k]['ms'] for k, _, _ in groups if k))
                values.append(max(0., value))
            ax.barh(range(3), values, left=cumulative, height=.65, label=label, color=color)
            cumulative = [a+b for a, b in zip(cumulative, values)]
        ax.set_yticks(range(3), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(case['name'], loc='left', fontsize=10)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('Exclusive C++ proposal stages (ms), one actual median-nearest call')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Exclusive stages — gray8, stop_layer=1, subpixel=0', fontsize=10)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .98), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.output/'exclusive-stages.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
