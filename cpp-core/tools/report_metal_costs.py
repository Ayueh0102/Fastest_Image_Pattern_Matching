"""Validate diagnostic Metal stage accounting and plot same-run proportions."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'cpp-core/docs/metal-cost-profile/report.json')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report['unchanged_during_measurement']
    assert all(row['exact_outputs'] for row in report['parity'])
    events_checked = 0
    rows = {}
    for entry in report['runs']:
        if entry['job']['variant'] != 'profile':
            continue
        assert len(entry['metal_setup']) == 1 and entry['metal_setup'][0]['unified_memory']
        for record in entry['result']['records']:
            assert record['profile']
            gpu_ms = 0.
            for event in record['profile']:
                assert event['status'] == 'ok' and event['gpu_ms'] >= 0
                assert event['gpu_ms'] <= event['commit_wait_ms']+1e-5
                assert event['pose_copy_bytes'] == 16*event['poses']
                assert event['result_read_bytes'] == event['poses']*(144 if event['operation'] == 'linearize' else 4)
                stages = ['workspace_ms', 'host_output_resize_ms', 'pose_copy_ms',
                          'encode_ms', 'commit_wait_ms', 'readback_assemble_ms']
                assert sum(event[k] for k in stages) <= event['backend_wall_ms']+1e-5
                gpu_ms += event['gpu_ms']
                events_checked += 1
            value = record['result']
            expected = value['metrics']['device_ms'] if entry['job']['kind'] == 'local' else value['gpu_command_ms']
            assert abs(gpu_ms-expected) < 1e-5
        summary = entry['summary']
        totals = summary['representative_profile_totals']
        api = summary['representative_api_metrics']
        wall = api['wall_ms'] if api else summary['representative_full_stages']['ncc_wall_ms']
        copies = totals['pose_copy_ms']+totals['readback_assemble_ms']
        parts = [totals['gpu_ms'], totals['commit_wait_ms']-totals['gpu_ms'],
                 totals['backend_wall_ms']-totals['commit_wait_ms']-copies,
                 copies, wall-totals['backend_wall_ms']]
        assert min(parts) >= 0 and abs(sum(parts)-wall) < 1e-5
        rows[entry['job']['case']['name']] = dict(api_wall_ms=wall, parts_ms=parts,
            commands=totals['chunks'], copy_assemble_percent=100*copies/wall,
            host_interface_logical_bytes=totals['pose_copy_bytes']+totals['result_read_bytes'])
    labels = [
        ('heldout_s0160_c1_n004', '160² gray / 4 poses'),
        ('calibration_s0768_c1_n032', '768² gray / 32 poses'),
        ('heldout_s0512_c3_n016', '512² RGB / 16 poses *'),
        ('chunked_768_c1_n256_one_iteration', '768² gray / 256 poses\n(one iteration only)'),
        ('src7', '4K rotor / 6 seeds\n(refinement only)'),
        ('held_out_scale', '12 objects / 34 seeds\n(forced Metal refinement)')]
    names = ['GPU command interval', 'Submit / schedule / completion residual',
             'Host encoding and other backend work', 'Pose copy + result read / assembly',
             'Outside backend (CPU scans / control / logging)']
    colors = ['#177b8d', '#dc9d45', '#6986a1', '#8fa647', '#d7dfe7']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(13, 6.5))
    y = np.arange(len(labels))
    values = np.asarray([rows[k]['parts_ms'] for k, _ in labels])
    percent = values/values.sum(axis=1)[:, None]*100
    start = np.zeros(len(labels))
    for i, (name, color) in enumerate(zip(names, colors)):
        ax.barh(y, percent[:, i], left=start, color=color, height=.62, label=name)
        start += percent[:, i]
    for index, (key, _) in enumerate(labels):
        row = rows[key]
        ax.text(101.5, index, f"{row['api_wall_ms']:.2f} ms\n{row['commands']} submissions", va='center', fontsize=9)
    ax.set_yticks(y, [label for _, label in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Share of refinement C API wall time; GPU interval is contained in commit/wait')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.13), frameon=False, ncol=2, fontsize=9)
    fig.suptitle('Metal refinement: copying is small; waiting and computation remain', x=.02, ha='left', fontsize=15, fontweight='bold')
    fig.text(.02, .015, 'Same run for all components. GPU interval is not utilization. * Instrumented / baseline wall differs by +20.9%; other cases within 2%.', fontsize=8.5)
    fig.subplots_adjust(left=.25, right=.98, top=.9, bottom=.24)
    image = args.report.with_name('breakdown.png')
    fig.savefig(image, dpi=170, facecolor='white')
    plt.close(fig)
    validation = dict(report_sha256=digest(args.report), renderer_sha256=digest(Path(__file__)),
        image_sha256=digest(image), events_checked=events_checked, all_checks_passed=True,
        checks=['Exact original/instrumented outputs for first and all measured calls',
                'Host stage sums within backend wall; valid GPU intervals overlap commit/wait',
                'Logical interface bytes match pose counts and layouts',
                'Summed GPU profile time agrees with C API metric',
                'Same-run additive decomposition (not sums of independent medians)'],
        series_order=[key for key, _ in labels], component_order=names, representative_runs=rows)
    args.report.with_name('validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(f'Validated {events_checked} events; saved {image}')


if __name__ == '__main__':
    main()
