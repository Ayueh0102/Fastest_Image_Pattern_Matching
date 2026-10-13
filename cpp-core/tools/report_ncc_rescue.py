"""Build compact tables and figures from saved rescue measurements (no workloads)."""
import argparse
import gzip
import json
from pathlib import Path

METHODS = ('old_cpu', 'old_resident_verified', 'rescue_cpu', 'rescue_resident_verified')
LABELS = ('CPU baseline', 'Metal baseline + verify', 'CPU + rescue', 'Metal + rescue + verify')
COLORS = ('#86909c', '#5c8fad', '#c79543', '#33846c')


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def rows(reports):
    output = []
    for path, report in reports:
        for run in report['runs']:
            job, result = run['job'], run['result']
            if result.get('status') == 'unavailable':
                output.append(dict(report=str(path), case=job['case']['name'], method=job['method'], unavailable=True))
                continue
            samples = result['measured']
            representative = sorted(range(len(samples)), key=lambda i: samples[i]['result']['metrics']['wall_ms'])[len(samples)//2]
            metrics = samples[representative]['result']['metrics']
            memory = result['memory']
            check = result['validation']
            output.append(dict(report=str(path), case=job['case']['name'], split=job['case']['split'],
                scope=job['case']['scope'], method=job['method'],
                source_shape=job['case'].get('source_shape'), template_size=job['case'].get('size'),
                channels=job['case'].get('channels'), seed_count=check['seed_count'],
                warm_outer_ms=result['algorithm_wall']['median_ms'], warm_api_ms=result['c_api_wall']['median_ms'],
                warm_gpu_ms=result['gpu_command']['median_ms'] if result.get('gpu_command') else None,
                constructor_ms=result['constructor_wall_ms'], first_ms=result['first']['wall_ms'],
                peak_rss_bytes=memory['process_peak_bytes'], buffers=memory['max_observed_owned_buffers'],
                precise_pass_count=check['precise_pass_count'], diagnostic_accuracy=check['accuracy_is_diagnostic'],
                passed=result['passed'], representative=metrics,
                rescue=metrics.get('rescue'), resident=metrics.get('resident'),
                full_truth=result.get('truth_accuracy'),
                samples=len(samples), checked_calls=len(check['calls'])))
    return output


def chart(table, split, destination):
    selected = [r for r in table if r.get('split') == split and not r.get('unavailable')]
    names = list(dict.fromkeys(r['case'] for r in selected))
    if not names:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup = {(r['case'], r['method']): r for r in selected}
    figure, axes = plt.subplots(1, 3, figsize=(17, max(4, .62*len(names)+1.8)), sharey=True)
    for ax, key, title, scale in zip(axes,
            ('warm_api_ms', 'constructor_ms', 'peak_rss_bytes'),
            ('Local refinement, warm median (ms)', 'Handle construction (ms)', 'Process peak RSS before validation (MiB)'),
            (1, 1, 1/(1024*1024))):
        for j, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            values = [lookup.get((name, method), {}).get(key, float('nan'))*scale for name in names]
            ax.barh([i+(j-1.5)*.19 for i in range(len(names))], values, height=.17, label=label, color=color)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    axes[0].invert_yaxis()
    axes[-1].legend(loc='lower right', fontsize=8)
    figure.suptitle('NCC bounded rescue — '+split.replace('_', ' '))
    figure.tight_layout()
    figure.savefig(destination/(split+'.png'), dpi=160)
    plt.close(figure)


def failure_image(reports, destination):
    """Visualize saved endpoints; sampling here is only a post-run illustration."""
    import numpy as np
    from benchmark_pose_stress import fixture
    selected = {run['job']['method']: run for _, report in reports for run in report['runs']
                if run['job']['case']['name'] == 'heldout_16'}
    if any(method not in selected for method in METHODS):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source, template = fixture(selected['old_cpu']['job']['case'])
    height, width = source.shape[:2]
    th, tw = template.shape[:2]
    yy, xx = np.indices((th, tw), dtype=np.float64)
    u, v = xx-tw/2, th/2-yy
    def sample(pose):
        theta, scale = np.deg2rad(pose[2]), np.exp(pose[3])
        x = width/2+pose[0]+scale*(np.cos(theta)*u-np.sin(theta)*v)
        y = height/2-pose[1]-scale*(np.sin(theta)*u+np.cos(theta)*v)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x-ix, y-iy
        value = np.zeros(template.shape, np.float64)
        for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                               (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
            valid = (ix+dx >= 0) & (ix+dx < width) & (iy+dy >= 0) & (iy+dy < height)
            value[valid] += source[iy[valid]+dy, ix[valid]+dx]*weight[valid]
        return value
    figure, axes = plt.subplots(1, 5, figsize=(13, 3.4))
    axes[0].imshow(template, cmap='gray', vmin=.2, vmax=.65)
    axes[0].set_title('Reference template')
    for ax, method, label in zip(axes[1:], METHODS, LABELS):
        result = selected[method]['result']['measured'][-1]['result']
        index = 3
        ax.imshow(sample(result['poses'][index]), cmap='gray', vmin=.2, vmax=.65)
        ax.set_title(label+'\nNCC %.6f' % result['scores'][index], fontsize=10)
    for ax in axes:
        ax.axis('off')
    figure.suptitle('Shared local-peak failure — heldout_16 seed 3, saved measured endpoints')
    figure.tight_layout()
    figure.savefig(destination/'common-failure.png', dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [(path, read(path)) for path in args.reports]
    table = rows(reports)
    (args.output/'table.json').write_text(json.dumps(table, indent=2)+'\n')
    summaries = []
    for path, report in reports:
        # Keep absolute accuracy and relative improvement separate. Challenges
        # with deliberately unidentifiable texture must remain visible in rows.
        flags = {key: value for key, value in report.items() if key.endswith('_passed') or key in (
            'complete', 'unavailable_runs', 'expected_runs', 'cross_method_nonregression', 'common_failure_recovered',
            'source_hashes_unchanged', 'binary_hashes_unchanged', 'performance_acceptance')}
        summaries.append(dict(report=str(path), flags=flags, comparisons=report.get('comparisons'),
            protocol=report.get('protocol'), policy=report.get('policy'),
            observed_runs=len(report['runs'])))
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    if args.plots:
        for split in ('source_size', 'workload'):
            chart(table, split, args.output)
        failure_image(reports, args.output)
    print(json.dumps(dict(rows=len(table), reports=len(reports), output=str(args.output))))


if __name__ == '__main__':
    main()
