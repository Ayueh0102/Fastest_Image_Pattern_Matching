"""Plot and preserve the sparse proposal study, including failed gates.

Run after all measured sources have been frozen. The archive contains original
worker records, protocol, input fixtures, source snapshots and validation logs;
its relative worker paths can be replayed without the original workspace.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def plots(report, out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    cases = [c for c in report['summary']['cases'] if c['backend'] == 1]
    names = dict(crop_small360='Small / 360 deg', crop_medium20='Six objects',
                 crop_lowcontrast20='Low contrast', thin_detail20='Thin detail',
                 scale_violation20='Scale +4% (outside assumption)',
                 repo_src4='Repository photo (no GT)', heldout_crowd128_20='128 objects')
    methods = [('pyramid', 'Pyramid + Metal', '#386b8a'),
               ('sparse_orientation', 'Sparse CPU proposals + Metal NCC', '#c88438')]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for i, (method, label, color) in enumerate(methods):
        for ax, field, factor in zip(axes, ['total_ms', 'peak_rss_bytes'], [1, 1/2**20]):
            values = []
            for c in cases:
                m = c['methods'][method]
                v = m.get(field)
                values.append((v['median'] if field == 'total_ms' else np.median(v))*factor if v else np.nan)
            bars = ax.barh(x+(i-.5)*.34, values, .32, label=label, color=color)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    for ax in axes:
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=.18)
    axes[0].set_yticks(x, [names.get(c['name'], c['name']) for c in cases], fontsize=9)
    axes[1].set_yticks(x, [])
    axes[0].set_xlabel('Algorithm time (ms; median of measured calls)')
    axes[1].set_xlabel('Peak process RSS (MiB; median of workers)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.6, .94), ncol=2, fontsize=9)
    fig.suptitle('Fixed-scale full-image rotation search | speed does not imply recall', fontsize=13)
    fig.tight_layout(rect=[0, .055, 1, .88])
    fig.text(.015, .02, 'Source response preparation and full NCC refinement included. Setup/decoding excluded. Failures are retained in results.', fontsize=9)
    fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    for case in report['protocol']['cases']:
        if case['name'] not in ('crop_medium20', 'thin_detail20', 'heldout_crowd128_20'):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={'width_ratios':[.6, 1.6, 1.6]})
        axes[0].imshow(plt.imread(case['template']), cmap='gray')
        axes[0].set_title('Native-pixel template')
        summary = next(c for c in cases if c['name'] == case['name'])
        for ax, (method, label, _) in zip(axes[1:], methods):
            ax.imshow(plt.imread(case['source']), cmap='gray', vmin=0, vmax=1)
            for gt in case['truth'] or []:
                ax.add_patch(Polygon(gt['corners'], closed=True, fill=False, edgecolor='#42bf84', linewidth=1.5))
            detected = summary['methods'][method].get('first_outputs', [[]])[0]
            for match in detected:
                ax.add_patch(Polygon(match['corners'], closed=True, fill=False, edgecolor='#df692b', linewidth=1, linestyle='--'))
            ax.set_title(label+'\n'+str(len(detected))+' detections', fontsize=10)
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('Full-image search: '+case['name'])
        fig.tight_layout(rect=[0, .09, 1, .94])
        fig.text(.015, .03, 'Green: synthetic ground truth. Orange dashed: detections. Photographed crop composited into synthetic scenes.', fontsize=9)
        fig.savefig(out/(case['name']+'.png'), dpi=160); plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'cpp-core/docs/sparse-orientation')
    p.add_argument('--validation-log', type=Path, action='append', default=[])
    p.add_argument('--pilot', type=Path, action='append', default=[])
    args = p.parse_args()
    result = args.results.resolve(); base = result.parent
    report = json.loads(result.read_text()); protocol = report['protocol']
    if not report['complete'] or not report['unchanged']:
        raise ValueError('Incomplete run or changed measured implementation')
    if sha(result.with_suffix('.protocol.json')) != report['protocol_sha256']:
        raise ValueError('Changed protocol')
    for group in ('source_sha256', 'binary_sha256', 'input_sha256'):
        if not all(sha(path) == digest for path, digest in protocol[group].items()):
            raise ValueError('Changed '+group)
    for entry in report['execution_order']:
        for key in ('job', 'result', 'stdout', 'stderr'):
            if key in entry and sha(base/entry[key]) != entry[key+'_sha256']:
                raise ValueError('Changed worker artifact')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    save(out/'results.json', report['summary'])
    copy(result.with_suffix('.protocol.json'), out/'protocol.json')
    plots(report, out)
    with tempfile.TemporaryDirectory(prefix='fpm-sparse-archive-') as tmp:
        folder = Path(tmp)/'archive'; shutil.copytree(base, folder/'formal')
        for pilot in args.pilot:
            shutil.copytree(pilot.resolve(), folder/'pilots'/pilot.name)
        for log in args.validation_log: copy(log, folder/'validation'/log.name)
        for name in ('Src3.bmp', 'Src4.bmp', 'Dst4.bmp'):
            copy(ROOT/'Test Images'/name, folder/'original-inputs'/name)
        validation = dict(source_binary_input_hashes_unchanged=True,
            workers=report['summary']['recorded_workers'],
            measured_calls=sum(len(json.loads((base/e['result']).read_text())['runs']) for e in report['execution_order'] if e['status']=='ok'),
            all_workers_succeeded=report['summary']['all_workers_succeeded'],
            route_integrity_passed=report['summary']['route_integrity_passed'],
            fixed_scale_gt_all_passed=report['summary']['fixed_scale_gt_all_passed'],
            fixed_scale_proposal_coverage_passed=report['summary']['fixed_scale_proposal_coverage_passed'],
            validation_logs={log.name:sha(log) for log in args.validation_log})
        save(out/'validation.json', validation); save(folder/'validation.json', validation)
        (folder/'README.txt').write_text('''Sparse orientation proposal study. All formal successes and failures are retained.
Proposal scores are unsigned-gradient similarity, NOT NCC. Final matches use full
NCC and fixed-scale continuous refinement. Source proposal work is CPU; backend1
uses Metal only for resident refinement (plus CPU verification/rescue control).
No image pyramid, GT positions or ROI hints enter the sparse worker.

Offline summary replay (NumPy and OpenCV installed, no GPU needed):
python3 formal/source-snapshot/cpp-core/tools/benchmark_sparse_orientation.py \\
 --summarize formal/results.json --output /tmp/replayed-summary.json
Paths in historical protocols describe original provenance. Original sources,
fixtures, worker records and a baseline source archive are preserved; binaries
are omitted but hashed. Setup and first-call costs are separate from warm timing.
''')
        manifest = {str(path.relative_to(folder)):sha(path) for path in sorted(folder.rglob('*')) if path.is_file()}
        save(folder/'sha256.json', manifest)
        archive = out/'raw-results.tar.gz'
        with archive.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode='w') as tar:
                    for path in sorted(folder.rglob('*')):
                        if not path.is_file(): continue
                        info = tar.gettarinfo(str(path), arcname=str(path.relative_to(folder)))
                        info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
                        with path.open('rb') as data: tar.addfile(info, data)
        relocated = Path(tmp)/'relocated'; relocated.mkdir()
        with tarfile.open(archive) as tar:
            for name, digest in manifest.items():
                if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=digest:
                    raise ValueError('Archive member hash mismatch')
            tar.extractall(relocated, filter='data')
        replay = relocated/'replayed.json'
        subprocess.run([sys.executable, str(relocated/'formal/source-snapshot/cpp-core/tools/benchmark_sparse_orientation.py'),
            '--summarize', str(relocated/'formal/results.json'), '--output', str(replay)], cwd='/tmp', check=True)
        if replay.read_bytes() != (out/'results.json').read_bytes():
            raise ValueError('Relocated summary does not match')
        save(out/'artifact-sha256.json', dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
            members=len(manifest), all_member_hashes_valid=True, relocated_summary_byte_identical=True,
            summary_sha256=sha(out/'results.json')))
    print(json.dumps(validation, indent=2))

if __name__ == '__main__':
    main()
