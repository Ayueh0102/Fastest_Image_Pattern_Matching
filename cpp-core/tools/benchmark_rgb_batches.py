"""RGB crossover supplement: same NCC workload, fresh serial CPU/Metal workers."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from benchmark_pose_experiments import ROOT, source_hashes
from pose_experiment import library_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, default=ROOT/'cpp-core/docs/rgb-batch-benchmark.json')
    args = parser.parse_args()
    tasks = [dict(suite='micro', case=dict(name=f'{size}_rgb_{count}', size=size, rgb=True, count=count),
                  backend=backend, threads=threads, build=str(args.build.resolve()),
                  repeats=7, warmups=2, warmup_ms=200)
             for size, count in [(64, 1), (64, 49), (64, 512), (128, 512)]
             for backend, threads in [('cpu', 1), ('cpu', 8), ('metal', 1)]]
    np.random.default_rng(71).shuffle(tasks)
    hashes = source_hashes()
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(created_at=datetime.now().astimezone().isoformat(), platform=platform.platform(),
                  numpy=np.__version__, source_sha256=hashes,
                  binary_sha256=hashlib.sha256(library_path(args.build).read_bytes()).hexdigest(),
                  protocol=dict(serial=True, fresh_process_per_variant=True, repeats=7, warmups=2, warmup_ms=200,
                                channels='three distinct analytic color channels, not replicated grayscale',
                                scope='resident fixed pose batch NCC only; no full-image search',
                                timing='wall includes submission, wait and output copy; excludes initialization and image generation',
                                memory='process RSS includes Python and setup; Metal shared allocations overlap RAM'), runs=[])
    with tempfile.TemporaryDirectory(dir=ROOT/'build', prefix='rgb-batch-') as directory:
        path = Path(directory)
        for job in tasks:
            (path/'request.json').write_text(json.dumps(job))
            subprocess.run([sys.executable, str(ROOT/'cpp-core/tools/benchmark_pose_experiments.py'),
                            '--worker', str(path/'request.json'), '--output', str(path/'result.json')],
                           check=True, timeout=300)
            entry = json.loads((path/'result.json').read_text())
            report['runs'].append(entry)
            print(job['case']['name'], job['backend'], job['threads'], entry['result']['wall']['median_ms'], flush=True)
    report['consistency'] = []
    for name in sorted({entry['job']['case']['name'] for entry in report['runs']}):
        variants = [r for r in report['runs'] if r['job']['case']['name'] == name]
        reference = next(r['result'] for r in variants if r['job']['backend'] == 'cpu' and r['job']['threads'] == 1)
        for entry in variants:
            maximum = float(np.max(np.abs(np.array(entry['result']['scores'])-reference['scores'])))
            assert maximum < 3e-5
            assert entry['result']['best_index'] == reference['best_index']
            report['consistency'].append(dict(case=name, backend=entry['job']['backend'],
                                             threads=entry['job']['threads'], max_score_error=maximum))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
