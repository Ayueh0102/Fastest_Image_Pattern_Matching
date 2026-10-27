"""Round 50: the GeCo2 CUDA runs on the lab PC (Windows, RTX PRO 4000), from a package
made by ``literature_compare.py export-geco2-cuda``.

Runs in the GeCo2 CUDA environment (``python -I``), detached from the SSH session
(started through Win32_Process.Create, so it is outside the SSH session's job object).
- B2: one batch over the 1,190 FSC-147 test images in split order (the M4 Pro MPS
  batch's command with ``--device cuda --deform-attn cuda``), 3-hour timeout.
- A and C: one fresh worker process per positive case, 1 warm-up and 1 measured call
  (the protocol's geco2 budget), 8 threads, threshold 0.11, 1-hour timeout.
Before every worker it records the host load (the 1-minute load average where there is
one; Windows has none, so the CPU LoadPercentage instead) and the GPU's utilisation,
memory in use and compute processes (nvidia-smi). Every attempt is one row of
``<out>/manifest.jsonl``. A restart skips every key that already has a row, failed ones
included, so failures are kept and nothing is rerun for nicer numbers; a worker that was
running when the host died has no row and runs again, which the row's ``attempt`` shows.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def query(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return repr(error)


def gpu_state():
    q = 'utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw'
    out = query(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'])
    state = dict(zip(q.split(','), (v.strip() for v in out.split(','))))
    state['compute_apps'] = query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
    return state


def host_state():
    state = dict(utc=now(), hostname=socket.gethostname(), gpu=gpu_state())
    if hasattr(os, 'getloadavg'):
        state['load1'] = os.getloadavg()[0]
    else:
        cpu = query(['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average'])
        state.update(load1=None, cpu_load_percent=float(cpu) if cpu.replace('.', '', 1).isdigit() else cpu)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='literature_geco2_cuda_setup.sh WORK directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', nargs='+', default=['B2', 'A', 'C'], choices=('B2', 'A', 'C'))
    args = p.parse_args()
    pkg, work, out = args.package.resolve(), args.work.resolve(), args.output.resolve()
    jobs = json.loads((pkg / 'jobs.json').read_text())
    worker = [sys.executable, '-I', str(pkg / 'tools/literature_geco2_worker.py'), '--repo', str(work / 'geco2'),
              '--weights', str(work / 'weights/GECO2FSCD.pth'), '--torch-home', str(work / 'torch-home'),
              '--threshold', '.11', '--device', 'cuda', '--deform-attn', 'cuda', '--threads', '8']
    manifest = out / 'manifest.jsonl'
    out.mkdir(parents=True, exist_ok=True)
    done = {(r['group'], r['case']) for r in map(json.loads, manifest.read_text().splitlines())} if manifest.exists() else set()
    attempts = out / 'attempts.jsonl'  # a row per start, so a worker cut off by a host failure is visible

    def run(group, case, cmd, result, timeout):
        start = dict(group=group, case=case, started=now())
        with open(attempts, 'a') as f:
            f.write(json.dumps(start) + '\n')
        starts = [json.loads(line) for line in attempts.read_text().splitlines()]
        attempt = sum(s['group'] == group and s['case'] == case for s in starts)
        host = host_state()
        t = time.perf_counter()
        try:
            # The lab PC's locale is cp950: decode explicitly so a stray byte cannot stop the run.
            r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=timeout)
            code, stderr = r.returncode, r.stderr
        except subprocess.TimeoutExpired as e:
            code, stderr = 'timeout', e.stderr if isinstance(e.stderr, str) else ''
        wall = (time.perf_counter() - t) * 1e3
        result.with_suffix('.stderr.txt').write_text(stderr or '', encoding='utf-8')
        row = dict(group=group, case=case, method='geco2_cuda', block=0, status='ok' if code == 0 else 'failed',
                   output=result.relative_to(out).as_posix(), returncode=code, wall_ms=wall, host=host, attempt=attempt,
                   cmd=[str(c) for c in cmd], finished=now())
        if code != 0:
            row['stderr_tail'] = (stderr or '')[-1500:]
        with open(manifest, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(group, case, row['status'], f'{wall:.0f} ms', flush=True)

    for group in args.groups:
        (out / group).mkdir(exist_ok=True)
        if group == 'B2':
            if ('B2', 'all') in done:
                continue
            items = [dict(it, image=str(pkg / it['image'])) for it in jobs['B2']]
            batch = out / 'B2' / 'items-cuda.json'
            batch.write_text(json.dumps(items) + '\n')
            result = out / 'B2' / 'geco2_cuda.jsonl'
            run('B2', 'all', worker + ['--batch', str(batch), '--output', str(result)], result, 3 * 3600)
            continue
        for job in jobs[group]:
            if (group, job['case']) in done:
                continue
            result = out / group / f"{job['case']}-b0.json"
            cmd = worker + ['--image', str(pkg / job['image']), '--exemplars', json.dumps(job['exemplars']),
                            '--warmup', '1', '--repeats', '1', '--output', str(result)]
            run(group, job['case'], cmd, result, 3600)
    (out / 'finished.txt').write_text(now() + '\n')


if __name__ == '__main__':
    main()
