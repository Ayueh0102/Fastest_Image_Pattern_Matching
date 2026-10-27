"""Round 52: opt-in match verification for small templates (border in-image NCC, size-aware threshold).

Run with python-port/venv. Arms are round 50's CURRENT_BEST research jobs (hybrid k3 and full GPU
peak2/fine4 k3) with the runner keys border_min_inside / small_template_area added; truth stays in
this parent and jobs carry images and settings only.

Subcommands
  dev        development selection of small_template_area (A0) on round 50's tiny scenes
             (scenes-dev and the group-C tiny cases), border_min_inside fixed at 0.5, hybrid k3
  run        holdout (round52_scenes.py output, after the freeze) or regression (round 50's
             group A and group C non-tiny cases) x arms into an output directory
  summarize  score every run with literature_score and decide the frozen hypotheses
Host rules as in round 50: every worker waits while another session's timing lock exists;
``run --formal`` holds this round's lock for the whole run.
"""
import argparse
import json
import statistics
from pathlib import Path
import sys
import time

import literature_compare as lc
import literature_score as score

BORDER_MIN_INSIDE = .5
A0_GRID = (256, 400, 576, 784, 1024, 1600)
DEV_TP_LOSS = .005  # of the development positives' truths
BACKENDS = ('ours_hybrid_k3', 'ours_fullgpu_k3')
RUNNER = lc.BUILD / 'stage52-dev/experiments/fpm_native_pipeline_runner'
LOCK_DIR = lc.LOCKS / 'round52-main'


def variants(a0):
    out = dict(off={}, border=dict(border_min_inside=BORDER_MIN_INSIDE))
    if a0:
        out.update(size=dict(small_template_area=float(a0)), both=dict(border_min_inside=BORDER_MIN_INSIDE, small_template_area=float(a0)))
    return out


def tiny_search():
    return dict(scales=[1.], angle=180, limit=100, score=.8, proposal_score=.5, overlap=.2, stop_layer=1, subpixel=False)


def scene_cases(path, split):
    """Driver cases from a scene generator's cases.json (tiny groups only); keeps the boundary edge flags."""
    out = []
    for c in json.loads(Path(path).read_text()):
        if c['group'] != 'tiny':
            continue
        truth = [dict(kind='pose', cx=t['pos_x'], cy=t['pos_y'], angle_deg=t['angle'], width=t['width'], height=t['height'],
                      edge=bool(t.get('edge', False))) for t in c['truth']]
        out.append(dict(name=c['name'], group='tiny', family=c['family'], condition=c['condition'], positive=c['positive'],
                        source=c['source'], template=c['template'], template_size=c['template_size'],
                        source_size=c.get('source_size'), truth=truth, search=tiny_search(), split=split))
    return out


def regression_cases(work):
    a = [dict(c, split='regression') for c in lc.cases_a(work)]
    c = [dict(x, split='regression') for x in lc.cases_c(lc.EXTERNAL / 'scenes-c') if x['family'] != 'tiny']
    return a + c


def job_for(case, arm, keys, repeats, warmup):
    job, from44 = lc.runner_job(case, arm, repeats, warmup)
    job.update(keys)
    return job, from44


def footprint(det):
    if det.get('corners'):
        return det['corners']
    n = score.normalise(det)
    return score.pose_corners(n['cx'], n['cy'], n['width'], n['height'], n['angle'])


def touches_border(det, width, height):
    """Corner beyond the outermost pixel centres: the footprint reaches the padded border."""
    return any(x < 0 or y < 0 or x > width - 1 or y > height - 1 for x, y in footprint(det))


def evaluate(case, text):
    """Counts for one runner output: strict TP, edge-truth TP, false positives split at the border, time, verify report."""
    parsed = lc.parse_runner(text, case)
    dets = parsed['detections']
    s = lc.case_score(case, dets)
    width, height = lc.image_size(case['source'])
    row = dict(gt=s['gt'], predictions=s['predictions'], time_ms=parsed['time_ms'], first_ms=parsed['first_ms'],
               verify=json.loads(text)['first'].get('match_verify'))
    if case['positive']:
        g = [score.normalise(t) for t in case['truth']]
        p = [score.normalise(d) for d in dets]
        pairs = score.match_poses(g, p, case.get('period', 360.))
        row.update(strict_tp=len(pairs), edge_gt=sum(t.get('edge', False) for t in case['truth']),
                   edge_tp=sum(case['truth'][i].get('edge', False) for i, _ in pairs),
                   extra=len(dets) - len(pairs))
    else:
        border = sum(touches_border(d, width, height) for d in dets)
        row.update(false_positives=s['false_positives'], border_fp=border, interior_fp=s['false_positives'] - border)
    return row


def run_jobs(cases, arms, out_dir, repeats, warmup, own_lock=None, runner=RUNNER):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'cases.json').write_text(json.dumps(cases, indent=1) + '\n')
    manifest = out_dir / 'manifest.jsonl'
    done = lc.manifest_done(manifest)
    for case in cases:
        for arm, variant, keys in arms:
            method = f'{arm}-{variant}'
            if (case['name'], method, 0) in done:
                continue
            out = out_dir / method / f"{case['name']}.json"
            job, from44 = job_for(case, arm, keys, repeats, warmup)
            job_path = out.with_suffix('.job.json')
            job_path.parent.mkdir(parents=True, exist_ok=True)
            job_path.write_text(json.dumps(job))
            w = lc.run_worker([runner, job_path], out, own_lock)
            if w['code'] == 0:
                out.write_text(w['stdout'])
            with open(manifest, 'a') as f:
                f.write(json.dumps(dict(case=case['name'], method=method, arm=arm, variant=variant, block=0,
                                        status='ok' if w['code'] == 0 else 'failed', output=str(out), job_from_round44=from44,
                                        returncode=w['code'], wall_ms=w['wall_ms'], host=w['host'], lock_wait_s=w['lock_wait_s'],
                                        stderr_tail=(out.with_suffix('.stderr.txt').read_text()[-400:] if w['code'] else None))) + '\n')


def table(out_dir):
    cases = {c['name']: c for c in json.loads((out_dir / 'cases.json').read_text())}
    rows, failures = [], []
    for e in lc.read_manifest(out_dir / 'manifest.jsonl'):
        case = cases[e['case']]
        if e['status'] != 'ok':
            failures.append(e)
            continue
        r = evaluate(case, lc.remapped(e['output']).read_text())
        rows.append(dict(r, case=e['case'], method=e['method'], arm=e['arm'], variant=e['variant'], positive=case['positive'],
                         family=case.get('family'), condition=case.get('condition')))
    return rows, failures


def totals(rows, method, positive):
    rs = [r for r in rows if r['method'] == method and r['positive'] == positive]
    keys = ('gt', 'strict_tp', 'edge_gt', 'edge_tp', 'extra') if positive else ('false_positives', 'border_fp', 'interior_fp')
    return dict(cases=len(rs), **{k: sum(r[k] for r in rs) for k in keys})


def select_a0(rows):
    """Frozen rule: border check on; among A0 whose positive strict-TP loss vs A0 off is <= 0.5% of the
    positives' truths, the fewest negative false positives; ties go to the smaller A0; none -> 0 (size check unused)."""
    base_p, base_n = totals(rows, 'ours_hybrid_k3-border', True), totals(rows, 'ours_hybrid_k3-border', False)
    grid = []
    for a0 in A0_GRID:
        p, n = totals(rows, f'ours_hybrid_k3-both{a0}', True), totals(rows, f'ours_hybrid_k3-both{a0}', False)
        loss = base_p['strict_tp'] - p['strict_tp']
        grid.append(dict(a0=a0, strict_tp=p['strict_tp'], tp_loss=loss, false_positives=n['false_positives'],
                         eligible=loss <= DEV_TP_LOSS * base_p['gt']))
    ok = [g for g in grid if g['eligible']]
    chosen = min(ok, key=lambda g: (g['false_positives'], g['a0']))['a0'] if ok else 0
    return dict(rule=select_a0.__doc__.strip(), border=dict(positives=base_p, negatives=base_n), grid=grid, chosen=chosen)


def dev(args):
    cases = scene_cases(lc.EXTERNAL / 'scenes-dev/cases.json', 'dev') + scene_cases(lc.EXTERNAL / 'scenes-c/cases.json', 'dev-round50c')
    arms = [('ours_hybrid_k3', 'off', {}), ('ours_hybrid_k3', 'border', dict(border_min_inside=BORDER_MIN_INSIDE))]
    arms += [('ours_hybrid_k3', f'both{a0}', dict(border_min_inside=BORDER_MIN_INSIDE, small_template_area=float(a0))) for a0 in A0_GRID]
    run_jobs(cases, arms, args.output, repeats=1, warmup=0.)
    rows, failures = table(args.output)
    record = dict(stage=52, cases=len(cases), failures=failures, selection=select_a0(rows),
                  off=dict(positives=totals(rows, 'ours_hybrid_k3-off', True), negatives=totals(rows, 'ours_hybrid_k3-off', False)))
    (args.output / 'dev-selection.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(dict(chosen=record['selection']['chosen'], grid=record['selection']['grid'], off=record['off'],
                          border=record['selection']['border']), indent=1))


def run(args):
    protocol = json.loads(args.protocol.read_text())
    a0 = protocol['selection']['small_template_area']
    if args.group == 'holdout':
        cases = scene_cases(args.scenes / 'cases.json', 'holdout')
        arms = [(b, v, k) for b in BACKENDS for v, k in variants(a0).items()]
        repeats, warmup = 3, .3
    else:
        cases = regression_cases(args.output / 'inputs')
        arms = [(b, v, k) for b in BACKENDS for v, k in variants(a0).items() if v in ('off', 'both' if a0 else 'border')]
        repeats, warmup = 1, 0.
    lock = None
    if args.formal:
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        lock = LOCK_DIR / f'timing-lock-round52-{args.group}.txt'
        while lc.foreign_locks(lock) or lc.heavy_processes():  # nothing heavy, ours included, may overlap formal timing
            time.sleep(30)
        lock.write_text(f'round 52 (small-template match verification) {args.group}\nstart_utc {lc.now()}\ndeleted when done\n')
    try:
        run_jobs(cases, arms, args.output, repeats, warmup, own_lock=lock, runner=args.runner)
    finally:
        if lock is not None and lock.exists():
            lock.unlink()


def median_ratio(rows, a, b):
    """Median over cases of time(b)/time(a) - 1, matched by case."""
    ta = {r['case']: r['time_ms'] for r in rows if r['method'] == a}
    ratios = [r['time_ms'] / ta[r['case']] - 1 for r in rows if r['method'] == b and r['case'] in ta and ta[r['case']]]
    return statistics.median(ratios) if ratios else None


def hypotheses(holdout, regression, a0):
    out = {}
    for b in BACKENDS:
        off_n, bor_n = totals(holdout, f'{b}-off', False), totals(holdout, f'{b}-border', False)
        off_p, bor_p = totals(holdout, f'{b}-off', True), totals(holdout, f'{b}-border', True)
        h1 = dict(off=off_n, border=bor_n, pass_=(bor_n['border_fp'] <= .1 * off_n['border_fp']) if off_n['border_fp'] >= 10 else None)
        h2 = dict(off=off_p, border=bor_p, pass_=bor_p['strict_tp'] >= off_p['strict_tp'] - .002 * off_p['gt']
                  and bor_p['edge_tp'] >= off_p['edge_tp'] - 1)
        res = dict(H1=h1, H2=h2)
        if a0:
            both_n, both_p = totals(holdout, f'{b}-both', False), totals(holdout, f'{b}-both', True)
            res['H3'] = dict(border=bor_n, both=both_n,
                             pass_=(both_n['interior_fp'] <= .5 * bor_n['interior_fp']) if bor_n['interior_fp'] >= 10 else None)
            res['H4'] = dict(border=bor_p, both=both_p, pass_=both_p['strict_tp'] >= bor_p['strict_tp'] - .005 * bor_p['gt'])
            res['H6'] = dict(median_time_increase=median_ratio(holdout, f'{b}-off', f'{b}-both'))
            res['H6']['pass_'] = res['H6']['median_time_increase'] is not None and res['H6']['median_time_increase'] <= .05
        if regression is not None:
            off = {r['case']: r for r in regression if r['method'] == f'{b}-off'}
            worse = []
            for r in regression:
                if r['method'] != f"{b}-{'both' if a0 else 'border'}" or r['case'] not in off:
                    continue
                o = off[r['case']]
                if (r['positive'] and r['strict_tp'] < o['strict_tp']) or (not r['positive'] and r['false_positives'] > o['false_positives']):
                    worse.append(r['case'])
            res['H5'] = dict(cases=len(off), worse=worse, pass_=not worse)
        out[b] = res
    return out


def identical_outputs(out_dir, a, b):
    """Cases whose first-call matches are byte-identical between methods a and b."""
    files = {}
    for e in lc.read_manifest(out_dir / 'manifest.jsonl'):
        if e['status'] == 'ok' and e['method'] in (a, b):
            files.setdefault(e['case'], {})[e['method']] = json.loads(lc.remapped(e['output']).read_text())['first']['matches']
    same = [c for c, v in files.items() if len(v) == 2 and v[a] == v[b]]
    return dict(cases=len(files), identical=len(same), different=sorted(c for c in files if c not in same))


def summarize(args):
    for pair in args.remap or []:
        lc.REMAP.append(tuple(pair.split('=', 1)))
    protocol = json.loads(args.protocol.read_text())
    a0 = protocol['selection']['small_template_area']
    holdout, hfail = table(args.holdout)
    regression, rfail = table(args.regression) if args.regression else (None, [])
    summary = dict(protocol_sha256=lc.sha(args.protocol), small_template_area=a0, created=lc.now(),
                   holdout=dict(rows=holdout, failures=hfail,
                                totals={m: dict(positives=totals(holdout, m, True), negatives=totals(holdout, m, False))
                                        for m in sorted({r['method'] for r in holdout})}),
                   hypotheses=hypotheses(holdout, regression, a0))
    if regression is not None:
        summary['regression'] = dict(rows=regression, failures=rfail,
                                     identity={b: identical_outputs(args.regression, f'{b}-off', f"{b}-{'both' if a0 else 'border'}")
                                               for b in BACKENDS})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(json.dumps({b: {h: v.get('pass_') for h, v in hs.items()} for b, hs in summary['hypotheses'].items()}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dev')
    d.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--group', required=True, choices=('holdout', 'regression'))
    r.add_argument('--protocol', type=Path, required=True)
    r.add_argument('--scenes', type=Path, help='round52_scenes.py output (holdout)')
    r.add_argument('--runner', type=Path, default=RUNNER)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--formal', action='store_true', help="hold this round's timing lock for the whole run")
    m = sub.add_parser('summarize')
    m.add_argument('--protocol', type=Path, required=True)
    m.add_argument('--holdout', type=Path, required=True)
    m.add_argument('--regression', type=Path)
    m.add_argument('--remap', nargs='*', help='OLD=NEW output-path prefixes for replaying archived runs')
    m.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    dict(dev=dev, run=run, summarize=summarize)[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
