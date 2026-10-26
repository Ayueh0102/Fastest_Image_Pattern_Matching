"""Round 50 phases 2-4: development selections, measured runs and scoring of the literature comparison.

Run with python-port/venv. Third-party methods run in their own environments via
their workers (literature_tdcm_worker.py, literature_geco2_worker.py); the
OpenCV baseline is literature_opencv_worker.py; our methods are the frozen
build's fpm_cli and fpm_native_pipeline_runner. Truth stays in this parent:
workers receive images, templates or exemplar boxes only.

Subcommands
  dev-tdcm-pipeline   S1 development pairs, both TDCM scripts; pick the primary
  dev-tdcm-threshold  development scenes, TDCM-adapted peak-threshold grid
  dev-ours-b2         FSC-147 val subset, our minimum-score grid (secondary B2 arm)
  run                 one group (A, C, B1, B2) x methods x blocks into an output dir
  summarize           score every run, check hypotheses, write summaries

Host rules: before every worker the parent checks the shared notes folder for
other sessions' timing locks and waits while one exists; ``run --formal`` holds
this round's own lock file for its whole duration. Every worker record keeps the
1-minute load and the processes above 50% CPU at its start (recorded, not gated).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import literature_score as score  # noqa: E402

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
PROJECT = Path('/Users/Ayueh/Fastest_Image_Pattern_Matching')
BUILD = PROJECT / 'build'
EXTERNAL = BUILD / 'stage50-external'
LOCKS = Path.home() / 'Library/Application Support/Claude/side-session-notes'
STAGE37 = BUILD / 'stage37-full-gpu/formal'
STAGE44 = BUILD / 'stage44-dedup-peak2/formal'
STAGE35 = BUILD / 'stage35-real-rgb/frozen-inputs/cases.json'
TESTBED = BUILD / 'testbed-v1'
PY = Path(sys.executable)

# CURRENT_BEST research settings (round-44 jobs); inputs, scales and budgets come from each case.
RUNNER_BASE = dict(angle=180, overlap=0.2, stop_layer=1, subpixel=False, backend=1, min_area=256, ncc_mode=2,
                   overlap_enabled=False, refine=True, invert_source=False, coarse_peak_mode=1, fine_ncc_mode=2,
                   fine_fused=True, fine_angle_reuse=True, pose_resident_mode=0, pose_input_mode=1, pose_storage_mode=1,
                   refinement_backend=1, full_gpu_profiling=False, candidate_dedup_radius=0.5, candidate_dedup_angle_steps=3.0)
ARMS = {
    'ours_hybrid_k0': dict(full_gpu=False, full_gpu_peak_mode=0, full_gpu_fine_mode=0, candidate_dedup_top_k=0),
    'ours_hybrid_k3': dict(full_gpu=False, full_gpu_peak_mode=0, full_gpu_fine_mode=0, candidate_dedup_top_k=3),
    'ours_fullgpu_k0': dict(full_gpu=True, full_gpu_peak_mode=2, full_gpu_fine_mode=4, full_gpu_peak_batch=8, candidate_dedup_top_k=0),
    'ours_fullgpu_k3': dict(full_gpu=True, full_gpu_peak_mode=2, full_gpu_fine_mode=4, full_gpu_peak_batch=8, candidate_dedup_top_k=3),
}
STAGE44_ARM = {'ours_hybrid_k0': 'hybrid_best_static_off', 'ours_hybrid_k3': 'hybrid_best_static_k3',
               'ours_fullgpu_k0': 'full_gpu_parallel_static_off', 'ours_fullgpu_k3': 'full_gpu_parallel_static_k3'}
SHALLOW = dict(min_area=512, proposal_score=0.25, limit=300)  # round-35 manual shallow pyramid
PAPER_S1 = dict(loc_err=0.44, rot_err=1.03, miou=0.955, succ=99.6)  # TDCM Table 2, S1, "Ours"
TDCM_THRESHOLD_GRID = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
B2_SCORE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
B2_DEV_SAMPLE, B2_DEV_SEED = 300, 5050


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- host and locks

def foreign_locks(own=None):
    return sorted(str(p) for p in LOCKS.glob('*/timing-lock-*.txt') if own is None or p.resolve() != Path(own).resolve())


def competing():
    out = subprocess.run(['ps', '-A', '-o', '%cpu=,comm='], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            try:
                cpu = float(parts[0])
            except ValueError:
                continue
            if cpu >= 50:
                rows.append(dict(cpu=cpu, command=Path(parts[1]).name))
    return sorted(rows, key=lambda r: -r['cpu'])[:6]


def host_state():
    return dict(utc=now(), load1=os.getloadavg()[0], busy=competing())


def wait_for_locks(own=None, poll=30):
    waited = 0
    while foreign_locks(own):
        time.sleep(poll)
        waited += poll
    return waited


class OwnLock:
    def __init__(self, phase, what, expected):
        self.path = LOCKS / 'local_1b4c2052-63a8-4a9f-94f8-77e6ad04fbca' / f'timing-lock-round50-{phase}.txt'
        self.text = f'round 50 (literature comparison) {phase}\nstart_utc {now()}\nwhat {what}\nexpected {expected}; deleted when done\n'

    def __enter__(self):
        wait_for_locks()
        self.path.write_text(self.text)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def run_worker(cmd, out, own_lock=None, timeout=3600, env=None):
    waited = wait_for_locks(own_lock)
    host = host_state()
    t = time.perf_counter()
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, env=env)
        code, stdout, stderr = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, stdout, stderr = 'timeout', e.stdout or '', (e.stderr or '') if isinstance(e.stderr, str) else ''
    wall = (time.perf_counter() - t) * 1e3
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.stderr.txt').write_text(stderr if isinstance(stderr, str) else '')
    return dict(code=code, stdout=stdout if isinstance(stdout, str) else '', wall_ms=wall, host=host, lock_wait_s=waited,
                cmd=[str(c) for c in cmd])


# ---------------------------------------------------------------- cases

def pose_truth(items, width, height, period=360.):
    return [dict(kind='pose', cx=t['pos_x'], cy=t['pos_y'], angle_deg=t['angle'], width=width * t.get('scale', 1.),
                 height=height * t.get('scale', 1.)) for t in items]


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def wrong_template(case, out_dir):
    """Dst9 (gray) resized to the case's template size: same size, other content."""
    import cv2
    tw, th = case['template_size']
    path = out_dir / f"wrong-{tw}x{th}.png"
    if not path.exists():
        image = cv2.imread(str(PROJECT / 'Test Images/Dst9.bmp'), cv2.IMREAD_GRAYSCALE)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA))
    return path


def stage44_job(name, arm):
    path = STAGE44 / 'jobs' / f'{name}-{STAGE44_ARM[arm]}-b0.json'
    return json.loads(path.read_text()) if path.exists() else None


def cases_a(work):
    manifest = json.loads((STAGE37 / 'manifest.json').read_text())
    cases = []
    positives = ['many128_pose', 'many512_pose', 'controlled6', 'heldout12', 'small_fullres', '8k_t256_angle180_3scales',
                 'boundary_one']
    for c in manifest['cases']:
        if c['name'] not in positives + ['empty_scene', 'constant_source']:
            continue
        tw, th = image_size(c['template'])
        truth = json.loads(c['truth']) if isinstance(c['truth'], str) else c['truth']
        scales = json.loads(c['scales']) if isinstance(c['scales'], str) else c['scales']
        cases.append(dict(name=c['name'], group='A', family='round37', positive=bool(truth), source=c['source'],
                          template=c['template'], template_size=[tw, th], truth=pose_truth(truth, tw, th),
                          search=dict(scales=scales, angle=c['angle'], limit=c['limit'], score=c['score'],
                                      proposal_score=c.get('proposal_score', .5), overlap=c['overlap'],
                                      stop_layer=c.get('stop_layer', 1), subpixel=bool(c.get('subpixel', 0)))))
    for c in [c for c in cases if c['name'] in positives]:
        cases.append(dict(c, name=c['name'] + '_wrongtemplate', family='wrong_template', positive=False, truth=[],
                          template=str(wrong_template(c, work / 'inputs'))))
    for c in json.loads(STAGE35.read_text()):
        if c['name'] not in ('development_clean', 'development_exposure', 'development_white_balance',
                             'heldout_clean', 'heldout_exposure', 'heldout_white_balance'):
            continue
        tw, th = image_size(c['template'])
        s = c['search']
        cases.append(dict(name='r35_' + c['name'], group='A', family='round35', positive=True, source=c['source'],
                          template=c['template'], template_size=[tw, th], truth=pose_truth(c['truth'], tw, th),
                          search=dict(scales=s['scales'], angle=s['angle'], limit=s['limit'], score=s['score'],
                                      proposal_score=s['proposal_score'], overlap=s['overlap'], stop_layer=s['stop_layer'],
                                      subpixel=bool(s['subpixel']))))
    for scene, ref, scales, period in (('bl_gray_crowded', 'bl_gray_reference', [.8, 1., 1.25], 360.),
                                       ('bl_gray_gloss_shadow', 'bl_gray_reference', [1.], 360.),
                                       ('bl_sym_scene', 'bl_sym_reference', [1.], 180.)):
        meta = json.loads((TESTBED / 'blender' / f'{scene}.json').read_text())
        template = TESTBED / 'final' / f'{ref}_template.png'
        tw, th = image_size(template)
        truth = [dict(pos_x=o['projected_center'][0], pos_y=o['projected_center'][1], angle=o['angle'], scale=o['scale'])
                 for o in meta['objects']]
        cases.append(dict(name=scene, group='A', family='testbed_blender', positive=True, source=str(TESTBED / 'blender' / meta['image']),
                          template=str(template), template_size=[tw, th], truth=pose_truth(truth, tw, th), period=period,
                          search=dict(scales=scales, angle=180, limit=64, score=.65, proposal_score=.5, overlap=.2,
                                      stop_layer=1, subpixel=False)))
    return cases


def cases_c(scenes_c):
    cases = []
    for c in json.loads((scenes_c / 'cases.json').read_text()):
        tw, th = c['template_size']
        truth = [dict(kind='pose', cx=t['pos_x'], cy=t['pos_y'], angle_deg=t['angle'], width=t['width'], height=t['height'])
                 for t in c['truth']]
        if c['group'] == 'tiny':
            search = dict(scales=[1.], angle=180, limit=100, score=.8, proposal_score=.5, overlap=.2, stop_layer=1, subpixel=False)
        elif c['group'] == 'challenge':
            search = dict(scales=[.75, 1., 1.25], angle=180, limit=70, score=.8, proposal_score=.5, overlap=.2, stop_layer=1,
                          subpixel=False)
        else:
            search = dict(scales=[.75, 1., 1.25], angle=180, limit=3 * c['expected_count'], score=.8, proposal_score=.5,
                          overlap=.2, stop_layer=1, subpixel=False)
        cases.append(dict(name=c['name'], group='C', family=c['group'], positive=c['positive'], source=c['source'],
                          template=c['template'], template_size=[tw, th], truth=truth, search=search,
                          kind=c.get('kind'), condition=c.get('condition')))
    for meta_path in sorted((TESTBED / 'synthetic').glob('syn_*.truth.json')):
        meta = json.loads(meta_path.read_text())
        if meta['condition'] == 'neutral':
            continue
        template = TESTBED / 'synthetic' / meta['template']
        tw, th = image_size(template)
        truth = pose_truth(meta['truth'], tw, th)
        for t, src in zip(truth, meta['truth']):
            t['same_colour'] = src['same_colour']
        cases.append(dict(name=meta['scene'], group='C', family='illumination', positive=True,
                          source=str(TESTBED / 'synthetic' / meta['image']), template=str(template), template_size=[tw, th],
                          truth=truth, search=dict(scales=[.85, 1., 1.15], angle=180, limit=64, score=.65, proposal_score=.5,
                                                   overlap=.2, stop_layer=1, subpixel=False), condition=meta['condition']))
    for scene in ('bl_rgb_white', 'bl_rgb_warm3200', 'bl_rgb_mixed'):
        meta = json.loads((TESTBED / 'blender' / f'{scene}.json').read_text())
        template = TESTBED / 'final' / 'bl_rgb_reference_template.png'
        tw, th = image_size(template)
        truth = [dict(pos_x=o['projected_center'][0], pos_y=o['projected_center'][1], angle=o['angle'], scale=o['scale'])
                 for o in meta['objects']]
        cases.append(dict(name=scene, group='C', family='illumination', positive=True, source=str(TESTBED / 'blender' / meta['image']),
                          template=str(template), template_size=[tw, th], truth=pose_truth(truth, tw, th),
                          search=dict(scales=[.9, 1., 1.1], angle=180, limit=64, score=.65, proposal_score=.5, overlap=.2,
                                      stop_layer=1, subpixel=False)))
    return cases


def exemplars_for(case, n=3):
    boxes = [list(score.normalise(t)['box']) for t in case['truth'][:n]]
    return [[round(v, 2) for v in b] for b in boxes]


# ---------------------------------------------------------------- method adapters

def ours_core_cmd(case, build):
    s = case['search']
    return [build / 'fpm_cli', '--src', case['source'], '--tmpl', case['template'], '--target-num', s['limit'],
            '--score', s['score'], '--angle', s['angle'], '--overlap', s['overlap'], '--subpixel']


def parse_core(stdout, case):
    d = json.loads(stdout)
    tw, th = case['template_size']
    dets = [dict(kind='pose', cx=m['pos_x'], cy=m['pos_y'], angle_deg=m['angle'], width=tw, height=th, corners=m['corners'],
                 score=m['score']) for m in d['matches']]
    return dict(detections=dets, time_ms=d['match_ms'], times_ms=[d['match_ms']], construct_ms=d['learn_ms'])


def runner_job(case, arm, repeats, warmup_seconds, shallow=False):
    reference = stage44_job(case['name'], arm) if case['group'] == 'A' and not shallow else None
    if reference is not None:
        job = dict(reference, repeats=repeats, warmup_seconds=warmup_seconds, full_gpu_profiling=False)
        job.update(source=case['source'], template=case['template'])
        return job, True
    s = case['search']
    job = dict(RUNNER_BASE, **ARMS[arm], source=case['source'], template=case['template'], scales=s['scales'], angle=s['angle'],
               limit=s['limit'], score=s['score'], proposal_score=s['proposal_score'], overlap=s['overlap'],
               stop_layer=s['stop_layer'], subpixel=s['subpixel'], repeats=repeats, warmup_seconds=warmup_seconds)
    if shallow:
        job.update(SHALLOW)
    return job, False


def parse_runner(stdout, case):
    d = json.loads(stdout)
    tw, th = case['template_size']
    runs = d['runs']
    call = runs[0]

    def dets(matches):
        return [dict(kind='pose', cx=m['pos_x'], cy=m['pos_y'], angle_deg=m['angle'], width=tw * m['scale'],
                     height=th * m['scale'], corners=m['corners'], score=m['score']) for m in matches]
    return dict(detections=dets(call['matches']), first_matches=d['first']['matches'],
                time_ms=statistics.median(r['total_ms'] for r in runs), times_ms=[r['total_ms'] for r in runs],
                first_ms=d['first']['total_ms'], construct_ms=d['construct_ms'], exact_repeat=d['exact_repeat'],
                peak_rss_bytes=d['final_memory']['peak_rss_bytes'], full_gpu_bytes=call['full_gpu']['gpu_bytes'],
                frame_gpu_bytes=call['frame_coarse_gpu_bytes'] + call['frame_fine_gpu_bytes'], pose_gpu_bytes=call['pose_gpu_bytes'])


def opencv_cmd(case, out, threads=8, max_maps=0, angle_step=1.):
    s = case['search']
    cmd = [PY, '-I', TOOLS / 'literature_opencv_worker.py', '--image', case['source'], '--template', case['template'],
           '--angle-step', angle_step, '--scales', *s['scales'], '--score', s['score'], '--limit', s['limit'],
           '--overlap', s['overlap'], '--threads', threads, '--output', out]
    if max_maps:
        cmd += ['--max-maps', max_maps]
    return cmd


def parse_opencv(out, case):
    d = json.loads(out.read_text())
    t = d['timing_ms']
    return dict(detections=d['detections'], time_ms=t['search'] + t['nms'], times_ms=[t['search'] + t['nms']],
                complete=d['maps'] == d['planned_maps'], maps=d['maps'], planned_maps=d['planned_maps'],
                extrapolated_ms=t['extrapolated_full_search'])


def tdcm_cmd(case, out, device, pipeline, threshold, threads=8, warmup=1, repeats=3, adapt=True):
    cmd = [EXTERNAL / 'envs/tdcm-py312/bin/python', '-I', TOOLS / 'literature_tdcm_worker.py', '--repo', EXTERNAL / 'repos/tdcm',
           '--weights', EXTERNAL / 'repos/tdcm/dict/model.pth', '--image', case['source'], '--template', case['template'],
           '--device', device, '--pipeline', pipeline, '--threads', threads, '--warmup', warmup, '--repeats', repeats,
           '--output', out]
    if adapt:
        cmd += ['--adapt', '--limit', case['search']['limit'], '--overlap', case['search']['overlap'], '--peak-threshold', threshold]
    return cmd


def parse_worker(out, case):
    d = json.loads(out.read_text())
    t = d['timing_ms']
    return dict(detections=d['detections'], time_ms=t['call_median'], times_ms=t['calls'], construct_ms=t['load_model'],
                exact_repeat=d['exact_repeat'], memory=d['memory'])


def geco2_cmd(case, out, device, threads=8, warmup=1, repeats=3, exemplars=None, deform='reference'):
    return [EXTERNAL / 'envs/geco2-py312/bin/python', '-I', TOOLS / 'literature_geco2_worker.py', '--repo', EXTERNAL / 'repos/geco2',
            '--weights', EXTERNAL / 'data/geco2-weights-hf/GECO2FSCD.pth', '--torch-home', EXTERNAL / 'torch-home',
            '--image', case['source'], '--exemplars', json.dumps(exemplars if exemplars is not None else exemplars_for(case)),
            '--threshold', .11, '--device', device, '--deform-attn', deform, '--threads', threads, '--warmup', warmup,
            '--repeats', repeats, '--output', out]


# ---------------------------------------------------------------- scoring helpers

def case_score(case, detections):
    period = case.get('period', 360.)
    if case['positive']:
        r = score.score_pose_image(case['truth'], detections, period) if all(d['kind'] == 'pose' for d in detections) else None
        boxes = score.score_box_images([([score.normalise(t)['box'] for t in case['truth']], detections)])
        out = dict(gt=len(case['truth']), predictions=len(detections), boxes=boxes)
        if r is not None:
            out.update(strict_tp=r['strict_tp'], strict_fp=r['strict_fp'], strict_fn=r['strict_fn'],
                       association_tp=r['association_tp'], strict_errors=score.error_summary(r['strict_errors']),
                       association_errors=score.error_summary(r['association_errors']),
                       association_pairs=r['association_errors'])
            if any('same_colour' in t for t in case['truth']):
                same = [t for t in case['truth'] if t.get('same_colour')]
                rs = score.score_pose_image(same, detections, period)
                out['same_colour'] = dict(gt=len(same), strict_tp=rs['strict_tp'], association_tp=rs['association_tp'])
        return out
    return dict(gt=0, predictions=len(detections), false_positives=len(detections))


# ---------------------------------------------------------------- development selections

def b1_paper_metrics(truth, outputs):
    """test_single.py definitions: success = score > 0.01 and IoU > 0.1; errors averaged over successes."""
    n, rows = len(truth), []
    for t in truth:
        dets = outputs.get(t['index'], [])
        if not dets:
            rows.append(None)
            continue
        d = dets[0]
        g = dict(cx=t['cx'], cy=t['cy'], scale_x=t['scale_x'], scale_y=t['scale_y'], angle_deg=t['angle'])
        p = dict(cx=d['cx'], cy=d['cy'], scale_x=d['width'] / 36, scale_y=d['height'] / 36, angle_deg=d['angle_deg'])
        iou = score.tdcm_paper_iou(g, p)
        ok = d.get('score', 1.) > 0.01 and iou > 0.1
        rows.append(dict(ok=ok, iou=iou, loc=math.hypot(p['cx'] - g['cx'], p['cy'] - g['cy']),
                         rot=score.angle_error(p['angle_deg'], g['angle_deg'])))
    good = [r for r in rows if r and r['ok']]
    mean = lambda k: sum(r[k] for r in good) / len(good) if good else None  # noqa: E731
    return dict(samples=n, successes=len(good), succ=100. * len(good) / n if n else None, miou=mean('iou'),
                loc_err=mean('loc'), rot_err=mean('rot'))


def paper_distance(m, paper=PAPER_S1):
    if m['successes'] == 0:
        return math.inf
    return (abs(m['miou'] - paper['miou']) / paper['miou'] + abs(m['succ'] - paper['succ']) / paper['succ']
            + abs(m['loc_err'] - paper['loc_err']) / paper['loc_err'] + abs(m['rot_err'] - paper['rot_err']) / paper['rot_err'])


def read_batch(path):
    out, header, footer = {}, None, None
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if 'header' in d:
            header = d['header']
        elif 'footer' in d:
            footer = d['footer']
        else:
            out[d['id']] = d
    return header, out, footer


def dev_tdcm_pipeline(args):
    fixtures = args.fixtures / 'S1-dev'
    truth = json.loads((fixtures / 'truth.json').read_text())
    items = [dict(id=t['index'], image=str(fixtures / f"{t['index']:04d}_search.png"),
                  template=str(fixtures / f"{t['index']:04d}_template.png")) for t in truth]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'items.json').write_text(json.dumps(items, indent=1) + '\n')
    record = dict(rule='smaller sum of relative differences to the paper S1 row (mIoU, Succ, Loc.Err, Rot.Err); tie -> test_single',
                  paper_s1=PAPER_S1, fixtures=str(fixtures), truth_sha256=sha(fixtures / 'truth.json'), pipelines={})
    for pipeline in ('test_single', 'test_py'):
        out = args.output / f'{pipeline}.jsonl'
        cmd = [EXTERNAL / 'envs/tdcm-py312/bin/python', '-I', TOOLS / 'literature_tdcm_worker.py', '--repo', EXTERNAL / 'repos/tdcm',
               '--weights', EXTERNAL / 'repos/tdcm/dict/model.pth', '--batch', args.output / 'items.json', '--device', 'cpu',
               '--pipeline', pipeline, '--threads', 8, '--output', out]
        w = run_worker(cmd, out)
        if w['code'] != 0:
            record['pipelines'][pipeline] = dict(status='failed', worker=w)
            continue
        _, results, footer = read_batch(out)
        m = b1_paper_metrics(truth, {k: v['detections'] for k, v in results.items()})
        record['pipelines'][pipeline] = dict(status='ok', metrics=m, distance=paper_distance(m), worker=w, footer=footer,
                                             output_sha256=sha(out))
    ok = {k: v for k, v in record['pipelines'].items() if v['status'] == 'ok'}
    choice = min(ok, key=lambda k: (ok[k]['distance'], k != 'test_single')) if ok else None
    record.update(choice=choice, secondary=[k for k in ('test_single', 'test_py') if k != choice])
    (args.output / 'dev-tdcm-pipeline.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: (v.get('metrics'), v.get('distance')) for k, v in record['pipelines'].items()}, indent=1), 'choice', choice)


def dev_tdcm_threshold(args):
    dev = json.loads((args.scenes / 'cases.json').read_text())
    cases = []
    for c in dev:
        truth = [dict(kind='pose', cx=t['pos_x'], cy=t['pos_y'], angle_deg=t['angle'], width=t['width'], height=t['height'])
                 for t in c['truth']]
        limit = 3 * c['expected_count'] if c['expected_count'] else 100
        cases.append(dict(name=c['name'], positive=c['positive'], source=c['source'], template=c['template'],
                          template_size=c['template_size'], truth=truth, search=dict(limit=limit if c['positive'] else 100, overlap=.2)))
    record = dict(rule='largest pooled strict F1 over the development scenes (TP, FP over positives and negatives, FN); tie -> higher threshold',
                  grid=TDCM_THRESHOLD_GRID, pipeline=args.pipeline, scenes_sha256=sha(args.scenes / 'cases.json'), thresholds={})
    for th in TDCM_THRESHOLD_GRID:
        tp = fp = fn = 0
        rows = []
        for c in cases:
            out = args.output / f'th{th}' / f"{c['name']}.json"
            w = run_worker(tdcm_cmd(c, out, args.device, args.pipeline, th, warmup=0, repeats=1), out)
            if w['code'] != 0:
                rows.append(dict(case=c['name'], status='failed', stderr=(out.with_suffix('.stderr.txt')).read_text()[-400:]))
                fn += len(c['truth'])
                continue
            dets = json.loads(out.read_text())['detections']
            s = case_score(c, dets)
            if c['positive']:
                tp, fp, fn = tp + s['strict_tp'], fp + s['strict_fp'], fn + s['strict_fn']
            else:
                fp += s['false_positives']
            rows.append(dict(case=c['name'], status='ok', **{k: s.get(k) for k in ('gt', 'predictions', 'strict_tp', 'association_tp',
                                                                                  'false_positives')}))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.
        record['thresholds'][str(th)] = dict(tp=tp, fp=fp, fn=fn, f1=f1, cases=rows)
        print(th, dict(tp=tp, fp=fp, fn=fn, f1=round(f1, 4)), flush=True)
    best = max(TDCM_THRESHOLD_GRID, key=lambda th: (record['thresholds'][str(th)]['f1'], th))
    record['choice'] = best
    (args.output / 'dev-tdcm-threshold.json').write_text(json.dumps(record, indent=1) + '\n')
    print('choice', best)


def fsc_cases(split, fsc_images, annotations, sample=None, seed=None):
    a = json.loads((annotations / 'annotation_FSC147_384.json').read_text())
    names = json.loads((annotations / 'Train_Test_Val_FSC_147.json').read_text())[split]
    if sample:
        names = sorted(random.Random(seed).sample(names, sample), key=lambda n: int(n.split('.')[0]))
    return [dict(name=n, image=str(fsc_images / n), annotation=a[n]) for n in names]


def fsc_template(case, out_dir):
    """Exemplar 0 crop (floor/ceil of its polygon), saved as the template every method gets."""
    import cv2
    import numpy as np
    box = np.asarray(case['annotation']['box_examples_coordinates'][0], float)
    x0, y0 = np.floor(box.min(axis=0)).astype(int)
    x1, y1 = np.ceil(box.max(axis=0)).astype(int)
    path = out_dir / f"{case['name'].split('.')[0]}-ex0.png"
    if not path.exists():
        image = cv2.imread(case['image'], cv2.IMREAD_COLOR)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image[max(0, y0):y1, max(0, x0):x1])
    return path, [int(x0), int(y0), int(x1), int(y1)]


def b2_job(case, template, arm, score_min, repeats=1, warmup_seconds=0.):
    tw, th = image_size(template)
    s = dict(scales=[.75, 1., 1.25], angle=180, limit=4000, score=score_min, proposal_score=.5, overlap=.2, stop_layer=1,
             subpixel=False)
    job = dict(RUNNER_BASE, **ARMS[arm], source=case['image'], template=str(template), scales=s['scales'], angle=s['angle'],
               limit=s['limit'], score=s['score'], proposal_score=s['proposal_score'], overlap=s['overlap'],
               stop_layer=s['stop_layer'], subpixel=s['subpixel'], repeats=repeats, warmup_seconds=warmup_seconds)
    return job, (tw, th)


def dev_ours_b2(args):
    cases = fsc_cases('val', args.fsc_images, args.annotations, B2_DEV_SAMPLE, B2_DEV_SEED)
    record = dict(rule='lowest count MAE on the FSC-147 val subset; tie -> higher score', grid=B2_SCORE_GRID, arm=args.arm,
                  subset=[c['name'] for c in cases], seed=B2_DEV_SEED, scores={})
    for sc in B2_SCORE_GRID:
        pairs, failures = [], 0
        for c in cases:
            template, _ = fsc_template(c, args.output / 'templates')
            job, size = b2_job(c, template, args.arm, sc)
            out = args.output / f's{sc}' / f"{c['name'].split('.')[0]}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix('.job.json').write_text(json.dumps(job, indent=1))
            w = run_worker([args.build / 'experiments/fpm_native_pipeline_runner', out.with_suffix('.job.json')], out)
            gt = len(c['annotation']['points'])
            if w['code'] != 0:
                failures += 1
                pairs.append((gt, 0))
                continue
            out.write_text(w['stdout'])
            pairs.append((gt, len(json.loads(w['stdout'])['runs'][0]['matches'])))
        record['scores'][str(sc)] = dict(count=score.count_errors(pairs), failures=failures)
        print(sc, record['scores'][str(sc)], flush=True)
    best = min(B2_SCORE_GRID, key=lambda sc: (record['scores'][str(sc)]['count']['mae'], -sc))
    record['choice'] = best
    (args.output / 'dev-ours-b2.json').write_text(json.dumps(record, indent=1) + '\n')
    print('choice', best)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('dev-tdcm-pipeline')
    a.add_argument('--fixtures', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('dev-tdcm-threshold')
    b.add_argument('--scenes', type=Path, required=True)
    b.add_argument('--pipeline', required=True, choices=('test_single', 'test_py'))
    b.add_argument('--device', default='mps', choices=('cpu', 'mps'))
    b.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('dev-ours-b2')
    c.add_argument('--fsc-images', type=Path, required=True)
    c.add_argument('--annotations', type=Path, required=True)
    c.add_argument('--build', type=Path, required=True)
    c.add_argument('--arm', default='ours_fullgpu_k3', choices=tuple(ARMS))
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    dict(**{'dev-tdcm-pipeline': dev_tdcm_pipeline, 'dev-tdcm-threshold': dev_tdcm_threshold,
            'dev-ours-b2': dev_ours_b2})[args.command](args)


if __name__ == '__main__':
    main()
