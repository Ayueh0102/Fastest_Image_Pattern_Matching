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
  export-geco2-cuda   package images, exemplar boxes and tools for the lab PC's CUDA run
  ingest-geco2-cuda   copy its outputs back into the run dirs as method geco2_cuda

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


HEAVY = ('literature_tdcm_worker', 'literature_geco2_worker', 'literature_opencv_worker', 'fpm_native_pipeline_runner',
         'fpm_color_verify_runner', 'fpm_cli', 'benchmark_', 'ctest', 'cmake', 'clang')


def heavy_processes():
    """Project or round-50 work still running anywhere on the machine (our own background jobs included)."""
    out = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        pid, _, command = line.strip().partition(' ')
        if (pid.isdigit() and int(pid) != os.getpid() and any(h in command for h in HEAVY) and 'literature_compare.py' not in command
                and 'clangd' not in command):
            rows.append(command[:160])
    return rows


class OwnLock:
    def __init__(self, phase, what, expected):
        self.path = LOCKS / 'local_1b4c2052-63a8-4a9f-94f8-77e6ad04fbca' / f'timing-lock-round50-{phase}.txt'
        self.text = f'round 50 (literature comparison) {phase}\nstart_utc {now()}\nwhat {what}\nexpected {expected}; deleted when done\n'

    def __enter__(self):
        while True:  # neither another session's lock nor any heavy work (ours included) may overlap formal timing
            wait_for_locks()
            if not heavy_processes():
                break
            time.sleep(30)
        self.path.write_text(self.text)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def run_worker(cmd, out, own_lock=None, timeout=3600, env=None):
    out.parent.mkdir(parents=True, exist_ok=True)  # workers write --output themselves
    waited = wait_for_locks(own_lock)
    host = host_state()
    t = time.perf_counter()
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, env=env)
        code, stdout, stderr = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, stdout, stderr = 'timeout', e.stdout or '', (e.stderr or '') if isinstance(e.stderr, str) else ''
    wall = (time.perf_counter() - t) * 1e3
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
    if 'compacted' in d:  # literature_archive.py compact: first-call matches (equal to every run, exact_repeat) and times
        return dict(detections=dets(d['first']['matches']), first_matches=d['first']['matches'],
                    time_ms=statistics.median(r['total_ms'] for r in runs), times_ms=[r['total_ms'] for r in runs],
                    first_ms=d['first']['total_ms'], construct_ms=d['construct_ms'], exact_repeat=d['exact_repeat'],
                    peak_rss_bytes=(d.get('final_memory') or {}).get('peak_rss_bytes'), full_gpu_bytes=None,
                    frame_gpu_bytes=None, pose_gpu_bytes=None)
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


# ---------------------------------------------------------------- measured runs

SCENE_METHODS = ('ours_core', 'ours_hybrid_k0', 'ours_hybrid_k3', 'ours_fullgpu_k0', 'ours_fullgpu_k3', 'ours_small_shallow',
                 'opencv', 'tdcm_adapted_cpu', 'tdcm_adapted_mps', 'geco2_mps', 'geco2_cpu')
ONE_BLOCK = {'opencv', 'tdcm_adapted_mps', 'geco2_mps', 'geco2_cpu'}
H4_SCENES = {'many128_pose', 'many512_pose', 'many8k_512'}
# (warm-up calls, measured calls); TDCM-adapted refines every peak, so outside H4 one call is the budget.
CALLS = dict(tdcm_h4=(1, 3), tdcm_other=(0, 1), geco2_mps=(1, 1), geco2_cpu=(0, 1))
OPENCV_FULL_MAX_PIXELS = 2.5e6


def applicable(case, method):
    if method == 'ours_core':
        if case['positive']:
            tw, th = case['template_size']
            return all(abs(t['width'] - tw) < 1e-6 and abs(t['height'] - th) < 1e-6 for t in case['truth'])
        return case['search']['scales'] == [1.0]
    if method == 'ours_small_shallow':
        return case.get('family') == 'tiny'
    if method == 'geco2_cpu':
        return case['positive'] and case['group'] == 'A'
    if method.startswith('geco2'):
        return case['positive']
    if method == 'tdcm_adapted_mps':
        return case['name'] in H4_SCENES
    return True


def blocks_for(case, method, blocks):
    if method in ONE_BLOCK or (method == 'tdcm_adapted_cpu' and case['name'] not in H4_SCENES):
        return 1
    return blocks


def manifest_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r['case'], r['method'], r['block']))
    return done


def execute(case, method, block, out_dir, cfg, own_lock):
    stem = out_dir / method / f"{case['name']}-b{block}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = stem.with_suffix('.json')
    extra, parsed = {}, None
    try:
        if method == 'ours_core':
            w = run_worker(ours_core_cmd(case, cfg.build), out, own_lock)
            if w['code'] == 0:
                out.write_text(w['stdout'])
                parsed = parse_core(w['stdout'], case)
        elif method in ARMS or method == 'ours_small_shallow':
            arm = 'ours_hybrid_k0' if method == 'ours_small_shallow' else method
            job, reused = runner_job(case, arm, cfg.repeats, cfg.warmup_seconds, shallow=method == 'ours_small_shallow')
            job_path = stem.with_suffix('.job.json')
            job_path.write_text(json.dumps(job, indent=1) + '\n')
            w = run_worker([cfg.build / 'experiments/fpm_native_pipeline_runner', job_path], out, own_lock)
            extra['job_from_round44'] = reused
            if w['code'] == 0:
                out.write_text(w['stdout'])
                parsed = parse_runner(w['stdout'], case)
        elif method == 'opencv':
            iw, ih = image_size(case['source'])
            max_maps = 6 if iw * ih > OPENCV_FULL_MAX_PIXELS else 0
            w = run_worker(opencv_cmd(case, out, max_maps=max_maps), out, own_lock, timeout=4 * 3600)
            if w['code'] == 0:
                parsed = parse_opencv(out, case)
        elif method.startswith('tdcm_adapted'):
            warm, reps = CALLS['tdcm_h4' if case['name'] in H4_SCENES else 'tdcm_other']
            w = run_worker(tdcm_cmd(case, out, method.split('_')[-1], cfg.tdcm_pipeline, cfg.tdcm_threshold, warmup=warm, repeats=reps),
                           out, own_lock, timeout=6 * 3600)
            if w['code'] == 0:
                parsed = parse_worker(out, case)
        elif method.startswith('geco2'):
            warm, reps = CALLS[method]
            w = run_worker(geco2_cmd(case, out, method.split('_')[-1], warmup=warm, repeats=reps), out, own_lock)
            if w['code'] == 0:
                parsed = parse_worker(out, case)
        else:
            raise ValueError(method)
    except Exception as error:  # a parse failure is a failed worker, kept as such
        w = dict(code='exception', stdout='', wall_ms=None, host=host_state(), lock_wait_s=None, cmd=[], error=repr(error))
        parsed = None
    record = dict(case=case['name'], group=case['group'], family=case.get('family'), method=method, block=block,
                  status='ok' if parsed is not None else 'failed', output=str(out), returncode=w['code'], wall_ms=w['wall_ms'],
                  host=w['host'], lock_wait_s=w['lock_wait_s'], cmd=w['cmd'], **extra)
    if parsed is None:
        err = out.with_suffix('.stderr.txt')
        record['stderr_tail'] = err.read_text()[-1500:] if err.exists() else w.get('error')
    else:
        record.update({k: v for k, v in parsed.items() if k not in ('detections', 'first_matches')}, detections=len(parsed['detections']))
    return record


class RunConfig:
    def __init__(self, args, protocol):
        self.build = args.build
        self.repeats, self.warmup_seconds = 3, 1.0
        self.tdcm_pipeline = protocol['dev_selection']['tdcm_pipeline']['choice']
        self.tdcm_threshold = protocol['dev_selection']['tdcm_adapted_peak_threshold']['choice']


def run_scenes(args, cases, cfg):
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'cases.json').write_text(json.dumps(cases, indent=1) + '\n')
    manifest = out_dir / 'manifest.jsonl'
    done = manifest_done(manifest)
    lock = OwnLock(f'{args.group}-{args.phase}', f'{args.group} x {",".join(args.methods)} x {args.blocks} blocks',
                   'several hours') if args.formal else None
    own = lock.path if lock else None
    if lock:
        lock.__enter__()
    try:
        for block in range(args.blocks):
            order = [(c, m) for c in cases for m in args.methods if applicable(c, m) and block < blocks_for(c, m, args.blocks)]
            if block % 2:
                order.reverse()
            for c, m in order:
                if (c['name'], m, block) in done:
                    continue
                r = execute(c, m, block, out_dir, cfg, own)
                with open(manifest, 'a') as f:
                    f.write(json.dumps(r) + '\n')
                print(block, c['name'], m, r['status'], r.get('time_ms'), r.get('detections'), flush=True)
    finally:
        if lock:
            lock.__exit__()


def b1_scales(setting, variant):
    if variant == 'native':
        return [1.0]
    lo, hi = dict(S1=(34 / 36, 38 / 36), S1_5=(.8, 1.5), S2=(.5, 2.), S2_5=(.4, 2.5))[setting]
    n = math.ceil(math.log(hi / lo) / math.log(1.04)) + 1
    return [lo * (hi / lo) ** (k / (n - 1)) for k in range(n)]


def b1_opencv_scales(setting):
    lo, hi = dict(S1=(34 / 36, 38 / 36), S1_5=(.8, 1.5), S2=(.5, 2.), S2_5=(.4, 2.5))[setting]
    n = math.ceil(math.log(hi / lo) / math.log(1.10)) + 1
    return [lo * (hi / lo) ** (k / (n - 1)) for k in range(n)]


def run_b1(args, cfg, protocol):
    pipelines = [cfg.tdcm_pipeline] + [p for p in ('test_single', 'test_py') if p != cfg.tdcm_pipeline]
    want = lambda key: any(m.startswith(key) for m in args.methods)  # noqa: E731  ('tdcm', 'ours', 'opencv')
    manifest = args.output / 'manifest.jsonl'
    args.output.mkdir(parents=True, exist_ok=True)
    done = manifest_done(manifest)

    def log(r):
        with open(manifest, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(r['case'], r['method'], r['status'], r.get('time_ms'), flush=True)

    for setting in args.settings:
        fixtures = args.fixtures / f'{setting}-eval'
        truth = json.loads((fixtures / 'truth.json').read_text())
        for variant in ('t36', 'native'):
            suffix = '_template36.png' if variant == 't36' else '_template.png'
            items = [dict(id=t['index'], image=str(fixtures / f"{t['index']:04d}_search.png"),
                          template=str(fixtures / f"{t['index']:04d}{suffix}")) for t in truth]
            items_path = args.output / 'items' / f'{setting}-{variant}.json'
            items_path.parent.mkdir(parents=True, exist_ok=True)
            items_path.write_text(json.dumps(items) + '\n')
            for pipeline in (pipelines if variant == 't36' else pipelines[:1]) if want('tdcm') else ():
                key = (f'{setting}-{variant}', f'tdcm_{pipeline}', 0)
                if key in done:
                    continue
                out = args.output / f'tdcm_{pipeline}' / f'{setting}-{variant}.jsonl'
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [EXTERNAL / 'envs/tdcm-py312/bin/python', '-I', TOOLS / 'literature_tdcm_worker.py', '--repo', EXTERNAL / 'repos/tdcm',
                       '--weights', EXTERNAL / 'repos/tdcm/dict/model.pth', '--batch', items_path, '--device', 'cpu',
                       '--pipeline', pipeline, '--threads', 8, '--output', out]
                w = run_worker(cmd, out)
                log(dict(case=key[0], method=key[1], block=0, status='ok' if w['code'] == 0 else 'failed', output=str(out),
                         wall_ms=w['wall_ms'], host=w['host'], returncode=w['code']))
            arms = (['ours_hybrid_k0', 'ours_fullgpu_k0'] + (['ours_core'] if variant == 'native' else [])) if want('ours') else []
            for t in truth:
                image, template = items[t['index']]['image'], items[t['index']]['template']
                tw, th = image_size(template)
                case = dict(name=f"{setting}-{variant}-{t['index']:04d}", group='B1', source=image, template=template,
                            template_size=[tw, th], positive=True, truth=[],
                            search=dict(scales=b1_scales(setting, variant), angle=180, limit=1, score=.5, proposal_score=.3,
                                        overlap=.2, stop_layer=0, subpixel=False))
                for arm in arms:
                    if (case['name'], arm, 0) in done:
                        continue
                    out = args.output / arm / f'{setting}-{variant}' / f"{t['index']:04d}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if arm == 'ours_core':
                        w = run_worker(ours_core_cmd(case, cfg.build), out)
                    else:
                        job, _ = runner_job(case, arm, 1, 0.)
                        out.with_suffix('.job.json').write_text(json.dumps(job) + '\n')
                        w = run_worker([cfg.build / 'experiments/fpm_native_pipeline_runner', out.with_suffix('.job.json')], out)
                    if w['code'] == 0:
                        out.write_text(w['stdout'])
                    with open(manifest, 'a') as f:
                        f.write(json.dumps(dict(case=case['name'], method=arm, block=0, status='ok' if w['code'] == 0 else 'failed',
                                                output=str(out), wall_ms=w['wall_ms'], load1=w['host']['load1'],
                                                returncode=w['code'])) + '\n')
                if want('opencv') and variant == 't36' and t['index'] < 200 and (case['name'], 'opencv', 0) not in done:
                    out = args.output / 'opencv' / f'{setting}-{variant}' / f"{t['index']:04d}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    case_cv = dict(case, search=dict(case['search'], scales=b1_opencv_scales(setting), score=-1.))
                    w = run_worker(opencv_cmd(case_cv, out, angle_step=2.), out)
                    with open(manifest, 'a') as f:
                        f.write(json.dumps(dict(case=case['name'], method='opencv', block=0, status='ok' if w['code'] == 0 else 'failed',
                                                output=str(out), wall_ms=w['wall_ms'], load1=w['host']['load1'],
                                                returncode=w['code'])) + '\n')
            print(setting, variant, 'done', flush=True)


def run_b2(args, cfg, protocol):
    cases = fsc_cases('test', args.fsc_images, args.annotations)
    manifest = args.output / 'manifest.jsonl'
    args.output.mkdir(parents=True, exist_ok=True)
    done = manifest_done(manifest)
    tuned = protocol['dev_selection']['ours_b2_score']['choice']
    arms = [('ours_fullgpu_k3', 'ours_fullgpu_k3', .8), ('ours_hybrid_k3', 'ours_hybrid_k3', .8),
            (f'ours_fullgpu_k3_s{tuned}', 'ours_fullgpu_k3', tuned)]
    items = []
    for c in cases:
        template, crop = fsc_template(c, args.output / 'templates')
        ex = [[min(p[0] for p in q), min(p[1] for p in q), max(p[0] for p in q), max(p[1] for p in q)]
              for q in c['annotation']['box_examples_coordinates'][:3]]
        items.append(dict(id=c['name'], image=c['image'], exemplars=ex, template=str(template), crop=crop))
    (args.output / 'items.json').write_text(json.dumps(items) + '\n')
    for name, arm, sc in arms if 'ours' in args.methods else []:
        for it in items:
            if (it['id'], name, 0) in done:
                continue
            job, _ = b2_job(dict(image=it['image']), it['template'], arm, sc)
            out = args.output / name / f"{it['id'].split('.')[0]}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix('.job.json').write_text(json.dumps(job) + '\n')
            w = run_worker([cfg.build / 'experiments/fpm_native_pipeline_runner', out.with_suffix('.job.json')], out)
            if w['code'] == 0:
                out.write_text(w['stdout'])
            with open(manifest, 'a') as f:
                f.write(json.dumps(dict(case=it['id'], method=name, block=0, status='ok' if w['code'] == 0 else 'failed',
                                        output=str(out), wall_ms=w['wall_ms'], load1=w['host']['load1'], returncode=w['code'],
                                        stderr_tail=None if w['code'] == 0 else out.with_suffix('.stderr.txt').read_text()[-600:])) + '\n')
        print(name, 'done', flush=True)
    for device in [m.split('_')[-1] for m in args.methods if m.startswith('geco2')]:
        key = ('all', f'geco2_{device}', 0)
        if key in done:
            continue
        out = args.output / f'geco2_{device}.jsonl'
        batch = args.output / 'items.json'
        if device == 'cpu':  # descriptive: the first 100 test images in split order
            batch = args.output / 'items-cpu-first100.json'
            batch.write_text(json.dumps(items[:100]) + '\n')
        cmd = [EXTERNAL / 'envs/geco2-py312/bin/python', '-I', TOOLS / 'literature_geco2_worker.py', '--repo', EXTERNAL / 'repos/geco2',
               '--weights', EXTERNAL / 'data/geco2-weights-hf/GECO2FSCD.pth', '--torch-home', EXTERNAL / 'torch-home',
               '--batch', batch, '--threshold', .11, '--device', device, '--threads', 8, '--output', out]
        w = run_worker(cmd, out, timeout=3 * 3600)  # protocol geco2_mps_stop: MPS must finish within 3 hours
        with open(manifest, 'a') as f:
            f.write(json.dumps(dict(case='all', method=f'geco2_{device}', block=0, status='ok' if w['code'] == 0 else 'failed',
                                    output=str(out), wall_ms=w['wall_ms'], host=w['host'], returncode=w['code'])) + '\n')
        print('geco2', device, w['code'], flush=True)


# ---------------------------------------------------------------- GeCo2 on the lab PC's CUDA

CUDA_TOOLS = ('literature_geco2_worker.py', 'literature_geco2_cuda_setup.ps1', 'literature_geco2_cuda_precheck.py',
              'literature_geco2_cuda_run.py')


def export_geco2_cuda(args):
    """Package for literature_geco2_cuda_run.py: images, exemplar boxes and tools only (truth stays here)."""
    pkg = args.output
    if pkg.exists():
        raise SystemExit(f'{pkg} exists; export into a new directory')
    jobs = {}
    for group, run_dir in (('A', args.a), ('C', args.c)):
        jobs[group] = []
        for case in json.loads((run_dir / 'cases.json').read_text()):
            if not applicable(case, 'geco2_cuda'):
                continue
            image = Path('images') / group / (case['name'] + Path(case['source']).suffix)
            (pkg / image).parent.mkdir(parents=True, exist_ok=True)
            (pkg / image).write_bytes(Path(case['source']).read_bytes())
            jobs[group].append(dict(case=case['name'], image=str(image), exemplars=exemplars_for(case)))
    jobs['B2'] = []
    for it in json.loads((args.b2 / 'items.json').read_text()):
        image = Path('images/B2') / it['id']
        (pkg / image).parent.mkdir(parents=True, exist_ok=True)
        (pkg / image).write_bytes(Path(it['image']).read_bytes())
        jobs['B2'].append(dict(id=it['id'], image=str(image), exemplars=it['exemplars']))
    (pkg / 'tools').mkdir()
    for name in CUDA_TOOLS:
        (pkg / 'tools' / name).write_bytes((TOOLS / name).read_bytes())
    (pkg / 'jobs.json').write_text(json.dumps(jobs, indent=1) + '\n')
    files = sorted(p for p in pkg.rglob('*') if p.is_file())
    (pkg / 'package.json').write_text(json.dumps(dict(created=now(), jobs={k: len(v) for k, v in jobs.items()},
                                                      files={str(p.relative_to(pkg)): sha(p) for p in files}), indent=1) + '\n')
    print({k: len(v) for k, v in jobs.items()}, len(files), 'files')


def ingest_geco2_cuda(args):
    """Copy the lab PC's outputs into the run directories and append manifest rows (method geco2_cuda)."""
    rows = read_manifest(args.results / 'manifest.jsonl')
    for group, run_dir in (('A', args.a), ('C', args.c), ('B2', args.b2)):
        if run_dir is None:
            continue
        manifest = run_dir / 'manifest.jsonl'
        done = manifest_done(manifest)
        cases = {c['name']: c for c in json.loads((run_dir / 'cases.json').read_text())} if group != 'B2' else {}
        for r in (r for r in rows if r['group'] == group):
            key = (r['case'], 'geco2_cuda', 0)
            if key in done:
                continue
            src = args.results / r['output']
            dst = run_dir / ('geco2_cuda.jsonl' if group == 'B2' else f"geco2_cuda/{r['case']}-b0.json")
            dst.parent.mkdir(parents=True, exist_ok=True)
            for a, b in ((src, dst), (src.with_suffix('.stderr.txt'), dst.with_suffix('.stderr.txt'))):
                if a.exists():
                    b.write_bytes(a.read_bytes())
            record = dict(case=r['case'], method='geco2_cuda', block=0, status=r['status'], output=str(dst), returncode=r['returncode'],
                          wall_ms=r['wall_ms'], host=r['host'], lock_wait_s=None, cmd=r['cmd'], attempt=r['attempt'],
                          remote_output=r['output'])
            if group != 'B2':
                case = cases[r['case']]
                record.update(group=group, family=case.get('family'))
                if r['status'] == 'ok':
                    parsed = parse_worker(dst, case)
                    record.update({k: v for k, v in parsed.items() if k != 'detections'}, detections=len(parsed['detections']))
                else:
                    record['stderr_tail'] = r.get('stderr_tail')
            with open(manifest, 'a') as f:
                f.write(json.dumps(record) + '\n')
            print(group, r['case'], r['status'], flush=True)


def check_methods(group, methods):
    # Attempt 1 of the formal A/C runs passed every method as one string (shell quoting) and failed 435 workers.
    known = SCENE_METHODS if group in ('A', 'C') else ('tdcm', 'ours', 'opencv', 'geco2_mps', 'geco2_cpu')
    unknown = [m for m in methods if m not in known]
    if unknown:
        raise SystemExit(f'unknown method(s) for group {group}: {unknown}')


def run(args):
    check_methods(args.group, args.methods)
    protocol = json.loads(args.protocol.read_text())
    cfg = RunConfig(args, protocol)
    if args.group == 'A':
        run_scenes(args, cases_a(args.output), cfg)
    elif args.group == 'C':
        run_scenes(args, cases_c(args.scenes), cfg)
    elif args.group == 'B1':
        run_b1(args, cfg, protocol)
    else:
        run_b2(args, cfg, protocol)


# ---------------------------------------------------------------- summaries

def read_manifest(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


REMAP = []  # (old prefix, new prefix) for replaying archived runs at another location


def remapped(path):
    for old, new in REMAP:
        if str(path).startswith(old):
            return Path(new + str(path)[len(old):])
    return Path(path)


def detections_of(entry, case):
    out = remapped(entry['output'])
    m = entry['method']
    if m == 'ours_core':
        return parse_core(out.read_text(), case)['detections']
    if m in ARMS or m == 'ours_small_shallow':
        return parse_runner(out.read_text(), case)['detections']
    if m == 'opencv':
        return parse_opencv(out, case)['detections']
    return parse_worker(out, case)['detections']


def memory_of(entry):
    if 'peak_rss_bytes' in entry:
        return dict(peak_rss_bytes=entry['peak_rss_bytes'], named_metal_bytes=entry.get('full_gpu_bytes'),
                    frame_gpu_bytes=entry.get('frame_gpu_bytes'), pose_gpu_bytes=entry.get('pose_gpu_bytes'))
    return entry.get('memory')


def summarize_scenes(run_dir):
    """Per (case, method): accuracy from the lowest ok block, time = median over ok blocks, failures kept."""
    cases = {c['name']: c for c in json.loads((run_dir / 'cases.json').read_text())}
    by = {}
    for e in read_manifest(run_dir / 'manifest.jsonl'):
        by.setdefault((e['case'], e['method']), []).append(e)
    table, failures, pairs = [], [], {}
    for (name, method), es in sorted(by.items()):
        case = cases[name]
        ok = sorted((e for e in es if e['status'] == 'ok'), key=lambda e: e['block'])
        failures += [dict(case=name, method=method, block=e['block'], returncode=e.get('returncode'), stderr_tail=e.get('stderr_tail'))
                     for e in es if e['status'] != 'ok']
        row = dict(case=name, group=case['group'], family=case.get('family'), positive=case['positive'], method=method,
                   blocks=len(es), ok_blocks=len(ok))
        if ok:
            dets = detections_of(ok[0], case)
            s = case_score(case, dets)
            pairs[(name, method)] = s.pop('association_pairs', [])
            row['score'] = s
            row['blocks_identical'] = all(detections_of(e, case) == dets for e in ok[1:])
            times = [e['time_ms'] for e in ok if e.get('time_ms') is not None]
            row['time_ms'] = statistics.median(times) if times else None
            row['time_ms_blocks'] = times
            constructs = [e['construct_ms'] for e in ok if e.get('construct_ms') is not None]
            row['construct_ms'] = statistics.median(constructs) if constructs else None
            row['memory'] = memory_of(ok[0])
            loads = [e['host']['load1'] for e in ok if e['host'].get('load1') is not None]  # the Windows lab PC has none
            row['load1_median'] = statistics.median(loads) if loads else None
            for key in ('complete', 'extrapolated_ms', 'maps', 'planned_maps', 'job_from_round44', 'exact_repeat'):
                if key in ok[0]:
                    row[key] = ok[0][key]
        table.append(row)
    return cases, table, failures, pairs


def identity_round44(run_dir):
    out = []
    for e in read_manifest(run_dir / 'manifest.jsonl'):
        if e['status'] != 'ok' or e['method'] not in STAGE44_ARM or not e.get('job_from_round44'):
            continue
        ref = STAGE44 / 'results' / f"{e['case']}-{STAGE44_ARM[e['method']]}-b0.json"
        if not ref.exists():
            continue
        ours = json.loads(remapped(e['output']).read_text())['first']['matches']
        theirs = json.loads(ref.read_text())['first']['matches']
        out.append(dict(case=e['case'], method=e['method'], block=e['block'], equal=ours == theirs, ours=len(ours), round44=len(theirs)))
    return out


def row_of(table, case, method):
    return next((r for r in table if r['case'] == case and r['method'] == method), None)


def strict_recall(row):
    s = (row or {}).get('score') or {}
    return s['strict_tp'] / s['gt'] if s.get('gt') and s.get('strict_tp') is not None else None


def pooled(table, names, method, key):
    rows = [row_of(table, n, method) for n in names]
    rows = [r for r in rows if r and r.get('score')]
    gt = sum(r['score']['gt'] for r in rows)
    if key == 'boxes':
        tp = sum(r['score']['boxes']['tp'] for r in rows)
    else:
        tp = sum(r['score'].get(key) or 0 for r in rows)
    return dict(cases=len(rows), of=len(names), gt=gt, tp=tp, recall=tp / gt if gt else None)


def pooled_errors(pairs, names, method):
    rows = [p for n in names for p in pairs.get((n, method), [])]
    return dict(pairs=len(rows), position_px=score.summary([r['position_px'] for r in rows]),
                angle_deg=score.summary([r['angle_deg'] for r in rows]))


def hypotheses(a, c, b2, geco2_method):
    out = {}
    if a:
        cases, table, _, pairs = a
        positives = sorted(n for n, x in cases.items() if x['positive'])
        rows = []
        for n in positives:
            ours, tdcm = strict_recall(row_of(table, n, 'ours_fullgpu_k3')), strict_recall(row_of(table, n, 'tdcm_adapted_cpu'))
            cv = row_of(table, n, 'opencv')
            cvr = strict_recall(cv) if cv and cv.get('complete') else None
            rows.append(dict(case=n, ours=ours, tdcm_adapted=tdcm, opencv=cvr,
                             ok=ours is not None and tdcm is not None and ours >= tdcm and (cvr is None or ours >= cvr)))
        e_ours, e_tdcm = pooled_errors(pairs, positives, 'ours_fullgpu_k3'), pooled_errors(pairs, positives, 'tdcm_adapted_cpu')
        b_ours, b_geco = pooled(table, positives, 'ours_fullgpu_k3', 'boxes'), pooled(table, positives, geco2_method, 'boxes')
        med = lambda e, k: e[k].get('median')  # noqa: E731
        b_ok = (e_ours['pairs'] and e_tdcm['pairs'] and med(e_ours, 'position_px') < med(e_tdcm, 'position_px')
                and med(e_ours, 'angle_deg') < med(e_tdcm, 'angle_deg'))
        c_ok = b_geco['recall'] is not None and b_ours['recall'] is not None and b_ours['recall'] >= b_geco['recall']
        out['H1'] = dict(a=rows, a_pass=all(r['ok'] for r in rows), b=dict(ours=e_ours, tdcm_adapted=e_tdcm, pass_=bool(b_ok)),
                         c=dict(ours=b_ours, geco2=b_geco, geco2_method=geco2_method, pass_=c_ok if b_geco['cases'] else None))
        # Any failed part decides H1; only when (a) and (b) pass does it wait for GeCo2's (c).
        if not out['H1']['a_pass'] or not b_ok:
            out['H1']['pass'] = False
        else:
            out['H1']['pass'] = c_ok if b_geco['cases'] else None
        negatives = sorted(n for n, x in cases.items() if not x['positive'])
        ours_arms = ('ours_core', 'ours_hybrid_k0', 'ours_hybrid_k3', 'ours_fullgpu_k0', 'ours_fullgpu_k3')
        h5 = []
        for n in negatives:
            for m in ours_arms + ('tdcm_adapted_cpu', 'opencv'):
                r = row_of(table, n, m)
                if r and r.get('score'):
                    h5.append(dict(case=n, method=m, false_positives=r['score']['false_positives']))
        out['H5'] = dict(rows=h5, pass_=all(x['false_positives'] == 0 for x in h5 if x['method'] in ours_arms))
    if c:
        cases, table, _, pairs = c
        tiny = sorted(n for n, x in cases.items() if x['positive'] and x.get('family') == 'tiny')
        t, o = pooled(table, tiny, 'tdcm_adapted_cpu', 'strict_tp'), pooled(table, tiny, 'ours_fullgpu_k3', 'strict_tp')
        out['H3'] = dict(tdcm_adapted=t, ours=o, association=dict(tdcm_adapted=pooled(table, tiny, 'tdcm_adapted_cpu', 'association_tp'),
                                                                   ours=pooled(table, tiny, 'ours_fullgpu_k3', 'association_tp')),
                         pass_=t['recall'] is not None and o['recall'] is not None and t['recall'] >= o['recall'])
    rows = []
    for tab, names in ((a, ('many128_pose', 'many512_pose')), (c, ('many8k_512',))):
        if not tab:
            continue
        for n in names:
            ours, tdcm = row_of(tab[1], n, 'ours_fullgpu_k3'), row_of(tab[1], n, 'tdcm_adapted_cpu')
            rows.append(dict(case=n, ours_ms=(ours or {}).get('time_ms'), tdcm_adapted_cpu_ms=(tdcm or {}).get('time_ms'),
                             tdcm_adapted_mps_ms=(row_of(tab[1], n, 'tdcm_adapted_mps') or {}).get('time_ms'),
                             ok=bool(ours and tdcm and ours.get('time_ms') is not None and tdcm.get('time_ms') is not None
                                     and ours['time_ms'] < tdcm['time_ms'])))
    if rows:
        out['H4'] = dict(rows=rows, pass_=len(rows) == 3 and all(r['ok'] for r in rows))
    if b2:
        g, o = b2['methods'].get(b2['geco2_method']), b2['methods'].get('ours_fullgpu_k3')
        if g and o:
            if b2['boxes_available']:
                out['H2'] = dict(metric='AP50', geco2=g['boxes']['ap50'], ours=o['boxes']['ap50'], pass_=g['boxes']['ap50'] > o['boxes']['ap50'])
            else:
                out['H2'] = dict(metric='count MAE vs FSC points', geco2=g['count_points']['mae'], ours=o['count_points']['mae'],
                                 pass_=g['count_points']['mae'] < o['count_points']['mae'])
    return out


def summarize_b1(run_dir, fixtures, settings):
    entries = read_manifest(run_dir / 'manifest.jsonl')
    status = {}
    for e in entries:
        status[(e['case'], e['method'])] = e
    out = {}
    for setting in settings:
        truth = json.loads((fixtures / f'{setting}-eval' / 'truth.json').read_text())
        gts = {t['index']: dict(kind='pose', cx=t['cx'], cy=t['cy'], angle_deg=t['angle'], width=36 * t['scale_x'],
                                height=36 * t['scale_y']) for t in truth}
        for variant in ('t36', 'native'):
            methods = {}
            for path in sorted(run_dir.glob(f'tdcm_*/{setting}-{variant}.jsonl')):
                _, results, footer = read_batch(path)
                methods[path.parent.name] = ({k: v['detections'] for k, v in results.items()},
                                             [v['timing_ms']['call_ms'] for v in results.values()], footer)
            for arm in ('ours_hybrid_k0', 'ours_fullgpu_k0', 'ours_core', 'opencv'):
                d = run_dir / arm / f'{setting}-{variant}'
                if not d.exists():
                    continue
                dets, times = {}, []
                for t in truth:
                    f = d / f"{t['index']:04d}.json"
                    if not f.exists() or f.stat().st_size == 0:
                        continue
                    case = dict(template_size=list(image_size(fixtures / f'{setting}-eval' / (f"{t['index']:04d}_template36.png" if variant == 't36' else f"{t['index']:04d}_template.png"))))
                    if arm == 'ours_core':
                        p = parse_core(f.read_text(), case)
                    elif arm == 'opencv':
                        p = parse_opencv(f, case)
                    else:
                        p = parse_runner(f.read_text(), case)
                    dets[t['index']], times = p['detections'][:1], times + [p['time_ms']]
                methods[arm] = (dets, times, None)
            rows = {}
            for m, (dets, times, footer) in methods.items():
                scored = [i for i in gts if i in dets] if m == 'opencv' else list(gts)
                strict = assoc = 0
                for i in scored:
                    r = score.score_pose_image([gts[i]], dets.get(i, []))
                    strict += r['strict_tp']
                    assoc += r['association_tp']
                paper = b1_paper_metrics([t for t in truth if t['index'] in scored],
                                         {i: [dict(cx=x['cx'], cy=x['cy'], width=x['width'] if 'width' in x else x['scale_x'] * x['scale_reference'][0],
                                                   height=x['height'] if 'height' in x else x['scale_y'] * x['scale_reference'][1],
                                                   angle_deg=x['angle_deg'], score=x.get('score', 1.)) for x in dets.get(i, [])]
                                          for i in scored})
                rows[m] = dict(samples=len(scored), outputs=sum(1 for i in scored if dets.get(i)), strict_tp=strict, association_tp=assoc,
                               strict_rate=strict / len(scored) if scored else None, paper=paper,
                               time_ms=score.summary(times), memory=(footer or {}).get('memory'))
            out[f'{setting}-{variant}'] = rows
    return out


def geco2_agreement(run_dir):
    """Per-image agreement between the GeCo2 devices on the images both ran (disclosed, not scored)."""
    runs = {f.stem: read_batch(f)[1] for f in sorted(run_dir.glob('geco2_*.jsonl'))}
    out = {}
    for a in runs:
        for b in runs:
            if a >= b:
                continue
            names = sorted(set(runs[a]) & set(runs[b]))
            diffs = [len(runs[a][n]['detections']) - len(runs[b][n]['detections']) for n in names]
            out[f'{a} vs {b}'] = dict(images=len(names), identical=sum(runs[a][n]['detections'] == runs[b][n]['detections'] for n in names),
                                      same_count=sum(d == 0 for d in diffs), count_difference=score.summary([abs(d) for d in diffs]))
    return out


def summarize_b2(run_dir, annotations, boxes_path):
    items = {i['id']: i for i in json.loads((run_dir / 'items.json').read_text())}
    a = json.loads((annotations / 'annotation_FSC147_384.json').read_text())
    boxes = {}
    if boxes_path and boxes_path.exists():
        inst = json.loads(boxes_path.read_text())
        names = {im['id']: im['file_name'] for im in inst['images']}
        for x in inst['annotations']:
            bx, by, bw, bh = x['bbox']
            boxes.setdefault(names[x['image_id']], []).append((bx, by, bx + bw, by + bh))
    methods = {}
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('ours')):
        dets, times, failed = {}, [], 0
        for name, it in items.items():
            f = d / f"{name.split('.')[0]}.json"
            if not f.exists() or f.stat().st_size == 0:
                failed += 1
                dets[name] = []
                continue
            tw, th = image_size(it['template'])
            p = parse_runner(f.read_text(), dict(template_size=[tw, th]))
            dets[name], times = p['detections'], times + [p['time_ms']]
        methods[d.name] = (dets, times, failed, None)
    subsets = {}
    for f in sorted(run_dir.glob('geco2_*.jsonl')):
        _, results, footer = read_batch(f)
        methods[f.stem] = ({k: v['detections'] for k, v in results.items()}, [v['timing_ms']['call_ms'] for v in results.values()],
                           0, footer)
        subsets[f.stem] = [n for n in items if n in results]  # the CPU batch covers the first 100 test images only

    def scored(dets, names):
        r = dict(images=len(names), count_points=score.count_errors([(len(a[n]['points']), len(dets.get(n, []))) for n in names]))
        if boxes:
            r['boxes'] = score.score_box_images([(boxes.get(n, []), dets.get(n, [])) for n in names])
            r['count_boxes'] = score.count_errors([(len(boxes.get(n, [])), len(dets.get(n, []))) for n in names])
        return r
    out = {}
    for m, (dets, times, failed, footer) in methods.items():
        names = subsets.get(m, list(items))
        out[m] = dict(scored(dets, names), failed=failed, time_ms=score.summary(times), memory=(footer or {}).get('memory'))
        if len(names) < len(items):
            out[m]['subset'] = f'first {len(names)} test images'
            out[m]['same_subset'] = {k: scored(methods[k][0], names) for k in methods if k != m}
    return out


def summarize(args):
    for pair in args.remap or []:
        REMAP.append(tuple(pair.split('=', 1)))
    protocol = json.loads(args.protocol.read_text())
    summary = dict(protocol_sha256=sha(args.protocol), created=now())
    a = summarize_scenes(args.a) if args.a else None
    c = summarize_scenes(args.c) if args.c else None
    geco2_method = args.geco2_accuracy
    if a:
        summary['A'] = dict(table=a[1], failures=a[2], identity_round44=identity_round44(args.a))
    if c:
        summary['C'] = dict(table=c[1], failures=c[2])
    b2 = None
    if args.b2:
        methods = summarize_b2(args.b2, args.annotations, args.boxes)
        summary['B2'] = methods
        summary['B2_geco2_agreement'] = geco2_agreement(args.b2)
        b2 = dict(methods=methods, geco2_method=geco2_method, boxes_available=bool(args.boxes and args.boxes.exists()))
    if args.b1:
        summary['B1'] = summarize_b1(args.b1, args.fixtures, args.settings)
    summary['hypotheses'] = hypotheses(a, c, b2, geco2_method)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('pass', 'pass_', 'a_pass')} for k, v in summary['hypotheses'].items()}, indent=1))


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
    r = sub.add_parser('run')
    r.add_argument('--group', required=True, choices=('A', 'C', 'B1', 'B2'))
    r.add_argument('--protocol', type=Path, required=True)
    r.add_argument('--build', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--methods', nargs='+', default=list(SCENE_METHODS))
    r.add_argument('--blocks', type=int, default=3)
    r.add_argument('--phase', default='formal')
    r.add_argument('--formal', action='store_true', help='hold this round\'s timing lock for the whole run')
    r.add_argument('--scenes', type=Path, help='group C scenes directory')
    r.add_argument('--fixtures', type=Path, help='B1 fixtures directory')
    r.add_argument('--settings', nargs='+', default=['S1', 'S1_5', 'S2', 'S2_5'])
    r.add_argument('--fsc-images', type=Path)
    r.add_argument('--annotations', type=Path)
    m = sub.add_parser('summarize')
    m.add_argument('--protocol', type=Path, required=True)
    m.add_argument('--a', type=Path)
    m.add_argument('--c', type=Path)
    m.add_argument('--b1', type=Path)
    m.add_argument('--b2', type=Path)
    m.add_argument('--fixtures', type=Path)
    m.add_argument('--settings', nargs='+', default=['S1', 'S1_5', 'S2', 'S2_5'])
    m.add_argument('--annotations', type=Path)
    m.add_argument('--boxes', type=Path, help='FSCD-147 instances_test.json')
    m.add_argument('--geco2-accuracy', default='geco2_cuda', help='method name whose GeCo2 outputs carry accuracy')
    m.add_argument('--remap', nargs='*', help='OLD=NEW output-path prefixes for replaying archived runs')
    m.add_argument('--output', type=Path, required=True)
    x = sub.add_parser('export-geco2-cuda')
    x.add_argument('--a', type=Path, required=True)
    x.add_argument('--c', type=Path, required=True)
    x.add_argument('--b2', type=Path, required=True)
    x.add_argument('--output', type=Path, required=True)
    g = sub.add_parser('ingest-geco2-cuda')
    g.add_argument('--results', type=Path, required=True, help='literature_geco2_cuda_run.py --output, copied back')
    g.add_argument('--a', type=Path)
    g.add_argument('--c', type=Path)
    g.add_argument('--b2', type=Path)
    args = p.parse_args()
    dict(**{'dev-tdcm-pipeline': dev_tdcm_pipeline, 'dev-tdcm-threshold': dev_tdcm_threshold,
            'dev-ours-b2': dev_ours_b2, 'run': run, 'summarize': summarize,
            'export-geco2-cuda': export_geco2_cuda, 'ingest-geco2-cuda': ingest_geco2_cuda})[args.command](args)


if __name__ == '__main__':
    main()
