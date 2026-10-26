"""Round 50 phase 1: smoke-run every method of the literature comparison on two image pairs.

Feasibility only, never timing evidence: no warm-up protocol, other sessions may
share the machine, and each worker is one fresh process. Run with python-port/venv;
third-party methods run in their own environments under the external root:

  <external>/repos/{tdcm,geco2}           pinned clones
  <external>/envs/{tdcm-py312,geco2-py312} isolated interpreters (always python -I)
  <external>/data/...                     downloads, one directory each
  <external>/torch-home/hub/checkpoints   SAM2 checkpoint link for GeCo2

Pairs: ``tdcm_demo`` (TDCM's own res/image.png + res/template.png with the pose
the authors typed into test.py) and ``many128`` (round-37 4096x4096 scene, 128
copies of one template, independent GT). The methods get the same image and the
same template file; GeCo2 takes exemplar boxes inside the image instead, here the
enclosing boxes of the first GT instances (FSCD protocol uses three).

Every worker refuses to start while a timing lock of another session exists.
Truth stays in this parent: workers get images and templates only, and the
literature_score module scores what they wrote.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import literature_score as score  # noqa: E402

TOOLS = Path(__file__).resolve().parent
SAM2_SHA256 = 'd0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071'
STAGE37 = 'stage37-full-gpu/formal/inputs'
# CURRENT_BEST research settings (round-44 jobs); only inputs, scales and budgets change per pair.
RUNNER_BASE = dict(angle=180, overlap=0.2, stop_layer=1, subpixel=False, backend=1, min_area=256, ncc_mode=2,
                   overlap_enabled=False, refine=True, invert_source=False, repeats=1, warmup_seconds=0.0,
                   coarse_peak_mode=1, fine_ncc_mode=2, fine_fused=True, fine_angle_reuse=True,
                   pose_resident_mode=0, pose_input_mode=1, pose_storage_mode=1, refinement_backend=1,
                   full_gpu_profiling=False, candidate_dedup_radius=0.5, candidate_dedup_angle_steps=3.0)
RUNNER_ARMS = {
    'ours_hybrid_k0': dict(full_gpu=False, full_gpu_peak_mode=0, full_gpu_fine_mode=0, candidate_dedup_top_k=0),
    'ours_hybrid_k3': dict(full_gpu=False, full_gpu_peak_mode=0, full_gpu_fine_mode=0, candidate_dedup_top_k=3),
    'ours_fullgpu_k0': dict(full_gpu=True, full_gpu_peak_mode=2, full_gpu_fine_mode=4, full_gpu_peak_batch=8,
                            candidate_dedup_top_k=0),
    'ours_fullgpu_k3': dict(full_gpu=True, full_gpu_peak_mode=2, full_gpu_fine_mode=4, full_gpu_peak_batch=8,
                            candidate_dedup_top_k=3),
}
METHODS = ['ours_core', *RUNNER_ARMS, 'opencv', 'tdcm_cpu', 'tdcm_mps', 'geco2_cpu', 'geco2_mps']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def locks(lock_root):
    return sorted(str(p) for p in Path(lock_root).glob('*/timing-lock-*.txt'))


def run(cmd, out_path, lock_root, timeout):
    held = locks(lock_root)
    if held:
        raise SystemExit('timing lock present, refusing to start: ' + ', '.join(held))
    t = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    wall = (time.perf_counter() - t) * 1e3
    out_path.with_suffix('.stderr.txt').write_text(r.stderr)
    out_path.with_suffix('.cmd.json').write_text(json.dumps([str(c) for c in cmd], indent=1) + '\n')
    return r, wall


def pairs(external, build_root):
    tdcm = external / 'repos/tdcm/res'
    # test.py: true_param = [45, 95, 0.89, 0.90, -50] (centre, 36-relative scales, angle).
    tdcm_gt = [dict(kind='pose', cx=45., cy=95., angle_deg=-50., scale_x=.89, scale_y=.90, scale_reference=[36, 36])]
    meta = json.loads((build_root / STAGE37 / 'many128_pose.json').read_text())
    th, tw = meta['template_shape']
    many_gt = [dict(kind='pose', cx=t['pos_x'], cy=t['pos_y'], angle_deg=t['angle'], width=tw * t['scale'],
                    height=th * t['scale'], corners=t['corners']) for t in meta['truth']]
    box = lambda g: [round(v, 2) for v in score.normalise(g)['box']]  # noqa: E731
    return {
        'tdcm_demo': dict(image=tdcm / 'image.png', template=tdcm / 'template.png', gt=tdcm_gt, gt_source='tdcm test.py',
                          exemplars=[box(tdcm_gt[0])], template_size=(31, 33), limit=1, score=.5,
                          scales=[1.0], runner=dict(limit=4, proposal_score=.3, score=.5)),
        'many128': dict(image=build_root / STAGE37 / 'many128_pose.png', template=build_root / STAGE37 / 'many_template.png',
                        gt=many_gt, gt_source='many_object_scenes.py truth', exemplars=[box(g) for g in many_gt[:3]],
                        template_size=(tw, th), limit=128, score=.8, scales=[.75, 1., 1.25],
                        runner=dict(limit=384, proposal_score=.5, score=.8)),
    }


def ours_core(build, pair, out, lock_root):
    cmd = [build / 'fpm_cli', '--src', pair['image'], '--tmpl', pair['template'], '--target-num', pair['limit'],
           '--score', pair['score'], '--angle', 180, '--overlap', .2, '--subpixel']
    r, wall = run(cmd, out, lock_root, 1800)
    if r.returncode:
        return dict(status='failed', returncode=r.returncode, stderr=r.stderr[-2000:], wall_ms=wall)
    d = json.loads(r.stdout)
    tw, th = pair['template_size']
    dets = [dict(kind='pose', cx=m['pos_x'], cy=m['pos_y'], angle_deg=m['angle'], width=tw, height=th,
                 corners=m['corners'], score=m['score']) for m in d['matches']]
    return dict(status='ok', device='cpu', backend=d['backend'], wall_ms=wall,
                timing_ms=dict(learn=d['learn_ms'], match=d['match_ms']), detections=dets)


def ours_runner(build, arm, pair, out, lock_root):
    job = dict(RUNNER_BASE, **RUNNER_ARMS[arm], **pair['runner'], source=str(pair['image']),
               template=str(pair['template']), scales=pair['scales'])
    job_path = out.with_suffix('.job.json')
    job_path.write_text(json.dumps(job, indent=1) + '\n')
    r, wall = run([build / 'experiments/fpm_native_pipeline_runner', job_path], out, lock_root, 1800)
    if r.returncode:
        return dict(status='failed', returncode=r.returncode, stderr=r.stderr[-2000:], wall_ms=wall)
    d = json.loads(r.stdout)
    call = d['runs'][0]
    tw, th = pair['template_size']
    dets = [dict(kind='pose', cx=m['pos_x'], cy=m['pos_y'], angle_deg=m['angle'], width=tw * m['scale'],
                 height=th * m['scale'], corners=m['corners'], score=m['score']) for m in call['matches']]
    return dict(status='ok', device='metal' if job['full_gpu'] else 'cpu+metal', wall_ms=wall,
                timing_ms=dict(construct=d['construct_ms'], first_call=d['first']['total_ms'], measured_call=call['total_ms']),
                peak_rss_bytes=d['final_memory']['peak_rss_bytes'], full_gpu_bytes=call['full_gpu']['gpu_bytes'],
                exact_repeat=d['exact_repeat'], detections=dets)


def worker_result(r, wall, out):
    if r.returncode:
        return dict(status='failed', returncode=r.returncode, stderr=r.stderr[-3000:], wall_ms=wall)
    d = json.loads(out.read_text())
    return dict(status='ok', wall_ms=wall, **d)


def opencv(pair, out, lock_root, threads, smoke_maps):
    py = Path(sys.executable)
    cmd = [py, '-I', TOOLS / 'literature_opencv_worker.py', '--image', pair['image'], '--template', pair['template'],
           '--angle-step', 1, '--scales', *pair['scales'], '--score', pair['score'], '--limit', pair['limit'],
           '--threads', threads, '--output', out]
    if smoke_maps:
        cmd += ['--max-maps', smoke_maps]
    return worker_result(*run(cmd, out, lock_root, 3600), out)


def tdcm(external, pair, out, lock_root, device, threads):
    py = external / 'envs/tdcm-py312/bin/python'
    cmd = [py, '-I', TOOLS / 'literature_tdcm_worker.py', '--repo', external / 'repos/tdcm',
           '--weights', external / 'repos/tdcm/dict/model.pth', '--image', pair['image'], '--template', pair['template'],
           '--device', device, '--threads', threads, '--output', out]
    return worker_result(*run(cmd, out, lock_root, 3600), out)


def geco2(external, pair, out, lock_root, device, threads):
    home = external / 'torch-home'
    link = home / 'hub/checkpoints/sam2_hiera_base_plus.pt'
    target = external / 'data/sam2-hiera-base-plus/sam2_hiera_base_plus.pt'
    if not link.exists():
        if sha256(target) != SAM2_SHA256:
            raise SystemExit('SAM2 checkpoint hash differs from the recorded download')
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    py = external / 'envs/geco2-py312/bin/python'
    cmd = [py, '-I', TOOLS / 'literature_geco2_worker.py', '--repo', external / 'repos/geco2',
           '--weights', external / 'data/geco2-weights-hf/GECO2FSCD.pth', '--torch-home', home,
           '--image', pair['image'], '--exemplars', json.dumps(pair['exemplars']), '--threshold', .11,
           '--device', device, '--threads', threads, '--output', out]
    return worker_result(*run(cmd, out, lock_root, 3600), out)


def scored(pair, result):
    if result.get('status') != 'ok':
        return None
    dets = result['detections']
    tw, th = pair['template_size']
    out = dict(count=len(dets), boxes_iou50=score.score_box_images([([score.normalise(g)['box'] for g in pair['gt']], dets)]))
    if all(d['kind'] == 'pose' for d in dets):
        out['pose'] = score.score_pose_image(pair['gt'], dets, loose_radius=min(tw, th) / 2)
        out['pose']['errors'] = {k: score.summary([e[k] for e in out['pose']['errors']]) for k in
                                 ('position_px', 'angle_deg', 'width_rel', 'height_rel')}
        out['pose']['loose_errors'] = {k: score.summary([abs(e[k]) for e in out['pose']['loose_errors']]) for k in
                                       ('position_px', 'angle_deg', 'width_rel', 'height_rel')}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external', type=Path, required=True)
    p.add_argument('--build-root', type=Path, required=True, help='directory holding stage37-full-gpu inputs')
    p.add_argument('--ours-build', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--lock-root', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--pairs', nargs='+', default=['tdcm_demo', 'many128'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--opencv-smoke-maps', type=int, default=6, help='many128 only: maps run before extrapolating')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    defined = pairs(args.external, args.build_root)
    summary_path = args.output_dir / 'smoke-summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(results={})
    summary['inputs'] = {name: dict(image=str(v['image']), image_sha256=sha256(v['image']), template=str(v['template']),
                                    template_sha256=sha256(v['template']), gt_source=v['gt_source'], gt_count=len(v['gt']),
                                    geco2_exemplars=v['exemplars']) for name, v in defined.items()}
    for name in args.pairs:
        pair = defined[name]
        for method in args.methods:
            out = args.output_dir / f'{name}-{method}.json'
            started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            try:
                if method == 'ours_core':
                    result = ours_core(args.ours_build, pair, out, args.lock_root)
                elif method in RUNNER_ARMS:
                    result = ours_runner(args.ours_build, method, pair, out, args.lock_root)
                elif method == 'opencv':
                    result = opencv(pair, out, args.lock_root, args.threads, args.opencv_smoke_maps if name == 'many128' else 0)
                elif method.startswith('tdcm'):
                    result = tdcm(args.external, pair, out, args.lock_root, method.split('_')[1], args.threads)
                else:
                    result = geco2(args.external, pair, out, args.lock_root, method.split('_')[1], args.threads)
            except subprocess.TimeoutExpired as e:
                result = dict(status='timeout', seconds=e.timeout)
            result['started'] = started
            if result.get('status') == 'ok' and not out.exists():
                out.write_text(json.dumps(result, indent=1) + '\n')
            result['score'] = scored(pair, result)
            keep = {k: v for k, v in result.items() if k != 'detections'}
            keep['detections_head'] = result.get('detections', [])[:3]
            summary['results'][f'{name}/{method}'] = keep
            summary_path.write_text(json.dumps(summary, indent=1) + '\n')
            print(name, method, result.get('status'), (result.get('score') or {}).get('count'), flush=True)


if __name__ == '__main__':
    main()
