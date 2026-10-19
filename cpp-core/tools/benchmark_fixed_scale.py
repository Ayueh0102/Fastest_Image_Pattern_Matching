"""Bounded full-image, fixed-scale rotation study using the frozen native Session.

Preparation writes a protocol before any measured process starts. Search workers
never receive ground truth or placement ROIs. Every method gets the same search
inputs; only the learned pyramid depth changes. No matching code is replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

import cv2
import numpy as np

from many_object_accuracy import evaluate
from many_object_scenes import _corners, _roi, render_scene, validate_placements


ROOT = Path(__file__).resolve().parents[2]
METHODS = ('pyramid', 'full_resolution')
CASE_NAMES = ('crop_small360', 'crop_medium20', 'crop_lowcontrast20',
              'thin_detail20', 'scale_violation20', 'repo_src4')
GIT = '/Library/Developer/CommandLineTools/usr/bin/git'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def distribution(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)


def levels(shape, min_area):
    area, limit, count = math.prod(shape), math.isqrt(min_area)**2, 1
    while area > limit:
        area //= 4
        count += 1
    return count


def original_resolution_min_area(shape):
    # Pattern uses floor(sqrt(min_area)) squared, so template area alone can
    # accidentally request another level when that area is not a square.
    root = math.isqrt(math.prod(shape))
    return (root+(root*root < math.prod(shape)))**2


def work_bound(case):
    """Conservative dense-NCC pixel products; a scheduling guard, not FLOPs.

    Rotated source canvases fit within (diagonal + 2*max(template_side)+8)^2.
    Bound applies to coarse full-resolution maps, excluding local refinement.
    Actual maps/windows are recorded independently by the native runner.
    """
    th, tw = case['template_shape']
    h, w = case['scene_shape']
    step = math.degrees(math.atan(2/max(th, tw)))
    angles = 1 if case['angle'] == 0 else 2*math.ceil(case['angle']/step)+1
    side = math.ceil(math.hypot(w, h))+2*max(th, tw)+8
    return dict(angle_step_deg=step, angle_count_upper=angles,
                rotated_canvas_pixels_upper=side*side,
                dense_pixel_products_upper=side*side*th*tw*angles)


def write_image(path, pixels):
    if not cv2.imwrite(str(path), pixels):
        raise RuntimeError('Cannot write image: '+str(path))


def make_cases(inputs):
    inputs.mkdir(parents=True, exist_ok=True)
    photo = ROOT/'Test Images/Src3.bmp'
    photographed = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
    if photographed is None:
        raise ValueError('Missing source photograph')
    x, y, w, h = 450, 8, 64, 48
    template = photographed[y:y+h, x:x+w].copy()
    template_path = inputs/'src3_native_crop.png'
    write_image(template_path, template)
    crop = dict(source='Test Images/Src3.bmp', source_sha256=sha(photo),
                crop_xywh=[x, y, w, h], resizing=False,
                meaning='Native-pixel photographed patch, not an isolated whole physical object')
    thin = np.full((48, 64), 127, np.uint8)
    # Sparse, asymmetric, one-pixel detail is intentionally hostile to blur
    # and decimation. This is a targeted stress case, not real-world evidence.
    rng = np.random.default_rng(2026101703)
    for row, col, length in zip(rng.integers(4, 44, 40), rng.integers(4, 53, 40), rng.integers(2, 10, 40)):
        thin[row, col:col+length] = 245 if (row+col)%2 else 10
    thin[8:39, 51] = 20
    thin[37, 18:51] = 235
    thin_path = inputs/'thin_detail_template.png'
    write_image(thin_path, thin)
    configurations = [
        ('crop_small360', (256, 256), [(127.35, 129.65, 37.3)], 180., 1., 1., 2.),
        ('crop_medium20', (512, 768), [(x+.35, y+.65, a) for (x, y, a) in
            [(128, 128, -17.3), (384, 128, 9.7), (640, 128, -3.4),
             (128, 384, 14.6), (384, 384, -11.8), (640, 384, 2.3)]], 20., 1., 1., 2.),
        ('crop_lowcontrast20', (384, 512), [(128.35, 128.65, -14.7), (384.35, 256.65, 11.3)], 20., 1., .3, 4.),
        ('thin_detail20', (256, 256), [(128.35, 127.65, 7.3)], 20., 1., 1., 1.),
        ('scale_violation20', (256, 256), [(127.35, 129.65, -12.7)], 20., 1.04, 1., 2.),
    ]
    cases = []
    for index, (name, shape, poses, angle, scale, gain, noise) in enumerate(configurations):
        texture = thin if name == 'thin_detail20' else template
        texture_path = thin_path if name == 'thin_detail20' else template_path
        truth = []
        for i, (px, py, a) in enumerate(poses):
            corners = _corners([px, py], a, scale, texture.shape)
            truth.append(dict(index=i, pos_x=px, pos_y=py, angle=a, scale=scale,
                              gain=gain, bias=128*(1-gain), corners=corners, warp_roi=_roi(corners)))
        seed = 2026101700+index
        source_path = inputs/(name+'.png')
        image = render_scene(texture, truth, shape, noise_seed=seed, noise_std=noise, background=128.)
        write_image(source_path, image)
        meta_path = inputs/(name+'.json')
        metadata = dict(name=name, source_kind='targeted synthetic detail' if name == 'thin_detail20' else 'photographed crop independently composited into a synthetic scene',
                        template_origin=None if name == 'thin_detail20' else crop,
                        renderer='OpenCV warpAffine INTER_LINEAR at native template size; synthetic noise',
                        numpy_version=np.__version__, opencv_version=cv2.__version__,
                        source_shape=list(shape), template_shape=list(texture.shape),
                        noise_seed=seed, noise_std=noise, background=128., truth=truth,
                        validation=validate_placements(truth, shape, texture.shape),
                        limitations=['Ground truth is synthetic, not a new independently photographed object.',
                                     'One template, separated instances, no occlusion.',
                                     'No search ROI or object count is supplied to the matcher.'])
        save(meta_path, metadata)
        cases.append(dict(name=name, source=str(source_path), template=str(texture_path),
                          metadata=str(meta_path), scene_shape=list(shape), template_shape=list(texture.shape),
                          angle=angle, truth=truth, validity='assumption_violation' if scale != 1. else 'fixed_scale_gt',
                          backends=[0, 1] if name == 'crop_small360' else [1]))
    # Preserve the original sample pixels. Without independently labelled poses,
    # report detections and agreement, never a recall/accuracy claim.
    real_source, real_template = ROOT/'Test Images/Src4.bmp', ROOT/'Test Images/Dst4.bmp'
    real_meta = inputs/'repo_src4.json'
    save(real_meta, dict(source_kind='unmodified repository real image/template pair',
                        source='Test Images/Src4.bmp', template='Test Images/Dst4.bmp',
                        source_sha256=sha(real_source), template_sha256=sha(real_template),
                        truth=None, limitations=['No independent ground truth; output agreement is not accuracy.']))
    cases.append(dict(name='repo_src4', source=str(real_source), template=str(real_template),
                      metadata=str(real_meta), scene_shape=list(cv2.imread(str(real_source), 0).shape),
                      template_shape=list(cv2.imread(str(real_template), 0).shape), angle=180.,
                      truth=None, validity='unlabelled_real', backends=[1]))
    for case in cases:
        case['work_bound'] = work_bound(case)
    return cases


def prepare(args, protocol_path):
    cases = make_cases(protocol_path.parent/'inputs')
    if args.cases:
        selected = args.cases.split(',')
        if len(set(selected)) != len(selected) or set(selected)-set(CASE_NAMES):
            raise ValueError('Select unique known cases: '+','.join(CASE_NAMES))
        cases = [case for case in cases if case['name'] in selected]
    sources = [Path(__file__).resolve(), Path(__file__).with_name('many_object_scenes.py'),
               Path(__file__).with_name('many_object_accuracy.py'), ROOT/'cpp-core/tools/native_pipeline_runner.cpp']
    for directory in ('src', 'include', 'experiments', 'tests'):
        sources += [p for p in (ROOT/'cpp-core'/directory).rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    sources += list((ROOT/'cpp-core/tools').glob('*.py'))
    sources += [ROOT/'cpp-core/CMakeLists.txt', ROOT/'macos-cli/mainwindow.cpp', ROOT/'macos-cli/mainwindow.h']
    binaries = [args.build/'experiments'/name for name in
                ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib', 'libfpm_pose_experiment.dylib')]
    protocol = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    baseline_commit=subprocess.check_output([GIT, 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    platform=platform.platform(), build=str(args.build.resolve()), cases=cases,
                    pilot=args.pilot, blocks=1 if args.pilot else 2, repeats=1 if args.pilot else 3,
                    warmup_seconds=0. if args.pilot else 1., timeout_seconds=120.,
                    dense_pixel_products_limit=200_000_000_000,
                    methods=dict(pyramid='min_area=256', full_resolution='min_area=ceil(sqrt(template_area))^2; one level'),
                    shared_settings=dict(scales=[1.], optimize_scale=False, stop_layer=0, refine=True,
                        subpixel=True, score=.8, proposal_score=.5, overlap=.2, limit=64,
                        ncc_mode=0, overlap_enabled=False),
                    timing='Fresh serial child processes, forward/reverse block order; construction and first call separate; prepare included in total_ms; JSON and input decoding excluded.',
                    memory='Peak/current process RSS separately from retained frame/pose capacities; never add overlapping counters.',
                    geometry='Per method, independent one-to-one GT matching at <=2px, <=1deg, and scale ratio 0.98..1.02; raw poses retained. No cross-method exactness requirement.',
                    gates=dict(correctness='All fixed_scale_gt cases must have TP=truth_count, FP=FN=0 independently. Negative control is descriptive.',
                        performance='Full-resolution per-block median <= pyramid median + max(2ms,10%); no claim if timeout/skip/error.',
                        memory='Full-resolution peak RSS <= pyramid peak RSS + max(32MiB,10%) for each paired block.',
                        scheduling='Skip an entire case/backend pair if the conservative dense work bound exceeds the limit; timeout retained as failure, no silent retry.'),
                    scope='Original-resolution FULL IMAGE search; template cropping defines what to find, not where to search. No automatic segmentation, ROI hints, or scale-bank search.',
                    limitations=['Fixed-scale synthetic studies and one unlabelled repository photo do not establish general real-camera accuracy.',
                                 'Single-level route retains local original-resolution refinement; it removes image/template downsampling, not all coarse-to-fine ideas.',
                                 'Template/source rotations follow the existing runtime; this experiment adds no rotated-template kernel.'],
                    source_sha256={str(p.resolve()):sha(p) for p in sources},
                    binary_sha256={str(p.resolve()):sha(p) for p in binaries},
                    input_sha256={case[key]:sha(case[key]) for case in cases for key in ('source', 'template', 'metadata')})
    save(protocol_path, protocol)
    return protocol


def native_job(case, method, backend, protocol):
    settings = dict(protocol['shared_settings'])
    settings.pop('optimize_scale')  # Session derives this from the single scale.
    return dict(settings, source=case['source'], template=case['template'], angle=case['angle'],
                backend=backend, min_area=256 if method == 'pyramid' else original_resolution_min_area(case['template_shape']),
                repeats=protocol['repeats'], warmup_seconds=protocol['warmup_seconds'])


def summarize(protocol, entries, destination):
    cases = []
    for case in protocol['cases']:
        for backend in case['backends']:
            group = [e for e in entries if e['case'] == case['name'] and e['backend'] == backend]
            methods = {}
            for method in METHODS:
                measured = [e for e in group if e['method'] == method and e['status'] == 'ok']
                workers = [(e, json.loads((destination/e['result']).read_text())) for e in measured]
                if not workers:
                    methods[method] = dict(status='unavailable', outcomes=[e['status'] for e in group if e['method'] == method])
                    continue
                calls = [r for _, worker in workers for r in [worker['first'], *worker['runs']]]
                runs = [r for _, worker in workers for r in worker['runs']]
                accuracy = [evaluate(r['matches'], case['truth']) for r in calls] if case['truth'] is not None else None
                pyramid_levels = levels(case['template_shape'], 256 if method == 'pyramid' else original_resolution_min_area(case['template_shape']))
                host_expected = 0
                height, width = case['scene_shape']
                for _ in range(pyramid_levels):
                    host_expected += height*width
                    height, width = (height+1)//2, (width+1)//2
                methods[method] = dict(status='complete' if len(workers) == protocol['blocks'] else 'partial',
                    total_ms=distribution([r['total_ms'] for r in runs]),
                    setup_ms=[w['construct_ms'] for _, w in workers], first_ms=[w['first']['total_ms'] for _, w in workers],
                    peak_rss_bytes=[w['final_memory']['peak_rss_bytes'] for _, w in workers],
                    current_rss_bytes=[w['final_memory']['current_rss_bytes'] for _, w in workers],
                    stage_medians={k:statistics.median(r[k] for r in runs) for k in ('prepare_ms', 'proposal_ms', 'refine_ms', 'nms_ms')},
                    stages_note='proposal_ms includes prepare_ms in the current runner; do not sum both.',
                    expected_pyramid_levels=pyramid_levels, expected_host_pyramid_bytes=host_expected,
                    host_pyramid_bytes_correct=all(r['frame_host_bytes'] == host_expected for r in calls),
                    fixed_scale_preserved=all(all(m['scale'] == 1. for m in r[k])
                        for r in calls for k in ('matches', 'proposals')),
                    one_model_used=all(r['models_matched'] == 1 for r in calls),
                    refinement_backend_valid=all(r['selected_refinement_backend'] == backend
                        for r in calls if r['proposals']),
                    repeat_exact=all(w['exact_repeat'] for _, w in workers),
                    accuracy=accuracy, all_strict_gt=None if accuracy is None else all(a['strict']['fn'] == a['strict']['fp'] == 0 for a in accuracy),
                    first_outputs=[w['first']['matches'] for _, w in workers],
                    blocks=[dict(block=e['block'], median_ms=statistics.median(r['total_ms'] for r in w['runs']),
                                 peak_rss_bytes=w['final_memory']['peak_rss_bytes'], work=w['runs'][-1]['work'],
                                 capacities={k:w['runs'][-1][k] for k in ('frame_host_bytes', 'frame_coarse_gpu_bytes', 'frame_fine_gpu_bytes', 'pose_host_bytes', 'pose_gpu_bytes')}) for e, w in workers])
            pairs = []
            if all(methods[m]['status'] == 'complete' for m in METHODS):
                reference = {b['block']:b for b in methods['pyramid']['blocks']}
                for b in methods['full_resolution']['blocks']:
                    a = reference[b['block']]
                    pairs.append(dict(block=b['block'], speed_ratio=b['median_ms']/a['median_ms'],
                                      time_delta_percent=100*(b['median_ms']/a['median_ms']-1),
                                      speed_gate=b['median_ms'] <= a['median_ms']+max(2., .1*a['median_ms']),
                                      rss_delta_bytes=b['peak_rss_bytes']-a['peak_rss_bytes'],
                                      memory_gate=b['peak_rss_bytes'] <= a['peak_rss_bytes']+max(32*2**20, .1*a['peak_rss_bytes'])))
            cases.append(dict(name=case['name'], backend=backend, validity=case['validity'], methods=methods,
                              paired_comparison=pairs, work_bound=case['work_bound'],
                              output_comparison='Cross-method ordered values may differ; GT evaluates both independently.'))
    expected_workers = sum(len(c['backends']) for c in protocol['cases'])*protocol['blocks']*len(METHODS)
    return dict(cases=cases, expected_workers=expected_workers, recorded_workers=len(entries),
                all_workers_succeeded=len(entries) == expected_workers and all(e['status'] == 'ok' for e in entries),
                route_integrity_passed=all(m.get('status') == 'complete' and all(m.get(k) for k in (
                    'host_pyramid_bytes_correct', 'fixed_scale_preserved', 'one_model_used',
                    'refinement_backend_valid', 'repeat_exact')) for c in cases for m in c['methods'].values()),
                fixed_scale_gt_all_passed=all(m.get('status') == 'complete' and m.get('all_strict_gt')
                    for c in cases if c['validity'] == 'fixed_scale_gt' for m in c['methods'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/gui-integration')
    parser.add_argument('--output', type=Path, default=ROOT/'build/fixed-scale-study/formal.json')
    parser.add_argument('--cases', help=','.join(CASE_NAMES))
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        for entry in report['execution_order']:
            for key, digest in (('job', 'job_sha256'), ('result', 'result_sha256')):
                if key in entry and sha(args.summarize.parent/entry[key]) != entry[digest]:
                    raise ValueError('Changed archived '+key)
        save(output, summarize(report['protocol'], report['execution_order'], args.summarize.parent))
        return
    protocol_path = output.with_suffix('.protocol.json')
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if args.pilot != protocol['pilot']:
            raise ValueError('Pilot flag differs from frozen protocol; use a separate output folder')
        if str(args.build.resolve()) != protocol['build']:
            raise ValueError('Build directory differs from frozen protocol')
        if args.cases and set(args.cases.split(',')) != {c['name'] for c in protocol['cases']}:
            raise ValueError('Cases differ from frozen protocol')
    else:
        protocol = prepare(args, protocol_path)
    if args.prepare_only:
        print(json.dumps(dict(protocol=str(protocol_path), cases=[dict(name=c['name'], work_bound=c['work_bound']) for c in protocol['cases']])))
        return
    if output.exists():
        raise ValueError('Report exists; preserve it and use a separate output folder')
    hash_groups = ('source_sha256', 'binary_sha256', 'input_sha256')
    if not all(sha(p) == digest for group in hash_groups for p, digest in protocol[group].items()):
        raise ValueError('Prepared source/binary/input changed; create a new protocol after freezing inputs')
    report = dict(version=1, protocol=protocol, protocol_sha256=sha(protocol_path), execution_order=[], complete=False)
    save(output, report)
    for case in protocol['cases']:
        for backend in case['backends']:
            for block in range(protocol['blocks']):
                for method in METHODS if block%2 == 0 else METHODS[::-1]:
                    label = f'{case["name"]}-b{backend}-{method}-block{block}'
                    entry = dict(case=case['name'], backend=backend, method=method, block=block)
                    if case['work_bound']['dense_pixel_products_upper'] > protocol['dense_pixel_products_limit']:
                        entry.update(status='skipped_work_budget', reason='Conservative full-resolution dense work exceeds the predeclared bound')
                    else:
                        jobpath = output.parent/(output.stem+'-jobs')/(label+'.json')
                        save(jobpath, native_job(case, method, backend, protocol))
                        entry.update(job=str(jobpath.relative_to(output.parent)), job_sha256=sha(jobpath))
                        print('Running '+label, flush=True)
                        command = [str(Path(protocol['build'])/'experiments/fpm_native_pipeline_runner'), str(jobpath)]
                        stdout, stderr = '', ''
                        try:
                            child = subprocess.run(command, capture_output=True, text=True, timeout=protocol['timeout_seconds'])
                            stdout, stderr = child.stdout, child.stderr
                            entry.update(status='ok' if child.returncode == 0 else 'process_error', returncode=child.returncode)
                        except subprocess.TimeoutExpired as error:
                            stdout, stderr = error.stdout or '', error.stderr or ''
                            if isinstance(stdout, bytes): stdout = stdout.decode(errors='replace')
                            if isinstance(stderr, bytes): stderr = stderr.decode(errors='replace')
                            entry.update(status='timeout', timeout_seconds=protocol['timeout_seconds'])
                        stdoutpath, stderrpath = jobpath.with_suffix('.stdout.txt'), jobpath.with_suffix('.stderr.txt')
                        stdoutpath.write_text(stdout); stderrpath.write_text(stderr)
                        entry.update(stdout=str(stdoutpath.relative_to(output.parent)), stdout_sha256=sha(stdoutpath),
                                     stderr=str(stderrpath.relative_to(output.parent)), stderr_sha256=sha(stderrpath))
                        if entry['status'] == 'ok':
                            try:
                                raw = json.loads(stdout)
                                if not raw.get('exact_repeat'):
                                    entry['status'] = 'repeat_mismatch'
                                resultpath = jobpath.with_suffix('.result.json')
                                save(resultpath, raw)
                                entry.update(result=str(resultpath.relative_to(output.parent)), result_sha256=sha(resultpath))
                            except (ValueError, KeyError) as error:
                                entry.update(status='invalid_output', error=str(error))
                    report['execution_order'].append(entry)
                    save(output, report)
    report['complete'] = True
    report['unchanged'] = all(sha(p) == digest for group in hash_groups for p, digest in protocol[group].items())
    report['summary'] = summarize(protocol, report['execution_order'], output.parent)
    save(output, report)
    # Accuracy and performance failures remain results, not reasons to discard
    # this report. Process/integrity failures still return a failing status.
    if not report['unchanged'] or not report['summary']['all_workers_succeeded']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
