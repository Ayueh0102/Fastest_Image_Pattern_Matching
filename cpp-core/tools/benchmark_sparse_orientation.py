"""Compare sparse original-resolution proposals with the frozen pyramid route.

Workers receive pixels and public search settings, never ground truth or ROIs.
Preparation snapshots the implementation and records source/binary/input hashes
before any measured worker starts. Accuracy failures remain in the report.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tarfile

import cv2
import numpy as np

from benchmark_fixed_scale import (CASE_NAMES as FIXED_CASE_NAMES, GIT, ROOT, distribution, levels,
                                   make_cases, original_resolution_min_area, save, sha, write_image)
from many_object_accuracy import _metric, _poses, evaluate
from many_object_scenes import _corners, _roi, render_scene, validate_placements

METHODS = ('pyramid', 'sparse_orientation')
CROWD_NAME = 'heldout_crowd128_20'
CASE_NAMES = (*FIXED_CASE_NAMES, CROWD_NAME)
BINARY_NAMES = ('fpm_native_pipeline_runner', 'libfpm_coarse_experiment.dylib',
                'libfpm_pose_experiment.dylib')
CAPACITIES = ('frame_host_bytes', 'frame_coarse_gpu_bytes', 'frame_fine_gpu_bytes',
              'pose_host_bytes', 'pose_gpu_bytes')
SPARSE_DEFAULTS = dict(features=128, stride=4, angle_step=6., spread=2,
                       threshold=.75, candidate_limit=256)
BASIN = dict(position_px=6., angle_deg=6., relative_scale=.02)


def proposal_accuracy(proposals, truths, basin=BASIN):
    """Measure candidate coverage, independently of final NCC acceptance.

    A candidate must lie within the predeclared position/angle/scale basin.
    One-to-one assignment prevents one candidate from covering several objects.
    This geometric coverage is not a guarantee of optimizer convergence.
    """
    predicted, expected = _poses(proposals, 'proposals'), _poses(truths, 'truths')
    if not (np.isfinite(expected).all(axis=1) & (expected[:, 3] > 0)).all():
        raise ValueError('Invalid ground truth')
    valid = np.isfinite(predicted).all(axis=1) & (predicted[:, 3] > 0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        position = np.hypot(predicted[None, :, 0]-expected[:, None, 0],
                            predicted[None, :, 1]-expected[:, None, 1])
        angle = np.abs((predicted[None, :, 2]-expected[:, None, 2]+180) % 360-180)
        scale_ratio = predicted[None, :, 3]/expected[:, None, 3]
        relative_scale = np.abs(scale_ratio-1)
        pose_edges = valid[None, :] & (position <= basin['position_px']) & (angle <= basin['angle_deg'])
        strict_edges = (pose_edges & (scale_ratio >= 1-basin['relative_scale'])
                        & (scale_ratio <= 1+basin['relative_scale']))
    return dict(thresholds=basin, truth_count=len(truths), proposal_count=len(proposals),
                meaning='Geometric candidate coverage, not final accuracy or guaranteed convergence',
                strict=_metric(strict_edges, position, angle, relative_scale),
                position_angle_only=_metric(pose_edges, position, angle, relative_scale))


def sparse_settings(args):
    settings = {key: getattr(args, key) for key in SPARSE_DEFAULTS}
    if not 1 <= settings['features'] <= 4096:
        raise ValueError('Feature count must be in [1,4096]')
    if not 1 <= settings['stride'] <= 64 or not 0 <= settings['spread'] <= 16:
        raise ValueError('Stride must be in [1,64] and spread in [0,16]')
    if not .1 <= settings['angle_step'] <= 180 or not 0 < settings['threshold'] <= 1:
        raise ValueError('Angle step must be in [0.1,180] and threshold in (0,1]')
    if not 1 <= settings['candidate_limit'] <= 65536:
        raise ValueError('Candidate limit must be in [1,65536]')
    return {'sparse_'+key: value for key, value in settings.items()}


def make_crowd_case(inputs):
    """Held-out dense instance load; fixed limits do not depend on its truth."""
    template_path = inputs/'src3_native_crop.png'
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    shape = (768, 1536)
    rng = np.random.default_rng(2026101407)
    truth = []
    for row in range(8):
        for column in range(16):
            px = 48+96*column+float(rng.uniform(-.4, .4))
            py = 48+96*row+float(rng.uniform(-.4, .4))
            angle = float(rng.uniform(-19., 19.))
            corners = _corners([px, py], angle, 1., template.shape)
            truth.append(dict(index=len(truth), pos_x=px, pos_y=py, angle=angle, scale=1.,
                              gain=1., bias=0., corners=corners, warp_roi=_roi(corners)))
    source = inputs/(CROWD_NAME+'.png')
    write_image(source, render_scene(template, truth, shape, noise_seed=2026101408,
                                     noise_std=2., background=128.))
    metadata = inputs/(CROWD_NAME+'.json')
    save(metadata, dict(name=CROWD_NAME, source_kind='held-out synthetic 128-instance load using photographed crop',
        template_origin=dict(source='Test Images/Src3.bmp', crop_xywh=[450, 8, 64, 48], resizing=False),
        source_shape=list(shape), template_shape=list(template.shape), truth=truth,
        pose_seed=2026101407, noise_seed=2026101408, noise_std=2.,
        numpy_version=np.__version__, opencv_version=cv2.__version__,
        validation=validate_placements(truth, shape, template.shape),
        limitations=['Separated identical-template instances with no occlusion or background clutter.',
                     'This case is excluded from pilot runs; no object count or placement is supplied to the matcher.']))
    return dict(name=CROWD_NAME, source=str(source), template=str(template_path), metadata=str(metadata),
                scene_shape=list(shape), template_shape=list(template.shape), angle=20., truth=truth,
                validity='fixed_scale_gt', backends=[1], held_out=True)


def prepare(args, protocol_path):
    cases = make_cases(protocol_path.parent/'inputs')
    if not args.pilot:
        cases.append(make_crowd_case(protocol_path.parent/'inputs'))
    elif args.cases and CROWD_NAME in args.cases.split(','):
        raise ValueError('The held-out crowd case is formal-only; do not tune against it')
    if args.cases:
        selected = args.cases.split(',')
        if len(set(selected)) != len(selected) or set(selected)-set(CASE_NAMES):
            raise ValueError('Select unique known cases: '+','.join(CASE_NAMES))
        cases = [case for case in cases if case['name'] in selected]
    if args.skip_cpu:
        for case in cases:
            case['backends'] = [backend for backend in case['backends'] if backend == 1]
    builds = dict(pyramid=str(args.baseline_build.resolve()), sparse_orientation=str(args.build.resolve()))
    binaries = {str((Path(build)/'experiments'/name).resolve()):
                sha(Path(build)/'experiments'/name) for build in builds.values() for name in BINARY_NAMES}
    baseline_protocol = json.loads(args.baseline_protocol.read_text())
    baseline_expected = {Path(path).name: digest for path, digest in baseline_protocol['binary_sha256'].items()}
    for name in BINARY_NAMES:
        path = str((args.baseline_build/'experiments'/name).resolve())
        if binaries[path] != baseline_expected.get(name):
            raise ValueError('Baseline binary differs from the archived fixed-scale protocol: '+name)
    sources = set()
    for directory in ('src', 'include', 'experiments', 'tests', 'tools'):
        sources.update(path for path in (ROOT/'cpp-core'/directory).rglob('*')
                       if path.is_file() and '__pycache__' not in path.parts and
                       (path.suffix in ('.cpp', '.hpp', '.h', '.mm', '.metal', '.in', '.py')
                        or path.name == 'CMakeLists.txt'))
    sources.update((ROOT/'cpp-core/CMakeLists.txt', ROOT/'macos-cli/mainwindow.cpp',
                    ROOT/'macos-cli/mainwindow.h'))
    snapshot_root = protocol_path.parent/'source-snapshot'
    if snapshot_root.exists():
        raise ValueError('Source snapshot already exists; choose a new output directory')
    for path in sorted(sources):
        target = snapshot_root/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    baseline_snapshot = protocol_path.parent/'baseline-source.tar'
    with baseline_snapshot.open('wb') as target:
        subprocess.run([GIT, 'archive', baseline_protocol['baseline_commit'],
                        'cpp-core/src', 'cpp-core/include', 'cpp-core/experiments',
                        'cpp-core/tests', 'cpp-core/tools', 'cpp-core/CMakeLists.txt',
                        'macos-cli/mainwindow.cpp', 'macos-cli/mainwindow.h', 'macos-cli/main.cpp'],
                       cwd=ROOT, stdout=target, check=True)
    baseline_protocol_copy = protocol_path.parent/'baseline-protocol.json'
    shutil.copyfile(args.baseline_protocol, baseline_protocol_copy)
    verified_baseline_sources = {}
    with tarfile.open(baseline_snapshot) as archive:
        for path, digest in baseline_protocol['source_sha256'].items():
            relative = Path(path).relative_to(ROOT)
            runtime = (relative.parts[:2] in (('cpp-core', 'src'), ('cpp-core', 'include'),
                                             ('cpp-core', 'experiments')) and
                       relative.suffix in ('.cpp', '.hpp', '.h', '.mm', '.metal', '.in'))
            if runtime or str(relative) == 'cpp-core/tools/native_pipeline_runner.cpp':
                member = archive.extractfile(str(relative))
                actual = hashlib.sha256(member.read()).hexdigest() if member is not None else None
                if actual != digest:
                    raise ValueError('Archived baseline runtime source differs from its original protocol: '+str(relative))
                verified_baseline_sources[str(relative)] = digest
    protocol = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        implementation_commit=subprocess.check_output([GIT, 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        baseline_commit=baseline_protocol['baseline_commit'],
        baseline_protocol_sha256=sha(args.baseline_protocol),
        baseline_protocol_copy=baseline_protocol_copy.name,
        baseline_runtime_source_verified=verified_baseline_sources,
        baseline_source_archive=dict(path=baseline_snapshot.name, sha256=sha(baseline_snapshot)),
        platform=platform.platform(), builds=builds, cases=cases,
        pilot=args.pilot, blocks=1 if args.pilot else 2, repeats=1 if args.pilot else 3,
        warmup_seconds=0. if args.pilot else 1., timeout_seconds=args.timeout,
        methods=dict(pyramid='Frozen fixed-scale native Session; min_area=256',
                     sparse_orientation='Original-resolution sparse orientation candidates, full NCC and resident/CPU pose refinement; no pyramid'),
        sparse_settings=sparse_settings(args), proposal_basin=BASIN,
        shared_settings=dict(scales=[1.], optimize_scale=False, stop_layer=0, refine=True,
            subpixel=True, score=.8, proposal_score=.5, overlap=.2, limit=256,
            ncc_mode=0, overlap_enabled=False),
        timing='Fresh serial workers; forward/reverse block order. Construction and first call separate. Algorithm times exclude decoding, JSON and parent-side accuracy.',
        memory='Peak/current process RSS separately from retained frame/pose/sparse capacities; do not add overlapping counters.',
        geometry='Independent maximum-cardinality one-to-one matching. Final <=2px, <=1deg, scale ratio [0.98,1.02]. Proposal basin <=6px, <=6deg, same scale tolerance, fixed before measurement.',
        gates=dict(correctness='Every normal fixed-scale GT case has final TP=truth_count and FP=FN=0 independently for each route.',
            proposal_coverage='Every normal fixed-scale GT case has proposal basin recall=1; coverage does not guarantee convergence.',
            performance='Sparse per-block median <= pyramid median + max(2ms,10%).',
            memory='Sparse peak RSS <= pyramid peak RSS + max(32MiB,10%) per paired block.',
            integrity='All workers succeed; all hashes unchanged; fixed scale, selected backend, no sparse pyramid and exact own repeats.'),
        scope='Native-pixel cropped template searched across the entire image. Rotation is searched; scale is fixed at 1. No ground truth, count or placement ROI reaches workers.',
        limitations=['Photographed crops are independently composited into synthetic scenes; they do not establish independent real-camera accuracy.',
            'Sparse orientation score is a heuristic, not an NCC upper bound; false-negative proposals are possible.',
            'The +4% scale case deliberately violates the fixed-scale assumption; the repository photo has no pose ground truth.',
            'The baseline executable is frozen and hash-checked against its earlier protocol; current implementation source is snapshotted before measurement.',
            'Both routes use final limit256 globally, so prior fixed-scale timings with limit64 are not a direct paired comparison.',
            'The 128-instance case is held out from pilot tuning; it measures a larger separated-object load, not arbitrary crowded-scene robustness.'],
        source_sha256={str(path.resolve()):sha(path) for path in sorted(sources)},
        source_snapshot_sha256={str(path.relative_to(protocol_path.parent)):sha(path)
                                for path in snapshot_root.rglob('*') if path.is_file()},
        binary_sha256=binaries,
        input_sha256={case[key]:sha(case[key]) for case in cases for key in ('source', 'template', 'metadata')})
    save(protocol_path, protocol)
    return protocol


def native_job(case, method, backend, protocol):
    settings = dict(protocol['shared_settings'])
    settings.pop('optimize_scale')
    result = dict(settings, source=case['source'], template=case['template'], angle=case['angle'],
                  backend=backend, min_area=256 if method == 'pyramid' else original_resolution_min_area(case['template_shape']),
                  repeats=protocol['repeats'], warmup_seconds=protocol['warmup_seconds'])
    if method == 'sparse_orientation':
        result.update(protocol['sparse_settings'], proposal_method=method)
    return result


def expected_frame_bytes(case):
    height, width = case['scene_shape']
    total = 0
    for _ in range(levels(case['template_shape'], 256)):
        total += height*width
        height, width = (height+1)//2, (width+1)//2
    return total


def numeric_metrics(records, key):
    """Keep all numeric native metrics without requiring a fixed schema."""
    names = {name for record in records for name, value in record.get(key, {}).items()
             if isinstance(value, (int, float)) and not isinstance(value, bool)}
    return {name: statistics.median(record[key][name] for record in records
                                    if name in record.get(key, {})) for name in sorted(names)}


def summarize(protocol, entries, destination):
    cases = []
    for case in protocol['cases']:
        for backend in case['backends']:
            group = [entry for entry in entries if entry['case'] == case['name'] and entry['backend'] == backend]
            methods = {}
            for method in METHODS:
                measured = [entry for entry in group if entry['method'] == method and entry['status'] == 'ok']
                workers = [(entry, json.loads((destination/entry['result']).read_text())) for entry in measured]
                if not workers:
                    methods[method] = dict(status='unavailable', outcomes=[entry['status'] for entry in group if entry['method'] == method])
                    continue
                calls = [record for _, worker in workers for record in [worker['first'], *worker['runs']]]
                runs = [record for _, worker in workers for record in worker['runs']]
                accuracy = [evaluate(record['matches'], case['truth']) for record in calls] if case['truth'] is not None else None
                proposal = [proposal_accuracy(record['proposals'], case['truth'], protocol['proposal_basin'])
                            for record in calls] if case['truth'] is not None else None
                route_valid = (all(record['frame_host_bytes'] == expected_frame_bytes(case) for record in calls)
                    if method == 'pyramid' else all(record.get('sparse', {}).get('enabled') is True and
                        record['sparse'].get('source_levels') == 1 for record in calls))
                methods[method] = dict(status='complete' if len(workers) == protocol['blocks'] else 'partial',
                    total_ms=distribution([record['total_ms'] for record in runs]),
                    setup_ms=[worker['construct_ms'] for _, worker in workers],
                    first_ms=[worker['first']['total_ms'] for _, worker in workers],
                    peak_rss_bytes=[worker['final_memory']['peak_rss_bytes'] for _, worker in workers],
                    current_rss_bytes=[worker['final_memory']['current_rss_bytes'] for _, worker in workers],
                    stage_medians={key:statistics.median(record[key] for record in runs)
                                   for key in ('prepare_ms', 'proposal_ms', 'refine_ms', 'nms_ms')},
                    stages_note='proposal_ms includes prepare_ms; do not sum both.',
                    sparse_metric_medians=numeric_metrics(runs, 'sparse'),
                    route_valid=route_valid,
                    fixed_scale_preserved=all(all(match['scale'] == 1. for match in record[key])
                        for record in calls for key in ('matches', 'proposals')),
                    refinement_backend_valid=all(record['selected_refinement_backend'] == backend
                                                 for record in calls if record['proposals']),
                    repeat_exact=all(worker['exact_repeat'] for _, worker in workers),
                    accuracy=accuracy, proposal_coverage=proposal,
                    all_strict_gt=None if accuracy is None else all(item['strict']['fn'] == item['strict']['fp'] == 0 for item in accuracy),
                    all_proposal_gt=None if proposal is None else all(item['strict']['fn'] == 0 for item in proposal),
                    first_outputs=[worker['first']['matches'] for _, worker in workers],
                    blocks=[dict(block=entry['block'], median_ms=statistics.median(record['total_ms'] for record in worker['runs']),
                        peak_rss_bytes=worker['final_memory']['peak_rss_bytes'],
                        work=worker['runs'][-1].get('work', []), sparse=worker['runs'][-1].get('sparse', {}),
                        capacities={key:worker['runs'][-1].get(key) for key in CAPACITIES}) for entry, worker in workers])
            pairs = []
            if all(methods[method]['status'] == 'complete' for method in METHODS):
                reference = {block['block']:block for block in methods['pyramid']['blocks']}
                for block in methods['sparse_orientation']['blocks']:
                    baseline = reference[block['block']]
                    ratio = block['median_ms']/baseline['median_ms']
                    pairs.append(dict(block=block['block'], sparse_over_pyramid_time=ratio,
                        time_delta_percent=100*(ratio-1),
                        speed_gate=block['median_ms'] <= baseline['median_ms']+max(2., .1*baseline['median_ms']),
                        rss_delta_bytes=block['peak_rss_bytes']-baseline['peak_rss_bytes'],
                        memory_gate=block['peak_rss_bytes'] <= baseline['peak_rss_bytes']+max(32*2**20, .1*baseline['peak_rss_bytes'])))
            cases.append(dict(name=case['name'], backend=backend, validity=case['validity'],
                              methods=methods, paired_comparison=pairs))
    expected = sum(len(case['backends']) for case in protocol['cases'])*protocol['blocks']*len(METHODS)
    return dict(cases=cases, expected_workers=expected, recorded_workers=len(entries),
        all_workers_succeeded=len(entries) == expected and all(entry['status'] == 'ok' for entry in entries),
        route_integrity_passed=all(method.get('status') == 'complete' and all(method.get(key) for key in
            ('route_valid', 'fixed_scale_preserved', 'refinement_backend_valid', 'repeat_exact'))
            for case in cases for method in case['methods'].values()),
        fixed_scale_gt_all_passed=all(method.get('status') == 'complete' and method.get('all_strict_gt')
            for case in cases if case['validity'] == 'fixed_scale_gt' for method in case['methods'].values()),
        fixed_scale_proposal_coverage_passed=all(method.get('status') == 'complete' and method.get('all_proposal_gt')
            for case in cases if case['validity'] == 'fixed_scale_gt' for method in case['methods'].values()))


def unchanged(protocol):
    return all(Path(path).is_file() and sha(path) == digest
               for group in ('source_sha256', 'binary_sha256', 'input_sha256')
               for path, digest in protocol[group].items())


def run_worker(protocol, case, backend, method, block, output):
    label = f'{case["name"]}-b{backend}-{method}-block{block}'
    entry = dict(case=case['name'], backend=backend, method=method, block=block)
    jobpath = output.parent/(output.stem+'-jobs')/(label+'.json')
    save(jobpath, native_job(case, method, backend, protocol))
    entry.update(job=str(jobpath.relative_to(output.parent)), job_sha256=sha(jobpath))
    print('Running '+label, flush=True)
    command = [str(Path(protocol['builds'][method])/'experiments/fpm_native_pipeline_runner'), str(jobpath)]
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
    stdoutpath.write_text(stdout)
    stderrpath.write_text(stderr)
    entry.update(stdout=str(stdoutpath.relative_to(output.parent)), stdout_sha256=sha(stdoutpath),
                 stderr=str(stderrpath.relative_to(output.parent)), stderr_sha256=sha(stderrpath))
    if entry['status'] == 'ok':
        try:
            raw = json.loads(stdout)
            if not raw.get('runs') or 'first' not in raw or 'final_memory' not in raw:
                raise ValueError('Missing native measurement records')
            resultpath = jobpath.with_suffix('.result.json')
            save(resultpath, raw)
            entry.update(result=str(resultpath.relative_to(output.parent)), result_sha256=sha(resultpath))
            if not raw.get('exact_repeat'):
                entry['status'] = 'repeat_mismatch'
        except (ValueError, KeyError, TypeError) as error:
            entry.update(status='invalid_output', error=str(error))
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/sparse-orientation')
    parser.add_argument('--baseline-build', type=Path, default=ROOT/'build/gui-integration')
    parser.add_argument('--baseline-protocol', type=Path, default=ROOT/'cpp-core/docs/fixed-scale/protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT/'build/sparse-orientation-study/formal/results.json')
    parser.add_argument('--cases', help=','.join(CASE_NAMES))
    parser.add_argument('--skip-cpu', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--timeout', type=float, default=120.)
    for key, value in SPARSE_DEFAULTS.items():
        parser.add_argument('--'+key.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        for entry in report['execution_order']:
            for key in ('job', 'result', 'stdout', 'stderr'):
                if key in entry and sha(args.summarize.parent/entry[key]) != entry[key+'_sha256']:
                    raise ValueError('Changed archived '+key)
        save(output, summarize(report['protocol'], report['execution_order'], args.summarize.parent))
        return
    if output.exists():
        raise ValueError('Report exists; preserve it and use a separate output directory')
    protocol_path = output.with_suffix('.protocol.json')
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        expected = dict(pyramid=str(args.baseline_build.resolve()), sparse_orientation=str(args.build.resolve()))
        if args.pilot != protocol['pilot'] or expected != protocol['builds']:
            raise ValueError('Pilot/build options differ from the frozen protocol')
        if sparse_settings(args) != protocol['sparse_settings'] or args.timeout != protocol['timeout_seconds']:
            raise ValueError('Sparse/timeout settings differ from the frozen protocol')
        if args.cases and set(args.cases.split(',')) != {case['name'] for case in protocol['cases']}:
            raise ValueError('Cases differ from the frozen protocol')
        if args.skip_cpu and any(0 in case['backends'] for case in protocol['cases']):
            raise ValueError('CPU option differs from the frozen protocol')
    else:
        protocol = prepare(args, protocol_path)
    if args.prepare_only:
        print(json.dumps(dict(protocol=str(protocol_path), cases=[case['name'] for case in protocol['cases']],
                              sparse_settings=protocol['sparse_settings'])))
        return
    if not unchanged(protocol):
        raise ValueError('Prepared sources/binaries/inputs changed; create a new protocol after freezing')
    report = dict(version=1, protocol=protocol, protocol_sha256=sha(protocol_path), execution_order=[], complete=False)
    save(output, report)
    for case in protocol['cases']:
        for backend in case['backends']:
            for block in range(protocol['blocks']):
                for method in METHODS if block%2 == 0 else METHODS[::-1]:
                    report['execution_order'].append(run_worker(protocol, case, backend, method, block, output))
                    save(output, report)
    report['complete'] = True
    report['unchanged'] = unchanged(protocol)
    report['summary'] = summarize(protocol, report['execution_order'], output.parent)
    save(output, report)
    # Performance and accuracy failures remain evidence, never silent retries.
    if not report['unchanged'] or not report['summary']['all_workers_succeeded']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
