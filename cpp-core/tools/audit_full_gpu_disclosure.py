"""Recompute the 2026-10-19 full-GPU disclosure numbers from archived results.

Nothing is re-measured: the committed stage37-40 archives are read in memory.

* Pose context: per round, phase and case, the hybrid formal median, its
  first-call pose_create_ms in each process and their median (every formal
  call must report zero), the hybrid median with that cost added back, and
  each full-GPU method against both. The sum estimates a new frame per call;
  it is not a measurement of one.
* Mode-4 routes: SIMD32 and fallback angle evaluations (49 windows each) in
  the archived stage-40 profile summaries, from the shader's route rule
  (cached ROI and uint32-safe four-lane stripes) and the level sizes.
* 8K workload: the archived case definition and fixture, checking that every
  pixel outside the six truth boxes is one constant and each box is the template.
"""
import argparse
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'cpp-core/docs'
HYBRID = 'hybrid_best'
ROUNDS = {37: ('stage37-full-gpu', ('formal',)),
          38: ('stage38-gpu-peaks', ('formal',)),
          39: ('stage39-fine-tiles', ('formal', 'stop0-formal')),
          40: ('stage40-ncc-closure', ('formal', 'stop0-formal', 'holdout'))}
PROFILES = ('profile-summary.json', 'stop0-profile-summary.json')
CASE_8K = '8k_t256_angle180_3scales'


def workers(archive, phase):
    """(case, method, worker) for every archived worker result of one phase."""
    prefix = phase+'/results/'
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name
            if member.isfile() and name.startswith(prefix) and name.endswith('.json') and '/' not in name[len(prefix):]:
                case, rest = name[len(prefix):-len('.json')].split('-', 1)
                method, _ = rest.rsplit('-', 1)
                yield case, method, json.load(tar.extractfile(member))


def median_ms(group):
    return statistics.median(run['total_ms'] for worker in group for run in worker['runs'])


def pose_context(archive, phase):
    groups = {}
    for case, method, worker in workers(archive, phase):
        groups.setdefault(case, {}).setdefault(method, []).append(worker)
    rows = []
    for case, methods in sorted(groups.items()):
        if HYBRID not in methods:
            continue
        hybrid = methods[HYBRID]
        formal = [run['pose_create_ms'] for worker in hybrid for run in worker['runs']]
        if any(formal):
            raise ValueError(case+': a formal hybrid call rebuilt its pose context')
        first = [worker['first']['pose_create_ms'] for worker in hybrid]
        base, added = median_ms(hybrid), statistics.median(first)
        rows.append(dict(case=case, hybrid_median_ms=base, first_call_pose_create_ms=first,
            first_call_pose_create_median_ms=added, hybrid_with_pose_create_ms=base+added,
            methods={method: dict(median_ms=median_ms(group), vs_hybrid_percent=100*(median_ms(group)/base-1),
                                  vs_hybrid_with_pose_create_percent=100*(median_ms(group)/(base+added)-1))
                     for method, group in sorted(methods.items()) if method != HYBRID}))
    return rows


def simd32_route(width, height, cache):
    return (width+6)*(height+6) <= cache and (width*height+3)//4 <= 66051


def mode4_routes(summary, method='full_gpu_simd'):
    rows = []
    for case in summary['cases']:
        row = case['methods'].get(method) or {}
        work = row.get('fine_work') or {}
        if not work.get('layers'):
            continue
        if row.get('effective_fine_modes') != [4]:
            raise ValueError(case['name']+': profile calls did not run effective mode 4')
        simd = fallback = 0
        for layer in work['layers']:
            angles = layer['counts']['angles']
            if angles['min'] != angles['max']:
                raise ValueError(case['name']+': angle counts changed between profile calls')
            if simd32_route(layer['template_width'], layer['template_height'], layer['roi_cache_bytes']):
                simd += angles['median']
            else:
                fallback += angles['median']
        total = simd+fallback
        rows.append(dict(case=case['name'], simd32_angles=simd, fallback_angles=fallback,
                         simd32_windows=49*simd, fallback_windows=49*fallback,
                         simd32_share=simd/total if total else None))
    return rows


def workload_8k(raw_archive, fixture_archive):
    import cv2
    import numpy as np
    with tarfile.open(raw_archive) as tar:
        manifest = json.load(tar.extractfile('formal/manifest.json'))
    case = next(row for row in manifest['cases'] if row['name'] == CASE_8K)
    images = {}
    with tarfile.open(fixture_archive) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith((Path(case['source']).name, Path(case['template']).name)):
                data = np.frombuffer(tar.extractfile(member).read(), np.uint8)
                images[Path(member.name).name] = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    source, template = images[Path(case['source']).name], images[Path(case['template']).name]
    height, width = template.shape
    outside = np.ones(source.shape, bool)
    for truth in case['truth']:
        left, top = round(truth['pos_x']-width/2), round(truth['pos_y']-height/2)
        if not np.array_equal(source[top:top+height, left:left+width], template):
            raise ValueError('8K truth box differs from the template')
        outside[top:top+height, left:left+width] = False
    background = np.unique(source[outside])
    return dict(case=CASE_8K, source_shape=list(source.shape), template_shape=list(template.shape),
                scales=case['scales'], angle_tolerance=case['angle'], workflow=case['workflow'],
                workload=case.get('workload'), truth=case['truth'],
                truth_angles=sorted({row['angle'] for row in case['truth']}),
                truth_scales=sorted({row['scale'] for row in case['truth']}),
                background_values=[int(value) for value in background], truth_boxes_equal_template=True)


def audit(docs=DOCS):
    result = dict(note='Archived values only; nothing re-measured. Added-back sums are estimates.', pose_context={})
    for round_number, (directory, phases) in ROUNDS.items():
        archive = docs/directory/'raw-results.tar.gz'
        result['pose_context'][str(round_number)] = {phase: pose_context(archive, phase) for phase in phases}
    stage40 = docs/ROUNDS[40][0]
    result['mode4_routes'] = {name: mode4_routes(json.loads((stage40/name).read_text())) for name in PROFILES}
    result['workload_8k'] = workload_8k(stage40/'raw-results.tar.gz', stage40/'fixtures.tar.gz')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', type=Path, default=DOCS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(args.docs), indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
