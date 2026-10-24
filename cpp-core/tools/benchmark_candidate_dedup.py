"""Stage 43: opt-in, SEMANTICS-CHANGING coarse-candidate dedup before fine search.

Options::candidateDedupTopK (default 0) drops a coarse candidate when at least
top_k higher-ranked candidates of the same scale model have centers within
radius*(1-maxOverlap)*min(template side) source pixels and search angles within
angle_steps top-level coarse steps. Outputs are therefore NOT expected to equal
the baseline; every arm is compared with a same-build top_k=0 baseline, and
ground truth and output identity are separate checks.

Phases write one manifest each; every worker is a fresh runner process:
  gt       every GT case x applicable path x arm: first call + one repeat
  pilot    one formal block; labelled screening, never decision evidence
  formal   timing cases x {hybrid, full_gpu} x {off, k3}, alternating blocks,
           1 s warm-up then measured calls (algorithm total_ms)
  profile  full-GPU encoder timestamps and fine-work counters, never timing
Truth stays in this parent. Failed gates and workers are retained.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor

import benchmark_gpu_ncc_closure as stage40
import benchmark_new_frame_cost as stage41
import candidate_dedup_scenes

previous = stage40.previous
ROOT = stage40.ROOT
DOCS = ROOT/'cpp-core/docs'
DATA_ROOT = Path('/Users/Ayueh/Fastest_Image_Pattern_Matching')
PATHS = {'hybrid': 'hybrid_best', 'full_gpu': 'full_gpu_simd'}
ARMS = {
    'off': dict(top_k=0, radius=.5, angle_steps=3.),
    'k3': dict(top_k=3, radius=.5, angle_steps=3.),
    'k2': dict(top_k=2, radius=.5, angle_steps=3.),
    'k1': dict(top_k=1, radius=.5, angle_steps=3.),
    # Negative-control ablations: the angle window ignored, or a radius above
    # the separation of objects that can both be output.
    'k1_position_only': dict(top_k=1, radius=.5, angle_steps=1000.),
    'k1_wide_radius': dict(top_k=1, radius=2., angle_steps=3.),
}
PRIMARY = 'k3'
ABLATIONS = {'k1_position_only': 'sym_marker12', 'k1_wide_radius': 'adjacent_grid16'}
FORMAL_CASES = ('many128_pose', 'many512_pose', 'controlled6', '8k_t256_angle180_3scales')
PRIMARY_TIMING_CASES = ('many128_pose', 'many512_pose')
FORMAL_ARMS = [(path, arm) for path in PATHS for arm in ('off', PRIMARY)]
PROFILE_CASES = ('many128_pose', 'many512_pose')
R40_NAMES = ('controlled6', 'heldout12', 'small_fullres', '8k_t256_angle180_3scales', 'many128_pose',
             'many512_pose', 'boundary_one', 'empty_scene', 'constant_source')
RESIDENCY = {'command_buffers': 1, 'cpu_waits': 1, 'intermediate_readbacks': 0, 'cpu_algorithm_steps': 0}
POSE = ('pos_x', 'pos_y', 'angle', 'score', 'scale', 'corners')
TIME_GATE = (2., .10)          # ms, fraction
RSS_GATE = (8*1024*1024, .10)  # bytes, fraction
SPEED_TARGET = -10.            # percent, primary timing cases
sha, save, distribution, evaluate = stage40.sha, stage40.save, stage40.distribution, stage40.evaluate


def truth_list(metadata):
    return json.loads(Path(metadata).read_text())['truth']


def r40_cases(data_root):
    manifest = data_root/'build/stage37-full-gpu/formal/manifest.json'
    cases = stage40.fixture_cases(manifest, list(R40_NAMES))
    names = set(R40_NAMES)
    for fixtures in ('holdout-fixtures.json', 'stop0-fixtures.json'):
        archived = json.loads((DOCS/'stage40-ncc-closure'/fixtures).read_text())
        if archived.get('complete') is not True:
            raise ValueError('Incomplete round-40 fixtures: '+fixtures)
        for case in archived['cases']:
            if case['name'] not in names:
                cases.append(copy.deepcopy(case)); names.add(case['name'])
    for case in cases:
        case.update(family='r40', role='negative' if not case['truth'] else 'pass')
    return cases


def listed_cases(path, family, roles, skip=()):
    cases = []
    for case in json.loads(Path(path).read_text()):
        if case['name'] in skip:
            continue
        # Older indexes embed the truth list; newer ones store a count and keep
        # the poses in the scene metadata.
        truth = case['truth'] if isinstance(case['truth'], list) else truth_list(case['metadata'])
        if not isinstance(case['truth'], list) and len(truth) != case['truth']:
            raise ValueError('Truth count differs from case index: '+case['name'])
        cases.append(dict(case, truth=truth, family=family,
                          role=roles.get(case.get('kind'), roles.get(case['name'], 'pass'))))
    return cases


def r34_cases(data_root):
    roles = dict(occlusion='known_failure', distractors='hard_negative', contact='adjacent')
    cases = listed_cases(data_root/'build/stage34-challenge-inputs/cases.json', 'r34_challenge', roles)
    for case in listed_cases(data_root/'build/stage34-blender/inputs/cases.json', 'r34_blender', roles):
        cases.append(case)
    return cases


def r33_cases(data_root):
    roles = dict(new_small12='known_failure', degraded12='known_failure', empty_scene='negative')
    cases = listed_cases(data_root/'build/resident-multiscale-inputs/cases.json', 'r33', roles,
                         skip=('controlled6', 'many128_pose', 'many512_pose'))
    for case in cases:
        if case['name'] == 'empty_scene':
            case['name'] = 'r33_empty_scene'
    return cases


def scale_cases(data_root, r33):
    import benchmark_scale_support as stage34
    manifest = json.loads((data_root/'build/stage34-scale/study/inputs/manifest.json').read_text())
    old = [dict(case, name=case['name']) for case in r33 if case['name'] == 'new_small12']
    cases = []
    for case in stage34.cases_for(manifest, old):
        case = copy.deepcopy(case)
        for key in ('source', 'template'):
            if not Path(case[key]).is_absolute():
                case[key] = str(data_root/case[key])
        base = dict(case, name='s34_'+case['name'], family='r34_scale', role='small_template')
        cases.append(base)
        cases.append(dict(base, name=base['name']+'+support', extra=dict(scale_support_refine=True), paths=('hybrid',)))
    return cases


def r35_cases(data_root):
    cases = []
    for case in json.loads((data_root/'build/stage35-real-rgb/protocol-inputs/cases.json').read_text()):
        if case['kind'] != 'semi_synthetic49':
            continue  # FSC/VisA have no pose ground truth
        if len(case['truth']) != 49:
            raise ValueError('Round-35 case without 49 poses: '+case['name'])
        tuned = dict(name='r35_'+case['name'], source=case['source'], template=case['template'],
                     native_search=copy.deepcopy(case['search']), truth=case['truth'], family='r35', role='pass')
        original = copy.deepcopy(tuned)
        original['native_search'].update(min_area=256, proposal_score=.5, limit=100)
        original.update(name=tuned['name']+'+original',
                        role='known_failure' if case['split'] == 'development' else 'pass')
        cases += [tuned, original]
    return cases


def control_cases(directory):
    return [dict(case, family='r43_control', role='control') for case in candidate_dedup_scenes.make_cases(directory)]


def all_cases(data_root, controls):
    r33 = r33_cases(data_root)
    cases = r40_cases(data_root)+r34_cases(data_root)+r33+scale_cases(data_root, r33)+r35_cases(data_root)+control_cases(controls)
    if len({case['name'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case names')
    for case in cases:
        case.setdefault('paths', tuple(PATHS))
        for pose in case['truth']:
            for key in ('pos_x', 'pos_y', 'angle', 'scale'):
                if not math.isfinite(float(pose[key])):
                    raise ValueError('Invalid truth in '+case['name'])
    return cases


def make_job(case, path, arm, repeats, warmup, profile=False):
    if 'native_search' in case:
        job = dict(source=case['source'], template=case['template'], **case['native_search'],
                   repeats=repeats, warmup_seconds=warmup)
        job.update(stage40.METHODS[PATHS[path]], full_gpu_profiling=profile)
    else:
        job = stage40.job(case, PATHS[path], repeats, warmup, profile)
    job.update(case.get('extra', {}))
    options = ARMS[arm]
    job.update(candidate_dedup_top_k=options['top_k'], candidate_dedup_radius=options['radius'],
               candidate_dedup_angle_steps=options['angle_steps'])
    return job


def poses(records):
    return [{key: pose[key] for key in POSE} for pose in records]


def record_check(record, path, arm, models):
    """Validate one call's route and dedup counters; return derived values."""
    full = path == 'full_gpu'
    if not previous.validate_record(record, full):
        raise ValueError('Invalid route or residency for '+path)
    dedup = record.get('candidate_dedup')
    if not isinstance(dedup, dict) or dedup.get('top_k') != ARMS[arm]['top_k']:
        raise ValueError('Missing or mismatched candidate_dedup.top_k')
    rows = dedup['models']
    enabled = ARMS[arm]['top_k'] > 0
    if full and not enabled:
        if dedup['available'] or rows or dedup['input_candidates'] is not None:
            raise ValueError('Disabled full-GPU dedup reports counters')
    else:
        if not dedup['available'] or len(rows) != models:
            raise ValueError('Dedup counters missing for a model')
        for row in rows:
            if not 0 <= row['kept_candidates'] <= row['input_candidates']:
                raise ValueError('Dedup kept more candidates than its input')
            if not enabled and (row['kept_candidates'] != row['input_candidates'] or row['filter_ms']):
                raise ValueError('Disabled dedup dropped candidates or ran')
            if full and row['filter_ms']:
                raise ValueError('Full-GPU dedup reports host filter time')
        if (dedup['input_candidates'] != sum(row['input_candidates'] for row in rows) or
                dedup['kept_candidates'] != sum(row['kept_candidates'] for row in rows)):
            raise ValueError('Dedup totals differ from model rows')
    if full:
        metrics = record['full_gpu']
        if any(metrics[key] != value for key, value in RESIDENCY.items()):
            raise ValueError('Full-GPU residency counters changed')
    return dict(input=dedup['input_candidates'], kept=dedup['kept_candidates'],
                fine_maps=None if full else sum(row['fine_maps'] for row in record['work']))


def worker_calls(worker):
    if not isinstance(worker.get('runs'), list) or 'first' not in worker:
        raise ValueError('Invalid worker output')
    return [worker['first'], *worker['runs']]


def run_worker(binary, root, label, job, timeout):
    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
    save(jp, job)
    state = stage41.host_state()
    child = subprocess.run([str(binary), str(jp)], capture_output=True, text=True, timeout=timeout)
    jp.with_suffix('.stdout.txt').write_text(child.stdout)
    jp.with_suffix('.stderr.txt').write_text(child.stderr)
    entry = dict(job=str(jp.relative_to(root)), job_sha256=sha(jp), returncode=child.returncode,
                 stdout=str(jp.with_suffix('.stdout.txt').relative_to(root)), host=state)
    if child.returncode == 0:
        save(rp, json.loads(child.stdout))
        entry.update(result=str(rp.relative_to(root)), result_sha256=sha(rp))
    else:
        entry['stderr_tail'] = child.stderr[-4000:]
    return entry


def load_worker(root, entry):
    for key in ('job', 'result'):
        if key not in entry:
            raise ValueError('Worker failed: '+entry['job'])
        if sha(stage40.previous.worker_path(root, entry[key])) != entry[key+'_sha256']:
            raise ValueError('Changed raw evidence: '+entry[key])
    return json.loads((root/entry['result']).read_text())


def gt_cell(case, path, arm, worker):
    calls = worker_calls(worker)
    derived = [record_check(record, path, arm, len(case.get('scales') or case.get('native_search', {}).get('scales')))
               for record in calls]
    assessments = [evaluate(record['matches'], case['truth']) for record in calls]
    strict = [dict(tp=a['strict']['tp'], fp=a['strict']['fp'], fn=a['strict']['fn']) for a in assessments]
    first = calls[0]
    return dict(strict=strict, localization_tp=[a['localization']['tp'] for a in assessments],
        max_errors=assessments[0]['strict']['max_errors'], exact_repeat=worker['exact_repeat'] is True,
        same_counts=all(s == strict[0] for s in strict), matches=len(first['matches']), proposals=len(first['proposals']),
        dedup_input=derived[0]['input'], dedup_kept=derived[0]['kept'], fine_maps=derived[0]['fine_maps'],
        kept_fraction=derived[0]['kept']/derived[0]['input'] if derived[0]['input'] else None,
        total_ms=[record['total_ms'] for record in calls], peak_rss_bytes=worker['final_memory']['peak_rss_bytes'],
        _matches=poses(first['matches']), _proposals=poses(first['proposals']))


def compare_cell(cell, baseline):
    left, right = baseline['strict'][0], cell['strict'][0]
    delta = {key: right[key]-left[key] for key in ('tp', 'fp', 'fn')}
    return dict(delta=delta, regression=delta['tp'] < 0 or delta['fp'] > 0,
                improvement=(delta['tp'] > 0 or delta['fp'] < 0) and not (delta['tp'] < 0 or delta['fp'] > 0),
                matches_identical=cell['_matches'] == baseline['_matches'],
                proposals_identical=cell['_proposals'] == baseline['_proposals'],
                match_count_delta=cell['matches']-baseline['matches'],
                proposal_count_delta=cell['proposals']-baseline['proposals'])


def summarize_gt(manifest, root):
    cases = {case['name']: case for case in manifest['cases']}
    cells, failures = {}, []
    for entry in manifest['executions']:
        key = (entry['case'], entry['path'], entry['arm'])
        if key in cells:
            raise ValueError('Duplicate GT worker: '+str(key))
        if entry['returncode']:
            failures.append(dict(gate='worker', case=key[0], path=key[1], arm=key[2], detail=entry.get('stderr_tail', '')[-400:]))
            continue
        cells[key] = gt_cell(cases[key[0]], key[1], key[2], load_worker(root, entry))
    expected = {(case['name'], path, arm) for case in manifest['cases'] for path in case['paths'] for arm in ARMS}
    if set(cells)|{(f['case'], f['path'], f['arm']) for f in failures if f['gate'] == 'worker'} != expected:
        raise ValueError('GT workers do not cover every case/path/arm exactly')
    rows, aggregates = [], {path: {arm: dict(cases=0, regressions=[], improvements=[], tp_delta=0, fp_delta=0, fn_delta=0,
        matches_identical=0, proposals_identical=0, input=0, kept=0, fine_maps_off=0, fine_maps=0) for arm in ARMS if arm != 'off'}
        for path in PATHS}
    for name, case in cases.items():
        row = dict(case=name, family=case['family'], role=case['role'], truth=len(case['truth']), paths={})
        for path in case['paths']:
            baseline = cells.get((name, path, 'off'))
            arms = {}
            for arm in ARMS:
                cell = cells.get((name, path, arm))
                if cell is None:
                    continue
                public = {k: v for k, v in cell.items() if not k.startswith('_')}
                if not cell['exact_repeat'] or not cell['same_counts']:
                    failures.append(dict(gate='repeat', case=name, path=path, arm=arm))
                if arm != 'off' and baseline is not None:
                    public['versus_off'] = comparison = compare_cell(cell, baseline)
                    total = aggregates[path][arm]
                    total['cases'] += 1
                    total['tp_delta'] += comparison['delta']['tp']; total['fp_delta'] += comparison['delta']['fp']
                    total['fn_delta'] += comparison['delta']['fn']
                    total['matches_identical'] += comparison['matches_identical']
                    total['proposals_identical'] += comparison['proposals_identical']
                    total['input'] += cell['dedup_input']; total['kept'] += cell['dedup_kept']
                    if path == 'hybrid':
                        total['fine_maps_off'] += baseline['fine_maps']; total['fine_maps'] += cell['fine_maps']
                    if comparison['regression']:
                        total['regressions'].append(dict(case=name, role=case['role'], **comparison['delta']))
                        failures.append(dict(gate='gt_regression', case=name, path=path, arm=arm, role=case['role'], **comparison['delta']))
                    if comparison['improvement']:
                        total['improvements'].append(dict(case=name, role=case['role'], **comparison['delta']))
                arms[arm] = public
            row['paths'][path] = arms
        rows.append(row)
    for path in aggregates:
        for total in aggregates[path].values():
            total['kept_fraction'] = total['kept']/total['input'] if total['input'] else None
            if path == 'hybrid':
                total['fine_maps_fraction'] = total['fine_maps']/total['fine_maps_off'] if total['fine_maps_off'] else None
            else:
                total.pop('fine_maps_off'); total.pop('fine_maps')
    decision = dict(primary=PRIMARY, gt_neutral={path: not aggregates[path][PRIMARY]['regressions'] for path in PATHS},
        rule='GT-neutral: no case on that path with fewer strict TP or more strict FP than its same-path off arm; '
             'known failures included. Output identity is reported separately and is not expected.',
        negative_controls={arm: {path: dict(case=case, demonstrated=any(r['case'] == case for r in aggregates[path][arm]['regressions']))
                                 for path in PATHS} for arm, case in ABLATIONS.items()})
    return dict(phase='gt', arms=ARMS, cases=rows, aggregates=aggregates, decision=decision, failures=failures)


def formal_worker(case, path, arm, worker, measured):
    calls = worker_calls(worker)
    if len(worker['runs']) != measured:
        raise ValueError('Formal worker lacks measured calls')
    derived = [record_check(record, path, arm, len(case['scales'])) for record in calls]
    strict = [evaluate(record['matches'], case['truth'])['strict'] for record in calls]
    return dict(total_ms=[record['total_ms'] for record in worker['runs']], first_ms=calls[0]['total_ms'],
        warmup_calls=len(worker['warmups_ms']), warmup_ms=sum(worker['warmups_ms']),
        gt_clean=all(s['fp'] == 0 and s['fn'] == 0 for s in strict), exact_repeat=worker['exact_repeat'] is True,
        peak_rss_bytes=worker['final_memory']['peak_rss_bytes'],
        gpu_bytes=calls[0]['full_gpu']['gpu_bytes'] if path == 'full_gpu' else None,
        dedup_kept=derived[0]['kept'], dedup_input=derived[0]['input'],
        stages={key: statistics.median(record[key] for record in worker['runs']) for key in ('proposal_ms', 'refine_ms', 'nms_ms')}
        if path == 'hybrid' else {key: statistics.median(record['full_gpu'][key] for record in worker['runs']) for key in ('wait_ms', 'encode_ms')},
        _matches=poses(calls[0]['matches']), _proposals=poses(calls[0]['proposals']))


def summarize_formal(manifest, root, gt_manifest=None, gt_root=None):
    cases = {case['name']: case for case in manifest['cases']}
    blocks, measured = manifest['blocks'], manifest['measured_calls']
    workers, failures = {}, []
    for entry in manifest['executions']:
        key = (entry['case'], entry['path'], entry['arm'], entry['block'])
        if key in workers:
            raise ValueError('Duplicate formal worker')
        for process in stage41.host_entry(entry['host']):
            failures.append(dict(gate='host', case=key[0], path=key[1], arm=key[2], block=key[3], process=process['command']))
        workers[key] = formal_worker(cases[key[0]], key[1], key[2], load_worker(root, entry), measured)
    expected = [(name, path, arm, block) for name in cases for block in range(blocks)
                for path, arm in (FORMAL_ARMS if block % 2 == 0 else FORMAL_ARMS[::-1])]
    if [(e['case'], e['path'], e['arm'], e['block']) for e in manifest['executions']] != expected:
        raise ValueError('Formal workers missing or out of the pre-registered order')
    reference = {}
    if gt_manifest is not None:
        gt = {(e['case'], e['path'], e['arm']): e for e in gt_manifest['executions']}
        for (name, path, arm) in {(k[0], k[1], k[2]) for k in workers}:
            first = worker_calls(load_worker(gt_root, gt[(name, path, arm)]))[0]
            reference[(name, path, arm)] = (poses(first['matches']), poses(first['proposals']))
    rows = []
    for name, case in cases.items():
        row = dict(case=name, paths={})
        for path in PATHS:
            arms = {}
            for arm in ('off', PRIMARY):
                group = [workers[(name, path, arm, block)] for block in range(blocks)]
                values = [value for worker in group for value in worker['total_ms']]
                for block, worker in enumerate(group):
                    if not worker['gt_clean']:
                        failures.append(dict(gate='gt', case=name, path=path, arm=arm, block=block))
                    if not worker['exact_repeat']:
                        failures.append(dict(gate='repeat', case=name, path=path, arm=arm, block=block))
                    # The runner warms up for at least 1 s of wall time; algorithm
                    # totals inside it can sum to slightly less and are reported.
                    if worker['warmup_calls'] < 1:
                        failures.append(dict(gate='warmup', case=name, path=path, arm=arm, block=block))
                    if (name, path, arm) in reference and (worker['_matches'], worker['_proposals']) != reference[(name, path, arm)]:
                        failures.append(dict(gate='gt_phase_identity', case=name, path=path, arm=arm, block=block))
                arms[arm] = dict(total_ms=distribution(values), block_medians=[statistics.median(w['total_ms']) for w in group],
                    peak_rss_bytes=[w['peak_rss_bytes'] for w in group], gpu_bytes=sorted({w['gpu_bytes'] for w in group}, key=str),
                    dedup_kept=sorted({w['dedup_kept'] for w in group}, key=str), dedup_input=sorted({w['dedup_input'] for w in group}, key=str),
                    stages={key: statistics.median(w['stages'][key] for w in group) for key in group[0]['stages']},
                    first_ms=distribution([w['first_ms'] for w in group]))
            off, on = arms['off'], arms[PRIMARY]
            paired = [dict(block=b, off_ms=o, k3_ms=k, delta_percent=100*(k/o-1))
                      for b, (o, k) in enumerate(zip(off['block_medians'], on['block_medians']))]
            pooled = 100*(on['total_ms']['median']/off['total_ms']['median']-1)
            signs = {math.copysign(1, p['delta_percent']) for p in paired}
            consistent = len(signs) == 1 and math.copysign(1, pooled) in signs
            comparison = dict(delta_percent=pooled, paired=paired, consistent_direction=consistent,
                              k3_faster_blocks=sum(p['delta_percent'] < 0 for p in paired), blocks=len(paired))
            if name in PRIMARY_TIMING_CASES and not (consistent and pooled <= SPEED_TARGET):
                failures.append(dict(gate='speed_target', case=name, path=path, delta_percent=pooled, consistent=consistent))
            for p in paired:
                if name not in PRIMARY_TIMING_CASES and p['k3_ms'] > p['off_ms']+max(TIME_GATE[0], TIME_GATE[1]*p['off_ms']):
                    failures.append(dict(gate='time_budget', case=name, path=path, block=p['block'], delta_percent=p['delta_percent']))
            for block, (o, k) in enumerate(zip(off['peak_rss_bytes'], on['peak_rss_bytes'])):
                if k > o+max(RSS_GATE[0], RSS_GATE[1]*o):
                    failures.append(dict(gate='rss', case=name, path=path, block=block, off=o, k3=k))
            row['paths'][path] = dict(arms=arms, k3_versus_off=comparison)
        rows.append(row)
    return dict(phase=manifest['phase'], cases=rows, failures=failures,
        gates=dict(speed_target=f'{SPEED_TARGET:g}% pooled median with 6/6 block direction on '+','.join(PRIMARY_TIMING_CASES),
                   time_budget=f'each block k3 <= off + max({TIME_GATE[0]:g} ms, {100*TIME_GATE[1]:g}%) on other cases',
                   rss=f'each block k3 <= off + max(8 MiB, {100*RSS_GATE[1]:g}%)'))


def summarize_profile(manifest, root):
    cases = {case['name']: case for case in manifest['cases']}
    rows = []
    for entry in manifest['executions']:
        worker = load_worker(root, entry)
        case = cases[entry['case']]
        calls = worker_calls(worker)
        record_check(calls[-1], 'full_gpu', entry['arm'], len(case['scales']))
        profile = stage40.profile_record(calls[-1], True)
        work = stage40.fine_work_record(calls[-1], True)
        rows.append(dict(case=entry['case'], arm=entry['arm'], profile=profile,
            reached_candidates={f'{m}/{l}': row['reached_candidates'] for (m, l), row in sorted(work.items())},
            angles=sum(row['angles'] for row in work.values()),
            dedup_kept=calls[-1]['candidate_dedup']['kept_candidates'], dedup_input=calls[-1]['candidate_dedup']['input_candidates'],
            gt_clean=all(evaluate(r['matches'], case['truth'])['strict']['fp'] == 0 and
                         evaluate(r['matches'], case['truth'])['strict']['fn'] == 0 for r in calls)))
    return dict(phase='profile', rows=rows, note='Encoder timestamps add overhead; never compared with formal timings.')


def execute(args, phase, cases, plan, binary, root):
    before = previous.fingerprints(binary, cases)
    host = stage41.host_state(samples=5)
    if phase == 'formal' and host['competing_processes']:
        raise RuntimeError('Competing build or benchmark processes: '+json.dumps(host['competing_processes']))
    manifest = dict(version=1, stage=43, phase=phase, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(args.protocol), arms=ARMS, paths=PATHS, cases=cases, blocks=args.blocks, measured_calls=args.measured,
        timeout_seconds=args.timeout, host=dict(before=host), executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        def one(item):
            label, case, path, arm, block, job = item
            return dict(case=case['name'], path=path, arm=arm, block=block, **run_worker(binary, root, label, job, args.timeout))
        if phase == 'gt' and args.jobs > 1:
            with ThreadPoolExecutor(args.jobs) as pool:
                for result in pool.map(one, plan):
                    manifest['executions'].append(result); save(args.output, manifest)
                    print('Ran '+result['job'], result['returncode'], flush=True)
        else:
            for item in plan:
                result = one(item)
                if phase != 'gt' and result['returncode']:
                    manifest['executions'].append(result); save(args.output, manifest)
                    raise RuntimeError(item[0]+': '+result.get('stderr_tail', ''))
                manifest['executions'].append(result); save(args.output, manifest)
                print('Ran '+result['job'], result['returncode'], flush=True)
        manifest['host']['after'] = stage41.host_state(samples=5)
        after = previous.fingerprints(binary, cases)
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), after_sha256=after, unchanged=before == after)
        save(args.output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(args.output, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('gt', 'pilot', 'formal', 'profile'))
    parser.add_argument('--build', type=Path, default=ROOT/'build/stage43-release')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT, help='Checkout holding the frozen per-round build/ fixtures')
    parser.add_argument('--protocol', type=Path, default=DOCS/'stage43-candidate-dedup/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--measured', type=int, default=3)
    parser.add_argument('--jobs', type=int, default=1, help='Parallel GT workers (never for timing phases)')
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--gt-manifest', type=Path, help='With --summarize of a formal manifest: GT-phase outputs to compare')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        root = args.summarize.resolve().parent
        if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
            raise ValueError('Summaries require a complete manifest with unchanged sources, binaries and inputs')
        if manifest['phase'] == 'gt':
            summary = summarize_gt(manifest, root)
        elif manifest['phase'] == 'profile':
            summary = summarize_profile(manifest, root)
        else:
            gt = json.loads(args.gt_manifest.read_text()) if args.gt_manifest else None
            summary = summarize_formal(manifest, root, gt, args.gt_manifest.resolve().parent if args.gt_manifest else None)
        save(args.output, dict(summary, manifest_sha256=sha(args.summarize)))
        return
    if not args.phase:
        parser.error('--phase or --summarize is required')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    if args.phase != 'gt' and args.jobs != 1:
        parser.error('Timing and profile phases run one worker at a time')
    if args.phase == 'formal' and (args.blocks < 3 or args.measured < 2):
        parser.error('Formal requires at least 3 blocks and 2 measured calls')
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    if args.phase == 'gt':
        cases = all_cases(args.data_root, root/'inputs')
        plan = [(f"{case['name']}-{path}-{arm}", case, path, arm, 0, make_job(case, path, arm, 1, 0.))
                for case in cases for path in case['paths'] for arm in ARMS]
    elif args.phase in ('pilot', 'formal'):
        cases = [case for case in r40_cases(args.data_root) if case['name'] in FORMAL_CASES]
        cases.sort(key=lambda case: FORMAL_CASES.index(case['name']))
        blocks = 1 if args.phase == 'pilot' else args.blocks
        args.blocks = blocks
        plan = [(f"{case['name']}-{path}-{arm}-b{block}", case, path, arm, block, make_job(case, path, arm, args.measured, 1.))
                for case in cases for block in range(blocks) for path, arm in (FORMAL_ARMS if block % 2 == 0 else FORMAL_ARMS[::-1])]
    else:
        cases = [case for case in r40_cases(args.data_root) if case['name'] in PROFILE_CASES]
        args.blocks = 1
        plan = [(f"{case['name']}-full_gpu-{arm}-profile", case, 'full_gpu', arm, 0, make_job(case, 'full_gpu', arm, args.measured, 0., True))
                for case in cases for arm in ('off', PRIMARY)]
    execute(args, args.phase, cases, plan, binary, root)


if __name__ == '__main__':
    main()
