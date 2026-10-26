"""Stage 49: choose the coarse top level from template information (small-template recall).

Round 35 found that a 43x30 stamp template reduced to 11x8 lost 45 of 49 rotated
copies at the original min_area=256, while min_area=512 (with a wider candidate
budget) found all of them but quadrupled the time, and a 58x71 heldout template
that was already 49/49 only got slower. This round tests the opt-in
research_pipeline Settings::pyramidRule: for each template the runner predicts
the true-pose NCC at every candidate top level and starts at the deepest level
whose prediction reaches pyramid_rule_min_score, never deeper than min_area 256.

Phases (each worker is a fresh fpm_color_verify_runner process, geometry_only):
  dev-sweep      development positives at every forced top level, plus one rule
                 report per template (predicted scores; threshold -1 keeps depth)
  dev-calibrate  development positives with the tuned rule, the original and the
                 equal-time budget grid (timing; own lock)
  formal         holdout positives and negatives, five frozen arms (own lock)
Offline: --tune (threshold from dev-sweep), --calibrate (budget recipe from
dev-calibrate), --summarize (formal summary and pre-registered verdicts).
Truth stays in the parent; jobs never contain it. Every worker, failed gate and
lock wait is kept. Time, RSS, named GPU bytes and GT are reported separately.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

from benchmark_frame_overheads import host_state
from many_object_accuracy import evaluate
from stage49_scenes import save, sha

ROOT = Path(__file__).resolve().parents[2]
NOTES = Path.home()/'Library/Application Support/Claude/side-session-notes'
# Round-35 geometry search (stage35_scenes.search_settings with scales [1]).
BASE_SEARCH = dict(backend=1, refinement_backend=1, scales=[1], angle=180, score=.8, overlap=.2, stop_layer=1,
                   subpixel=False, cpu_threads=8, pose_resident_mode=0, pose_storage_mode=1, pose_input_mode=1,
                   ncc_mode=2, fine_ncc_mode=2, fine_fused=True, fine_angle_reuse=True)
ORIGINAL = dict(min_area=256, proposal_score=.5, limit=100)
STAGE35_MANUAL = dict(min_area=512, proposal_score=.25, limit=300)
FIXED512 = dict(min_area=512, proposal_score=.5, limit=100)
BUDGET_GRID = ((.45, 150), (.40, 200), (.35, 200), (.30, 300), (.25, 300), (.20, 400))
TAU_GRID = tuple(round(.30+.05*k, 2) for k in range(11))
NEEDED_TOLERANCE = .02  # needed depth: deepest level within 2% of the template's objects of its best depth
PHASES = dict(dev_sweep=dict(warmups=1, repeats=3, blocks=1, lock=False),
              dev_calibrate=dict(warmups=1, repeats=3, blocks=1, lock=True),
              formal=dict(warmups=1, repeats=3, blocks=3, negative_blocks=1, lock=True))
LOCK_POLL_SECONDS = 30
# 1-minute load on this 12-core host (8P+4E). Round 48 used 4, but during this
# round a system ANECompilerService held ~90% of one core for days, so the
# idle load was already ~4.1-4.4; 6 still leaves half the cores free.
HOST_LOAD_LIMIT = 6.


def depth(width, height, min_area):
    """fpm::Pattern's top level: area/4 per level until area <= floor(sqrt(min_area))^2, or 1x1."""
    side, area, levels = math.isqrt(min_area), width*height, 0
    while area > side*side and (width, height) != (1, 1):
        width, height, area, levels = (width+1)//2, (height+1)//2, area//4, levels+1
    return levels


def min_area_for_top(width, height, top):
    """Mirror of research::minAreaForTop: ceil(sqrt(area at `top`))^2."""
    area = width*height
    for _ in range(top):
        area //= 4
    side = math.isqrt(max(area, 1))
    return max(side*side if side*side >= area else (side+1)**2, 1)


def arm_search(name, protocol=None):
    if name == 'original':
        return dict(ORIGINAL)
    if name == 'stage35_manual':
        return dict(STAGE35_MANUAL)
    if name == 'fixed512':
        return dict(FIXED512)
    if name == 'adaptive':
        return dict(ORIGINAL, pyramid_rule=1, pyramid_rule_min_score=protocol['frozen']['pyramid_rule_min_score'])
    if name == 'budget':
        recipe = protocol['frozen']['budget']
        return dict(ORIGINAL, proposal_score=recipe['proposal_score'], limit=recipe['limit'])
    raise ValueError('Unknown arm '+name)


FORMAL_ARMS = ('original', 'stage35_manual', 'adaptive', 'budget', 'fixed512')


def job(case, search, warmups, repeats):
    return dict(source=case['source'], template=case['template'], method='geometry_only', backend='cpu', threads=1,
                warmups=warmups, repeats=repeats, search=dict(BASE_SEARCH, **search))


def phase_plan(phase, cases, tuning=None, protocol=None):
    """(case, arm, search, blocks) rows in a fixed order."""
    positives = [case for case in cases if case['positive']]
    plan = []
    if phase == 'dev_sweep':
        reported = set()
        for case in positives:
            w, h = case['template_size']
            top = depth(w, h, 256)
            for level in range(top+1):
                area = 256 if level == top else min_area_for_top(w, h, level)
                plan.append((case, f'depth{level}', dict(ORIGINAL, min_area=area), 1))
            if case['template_name'] not in reported:
                reported.add(case['template_name'])
                plan.append((case, 'rule_report', dict(ORIGINAL, pyramid_rule=1, pyramid_rule_min_score=-1), 1))
    elif phase == 'dev_calibrate':
        for case in positives:
            plan.append((case, 'original', dict(ORIGINAL), 1))
            plan.append((case, 'adaptive', dict(ORIGINAL, pyramid_rule=1, pyramid_rule_min_score=tuning['selected_min_score']), 1))
            for proposal, limit in BUDGET_GRID:
                plan.append((case, f'budget_{proposal:.2f}_{limit}', dict(ORIGINAL, proposal_score=proposal, limit=limit), 1))
    elif phase == 'formal':
        spec = PHASES['formal']
        for case in cases:
            for arm in FORMAL_ARMS:
                plan.append((case, arm, arm_search(arm, protocol), spec['blocks'] if case['positive'] else spec['negative_blocks']))
    else:
        raise ValueError('Unknown phase '+phase)
    return plan


def foreign_locks(own):
    return sorted(str(path) for path in NOTES.glob('*/timing-lock-*.txt') if own is None or path.resolve() != own.resolve())


def wait_for_locks(own, sleep=time.sleep):
    """Wait while another session holds a timing lock; returns the waits observed."""
    waits = []
    while True:
        locks = foreign_locks(own)
        if not locks:
            return waits
        waits.append(dict(utc=datetime.now(timezone.utc).isoformat(), locks=locks))
        print('Waiting for timing lock(s): '+', '.join(locks), flush=True)
        sleep(LOCK_POLL_SECONDS)


def fingerprints(binary, cases):
    sources = [path for folder in ('src', 'include', 'experiments', 'tools', 'tests')
               for path in (ROOT/'cpp-core'/folder).rglob('*') if path.is_file() and
               (path.suffix in ('.cpp', '.hpp', '.h', '.mm', '.metal', '.py', '.in') or path.name == 'CMakeLists.txt')]
    sources += [ROOT/'cpp-core/CMakeLists.txt']
    binaries = [binary, *binary.parent.glob('*.dylib')]
    return dict(source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sorted(set(sources))},
                binary_sha256={str(path): sha(path) for path in sorted(set(binaries))},
                input_sha256={case[key]: sha(case[key]) for case in cases for key in ('source', 'template')})


def run_phase(phase, cases_path, build, output, tuning_path=None, protocol_path=None, lock_dir=None, timeout=600.):
    if output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    root = output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    cases = json.loads(cases_path.read_text())
    for case in cases:
        for key, digest in case['input_sha256'].items():
            if sha(case[key]) != digest:
                raise ValueError(f"Input changed: {case['name']} {key}")
    tuning = json.loads(tuning_path.read_text()) if tuning_path else None
    protocol = json.loads(protocol_path.read_text()) if protocol_path else None
    if phase == 'dev_calibrate' and tuning is None:
        raise ValueError('dev_calibrate needs --tuning')
    if phase == 'formal' and (protocol is None or 'frozen' not in protocol):
        raise ValueError('formal needs the frozen --protocol')
    if phase != 'formal' and any(case['split'] != 'development' for case in cases):
        raise ValueError('Development phases accept development cases only')
    if phase == 'formal' and any(case['split'] != 'holdout' for case in cases):
        raise ValueError('The formal phase accepts holdout cases only')
    spec = PHASES[phase]
    plan = phase_plan(phase, cases, tuning, protocol)
    binary = (build/'experiments/fpm_color_verify_runner').resolve()
    before = fingerprints(binary, cases)
    own = None
    if spec['lock']:
        if lock_dir is None:
            raise ValueError('Timed phases need --lock-dir')
        own = lock_dir/f'timing-lock-round49-{phase}.txt'
    waits = wait_for_locks(own)
    host = host_state(samples=5)
    # Timed phases refuse a busy host; the recall sweep only records it (its times only break ties).
    if spec['lock'] and (host['competing_processes'] or host['load_average'][0] > HOST_LOAD_LIMIT):
        raise RuntimeError('Host not quiet: '+json.dumps(dict(load=host['load_average'], competing=host['competing_processes'])))
    manifest = dict(version=1, phase=phase, platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
                    spec=spec, cases_path=str(cases_path.resolve()), cases_sha256=sha(cases_path), binary=str(binary),
                    tuning=tuning, tuning_sha256=sha(tuning_path) if tuning_path else None,
                    protocol_path=str(protocol_path.resolve()) if protocol_path else None,
                    protocol_sha256=sha(protocol_path) if protocol_path else None,
                    base_search=BASE_SEARCH, timeout_seconds=timeout, host=dict(before=host, initial_lock_waits=waits),
                    executions=[], complete=False, **before)
    save(output, manifest)
    if own:
        own.write_text(f"round 49 (adaptive pyramid) {phase}\nstart_utc {manifest['started_utc']}\n"
                       f"workers {sum(rows for _, _, _, rows in plan)}\nexpected well under 2 h; deleted when done\n")
    try:
        blocks = max(rows for _, _, _, rows in plan)
        for block in range(blocks):
            rows = [row for row in plan if block < row[3]]
            for case, arm, search, _ in rows[::1 if block % 2 == 0 else -1]:
                label = f"{case['name']}-{arm}-b{block}"
                jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                save(jp, job(case, search, spec['warmups'], spec['repeats']))
                lock_waits = wait_for_locks(own)
                state = host_state()
                print('Running '+label, flush=True)
                child = subprocess.run([str(binary), str(jp)], capture_output=True, text=True, timeout=timeout)
                jp.with_suffix('.stdout.txt').write_text(child.stdout)
                jp.with_suffix('.stderr.txt').write_text(child.stderr)
                entry = dict(case=case['name'], arm=arm, block=block, job=str(jp.relative_to(root)), job_sha256=sha(jp),
                             returncode=child.returncode, host=state, lock_waits=lock_waits)
                if child.stdout.strip():
                    save(rp, json.loads(child.stdout))
                    entry.update(result=str(rp.relative_to(root)), result_sha256=sha(rp))
                elif child.returncode:
                    raise RuntimeError(label+': '+child.stderr[-4000:])
                manifest['executions'].append(entry)
                save(output, manifest)
        after = fingerprints(binary, cases)
        manifest['host']['after'] = host_state(samples=5)
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), after_sha256=after,
                        unchanged=before == {key: after[key] for key in before})
        save(output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(output, manifest)
        raise
    finally:
        if own and own.exists():
            own.unlink()
    return manifest


def load_rows(manifest, root):
    """One record per execution with GT, times and memory; truth comes from the cases file."""
    cases = {case['name']: case for case in json.loads(Path(manifest['cases_path']).read_text())}
    if sha(manifest['cases_path']) != manifest['cases_sha256']:
        raise ValueError('Cases file changed after the run')
    rows = []
    for entry in manifest['executions']:
        if sha(root/entry['job']) != entry['job_sha256'] or ('result' in entry and sha(root/entry['result']) != entry['result_sha256']):
            raise ValueError('Altered worker '+entry['job'])
        result = json.loads((root/entry['result']).read_text())
        case = cases[entry['case']]
        first, runs = result['first'], result['runs']
        strict = evaluate(first['search_matches'], case['truth'])['strict']
        per_run = [evaluate(run['search_matches'], case['truth'])['strict'] for run in runs]
        diagnostics = first['search_diagnostics']
        rows.append(dict(case=entry['case'], template=case['template_name'], condition=case['condition'],
                         positive=case['positive'], expected=case['expected_count'], arm=entry['arm'], block=entry['block'],
                         returncode=entry['returncode'], exact_repeat=result['exact_repeat'],
                         tp=strict['tp'], fp=strict['fp'], fn=strict['fn'],
                         runs_same_gt=all((r['tp'], r['fp'], r['fn']) == (strict['tp'], strict['fp'], strict['fn']) for r in per_run),
                         max_errors=strict['max_errors'], matches=first['search_matches'],
                         search_ms=statistics.median(run['search_ms'] for run in runs), first_ms=first['search_ms'],
                         construct_ms=result['search_construct_ms'], decode_ms=result['decode_ms'],
                         peak_rss_bytes=result['final_memory']['peak_rss_bytes'],
                         named_gpu_bytes=dict(frame_coarse=diagnostics['frame_coarse_gpu_bytes'],
                                              frame_fine=diagnostics['frame_fine_gpu_bytes'], pose=diagnostics['pose_gpu_bytes']),
                         pyramid_rule=result.get('pyramid_rule'),
                         host_busy=bool(entry['host']['competing_processes']), host_load=entry['host']['load_average'][0],
                         lock_waits=len(entry['lock_waits'])))
    return rows


def cell(rows):
    """Median over blocks of the per-process median; GT from block 0 and its cross-block identity."""
    rows = sorted(rows, key=lambda row: row['block'])
    head = rows[0]
    return dict(tp=head['tp'], fp=head['fp'], fn=head['fn'], expected=head['expected'], blocks=len(rows),
                search_ms=statistics.median(row['search_ms'] for row in rows),
                search_ms_blocks=[row['search_ms'] for row in rows],
                first_ms=statistics.median(row['first_ms'] for row in rows),
                construct_ms=statistics.median(row['construct_ms'] for row in rows),
                peak_rss_mib=statistics.median(row['peak_rss_bytes'] for row in rows)/2**20,
                named_gpu_mib={key: value/2**20 for key, value in head['named_gpu_bytes'].items()},
                matches_identical_across_blocks=all(row['matches'] == head['matches'] for row in rows),
                exact_repeat=all(row['exact_repeat'] and row['returncode'] == 0 and row['runs_same_gt'] for row in rows),
                host_busy_blocks=sum(row['host_busy'] or row['host_load'] > HOST_LOAD_LIMIT for row in rows),
                lock_waits=sum(row['lock_waits'] for row in rows),
                pyramid_rule=head['pyramid_rule'], max_errors=head['max_errors'], matches=head['matches'])


def cells(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['case'], row['arm']), []).append(row)
    return {key: cell(value) for key, value in grouped.items()}


def tune(manifest, root):
    """Threshold from the development depth sweep (pre-registered objective)."""
    rows = load_rows(manifest, root)
    table = cells(rows)
    templates = {}
    for (case, arm), value in table.items():
        name = next(row['template'] for row in rows if row['case'] == case)
        entry = templates.setdefault(name, dict(depths={}, scores=None, cases=set()))
        if arm == 'rule_report':
            model = value['pyramid_rule']['models'][0]
            entry.update(scores=model['predicted_scores'], original_top=model['original_top'])
            continue
        level = int(arm[len('depth'):])
        entry['cases'].add(case)
        slot = entry['depths'].setdefault(level, dict(tp=0, fp=0, expected=0, search_ms=0.))
        for key in ('tp', 'fp', 'expected', 'search_ms'):
            slot[key] += value[key]
    out = {}
    for name, entry in sorted(templates.items()):
        depths = entry['depths']
        best = max(slot['tp'] for slot in depths.values())
        expected = next(iter(depths.values()))['expected']
        tolerance = NEEDED_TOLERANCE*expected
        needed = max(level for level, slot in depths.items() if slot['tp'] >= best-tolerance)
        out[name] = dict(original_top=entry['original_top'], predicted_scores=entry['scores'], best_tp=best,
                         expected=expected, needed_top=needed, cases=sorted(entry['cases']),
                         depths={str(level): slot for level, slot in sorted(depths.items())})

    def decision(scores, threshold):
        for level in range(len(scores)-1, -1, -1):
            if scores[level] >= threshold:
                return level
        best = max(scores)
        return max(level for level, score in enumerate(scores) if score == best)

    candidates = []
    for threshold in TAU_GRID:
        chosen = {name: decision(entry['predicted_scores'], threshold) for name, entry in out.items()}
        misses = sorted(name for name, level in chosen.items() if level > out[name]['needed_top'])
        shallower = sorted(name for name, level in chosen.items() if level < out[name]['needed_top'])
        time_ms = sum(out[name]['depths'][str(level)]['search_ms'] for name, level in chosen.items())
        tp = sum(out[name]['depths'][str(level)]['tp'] for name, level in chosen.items())
        candidates.append(dict(min_score=threshold, misses=misses, shallower_than_needed=shallower,
                               dev_search_ms=time_ms, dev_tp=tp, selected_tops=chosen))
    best = min(candidates, key=lambda c: (len(c['misses']), c['dev_search_ms'], c['min_score']))
    return dict(objective='fewest templates started deeper than needed, then least summed development time, then smallest threshold',
                needed_rule=f'deepest top level whose summed strict TP is within {NEEDED_TOLERANCE:.0%} of the template objects of its best level',
                templates=out, candidates=candidates, selected_min_score=best['min_score'],
                expected_total=sum(entry['expected'] for entry in out.values()),
                oracle_tp=sum(entry['depths'][str(entry['needed_top'])]['tp'] for entry in out.values()),
                original_tp=sum(entry['depths'][str(entry['original_top'])]['tp'] for entry in out.values()))


def calibrate(manifest, root):
    """Equal-time budget recipe: the largest summed development time not above the adaptive arm's."""
    table = cells(load_rows(manifest, root))
    totals = {}
    for (case, arm), value in table.items():
        slot = totals.setdefault(arm, dict(search_ms=0., tp=0, fp=0, expected=0))
        for key in slot:
            slot[key] += value[key]
    budget = totals['adaptive']['search_ms']
    grid = [dict(name=f'budget_{p:.2f}_{l}', proposal_score=p, limit=l, **totals[f'budget_{p:.2f}_{l}']) for p, l in BUDGET_GRID]
    within = [row for row in grid if row['search_ms'] <= budget]
    chosen = max(within, key=lambda row: row['search_ms']) if within else min(grid, key=lambda row: row['search_ms'])
    return dict(rule='largest summed development search time not above the adaptive arm; cheapest if none',
                totals=totals, grid=grid, selected=dict(proposal_score=chosen['proposal_score'], limit=chosen['limit']),
                selected_within_budget=bool(within), adaptive_search_ms=budget)


def summarize(manifest, root, protocol):
    rows = load_rows(manifest, root)
    table = cells(rows)
    gates = protocol['gates']
    templates = sorted({row['template'] for row in rows})
    per_template = {}
    for name in templates:
        cases = sorted({row['case'] for row in rows if row['template'] == name})
        entry = dict(cases=cases)
        for arm in FORMAL_ARMS:
            pos = [table[(case, arm)] for case in cases if (case, arm) in table and table[(case, arm)]['expected']]
            neg = [table[(case, arm)] for case in cases if (case, arm) in table and not table[(case, arm)]['expected']]
            entry[arm] = dict(tp=sum(c['tp'] for c in pos), fp=sum(c['fp'] for c in pos), expected=sum(c['expected'] for c in pos),
                              search_ms=sum(c['search_ms'] for c in pos), negative_fp=sum(c['fp'] for c in neg),
                              negative_search_ms=sum(c['search_ms'] for c in neg))
        original = entry['original']
        entry['deficient'] = original['expected'] > 0 and original['tp'] < gates['deficient_recall_below']*original['expected']
        rule = next((table[(case, 'adaptive')]['pyramid_rule'] for case in cases if (case, 'adaptive') in table), None)
        model = rule['models'][0] if rule else None
        entry['rule'] = model
        entry['rule_kept_depth'] = bool(model) and model['selected_top'] == model['original_top']
        entry['adaptive_identical_to_original'] = all(
            table[(case, 'adaptive')]['matches'] == table[(case, 'original')]['matches'] for case in cases)
        per_template[name] = entry
    totals = {}
    for arm in FORMAL_ARMS:
        totals[arm] = {key: sum(per_template[name][arm][key] for name in templates)
                       for key in ('tp', 'fp', 'expected', 'search_ms', 'negative_fp', 'negative_search_ms')}
    deficient = [name for name in templates if per_template[name]['deficient']]
    sufficient = [name for name in templates if not per_template[name]['deficient'] and per_template[name]['original']['expected']]

    def total(names, arm, key):
        return sum(per_template[name][arm][key] for name in names)

    verdicts = {}
    if deficient:
        missed = total(deficient, 'original', 'expected')-total(deficient, 'original', 'tp')
        recovered = total(deficient, 'adaptive', 'tp')-total(deficient, 'original', 'tp')
        worse = [case for name in deficient for case in per_template[name]['cases']
                 if table[(case, 'original')]['expected'] and table[(case, 'adaptive')]['tp'] < table[(case, 'original')]['tp']-1]
        verdicts['H1'] = dict(passed=recovered >= gates['H1_recovered_fraction']*missed and not worse,
                              missed_by_original=missed, recovered=recovered, cases_worse_by_more_than_one=worse)
    else:
        verdicts['H1'] = dict(passed=None, reason='no deficient holdout template')
    kept = [name for name in sufficient if per_template[name]['rule_kept_depth']]
    identical = [name for name in kept if per_template[name]['adaptive_identical_to_original']]
    ratio = total(sufficient, 'adaptive', 'search_ms')/total(sufficient, 'original', 'search_ms') if sufficient else None
    verdicts['H2'] = dict(passed=bool(sufficient) and len(kept) >= gates['H2_kept_fraction']*len(sufficient)
                          and len(identical) == len(kept) and ratio <= gates['H2_time_ratio'],
                          sufficient=len(sufficient), kept=len(kept), kept_identical=len(identical),
                          changed=[name for name in sufficient if name not in kept], time_ratio=ratio)
    regression = gates['H2r_template']
    if regression in per_template:
        entry = per_template[regression]
        r = entry['adaptive']['search_ms']/entry['original']['search_ms']
        verdicts['H2r'] = dict(passed=entry['rule_kept_depth'] and entry['adaptive']['tp'] == entry['adaptive']['expected']
                               and r <= gates['H2_time_ratio'], kept_depth=entry['rule_kept_depth'],
                               tp=entry['adaptive']['tp'], expected=entry['adaptive']['expected'], time_ratio=r)
    else:
        verdicts['H2r'] = dict(passed=False, reason='regression template missing')
    a, b, s = totals['adaptive'], totals['budget'], totals['stage35_manual']
    verdicts['H3'] = dict(passed=a['tp'] >= b['tp'] and a['search_ms'] <= gates['H3_time_ratio']*b['search_ms'],
                          adaptive_tp=a['tp'], budget_tp=b['tp'], time_ratio=a['search_ms']/b['search_ms'])
    verdicts['H4'] = dict(passed=a['search_ms'] <= gates['H4_time_ratio']*s['search_ms']
                          and a['tp'] >= s['tp']-gates['H4_tp_tolerance']*a['expected'],
                          adaptive_tp=a['tp'], stage35_tp=s['tp'], time_ratio=a['search_ms']/s['search_ms'])
    verdicts['H5'] = dict(passed=a['negative_fp'] == 0 and a['negative_fp'] <= totals['original']['negative_fp'],
                          negative_fp={arm: totals[arm]['negative_fp'] for arm in FORMAL_ARMS})
    verdicts['H6'] = dict(passed=a['fp'] <= totals['original']['fp']+gates['H6_fp_tolerance']*a['expected'],
                          positive_fp={arm: totals[arm]['fp'] for arm in FORMAL_ARMS})
    integrity = dict(all_exact_repeat=all(value['exact_repeat'] for value in table.values()),
                     all_blocks_identical=all(value['matches_identical_across_blocks'] for value in table.values()),
                     host_busy_cells=sorted(f'{case}/{arm}' for (case, arm), value in table.items() if value['host_busy_blocks']),
                     lock_wait_cells=sorted(f'{case}/{arm}' for (case, arm), value in table.items() if value['lock_waits']),
                     unchanged_binary_and_inputs=manifest.get('unchanged'))
    cell_rows = [dict(case=case, arm=arm, **{key: value for key, value in data.items() if key != 'matches'})
                 for (case, arm), data in sorted(table.items())]
    return dict(totals=totals, deficient=deficient, sufficient=sufficient, per_template=per_template,
                verdicts=verdicts, integrity=integrity, cells=cell_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=list(PHASES))
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tuning', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--lock-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--tune', type=Path, help='dev_sweep manifest')
    parser.add_argument('--calibrate', type=Path, help='dev_calibrate manifest')
    parser.add_argument('--summarize', type=Path, help='formal manifest (needs --protocol)')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite output')
    if args.tune:
        save(args.output, tune(json.loads(args.tune.read_text()), args.tune.resolve().parent))
    elif args.calibrate:
        save(args.output, calibrate(json.loads(args.calibrate.read_text()), args.calibrate.resolve().parent))
    elif args.summarize:
        save(args.output, summarize(json.loads(args.summarize.read_text()), args.summarize.resolve().parent,
                                    json.loads(args.protocol.read_text())))
    else:
        if not (args.phase and args.cases and args.build):
            parser.error('--phase, --cases and --build are required for a run')
        run_phase(args.phase, args.cases, args.build, args.output, args.tuning, args.protocol, args.lock_dir, args.timeout)


if __name__ == '__main__':
    main()
