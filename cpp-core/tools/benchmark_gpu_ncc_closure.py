"""Stage 40: exact NCC variants, screening, and phase-closure benchmarks.

Stage-37/38/39 helpers and validators remain unmodified. New-mode fine-work
validation is local, including exact source-sampler formulas; since
2026-10-19 it also checks the runner's 24-byte mode-4 route records, which the
earlier-stage validators (16 bytes per level) reject. Every measured
and first call is checked against independent truth; every GPU candidate is
checked against complete ordered baseline matches AND proposals. Pilots are
explicitly labelled and cannot support adoption. Profile timings are separate.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

import benchmark_gpu_fine_tiles as stage39

peaks = stage39.peaks

previous = peaks.previous
ROOT = peaks.ROOT
DEFAULT_CASES = peaks.DEFAULT_CASES
PROFILE_CASES = peaks.PROFILE_CASES
METHODS = {
    'hybrid_best': dict(full_gpu=False, coarse_peak_mode=1, full_gpu_peak_mode=0, full_gpu_fine_mode=0),
    'full_gpu_tiled': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=1, full_gpu_fine_mode=0),
    'full_gpu_fine_tiled': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=1, full_gpu_fine_mode=1),
    'full_gpu_u32': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=1, full_gpu_fine_mode=2),
    'full_gpu_moments': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=1, full_gpu_fine_mode=3),
    'full_gpu_simd': dict(full_gpu=True, coarse_peak_mode=1, full_gpu_peak_mode=1, full_gpu_fine_mode=4),
}
GPU_METHODS = {name for name, options in METHODS.items() if options['full_gpu']}
BASELINE = 'full_gpu_tiled'
CANDIDATES = GPU_METHODS - {BASELINE}
DEFAULT_METHODS = ','.join(name for name in METHODS if name != 'full_gpu_simd')
PRIMARY_CASES = peaks.PRIMARY_CASES
sha, save, distribution = peaks.sha, peaks.save, peaks.distribution
fixture_cases, integer, profile_record = peaks.fixture_cases, peaks.integer, peaks.profile_record
profile_overhead, evaluate = peaks.profile_overhead, peaks.evaluate


def job(case, method, repeats, warmup, profile=False):
    result = previous.job(case, 'full_gpu' if method in GPU_METHODS else 'hybrid_best', repeats, warmup)
    result.update(METHODS[method])
    result['full_gpu_profiling'] = profile
    return result


FINE_COUNT_FIELDS = ('reached_candidates', 'angles', 'cached_angles', 'direct_angles',
                     'tiled_angles', 'source_samples', 'tile_loads')
FINE_GEOMETRY_FIELDS = ('template_width', 'template_height', 'roi_cache_bytes')
# Since 2026-10-19 the runner also reports which mode-4 windows used the SIMD32
# kernel or its mode-2 fallback; each level record is then 24 diagnostic bytes.
# Archived records carry none of these fields and are validated as before.
FINE_ROUTE_FIELDS = ('simd32_windows', 'fallback_windows')
FINE_ROUTE_TOTALS = ('fine_simd32_windows', 'fine_fallback_windows')


def fine_work_record(record, requested):
    """Audit final diagnostic readback, independently of timestamp support."""
    metrics = record['full_gpu']
    if type(metrics.get('fine_mode')) is not int or metrics['fine_mode'] not in range(5):
        raise ValueError('Invalid fine_mode')
    effective = integer(metrics.get('fine_effective_mode'), 'fine_effective_mode')
    simd_width = integer(metrics.get('fine_simd_width'), 'fine_simd_width')
    if simd_width < 1:
        raise ValueError('Empty fine SIMD width')
    expected_effective = 2 if metrics['fine_mode'] == 4 and simd_width != 32 else metrics['fine_mode']
    if effective != expected_effective:
        raise ValueError('Effective fine mode differs from requested mode and SIMD fallback contract')
    if type(metrics.get('fine_work_available')) is not bool or metrics['fine_work_available'] != requested:
        raise ValueError('Mismatched fine_work_available')
    for field in ('fine_diagnostic_bytes', 'fine_diagnostic_readback_bytes', 'fine_threadgroup_bytes'):
        integer(metrics.get(field), field)
    if metrics['fine_diagnostic_readback_bytes'] > metrics['fine_diagnostic_bytes']:
        raise ValueError('Fine diagnostic payload exceeds retained allocation')
    if metrics['fine_diagnostic_bytes'] > metrics['gpu_bytes']:
        raise ValueError('Fine diagnostic allocation exceeds named Metal bytes')
    rows = metrics.get('fine_work')
    if not isinstance(rows, list):
        raise ValueError('Missing fine_work array')
    routes = any(field in metrics for field in FINE_ROUTE_TOTALS)
    if routes and not all(field in metrics for field in FINE_ROUTE_TOTALS):
        raise ValueError('Partial fine route totals')
    if not requested:
        if rows or metrics['fine_diagnostic_readback_bytes']:
            raise ValueError('Disabled fine diagnostics contain work or current payload')
        if routes and any(metrics[field] is not None for field in FINE_ROUTE_TOTALS):
            raise ValueError('Disabled fine diagnostics report route windows')
        return None
    if metrics['fine_diagnostic_readback_bytes'] != (24 if routes else 16)*len(rows):
        raise ValueError('Fine diagnostic payload differs from level coverage')
    for field in FINE_ROUTE_TOTALS if routes else ():
        integer(metrics[field], field)
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid fine-work row')
        if not routes and any(field in row for field in FINE_ROUTE_FIELDS):
            raise ValueError('Fine route windows without call totals')
        for field in ('model', 'layer', *FINE_GEOMETRY_FIELDS, *FINE_COUNT_FIELDS, *(FINE_ROUTE_FIELDS if routes else ())):
            integer(row.get(field), 'fine_work.'+field)
        if any(row[field] == 0 for field in FINE_GEOMETRY_FIELDS):
            raise ValueError('Empty fine-search geometry')
        key = (row['model'], row['layer'])
        if key in result:
            raise ValueError('Duplicate fine-work model/layer')
        width, height = row['template_width'], row['template_height']
        roi_area = (width+6)*(height+6)
        tile_rows = min(16, row['roi_cache_bytes']//(width+6)-6)
        cache_fits = roi_area <= row['roi_cache_bytes']
        tiled_eligible = not cache_fits and effective >= 1 and tile_rows >= 1
        if row['cached_angles'] and not cache_fits:
            raise ValueError('Cached fine angles exceed cache capacity')
        if row['tiled_angles'] and not tiled_eligible:
            raise ValueError('Tiled fine angles cannot use the declared cache/mode')
        if row['direct_angles'] and (cache_fits or tiled_eligible):
            raise ValueError('Direct fine angles bypass an eligible cache path')
        if row['angles'] != row['cached_angles']+row['direct_angles']+row['tiled_angles']:
            raise ValueError('Fine angle categories do not sum to angles')
        if not row['reached_candidates'] and row['angles']:
            raise ValueError('Fine angles without reached candidates')
        if metrics['fine_mode'] == 0 and row['tiled_angles']:
            raise ValueError('Baseline fine mode reports tiled angles')
        if not row['tiled_angles'] and row['tile_loads']:
            raise ValueError('Fine tile loads without tiled angles')
        expected_loads = row['tiled_angles']*((height+tile_rows-1)//tile_rows) if tiled_eligible else 0
        if row['tile_loads'] != expected_loads:
            raise ValueError('Fine strip count differs from cache capacity and geometry')
        # Last strip uses its actual remaining rows, retaining six halo rows.
        expected_samples = (row['cached_angles']*roi_area +
                            row['direct_angles']*49*width*height +
                            (width+6)*(height*row['tiled_angles']+6*row['tile_loads']))
        if row['source_samples'] != expected_samples:
            raise ValueError('Fine sampler-call count differs from executed categories')
        if routes:
            # SIMD32 needs a cached ROI and uint32-safe four-lane stripes.
            simd = effective == 4 and cache_fits and (width*height+3)//4 <= 66051
            windows = 49*row['angles']
            expected = (windows, 0) if simd else (0, windows if effective == 4 else 0)
            if (row['simd32_windows'], row['fallback_windows']) != expected:
                raise ValueError('Fine SIMD32/fallback windows differ from the cache and uint32 route rule')
        result[key] = row
    if routes and tuple(metrics[field] for field in FINE_ROUTE_TOTALS) != tuple(
            sum(row[field] for row in rows) for field in FINE_ROUTE_FIELDS):
        raise ValueError('Fine route totals differ from levels')
    return result


def fine_work_summary(calls, runs, requested):
    all_work = [fine_work_record(record, requested) for record in calls]
    measured = [fine_work_record(record, requested) for record in runs]
    available = [row for row in measured if row is not None]
    totals = [{name: sum(level[name] for level in work.values()) for name in FINE_COUNT_FIELDS}
              for work in available]
    layers = []
    for key in sorted({key for work in available for key in work}):
        rows = [work[key] for work in available if key in work]
        if len({tuple(row[name] for name in FINE_GEOMETRY_FIELDS) for row in rows}) != 1:
            raise ValueError('Fine-work geometry changes between measured calls')
        layers.append(dict(model=key[0], layer=key[1],
            **{name: rows[0][name] for name in FINE_GEOMETRY_FIELDS},
            roi_area_bytes=(rows[0]['template_width']+6)*(rows[0]['template_height']+6),
            tile_rows_capacity=max(0, min(16, rows[0]['roi_cache_bytes']//(rows[0]['template_width']+6)-6)),
            measured_available_calls=len(rows),
            counts={name: distribution([row[name] for row in rows]) for name in FINE_COUNT_FIELDS}))
    def fractions(field):
        values = [row[field]/row['angles'] for row in totals if row['angles']]
        return distribution(values) if values else None
    return dict(requested=requested, available_calls=sum(row is not None for row in all_work),
        total_calls=len(calls), measured_available_calls=len(available), measured_calls=len(runs),
        measured_totals={name: distribution([row[name] for row in totals]) for name in FINE_COUNT_FIELDS} if totals else None,
        cached_angle_fraction=fractions('cached_angles'), direct_angle_fraction=fractions('direct_angles'),
        tiled_angle_fraction=fractions('tiled_angles'), layers=layers,
        stable_measured_counts=all(work == available[0] for work in available) if available else None,
        note='Counters are final readback only. source_samples counts analytic sampler invocations from executed angle categories; it is not hardware memory traffic, cache hits or GPU occupancy.')


def fine_work_comparison(method_rows, candidate):
    if not {BASELINE, candidate} <= method_rows.keys():
        return None
    old = method_rows['full_gpu_tiled']['fine_work']
    new = method_rows[candidate]['fine_work']
    if old['measured_totals'] is None or new['measured_totals'] is None:
        return None
    def workload(summary):
        return [(row['model'], row['layer'], *(row[name] for name in FINE_GEOMETRY_FIELDS),
                 row['counts']['reached_candidates']['median'], row['counts']['angles']['median'])
                for row in summary['layers']]
    comparable = (old['measured_available_calls'] == old['measured_calls'] and
                  new['measured_available_calls'] == new['measured_calls'] and
                  old['stable_measured_counts'] and new['stable_measured_counts'] and workload(old) == workload(new))
    left = old['measured_totals']['source_samples']['median']
    right = new['measured_totals']['source_samples']['median']
    return dict(same_measured_workload=comparable,
        source_sampler_calls_delta_percent=100*(right/left-1) if comparable and left else None,
        source_sampler_calls_ratio=right/left if comparable and left else None,
        note='Sampler-call reduction is an operation count, not an end-to-end speedup; arithmetic and synchronization remain.')


def summarize(manifest, root):
    if manifest.get('complete') is not True or manifest.get('unchanged') is not True:
        raise ValueError('Experiment must be complete with unchanged fingerprints')
    fingerprints = {name: manifest.get(name) for name in ('source_sha256', 'binary_sha256', 'input_sha256')}
    if not all(fingerprints.values()) or manifest.get('after_sha256') != fingerprints:
        raise ValueError('Missing or inconsistent before/after fingerprints')
    requested = manifest.get('profiling')
    if type(requested) is not bool:
        raise ValueError('Missing profiling boolean')
    methods = manifest['methods']
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or set(methods)-METHODS.keys():
        raise ValueError('Empty, duplicate, or unknown methods')
    if requested and not set(methods) <= GPU_METHODS:
        raise ValueError('Profile runs require full-GPU methods')
    if not requested and (BASELINE not in methods or not set(methods) & CANDIDATES):
        raise ValueError('Unprofiled runs require baseline and at least one GPU candidate')
    phase = manifest.get('phase')
    if phase not in ('pilot', 'formal', 'profile') or (phase == 'profile') != requested:
        raise ValueError('Missing or inconsistent experiment phase')
    if json.dumps(manifest.get('method_options'), sort_keys=True) != json.dumps({name: METHODS[name] for name in methods}, sort_keys=True):
        raise ValueError('Manifest method_options differ from declared modes')
    for name in ('blocks', 'repeats'):
        if integer(manifest[name], name) < 1:
            raise ValueError('Empty '+name)
    previous.finite_nonnegative(manifest['warmup_seconds'], 'warmup_seconds')
    if manifest['warmup_seconds'] > 60:
        raise ValueError('Warmup exceeds protocol maximum')
    if phase == 'formal' and (manifest['blocks'] < 3 or manifest['repeats'] < 2 or manifest['warmup_seconds'] < 1):
        raise ValueError('Formal run requires at least 3 blocks, 2 repeats and 1 second warmup')
    cases = {case['name']: case for case in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Empty or duplicate cases')
    expected = {(name, method, block) for name in cases for method in methods for block in range(manifest['blocks'])}
    entries = {}
    for entry in manifest['executions']:
        integer(entry['block'], 'execution.block')
        key = (entry['case'], entry['method'], entry['block'])
        if key not in expected or key in entries:
            raise ValueError('Unexpected or duplicate worker '+str(key))
        paths = {name: previous.worker_path(root, entry[name]) for name in ('job', 'result')}
        for name, path in paths.items():
            if sha(path) != entry[name+'_sha256']:
                raise ValueError('Changed '+name+': '+entry[name])
        work_job = json.loads(paths['job'].read_text())
        expected_job = job(cases[entry['case']], entry['method'], manifest['repeats'], manifest['warmup_seconds'], requested)
        # Python equality treats bool as int; serialized equality preserves the
        # worker's strict boolean/mode contracts and every shared setting.
        if json.dumps(work_job, sort_keys=True) != json.dumps(expected_job, sort_keys=True):
            raise ValueError('Worker job differs from declared shared search settings')
        worker = json.loads(paths['result'].read_text())
        if len(worker['runs']) != manifest['repeats']:
            raise ValueError('Missing or extra measured repeats')
        previous.finite_nonnegative(worker['construct_ms'], 'construct_ms')
        integer(worker['final_memory']['peak_rss_bytes'], 'peak_rss_bytes')
        if type(worker.get('exact_repeat')) is not bool:
            raise ValueError('Missing exact_repeat boolean')
        entries[key] = worker
    if set(entries) != expected:
        raise ValueError('Missing measurement blocks')
    failures, rows = [], []
    for case_name, case in cases.items():
        reference = entries[(case_name, BASELINE, 0)]['first'] if BASELINE in methods else None
        method_rows = {}
        for method in methods:
            blocks = [(block, entries[(case_name, method, block)]) for block in range(manifest['blocks'])]
            calls = [record for _, worker in blocks for record in [worker['first'], *worker['runs']]]
            runs = [record for _, worker in blocks for record in worker['runs']]
            gpu = method in GPU_METHODS
            routes = [previous.validate_record(record, gpu) for record in calls]
            profiles = []
            if gpu:
                for record in calls:
                    if type(record['full_gpu'].get('peak_mode')) is not int:
                        raise ValueError('Invalid peak_mode')
                    routes.append(record['full_gpu']['peak_mode'] == METHODS[method]['full_gpu_peak_mode'])
                    fine_work_record(record, requested)
                    routes.append(record['full_gpu']['fine_mode'] == METHODS[method]['full_gpu_fine_mode'])
                    profiles.append(profile_record(record, requested))
            accuracy = [evaluate(record['matches'], case['truth'])['strict'] for record in calls]
            gt_pass = all(row['fp'] == row['fn'] == 0 for row in accuracy)
            exact_matches = all(record['matches'] == reference['matches'] for record in calls) if reference is not None else None
            exact_proposals = all(record['proposals'] == reference['proposals'] for record in calls) if reference is not None else None
            repeat_exact = all(worker['exact_repeat'] for _, worker in blocks) and all(
                record['matches'] == calls[0]['matches'] and record['proposals'] == calls[0]['proposals'] for record in calls)
            algorithm_executed = all(record['full_gpu']['fine_effective_mode'] == METHODS[method]['full_gpu_fine_mode'] for record in calls) if gpu else True
            measured_metrics = [record['full_gpu'] for record in runs] if gpu else []
            timing_flags = [previous.device_timestamp_available(record) for record in measured_metrics]
            gpu_fields = (*previous.GPU_FIELDS, 'peak_scratch_bytes', 'profile_counter_bytes', 'profile_resolve_bytes', 'profile_host_ms', 'fine_diagnostic_bytes', 'fine_diagnostic_readback_bytes', 'fine_threadgroup_bytes')
            gpu_metrics = {name: distribution([record[name] for record in measured_metrics]) for name in gpu_fields if name != 'device_ms'} if gpu else None
            if gpu:
                values = [record['device_ms'] for record, available in zip(measured_metrics, timing_flags) if available]
                gpu_metrics['device_ms'] = distribution(values) if values else None
            measured_profile = [profile_record(record, requested) for record in runs] if gpu else []
            available_profile = [row for row in measured_profile if row is not None]
            stage_names = sorted({name for row in available_profile for name in row['stages']})
            profiling = None
            if gpu:
                profiling = dict(requested=requested, available_calls=sum(row is not None for row in profiles),
                    total_calls=len(calls), measured_available_calls=len(available_profile), measured_calls=len(runs),
                    status_counts={status: sum(record['full_gpu']['profile_status'] == status for record in calls)
                                   for status in sorted({record['full_gpu']['profile_status'] for record in calls})},
                    sample_counts=sorted({record['full_gpu']['profile_samples'] for record in calls}),
                    stages={name: dict(gpu_ms=distribution([row['stages'][name]['gpu_ms'] for row in available_profile if name in row['stages']]),
                                       encoders=sorted({row['stages'][name]['encoders'] for row in available_profile if name in row['stages']}),
                                       measured_available_calls=sum(name in row['stages'] for row in available_profile)) for name in stage_names},
                    stage_sum_ms=distribution([row['stage_sum_ms'] for row in available_profile]) if available_profile else None,
                    device_remainder_ms=distribution([row['device_remainder_ms'] for row in available_profile if row['device_remainder_ms'] is not None])
                        if any(row['device_remainder_ms'] is not None for row in available_profile) else None)
            method_rows[method] = dict(total_ms=distribution([row['total_ms'] for row in runs]),
                blocks=[dict(block=block, median_ms=statistics.median(row['total_ms'] for row in worker['runs'])) for block, worker in blocks],
                first_ms=[worker['first']['total_ms'] for _, worker in blocks], construct_ms=[worker['construct_ms'] for _, worker in blocks],
                peak_rss_bytes=max(worker['final_memory']['peak_rss_bytes'] for _, worker in blocks),
                retained_gpu_bytes=max(record['full_gpu']['gpu_bytes'] if gpu else record['frame_coarse_gpu_bytes']+record['frame_fine_gpu_bytes'] for record in calls),
                retained_gpu_scope='all named full-GPU Metal buffers' if gpu else 'frame coarse + fine buffers only; other owners excluded',
                measured_gpu_metrics=gpu_metrics, profiling=profiling,
                requested_fine_mode=METHODS[method]['full_gpu_fine_mode'] if gpu else None,
                effective_fine_modes=sorted({record['full_gpu']['fine_effective_mode'] for record in calls}) if gpu else None,
                fine_simd_widths=sorted({record['full_gpu']['fine_simd_width'] for record in calls}) if gpu else None,
                fallback_calls=sum(record['full_gpu']['fine_effective_mode'] != record['full_gpu']['fine_mode'] for record in calls) if gpu else 0,
                requested_algorithm_executed=algorithm_executed,
                fine_work=fine_work_summary(calls, runs, requested) if gpu else None,
                device_timestamp_available_measured_calls=sum(timing_flags) if gpu else None,
                all_gt_pass=gt_pass, accuracy_counts=sorted({(row['tp'], row['fp'], row['fn']) for row in accuracy}),
                max_errors={name: max((row['max_errors'][name] for row in accuracy if row['max_errors'][name] is not None), default=None)
                            for name in ('position_px', 'angle_deg', 'relative_scale')},
                route_valid=all(routes), full_pipeline_residency_verified=all(routes) if gpu else False,
                ordered_matches_equal_baseline_gpu=exact_matches, ordered_proposals_equal_baseline_gpu=exact_proposals,
                repeat_exact=repeat_exact, proposal_counts=sorted({len(row['proposals']) for row in calls}),
                first_calls=len(blocks), measured_calls=len(runs), gt_evaluated_calls=len(calls))
            for name, passed in [('independent_ground_truth', gt_pass), ('route_and_residency', all(routes)), ('exact_repeat', repeat_exact)]:
                if not passed:
                    failures.append(dict(case=case_name, method=method, gate=name))
            if method in CANDIDATES and not algorithm_executed:
                failures.append(dict(case=case_name, method=method, gate='requested_algorithm_not_executed'))
            if method in CANDIDATES and BASELINE in methods:
                for name, passed in [('exact_ordered_matches', exact_matches), ('exact_ordered_proposals', exact_proposals)]:
                    if not passed:
                        failures.append(dict(case=case_name, method=method, gate=name))
        comparisons = {}
        for candidate in (name for name in methods if name in CANDIDATES):
            comparisons[candidate] = {}
            if requested:
                continue
            measured = method_rows[candidate]
            for baseline_name in (BASELINE, 'hybrid_best', 'full_gpu_fine_tiled'):
                if baseline_name not in method_rows or baseline_name == candidate:
                    continue
                baseline = method_rows[baseline_name]
                paired = []
                for left, right in zip(baseline['blocks'], measured['blocks']):
                    passed = right['median_ms'] <= left['median_ms']+max(2., .1*left['median_ms'])
                    paired.append(dict(block=left['block'], delta_percent=100*(right['median_ms']/left['median_ms']-1), within_regression_budget=passed))
                    if not passed:
                        failures.append(dict(case=case_name, method=candidate, baseline=baseline_name, gate='time_budget', block=left['block']))
                memory_ok = measured['peak_rss_bytes'] <= baseline['peak_rss_bytes']+max(8*1024**2, .1*baseline['peak_rss_bytes'])
                if not memory_ok:
                    failures.append(dict(case=case_name, method=candidate, baseline=baseline_name, gate='rss_budget'))
                semantic_pass = all(row['all_gt_pass'] and row['route_valid'] and row['repeat_exact'] and row['requested_algorithm_executed'] for row in (baseline, measured)) and measured['ordered_matches_equal_baseline_gpu'] and measured['ordered_proposals_equal_baseline_gpu']
                delta = 100*(measured['total_ms']['median']/baseline['total_ms']['median']-1)
                target = delta <= -10. if baseline_name == BASELINE and case_name in PRIMARY_CASES else None
                if target is False:
                    failures.append(dict(case=case_name, method=candidate, baseline=baseline_name, gate='primary_10_percent_improvement'))
                observed_pass = bool(semantic_pass and memory_ok and all(row['within_regression_budget'] for row in paired))
                comparisons[candidate][baseline_name] = dict(delta_percent=delta, primary_10_percent_improvement=target,
                    paired=paired, memory_within_budget=memory_ok, observed_gates_pass=observed_pass,
                    candidate_for_adoption=phase == 'formal' and observed_pass)
        rows.append(dict(name=case_name, truth_count=len(case['truth']), methods=method_rows,
            candidate_comparisons=comparisons,
            fine_work_comparisons={name: fine_work_comparison(method_rows, name) for name in methods if name in CANDIDATES}))
    candidate_summary = {}
    for candidate in (name for name in methods if name in CANDIDATES):
        exact = all(row['methods'][candidate]['ordered_matches_equal_baseline_gpu'] and row['methods'][candidate]['ordered_proposals_equal_baseline_gpu'] for row in rows) if BASELINE in methods else None
        baselines = {}
        if not requested:
            for baseline_name in (BASELINE, 'hybrid_best', 'full_gpu_fine_tiled'):
                if baseline_name not in methods or baseline_name == candidate:
                    continue
                comparisons = [row['candidate_comparisons'][candidate][baseline_name] for row in rows]
                baselines[baseline_name] = dict(
                    all_observed_gates_pass=all(row['observed_gates_pass'] for row in comparisons),
                    all_candidate_for_adoption=all(row['candidate_for_adoption'] for row in comparisons))
        primary_covered = PRIMARY_CASES <= cases.keys()
        target_pass = all(row['candidate_comparisons'][candidate][BASELINE]['primary_10_percent_improvement'] and
                          row['candidate_comparisons'][candidate][BASELINE]['observed_gates_pass']
                          for row in rows if row['name'] in PRIMARY_CASES) if not requested and primary_covered else None
        candidate_summary[candidate] = dict(all_exact_baseline=exact, comparisons=baselines,
            all_requested_algorithm_executed=all(row['methods'][candidate]['requested_algorithm_executed'] for row in rows),
            fallback_calls=sum(row['methods'][candidate]['fallback_calls'] for row in rows),
            primary_cases_present=sorted(PRIMARY_CASES & cases.keys()),
            primary_10_percent_target_pass=target_pass,
            formal_primary_target_pass=target_pass if phase == 'formal' else None)
    return dict(phase=phase, profiling=requested, formal_adoption_eligible=phase == 'formal',
        cases=rows, failed_gates=failures, candidates=candidate_summary,
        all_accuracy_pass=all(row['methods'][method]['all_gt_pass'] for row in rows for method in methods),
        all_gpu_resident=all(row['methods'][method]['full_pipeline_residency_verified'] for row in rows for method in methods if method in GPU_METHODS),
        all_candidates_exact_baseline=all(row['all_exact_baseline'] for row in candidate_summary.values()) if candidate_summary and BASELINE in methods else None,
        counts=dict(fresh_processes=len(entries), first_calls=len(entries), measured_calls=len(entries)*manifest['repeats'], gt_evaluated_calls=len(entries)*(manifest['repeats']+1)),
        accuracy_thresholds=dict(position_px=2., angle_deg=1., relative_scale=.02),
        timing_note='First calls are separate. Distributions use measured repeats only. Pilots are screening only, not adoption evidence. Formal requires at least 3 alternating blocks, 2 measured repeats and 1 second warmup. Profile diagnostics never enter unprofiled speed gates; unavailable stages and remainder are null. Signed device remainder includes unsampled work, gaps, and timestamp-instrument differences.',
        memory_note='Peak process RSS and retained named Metal buffer lengths differ in scope and must not be added; GPU buffer lengths are not physical VRAM. Counter payload capacity is logical and opaque to Metal allocation accounting. Resolved timestamp bytes are transient CPU NSData payload, excluded from gpu_bytes; peak scratch and retained fine diagnostic buffers are already included in full-GPU gpu_bytes. Fine diagnostic current readback payload is separate from algorithm readback_bytes. fine_threadgroup_bytes is compiled kernel static threadgroup storage, not retained buffer allocation.',
        parity_note='Independent one-to-one GT accuracy plus exact ordered matches AND proposals against full_gpu_tiled are required for every candidate. A single-method profile without that baseline makes no cross-mode parity claim.',
        residency_note='Audited implementation counters, not a hardware trace: one command buffer, one final wait, no intermediate readback or image-dependent CPU algorithm.')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build/fourier-gpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures-manifest', type=Path, default=ROOT/'build/stage37-full-gpu/formal/manifest.json')
    parser.add_argument('--cases')
    parser.add_argument('--methods')
    parser.add_argument('--blocks', type=int)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--timeout', type=float, default=600.)
    purpose = parser.add_mutually_exclusive_group()
    purpose.add_argument('--profile', action='store_true')
    purpose.add_argument('--pilot', action='store_true', help='Fast screen, never eligible for adoption')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--unprofiled-manifest', type=Path, help='Separate formal run used only to report profiler overhead')
    args = parser.parse_args()
    args.blocks = args.blocks if args.blocks is not None else (1 if args.profile or args.pilot else 3)
    args.warmup = args.warmup if args.warmup is not None else (0. if args.profile or args.pilot else 1.)
    if args.output.exists():
        raise ValueError('Refusing to overwrite experiment output')
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        summary = summarize(manifest, args.summarize.resolve().parent)
        if args.unprofiled_manifest:
            baseline = json.loads(args.unprofiled_manifest.read_text())
            summary['profile_overhead'] = profile_overhead(manifest, summary, baseline, summarize(baseline, args.unprofiled_manifest.resolve().parent))
            summary['unprofiled_manifest_sha256'] = sha(args.unprofiled_manifest)
        save(args.output, summary)
        return
    if args.unprofiled_manifest:
        parser.error('--unprofiled-manifest is available only with --summarize')
    if args.blocks < 1 or args.repeats < 1 or not 0 <= args.warmup <= 60 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Positive blocks/repeats/timeout and warmup in 0..60 required')
    if not args.profile and not args.pilot and (args.blocks < 3 or args.repeats < 2 or args.warmup < 1):
        parser.error('Formal requires at least 3 blocks, 2 repeats and 1 second warmup; use --pilot for screening')
    names = previous.selected_names(args.cases or (PROFILE_CASES if args.profile else DEFAULT_CASES), None, 'cases')
    methods = previous.selected_names(args.methods or ('full_gpu_tiled,full_gpu_u32,full_gpu_moments' if args.profile else DEFAULT_METHODS), METHODS, 'methods')
    if args.profile and not set(methods) <= GPU_METHODS or not args.profile and (BASELINE not in methods or not set(methods) & CANDIDATES):
        parser.error('Profile requires only GPU methods; unprofiled requires baseline and at least one GPU candidate')
    root = args.output.resolve().parent
    if any((root/name).exists() for name in ('jobs', 'results')):
        raise ValueError('Refusing to overwrite an existing experiment directory')
    cases = fixture_cases(args.fixtures_manifest, names)
    binary = (args.build/'experiments/fpm_native_pipeline_runner').resolve()
    before = previous.fingerprints(binary, cases)
    manifest = dict(version=1, phase='profile' if args.profile else 'pilot' if args.pilot else 'formal', platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        profiling=args.profile, blocks=args.blocks, repeats=args.repeats, warmup_seconds=args.warmup,
        timeout_seconds=args.timeout, cases=cases, methods=methods, method_options={name: METHODS[name] for name in methods},
        fixtures_manifest=str(args.fixtures_manifest.resolve()), fixtures_manifest_sha256=sha(args.fixtures_manifest),
        executions=[], complete=False, **before)
    save(args.output, manifest)
    try:
        for case in cases:
            for block in range(args.blocks):
                for method in methods[::1 if block % 2 == 0 else -1]:
                    label = f"{case['name']}-{method}-b{block}"
                    jp, rp = root/'jobs'/(label+'.json'), root/'results'/(label+'.json')
                    save(jp, job(case, method, args.repeats, args.warmup, args.profile))
                    print('Running '+label, flush=True)
                    try:
                        child = subprocess.run([str(binary), str(jp)], capture_output=True, text=True, timeout=args.timeout)
                    except subprocess.TimeoutExpired as error:
                        jp.with_suffix('.stdout.txt').write_text(error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or '')
                        jp.with_suffix('.stderr.txt').write_text(error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or '')
                        raise
                    jp.with_suffix('.stdout.txt').write_text(child.stdout)
                    jp.with_suffix('.stderr.txt').write_text(child.stderr)
                    if child.returncode:
                        raise RuntimeError(label+': '+child.stderr[-4000:])
                    save(rp, json.loads(child.stdout))
                    manifest['executions'].append(dict(case=case['name'], method=method, block=block,
                        job=str(jp.relative_to(root)), job_sha256=sha(jp), result=str(rp.relative_to(root)), result_sha256=sha(rp)))
                    save(args.output, manifest)
        after = previous.fingerprints(binary, cases)
        manifest.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), after_sha256=after, unchanged=before == after)
        save(args.output, manifest)
        manifest['summary'] = summarize(manifest, root)
        save(args.output, manifest)
    except Exception as error:
        manifest['failure'] = dict(type=type(error).__name__, message=str(error))
        save(args.output, manifest)
        raise


if __name__ == '__main__':
    main()
