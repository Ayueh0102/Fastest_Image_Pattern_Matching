"""Read existing profiling records; never run matching or GPU computation."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics


STAGES = ('source_copy_invert_ms', 'pyramid_ms', 'coarse_rotation_ms', 'coarse_ncc_ms',
          'peak_selection_ms', 'fine_rotation_ms', 'fine_ncc_ms', 'fine_control_ms',
          'nms_ms', 'unclassified_ms')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_logical_path(recorded_path):
    """Keep the observed jobs directory and filename, even across OS styles."""
    original = PurePosixPath(str(recorded_path).replace('\\', '/'))
    if not original.name or original.name in ('.', '..'):
        raise ValueError('Worker result has no valid filename: ' + str(recorded_path))
    return PurePosixPath(original.parent.name)/original.name


def resolve_worker_path(entry, report_path, worker_root=None):
    """Explicit archive roots never fall back to the original machine's files.

    An archive root may contain the original jobs directory or be that directory
    itself. The first existing candidate is authoritative: an incorrect SHA is
    an error, never a reason to try a different copy of the same result.
    """
    logical = worker_logical_path(entry['result'])
    if worker_root is not None:
        root = Path(worker_root).resolve()
        candidates = (root/str(logical), root/logical.name)
    else:
        root = None
        original = Path(entry['result'])
        if not original.is_absolute():
            original = Path(report_path).parent/original
        candidates = (original, Path(report_path).parent/str(logical))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            resolved = candidate.resolve()
            if root is not None and not resolved.is_relative_to(root):
                raise ValueError('Archive worker path escapes the explicit root: ' + str(candidate))
            return resolved
    raise FileNotFoundError('Worker result is missing; checked: ' + ', '.join(map(str, candidates)))


def load_worker(entry, report_path, worker_root=None):
    path = resolve_worker_path(entry, report_path, worker_root)
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != entry['result_sha256']:
        raise ValueError('Raw worker result changed: ' + str(path))
    # Parse the exact bytes just verified, rather than rereading an unchecked
    # second snapshot. Logical hash keys remain stable after archive relocation.
    return str(worker_logical_path(entry['result'])), actual, json.loads(content)


def fraction(value, total):
    return value/total if total > 0 else None


def representative(records):
    center = statistics.median(record['proposal_wall_ms'] for record in records)
    return min(records, key=lambda record: abs(record['proposal_wall_ms']-center))


def paired_regression(methods, candidate, baseline):
    if candidate not in methods or baseline not in methods:
        return dict(available=False, passed=None, reason='comparison method not measured')
    left = {block['block']: block for block in methods[baseline]['blocks']}
    right = {block['block']: block for block in methods[candidate]['blocks']}
    common = sorted(left.keys() & right.keys())
    if not common:
        return dict(available=False, passed=None, reason='no matching order blocks')
    blocks = []
    for block in common:
        old, new = left[block]['total_ms'], right[block]['total_ms']
        allowance = max(.1*old, 2.)
        blocks.append(dict(block=block, baseline_total_ms=old, candidate_total_ms=new,
            delta_ms=new-old, delta_percent=100*(new/old-1) if old else None,
            allowed_regression_ms=allowance, passed=new <= old+allowance))
    return dict(available=True, passed=all(block['passed'] for block in blocks), blocks=blocks)


def bottleneck_comparison(workers, candidate, baseline, stage):
    if candidate not in workers or baseline not in workers:
        return dict(available=False, repeatable_improvement=None, reason='profile comparison not measured')
    def times(name):
        return {worker['block']: statistics.median(sum(p['profile'][stage] for p in record['patterns'])
                    for record in worker['runs']) for worker in workers[name]}
    left, right = times(baseline), times(candidate)
    blocks = [dict(block=block, baseline_ms=left[block], candidate_ms=right[block],
                   improved=right[block] < left[block]) for block in sorted(left.keys() & right.keys())]
    return dict(available=bool(blocks), stage=stage,
        repeatable_improvement=all(block['improved'] for block in blocks) if len(blocks) >= 2 else None,
        blocks=blocks, note='profiled stage medians; at least two matching order blocks required; inspect clock overhead separately')


def performance_case(methods, workers):
    regressions, bottlenecks = {}, {}
    candidates = [name for name in methods if not name.startswith('frozen_') and not name.endswith('_profile')]
    for candidate in candidates:
        regressions[candidate] = {baseline: paired_regression(methods, candidate, baseline)
                                 for baseline in ('frozen_cpu', 'frozen_metal', 'metal') if baseline != candidate}
    for candidate in ('pyramid_direct', 'pyramid_separable', 'batched', 'combined'):
        if candidate+'_profile' not in workers:
            continue
        stages = ('pyramid_ms', 'coarse_ncc_ms') if candidate == 'combined' else (
            ('coarse_ncc_ms',) if candidate == 'batched' else ('pyramid_ms',))
        bottlenecks[candidate] = {stage: bottleneck_comparison(workers, candidate+'_profile',
                                                             'metal_profile', stage) for stage in stages}
    return dict(whole_pipeline_regression=regressions, targeted_profiled_stage_improvement=bottlenecks)


def summarize_record(record):
    profiles = [p['profile'] for p in record['patterns'] if p['profile'] is not None]
    total, proposal = record['total_wall_ms'], record['proposal_wall_ms']
    output = dict(total_wall_ms=total, proposal_wall_ms=proposal,
                  output_sha256=record['output_sha256'])
    if not profiles:
        return output
    core = sum(p['core_wall_ms'] for p in profiles)
    stages = {key: sum(p[key] for p in profiles) for key in STAGES}
    output.update(core_wall_ms=core,
        api_wall_ms=sum(p['api_wall_ms'] for p in profiles),
        exclusive_stages={key: dict(ms=value, fraction_of_core=fraction(value, core),
                                   fraction_of_total=fraction(value, total)) for key, value in stages.items()},
        ideal_zero_cost_stage_speed_bound={key: total/(total-value) if total > value else None
            for key, value in stages.items() if key in ('pyramid_ms', 'coarse_rotation_ms',
                                                       'fine_rotation_ms', 'fine_ncc_ms')})
    metrics = [p['metrics'] for p in record['patterns'] if p['metrics'] is not None]
    if profiles[0]['backend'] == 1:
        nested = {key: sum(m[key] for m in metrics) for key in
                  ('upload_ms', 'device_ms', 'wait_ms', 'normalization_ms', 'ncc_ms', 'command_buffers')}
        device_valid = all(m['device_ms'] >= 0 for m in metrics)
        if not device_valid:
            nested['device_ms'] = None
        output['nested_metal'] = dict(values=nested,
            shared_memcpy_fraction_of_total=fraction(nested['upload_ms'], total),
            wait_fraction_of_total=fraction(nested['wait_ms'], total),
            device_fraction_of_total=fraction(nested['device_ms'], total) if device_valid else None,
            wait_minus_device_ms=nested['wait_ms']-nested['device_ms'] if device_valid else None)
    output['retained_coarse_buffers'] = {key: sum(m.get(key, 0) for m in metrics) for key in
        ('host_bytes', 'gpu_bytes', 'workspace_bytes', 'source_buffer_bytes', 'template_buffer_bytes')}
    execution = [p['execution_metrics'] for p in record['patterns'] if p.get('execution_metrics')]
    if execution:
        output['execution_config'] = [p.get('execution_config') for p in record['patterns']]
        fields = ('pyramid_wall_ms', 'pyramid_setup_ms', 'pyramid_upload_ms', 'pyramid_encode_ms',
                  'pyramid_wait_ms', 'pyramid_device_ms', 'pyramid_copyout_ms', 'batch_encode_ms')
        nested = {key: sum(m[key] for m in execution) for key in fields}
        if any(m['pyramid_device_ms'] < 0 for m in execution):
            nested['pyramid_device_ms'] = None
        output['nested_execution'] = dict(values=nested,
            pyramid_copy_fraction_of_total=fraction(nested['pyramid_upload_ms']+nested['pyramid_copyout_ms'], total),
            pyramid_wait_fraction_of_total=fraction(nested['pyramid_wait_ms'], total),
            pyramid_device_fraction_of_total=fraction(nested['pyramid_device_ms'], total)
                if nested['pyramid_device_ms'] is not None else None,
            counters={key: sum(m[key] for m in execution) for key in
                ('pyramid_command_buffers', 'pyramid_levels', 'batch_calls', 'batch_maps')},
            max_batch_maps=max(m['max_batch_maps'] for m in execution),
            buffer_breakdown={key: sum(m[key] for m in execution) for key in
                ('pyramid_input_bytes', 'pyramid_output_bytes', 'pyramid_scratch_bytes',
                 'pyramid_buffer_bytes', 'batch_source_arena_bytes', 'batch_source_upload_bytes',
                 'batch_template_buffer_bytes', 'batch_statistics_bytes')})
    output['truth_accuracy'] = record.get('truth_accuracy')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    parser.add_argument('--output')
    parser.add_argument('--worker-root', help='Archive root or archived jobs directory; overrides original worker paths')
    args = parser.parse_args()
    path = Path(args.report).resolve()
    report = json.loads(path.read_text())
    grouped, raw_hashes = {}, {}
    for entry in report['execution_order']:
        logical, actual, result = load_worker(entry, path, args.worker_root)
        if logical in raw_hashes:
            raise ValueError('Duplicate worker artifact logical path: ' + logical)
        raw_hashes[logical] = actual
        if not entry['reference_only']:
            grouped.setdefault((entry['case'], entry['method']), []).append(result)
    rows = []
    for case in report['cases']:
        methods, case_workers = {}, {}
        for method, summary in case['methods'].items():
            workers = grouped[(case['case']['name'], method)]
            case_workers[method] = workers
            records = [record for worker in workers for record in worker['runs']]
            measured = summary['measured']
            methods[method] = dict(
                total_ms=measured['total_wall_ms'], proposal_ms=measured['proposal_wall_ms'],
                first_total_ms=summary['first_total_ms'], construct_ms=summary['construct_ms'],
                peak_rss_bytes=summary['peak_rss_bytes'], blocks=summary['blocks'],
                representative_median_proposal_call=summarize_record(representative(records)),
                nominal_output_buffers=summary.get('nominal_output_buffers'),
                execution_config=summary.get('execution_config'),
                work_counters=summary.get('work_counters'),
                owned_pose_buffers=[worker['memory']['after_measurements']['pose_owned_buffers'] for worker in workers])
        rows.append(dict(name=case['case']['name'], workflow=case['case']['workflow'],
            exact_all_outputs=case['exact_all_outputs'], truth_passed=case['truth_passed'],
            profile_valid=case['profile_valid'], methods=methods,
            performance=performance_case(methods, case_workers),
            profile_overhead_percent_by_block=case['profile_overhead_percent_by_block']))
    result = dict(report=path.name, report_sha256=sha(path), raw_worker_sha256=raw_hashes,
        artifact_resolution='raw_worker_sha256 keys are original jobs-directory/basename, linked to report execution_order entries by the same logical path. Explicit --worker-root is authoritative and never falls back to original absolute paths; otherwise an absent original path falls back relative to the report. Every raw result is SHA-verified before parsing.',
        correctness_and_integrity_passed=report.get('passed'), complete=report.get('complete'), design=report['design'],
        performance_policy=dict(regression_limit='candidate warm whole-pipeline median <= baseline + max(10% of baseline, 2 ms), separately in every matching order block',
            references='report frozen 91842dd CPU, frozen 91842dd Metal, and current unprofiled Metal separately; never substitute profile-on timings for unprofiled production speed',
            targeted_bottleneck='require improvement in at least two matching profile blocks, with profile overhead reported; a failed/no-improvement workload remains visible and is not averaged away',
            unavailable='missing tuning-only baselines or insufficient blocks remain unassessed, never silently passed'),
        interpretation=[
            'Exclusive stage ratios are from one actual call nearest the median proposal wall time; independent stage medians are not added.',
            'GPU copy, command interval, wait and normalization are nested in coarse NCC. Wait overlaps device time; wait minus device is not CPU busy time.',
            'Upload is shared-memory CPU memcpy on Apple Silicon, not a PCIe transfer measurement.',
            'Ideal speed bounds assume the named stage becomes free and all remaining work stays fixed; they are not measured GPU predictions.',
            'Forward/reverse fresh-process blocks reduce order bias but do not control thermal/frequency variation. Profile-on clocks have measurable overhead.',
            'Nominal output bytes and the CPU pyramid scratch proxy are not measured resident memory. Retained buffers overlap RSS and Metal allocated totals.',
            'Execution metrics are a breakdown: pyramid and batch buffer sizes are already included in coarse gpu_bytes. Upload/output/scratch component byte counts overlap pyramid_buffer_bytes; do not add them again.',
            'The 8K full-angle stress searches six upright copies. Rotated/scaled ground truth is supplied by the separate six/twelve-object scenes.',
        ], cases=rows)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output).resolve()
        if target == path:
            raise ValueError('Summary output must not overwrite the raw report')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(str(target))
    else:
        print(text)


if __name__ == '__main__':
    main()
