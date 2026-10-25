"""Negative controls for the stage-44 dedup + parallel peaks harness."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_dedup_peak2 as bench
import test_parallel_peaks_harness as fixture

HYBRID, GPU = bench.HYBRID, bench.GPU
POSES = fixture.POSES
K3_FACTOR = .7  # k3 calls take 70% of the off time unless a test changes it
SCALES = 3


def add_dedup(row, method, dedup):
    gpu, enabled = method == GPU, dedup == 'k3'
    if gpu and not enabled:
        row['candidate_dedup'] = dict(top_k=0, available=False, models=[], input_candidates=None, kept_candidates=None)
        return
    models = [dict(input_candidates=10, kept_candidates=6 if enabled else 10, radius_px=8., angle_window_deg=9.,
                   filter_ms=.01 if enabled and not gpu else 0.) for _ in range(SCALES)]
    row['candidate_dedup'] = dict(top_k=3 if enabled else 0, available=True, models=models,
                                  input_candidates=10*SCALES, kept_candidates=sum(m['kept_candidates'] for m in models))


def worker(case, arm, measured, warmup, profile, factor):
    method, mode, dedup = bench.ARMS[arm]
    if mode == 'frames':
        result = fixture.frames_worker(case, method, warmup, measured)
        for row in result['runs'][1:]:
            for field in ('total_ms', 'frame_wall_ms', 'source_update_ms'):
                row[field] *= factor
        rows = result['runs']
    else:
        result = fixture.static_worker(case, method, measured, profile)
        for row in result['runs']:
            row['total_ms'] *= factor
        result['warmups_ms'] = [value*factor for value in result['warmups_ms']]
        rows = [result['first'], *result['runs']]
    for row in rows:
        add_dedup(row, method, dedup)
    return result


def reference(cases, change=None):
    jobs, results = {}, {}
    for case in cases:
        for method, path in bench.PATHS.items():
            for dedup in bench.DEDUP:
                old = bench.job(case, f'{method}_static_{dedup}', 3, 1., False, None)
                for key in ('full_gpu_peak_mode', 'full_gpu_peak_batch'):
                    old.pop(key, None)
                if method == GPU:
                    old['full_gpu_peak_mode'] = 1
                jobs[(case['name'], path, dedup)] = old
                results[(case['name'], path, dedup)] = dict(first=dict(matches=copy.deepcopy(POSES), proposals=copy.deepcopy(POSES)))
    ref = dict(sha256='r43', archive='raw-results-timing.tar.gz', jobs=jobs, results=results)
    if change:
        change(ref)
    return ref


class DedupPeak2Harness(unittest.TestCase):
    def fixture(self, root, phase='formal', cases=None, change=None, factor=None):
        cases = cases or [fixture.make_case('many128_pose'), fixture.make_case('many512_pose')]
        blocks, measured, warmup_seconds, profile = bench.stage42.phase_settings(phase)
        arms = bench.PHASE_ARMS[phase]
        framed = any(bench.ARMS[arm][1] == 'frames' for arm in arms)
        warmups = {case['name']: bench.FRAME_WARMUP_CALLS[case['name']] for case in cases} if framed else None
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'input': '3'})
        manifest = dict(version=1, stage=44, phase=phase, blocks=blocks, measured_calls=measured, static_warmup_seconds=warmup_seconds,
                        profiling=profile, frame_warmup_calls=warmups, cases=cases, arms=list(arms),
                        method_options=copy.deepcopy(bench.METHODS), dedup_options=copy.deepcopy(bench.DEDUP),
                        reference_sha256='r43', executions=[], complete=True, unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
        for case in cases:
            for block in range(blocks):
                for arm in arms:
                    warmup = warmups[case['name']] if framed else None
                    scale = factor(case['name'], arm, block) if factor else (K3_FACTOR if arm.endswith('_k3') else 1.)
                    result = worker(case, arm, measured, warmup, profile, scale)
                    if change:
                        change(case['name'], arm, block, result)
                    stem = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/(stem+'.job.json'), root/(stem+'.result.json')
                    bench.save(jp, bench.job(case, arm, measured, warmup_seconds, profile, warmup))
                    bench.save(rp, result)
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block, job=jp.name, result=rp.name,
                        job_sha256=bench.sha(jp), result_sha256=bench.sha(rp), returncode=0,
                        host=dict(utc='2026-10-25T00:00:00+00:00', load_average=[1., 1., 1.],
                                  gpu_device_utilization_percent=[0], competing_processes=[])))
        return manifest, cases

    def summary(self, change=None, factor=None, ref_change=None, cases=None, phase='formal'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cases = self.fixture(root, phase, cases, change, factor)
            return bench.summarize(manifest, root, reference(cases, ref_change))

    def gates(self, summary):
        return {(row['case'], row.get('arm'), row['gate']) for row in summary['failed_gates']}

    def test_clean_fixture_passes(self):
        summary = self.summary()
        self.assertEqual(summary['failed_gates'], [])
        self.assertTrue(summary['primary_target_met'])
        self.assertTrue(summary['all_exact_round43'] and summary['all_accuracy_pass'] and summary['all_routes_pass'])
        self.assertEqual(summary['frames_fastest'], {name: f'{GPU}_frames_k3' for name in ('many128_pose', 'many512_pose')})
        comparison = summary['cases'][0]['comparisons'][f'{GPU}_static_k3_vs_off']
        self.assertAlmostEqual(comparison['delta_percent'], -30.)

    def test_one_more_false_positive_fails(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many128_pose', f'{GPU}_frames_k3', 3):
                result['runs'][-1]['matches'].append(dict(POSES[0], pos_x=400., pos_y=300.))
        gates = self.gates(self.summary(change))
        self.assertIn(('many128_pose', f'{GPU}_frames_k3', 'independent_ground_truth'), gates)

    def test_frames_output_must_equal_static(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many512_pose', f'{HYBRID}_frames_off', 2):
                result['runs'][4]['proposals'][0]['score'] -= 1e-6
        gates = self.gates(self.summary(change))
        self.assertIn(('many512_pose', f'{HYBRID}_frames_off', 'same_arm_output_parity'), gates)

    def test_peak_mode1_is_a_route_failure(self):
        def change(name, arm, block, result):
            if arm == f'{GPU}_static_k3' and block == 2:
                for row in [result['first'], *result['runs']]:
                    row['full_gpu'].update(peak_mode=1, peak_batch_maps=1)
        gates = self.gates(self.summary(change))
        self.assertIn(('many128_pose', f'{GPU}_static_k3', 'peak_mode2_route'), gates)

    def test_dedup_counter_contract_raises(self):
        def change(name, arm, block, result):
            if arm == f'{HYBRID}_static_k3' and block == 0:
                result['runs'][0]['candidate_dedup']['top_k'] = 0
        with self.assertRaises(ValueError):
            self.summary(change)

    def test_round43_archive_difference_is_detected(self):
        def ref_change(ref):
            ref['results'][('many512_pose', 'full_gpu', 'k3')]['first']['matches'][1]['score'] += 1e-7
        gates = self.gates(self.summary(ref_change=ref_change))
        self.assertEqual({gate for gate in gates if gate[2] == 'exact_vs_round43_archive'},
                         {('many512_pose', f'{GPU}_static_k3', 'exact_vs_round43_archive'),
                          ('many512_pose', f'{GPU}_frames_k3', 'exact_vs_round43_archive')})

    def test_round43_job_difference_is_detected(self):
        def ref_change(ref):
            ref['jobs'][('many128_pose', 'hybrid', 'off')]['min_area'] += 1
        gates = self.gates(self.summary(ref_change=ref_change))
        self.assertIn(('many128_pose', f'{HYBRID}_static_off', 'job_equals_round43'), gates)

    def test_speed_target_needs_ten_percent(self):
        gates = self.gates(self.summary(factor=lambda name, arm, block: .95 if arm.endswith('_k3') else 1.))
        self.assertIn(('many128_pose', f'{GPU}_static_k3', 'primary_10_percent_improvement'), gates)
        self.assertNotIn(('many128_pose', f'{HYBRID}_static_k3', 'primary_10_percent_improvement'), gates)

    def test_one_disagreeing_block_fails_the_target(self):
        def factor(name, arm, block):
            if arm == f'{GPU}_static_k3':
                return 1.2 if block == 4 else .6
            return K3_FACTOR if arm.endswith('_k3') else 1.
        summary = self.summary(factor=factor)
        self.assertIn(('many128_pose', f'{GPU}_static_k3', 'primary_10_percent_improvement'), self.gates(summary))
        self.assertFalse(summary['primary_target_met'])

    def test_time_budget_on_secondary_case(self):
        cases = [fixture.make_case('many128_pose'), fixture.make_case('many512_pose'), fixture.make_case('controlled6', 'proposal')]
        def factor(name, arm, block):
            if name == 'controlled6' and arm == f'{GPU}_static_k3':
                return 1.5 if block == 1 else .9  # 9.45 ms against 6.3 ms + 2 ms
            return K3_FACTOR if arm.endswith('_k3') else 1.
        gates = {(row['case'], row.get('arm'), row['gate'], row.get('block')) for row in self.summary(factor=factor, cases=cases)['failed_gates']}
        self.assertIn(('controlled6', f'{GPU}_static_k3', 'time_budget', 1), gates)

    def test_rss_budget(self):
        def change(name, arm, block, result):
            if (name, arm) == ('many128_pose', f'{HYBRID}_frames_k3'):
                result['final_memory']['peak_rss_bytes'] += 30*1024*1024
        self.assertIn(('many128_pose', f'{HYBRID}_frames_k3', 'rss_budget'), self.gates(self.summary(change)))

    def test_fastest_needs_every_block(self):
        def factor(name, arm, block):
            if name == 'many128_pose' and arm == f'{HYBRID}_frames_k3' and block == 5:
                return .3
            return K3_FACTOR if arm.endswith('_k3') else 1.
        fastest = self.summary(factor=factor)['frames_fastest']
        self.assertIsNone(fastest['many128_pose'])
        self.assertEqual(fastest['many512_pose'], f'{GPU}_frames_k3')

    def test_changed_evidence_and_missing_worker_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cases = self.fixture(root)
            (root/manifest['executions'][5]['result']).write_text('{}')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root, reference(cases))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cases = self.fixture(root)
            del manifest['executions'][7]
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root, reference(cases))

    def test_jobs_carry_peak2_and_dedup(self):
        case = fixture.make_case()
        gpu = bench.job(case, f'{GPU}_frames_k3', 3, 1., False, 2)
        self.assertEqual((gpu['full_gpu_peak_mode'], gpu['full_gpu_peak_batch'], gpu['candidate_dedup_top_k']), (2, 8, 3))
        self.assertNotIn('truth', json.dumps(gpu))
        hybrid = bench.job(case, f'{HYBRID}_static_off', 3, 1., False, None)
        self.assertEqual((hybrid['full_gpu'], hybrid['candidate_dedup_top_k']), (False, 0))
        gt = bench.gt_job(dict(case, paths=('hybrid', 'full_gpu')), 'k3')
        old = bench.stage43.make_job(dict(case, paths=('hybrid', 'full_gpu')), 'full_gpu', 'k3', 1, 0.)
        self.assertEqual(bench.comparable(gt), bench.comparable(old))
        self.assertEqual((gt['full_gpu_peak_mode'], old['full_gpu_peak_mode']), (2, 1))


class DedupPeak2GtHarness(unittest.TestCase):
    def gt(self, change=None, ref_change=None, failed=None):
        cases = [dict(fixture.make_case(name), family='r40', role='pass', paths=('hybrid', 'full_gpu')) for name in ('alpha', 'beta')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'input': '3'})
            manifest = dict(version=1, stage=44, phase='gt', cases=cases, arms=list(bench.DEDUP), method_options=copy.deepcopy(bench.METHODS),
                            dedup_options=copy.deepcopy(bench.DEDUP), reference_sha256='r43gt', executions=[], complete=True,
                            unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
            ref = dict(sha256='r43gt', archive='raw-results-gt.tar.gz', jobs={}, results={})
            for case in cases:
                for dedup in bench.DEDUP:
                    rows = [fixture.record(GPU, False, 9., 0.) for _ in range(2)]
                    for row in rows:
                        add_dedup(row, GPU, dedup)
                        for pose in row['matches']+row['proposals']:
                            pose['corners'] = [[pose['pos_x']-5, pose['pos_y']-5], [pose['pos_x']+5, pose['pos_y']+5]]
                    result = dict(first=rows[0], runs=rows[1:], exact_repeat=True, warmups_ms=[],
                                  final_memory=dict(current_rss_bytes=1, peak_rss_bytes=2))
                    old = bench.stage43.make_job(case, 'full_gpu', dedup, 1, 0.)
                    ref['jobs'][(case['name'], 'full_gpu', dedup)] = old
                    ref['results'][(case['name'], 'full_gpu', dedup)] = copy.deepcopy(result)
                    if change:
                        change(case['name'], dedup, result)
                    label = f"{case['name']}-full_gpu-{dedup}"
                    jp, rp = root/(label+'.json'), root/(label+'.result.json')
                    bench.save(jp, bench.gt_job(case, dedup))
                    entry = dict(case=case['name'], arm=dedup, block=0, job=jp.name, job_sha256=bench.sha(jp), returncode=0)
                    if failed == (case['name'], dedup):
                        entry.update(returncode=1, stderr_tail='Metal unavailable')
                    else:
                        bench.save(rp, result)
                        entry.update(result=rp.name, result_sha256=bench.sha(rp))
                    manifest['executions'].append(entry)
            if ref_change:
                ref_change(ref)
            return bench.summarize_gt(manifest, root, ref)

    def gates(self, summary):
        return {(row['case'], row['arm'], row['gate']) for row in summary['failures']}

    def test_clean_gt_passes(self):
        summary = self.gt()
        self.assertEqual(summary['failures'], [])
        self.assertTrue(summary['gt_neutral'] and summary['all_exact_round43'] and summary['all_jobs_equal_round43'])

    def test_lost_object_is_a_regression(self):
        def change(name, dedup, result):
            if (name, dedup) == ('beta', 'k3'):
                for row in [result['first'], *result['runs']]:
                    row['matches'] = row['matches'][:1]
        summary = self.gt(change)
        self.assertIn(('beta', 'k3', 'gt_regression'), self.gates(summary))
        self.assertFalse(summary['gt_neutral'])

    def test_dedup_count_difference_from_archive(self):
        def ref_change(ref):
            ref['results'][('alpha', 'full_gpu', 'k3')]['first']['candidate_dedup']['kept_candidates'] += 1
        self.assertEqual(self.gates(self.gt(ref_change=ref_change)), {('alpha', 'k3', 'exact_vs_round43_archive')})

    def test_peak_mode1_in_gt_phase(self):
        def change(name, dedup, result):
            if (name, dedup) == ('alpha', 'off'):
                result['runs'][0]['full_gpu'].update(peak_mode=1, peak_batch_maps=1)
        self.assertIn(('alpha', 'off', 'peak_mode2_route'), self.gates(self.gt(change)))

    def test_failed_worker_is_not_neutral(self):
        summary = self.gt(failed=('alpha', 'k3'))
        self.assertIn(('alpha', 'k3', 'worker'), {(row['case'], row['arm'], row['gate']) for row in summary['failures']})
        self.assertFalse(summary['gt_neutral'])


if __name__ == '__main__':
    unittest.main()
