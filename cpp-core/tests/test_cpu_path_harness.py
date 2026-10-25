"""Negative controls for the stage-45 full CPU path harness."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_cpu_path as bench
import test_dedup_peak2_harness as round44
import test_parallel_peaks_harness as fixture

CPU, HYBRID, GPU = bench.CPU, bench.HYBRID, bench.GPU
POSES = fixture.POSES
CPU_FACTOR = 5.  # CPU frames take five times the hybrid fixture time unless a test changes it


def to_cpu(row):
    active = row['refinement_candidates'] > 0
    row.update(frame_coarse_gpu_bytes=0, frame_fine_gpu_bytes=0, pose_gpu_bytes=0, refinement_dispatches=0,
               refinement_command_buffers=0, selected_refinement_backend=0, configured_refinement_backend=0, pose_storage_mode=0,
               pose_host_bytes=3332652 if active else 0,
               work=[dict(coarse_commands=0, fine_commands=0, coarse_gpu_bytes=0, fine_maps=5) for _ in range(round44.SCALES)])
    return row


def worker(case, arm, measured, warmup, factor):
    method, dedup = bench.ARMS[arm]
    proxy = f'{HYBRID if method == CPU else method}_frames_{dedup}'
    result = round44.worker(case, proxy, measured, warmup, False, factor)
    if method == CPU:
        for row in result['runs']:
            to_cpu(row)
    return result


def default_factor(name, arm, block):
    method, dedup = bench.ARMS[arm]
    return (CPU_FACTOR if method == CPU else 1.)*(round44.K3_FACTOR if dedup == 'k3' else 1.)


class CpuPathHarness(unittest.TestCase):
    def fixture(self, root, cases=None, change=None, factor=default_factor):
        cases = cases or [fixture.make_case('many128_pose'), fixture.make_case('many512_pose')]
        blocks, measured = bench.PHASE_SETTINGS['formal']
        warmups = {case['name']: bench.FRAME_WARMUP_CALLS[case['name']] for case in cases}
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'input': '3'})
        manifest = dict(version=1, stage=45, phase='formal', blocks=blocks, measured_calls=measured, frame_warmup_calls=warmups,
                        cases=cases, arms=list(bench.ARMS), method_options=copy.deepcopy(bench.METHODS),
                        dedup_options=copy.deepcopy(bench.DEDUP), reference_sha256='r43', executions=[], complete=True,
                        unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
        for case in cases:
            for block in range(blocks):
                for arm in bench.ARMS:
                    result = worker(case, arm, measured, warmups[case['name']], factor(case['name'], arm, block))
                    if change:
                        change(case['name'], arm, block, result)
                    stem = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/(stem+'.job.json'), root/(stem+'.result.json')
                    bench.save(jp, bench.job(case, arm, measured, warmups[case['name']]))
                    bench.save(rp, result)
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block, job=jp.name, result=rp.name,
                        job_sha256=bench.sha(jp), result_sha256=bench.sha(rp), returncode=0,
                        host=dict(utc='2026-10-25T00:00:00+00:00', load_average=[1., 1., 1.],
                                  gpu_device_utilization_percent=[0], competing_processes=[])))
        return manifest, cases

    def summary(self, change=None, factor=default_factor, ref_change=None, cases=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cases = self.fixture(root, cases, change, factor)
            return bench.summarize(manifest, root, round44.reference(cases, ref_change))

    def gates(self, summary):
        return {(row['case'], row.get('arm'), row['gate']) for row in summary['failed_gates']}

    def test_clean_fixture_passes(self):
        summary = self.summary()
        self.assertEqual(summary['failed_gates'], [])
        self.assertTrue(summary['all_accuracy_pass'] and summary['all_routes_pass'] and summary['all_exact_round43'])
        self.assertEqual(summary['frames_fastest'], {'many128_pose': f'{GPU}_frames_k3', 'many512_pose': f'{GPU}_frames_k3'})
        comparison = summary['cases'][0]['comparisons'][f'{GPU}_frames_k3_vs_{CPU}_frames_off']
        self.assertTrue(comparison['consistent_direction'] and comparison['delta_percent'] < -80)
        self.assertEqual(summary['cpu_outputs_equal_round43_hybrid']['many128_pose'], dict(off=True, k3=True))

    def test_cpu_call_that_used_metal_fails_route(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many128_pose', f'{CPU}_frames_k3', 2):
                result['runs'][-1]['pose_gpu_bytes'] = 4096
        self.assertIn(('many128_pose', f'{CPU}_frames_k3', 'route'), self.gates(self.summary(change)))

    def test_cpu_work_row_with_metal_commands_fails_route(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many512_pose', f'{CPU}_frames_off', 4):
                result['runs'][3]['work'][1]['fine_commands'] = 1
        self.assertIn(('many512_pose', f'{CPU}_frames_off', 'route'), self.gates(self.summary(change)))

    def test_cpu_refinement_on_metal_fails_route(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many128_pose', f'{CPU}_frames_off', 0):
                for row in result['runs']:
                    row['selected_refinement_backend'] = 1
        self.assertIn(('many128_pose', f'{CPU}_frames_off', 'route'), self.gates(self.summary(change)))

    def test_one_more_false_positive_on_cpu_fails(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many512_pose', f'{CPU}_frames_k3', 5):
                result['runs'][-1]['matches'].append(dict(POSES[0], pos_x=400., pos_y=300.))
        self.assertIn(('many512_pose', f'{CPU}_frames_k3', 'independent_ground_truth'), self.gates(self.summary(change)))

    def test_hybrid_archive_difference_is_a_gate_cpu_difference_is_not(self):
        def ref_change(ref):
            ref['results'][('many128_pose', 'hybrid', 'off')]['first']['proposals'][0]['score'] += 1e-7
        summary = self.summary(ref_change=ref_change)
        self.assertEqual({gate for gate in self.gates(summary) if gate[2] == 'exact_vs_round43_archive'},
                         {('many128_pose', f'{HYBRID}_frames_off', 'exact_vs_round43_archive')})
        self.assertFalse(summary['cpu_outputs_equal_round43_hybrid']['many128_pose']['off'])

    def test_cpu_output_change_in_one_block_breaks_cross_block_identity(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many128_pose', f'{CPU}_frames_off', 3):
                for row in result['runs']:
                    row['matches'][1]['score'] -= 1e-6
        self.assertIn(('many128_pose', f'{CPU}_frames_off', 'cross_block_identity'), self.gates(self.summary(change)))

    def test_fastest_needs_every_block(self):
        def factor(name, arm, block):
            if (name, arm, block) == ('many512_pose', f'{HYBRID}_frames_k3', 1):
                return .2
            return default_factor(name, arm, block)
        fastest = self.summary(factor=factor)['frames_fastest']
        self.assertIsNone(fastest['many512_pose'])
        self.assertEqual(fastest['many128_pose'], f'{GPU}_frames_k3')

    def test_short_warmup_is_a_gate(self):
        def change(name, arm, block, result):
            if (name, arm, block) == ('many512_pose', f'{CPU}_frames_k3', 0):
                for index, row in enumerate(result['runs']):
                    row['elapsed_ms'] = 300.*(index+1)  # still later than decode plus frame wall; 3 warm-ups = 900 ms
        self.assertIn(('many512_pose', f'{CPU}_frames_k3', 'warmup_shorter_than_1s'), self.gates(self.summary(change)))

    def test_rss_budget(self):
        def change(name, arm, block, result):
            if (name, arm) == ('many128_pose', f'{CPU}_frames_k3'):
                result['final_memory']['peak_rss_bytes'] += 40*1024*1024
        self.assertIn(('many128_pose', f'{CPU}_frames_k3', 'rss_budget'), self.gates(self.summary(change)))

    def test_changed_evidence_and_missing_worker_raise(self):
        for damage in ('result', 'missing'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, cases = self.fixture(root)
                if damage == 'result':
                    (root/manifest['executions'][3]['result']).write_text('{}')
                else:
                    del manifest['executions'][9]
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root, round44.reference(cases))

    def test_job_contract(self):
        case = fixture.make_case()
        cpu = bench.job(case, f'{CPU}_frames_k3', 3, 6)
        self.assertEqual({key: cpu[key] for key in bench.CPU_OPTIONS}, bench.CPU_OPTIONS)
        self.assertEqual((cpu['backend'], cpu['refinement_backend'], cpu['fine_fused'], cpu['candidate_dedup_top_k']), (0, 0, False, 3))
        for method in (HYBRID, GPU):
            self.assertEqual(bench.job(case, f'{method}_frames_off', 3, 6),
                             bench.stage44.job(case, f'{method}_frames_off', 3, 1., False, 6))
        self.assertNotIn('truth', json.dumps(cpu))


class CpuPathGtHarness(unittest.TestCase):
    def gt(self, change=None, ref_change=None, failed=None, inputs_change=None):
        cases = [dict(fixture.make_case(name), family='r40', role='pass', paths=('hybrid', 'full_gpu')) for name in ('alpha', 'beta')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'},
                          input_sha256={path: 'h-'+Path(path).name for case in cases for path in (case['source'], case['template'])})
            inputs43 = {Path(path).name: digest for path, digest in hashes['input_sha256'].items()}
            manifest = dict(version=1, stage=45, phase='gt', cases=cases, arms=list(bench.DEDUP), method_options=copy.deepcopy(bench.METHODS),
                            dedup_options=copy.deepcopy(bench.DEDUP), reference_sha256='r43gt', executions=[], complete=True,
                            unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
            ref = dict(sha256='r43gt', archive='raw-results-gt.tar.gz', jobs={}, results={})
            for case in cases:
                for dedup in bench.DEDUP:
                    rows = [fixture.record(HYBRID, True, 9., 0.) for _ in range(2)]
                    for row in rows:
                        round44.add_dedup(row, HYBRID, dedup)
                        for pose in row['matches']+row['proposals']:
                            pose['corners'] = [[pose['pos_x']-5, pose['pos_y']-5], [pose['pos_x']+5, pose['pos_y']+5]]
                    result = dict(first=rows[0], runs=rows[1:], exact_repeat=True, warmups_ms=[],
                                  final_memory=dict(current_rss_bytes=1, peak_rss_bytes=2))
                    ref['jobs'][(case['name'], 'hybrid', dedup)] = bench.stage43.make_job(case, 'hybrid', dedup, 1, 0.)
                    ref['results'][(case['name'], 'hybrid', dedup)] = copy.deepcopy(result)
                    for row in rows:
                        to_cpu(row)
                    if change:
                        change(case['name'], dedup, result)
                    label = f"{case['name']}-{CPU}-{dedup}"
                    jp, rp = root/(label+'.json'), root/(label+'.result.json')
                    bench.save(jp, bench.gt_job(case, dedup))
                    entry = dict(case=case['name'], arm=dedup, block=0, job=jp.name, job_sha256=bench.sha(jp), returncode=0)
                    if failed == (case['name'], dedup):
                        entry.update(returncode=1, stderr_tail='timeout')
                    else:
                        bench.save(rp, result)
                        entry.update(result=rp.name, result_sha256=bench.sha(rp))
                    manifest['executions'].append(entry)
            if ref_change:
                ref_change(ref)
            if inputs_change:
                inputs_change(inputs43)
            return bench.summarize_gt(manifest, root, ref, inputs43)

    def gates(self, summary):
        return {(row['case'], row['arm'], row['gate']) for row in summary['failures']}

    def test_clean_gt_passes(self):
        summary = self.gt()
        self.assertEqual(summary['failures'], [])
        self.assertTrue(summary['gt_neutral'] and summary['all_cpu_routes'] and summary['all_jobs_equal_round43'])
        self.assertEqual(summary['cpu_versus_round43_hybrid']['outputs_identical'], 4)

    def test_lost_object_is_a_regression(self):
        def change(name, dedup, result):
            if (name, dedup) == ('alpha', 'k3'):
                for row in [result['first'], *result['runs']]:
                    row['matches'] = row['matches'][:1]
        summary = self.gt(change)
        self.assertIn(('alpha', 'k3', 'gt_regression'), self.gates(summary))
        self.assertFalse(summary['gt_neutral'])
        self.assertEqual(summary['cpu_versus_round43_hybrid']['fewer_tp'], [dict(case='alpha', arm='k3', delta=-1)])

    def test_metal_use_in_gt_phase(self):
        def change(name, dedup, result):
            if (name, dedup) == ('beta', 'off'):
                result['runs'][0]['frame_fine_gpu_bytes'] = 1
        self.assertIn(('beta', 'off', 'cpu_route'), self.gates(self.gt(change)))

    def test_inputs_compared_by_content_not_path(self):
        def moved(ref):
            for key, job in ref['jobs'].items():
                job['source'] = '/elsewhere/'+Path(job['source']).name
        self.assertEqual(self.gt(ref_change=moved)['failures'], [])
        def different(inputs43):
            inputs43['alpha.png'] = 'other'
        self.assertEqual(self.gates(self.gt(inputs_change=different)), {('alpha', 'off', 'job_equals_round43'), ('alpha', 'k3', 'job_equals_round43')})

    def test_failed_worker_is_not_neutral(self):
        summary = self.gt(failed=('beta', 'k3'))
        self.assertIn(('beta', 'k3', 'worker'), self.gates(summary))
        self.assertFalse(summary['gt_neutral'])


if __name__ == '__main__':
    unittest.main()
