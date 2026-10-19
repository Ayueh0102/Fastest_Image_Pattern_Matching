"""Negative controls for exact tiled parity, stage timestamps, and archives."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_gpu_peaks as bench


class GpuPeaksHarness(unittest.TestCase):
    def fixture(self, root, profile=False, change=None, methods=None):
        poses = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
                 dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
        case = dict(name='one', truth=copy.deepcopy(poses), source='source.png', template='template.png',
                    scales=[1.], angle=180, limit=8, score=.8, overlap=.2,
                    stop_layer=0, subpixel=1, workflow='full_rescue')
        methods = methods or (['full_gpu', 'full_gpu_tiled'] if profile else list(bench.METHODS))
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'source.png': '3', 'template.png': '4'})
        manifest = dict(cases=[case], blocks=1, repeats=1, warmup_seconds=0., methods=methods, profiling=profile,
                        executions=[], complete=True, unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
        for method in methods:
            gpu = method in bench.GPU_METHODS
            metric = dict(enabled=gpu, command_buffers=1, cpu_waits=1, intermediate_readbacks=0,
                          cpu_algorithm_steps=0, gpu_bytes=300, upload_bytes=120, readback_bytes=40,
                          encode_ms=.1, wait_ms=4., device_ms=10., device_timestamp_available=True,
                          peak_mode=bench.METHODS[method]['full_gpu_peak_mode'], peak_scratch_bytes=30,
                          profile_requested=profile, profile_available=profile, profile_status='ok' if profile else 'disabled',
                          profile_samples=4 if profile else 0, profile_counter_bytes=32 if profile else 0,
                          profile_resolve_bytes=32 if profile else 0, profile_host_ms=.1 if profile else 0.,
                          profile_gpu_ms=9. if profile else None,
                          stages=[dict(name='coarse_peak', gpu_ms=6., encoders=1),
                                  dict(name='fine', gpu_ms=3., encoders=1)] if profile else [])
            row = dict(total_ms=5. if method == 'full_gpu_tiled' else 10., prepare_ms=None if gpu else 1.,
                       proposal_ms=None if gpu else 7., refine_ms=None if gpu else 1., nms_ms=None if gpu else 1.,
                       stage_timings_available=not gpu, matches=copy.deepcopy(poses), proposals=copy.deepcopy(poses),
                       frame_coarse_gpu_bytes=100, frame_fine_gpu_bytes=100, work=[], full_gpu=metric)
            if change:
                change(method, row)
            worker = dict(first=row, runs=[copy.deepcopy(row)], exact_repeat=True, construct_ms=1.,
                          final_memory=dict(peak_rss_bytes=1000000))
            jp, rp = root/(method+'.job.json'), root/(method+'.result.json')
            bench.save(jp, bench.job(case, method, 1, 0., profile))
            bench.save(rp, worker)
            manifest['executions'].append(dict(case='one', method=method, block=0, job=jp.name,
                result=rp.name, job_sha256=bench.sha(jp), result_sha256=bench.sha(rp)))
        return manifest

    def revise(self, root, manifest, method, name, operation):
        entry = next(row for row in manifest['executions'] if row['method'] == method)
        path = root/entry[name]
        data = json.loads(path.read_text())
        operation(data)
        bench.save(path, data)
        entry[name+'_sha256'] = bench.sha(path)

    def test_formal_positive_and_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root), root)
            self.assertTrue(summary['all_tiled_vs_original_gates_pass'])
            self.assertTrue(summary['all_tiled_vs_hybrid_gates_pass'])
            self.assertTrue(summary['all_tiled_exact_original'])
            self.assertTrue(summary['all_accuracy_pass'])
            self.assertEqual(summary['failed_gates'], [])
            self.assertEqual(summary['counts'], dict(fresh_processes=3, first_calls=3, measured_calls=3, gt_evaluated_calls=6))
            self.assertIsNone(summary['cases'][0]['methods']['full_gpu']['profiling']['stage_sum_ms'])

    def test_profile_separate_from_adoption_and_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True), root)
            self.assertIsNone(summary['all_tiled_vs_original_gates_pass'])
            self.assertIsNone(summary['all_tiled_vs_hybrid_gates_pass'])
            row = summary['cases'][0]['methods']['full_gpu']['profiling']
            self.assertEqual(row['stage_sum_ms']['values'], [9.])
            self.assertEqual(row['device_remainder_ms']['values'], [1.])
            self.assertEqual(row['sample_counts'], [4])
            self.assertEqual(row['available_calls'], 2)
            self.assertEqual(row['stages']['coarse_peak']['gpu_ms']['median'], 6.)

    def test_original_only_profile_can_precede_tiled_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, methods=['full_gpu']), root)
            self.assertIsNone(summary['all_tiled_exact_original'])
            self.assertTrue(summary['all_gpu_resident'])

    def test_exact_matches_and_proposals_are_required_beyond_gt_accuracy(self):
        for field, mutation in [('matches', lambda rows: rows.reverse()), ('proposals', lambda rows: rows.reverse()),
                                ('matches', lambda rows: rows[0].update(score=.9500001)),
                                ('proposals', lambda rows: rows[0].update(pos_x=50.001))]:
            def change(method, row):
                if method == 'full_gpu_tiled':
                    mutation(row[field])
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, change=change), root)
                self.assertTrue(summary['all_accuracy_pass'])
                self.assertFalse(summary['all_tiled_exact_original'])
                self.assertFalse(summary['all_tiled_vs_original_gates_pass'])
                self.assertTrue(any(row['gate'] == 'exact_ordered_'+field for row in summary['failed_gates']))

    def test_independent_gt_failure_not_hidden_by_shared_exact_results(self):
        def change(method, row):
            row['matches'][0]['pos_x'] += 10.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, change=change), root)
            self.assertTrue(summary['all_tiled_exact_original'])
            self.assertFalse(summary['all_accuracy_pass'])
            self.assertFalse(summary['all_tiled_vs_original_gates_pass'])

    def test_each_residency_contract_and_route_is_enforced(self):
        for field, value in [('enabled', False), ('command_buffers', 2), ('cpu_waits', 2),
                             ('intermediate_readbacks', 1), ('cpu_algorithm_steps', 1), ('peak_mode', 0)]:
            def change(method, row):
                if method == 'full_gpu_tiled':
                    row['full_gpu'][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, change=change), root)
                self.assertFalse(summary['all_gpu_resident'])
                self.assertFalse(summary['all_tiled_vs_original_gates_pass'])

    def test_time_and_memory_failed_gates_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            def mutate(worker):
                worker['runs'][0]['total_ms'] = 30.
                worker['final_memory']['peak_rss_bytes'] = 100000000
            self.revise(root, manifest, 'full_gpu_tiled', 'result', mutate)
            summary = bench.summarize(manifest, root)
            self.assertTrue(summary['all_accuracy_pass'])
            self.assertFalse(summary['all_tiled_vs_original_gates_pass'])
            self.assertEqual(len([row for row in summary['failed_gates'] if row['gate'] == 'time_budget']), 2)
            self.assertEqual(len([row for row in summary['failed_gates'] if row['gate'] == 'rss_budget']), 2)

    def test_first_call_is_gt_checked_but_not_timed_as_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            def mutate(worker):
                worker['first']['total_ms'] = 999.
                worker['first']['matches'][0]['angle'] += 10.
            self.revise(root, manifest, 'full_gpu_tiled', 'result', mutate)
            summary = bench.summarize(manifest, root)
            row = summary['cases'][0]['methods']['full_gpu_tiled']
            self.assertEqual(row['total_ms']['median'], 5.)
            self.assertEqual(row['first_ms'], [999.])
            self.assertFalse(row['all_gt_pass'])
            self.assertFalse(row['repeat_exact'])

    def test_unavailable_profile_preserves_null_and_status(self):
        def change(method, row):
            row['full_gpu'].update(profile_available=False, profile_status='unsupported', stages=[], profile_gpu_ms=None,
                                   profile_samples=0, profile_counter_bytes=0, profile_resolve_bytes=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, change=change), root)
            profile = summary['cases'][0]['methods']['full_gpu']['profiling']
            self.assertIsNone(profile['stage_sum_ms'])
            self.assertIsNone(profile['device_remainder_ms'])
            self.assertEqual(profile['stages'], {})
            self.assertEqual(profile['status_counts'], {'unsupported': 2})
            self.assertTrue(summary['all_gpu_resident'])

    def test_missing_device_timestamp_does_not_erase_valid_stage_intervals(self):
        def change(method, row):
            row['full_gpu'].update(device_ms=None, device_timestamp_available=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, change=change), root)
            profile = summary['cases'][0]['methods']['full_gpu']['profiling']
            self.assertEqual(profile['stage_sum_ms']['median'], 9.)
            self.assertIsNone(profile['device_remainder_ms'])

    def test_inconsistent_profile_evidence_rejected(self):
        mutations = [lambda gpu: gpu.update(profile_requested=False), lambda gpu: gpu.pop('profile_available'),
                     lambda gpu: gpu.update(profile_available=True, stages=[]),
                     lambda gpu: gpu.update(profile_available=False),
                     lambda gpu: gpu.update(profile_gpu_ms=99.), lambda gpu: gpu.update(profile_gpu_ms=None),
                     lambda gpu: gpu.update(profile_samples=8), lambda gpu: gpu.update(profile_samples=4.),
                     lambda gpu: gpu['stages'][0].update(encoders=0), lambda gpu: gpu['stages'][0].update(gpu_ms=-1.),
                     lambda gpu: gpu['stages'][0].update(name='fine'), lambda gpu: gpu.update(profile_resolve_bytes=True),
                     lambda gpu: gpu.update(peak_scratch_bytes=301), lambda gpu: gpu.update(profile_host_ms=float('nan'))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root, profile=True)
                entry = next(row for row in manifest['executions'] if row['method'] == 'full_gpu')
                worker = json.loads((root/entry['result']).read_text())
                mutation(worker['first']['full_gpu'])
                # Deliberately permit a malformed NaN worker to exercise the
                # reader rather than the archive writer's no-NaN guarantee.
                (root/entry['result']).write_text(json.dumps(worker))
                entry['result_sha256'] = bench.sha(root/entry['result'])
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_cpu_resolve_payload_does_not_share_named_gpu_allocation_scope(self):
        def change(method, row):
            row['full_gpu'].update(profile_resolve_bytes=10000, profile_counter_bytes=16384)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, change=change), root)
            measured = summary['cases'][0]['methods']['full_gpu']
            self.assertEqual(measured['retained_gpu_bytes'], 300)
            self.assertEqual(measured['measured_gpu_metrics']['profile_resolve_bytes']['median'], 10000)
            self.assertTrue(summary['all_gpu_resident'])

    def test_mixed_profile_availability_reports_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True)
            def mutate(worker):
                worker['first']['full_gpu'].update(profile_available=False, profile_status='unavailable', profile_gpu_ms=None, stages=[])
            self.revise(root, manifest, 'full_gpu', 'result', mutate)
            profile = bench.summarize(manifest, root)['cases'][0]['methods']['full_gpu']['profiling']
            self.assertEqual(profile['available_calls'], 1)
            self.assertEqual(profile['total_calls'], 2)
            self.assertEqual(profile['measured_available_calls'], 1)
            self.assertEqual(profile['stage_sum_ms']['values'], [9.])

    def test_source_and_execution_completeness_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            for invalid in [dict(manifest, complete=False), dict(manifest, unchanged=False), dict(manifest, after_sha256={}),
                            dict(manifest, blocks=2), dict(manifest, repeats=2), dict(manifest, methods=['full_gpu']),
                            dict(manifest, methods=['full_gpu','full_gpu_tiled','unknown']),
                            dict(manifest, executions=manifest['executions']*2), dict(manifest, profiling=1),
                            dict(manifest, cases=manifest['cases']*2)]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    bench.summarize(invalid, root)
            (root/manifest['executions'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)

    def test_worker_truth_leaks_setting_changes_and_bool_integer_alias_rejected(self):
        for changes in [dict(truth=[]), dict(score=.1), dict(full_gpu_peak_mode=True),
                        dict(full_gpu_profiling=0), dict(full_gpu=False), dict(repeats=2)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, 'full_gpu_tiled', 'job', lambda row: row.update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_fixture_input_integrity_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, template = root/'source.png', root/'template.png'
            source.write_bytes(b'source')
            template.write_bytes(b'template')
            manifest = dict(complete=True, unchanged=True, cases=[dict(name='one', source=str(source), template=str(template))],
                            input_sha256={str(source): bench.sha(source), str(template): bench.sha(template)})
            path = root/'manifest.json'
            bench.save(path, manifest)
            self.assertEqual(bench.fixture_cases(path, ['one'])[0]['source'], str(source))
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                bench.fixture_cases(path, ['one'])

    def test_profile_overhead_requires_identical_binary_source_and_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_root, formal_root = root/'profile', root/'formal'
            profile = self.fixture(profile_root, profile=True)
            formal = self.fixture(formal_root)
            psummary, fsummary = bench.summarize(profile, profile_root), bench.summarize(formal, formal_root)
            overhead = bench.profile_overhead(profile, psummary, formal, fsummary)
            self.assertTrue(all(row['comparable'] for row in overhead))
            self.assertEqual(overhead[0]['total_delta_percent'], 0.)
            for field, value in [('binary_sha256', {'worker': 'different'}), ('source_sha256', {'source': 'different'}),
                                 ('input_sha256', {'source.png': 'different', 'template.png': '4'})]:
                changed = dict(formal, **{field: value})
                rows = bench.profile_overhead(profile, psummary, changed, fsummary)
                self.assertTrue(all(not row['comparable'] and row['total_delta_percent'] is None for row in rows))
            with self.assertRaises(ValueError):
                bench.profile_overhead(formal, fsummary, profile, psummary)

    def test_paths_cannot_escape_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest['executions'][0]['result'] = '../outside.json'
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)


if __name__ == '__main__':
    unittest.main()
