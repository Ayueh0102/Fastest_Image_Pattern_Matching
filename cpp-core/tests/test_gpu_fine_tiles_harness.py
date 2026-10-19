"""Negative controls for exact tiled parity, stage timestamps, and archives."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_gpu_fine_tiles as bench


class GpuFineTilesHarness(unittest.TestCase):
    def fixture(self, root, profile=False, change=None, methods=None):
        poses = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
                 dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
        case = dict(name='one', truth=copy.deepcopy(poses), source='source.png', template='template.png',
                    scales=[1.], angle=180, limit=8, score=.8, overlap=.2,
                    stop_layer=0, subpixel=1, workflow='full_rescue')
        methods = methods or (['full_gpu_tiled', 'full_gpu_fine_tiled'] if profile else list(bench.METHODS))
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'source.png': '3', 'template.png': '4'})
        manifest = dict(cases=[case], blocks=1, repeats=1, warmup_seconds=0., methods=methods, method_options={name: bench.METHODS[name] for name in methods}, profiling=profile,
                        executions=[], complete=True, unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
        for method in methods:
            gpu = method in bench.GPU_METHODS
            fine_mode = bench.METHODS[method]['full_gpu_fine_mode']
            fine_work = [dict(model=0, layer=0, template_width=175, template_height=128, roi_cache_bytes=16384,
                              reached_candidates=2, angles=6, cached_angles=0,
                              direct_angles=0 if fine_mode else 6, tiled_angles=6 if fine_mode else 0,
                              source_samples=6*181*(128+6*8) if fine_mode else 6*49*175*128,
                              tile_loads=48 if fine_mode else 0),
                         dict(model=0, layer=1, template_width=87, template_height=64, roi_cache_bytes=16384,
                              reached_candidates=0, angles=0, cached_angles=0, direct_angles=0, tiled_angles=0,
                              source_samples=0, tile_loads=0)]
            metric = dict(fine_mode=fine_mode, fine_work_available=profile, fine_work=fine_work if profile else [],
                          fine_diagnostic_bytes=64 if profile else 0, fine_diagnostic_readback_bytes=32 if profile else 0,
                          fine_threadgroup_bytes=16384, enabled=gpu, command_buffers=1, cpu_waits=1, intermediate_readbacks=0,
                          cpu_algorithm_steps=0, gpu_bytes=300, upload_bytes=120, readback_bytes=40,
                          encode_ms=.1, wait_ms=4., device_ms=10., device_timestamp_available=True,
                          peak_mode=bench.METHODS[method]['full_gpu_peak_mode'], peak_scratch_bytes=30,
                          profile_requested=profile, profile_available=profile, profile_status='ok' if profile else 'disabled',
                          profile_samples=4 if profile else 0, profile_counter_bytes=32 if profile else 0,
                          profile_resolve_bytes=32 if profile else 0, profile_host_ms=.1 if profile else 0.,
                          profile_gpu_ms=9. if profile else None,
                          stages=[dict(name='coarse_peak', gpu_ms=6., encoders=1),
                                  dict(name='fine', gpu_ms=3., encoders=1)] if profile else [])
            row = dict(total_ms=5. if method == 'full_gpu_fine_tiled' else 10., prepare_ms=None if gpu else 1.,
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
            self.assertTrue(summary['all_fine_tiled_vs_baseline_gates_pass'])
            self.assertTrue(summary['all_fine_tiled_vs_hybrid_gates_pass'])
            self.assertTrue(summary['all_fine_tiled_exact_baseline'])
            self.assertTrue(summary['all_accuracy_pass'])
            self.assertEqual(summary['failed_gates'], [])
            self.assertEqual(summary['counts'], dict(fresh_processes=3, first_calls=3, measured_calls=3, gt_evaluated_calls=6))
            self.assertIsNone(summary['cases'][0]['methods']['full_gpu_tiled']['profiling']['stage_sum_ms'])

    def test_profile_separate_from_adoption_and_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True), root)
            self.assertIsNone(summary['all_fine_tiled_vs_baseline_gates_pass'])
            self.assertIsNone(summary['all_fine_tiled_vs_hybrid_gates_pass'])
            row = summary['cases'][0]['methods']['full_gpu_tiled']['profiling']
            self.assertEqual(row['stage_sum_ms']['values'], [9.])
            self.assertEqual(row['device_remainder_ms']['values'], [1.])
            self.assertEqual(row['sample_counts'], [4])
            self.assertEqual(row['available_calls'], 2)
            self.assertEqual(row['stages']['coarse_peak']['gpu_ms']['median'], 6.)

    def test_baseline_only_profile_can_precede_fine_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, methods=['full_gpu_tiled']), root)
            self.assertIsNone(summary['all_fine_tiled_exact_baseline'])
            self.assertTrue(summary['all_gpu_resident'])

    def test_exact_matches_and_proposals_are_required_beyond_gt_accuracy(self):
        for field, mutation in [('matches', lambda rows: rows.reverse()), ('proposals', lambda rows: rows.reverse()),
                                ('matches', lambda rows: rows[0].update(score=.9500001)),
                                ('proposals', lambda rows: rows[0].update(pos_x=50.001))]:
            def change(method, row):
                if method == 'full_gpu_fine_tiled':
                    mutation(row[field])
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, change=change), root)
                self.assertTrue(summary['all_accuracy_pass'])
                self.assertFalse(summary['all_fine_tiled_exact_baseline'])
                self.assertFalse(summary['all_fine_tiled_vs_baseline_gates_pass'])
                self.assertTrue(any(row['gate'] == 'exact_ordered_'+field for row in summary['failed_gates']))

    def test_independent_gt_failure_not_hidden_by_shared_exact_results(self):
        def change(method, row):
            row['matches'][0]['pos_x'] += 10.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, change=change), root)
            self.assertTrue(summary['all_fine_tiled_exact_baseline'])
            self.assertFalse(summary['all_accuracy_pass'])
            self.assertFalse(summary['all_fine_tiled_vs_baseline_gates_pass'])

    def test_each_residency_contract_and_route_is_enforced(self):
        for field, value in [('enabled', False), ('command_buffers', 2), ('cpu_waits', 2),
                             ('intermediate_readbacks', 1), ('cpu_algorithm_steps', 1), ('peak_mode', 0), ('fine_mode', 0)]:
            def change(method, row):
                if method == 'full_gpu_fine_tiled':
                    row['full_gpu'][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, change=change), root)
                self.assertFalse(summary['all_gpu_resident'])
                self.assertFalse(summary['all_fine_tiled_vs_baseline_gates_pass'])

    def test_time_and_memory_failed_gates_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            def mutate(worker):
                worker['runs'][0]['total_ms'] = 30.
                worker['final_memory']['peak_rss_bytes'] = 100000000
            self.revise(root, manifest, 'full_gpu_fine_tiled', 'result', mutate)
            summary = bench.summarize(manifest, root)
            self.assertTrue(summary['all_accuracy_pass'])
            self.assertFalse(summary['all_fine_tiled_vs_baseline_gates_pass'])
            self.assertEqual(len([row for row in summary['failed_gates'] if row['gate'] == 'time_budget']), 2)
            self.assertEqual(len([row for row in summary['failed_gates'] if row['gate'] == 'rss_budget']), 2)

    def test_first_call_is_gt_checked_but_not_timed_as_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            def mutate(worker):
                worker['first']['total_ms'] = 999.
                worker['first']['matches'][0]['angle'] += 10.
            self.revise(root, manifest, 'full_gpu_fine_tiled', 'result', mutate)
            summary = bench.summarize(manifest, root)
            row = summary['cases'][0]['methods']['full_gpu_fine_tiled']
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
            profile = summary['cases'][0]['methods']['full_gpu_tiled']['profiling']
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
            profile = summary['cases'][0]['methods']['full_gpu_tiled']['profiling']
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
                entry = next(row for row in manifest['executions'] if row['method'] == 'full_gpu_tiled')
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
            measured = summary['cases'][0]['methods']['full_gpu_tiled']
            self.assertEqual(measured['retained_gpu_bytes'], 300)
            self.assertEqual(measured['measured_gpu_metrics']['profile_resolve_bytes']['median'], 10000)
            self.assertTrue(summary['all_gpu_resident'])

    def test_mixed_profile_availability_reports_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True)
            def mutate(worker):
                worker['first']['full_gpu'].update(profile_available=False, profile_status='unavailable', profile_gpu_ms=None, stages=[])
            self.revise(root, manifest, 'full_gpu_tiled', 'result', mutate)
            profile = bench.summarize(manifest, root)['cases'][0]['methods']['full_gpu_tiled']['profiling']
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
                            dict(manifest, methods=['full_gpu_tiled','full_gpu_fine_tiled','unknown']),
                            dict(manifest, executions=manifest['executions']*2), dict(manifest, profiling=1),
                            dict(manifest, cases=manifest['cases']*2)]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    bench.summarize(invalid, root)
            (root/manifest['executions'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)

    def test_worker_truth_leaks_setting_changes_and_bool_integer_alias_rejected(self):
        for changes in [dict(truth=[]), dict(score=.1), dict(full_gpu_peak_mode=True), dict(full_gpu_fine_mode=True), dict(full_gpu_fine_mode=0),
                        dict(full_gpu_profiling=0), dict(full_gpu=False), dict(repeats=2)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, 'full_gpu_fine_tiled', 'job', lambda row: row.update(changes))
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

    def test_only_fine_mode_differs_between_gpu_worker_jobs(self):
        case = dict(name='one', source='a', template='b', scales=[1.], angle=180,
                    limit=8, score=.8, overlap=.2, stop_layer=0, subpixel=0,
                    workflow='core_only', truth=[dict(pos_x=0, pos_y=0, angle=0, scale=1)])
        old = bench.job(case, 'full_gpu_tiled', 2, 1.)
        new = bench.job(case, 'full_gpu_fine_tiled', 2, 1.)
        self.assertEqual({key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}, {'full_gpu_fine_mode'})
        self.assertNotIn('truth', old)

    def test_formal_diagnostics_are_unavailable_not_zero_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root), root)
            self.assertIsNone(summary['cases'][0]['fine_work_comparison'])
            for method in bench.GPU_METHODS:
                row = summary['cases'][0]['methods'][method]['fine_work']
                self.assertIsNone(row['measured_totals'])
                self.assertIsNone(row['direct_angle_fraction'])
                self.assertEqual(row['layers'], [])
                self.assertEqual(row['available_calls'], 0)

    def test_measured_diagnostics_include_zero_work_levels_and_exact_sample_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True), root)
            row = summary['cases'][0]
            work = row['methods']['full_gpu_fine_tiled']['fine_work']
            self.assertEqual(work['measured_totals']['source_samples']['values'], [191136])
            self.assertEqual(work['measured_totals']['tile_loads']['values'], [48])
            self.assertEqual(work['measured_totals']['reached_candidates']['values'], [2])
            self.assertEqual(work['tiled_angle_fraction']['median'], 1.)
            self.assertEqual(work['layers'][1]['counts']['angles']['values'], [0])
            self.assertTrue(work['stable_measured_counts'])
            self.assertTrue(row['fine_work_comparison']['same_measured_workload'])
            self.assertAlmostEqual(row['fine_work_comparison']['source_sampler_calls_ratio'], 191136/(6*49*175*128))
            self.assertEqual(summary['failed_gates'], [])

    def test_fine_work_remains_available_when_timestamps_are_unsupported(self):
        def change(method, row):
            row['full_gpu'].update(profile_available=False, profile_status='unsupported', stages=[], profile_gpu_ms=None,
                                   profile_samples=0, profile_counter_bytes=0, profile_resolve_bytes=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, change=change), root)
            method = summary['cases'][0]['methods']['full_gpu_fine_tiled']
            self.assertEqual(method['profiling']['available_calls'], 0)
            self.assertEqual(method['fine_work']['available_calls'], 2)
            self.assertIsNotNone(method['fine_work']['measured_totals'])
            self.assertTrue(summary['all_gpu_resident'])

    def test_malformed_fine_diagnostics_are_rejected(self):
        mutations = [lambda gpu: gpu.update(fine_work_available=False),
                     lambda gpu: gpu.update(fine_work_available=1),
                     lambda gpu: gpu.update(fine_mode=True),
                     lambda gpu: gpu.update(fine_mode=2),
                     lambda gpu: gpu.update(fine_work=None),
                     lambda gpu: gpu.update(fine_diagnostic_readback_bytes=31),
                     lambda gpu: gpu.update(fine_diagnostic_bytes=31),
                     lambda gpu: gpu.update(fine_diagnostic_bytes=301),
                     lambda gpu: gpu.update(fine_threadgroup_bytes=True),
                     lambda gpu: gpu['fine_work'][0].update(angles=5),
                     lambda gpu: gpu['fine_work'][0].update(reached_candidates=0),
                     lambda gpu: gpu['fine_work'][0].update(source_samples=191135),
                     lambda gpu: gpu['fine_work'][0].update(roi_cache_bytes=1),
                     lambda gpu: gpu['fine_work'][0].update(template_width=0),
                     lambda gpu: gpu['fine_work'][0].update(layer=True),
                     lambda gpu: gpu['fine_work'][1].update(layer=0),
                     lambda gpu: gpu['fine_work'][1].update(tile_loads=1),
                     lambda gpu: gpu['fine_work'][0].update(tile_loads=47, source_samples=190050),
                     lambda gpu: gpu.update(fine_mode=0)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root, profile=True)
                def mutate(worker):
                    mutation(worker['first']['full_gpu'])
                self.revise(root, manifest, 'full_gpu_fine_tiled', 'result', mutate)
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_formal_cannot_contain_diagnostic_readback(self):
        for changes in [dict(fine_work_available=True), dict(fine_diagnostic_readback_bytes=16),
                        dict(fine_work=[{}])]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, 'full_gpu_fine_tiled', 'result',
                            lambda worker: worker['first']['full_gpu'].update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_different_reached_workload_suppresses_sampler_reduction_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True)
            def mutate(worker):
                for record in [worker['first'], *worker['runs']]:
                    record['full_gpu']['fine_work'][0]['reached_candidates'] = 3
            self.revise(root, manifest, 'full_gpu_fine_tiled', 'result', mutate)
            compare = bench.summarize(manifest, root)['cases'][0]['fine_work_comparison']
            self.assertFalse(compare['same_measured_workload'])
            self.assertIsNone(compare['source_sampler_calls_delta_percent'])

    def test_first_call_diagnostics_do_not_enter_measured_work_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True)
            def mutate(worker):
                for field in bench.FINE_COUNT_FIELDS:
                    worker['first']['full_gpu']['fine_work'][0][field] *= 2
            self.revise(root, manifest, 'full_gpu_fine_tiled', 'result', mutate)
            work = bench.summarize(manifest, root)['cases'][0]['methods']['full_gpu_fine_tiled']['fine_work']
            self.assertEqual(work['measured_totals']['source_samples']['values'], [191136])
            self.assertEqual(work['available_calls'], 2)

    def test_primary_ten_percent_target_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest['cases'][0]['name'] = 'many128_pose'
            for entry in manifest['executions']:
                entry['case'] = 'many128_pose'
            self.revise(root, manifest, 'full_gpu_fine_tiled', 'result',
                        lambda worker: worker['runs'][0].update(total_ms=9.5))
            summary = bench.summarize(manifest, root)
            compare = summary['cases'][0]['fine_tiled_comparisons']['full_gpu_tiled']
            self.assertTrue(compare['candidate_for_adoption'])
            self.assertFalse(compare['primary_10_percent_improvement'])
            self.assertEqual([row['gate'] for row in summary['failed_gates']], ['primary_10_percent_improvement'])

    def test_manifest_method_options_are_strictly_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest['method_options'] = copy.deepcopy(manifest['method_options'])
            manifest['method_options']['full_gpu_fine_tiled']['full_gpu_fine_mode'] = True
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)

    def test_cache_boundary_partial_last_strip_and_wide_fallback(self):
        fixtures = [(87, 64, 16384, 'cached_angles'),
                    (175, 129, 16384, 'tiled_angles'),
                    (1200, 100, 16384, 'tiled_angles'),
                    (600, 20, 4096, 'direct_angles'),
                    (600, 20, 4242, 'tiled_angles')]
        for width, height, capacity, category in fixtures:
            with self.subTest(width=width, height=height, capacity=capacity), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root, profile=True)
                tile_rows = min(16, capacity//(width+6)-6)
                tile_loads = 6*((height+tile_rows-1)//tile_rows) if category == 'tiled_angles' else 0
                samples = (6*(width+6)*(height+6) if category == 'cached_angles' else
                           6*49*width*height if category == 'direct_angles' else
                           (width+6)*(height*6+tile_loads*6))
                def mutate(worker):
                    for record in [worker['first'], *worker['runs']]:
                        level = record['full_gpu']['fine_work'][0]
                        level.update(template_width=width, template_height=height, roi_cache_bytes=capacity,
                                     cached_angles=0, tiled_angles=0, direct_angles=0,
                                     tile_loads=tile_loads, source_samples=samples)
                        level[category] = 6
                self.revise(root, manifest, 'full_gpu_fine_tiled', 'result', mutate)
                work = bench.summarize(manifest, root)['cases'][0]['methods']['full_gpu_fine_tiled']['fine_work']
                self.assertEqual(work['measured_totals']['source_samples']['median'], samples)
                self.assertEqual(work['layers'][0]['tile_rows_capacity'], max(0, tile_rows))

    def test_zero_fine_work_is_distinct_from_unavailable_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True)
            def mutate(worker):
                for record in [worker['first'], *worker['runs']]:
                    for row in record['full_gpu']['fine_work']:
                        for field in bench.FINE_COUNT_FIELDS:
                            row[field] = 0
            for method in bench.GPU_METHODS:
                self.revise(root, manifest, method, 'result', mutate)
            summary = bench.summarize(manifest, root)
            work = summary['cases'][0]['methods']['full_gpu_fine_tiled']['fine_work']
            self.assertEqual(work['measured_totals']['source_samples']['median'], 0)
            self.assertEqual(work['available_calls'], 2)
            self.assertIsNone(work['tiled_angle_fraction'])
            self.assertTrue(summary['cases'][0]['fine_work_comparison']['same_measured_workload'])
            self.assertIsNone(summary['cases'][0]['fine_work_comparison']['source_sampler_calls_ratio'])


if __name__ == '__main__':
    unittest.main()
