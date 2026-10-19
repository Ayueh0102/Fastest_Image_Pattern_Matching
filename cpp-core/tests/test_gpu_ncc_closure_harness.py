"""Negative controls for exact tiled parity, stage timestamps, and archives."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_gpu_ncc_closure as bench


class GpuNccClosureHarness(unittest.TestCase):
    def fixture(self, root, profile=False, change=None, methods=None, formal=False, names=None):
        poses = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
                 dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
        case = dict(name='one', truth=copy.deepcopy(poses), source='source.png', template='template.png',
                    scales=[1.], angle=180, limit=8, score=.8, overlap=.2,
                    stop_layer=0, subpixel=1, workflow='full_rescue')
        methods = methods or (list(bench.METHODS) if not profile else list(name for name in bench.METHODS if name in bench.GPU_METHODS))
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'source.png': '3', 'template.png': '4'})
        blocks, repeats, warmup = (3, 2, 1.) if formal else (1, 1, 0.)
        names = names or ['one']
        manifest = dict(phase='profile' if profile else 'formal' if formal else 'pilot', cases=[dict(case, name=name) for name in names], blocks=blocks, repeats=repeats, warmup_seconds=warmup, methods=methods, method_options={name: bench.METHODS[name] for name in methods}, profiling=profile,
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
            metric = dict(fine_mode=fine_mode, fine_effective_mode=fine_mode, fine_simd_width=32, fine_work_available=profile, fine_work=fine_work if profile else [],
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
            row = dict(total_ms=5. if method in bench.CANDIDATES else 10., prepare_ms=None if gpu else 1.,
                       proposal_ms=None if gpu else 7., refine_ms=None if gpu else 1., nms_ms=None if gpu else 1.,
                       stage_timings_available=not gpu, matches=copy.deepcopy(poses), proposals=copy.deepcopy(poses),
                       frame_coarse_gpu_bytes=100, frame_fine_gpu_bytes=100, work=[], full_gpu=metric)
            if change:
                change(method, row)
            for fixture in manifest['cases']:
                for block in range(blocks):
                    worker = dict(first=copy.deepcopy(row), runs=[copy.deepcopy(row) for _ in range(repeats)], exact_repeat=True, construct_ms=1.,
                                  final_memory=dict(peak_rss_bytes=1000000))
                    stem = f"{fixture['name']}-{method}-{block}"
                    jp, rp = root/(stem+'.job.json'), root/(stem+'.result.json')
                    bench.save(jp, bench.job(fixture, method, repeats, warmup, profile))
                    bench.save(rp, worker)
                    manifest['executions'].append(dict(case=fixture['name'], method=method, block=block, job=jp.name,
                        result=rp.name, job_sha256=bench.sha(jp), result_sha256=bench.sha(rp)))
        return manifest

    def revise(self, root, manifest, method, name, operation):
        entry = next(row for row in manifest['executions'] if row['method'] == method)
        path = root/entry[name]
        data = json.loads(path.read_text())
        operation(data)
        bench.save(path, data)
        entry[name+'_sha256'] = bench.sha(path)

    def test_formal_counts_and_each_candidate_has_real_primary_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, formal=True, names=sorted(bench.PRIMARY_CASES)), root)
            self.assertTrue(summary['formal_adoption_eligible'])
            self.assertTrue(summary['all_accuracy_pass'])
            self.assertTrue(summary['all_candidates_exact_baseline'])
            self.assertEqual(summary['failed_gates'], [])
            self.assertEqual(summary['counts'], dict(fresh_processes=36, first_calls=36, measured_calls=72, gt_evaluated_calls=108))
            for candidate in bench.CANDIDATES:
                result = summary['candidates'][candidate]
                self.assertTrue(result['primary_10_percent_target_pass'])
                self.assertTrue(result['formal_primary_target_pass'])
                self.assertTrue(result['comparisons'][bench.BASELINE]['all_candidate_for_adoption'])

    def test_pilot_is_labelled_and_cannot_support_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root), root)
            self.assertEqual(summary['phase'], 'pilot')
            self.assertFalse(summary['formal_adoption_eligible'])
            for candidate, row in summary['candidates'].items():
                self.assertTrue(row['comparisons'][bench.BASELINE]['all_observed_gates_pass'])
                self.assertFalse(row['comparisons'][bench.BASELINE]['all_candidate_for_adoption'])
                self.assertIsNone(row['formal_primary_target_pass'])
                self.assertIsNone(row['primary_10_percent_target_pass'])

    def test_formal_cannot_relabel_a_fast_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            for phase in ['formal', None, 'profile', 'unknown']:
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    bench.summarize(dict(manifest, phase=phase), root)

    def test_every_candidate_can_be_selected_without_other_candidates(self):
        for candidate in bench.CANDIDATES:
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, methods=[bench.BASELINE, candidate]), root)
                self.assertEqual(set(summary['candidates']), {candidate})
                self.assertTrue(summary['all_candidates_exact_baseline'])
                self.assertEqual(summary['counts']['gt_evaluated_calls'], 4)

    def test_unprofiled_requires_baseline_and_at_least_one_candidate(self):
        for methods in [[bench.BASELINE], ['hybrid_best', bench.BASELINE], ['full_gpu_u32', 'full_gpu_moments']]:
            with self.subTest(methods=methods), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    bench.summarize(self.fixture(root, methods=methods), root)

    def test_profile_candidate_without_baseline_makes_no_parity_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, methods=['full_gpu_u32']), root)
            self.assertIsNone(summary['all_candidates_exact_baseline'])
            self.assertIsNone(summary['candidates']['full_gpu_u32']['all_exact_baseline'])
            method = summary['cases'][0]['methods']['full_gpu_u32']
            self.assertIsNone(method['ordered_matches_equal_baseline_gpu'])
            self.assertEqual(method['profiling']['stage_sum_ms']['median'], 9.)
            self.assertEqual(method['profiling']['device_remainder_ms']['median'], 1.)
            self.assertEqual(summary['failed_gates'], [])

    def test_profile_is_separate_and_validates_all_new_mode_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True), root)
            self.assertFalse(summary['formal_adoption_eligible'])
            self.assertTrue(summary['all_candidates_exact_baseline'])
            for candidate in bench.CANDIDATES:
                method = summary['cases'][0]['methods'][candidate]
                self.assertEqual(method['fine_work']['measured_totals']['source_samples']['median'], 191136)
                self.assertEqual(method['fine_work']['measured_totals']['tile_loads']['median'], 48)
                comparison = summary['cases'][0]['fine_work_comparisons'][candidate]
                self.assertTrue(comparison['same_measured_workload'])
                self.assertAlmostEqual(comparison['source_sampler_calls_ratio'], 191136/(6*49*175*128))
                self.assertEqual(summary['candidates'][candidate]['comparisons'], {})

    def test_gt_and_exact_matches_and_proposals_are_independent(self):
        for field in ('matches', 'proposals'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def change(method, row):
                    if method == 'full_gpu_u32':
                        row[field].reverse()
                summary = bench.summarize(self.fixture(root, change=change), root)
                self.assertTrue(summary['all_accuracy_pass'])
                self.assertFalse(summary['all_candidates_exact_baseline'])
                self.assertIn('exact_ordered_'+field, [row['gate'] for row in summary['failed_gates']])
                self.assertFalse(summary['candidates']['full_gpu_u32']['comparisons'][bench.BASELINE]['all_observed_gates_pass'])

    def test_first_call_gt_and_residency_failures_cannot_be_hidden_by_measured_success(self):
        for issue in ('ground_truth', 'residency'):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                def mutate(worker):
                    if issue == 'ground_truth':
                        worker['first']['matches'].pop()
                    else:
                        worker['first']['full_gpu']['cpu_algorithm_steps'] = 1
                self.revise(root, manifest, 'full_gpu_u32', 'result', mutate)
                summary = bench.summarize(manifest, root)
                expected_gate = 'independent_ground_truth' if issue == 'ground_truth' else 'route_and_residency'
                self.assertIn(expected_gate, [row['gate'] for row in summary['failed_gates']])
                self.assertFalse(summary['candidates']['full_gpu_u32']['comparisons'][bench.BASELINE]['all_observed_gates_pass'])

    def test_primary_gate_is_not_vacuously_true_for_missing_or_single_primary_case(self):
        for names in [['holdout_case'], ['many128_pose']]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, formal=True, names=names), root)
                for row in summary['candidates'].values():
                    self.assertIsNone(row['primary_10_percent_target_pass'])
                    self.assertEqual(row['primary_cases_present'], sorted(bench.PRIMARY_CASES & set(names)))

    def test_primary_failure_kept_separately_for_each_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def change(method, row):
                if method == 'full_gpu_u32':
                    row['total_ms'] = 9.5
            summary = bench.summarize(self.fixture(root, formal=True, names=sorted(bench.PRIMARY_CASES), change=change), root)
            self.assertFalse(summary['candidates']['full_gpu_u32']['primary_10_percent_target_pass'])
            self.assertTrue(summary['candidates']['full_gpu_moments']['primary_10_percent_target_pass'])
            failures = [row for row in summary['failed_gates'] if row['gate'] == 'primary_10_percent_improvement']
            self.assertEqual(len(failures), 2)
            self.assertTrue(all(row['method'] == 'full_gpu_u32' for row in failures))

    def test_slow_and_memory_regressions_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            def mutate(worker):
                worker['runs'][0]['total_ms'] = 20.
                worker['final_memory']['peak_rss_bytes'] = 30000000
            self.revise(root, manifest, 'full_gpu_u32', 'result', mutate)
            summary = bench.summarize(manifest, root)
            failures = {row['gate'] for row in summary['failed_gates'] if row['method'] == 'full_gpu_u32'}
            self.assertIn('time_budget', failures)
            self.assertIn('rss_budget', failures)
            self.assertFalse(summary['candidates']['full_gpu_u32']['comparisons'][bench.BASELINE]['all_observed_gates_pass'])

    def test_truth_leaks_changed_settings_and_bool_mode_alias_rejected(self):
        for changes in [dict(truth=[]), dict(score=.1), dict(full_gpu_peak_mode=True), dict(full_gpu_fine_mode=True),
                        dict(full_gpu_fine_mode=2.), dict(full_gpu_fine_mode=3), dict(full_gpu_profiling=0),
                        dict(full_gpu=False), dict(repeats=2), dict(stop_layer=1)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, 'full_gpu_u32', 'job', lambda row: row.update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_manifest_modes_and_execution_integer_aliases_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            for changes in [dict(blocks=True), dict(repeats=True), dict(warmup_seconds=False), dict(profiling=0),
                            dict(complete=False), dict(unchanged=False), dict(after_sha256={}),
                            dict(executions=manifest['executions']*2), dict(cases=manifest['cases']*2)]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    bench.summarize(dict(manifest, **changes), root)
            malformed = copy.deepcopy(manifest)
            malformed['method_options']['full_gpu_u32']['full_gpu_fine_mode'] = 2.
            with self.assertRaises(ValueError):
                bench.summarize(malformed, root)
            malformed = copy.deepcopy(manifest)
            malformed['executions'][0]['block'] = False
            with self.assertRaises(ValueError):
                bench.summarize(malformed, root)

    def test_new_mode_counter_types_geometry_and_sampler_formula_rejected(self):
        mutations = [lambda gpu: gpu.update(fine_mode=True), lambda gpu: gpu.update(fine_mode=2.),
                     lambda gpu: gpu.update(fine_mode=5), lambda gpu: gpu.update(fine_work_available=1),
                     lambda gpu: gpu.update(fine_diagnostic_bytes=True), lambda gpu: gpu.update(fine_threadgroup_bytes=1.5),
                     lambda gpu: gpu.update(fine_diagnostic_readback_bytes=31), lambda gpu: gpu.update(fine_diagnostic_bytes=31),
                     lambda gpu: gpu.update(fine_diagnostic_bytes=301), lambda gpu: gpu.update(fine_work=None),
                     lambda gpu: gpu['fine_work'][0].update(angles=5), lambda gpu: gpu['fine_work'][0].update(angles=6.),
                     lambda gpu: gpu['fine_work'][0].update(reached_candidates=0), lambda gpu: gpu['fine_work'][0].update(template_width=0),
                     lambda gpu: gpu['fine_work'][0].update(source_samples=191135), lambda gpu: gpu['fine_work'][0].update(layer=True),
                     lambda gpu: gpu['fine_work'][1].update(layer=0), lambda gpu: gpu['fine_work'][0].update(tile_loads=47),
                     lambda gpu: gpu['fine_work'][1].update(tile_loads=1), lambda gpu: gpu['fine_work'][0].update(roi_cache_bytes=1)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root, profile=True)
                self.revise(root, manifest, 'full_gpu_u32', 'result', lambda worker: mutation(worker['first']['full_gpu']))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_simd_fallback_is_explicit_and_cannot_support_adoption(self):
        def change(method, row):
            if method == 'full_gpu_simd':
                row['full_gpu'].update(fine_simd_width=16, fine_effective_mode=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, formal=True, names=sorted(bench.PRIMARY_CASES), change=change), root)
            self.assertTrue(summary['all_gpu_resident'])
            self.assertTrue(summary['all_candidates_exact_baseline'])
            simd = summary['candidates']['full_gpu_simd']
            self.assertFalse(simd['all_requested_algorithm_executed'])
            self.assertFalse(simd['formal_primary_target_pass'])
            self.assertEqual(simd['fallback_calls'], 18)
            self.assertFalse(simd['comparisons'][bench.BASELINE]['all_candidate_for_adoption'])
            failures = [row for row in summary['failed_gates'] if row['gate'] == 'requested_algorithm_not_executed']
            self.assertEqual(len(failures), 2)
            self.assertTrue(all(row['method'] == 'full_gpu_simd' for row in failures))

    def test_effective_mode_and_simd_width_contract_is_strict(self):
        mutations = [dict(fine_effective_mode=True), dict(fine_effective_mode=2.), dict(fine_effective_mode=3),
                     dict(fine_simd_width=True), dict(fine_simd_width=32.), dict(fine_simd_width=0),
                     dict(fine_mode=4, fine_effective_mode=4, fine_simd_width=16),
                     dict(fine_mode=4, fine_effective_mode=2, fine_simd_width=32)]
        for changes in mutations:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root, profile=True)
                self.revise(root, manifest, 'full_gpu_u32', 'result', lambda worker: worker['first']['full_gpu'].update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_cache_boundary_partial_strip_and_wide_fallback_all_new_modes(self):
        geometries = [(87, 64, 16384, 'cached_angles'), (175, 129, 16384, 'tiled_angles'),
                      (1200, 100, 16384, 'tiled_angles'), (600, 20, 4096, 'direct_angles'),
                      (600, 20, 4242, 'tiled_angles')]
        for mode in (2, 3, 4):
            for width, height, capacity, category in geometries:
                with self.subTest(mode=mode, geometry=(width, height, capacity)), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = self.fixture(root, profile=True, methods=['full_gpu_u32'])
                    worker = json.loads((root/manifest['executions'][0]['result']).read_text())
                    row = worker['first']
                    gpu = row['full_gpu']
                    gpu['fine_mode'] = mode
                    gpu['fine_effective_mode'] = mode
                    tile_rows = min(16, capacity//(width+6)-6)
                    loads = 6*((height+tile_rows-1)//tile_rows) if category == 'tiled_angles' else 0
                    samples = 6*(width+6)*(height+6) if category == 'cached_angles' else 6*49*width*height if category == 'direct_angles' else (width+6)*(height*6+loads*6)
                    gpu['fine_work'][0].update(template_width=width, template_height=height, roi_cache_bytes=capacity,
                        cached_angles=0, tiled_angles=0, direct_angles=0, tile_loads=loads, source_samples=samples)
                    gpu['fine_work'][0][category] = 6
                    self.assertEqual(bench.fine_work_record(row, True)[(0, 0)]['source_samples'], samples)

    @staticmethod
    def current_routes(method, row):
        """Current runner: 24-byte level records with mode-4 route windows."""
        gpu = row['full_gpu']
        fallback = gpu['fine_effective_mode'] == 4  # The 181x134 level-0 ROI exceeds 16 KiB.
        for level in gpu['fine_work']:
            level.update(simd32_windows=0, fallback_windows=49*level['angles'] if fallback else 0)
        profiled = gpu['fine_work_available']
        gpu.update(fine_diagnostic_readback_bytes=24*len(gpu['fine_work']),
                   fine_simd32_windows=0 if profiled else None,
                   fine_fallback_windows=sum(level['fallback_windows'] for level in gpu['fine_work']) if profiled else None)

    def test_current_route_records_and_legacy_records_validate(self):
        for profile in (False, True):
            for change in (None, self.current_routes):
                with self.subTest(profile=profile, routes=change is not None), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.assertTrue(bench.summarize(self.fixture(root, profile=profile, change=change), root)['all_gpu_resident'])

    def test_route_windows_follow_cache_and_uint32_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True, methods=['full_gpu_simd'], change=self.current_routes)
            row = json.loads((root/manifest['executions'][0]['result']).read_text())['first']
        self.assertEqual(bench.fine_work_record(row, True)[(0, 0)]['fallback_windows'], 294)
        cached = copy.deepcopy(row)
        cached['full_gpu']['fine_work'][0].update(template_width=87, template_height=64, cached_angles=6, tiled_angles=0,
                                                  tile_loads=0, source_samples=6*93*70, simd32_windows=294, fallback_windows=0)
        cached['full_gpu'].update(fine_simd32_windows=294, fine_fallback_windows=0)
        self.assertEqual(bench.fine_work_record(cached, True)[(0, 0)]['simd32_windows'], 294)
        unprofiled = copy.deepcopy(row)
        unprofiled['full_gpu'].update(fine_work_available=False, fine_work=[], fine_diagnostic_readback_bytes=0,
                                      fine_simd32_windows=None, fine_fallback_windows=None)
        self.assertIsNone(bench.fine_work_record(unprofiled, False))
        mutations = [
            (row, True, lambda gpu: gpu['fine_work'][0].update(simd32_windows=294, fallback_windows=0)),
            (row, True, lambda gpu: gpu['fine_work'][0].update(fallback_windows=245)),
            (row, True, lambda gpu: gpu['fine_work'][0].update(fallback_windows=294.)),
            (row, True, lambda gpu: gpu.update(fine_fallback_windows=293)),
            (row, True, lambda gpu: gpu.update(fine_diagnostic_readback_bytes=32)),
            (row, True, lambda gpu: gpu['fine_work'][1].pop('simd32_windows')),
            (row, True, lambda gpu: gpu.pop('fine_simd32_windows')),
            (row, True, lambda gpu: gpu.update(fine_mode=2, fine_effective_mode=2)),
            (cached, True, lambda gpu: gpu['fine_work'][0].update(simd32_windows=0, fallback_windows=294)),
            (unprofiled, False, lambda gpu: gpu.update(fine_simd32_windows=0)),
        ]
        for index, (source, requested, mutate) in enumerate(mutations):
            with self.subTest(index=index):
                changed = copy.deepcopy(source)
                mutate(changed['full_gpu'])
                with self.assertRaises(ValueError):
                    bench.fine_work_record(changed, requested)

    def test_unavailable_timestamps_do_not_erase_diagnostics(self):
        def change(method, row):
            row['full_gpu'].update(profile_available=False, profile_status='unsupported', stages=[], profile_gpu_ms=None,
                                   profile_samples=0, profile_counter_bytes=0, profile_resolve_bytes=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, profile=True, change=change), root)
            for name in bench.CANDIDATES:
                method = summary['cases'][0]['methods'][name]
                self.assertEqual(method['profiling']['available_calls'], 0)
                self.assertEqual(method['fine_work']['available_calls'], 2)
                self.assertIsNotNone(method['fine_work']['measured_totals'])

    def test_formal_diagnostics_are_null_not_fake_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root), root)
            for name in bench.CANDIDATES:
                self.assertIsNone(summary['cases'][0]['fine_work_comparisons'][name])
                self.assertIsNone(summary['cases'][0]['methods'][name]['fine_work']['measured_totals'])

    def test_nonzero_diagnostics_in_unprofiled_run_rejected(self):
        for changes in [dict(fine_work_available=True), dict(fine_work=[{}]), dict(fine_diagnostic_readback_bytes=16)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, 'full_gpu_u32', 'result', lambda worker: worker['first']['full_gpu'].update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_first_calls_excluded_from_timing_and_diagnostic_distributions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, profile=True)
            def mutate(worker):
                worker['first']['total_ms'] = 2000.
                for field in bench.FINE_COUNT_FIELDS:
                    worker['first']['full_gpu']['fine_work'][0][field] *= 2
            self.revise(root, manifest, 'full_gpu_u32', 'result', mutate)
            method = bench.summarize(manifest, root)['cases'][0]['methods']['full_gpu_u32']
            self.assertEqual(method['first_ms'], [2000.])
            self.assertEqual(method['total_ms']['values'], [5.])
            self.assertEqual(method['fine_work']['measured_totals']['source_samples']['values'], [191136])

    def test_archive_hash_and_path_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            changed = copy.deepcopy(manifest)
            changed['executions'][0]['result'] = '../outside.json'
            with self.assertRaises(ValueError):
                bench.summarize(changed, root)
            (root/manifest['executions'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)

    def test_jobs_have_no_truth_and_differ_only_in_fine_mode(self):
        case = dict(name='holdout', source='a', template='b', scales=[1.], angle=180, limit=8,
                    score=.8, overlap=.2, stop_layer=0, subpixel=0, workflow='core_only', truth=[])
        baseline = bench.job(case, bench.BASELINE, 2, 1.)
        self.assertNotIn('truth', baseline)
        for candidate in bench.CANDIDATES:
            job = bench.job(case, candidate, 2, 1.)
            self.assertEqual({key for key in baseline.keys() | job.keys() if baseline.get(key) != job.get(key)}, {'full_gpu_fine_mode'})

    def test_offline_replay_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, formal=True)
            expected = bench.summarize(manifest, root)
            path = root/'manifest.json'
            bench.save(path, manifest)
            actual = bench.summarize(json.loads(path.read_text()), root)
            self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(actual, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
