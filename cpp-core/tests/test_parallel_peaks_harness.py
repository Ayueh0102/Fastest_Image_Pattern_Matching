"""Negative controls for the stage-42 parallel coarse peak harness."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_parallel_peaks as bench

HYBRID, BASELINE, ABLATION, CANDIDATE = bench.HYBRID, bench.BASELINE, bench.ABLATION, bench.CANDIDATE
STATIC = {method: method+'_static' for method in bench.METHODS}
FRAMES = {method: method+'_frames' for method in (HYBRID, CANDIDATE)}
POSES = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
         dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
# Hand-chosen per-call values; the expected summaries below follow from them by hand.
STATIC_RUN_MS = {HYBRID: 10., BASELINE: 9., ABLATION: 7.2, CANDIDATE: 6.3}
FRAME_WALL = {HYBRID: 12., CANDIDATE: 6.8}
RUN_MS = {HYBRID: 11., CANDIDATE: 6.4}
PEAK = {BASELINE: (1, 1), ABLATION: (2, 1), CANDIDATE: (2, 8)}


def full_gpu_metrics(method, profile=False):
    if method == HYBRID:
        return dict(command_buffers=0, cpu_algorithm_steps=0, cpu_waits=0, device_ms=None, device_timestamp_available=False,
                    dispatches=0, enabled=False, encode_ms=0, fine_diagnostic_bytes=0, fine_diagnostic_readback_bytes=0,
                    fine_effective_mode=0, fine_fallback_windows=None, fine_mode=0, fine_roi_cache_bytes=0,
                    fine_simd32_windows=None, fine_simd_width=0, fine_threadgroup_bytes=0, fine_work=[],
                    fine_work_available=False, gpu_bytes=0, intermediate_readbacks=0, peak_mode=0, peak_scratch_bytes=0,
                    peak_batch_maps=0, coarse_score_bytes=0, profile_available=False, profile_counter_bytes=0,
                    profile_gpu_ms=None, profile_host_ms=0, profile_requested=False, profile_resolve_bytes=0,
                    profile_samples=0, profile_status='disabled', readback_bytes=0, readback_ms=0, stages=[],
                    upload_bytes=0, upload_ms=0, wait_ms=0)
    mode, batch = PEAK[method]
    stages = [dict(name='coarse_peaks', gpu_ms=1.5 if mode == 1 else .25, encoders=4),
              dict(name='fine_search', gpu_ms=4., encoders=2)] if profile else []
    # Profiling also reads back fine work; one level with no reached candidates is consistent.
    work = [dict(model=0, layer=0, template_width=18, template_height=13, roi_cache_bytes=16384, reached_candidates=0,
                 angles=0, cached_angles=0, direct_angles=0, tiled_angles=0, source_samples=0, tile_loads=0,
                 simd32_windows=0, fallback_windows=0)] if profile else []
    return dict(command_buffers=1, cpu_algorithm_steps=0, cpu_waits=1, device_ms=6., device_timestamp_available=True,
                dispatches=754, enabled=True, encode_ms=.75, fine_diagnostic_bytes=48 if profile else 0,
                fine_diagnostic_readback_bytes=24 if profile else 0, fine_effective_mode=4,
                fine_fallback_windows=0 if profile else None, fine_mode=4, fine_roi_cache_bytes=16384,
                fine_simd32_windows=0 if profile else None, fine_simd_width=32, fine_threadgroup_bytes=28560, fine_work=work,
                fine_work_available=profile, gpu_bytes=6451456, intermediate_readbacks=0, peak_mode=mode,
                peak_scratch_bytes=158208, peak_batch_maps=batch, coarse_score_bytes=200000*batch,
                profile_available=profile, profile_counter_bytes=16384 if profile else 0,
                profile_gpu_ms=sum(stage['gpu_ms'] for stage in stages) if profile else None, profile_host_ms=.1 if profile else 0,
                profile_requested=profile, profile_resolve_bytes=192 if profile else 0, profile_samples=12 if profile else 0,
                profile_status='encoder_boundary_timestamps' if profile else 'disabled', readback_bytes=1376, readback_ms=.008,
                stages=stages, upload_bytes=828704, upload_ms=.04, wait_ms=6.)


def record(method, refine, total_ms, pose_create_ms, profile=False):
    gpu = method != HYBRID
    active = refine and not gpu
    return dict(total_ms=total_ms, prepare_ms=0. if gpu else 1., proposal_ms=0. if gpu else 6., refine_ms=0. if gpu else 3.,
                nms_ms=0. if gpu else .1, stage_timings_available=not gpu, legacy_backend_metrics_available=not gpu,
                generation=1, models_matched=0 if gpu else 3, frame_host_bytes=0 if gpu else 4000000,
                frame_coarse_gpu_bytes=0 if gpu else 2091081, frame_fine_gpu_bytes=0 if gpu else 4160256,
                pose_host_bytes=3332652 if active else 0, pose_gpu_bytes=3479808 if active else 0,
                selected_refinement_backend=1 if active else 0, rescued_seeds=0,
                memory=dict(current_rss_bytes=40000000, peak_rss_bytes=41000000), full_gpu=full_gpu_metrics(method, profile),
                refinement_mode=0, configured_refinement_backend=1, refinement_candidates=len(POSES) if refine else 0,
                pose_storage_mode=1, pose_input_mode=1, pose_create_ms=pose_create_ms,
                refinement_dispatches=12 if active else 0, refinement_command_buffers=2 if active else 0,
                matches=copy.deepcopy(POSES), proposals=copy.deepcopy(POSES), work=[])


def frames_worker(case, method, warmup, measured):
    gpu, refine = method != HYBRID, case['workflow'] == 'full_rescue'
    rows, elapsed = [], 0.
    for call in range(1+warmup+measured):
        cold, warming = call == 0, 0 < call <= warmup
        total = 20. if cold else RUN_MS[method]+(1. if warming else 0.)
        pose = 0. if gpu or not refine else 2.5 if cold else 1.5
        update = 0. if cold else .4
        construct, release = (5., 4.2e-05) if cold else (0., 0.)
        wall = total+update+construct+release if cold else FRAME_WALL[method]+(1. if warming else 0.)
        elapsed += 1200.
        row = record(method, refine, total, pose)
        row.update(source_index=0, sequence_cycle=call, call=call, source_path=case['source'], source_width=1100,
                   source_height=740, source_decode_ms=0. if cold else 3.5, release_ms=release, construct_ms=construct,
                   source_update_ms=update, source_copy_bytes=0 if cold else 814000,
                   retained_template_models=0 if cold else 3, pose_invalidated=not cold and refine and not gpu,
                   source_dimensions_changed=False, frame_wall_ms=wall, exact_sequence_repeat=True,
                   reference_cycle=cold, elapsed_ms=elapsed, source_generation=call)
        rows.append(row)
    return dict(sequence_mode='reuse', source_frames=[case['source']], cycles=len(rows), calls=len(rows),
                duration_ms=elapsed, runs=rows, exact_sequence_repeat=True, drained=True,
                final_memory=dict(current_rss_bytes=44000000, peak_rss_bytes=50000000 if gpu else 60000000))


def static_worker(case, method, measured, profile=False):
    gpu, refine = method != HYBRID, case['workflow'] == 'full_rescue'
    first = record(method, refine, 13. if not gpu else 9.5, 2. if refine and not gpu else 0., profile)
    runs = [record(method, refine, STATIC_RUN_MS[method], 0., profile) for _ in range(measured)]
    for row in [first, *runs]:
        row['source_generation'] = 0
    return dict(construct_ms=100., first=first, warmups_ms=[STATIC_RUN_MS[method]]*3, runs=runs, exact_repeat=True,
                final_memory=dict(current_rss_bytes=43000000, peak_rss_bytes=45000000 if gpu else 55000000))


def make_case(name='many128_pose', workflow='full_rescue'):
    return dict(name=name, truth=copy.deepcopy(POSES), source=name+'.png', template='template.png', scales=[.75, 1., 1.25],
                angle=180, limit=8, proposal_score=.5, score=.8, overlap=.2, stop_layer=1, subpixel=0, workflow=workflow)


class ParallelPeaksHarness(unittest.TestCase):
    def fixture(self, root, phase='formal', cases=None, change=None):
        cases = cases or [make_case()]
        blocks, measured, warmup_seconds, profile = bench.phase_settings(phase)
        arms = bench.PHASE_ARMS[phase]
        framed = any(bench.ARMS[arm][1] == 'frames' for arm in arms)
        warmups = {case['name']: bench.FRAME_WARMUP_CALLS[case['name']] for case in cases} if framed else None
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'},
                      input_sha256={name: str(index) for index, name in enumerate(sorted({c[k] for c in cases for k in ('source', 'template')}))})
        manifest = dict(version=1, phase=phase, blocks=blocks, measured_calls=measured, static_warmup_seconds=warmup_seconds,
                        profiling=profile, frame_warmup_calls=warmups, cases=cases, arms=list(arms),
                        method_options=copy.deepcopy(bench.METHODS), executions=[], complete=True, unchanged=True,
                        after_sha256=copy.deepcopy(hashes), **hashes)
        for case in cases:
            for block in range(blocks):
                for arm in arms:
                    method, mode = bench.ARMS[arm]
                    warmup = warmups[case['name']] if framed else None
                    worker = frames_worker(case, method, warmup, measured) if mode == 'frames' else static_worker(case, method, measured, profile)
                    if change:
                        change(case['name'], arm, block, worker)
                    stem = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/(stem+'.job.json'), root/(stem+'.result.json')
                    bench.save(jp, bench.job(case, arm, measured, warmup_seconds, profile, warmup))
                    bench.save(rp, worker)
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block, job=jp.name,
                        result=rp.name, job_sha256=bench.sha(jp), result_sha256=bench.sha(rp),
                        host=dict(utc='2026-10-24T00:00:00+00:00', load_average=[1., 1., 1.],
                                  gpu_device_utilization_percent=[0], competing_processes=[])))
        return manifest

    def revise(self, root, manifest, arm, name, operation, block=0):
        entry = next(row for row in manifest['executions'] if row['arm'] == arm and row['block'] == block)
        path = root/entry[name]
        data = json.loads(path.read_text())
        operation(data)
        bench.save(path, data)
        entry[name+'_sha256'] = bench.sha(path)

    def summary(self, change=None, **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            return bench.summarize(self.fixture(root, change=change, **options), root)

    def gates(self, summary):
        return {row['gate'] for row in summary['failed_gates']}

    def test_formal_positive_counts_and_comparisons(self):
        summary = self.summary()
        self.assertEqual(summary['failed_gates'], [])
        self.assertTrue(summary['formal_decision_eligible'] and summary['all_accuracy_pass'] and summary['all_parallel_exact_mode1'])
        self.assertTrue(summary['primary_target_met'] is None)  # one primary case only
        # 6 arms x 6 blocks; frames GT on 1+2+3 calls (many128 warm-up 2), static on first+3 runs.
        self.assertEqual(summary['counts'], dict(fresh_processes=36, cold_calls=36, gt_checked_warmup_calls=24,
            native_only_warmup_calls=72, measured_calls=108, gt_evaluated_calls=168))
        case = summary['cases'][0]['comparisons']
        primary = case[CANDIDATE+'_vs_mode1_static']
        self.assertEqual((primary['baseline_median_ms'], primary['candidate_median_ms']), (9., 6.3))
        self.assertAlmostEqual(primary['delta_percent'], -30.)
        self.assertTrue(primary['primary_target_met'] and primary['consistent_direction'])
        self.assertEqual(primary['faster'], CANDIDATE)
        self.assertAlmostEqual(case[ABLATION+'_vs_mode1_static']['delta_percent'], -20.)
        self.assertNotIn('primary_target_met', case[ABLATION+'_vs_mode1_static'])
        self.assertAlmostEqual(case['batch_vs_b1_static']['delta_percent'], -12.5)
        hybrid = case['vs_hybrid_frames_frame_wall']
        self.assertEqual((hybrid['baseline_median_ms'], hybrid['candidate_median_ms'], hybrid['faster']), (12., 6.8, CANDIDATE))
        self.assertAlmostEqual(case['vs_hybrid_static_run_only']['delta_percent'], -37.)
        self.assertAlmostEqual(case['mode1_vs_hybrid_static_run_only']['delta_percent'], -10.)

    def test_both_primary_cases_set_the_phase_target(self):
        summary = self.summary(cases=[make_case(), make_case('many512_pose')])
        self.assertTrue(summary['primary_target_met'])
        def change(name, arm, block, worker):
            if name == 'many512_pose' and arm == STATIC[CANDIDATE]:
                for row in worker['runs']:
                    row['total_ms'] = 8.2  # -8.9%: misses the 10% target, within the time budget
        summary = self.summary(cases=[make_case(), make_case('many512_pose')], change=change)
        self.assertFalse(summary['primary_target_met'])
        self.assertEqual({(row['case'], row['gate']) for row in summary['failed_gates']}, {('many512_pose', 'primary_10_percent_improvement')})

    def test_first_and_warmup_calls_excluded_from_distributions(self):
        arms = self.summary()['cases'][0]['arms']
        self.assertEqual(arms[STATIC[CANDIDATE]]['total_ms']['values'], [6.3]*18)
        self.assertEqual(arms[FRAMES[CANDIDATE]]['frame_wall_ms']['values'], [6.8]*18)
        self.assertEqual(arms[FRAMES[HYBRID]]['total_ms']['values'], [11.]*18)
        self.assertEqual(arms[STATIC[CANDIDATE]]['peak_batch_maps'], [8])
        self.assertEqual(arms[STATIC[ABLATION]]['coarse_score_bytes'], [200000])

    def test_exactness_against_mode1_is_separate_from_gt(self):
        for target, field, gate in [(STATIC[CANDIDATE], 'matches', 'exact_ordered_matches_vs_mode1'),
                                    (STATIC[ABLATION], 'proposals', 'exact_ordered_proposals_vs_mode1'),
                                    (FRAMES[CANDIDATE], 'matches', 'exact_ordered_matches_vs_mode1')]:
            def change(name, arm, block, worker, target=target, field=field):
                if arm == target:
                    calls = worker['runs'] if arm.endswith('_frames') else [worker['first'], *worker['runs']]
                    for row in calls:
                        row[field].reverse()
            with self.subTest(target=target):
                summary = self.summary(change=change)
                self.assertTrue(summary['all_accuracy_pass'])
                self.assertIn(gate, self.gates(summary))
                self.assertFalse(summary['all_parallel_exact_mode1'])

    def test_one_changed_score_in_one_call_breaks_exactness(self):
        def change(name, arm, block, worker):
            if arm == STATIC[CANDIDATE] and block == 4:
                worker['runs'][1]['proposals'][1]['score'] = .9400001
        summary = self.summary(change=change)
        self.assertTrue(summary['all_accuracy_pass'])
        self.assertIn('exact_ordered_proposals_vs_mode1', self.gates(summary))
        self.assertIn('exact_repeat', self.gates(summary))

    def test_mode1_reference_change_breaks_every_parallel_arm(self):
        def change(name, arm, block, worker):
            if arm == STATIC[BASELINE] and block == 0:
                worker['first']['matches'].reverse()
        failed = {(row['arm'], row['gate']) for row in self.summary(change=change)['failed_gates']}
        for arm in (STATIC[ABLATION], STATIC[CANDIDATE], FRAMES[CANDIDATE]):
            self.assertIn((arm, 'exact_ordered_matches_vs_mode1'), failed)

    def test_gt_failures_in_first_warmup_or_measured_calls(self):
        targets = [(FRAMES[CANDIDATE], 3, lambda worker: worker['runs'][1]['matches'].pop()),
                   (FRAMES[HYBRID], 3, lambda worker: worker['runs'][0]['matches'].pop()),
                   (STATIC[ABLATION], 3, lambda worker: worker['first']['matches'].pop()),
                   (STATIC[CANDIDATE], 3, lambda worker: worker['runs'][2]['matches'].append(dict(POSES[0], pos_x=150.))),
                   # The very first call of an arm is checked too.
                   (STATIC[BASELINE], 0, lambda worker: worker['first']['matches'].pop()),
                   (FRAMES[CANDIDATE], 0, lambda worker: worker['runs'][0]['matches'].pop())]
        for index, (target, at, mutate) in enumerate(targets):
            def change(name, arm, block, worker, target=target, at=at, mutate=mutate):
                if arm == target and block == at:
                    mutate(worker)
            with self.subTest(index=index):
                summary = self.summary(change=change)
                self.assertIn('independent_ground_truth', self.gates(summary))
                self.assertFalse(summary['all_accuracy_pass'])

    def test_routes_and_requested_kernels(self):
        mutations = [(STATIC[CANDIDATE], lambda row: row['full_gpu'].update(peak_mode=1), 'route_and_residency'),
                     (STATIC[CANDIDATE], lambda row: row['full_gpu'].update(peak_batch_maps=9), 'route_and_residency'),
                     (STATIC[ABLATION], lambda row: row['full_gpu'].update(peak_batch_maps=2), 'route_and_residency'),
                     (STATIC[ABLATION], lambda row: row['full_gpu'].update(peak_batch_maps=0), 'route_and_residency'),
                     (STATIC[BASELINE], lambda row: row['full_gpu'].update(peak_mode=2), 'route_and_residency'),
                     (FRAMES[CANDIDATE], lambda row: row['full_gpu'].update(cpu_waits=2), 'route_and_residency'),
                     (STATIC[CANDIDATE], lambda row: row['full_gpu'].update(intermediate_readbacks=1), 'route_and_residency'),
                     (STATIC[CANDIDATE], lambda row: row['full_gpu'].update(coarse_score_bytes=10**9), 'route_and_residency'),
                     (FRAMES[HYBRID], lambda row: row.update(selected_refinement_backend=0), 'route_and_residency'),
                     (STATIC[CANDIDATE], lambda row: row['full_gpu'].update(fine_simd_width=16, fine_effective_mode=2),
                      'requested_algorithm_not_executed')]
        for index, (target, mutate, gate) in enumerate(mutations):
            def change(name, arm, block, worker, target=target, mutate=mutate):
                if arm == target:
                    mutate(worker['runs'][-1])
            with self.subTest(index=index):
                self.assertIn(gate, self.gates(self.summary(change=change)))

    def test_malformed_peak_metrics_rejected(self):
        for mutate in [lambda row: row['full_gpu'].pop('peak_batch_maps'), lambda row: row['full_gpu'].update(coarse_score_bytes=1.5),
                       lambda row: row['full_gpu'].update(peak_mode=True)]:
            with self.subTest(), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, STATIC[CANDIDATE], 'result', lambda worker: mutate(worker['runs'][0]))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_time_and_rss_budgets_against_mode1_and_hybrid(self):
        def change(name, arm, block, worker):
            if arm == STATIC[CANDIDATE] and block == 2:
                for row in worker['runs']:
                    row['total_ms'] = 11.1  # mode-1 budget is 9+max(2, .9)=11 ms
            if arm == STATIC[ABLATION]:
                worker['final_memory']['peak_rss_bytes'] = 45000000+8*1024**2+1
            if arm == FRAMES[CANDIDATE] and block == 5:
                for row in worker['runs'][3:]:
                    row['frame_wall_ms'] = 14.5  # hybrid budget is 12+max(2, 1.2)=14 ms
                worker['final_memory']['peak_rss_bytes'] = 60000000+8*1024**2+1
        summary = self.summary(change=change)
        failed = {(row['arm'], row['gate'], row.get('block')) for row in summary['failed_gates']}
        self.assertIn((STATIC[CANDIDATE], 'time_budget', 2), failed)
        self.assertIn((STATIC[ABLATION], 'rss_budget', None), failed)
        self.assertIn((FRAMES[CANDIDATE], 'time_budget', 5), failed)
        self.assertIn((FRAMES[CANDIDATE], 'rss_budget', None), failed)
        # One slower block: the pooled median still favours mode 2 but no direction is named.
        primary = summary['cases'][0]['comparisons'][CANDIDATE+'_vs_mode1_static']
        self.assertFalse(primary['consistent_direction'])
        self.assertIsNone(primary['faster'])
        self.assertFalse(primary['primary_target_met'])
        self.assertIn((STATIC[CANDIDATE], 'primary_10_percent_improvement', None), failed)

    def test_budgets_allow_their_documented_tolerance(self):
        def change(name, arm, block, worker):
            if arm == STATIC[ABLATION] and block == 0:
                for row in worker['runs']:
                    row['total_ms'] = 11.  # exactly 9+2
                worker['final_memory']['peak_rss_bytes'] = 45000000+8*1024**2
        summary = self.summary(change=change)
        self.assertEqual(summary['failed_gates'], [])

    def test_frames_routes_parity_and_warmup(self):
        def short(name, arm, block, worker):
            if arm == FRAMES[CANDIDATE] and block == 1:
                for index, row in enumerate(worker['runs']):
                    row['elapsed_ms'] = 400.*(index+1)
        failures = [(row['arm'], row['block']) for row in self.summary(change=short)['failed_gates'] if row['gate'] == 'warmup_shorter_than_1s']
        self.assertEqual(failures, [(FRAMES[CANDIDATE], 1)])
        def pose(name, arm, block, worker):
            if arm == FRAMES[CANDIDATE]:
                worker['runs'][-1]['pose_create_ms'] = .1
        self.assertIn('sequence_route', self.gates(self.summary(change=pose)))
        def parity(name, arm, block, worker):
            if arm == FRAMES[HYBRID]:
                for row in worker['runs']:
                    row['proposals'].reverse()
        gates = self.gates(self.summary(change=parity))
        self.assertIn('same_method_output_parity', gates)
        self.assertNotIn('exact_ordered_proposals_vs_mode1', gates)

    def test_static_control_must_reuse_the_hybrid_pose_context(self):
        def change(name, arm, block, worker):
            if arm == STATIC[HYBRID]:
                worker['runs'][1]['pose_create_ms'] = 3.
        self.assertIn('static_route', self.gates(self.summary(change=change)))

    def test_host_busy_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest['executions'][7]['host']['competing_processes'] = [dict(command='ctest', pcpu=50.)]
            summary = bench.summarize(manifest, root)
            self.assertEqual([row['gate'] for row in summary['failed_gates']], ['host_busy'])

    def test_profile_phase_reports_stages_without_time_gates(self):
        def change(name, arm, block, worker):
            if arm == STATIC[CANDIDATE]:
                for row in worker['runs']:
                    row['total_ms'] = 50.
        summary = self.summary(phase='profile', change=change)
        self.assertEqual(summary['failed_gates'], [])
        arms = summary['cases'][0]['arms']
        self.assertEqual(set(arms), {STATIC[BASELINE], STATIC[ABLATION], STATIC[CANDIDATE]})
        self.assertEqual(arms[STATIC[BASELINE]]['profile']['stages']['coarse_peaks']['median'], 1.5)
        self.assertEqual(arms[STATIC[CANDIDATE]]['profile']['stages']['coarse_peaks']['median'], .25)
        self.assertAlmostEqual(arms[STATIC[CANDIDATE]]['profile']['device_remainder_ms']['median'], 1.75)
        self.assertEqual(summary['cases'][0]['comparisons'], {})
        def missing(name, arm, block, worker):
            if arm == STATIC[ABLATION]:
                worker['runs'][0]['full_gpu']['stages'] = []
        with self.assertRaises(ValueError):
            self.summary(phase='profile', change=missing)

    def test_holdout_phase_gates_exactness_not_time(self):
        cases = [make_case('heldout12'), make_case('empty_scene', workflow='core_only')]
        cases[1]['truth'] = []
        def matchless(name, arm, block, worker):
            if name == 'empty_scene':
                for row in [worker['first'], *worker['runs']]:
                    row['matches'], row['proposals'], row['refinement_candidates'] = [], [], 0
        def slow(name, arm, block, worker):
            matchless(name, arm, block, worker)
            if arm == STATIC[CANDIDATE]:
                for row in worker['runs']:
                    row['total_ms'] = 50.
        summary = self.summary(phase='holdout', cases=cases, change=slow)
        self.assertEqual(summary['failed_gates'], [])
        self.assertEqual(summary['counts']['fresh_processes'], 12)
        def drift(name, arm, block, worker):
            matchless(name, arm, block, worker)
            if name == 'heldout12' and arm == STATIC[ABLATION] and block == 1:
                worker['first']['proposals'].reverse()
        self.assertIn('exact_ordered_proposals_vs_mode1', self.gates(self.summary(phase='holdout', cases=cases, change=drift)))

    def test_pilot_is_not_decision_evidence(self):
        summary = self.summary(phase='pilot')
        self.assertFalse(summary['formal_decision_eligible'])
        self.assertIsNone(summary['primary_target_met'])

    def test_jobs_differ_only_in_declared_keys(self):
        case = make_case()
        static = {method: bench.job(case, STATIC[method], 3, 1., False, None) for method in bench.METHODS}
        self.assertEqual({key for key in static[BASELINE].keys() | static[CANDIDATE].keys()
                          if static[BASELINE].get(key) != static[CANDIDATE].get(key)}, {'full_gpu_peak_mode', 'full_gpu_peak_batch'})
        self.assertEqual((static[CANDIDATE]['full_gpu_peak_mode'], static[CANDIDATE]['full_gpu_peak_batch']), (2, 8))
        self.assertEqual(static[ABLATION]['full_gpu_peak_batch'], 1)
        self.assertNotIn('full_gpu_peak_batch', static[BASELINE])
        frames = bench.job(case, FRAMES[CANDIDATE], 3, 1., False, 2)
        self.assertEqual({key for key in frames.keys() | static[CANDIDATE].keys() if frames.get(key) != static[CANDIDATE].get(key)},
                         {'source', 'source_frames', 'sequence_mode', 'repeats', 'warmup_seconds'})
        self.assertEqual((frames['repeats'], frames['warmup_seconds'], frames['sequence_mode']), (6, 0., 'reuse'))
        for row in [*static.values(), frames]:
            self.assertFalse({'truth', 'truths', 'ground_truth'} & row.keys())

    def test_changed_jobs_and_truth_leaks_rejected(self):
        for arm, changes in [(STATIC[CANDIDATE], dict(full_gpu_peak_batch=4)), (STATIC[CANDIDATE], dict(full_gpu_peak_mode=1)),
                             (STATIC[ABLATION], dict(full_gpu_peak_batch=True)), (STATIC[BASELINE], dict(full_gpu_peak_batch=0)),
                             (STATIC[CANDIDATE], dict(full_gpu_profiling=True)), (STATIC[CANDIDATE], dict(warmup_seconds=0.)),
                             (FRAMES[CANDIDATE], dict(repeats=5)), (FRAMES[CANDIDATE], dict(truth=[])),
                             (STATIC[HYBRID], dict(full_gpu=True))]:
            with self.subTest(arm=arm, changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, arm, 'job', lambda row: row.update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_manifest_integrity_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            arms = bench.PHASE_ARMS['formal']
            options = copy.deepcopy(bench.METHODS)
            options[CANDIDATE]['full_gpu_peak_batch'] = 16
            for changes in [dict(complete=False), dict(unchanged=False), dict(after_sha256={}), dict(blocks=3),
                            dict(measured_calls=2), dict(static_warmup_seconds=0.), dict(profiling=True), dict(phase='other'),
                            dict(arms=arms[:-1]), dict(arms=arms[::-1]), dict(method_options=options),
                            dict(frame_warmup_calls={'many128_pose': 1}), dict(frame_warmup_calls=None),
                            dict(executions=manifest['executions'][:-1]), dict(executions=manifest['executions']*2), dict(cases=[])]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    bench.summarize(dict(manifest, **changes), root)
            entry = manifest['executions'][3]
            (root/entry['result']).write_text((root/entry['result']).read_text()+' ')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)


if __name__ == '__main__':
    unittest.main()
