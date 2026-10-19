"""Negative controls for the stage-41 per-new-frame cost harness."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_new_frame_cost as bench

HYBRID, GPU = 'hybrid_best', 'full_gpu_simd'
FRAMES = {HYBRID: 'hybrid_best_frames', GPU: 'full_gpu_simd_frames'}
STATIC = {HYBRID: 'hybrid_best_static', GPU: 'full_gpu_simd_static'}
POSES = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
         dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
# Hand-chosen per-call values; the expected summaries below are derived from these by hand.
FRAME_WALL = {HYBRID: 12., GPU: 9.}
RUN_MS = {HYBRID: 11., GPU: 8.5}
UPDATE_MS = {HYBRID: .5, GPU: .4}
STATIC_RUN_MS = {HYBRID: 10., GPU: 8.4}


def full_gpu_metrics(enabled):
    """Field set of the current runner's full_gpu object, for both routes."""
    if not enabled:
        return dict(command_buffers=0, cpu_algorithm_steps=0, cpu_waits=0, device_ms=None, device_timestamp_available=False,
                    dispatches=0, enabled=False, encode_ms=0, fine_diagnostic_bytes=0, fine_diagnostic_readback_bytes=0,
                    fine_effective_mode=0, fine_fallback_windows=None, fine_mode=0, fine_roi_cache_bytes=0,
                    fine_simd32_windows=None, fine_simd_width=0, fine_threadgroup_bytes=0, fine_work=[],
                    fine_work_available=False, gpu_bytes=0, intermediate_readbacks=0, peak_mode=0, peak_scratch_bytes=0,
                    profile_available=False, profile_counter_bytes=0, profile_gpu_ms=None, profile_host_ms=0,
                    profile_requested=False, profile_resolve_bytes=0, profile_samples=0, profile_status='disabled',
                    readback_bytes=0, readback_ms=0, stages=[], upload_bytes=0, upload_ms=0, wait_ms=0)
    return dict(command_buffers=1, cpu_algorithm_steps=0, cpu_waits=1, device_ms=8., device_timestamp_available=True,
                dispatches=754, enabled=True, encode_ms=.75, fine_diagnostic_bytes=0, fine_diagnostic_readback_bytes=0,
                fine_effective_mode=4, fine_fallback_windows=None, fine_mode=4, fine_roi_cache_bytes=16384,
                fine_simd32_windows=None, fine_simd_width=32, fine_threadgroup_bytes=28560, fine_work=[],
                fine_work_available=False, gpu_bytes=6451456, intermediate_readbacks=0, peak_mode=1,
                peak_scratch_bytes=158208, profile_available=False, profile_counter_bytes=0, profile_gpu_ms=None,
                profile_host_ms=0, profile_requested=False, profile_resolve_bytes=0, profile_samples=0,
                profile_status='disabled', readback_bytes=1376, readback_ms=.008, stages=[], upload_bytes=828704,
                upload_ms=.04, wait_ms=8.)


def record(method, refine, total_ms, pose_create_ms):
    gpu = method == GPU
    active = refine and not gpu
    return dict(total_ms=total_ms, prepare_ms=0. if gpu else 1., proposal_ms=0. if gpu else 6., refine_ms=0. if gpu else 3.,
                nms_ms=0. if gpu else .1, stage_timings_available=not gpu, legacy_backend_metrics_available=not gpu,
                generation=1, models_matched=0 if gpu else 3, frame_host_bytes=0 if gpu else 4000000,
                frame_coarse_gpu_bytes=0 if gpu else 2091081, frame_fine_gpu_bytes=0 if gpu else 4160256,
                pose_host_bytes=3332652 if active else 0, pose_gpu_bytes=3479808 if active else 0,
                selected_refinement_backend=1 if active else 0, rescued_seeds=0,
                memory=dict(current_rss_bytes=40000000, peak_rss_bytes=41000000), full_gpu=full_gpu_metrics(gpu),
                refinement_mode=0, configured_refinement_backend=1, refinement_candidates=len(POSES) if refine else 0,
                pose_storage_mode=1, pose_input_mode=1, pose_create_ms=pose_create_ms,
                refinement_dispatches=12 if active else 0, refinement_command_buffers=2 if active else 0,
                matches=copy.deepcopy(POSES), proposals=copy.deepcopy(POSES), work=[])


def frames_worker(case, method, warmup, measured):
    gpu, refine = method == GPU, case['workflow'] == 'full_rescue'
    rows, elapsed = [], 0.
    for call in range(1+warmup+measured):
        cold, warming = call == 0, 0 < call <= warmup
        total = 20. if cold else RUN_MS[method]+(1. if warming else 0.)
        pose = 0. if gpu or not refine else 2.5 if cold else 1.5
        update = 0. if cold else UPDATE_MS[method]
        construct, release = (5., 4.2e-05) if cold else (0., 0.)
        wall = total+update+construct+release if cold else FRAME_WALL[method]+(1. if warming else 0.)
        elapsed += 600.
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
                load=dict(decode_ms=4.8, mode='direct_gray8'), after_load_memory=dict(current_rss_bytes=19000000, peak_rss_bytes=19000000),
                final_memory=dict(current_rss_bytes=44000000, peak_rss_bytes=50000000 if gpu else 60000000),
                comparison_reference='first cycle per frame index', frame_wall_scope='update+run',
                final_release_ms=.9, after_release_memory=dict(current_rss_bytes=40000000, peak_rss_bytes=50000000))


def static_worker(case, method, measured):
    gpu, refine = method == GPU, case['workflow'] == 'full_rescue'
    first = record(method, refine, 13. if not gpu else 9.5, 2. if refine and not gpu else 0.)
    runs = [record(method, refine, STATIC_RUN_MS[method], 0.) for _ in range(measured)]
    for row in [first, *runs]:
        row['source_generation'] = 0
    return dict(construct_ms=100., first=first, warmups_ms=[STATIC_RUN_MS[method]]*3, runs=runs, exact_repeat=True,
                final_memory=dict(current_rss_bytes=43000000, peak_rss_bytes=45000000 if gpu else 55000000),
                load=dict(decode_ms=5.5, mode='direct_gray8'), after_load_memory=dict(current_rss_bytes=19000000, peak_rss_bytes=19000000))


def make_case(name='many128_pose', workflow='full_rescue'):
    return dict(name=name, truth=copy.deepcopy(POSES), source=name+'.png', template='template.png', scales=[.75, 1., 1.25],
                angle=180, limit=8, proposal_score=.5, score=.8, overlap=.2, stop_layer=1, subpixel=0, workflow=workflow)


class NewFrameCostHarness(unittest.TestCase):
    def fixture(self, root, phase='formal', blocks=3, measured=2, warmup=2, cases=None, change=None):
        cases = cases or [make_case()]
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'},
                      input_sha256={name: str(index) for index, name in enumerate(sorted({c[k] for c in cases for k in ('source', 'template')}))})
        manifest = dict(version=1, phase=phase, blocks=blocks, measured_calls=measured, static_warmup_seconds=1.,
                        warmup_calls={case['name']: warmup for case in cases}, cases=cases, arms=list(bench.ARMS),
                        arm_options=copy.deepcopy(bench.ARM_OPTIONS), executions=[], complete=True, unchanged=True,
                        after_sha256=copy.deepcopy(hashes), **hashes)
        for case in cases:
            for block in range(blocks):
                for arm, (method, mode) in bench.ARMS.items():
                    worker = (frames_worker(case, method, warmup, measured) if mode == 'frames'
                              else static_worker(case, method, measured))
                    if change:
                        change(case['name'], arm, block, worker)
                    stem = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/(stem+'.job.json'), root/(stem+'.result.json')
                    bench.save(jp, bench.job(case, arm, warmup, measured))
                    bench.save(rp, worker)
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block, job=jp.name,
                        result=rp.name, job_sha256=bench.sha(jp), result_sha256=bench.sha(rp),
                        host=dict(utc='2026-10-19T00:00:00+00:00', load_average=[1., 1., 1.],
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

    def test_formal_positive_counts_and_per_frame_decision(self):
        summary = self.summary()
        self.assertEqual(summary['failed_gates'], [])
        self.assertTrue(summary['formal_decision_eligible'])
        self.assertTrue(summary['all_accuracy_pass'])
        # 4 arms x 3 blocks; frames GT on 1+2+2 calls, static on first+2 runs.
        self.assertEqual(summary['counts'], dict(fresh_processes=12, cold_calls=12, gt_checked_warmup_calls=12,
            native_only_warmup_calls=18, measured_calls=24, gt_evaluated_calls=48))
        case = summary['cases'][0]
        primary = case['comparisons']['frames_frame_wall']
        self.assertEqual((primary['hybrid_median_ms'], primary['gpu_median_ms']), (12., 9.))
        self.assertAlmostEqual(primary['delta_percent'], -25.)
        self.assertEqual([row['delta_percent'] for row in primary['paired']], [-25., -25., -25.])
        self.assertTrue(primary['consistent_direction'])
        self.assertEqual(primary['faster'], GPU)
        self.assertEqual(case['decision']['faster_per_frame'], GPU)
        self.assertAlmostEqual(case['comparisons']['frames_run_only']['delta_percent'], 100*(8.5/11.-1))
        self.assertAlmostEqual(case['comparisons']['static_run_only']['delta_percent'], -16.)

    def test_cold_and_warmup_calls_excluded_from_measured_distributions(self):
        arms = self.summary()['cases'][0]['arms']
        self.assertEqual(arms[FRAMES[HYBRID]]['frame_wall_ms']['values'], [12.]*6)
        self.assertEqual(arms[FRAMES[HYBRID]]['total_ms']['values'], [11.]*6)
        self.assertEqual(arms[FRAMES[HYBRID]]['pose_create_ms']['values'], [1.5]*6)
        self.assertEqual(arms[FRAMES[HYBRID]]['cold_pose_create_ms'], [2.5]*3)
        self.assertEqual(arms[FRAMES[GPU]]['frame_wall_ms']['values'], [9.]*6)
        self.assertEqual(arms[STATIC[HYBRID]]['total_ms']['values'], [10.]*6)
        self.assertEqual(arms[STATIC[HYBRID]]['first_pose_create_ms'], [2.]*3)
        block = arms[FRAMES[HYBRID]]['blocks'][0]
        self.assertEqual((block['warmup_calls'], block['warmup_wall_ms'], block['cold_frame_wall_ms']), (2, 1200., 25.000042))

    def test_same_build_estimate_and_reuse_effect(self):
        case = self.summary()['cases'][0]
        estimate = case['same_build_estimate']
        self.assertEqual((estimate['hybrid_static_median_ms'], estimate['first_call_pose_create_median_ms']), (10., 2.))
        self.assertEqual(estimate['estimate_ms'], 12.)
        self.assertEqual(estimate['measured_run_minus_estimate_ms'], -1.)
        self.assertEqual(estimate['measured_frame_wall_minus_estimate_ms'], 0.)
        self.assertEqual(case['reuse_effect'][HYBRID]['run_ms'], 1.)
        self.assertEqual(case['reuse_effect'][HYBRID]['frame_wall_minus_static_run_ms'], 2.)
        self.assertAlmostEqual(case['reuse_effect'][GPU]['frame_wall_minus_static_run_ms'], .6)

    def test_pilot_cannot_support_the_per_frame_decision(self):
        summary = self.summary(phase='pilot', blocks=1, measured=1)
        self.assertFalse(summary['formal_decision_eligible'])
        self.assertFalse(summary['cases'][0]['decision']['formal_decision_eligible'])

    def test_formal_protocol_minimums_rejected(self):
        for options in [dict(blocks=2), dict(measured=1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.summary(**options)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            for changes in [dict(static_warmup_seconds=.5), dict(phase='profile'), dict(phase=None)]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    bench.summarize(dict(manifest, **changes), root)

    def test_hybrid_must_rebuild_and_gpu_must_not_create_pose_context(self):
        mutations = [
            (HYBRID, lambda row: row.update(pose_create_ms=0.)),
            (HYBRID, lambda row: row.update(pose_invalidated=False)),
            (GPU, lambda row: row.update(pose_create_ms=.1)),
            (GPU, lambda row: row.update(source_generation=0)),
            (GPU, lambda row: row.update(source_copy_bytes=1)),
            (HYBRID, lambda row: row.update(retained_template_models=0)),
            (HYBRID, lambda row: row.update(construct_ms=1., frame_wall_ms=13.)),
        ]
        for index, (method, mutate) in enumerate(mutations):
            def change(name, arm, block, worker):
                if arm == FRAMES[method] and block == 1:
                    mutate(worker['runs'][-1])
            with self.subTest(index=index):
                summary = self.summary(change=change)
                self.assertIn('sequence_route', self.gates(summary))
                self.assertFalse(summary['cases'][0]['arms'][FRAMES[method]]['sequence_route_valid'])

    def test_undrained_sequence_is_a_route_failure(self):
        def change(name, arm, block, worker):
            if arm == FRAMES[GPU]:
                worker['drained'] = False
        self.assertIn('sequence_route', self.gates(self.summary(change=change)))

    def test_static_control_must_reuse_the_pose_context(self):
        def change(name, arm, block, worker):
            if arm == STATIC[HYBRID]:
                worker['runs'][1]['pose_create_ms'] = 3.
        summary = self.summary(change=change)
        self.assertIn('static_route', self.gates(summary))

    def test_core_only_case_never_creates_a_pose_context(self):
        case = make_case('8k_t256_angle180_3scales', workflow='core_only')
        summary = self.summary(cases=[case])
        self.assertEqual(summary['failed_gates'], [])
        self.assertEqual(summary['cases'][0]['arms'][FRAMES[HYBRID]]['pose_create_ms']['values'], [0.]*6)
        def change(name, arm, block, worker):
            if arm == FRAMES[HYBRID]:
                worker['runs'][2]['pose_create_ms'] = 1.
        self.assertIn('sequence_route', self.gates(self.summary(cases=[case], change=change)))

    def test_gt_failures_in_cold_warmup_or_static_calls_cannot_be_hidden(self):
        targets = [(FRAMES[HYBRID], lambda worker: worker['runs'][1]['matches'].pop()),
                   (FRAMES[GPU], lambda worker: worker['runs'][0]['matches'].pop()),
                   (STATIC[GPU], lambda worker: worker['first']['matches'].pop())]
        for index, (target, mutate) in enumerate(targets):
            def change(name, arm, block, worker):
                if arm == target and block == 2:
                    mutate(worker)
            with self.subTest(index=index):
                summary = self.summary(change=change)
                self.assertIn('independent_ground_truth', self.gates(summary))
                self.assertFalse(summary['all_accuracy_pass'])

    def test_one_extra_false_positive_fails_ground_truth(self):
        # Every truth object stays matched; one extra prediction alone must fail the GT gate.
        extras = {'far_from_truth': dict(pos_x=150., pos_y=20., angle=0., scale=1., score=.9),
                  'duplicate_of_truth': dict(POSES[0], score=.9)}
        targets = [(FRAMES[HYBRID], lambda worker: worker['runs'][-1]),
                   (FRAMES[GPU], lambda worker: worker['runs'][0]),
                   (STATIC[GPU], lambda worker: worker['first'])]
        for label, extra in extras.items():
            for target, call in targets:
                def change(name, arm, block, worker):
                    if arm == target and block == 2:
                        call(worker)['matches'].append(copy.deepcopy(extra))
                with self.subTest(extra=label, arm=target):
                    summary = self.summary(change=change)
                    self.assertIn('independent_ground_truth', self.gates(summary))
                    self.assertFalse(summary['all_accuracy_pass'])

    def test_output_parity_with_static_control_is_separate_from_gt(self):
        def change(name, arm, block, worker):
            if arm == FRAMES[GPU]:
                for row in worker['runs']:
                    row['matches'].reverse()
        summary = self.summary(change=change)
        self.assertTrue(summary['all_accuracy_pass'])
        self.assertEqual(self.gates(summary), {'same_method_output_parity'})
        self.assertFalse(summary['cases'][0]['arms'][FRAMES[GPU]]['same_method_output_parity'])

    def test_exact_repeat_failures_retained(self):
        mutations = [lambda worker: worker.update(exact_sequence_repeat=False),
                     lambda worker: worker['runs'][-1]['proposals'].reverse()]
        for index, mutate in enumerate(mutations):
            def change(name, arm, block, worker):
                if arm == FRAMES[HYBRID] and block == 0:
                    mutate(worker)
            with self.subTest(index=index):
                self.assertIn('exact_repeat', self.gates(self.summary(change=change)))

    def test_residency_algorithm_and_refinement_routes(self):
        mutations = [(FRAMES[GPU], lambda worker: worker['runs'][1]['full_gpu'].update(cpu_waits=2), 'route_and_residency'),
                     (STATIC[GPU], lambda worker: worker['first']['full_gpu'].update(intermediate_readbacks=1), 'route_and_residency'),
                     (FRAMES[HYBRID], lambda worker: worker['runs'][-1].update(selected_refinement_backend=0), 'route_and_residency'),
                     (FRAMES[GPU], lambda worker: [row['full_gpu'].update(fine_simd_width=16, fine_effective_mode=2) for row in worker['runs']],
                      'requested_algorithm_not_executed')]
        for index, (target, mutate, gate) in enumerate(mutations):
            def change(name, arm, block, worker):
                if arm == target:
                    mutate(worker)
            with self.subTest(index=index):
                self.assertIn(gate, self.gates(self.summary(change=change)))

    def test_short_warmup_retained_per_block(self):
        def change(name, arm, block, worker):
            if arm == FRAMES[GPU] and block == 1:
                for index, row in enumerate(worker['runs']):
                    row['elapsed_ms'] = 400.*(index+1)
        summary = self.summary(change=change)
        failures = [row for row in summary['failed_gates'] if row['gate'] == 'warmup_shorter_than_1s']
        self.assertEqual([(row['arm'], row['block']) for row in failures], [(FRAMES[GPU], 1)])

    def test_time_and_rss_budget_failures_retained_and_mixed_blocks_make_no_claim(self):
        def change(name, arm, block, worker):
            if arm == FRAMES[GPU] and block == 1:
                # Measured calls only; the hybrid budget is 12+max(2, 1.2)=14 ms.
                for row in worker['runs'][3:]:
                    row['frame_wall_ms'] = 15.
                worker['final_memory']['peak_rss_bytes'] = 90000000
        summary = self.summary(change=change)
        failures = [(row['gate'], row.get('block')) for row in summary['failed_gates']]
        self.assertIn(('time_budget', 1), failures)
        self.assertIn(('rss_budget', None), failures)
        primary = summary['cases'][0]['comparisons']['frames_frame_wall']
        self.assertEqual(primary['gpu_median_ms'], 9.)
        self.assertFalse(primary['consistent_direction'])
        self.assertIsNone(primary['faster'])
        self.assertIsNone(summary['cases'][0]['decision']['faster_per_frame'])

    def test_budgets_allow_their_documented_tolerance(self):
        def change(name, arm, block, worker):
            if arm == FRAMES[GPU] and block == 0:
                # 12+max(2, 1.2)=14 ms and 60 MB+max(8 MiB, 6 MB) are still within budget.
                for row in worker['runs'][3:]:
                    row['frame_wall_ms'] = 14.
                worker['final_memory']['peak_rss_bytes'] = 68000000
        summary = self.summary(change=change)
        self.assertEqual(summary['failed_gates'], [])
        self.assertIsNone(summary['cases'][0]['decision']['faster_per_frame'])

    def test_sequence_structure_violations_rejected(self):
        mutations = [lambda worker: worker['runs'][1].update(call=2),
                     lambda worker: worker['runs'][2].update(sequence_cycle=1),
                     lambda worker: worker['runs'][1].update(source_index=1),
                     lambda worker: worker['runs'][1].update(source_path='other.png'),
                     lambda worker: worker['runs'][1].update(reference_cycle=True),
                     lambda worker: worker['runs'][0].update(reference_cycle=1),
                     lambda worker: worker['runs'].pop(),
                     lambda worker: worker.update(cycles=4),
                     lambda worker: worker.update(sequence_mode='fresh'),
                     lambda worker: worker.update(source_frames=['other.png']),
                     lambda worker: worker.update(exact_sequence_repeat=1),
                     lambda worker: worker['runs'][3].update(frame_wall_ms=11.),
                     lambda worker: worker['runs'][3].update(source_update_ms=-.1),
                     lambda worker: worker['runs'][3].update(pose_create_ms='1'),
                     lambda worker: worker['runs'][3].update(elapsed_ms=1200.),
                     lambda worker: worker['runs'][3].pop('source_decode_ms'),
                     lambda worker: worker['final_memory'].update(peak_rss_bytes=1.5)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, FRAMES[HYBRID], 'result', mutate)
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_static_structure_violations_rejected(self):
        mutations = [lambda worker: worker['runs'].pop(), lambda worker: worker.update(exact_repeat=1),
                     lambda worker: worker.update(warmups_ms=[]), lambda worker: worker.update(construct_ms=-1.),
                     lambda worker: worker['first'].update(total_ms=0.)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, STATIC[GPU], 'result', mutate)
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_jobs_differ_only_in_declared_sequence_and_method_keys(self):
        case = make_case()
        frames, static = bench.job(case, FRAMES[HYBRID], 2, 2), bench.job(case, STATIC[HYBRID], 2, 2)
        self.assertEqual({key for key in frames.keys() | static.keys() if frames.get(key) != static.get(key)},
                         {'source', 'source_frames', 'sequence_mode', 'repeats', 'warmup_seconds'})
        self.assertEqual((frames['source_frames'], frames['sequence_mode'], frames['repeats'], frames['warmup_seconds']),
                         (['many128_pose.png'], 'reuse', 5, 0.))
        self.assertEqual((static['source'], static['repeats'], static['warmup_seconds']), ('many128_pose.png', 2, 1.))
        gpu = bench.job(case, FRAMES[GPU], 2, 2)
        self.assertEqual({key for key in frames.keys() | gpu.keys() if frames.get(key) != gpu.get(key)},
                         {'full_gpu', 'full_gpu_peak_mode', 'full_gpu_fine_mode'})
        self.assertEqual((gpu['full_gpu'], gpu['full_gpu_peak_mode'], gpu['full_gpu_fine_mode']), (True, 1, 4))
        for row in (frames, static, gpu):
            self.assertFalse({'truth', 'truths', 'ground_truth'} & row.keys())

    def test_changed_jobs_and_truth_leaks_rejected(self):
        for changes in [dict(sequence_mode='fresh'), dict(repeats=6), dict(warmup_seconds=1.), dict(full_gpu_fine_mode=2),
                        dict(full_gpu_profiling=True), dict(truth=[]), dict(source='many128_pose.png'),
                        dict(source_frames=['many128_pose.png']*2), dict(full_gpu_peak_mode=True)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, FRAMES[GPU], 'job', lambda row: row.update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_manifest_integrity_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            arms = list(bench.ARMS)
            for changes in [dict(complete=False), dict(unchanged=False), dict(after_sha256={}), dict(blocks=True),
                            dict(measured_calls=True), dict(arms=arms[:3]), dict(arms=arms+['other']), dict(arms=arms+arms[:1]),
                            dict(warmup_calls={}), dict(warmup_calls={'many128_pose': 0}), dict(warmup_calls={'many128_pose': True}),
                            dict(executions=manifest['executions'][:-1]), dict(executions=manifest['executions']*2),
                            dict(cases=manifest['cases']*2)]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    bench.summarize(dict(manifest, **changes), root)
            changed = copy.deepcopy(manifest)
            changed['arm_options'][FRAMES[GPU]]['full_gpu_fine_mode'] = 2
            with self.assertRaises(ValueError):
                bench.summarize(changed, root)
            changed = copy.deepcopy(manifest)
            changed['executions'][0]['result'] = '../outside.json'
            with self.assertRaises(ValueError):
                bench.summarize(changed, root)
            (root/manifest['executions'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)

    def test_competing_host_processes_retained_and_missing_host_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            entry = next(row for row in manifest['executions'] if row['arm'] == FRAMES[GPU] and row['block'] == 2)
            entry['host']['competing_processes'] = [dict(command='clang++', pcpu=95.)]
            failures = [row for row in bench.summarize(manifest, root)['failed_gates'] if row['gate'] == 'host_busy']
            self.assertEqual([(row['arm'], row['block']) for row in failures], [(FRAMES[GPU], 2)])
            for host in [None, dict(competing_processes=None), dict(competing_processes=[dict(pcpu=1.)])]:
                changed = copy.deepcopy(manifest)
                changed['executions'][0]['host'] = host
                with self.subTest(host=host), self.assertRaises(ValueError):
                    bench.summarize(changed, root)

    def test_memory_scopes_reported_separately(self):
        arms = self.summary()['cases'][0]['arms']
        self.assertEqual(arms[FRAMES[GPU]]['named_gpu_bytes'], [6451456])
        self.assertEqual(arms[FRAMES[HYBRID]]['frame_gpu_bytes'], [2091081+4160256])
        self.assertEqual(arms[FRAMES[HYBRID]]['pose_gpu_bytes'], [3479808])
        self.assertEqual(arms[FRAMES[HYBRID]]['peak_rss_bytes'], 60000000)
        self.assertEqual(arms[STATIC[GPU]]['peak_rss_bytes'], 45000000)

    def test_warmup_calls_follow_round40_medians(self):
        medians = {'many128_pose': (824.59, 806.46), 'many512_pose': (2391.95, 2805.18),
                   'controlled6': (59.85, 70.03), '8k_t256_angle180_3scales': (265.28, 66.80)}
        summary = dict(cases=[dict(name=name, methods={HYBRID: dict(total_ms=dict(median=a)), GPU: dict(total_ms=dict(median=b))})
                              for name, (a, b) in medians.items()])
        self.assertEqual(bench.warmup_calls(summary, list(medians)),
                         {'many128_pose': 2, 'many512_pose': 1, 'controlled6': 26, '8k_t256_angle180_3scales': 23})
        with self.assertRaises(ValueError):
            bench.warmup_calls(summary, ['missing'])

    def test_round40_estimate_direction_check(self):
        archived = [dict(case='many128_pose', hybrid_median_ms=824.59, first_call_pose_create_median_ms=19.08,
                         hybrid_with_pose_create_ms=843.67,
                         methods={GPU: dict(median_ms=806.46, vs_hybrid_percent=-2.2, vs_hybrid_with_pose_create_percent=-4.4)})]
        held = bench.estimate_check(self.summary(), archived)['many128_pose']
        self.assertTrue(held['direction_held'])
        self.assertAlmostEqual(held['measured_minus_estimate_points'], -25.+4.4)
        self.assertEqual(held['archived_first_call_pose_create_median_ms'], 19.08)
        self.assertEqual(held['measured_pose_create_median_ms'], 1.5)
        def slower(name, arm, block, worker):
            if arm == FRAMES[GPU]:
                for row in worker['runs'][3:]:
                    row['frame_wall_ms'] = 13.
        self.assertFalse(bench.estimate_check(self.summary(change=slower), archived)['many128_pose']['direction_held'])
        def mixed(name, arm, block, worker):
            if arm == FRAMES[GPU] and block == 0:
                for row in worker['runs'][3:]:
                    row['frame_wall_ms'] = 13.
        self.assertIsNone(bench.estimate_check(self.summary(change=mixed), archived)['many128_pose']['direction_held'])
        missing = bench.estimate_check(self.summary(cases=[make_case('controlled6')]), archived)
        self.assertIsNone(missing['controlled6'])

    def test_offline_replay_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            expected = bench.summarize(manifest, root)
            path = root/'manifest.json'
            bench.save(path, manifest)
            actual = bench.summarize(json.loads(path.read_text()), root)
            self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(actual, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
