"""Negative controls for the stage-48 per-new-frame overhead diagnosis harness."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_frame_overheads as bench

HYBRID, G1, G2 = 'hybrid_best', 'full_gpu_simd', 'full_gpu_parallel'
C128, C512, C8K = 'many128_pose', 'many512_pose', '8k_t256_angle180_3scales'
POSES = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
         dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
# Hand-chosen measured values per condition. A gap (sleep or decode) slows the
# first GPU stage; an update adds the pose rebuild, one ms of refinement on new
# buffers and half a ms of frame preparation.
HYBRID_VALUES = {
    'static_none':   dict(total=100., proposal=80., prepare=2., refine=20., pose=0., refdev=15., coarse=10., update=0.),
    'static_sleep':  dict(total=102., proposal=82., prepare=2., refine=20., pose=0., refdev=15., coarse=12., update=0.),
    'static_decode': dict(total=102., proposal=82., prepare=2., refine=20., pose=0., refdev=15., coarse=12., update=0.),
    'reuse_none':    dict(total=106.5, proposal=80.5, prepare=2.5, refine=26., pose=5., refdev=16., coarse=10., update=.5),
    'reuse_decode':  dict(total=108.5, proposal=82.5, prepare=2.5, refine=26., pose=5., refdev=16., coarse=12., update=.5),
}
GPU_BASE = {G1: 60., G2: 50.}
GPU_GAP = {'static_none': 0., 'static_sleep': 5., 'static_decode': 5., 'reuse_none': 0., 'reuse_decode': 5.}


def full_gpu_metrics(method, device=0., profile=False):
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
    metrics = dict(command_buffers=1, cpu_algorithm_steps=0, cpu_waits=1, device_ms=device, device_timestamp_available=True,
                   dispatches=754, enabled=True, encode_ms=.3, fine_diagnostic_bytes=0, fine_diagnostic_readback_bytes=0,
                   fine_effective_mode=4, fine_fallback_windows=None, fine_mode=4, fine_roi_cache_bytes=16384,
                   fine_simd32_windows=None, fine_simd_width=32, fine_threadgroup_bytes=28560, fine_work=[],
                   fine_work_available=False, gpu_bytes=6451456, intermediate_readbacks=0,
                   peak_mode=bench.METHODS[method]['full_gpu_peak_mode'], peak_scratch_bytes=158208,
                   peak_batch_maps=bench.METHODS[method].get('full_gpu_peak_batch', 0), coarse_score_bytes=1000,
                   profile_available=False, profile_counter_bytes=0, profile_gpu_ms=None, profile_host_ms=0,
                   profile_requested=False, profile_resolve_bytes=0, profile_samples=0, profile_status='disabled',
                   readback_bytes=1376, readback_ms=.008, stages=[], upload_bytes=828704, upload_ms=1., wait_ms=device+.5)
    if profile:
        # Round-40 harness fixture shape: one tiled level-0 and one empty level-1 record.
        metrics.update(fine_work_available=True, fine_diagnostic_bytes=64, fine_diagnostic_readback_bytes=48,
                       fine_simd32_windows=0, fine_fallback_windows=294,
                       fine_work=[dict(model=0, layer=0, template_width=175, template_height=128, roi_cache_bytes=16384,
                                       reached_candidates=2, angles=6, cached_angles=0, direct_angles=0, tiled_angles=6,
                                       source_samples=6*181*(128+6*8), tile_loads=48, simd32_windows=0, fallback_windows=294),
                                  dict(model=0, layer=1, template_width=87, template_height=64, roi_cache_bytes=16384,
                                       reached_candidates=0, angles=0, cached_angles=0, direct_angles=0, tiled_angles=0,
                                       source_samples=0, tile_loads=0, simd32_windows=0, fallback_windows=0)],
                       profile_requested=True, profile_available=True, profile_status='ok', profile_samples=4,
                       profile_counter_bytes=32, profile_resolve_bytes=32, profile_host_ms=.1, profile_gpu_ms=device-1.,
                       stages=[dict(name='coarse', gpu_ms=device/2-1., encoders=1), dict(name='fine', gpu_ms=device/2, encoders=1)])
    return metrics


def record(case, method, condition, call, warmup, profile=False):
    refine, gpu = case['workflow'] == 'full_rescue', method != HYBRID
    cold, warming = call == 0, 0 < call <= warmup
    mode, gap = bench.CONDITIONS[condition]
    reused = mode == 'reuse' and call > 0
    if gpu:
        # Warm-up calls run 2 ms slower so a diagnostic that also counts them shows it.
        device = GPU_BASE[method]+(GPU_GAP[condition] if call else 20.)+(2. if warming else 0.)
        total, update, pose = device+1., (1. if reused else 0.), 0.
        stages = dict(prepare_ms=0., proposal_ms=0., refine_ms=0., nms_ms=0.)
        work, refdev = [], 0.
    else:
        values = HYBRID_VALUES[condition] if call else dict(HYBRID_VALUES['static_none'], total=150., pose=6., refine=26.)
        if not refine:
            values = dict(values, refine=0., pose=0., refdev=0., total=values['total']-values['refine'])
        total, update = values['total'], values['update'] if reused else 0.
        pose = values['pose'] if refine and (call == 0 or mode == 'reuse') else 0.
        stages = dict(prepare_ms=values['prepare'], proposal_ms=values['proposal'], refine_ms=values['refine'], nms_ms=.1)
        refdev = values['refdev']
        # Two scale models share the coarse device time.
        work = [dict(device_ms=values['coarse']/2, wait_ms=values['coarse']/2+1., fine_wait_ms=25., fine_ncc_ms=27.,
                     normalization_ms=4., maps=10, windows=100, fine_maps=5) for _ in range(2)]
    if warming:
        total += 1.
    active = refine and not gpu
    row = dict(total_ms=total, **stages, stage_timings_available=not gpu, legacy_backend_metrics_available=not gpu,
               generation=1, models_matched=0 if gpu else 3, frame_host_bytes=0 if gpu else 4000000,
               frame_coarse_gpu_bytes=0 if gpu else 2091081, frame_fine_gpu_bytes=0 if gpu else 4160256,
               pose_host_bytes=3332652 if active else 0, pose_gpu_bytes=3479808 if active else 0,
               selected_refinement_backend=1 if active else 0, rescued_seeds=0,
               memory=dict(current_rss_bytes=40000000, peak_rss_bytes=41000000),
               full_gpu=full_gpu_metrics(method, device if gpu else 0., profile and gpu),
               refinement_mode=0, configured_refinement_backend=1, refinement_candidates=len(POSES) if refine else 0,
               pose_storage_mode=1, pose_input_mode=1, pose_create_ms=pose, refinement_device_ms=refdev,
               refinement_dispatches=12 if active else 0, refinement_command_buffers=2 if active else 0,
               matches=copy.deepcopy(POSES), proposals=copy.deepcopy(POSES), work=work)
    return row, update


def worker(case, arm, spec, warmup, measured, profile=False):
    rows, elapsed = [], 0.
    sleep = spec['sleep_ms']
    for call in range(1+warmup+measured):
        row, update = record(case, spec['method'], bench.condition_of(spec), call, warmup, profile)
        cold = call == 0
        decode = 130. if spec['gap'] == 'decode' and not cold else 0.
        gap = 0. if cold else decode+.01 if spec['gap'] == 'decode' else sleep+.2 if spec['gap'] == 'sleep' else .001
        construct, release = (5., 4.2e-05) if cold else (0., 0.)
        wall = row['total_ms']+update+construct+release+.01
        elapsed += gap+wall+1000.
        reused = spec['mode'] == 'reuse' and not cold
        row.update(source_index=0, sequence_cycle=call, call=call, source_path=case['source'], source_width=1100,
                   source_height=740, source_decode_ms=decode, sequence_gap=spec['gap'], gap_ms=gap,
                   release_ms=release, construct_ms=construct, source_update_ms=update,
                   source_copy_bytes=1100*740 if reused else 0, retained_template_models=3 if reused else 0,
                   pose_invalidated=reused and case['workflow'] == 'full_rescue' and spec['method'] == HYBRID,
                   source_dimensions_changed=False, frame_wall_ms=wall, exact_sequence_repeat=True,
                   reference_cycle=cold, elapsed_ms=elapsed, source_generation=call if reused else 0)
        rows.append(row)
    return dict(sequence_mode=spec['mode'], sequence_gap=spec['gap'],
                sequence_gap_sleep_ms=float(sleep) if spec['gap'] == 'sleep' else None,
                source_frames=[case['source']], cycles=len(rows), calls=len(rows), duration_ms=elapsed, runs=rows,
                exact_sequence_repeat=True, drained=True, load=dict(decode_ms=4.8, mode='direct_gray8'),
                after_load_memory=dict(current_rss_bytes=19000000, peak_rss_bytes=19000000),
                final_memory=dict(current_rss_bytes=44000000, peak_rss_bytes=60000000),
                comparison_reference='first cycle per frame index', frame_wall_scope='update+run',
                final_release_ms=.9, after_release_memory=dict(current_rss_bytes=40000000, peak_rss_bytes=50000000))


def make_case(name):
    return dict(name=name, truth=copy.deepcopy(POSES), source=name+'.png', template='template.png', scales=[.75, 1., 1.25],
                angle=180, limit=8, proposal_score=.5, score=.8, overlap=.2, stop_layer=1, subpixel=0,
                workflow='core_only' if name == C8K else 'full_rescue')


class FrameOverheadsHarness(unittest.TestCase):
    def fixture(self, root, phase='formal', blocks=3, measured=2, change=None, cases=None):
        names = cases or bench.PHASE_CASES[phase]
        cases = [make_case(name) for name in names]
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'},
                      input_sha256={name: str(index) for index, name in enumerate(sorted({c[k] for c in cases for k in ('source', 'template')}))})
        arms = {case['name']: bench.phase_arms(phase, case['name']) for case in cases}
        manifest = dict(version=1, phase=phase, blocks=blocks, measured_calls=measured, profiling=phase == 'profile',
                        warmup_calls={name: bench.WARMUP_CALLS[name] for name in names},
                        sleep_ms={name: bench.SLEEP_MS[name] for name in names}, cases=cases,
                        arms={name: [arm for arm, _ in rows] for name, rows in arms.items()},
                        arm_specs={name: {arm: spec for arm, spec in rows} for name, rows in arms.items()},
                        executions=[], complete=True, unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
        for case in cases:
            for block in range(blocks):
                for arm, spec in arms[case['name']]:
                    data = worker(case, arm, spec, bench.WARMUP_CALLS[case['name']], measured, phase == 'profile')
                    if change:
                        change(case['name'], arm, block, data)
                    stem = f"{case['name']}-{arm}-b{block}"
                    jp, rp = root/(stem+'.job.json'), root/(stem+'.result.json')
                    bench.save(jp, bench.job(case, spec, bench.WARMUP_CALLS[case['name']], measured, phase == 'profile'))
                    bench.save(rp, data)
                    manifest['executions'].append(dict(case=case['name'], arm=arm, block=block, job=jp.name, result=rp.name,
                        job_sha256=bench.sha(jp), result_sha256=bench.sha(rp),
                        host=dict(utc='2026-10-25T00:00:00+00:00', load_average=[1., 1., 1.],
                                  gpu_device_utilization_percent=[0], competing_processes=[])))
        return manifest

    def summary(self, change=None, **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            return bench.summarize(self.fixture(root, change=change, **options), root)

    def revise(self, root, manifest, case, arm, name, operation, block=0):
        entry = next(row for row in manifest['executions'] if row['case'] == case and row['arm'] == arm and row['block'] == block)
        path = root/entry[name]
        data = json.loads(path.read_text())
        operation(data)
        bench.save(path, data)
        entry[name+'_sha256'] = bench.sha(path)

    def gates(self, summary):
        return {row['gate'] for row in summary['failed_gates']}

    def case(self, summary, name):
        return next(row for row in summary['cases'] if row['name'] == name)

    def test_formal_effects_and_hypothesis_verdicts(self):
        summary = self.summary()
        self.assertEqual(summary['failed_gates'], [])
        self.assertTrue(summary['all_accuracy_pass'])
        c512 = self.case(summary, C512)['effects'][HYBRID]
        self.assertEqual((c512['round41']['total_ms']['delta_ms'], c512['round41']['total_ms']['direction']), (8.5, 'up'))
        self.assertEqual(c512['update']['total_ms']['delta_ms'], 6.5)
        self.assertEqual(c512['decode_gap']['coarse_device_ms']['delta_ms'], 2.)
        self.assertEqual(c512['sleep_gap']['coarse_device_ms']['direction'], 'up')
        self.assertIsNone(c512['update']['coarse_device_ms']['direction'])
        self.assertEqual((c512['update']['refinement_device_ms']['delta_ms'], c512['update']['refinement_device_ms']['direction']), (1., 'up'))
        self.assertIsNone(c512['sleep_gap']['refinement_device_ms']['direction'])
        self.assertEqual(c512['update']['refine_minus_pose_ms']['delta_ms'], 1.)
        self.assertEqual(c512['update']['prepare_ms']['delta_ms'], .5)
        g8k = self.case(summary, C8K)['effects'][G1]
        self.assertEqual((g8k['sleep_gap']['device_ms']['delta_ms'], g8k['update']['device_ms']['direction']), (5., None))
        verdicts = summary['hypotheses']
        self.assertTrue(verdicts['round41_reproduced']['reproduced'])
        self.assertEqual(verdicts['H1_gap_slows_first_gpu_stage']['verdict'], 'supported')
        self.assertEqual(verdicts['H2_new_pose_buffers_slow_refinement']['verdict'], 'supported')
        self.assertEqual(verdicts['H3_new_source_slows_frame_prepare']['verdict'], 'supported')
        self.assertEqual(verdicts['H4_8k_full_gpu_attribution']['verdict'], 'gap')
        self.assertEqual(verdicts['S1_8k_peak2_attribution']['verdict'], 'gap')

    def test_counts_and_cold_warmup_exclusion(self):
        summary = self.summary()
        # 10+10+15 arms x 3 blocks; frames rows per process are 1+W+2 with W=2/1/23.
        self.assertEqual(summary['counts'], dict(fresh_processes=105, cold_calls=105,
            warmup_calls=3*(10*2+10*1+15*23), measured_calls=105*2, gt_evaluated_calls=3*(10*5+10*4+15*26)))
        arm = self.case(summary, C128)['arms'][f'{HYBRID}__static_none']
        self.assertEqual(arm['metrics']['total_ms']['values'], [100.]*6)
        self.assertEqual(arm['blocks'][0]['warmup_calls'], 2)

    def test_mixed_blocks_make_no_direction_and_flip_verdicts(self):
        def change(name, arm, block, data):
            if name == C512 and arm == f'{HYBRID}__static_sleep' and block == 1:
                for row in data['runs'][2:]:
                    for model in row['work']:
                        model['device_ms'] = 4.
        summary = self.summary(change=change)
        effect = self.case(summary, C512)['effects'][HYBRID]['sleep_gap']['coarse_device_ms']
        self.assertIsNone(effect['direction'])
        self.assertEqual(summary['hypotheses']['H1_gap_slows_first_gpu_stage']['verdict'], 'decode_only')

    def test_verdicts_follow_their_rules(self):
        def no_gap_effect(name, arm, block, data):
            if arm.endswith(('static_sleep', 'static_decode', 'reuse_decode')):
                for row in data['runs'][1:]:
                    if row['work']:
                        for model in row['work']:
                            model['device_ms'] = HYBRID_VALUES['static_none']['coarse']/2
                    else:
                        row['full_gpu']['device_ms'] -= GPU_GAP['static_sleep']
        verdicts = self.summary(change=no_gap_effect)['hypotheses']
        self.assertEqual(verdicts['H1_gap_slows_first_gpu_stage']['verdict'], 'rejected')
        self.assertEqual(verdicts['H4_8k_full_gpu_attribution']['verdict'], 'neither')
        def gap_on_refine(name, arm, block, data):
            if name == C512 and arm in (f'{HYBRID}__static_sleep', f'{HYBRID}__static_decode'):
                for row in data['runs'][1:]:
                    row['refinement_device_ms'] = 18.
        self.assertEqual(self.summary(change=gap_on_refine)['hypotheses']['H2_new_pose_buffers_slow_refinement']['verdict'], 'confounded')
        def no_update_on_prepare(name, arm, block, data):
            if name == C512 and arm.startswith(f'{HYBRID}__reuse'):
                for row in data['runs'][1:]:
                    row['prepare_ms'] = 2.
        self.assertEqual(self.summary(change=no_update_on_prepare)['hypotheses']['H3_new_source_slows_frame_prepare']['verdict'], 'rejected')
        def update_on_8k(name, arm, block, data):
            if name == C8K and arm.startswith(f'{G1}__reuse'):
                for row in data['runs'][1:]:
                    row['full_gpu']['device_ms'] += 3.
        self.assertEqual(self.summary(change=update_on_8k)['hypotheses']['H4_8k_full_gpu_attribution']['verdict'], 'both')

    def test_static_rows_must_keep_source_and_pose_context(self):
        mutations = [lambda row: row.update(source_generation=1), lambda row: row.update(pose_create_ms=1.),
                     lambda row: row.update(source_copy_bytes=1), lambda row: row.update(pose_invalidated=True)]
        for index, mutate in enumerate(mutations):
            def change(name, arm, block, data):
                if name == C128 and arm == f'{HYBRID}__static_decode' and block == 1:
                    mutate(data['runs'][-1])
            with self.subTest(index=index):
                self.assertIn('sequence_route', self.gates(self.summary(change=change)))

    def test_reuse_rows_must_update_and_rebuild(self):
        mutations = [lambda row: row.update(source_generation=0), lambda row: row.update(pose_create_ms=0.),
                     lambda row: row.update(pose_invalidated=False)]
        for index, mutate in enumerate(mutations):
            def change(name, arm, block, data):
                if name == C512 and arm == f'{HYBRID}__reuse_none' and block == 2:
                    mutate(data['runs'][-1])
            with self.subTest(index=index):
                self.assertIn('sequence_route', self.gates(self.summary(change=change)))

    def test_gap_contract_failures_retained(self):
        mutations = [(f'{G1}__static_sleep', lambda row: row.update(gap_ms=bench.SLEEP_MS[C8K]-5.)),
                     (f'{G1}__static_sleep', lambda row: row.update(gap_ms=bench.SLEEP_MS[C8K]+80.)),
                     (f'{G1}__reuse_none', lambda row: row.update(gap_ms=3.)),
                     (f'{G1}__static_decode', lambda row: row.update(source_decode_ms=0., gap_ms=0.))]
        for index, (target, mutate) in enumerate(mutations):
            def change(name, arm, block, data):
                if name == C8K and arm == target and block == 0:
                    mutate(data['runs'][5])
            with self.subTest(index=index):
                summary = self.summary(change=change)
                self.assertIn('gap_contract', self.gates(summary))

    def test_gt_parity_and_peak2_equality_are_separate(self):
        def extra(name, arm, block, data):
            if name == C128 and arm == f'{G1}__static_sleep' and block == 2:
                data['runs'][1]['matches'].append(dict(pos_x=150., pos_y=20., angle=0., scale=1., score=.9))
        summary = self.summary(change=extra)
        self.assertIn('independent_ground_truth', self.gates(summary))
        self.assertFalse(summary['all_accuracy_pass'])
        def reorder(name, arm, block, data):
            if name == C8K and arm.startswith(G2):
                for row in data['runs']:
                    row['proposals'].reverse()
        summary = self.summary(change=reorder)
        self.assertTrue(summary['all_accuracy_pass'])
        self.assertEqual(self.gates(summary), {'peak2_equals_peak1'})
        def one_arm(name, arm, block, data):
            if name == C512 and arm == f'{HYBRID}__reuse_none':
                for row in data['runs']:
                    row['matches'].reverse()
        self.assertEqual(self.gates(self.summary(change=one_arm)), {'cross_arm_output_parity'})

    def test_residency_algorithm_and_exactness_failures_retained(self):
        mutations = [(f'{G2}__reuse_decode', lambda data: data['runs'][3]['full_gpu'].update(cpu_waits=2), 'route_and_residency'),
                     (f'{G2}__static_none', lambda data: [row['full_gpu'].update(peak_mode=1) for row in data['runs']], 'route_and_residency'),
                     (f'{G1}__static_none', lambda data: [row['full_gpu'].update(fine_simd_width=16, fine_effective_mode=2) for row in data['runs']],
                      'requested_algorithm_not_executed'),
                     (f'{HYBRID}__static_none', lambda data: data.update(exact_sequence_repeat=False), 'exact_repeat'),
                     (f'{HYBRID}__static_none', lambda data: data.update(drained=False), 'sequence_route')]
        for index, (target, mutate, gate) in enumerate(mutations):
            def change(name, arm, block, data):
                if name == C8K and arm == target:
                    mutate(data)
            with self.subTest(index=index):
                self.assertIn(gate, self.gates(self.summary(change=change)))

    def test_short_warmup_and_host_busy_retained(self):
        def short(name, arm, block, data):
            if name == C512 and arm == f'{G1}__reuse_none' and block == 1:
                for index, row in enumerate(data['runs']):
                    row['elapsed_ms'] = 400.*(index+1)+sum(r['gap_ms']+r['frame_wall_ms'] for r in data['runs'][:index+1])
        failures = [(row['gate'], row['case'], row['arm'], row.get('block')) for row in self.summary(change=short)['failed_gates']]
        self.assertEqual(failures, [('warmup_shorter_than_1s', C512, f'{G1}__reuse_none', 1)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest['executions'][7]['host']['competing_processes'] = [dict(command='clang++', pcpu=90.)]
            self.assertIn('host_busy', self.gates(bench.summarize(manifest, root)))
            manifest['executions'][7]['host']['competing_processes'] = []
            manifest['executions'][9]['host']['load_average'] = [bench.HOST_LOAD_LIMIT+.5, 1., 1.]
            loaded = [(row['arm'], row['block']) for row in bench.summarize(manifest, root)['failed_gates'] if row['gate'] == 'host_load_high']
            self.assertEqual(loaded, [(manifest['executions'][9]['arm'], manifest['executions'][9]['block'])])
            manifest['executions'][9]['host']['load_average'] = [bench.HOST_LOAD_LIMIT, 1., 1.]
            self.assertEqual(bench.summarize(manifest, root)['failed_gates'], [])
            for host in [None, dict(manifest['executions'][9]['host'], load_average=None), dict(manifest['executions'][9]['host'], load_average=['1', 1, 1])]:
                changed = copy.deepcopy(manifest)
                changed['executions'][9]['host'] = host
                with self.subTest(host=host), self.assertRaises(ValueError):
                    bench.summarize(changed, root)

    def test_competing_process_detection(self):
        # ucomm names and full argument lines as two ps listings, joined by pid.
        names = {10: (90., 'Python'), 11: (795.2, 'Blender'), 12: (14.1, 'WindowServer'), 13: (0., 'python3.12'),
                 14: (150.6, 'spotlightknowledged.updater'), 15: (99.9, 'Codex (Service)'), 16: (100., 'Python'),
                 17: (3., 'clang++'), 18: (0., 'fpm_native_pipeline_runner')}
        args = {10: '/opt/homebrew/Python.app/Contents/MacOS/Python cpp-core/tools/benchmark_chroma_margin.py holdout',
                11: '/Applications/Blender.app/Contents/MacOS/Blender --background --python testbed_blender_render.py',
                12: 'WindowServer -daemon', 13: '/Users/me/.cache/uv/bin/python /Users/me/.cache/uv/bin/blender-mcp',
                14: '/usr/libexec/spotlightknowledged.updater', 15: 'Codex (Service)', 16: 'Python driver.py',
                17: 'clang++ -c x.cpp', 18: 'fpm_native_pipeline_runner job.json'}
        found = bench.competing_processes(names, args, own={16})
        self.assertEqual(sorted((row['command'], row['reason']) for row in found),
                         [('Blender', 'tool'), ('benchmark_chroma_margin.py', 'script'), ('clang++', 'tool'),
                          ('fpm_native_pipeline_runner', 'tool'), ('spotlightknowledged.updater', 'cpu')])

    def test_sequence_structure_violations_rejected(self):
        mutations = [lambda data: data.update(sequence_gap='none'), lambda data: data.update(sequence_mode='reuse'),
                     lambda data: data.update(sequence_gap_sleep_ms=10.), lambda data: data['runs'].pop(),
                     lambda data: data['runs'][2].update(call=3), lambda data: data['runs'][2].update(sequence_gap='sleep'),
                     lambda data: data['runs'][2].update(frame_wall_ms=1.), lambda data: data['runs'][2].update(gap_ms=-1.),
                     lambda data: data['runs'][2].update(elapsed_ms=1.), lambda data: data['runs'][2].pop('gap_ms'),
                     lambda data: data['runs'][2].update(reference_cycle=True)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, C512, f'{HYBRID}__static_decode', 'result', mutate)
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_jobs_follow_round41_and_differ_only_in_gap_keys(self):
        case = make_case(C512)
        frames = bench.job(case, bench.spec(HYBRID, 'reuse', 'decode'), 1, 3)
        self.assertEqual(frames, bench.stage41.job(case, 'hybrid_best_frames', 1, 3))
        static = bench.job(case, bench.spec(HYBRID, 'static', 'sleep', 130), 1, 3)
        self.assertEqual({key for key in frames.keys() | static.keys() if frames.get(key) != static.get(key)},
                         {'sequence_mode', 'sequence_gap', 'sequence_gap_sleep_ms'})
        self.assertEqual((static['sequence_mode'], static['sequence_gap'], static['sequence_gap_sleep_ms']), ('static', 'sleep', 130.))
        none = bench.job(case, bench.spec(HYBRID, 'reuse', 'none'), 1, 3)
        self.assertEqual({key for key in frames.keys() | none.keys() if frames.get(key) != none.get(key)}, {'sequence_gap'})
        peak2 = bench.job(case, bench.spec(G2, 'reuse', 'decode'), 1, 3)
        peak1 = bench.job(case, bench.spec(G1, 'reuse', 'decode'), 1, 3)
        self.assertEqual({key for key in peak1.keys() | peak2.keys() if peak1.get(key) != peak2.get(key)},
                         {'full_gpu_peak_mode', 'full_gpu_peak_batch'})
        self.assertEqual((peak2['full_gpu_peak_mode'], peak2['full_gpu_peak_batch']), (2, 8))
        profiled = bench.job(case, bench.spec(G1, 'static', 'none'), 1, 3, True)
        self.assertTrue(profiled['full_gpu_profiling'])
        for row in (frames, static, none, peak2, profiled):
            self.assertFalse({'truth', 'truths', 'ground_truth', 'source'} & row.keys())

    def test_changed_jobs_and_manifest_guards_rejected(self):
        for changes in [dict(sequence_gap='decode'), dict(sequence_gap_sleep_ms=130.), dict(repeats=9), dict(full_gpu_peak_mode=1),
                        dict(full_gpu_profiling=True), dict(truth=[]), dict(sequence_mode='fresh')]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                self.revise(root, manifest, C8K, f'{G2}__static_sleep', 'job', lambda row: row.update(changes))
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            for changes in [dict(complete=False), dict(unchanged=False), dict(after_sha256={}), dict(blocks=2), dict(measured_calls=1),
                            dict(phase='nonsense'), dict(warmup_calls=dict(manifest['warmup_calls'], **{C128: 5})),
                            dict(sleep_ms=dict(manifest['sleep_ms'], **{C8K: 30})), dict(cases=manifest['cases'][:2]),
                            dict(executions=manifest['executions'][:-1]), dict(executions=manifest['executions']*2)]:
                with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                    bench.summarize(dict(manifest, **changes), root)
            changed = copy.deepcopy(manifest)
            changed['arms'][C128] = changed['arms'][C128][:-1]
            with self.assertRaises(ValueError):
                bench.summarize(changed, root)
            changed = copy.deepcopy(manifest)
            changed['arm_specs'][C8K][f'{G1}__static_sleep']['sleep_ms'] = 30
            with self.assertRaises(ValueError):
                bench.summarize(changed, root)

    def test_dose_phase_reports_levels_and_monotone_trend(self):
        def dose(name, arm, block, data):
            spec = bench.parse_arm(arm)
            for row in data['runs'][1:]:
                if spec['method'] == HYBRID:
                    for model in row['work']:
                        model['device_ms'] = 5.+spec['sleep_ms']/100
                else:
                    row['full_gpu']['device_ms'] = GPU_BASE[spec['method']]+spec['sleep_ms']/30
        summary = self.summary(phase='dose', change=dose)
        self.assertEqual(summary['failed_gates'], [])
        g1 = self.case(summary, C8K)['dose'][G1]
        self.assertEqual(g1['levels_ms'], [0, 10, 30, 100, 300])
        self.assertEqual([row['median'] for row in g1['device_ms']], [60., 60.+1/3, 61., 60.+10/3, 70.])
        self.assertTrue(g1['monotone_nondecreasing'])
        self.assertEqual(g1['top_vs_none']['direction'], 'up')
        hybrid = self.case(summary, C512)['dose'][HYBRID]
        self.assertEqual([row['median'] for row in hybrid['coarse_device_ms']], [10., 10.2, 10.6, 12., 16.])
        def dip(name, arm, block, data):
            if name == C8K and arm == f'{G1}__static_sleep100':
                for row in data['runs'][1:]:
                    row['full_gpu']['device_ms'] = 55.
        g1 = self.case(self.summary(phase='dose', change=dip), C8K)['dose'][G1]
        self.assertEqual([row['median'] for row in g1['device_ms']], [60., 65., 65., 55., 65.])
        self.assertFalse(g1['monotone_nondecreasing'])

    def test_profile_phase_reports_stages(self):
        # Measured calls only, as the protocol states: cold and warm-up stage times must not enter.
        summary = self.summary(phase='profile', blocks=2)
        stages = self.case(summary, C8K)['arms'][f'{G1}__static_sleep']['profile_stages']
        self.assertEqual(stages['coarse']['median'], (GPU_BASE[G1]+5.)/2-1.)
        self.assertEqual(stages['fine']['median'], (GPU_BASE[G1]+5.)/2)

    def test_sleep_and_warmup_constants_follow_round41(self):
        summary = dict(cases=[dict(name=name, arms={'hybrid_best_frames': dict(source_decode_ms=dict(median=value))})
                              for name, value in [(C128, 65.5), (C512, 129.7), (C8K, 131.4)]])
        self.assertEqual(bench.sleep_from_round41(summary), {C128: 66, C512: 130, C8K: 131})
        self.assertEqual(bench.WARMUP_CALLS, {C128: 2, C512: 1, C8K: 23})
        archived = json.loads((bench.DOCS/'stage41-new-frame-cost/formal-summary.json').read_text())
        self.assertEqual(bench.SLEEP_MS, bench.sleep_from_round41(archived))

    def test_offline_replay_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            expected = bench.summarize(manifest, root)
            bench.save(root/'manifest.json', manifest)
            actual = bench.summarize(json.loads((root/'manifest.json').read_text()), root)
            self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(actual, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
