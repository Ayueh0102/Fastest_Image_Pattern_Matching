"""Stage33 CPU-only synthetic replay and negative controls; no native/GPU calls."""
import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from benchmark_resident_multiscale import (METHODS, CASE_NAMES, WORK_COUNTS, WORK_CLOCKS,
    SEARCH_COUNTS, CLOCKS, FUSION_COUNTS, FUSION_CLOCKS, REFINEMENT_COUNTS,
    REFINEMENT_CLOCKS, POSE_BYTES, BINARY_NAMES, baseline, ensure_empty_output,
    job_for, route, image_shape, loader_valid, summarize, sha, save, worker_order)


class ResidentMultiscaleHarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.fixture = json.loads((Path(__file__).parent / 'fixtures/fine_search_harness.json').read_text())
        self.make_report()

    def tearDown(self):
        self.temp.cleanup()

    def reference(self, method, first, negative=False):
        run = copy.deepcopy(self.fixture['first'])
        work = dict.fromkeys((*WORK_COUNTS, *WORK_CLOCKS), 0)
        work.update(maps=2, windows=10, fine_maps=3, fine_ncc_mode=2, fine_stats_dispatches=1,
                    fine_commands=1, fine_stats_device_ms=-1, fine_wait_ms=3., fine_repair_pixels=1)
        search = dict.fromkeys((*SEARCH_COUNTS, *CLOCKS), 0)
        search.update(angle_reuse_enabled=1, logical_requests=4, reused_maps=1, cache_peak_bytes=1024,
                      warp_device_ms=-1)
        fusion = dict.fromkeys((*FUSION_COUNTS, *FUSION_CLOCKS), 0)
        fusion.update(enabled=1, fused_batches=1, fused_wait_ms=3., fused_device_ms=2.,
                      changed_pixels=1, corrected_windows=1)
        run.update(work=[work], fine_search=[search], fine_fusion=[fusion], generation=1 if first else 2,
            total_ms=20. if first else 10., models_matched=1, selected_refinement_backend=1,
            refinement_mode=METHODS.index(method), pose_input_mode=1, pose_storage_mode=1,
            pose_host_image_bytes=32768, pose_gpu_image_bytes=32768, pose_shared_image_bytes=32768,
            pose_unique_image_bytes=32768, pose_host_bytes=32868, pose_gpu_bytes=32968,
            pose_conversion_temporary_bytes=0, pose_create_ms=2. if first else 0.)
        run.update(dict.fromkeys(REFINEMENT_COUNTS, 0))
        run.update(dict.fromkeys(REFINEMENT_CLOCKS, 0.))
        run.update(pose_create_ms=2. if first else 0., refinement_dispatches=8 if method == 'resident0' else 1,
            refinement_command_buffers=1, refinement_cpu_waits=1, refinement_auxiliary_bytes=200,
            refinement_device_ms=1.)
        if method != 'resident0':
            run.update(refinement_threads=256 if method == 'fused256' else 128, refinement_simd_width=32,
                       refinement_threadgroup_bytes=16384)
        if negative:
            run.update(matches=[], proposals=[], selected_refinement_backend=0)
            run.update(dict.fromkeys((*POSE_BYTES, *REFINEMENT_COUNTS, *REFINEMENT_CLOCKS), 0))
            work.update(fine_maps=0, fine_stats_dispatches=0, fine_commands=0, fine_wait_ms=0., fine_repair_pixels=0)
            search.update(logical_requests=0, reused_maps=0, cache_peak_bytes=0)
            fusion.update(fused_batches=0, fused_wait_ms=0., fused_device_ms=-1, changed_pixels=0, corrected_windows=0)
        return run

    def make_report(self):
        cases = [dict(self.fixture['case'], name=name, workflow='full_rescue', source_shape=[32, 16],
            template_shape=[7, 3], validity='negative' if name == 'empty_scene' else 'challenge' if name == 'degraded12' else 'normal',
            origin='new' if name in CASE_NAMES[4:] else 'existing', expect_refine=name != 'empty_scene') for name in CASE_NAMES]
        cases[-1]['truth'] = []
        self.case = cases[0]
        binaries = {'/prior/run-diagnostics/experiments/' + n: n for n in BINARY_NAMES}
        prior = self.base / 'baseline-protocol.json'
        save(prior, dict(binary_sha256=binaries))
        self.report = dict(version=1, methods=METHODS, blocks=2, repeats=1, warm_seconds=0., memory_page_bytes=16384,
            cases=cases, expected_workers=len(worker_order(cases)), execution_order=[], complete=False,
            source_sha256={'source': 'hash'}, binary_sha256=binaries, input_sha256={'input': 'hash'},
            baseline_protocol_sha256=sha(prior), frozen_binary_verified=dict(zip(BINARY_NAMES, BINARY_NAMES)))
        save(self.base / 'protocol.json', self.report)
        self.report['protocol_sha256'] = sha(self.base / 'protocol.json')
        for name, method, block in worker_order(cases):
            case = next(c for c in cases if c['name'] == name)
            first, repeat = [self.reference(method, first, name == 'empty_scene') for first in (True, False)]
            raw = dict(first=first, runs=[repeat], exact_repeat=True, construct_ms=1., warmups_ms=[],
                final_memory=self.fixture['final_memory'],
                load=dict(mode='direct_gray8', decode_ms=1., source=dict(direct_gray8=True, rgb_temporary_bytes=0),
                          template=dict(direct_gray8=True, rgb_temporary_bytes=0)),
                after_load_memory=dict(current_rss_bytes=1024, peak_rss_bytes=2048))
            job = self.base / f'{name}-{method}-{block}.json'
            result, stderr = job.with_suffix('.result.json'), job.with_suffix('.stderr.txt')
            save(job, job_for(case, method)); save(result, raw); stderr.write_text('')
            self.report['execution_order'].append(dict(case=name, method=method, block=block,
                job=job.name, job_sha256=sha(job), result=result.name, result_sha256=sha(result),
                stderr=stderr.name, stderr_sha256=sha(stderr), returncode=0))
        self.report.update(complete=True, unchanged=True,
            post_sha256={k: copy.deepcopy(self.report[k]) for k in ('source_sha256', 'binary_sha256', 'input_sha256')})

    def mutate(self, method, fn, case=None, block=None):
        for entry in self.report['execution_order']:
            if ((method is not None and entry['method'] != method) or
                    (case is not None and entry['case'] != case) or (block is not None and entry['block'] != block)):
                continue
            path = self.base / entry['result']
            raw = json.loads(path.read_text()); fn(raw); save(path, raw)
            entry['result_sha256'] = sha(path)

    def summary(self):
        return summarize(self.report, self.base)

    def test_valid_96_calls_relocatable(self):
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertTrue(result['all_performance_tolerances_passed'])
        self.assertEqual((result['workers'], result['algorithm_calls'], result['first_calls'], result['repeat_calls']), (48, 96, 48, 48))
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / 'moved'; shutil.copytree(self.base, moved)
            self.assertEqual(result, summarize(self.report, moved))

    def test_jobs_keep_truth_metadata_out(self):
        case = dict(self.case, truth=[{'hidden': 1}], roi=[0, 0, 2, 2], metadata='secret', renderer_seed=9)
        for mode, method in enumerate(METHODS):
            job = job_for(case, method)
            self.assertEqual((job['pose_resident_mode'], job['pose_input_mode'], job['pose_storage_mode']), (mode, 1, 1))
            self.assertEqual((job['repeats'], job['warmup_seconds']), (1, 0.))
            self.assertTrue(job['fine_fused'] and job['fine_angle_reuse'] and job['refine'])
            self.assertFalse(job['run_resource_diagnostics'] or job['overlap_enabled'])
            self.assertFalse(set(job) & {'truth', 'roi', 'metadata', 'renderer_seed', 'validity', 'expect_refine', 'source_shape'})

    def test_missing_duplicate_order_failed_workers(self):
        for kind in ('missing', 'duplicate', 'reorder', 'failed'):
            report = copy.deepcopy(self.report)
            if kind == 'missing': report['execution_order'].pop()
            elif kind == 'duplicate': report['execution_order'][1] = report['execution_order'][0]
            elif kind == 'reorder': report['execution_order'][:2] = report['execution_order'][1::-1]
            else: report['execution_order'][0]['returncode'] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): summarize(report, self.base)

    def test_hash_and_semantic_tamper(self):
        entry = self.report['execution_order'][0]
        for field in ('job', 'result', 'stderr'):
            p = self.base / entry[field]; original = p.read_bytes(); p.write_text('{}')
            with self.subTest(field=field), self.assertRaises(ValueError): self.summary()
            p.write_bytes(original)
        job = self.base / entry['job']; value = json.loads(job.read_text()); value['pose_input_mode'] = 0
        save(job, value); entry['job_sha256'] = sha(job)
        with self.assertRaises(ValueError): self.summary()

    def test_hidden_warmup_and_repeat_count_rejected(self):
        for change in (lambda r: r.update(warmups_ms=[1.]), lambda r: r['runs'].pop()):
            self.make_report(); self.mutate('simd128', change, case='controlled6')
            with self.assertRaises(ValueError): self.summary()

    def test_frontend_work_change_cannot_hide_across_whole_block(self):
        self.mutate(None, lambda raw: raw['first']['work'][0].update(windows=999), case='controlled6', block=1)
        self.assertFalse(self.summary()['integrity_passed'])

    def test_missing_repeat_flag_and_reordered_proposals(self):
        self.mutate('fused256', lambda raw: raw.update(exact_repeat=False), case='controlled6')
        self.assertFalse(self.summary()['integrity_passed'])
        self.make_report()
        self.mutate('simd128', lambda raw: raw['first']['proposals'].reverse(), case='controlled6')
        self.assertFalse(self.summary()['integrity_passed'])

    def test_protocol_prepost_and_baseline_provenance(self):
        report = copy.deepcopy(self.report); report['repeats'] = 2
        with self.assertRaises(ValueError): summarize(report, self.base)
        for field in ('source_sha256', 'binary_sha256', 'input_sha256'):
            report = copy.deepcopy(self.report); report['post_sha256'][field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError): summarize(report, self.base)
        (self.base / 'baseline-protocol.json').write_text('{}')
        with self.assertRaises(ValueError): self.summary()

    def test_bad_route_info_and_backend(self):
        changes = [('selected_refinement_backend', 0), ('refinement_mode', 0), ('refinement_threads', 256),
            ('refinement_simd_width', 16), ('refinement_threadgroup_bytes', 999999),
            ('refinement_dispatches', 2), ('refinement_command_buffers', 0), ('refinement_cpu_waits', 0)]
        for key, value in changes:
            run = self.reference('simd128', True); run[key] = value
            self.assertFalse(route(run, self.reference('resident0', True), 'simd128', self.case, True, 16384), key)

    def test_missing_counters_even_in_common_reference(self):
        for family, field in [('work', 'fine_commands'), ('work', 'async_submitted'),
                              ('fine_search', 'logical_requests'), ('fine_fusion', 'changed_pixels')]:
            run = self.reference('resident0', True); del run[family][0][field]
            self.assertFalse(route(run, run, 'resident0', self.case, True, 16384), field)
        run = self.reference('simd128', True); del run['refinement_auxiliary_bytes']
        self.assertFalse(route(run, self.reference('resident0', True), 'simd128', self.case, True, 16384))

    def test_shared_capacity_formulas_and_initialization(self):
        for key in (*POSE_BYTES, 'pose_input_mode', 'pose_storage_mode'):
            run = self.reference('simd128', True); run[key] += 1
            # Host and GPU total workspace are allowed to differ by method.
            if key in ('pose_host_bytes', 'pose_gpu_bytes'): continue
            self.assertFalse(route(run, self.reference('resident0', True), 'simd128', self.case, True, 16384), key)
        run = self.reference('simd128', False); run['pose_create_ms'] = 1.
        self.assertFalse(route(run, self.reference('resident0', False), 'simd128', self.case, False, 16384))
        run = self.reference('simd128', True); run['pose_gpu_bytes'] = 32767
        self.assertFalse(route(run, self.reference('resident0', True), 'simd128', self.case, True, 16384))

    def test_alternate_workspace_does_not_require_exact(self):
        run = self.reference('simd128', True)
        run.update(pose_host_bytes=40000, pose_gpu_bytes=36000, refinement_auxiliary_bytes=3232)
        self.assertTrue(route(run, self.reference('resident0', True), 'simd128', self.case, True, 16384))

    def test_duplicate_final_pairing_cannot_pass(self):
        self.mutate('simd128', lambda raw: [r.update(matches=[r['matches'][0], copy.deepcopy(r['matches'][0])])
                                          for r in [raw['first'], *raw['runs']]], case='controlled6')
        result = self.summary(); self.assertTrue(result['integrity_passed'])
        self.assertFalse(result['paired_final_passed']); self.assertFalse(result['accuracy_passed'])

    def test_common_gt_errors_separate_from_integrity(self):
        self.mutate(None, lambda raw: [r.update(matches=[]) for r in [raw['first'], *raw['runs']]], case='new_small12')
        result = self.summary()
        self.assertTrue(result['integrity_passed'] and result['paired_final_passed'])
        self.assertFalse(result['normal_accuracy_passed'] or result['correctness_passed'])
        self.assertEqual(len(result['cases']), 8)
        self.assertEqual(len(result['accuracy_failures']), 3)

    def test_challenge_errors_retained(self):
        self.mutate(None, lambda raw: [r.update(matches=[]) for r in [raw['first'], *raw['runs']]], case='degraded12')
        result = self.summary(); self.assertTrue(result['normal_accuracy_passed'])
        self.assertFalse(result['accuracy_passed'] or result['correctness_passed'])
        self.assertTrue(all(f['validity'] == 'challenge' for f in result['accuracy_failures']))

    def test_empty_negative_is_valid_zero_execution(self):
        result = self.summary(); empty = result['cases'][-1]
        for method in METHODS:
            self.assertTrue(empty['methods'][method]['route'])
            self.assertEqual((empty['methods'][method]['accuracy']['max_fp'], empty['methods'][method]['accuracy']['max_fn']), (0, 0))
        self.mutate('simd128', lambda raw: raw['first'].update(refinement_threads=128), case='empty_scene')
        self.assertFalse(self.summary()['integrity_passed'])

    def test_zero_positive_route_is_not_counted_as_executed(self):
        run = self.reference('simd128', True, True)
        self.assertFalse(route(run, run, 'simd128', self.case, True, 16384))

    def test_final_differences_within_bounds_need_not_be_exact(self):
        def alter(raw):
            for run in [raw['first'], *raw['runs']]: run['matches'][0]['pos_x'] += .1
        self.mutate('simd128', alter, case='controlled6')
        result = self.summary(); self.assertTrue(result['correctness_passed'])
        method = result['cases'][0]['methods']['simd128']
        self.assertFalse(method['final_exact']); self.assertTrue(method['final_paired'])

    def test_performance_failures_retained_per_phase(self):
        self.mutate('simd128', lambda raw: raw['first'].update(total_ms=40.), case='controlled6', block=1)
        self.mutate('fused256', lambda raw: raw['final_memory'].update(peak_rss_bytes=2**32), case='new_texture12', block=0)
        result = self.summary(); self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_performance_tolerances_passed'])
        self.assertTrue(any(f['failed_time_phases'] == ['first'] for f in result['performance_failures']))
        self.assertTrue(any(f['rss_failed'] for f in result['performance_failures']))

    def test_loader_fallback_equal_and_capacity_required(self):
        entry = self.report['execution_order'][0]
        raw = json.loads((self.base / entry['result']).read_text())
        raw['load']['template'].update(direct_gray8=False, rgb_temporary_bytes=100)
        self.assertTrue(loader_valid(raw, raw))
        bad = copy.deepcopy(raw); bad['load']['template']['rgb_temporary_bytes'] = 0
        self.assertFalse(loader_valid(bad, bad)); self.assertFalse(loader_valid(bad, raw))

    def test_frozen_hash_and_output_overwrite(self):
        build = self.base / 'build'; records = {}
        for name in BINARY_NAMES:
            path = build / 'experiments' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
            records['/prior/run-diagnostics/experiments/' + name] = sha(path)
        prior = self.base / 'prior.json'; save(prior, dict(binary_sha256=records))
        self.assertEqual(set(baseline(build, prior)), set(BINARY_NAMES))
        path.write_text('changed')
        with self.assertRaises(ValueError): baseline(build, prior)
        with self.assertRaises(ValueError): ensure_empty_output(self.base)

    def test_png_and_bmp_shapes_without_decoding(self):
        path = self.base / 'shape'
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 21, 11))
        self.assertEqual(image_shape(path), [21, 11])
        path.write_bytes(b'BM' + bytes(12) + struct.pack('<Iii', 40, 21, -11) + bytes(28))
        self.assertEqual(image_shape(path), [21, 11])
        path.write_bytes(b'invalid')
        with self.assertRaises(ValueError): image_shape(path)


if __name__ == '__main__':
    unittest.main()
