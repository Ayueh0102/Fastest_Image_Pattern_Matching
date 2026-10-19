"""Grayscale-loader benchmark replay/negative controls; launches no Metal work."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from benchmark_fine_fused import CLOCKS, FUSION_COUNTS, FUSION_CLOCKS
from benchmark_gray_loader import (METHODS, BINARY_NAMES, baseline, ensure_empty_output,
                                   job_for, nonclock_state, route, loader_route,
                                   summarize, sha, save, worker_order)


class GrayLoaderHarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        fixture = json.loads((Path(__file__).parent / 'fixtures/fine_search_harness.json').read_text())
        self.case = dict(fixture['case'], expect_reuse=True, benchmark_repeats=1)
        first = fixture['first']
        ds, fs = [], []
        for work in first['work']:
            fields = ('calls', 'requests', 'within_call_duplicates', 'cross_call_duplicates',
                      'untracked_call_keys', 'tracked_unique_keys', 'history_limit_keys',
                      'history_key_payload_bytes', 'history_bucket_count', 'peak_call_key_capacity_bytes',
                      'warp_batches', 'zero_repair_batches', 'guard_words', 'nonzero_guard_words')
            diagnostics = dict.fromkeys((*CLOCKS, *fields), 0)
            diagnostics.update(profiling_enabled=0, angle_reuse_enabled=1, warp_device_ms=-1,
                               logical_requests=work['fine_maps'], reused_maps=1, cache_peak_bytes=1024)
            work['fine_maps'] -= 1
            work['fine_commands'] = work['fine_stats_dispatches']
            work['fine_stats_device_ms'] = -1
            fusion = dict.fromkeys((*FUSION_COUNTS, *FUSION_CLOCKS), 0)
            fusion.update(enabled=1, fused_batches=work['fine_stats_dispatches'],
                          changed_pixels=1, corrected_windows=5,
                          correction_ms=.01, fused_wait_ms=work['fine_wait_ms'], fused_device_ms=3.)
            ds.append(diagnostics)
            fs.append(fusion)
        first.update(fine_search=ds, fine_fusion=fs, total_ms=20.)
        repeat = copy.deepcopy(first)
        repeat.update(generation=2, total_ms=10.)
        # Distinct first/repeat upload counts must not be mistaken for drift.
        first['work'][0].update(fine_uploads=1, fine_upload_bytes=1024)
        reference = dict(first=first, runs=[repeat], exact_repeat=True,
                         construct_ms=1., warmups_ms=[], final_memory=fixture['final_memory'])
        self.report = dict(version=1, methods=METHODS, blocks=2, repeats=1, warm_seconds=0.,
                           cases=[self.case], expected_workers=6, execution_order=[], complete=False)
        save(self.base / 'protocol.json', self.report)
        self.report['protocol_sha256'] = sha(self.base / 'protocol.json')
        for name, method, block in worker_order([self.case]):
            raw = copy.deepcopy(reference)
            if method != 'frozen':
                direct = method == 'current_direct'
                info = dict(direct_gray8=direct, rgb_temporary_bytes=0 if direct else 4096)
                raw['load'] = dict(mode='direct_gray8' if direct else 'legacy_rgb',
                                   decode_ms=1. if direct else 2.,
                                   source=copy.deepcopy(info), template=copy.deepcopy(info))
                raw['after_load_memory'] = dict(current_rss_bytes=1024 * (1 if direct else 2),
                                               peak_rss_bytes=1024 * (2 if direct else 3))
            job = self.base / f'{method}{block}.json'
            result = self.base / f'{method}{block}.result.json'
            save(job, job_for(self.case, method))
            save(result, raw)
            self.report['execution_order'].append(dict(case=name, method=method, block=block,
                job=job.name, job_sha256=sha(job), result=result.name, result_sha256=sha(result), returncode=0))
        self.report.update(complete=True, unchanged=True)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, method, fn, block=None):
        for entry in self.report['execution_order']:
            if entry['method'] != method or (block is not None and entry['block'] != block):
                continue
            path = self.base / entry['result']
            raw = json.loads(path.read_text())
            fn(raw)
            save(path, raw)
            entry['result_sha256'] = sha(path)

    def summary(self):
        return summarize(self.report, self.base)

    def test_valid_and_relocatable(self):
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertTrue(result['all_performance_tolerances_passed'])
        self.assertEqual((result['workers'], result['algorithm_calls'], result['warmup_calls']), (6, 12, 0))
        self.assertEqual(result['first_calls'], result['repeat_calls'])
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / 'relocated'
            shutil.copytree(self.base, moved)
            self.assertEqual(result, summarize(self.report, moved))

    def test_worker_has_no_truth_and_all_arms_have_same_algorithm(self):
        for method in METHODS:
            job = job_for(self.case, method)
            self.assertNotIn('truth', job)
            self.assertNotIn('expect_reuse', job)
            self.assertTrue(job['fine_fused'])
            self.assertTrue(job['fine_angle_reuse'])
            self.assertEqual(job['fine_ncc_mode'], 2)
            self.assertEqual((job['repeats'], job['warmup_seconds']), (1, 0.))
        self.assertNotIn('gray_load_mode', job_for(self.case, 'frozen'))
        self.assertEqual(job_for(self.case, 'current_legacy')['gray_load_mode'], 'legacy_rgb')
        self.assertEqual(job_for(self.case, 'current_direct')['gray_load_mode'], 'direct_gray8')
        with self.assertRaises(ValueError):
            job_for(self.case, 'unknown')

    def test_frozen_loader_data_remains_unavailable(self):
        methods = self.summary()['cases'][0]['methods']
        self.assertIsNone(methods['frozen']['load'])
        self.assertIsNone(methods['frozen']['after_load_memory'])
        pair = self.summary()['cases'][0]['comparisons']['current_direct_vs_current_legacy'][0]
        self.assertEqual(pair['decode_delta_ms'], -1.)
        self.assertEqual(pair['after_load_current_rss_delta_bytes'], -1024)
        self.assertEqual(pair['after_load_peak_rss_delta_bytes'], -1024)

    def test_undisclosed_warmup_rejected(self):
        self.mutate('current_direct', lambda raw: raw.update(warmups_ms=[1.]))
        with self.assertRaises(ValueError):
            self.summary()

    def test_wrong_sample_count_rejected(self):
        self.mutate('current_direct', lambda raw: raw['runs'].append(copy.deepcopy(raw['runs'][0])))
        with self.assertRaises(ValueError):
            self.summary()

    def test_job_tamper_rejected_even_with_new_hash(self):
        entry = self.report['execution_order'][0]
        path = self.base / entry['job']
        job = json.loads(path.read_text())
        job['truth'] = []
        save(path, job)
        entry['job_sha256'] = sha(path)
        with self.assertRaises(ValueError):
            self.summary()

    def test_raw_and_protocol_hash_tamper_rejected(self):
        for filename in (self.report['execution_order'][0]['result'], 'protocol.json'):
            with self.subTest(filename=filename):
                path = self.base / filename
                original = path.read_bytes()
                path.write_text('{}')
                with self.assertRaises(ValueError):
                    self.summary()
                path.write_bytes(original)

    def test_duplicate_missing_reorder_or_failed_worker_rejected(self):
        for kind in ('duplicate', 'missing', 'reorder', 'failed'):
            with self.subTest(kind=kind):
                report = copy.deepcopy(self.report)
                if kind == 'duplicate':
                    report['execution_order'][1] = report['execution_order'][0]
                elif kind == 'missing':
                    report['execution_order'].pop()
                elif kind == 'reorder':
                    report['execution_order'][:2] = report['execution_order'][1::-1]
                else:
                    report['execution_order'][0]['returncode'] = 1
                with self.assertRaises(ValueError):
                    summarize(report, self.base)

    def test_changed_predeclared_protocol_rejected(self):
        self.report['cases'][0]['score'] = .1
        with self.assertRaises(ValueError):
            self.summary()

    def test_first_and_repeat_are_independent_performance_gates(self):
        self.mutate('current_direct', lambda raw: raw['first'].update(total_ms=100.), block=1)
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_performance_tolerances_passed'])
        pairs = result['cases'][0]['comparisons']['current_direct_vs_current_legacy']
        self.assertTrue(pairs[0]['first_time_tolerance_pass'])
        self.assertFalse(pairs[1]['first_time_tolerance_pass'])
        self.assertTrue(pairs[1]['repeat_time_tolerance_pass'])

    def test_single_block_rss_failure_retained(self):
        self.mutate('current_direct', lambda raw: raw['final_memory'].update(peak_rss_bytes=2 ** 32), block=0)
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_performance_tolerances_passed'])
        pairs = result['cases'][0]['comparisons']['current_direct_vs_current_legacy']
        self.assertFalse(pairs[0]['rss_tolerance_pass'])
        self.assertTrue(pairs[1]['rss_tolerance_pass'])

    def test_ordered_output_drift_rejected(self):
        self.mutate('current_direct', lambda raw: raw['first']['proposals'].reverse())
        self.assertFalse(self.summary()['correctness_passed'])

    def test_repeated_output_drift_rejected(self):
        self.mutate('current_legacy', lambda raw: raw['runs'][0]['matches'][0].update(score=.123))
        self.assertFalse(self.summary()['correctness_passed'])

    def test_pose_and_frame_capacities_checked_for_first_and_repeat(self):
        entry = self.report['execution_order'][0]
        reference = json.loads((self.base / entry['result']).read_text())['first']
        for key in ('frame_fine_gpu_bytes', 'pose_gpu_bytes', 'pose_host_image_bytes',
                    'pose_conversion_temporary_bytes', 'refinement_auxiliary_bytes'):
            with self.subTest(key=key):
                run = copy.deepcopy(reference)
                run[key] += 1
                self.assertFalse(route(run, reference))
        self.mutate('current_direct', lambda raw: raw['runs'][0].update(pose_gpu_bytes=42))
        self.assertFalse(self.summary()['correctness_passed'])

    def test_nonclock_coarse_fine_fusion_and_pose_work_checked(self):
        entry = self.report['execution_order'][0]
        ref = json.loads((self.base / entry['result']).read_text())['first']
        for family, key in [('work', 'async_completed'), ('work', 'fine_commands'),
                            ('fine_search', 'cache_peak_bytes'), ('fine_fusion', 'changed_pixels')]:
            with self.subTest(family=family, key=key):
                run = copy.deepcopy(ref)
                run[family][0][key] += 1
                self.assertFalse(route(run, ref))
        run = copy.deepcopy(ref)
        run['refinement_dispatches'] += 1
        self.assertFalse(route(run, ref))

    def test_only_clocks_are_excluded_from_algorithm_state(self):
        entry = self.report['execution_order'][0]
        ref = json.loads((self.base / entry['result']).read_text())['first']
        run = copy.deepcopy(ref)
        run['work'][0]['copy_ms'] *= 2
        run['pose_create_ms'] += 4
        self.assertEqual(nonclock_state(run), nonclock_state(ref))
        self.assertTrue(route(run, ref))

    def test_loader_route_and_temporary_bytes_validated(self):
        for method in ('current_direct', 'current_legacy'):
            entry = next(e for e in self.report['execution_order'] if e['method'] == method)
            original = json.loads((self.base / entry['result']).read_text())
            for kind in ('mode', 'bool', 'bytes', 'missing', 'negative_time'):
                with self.subTest(method=method, kind=kind):
                    raw = copy.deepcopy(original)
                    if kind == 'mode':
                        raw['load']['mode'] = 'unknown'
                    elif kind == 'bool':
                        raw['load']['source']['direct_gray8'] = not raw['load']['source']['direct_gray8']
                    elif kind == 'bytes':
                        raw['load']['template']['rgb_temporary_bytes'] = 4 if method == 'current_direct' else 0
                    elif kind == 'missing':
                        del raw['after_load_memory']
                    else:
                        raw['load']['decode_ms'] = -1
                    self.assertFalse(loader_route(raw, method))

    def test_missing_loader_data_is_reportable_failure(self):
        self.mutate('current_direct', lambda raw: raw.pop('load'))
        self.assertFalse(self.summary()['correctness_passed'])

    def test_ground_truth_failures_remain_reportable(self):
        self.mutate('current_direct', lambda raw: [r.update(matches=[]) for r in [raw['first'], *raw['runs']]])
        result = self.summary()
        self.assertFalse(result['correctness_passed'])
        accuracy = result['cases'][0]['methods']['current_direct']['accuracy']
        self.assertGreater(accuracy['max_fn'], 0)
        self.assertTrue(all(value is None for value in accuracy['max_errors'].values()))

    def test_nonfinite_memory_and_clock_rejected(self):
        self.mutate('current_direct', lambda raw: raw['final_memory'].update(current_rss_bytes=-1))
        with self.assertRaises(ValueError):
            self.summary()
        entry = self.report['execution_order'][0]
        run = json.loads((self.base / entry['result']).read_text())['first']
        bad = copy.deepcopy(run)
        bad['total_ms'] = float('nan')
        self.assertFalse(route(bad, run))

    def test_zero_algorithm_time_rejected_before_ratio_calculation(self):
        self.mutate('frozen', lambda raw: raw['first'].update(total_ms=0))
        with self.assertRaises(ValueError):
            self.summary()

    def test_baseline_binary_hashes_are_stage30_evidence(self):
        build = self.base / 'baseline'
        protocol = self.base / 'baseline.json'
        records = {}
        for name in BINARY_NAMES:
            path = build / 'experiments' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            records['/prior/fine-fused/experiments/' + name] = sha(path)
        save(protocol, dict(binary_sha256=records))
        self.assertEqual(set(baseline(build, protocol)), set(BINARY_NAMES))
        (build / 'experiments' / BINARY_NAMES[0]).write_text('changed')
        with self.assertRaises(ValueError):
            baseline(build, protocol)

    def test_nonempty_output_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            ensure_empty_output(self.base)
        empty = self.base / 'empty'
        ensure_empty_output(empty)
        empty.mkdir()
        ensure_empty_output(empty)
        file = self.base / 'file'
        file.write_text('existing')
        with self.assertRaises(ValueError):
            ensure_empty_output(file)


if __name__ == '__main__':
    unittest.main()
