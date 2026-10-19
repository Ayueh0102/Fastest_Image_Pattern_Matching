"""Movable fusion replay and negative controls; no Metal jobs are launched."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from benchmark_fine_fused import (METHODS, CLOCKS, FUSION_COUNTS, FUSION_CLOCKS,
                                  job_for, route, summarize, sha, save, sum_gpu_clock)


class FusionHarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        fixture = json.loads((Path(__file__).parent / 'fixtures/fine_search_harness.json').read_text())
        self.case = dict(fixture['case'], expect_reuse=True, benchmark_repeats=3)
        reference = dict(first=fixture['first'], runs=[copy.deepcopy(fixture['first']) for _ in range(3)],
                         exact_repeat=True, construct_ms=0, final_memory=fixture['final_memory'])
        self.report = dict(version=1, methods=METHODS, blocks=2, repeats=3, warm_seconds=1.,
                           cases=[self.case], expected_workers=6, execution_order=[], complete=False)
        save(self.base / 'protocol.json', self.report)
        self.report['protocol_sha256'] = sha(self.base / 'protocol.json')
        for block in range(2):
            for method in (METHODS if block == 0 else METHODS[::-1]):
                raw = copy.deepcopy(reference)
                for run in [raw['first'], *raw['runs']]:
                    ds, fs = [], []
                    for w in run['work']:
                        fields = ('calls', 'requests', 'within_call_duplicates', 'cross_call_duplicates',
                                  'untracked_call_keys', 'tracked_unique_keys', 'history_limit_keys',
                                  'history_key_payload_bytes', 'history_bucket_count',
                                  'peak_call_key_capacity_bytes', 'warp_batches', 'zero_repair_batches',
                                  'guard_words', 'nonzero_guard_words')
                        d = dict.fromkeys((*CLOCKS, *fields), 0)
                        d.update(profiling_enabled=0, angle_reuse_enabled=1, warp_device_ms=-1,
                                 logical_requests=w['fine_maps'], reused_maps=1, cache_peak_bytes=1024)
                        w['fine_maps'] -= 1
                        ds.append(d)
                        f = dict.fromkeys((*FUSION_COUNTS, *FUSION_CLOCKS), 0)
                        f['fused_device_ms'] = -1
                        if method == 'fused':
                            w['fine_commands'] = w['fine_stats_dispatches']
                            w['fine_stats_device_ms'] = -1
                            f.update(enabled=1, fused_batches=w['fine_stats_dispatches'],
                                     changed_pixels=1, corrected_windows=5,
                                     correction_ms=.01, fused_wait_ms=w['fine_wait_ms'], fused_device_ms=3.)
                        fs.append(f)
                    run['fine_search'] = ds
                    if method != 'frozen':
                        run['fine_fusion'] = fs
                job = self.base / f'{method}{block}.json'
                result = self.base / f'{method}{block}.result.json'
                save(job, job_for(self.case, method))
                save(result, raw)
                self.report['execution_order'].append(dict(case=self.case['name'], method=method, block=block,
                    job=job.name, job_sha256=sha(job), result=result.name, result_sha256=sha(result), returncode=0))
        self.report.update(complete=True, unchanged=True)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, method, fn):
        for entry in self.report['execution_order']:
            if entry['method'] != method:
                continue
            path = self.base / entry['result']
            raw = json.loads(path.read_text())
            fn(raw)
            save(path, raw)
            entry['result_sha256'] = sha(path)

    def test_valid_and_relocatable(self):
        summary = summarize(self.report, self.base)
        self.assertTrue(summary['correctness_passed'])
        self.assertTrue(summary['all_performance_tolerances_passed'])
        self.assertEqual(summary['formal_calls'], 18)
        with tempfile.TemporaryDirectory() as other:
            moved = Path(other) / 'relocated'
            shutil.copytree(self.base, moved)
            self.assertEqual(summary, summarize(self.report, moved))

    def test_worker_has_no_ground_truth_and_frozen_has_no_new_switch(self):
        for method in METHODS:
            job = job_for(self.case, method)
            self.assertNotIn('truth', job)
            self.assertNotIn('expect_reuse', job)
            self.assertTrue(job['fine_angle_reuse'])
            self.assertEqual(job['repeats'], 3)
        self.assertNotIn('fine_fused', job_for(self.case, 'frozen'))
        with self.assertRaises(ValueError):
            job_for(self.case, 'unknown')

    def test_worker_truth_leak_rejected(self):
        entry = self.report['execution_order'][0]
        path = self.base / entry['job']
        job = json.loads(path.read_text())
        job['truth'] = []
        save(path, job)
        entry['job_sha256'] = sha(path)
        with self.assertRaises(ValueError):
            summarize(self.report, self.base)

    def test_raw_hash_tamper_rejected(self):
        (self.base / self.report['execution_order'][0]['result']).write_text('{}')
        with self.assertRaises(ValueError):
            summarize(self.report, self.base)

    def test_duplicate_missing_reorder_or_failure_rejected(self):
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

    def test_changed_protocol_rejected(self):
        self.report['cases'][0]['score'] = .1
        with self.assertRaises(ValueError):
            summarize(self.report, self.base)

    def test_duplicate_cases_rejected_even_with_updated_protocol_hash(self):
        self.report['cases'].append(copy.deepcopy(self.case))
        save(self.base / 'protocol.json', {k: v for k, v in self.report.items() if k not in ('protocol_sha256', 'unchanged')})
        self.report['protocol_sha256'] = sha(self.base / 'protocol.json')
        with self.assertRaises(ValueError):
            summarize(self.report, self.base)

    def test_incorrect_reuse_accounting_rejected(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['fine_search'][0].update(reused_maps=999))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_incorrect_fused_command_count_rejected(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['work'][0].update(fine_commands=0))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_fallback_batch_accounted_without_claiming_fusion(self):
        def fallback(raw):
            for run in [raw['first'], *raw['runs']]:
                for f, w in zip(run['fine_fusion'], run['work']):
                    f['fused_batches'] -= 1
                    f['fallback_batches'] += 1
                    w['fine_commands'] += 1
        self.mutate('fused', fallback)
        self.assertTrue(summarize(self.report, self.base)['correctness_passed'])

    def test_all_fallback_cannot_claim_successful_fusion(self):
        def fallback(raw):
            for run in [raw['first'], *raw['runs']]:
                for f, w in zip(run['fine_fusion'], run['work']):
                    f.update(fused_batches=0, fallback_batches=w['fine_stats_dispatches'], fused_device_ms=-1)
                    w['fine_commands'] = 2 * w['fine_stats_dispatches']
        self.mutate('fused', fallback)
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_combined_device_timestamp_cannot_be_reported_as_ncc(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['work'][0].update(fine_stats_device_ms=3.))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_disabled_fusion_metrics_must_be_empty(self):
        self.mutate('current', lambda raw: raw['runs'][0]['fine_fusion'][0].update(changed_pixels=1))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_ordered_output_equality(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['proposals'][0].update(score=.123456))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_all_missed_retains_null_errors(self):
        self.mutate('fused', lambda raw: [r.update(matches=[]) for r in [raw['first'], *raw['runs']]])
        summary = summarize(self.report, self.base)
        self.assertFalse(summary['correctness_passed'])
        accuracy = summary['cases'][0]['methods']['fused']['accuracy']
        self.assertGreater(accuracy['max_fn'], 0)
        self.assertTrue(all(v is None for v in accuracy['max_errors'].values()))

    def test_perf_regression_is_retained_separately_from_correctness(self):
        def regression(raw):
            raw['final_memory']['peak_rss_bytes'] *= 3
            for r in raw['runs']:
                r['total_ms'] *= 3
        self.mutate('fused', regression)
        summary = summarize(self.report, self.base)
        self.assertTrue(summary['correctness_passed'])
        self.assertFalse(summary['all_performance_tolerances_passed'])
        for pair in summary['cases'][0]['comparisons']['fused_vs_current']:
            self.assertFalse(pair['rss_tolerance_pass'])
            self.assertFalse(pair['time_tolerance_pass'])

    def test_finite_and_missing_metric_checks(self):
        raw = json.loads((self.base / self.report['execution_order'][0]['result']).read_text())
        run = raw['runs'][0]
        for field, value in (('total_ms', float('nan')), ('total_ms', 0)):
            bad = copy.deepcopy(run)
            bad[field] = value
            self.assertFalse(route(bad, run, 'frozen'))
        bad = copy.deepcopy(run)
        del bad['fine_search']
        self.assertFalse(route(bad, run, 'frozen'))

    def test_wrong_sample_count_rejected(self):
        self.mutate('fused', lambda raw: raw['runs'].pop())
        with self.assertRaises(ValueError):
            summarize(self.report, self.base)

    def test_changed_pixels_cannot_exceed_physical_repairs(self):
        def excessive(raw):
            run = raw['runs'][0]
            run['fine_fusion'][0]['changed_pixels'] = run['work'][0]['fine_repair_pixels'] + 1
        self.mutate('fused', excessive)
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_corrected_windows_bound_is_seven_by_seven(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['fine_fusion'][0].update(corrected_windows=50))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_fused_wait_cannot_exceed_total_wait(self):
        def excessive(raw):
            run = raw['runs'][0]
            run['fine_fusion'][0]['fused_wait_ms'] = run['work'][0]['fine_wait_ms'] + 1e-5
        self.mutate('fused', excessive)
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_fused_wait_must_equal_total_without_fallback(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['fine_fusion'][0].update(fused_wait_ms=0))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])

    def test_fused_wait_tolerates_only_rounding(self):
        def roundoff(raw):
            run = raw['runs'][0]
            run['fine_fusion'][0]['fused_wait_ms'] = run['work'][0]['fine_wait_ms'] + 1e-8
        self.mutate('fused', roundoff)
        self.assertTrue(summarize(self.report, self.base)['correctness_passed'])

    def test_dense_maps_require_actual_changes(self):
        self.mutate('fused', lambda raw: raw['runs'][0]['fine_fusion'][0].update(
            dense_maps=1, changed_pixels=0, corrected_windows=0))
        self.assertFalse(summarize(self.report, self.base)['correctness_passed'])


    def test_gpu_clock_unavailable_sentinel_survives_aggregation(self):
        self.assertEqual(sum_gpu_clock([1.5, 2.5]), 4.)
        self.assertEqual(sum_gpu_clock([0., 0.]), 0.)
        self.assertEqual(sum_gpu_clock([-1, -1, -1]), -1)
        self.assertEqual(sum_gpu_clock([7., -1, 3.]), -1)

    def test_multimodel_mixed_gpu_clock_summary(self):
        def two_models(raw):
            for run in [raw['first'], *raw['runs']]:
                run['work'].append(copy.deepcopy(run['work'][0]))
                run['fine_search'].append(copy.deepcopy(run['fine_search'][0]))
                run['models_matched'] += 1
                if 'fine_fusion' in run:
                    run['fine_fusion'].append(copy.deepcopy(run['fine_fusion'][0]))
                    f = run['fine_fusion'][1]
                    w = run['work'][1]
                    if f['enabled']:
                        f.update(fused_batches=0, fallback_batches=w['fine_stats_dispatches'],
                                 fused_device_ms=-1, fused_wait_ms=0,
                                 changed_pixels=0, corrected_windows=0, correction_ms=0)
                        w['fine_commands'] = 2 * w['fine_stats_dispatches']
        for method in METHODS:
            self.mutate(method, two_models)
        summary = summarize(self.report, self.base)
        self.assertTrue(summary['correctness_passed'])
        methods = summary['cases'][0]['methods']
        self.assertEqual(methods['fused']['stages']['fine_stats_device_ms']['values'], [-1] * 6)
        self.assertEqual(methods['fused']['fusion']['fused_device_ms']['values'], [-1] * 6)
        self.assertEqual(methods['current']['fusion']['fused_device_ms']['values'], [-1] * 6)
        # Ordinary counters and timings must still be summed, not sentinel-aware.
        self.assertEqual(methods['fused']['fusion']['changed_pixels']['values'], [1] * 6)
        self.assertEqual(methods['fused']['reuse']['reused_maps']['values'], [2] * 6)


if __name__ == '__main__':
    unittest.main()
