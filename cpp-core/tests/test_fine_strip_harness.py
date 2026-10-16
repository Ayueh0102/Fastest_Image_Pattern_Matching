"""Fail-closed experiment bookkeeping and ground-truth isolation checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_fine_strip as bench


class FineStripHarnessTest(unittest.TestCase):
    def test_cold_and_warm_counters_are_distinct(self):
        def sample(n):
            return {k:n for k in ('maps','command_buffers','roi_pixels','repair_pixels','stats_dispatches','peak_batch','gpu_bytes','statistics_bytes')}
        first, warm = sample(2), sample(1)
        expected = bench.counters(dict(first=first, runs=[warm]*10), 'micro')
        self.assertNotEqual(expected, bench.counters(dict(first=warm, runs=[first]*10), 'micro'))
        warm['wall_ms'] = 999.
        self.assertEqual(expected, bench.counters(dict(first=first, runs=[warm]*10), 'micro'))

    def test_frozen_binary_must_match_saved_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);(root/'experiments').mkdir()
            hashes = {}
            for name in ('fpm_native_pipeline_runner','libfpm_coarse_experiment.dylib','libfpm_pose_experiment.dylib'):
                p = root/'experiments'/name;p.write_bytes(name.encode())
                hashes['/old/fine-reuse/experiments/'+name] = bench.sha(p)
            protocol = root/'protocol.json';bench.save(protocol, dict(binary_sha256=hashes))
            self.assertEqual(len(bench.verify_baseline(root, protocol)), 3)
            (root/'experiments/fpm_native_pipeline_runner').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                bench.verify_baseline(root, protocol)

    def test_no_true_positives_remain_reportable(self):
        self.assertIsNone(bench.max_present([None, None]))
        self.assertEqual(bench.max_present([None, .5, 0.]), .5)

    def test_failed_summary_returns_failure(self):
        with tempfile.TemporaryDirectory() as d:
            report = Path(d)/'formal.json';bench.save(report, {})
            output = Path(d)/'summary.json'
            for correct, unchanged in ((False, True), (True, False)):
                result = dict(correctness_passed=correct, unchanged=unchanged)
                with patch.object(sys, 'argv', ['bench', '--summarize', str(report), '--output', str(output)]), patch.object(bench, 'summarize', return_value=result):
                    with self.assertRaises(SystemExit) as error:
                        bench.main()
                    self.assertEqual(error.exception.code, 1)
                    self.assertEqual(json.loads(output.read_text()), result)

    def test_truth_and_descriptive_metadata_do_not_enter_worker(self):
        case = dict(source='image', template='template', scales=[1.], angle=180,
                    limit=8, score=.8, overlap=.2, stop_layer=1, subpixel=0,
                    workflow='full_rescue', truth=[dict(secret='truth')],
                    placement_coordinates=[1, 2], name='test')
        for method in ('frozen', 'original', 'strip32', 'strip16'):
            job = bench.full_job(case, method)
            self.assertNotIn('truth', job)
            self.assertNotIn('placement_coordinates', job)
            self.assertEqual(job['ncc_mode'], 2)
            self.assertFalse(job['overlap_enabled'])
            self.assertEqual(job['repeats'], 5)
            self.assertEqual(job['warmup_seconds'], 1.)
            if method == 'frozen':
                self.assertNotIn('fine_ncc_mode', job)
            else:
                self.assertEqual(job['fine_ncc_mode'], bench.MODES[method])

    def test_incomplete_runs_are_rejected(self):
        for report in (dict(complete=False), dict(complete=True, execution_order=[], expected_workers=76)):
            with self.assertRaises(ValueError):
                bench.summarize(report, Path('.'))

    def test_changed_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'sample.json'
            bench.save(p, dict(time=1))
            digest = bench.sha(p)
            self.assertEqual(bench.checked(p, digest), dict(time=1))
            bench.save(p, dict(time=2))
            with self.assertRaises(ValueError):
                bench.checked(p, digest)

    def test_failed_worker_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'job.json';bench.save(p, {})
            report = dict(complete=True, expected_workers=1, execution_order=[dict(
                job='job.json', job_sha256=bench.sha(p), returncode=1)])
            with self.assertRaises(ValueError):
                bench.summarize(report, Path(d))

    def test_accidental_truth_is_rejected_on_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'job.json';bench.save(p, dict(truth=[]))
            report = dict(complete=True, expected_workers=1, execution_order=[dict(
                job='job.json', job_sha256=bench.sha(p), returncode=0)])
            with self.assertRaises(ValueError):
                bench.summarize(report, Path(d))


if __name__ == '__main__':
    unittest.main()
