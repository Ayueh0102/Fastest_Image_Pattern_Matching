"""Pure-data checks for profiling acceptance; no C++ calls or GPU workloads."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_pipeline_stages as bench
import summarize_pipeline_stages as summarize


def profile_record():
    profile = {key: .1 for key in bench.EXCLUSIVE_STAGES}
    profile.update(version=1, struct_bytes=296, backend=1, reserved=0,
        core_wall_ms=1., api_wall_ms=1.2, api_residual_ms=.2, coarse_maps=2,
        pyramid_levels=2, pyramid_generated_bytes=16, pyramid_live_bytes=80,
        scoremap_peak_bytes=128, scoremap_total_bytes=256)
    return dict(scale=1., profile=profile, metrics=dict(score_maps=2))


def worker_result(method='cpu_profile'):
    record = dict(matches=[dict(pos_x=1., pos_y=2., score=.99)],
        patterns=[profile_record()] if method.endswith('_profile') else [],
        profile_errors=[], truth_accuracy=dict(pose_tp=1, pose_fp=0, pose_fn=0),
        total_wall_ms=2., proposal_wall_ms=1.5, call_wall_ms=2.1)
    return dict(method=method, block=0, first=copy.deepcopy(record),
        warmups=[copy.deepcopy(record)], runs=[copy.deepcopy(record)],
        source_shape=[8, 8], construct_ms=1.,
        memory=dict(after_measurements=dict(process_peak_rss_bytes=1024, pose_owned_buffers=None)))


def archive_fixture(root):
    """One fully synthetic worker/report, with no library or matching calls."""
    original = root/'original'/'formal-jobs'/'fixture.result.json'
    original.parent.mkdir(parents=True)
    result = worker_result('frozen_cpu')
    for record in bench.calls(result):
        record['output_sha256'] = bench.output_digest(record)
    content = json.dumps(result).encode()
    original.write_bytes(content)
    entry = dict(result=str(original), result_sha256=hashlib.sha256(content).hexdigest(),
                 reference_only=False, case='fixture', method='frozen_cpu')
    archive = root/'archive'
    raw = archive/'formal-jobs'/'fixture.result.json'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(content)
    report = archive/'formal.json'
    row = bench.summarize_case(dict(name='fixture', workflow='core_only'), [result], result)
    report.write_text(json.dumps(dict(execution_order=[entry], cases=[row],
                                     passed=True, complete=True, design={})))
    return original, archive, raw, report, entry


class PipelineAcceptanceTest(unittest.TestCase):
    def test_explicit_archive_root_never_falls_back_to_original(self):
        with tempfile.TemporaryDirectory() as directory:
            original, archive, raw, report, entry = archive_fixture(Path(directory))
            self.assertEqual(summarize.resolve_worker_path(entry, report, archive), raw.resolve())
            self.assertEqual(summarize.resolve_worker_path(entry, report, raw.parent), raw.resolve())
            raw.unlink()
            self.assertTrue(original.is_file())
            with self.assertRaises(FileNotFoundError):
                summarize.load_worker(entry, report, archive)
            raw.write_bytes(b'wrong archived result')
            with self.assertRaisesRegex(ValueError, 'Raw worker result changed'):
                summarize.load_worker(entry, report, archive)

    def test_archive_sha_mismatch_is_rejected_without_trying_another_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            original, _, raw, report, entry = archive_fixture(Path(directory))
            original.write_bytes(b'wrong original result')
            # The automatic lookup must reject the existing corrupt original,
            # even though an intact archive copy is available next to report.
            with self.assertRaisesRegex(ValueError, 'Raw worker result changed'):
                summarize.load_worker(entry, report)
            original.unlink()
            logical, actual, loaded = summarize.load_worker(entry, report)
            self.assertEqual(logical, 'formal-jobs/fixture.result.json')
            self.assertEqual(actual, summarize.sha(raw))
            self.assertEqual(loaded['method'], 'frozen_cpu')

    def test_relocated_archive_reproduces_identical_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, archive, _, _, _ = archive_fixture(root)
            original.unlink()
            relocated = root/'relocated'
            shutil.copytree(archive, relocated)
            summaries = []
            for location in (archive, relocated):
                output = io.StringIO()
                argv = ['summarize_pipeline_stages.py', str(location/'formal.json'),
                        '--worker-root', str(location)]
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
                    summarize.main()
                summaries.append(output.getvalue())
            self.assertEqual(summaries[0], summaries[1])
            result = json.loads(summaries[0])
            self.assertEqual(result['cases'][0]['methods']['frozen_cpu']['total_ms']['median'], 2.)
            self.assertEqual(list(result['raw_worker_sha256']), ['formal-jobs/fixture.result.json'])

    def test_performance_gate_requires_each_matching_block(self):
        methods = dict(frozen_metal=dict(blocks=[dict(block=0, total_ms=10.), dict(block=1, total_ms=40.)]),
                       combined=dict(blocks=[dict(block=0, total_ms=12.), dict(block=1, total_ms=44.)]))
        self.assertTrue(summarize.paired_regression(methods, 'combined', 'frozen_metal')['passed'])
        methods['combined']['blocks'][1]['total_ms'] = 44.001
        self.assertFalse(summarize.paired_regression(methods, 'combined', 'frozen_metal')['passed'])
        self.assertIsNone(summarize.paired_regression(methods, 'combined', 'absent')['passed'])

    def test_exclusive_times_and_nested_maps(self):
        record = profile_record()
        self.assertEqual(bench.profile_checks([record]), [])
        bad = copy.deepcopy(record)
        bad['profile']['coarse_ncc_ms'] += .01
        self.assertIn('exclusive stage sum differs from core wall', bench.profile_checks([bad]))
        bad = copy.deepcopy(record)
        bad['profile']['api_residual_ms'] = .4
        self.assertIn('API residual does not reconcile', bench.profile_checks([bad]))
        bad = copy.deepcopy(record)
        bad['metrics']['score_maps'] = 1
        self.assertIn('Metal score map count differs from core count', bench.profile_checks([bad]))
        bad = copy.deepcopy(record)
        bad['profile']['fine_ncc_ms'] = float('nan')
        self.assertIn('nonfinite/negative profile time', bench.profile_checks([bad]))

    def test_warmups_and_reference_repeats_are_checked(self):
        case = dict(name='fixture', truth=[{}])
        reference, result = worker_result('frozen_cpu'), worker_result()
        valid = bench.summarize_case(case, [result], reference)
        self.assertTrue(valid['exact_all_outputs'])
        self.assertTrue(valid['truth_passed'])
        self.assertEqual(valid['methods']['cpu_profile']['nominal_output_buffers'][
            'cpu_pyramid_horizontal_scratch_peak_proxy_bytes'], 128)
        result['warmups'][0]['matches'][0]['score'] = .98
        self.assertFalse(bench.summarize_case(case, [result], reference)['exact_all_outputs'])
        result = worker_result()
        result['runs'][0]['truth_accuracy']['pose_fn'] = 1
        self.assertFalse(bench.summarize_case(case, [result], reference)['truth_passed'])
        reference['runs'][0]['matches'][0]['score'] = .97
        invalid = bench.summarize_case(case, [worker_result()], reference)
        self.assertFalse(invalid['reference_all_calls_exact'])
        self.assertFalse(invalid['exact_all_outputs'])

    def test_src7_does_not_claim_truth(self):
        result, reference = worker_result(), worker_result('frozen_cpu')
        summary = bench.summarize_case(dict(name='src7'), [result], reference)
        self.assertIsNone(summary['truth_passed'])
        self.assertIsNone(summary['methods']['cpu_profile']['truth_passed'])

    def test_compact_warmup_digest_preserves_exact_gate(self):
        result, reference = worker_result(), worker_result('frozen_cpu')
        result['warmups'] = [dict(compact_warmup=True,
            output_sha256=bench.output_digest(reference['first']), profile_errors=[])]
        summary = bench.summarize_case(dict(name='fixture', truth=[{}]), [result], reference)
        self.assertTrue(summary['exact_all_outputs'])
        self.assertTrue(summary['truth_passed'])
        result['warmups'][0]['output_sha256'] = 'wrong'
        summary = bench.summarize_case(dict(name='fixture', truth=[{}]), [result], reference)
        self.assertFalse(summary['exact_all_outputs'])
        self.assertFalse(summary['truth_passed'])

    def test_profile_overhead_uses_matching_order_block(self):
        plain, profiled = worker_result('cpu'), worker_result()
        profiled['runs'][0]['proposal_wall_ms'] = 1.65
        reference = worker_result('frozen_cpu')
        summary = bench.summarize_case(dict(name='fixture'), [plain, profiled], reference)
        self.assertAlmostEqual(summary['profile_overhead_percent_by_block']['cpu'][0]['proposal_percent'], 10.)

    def test_strict_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = [root/'core.so', root/'coarse.so', root/'pose.so']
            for index, path in enumerate(libraries):
                path.write_bytes(bytes([index]))
            manifest = dict(commit=bench.BASELINE_COMMIT,
                            files={p.name: bench.file_hash(p) for p in libraries})
            manifest_path = root/'baseline-provenance.json'
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(bench, 'core_library_path', return_value=libraries[0]), \
                 patch.object(bench, 'coarse_library_path', return_value=libraries[1]), \
                 patch.object(bench, 'pose_library_path', return_value=libraries[2]):
                self.assertTrue(bench.verify_baseline(root)['verified'])
                libraries[1].write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'differs from manifest'):
                    bench.verify_baseline(root)
                manifest['commit'] = '958ab79'
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, '91842dd'):
                    bench.verify_baseline(root)


if __name__ == '__main__':
    unittest.main()
