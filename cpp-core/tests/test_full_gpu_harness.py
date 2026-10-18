"""Negative controls for residency, ground-truth accuracy, and result archival."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_full_gpu as bench


class FullGpuHarness(unittest.TestCase):
    def fixture(self, root, change=None, truth=None):
        poses = [dict(pos_x=50., pos_y=50., angle=0., scale=1., score=.95),
                 dict(pos_x=100., pos_y=80., angle=20., scale=1., score=.94)]
        case = dict(name='one', truth=copy.deepcopy(poses) if truth is None else truth,
                    source='source.png', template='template.png', scales=[1.], angle=180,
                    limit=8, score=.8, overlap=.2, stop_layer=0, subpixel=1, workflow='full_rescue')
        hashes = dict(source_sha256={'source': '1'}, binary_sha256={'worker': '2'}, input_sha256={'input': '3'})
        manifest = dict(cases=[case], blocks=1, repeats=1, warmup_seconds=0., methods=list(bench.METHODS),
                        executions=[], complete=True, unchanged=True, after_sha256=copy.deepcopy(hashes), **hashes)
        for method in bench.METHODS:
            row = dict(total_ms=10., prepare_ms=1., proposal_ms=7., refine_ms=1., nms_ms=1.,
                       stage_timings_available=method != 'full_gpu',
                       matches=copy.deepcopy(poses), proposals=copy.deepcopy(poses),
                       frame_coarse_gpu_bytes=100, frame_fine_gpu_bytes=100, work=[],
                       full_gpu=dict(enabled=method == 'full_gpu', command_buffers=1, cpu_waits=1,
                                     intermediate_readbacks=0, cpu_algorithm_steps=0, gpu_bytes=300,
                                     upload_bytes=120, readback_bytes=40, encode_ms=.1, wait_ms=4., device_ms=3.,
                                     device_timestamp_available=True))
            if method == 'full_gpu':
                row['total_ms'] = 5.
                if change:
                    change(row)
            worker = dict(first=row, runs=[copy.deepcopy(row)], exact_repeat=True, construct_ms=1.,
                          final_memory=dict(peak_rss_bytes=1000000))
            jp, rp = root/(method+'.job.json'), root/(method+'.result.json')
            bench.save(jp, bench.job(case, method, 1, 0.))
            bench.save(rp, worker)
            manifest['executions'].append(dict(case='one', method=method, block=0, job=jp.name,
                result=rp.name, job_sha256=bench.sha(jp), result_sha256=bench.sha(rp)))
        return manifest

    def test_positive_and_negative_residency_accuracy_performance(self):
        for change, expected in [
            (None, True),
            (lambda row: row['matches'][0].update(angle=10.), False),
            (lambda row: row['matches'].pop(), False),
            (lambda row: row['matches'].append(copy.deepcopy(row['matches'][0])), False),
            (lambda row: row['full_gpu'].update(enabled=False), False),
            (lambda row: row['full_gpu'].update(command_buffers=2), False),
            (lambda row: row['full_gpu'].update(cpu_waits=2), False),
            (lambda row: row['full_gpu'].update(intermediate_readbacks=1), False),
            (lambda row: row['full_gpu'].update(cpu_algorithm_steps=1), False),
            (lambda row: row['full_gpu'].update(gpu_bytes=0), False),
            (lambda row: row.update(total_ms=20.), False),
        ]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = bench.summarize(self.fixture(root, change), root)
                self.assertEqual(summary['all_full_gpu_adoption_gates_pass'], expected)

    def test_order_and_small_fp32_differences_are_not_false_failures(self):
        def change(row):
            row['matches'].reverse()
            row['matches'][0]['score'] -= 1e-7
            row['matches'][0]['pos_x'] += .01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = bench.summarize(self.fixture(root, change), root)
            measured = result['cases'][0]['methods']['full_gpu']
            self.assertTrue(result['all_full_gpu_adoption_gates_pass'])
            self.assertFalse(measured['ordered_matches_equal_control'])
            self.assertTrue(measured['strict_pose_equivalence_to_control'])
            self.assertGreater(measured['max_matched_score_delta_from_control'], 0.)
            self.assertEqual(measured['retained_gpu_bytes'], 300)

    def test_empty_predictions_and_empty_truth_are_valid_negative_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root, truth=[])
            for entry in manifest['executions']:
                path = root/entry['result']
                worker = json.loads(path.read_text())
                for row in [worker['first'], *worker['runs']]:
                    row.update(matches=[], proposals=[])
                bench.save(path, worker)
                entry['result_sha256'] = bench.sha(path)
            summary = bench.summarize(manifest, root)
            self.assertTrue(summary['all_full_gpu_adoption_gates_pass'])
            self.assertIsNone(summary['cases'][0]['methods']['full_gpu']['scores'])

    def test_missing_duplicate_changed_or_extra_workers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            for invalid in (dict(manifest, complete=False), dict(manifest, unchanged=False),
                            dict(manifest, blocks=2), dict(manifest, repeats=2),
                            dict(manifest, executions=manifest['executions']*2),
                            dict(manifest, executions=manifest['executions']+[dict(manifest['executions'][0], case='unknown')]),
                            dict(manifest, methods=['full_gpu']), dict(manifest, after_sha256={})):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    bench.summarize(invalid, root)
            (root/manifest['executions'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):
                bench.summarize(manifest, root)

    def test_wrong_worker_settings_and_truth_leaks_rejected(self):
        for changes in (dict(full_gpu=True), dict(coarse_peak_mode=2), dict(score=.1),
                        dict(truth=[]), dict(repeats=2)):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                entry = manifest['executions'][0]
                path = root/entry['job']
                data = json.loads(path.read_text())
                data.update(changes)
                bench.save(path, data)
                entry['job_sha256'] = bench.sha(path)
                with self.assertRaises(ValueError):
                    bench.summarize(manifest, root)

    def test_invalid_or_missing_residency_metrics_rejected(self):
        for change in (lambda row: row['full_gpu'].pop('enabled'),
                       lambda row: row['full_gpu'].update(device_ms=-1.),
                       lambda row: row['full_gpu'].update(cpu_waits=1.),
                       lambda row: row.update(total_ms=0),
                       lambda row: row['matches'][0].update(scale=0),
                       lambda row: row['matches'][0].update(score=2.)):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    bench.summarize(self.fixture(root, change), root)

    def test_unavailable_timestamps_and_stages_do_not_fail_residency(self):
        def unavailable(row):
            row['full_gpu'].update(device_ms=None, device_timestamp_available=False)
            row.update(prepare_ms=None, proposal_ms=None, refine_ms=None, nms_ms=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, unavailable), root)
            measured = summary['cases'][0]['methods']['full_gpu']
            self.assertTrue(summary['all_full_gpu_resident'])
            self.assertTrue(summary['all_full_gpu_adoption_gates_pass'])
            self.assertEqual(measured['stages']['total_ms']['median'], 5.)
            self.assertTrue(all(measured['stages'][key] is None for key in bench.STAGE_FIELDS if key != 'total_ms'))
            self.assertIsNone(measured['residency_metrics']['device_ms'])
            self.assertEqual(measured['device_timestamp_availability'],
                dict(available_calls=0, total_calls=2, legacy_assumed_available_calls=0))
            self.assertEqual(measured['stage_timing_availability']['available_calls'], 0)

    def test_legacy_numeric_timestamps_and_stages_remain_replayable(self):
        def legacy(row):
            row.pop('stage_timings_available')
            row['full_gpu'].pop('device_timestamp_available')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = bench.summarize(self.fixture(root, legacy), root)
            measured = summary['cases'][0]['methods']['full_gpu']
            self.assertTrue(summary['all_full_gpu_adoption_gates_pass'])
            self.assertEqual(measured['residency_metrics']['device_ms']['median'], 3.)
            self.assertEqual(measured['stages']['proposal_ms']['median'], 7.)
            self.assertEqual(measured['device_timestamp_availability'],
                dict(available_calls=2, total_calls=2, legacy_assumed_available_calls=2))
            self.assertEqual(measured['stage_timing_availability']['legacy_assumed_available_calls'], 2)

    def test_missing_or_inconsistent_availability_flags_rejected(self):
        def missing_timestamp_flag(row):
            row['full_gpu'].pop('device_timestamp_available')
            row['full_gpu']['device_ms'] = None
        def missing_timestamp_value(row):
            row['full_gpu'].update(device_timestamp_available=False)
            row['full_gpu'].pop('device_ms')
        for change in (missing_timestamp_flag, missing_timestamp_value,
                       lambda row: row['full_gpu'].update(device_timestamp_available=1),
                       lambda row: row['full_gpu'].update(device_timestamp_available=False),
                       lambda row: row['full_gpu'].update(device_timestamp_available=True, device_ms=None),
                       lambda row: row.update(stage_timings_available=None),
                       lambda row: row.update(stage_timings_available=1),
                       lambda row: row.update(stage_timings_available=True, proposal_ms=None)):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    bench.summarize(self.fixture(root, change), root)

    def test_mixed_timestamp_availability_counts_only_measured_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            entry = next(entry for entry in manifest['executions'] if entry['method'] == 'full_gpu')
            path = root/entry['result']
            worker = json.loads(path.read_text())
            worker['first']['full_gpu'].update(device_timestamp_available=False, device_ms=None)
            bench.save(path, worker)
            entry['result_sha256'] = bench.sha(path)
            summary = bench.summarize(manifest, root)
            measured = summary['cases'][0]['methods']['full_gpu']
            self.assertEqual(measured['residency_metrics']['device_ms']['values'], [3.])
            self.assertEqual(measured['device_timestamp_availability']['available_calls'], 1)
            self.assertEqual(measured['device_timestamp_availability']['total_calls'], 2)
            self.assertTrue(summary['all_full_gpu_resident'])

    def test_truth_never_sent_and_method_selection_is_explicit(self):
        case = dict(name='one', truth=[dict(secret=1)], source='s', template='t', scales=[1.],
                    angle=180, limit=8, score=.8, overlap=.2, stop_layer=0, subpixel=1, workflow='full_rescue')
        for method, expected in bench.METHODS.items():
            worker_job = bench.job(case, method, 3, 1.)
            self.assertNotIn('truth', worker_job)
            self.assertNotIn('name', worker_job)
            self.assertEqual({key: worker_job[key] for key in expected}, expected)
            self.assertEqual(worker_job['refinement_backend'], 1)
        for value in ('', 'full_gpu,full_gpu', 'unknown'):
            with self.assertRaises(ValueError):
                bench.selected_names(value, bench.METHODS, 'methods')

    def test_worker_paths_cannot_escape_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                bench.worker_path(Path(directory), '../outside.json')


if __name__ == '__main__':
    unittest.main()
