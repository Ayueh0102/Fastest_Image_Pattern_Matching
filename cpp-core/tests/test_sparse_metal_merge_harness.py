"""Negative controls for the Metal source/search submission merge study."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_sparse_metal_merge as b


class MetalMergeHarnessTest(unittest.TestCase):
    def evaluate(self, mutate=None, omit=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pose = dict(pos_x=20., pos_y=20., angle=0., score=.95, scale=1.,
                        corners=[[10, 10], [30, 10], [30, 30], [10, 30]])
            case = dict(name='x', validity='fixed_scale_gt', truth=[pose])
            report = dict(protocol=dict(cases=[case], blocks=1, repeats=1,
                          proposal_basin=dict(position_px=6., angle_deg=6., relative_scale=.02)),
                          execution_order=[])
            for method in b.METHODS:
                if method == omit:
                    continue
                merged = method == 'merged'
                sparse = dict(enabled=True, source_levels=1, response_mode=1,
                              search_backend=1, gpu_buffer_bytes=1000,
                              command_buffers=2-int(merged), cpu_waits=2-int(merged),
                              dispatches=5, peak_overflow_retries=0,
                              gradient_mode=1, gradient_lookup_bytes=b.GRADIENT_LOOKUP_BYTES,
                              **{key: 1 for key in b.COUNTS})
                if method != 'frozen_metal':
                    sparse['metal_submission_mode'] = int(merged)
                duration = dict(frozen_metal=10., split=8., merged=4.)[method]
                call = dict(matches=[copy.deepcopy(pose)], proposals=[copy.deepcopy(pose)],
                            sparse=sparse, frame_host_bytes=0, selected_refinement_backend=0,
                            pose_gpu_bytes=0, total_ms=duration, prepare_ms=.1,
                            proposal_ms=.4, refine_ms=.5, nms_ms=.1)
                if mutate:
                    mutate(method, call)
                raw = dict(first=call, runs=[copy.deepcopy(call)], exact_repeat=True,
                           construct_ms=.2, final_memory=dict(peak_rss_bytes=10000))
                path = root / (method + '.json')
                path.write_text(json.dumps(raw))
                report['execution_order'].append(dict(case='x', method=method, block=0,
                                                     status='ok', result=path.name))
            return b.summarize(report, root)

    def test_exact_baseline(self):
        summary = self.evaluate()
        self.assertTrue(all(summary[key] for key in (
            'all_workers_succeeded', 'exact_proposals_all', 'exact_cpu_final_all',
            'route_integrity_all', 'synchronization_all', 'normal_gt_all')))

    def test_attribution_uses_same_build_split(self):
        pairs = {item['attribution']: item for item in self.evaluate()['cases'][0]['paired_comparison']}
        self.assertEqual(pairs['merged_vs_split']['speedup'], 2.)
        self.assertEqual(pairs['merged_vs_split']['reference'], 'split')
        self.assertEqual(pairs['split_vs_frozen']['speedup'], 1.25)
        self.assertEqual(pairs['merged_vs_frozen']['speedup'], 2.5)

    def test_changed_candidate(self):
        def change(method, call):
            if method == 'merged':
                call['proposals'][0]['score'] -= .001
        self.assertFalse(self.evaluate(change)['exact_proposals_all'])

    def test_changed_work(self):
        def change(method, call):
            if method == 'split':
                call['sparse']['feature_evaluations'] += 1
        self.assertFalse(self.evaluate(change)['exact_proposals_all'])

    def test_submission_mode_silently_ignored_fails(self):
        def change(method, call):
            if method == 'merged':
                call['sparse']['metal_submission_mode'] = 0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_missing_submission_metric_fails(self):
        def change(method, call):
            if method == 'split':
                del call['sparse']['metal_submission_mode']
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_invalid_lookup_bytes_fails(self):
        for target in ('merged', 'split'):
            def change(method, call):
                if method == target:
                    call['sparse']['gradient_lookup_bytes'] -= 1
            with self.subTest(method=target):
                self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_gpu_allocation_missing_fails(self):
        def change(method, call):
            call['sparse']['gpu_buffer_bytes'] = 0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_wait_not_removed_fails(self):
        def change(method, call):
            if method == 'merged':
                call['sparse']['cpu_waits'] += 1
        self.assertFalse(self.evaluate(change)['synchronization_all'])

    def test_dispatch_or_retry_change_fails(self):
        for key in ('dispatches', 'peak_overflow_retries'):
            def change(method, call):
                if method == 'merged':
                    call['sparse'][key] += 1
            with self.subTest(key=key):
                self.assertFalse(self.evaluate(change)['synchronization_all'])

    def test_submission_counter_missing_fails(self):
        def change(method, call):
            if method == 'merged':
                del call['sparse']['command_buffers']
        result = self.evaluate(change)
        self.assertFalse(result['route_integrity_all'])
        self.assertFalse(result['synchronization_all'])

    def test_split_baseline_counter_drift_fails(self):
        def change(method, call):
            if method == 'frozen_metal':
                call['sparse']['command_buffers'] += 1
        self.assertFalse(self.evaluate(change)['synchronization_all'])

    def test_fractional_counter_fails(self):
        def change(method, call):
            if method == 'merged':
                call['sparse']['dispatches'] = 5.5
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_cpu_fallback_fails(self):
        def change(method, call):
            call['sparse']['search_backend'] = 0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_metal_refinement_substitution_fails(self):
        def change(method, call):
            if method == 'split':
                call['selected_refinement_backend'] = 1
                call['pose_gpu_bytes'] = 100
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_final_difference_fails_all_current_routes(self):
        for target in ('split', 'merged'):
            def change(method, call):
                if method == target:
                    call['matches'][0]['score'] -= .001
            with self.subTest(method=target):
                self.assertFalse(self.evaluate(change)['exact_cpu_final_all'])

    def test_independent_gt_detects_common_failure(self):
        def change(method, call):
            call['matches'][0]['pos_x'] += 10
        summary = self.evaluate(change)
        self.assertTrue(summary['exact_cpu_final_all'])
        self.assertFalse(summary['normal_gt_all'])

    def test_missing_worker_fails(self):
        summary = self.evaluate(omit='merged')
        self.assertFalse(summary['synchronization_all'])
        self.assertFalse(summary['all_workers_succeeded'])
        self.assertFalse(summary['exact_proposals_all'])
        self.assertFalse(summary['exact_cpu_final_all'])

    def test_changed_worker_artifact_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'worker.json'
            path.write_text('{}')
            report = dict(execution_order=[dict(result=path.name, result_sha256=b.sha(path))])
            b.verify_workers(report, root)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Changed worker artifact'):
                b.verify_workers(report, root)


if __name__ == '__main__':
    unittest.main()
