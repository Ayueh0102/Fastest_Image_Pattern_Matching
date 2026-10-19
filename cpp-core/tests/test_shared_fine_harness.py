"""Pure-data/fake-backend harness tests; no image matching or GPU dispatch."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_shared_fine as bench


class SharedFineHarnessTest(unittest.TestCase):
    def test_prepare_repeats_inside_pipeline_before_every_scale(self):
        clock, events = SimpleNamespace(value=0.), []

        class FakeFrame:
            def prepare(self, source, invert=False):
                events.append(('prepare', source.data[0], invert))
                clock.value += .010

        class FakePattern:
            def __init__(self, name):
                self.name = name

            def match(self, source, config, frame):
                events.append(('match', self.name))
                clock.value += .003
                return [dict(index=0, score=.99, pos_x=8., pos_y=8., angle=0.)]

        search = object.__new__(bench.Search)
        search.source = np.zeros((16, 16), np.uint8)
        search.shape = (2, 2)
        search.frame, search.prepare_calls = FakeFrame(), 0
        search.patterns = [(1., FakePattern('a')), (1.25, FakePattern('b'))]
        search.config = SimpleNamespace(invert_source=0)
        search.hybrid = False
        search.job = dict(case=dict(limit=4, overlap=.2, score=.7))
        with patch.object(bench.time, 'perf_counter', side_effect=lambda: clock.value):
            first = search.run()
            search.source[0, 0] = 91  # same allocation, new frame contents
            second = search.run()
        self.assertEqual(events, [('prepare', 0, False), ('match', 'a'), ('match', 'b'),
                                  ('prepare', 91, False), ('match', 'a'), ('match', 'b')])
        self.assertEqual(search.prepare_calls, 2)
        self.assertAlmostEqual(search.last_prepare_wall_ms, 10.)
        self.assertAlmostEqual(first['algorithm_ms'], 16.)
        self.assertAlmostEqual(second['algorithm_ms'], 16.)
        self.assertEqual([p['scale'] for p in search.last_proposals], [1., 1.25])

    def test_each_scale_retains_its_own_returned_snapshot(self):
        class FakeNativeFrame:
            def __init__(self):
                self.counter = 0

            def match(self, pattern, config, profile=False):
                self.counter += 1
                self.last_metrics = dict(score_maps=self.counter)
                self.last_profile = dict(version=1, coarse_maps=self.counter) if profile else None
                self.last_execution_metrics = dict(batch_maps=self.counter)
                self.last_fine_metrics = dict(maps=self.counter)
                return []

        frame = object.__new__(bench.FrameAdapter)
        frame.frame = FakeNativeFrame()
        adapters = []
        for _ in range(2):
            adapter = object.__new__(bench.PatternAdapter)
            adapter.method, adapter.pattern = 'shared_metal_fine_profile', object()
            adapter.frame_snapshot = None
            adapters.append(adapter)
        for adapter in adapters:
            adapter.match(None, None, frame)
        first, second = [adapter.snapshot(1.) for adapter in adapters]
        self.assertEqual(first['metrics']['score_maps'], 1)
        self.assertEqual(second['metrics']['score_maps'], 2)
        self.assertEqual(first['fine_metrics']['maps'], 1)
        self.assertEqual(second['fine_metrics']['maps'], 2)

    def test_shared_profile_does_not_require_per_scale_pyramid_commands(self):
        profile = {key: 0. for key in bench.previous.EXCLUSIVE_STAGES}
        profile.update(version=1, core_wall_ms=0., api_wall_ms=0., api_residual_ms=0.,
                       backend=1, coarse_maps=2, pyramid_levels=4)
        record = dict(profile=profile, metrics=dict(score_maps=2),
            execution_config=dict(pyramid_mode=1, batch_angles=4),
            execution_metrics=dict(pyramid_mode=1, batch_angles=4, max_batch_maps=2,
                                   pyramid_command_buffers=0))
        self.assertEqual(bench.profile_checks([record], shared=True), [])
        self.assertIn('GPU pyramid requested but no command executed',
                      bench.profile_checks([record], shared=False))
        corrupted = copy.deepcopy(record)
        corrupted['profile']['coarse_maps'] = 3
        self.assertIn('Metal score map count differs from core count',
                      bench.profile_checks([corrupted], shared=True))
        corrupted = copy.deepcopy(record)
        corrupted['execution_metrics']['max_batch_maps'] = 5
        self.assertIn('shared angle batch exceeds configured cap',
                      bench.profile_checks([corrupted], shared=True))
        corrupted['execution_metrics']['pyramid_command_buffers'] = 1
        self.assertIn('shared match rebuilt its source pyramid',
                      bench.profile_checks([corrupted], shared=True))

    def test_fine_batch_size_propagates_only_to_gpu_route(self):
        with patch.object(bench.coarse, 'SourceFrame', return_value=object()) as constructor:
            gpu = bench.FrameAdapter('unused', True, [], fine_batch_size=512)
            self.assertEqual(gpu.config['fine_batch_size'], 512)
            self.assertEqual(constructor.call_args.kwargs['fine_batch_size'], 512)
            cpu = bench.FrameAdapter('unused', False, [], fine_batch_size=512)
            self.assertEqual(cpu.config['fine_batch_size'], 0)
            self.assertEqual(constructor.call_args.kwargs['fine_batch_size'], 0)

    def test_fine_diagnostics_reject_wrong_route_cap_and_counts(self):
        frame = dict(config=dict(fine_batch_size=128))
        item = dict(profile=dict(fine_maps=100), fine_metrics=dict(version=1, maps=100,
            peak_batch_maps=100, repair_pixels=3, roi_pixels=200, command_buffers=2))
        self.assertEqual(bench.fine_checks([item], frame), [])
        invalid = copy.deepcopy(item)
        invalid['fine_metrics'].update(peak_batch_maps=129, maps=101, repair_pixels=201)
        errors = bench.fine_checks([invalid], frame)
        self.assertIn('fine map batch exceeds configured cap', errors)
        self.assertIn('fine repaired pixels exceed sampled pixels', errors)
        self.assertIn('fine GPU map count differs from core count', errors)
        self.assertIn('CPU fine route unexpectedly dispatched fine GPU work',
                      bench.fine_checks([item], dict(config=dict(fine_batch_size=0))))

    def test_new_frozen_baseline_is_strict_and_symlink_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = [root/'core.so', root/'coarse.so', root/'pose.so']
            for i, path in enumerate(libraries):
                path.write_bytes(bytes([i]))
            manifest = dict(commit=bench.BASELINE_COMMIT,
                            files={p.name: bench.file_hash(p) for p in libraries})
            path = root/'baseline-provenance.json'
            path.write_text(json.dumps(manifest))
            with patch.object(bench, 'core_library_path', return_value=libraries[0]), \
                 patch.object(bench.coarse, 'library_path', return_value=libraries[1]), \
                 patch.object(bench, 'pose_library_path', return_value=libraries[2]):
                self.assertTrue(bench.verify_baseline(root)['verified'])
                libraries[2].write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'differs from manifest'):
                    bench.verify_baseline(root)
                manifest['commit'] = '91842dd'
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, '2c28ffb'):
                    bench.verify_baseline(root)


if __name__ == '__main__':
    unittest.main()
