"""No GPU calls: verify coarse experiment routing and frozen provenance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_coarse_next as bench


class CoarseNextTest(unittest.TestCase):
    def test_frozen_route_does_not_call_unavailable_setter(self):
        class OldFrame:
            pass
        calls = []
        def init(adapter, build, fine, patterns, fine_batch_size):
            calls.append((build, fine, patterns, fine_batch_size))
            adapter.frame, adapter.config = OldFrame(), {}
        with patch.object(bench.shared.FrameAdapter, '__init__', init):
            adapter = bench.FrameAdapter(dict(method='previous_shared', baseline='frozen', build='new'), ['a'])
        self.assertEqual(calls, [('frozen', True, ['a'], 512)])
        self.assertTrue(adapter.config['coarse_optimization']['frozen_api'])

    def test_ablations_set_only_intended_modes(self):
        calls = []
        class NewFrame:
            def set_coarse_optimization(self, **kwargs):
                calls.append(kwargs)
                self.coarse_optimization = kwargs
        def init(adapter, build, fine, patterns, fine_batch_size):
            self.assertEqual(build, 'new')
            adapter.frame, adapter.config = NewFrame(), {}
        with patch.object(bench.shared.FrameAdapter, '__init__', init):
            for name, modes in list(bench.ROUTES.items())[1:]:
                for suffix in ('', '_profile'):
                    bench.FrameAdapter(dict(method=name+suffix, baseline='frozen', build='new'), [])
                    self.assertEqual(calls[-1], dict(peak_mode=modes[0], ncc_mode=modes[1]))

    def test_frozen_hashes_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ('libfpm_c.dylib', 'experiments/libfpm_coarse_experiment.dylib',
                         'experiments/libfpm_pose_experiment.dylib'):
                path = root/name; path.parent.mkdir(exist_ok=True)
                path.write_bytes(name.encode())
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'baseline-provenance.json').write_text(json.dumps(dict(commit=bench.BASELINE_COMMIT, files=files)))
            self.assertTrue(bench.verify_baseline(root)['verified'])
            (root/'libfpm_c.dylib').write_bytes(b'tampered')
            with self.assertRaises(ValueError): bench.verify_baseline(root)

    def test_work_counts_preserve_scale_order(self):
        record = dict(patterns=[dict(scale=.75, metrics=dict(score_maps=10), fine_metrics=dict(maps=30)),
                                dict(scale=1., metrics=dict(score_maps=20), fine_metrics=dict(maps=60))])
        counts = bench.work_counts(record)
        self.assertEqual(counts, [dict(scale=.75, coarse_maps=10, fine_maps=30),
                                  dict(scale=1., coarse_maps=20, fine_maps=60)])
        record['patterns'].reverse()
        self.assertNotEqual(bench.work_counts(record), counts)


if __name__ == '__main__':
    unittest.main()
