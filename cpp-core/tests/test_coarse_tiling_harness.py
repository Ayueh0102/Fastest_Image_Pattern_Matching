"""CPU-only tests of tiling benchmark isolation and mode selection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_coarse_tiling as bench


class TilingHarnessTest(unittest.TestCase):
    def test_all_routes_are_synchronous_and_use_explicit_injection(self):
        for method in bench.METHODS:
            calls = []
            def init(search, job, **dependencies): calls.append((job, dependencies))
            with patch.object(bench.coarse.Search, '__init__', init):
                bench.Search(dict(method=method))
            job, dependencies = calls[0]
            base = method.removesuffix('_profile')
            self.assertEqual(job['_ncc_mode'], bench.ROUTES[base])
            self.assertIs(job['_overlap'], False)
            self.assertEqual(job['_frozen'], base == 'previous_peak')
            self.assertIs(dependencies['frame_adapter'], bench.FrameAdapter)
            self.assertIs(dependencies['pattern_adapter'], bench.overlap.PatternAdapter)
            self.assertEqual(job['method'], 'previous_shared' if base == 'previous_peak' else
                             'peak_only'+('_profile' if method.endswith('_profile') else ''))

    def test_adapter_applies_exact_requested_modes(self):
        for mode in (0, 2, 3):
            calls = []
            class Frame:
                def set_coarse_optimization(self, **values):
                    calls.append(values); self.coarse_optimization = values
            def init(adapter, job, patterns): adapter.frame = Frame(); adapter.config = {}
            with patch.object(bench.overlap.FrameAdapter, '__init__', init):
                adapter = bench.FrameAdapter(dict(_ncc_mode=mode, _frozen=False), [])
            self.assertEqual(calls, [dict(peak_mode=1, ncc_mode=mode)])
            self.assertEqual(adapter.config['coarse_optimization']['ncc_mode'], mode)

    def test_worker_rejects_truth_before_constructing_search(self):
        with patch.object(bench.coarse, 'worker') as worker:
            with self.assertRaises(ValueError):
                bench.worker(dict(case=dict(name='fixture', truth=[])))
            worker.assert_not_called()


if __name__ == '__main__': unittest.main()
