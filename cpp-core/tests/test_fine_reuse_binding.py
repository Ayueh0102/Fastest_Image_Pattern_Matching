"""Exercise the additive Python adapter against a built research library.

FPM_FINE_REUSE_BUILD selects the build; default is build/fine-reuse.
"""
import ctypes as C
import os
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from coarse_experiment import SourceFrame, CoarsePattern, FineReuseMetricsV1, Image, Config

BUILD = Path(os.environ.get('FPM_FINE_REUSE_BUILD', Path(__file__).resolve().parents[2]/'build/fine-reuse'))


class FineReuseBindingTest(unittest.TestCase):
    def test_cpu_gate_and_closed_frame(self):
        self.assertEqual(C.sizeof(FineReuseMetricsV1), 48)
        with SourceFrame(BUILD, backend='cpu') as frame:
            frame.set_fine_ncc_mode(0)
            self.assertEqual(frame.fine_reuse_metrics()['ncc_mode'], 0)
            for mode in (1, 2, 3):
                with self.assertRaises(RuntimeError):
                    frame.set_fine_ncc_mode(mode)
                self.assertEqual(frame.fine_reuse_metrics()['ncc_mode'], 0)
            for mode in (-1, 4, .0, True, '1'):
                with self.assertRaises(ValueError):
                    frame.set_fine_ncc_mode(mode)
        with self.assertRaises(ValueError):
            frame.fine_reuse_metrics()
        with self.assertRaises(ValueError):
            frame.set_fine_ncc_mode(0)

    def test_metal_matching_and_mode_switch(self):
        rng = random.Random(19)
        # Padded input rows also exercise snapshot copying through the adapter.
        sw, sh, stride, tw, th = 192, 144, 197, 40, 32
        source = (C.c_uint8*(stride*sh))(*[rng.randrange(256) for _ in range(stride*sh)])
        template = (C.c_uint8*(tw*th))(*[source[(y+53)*stride+x+72] for y in range(th) for x in range(tw)])
        with CoarsePattern(BUILD, Image(template,tw,th,tw), min_area=64) as pattern, SourceFrame(BUILD, fine_batch_size=7) as frame:
            frame.prepare(Image(source,sw,sh,stride), [pattern])
            config = Config(8,.8,.2,20,1,0,0,0,None,0)
            before = frame.match(pattern, config)
            self.assertGreater(len(before), 0)
            original = frame.fine_reuse_metrics()
            self.assertGreater(original['statistics_dispatches'], 0)
            for mode in (2, 3):
                frame.set_fine_ncc_mode(1)
                self.assertEqual(before, frame.match(pattern, config))
                metrics = frame.fine_reuse_metrics()
                self.assertEqual(metrics['ncc_mode'], 1)
                self.assertGreater(metrics['row_buffer_bytes'], 0)
                self.assertEqual(metrics['row_dispatches'], metrics['statistics_dispatches'])
                self.assertGreater(metrics['row_dispatches'], 0)
                frame.set_fine_ncc_mode(mode)
                self.assertEqual(frame.fine_reuse_metrics()['row_buffer_bytes'], 0)
                self.assertEqual(before, frame.match(pattern, config))
                metrics = frame.fine_reuse_metrics()
                self.assertEqual(metrics['ncc_mode'], mode)
                self.assertEqual(metrics['row_buffer_bytes'], 0)
                self.assertEqual(metrics['row_dispatches'], 0)
                self.assertEqual(metrics['statistics_dispatches'], original['statistics_dispatches'])
                with self.assertRaises(ValueError):
                    frame.set_fine_ncc_mode(4)
                self.assertEqual(frame.fine_reuse_metrics()['ncc_mode'], mode)
            frame.set_fine_ncc_mode(0)
            self.assertEqual(frame.fine_reuse_metrics()['row_buffer_bytes'], 0)
            self.assertEqual(frame.match(pattern, config), before)


if __name__ == '__main__':
    unittest.main()
