"""Stage29 additive options/metrics ABI and exact full search integration."""
import ctypes as C
import os
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from coarse_experiment import SourceFrame, CoarsePattern, FineSearchMetricsV1, Image, Config
BUILD = Path(os.environ.get('FPM_FINE_SEARCH_BUILD', Path(__file__).resolve().parents[2]/'build/fine-search'))

class FineSearchBindingTest(unittest.TestCase):
    def test_cpu_abi_invalid_and_closed(self):
        self.assertEqual(C.sizeof(FineSearchMetricsV1),232)
        with SourceFrame(BUILD, backend='cpu') as f:
            f.set_fine_search_options()
            self.assertEqual(f.fine_search_metrics()['profiling_enabled'],0)
            for options in ({'profiling':True},{'angle_reuse':True}):
                with self.assertRaises(RuntimeError): f.set_fine_search_options(**options)
            for v in (0,1,None,'yes'):
                with self.assertRaises(ValueError): f.set_fine_search_options(profiling=v)
                with self.assertRaises(ValueError): f.set_fine_search_options(angle_reuse=v)
            setter=f.dll.fpm_frame_set_fine_search_options
            for p,r in ((-1,0),(2,0),(0,-1),(0,2)):
                self.assertNotEqual(setter(f.handle,p,r,f.error,len(f.error)),0)
            self.assertNotEqual(setter(None,0,0,f.error,len(f.error)),0)
            getter=f.dll.fpm_frame_get_fine_search_metrics
            m=FineSearchMetricsV1()
            for handle,ptr,size in ((f.handle,C.byref(m),C.sizeof(m)-1),(None,C.byref(m),C.sizeof(m)),(f.handle,None,C.sizeof(m))):
                self.assertNotEqual(getter(handle,ptr,size,f.error,len(f.error)),0)
        with self.assertRaises(ValueError): f.set_fine_search_options()
        with self.assertRaises(ValueError): f.fine_search_metrics()

    def test_profile_reuse_reset_and_changed_source(self):
        rng=random.Random(61); sw,sh,stride,tw,th=192,144,197,40,32
        source=(C.c_uint8*(stride*sh))(*[rng.randrange(256) for _ in range(stride*sh)])
        template=(C.c_uint8*(tw*th))(*[source[(y+53)*stride+x+72] for y in range(th) for x in range(tw)])
        image=Image(source,sw,sh,stride)
        with CoarsePattern(BUILD,Image(template,tw,th,tw),min_area=64) as pattern, SourceFrame(BUILD,fine_batch_size=7) as f:
            f.set_fine_ncc_mode(2); f.prepare(image,[pattern])
            config=Config(8,.8,.2,20,1,0,0,0,None,0)
            expected=f.match(pattern,config); self.assertTrue(expected)
            original_maps=f.last_fine_metrics['maps']
            for profiling,reuse in ((True,False),(False,True),(True,True)):
                f.set_fine_search_options(profiling=profiling,angle_reuse=reuse)
                for _ in range(2):
                    self.assertEqual(f.match(pattern,config),expected)
                    d=f.fine_search_metrics();m=f.last_fine_metrics
                    self.assertEqual(d['profiling_enabled'],profiling);self.assertEqual(d['angle_reuse_enabled'],reuse)
                    if reuse:
                        self.assertEqual(d['logical_requests'],original_maps)
                        self.assertGreater(d['reused_maps'],0)
                        self.assertEqual(d['reused_maps']+m['maps'],d['logical_requests'])
                        self.assertGreater(d['cache_peak_bytes'],0)
                    if profiling:
                        self.assertEqual(d['requests'],m['maps'])
                        self.assertEqual(d['requests'],sum(d[k] for k in ('within_call_duplicates','cross_call_duplicates','tracked_unique_keys','untracked_call_keys')))
                        self.assertLessEqual(d['tracked_unique_keys'],65536)
                        self.assertEqual(d['warp_batches'],f.fine_reuse_metrics()['statistics_dispatches'])
                        self.assertAlmostEqual(d['warp_wait_ms']+d['ncc_wait_ms'],m['wait_ms'],places=7)
                        self.assertLessEqual(d['zero_repair_batches'],d['warp_batches'])
                    else:self.assertEqual(d['requests'],0)
            f.set_fine_search_options();self.assertEqual(f.fine_search_metrics()['history_key_payload_bytes'],0)
            self.assertEqual(f.match(pattern,config),expected)
            # Same pointer with changed bytes must not reuse old pixels/angles.
            for i in range(len(source)): source[i]=0
            f.prepare(image,[pattern]);f.set_fine_search_options(profiling=True,angle_reuse=True)
            changed=f.match(pattern,config)
            f.set_fine_search_options();self.assertEqual(f.match(pattern,config),changed)
            self.assertNotEqual(changed,expected)

if __name__=='__main__':unittest.main()
