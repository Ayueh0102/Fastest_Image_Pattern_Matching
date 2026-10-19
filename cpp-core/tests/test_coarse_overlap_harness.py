"""No GPU: verify frozen/synchronous/overlap route setup and snapshot ownership."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_coarse_overlap as bench

class OverlapHarnessTest(unittest.TestCase):
    def test_routes_do_not_call_new_api_on_frozen_build(self):
        for name,mode in [('previous_peak',None),('current_peak',False),('overlap',True),
                          ('current_peak_profile',False),('overlap_profile',True)]:
            captured=[]
            def init(search,job,**dependencies):captured.append((job,dependencies))
            with patch.object(bench.coarse.Search,'__init__',init):bench.Search(dict(method=name))
            job,deps=captured[0]
            self.assertEqual(job['_overlap'],mode)
            self.assertEqual(job['method'], 'previous_shared' if mode is None else 'peak_only'+('_profile' if name.endswith('_profile') else ''))
            self.assertIs(deps['frame_adapter'],bench.FrameAdapter)
            self.assertIs(deps['pattern_adapter'],bench.PatternAdapter)
    def test_frame_sets_peak_tree_on_frozen_and_explicit_toggle_on_current(self):
        for toggle in (None,False,True):
            calls=[]
            class Frame:
                def set_coarse_optimization(self,**values):calls.append(('peak',values))
                def set_coarse_overlap(self,**values):calls.append(('overlap',values))
            def init(adapter,job,patterns):adapter.frame=Frame();adapter.config={}
            with patch.object(bench.coarse.FrameAdapter,'__init__',init):
                a=bench.FrameAdapter(dict(_overlap=toggle),[])
            self.assertEqual(calls,[('peak',dict(peak_mode=1,ncc_mode=0))] if toggle is None else [('overlap',dict(enabled=toggle))])
            self.assertEqual(a.config['coarse_overlap'],toggle)

if __name__=='__main__':unittest.main()
