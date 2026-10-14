from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_native_pipeline as b
class NativeHarnessTest(unittest.TestCase):
    def test_gt_and_metadata_are_not_sent_to_native(self):
        c=dict(source='s',template='t',scales=[.75,1],angle=180,limit=5,score=.8,overlap=.2,stop_layer=1,subpixel=0,workflow='full_rescue',truth=[dict(pos_x=99)],metadata='private',name='fixture')
        j=b.native_job(c,'native_combined',5,1.)
        self.assertNotIn('truth',j);self.assertNotIn('metadata',j)
        self.assertEqual(j['ncc_mode'],2);self.assertTrue(j['overlap_enabled']);self.assertTrue(j['refine'])
    def test_parity_does_not_sort_or_ignore_geometry(self):
        m=dict(pos_x=1.,pos_y=2.,angle=3.,score=.8,scale=1.,corners=[[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
        n=dict(m,pos_x=8.)
        self.assertFalse(b.outputs_equal([m,n],[n,m]));self.assertFalse(b.outputs_equal([m],[]))
        self.assertFalse(b.outputs_equal([m],[dict(m,scale=1.001)]))
        self.assertTrue(b.outputs_equal([m],[dict(m,pos_x=1.+1e-7)]))
    def test_modes_are_independent(self):
        self.assertEqual(b.METHODS['native_original'],(0,False));self.assertEqual(b.METHODS['native_tile'],(2,False))
        self.assertEqual(b.METHODS['native_overlap'],(0,True));self.assertEqual(b.METHODS['native_combined'],(2,True))
if __name__=='__main__':unittest.main()
