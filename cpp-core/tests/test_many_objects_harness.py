"""Ensure truth and scene-generation hints never cross the timed worker boundary."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_many_objects import worker_case

class ManyObjectsHarnessTest(unittest.TestCase):
    def test_truth_and_rendering_hints_removed(self):
        case=dict(name='many128_pose',source='input.png',template='template.png',scales=[.75,1,1.25],
                  angle=180,limit=384,proposal_score=.5,score=.8,overlap=.2,stop_layer=1,subpixel=0,
                  workflow='full_rescue',truth=[dict(pos_x=100,pos_y=200)],metadata='truth.json',
                  object_count=128,scene_shape=[4096,4096],seed=11)
        result=worker_case(case)
        for key in ('truth','metadata','object_count','scene_shape','seed'):self.assertNotIn(key,result)
        self.assertEqual(result['limit'],384)
        self.assertEqual(result['scales'],case['scales'])
        self.assertEqual(result['score'],.8)
        self.assertIn('truth',case)

if __name__=='__main__':unittest.main()
