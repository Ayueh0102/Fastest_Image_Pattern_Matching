"""Negative controls for the GPU sparse frontend evidence checker."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_sparse_metal_front as b

class FrontHarnessTest(unittest.TestCase):
    def evaluate(self, mutate=None, omit=None):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pose=dict(pos_x=20.,pos_y=20.,angle=0.,score=.95,scale=1.,corners=[[10,10],[30,10],[30,30],[10,30]])
            case=dict(name='x',validity='fixed_scale_gt',truth=[pose])
            report=dict(protocol=dict(cases=[case],blocks=1,repeats=1,proposal_basin=dict(position_px=6.,angle_deg=6.,relative_scale=.02)),execution_order=[])
            for method in b.METHODS:
                if method==omit:continue
                metal=method.startswith('metal_');refine=method=='metal_metal'
                sparse=dict(enabled=True,source_levels=1,response_mode=1,search_backend=int(metal),gpu_buffer_bytes=1000*metal,command_buffers=int(metal),**{k:1 for k in b.COUNTS})
                r=dict(matches=[copy.deepcopy(pose)],proposals=[copy.deepcopy(pose)],sparse=sparse,
                    frame_host_bytes=0,selected_refinement_backend=int(refine),pose_gpu_bytes=1000*refine,
                    total_ms=1.,prepare_ms=.1,proposal_ms=.4,refine_ms=.5,nms_ms=.1)
                if mutate:mutate(method,r)
                raw=dict(first=r,runs=[copy.deepcopy(r)],exact_repeat=True,construct_ms=.2,final_memory=dict(peak_rss_bytes=10000))
                p=root/(method+'.json');p.write_text(json.dumps(raw))
                report['execution_order'].append(dict(case='x',method=method,block=0,status='ok',result=p.name))
            return b.summarize(report,root)
    def test_exact_baseline(self):
        s=self.evaluate()
        self.assertTrue(all(s[k] for k in ('all_workers_succeeded','exact_proposals_all','exact_cpu_final_all','route_integrity_all','normal_gt_all')))
    def test_changed_candidate(self):
        def f(m,r):
            if m=='metal_cpu':r['proposals'][0]['score']-=.001
        self.assertFalse(self.evaluate(f)['exact_proposals_all'])
    def test_changed_work(self):
        def f(m,r):
            if m=='metal_cpu':r['sparse']['feature_evaluations']+=1
        self.assertFalse(self.evaluate(f)['exact_proposals_all'])
    def test_cpu_fallback_fails(self):
        def f(m,r):r['sparse']['search_backend']=0
        self.assertFalse(self.evaluate(f)['route_integrity_all'])
    def test_final_cpu_difference_fails(self):
        def f(m,r):
            if m=='metal_cpu':r['matches'][0]['score']-=.001
        self.assertFalse(self.evaluate(f)['exact_cpu_final_all'])
    def test_metal_float_allowed_but_gt_independent(self):
        def f(m,r):
            if m=='metal_metal':r['matches'][0]['pos_x']+=.001
        self.assertTrue(self.evaluate(f)['normal_gt_all'])
        def bad(m,r):
            if m=='metal_metal':r['matches'][0]['pos_x']+=10
        self.assertFalse(self.evaluate(bad)['normal_gt_all'])
    def test_missing_worker_fails(self):
        s=self.evaluate(omit='cpu')
        self.assertFalse(s['all_workers_succeeded']);self.assertFalse(s['exact_proposals_all'])

if __name__=='__main__':unittest.main()
