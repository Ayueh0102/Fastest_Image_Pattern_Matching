"""CPU/Metal comparison must preserve proposals, not require identical final floats."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_sparse_backends as b

class BackendHarnessTest(unittest.TestCase):
    def report(self,root,mutate=None):
        pose=dict(pos_x=20.,pos_y=20.,angle=0.,score=.95,scale=1.,corners=[[10,10],[30,10],[30,30],[10,30]])
        case=dict(name='x',scene_shape=[40,40],validity='fixed_scale_gt',truth=[pose])
        protocol=dict(cases=[case],methods=['compact'],blocks=1,repeats=1,proposal_basin=dict(position_px=6.,angle_deg=6.,relative_scale=.02))
        report=dict(protocol=protocol,execution_order=[])
        for backend in (0,1):
            sparse=dict(enabled=True,source_levels=1,response_mode=1,source_bytes=1600,**{key:1 for key in b.COUNTS})
            record=dict(matches=[copy.deepcopy(pose)],proposals=[copy.deepcopy(pose)],sparse=sparse,
                frame_host_bytes=0,selected_refinement_backend=backend,pose_host_bytes=1000,pose_gpu_bytes=1000*backend,
                rescued_seeds=0,total_ms=1.,prepare_ms=.1,proposal_ms=.4,refine_ms=.5,nms_ms=.1)
            if mutate:mutate(backend,record)
            raw=dict(first=record,runs=[copy.deepcopy(record)],exact_repeat=True,construct_ms=.2,final_memory=dict(peak_rss_bytes=10000,current_rss_bytes=9000))
            path=root/f'{backend}.json';path.write_text(json.dumps(raw))
            report['execution_order'].append(dict(case='x',method='compact',backend=backend,block=0,status='ok',result=path.name))
        return report
    def evaluate(self,mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);return b.summarize(self.report(path,mutate),path)
    def test_final_rounding_does_not_fail_proposal_equality(self):
        def change(backend,r):
            if backend:r['matches'][0]['pos_x']+=.001
        s=self.evaluate(change);self.assertTrue(s['proposal_equality_all']);self.assertTrue(s['normal_gt_all'])
    def test_candidate_change_fails(self):
        def change(backend,r):
            if backend:r['proposals'][0]['score']-=.001
        self.assertFalse(self.evaluate(change)['proposal_equality_all'])
    def test_wrong_backend_fails(self):
        def change(backend,r):r['selected_refinement_backend']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])
    def test_cpu_gpu_capacity_fails(self):
        def change(backend,r):r['pose_gpu_bytes']=1000
        self.assertFalse(self.evaluate(change)['route_integrity_all'])
    def test_large_final_error_fails_gt(self):
        def change(backend,r):
            if backend:r['matches'][0]['pos_x']+=10
        s=self.evaluate(change);self.assertTrue(s['proposal_equality_all']);self.assertFalse(s['normal_gt_all'])

if __name__=='__main__':unittest.main()
