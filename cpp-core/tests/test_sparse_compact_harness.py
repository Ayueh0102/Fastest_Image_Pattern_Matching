"""Audit-contract tests; no matcher or GPU is launched."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_sparse_compact as b


class CompactHarnessTest(unittest.TestCase):
    def fixture(self):
        pose=dict(pos_x=20.,pos_y=20.,angle=0.,score=.95,scale=1.,corners=[[10,10],[30,10],[30,30],[10,30]])
        case=dict(name='x',backends=[0],scene_shape=[40,40],validity='fixed_scale_gt',truth=[pose])
        protocol=dict(cases=[case],blocks=1,repeats=1,proposal_basin=dict(position_px=6.,angle_deg=6.,relative_scale=.02))
        return pose,protocol

    def report(self,base,change=None):
        pose,protocol=self.fixture();report=dict(protocol=protocol,execution_order=[])
        for method in b.METHODS:
            s=dict(enabled=True,source_levels=1,source_bytes=1600*(1 if method=='compact' else 8),lookup_bytes=2048,response_mode=int(method=='compact'),
                   **{key:1 for key in b.COUNTS})
            record=dict(matches=[copy.deepcopy(pose)],proposals=[copy.deepcopy(pose)],sparse=s,frame_host_bytes=0,selected_refinement_backend=0,
                        total_ms=1.,prepare_ms=.1,proposal_ms=.4,refine_ms=.5,nms_ms=.1)
            if change:change(method,record)
            raw=dict(first=record,runs=[copy.deepcopy(record)],exact_repeat=True,construct_ms=.2,final_memory=dict(peak_rss_bytes=10000,current_rss_bytes=9000))
            path=base/(method+'.json');path.write_text(json.dumps(raw))
            report['execution_order'].append(dict(case='x',backend=0,method=method,block=0,status='ok',result=path.name))
        return report

    def test_complete_parity_and_memory_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);summary=b.summarize(self.report(p),p)
            self.assertTrue(summary['exact_frozen_all']);self.assertTrue(summary['route_integrity_all'])
            self.assertTrue(summary['fixed_scale_gt_all']);self.assertEqual(summary['measured_calls'],3)

    def test_candidate_score_change_fails_exactness(self):
        def mutate(method,record):
            if method=='compact':record['proposals'][0]['score']-=.001
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);summary=b.summarize(self.report(p,mutate),p)
            self.assertFalse(summary['exact_frozen_all']);self.assertTrue(summary['fixed_scale_gt_all'])

    def test_wrong_response_capacity_fails_route(self):
        def mutate(method,record):
            if method=='compact':record['sparse']['source_bytes']*=8
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);summary=b.summarize(self.report(p,mutate),p)
            self.assertFalse(summary['route_integrity_all'])

    def test_missing_worker_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);report=self.report(p);report['execution_order'].pop()
            summary=b.summarize(report,p)
            self.assertFalse(summary['all_workers_succeeded']);self.assertFalse(summary['exact_frozen_all'])

    def test_changed_raw_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'raw.json';f.write_text('{}')
            report=dict(execution_order=[dict(result='raw.json',result_sha256=b.sha(f))]);f.write_text('{"changed":true}')
            with self.assertRaises(ValueError):b.verify_workers(report,p)

if __name__=='__main__':unittest.main()
