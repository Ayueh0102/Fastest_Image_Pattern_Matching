"""Negative controls: speed alone must never pass the resident experiment."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_coarse_resident as bench


class Harness(unittest.TestCase):
    def fixture(self, root, change=None):
        pose=dict(pos_x=50.,pos_y=50.,angle=0.,scale=1.,score=1.)
        case=dict(name='one',truth=[pose])
        manifest=dict(cases=[case],blocks=1,executions=[],complete=True)
        for method in bench.METHODS:
            row=dict(total_ms=10.,prepare_ms=1.,proposal_ms=7.,refine_ms=1.,nms_ms=1.,
                     matches=[pose],proposals=[pose],frame_coarse_gpu_bytes=100,frame_fine_gpu_bytes=100,
                     work=[dict(device_ms=3.,wait_ms=4.,copy_ms=.1,normalization_ms=0.,coarse_commands=1,
                                fine_maps=1,maps=1,windows=100,coarse_host_scoremap_peak_bytes=0)])
            if method=='metal_resident_peaks':
                row=copy.deepcopy(row);row['total_ms']=5.
                if change: change(row)
            raw=dict(first=row,runs=[row],exact_repeat=True,construct_ms=1.,final_memory=dict(peak_rss_bytes=1000000))
            jp=root/(method+'.job.json');rp=root/(method+'.result.json')
            bench.save(jp,dict(coarse_peak_mode=bench.METHODS[method]));bench.save(rp,raw)
            manifest['executions'].append(dict(case='one',method=method,block=0,job=jp.name,
                result=rp.name,job_sha256=bench.sha(jp),result_sha256=bench.sha(rp)))
        return manifest

    def test_positive_and_reject_accuracy_work_and_cpu_fallback(self):
        for change, expected in [
            (None,True),
            (lambda r:r['matches'][0].update(angle=10.),False),
            (lambda r:r['work'][0].update(windows=99),False),
            (lambda r:r['work'][0].update(normalization_ms=.1),False),
            (lambda r:r['work'][0].update(coarse_host_scoremap_peak_bytes=8),False),
            (lambda r:r.update(total_ms=20.),False),
        ]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root=Path(d);m=self.fixture(root,change)
                self.assertEqual(bench.summarize(m,root)['cases'][0]['candidate_for_adoption'],expected)

    def test_missing_duplicate_and_modified_workers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=self.fixture(root)
            for invalid in (dict(m,complete=False),dict(m,blocks=2),dict(m,executions=m['executions']*2)):
                with self.assertRaises(ValueError):bench.summarize(invalid,root)
            (root/m['executions'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):bench.summarize(m,root)

    def test_truth_never_sent_to_worker(self):
        case=dict(name='one',truth=[dict(secret=1)],source='s',template='t',scales=[1.],
                  angle=180,limit=8,score=.8,overlap=.2,stop_layer=0,subpixel=1,workflow='full_rescue')
        j=bench.job(case,2,4,1)
        self.assertNotIn('truth',j)
        self.assertNotIn('name',j)
        self.assertEqual(j['coarse_peak_mode'],2)


if __name__=='__main__':unittest.main()
