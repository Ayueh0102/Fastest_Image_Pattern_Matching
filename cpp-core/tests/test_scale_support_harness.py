"""Evidence replay negative controls; no native/GPU work."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_scale_support import make_job, summarize, save, sha


class ScaleSupportHarness(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        pose=dict(pos_x=20.,pos_y=30.,angle=0.,scale=1.,score=.95,corners=[[1.,2.]]*4)
        self.case=dict(name='synthetic',source='/image',template='/template',truth=[pose],
            scales=[.75,1.,1.25],angle=180,limit=8,proposal_score=.5,score=.8,overlap=.2,
            stop_layer=1,subpixel=0,workflow='full_rescue',split='heldout',group='small')
        p=dict(cases=[self.case]);save(self.base/'protocol.json',p)
        self.report=dict(protocol=p,protocol_sha256=sha(self.base/'protocol.json'),complete=True,unchanged=True,entries=[])
        for b in range(2):
            for m in (('baseline','support') if b==0 else ('support','baseline')):
                r=dict(matches=[copy.deepcopy(pose)],proposals=[copy.deepcopy(pose)],total_ms=10.,refine_ms=1.,generation=1,
                    models_matched=3,frame_host_bytes=4,frame_coarse_gpu_bytes=4,frame_fine_gpu_bytes=4,
                    work=[],fine_search=[],fine_fusion=[],scale_support_enabled=m=='support')
                r.update(dict.fromkeys(('scale_support_refine_ms','scale_support_eligible','scale_support_changed',
                    'scale_support_boundary','scale_support_evaluations','scale_support_samples'),0))
                raw=dict(first=r,runs=[copy.deepcopy(r)],warmups_ms=[],exact_repeat=True,final_memory=dict(peak_rss_bytes=1000))
                paths={k:self.base/f'{b}-{m}-{k}.json' for k in ('job','raw','stderr')}
                save(paths['job'],make_job(self.case,m));save(paths['raw'],raw);paths['stderr'].write_text('')
                e=dict(case='synthetic',block=b,method=m,returncode=0)
                for k,p in paths.items():e[k]=p.name;e[k+'_sha256']=sha(p)
                self.report['entries'].append(e)

    def tearDown(self):self.tmp.cleanup()

    def mutate(self,fn,method='support'):
        for e in self.report['entries']:
            if e['method']==method:
                p=self.base/e['raw'];r=json.loads(p.read_text());fn(r);save(p,r);e['raw_sha256']=sha(p)

    def test_valid(self):
        s=summarize(self.report,self.base)
        self.assertTrue(s['integrity_passed'] and s['all_support_accuracy_passed'])

    def test_common_scale_failure_is_not_success(self):
        for method in ('baseline','support'):
            self.mutate(lambda r:[x['matches'][0].update(scale=.97) for x in [r['first'],*r['runs']]],method)
        s=summarize(self.report,self.base)
        self.assertTrue(s['integrity_passed']);self.assertFalse(s['all_support_accuracy_passed'])

    def test_missing_workers(self):
        self.report['entries'].pop()
        with self.assertRaises(ValueError):summarize(self.report,self.base)

    def test_raw_tamper(self):
        (self.base/self.report['entries'][0]['raw']).write_text('{}')
        with self.assertRaises(ValueError):summarize(self.report,self.base)

    def test_semantic_job_tamper(self):
        e=self.report['entries'][0];p=self.base/e['job'];j=json.loads(p.read_text());j['truth']=[1];save(p,j);e['job_sha256']=sha(p)
        with self.assertRaises(ValueError):summarize(self.report,self.base)

    def test_frontend_mismatch(self):
        self.mutate(lambda r:r['first']['proposals'].clear())
        self.assertFalse(summarize(self.report,self.base)['integrity_passed'])

    def test_ignored_support_flag(self):
        self.mutate(lambda r:r['first'].update(scale_support_enabled=False))
        self.assertFalse(summarize(self.report,self.base)['integrity_passed'])

    def test_slow_and_high_memory_retained(self):
        self.mutate(lambda r:(r['first'].update(total_ms=30),r['final_memory'].update(peak_rss_bytes=50*2**20)))
        s=summarize(self.report,self.base)
        self.assertFalse(s['all_time_tolerances_passed']);self.assertFalse(s['all_rss_tolerances_passed'])

    def test_truth_does_not_reach_worker(self):
        self.assertFalse(set(make_job(self.case,'support'))&{'truth','group','split'})


if __name__=='__main__':unittest.main()
