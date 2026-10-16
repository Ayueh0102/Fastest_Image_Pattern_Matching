"""Replay integrity and route checks without Metal or performance jobs."""
import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_fine_search import METHODS,ROOT,job_for,route,summarize,sha,save,CLOCKS

class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        fixture=json.loads((Path(__file__).parent/'fixtures/fine_search_harness.json').read_text())
        self.case=dict(fixture['case'],expect_reuse=True,benchmark_repeats=5)
        reference=dict(first=fixture['first'],runs=[copy.deepcopy(fixture['first']) for _ in range(5)],exact_repeat=True,construct_ms=0,final_memory=fixture['final_memory'])
        self.report=dict(version=1,cases=[self.case],expected_workers=8,execution_order=[],complete=False)
        save(self.base/'protocol.json',self.report);self.report['protocol_sha256']=sha(self.base/'protocol.json')
        for block in range(2):
            for method in (METHODS if block==0 else METHODS[::-1]):
                raw=copy.deepcopy(reference)
                for run in [raw['first'],*raw['runs']]:
                    ds=[]
                    for w in run['work']:
                        fields=('calls','requests','within_call_duplicates','cross_call_duplicates','untracked_call_keys','tracked_unique_keys','history_limit_keys','history_key_payload_bytes','history_bucket_count','peak_call_key_capacity_bytes','logical_requests','reused_maps','cache_peak_bytes','warp_batches','zero_repair_batches','guard_words','nonzero_guard_words')
                        d=dict.fromkeys((*CLOCKS,*fields),0);d.update(profiling_enabled=int(method=='profile'),angle_reuse_enabled=int(method=='reuse'),warp_device_ms=-1)
                        if method=='profile':d.update(requests=w['fine_maps'],tracked_unique_keys=w['fine_maps'],history_limit_keys=65536,warp_batches=w['fine_stats_dispatches'],warp_device_ms=0,warp_wait_ms=w['fine_wait_ms'])
                        if method=='reuse':
                            d.update(logical_requests=w['fine_maps'],reused_maps=1,cache_peak_bytes=1024);w['fine_maps']-=1
                        ds.append(d)
                    if method!='frozen':run['fine_search']=ds
                job=self.base/f'{method}{block}.json';result=self.base/f'{method}{block}.result.json';save(job,job_for(self.case,method));save(result,raw)
                self.report['execution_order'].append(dict(case=self.case['name'],method=method,block=block,job=job.name,job_sha256=sha(job),result=result.name,result_sha256=sha(result),returncode=0))
        self.report.update(complete=True,unchanged=True)
    def tearDown(self):self.temp.cleanup()
    def mutate(self,method,fn):
        for e in self.report['execution_order']:
            if e['method']!=method:continue
            p=self.base/e['result'];r=json.loads(p.read_text());fn(r);save(p,r);e['result_sha256']=sha(p)
    def test_valid_and_relocatable(self):
        s=summarize(self.report,self.base);self.assertTrue(s['correctness_passed']);self.assertEqual(s['formal_calls'],40)
        with tempfile.TemporaryDirectory() as other:
            moved=Path(other)/'relocated';shutil.copytree(self.base,moved)
            self.assertEqual(s,summarize(self.report,moved))
    def test_job_whitelist(self):
        e=self.report['execution_order'][0];p=self.base/e['job'];j=json.loads(p.read_text());j['truth']=[];save(p,j);e['job_sha256']=sha(p)
        with self.assertRaises(ValueError):summarize(self.report,self.base)
    def test_hash(self):
        e=self.report['execution_order'][0];(self.base/e['result']).write_text('{}')
        with self.assertRaises(ValueError):summarize(self.report,self.base)
    def test_duplicate_missing_failure(self):
        for kind in ('duplicate','missing','failed'):
            r=copy.deepcopy(self.report)
            if kind=='duplicate':r['execution_order'][1]=r['execution_order'][0]
            elif kind=='missing':r['execution_order'].pop()
            else:r['execution_order'][0]['returncode']=1
            with self.assertRaises(ValueError):summarize(r,self.base)
    def test_protocol_and_duplicate_cases(self):
        r=copy.deepcopy(self.report);r['cases'][0]['score']=.1
        with self.assertRaises(ValueError):summarize(r,self.base)
        r=copy.deepcopy(self.report);r['cases'].append(r['cases'][0]);save(self.base/'protocol.json',{k:v for k,v in r.items() if k not in ('protocol_sha256','unchanged')});r['protocol_sha256']=sha(self.base/'protocol.json')
        with self.assertRaises(ValueError):summarize(r,self.base)
    def test_bad_reuse_accounting(self):
        self.mutate('reuse',lambda raw:raw['runs'][0]['fine_search'][0].update(reused_maps=999))
        self.assertFalse(summarize(self.report,self.base)['correctness_passed'])
    def test_bad_profile_counter(self):
        self.mutate('profile',lambda raw:raw['runs'][0]['fine_search'][0].update(requests=-1))
        self.assertFalse(summarize(self.report,self.base)['correctness_passed'])
    def test_off_clock_and_capacity_regression_rejected(self):
        def bad_clock(raw):raw['runs'][0]['fine_search'][0]['plan_ms']=1
        self.mutate('current',bad_clock)
        self.assertFalse(summarize(self.report,self.base)['correctness_passed'])
        self.mutate('reuse',lambda raw:raw['runs'][0].update(frame_fine_gpu_bytes=10**12))
        self.assertFalse(summarize(self.report,self.base)['correctness_passed'])
    def test_nan_rejected(self):
        raw=json.loads((self.base/self.report['execution_order'][0]['result']).read_text());run=raw['runs'][0];bad=copy.deepcopy(run);bad['total_ms']=float('nan')
        self.assertFalse(route(bad,run,'frozen'))
    def test_all_missed_reported(self):
        self.mutate('reuse',lambda raw:[r.update(matches=[]) for r in [raw['first'],*raw['runs']]])
        s=summarize(self.report,self.base);self.assertFalse(s['correctness_passed']);a=s['cases'][0]['methods']['reuse']['accuracy'];self.assertGreater(a['max_fn'],0);self.assertTrue(all(v is None for v in a['max_errors'].values()))
    def test_positive_case_cannot_silently_disable_reuse(self):
        def disable(raw):
            for r in [raw['first'],*raw['runs']]:
                for d,w in zip(r['fine_search'],r['work']):w['fine_maps']+=d['reused_maps'];d['reused_maps']=0
        self.mutate('reuse',disable);self.assertFalse(summarize(self.report,self.base)['correctness_passed'])

if __name__=='__main__':unittest.main()
