"""CPU-only replay/negative controls for the stage34 lifecycle evidence reader."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from closure_lifecycle import sequence_summary, soak_summary
from benchmark_resident_multiscale import job_for
from benchmark_native_pipeline import save,sha


class LifecycleReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        fixture=json.loads((Path(__file__).parent/'fixtures/fine_search_harness.json').read_text())
        self.case=fixture['case'];self.pose=dict(pos_x=12.,pos_y=12.,angle=0.,scale=1.,score=.95)
        self.case['truth']=[self.pose];self.case['scales']=[1.]
        self.frames=[dict(index=0,path='/frame0.png',shape=[32,32],empty=False,truth=[self.pose],sha256='frame0'),
            dict(index=1,path='/frame1.png',shape=[48,48],empty=True,truth=[],sha256='frame1')]

    def tearDown(self):self.temp.cleanup()

    def row(self,active=True):
        return dict(matches=[self.pose] if active else [],proposals=[self.pose] if active else [],
            work=[dict(maps=int(active),wait_ms=1.)],total_ms=10.,frame_wall_ms=13.,source_update_ms=0.,
            construct_ms=2.,release_ms=0.,source_decode_ms=1.,pose_create_ms=1. if active else 0.,
            configured_refinement_backend=1,selected_refinement_backend=int(active),refinement_mode=0,
            pose_input_mode=1,pose_storage_mode=1,refinement_candidates=int(active),
            pose_gpu_bytes=128 if active else 0,refinement_dispatches=8 if active else 0,
            refinement_command_buffers=int(active),source_generation=0,retained_template_models=0,
            source_copy_bytes=0,pose_invalidated=False,source_dimensions_changed=False,
            exact_sequence_repeat=True,memory=dict(current_rss_bytes=2**20,peak_rss_bytes=2**20))

    def entry(self,label,job,raw):
        e=dict(label=label,returncode=0)
        for key,value in [('job',job),('raw',raw),('stderr','')]:
            path=self.base/(label+'.'+key+'.json')
            if key=='stderr':path.write_text(value)
            else:save(path,value)
            e[key]=path.name;e[key+'_sha256']=sha(path)
        return e

    def finish(self,p,entries):
        save(self.base/'protocol.json',p)
        return dict(complete=True,protocol=p,protocol_sha256=sha(self.base/'protocol.json'),entries=entries,
            unchanged=True,post_hashes=copy.deepcopy(p['hashes']))

    def sequence(self):
        p=dict(mode='sequence',case=self.case,frames=self.frames,blocks=2,cycles=2,
            job={**job_for(self.case,'resident0'),'refinement_backend':1,'run_resource_diagnostics':False},
            hashes={f['path']:f['sha256'] for f in self.frames})
        entries=[]
        for block in range(2):
            for mode in (['fresh','reuse'] if block==0 else ['reuse','fresh']):
                rows=[]
                for call in range(4):
                    index=call%2;frame=self.frames[index];r=self.row(not frame['empty']);reused=mode=='reuse' and call>0
                    r.update(call=call,source_index=index,sequence_cycle=call//2,source_width=frame['shape'][1],
                        source_height=frame['shape'][0],source_path=frame['path'],reference_cycle=call<2,
                        source_generation=call if mode=='reuse' else 0,retained_template_models=int(reused),
                        source_copy_bytes=frame['shape'][0]*frame['shape'][1] if reused else 0,
                        pose_invalidated=reused and index==1,source_dimensions_changed=reused,
                        source_update_ms=.5 if reused else 0.,construct_ms=0. if reused else 2.)
                    rows.append(r)
                raw=dict(sequence_mode=mode,source_frames=[f['path'] for f in self.frames],cycles=2,runs=rows,calls=4,
                    exact_sequence_repeat=True,drained=True,final_memory=dict(peak_rss_bytes=2**20),
                    after_release_memory=dict(current_rss_bytes=2**20),final_release_ms=.1)
                job={**p['job'],'source_frames':raw['source_frames'],'sequence_mode':mode,'repeats':2}
                e=self.entry(f'{mode}-{block}',job,raw);e.update(block=block,mode=mode);entries.append(e)
        return self.finish(p,entries)

    def mutate_raw(self,e,fn):
        path=self.base/e['raw'];r=json.loads(path.read_text());fn(r);save(path,r);e['raw_sha256']=sha(path)

    def soak(self):
        reference=dict(first=self.row());save(self.base/'reference.json',reference)
        p=dict(mode='soak',case=self.case,seconds=1.,reference_sha256=sha(self.base/'reference.json'),hashes={'source':'abc'})
        p['job']={**job_for(self.case,'resident0'),'refinement_backend':1,'run_resource_diagnostics':False,
            'soak_seconds':1.,'stream_path':'/samples.jsonl','warmup_seconds':1.}
        rows=[]
        for i in range(2):
            r=self.row();r.update(call=i,context_index=0,generation=6+i,recreate_ms=0.,elapsed_ms=10.*(i+1),exact=True)
            r.update(dict.fromkeys(('frame_host_bytes','frame_coarse_gpu_bytes','frame_fine_gpu_bytes','pose_host_bytes'),128))
            rows.append(r)
        raw=dict(calls=2,duration_ms=1000.,contexts_recreated=0,exact=True,drained=True,first=reference['first'])
        e=self.entry('soak',p['job'],raw)
        report=self.finish(p,[e]);self.write_stream(report,rows);return report

    def write_stream(self,report,rows):
        path=self.base/'samples.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows));report['stream_sha256']=sha(path)

    def mutate_stream(self,report,fn):
        rows=[json.loads(s) for s in (self.base/'samples.jsonl').read_text().splitlines()];fn(rows);self.write_stream(report,rows)

    def test_sequence_positive(self):
        r=sequence_summary(self.sequence(),self.base);self.assertTrue(r['passed']);self.assertEqual(r['calls'],16)

    def test_soak_positive(self):self.assertTrue(soak_summary(self.soak(),self.base)['passed'])

    def test_source_changed(self):
        r=self.sequence();r['post_hashes']['/frame0.png']='changed'
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_missing_worker(self):
        r=self.sequence();r['entries'].pop()
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_tampered_evidence(self):
        r=self.sequence();(self.base/r['entries'][0]['raw']).write_text('{}')
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_changed_job(self):
        r=self.sequence();e=r['entries'][0];p=self.base/e['job'];j=json.loads(p.read_text());j['sequence_mode']='reuse';save(p,j);e['job_sha256']=sha(p)
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_frame_order(self):
        r=self.sequence();self.mutate_raw(r['entries'][0],lambda raw:raw['runs'][1].update(source_index=0))
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_negative_timing(self):
        r=self.sequence();self.mutate_raw(r['entries'][0],lambda raw:raw['runs'][0].update(source_update_ms=-1))
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_narrowed_timing_scope(self):
        r=self.sequence();self.mutate_raw(r['entries'][0],lambda raw:raw['runs'][0].update(frame_wall_ms=10.))
        with self.assertRaises(ValueError):sequence_summary(r,self.base)

    def test_backend_route(self):
        r=self.sequence();self.mutate_raw(r['entries'][0],lambda raw:raw['runs'][0].update(selected_refinement_backend=0))
        self.assertFalse(sequence_summary(r,self.base)['passed'])

    def test_stale_generation(self):
        r=self.sequence();self.mutate_raw(r['entries'][1],lambda raw:raw['runs'][1].update(source_generation=0))
        self.assertFalse(sequence_summary(r,self.base)['routes_passed'])

    def test_both_arms_repeat_drift(self):
        r=self.sequence()
        for e in r['entries']:
            self.mutate_raw(e,lambda raw:raw['runs'][2]['matches'][0].update(pos_x=13.))
        s=sequence_summary(r,self.base);self.assertTrue(s['fresh_reuse_ordered_exact']);self.assertFalse(s['repeat_exact']);self.assertFalse(s['passed'])

    def test_common_accuracy_failure(self):
        r=self.sequence()
        for e in r['entries']:
            self.mutate_raw(e,lambda raw:[row.update(matches=[]) for row in raw['runs']])
        s=sequence_summary(r,self.base);self.assertTrue(s['fresh_reuse_ordered_exact']);self.assertFalse(s['accuracy_passed']);self.assertFalse(s['passed'])

    def test_soak_wrong_job(self):
        r=self.soak();e=r['entries'][0];p=self.base/e['job'];j=json.loads(p.read_text());j['fine_fused']=False;save(p,j);e['job_sha256']=sha(p)
        with self.assertRaises(ValueError):soak_summary(r,self.base)

    def test_soak_changed_reference(self):
        r=self.soak();save(self.base/'reference.json',{})
        with self.assertRaises(ValueError):soak_summary(r,self.base)

    def test_soak_missing_stream_call(self):
        r=self.soak();self.mutate_stream(r,lambda rows:rows.pop())
        with self.assertRaises(ValueError):soak_summary(r,self.base)

    def test_soak_bad_generation(self):
        r=self.soak();self.mutate_stream(r,lambda rows:rows[1].update(generation=99))
        self.assertFalse(soak_summary(r,self.base)['passed'])

    def test_soak_capacity_growth(self):
        r=self.soak();self.mutate_stream(r,lambda rows:rows[1].update(frame_host_bytes=256))
        self.assertFalse(soak_summary(r,self.base)['retained_capacity_constant'])

    def test_soak_rss_growth(self):
        r=self.soak();self.mutate_stream(r,lambda rows:rows[1]['memory'].update(current_rss_bytes=100*2**20))
        self.assertFalse(soak_summary(r,self.base)['passed'])

    def test_soak_short_duration(self):
        r=self.soak();self.mutate_raw(r['entries'][0],lambda raw:raw.update(duration_ms=999.))
        self.assertFalse(soak_summary(r,self.base)['passed'])

    def test_soak_changed_backend(self):
        r=self.soak();self.mutate_stream(r,lambda rows:rows[1].update(selected_refinement_backend=0))
        self.assertFalse(soak_summary(r,self.base)['routes_passed'])

    def test_soak_invalid_memory(self):
        r=self.soak();self.mutate_stream(r,lambda rows:rows[1]['memory'].update(current_rss_bytes=-1))
        with self.assertRaises(ValueError):soak_summary(r,self.base)


if __name__=='__main__':unittest.main()
