"""Negative controls for the grayscale fused resident refinement study."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_pose_fused_pipeline as b


def pose(x=20.,angle=0.,score=.95):
    return dict(pos_x=x,pos_y=20.,angle=angle,score=score,scale=1.,
                corners=[[10,10],[30,10],[30,30],[10,30]])


class PoseFusedHarnessTest(unittest.TestCase):
    def evaluate(self,mutate=None,omit=None,duplicate=None):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            case=dict(name='x',validity='fixed_scale_gt',truth=[pose()])
            report=dict(protocol=dict(cases=[case],blocks=1,repeats=1,
                proposal_basin=dict(position_px=6.,angle_deg=6.,relative_scale=.02)),execution_order=[])
            for method in b.METHODS:
                if method==omit:continue
                cpu=method=='current_cpu';fused=method=='current_fused'
                sparse=dict(enabled=True,source_levels=1,response_mode=1,search_backend=1,gpu_buffer_bytes=1000,
                    command_buffers=2,cpu_waits=2,dispatches=5,peak_overflow_retries=0,
                    gradient_mode=1,gradient_lookup_bytes=b.GRADIENT_LOOKUP_BYTES,metal_submission_mode=0,
                    **{key:1 for key in b.COUNTS})
                duration=dict(frozen_metal=10.,current_cpu=6.,current_resident=8.,current_fused=4.)[method]
                call=dict(matches=[pose()],proposals=[pose()],sparse=sparse,frame_host_bytes=0,
                    selected_refinement_backend=0 if cpu else 1,pose_gpu_bytes=0 if cpu else 1000,pose_host_bytes=100,
                    total_ms=duration,prepare_ms=.1,proposal_ms=.4,refine_ms=duration/2,nms_ms=.1)
                if method!='frozen_metal':
                    call.update(refinement_mode=int(fused),refinement_dispatches=0 if cpu else (1 if fused else 20),
                        refinement_command_buffers=0 if cpu else 1,refinement_cpu_waits=0 if cpu else 1,
                        refinement_auxiliary_bytes=0 if cpu else 100)
                    call.update({k:0. if cpu else .1 for k in b.REFINEMENT_TIMES})
                    if cpu:call['refinement_device_ms']=-1
                if mutate:mutate(method,call)
                raw=dict(first=call,runs=[copy.deepcopy(call)],exact_repeat=True,construct_ms=.2,final_memory=dict(peak_rss_bytes=10000))
                path=root/(method+'.json');path.write_text(json.dumps(raw))
                entry=dict(case='x',method=method,block=0,status='ok',result=path.name)
                report['execution_order'].append(entry)
                if method==duplicate:report['execution_order'].append(dict(entry))
            return b.summarize(report,root)

    def test_exact_baseline(self):
        summary=self.evaluate()
        self.assertTrue(all(summary[key] for key in b.GATES))
        self.assertTrue(summary['paired_final_all'])
        self.assertTrue(summary['cpu_final_comparison_all'])

    def test_attribution_uses_same_build_resident(self):
        pairs={p['attribution']:p for p in self.evaluate()['cases'][0]['paired_comparison']}
        self.assertEqual(pairs['fused_vs_resident']['speedup'],2.)
        self.assertEqual(pairs['fused_vs_resident']['reference'],'current_resident')
        self.assertEqual(pairs['resident_vs_frozen']['speedup'],1.25)
        self.assertEqual(pairs['fused_vs_cpu']['speedup'],1.5)
        self.assertEqual(pairs['fused_vs_frozen']['speedup'],2.5)

    def test_fused_mode_silently_ignored_fails(self):
        def change(method,call):
            if method=='current_fused':call['refinement_mode']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_missing_current_metric_fails(self):
        for key in ('refinement_mode',*b.REFINEMENT_COUNTS,*b.REFINEMENT_TIMES):
            def change(method,call):
                if method=='current_fused':del call[key]
            with self.subTest(key=key):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_frozen_may_omit_new_metrics_only(self):
        self.assertTrue(self.evaluate()['route_integrity_all'])
        def change(method,call):
            if method=='frozen_metal':del call['sparse']['metal_submission_mode']
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_unknown_device_time_is_not_fallback(self):
        def change(method,call):
            if method=='current_fused':call['refinement_device_ms']=-1
        self.assertTrue(self.evaluate(change)['route_integrity_all'])
        def invalid(method,call):
            if method=='current_fused':call['refinement_wait_ms']=-1
        self.assertFalse(self.evaluate(invalid)['route_integrity_all'])

    def test_fused_dispatch_not_fused_fails(self):
        def change(method,call):
            if method=='current_fused':call['refinement_dispatches']=20
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_resident_dispatch_silently_fused_fails(self):
        def change(method,call):
            if method=='current_resident':call['refinement_dispatches']=1
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_no_gpu_work_fails(self):
        def change(method,call):
            if method=='current_fused':
                for key in b.REFINEMENT_COUNTS:call[key]=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_cpu_fallback_fails(self):
        def change(method,call):
            if method=='current_fused':call['selected_refinement_backend']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_cpu_gpu_work_fails(self):
        def change(method,call):
            if method=='current_cpu':call['refinement_command_buffers']=1
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_fractional_negative_bool_counter_fails(self):
        for value in (1.5,-1,True,float('nan')):
            def change(method,call):
                if method=='current_fused':call['refinement_dispatches']=value
            with self.subTest(value=value):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_frontend_mode_change_fails(self):
        def change(method,call):
            if method=='current_fused':call['sparse']['metal_submission_mode']=1
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_frontend_work_change_fails(self):
        def change(method,call):
            if method=='current_fused':call['sparse']['feature_evaluations']+=1
        self.assertFalse(self.evaluate(change)['exact_proposals_all'])

    def test_frontend_dispatch_change_fails(self):
        def change(method,call):
            if method=='current_fused':call['sparse']['dispatches']+=1
        self.assertFalse(self.evaluate(change)['frontend_identical_all'])

    def test_proposal_difference_fails(self):
        def change(method,call):
            if method=='current_fused':call['proposals'][0]['score']-=.001
        self.assertFalse(self.evaluate(change)['exact_proposals_all'])

    def test_current_resident_must_be_exact(self):
        def change(method,call):
            if method=='current_resident':call['matches'][0]['pos_x']+=.001
        result=self.evaluate(change)
        self.assertFalse(result['exact_resident_final_all'])
        self.assertTrue(result['paired_final_all'])

    def test_fused_numerical_difference_within_bounds_passes(self):
        def change(method,call):
            if method=='current_fused':
                call['matches'][0]['pos_x']+=.2
                call['matches'][0]['angle']+=.2
                call['matches'][0]['score']-=.00005
        self.assertTrue(all(self.evaluate(change)[k] for k in b.GATES))

    def test_fused_pair_error_fails_without_hiding_gt(self):
        def change(method,call):
            if method=='current_fused':call['matches'][0]['angle']+=.3
        result=self.evaluate(change)
        self.assertFalse(result['fused_final_all'])
        self.assertTrue(result['normal_gt_all'])

    def test_cpu_cross_solver_error_separate(self):
        def change(method,call):
            if method=='current_cpu':call['matches'][0]['angle']+=.3
        result=self.evaluate(change)
        self.assertFalse(result['cpu_final_comparison_all'])
        self.assertFalse(result['paired_final_all'])
        self.assertTrue(all(result[k] for k in b.GATES))

    def test_score_tolerance_enforced(self):
        def change(method,call):
            if method=='current_fused':call['matches'][0]['score']-=.001
        self.assertFalse(self.evaluate(change)['fused_final_all'])

    def test_missing_final_or_duplicate_rejected(self):
        for duplicate in (False,True):
            def change(method,call):
                if method=='current_fused':call['matches']=[pose(),pose()] if duplicate else []
            result=self.evaluate(change)
            self.assertFalse(result['fused_final_all'])
            self.assertFalse(result['normal_gt_all'])

    def test_common_gt_failure_not_hidden_by_baseline(self):
        def change(method,call):call['matches'][0]['pos_x']+=10
        result=self.evaluate(change)
        self.assertTrue(result['exact_resident_final_all'])
        self.assertTrue(result['paired_final_all'])
        self.assertFalse(result['normal_gt_all'])

    def test_missing_worker_fails(self):
        result=self.evaluate(omit='current_fused')
        for key in ('all_workers_succeeded','exact_proposals_all','fused_final_all','frontend_identical_all','normal_gt_all'):
            self.assertFalse(result[key])

    def test_duplicate_worker_fails(self):
        result=self.evaluate(duplicate='current_fused')
        self.assertFalse(result['all_workers_succeeded'])
        self.assertFalse(result['fused_final_all'])

    def test_artifact_tamper_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'worker.json';path.write_text('{}')
            report=dict(execution_order=[dict(result=path.name,result_sha256=b.sha(path))])
            b.verify_workers(report,root);path.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError,'Changed worker artifact'):b.verify_workers(report,root)

    def test_one_to_one_not_nearest_neighbor_only(self):
        self.assertTrue(b.final_comparison([pose(20.1),pose(20.)],[pose(20.),pose(20.3)])['valid'])
        self.assertFalse(b.final_comparison([pose(20.),pose(20.)],[pose(20.),pose(21.)])['valid'])

    def test_angle_wrap_and_output_reorder(self):
        result=b.final_comparison([pose(40.,-179.9),pose(20.)],[pose(20.),pose(40.,179.9)])
        self.assertTrue(result['valid'])

    def test_invalid_pose_never_matches(self):
        for key,value in (('scale',0.),('score',float('nan')),('pos_x',float('inf'))):
            bad=pose();bad[key]=value
            self.assertFalse(b.final_comparison([bad],[pose()])['valid'])

    def test_job_contains_no_truth_or_roi(self):
        protocol=dict(shared_settings=dict(optimize_scale=False),sparse_settings={},repeats=1,warmup_seconds=0.,
                      timeout_seconds=1.,builds={m:'/fake-build' for m in b.METHODS})
        case=dict(name='x',source='x.png',template='t.png',angle=20.,truth=[pose()],roi=[0,0,10,10])
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(b.subprocess,'run') as run:
                run.return_value.returncode=1;run.return_value.stdout='';run.return_value.stderr='test'
                for method in b.METHODS:
                    b.run_worker(protocol,case,method,0,Path(folder)/'results.json')
                    job=json.loads((Path(folder)/'jobs'/('x-'+method+'-block0.json')).read_text())
                    self.assertFalse({'truth','roi','expected_count'} & set(job))
                    self.assertEqual(job['backend'],0 if method=='current_cpu' else 1)
                    self.assertEqual(job['sparse_metal_submission_mode'],0)
                    if method=='frozen_metal':self.assertNotIn('pose_resident_mode',job)
                    else:self.assertEqual(job['pose_resident_mode'],int(method=='current_fused'))


if __name__=='__main__':unittest.main()
