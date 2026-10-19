"""Negative controls for the direct uint8 initialization study."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_pose_u8_pipeline as b


def pose(x=20.,angle=0.,score=.95):
    return dict(pos_x=x,pos_y=20.,angle=angle,score=score,scale=1.,
                corners=[[10,10],[30,10],[30,30],[10,30]])


class PoseU8HarnessTest(unittest.TestCase):
    def evaluate(self,mutate=None,omit=None,duplicate=None):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            case=dict(name='x',validity='fixed_scale_gt',truth=[pose()])
            report=dict(protocol=dict(cases=[case],blocks=1,repeats=1,
                proposal_basin=dict(position_px=6.,angle_deg=6.,relative_scale=.02)),execution_order=[])
            for method in b.METHODS:
                if method==omit:continue
                cpu=method in b.CPU_METHODS;direct=method in b.DIRECT_METHODS
                sparse=dict(enabled=True,source_levels=1,response_mode=1,search_backend=1,gpu_buffer_bytes=1000,
                    command_buffers=2,cpu_waits=2,dispatches=5,peak_overflow_retries=0,
                    gradient_mode=1,gradient_lookup_bytes=b.GRADIENT_LOOKUP_BYTES,metal_submission_mode=0,
                    **{key:1 for key in b.COUNTS})
                duration=dict(frozen_shared=10.,current_cpu=6.,current_cpu_direct=5.,current_shared=8.,current_direct=4.)[method]
                call=dict(matches=[pose()],proposals=[pose()],sparse=sparse,frame_host_bytes=0,
                    selected_refinement_backend=0 if cpu else 1,pose_gpu_bytes=0 if cpu else 1000,pose_host_bytes=100,
                    total_ms=duration,prepare_ms=.1,proposal_ms=.4,refine_ms=duration/2,nms_ms=.1)
                call.update(refinement_mode=0 if cpu else 2,refinement_dispatches=0 if cpu else 1,
                    refinement_command_buffers=0 if cpu else 1,refinement_cpu_waits=0 if cpu else 1,
                    refinement_auxiliary_bytes=0 if cpu else 100)
                call.update({k:0. if cpu else .1 for k in b.REFINEMENT_TIMES})
                if cpu:call['refinement_device_ms']=-1
                call.update(refinement_threads=0 if cpu else 128,refinement_simd_width=0 if cpu else 32,refinement_threadgroup_bytes=0 if cpu else 7000,
                    pose_storage_mode=0 if cpu else 1,pose_host_image_bytes=400,
                    pose_gpu_image_bytes=0 if cpu else 400,pose_shared_image_bytes=0 if cpu else 400,
                    pose_unique_image_bytes=400,pose_create_ms=.5)
                if method!='frozen_shared':call.update(pose_input_mode=int(direct),pose_conversion_temporary_bytes=0 if direct else 400)
                warm=copy.deepcopy(call);warm['pose_create_ms']=0.
                if method!='frozen_shared':warm['pose_conversion_temporary_bytes']=0
                if mutate:
                    mutate(method,call);mutate(method,warm)
                raw=dict(first=call,runs=[warm],exact_repeat=True,construct_ms=.2,final_memory=dict(peak_rss_bytes=10000))
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
        self.assertEqual(pairs['direct_vs_shared']['speedup'],2.)
        self.assertEqual(pairs['direct_vs_shared']['reference'],'current_shared')
        self.assertEqual(pairs['shared_vs_frozen']['speedup'],1.25)
        self.assertEqual(pairs['cpu_direct_vs_cpu']['speedup'],1.2)
        self.assertEqual(pairs['direct_vs_cpu_direct']['speedup'],1.25)

    def test_fused_mode_silently_ignored_fails(self):
        def change(method,call):
            if method=='current_direct':call['refinement_mode']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_missing_current_metric_fails(self):
        for key in ('refinement_mode',*b.REFINEMENT_COUNTS,*b.REFINEMENT_TIMES,'refinement_threads','refinement_simd_width','refinement_threadgroup_bytes','pose_storage_mode','pose_create_ms','pose_input_mode','pose_conversion_temporary_bytes',*b.STORAGE_BYTES):
            def change(method,call):
                if method=='current_direct':del call[key]
            with self.subTest(key=key):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_frozen_may_omit_new_metrics_only(self):
        self.assertTrue(self.evaluate()['route_integrity_all'])
        def change(method,call):
            if method=='frozen_shared':del call['sparse']['metal_submission_mode']
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_unknown_device_time_is_not_fallback(self):
        def change(method,call):
            if method=='current_direct':call['refinement_device_ms']=-1
        self.assertTrue(self.evaluate(change)['route_integrity_all'])
        def invalid(method,call):
            if method=='current_direct':call['refinement_wait_ms']=-1
        self.assertFalse(self.evaluate(invalid)['route_integrity_all'])

    def test_fused_dispatch_not_fused_fails(self):
        def change(method,call):
            if method=='current_direct':call['refinement_dispatches']=20
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_resident_dispatch_silently_fused_fails(self):
        def change(method,call):
            if method=='current_shared':call['refinement_dispatches']=20
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_simd_kernel_geometry_not_ignored(self):
        def change(method,call):
            if method=='current_direct':call['refinement_threads']=256
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_no_gpu_work_fails(self):
        def change(method,call):
            if method=='current_direct':
                for key in b.REFINEMENT_COUNTS:call[key]=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_cpu_fallback_fails(self):
        def change(method,call):
            if method=='current_direct':call['selected_refinement_backend']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_cpu_gpu_work_fails(self):
        def change(method,call):
            if method=='current_cpu':call['refinement_command_buffers']=1
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_fractional_negative_bool_counter_fails(self):
        for value in (1.5,-1,True,float('nan')):
            def change(method,call):
                if method=='current_direct':call['refinement_dispatches']=value
            with self.subTest(value=value):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_frontend_mode_change_fails(self):
        def change(method,call):
            if method=='current_direct':call['sparse']['metal_submission_mode']=1
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_frontend_work_change_fails(self):
        def change(method,call):
            if method=='current_direct':call['sparse']['feature_evaluations']+=1
        self.assertFalse(self.evaluate(change)['exact_proposals_all'])

    def test_frontend_dispatch_change_fails(self):
        def change(method,call):
            if method=='current_direct':call['sparse']['dispatches']+=1
        self.assertFalse(self.evaluate(change)['frontend_identical_all'])

    def test_proposal_difference_fails(self):
        def change(method,call):
            if method=='current_direct':call['proposals'][0]['score']-=.001
        self.assertFalse(self.evaluate(change)['exact_proposals_all'])

    def test_current_simd_must_be_exact(self):
        def change(method,call):
            if method=='current_shared':call['matches'][0]['pos_x']+=.001
        result=self.evaluate(change)
        self.assertFalse(result['exact_resident_final_all'])
        self.assertTrue(result['paired_final_all'])

    def test_shared_numerical_difference_within_tolerance_still_fails_exactness(self):
        def change(method,call):
            if method=='current_direct':
                call['matches'][0]['pos_x']+=.2
                call['matches'][0]['angle']+=.2
                call['matches'][0]['score']-=.00005
        self.assertFalse(self.evaluate(change)['exact_resident_final_all'])

    def test_fused_pair_error_fails_without_hiding_gt(self):
        def change(method,call):
            if method=='current_direct':call['matches'][0]['angle']+=.3
        result=self.evaluate(change)
        self.assertFalse(result['fused_final_all'])
        self.assertTrue(result['normal_gt_all'])

    def test_cpu_cross_solver_error_separate(self):
        def change(method,call):
            if method in b.CPU_METHODS:call['matches'][0]['angle']+=.3
        result=self.evaluate(change)
        self.assertFalse(result['cpu_final_comparison_all'])
        self.assertFalse(result['paired_final_all'])
        self.assertTrue(all(result[k] for k in b.GATES))

    def test_score_tolerance_enforced(self):
        def change(method,call):
            if method=='current_direct':call['matches'][0]['score']-=.001
        self.assertFalse(self.evaluate(change)['fused_final_all'])

    def test_missing_final_or_duplicate_rejected(self):
        for duplicate in (False,True):
            def change(method,call):
                if method=='current_direct':call['matches']=[pose(),pose()] if duplicate else []
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
        result=self.evaluate(omit='current_direct')
        for key in ('all_workers_succeeded','exact_proposals_all','fused_final_all','frontend_identical_all','normal_gt_all'):
            self.assertFalse(result[key])

    def test_duplicate_worker_fails(self):
        result=self.evaluate(duplicate='current_direct')
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

    def test_shared_copy_and_overlap_accounting_rejected(self):
        for field,value in [('pose_storage_mode',0),('pose_shared_image_bytes',0),('pose_unique_image_bytes',800),('pose_host_image_bytes',0),('pose_gpu_image_bytes',0)]:
            def change(method,call):
                if method=='current_direct':call[field]=value
            with self.subTest(field=field):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_repeated_image_initialization_rejected(self):
        def change(method,call):
            if method=='current_direct':call['pose_create_ms']=.01
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_copied_images_cannot_claim_shared(self):
        def change(method,call):
            if method=='current_cpu':
                call['pose_shared_image_bytes']=400
                call['pose_unique_image_bytes']=400
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_direct_cpu_numerical_difference_not_tolerance_only(self):
        def change(method,call):
            if method=='current_cpu_direct':call['matches'][0]['pos_x']+=.001
        result=self.evaluate(change)
        self.assertFalse(result['exact_cpu_direct_all'])
        self.assertTrue(result['paired_final_all'])

    def test_input_route_ignored(self):
        for route in b.METHODS[1:]:
            def change(method,call):
                if method==route:call['pose_input_mode']=1-call['pose_input_mode']
            with self.subTest(route=route):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_direct_temporary_forbidden(self):
        for route in b.DIRECT_METHODS:
            def change(method,call):
                if method==route:call['pose_conversion_temporary_bytes']=400
            with self.subTest(route=route):self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_float_first_temporary_must_be_reported(self):
        def change(method,call):
            if method=='current_shared':call['pose_conversion_temporary_bytes']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_retained_storage_cannot_change(self):
        def change(method,call):
            if method=='current_direct':
                for key in b.STORAGE_BYTES:call[key]+=16
                call['pose_unique_image_bytes']=call['pose_host_image_bytes']
        self.assertTrue(self.evaluate(change)['route_integrity_all'])
        self.assertFalse(self.evaluate(change)['initialization_only_all'])

    def test_solver_work_cannot_change(self):
        def change(method,call):
            if method=='current_direct':
                for key in b.REFINEMENT_COUNTS[:3]:call[key]+=1
        self.assertTrue(self.evaluate(change)['route_integrity_all'])
        self.assertFalse(self.evaluate(change)['initialization_only_all'])

    def test_warm_float_temporary_forbidden(self):
        def change(method,call):
            if method=='current_cpu' and call['pose_create_ms']==0:call['pose_conversion_temporary_bytes']=400
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

    def test_first_creation_must_be_positive(self):
        def change(method,call):
            if method=='current_direct':call['pose_create_ms']=0
        self.assertFalse(self.evaluate(change)['route_integrity_all'])

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
                    self.assertEqual(job['backend'],0 if method in b.CPU_METHODS else 1)
                    self.assertEqual(job['sparse_metal_submission_mode'],0)
                    self.assertEqual(job['pose_resident_mode'],0 if method in b.CPU_METHODS else 2)
                    self.assertEqual(job['pose_storage_mode'],0 if method in b.CPU_METHODS else 1)
                    self.assertEqual(job['pose_input_mode'],int(method in b.DIRECT_METHODS))


if __name__=='__main__':unittest.main()
