"""Synthetic geometry and diagnostic invariants, no GPU/matcher invocation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import challenge_coverage_scenes as scenes
import challenge_coverage_benchmark as bench


class ChallengeCoverageTest(unittest.TestCase):
    def setUp(self):
        self.template=np.arange(192,dtype=np.uint8).reshape(12,16)

    def test_upright_integer_pixels_match_template(self):
        item=scenes.placement(0,[24.,24.],0,1,self.template.shape)
        image,visible,_=scenes.composite(self.template,[item],(48,48),8,noise_std=0)
        np.testing.assert_array_equal(image[18:30,16:32],self.template)
        self.assertEqual(visible,[1.])

    def test_strip_visibility_and_only_declared_columns(self):
        item=scenes.placement(0,[24.,24.],0,1,self.template.shape,strip_fraction=.25)
        image,visible,_=scenes.composite(self.template,[item],(48,48),8,noise_std=0)
        np.testing.assert_array_equal(image[18:30,20:32],self.template[:,4:])
        np.testing.assert_array_equal(image[18:30,16:20],np.full((12,4),150))
        self.assertEqual(visible,[.75])

    def test_depth_occlusion_is_measured(self):
        a=scenes.placement(0,[24.,24.],0,1,self.template.shape)
        b=scenes.placement(1,[32.,24.],0,1,self.template.shape)
        _,visible,_=scenes.composite(self.template,[a,b],(48,64),8,noise_std=0)
        self.assertEqual(visible,[.5,1.])

    def test_negative_identity_changes_only_patch(self):
        variant,(x0,y0,x1,y1)=scenes.altered_identity(self.template)
        expected=self.template.copy();expected[y0:y1,x0:x1]=255-expected[y0:y1,x0:x1]
        np.testing.assert_array_equal(variant,expected)
        self.assertGreater(np.count_nonzero(variant!=self.template),0)

    def test_invalid_inputs_are_rejected(self):
        for fraction in (-.1,1.1):
            item=scenes.placement(0,[24.,24.],0,1,self.template.shape,strip_fraction=fraction)
            with self.assertRaises(ValueError):scenes.composite(self.template,[item],(48,48),1)
        clipped=scenes.placement(0,[0.,0.],0,1,self.template.shape)
        with self.assertRaises(ValueError):scenes.composite(self.template,[clipped],(48,48),1)

    def test_splits_are_reproducible_with_distinct_templates(self):
        a,b=scenes.scenes(),scenes.scenes()
        self.assertEqual(len(a),6)
        self.assertEqual(sum(len(x['truth']) for x in a),32)
        self.assertEqual(sum(len(x['distractors']) for x in a),8)
        self.assertNotEqual(a[0]['template_fixture'],a[3]['template_fixture'])
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x['image'],y['image'])
            self.assertEqual(x['objects'],y['objects'])
            self.assertTrue(all(0<=o['visible_content_fraction']<=1.000001 for o in x['objects']))
        for scene in (a[2],a[5]):
            self.assertLess(scene['objects'][4]['visible_content_fraction'],.8)
            self.assertAlmostEqual(scene['objects'][5]['visible_content_fraction'],1.,places=5)

    def test_worker_never_receives_ground_truth(self):
        case=dict(source='s',template='t',scales=[1.],angle=180,limit=70,proposal_score=.5,score=.8,
                  overlap=.2,stop_layer=1,subpixel=0,workflow='full_rescue',truth=[{'secret':1}],
                  distractors=[{'secret':2}],metadata='secret',split='heldout')
        a,b=bench.worker_job(case,'baseline'),bench.worker_job(case,'expanded')
        self.assertFalse({'truth','distractors','metadata','split'} & set(a))
        self.assertEqual({k for k in a if a[k]!=b[k]},{'proposal_score','limit'})
        self.assertEqual(a['score'],b['score'])

    def test_coverage_bound_is_necessary_not_strict_accuracy(self):
        t=dict(pos_x=100.,pos_y=100.,angle=179.,scale=1.)
        p=dict(t,pos_x=111.,angle=-178.,scale=1.2,score=.99)
        self.assertEqual(bench.proposal_coverage([p],[t])['covered'],1)
        case=dict(truth=[dict(t,visible_content_fraction=1.)],distractors=[])
        d=bench.diagnose(dict(matches=[p],proposals=[p]),case)
        self.assertEqual(d['accuracy']['strict']['tp'],0)
        self.assertEqual(d['failures'][0]['reason'],'accepted_wrong_pose')

    def test_duplicate_predictions_remain_false_positive(self):
        t=dict(pos_x=100.,pos_y=100.,angle=0.,scale=1.,visible_content_fraction=1.)
        p=dict(t,score=.99)
        d=bench.diagnose(dict(matches=[p,p],proposals=[p]),dict(truth=[t],distractors=[]))
        self.assertEqual(d['accuracy']['strict']['tp'],1)
        self.assertEqual(d['accuracy']['strict']['fp'],1)

    def test_identity_distractor_remains_false_positive(self):
        t=dict(pos_x=100.,pos_y=100.,angle=0.,scale=1.,visible_content_fraction=1.)
        p=dict(t,score=.99)
        d=bench.diagnose(dict(matches=[p],proposals=[p]),dict(truth=[],distractors=[t]))
        self.assertEqual(d['accuracy']['strict']['fp'],1)
        self.assertEqual(len(d['distractor_false_acceptances']),1)

    def test_generation_freezes_input_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as path:
            cases=scenes.generate(Path(path)/'inputs')
            protocol=bench.prepare(Path(path)/'inputs/cases.json',Path(path)/'results')
            self.assertEqual(len(protocol['order']),24)
            self.assertEqual(len(protocol['inputs_sha256']),18)
            self.assertEqual(len(cases),6)
            with self.assertRaises(ValueError):scenes.generate(Path(path)/'inputs')
            with self.assertRaises(ValueError):bench.prepare(Path(path)/'inputs/cases.json',Path(path)/'results')

    def test_historical_periodic_failures_not_relabelled(self):
        old=bench.historic_periodic()
        self.assertTrue(old['historical_only'])
        self.assertFalse(old['retested_current_binary'])
        self.assertGreater(old['geometric_failures_preserved'],0)

class ChallengeSummaryTest(unittest.TestCase):
    def evidence(self, folder):
        root=Path(folder)
        target=dict(pos_x=100.,pos_y=100.,angle=0.,scale=1.,visible_content_fraction=1.)
        wrong=dict(target,pos_x=108.,score=.99999)
        case=dict(name='test',split='heldout',kind='occlusion',source='s',template='t',scales=[1.],
            angle=180,limit=70,proposal_score=.5,score=.8,overlap=.2,stop_layer=1,subpixel=0,
            workflow='full_rescue',truth=[target],distractors=[])
        order=[dict(case='test',method=m,block=0) for m in bench.METHODS]
        protocol=dict(cases=[case],order=order,methods=bench.METHODS)
        bench.save(root/'protocol.json',protocol)
        report=dict(protocol,protocol_sha256=bench.sha(root/'protocol.json'),complete=True,unchanged=True,execution=[])
        for entry in order:
            label=entry['method']
            job=root/(label+'.json'); result=root/(label+'.result.json'); stderr=root/(label+'.stderr.txt')
            bench.save(job,bench.worker_job(case,label));stderr.write_text('')
            call=dict(total_ms=12. if label=='expanded' else 10.,proposal_ms=7.,refine_ms=3.,
                proposals=[wrong],matches=[wrong],selected_refinement_backend=1,pose_input_mode=1,
                pose_storage_mode=1,refinement_mode=0)
            raw=dict(first=call,runs=[call],warmups_ms=[],exact_repeat=True,construct_ms=1.,
                final_memory=dict(peak_rss_bytes=2**20),load=dict(decode_ms=1.))
            bench.save(result,raw)
            report['execution'].append(dict(entry,job=job.name,job_sha256=bench.sha(job),
                result=result.name,result_sha256=bench.sha(result),stderr=stderr.name,stderr_sha256=bench.sha(stderr)))
        return root,report

    def test_common_high_score_wrong_peak_is_failed_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report=self.evidence(folder)
            result=bench.summarize(report,root)
            self.assertFalse(result['all_strict_passed'])
            self.assertTrue(result['all_performance_passed'])
            self.assertEqual(result['workers'],2)

    def test_tampered_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report=self.evidence(folder)
            (root/report['execution'][0]['result']).write_text('{}')
            with self.assertRaises(ValueError):bench.summarize(report,root)

    def test_changed_protocol_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report=self.evidence(folder)
            report['cases'][0]['score']=.1
            with self.assertRaises(ValueError):bench.summarize(report,root)

    def test_missing_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report=self.evidence(folder)
            report['execution'].pop()
            with self.assertRaises(ValueError):bench.summarize(report,root)

    def test_wrong_backend_rejected_even_with_updated_result_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root,report=self.evidence(folder)
            e=report['execution'][0];p=root/e['result'];raw=json.loads(p.read_text())
            raw['first']['selected_refinement_backend']=0
            bench.save(p,raw);e['result_sha256']=bench.sha(p)
            with self.assertRaises(ValueError):bench.summarize(report,root)

if __name__=='__main__':unittest.main()
