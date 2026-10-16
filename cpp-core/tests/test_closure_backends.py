"""CPU-only replay/negative controls for the same-frontend backend comparison."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_closure_backends as harness
import test_resident_multiscale_harness as prior


class ClosureBackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.helper = prior.ResidentMultiscaleHarnessTest()
        self.helper.fixture = json.loads((Path(__file__).parent/'fixtures/fine_search_harness.json').read_text())
        self.make_report()

    def tearDown(self):
        self.temp.cleanup()

    def test_top_level_only_still_exercises_coarse_gpu(self):
        def coarse_only(raw):
            for row in [raw['first'], *raw['runs']]:
                row['frame_fine_gpu_bytes']=0
                for work in row['work']:
                    for key in work:
                        if key.startswith('fine_'):work[key]=0
                    work.update(fine_ncc_mode=2,fine_stats_device_ms=-1)
                for search in row['fine_search']:
                    for key in search:search[key]=0
                    search.update(angle_reuse_enabled=1,warp_device_ms=-1)
                for fusion in row['fine_fusion']:
                    for key in fusion:fusion[key]=0
                    fusion.update(enabled=1,fused_device_ms=-1)
        self.mutate(None,coarse_only,case='new_small12')
        self.assertTrue(harness.summarize(self.report,self.base)['integrity_passed'])

    def make_report(self):
        cases = [dict(self.helper.fixture['case'], name=n) for n in harness.CASES]
        protocol = dict(cases=cases, blocks=2, first_and_one_repeat=True, cpu_threads=8,
                        order=harness.order(harness.CASES, 2), hashes={'frozen':'sha'},
                        time_gate='max(10%,2ms)', rss_gate='max(10%,16MiB)')
        harness.save(self.base/'protocol.json', protocol)
        self.report = dict(protocol=protocol, protocol_sha256=harness.sha(self.base/'protocol.json'),
                           entries=[], complete=True, unchanged=True, post_hashes=protocol['hashes'])
        for name, block, method in harness.order(harness.CASES, 2):
            case = next(c for c in cases if c['name'] == name)
            calls = [self.helper.reference('resident0', first) for first in (True, False)]
            for call in calls:
                call.update(refine_ms=2., selected_refinement_backend=int(method=='metal'), pose_storage_mode=int(method=='metal'))
                if method == 'cpu':
                    call.update(pose_gpu_bytes=0, refinement_dispatches=0)
                call['resources'] = dict.fromkeys(('wall_ms','process_user_ms','process_system_ms','minor_faults','major_faults','voluntary_switches','involuntary_switches'), 1.)
            raw = dict(first=calls[0], runs=[calls[1]], warmups_ms=[], exact_repeat=True,
                       final_memory=dict(peak_rss_bytes=2**20))
            entry = dict(case=name, block=block, method=method, returncode=0)
            for field, value in [('job', harness.job(case,method)), ('raw',raw), ('stderr','')]:
                path = self.base/f'{name}-{block}-{method}-{field}.json'
                if field == 'stderr': path.write_text(value)
                else: harness.save(path,value)
                entry[field], entry[field+'_sha256'] = path.name, harness.sha(path)
            self.report['entries'].append(entry)

    def mutate(self, method, change, case='controlled6', block=None):
        for entry in self.report['entries']:
            if entry['case'] != case or method and entry['method'] != method or block is not None and entry['block'] != block:
                continue
            path = self.base/entry['raw']
            raw = json.loads(path.read_text())
            change(raw)
            harness.save(path, raw)
            entry['raw_sha256'] = harness.sha(path)

    def summary(self):
        return harness.summarize(self.report, self.base)

    def test_valid_balanced_relocatable_and_serialized(self):
        result = self.summary()
        self.assertTrue(result['correctness_passed'] and result['all_performance_tolerances_passed'])
        self.assertEqual((result['workers'], result['algorithm_calls']), (16,32))
        self.assertEqual(result, harness.summarize(json.loads(json.dumps(self.report)), self.base))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'moved'
            shutil.copytree(self.base, target)
            self.assertEqual(result, harness.summarize(self.report,target))

    def test_common_accuracy_failure_remains_false(self):
        self.mutate(None, lambda raw:[r.update(matches=[]) for r in [raw['first'],*raw['runs']]], case='new_small12')
        result = self.summary()
        self.assertTrue(result['integrity_passed'] and result['paired_final_passed'])
        self.assertFalse(result['accuracy_passed'] or result['correctness_passed'])
        self.assertEqual(len(result['accuracy_failures']),2)

    def test_performance_and_memory_failures_not_correctness(self):
        self.mutate('metal',lambda raw:raw['first'].update(total_ms=99.),block=1)
        self.mutate('metal',lambda raw:raw['final_memory'].update(peak_rss_bytes=2**30),block=1)
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_time_tolerances_passed'] or result['all_rss_tolerances_passed'])
        self.assertEqual(len(result['performance_failures']),2)

    def test_job_truth_is_not_sent(self):
        case = dict(self.report['protocol']['cases'][0], truth=[dict(secret=1)], metadata='secret')
        for method in harness.METHODS:
            job = harness.job(case,method)
            self.assertFalse({'truth','metadata'} & set(job))
            self.assertEqual(job['refinement_backend'],int(method=='metal'))
            self.assertEqual(job['cpu_threads'],8)
            self.assertTrue(job['run_resource_diagnostics'])

    def test_incomplete_missing_duplicate_reordered_worker_rejected(self):
        original = copy.deepcopy(self.report)
        for kind in ('incomplete','missing','duplicate','reorder','failed'):
            self.report = copy.deepcopy(original)
            if kind=='incomplete': self.report['complete']=False
            if kind=='missing': self.report['entries'].pop()
            if kind=='duplicate': self.report['entries'][1]=self.report['entries'][0]
            if kind=='reorder': self.report['entries'][:2]=self.report['entries'][1::-1]
            if kind=='failed': self.report['entries'][0]['returncode']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError): self.summary()

    def test_job_hash_and_semantic_tamper(self):
        entry = self.report['entries'][0]
        path = self.base/entry['job']
        value=json.loads(path.read_text());value['refinement_backend']=9
        harness.save(path,value)
        with self.assertRaises(ValueError):self.summary()
        entry['job_sha256']=harness.sha(path)
        with self.assertRaises(ValueError):self.summary()

    def test_provenance_and_protocol_tamper(self):
        self.report['post_hashes']={}
        with self.assertRaises(ValueError):self.summary()
        self.make_report()
        self.report['protocol']['cpu_threads']=1
        harness.save(self.base/'protocol.json',self.report['protocol'])
        self.report['protocol_sha256']=harness.sha(self.base/'protocol.json')
        with self.assertRaises(ValueError):self.summary()

    def test_hidden_warmup_rejected(self):
        self.mutate('metal',lambda raw:raw.update(warmups_ms=[1.]))
        with self.assertRaises(ValueError):self.summary()

    def test_cpu_gpu_route_cannot_silently_fallback(self):
        self.mutate('metal',lambda raw:raw['first'].update(selected_refinement_backend=0))
        self.assertFalse(self.summary()['integrity_passed'])
        self.make_report()
        self.mutate('cpu',lambda raw:raw['first'].update(pose_gpu_bytes=1))
        self.assertFalse(self.summary()['integrity_passed'])

    def test_equal_wrong_frontends_are_not_valid(self):
        self.mutate(None,lambda raw:[r['fine_fusion'][0].update(enabled=0) for r in [raw['first'],*raw['runs']]])
        result=self.summary()
        self.assertFalse(result['integrity_passed'])
        self.assertTrue(all(p['frontend_exact'] for p in result['cases'][0]['pairs']))

    def test_frontend_change_across_both_arms_in_block_detected(self):
        self.mutate(None,lambda raw:raw['first']['work'][0].update(windows=999),block=1)
        self.assertFalse(self.summary()['integrity_passed'])

    def test_distributions_and_bootstrap_process_pairs(self):
        result=harness.distribution([1,2,3,4])
        self.assertEqual(result['median'],2.5)
        self.assertAlmostEqual(result['p95'],3.85)
        paired=harness.paired_change([1,2,3],[2,4,6])
        self.assertEqual(paired['percentile_bootstrap_95'],[100.,100.])
        self.assertEqual(paired['independent_pairs'],3)
        for values in ([],[-1],[float('nan')]):
            with self.assertRaises(ValueError):harness.distribution(values)
        with self.assertRaises(ValueError):harness.order(['case'],3)


if __name__=='__main__':
    unittest.main()
