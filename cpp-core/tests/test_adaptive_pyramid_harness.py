"""Negative controls for the round-49 adaptive pyramid harness (no binaries, no GPU)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_adaptive_pyramid as harness  # noqa: E402
import stage49_scenes as scenes  # noqa: E402


def truth(count):
    return [dict(index=i, pos_x=50.*i+20, pos_y=30., angle=10.*i, scale=1.) for i in range(count)]


def found(objects, missing=0, extra=0):
    matches = [dict(pos_x=o['pos_x']+.1, pos_y=o['pos_y'], angle=o['angle']+.2, scale=1., score=.9) for o in objects[missing:]]
    return matches+[dict(pos_x=-500.-k, pos_y=-500., angle=0., scale=1., score=.85) for k in range(extra)]


class Fake:
    """Writes runner-shaped results and a manifest the summaries accept."""

    def __init__(self, root, cases):
        self.root, self.cases = root, cases
        self.cases_path = root/'cases.json'
        scenes.save(self.cases_path, cases)
        self.executions = []

    def add(self, case, arm, matches, ms, block=0, rule=None, exact=True, busy=False):
        label = f"{case['name']}-{arm}-b{block}"
        job, result = self.root/'jobs'/(label+'.json'), self.root/'results'/(label+'.json')
        scenes.save(job, dict(source=case['source']))
        run = dict(search_matches=matches, search_ms=ms,
                   search_diagnostics=dict(frame_coarse_gpu_bytes=1., frame_fine_gpu_bytes=2., pose_gpu_bytes=3.))
        value = dict(first=dict(run, search_ms=ms*2), runs=[run, run, run], exact_repeat=exact, search_construct_ms=1.,
                     decode_ms=1., final_memory=dict(peak_rss_bytes=2.**25))
        if rule:
            value['pyramid_rule'] = rule
        scenes.save(result, value)
        self.executions.append(dict(case=case['name'], arm=arm, block=block, job=str(job.relative_to(self.root)),
                                    job_sha256=scenes.sha(job), result=str(result.relative_to(self.root)),
                                    result_sha256=scenes.sha(result), returncode=0 if exact else 1, lock_waits=[],
                                    host=dict(competing_processes=[dict(command='cmake')] if busy else [], load_average=[1., 1., 1.])))

    def manifest(self):
        return dict(cases_path=str(self.cases_path), cases_sha256=scenes.sha(self.cases_path), executions=self.executions, unchanged=True)


def case(name, template, objects, condition='clean', split='holdout', size=(43, 30)):
    return dict(name=name, template_name=template, condition=condition, positive=bool(objects), split=split,
                expected_count=len(objects), truth=objects, source=f'/{name}.png', template=f'/{template}.png',
                template_size=list(size))


def rule(original, selected, scores):
    return dict(min_score=.5, models=[dict(scale=1, original_top=original, selected_top=selected, effective_min_area=256,
                                           predicted_scores=scores, low_background=10, high_background=200, ms=1.)])


PROTOCOL = dict(frozen=dict(pyramid_rule_min_score=.5, budget=dict(proposal_score=.3, limit=300)),
                gates=dict(deficient_recall_below=.9, H1_recovered_fraction=.8, H2_kept_fraction=.9, H2_time_ratio=1.05,
                           H2r_template='keep', H3_time_ratio=1.1, H4_time_ratio=.75, H4_tp_tolerance=.02, H6_fp_tolerance=.01))


class DepthMirror(unittest.TestCase):
    def test_round35_depths(self):
        self.assertEqual(harness.depth(43, 30, 256), 2)
        self.assertEqual(harness.depth(43, 30, 512), 1)
        self.assertEqual(harness.depth(58, 71, 256), 3)
        self.assertEqual(harness.depth(58, 71, 512), 2)

    def test_area_for_every_top(self):
        for w, h in ((43, 30), (58, 71), (17, 33), (1, 1), (255, 1), (3, 500), (102, 140), (33, 33)):
            for level in range(harness.depth(w, h, 1)+1):
                self.assertEqual(harness.depth(w, h, harness.min_area_for_top(w, h, level)), level, (w, h, level))

    def test_jobs_have_no_truth_and_the_round35_search(self):
        item = case('a_clean', 'a', truth(3))
        job = harness.job(item, dict(harness.ORIGINAL), 1, 3)
        self.assertNotIn('truth', json.dumps(job))
        self.assertEqual(job['method'], 'geometry_only')
        self.assertEqual(job['search']['stop_layer'], 1)
        self.assertEqual(job['search']['min_area'], 256)

    def test_plans(self):
        cases = [case('a_clean', 'a', truth(3), split='development'), case('a_noise', 'a', truth(3), 'noise', 'development'),
                 case('a_empty', 'a', [], 'empty', 'development')]
        sweep = harness.phase_plan('dev_sweep', cases)
        self.assertEqual([arm for _, arm, _, _ in sweep], ['depth0', 'depth1', 'depth2', 'rule_report', 'depth0', 'depth1', 'depth2'])
        self.assertEqual(sweep[2][2]['min_area'], 256)
        formal = harness.phase_plan('formal', [dict(c, split='holdout') for c in cases], protocol=PROTOCOL)
        self.assertEqual(len(formal), 15)
        self.assertEqual({rows for c, _, _, rows in formal if not c['positive']}, {1})
        self.assertEqual({rows for c, _, _, rows in formal if c['positive']}, {3})
        adaptive = [search for _, arm, search, _ in formal if arm == 'adaptive'][0]
        self.assertEqual((adaptive['pyramid_rule'], adaptive['pyramid_rule_min_score'], adaptive['min_area']), (1, .5, 256))
        budget = [search for _, arm, search, _ in formal if arm == 'budget'][0]
        self.assertEqual((budget['proposal_score'], budget['limit'], budget['min_area']), (.3, 300, 256))

    def test_split_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'s.png'
            source.write_bytes(b'x')
            item = dict(case('a_clean', 'a', truth(2)), source=str(source), template=str(source),
                        input_sha256=dict(source=scenes.sha(source), template=scenes.sha(source)))
            scenes.save(root/'holdout.json', [item])
            with self.assertRaisesRegex(ValueError, 'development cases only'):
                harness.run_phase('dev_sweep', root/'holdout.json', root, root/'a'/'manifest.json')
            scenes.save(root/'dev.json', [dict(item, split='development')])
            with self.assertRaisesRegex(ValueError, 'frozen --protocol'):
                harness.run_phase('formal', root/'dev.json', root, root/'b'/'manifest.json')
            with self.assertRaisesRegex(ValueError, 'holdout cases only'):
                scenes.save(root/'p.json', PROTOCOL)
                harness.run_phase('formal', root/'dev.json', root, root/'c'/'manifest.json', protocol_path=root/'p.json')
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                harness.run_phase('dev_sweep', root/'dev.json', root, root/'d'/'manifest.json')


class Locks(unittest.TestCase):
    def test_waits_for_foreign_locks_only(self):
        with tempfile.TemporaryDirectory() as directory:
            notes, saved = Path(directory), harness.NOTES
            harness.NOTES = notes
            try:
                (notes/'a').mkdir()
                (notes/'b').mkdir()
                own, other = notes/'a'/'timing-lock-round49-formal.txt', notes/'b'/'timing-lock-other.txt'
                own.write_text('mine')
                self.assertEqual(harness.wait_for_locks(own, sleep=lambda s: self.fail('own lock waited')), [])
                other.write_text('theirs')
                calls = []
                waits = harness.wait_for_locks(own, sleep=lambda s: (calls.append(s), other.unlink()))
                self.assertEqual(len(waits), 1)
                self.assertEqual(calls, [harness.LOCK_POLL_SECONDS])
                self.assertIn(str(other), waits[0]['locks'])
            finally:
                harness.NOTES = saved


class Offline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_tune_needed_depth_and_threshold(self):
        objects = truth(10)
        cases = [case('fine_clean', 'fine', objects, split='development'), case('coarse_clean', 'coarse', objects, split='development')]
        fake = Fake(self.root, cases)
        # fine: depth2 misses 8, depth1 and depth0 find all -> needed 1. coarse: every depth finds all -> needed 2.
        for level, missing, ms in ((0, 0, 50.), (1, 0, 20.), (2, 8, 5.)):
            fake.add(cases[0], f'depth{level}', found(objects, missing), ms)
            fake.add(cases[1], f'depth{level}', found(objects), ms)
        fake.add(cases[0], 'rule_report', [], 1., rule=rule(2, 2, [.9, .6, .3]))
        fake.add(cases[1], 'rule_report', [], 1., rule=rule(2, 2, [.95, .9, .7]))
        tuned = harness.tune(fake.manifest(), self.root)
        self.assertEqual(tuned['templates']['fine']['needed_top'], 1)
        self.assertEqual(tuned['templates']['coarse']['needed_top'], 2)
        # .30 keeps fine at depth 2 (a miss); .35-.60 start fine at 1 and keep coarse at 2 (25 ms);
        # .65-.70 start fine at 0 (55 ms); .75+ also moves coarse higher. The smallest fastest wins.
        self.assertEqual(tuned['selected_min_score'], .35)
        by = {row['min_score']: row for row in tuned['candidates']}
        self.assertEqual(by[.3]['misses'], ['fine'])
        self.assertEqual(by[.6]['dev_search_ms'], 25.)
        self.assertEqual(by[.65]['dev_search_ms'], 55.)
        self.assertEqual(by[.8]['shallower_than_needed'], ['coarse', 'fine'])
        self.assertEqual(tuned['oracle_tp'], 20)
        self.assertEqual(tuned['original_tp'], 12)

    def test_tune_rejects_altered_workers(self):
        objects = truth(2)
        cases = [case('a_clean', 'a', objects, split='development')]
        fake = Fake(self.root, cases)
        fake.add(cases[0], 'depth0', found(objects), 1.)
        fake.add(cases[0], 'rule_report', [], 1., rule=rule(0, 0, [.9]))
        (self.root/fake.executions[0]['result']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Altered worker'):
            harness.tune(fake.manifest(), self.root)
        manifest = fake.manifest()
        scenes.save(fake.cases_path, [])
        with self.assertRaisesRegex(ValueError, 'Cases file changed'):
            harness.load_rows(manifest, self.root)

    def test_calibrate_picks_the_largest_recipe_within_budget(self):
        objects = truth(4)
        cases = [case('a_clean', 'a', objects, split='development')]
        fake = Fake(self.root, cases)
        fake.add(cases[0], 'original', found(objects, 2), 10.)
        fake.add(cases[0], 'adaptive', found(objects), 30.)
        for k, (p, l) in enumerate(harness.BUDGET_GRID):
            fake.add(cases[0], f'budget_{p:.2f}_{l}', found(objects, 1), 12.+6*k)
        out = harness.calibrate(fake.manifest(), self.root)
        self.assertEqual(out['selected'], dict(proposal_score=.3, limit=300))  # 30 ms <= 30 ms < 36 ms
        self.assertTrue(out['selected_within_budget'])
        fake2 = Fake(self.root/'two', cases)
        fake2.add(cases[0], 'original', found(objects), 10.)
        fake2.add(cases[0], 'adaptive', found(objects), 5.)
        for k, (p, l) in enumerate(harness.BUDGET_GRID):
            fake2.add(cases[0], f'budget_{p:.2f}_{l}', found(objects), 12.+6*k)
        out = harness.calibrate(fake2.manifest(), self.root/'two')
        self.assertEqual(out['selected'], dict(proposal_score=.45, limit=150))
        self.assertFalse(out['selected_within_budget'])

    def formal(self, adaptive_missing=0, keep_changed=False, negative_fp=0, slow=1.):
        objects = truth(10)
        cases = [case('small_clean', 'small', objects), case('small_empty', 'small', [], 'empty'),
                 case('keep_clean', 'keep', objects), case('keep_empty', 'keep', [], 'empty')]
        fake = Fake(self.root, cases)
        times = dict(original=10., stage35_manual=40., adaptive=20., budget=19., fixed512=25.)
        for arm in harness.FORMAL_ARMS:
            small_missing = dict(original=6, adaptive=adaptive_missing).get(arm, 0)
            fake.add(cases[0], arm, found(objects, small_missing), times[arm], rule=rule(2, 1, [.9, .7, .2]) if arm == 'adaptive' else None)
            fake.add(cases[1], arm, found([], extra=negative_fp if arm == 'adaptive' else 0), 1.,
                     rule=rule(2, 1, [.9, .7, .2]) if arm == 'adaptive' else None)
            keep = found(objects, 1 if (keep_changed and arm == 'adaptive') else 0)
            fake.add(cases[2], arm, keep, 10.*(slow if arm == 'adaptive' else 1),
                     rule=rule(3, 3, [.9, .9, .8, .7]) if arm == 'adaptive' else None)
            fake.add(cases[3], arm, [], 1., rule=rule(3, 3, [.9, .9, .8, .7]) if arm == 'adaptive' else None)
        return harness.summarize(fake.manifest(), self.root, PROTOCOL)

    def test_verdicts_pass(self):
        out = self.formal()
        self.assertEqual(out['deficient'], ['small'])
        self.assertEqual(out['sufficient'], ['keep'])
        for name in ('H1', 'H2', 'H2r', 'H5', 'H6'):
            self.assertTrue(out['verdicts'][name]['passed'], name)
        self.assertTrue(out['verdicts']['H3']['passed'])  # equal TP, 30 vs 29 ms is within 1.10
        self.assertTrue(out['verdicts']['H4']['passed'])  # 30 vs 50 ms

    def test_verdicts_fail(self):
        self.assertFalse(self.formal(adaptive_missing=3)['verdicts']['H1']['passed'])
        changed = self.formal(keep_changed=True)['verdicts']
        self.assertFalse(changed['H2']['passed'])
        self.assertFalse(self.formal(negative_fp=1)['verdicts']['H5']['passed'])
        self.assertFalse(self.formal(slow=1.2)['verdicts']['H2r']['passed'])

    def test_h3_fails_when_the_budget_finds_more(self):
        out = self.formal(adaptive_missing=1)
        self.assertTrue(out['verdicts']['H1']['passed'])  # 5 of 6 recovered >= 4.8
        self.assertFalse(out['verdicts']['H3']['passed'])


class Scenes(unittest.TestCase):
    def test_holdout_needs_the_freezing_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'fsc.json'
            manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'frozen protocol'):
                scenes.generate('holdout', manifest, root, root/'out', root)
            scenes.save(root/'protocol.json', dict(scenes=dict(generator_sha256='0'*64)))
            with self.assertRaisesRegex(ValueError, 'does not freeze'):
                scenes.generate('holdout', manifest, root, root/'out2', root, root/'protocol.json')

    def test_layout_and_textures_are_deterministic(self):
        self.assertEqual(scenes.layout(43, 30, 5), scenes.layout(43, 30, 5))
        size, background, objects = scenes.layout(43, 30, 5)
        self.assertEqual(len(objects), 49)
        self.assertTrue(60 <= background <= 190)
        self.assertTrue(all(0 < o['pos_x'] < size and 0 < o['pos_y'] < size for o in objects))
        a, b = scenes.procedural(7, 2.), scenes.procedural(7, 2.)
        self.assertTrue((a == b).all())
        self.assertAlmostEqual(float(a.mean()), 128, delta=2)
        self.assertEqual(scenes.procedural(7, 2., (20, 10)).shape, (10, 20))

    def test_splits_do_not_share_sources(self):
        self.assertFalse(set(scenes.FSC_IMAGES['development']) & set(scenes.FSC_IMAGES['holdout']))
        self.assertFalse(set(scenes.PROCEDURAL_SEEDS['development']) & set(scenes.PROCEDURAL_SEEDS['holdout']))
        self.assertFalse(set(scenes.TEST_IMAGES['development']) & set(scenes.TEST_IMAGES['holdout']))
        self.assertIn('7417', scenes.FSC_IMAGES['holdout'])
        self.assertIn('heldout_clean', scenes.STAGE35_CASES['holdout'])


if __name__ == '__main__':
    unittest.main()
