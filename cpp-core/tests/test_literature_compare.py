"""Negative controls for the round-50 comparison driver: rules, budgets and truth isolation."""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import literature_compare as lc  # noqa: E402


def pose(cx, cy, angle, w=40., h=20.):
    return dict(kind='pose', cx=cx, cy=cy, angle_deg=angle, width=w, height=h, score=1.)


def case(name='c', positive=True, scales=(1.,), truth=None, family=None, size=(40, 20)):
    return dict(name=name, group='C', family=family, positive=positive, source='s.png', template='t.png', template_size=list(size),
                truth=truth if truth is not None else ([pose(10, 10, 0)] if positive else []),
                search=dict(scales=list(scales), angle=180, limit=5, score=.8, proposal_score=.5, overlap=.2, stop_layer=1, subpixel=False))


class ScaleGridTest(unittest.TestCase):
    def test_b1_grid_ratio_and_counts(self):
        for setting, n in (('S1', 4), ('S1_5', 18), ('S2', 37), ('S2_5', 48)):
            grid = lc.b1_scales(setting, 't36')
            self.assertEqual(len(grid), n, setting)
            self.assertTrue(all(b / a <= 1.04 + 1e-12 for a, b in zip(grid, grid[1:])))
        self.assertAlmostEqual(lc.b1_scales('S2_5', 't36')[0], .4)
        self.assertAlmostEqual(lc.b1_scales('S2_5', 't36')[-1], 2.5)
        self.assertEqual(lc.b1_scales('S2', 'native'), [1.0])
        self.assertTrue(all(b / a <= 1.10 + 1e-12 for a, b in zip(lc.b1_opencv_scales('S2_5'), lc.b1_opencv_scales('S2_5')[1:])))


class SelectionRuleTest(unittest.TestCase):
    def test_paper_distance_prefers_closer_and_rejects_empty(self):
        exact = dict(successes=10, miou=.955, succ=99.6, loc_err=.44, rot_err=1.03)
        self.assertAlmostEqual(lc.paper_distance(exact), 0.)
        worse = dict(exact, rot_err=2.06)
        self.assertAlmostEqual(lc.paper_distance(worse), 1.)
        self.assertEqual(lc.paper_distance(dict(successes=0)), math.inf)

    def test_paper_metrics_follow_test_single(self):
        truth = [dict(index=0, cx=50, cy=50, scale_x=1., scale_y=1., angle=10.), dict(index=1, cx=80, cy=80, scale_x=1., scale_y=1., angle=0.)]
        outputs = {0: [dict(cx=50, cy=50, width=36, height=36, angle_deg=10., score=.9)],
                   1: [dict(cx=80, cy=80, width=36, height=36, angle_deg=0., score=.005)]}  # score <= 0.01 fails
        m = lc.b1_paper_metrics(truth, outputs)
        self.assertEqual((m['samples'], m['successes'], m['succ']), (2, 1, 50.))
        self.assertAlmostEqual(m['miou'], 1.)
        self.assertAlmostEqual(m['loc_err'], 0.)


class ApplicabilityTest(unittest.TestCase):
    def test_core_only_on_fixed_scale(self):
        self.assertTrue(lc.applicable(case(), 'ours_core'))
        self.assertFalse(lc.applicable(case(truth=[pose(10, 10, 0, 44., 22.)]), 'ours_core'))
        self.assertTrue(lc.applicable(case(positive=False), 'ours_core'))
        self.assertFalse(lc.applicable(case(positive=False, scales=(.75, 1., 1.25)), 'ours_core'))

    def test_geco2_needs_in_image_exemplars_and_shallow_is_tiny_only(self):
        self.assertFalse(lc.applicable(case(positive=False), 'geco2_mps'))
        self.assertTrue(lc.applicable(case(), 'geco2_mps'))
        self.assertFalse(lc.applicable(case(), 'ours_small_shallow'))
        self.assertTrue(lc.applicable(case(family='tiny'), 'ours_small_shallow'))

    def test_exemplars_are_first_three_enclosing_boxes(self):
        c = case(truth=[pose(10 * k, 10, 0) for k in range(1, 6)])
        ex = lc.exemplars_for(c)
        self.assertEqual(len(ex), 3)
        self.assertEqual(ex[0], [-10.0, 0.0, 30.0, 20.0])


class TruthIsolationTest(unittest.TestCase):
    def test_runner_job_carries_no_truth_and_reuses_round44_only_in_a(self):
        job, reused = lc.runner_job(case(), 'ours_fullgpu_k3', 3, 1.)
        self.assertFalse(reused)
        self.assertNotIn('truth', json.dumps(job))
        self.assertEqual(job['candidate_dedup_top_k'], 3)
        self.assertEqual((job['full_gpu_peak_mode'], job['full_gpu_fine_mode']), (2, 4))
        shallow, _ = lc.runner_job(case(family='tiny'), 'ours_hybrid_k0', 3, 1., shallow=True)
        self.assertEqual((shallow['min_area'], shallow['proposal_score'], shallow['limit']), (512, .25, 300))

    def test_no_arm_enables_pyramid_rule(self):
        for arm in lc.ARMS:
            job, _ = lc.runner_job(case(), arm, 1, 0.)
            self.assertNotIn('pyramid_rule', job)

    def test_worker_commands_never_receive_truth(self):
        c = case()
        for cmd in (lc.tdcm_cmd(c, Path('o.json'), 'cpu', 'test_py', .1), lc.opencv_cmd(c, Path('o.json')), lc.ours_core_cmd(c, Path('b'))):
            self.assertNotIn('truth', ' '.join(map(str, cmd)))
        self.assertIn('--adapt', lc.tdcm_cmd(c, Path('o.json'), 'cpu', 'test_py', .1))


class ScoringTest(unittest.TestCase):
    def test_negative_case_counts_every_detection(self):
        s = lc.case_score(case(positive=False), [pose(1, 1, 0), pose(5, 5, 0)])
        self.assertEqual(s['false_positives'], 2)

    def test_hypotheses_h4_and_h5(self):
        a_cases = {'many128_pose': dict(positive=True), 'many512_pose': dict(positive=True), 'empty_scene': dict(positive=False)}

        def row(n, m, t=None, fp=None, gt=0, tp=0):
            s = dict(false_positives=fp) if fp is not None else dict(gt=gt, strict_tp=tp, association_tp=tp, boxes=dict(tp=tp))
            return dict(case=n, method=m, score=s, time_ms=t)
        table = [row('many128_pose', 'ours_fullgpu_k3', 100, gt=4, tp=4), row('many128_pose', 'tdcm_adapted_cpu', 900, gt=4, tp=1),
                 row('many512_pose', 'ours_fullgpu_k3', 300, gt=4, tp=4), row('many512_pose', 'tdcm_adapted_cpu', 200, gt=4, tp=1),
                 row('empty_scene', 'ours_hybrid_k0', fp=0), row('empty_scene', 'ours_fullgpu_k3', fp=1)]
        c_cases = {'many8k_512': dict(positive=True)}
        c_table = [row('many8k_512', 'ours_fullgpu_k3', 50, gt=4, tp=4), row('many8k_512', 'tdcm_adapted_cpu', 500, gt=4, tp=0)]
        h = lc.hypotheses((a_cases, table, [], {}), (c_cases, c_table, [], {}), None, 'geco2_cuda')
        self.assertFalse(h['H4']['pass_'])  # many512: 300 ms is not below 200 ms
        self.assertEqual([r['ok'] for r in h['H4']['rows']], [True, False, True])
        self.assertFalse(h['H5']['pass_'])  # one ours false positive fails H5
        self.assertTrue(h['H1']['a_pass'])
        self.assertIsNone(h['H1']['pass'])  # GeCo2 accuracy outputs missing -> not judged


class WorkerDirectoryTest(unittest.TestCase):
    def test_output_directory_exists_before_the_worker_starts(self):
        # Attempt 1 of the threshold selection lost every first case: the worker wrote --output into a missing directory.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new' / 'dir' / 'o.json'
            code = 'import sys, pathlib; pathlib.Path(sys.argv[1]).write_text("{}")'
            w = lc.run_worker([sys.executable, '-c', code, out], out)
            self.assertEqual(w['code'], 0)
            self.assertEqual(out.read_text(), '{}')


class HeavyProcessTest(unittest.TestCase):
    def test_formal_lock_sees_our_own_workers_but_not_clangd(self):
        from unittest import mock
        ps = ('101 /usr/bin/python -I literature_tdcm_worker.py --batch x\n'
              '102 /Applications/Xcode.app/.../clangd --background-index\n'
              '103 /bin/zsh\n'
              '104 python literature_compare.py run --group A --formal\n')
        with mock.patch.object(lc.subprocess, 'run', return_value=mock.Mock(stdout=ps)):
            rows = lc.heavy_processes()
        self.assertEqual(len(rows), 1)
        self.assertIn('literature_tdcm_worker', rows[0])


class MethodNameTest(unittest.TestCase):
    def test_unknown_or_joined_method_names_are_refused(self):
        with self.assertRaises(SystemExit):
            lc.check_methods('A', ['ours_core ours_hybrid_k0'])
        with self.assertRaises(SystemExit):
            lc.check_methods('B1', ['ours_core'])
        lc.check_methods('C', ['ours_core', 'tdcm_adapted_cpu'])
        lc.check_methods('B2', ['ours', 'geco2_cpu'])


if __name__ == '__main__':
    unittest.main()
