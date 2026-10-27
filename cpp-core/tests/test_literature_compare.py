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

    def test_paper_success_needs_only_iou_above_a_tenth(self):
        truth = [dict(index=0, cx=50, cy=50, scale_x=1., scale_y=1., angle=0.)]
        shifted = {0: [dict(cx=68, cy=50, width=36, height=36, angle_deg=0., score=.9)]}  # IoU 1/3
        self.assertEqual(lc.b1_paper_metrics(truth, shifted)['successes'], 1)
        far = {0: [dict(cx=83, cy=50, width=36, height=36, angle_deg=0., score=.9)]}  # IoU 108/2484
        self.assertEqual(lc.b1_paper_metrics(truth, far)['successes'], 0)


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
        pairs = {('many128_pose', 'ours_fullgpu_k3'): [dict(position_px=.1, angle_deg=.1)],
                 ('many128_pose', 'tdcm_adapted_cpu'): [dict(position_px=1., angle_deg=1.)]}
        h = lc.hypotheses((a_cases, table, [], pairs), (c_cases, c_table, [], {}), None, 'geco2_cuda')
        self.assertFalse(h['H4']['pass_'])  # many512: 300 ms is not below 200 ms
        self.assertEqual([r['ok'] for r in h['H4']['rows']], [True, False, True])
        self.assertFalse(h['H5']['pass_'])  # one ours false positive fails H5
        self.assertTrue(h['H1']['a_pass'])
        self.assertIsNone(h['H1']['pass'])  # (a) and (b) pass, GeCo2 accuracy outputs missing -> not judged
        table.append(row('many128_pose', 'opencv', gt=4, tp=4))
        table[-1]['complete'] = True
        table[1]['score']['strict_tp'] = 1
        table[0]['score']['strict_tp'] = 0  # ours below a competitor: H1 fails without waiting for GeCo2
        h = lc.hypotheses((a_cases, table, [], pairs), None, None, 'geco2_cuda')
        self.assertFalse(h['H1']['a_pass'])
        self.assertIs(h['H1']['pass'], False)

    def test_h1_and_h5_rules_follow_the_protocol(self):
        def row(n, m, tp=None, fp=None, complete=None):
            s = dict(false_positives=fp) if fp is not None else dict(gt=4, strict_tp=tp, association_tp=tp, boxes=dict(tp=tp))
            r = dict(case=n, method=m, score=s, time_ms=None)
            if complete is not None:
                r['complete'] = complete
            return r
        cases = {'p': dict(positive=True), 'q': dict(positive=True), 'neg': dict(positive=False)}
        pairs = {('p', 'ours_fullgpu_k3'): [dict(position_px=.1, angle_deg=.1)], ('p', 'tdcm_adapted_cpu'): [dict(position_px=1., angle_deg=1.)]}
        base = [row('p', 'ours_fullgpu_k3', 4), row('p', 'tdcm_adapted_cpu', 1), row('q', 'ours_fullgpu_k3', 2), row('q', 'tdcm_adapted_cpu', 2)]

        def judge(rows, pairs=pairs):
            return lc.hypotheses((cases, rows, [], pairs), None, None, 'geco2_cuda')
        # (a): a completed OpenCV search above ours fails that case; an incomplete one is not compared.
        self.assertFalse(judge(base + [row('q', 'opencv', 3, complete=True)])['H1']['a_pass'])
        self.assertTrue(judge(base + [row('q', 'opencv', 3, complete=False)])['H1']['a_pass'])
        # (b): the median position and the median angle error must both be below TDCM-adapted's.
        h = judge(base, {**pairs, ('p', 'ours_fullgpu_k3'): [dict(position_px=.1, angle_deg=2.)]})
        self.assertFalse(h['H1']['b']['pass_'])
        self.assertIs(h['H1']['pass'], False)
        # H5 judges our arms only; competitors' false positives are reported beside them.
        h = judge(base + [row('neg', 'ours_fullgpu_k3', fp=0), row('neg', 'tdcm_adapted_cpu', fp=5)])
        self.assertTrue(h['H5']['pass_'])
        self.assertEqual(len(h['H5']['rows']), 2)

    def test_h3_and_h4_directions(self):
        tiny = {'t1': dict(positive=True, family='tiny'), 't2': dict(positive=False, family='tiny')}

        def tiny_table(tdcm_tp):
            return [dict(case='t1', method='ours_fullgpu_k3', score=dict(gt=10, strict_tp=8, association_tp=10)),
                    dict(case='t1', method='tdcm_adapted_cpu', score=dict(gt=10, strict_tp=tdcm_tp, association_tp=10))]
        self.assertFalse(lc.hypotheses(None, (tiny, tiny_table(7), [], {}), None, 'geco2_cuda')['H3']['pass_'])
        self.assertTrue(lc.hypotheses(None, (tiny, tiny_table(8), [], {}), None, 'geco2_cuda')['H3']['pass_'])

        def timed(n, ours, tdcm):
            return [dict(case=n, method='ours_fullgpu_k3', time_ms=ours), dict(case=n, method='tdcm_adapted_cpu', time_ms=tdcm)]
        a = ({}, timed('many128_pose', 1, 2) + timed('many512_pose', 1, 2), [], {})
        h = lc.hypotheses(a, ({}, timed('many8k_512', 5, 5), [], {}), None, 'geco2_cuda')
        self.assertEqual([r['ok'] for r in h['H4']['rows']], [True, True, False])  # a tie is not faster
        self.assertFalse(h['H4']['pass_'])
        self.assertFalse(lc.hypotheses(a, None, None, 'geco2_cuda')['H4']['pass_'])  # two of the three scenes cannot pass


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


FAKE_WORKER = '''import json, sys
a = sys.argv[1:]
out = a[a.index('--output') + 1]
if '--batch' in a:
    open(out, 'w').write(json.dumps(dict(header={})) + '\\n' + json.dumps(dict(footer={})) + '\\n')
    sys.exit(0)
if 'bad' in a[a.index('--image') + 1]:
    sys.exit('boom')
open(out, 'w').write(json.dumps(dict(detections=[dict(kind='box', x0=0, y0=0, x1=4, y1=4, score=.9)], exact_repeat=True,
                                     memory=dict(peak_rss_bytes=1), timing_ms=dict(call_median=5., calls=[5.], load_model=9.))))
'''


class GeCo2CudaPackageTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        image = self.root / 'src.png'
        image.write_bytes(b'png')
        for group, names in (('A', ['good', 'neg']), ('C', ['bad'])):
            d = self.root / 'runs' / group
            d.mkdir(parents=True)
            cases = [dict(case(n, positive=n != 'neg'), group=group, source=str(image)) for n in names]
            (d / 'cases.json').write_text(json.dumps(cases))
        (self.root / 'runs/B2').mkdir()
        (self.root / 'runs/B2/items.json').write_text(json.dumps([dict(id='2.jpg', image=str(image), exemplars=[[0, 0, 5, 5]])]))
        self.runs = {g: self.root / 'runs' / g for g in ('A', 'C', 'B2')}

    def tearDown(self):
        self.tmp.cleanup()

    def export(self):
        from argparse import Namespace
        pkg = self.root / 'pkg'
        lc.export_geco2_cuda(Namespace(a=self.runs['A'], c=self.runs['C'], b2=self.runs['B2'], output=pkg))
        return pkg

    def test_export_sends_positive_cases_without_truth(self):
        pkg = self.export()
        jobs = json.loads((pkg / 'jobs.json').read_text())
        self.assertEqual({k: [j.get('case', j.get('id')) for j in v] for k, v in jobs.items()}, dict(A=['good'], C=['bad'], B2=['2.jpg']))
        self.assertNotIn('truth', (pkg / 'jobs.json').read_text())
        self.assertEqual(jobs['A'][0]['exemplars'], lc.exemplars_for(case('good')))
        files = json.loads((pkg / 'package.json').read_text())['files']
        self.assertIn('tools/literature_geco2_cuda_run.py', files)
        self.assertEqual(files['images/A/good.png'], lc.sha(pkg / 'images/A/good.png'))
        with self.assertRaises(SystemExit):  # never into an existing package
            self.export()

    def test_runner_keeps_failures_and_skips_them_on_restart_and_ingest_is_idempotent(self):
        pkg = self.export()
        (pkg / 'tools/literature_geco2_worker.py').write_text(FAKE_WORKER)
        out, work = self.root / 'results', self.root / 'work'
        runner = [sys.executable, '-I', pkg / 'tools/literature_geco2_cuda_run.py', '--package', pkg, '--work', work, '--output', out]
        import subprocess
        for _ in range(2):
            subprocess.run([str(c) for c in runner], check=True, capture_output=True)
        rows = lc.read_manifest(out / 'manifest.jsonl')
        self.assertEqual(sorted((r['group'], r['case'], r['status']) for r in rows),
                         [('A', 'good', 'ok'), ('B2', 'all', 'ok'), ('C', 'bad', 'failed')])
        self.assertIn('boom', rows[[r['case'] for r in rows].index('bad')]['stderr_tail'])
        self.assertEqual(len(lc.read_manifest(out / 'attempts.jsonl')), 3)
        from argparse import Namespace
        for _ in range(2):
            lc.ingest_geco2_cuda(Namespace(results=out, a=self.runs['A'], c=self.runs['C'], b2=self.runs['B2']))
        a = lc.read_manifest(self.runs['A'] / 'manifest.jsonl')
        self.assertEqual(len(a), 1)
        self.assertEqual((a[0]['method'], a[0]['status'], a[0]['time_ms'], a[0]['detections']), ('geco2_cuda', 'ok', 5., 1))
        self.assertEqual(lc.read_manifest(self.runs['C'] / 'manifest.jsonl')[0]['status'], 'failed')
        self.assertTrue((self.runs['B2'] / 'geco2_cuda.jsonl').exists())
        table = lc.summarize_scenes(self.runs['A'])[1]
        self.assertEqual([(r['method'], r['ok_blocks']) for r in table], [('geco2_cuda', 1)])

    def test_windows_host_without_load_average_still_summarises(self):
        d = self.runs['A']
        out = d / 'geco2_cuda' / 'good-b0.json'
        out.parent.mkdir()
        out.write_text(json.dumps(dict(detections=[], exact_repeat=True, memory={}, timing_ms=dict(call_median=1., calls=[1.], load_model=1.))))
        row = dict(case='good', method='geco2_cuda', block=0, status='ok', output=str(out), time_ms=1.,
                   host=dict(load1=None, cpu_load_percent=3.))
        (d / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
        self.assertIsNone(lc.summarize_scenes(d)[1][0]['load1_median'])


if __name__ == '__main__':
    unittest.main()
