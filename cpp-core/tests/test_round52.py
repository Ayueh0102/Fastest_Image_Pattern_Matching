"""Round 52 driver and holdout generator: selection rule, border split, edge truth and hypothesis directions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import round52_compare as r52  # noqa: E402


def neg(method, fp, border):
    return dict(method=method, positive=False, false_positives=fp, border_fp=border, interior_fp=fp - border, case='n', time_ms=10.)


def pos(method, gt, tp, edge_gt=0, edge_tp=0, case='p', time_ms=10.):
    return dict(method=method, positive=True, gt=gt, strict_tp=tp, edge_gt=edge_gt, edge_tp=edge_tp, extra=0, case=case, time_ms=time_ms)


class SelectionTest(unittest.TestCase):
    def rows(self, grid):
        rows = [pos('ours_hybrid_k3-border', 1000, 990), neg('ours_hybrid_k3-border', 100, 0)]
        for a0, (tp, fp) in grid.items():
            rows += [pos(f'ours_hybrid_k3-both{a0}', 1000, tp), neg(f'ours_hybrid_k3-both{a0}', fp, 0)]
        return rows

    def test_fewest_false_positives_within_the_loss_bound(self):
        grid = {256: (990, 60), 400: (989, 40), 576: (986, 20), 784: (984, 10), 1024: (980, 5), 1600: (970, 0)}
        s = r52.select_a0(self.rows(grid))
        self.assertEqual(s['chosen'], 576)  # 784 loses 6 > 0.5% of 1000; 576 loses 4
        self.assertEqual([g['eligible'] for g in s['grid']], [True, True, True, False, False, False])

    def test_ties_go_to_the_smaller_area_and_none_means_off(self):
        grid = {a0: (990, 7) for a0 in r52.A0_GRID}
        self.assertEqual(r52.select_a0(self.rows(grid))['chosen'], 256)
        grid = {a0: (900, 0) for a0 in r52.A0_GRID}
        self.assertEqual(r52.select_a0(self.rows(grid))['chosen'], 0)

    def test_variants(self):
        self.assertEqual(set(r52.variants(0)), {'off', 'border'})
        v = r52.variants(784)
        self.assertEqual(v['both'], dict(border_min_inside=.5, small_template_area=784.))
        self.assertEqual(v['size'], dict(small_template_area=784.))


class GeometryTest(unittest.TestCase):
    def test_border_touch_uses_pixel_centres(self):
        inside = dict(kind='pose', cx=10, cy=10, angle_deg=0, width=20, height=20, corners=[[0, 0], [19, 0], [19, 19], [0, 19]])
        self.assertFalse(r52.touches_border(inside, 20, 20))
        self.assertTrue(r52.touches_border(dict(inside, corners=[[-.5, 0], [19, 0], [19, 19], [0, 19]]), 20, 20))
        self.assertTrue(r52.touches_border(dict(kind='pose', cx=5, cy=5, angle_deg=30, width=20, height=10), 40, 40))

    def test_boundary_scene_crosses_the_edges_with_centres_inside(self):
        import numpy as np
        import round52_scenes as gen
        pixels = (np.arange(16 * 12, dtype=np.uint8).reshape(12, 16) * 7) % 251
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'templates').mkdir()
            a = gen.boundary_case('t', pixels, 52999, out, 'procedural', {})
            image = (out / 'scenes' / 't-boundary.png').read_bytes()
            b = gen.boundary_case('t', pixels, 52999, out, 'procedural', {})
            self.assertEqual(image, (out / 'scenes' / 't-boundary.png').read_bytes())  # seeded
        self.assertEqual(a, b)
        side = a['source_size'][0]
        self.assertEqual(len(a['truth']), 49)
        for t in a['truth']:
            self.assertTrue(0 <= t['pos_x'] <= side - 1 and 0 <= t['pos_y'] <= side - 1)
        first = [t for t in a['truth'] if t['index'] % 7 == 0 or t['index'] < 7]
        self.assertGreater(sum(t['edge'] for t in first), 6)  # most of the first row and column cross an edge
        self.assertFalse(any(t['edge'] for t in a['truth'] if t not in first))
        self.assertEqual(a['edge_count'], sum(t['edge'] for t in a['truth']))

    def test_generator_needs_the_protocol_naming_it(self):
        import round52_scenes as gen
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp) / 'protocol.json'
            protocol.write_text(json.dumps(dict(generators={'round52_scenes.py': 'not this file'})))
            sys_argv = sys.argv
            sys.argv = ['round52_scenes.py', '--output', str(Path(tmp) / 'out'), '--protocol', str(protocol)]
            try:
                with self.assertRaises(SystemExit):
                    gen.main()
            finally:
                sys.argv = sys_argv


class HypothesisTest(unittest.TestCase):
    def holdout(self, border_fp_after, interior_after, tp_border, tp_both, edge_after=3, t_both=10.4):
        rows = []
        for b in r52.BACKENDS:
            rows += [neg(f'{b}-off', 120, 100), neg(f'{b}-border', border_fp_after + 20, border_fp_after),
                     dict(neg(f'{b}-both', interior_after, 0), time_ms=t_both),
                     pos(f'{b}-off', 1000, 990, 40, 3), pos(f'{b}-border', 1000, tp_border, 40, edge_after),
                     pos(f'{b}-both', 1000, tp_both, 40, edge_after, time_ms=t_both)]
        return rows

    def test_directions_and_bounds(self):
        h = r52.hypotheses(self.holdout(10, 9, 989, 985), None, 784)['ours_hybrid_k3']
        self.assertTrue(h['H1']['pass_'])  # 100 border FP -> 10
        self.assertTrue(h['H2']['pass_'])  # 990 -> 989 is within 0.2% of 1000
        self.assertTrue(h['H3']['pass_'])  # 20 interior FP -> 9
        self.assertTrue(h['H4']['pass_'])  # 989 -> 985 within 0.5%
        self.assertTrue(h['H6']['pass_'])  # 4% slower
        h = r52.hypotheses(self.holdout(11, 11, 987, 983, edge_after=1, t_both=10.6), None, 784)['ours_fullgpu_k3']
        self.assertFalse(h['H1']['pass_'])
        self.assertFalse(h['H2']['pass_'])  # loses 3 > 2 truths and 2 edge truths
        self.assertFalse(h['H3']['pass_'])
        self.assertTrue(h['H4']['pass_'])
        self.assertFalse(h['H6']['pass_'])
        self.assertNotIn('H3', r52.hypotheses(self.holdout(10, 9, 989, 985), None, 0)['ours_hybrid_k3'])

    def test_too_few_false_positives_are_not_judged(self):
        rows = [neg(f'{b}-{v}', 5, 5) for b in r52.BACKENDS for v in ('off', 'border', 'both')]
        rows += [pos(f'{b}-{v}', 10, 10) for b in r52.BACKENDS for v in ('off', 'border', 'both')]
        h = r52.hypotheses(rows, None, 784)['ours_hybrid_k3']
        self.assertIsNone(h['H1']['pass_'])
        self.assertIsNone(h['H3']['pass_'])

    def test_regression_flags_any_case_that_gets_worse(self):
        reg = []
        for b in r52.BACKENDS:
            reg += [pos(f'{b}-off', 10, 10, case='a'), pos(f'{b}-both', 10, 10, case='a'),
                    dict(neg(f'{b}-off', 0, 0), case='n'), dict(neg(f'{b}-both', 1, 1), case='n')]
        h = r52.hypotheses(self.holdout(10, 9, 989, 985), reg, 784)
        self.assertEqual(h['ours_hybrid_k3']['H5']['worse'], ['n'])
        self.assertFalse(h['ours_hybrid_k3']['H5']['pass_'])


if __name__ == '__main__':
    unittest.main()
