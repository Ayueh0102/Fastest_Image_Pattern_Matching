import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import literature_score as score  # noqa: E402


def pose(cx, cy, angle, width=40., height=20., s=1.):
    return dict(kind='pose', cx=cx, cy=cy, angle_deg=angle, width=width, height=height, score=s)


class GeometryTest(unittest.TestCase):
    def test_angle_error_wraps_and_respects_symmetry(self):
        self.assertAlmostEqual(score.angle_error(179., -179.), 2.)
        self.assertAlmostEqual(score.angle_error(10., 190., period=180.), 0.)
        self.assertAlmostEqual(score.angle_error(0., 90., period=360.), 90.)

    def test_corners_match_round37_generator(self):
        # many128_pose.json truth[0]: template 140x102, scale 0.7342201613947882.
        corners = score.pose_corners(1150.1604891846423, 127.43275329326241, 140 * .7342201613947882,
                                     102 * .7342201613947882, 52.32773217006542)
        expected = [(1089.1118661074277, 145.22879330532626), (1151.9318608375393, 63.867859977831394),
                    (1211.209112261857, 109.63671328119855), (1148.3891175317453, 190.99764660869343)]
        for got, want in zip(corners, expected):
            self.assertAlmostEqual(got[0], want[0], places=6)
            self.assertAlmostEqual(got[1], want[1], places=6)

    def test_scale_reference_gives_physical_size(self):
        n = score.normalise(dict(kind='pose', cx=0, cy=0, angle_deg=0, scale_x=.5, scale_y=2., scale_reference=[36, 36]))
        self.assertEqual((n['width'], n['height']), (18., 72.))
        self.assertEqual(n['box'], (-9., -36., 9., 36.))

    def test_rotated_pose_box_encloses_corners(self):
        n = score.normalise(pose(0., 0., 90.))
        for got, want in zip(n['box'], (-10., -20., 10., 20.)):
            self.assertAlmostEqual(got, want)

    def test_rotated_iou(self):
        a = score.pose_corners(0, 0, 40, 20, 0)
        self.assertAlmostEqual(score.rotated_iou(a, a), 1.)
        self.assertEqual(score.rotated_iou(a, score.pose_corners(100, 0, 40, 20, 0)), 0.)
        b = score.pose_corners(10, 0, 40, 20, 0)
        self.assertAlmostEqual(score.rotated_iou(a, b), score.iou((-20, -10, 20, 10), (-10, -10, 30, 10)))
        # A 40x20 rectangle and itself turned 90 deg share a 20x20 square: 400 / (800 + 800 - 400).
        self.assertAlmostEqual(score.rotated_iou(a, score.pose_corners(0, 0, 40, 20, 90)), 1 / 3)
        # Two squares at 45 deg: octagon intersection, area 2*(sqrt2-1)*s^2 for unit half-size s.
        sq, diamond = score.pose_corners(0, 0, 2, 2, 0), score.pose_corners(0, 0, 2, 2, 45)
        inter = 8 * (math.sqrt(2) - 1)
        self.assertAlmostEqual(score.rotated_iou(sq, diamond), inter / (8 - inter))

    def test_tdcm_paper_iou_follows_get_iou(self):
        # TDCM utils/tool.py: getIOU draws rotated_rect(x, y, w, h, -r), and rotated_rect turns the
        # corners by [[cos, -sin], [sin, cos]] in y-down image coordinates.
        def rotated_rect(x, y, w, h, angle):
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            return [(x + u * c - v * s, y + u * s + v * c) for u, v in ((w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2))]
        g = dict(cx=50, cy=50, scale_x=1., scale_y=.5, angle_deg=30)
        p = dict(cx=52, cy=47, scale_x=1., scale_y=.5, angle_deg=40)
        self.assertAlmostEqual(score.tdcm_paper_iou(g, g), 1.)
        expected = score.rotated_iou(rotated_rect(50, 50, 36, 18, -30), rotated_rect(52, 47, 36, 18, -40))
        mirrored = score.rotated_iou(rotated_rect(50, 50, 36, 18, 30), rotated_rect(52, 47, 36, 18, 40))
        self.assertGreater(abs(expected - mirrored), .01)  # the example tells the two conventions apart
        self.assertAlmostEqual(score.tdcm_paper_iou(g, p), expected)


class PoseMatchTest(unittest.TestCase):
    def test_strict_tolerances_are_inclusive_and_separate(self):
        gt = [pose(100, 100, 0)]
        self.assertEqual(score.score_pose_image(gt, [pose(102, 100, 1)])['strict_tp'], 1)
        self.assertEqual(score.score_pose_image(gt, [pose(100, 100, 0, width=40.8)])['strict_tp'], 1)
        self.assertEqual(score.score_pose_image(gt, [pose(102.1, 100, 0)])['strict_tp'], 0)
        self.assertEqual(score.score_pose_image(gt, [pose(100, 100, 1.1)])['strict_tp'], 0)
        self.assertEqual(score.score_pose_image(gt, [pose(100, 100, 0, width=40.9)])['strict_tp'], 0)
        self.assertEqual(score.score_pose_image(gt, [pose(100, 100, 0, height=19.5)])['strict_tp'], 0)

    def test_one_object_cannot_score_twice(self):
        gt = [pose(100, 100, 0)]
        r = score.score_pose_image(gt, [pose(100, 100, 0, s=.9), pose(100.5, 100, 0, s=.8)])
        self.assertEqual((r['strict_tp'], r['strict_fp'], r['strict_fn']), (1, 1, 0))

    def test_maximum_matching_beats_greedy(self):
        # The high-score prediction can match either truth; greedy would take truth 0 and leave truth 1 unmatched.
        gt = [pose(100, 100, 0), pose(103, 100, 0)]
        preds = [pose(101.5, 100, 0, s=.9), pose(99, 100, 0, s=.5)]
        self.assertEqual(score.score_pose_image(gt, preds)['strict_tp'], 2)

    def test_association_radius_is_descriptive(self):
        gt = [pose(100, 100, 0)]
        r = score.score_pose_image(gt, [pose(115, 100, 20)])
        self.assertEqual((r['strict_tp'], r['association_tp']), (0, 1))
        self.assertAlmostEqual(r['association_errors'][0]['angle_deg'], 20.)
        self.assertEqual(score.score_pose_image(gt, [pose(121, 100, 0)])['association_tp'], 0)

    def test_nonfinite_prediction_is_an_unmatched_false_positive(self):
        r = score.score_pose_image([pose(0, 0, 0)], [pose(float('nan'), 0, 0)])
        self.assertEqual((r['strict_tp'], r['strict_fp'], r['invalid_predictions']), (0, 1, 1))

    def test_nonfinite_angle_or_empty_size_is_invalid(self):
        for bad in (pose(0, 0, float('nan')), pose(0, 0, 0, 0., 20.)):
            r = score.score_pose_image([pose(0, 0, 0)], [bad])
            self.assertEqual((r['invalid_predictions'], r['association_tp']), (1, 0))

    def test_counts_equal_many_object_accuracy(self):
        try:
            from many_object_accuracy import evaluate
        except ImportError as error:  # numpy lives in python-port/venv
            self.skipTest(str(error))
        rng = random.Random(50)
        for trial in range(20):
            truths, preds = [], []
            for k in range(30):
                x, y, a, s = rng.uniform(0, 300), rng.uniform(0, 300), rng.uniform(-180, 180), rng.uniform(.7, 1.3)
                truths.append(dict(pos_x=x, pos_y=y, angle=a, scale=s))
                for _ in range(rng.randint(0, 2)):
                    preds.append(dict(pos_x=x + rng.gauss(0, 1.2), pos_y=y + rng.gauss(0, 1.2), angle=a + rng.gauss(0, .7),
                                      scale=s * (1 + rng.gauss(0, .015)), score=rng.random()))
            for _ in range(10):
                preds.append(dict(pos_x=rng.uniform(0, 300), pos_y=rng.uniform(0, 300), angle=0, scale=1, score=rng.random()))
            ref = evaluate(preds, truths)
            to = lambda d: pose(d['pos_x'], d['pos_y'], d['angle'], 40 * d['scale'], 20 * d['scale'], d.get('score', 1.))  # noqa: E731
            mine = score.score_pose_image([to(t) for t in truths], [to(p) for p in preds])
            self.assertEqual(mine['strict_tp'], ref['strict']['tp'], trial)
            self.assertEqual(mine['association_tp'], ref['localization']['tp'], trial)


class BoxAndCountTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(score.iou((0, 0, 2, 2), (1, 0, 3, 2)), 1 / 3)
        self.assertEqual(score.iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.)

    def test_perfect_and_empty_ap(self):
        boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
        preds = [dict(kind='box', x0=b[0], y0=b[1], x1=b[2], y1=b[3], score=.9) for b in boxes]
        r = score.score_box_images([(boxes, preds)])
        self.assertAlmostEqual(r['ap50'], 1.)
        self.assertEqual((r['precision'], r['recall']), (1., 1.))
        self.assertEqual(score.score_box_images([(boxes, [])])['ap50'], 0.)
        self.assertIsNone(score.score_box_images([([], [])])['ap50'])

    def test_ap_ranks_false_positive_first(self):
        boxes = [(0, 0, 10, 10)]
        preds = [dict(kind='box', x0=50, y0=50, x1=60, y1=60, score=.9),
                 dict(kind='box', x0=0, y0=0, x1=10, y1=10, score=.5)]
        # Precision at recall 1 is 1/2 after the higher-scored false positive.
        self.assertAlmostEqual(score.score_box_images([(boxes, preds)])['ap50'], .5)

    def test_box_matching_finds_large_far_boxes(self):
        boxes = [(0, 0, 400, 400)]
        preds = [dict(kind='box', x0=10, y0=10, x1=390, y1=390, score=.9)]
        self.assertEqual(score.score_box_images([(boxes, preds)])['tp'], 1)

    def test_box_matching_follows_descending_score_and_uses_each_truth_once(self):
        boxes = [(0, 0, 10, 10)]
        high = dict(kind='box', x0=0, y0=0, x1=10, y1=12, score=.9)  # IoU 10/12
        low = dict(kind='box', x0=0, y0=0, x1=10, y1=10, score=.5)  # exact, but ranked second
        flags = score.match_boxes(boxes, [score.normalise(low), score.normalise(high)])
        self.assertEqual([f[:2] for f in flags], [(.9, True), (.5, False)])
        self.assertEqual(score.score_box_images([(boxes, [low, high])])['tp'], 1)

    def test_box_iou_threshold_is_half_and_inclusive(self):
        boxes = [(0, 0, 10, 10)]
        half = dict(kind='box', x0=0, y0=0, x1=10, y1=20, score=.9)  # IoU exactly 0.5
        below = dict(kind='box', x0=0, y0=0, x1=10, y1=25, score=.9)  # IoU 0.4
        self.assertEqual(score.score_box_images([(boxes, [half])])['tp'], 1)
        self.assertEqual(score.score_box_images([(boxes, [below])])['tp'], 0)

    def test_pose_converts_to_enclosing_box(self):
        boxes = [(80, 90, 120, 110)]
        r = score.score_box_images([(boxes, [pose(100, 100, 0)])])
        self.assertEqual(r['tp'], 1)
        self.assertEqual(r['centre_error_px']['max'], 0.)

    def test_count_errors_and_empty_scenes(self):
        r = score.count_errors([(10, 12), (0, 3), (5, 5)])
        self.assertAlmostEqual(r['mae'], 5 / 3)
        self.assertAlmostEqual(r['rmse'], math.sqrt(13 / 3))
        self.assertEqual(r['empty_scene_false_positives'], 3)

    def test_count_errors_use_absolute_differences(self):
        r = score.count_errors([(10, 12), (10, 8)])
        self.assertAlmostEqual(r['mae'], 2.)
        self.assertAlmostEqual(r['rmse'], 2.)

    def test_summary_median_of_even_count(self):
        s = score.summary([4., 1., 3., 2.])
        self.assertEqual((s['n'], s['median'], s['p90'], s['max']), (4, 2.5, 4., 4.))


if __name__ == '__main__':
    unittest.main()
