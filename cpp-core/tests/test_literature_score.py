import math
from pathlib import Path
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


class PoseMatchTest(unittest.TestCase):
    def test_strict_tolerances_are_inclusive_and_separate(self):
        gt = [pose(100, 100, 0)]
        self.assertEqual(score.score_pose_image(gt, [pose(102, 100, 1)])['strict_tp'], 1)
        self.assertEqual(score.score_pose_image(gt, [pose(102.1, 100, 0)])['strict_tp'], 0)
        self.assertEqual(score.score_pose_image(gt, [pose(100, 100, 1.1)])['strict_tp'], 0)
        self.assertEqual(score.score_pose_image(gt, [pose(100, 100, 0, width=40.9)])['strict_tp'], 0)

    def test_one_object_cannot_score_twice(self):
        gt = [pose(100, 100, 0)]
        r = score.score_pose_image(gt, [pose(100, 100, 0, s=.9), pose(100.5, 100, 0, s=.8)])
        self.assertEqual((r['strict_tp'], r['strict_fp'], r['strict_fn']), (1, 1, 0))
        self.assertEqual(r['errors'][0]['position_px'], 0.)

    def test_loose_radius_separates_localisation_from_precision(self):
        gt = [pose(100, 100, 0)]
        r = score.score_pose_image(gt, [pose(104, 100, 20)], loose_radius=10)
        self.assertEqual((r['strict_tp'], r['loose_tp']), (0, 1))
        self.assertAlmostEqual(r['loose_errors'][0]['angle_deg'], 20.)


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


if __name__ == '__main__':
    unittest.main()
