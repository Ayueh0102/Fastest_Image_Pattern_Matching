"""Independent geometry and model controls for stage34 scale diagnostics."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import scale_sampling_study as study


class ScaleSamplingTest(unittest.TestCase):
    def test_bilinear_hand_computed_and_zero_extension(self):
        image = np.array([[1., 3.], [5., 7.]])
        values = study.bilinear(image, np.array([.5, -.5, 2.]), np.array([.5, 0., 0.]))
        np.testing.assert_array_equal(values, [4., .5, 0.])

    def test_ncc_photometric_invariance_and_constant(self):
        a = np.arange(10, dtype=float)
        self.assertAlmostEqual(study.ncc(a, 3*a+7), 1.)
        self.assertEqual(study.ncc(a, np.ones(10)), -2.)

    def test_geometry_quarter_turn(self):
        pose = dict(pos_x=12., pos_y=16., angle=90.)
        x, y = study.source_to_template(np.array([12., 14.]), np.array([16., 16.]), pose, (8, 10), 2.)
        np.testing.assert_allclose(x, [5, 5], atol=1e-12)
        np.testing.assert_allclose(y, [4, 5], atol=1e-12)

    def test_integer_copy_inverse_and_forward(self):
        template = np.random.default_rng(7).random((17, 33))
        source = np.full((64, 64), .5)
        source[20:37, 11:44] = template
        pose = dict(pos_x=27.5, pos_y=28.5, angle=0.)
        for method in ('inverse_raw', 'inverse_interior2', 'forward_pixel'):
            self.assertAlmostEqual(study.score(source, template, pose, 1., method), 1., places=12)
            self.assertLess(study.score(source, template, {**pose, 'pos_x':28.5}, 1., method), .99)

    def test_render_integer_legacy_copy_and_photometry(self):
        template = np.random.default_rng(12).integers(30, 200, (17, 33), dtype=np.uint8)
        pose = dict(pos_x=27.5, pos_y=28.5, angle=0., scale=1., gain=1., bias=0.)
        image = study.render(template, pose, (64, 64), 'legacy_linear', 1, noise_std=0)
        np.testing.assert_array_equal(image[20:37, 11:44], template)
        self.assertTrue(np.all(image[:19] == 150))

    def test_area_model_changes_high_frequency_signal(self):
        template = ((np.indices((17, 33)).sum(axis=0) % 2)*160 + 40).astype(np.uint8)
        pose = dict(pos_x=27.5, pos_y=28.5, angle=0., scale=1., gain=1., bias=0.)
        area = study.render(template, pose, (64, 64), 'area4', 1, noise_std=0)
        linear = study.render(template, pose, (64, 64), 'legacy_linear', 1, noise_std=0)
        self.assertLess(area[22:35, 13:42].std(), linear[22:35, 13:42].std())

    def test_forward_area_independent_renderer_truth_peak(self):
        template = np.random.default_rng(123).integers(40, 200, (17, 33), dtype=np.uint8)
        pose = dict(pos_x=33.25, pos_y=30.75, angle=31., scale=.88, gain=1.02, bias=-3.)
        image = study.render(template, pose, (64, 64), 'area4', 1, noise_std=0)/255.
        template = template/255.
        scores = [study.score(image, template, pose, pose['scale']*factor, 'forward_area4') for factor in (.96, 1., 1.04)]
        self.assertEqual(int(np.argmax(scores)), 1)
        self.assertGreater(scores[1], .999)

    def test_predeclared_grid_and_seeds(self):
        self.assertEqual(len(study.FACTORS), 41)
        self.assertEqual(study.FACTORS[20], 1.)
        self.assertEqual(study.SCALE_LIMIT, .02)
        self.assertNotIn(2026101602, study.SEEDS)


if __name__ == '__main__':
    unittest.main()
