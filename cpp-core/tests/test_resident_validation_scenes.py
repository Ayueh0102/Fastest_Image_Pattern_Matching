"""Data/geometry checks for held-out fixtures, without invoking the matcher."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import resident_validation_scenes as scenes
from many_object_scenes import make_placements, render_scene


class ValidationScenesTest(unittest.TestCase):
    def test_zero_degradation_matches_independent_renderer(self):
        template = np.random.default_rng(4).integers(30, 220, (12, 16), dtype=np.uint8)
        truth = make_placements(template.shape, rows=1, columns=2, cell_size=48, seed=912)
        reference = render_scene(template, truth, (48, 96), noise_std=0)
        image, strips = scenes.occluded_scene(template, truth, (48, 96), fractions=[0., 0.], sigma=0, noise_std=0)
        np.testing.assert_array_equal(image, reference)
        self.assertEqual(strips, [0, 0])

    def test_integer_upright_strip_only_changes_occluded_pixels(self):
        template = np.full((12, 16), 50, np.uint8)
        truth = make_placements(template.shape, rows=1, columns=1, cell_size=48, jitter=0, upright=True)
        truth[0].update(gain=1., bias=0.)
        image, strips = scenes.occluded_scene(template, truth, (48, 48), fractions=[.25], sigma=0, noise_std=0)
        expected = np.full((48, 48), 150, np.uint8)
        expected[18:30, 20:32] = 50
        np.testing.assert_array_equal(image, expected)
        self.assertEqual(strips, [4])

    def test_truth_invariant_across_degradation_and_new_seed_is_distinct(self):
        a, b = scenes.new_scenes(), scenes.new_scenes()
        self.assertEqual([x['name'] for x in a], list(scenes.NEW_CASES))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x['image'], y['image'])
            self.assertEqual(x['truth'], y['truth'])
        self.assertEqual(a[0]['truth'], a[2]['truth'])
        self.assertFalse(np.array_equal(a[0]['image'], a[2]['image']))
        self.assertNotEqual(a[0]['truth'], a[1]['truth'])
        self.assertEqual(a[3]['truth'], [])
        self.assertEqual(a[2]['validity'], 'challenge')
        self.assertEqual(len(a[0]['truth']) + len(a[1]['truth']), 24)

    def test_invalid_occlusion_configuration_is_rejected(self):
        template = np.ones((12, 16), np.uint8)
        truth = make_placements(template.shape, rows=1, columns=1, cell_size=48)
        for fractions in ([], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                scenes.occluded_scene(template, truth, (48, 48), fractions=fractions)
        with self.assertRaises(ValueError):
            scenes.occluded_scene(template, truth, (48, 48), fractions=[0], noise_std=-1)


if __name__ == '__main__':
    unittest.main()
