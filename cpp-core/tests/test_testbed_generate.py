"""Checks for the test-bed generator: colour science, labels, geometry and determinism."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import testbed_generate as tb


class ColourScience(unittest.TestCase):
    def test_ciede2000_matches_sharma_reference_pairs(self):
        # Sharma, Wu, Dalal (2005) test data, pairs 1, 7, 17 and 25.
        pairs = [((50., 2.6772, -79.7751), (50., 0., -82.7485), 2.0425),
                 ((50., 0., 0.), (50., -1., 2.), 2.3669),
                 ((50., 2.5, 0.), (73., 25., -18.), 27.1492),
                 ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]
        for lab1, lab2, expected in pairs:
            self.assertAlmostEqual(tb.ciede2000(lab1, lab2), expected, places=4)
            self.assertAlmostEqual(tb.ciede2000(lab2, lab1), expected, places=4)

    def test_lab_round_trip(self):
        for rgb in ((196, 58, 48), (40, 140, 150), (236, 234, 226), (0, 0, 0), (255, 255, 255)):
            np.testing.assert_allclose(tb.lab_to_rgb(tb.rgb8_to_lab(rgb)), rgb, atol=1e-6)

    def test_gray8_is_the_repository_conversion(self):
        self.assertEqual(int(tb.gray8([255, 255, 255])), 255)
        self.assertEqual(int(tb.gray8([196, 58, 48])), (77*196 + 150*58 + 29*48 + 128) >> 8)


class Variants(unittest.TestCase):
    def test_every_direction_hits_its_target_within_quantization(self):
        base = np.array(tb.FAMILIES['red'])
        base_lab = tb.rgb8_to_lab(base)
        for direction in ('hue+', 'hue-', 'chroma-', 'isogray0', 'isogray90', 'lightness+'):
            for target in (4., 8.):
                rgb = tb.colour_variant(base, direction, target)
                self.assertLess(abs(tb.ciede2000(base_lab, tb.rgb8_to_lab(rgb)) - target), .75, (direction, target))

    def test_isogray_variants_have_identical_gray8_textures(self):
        base = np.array(tb.FAMILIES['teal'])
        for direction in ('isogray0', 'isogray90'):
            rgb = tb.colour_variant(base, direction, 8.)
            self.assertEqual(int(tb.gray8(rgb)), int(tb.gray8(base)))
            self.assertTrue(np.array_equal(tb.gray8(tb.capsule_texture(rgb)), tb.gray8(tb.capsule_texture(base))))

    def test_non_colour_content_is_shared_by_all_variants(self):
        a, b = tb.capsule_texture((10, 20, 30)), tb.capsule_texture((200, 100, 50))
        x0, y0, x1, y1 = tb.COLOUR_FIELD
        outside = np.ones(a.shape[:2], bool); outside[y0:y1, x0:x1] = False
        self.assertTrue(np.array_equal(a[outside], b[outside]))
        self.assertFalse(np.array_equal(a, b))


class Scenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.records = tb.make_textures(cls.root/'textures')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_labels_are_balanced_and_negative_colours_really_differ(self):
        for family in tb.FAMILIES:
            family_records = [r for r in self.records if r['family'] == family]
            self.assertEqual(sum(r['same_colour'] for r in family_records), 1)
            for r in family_records:
                if not r['same_colour']:
                    self.assertGreater(r['delta_e00_achieved'], 1.25)
                    # Isogray variants are gray-identical by construction; a small hue step
                    # can coincide by rounding, and the label records that truthfully.
                    if r['direction'].startswith('isogray'):
                        self.assertTrue(r['gray_texture_identical_to_base'])

    def test_layout_is_valid_and_covers_every_variant(self):
        placements = tb.layout('red', self.records)
        names = [p['texture'] for p in placements]
        self.assertEqual(names.count('red_base'), 10)
        self.assertEqual(len(set(names)), 1 + sum(1 for r in self.records if r['family'] == 'red' and not r['same_colour']))

    def test_rendering_is_deterministic_and_labels_survive_illumination(self):
        textures = {r['name']: tb.read_rgb(self.root/'textures'/r['file']).astype(np.float64) for r in self.records}
        placements = tb.layout('red', self.records)[:3]
        a = tb.render_rgb_scene(placements, textures, 'wb_warm', noise_seed=1, background_seed=2)
        b = tb.render_rgb_scene(placements, textures, 'wb_warm', noise_seed=1, background_seed=2)
        self.assertTrue(np.array_equal(a, b))
        neutral = tb.render_rgb_scene(placements, textures, 'neutral', noise_seed=1, background_seed=2)
        self.assertFalse(np.array_equal(a, neutral))

    def test_synthetic_truth_is_separate_and_consistent(self):
        output = self.root/'synthetic'
        tb.CONDITIONS_BACKUP = dict(tb.CONDITIONS)
        try:
            for key in list(tb.CONDITIONS):
                if key not in ('neutral', 'spot_warm'):
                    del tb.CONDITIONS[key]
            scenes = tb.make_synthetic(self.root/'textures', output)
        finally:
            tb.CONDITIONS.clear(); tb.CONDITIONS.update(tb.CONDITIONS_BACKUP)
        self.assertEqual(len(scenes), 2*len(tb.FAMILIES))
        report = tb.finalize(self.root/'no-blender', output, self.root/'final')
        self.assertEqual(len(report['synthetic']), len(scenes))
        for scene in scenes:
            truth = json.loads((output/f'{scene}.truth.json').read_text())
            self.assertEqual(len(truth['truth']), 32)
            self.assertEqual(sum(t['same_colour'] for t in truth['truth']), 10)

    def test_finalize_rejects_inconsistent_corners(self):
        output = self.root/'bad'
        output.mkdir()
        tb.save(output/'syn_bad.truth.json', dict(scene='syn_bad', truth=[dict(pos_x=100., pos_y=100., angle=10., scale=1.,
            corners=[[0., 0.]]*4)]))
        with self.assertRaises(ValueError):
            tb.finalize(self.root/'no-blender', output, self.root/'final-bad')


if __name__ == '__main__':
    unittest.main()
