"""Small, pure-data checks for the independent crowded-scene renderer."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import many_object_scenes as scenes


class ManyObjectScenesTest(unittest.TestCase):
    def setUp(self):
        self.template = np.random.default_rng(37).integers(20, 235, (10, 14), np.uint8)
        self.options = dict(rows=12, columns=12, cell_size=32, jitter=.8, seed=413)
        self.shape = (384, 384)

    def test_deterministic_prefix_and_more_than_one_hundred(self):
        bank = scenes.make_placements(self.template.shape, **self.options)
        self.assertEqual(len(bank), 144)
        self.assertEqual(bank, scenes.make_placements(self.template.shape, **self.options))
        low, high = bank[:64], bank[:128]
        self.assertEqual(low, high[:64])
        a = scenes.render_scene(self.template, low, self.shape, noise_seed=824)
        b = scenes.render_scene(self.template, high, self.shape, noise_seed=824)
        for item in low:
            x0, y0, x1, y1 = item['warp_roi']
            np.testing.assert_array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
        self.assertFalse(np.array_equal(a, b))
        self.assertGreater(scenes.validate_placements(high, self.shape,
                           self.template.shape)['min_roi_separation_px'], 0)

    def test_geometry_is_independent_of_the_matcher(self):
        bank = scenes.make_placements(self.template.shape, **self.options)
        for item in bank:
            matrix = cv2.getRotationMatrix2D((self.template.shape[1]/2,
                self.template.shape[0]/2), item['angle'], item['scale'])
            matrix[:, 2] += [item['pos_x']-self.template.shape[1]/2,
                            item['pos_y']-self.template.shape[0]/2]
            points = np.array([[[0., 0.], [14., 0.], [14., 10.], [0., 10.]]])
            np.testing.assert_allclose(cv2.transform(points, matrix)[0], item['corners'],
                                       rtol=0, atol=1e-10)
            center = matrix @ [7., 5., 1.]
            np.testing.assert_allclose(center, [item['pos_x'], item['pos_y']], atol=1e-12)
        validation = scenes.validate_placements(bank, self.shape, self.template.shape)
        self.assertEqual(validation['pairwise_roi_checks'], 144*143//2)
        self.assertGreaterEqual(validation['min_roi_boundary_px'], 0)

    def test_upright_control_shares_positions_and_photometry(self):
        pose = scenes.make_placements(self.template.shape, **self.options)
        upright = scenes.make_placements(self.template.shape, **self.options, upright=True)
        for a, b in zip(pose, upright):
            for key in ('pos_x', 'pos_y', 'cell', 'gain', 'bias'):
                self.assertEqual(a[key], b[key])
            self.assertEqual((b['angle'], b['scale']), (0., 1.))

    def test_upright_integer_center_renders_at_ground_truth_pixels(self):
        truth = scenes.make_placements(self.template.shape, rows=1, columns=1,
            cell_size=32, jitter=0, upright=True)
        truth[0].update(gain=1., bias=0.)
        image = scenes.render_scene(self.template, truth, (32, 32), noise_std=0)
        expected = np.full((32, 32), 150, np.uint8)
        expected[11:21, 9:23] = self.template
        np.testing.assert_array_equal(image, expected)

    def test_reconstruction_from_serialized_truth_and_png(self):
        truth = scenes.make_placements(self.template.shape, **self.options)[:128]
        original = scenes.render_scene(self.template, truth, self.shape, noise_seed=824)
        metadata = dict(truth=truth, shape=self.shape, noise_seed=824,
            pixels_sha256=hashlib.sha256(original.tobytes()).hexdigest())
        restored = json.loads(json.dumps(metadata, allow_nan=False))
        rebuilt = scenes.render_scene(self.template, restored['truth'], tuple(restored['shape']),
                                      noise_seed=restored['noise_seed'])
        self.assertEqual(hashlib.sha256(rebuilt.tobytes()).hexdigest(), restored['pixels_sha256'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scene.png'
            self.assertTrue(cv2.imwrite(str(path), original))
            np.testing.assert_array_equal(cv2.imread(str(path), 0), rebuilt)

    def test_invalid_overlap_boundary_and_corners_rejected(self):
        bank = scenes.make_placements(self.template.shape, **self.options)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            scenes.validate_placements([bank[0], bank[0]], self.shape, self.template.shape)
        with self.assertRaisesRegex(ValueError, 'leaves the image'):
            scenes.validate_placements(bank, (32, 32), self.template.shape)
        corrupt = copy.deepcopy(bank[:1])
        corrupt[0]['pos_x'] += 1
        with self.assertRaisesRegex(ValueError, 'corners disagree'):
            scenes.validate_placements(corrupt, self.shape, self.template.shape)

    def test_real_template_all_full_size_banks_fit_without_rendering(self):
        template = cv2.imread(str(scenes.TEMPLATE_SOURCE), 0)
        self.assertEqual(template.shape, (102, 140))
        for columns in (16, 32):
            bank = scenes.make_placements(template.shape, columns=columns)
            validation = scenes.validate_placements(bank, (4096, columns*256), template.shape)
            self.assertEqual(validation['object_count'], 16*columns)
            self.assertGreater(validation['min_roi_separation_px'], 0)


if __name__ == '__main__':
    unittest.main()
