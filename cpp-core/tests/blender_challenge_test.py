"""Independent projection checks; never invoke Blender or the matcher."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import blender_challenge_prepare as tool


class BlenderChallengeTest(unittest.TestCase):
    def metadata(self,angle=0.,scale=1.):
        item=dict(pos_x=256.,pos_y=256.,angle=angle,scale=scale,identity='target',projected_center=[256.,256.])
        item['corners']=tool.analytic_corners(item).tolist()
        return dict(objects=[item],projection='orthographic frontal',perspective_out_of_model_excluded=True)

    def test_reference_crop_exact(self):
        data=self.metadata();np.testing.assert_array_equal(data['objects'][0]['corners'],[[208,220],[304,220],[304,292],[208,292]])
        self.assertEqual(tool.check_geometry(data)['corner_max_error_px'],0)

    def test_rotation_sign_and_scaled_dimensions(self):
        data=self.metadata(90,2)
        np.testing.assert_allclose(data['objects'][0]['corners'],[[184,352],[184,160],[328,160],[328,352]],atol=1e-12)
        tool.check_geometry(data)

    def test_stale_object_matrix_is_rejected(self):
        data=self.metadata();data['objects'][0]['corners']=[[255,255],[257,255],[257,257],[255,257]]
        with self.assertRaises(ValueError):tool.check_geometry(data)

    def test_wrong_camera_center_is_rejected(self):
        data=self.metadata();data['objects'][0]['projected_center'][0]+=.5
        with self.assertRaises(ValueError):tool.check_geometry(data)

    def test_perspective_not_mislabelled_2d(self):
        data=self.metadata();data['projection']='perspective'
        with self.assertRaises(ValueError):tool.check_geometry(data)

    def test_render_float_precision_is_allowed(self):
        data=self.metadata();data['objects'][0]['corners'][0][0]+=3e-5
        self.assertLess(tool.check_geometry(data)['corner_max_error_px'],tool.TOLERANCE)

if __name__=='__main__':unittest.main()
