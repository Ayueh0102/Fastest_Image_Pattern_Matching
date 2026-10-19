"""No GPU calls: verify independent candidate coverage and worker boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_sparse_orientation as bench


def pose(x=0., y=0., angle=0., scale=1.):
    return dict(pos_x=x, pos_y=y, angle=angle, scale=scale)


class SparseOrientationHarnessTest(unittest.TestCase):
    def test_proposal_coverage_uses_one_to_one_assignment(self):
        result = bench.proposal_accuracy([pose(2), pose(-4)], [pose(0), pose(4)])
        self.assertEqual(result['strict']['tp'], 2)
        # One coarse candidate cannot claim two nearby objects.
        result = bench.proposal_accuracy([pose(2)], [pose(0), pose(4)])
        self.assertEqual(result['strict']['tp'], 1)
        self.assertEqual(result['strict']['fn'], 1)

    def test_basin_inclusive_and_angle_wrapped(self):
        result = bench.proposal_accuracy([pose(6, angle=-177)], [pose(angle=177)])
        self.assertEqual(result['strict']['recall'], 1.)
        result = bench.proposal_accuracy([pose(6.0001, angle=-177)], [pose(angle=177)])
        self.assertEqual(result['strict']['recall'], 0.)

    def test_scale_violation_is_not_hidden_by_localization(self):
        result = bench.proposal_accuracy([pose()], [pose(scale=1.04)])
        self.assertEqual(result['strict']['fn'], 1)
        self.assertEqual(result['position_angle_only']['fn'], 0)
        result = bench.proposal_accuracy([pose(float('nan'))], [pose()])
        self.assertEqual(result['strict']['fn'], 1)

    def test_worker_job_has_no_truth_or_object_count(self):
        case = dict(source='source.png', template='template.png', angle=20.,
                    template_shape=[48, 64], truth=[pose()]*128, metadata='truth.json')
        protocol = dict(shared_settings=dict(scales=[1.], optimize_scale=False, limit=256),
                        sparse_settings={'sparse_'+key:value for key, value in bench.SPARSE_DEFAULTS.items()},
                        repeats=3, warmup_seconds=1.)
        job = bench.native_job(case, 'sparse_orientation', 1, protocol)
        self.assertEqual(job['limit'], 256)
        self.assertEqual(job['proposal_method'], 'sparse_orientation')
        self.assertEqual(bench.levels(case['template_shape'], job['min_area']), 1)
        self.assertFalse({'truth', 'metadata', 'count', 'roi', 'optimize_scale'} & job.keys())
        baseline = bench.native_job(case, 'pyramid', 1, protocol)
        self.assertEqual(baseline['limit'], job['limit'])
        self.assertNotIn('proposal_method', baseline)

    def test_heldout_fixture_is_fixed_scale_and_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cases = bench.make_cases(path)
            crowd = bench.make_crowd_case(path)
            self.assertEqual(len(crowd['truth']), 128)
            self.assertTrue(crowd['held_out'])
            self.assertEqual(crowd['scene_shape'], [768, 1536])
            self.assertTrue(all(item['scale'] == 1. and -20 < item['angle'] < 20
                                for item in crowd['truth']))
            self.assertEqual(crowd['template'], cases[0]['template'])
            self.assertEqual(bench.sha(crowd['source']), bench.sha(bench.make_crowd_case(path)['source']))


if __name__ == '__main__':
    unittest.main()
