"""Pure CPU evaluation checks; no image matching library or GPU is loaded."""
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from many_object_accuracy import evaluate


def pose(x=0., y=0., angle=0., scale=1.):
    return dict(pos_x=x, pos_y=y, angle=angle, scale=scale)


class ManyObjectAccuracyTest(unittest.TestCase):
    def test_duplicate_wrong_pose_does_not_steal_strict_match(self):
        result = evaluate([pose(angle=20), pose(x=1)], [pose()])
        self.assertEqual(result['strict']['tp'], 1)
        self.assertEqual(result['strict']['fp'], 1)
        self.assertEqual(result['strict']['pairs'][0]['prediction_index'], 1)
        self.assertEqual(result['localization']['pairs'][0]['prediction_index'], 0)
        self.assertEqual(result['strict']['unmatched_prediction_indices'], [0])

    def test_each_extra_prediction_is_one_false_positive(self):
        truths = [pose(), pose(x=100)]
        for label, extra in (('far_from_truth', pose(x=50, y=50)), ('duplicate_of_truth', pose(x=.5))):
            with self.subTest(extra=label):
                result = evaluate([pose(), pose(x=100), extra], truths)
                for metric in ('strict', 'localization'):
                    self.assertEqual(tuple(result[metric][key] for key in ('tp', 'fp', 'fn')), (2, 1, 0))
                self.assertEqual(result['strict']['unmatched_prediction_indices'], [2])

    def test_augmenting_path_recovers_both_and_order_invariance(self):
        truths = [pose(), pose(x=3)]
        # The first truth prefers prediction 0, but it is truth 1's only edge.
        predictions = [pose(x=1), pose(x=-2)]
        for gt in itertools.permutations(truths):
            for matches in itertools.permutations(predictions):
                result = evaluate(matches, gt)
                for metric in ('strict', 'localization'):
                    self.assertEqual(tuple(result[metric][key] for key in ('tp', 'fp', 'fn')), (2, 0, 0))
                    self.assertEqual(len({pair['prediction_index'] for pair in result[metric]['pairs']}), 2)
                    self.assertEqual(result[metric]['precision'], 1.)
                    self.assertEqual(result[metric]['recall'], 1.)

    def test_wrapped_angles_and_inclusive_thresholds(self):
        result = evaluate([pose(x=2, angle=-179.5, scale=1.02)], [pose(angle=179.5)])
        self.assertEqual(result['strict']['tp'], 1)
        errors = result['strict']['max_errors']
        self.assertEqual(errors['position_px'], 2.)
        self.assertEqual(errors['angle_deg'], 1.)
        self.assertAlmostEqual(errors['relative_scale'], .02)
        for prediction in (pose(x=2.001), pose(angle=1.001), pose(scale=1.02001), pose(scale=.97999)):
            self.assertEqual(evaluate([prediction], [pose()])['strict']['tp'], 0)

    def test_independent_localization_gate(self):
        result = evaluate([pose(x=20, angle=90, scale=2)], [pose()])
        self.assertEqual(result['strict']['tp'], 0)
        self.assertEqual(result['localization']['tp'], 1)
        self.assertEqual(evaluate([pose(x=20.01)], [pose()])['localization']['tp'], 0)

    def test_empty_inputs_and_json_serialization(self):
        for matches, truths, counts in (([], [], (0, 0, 0)), ([pose()], [], (0, 1, 0)),
                                        ([], [pose()], (0, 0, 1))):
            result = evaluate(matches, truths)
            json.dumps(result, allow_nan=False)
            for metric in ('strict', 'localization'):
                self.assertEqual(tuple(result[metric][key] for key in ('tp', 'fp', 'fn')), counts)
                self.assertEqual(result[metric]['precision'], 0.)
                self.assertEqual(result[metric]['recall'], 0.)
                self.assertTrue(all(value is None for value in result[metric]['max_errors'].values()))

    def test_invalid_predictions_count_as_false_positives(self):
        result = evaluate([pose(x=float('nan')), pose(scale=0), pose()], [pose()])
        self.assertEqual(result['invalid_prediction_indices'], [0, 1])
        self.assertEqual(result['strict']['fp'], 2)
        self.assertEqual(result['strict']['details'][0]['prediction_index'], 2)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            evaluate([], [pose(angle=float('inf'))])
        with self.assertRaises(ValueError):
            evaluate([], [pose(scale=-1)])


if __name__ == '__main__':
    unittest.main()
