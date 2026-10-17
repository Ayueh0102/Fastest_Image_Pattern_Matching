"""Stage35 replay and adversarial evidence checks; no native/GPU jobs run.

The tiny fabricated worker records test the audit boundary, not image-matching
quality. They deliberately preserve file hashes when testing semantic checks.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_real_rgb as harness


class RealRgbBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.mask = self.base / 'discolor.png'
        pixels = np.zeros((16, 16), np.uint8)
        pixels[6, 6] = 2
        self.assertTrue(cv2.imwrite(str(self.mask), pixels))
        template = self.base / 'template.png'
        self.assertTrue(cv2.imwrite(str(template), np.full((8, 8, 3), 127, np.uint8)))
        names = ['normal050', 'normal200', 'normal400', 'anomaly']
        cases = []
        for name in names:
            source = self.base / (name + '.png')
            shutil.copyfile(template, source)
            case = dict(name=name, kind='visa_real', split='heldout' if name == 'anomaly' else 'pilot',
                        color_anomaly=name == 'anomaly', template_reference=False,
                        source=str(source), template=str(template),
                        search={'scales': [1.], 'min_area': 512, 'proposal_score': .25, 'limit': 300})
            if name == 'anomaly':
                case['mask_gt'] = str(self.mask)
            case['input_sha256'] = {key: harness.sha(case[key])
                                   for key in ('source', 'template', 'mask_gt') if key in case}
            cases.append(case)
        hashes = {case[key]: digest for case in cases
                  for key, digest in case['input_sha256'].items()}
        hashes['frozen-source.cpp'] = 'recorded-source-digest'
        protocol = dict(cases=cases, blocks=2, methods=list(harness.METHODS),
                        first_and_one_repeat=True, serial_fresh_processes=True, hashes=hashes)
        harness.save(self.base / 'protocol.json', protocol)
        self.report = dict(protocol=protocol, protocol_sha256=harness.sha(self.base / 'protocol.json'),
                           post_hashes=copy.deepcopy(hashes), complete=True, unchanged=True, entries=[])
        calibration = harness.calibration_names(cases)
        order = harness.sequence(calibration, 2) + harness.sequence(['anomaly'], 2)
        for name, block, method in order:
            case = next(case for case in cases if case['name'] == name)
            expected_job = harness.job(case, method)
            score = .5 if name == 'anomaly' else .95
            match = dict(pos_x=6., pos_y=6., angle=0., scale=1., score=score,
                         corners=[[4., 4.], [8., 4.], [8., 8.], [4., 8.]])
            scores = [dict(status=0, changed_fraction=1-score)] if method == 'chroma' else [score]
            row = dict(search_matches=[match], scores=scores, pipeline_ms=1., search_ms=.9,
                       verification_ms=.1)
            raw = dict(method=expected_job['method'], backend=expected_job['backend'],
                       threads=expected_job['threads'], search_config=expected_job['search'],
                       integrated_search=True, first=row, runs=[copy.deepcopy(row)],
                       warmups_ms=[], exact_repeat=True, construct_ms=.5, color_construct_ms=.2,
                       decode_ms=.1, final_memory=dict(peak_rss_bytes=2**20))
            entry = dict(case=name, block=block, method=method, returncode=0)
            for field, value in [('job', expected_job), ('raw', raw), ('stderr', '')]:
                path = self.base / f'{name}-{block}-{method}-{field}.json'
                if field == 'stderr':
                    path.write_text(value)
                else:
                    harness.save(path, value)
                entry[field], entry[field + '_sha256'] = path.name, harness.sha(path)
            self.report['entries'].append(entry)
        harness.save(self.base / 'calibration.json', harness.calibrate(cases, self.report['entries'], self.base))
        self.report['calibration_sha256'] = harness.sha(self.base / 'calibration.json')
        self.report['calibration_completed_workers'] = len(calibration) * 2 * len(harness.METHODS)

    def summary(self, input_map=None):
        return harness.summarize(self.report, self.base, input_map)

    def refresh_calibration(self):
        harness.save(self.base / 'calibration.json',
                     harness.calibrate(self.report['protocol']['cases'], self.report['entries'], self.base))
        self.report['calibration_sha256'] = harness.sha(self.base / 'calibration.json')

    @staticmethod
    def empty_candidates(raw):
        for row in [raw['first'], *raw['runs']]:
            row.update(scores=[], search_matches=[])

    def mutate_worker(self, change, method=None, case='anomaly', block=None):
        for entry in self.report['entries']:
            if entry['case'] != case or method is not None and entry['method'] != method:
                continue
            if block is not None and entry['block'] != block:
                continue
            path = self.base / entry['raw']
            raw = json.loads(path.read_text())
            change(raw)
            harness.save(path, raw)
            entry['raw_sha256'] = harness.sha(path)

    def test_valid_replay_is_deterministic_and_keeps_calibration_first(self):
        result = self.summary()
        self.assertTrue(result['integrity_passed'])
        self.assertEqual((result['workers'], result['algorithm_calls']), (40, 80))
        self.assertEqual(result, harness.summarize(json.loads(json.dumps(self.report)), self.base))
        self.assertEqual(result['thresholds']['names'], ['normal050', 'normal200', 'normal400'])
        self.assertTrue(all(entry['case'] != 'anomaly' for entry in self.report['entries'][:30]))
        for method in harness.METHODS:
            accuracy = result['cases'][-1]['methods'][method]['accuracy']
            self.assertEqual((accuracy['tp'], accuracy['fn']), (1, 0))
            self.assertTrue(accuracy['localized_color_tp'])

    def test_ground_truth_and_evaluation_masks_never_enter_jobs(self):
        case = dict(self.report['protocol']['cases'][-1], truth=[{'secret': 1}],
                    labels='evaluation only', points=[[6, 6]], expected_count=1)
        for method in harness.METHODS:
            saved = harness.job(case, method)
            for forbidden in ('truth', 'labels', 'points', 'expected_count', 'mask_gt', 'color_anomaly', 'poses'):
                self.assertNotIn(forbidden, saved)

    def test_original_budget_is_isolated_and_jobs_do_not_mutate_case(self):
        case = self.report['protocol']['cases'][0]
        before = copy.deepcopy(case)
        original = harness.job(case, 'geometry_original')
        self.assertEqual((original['method'], original['backend'], original['threads']),
                         ('geometry_only', 'cpu', 1))
        self.assertEqual({key: original['search'][key] for key in ('min_area', 'proposal_score', 'limit')},
                         dict(min_area=256, proposal_score=.5, limit=100))
        for method in harness.SHARED_GEOMETRY_METHODS:
            self.assertEqual(harness.job(case, method)['search'], before['search'])
        original['search']['scales'].append(2.)
        self.assertEqual(case, before)

    def test_original_geometry_may_differ_and_still_has_cost_accuracy(self):
        self.mutate_worker(self.empty_candidates, method='geometry_original')
        result = self.summary()
        self.assertTrue(result['integrity_passed'])
        case = result['cases'][-1]
        self.assertNotIn('geometry_original', case['geometry_comparison_methods'])
        original = case['methods']['geometry_original']
        self.assertEqual((original['accuracy']['tp'], original['accuracy']['fn']), (0, 1))
        self.assertTrue(original['cross_process_geometry_exact'])
        pairs = [pair for pair in case['pairs'] if pair['method'] == 'geometry_original']
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(pair['reference_method'] == 'geometry' and not pair['shared_geometry'] for pair in pairs))

    def test_original_geometry_still_requires_its_own_cross_process_consistency(self):
        self.mutate_worker(self.empty_candidates, method='geometry_original', block=1)
        result = self.summary()
        self.assertFalse(result['integrity_passed'])
        self.assertTrue(result['cases'][-1]['geometry_exact_between_methods'])
        self.assertFalse(result['cases'][-1]['methods']['geometry_original']['cross_process_geometry_exact'])

    def test_original_worker_config_is_checked_against_its_actual_job(self):
        shared = self.report['protocol']['cases'][-1]['search']
        self.mutate_worker(lambda raw: raw.update(search_config=shared), method='geometry_original', block=0)
        with self.assertRaisesRegex(ValueError, 'backend/configuration'):
            self.summary()

    def test_partial_calibration_retains_nulls_and_marks_provisional(self):
        for name in ('normal050', 'normal200'):
            self.mutate_worker(self.empty_candidates, case=name)
        self.refresh_calibration()
        result = self.summary()
        self.assertTrue(result['integrity_passed'])
        calibration = result['thresholds']['methods']['chroma']
        self.assertEqual(calibration['normal_image_values'][:2], [None, None])
        self.assertEqual(calibration['valid_images'], ['normal400'])
        self.assertEqual(calibration['abstained_images'], ['normal050', 'normal200'])
        self.assertEqual((calibration['valid_image_count'], calibration['abstained_image_count']), (1, 2))
        self.assertEqual(calibration['coverage'], 1/3)
        self.assertTrue(calibration['provisional'])
        self.assertFalse(calibration['uncalibrated'])
        self.assertAlmostEqual(calibration['threshold'], .05 + 1e-6)

    def test_no_valid_calibration_is_uncalibrated_and_preserves_positive_fn(self):
        for name in ('normal050', 'normal200', 'normal400'):
            self.mutate_worker(self.empty_candidates, case=name)
        self.refresh_calibration()
        result = self.summary()
        self.assertTrue(result['integrity_passed'])
        calibration = result['thresholds']['methods']['chroma']
        self.assertEqual(calibration['normal_image_values'], [None, None, None])
        self.assertIsNone(calibration['threshold'])
        self.assertEqual(calibration['coverage'], 0.)
        self.assertTrue(calibration['provisional'] and calibration['uncalibrated'])
        anomaly = result['cases'][-1]['methods']['chroma']['accuracy']
        self.assertTrue(anomaly['uncalibrated'] and anomaly['abstained'])
        self.assertEqual((anomaly['tp'], anomaly['fn'], anomaly['flagged_candidates']), (0, 1, 0))
        self.assertFalse(anomaly['localized_color_tp'])
        # Even valid candidates in a normal evaluation image cannot create TN
        # when the method has no calibrated threshold.
        entry = next(entry for entry in self.report['entries']
                     if entry['case'] == 'anomaly' and entry['method'] == 'chroma')
        row = harness.load_worker(self.base, entry)['first']
        normal = harness.accuracy(row, 'chroma', self.report['protocol']['cases'][0], None)
        self.assertTrue(normal['uncalibrated'] and normal['abstained'])
        self.assertEqual((normal['tp'], normal['fp'], normal['fn'], normal['tn']), (0, 0, 0, 0))

    def test_category_coverage_requires_no_color_threshold(self):
        case = dict(kind='real_category_probe', points=[[6., 6.]], point_tolerance_px=10.)
        result = harness.accuracy(dict(search_matches=[dict(pos_x=6., pos_y=6.)]), 'geometry_original', case, None)
        self.assertEqual(result['covered_points'], 1)
        self.assertNotIn('threshold', result)

    def test_controls_keep_fixed_threshold_when_visa_calibration_is_empty(self):
        for name in ('normal050', 'normal200', 'normal400'):
            self.mutate_worker(self.empty_candidates, case=name)
        control = self.report['protocol']['cases'][-1]
        control.update(kind='semi_synthetic49', truth=[dict(pos_x=6., pos_y=6., angle=0., scale=1., color_anomaly=True)])
        harness.save(self.base / 'protocol.json', self.report['protocol'])
        self.report['protocol_sha256'] = harness.sha(self.base / 'protocol.json')
        self.refresh_calibration()
        result = self.summary()
        self.assertTrue(result['integrity_passed'])
        for method, data in result['cases'][-1]['methods'].items():
            self.assertIsNone(result['thresholds']['methods'][method]['threshold'])
            self.assertEqual(data['accuracy']['threshold'], .1 if method == 'chroma' else .2)
            self.assertEqual((data['accuracy']['color_tp'], data['accuracy']['color_fn']), (1, 0))

    def test_recorded_pre_post_source_hash_mismatch_rejected(self):
        self.report['post_hashes']['frozen-source.cpp'] = 'different-source-digest'
        with self.assertRaisesRegex(ValueError, 'Recorded sources'):
            self.summary()

    def test_false_unchanged_flag_not_accepted(self):
        self.report['unchanged'] = False
        with self.assertRaisesRegex(ValueError, 'Recorded sources'):
            self.summary()

    def test_case_input_changed_after_freeze_rejected_before_worker(self):
        cases = self.base / 'cases.json'
        harness.save(cases, self.report['protocol']['cases'])
        Path(self.report['protocol']['cases'][0]['source']).write_bytes(b'altered photo')
        args = ['benchmark_real_rgb.py', '--cases', str(cases), '--build', str(self.base / 'missing-build'),
                '--output', str(self.base / 'should-not-run')]
        with mock.patch.object(sys, 'argv', args), mock.patch.object(harness.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'Input changed since scene freeze'):
                harness.main()
            child.assert_not_called()

    def test_changed_evaluation_mask_rejected_before_accuracy(self):
        pixels = np.zeros((16, 16), np.uint8)
        pixels[13, 13] = 2
        self.assertTrue(cv2.imwrite(str(self.mask), pixels))
        with self.assertRaisesRegex(ValueError, 'Evaluation mask differs'):
            self.summary()

    def test_evaluation_mask_can_move_without_changing_replayed_results(self):
        expected = self.summary()
        moved = self.base / 'relocated-mask.png'
        self.mask.rename(moved)
        self.assertEqual(expected, self.summary({str(self.mask): str(moved)}))

    def test_missing_evaluation_mask_is_not_silently_ignored(self):
        self.mask.unlink()
        with self.assertRaises(FileNotFoundError):
            self.summary()

    def test_worker_method_backend_threads_and_search_are_checked(self):
        for key, value in [('method', 'chroma'), ('backend', 'cpu'), ('threads', 8),
                           ('search_config', {'scales': [2.]})]:
            with self.subTest(field=key):
                entry = next(entry for entry in self.report['entries']
                             if entry['case'] == 'anomaly' and entry['method'] == 'rgb_metal' and entry['block'] == 0)
                path = self.base / entry['raw']
                original, original_hash = path.read_bytes(), entry['raw_sha256']
                try:
                    self.mutate_worker(lambda raw: raw.update({key: value}), method='rgb_metal', block=0)
                    with self.assertRaisesRegex(ValueError, 'backend/configuration'):
                        self.summary()
                finally:
                    path.write_bytes(original)
                    entry['raw_sha256'] = original_hash

    def test_false_exact_repeat_flag_rejected_even_with_matching_rows(self):
        self.mutate_worker(lambda raw: raw.update(exact_repeat=False), method='rgb_cpu', block=0)
        with self.assertRaisesRegex(ValueError, 'Incorrect repeat flag'):
            self.summary()

    def test_true_exact_repeat_flag_cannot_hide_changed_repeat(self):
        self.mutate_worker(lambda raw: raw['runs'][0]['search_matches'][0].update(pos_x=15.),
                           method='rgb_cpu', block=0)
        with self.assertRaisesRegex(ValueError, 'Incorrect repeat flag'):
            self.summary()

    def test_common_cross_process_geometry_shift_fails_integrity(self):
        def shift(raw):
            for row in [raw['first'], *raw['runs']]:
                row['search_matches'][0]['pos_x'] += 10
        self.mutate_worker(shift, block=1)
        result = self.summary()
        self.assertFalse(result['integrity_passed'])
        case = result['cases'][-1]
        self.assertTrue(case['geometry_exact_between_methods'])
        for method in case['methods'].values():
            self.assertTrue(method['cross_process_scores_exact'])
            self.assertFalse(method['cross_process_geometry_exact'])

    def test_changed_threshold_cannot_be_reauthorized_by_only_updating_file_hash(self):
        path = self.base / 'calibration.json'
        data = json.loads(path.read_text())
        data['methods']['chroma']['threshold'] = .99
        harness.save(path, data)
        self.report['calibration_sha256'] = harness.sha(path)
        with self.assertRaisesRegex(ValueError, 'Calibration differs'):
            self.summary()

    def test_missing_worker_and_early_heldout_are_rejected(self):
        original = list(self.report['entries'])
        try:
            self.report['entries'] = original[:-1]
            with self.assertRaisesRegex(ValueError, 'Missing/reordered workers'):
                self.summary()
            self.report['entries'] = [original[-1], *original[:-1]]
            with self.assertRaisesRegex(ValueError, 'Missing/reordered workers'):
                self.summary()
        finally:
            self.report['entries'] = original

    def test_missing_and_modified_raw_files_are_rejected(self):
        path = self.base / self.report['entries'][-1]['raw']
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'Altered worker raw'):
            self.summary()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.summary()

    def test_duplicate_color_prediction_counts_only_one_true_positive(self):
        match = dict(pos_x=6., pos_y=6., angle=0., scale=1.)
        case = dict(kind='semi_synthetic49', truth=[dict(match, color_anomaly=True)])
        result = harness.accuracy(dict(search_matches=[match, match], scores=[.1, .1]), 'rgb_cpu', case, .2)
        self.assertEqual((result['color_tp'], result['color_fp'], result['color_fn']), (1, 1, 0))
        missing = harness.accuracy(dict(search_matches=[], scores=[]), 'rgb_cpu', case, .2)
        self.assertEqual((missing['color_tp'], missing['color_fn']), (0, 1))

    def test_category_point_cannot_be_counted_twice(self):
        case = dict(points=[[6., 6.]], point_tolerance_px=10.)
        result = harness.category_coverage([dict(pos_x=6., pos_y=6.), dict(pos_x=6., pos_y=6.)], case)
        self.assertEqual((result['covered_points'], result['unassigned_detections']), (1, 1))

    def test_normal_image_without_candidates_is_an_explicit_abstention(self):
        case = self.report['protocol']['cases'][0]
        result = harness.accuracy(dict(search_matches=[], scores=[]), 'geometry', case, .2)
        self.assertTrue(result['abstained'])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']), (0, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()
