"""Stage-46 harness controls: thresholds, decisions, gates and integrity checks without running the worker."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_illuminant_colour as stage46  # noqa: E402
import testbed_generate as tb  # noqa: E402

SYNTHETIC_CONDITIONS = ('neutral', 'exposure_dark', 'wb_cool', 'shading', 'wb_warm', 'exposure_bright', 'spot_warm', 'noise_jpeg')
DIFFERENT = [(d, t) for d in ('hue+', 'hue-', 'chroma-', 'isogray0', 'isogray90') for t in (2., 4., 8., 16.)] + \
            [('lightness+', 4.), ('lightness+', 8.)]
BLENDER_DIFFERENT = [('isogray90', 4.), ('chroma-', 8.), ('hue-', 8.), ('isogray0', 8.), ('hue+', 8.), ('chroma-', 4.),
                     ('lightness+', 8.), ('hue+', 4.)]


def row(group, condition, same, direction='base', target=0., value=None, index=0):
    source = 'blender' if group == 'bl_red' else 'synthetic'
    if value is None:
        value = .001 if same else .002 * target + .01
    return dict(scene=f'{group}_{condition}', source=source, group=group, family=group.split('_')[1], condition=condition,
                split=stage46.split(condition), index=index, same_colour=same, direction=direction, delta_e00_target=target,
                delta_e00_achieved=target, gray_identical=direction.startswith('isogray'), found=True, prediction=index,
                values={arm: value for arm in stage46.ARMS if group != 'bl_red' or not arm.startswith('card')}, diagnostics={})


def fixture():
    rows = []
    for group in ('syn_red', 'syn_teal'):
        for condition in SYNTHETIC_CONDITIONS:
            k = 0
            for _ in range(10):
                rows.append(row(group, condition, True, index=k)); k += 1
            for direction, target in DIFFERENT:
                rows.append(row(group, condition, False, direction, target, index=k)); k += 1
    for condition in ('white', 'warm3200', 'mixed'):
        k = 0
        for _ in range(4):
            rows.append(row('bl_red', condition, True, index=k)); k += 1
        for direction, target in BLENDER_DIFFERENT:
            rows.append(row('bl_red', condition, False, direction, target, index=k)); k += 1
    # A weaker baseline so that H3 is a real comparison.
    for r in rows:
        if not r['same_colour'] and r['delta_e00_target'] <= (4 if r['group'] == 'bl_red' else 2):
            for arm in stage46.COMPARED:
                r['values'][arm] = .0005
    return rows


def failed(checks):
    return [(c['name'], c.get('condition'), c.get('group')) for c in checks if not c['passed']]


def set_values(rows, arm, value, **where):
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    for r in hits:
        r['values'][arm] = value
    return hits


class Decisions(unittest.TestCase):
    def evaluate(self, rows):
        return stage46.hypotheses(rows, stage46.thresholds(rows))

    def test_clean_fixture_passes(self):
        checks = self.evaluate(fixture())
        self.assertEqual(failed(checks), [])
        self.assertEqual(len([c for c in checks if c['name'].startswith('H3')]), 16)
        self.assertEqual([c['n'] for c in checks if c['name'] == 'H1b gated different-colour rejection'], [30, 30, 30])
        self.assertEqual(next(c for c in checks if c['name'].startswith('H1c'))['n'], 36)
        self.assertEqual([c['n'] for c in checks if c['name'].startswith('H2b')], [7, 7])

    def test_one_extra_wrong_accept_fails_blender(self):
        rows = fixture()
        set_values(rows, stage46.PRIMARY, 0., group='bl_red', condition='mixed', direction='hue+', delta_e00_target=8.)
        self.assertEqual(failed(self.evaluate(rows)), [('H2b Blender gated different-colour rejection', 'mixed', None)])

    def test_one_extra_wrong_accept_fails_synthetic_boundary(self):
        rows = fixture()
        set_values(rows, stage46.PRIMARY, 0., group='syn_red', condition='wb_warm', direction='hue-', delta_e00_target=4.)
        self.assertEqual(failed(self.evaluate(rows)), [], '29/30 is within the pre-registered .95')
        set_values(rows, stage46.PRIMARY, 0., group='syn_teal', condition='wb_warm', direction='chroma-', delta_e00_target=16.)
        self.assertEqual(failed(self.evaluate(rows)), [('H1b gated different-colour rejection', 'wb_warm', None)])

    def test_isogray_pooled_boundary(self):
        rows = fixture()
        set_values(rows, stage46.PRIMARY, 0., group='syn_red', condition='noise_jpeg', direction='isogray0', delta_e00_target=16.)
        self.assertEqual(failed(self.evaluate(rows)), [])
        set_values(rows, stage46.PRIMARY, 0., group='syn_red', condition='spot_warm', direction='isogray90', delta_e00_target=4.)
        names = [name for name, _, _ in failed(self.evaluate(rows))]
        self.assertIn('H1c isogray rejection (pooled)', names)

    def test_abstained_same_colour_is_not_accepted(self):
        rows = fixture()
        same = [r for r in rows if r['condition'] == 'spot_warm' and r['same_colour']][:3]
        for r in same:
            for arm in (stage46.PRIMARY, *stage46.COMPARED):
                r['values'][arm] = None
        self.assertEqual(failed(self.evaluate(rows)), [('H1a same-colour acceptance', 'spot_warm', None)])
        table = stage46.thresholds(rows)
        result = stage46.cell([r for r in rows if r['group'] == 'syn_red' and r['condition'] == 'spot_warm'], stage46.PRIMARY, table)
        self.assertEqual((result['same']['abstained'], result['same']['accepted']), (3, 7))

    def test_abstained_different_colour_is_never_accepted(self):
        rows = fixture()
        hits = set_values(rows, stage46.PRIMARY, None, condition='wb_warm', same_colour=False)
        table = stage46.thresholds(rows)
        self.assertFalse(any(stage46.accepted(r, stage46.PRIMARY, table) for r in hits))
        result = stage46.cell([r for r in rows if r['group'] == 'syn_red' and r['condition'] == 'wb_warm'], stage46.PRIMARY, table)
        self.assertEqual((result['different']['abstained'], result['different']['wrongly_accepted']), (22, 0))

    def test_thresholds_use_calibration_only(self):
        rows = fixture()
        hits = [r for r in rows if r['group'] == 'syn_red' and r['split'] == 'calibration' and r['same_colour']]
        for k, r in enumerate(hits):
            r['values'][stage46.PRIMARY] = .0005 + .00001 * k
        table = stage46.thresholds(rows)
        self.assertLess(table[(stage46.PRIMARY, 'syn_red')], max(r['values'][stage46.PRIMARY] for r in hits))
        calibration = [r['values'][stage46.PRIMARY] for r in rows if r['group'] == 'syn_red' and r['split'] == 'calibration'
                       and r['same_colour']]
        self.assertEqual(len(calibration), 40)
        self.assertEqual(table[(stage46.PRIMARY, 'syn_red')], float(np.percentile(calibration, 95)))
        set_values(rows, stage46.PRIMARY, 1., split='holdout', same_colour=True)
        self.assertEqual(stage46.thresholds(rows), table)
        set_values(rows, stage46.PRIMARY, None, split='calibration', same_colour=True, condition='neutral')
        self.assertEqual(stage46.thresholds(rows)[(stage46.PRIMARY, 'syn_red')],
                         float(np.percentile([v for r in rows if r['group'] == 'syn_red' and r['split'] == 'calibration'
                                              and r['same_colour'] for v in [r['values'][stage46.PRIMARY]] if v is not None], 95)))
        set_values(rows, stage46.PRIMARY, .002, group='syn_red', condition='shading', same_colour=True)
        self.assertGreater(stage46.thresholds(rows)[(stage46.PRIMARY, 'syn_red')], table[(stage46.PRIMARY, 'syn_red')])
        self.assertNotIn(('card_white_region', 'bl_red'), table)

    def test_max_rule_is_the_calibration_maximum(self):
        rows = fixture()
        hits = set_values(rows, 'chroma', None, group='bl_red', condition='white', same_colour=True)
        for k, r in enumerate(hits):
            r['values']['chroma'] = .001 * (k + 1)
        values = [r['values']['chroma'] for r in hits]
        self.assertEqual(stage46.thresholds(rows, 'max')[('chroma', 'bl_red')], max(values))

    def test_lightness_and_delta_e2_are_reported_not_gated(self):
        rows = fixture()
        hits = [r for r in rows if not r['same_colour'] and (r['direction'] == 'lightness+' or r['delta_e00_target'] == 2.)]
        for r in hits:
            for arm in (stage46.PRIMARY, *stage46.COMPARED):
                r['values'][arm] = 0.
        self.assertEqual(failed(self.evaluate(rows)), [])
        table = stage46.thresholds(rows)
        report = stage46.direction_table([r for r in rows if r['group'] == 'syn_red' and r['split'] == 'holdout'], stage46.PRIMARY, table)
        self.assertEqual(report['lightness+@4']['rate'], 0)
        self.assertEqual(report['hue+@2']['rate'], 0)
        self.assertEqual(report['same']['rate_kind'], 'acceptance')

    def test_exposure_bright_is_reported_not_gated(self):
        rows = fixture()
        set_values(rows, stage46.PRIMARY, None, condition='exposure_bright')
        self.assertEqual(failed(self.evaluate(rows)), [])
        summary = stage46.evaluate_rows(rows)
        bright = summary['per_condition'][stage46.PRIMARY]['syn_red']['exposure_bright']
        self.assertEqual((bright['same']['acceptance'], bright['same']['abstained'], bright['auc']), (0, 10, .5))

    def test_h3_needs_auc_not_below_each_baseline(self):
        rows = fixture()
        for r in rows:
            if r['group'] == 'syn_teal' and r['condition'] == 'noise_jpeg' and not r['same_colour'] and r['delta_e00_target'] == 2.:
                r['values'][stage46.PRIMARY] = .0005
                r['values']['masked_region'] = .01
        self.assertEqual(failed(self.evaluate(rows)), [('H3 AUC vk_region >= masked_region', 'noise_jpeg', 'syn_teal')])

    def test_auc_ties_and_abstentions(self):
        self.assertEqual(stage46.auc([1, 2], [2, 3]), .875)
        self.assertEqual(stage46.auc([None], [None]), .5)
        self.assertEqual(stage46.auc([1.], [None]), 1.)
        self.assertEqual(stage46.auc([None], [5.]), 0.)
        self.assertIsNone(stage46.auc([], [1.]))

    def test_fallback_only_replaces_abstentions(self):
        rows = fixture()
        table = stage46.thresholds(rows)
        r = copy.deepcopy(rows[0])
        r['values'].update(vk_region=None, vk_uncorrected_region=0.)
        self.assertTrue(stage46.fallback_accepted(r, table))
        r['values'].update(vk_region=1., vk_uncorrected_region=0.)
        self.assertFalse(stage46.fallback_accepted(r, table))

    def test_blender_transfer_uses_synthetic_red_threshold(self):
        rows = fixture()
        set_values(rows, stage46.PRIMARY, .0008, group='bl_red', same_colour=True)
        set_values(rows, stage46.PRIMARY, .0005, group='bl_red', condition='white', same_colour=True)
        summary = stage46.evaluate_rows(rows)
        self.assertEqual(summary['per_condition'][stage46.PRIMARY]['bl_red']['mixed']['same']['acceptance'], 0)
        self.assertEqual(summary['blender_transfer_syn_red'][stage46.PRIMARY]['mixed']['same']['acceptance'], 1)
        self.assertNotIn('card_white_region', summary['blender_transfer_syn_red'][stage46.PRIMARY])


class Inputs(unittest.TestCase):
    def test_split_covers_every_condition_once(self):
        conditions = list(tb.CONDITIONS) + [c for _, c in stage46.BLENDER]
        self.assertEqual(sorted(conditions), sorted(stage46.CALIBRATION + stage46.HOLDOUT))
        self.assertFalse(set(stage46.CALIBRATION) & set(stage46.HOLDOUT))
        self.assertTrue(set(stage46.GATED_SYNTHETIC + stage46.GATED_BLENDER) <= set(stage46.HOLDOUT))
        self.assertIn('warm3200', stage46.HOLDOUT)
        self.assertIn('mixed', stage46.HOLDOUT)

    def test_jobs_carry_masks_and_no_gt(self):
        scene_list = stage46.scenes(Path('/testbed'))
        self.assertEqual(len(scene_list), 19)
        bl = next(s for s in scene_list if s['name'] == 'bl_rgb_mixed')
        self.assertEqual(stage46.methods_for(bl), [m for m in stage46.METHODS if not m.startswith('card')])
        syn = scene_list[0]
        vk = stage46.job(syn, 'vk_object', '/masks')
        self.assertTrue(vk['mask'].endswith('red_colour.png') and vk['reference_mask'].endswith('red_reference.png'))
        self.assertEqual(vk['search'], dict(stage46.SEARCH, scales=stage46.SYNTHETIC_SCALES, limit=stage46.LIMIT))
        card = stage46.job(syn, 'card_gray', '/masks')
        self.assertEqual(card['reference_rect'], stage46.CARD_RECTS['gray18'])
        self.assertNotIn('reference_mask', card)
        for scene in scene_list:
            for method in stage46.methods_for(scene):
                self.assertEqual(stage46.gt_problems(stage46.job(scene, method, '/masks')), [])
        self.assertEqual(stage46.gt_problems(dict(vk, truth=[])), ['job.truth'])
        self.assertEqual(len(stage46.gt_problems(dict(vk, search=dict(vk['search'], labels=1)))), 1)
        self.assertEqual(len(stage46.gt_problems(dict(vk, mask='/x/syn_red_neutral.truth.json'))), 1)

    def test_card_rects_inside_their_patches(self):
        x0, y0, x1, y1 = tb.REFERENCE_CARD['xyxy']
        step = (x1 - x0) // 3
        for k, name in enumerate(('white', 'gray18')):
            a, b, c, d = stage46.CARD_RECTS[name]
            self.assertTrue(x0 + k * step < a < c < x0 + (k + 1) * step and y0 < b < d < y1)

    def test_masks_come_from_the_artwork(self):
        with tempfile.TemporaryDirectory() as directory:
            records = stage46.make_masks(Path(directory))
            self.assertEqual(len(records), 4)
            import cv2
            for family in tb.FAMILIES:
                colour = cv2.imread(str(Path(directory) / f'{family}_colour.png'), cv2.IMREAD_GRAYSCALE) > 0
                reference = cv2.imread(str(Path(directory) / f'{family}_reference.png'), cv2.IMREAD_GRAYSCALE) > 0
                self.assertEqual(colour.shape, (tb.TEXTURE_H, tb.TEXTURE_W))
                self.assertFalse((colour & reference).any())
                fx0, fy0, fx1, fy1 = tb.COLOUR_FIELD
                ys, xs = np.nonzero(colour)
                self.assertTrue(xs.min() >= fx0 + 2 and xs.max() < fx1 - 2 and ys.min() >= fy0 + 2 and ys.max() < fy1 - 2)
                ys, xs = np.nonzero(reference)
                self.assertTrue(xs.min() >= 75 and xs.max() <= 122)
                texture = tb.capsule_texture(tb.FAMILIES[family])
                self.assertTrue(np.all(texture[reference] == stage46.PANEL))
                self.assertTrue(np.all(texture[colour] == tb.FAMILIES[family]))

    def test_dissimilarity_and_abstention(self):
        self.assertEqual(stage46.dissimilarity(.75, 'ncc'), .25)
        self.assertIsNone(stage46.dissimilarity(-2., 'ncc'))
        self.assertIsNone(stage46.dissimilarity(dict(status=3, region_distance=-1.), 'region_distance'))
        self.assertEqual(stage46.dissimilarity(dict(status=0, region_distance=.002), 'region_distance'), .002)


def fake_result(matches, method):
    if method in ('gray_ncc', 'rgb_ncc'):
        scores = [.9 for _ in matches]
    elif method.startswith('chroma'):
        scores = [dict(status=0, mean_distance=.01) for _ in matches]
    else:
        scores = [dict(status=0, mean_distance=.01, region_distance=.001, uncorrected_mean_distance=.02,
                       uncorrected_region_distance=.01, valid_fraction=1., clipped_pixels=0, reference_fraction=1.,
                       reference_clip_fraction=[0, 0, 0], gain=[1, 1, 1]) for _ in matches]
    return dict(exact_repeat=True, first=dict(search_matches=matches, scores=scores))


class Integrity(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        image = self.root / 'image.png'
        image.write_bytes(b'image')
        self.scene = dict(name='syn_red_wb_warm', source='synthetic', family='red', group='syn_red', condition='wb_warm',
                          image=str(image), template=str(image), truth=str(self.root / 'x.truth.json'), scales=[1.])
        truth = [dict(pos_x=10. + 100 * k, pos_y=20., angle=5., scale=1., index=k, same_colour=k == 0, direction='base' if k == 0 else 'hue+',
                      delta_e00_target=0. if k == 0 else 8., delta_e00_achieved=0., gray_texture_identical_to_base=True) for k in range(3)]
        Path(self.scene['truth']).write_text(json.dumps(dict(truth=truth, reference_card=dict(xyxy=list(tb.REFERENCE_CARD['xyxy'])))))
        # Two found objects (one exact, one within 2 px), one missed, one false positive.
        self.matches = [dict(pos_x=10., pos_y=20., angle=5., scale=1., score=.9), dict(pos_x=111., pos_y=20.5, angle=5.5, scale=1., score=.9),
                        dict(pos_x=500., pos_y=500., angle=0., scale=1., score=.8)]
        self.manifest = dict(executions=[], input_sha256={str(image): stage46.sha(image)}, complete=True, unchanged=True)
        for method in stage46.methods_for(self.scene):
            for kind in (('candidate', 'base') if method in stage46.EXISTING else ('candidate',)):
                self.write(method, kind, stage46.job(self.scene, method, self.root), fake_result(self.matches, method))

    def tearDown(self):
        self.directory.cleanup()

    def write(self, method, kind, job_dict, result, returncode=0):
        label = f'{method}-{kind}'
        jp, rp = self.root / 'jobs' / f'{label}.json', self.root / 'results' / f'{label}.json'
        stage46.save(jp, job_dict)
        stage46.save(rp, result)
        entry = dict(scene=self.scene['name'], method=method, runner_kind=kind, job=str(jp.relative_to(self.root)),
                     job_sha256=stage46.sha(jp), returncode=returncode)
        if not returncode:
            entry.update(result=str(rp.relative_to(self.root)), result_sha256=stage46.sha(rp))
        self.manifest['executions'] = [e for e in self.manifest['executions'] if (e['method'], e['runner_kind']) != (method, kind)] + [entry]

    def gates(self):
        return [f['gate'] for f in stage46.integrity(self.manifest, self.root, [self.scene])[1]]

    def test_clean_run_and_rows(self):
        results, failures = stage46.integrity(self.manifest, self.root, [self.scene])
        self.assertEqual(failures, [])
        rows, counts, fps = stage46.scene_rows(self.scene, json.loads(Path(self.scene['truth']).read_text()), results[self.scene['name']])
        self.assertEqual((counts['tp'], counts['fp'], counts['fn']), (2, 1, 1))
        self.assertEqual([r['found'] for r in rows], [True, True, False])
        self.assertAlmostEqual(rows[0]['values']['gray_ncc'], .1)
        self.assertEqual(rows[1]['values']['vk_region'], .001)
        self.assertEqual(rows[1]['values']['masked_region'], .01)
        self.assertEqual(rows[2]['values'], {})
        self.assertEqual([fp['prediction'] for fp in fps], [2])

    def test_localization_difference_between_methods_fails(self):
        moved = copy.deepcopy(self.matches)
        moved[0]['angle'] = 5.0001
        self.write('vk_object', 'candidate', stage46.job(self.scene, 'vk_object', self.root), fake_result(moved, 'vk_object'))
        self.assertEqual(self.gates(), ['localization_differs_between_methods'])

    def test_existing_method_change_fails(self):
        result = fake_result(self.matches, 'chroma_masked')
        result['first']['scores'][1]['mean_distance'] = .0100001
        self.write('chroma_masked', 'base', stage46.job(self.scene, 'chroma_masked', self.root), result)
        self.assertEqual(self.gates(), ['existing_method_changed'])

    def test_gt_in_job_fails(self):
        self.write('vk_object', 'candidate', dict(stage46.job(self.scene, 'vk_object', self.root), labels=[1]),
                   fake_result(self.matches, 'vk_object'))
        self.assertEqual(self.gates(), ['gt_in_job'])

    def test_inexact_repeat_and_failed_worker(self):
        result = fake_result(self.matches, 'card_gray')
        result['exact_repeat'] = False
        self.write('card_gray', 'candidate', stage46.job(self.scene, 'card_gray', self.root), result)
        self.write('rgb_ncc', 'candidate', stage46.job(self.scene, 'rgb_ncc', self.root), {}, returncode=1)
        self.assertEqual(sorted(self.gates()), ['exact_repeat', 'worker_failed'])

    def test_changed_input_and_incomplete_run(self):
        Path(self.scene['image']).write_bytes(b'other')
        self.manifest['unchanged'] = False
        self.assertEqual(self.gates(), ['input_changed', 'run_incomplete_or_binaries_changed'])

    def test_tampered_result_hash(self):
        path = self.root / 'results' / 'gray_ncc-candidate.json'
        path.write_text(path.read_text() + ' ')
        self.assertEqual(self.gates(), ['result_hash'])


if __name__ == '__main__':
    unittest.main()
