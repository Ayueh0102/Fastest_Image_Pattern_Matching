"""Stage-47 harness controls: thresholds, decisions, gates and integrity checks without running the worker."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import benchmark_chroma_margin as stage47  # noqa: E402
import testbed_generate as tb  # noqa: E402

CATCH = dict(syn_red=.008, syn_teal=.0147, bl_red=.008)
DIFFERENT = [(d, t) for d in ('hue+', 'hue-', 'chroma-', 'isogray0', 'isogray90') for t in (2., 4., 8., 16.)] + \
            [('lightness+', 4.), ('lightness+', 8.)]
BLENDER_DIFFERENT = [('isogray90', 4.), ('chroma-', 8.), ('hue-', 8.), ('isogray0', 8.), ('hue+', 8.), ('chroma-', 4.),
                     ('lightness+', 8.), ('hue+', 4.)]
SCENES = [('v1', c) for c in ('neutral', 'exposure_dark', 'wb_cool', 'shading')] + \
         [('r47', c) for c in tb.CONDITIONS]
BLENDER_SCENES = [('v1', 'white'), ('r47', 'white'), ('r47', 'warm3200'), ('r47', 'mixed')]


def row(dataset, group, condition, same, direction='base', target=0., value=None, index=0):
    source = 'blender' if group == 'bl_red' else 'synthetic'
    if value is None:
        value = .001 if same else .002 * target + .01
    values = {arm: value for arm in stage47.ARMS if group != 'bl_red' or not arm.startswith('card')}
    for arm in stage47.NCC_ARMS:
        values[arm] = .01 if same else .02
    return dict(scene=f'{dataset}/{group}_{condition}', dataset=dataset, source=source, group=group,
                family=group.split('_')[1], condition=condition, split=stage47.split(dataset, condition), index=index,
                same_colour=same, direction=direction, delta_e00_target=target, delta_e00_achieved=target,
                gray_identical=direction.startswith('isogray'), found=True, prediction=index, values=values, diagnostics={})


def fixture():
    rows = []
    for group in ('syn_red', 'syn_teal'):
        for dataset, condition in SCENES:
            k = 0
            for _ in range(10):
                rows.append(row(dataset, group, condition, True, index=k)); k += 1
            for direction, target in DIFFERENT:
                rows.append(row(dataset, group, condition, False, direction, target, index=k)); k += 1
    for dataset, condition in BLENDER_SCENES:
        k = 0
        for _ in range(4):
            rows.append(row(dataset, 'bl_red', condition, True, index=k)); k += 1
        for direction, target in BLENDER_DIFFERENT:
            rows.append(row(dataset, 'bl_red', condition, False, direction, target, index=k)); k += 1
    return rows


def failed(checks):
    return [(c['name'], c.get('condition'), c.get('group')) for c in checks if not c['passed']]


def set_values(rows, arm, value, **where):
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    for r in hits:
        r['values'][arm] = value
    return hits


def evaluate(rows, rule='catch', arm=stage47.PRIMARY):
    return stage47.hypotheses(rows, stage47.thresholds(rows, rule, CATCH), arm)


class CatchThreshold(unittest.TestCase):
    def test_artwork_thresholds(self):
        catch = stage47.catch_thresholds()
        self.assertAlmostEqual(catch['syn_red']['tau'], .0081575, delta=1e-6)
        self.assertAlmostEqual(catch['syn_teal']['tau'], .0147011, delta=1e-6)
        self.assertEqual(catch['bl_red']['tau'], catch['syn_red']['tau'], 'Blender uses the same red artwork')
        # In-gamut means unclipped linear sRGB in [0, 1]; teal's circle leaves the gamut on 124 of 360 steps.
        self.assertEqual({group: value['circle_in_gamut'] for group, value in catch.items()}, dict(syn_red=360, syn_teal=236, bl_red=360))
        self.assertEqual(catch['syn_teal']['circle_min']['degrees'], 316)
        self.assertAlmostEqual(catch['syn_teal']['circle_min']['distance'], .0314943, delta=1e-6)
        for group, value in catch.items():
            self.assertEqual(value['tau'], .5 * min(value['circle_min']['distance'], value['variant_min']['distance']))
        # teal: an artwork variant (isogray0 at DeltaE 4) is closer than the constant-L* circle
        self.assertEqual((catch['syn_teal']['variant_min']['direction'], catch['syn_teal']['variant_min']['target']), ('isogray0', 4.))
        self.assertLess(catch['syn_teal']['variant_min']['distance'], catch['syn_teal']['circle_min']['distance'])

    def test_every_gated_variant_is_at_least_twice_tau(self):
        for family, base in tb.FAMILIES.items():
            tau = stage47.catch_thresholds()[f'syn_{family}']['tau']
            lin = lambda rgb: tb.srgb_to_linear(__import__('numpy').asarray(rgb, float) / 255.)
            for direction in stage47.GATED_DIRECTIONS:
                for target in (4., 8., 16.):
                    rgb = tb.colour_variant(__import__('numpy').array(base), direction, target)
                    self.assertGreaterEqual(stage47.region_chroma(lin(base), lin(rgb)), 2 * tau)

    def test_region_chroma_is_scale_invariant(self):
        self.assertEqual(stage47.region_chroma([.2, .3, .4], [.4, .6, .8]), 0)
        self.assertAlmostEqual(stage47.region_chroma([1, 0, 0], [0, 1, 0]), 1.)

    def test_catch_does_not_use_scenes(self):
        rows = fixture()
        set_values(rows, stage47.PRIMARY, .5, split='calibration', same_colour=True)
        table = stage47.thresholds(rows, 'catch', CATCH)
        self.assertEqual(table[(stage47.PRIMARY, 'syn_red')], CATCH['syn_red'])
        self.assertIsNone(table[('gray_ncc', 'syn_red')])
        self.assertIsNotNone(stage47.thresholds(rows, 'p95', CATCH)[('gray_ncc', 'syn_red')])


class Decisions(unittest.TestCase):
    def test_clean_fixture_passes(self):
        checks = evaluate(fixture())
        self.assertEqual(failed(checks), [])
        n = {(c['name'], c.get('condition'), c.get('group')): c['n'] for c in checks}
        self.assertEqual([n[('H1b gated different-colour rejection', c, None)] for c in stage47.GATED_SYNTHETIC], [30, 30, 30])
        self.assertEqual([n[('H1a decided same-colour acceptance', c, None)] for c in stage47.GATED_SYNTHETIC], [20, 20, 20])
        self.assertEqual(n[('H1c isogray rejection (pooled)', 'wb_warm+spot_warm+noise_jpeg', None)], 36)
        self.assertEqual([n[('H2b gated different-colour rejection', c, None)] for c in stage47.GATED_BLENDER], [7, 7])
        self.assertEqual([n[('F2 calibration gated different-colour rejection', None, g)] for g in stage47.GROUPS], [60, 60, 7])
        self.assertEqual([n[('F1 calibration same-colour within tau/2', None, g)] for g in stage47.GROUPS], [40, 40, 4])
        self.assertEqual(len(checks), 3 * 2 + 3 * 2 + 1 + 2 * 2)

    def test_threshold_is_inclusive(self):
        rows = fixture()
        set_values(rows, stage47.PRIMARY, CATCH['syn_red'], split='holdout', group='syn_red', same_colour=True)
        self.assertEqual(failed(evaluate(rows)), [])
        set_values(rows, stage47.PRIMARY, CATCH['syn_red'] * 1.0001, split='holdout', group='syn_red', condition='wb_warm',
                   same_colour=True)
        self.assertEqual(failed(evaluate(rows)), [('H1a decided same-colour acceptance', 'wb_warm', None)])

    def test_abstained_same_colour_excluded_from_decided_but_counts_end_to_end(self):
        rows = fixture()
        same = [r for r in rows if r['split'] == 'holdout' and r['condition'] == 'spot_warm' and r['same_colour']
                and r['source'] == 'synthetic']
        for r in same[:12]:
            r['values'][stage47.PRIMARY] = None
        self.assertEqual(failed(evaluate(rows)), [], '8 decided objects, all accepted')
        table = stage47.thresholds(rows, 'catch', CATCH)
        e2e = {c['condition']: c for c in stage47.end_to_end(rows, table, stage47.PRIMARY)}
        self.assertEqual((e2e['spot_warm']['value'], e2e['spot_warm']['abstained'], e2e['spot_warm']['passed']), (8 / 20, 12, False))
        result = stage47.cell([r for r in rows if r['group'] == 'syn_red' and r['split'] == 'holdout'
                               and r['condition'] == 'spot_warm'], stage47.PRIMARY, table)
        self.assertEqual((result['same']['decided'], result['same']['abstained'], result['same']['accepted']), (0, 10, 0))
        self.assertIsNone(result['same']['acceptance_decided'])

    def test_minimum_decided_count(self):
        rows = fixture()
        same = [r for r in rows if r['split'] == 'holdout' and r['condition'] == 'noise_jpeg' and r['same_colour']
                and r['source'] == 'synthetic']
        for r in same[:15]:
            r['values'][stage47.PRIMARY] = None
        self.assertEqual(failed(evaluate(rows)), [], '5 decided objects meet the minimum')
        same[15]['values'][stage47.PRIMARY] = None
        self.assertEqual(failed(evaluate(rows)), [('H1a decided same-colour acceptance', 'noise_jpeg', None)])
        rows = fixture()
        blender = [r for r in rows if r['split'] == 'holdout' and r['condition'] == 'mixed' and r['same_colour']]
        for r in blender[:2]:
            r['values'][stage47.PRIMARY] = None
        self.assertEqual(failed(evaluate(rows)), [])
        blender[2]['values'][stage47.PRIMARY] = None
        self.assertEqual(failed(evaluate(rows)), [('H2a decided same-colour acceptance', 'mixed', None)])

    def test_abstained_different_colour_is_never_accepted(self):
        rows = fixture()
        hits = set_values(rows, stage47.PRIMARY, None, split='holdout', condition='wb_warm', same_colour=False)
        table = stage47.thresholds(rows, 'catch', CATCH)
        self.assertFalse(any(stage47.accepted(r, stage47.PRIMARY, table) for r in hits))
        self.assertEqual(failed(evaluate(rows)), [])

    def test_wrong_accept_boundaries(self):
        rows = fixture()
        set_values(rows, stage47.PRIMARY, 0., split='holdout', group='syn_red', condition='wb_warm', direction='hue-',
                   delta_e00_target=4.)
        self.assertEqual(failed(evaluate(rows)), [], '29/30 is within .95')
        set_values(rows, stage47.PRIMARY, 0., split='holdout', group='syn_teal', condition='wb_warm', direction='chroma-',
                   delta_e00_target=16.)
        self.assertEqual(failed(evaluate(rows)), [('H1b gated different-colour rejection', 'wb_warm', None)])
        rows = fixture()
        set_values(rows, stage47.PRIMARY, 0., split='holdout', group='bl_red', condition='mixed', direction='hue+',
                   delta_e00_target=8.)
        self.assertEqual(failed(evaluate(rows)), [('H2b gated different-colour rejection', 'mixed', None)])

    def test_isogray_pooled_boundary(self):
        rows = fixture()
        set_values(rows, stage47.PRIMARY, 0., split='holdout', group='syn_red', condition='noise_jpeg', direction='isogray0',
                   delta_e00_target=16.)
        self.assertEqual(failed(evaluate(rows)), [])
        set_values(rows, stage47.PRIMARY, 0., split='holdout', group='syn_red', condition='spot_warm', direction='isogray90',
                   delta_e00_target=4.)
        self.assertIn('H1c isogray rejection (pooled)', [name for name, _, _ in failed(evaluate(rows))])

    def test_feasibility_on_calibration(self):
        rows = fixture()
        hit = [r for r in rows if r['split'] == 'calibration' and r['group'] == 'syn_teal' and r['same_colour']][0]
        hit['values'][stage47.PRIMARY] = CATCH['syn_teal'] / 2
        self.assertEqual(failed(evaluate(rows)), [], 'tau/2 itself is feasible')
        hit['values'][stage47.PRIMARY] = CATCH['syn_teal'] / 2 * 1.001
        self.assertEqual(failed(evaluate(rows)), [('F1 calibration same-colour within tau/2', None, 'syn_teal')])
        rows = fixture()
        set_values(rows, stage47.PRIMARY, 0., split='calibration', group='bl_red', direction='hue+', delta_e00_target=8.)
        self.assertEqual(failed(evaluate(rows)), [('F2 calibration gated different-colour rejection', None, 'bl_red')])

    def test_fresh_calibration_lighting_is_never_gated_or_calibrating(self):
        rows = fixture()
        set_values(rows, stage47.PRIMARY, 1., split='fresh_calibration')
        self.assertEqual(failed(evaluate(rows)), [])
        self.assertEqual(failed(evaluate(rows, 'rayleigh')), [])
        before = stage47.thresholds(fixture(), 'rayleigh', CATCH)
        self.assertEqual(stage47.thresholds(rows, 'rayleigh', CATCH), before)

    def test_gates_read_the_split_label(self):
        # A row's split decides; a gated condition name in another split is never gated (defensive: split()
        # never produces such rows).
        rows = fixture()
        extra = [dict(copy.deepcopy(r), split='fresh_calibration', scene='r47x/' + r['scene']) for r in rows
                 if r['split'] == 'holdout' and r['condition'] in ('wb_warm', 'mixed')]
        for r in extra:
            r['values'][stage47.PRIMARY] = 0. if not r['same_colour'] else 1.
        self.assertEqual(failed(evaluate(rows + extra)), [])

    def test_ungated_cells(self):
        rows = fixture()
        set_values(rows, stage47.PRIMARY, 0., direction='lightness+')
        set_values(rows, stage47.PRIMARY, 0., delta_e00_target=2.)
        set_values(rows, stage47.PRIMARY, 1., condition='exposure_bright')
        self.assertEqual(failed(evaluate(rows)), [])

    def test_secondary_rules(self):
        rows = fixture()
        values = [r for r in rows if r['split'] == 'calibration' and r['group'] == 'syn_red' and r['same_colour']]
        for k, r in enumerate(values):
            r['values'][stage47.PRIMARY] = .0001 * (k + 1)
        table = stage47.thresholds(rows, 'rayleigh', CATCH)
        squares = [(.0001 * (k + 1)) ** 2 for k in range(len(values))]
        expected = math.sqrt(sum(squares) / len(squares) / 2) * math.sqrt(-2 * math.log(1e-3))
        self.assertAlmostEqual(table[(stage47.PRIMARY, 'syn_red')], expected, delta=1e-15)
        import numpy as np
        p95 = stage47.thresholds(rows, 'p95', CATCH)[(stage47.PRIMARY, 'syn_red')]
        self.assertEqual(p95, float(np.percentile([.0001 * (k + 1) for k in range(len(values))], 95)))
        self.assertIsNone(table[('gray_ncc', 'syn_red')])
        holdout = set_values(rows, stage47.PRIMARY, .9, split='holdout', group='syn_red', same_colour=True)
        self.assertEqual(stage47.thresholds(rows, 'rayleigh', CATCH)[(stage47.PRIMARY, 'syn_red')], table[(stage47.PRIMARY, 'syn_red')])
        self.assertTrue(holdout)


class Gradient(unittest.TestCase):
    def test_sign_test(self):
        self.assertEqual(stage47.sign_test(5, 0), 1 / 32)
        self.assertEqual(stage47.sign_test(0, 3), 1.)
        self.assertEqual(stage47.sign_test(3, 2), (10 + 5 + 1) / 32)
        self.assertIsNone(stage47.sign_test(0, 0))

    def test_paired_uses_holdout_same_colour_decided_by_both(self):
        rows = fixture()
        same = set_values(rows, stage47.GRADIENT, .0005, split='holdout', condition='spot_warm', same_colour=True)
        set_values(rows, stage47.GRADIENT, .0005, split='holdout', condition='mixed', same_colour=True)
        set_values(rows, stage47.GRADIENT, .0001, split='holdout', condition='spot_warm', same_colour=False)
        set_values(rows, stage47.GRADIENT, .0001, split='fresh_calibration', same_colour=True)
        same[0]['values'][stage47.PRIMARY] = None
        same[1]['values'][stage47.GRADIENT] = None
        same[2]['values'][stage47.GRADIENT] = .001
        result = stage47.paired(rows, stage47.GRADIENT_CONDITIONS)
        self.assertEqual((result['n'], result['wins'], result['losses'], result['ties']), (20 + 4 - 2, 21, 0, 1))
        self.assertEqual(result['p_one_sided'], 1 / 2 ** 21)
        self.assertEqual(result['median_ratio'], .5)

    def test_paired_reads_the_split_label(self):
        rows = fixture()
        set_values(rows, stage47.GRADIENT, .0005, split='holdout', condition='spot_warm', same_colour=True)
        extra = [dict(copy.deepcopy(r), split='calibration') for r in rows if r['split'] == 'holdout' and r['condition'] == 'spot_warm']
        before = stage47.paired(rows, stage47.GRADIENT_CONDITIONS)
        self.assertEqual(stage47.paired(rows + extra, stage47.GRADIENT_CONDITIONS), before)
        extra = [dict(r, split='fresh_calibration') for r in extra]
        self.assertEqual(stage47.paired(rows + extra, stage47.GRADIENT_CONDITIONS), before)

    def test_gradient_preference_needs_its_own_checks(self):
        rows = fixture()
        set_values(rows, stage47.GRADIENT, .0005, split='holdout', same_colour=True)
        evaluation = stage47.evaluate_rows(rows, CATCH)
        self.assertTrue(evaluation['gradient_better_on_gradients'] and evaluation['gradient_checks_passed'])
        set_values(rows, stage47.GRADIENT, 0., split='holdout', group='syn_red', condition='noise_jpeg', direction='hue+')
        evaluation = stage47.evaluate_rows(rows, CATCH)
        self.assertFalse(evaluation['gradient_checks_passed'])
        self.assertTrue(evaluation['primary_passed'])


def manifest_fixture(root):
    """A miniature complete run: one synthetic and one Blender scene, every method, both runners."""
    scene_list = [dict(key='r47/syn_red_wb_warm', source='synthetic'), dict(key='r47/bl_rgb_mixed', source='blender')]
    executions = []
    (root / 'jobs').mkdir(parents=True)
    (root / 'results').mkdir()
    for scene in scene_list:
        for method in stage47.methods_for(scene):
            for kind in (('candidate', 'base') if method in stage47.EXISTING else ('candidate',)):
                label = f"{scene['key'].replace('/', '-')}-{method}-{kind}"
                stage47.save(root / 'jobs' / f'{label}.json', dict(method=method, source='/x/syn.png'))
                stage47.save(root / 'results' / f'{label}.json', dict(exact_repeat=True, first=dict(
                    scores=[dict(region_distance=.001)], search_matches=[dict(pos_x=1)])))
                executions.append(dict(scene=scene['key'], method=method, runner_kind=kind, returncode=0,
                                       job=f'jobs/{label}.json', job_sha256=stage47.sha(root / 'jobs' / f'{label}.json'),
                                       result=f'results/{label}.json', result_sha256=stage47.sha(root / 'results' / f'{label}.json')))
    return dict(executions=executions, input_sha256={}, complete=True, unchanged=True), scene_list


class Integrity(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='stage47-test-')
        self.root = Path(self.directory.name)
        self.manifest, self.scenes = manifest_fixture(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def reset(self):
        self.tearDown()
        self.setUp()

    def gates(self, manifest=None):
        return [f['gate'] for f in stage47.integrity(manifest or self.manifest, self.root, self.scenes)[1]]

    def entry(self, scene, method, kind='candidate'):
        return next(e for e in self.manifest['executions'] if e['scene'] == scene and e['method'] == method and e['runner_kind'] == kind)

    def rewrite(self, entry, key, value):
        path = self.root / entry[key]
        path.write_text(json.dumps(value))
        entry[key + '_sha256'] = stage47.sha(path)

    def test_clean(self):
        self.assertEqual(self.gates(), [])
        self.assertNotIn(('r47/bl_rgb_mixed', 'card_gray', 'candidate'),
                         {(e['scene'], e['method'], e['runner_kind']) for e in self.manifest['executions']})
        self.assertNotIn('vk_linear', {e['method'] for e in self.manifest['executions'] if e['runner_kind'] == 'base'})

    def test_existing_and_round46_methods_must_equal_base(self):
        for method in ('chroma_masked', 'vk_object', 'vk_object_none', 'card_gray'):
            self.reset()
            entry = self.entry('r47/syn_red_wb_warm', method, 'base')
            self.rewrite(entry, 'result', dict(exact_repeat=True, first=dict(scores=[dict(region_distance=.002)],
                                                                             search_matches=[dict(pos_x=1)])))
            self.assertEqual(self.gates(), ['existing_method_changed'], method)

    def test_localization_must_be_shared(self):
        entry = self.entry('r47/bl_rgb_mixed', 'vk_linear')
        self.rewrite(entry, 'result', dict(exact_repeat=True, first=dict(scores=[dict(region_distance=.001)],
                                                                         search_matches=[dict(pos_x=2)])))
        self.assertEqual(self.gates(), ['localization_differs_between_methods'])

    def test_gt_in_job(self):
        entry = self.entry('r47/syn_red_wb_warm', 'vk_linear')
        self.rewrite(entry, 'job', dict(method='chroma_vk', truth=[]))
        self.assertEqual(self.gates(), ['gt_in_job'])
        self.rewrite(entry, 'job', dict(method='chroma_vk', source='/x/blender/bl_rgb_mixed.json'))
        self.assertEqual(self.gates(), ['gt_in_job'])

    def test_repeat_hashes_and_completion(self):
        entry = self.entry('r47/syn_red_wb_warm', 'vk_object')
        self.rewrite(entry, 'result', dict(exact_repeat=False, first=dict(scores=[dict(region_distance=.001)],
                                                                          search_matches=[dict(pos_x=1)])))
        self.assertEqual(self.gates(), ['exact_repeat'])
        self.reset()
        (self.root / self.entry('r47/syn_red_wb_warm', 'vk_linear')['result']).write_text('{}')
        self.assertEqual(self.gates(), ['result_hash'])
        self.reset()
        (self.root / self.entry('r47/syn_red_wb_warm', 'vk_linear')['job']).write_text('{}')
        self.assertEqual(self.gates(), ['job_hash'])
        self.reset()
        self.assertEqual(self.gates(dict(self.manifest, unchanged=False)), ['run_incomplete_or_binaries_changed'])
        self.assertEqual(self.gates(dict(self.manifest, complete=False)), ['run_incomplete_or_binaries_changed'])
        path = self.root / 'input.png'
        path.write_bytes(b'a')
        self.assertEqual(self.gates(dict(self.manifest, input_sha256={str(path): stage47.sha(path)})), [])
        self.assertEqual(self.gates(dict(self.manifest, input_sha256={str(path): '0' * 64})), ['input_changed'])
        self.manifest['executions'] = [e for e in self.manifest['executions']
                                       if not (e['method'] == 'vk_linear' and e['scene'] == 'r47/bl_rgb_mixed')]
        self.assertEqual(self.gates(), ['missing_worker'])


class Holdout(unittest.TestCase):
    SEEDS = dict(synthetic=dict(texture=tb.SEEDS['texture'], layout=dict(red=11, teal=12), noise=13, background=14),
                 blender=dict(cycles=15, order=16, layout=17))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='stage47-holdout-test-')
        root = self.root = Path(self.directory.name)
        for dataset in ('v1', 'r47'):
            (root / dataset / 'textures').mkdir(parents=True)
            (root / dataset / 'textures' / 'red_base.png').write_bytes(b'art')
        (root / 'r47' / 'synthetic').mkdir()
        (root / 'r47' / 'blender').mkdir()
        stage47.save(root / 'r47' / 'synthetic' / 'synthetic.json', dict(seeds=self.SEEDS['synthetic']))
        stage47.save(root / 'r47' / 'blender' / 'blender_generation.json', dict(seed=15, rgb_order_seed=16, rgb_layout_seed=17))
        stage47.save(root / 'r47' / 'holdout.json', dict(seeds=self.SEEDS))
        self.manifest = dict(holdout=str(root / 'r47'), v1=str(root / 'v1'), holdout_json_sha256=stage47.sha(root / 'r47' / 'holdout.json'))
        self.protocol = dict(data=dict(holdout_seeds=copy.deepcopy(self.SEEDS), used_seeds=[2026102402, 46000001]))

    def tearDown(self):
        self.directory.cleanup()

    def reset(self):
        self.tearDown()
        self.setUp()

    def gates(self):
        return [f['gate'] for f in stage47.holdout_failures(self.manifest, self.protocol)]

    def test_clean(self):
        self.assertEqual(self.gates(), [])

    def test_reused_or_duplicated_seed(self):
        self.protocol['data']['used_seeds'].append(13)
        self.assertEqual(self.gates(), ['holdout_seed_reused'])
        self.protocol['data']['used_seeds'].pop()
        self.protocol['data']['holdout_seeds']['blender']['order'] = 15
        self.assertIn('holdout_seed_reused', self.gates())

    def test_seed_records_must_match(self):
        self.protocol['data']['holdout_seeds']['synthetic']['noise'] = 99
        self.assertEqual(self.gates(), ['holdout_seeds_differ_from_protocol', 'synthetic_seeds_differ_from_protocol'])
        self.reset()
        stage47.save(self.root / 'r47' / 'blender' / 'blender_generation.json', dict(seed=15, rgb_order_seed=17, rgb_layout_seed=16))
        self.assertEqual(self.gates(), ['blender_seeds_differ_from_protocol'])

    def test_artwork_and_record(self):
        (self.root / 'r47' / 'textures' / 'red_base.png').write_bytes(b'other')
        self.assertEqual(self.gates(), ['artwork_differs_from_v1'])
        self.reset()
        (self.root / 'v1' / 'textures' / 'extra.png').write_bytes(b'x')
        self.assertEqual(self.gates(), ['artwork_differs_from_v1'])
        self.reset()
        stage47.save(self.root / 'r47' / 'holdout.json', dict(seeds=self.SEEDS, note='edited'))
        self.assertEqual(self.gates(), ['holdout_record_changed'])


class Verdict(unittest.TestCase):
    def test_verdicts_need_integrity_and_every_check(self):
        good = dict(primary_passed=True, gradient_checks_passed=True, gradient_better_on_gradients=True)
        self.assertEqual(stage47.verdict(True, good), dict(recommended=True, gradient_preferred=True))
        self.assertEqual(stage47.verdict(False, good), dict(recommended=False, gradient_preferred=False))
        self.assertEqual(stage47.verdict(True, dict(good, primary_passed=False)), dict(recommended=False, gradient_preferred=True))
        self.assertFalse(stage47.verdict(True, dict(good, gradient_checks_passed=False))['gradient_preferred'])
        self.assertFalse(stage47.verdict(True, dict(good, gradient_better_on_gradients=False))['gradient_preferred'])

    def test_catch_thresholds_must_equal_the_protocol(self):
        catch = {g: dict(tau=v) for g, v in CATCH.items()}
        protocol = dict(thresholds=dict(catch_tau={g: dict(tau=v) for g, v in CATCH.items()}))
        self.assertEqual(stage47.catch_failures(protocol, catch), [])
        protocol['thresholds']['catch_tau']['syn_teal']['tau'] += 1e-9
        self.assertEqual([f['gate'] for f in stage47.catch_failures(protocol, catch)], ['catch_threshold_differs_from_protocol'])
        del protocol['thresholds']['catch_tau']['syn_teal']
        self.assertEqual(len(stage47.catch_failures(protocol, catch)), 1)
        frozen = json.loads((Path(__file__).resolve().parents[1] / 'docs/stage47-chroma-margin/protocol.json').read_text()) \
            if (Path(__file__).resolve().parents[1] / 'docs/stage47-chroma-margin/protocol.json').exists() else None
        if frozen:
            self.assertEqual(stage47.catch_failures(frozen, stage47.catch_thresholds()), [])

    def test_masks_must_equal_stage46(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'red_colour.png').write_bytes(b'm')
            protocol = dict(masks=dict(files={'red_colour.png': stage47.sha(Path(directory) / 'red_colour.png')}))
            self.assertEqual(stage47.mask_failures(protocol, directory), [])
            (Path(directory) / 'red_colour.png').write_bytes(b'n')
            self.assertEqual([f['gate'] for f in stage47.mask_failures(protocol, directory)], ['mask_differs_from_stage46'])
            protocol['masks']['files']['teal_colour.png'] = '0' * 64
            self.assertEqual(len(stage47.mask_failures(protocol, directory)), 2)


class Plan(unittest.TestCase):
    def test_scenes_and_splits(self):
        scene_list = stage47.scenes('/v1', '/r47')
        self.assertEqual(len(scene_list), 9 + 19)
        self.assertEqual(sum(s['split'] == 'calibration' for s in scene_list), 9)
        self.assertEqual(sum(s['split'] == 'holdout' for s in scene_list), 10)
        self.assertEqual(sum(s['split'] == 'fresh_calibration' for s in scene_list), 9)
        self.assertTrue(all(str(s['template']).startswith('/v1/') for s in scene_list), 'templates always come from v1')
        self.assertTrue(all(str(s['image']).startswith('/v1/') == (s['dataset'] == 'v1') for s in scene_list))
        for condition in stage47.HOLDOUT:
            with self.assertRaises(ValueError):
                stage47.split('v1', condition)

    def test_jobs_carry_no_gt_and_the_linear_model(self):
        for scene in stage47.scenes('/v1', '/r47'):
            for method in stage47.methods_for(scene):
                job = stage47.job(scene, method, '/masks')
                self.assertEqual(stage47.gt_problems(job), [])
                self.assertEqual(job.get('reference_model'), 'linear' if method == 'vk_linear' else None)
                self.assertEqual(job['search']['limit'], stage47.LIMIT)
        self.assertEqual(stage47.EXISTING, ('gray_ncc', 'rgb_ncc', 'chroma', 'chroma_masked', 'vk_object', 'vk_object_none', 'card_gray'))

    def test_inputs_checked_against_their_own_manifest(self):
        testbeds = [(Path('/v1'), {'synthetic/a.png': 'x'}), (Path('/r47'), {'synthetic/a.png': 'y'})]
        stage47.check_inputs({'/v1/synthetic/a.png': 'x', '/r47/synthetic/a.png': 'y', '/masks/m.png': 'z'}, testbeds)
        with self.assertRaises(ValueError):
            stage47.check_inputs({'/r47/synthetic/a.png': 'x'}, testbeds)


if __name__ == '__main__':
    unittest.main()
