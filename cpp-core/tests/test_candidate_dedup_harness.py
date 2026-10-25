"""Negative controls for the stage-43 candidate-dedup harness."""
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import benchmark_candidate_dedup as bench

TRUTH = [dict(pos_x=50., pos_y=50., angle=0., scale=1.), dict(pos_x=150., pos_y=80., angle=30., scale=1.)]
CORNERS = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]


def pose(t, score=.95):
    return dict(pos_x=t['pos_x'], pos_y=t['pos_y'], angle=t['angle'], scale=t['scale'], score=score, corners=CORNERS)


def gpu_metrics(full):
    if not full:
        return dict(enabled=False)
    return dict(enabled=True, command_buffers=1, cpu_waits=1, intermediate_readbacks=0, cpu_algorithm_steps=0,
                gpu_bytes=4096, upload_bytes=64, readback_bytes=64, encode_ms=.5, wait_ms=5., device_ms=5.,
                device_timestamp_available=True)


def record(path, arm, found=TRUTH, total_ms=10., kept=None, score=.95):
    full, k = path == 'full_gpu', bench.ARMS[arm]['top_k']
    kept = (40 if k else 100) if kept is None else kept
    models = [] if full and not k else [dict(input_candidates=100, kept_candidates=kept, radius_px=40.8,
                                              angle_window_deg=19., filter_ms=0 if full or not k else .1)]
    result = dict(total_ms=total_ms, prepare_ms=1., proposal_ms=5., refine_ms=2., nms_ms=1., stage_timings_available=not full,
                  matches=[pose(t, score) for t in found], proposals=[pose(t, score) for t in found],
                  candidate_dedup=dict(top_k=k, available=bool(models), models=models,
                                       input_candidates=100 if models else None, kept_candidates=kept if models else None),
                  full_gpu=gpu_metrics(full), work=[dict(fine_maps=kept*10)])
    return result


def worker(path, arm, runs=1, warmups=0, rss=100*1024*1024, **options):
    first = record(path, arm, **options)
    return dict(first=first, runs=[copy.deepcopy(first) for _ in range(runs)], warmups_ms=[400.]*warmups,
                exact_repeat=True, final_memory=dict(peak_rss_bytes=rss))


def case(name='scene', scales=(1.,)):
    return dict(name=name, family='test', role='pass', truth=TRUTH, scales=list(scales), paths=list(bench.PATHS),
                source='s.png', template='t.png')


SCRATCH = None


def setUpModule():
    global SCRATCH
    SCRATCH = tempfile.TemporaryDirectory()


def tearDownModule():
    SCRATCH.cleanup()


class Harness:
    def __init__(self, phase, cases):
        self.root = Path(tempfile.mkdtemp(dir=SCRATCH.name))
        self.manifest = dict(phase=phase, cases=cases, executions=[], blocks=1, measured_calls=1)

    def add(self, name, path, arm, payload, block=0, competing=()):
        label = f'{name}-{path}-{arm}-b{block}'
        job, result = self.root/'jobs'/(label+'.json'), self.root/'results'/(label+'.json')
        bench.save(job, dict(label=label)); bench.save(result, payload)
        self.manifest['executions'].append(dict(case=name, path=path, arm=arm, block=block, returncode=0,
            job=str(job.relative_to(self.root)), job_sha256=bench.sha(job),
            result=str(result.relative_to(self.root)), result_sha256=bench.sha(result),
            host=dict(competing_processes=[dict(command=c, pcpu=1.) for c in competing])))


def gt_harness(overrides=None):
    h = Harness('gt', [case()])
    for path in bench.PATHS:
        for arm in bench.ARMS:
            options = (overrides or {}).get((path, arm), {})
            h.add('scene', path, arm, worker(path, arm, **options))
    return h


class GroundTruthPhase(unittest.TestCase):
    def test_clean_arms_are_gt_neutral(self):
        h = gt_harness()
        summary = bench.summarize_gt(h.manifest, h.root)
        self.assertEqual(summary['failures'], [])
        self.assertTrue(all(summary['decision']['gt_neutral'].values()))
        self.assertEqual(summary['aggregates']['hybrid']['k3']['kept_fraction'], .4)
        self.assertEqual(summary['aggregates']['hybrid']['k3']['fine_maps_fraction'], .4)

    def test_lost_object_is_a_regression(self):
        h = gt_harness({('full_gpu', 'k3'): dict(found=TRUTH[:1])})
        summary = bench.summarize_gt(h.manifest, h.root)
        self.assertFalse(summary['decision']['gt_neutral']['full_gpu'])
        self.assertTrue(summary['decision']['gt_neutral']['hybrid'])
        self.assertIn(dict(gate='gt_regression', case='scene', path='full_gpu', arm='k3', role='pass', tp=-1, fp=0, fn=1),
                      summary['failures'])

    def test_output_change_is_reported_apart_from_gt(self):
        h = gt_harness({('hybrid', 'k3'): dict(score=.94)})
        summary = bench.summarize_gt(h.manifest, h.root)
        cell = summary['cases'][0]['paths']['hybrid']['k3']['versus_off']
        self.assertFalse(cell['matches_identical'])
        self.assertFalse(cell['regression'])
        self.assertEqual(summary['failures'], [])

    def test_ablation_failure_marks_negative_control(self):
        h = Harness('gt', [case('sym_marker12')])
        for path in bench.PATHS:
            for arm in bench.ARMS:
                found = TRUTH[:1] if arm == 'k1_position_only' else TRUTH
                h.add('sym_marker12', path, arm, worker(path, arm, found=found))
        decision = bench.summarize_gt(h.manifest, h.root)['decision']['negative_controls']
        self.assertTrue(decision['k1_position_only']['hybrid']['demonstrated'])
        self.assertFalse(decision['k1_wide_radius']['hybrid']['demonstrated'])

    def test_counter_contracts(self):
        for overrides, message in (({('hybrid', 'k3'): dict(kept=101)}, 'kept more'),
                                   ({('hybrid', 'off'): dict(kept=99)}, 'Disabled dedup dropped')):
            h = gt_harness(overrides)
            with self.assertRaisesRegex(ValueError, message):
                bench.summarize_gt(h.manifest, h.root)
        h = gt_harness()
        payload = worker('full_gpu', 'off')
        payload['first']['candidate_dedup'].update(available=True, input_candidates=100)
        bench.save(h.root/h.manifest['executions'][len(bench.ARMS)]['result'], payload)
        h.manifest['executions'][len(bench.ARMS)]['result_sha256'] = bench.sha(h.root/h.manifest['executions'][len(bench.ARMS)]['result'])
        with self.assertRaisesRegex(ValueError, 'Disabled full-GPU'):
            bench.summarize_gt(h.manifest, h.root)

    def test_ignored_option_is_rejected(self):
        h = gt_harness()
        index = len(bench.ARMS)+1
        payload = worker('full_gpu', 'k3')
        payload['first']['candidate_dedup']['top_k'] = 0
        path = h.root/h.manifest['executions'][index]['result']
        bench.save(path, payload)
        h.manifest['executions'][index]['result_sha256'] = bench.sha(path)
        with self.assertRaisesRegex(ValueError, 'mismatched candidate_dedup.top_k'):
            bench.summarize_gt(h.manifest, h.root)

    def test_residency_change_is_rejected(self):
        h = gt_harness()
        index = len(bench.ARMS)+1
        payload = worker('full_gpu', 'k3')
        payload['first']['full_gpu']['cpu_waits'] = 2
        path = h.root/h.manifest['executions'][index]['result']
        bench.save(path, payload)
        h.manifest['executions'][index]['result_sha256'] = bench.sha(path)
        with self.assertRaises(ValueError):
            bench.summarize_gt(h.manifest, h.root)

    def test_changed_evidence_and_missing_arm(self):
        h = gt_harness()
        (h.root/h.manifest['executions'][0]['result']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Changed raw evidence'):
            bench.summarize_gt(h.manifest, h.root)
        h = gt_harness()
        h.manifest['executions'].pop()
        with self.assertRaisesRegex(ValueError, 'cover every'):
            bench.summarize_gt(h.manifest, h.root)

    def test_nonrepeating_worker_is_a_failure(self):
        h = gt_harness()
        payload = worker('hybrid', 'k2')
        payload['exact_repeat'] = False
        entry = next(e for e in h.manifest['executions'] if e['path'] == 'hybrid' and e['arm'] == 'k2')
        bench.save(h.root/entry['result'], payload)
        entry['result_sha256'] = bench.sha(h.root/entry['result'])
        self.assertIn(dict(gate='repeat', case='scene', path='hybrid', arm='k2'), bench.summarize_gt(h.manifest, h.root)['failures'])


def formal_harness(blocks=6, times=None, rss=None, competing=(), found=TRUTH):
    names = ('many128_pose', 'controlled6')
    h = Harness('formal', [case(name) for name in names])
    h.manifest.update(blocks=blocks, measured_calls=2)
    for name in names:
        for block in range(blocks):
            for path, arm in (bench.FORMAL_ARMS if block % 2 == 0 else bench.FORMAL_ARMS[::-1]):
                ms = (times or {}).get((name, path, arm), 100. if arm == 'off' else 80.)
                h.add(name, path, arm, worker(path, arm, runs=2, warmups=3, total_ms=ms,
                      rss=(rss or {}).get((name, path, arm), 100*1024*1024), found=found if arm != 'off' else TRUTH),
                      block, competing if block == 0 and arm == 'k3' else ())
    return h


class FormalPhase(unittest.TestCase):
    def test_clean_formal_passes(self):
        h = formal_harness(times={('controlled6', path, arm): 50. for path in bench.PATHS for arm in ('off', 'k3')})
        summary = bench.summarize_formal(h.manifest, h.root)
        self.assertEqual(summary['failures'], [])
        row = summary['cases'][0]['paths']['hybrid']['k3_versus_off']
        self.assertAlmostEqual(row['delta_percent'], -20.)
        self.assertTrue(row['consistent_direction'])

    def test_speed_target_time_rss_host_and_gt_gates(self):
        times = {('many128_pose', 'hybrid', 'k3'): 95.}
        times.update({('controlled6', path, 'k3'): 120. for path in bench.PATHS})
        h = formal_harness(times=times, rss={('controlled6', 'full_gpu', 'k3'): 200*1024*1024},
                           competing=('cmake',), found=TRUTH[:1])
        gates = {f['gate'] for f in bench.summarize_formal(h.manifest, h.root)['failures']}
        self.assertEqual(gates, {'speed_target', 'time_budget', 'rss', 'host', 'gt'})

    def test_order_and_measured_count_are_enforced(self):
        h = formal_harness()
        h.manifest['executions'][0], h.manifest['executions'][1] = h.manifest['executions'][1], h.manifest['executions'][0]
        with self.assertRaisesRegex(ValueError, 'order'):
            bench.summarize_formal(h.manifest, h.root)
        h = formal_harness()
        h.manifest['measured_calls'] = 3
        with self.assertRaisesRegex(ValueError, 'measured'):
            bench.summarize_formal(h.manifest, h.root)


class Jobs(unittest.TestCase):
    def test_every_job_key_is_accepted_by_the_runner(self):
        source = (Path(bench.__file__).resolve().parent/'native_pipeline_runner.cpp').read_text()
        keys = set(re.findall(r'"([a-z0-9_]+)"', source[source.index('jobKeys[]'):source.index('};', source.index('jobKeys[]'))]))
        std = dict(name='x', source='s.png', template='t.png', scales=[1.], angle=180, limit=8, score=.8, overlap=.2,
                   proposal_score=.5, stop_layer=1, subpixel=0, workflow='full_rescue', truth=TRUTH)
        native = dict(name='y', source='s.png', template='t.png', truth=TRUTH,
                      native_search=dict(backend=1, scales=[1.], angle=180, limit=300, min_area=512, ncc_mode=2))
        for c in (std, native, dict(std, extra=dict(scale_support_refine=True))):
            for path in bench.PATHS:
                for arm in bench.ARMS:
                    job = bench.make_job(c, path, arm, 1, 0.)
                    self.assertLessEqual(set(job), keys)
                    self.assertEqual(job['candidate_dedup_top_k'], bench.ARMS[arm]['top_k'])
                    self.assertEqual(job['full_gpu'], path == 'full_gpu')
                    self.assertNotIn('truth', job)

    def test_arms_and_ablations(self):
        self.assertEqual(bench.ARMS['off']['top_k'], 0)
        self.assertEqual(bench.ARMS[bench.PRIMARY], dict(top_k=3, radius=.5, angle_steps=3.))
        self.assertGreaterEqual(bench.ARMS['k1_position_only']['angle_steps'], 180)
        self.assertGreater(bench.ARMS['k1_wide_radius']['radius'], 1)


@unittest.skipUnless((bench.DATA_ROOT/'build/stage37-full-gpu/formal/manifest.json').exists(), 'frozen fixtures unavailable')
class Registry(unittest.TestCase):
    def test_every_family_loads_with_unique_names(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = bench.all_cases(bench.DATA_ROOT, Path(directory))
        families = {c['family'] for c in cases}
        self.assertEqual(families, {'r40', 'r34_challenge', 'r34_blender', 'r33', 'r34_scale', 'r35', 'r43_control'})
        names = {c['name'] for c in cases}
        self.assertEqual(len(names), len(cases))
        self.assertTrue({'many512_pose', 'many512_holdout', 'many512_pose_stop0', 'heldout_contact', 'blender_occlusion',
                         'degraded12', 'r35_heldout_clean+original', 'sym_marker12', 'adjacent_grid16'} <= names)
        self.assertTrue(all(c['paths'] == ('hybrid',) for c in cases if c['name'].endswith('+support')))
        self.assertEqual(sum(c['family'] == 'r34_scale' for c in cases), 66)
        self.assertFalse(any('fsc' in c['name'] or 'capsules' in c['name'] for c in cases))


if __name__ == '__main__':
    unittest.main()
