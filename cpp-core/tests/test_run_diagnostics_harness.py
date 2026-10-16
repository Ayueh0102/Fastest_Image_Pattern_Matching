"""Stage32 replay and negative controls; no native or Metal workload launched."""
import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from benchmark_fine_fused import CLOCKS, FUSION_COUNTS, FUSION_CLOCKS
from benchmark_run_diagnostics import (PHASES, BINARY_NAMES, baseline, ensure_empty_output,
    job_for, nonclock_state, pose_capacities_valid, resources_valid, stage_clocks,
    summarize, sha, save, worker_order, png_shape)


class RunDiagnosticsHarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.fixture = json.loads((Path(__file__).parent / 'fixtures/fine_search_harness.json').read_text())
        self.phase = 'reproduction'
        self.make_report(self.phase)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def resource_record():
        return dict(available=True, wall_ms=21., process_user_ms=9., process_system_ms=1.,
                    minor_faults=5, major_faults=0, voluntary_switches=8, involuntary_switches=1,
                    host_load_available=True, host_busy_ticks=100, host_idle_ticks=200)

    def reference(self):
        first = copy.deepcopy(self.fixture['first'])
        diagnostics, fusion = [], []
        for work in first['work']:
            fields = ('calls', 'requests', 'within_call_duplicates', 'cross_call_duplicates',
                      'untracked_call_keys', 'tracked_unique_keys', 'history_limit_keys',
                      'history_key_payload_bytes', 'history_bucket_count', 'peak_call_key_capacity_bytes',
                      'warp_batches', 'zero_repair_batches', 'guard_words', 'nonzero_guard_words')
            d = dict.fromkeys((*CLOCKS, *fields), 0)
            d.update(profiling_enabled=0, angle_reuse_enabled=1, warp_device_ms=-1,
                     logical_requests=work['fine_maps'], reused_maps=1, cache_peak_bytes=1024)
            work['fine_maps'] -= 1
            work['fine_commands'] = work['fine_stats_dispatches']
            work['fine_stats_device_ms'] = -1
            f = dict.fromkeys((*FUSION_COUNTS, *FUSION_CLOCKS), 0)
            f.update(enabled=1, fused_batches=work['fine_stats_dispatches'], changed_pixels=1,
                     corrected_windows=5, correction_ms=.01, fused_wait_ms=work['fine_wait_ms'], fused_device_ms=3.)
            diagnostics.append(d)
            fusion.append(f)
        first.update(fine_search=diagnostics, fine_fusion=fusion, total_ms=20.,
                     selected_refinement_backend=1, refinement_mode=0, pose_input_mode=0,
                     pose_storage_mode=0, pose_host_image_bytes=2132, pose_gpu_image_bytes=2132,
                     pose_shared_image_bytes=0, pose_unique_image_bytes=4264,
                     pose_host_bytes=2232, pose_gpu_bytes=2332, pose_conversion_temporary_bytes=2132,
                     pose_create_ms=2.)
        first['work'][0].update(fine_uploads=1, fine_upload_bytes=1024)
        return first

    def make_report(self, phase):
        self.phase = phase
        settings = PHASES[phase]
        cases = [dict(self.fixture['case'], name=name, expect_reuse=True, workflow='full_rescue',
                      source_shape=[32, 16], template_shape=[7, 3], benchmark_repeats=settings['repeats'])
                 for name in settings['cases']]
        self.case = cases[0]
        self.report = dict(version=1, phase=phase, methods=settings['methods'], blocks=2,
                           repeats=settings['repeats'], warm_seconds=0., memory_page_bytes=16384,
                           cases=cases, expected_workers=len(worker_order(cases, phase)),
                           execution_order=[], complete=False)
        save(self.base / 'protocol.json', self.report)
        self.report['protocol_sha256'] = sha(self.base / 'protocol.json')
        for name, method, block in worker_order(cases, phase):
            first = self.reference()
            runs = []
            for index in range(settings['repeats']):
                run = copy.deepcopy(first)
                run.update(generation=index + 2, total_ms=10. + index,
                           pose_create_ms=0., pose_conversion_temporary_bytes=0)
                run['work'][0].update(fine_uploads=0, fine_upload_bytes=0)
                runs.append(run)
            raw = dict(first=first, runs=runs, exact_repeat=True, construct_ms=1.,
                       warmups_ms=[], final_memory=self.fixture['final_memory'])
            if phase == 'reproduction' and method.startswith('probe_'):
                for run in [first, *runs]:
                    run['resources'] = self.resource_record()
            if phase == 'pose_memory':
                shared = method in ('shared', 'combined')
                direct = method in ('u8', 'combined')
                images = 32768 if shared else 2132
                for index, run in enumerate([first, *runs]):
                    run.update(pose_input_mode=int(direct), pose_storage_mode=int(shared),
                        pose_host_image_bytes=images, pose_gpu_image_bytes=images,
                        pose_shared_image_bytes=images if shared else 0,
                        pose_unique_image_bytes=images if shared else images * 2,
                        pose_host_bytes=images + 100, pose_gpu_bytes=images + 200,
                        pose_conversion_temporary_bytes=2132 if index == 0 and not direct else 0)
            direct = phase == 'pose_memory' or method.endswith('_direct')
            info = dict(direct_gray8=direct, rgb_temporary_bytes=0 if direct else 4096)
            raw['load'] = dict(mode='direct_gray8' if direct else 'legacy_rgb',
                               decode_ms=1. if direct else 2., source=copy.deepcopy(info), template=copy.deepcopy(info))
            raw['after_load_memory'] = dict(current_rss_bytes=1024, peak_rss_bytes=2048)
            job = self.base / f'{name}-{method}-{block}.json'
            result = job.with_suffix('.result.json')
            stderr = job.with_suffix('.stderr.txt')
            case = next(case for case in cases if case['name'] == name)
            save(job, job_for(case, method, phase))
            save(result, raw)
            stderr.write_text('')
            self.report['execution_order'].append(dict(case=name, method=method, block=block,
                job=job.name, job_sha256=sha(job), result=result.name, result_sha256=sha(result),
                stderr=stderr.name, stderr_sha256=sha(stderr), returncode=0))
        self.report.update(complete=True, unchanged=True)

    def mutate(self, method, fn, block=None, case=None):
        for entry in self.report['execution_order']:
            if (entry['method'] != method or (block is not None and entry['block'] != block) or
                    (case is not None and entry['case'] != case)):
                continue
            path = self.base / entry['result']
            raw = json.loads(path.read_text())
            fn(raw)
            save(path, raw)
            entry['result_sha256'] = sha(path)

    def summary(self):
        return summarize(self.report, self.base)

    def test_reproduction_valid_48_calls_and_relocatable(self):
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertTrue(result['all_performance_tolerances_passed'])
        self.assertEqual((result['workers'], result['algorithm_calls'], result['first_calls'],
                          result['repeat_calls'], result['warmup_calls']), (12, 48, 12, 36, 0))
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / 'moved'
            shutil.copytree(self.base, moved)
            self.assertEqual(result, summarize(self.report, moved))

    def test_pose_memory_valid_32_calls_and_relocatable(self):
        self.make_report('pose_memory')
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertTrue(result['all_performance_tolerances_passed'])
        self.assertEqual((result['workers'], result['algorithm_calls'], result['repeat_calls']), (16, 32, 16))
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / 'moved'
            shutil.copytree(self.base, moved)
            self.assertEqual(result, summarize(self.report, moved))

    def test_all_jobs_keep_solver_fusion_modes_and_no_truth(self):
        for phase, settings in PHASES.items():
            for method in settings['methods']:
                with self.subTest(phase=phase, method=method):
                    job = job_for(self.case, method, phase)
                    self.assertEqual((job['fine_ncc_mode'], job['pose_resident_mode']), (2, 0))
                    self.assertTrue(job['fine_fused'] and job['fine_angle_reuse'])
                    self.assertFalse(job['fine_profiling'] or job['overlap_enabled'])
                    self.assertEqual((job['repeats'], job['warmup_seconds']), (settings['repeats'], 0.))
                    self.assertFalse(set(job) & {'truth', 'expect_reuse', 'source_shape', 'template_shape'})
                    if method.startswith('frozen_'):
                        self.assertNotIn('run_resource_diagnostics', job)
                    else:
                        self.assertIs(job['run_resource_diagnostics'], method.startswith('probe_'))
        with self.assertRaises(ValueError):
            job_for(self.case, 'combined', 'reproduction')

    def test_phase_pose_modes_are_independent_two_by_two(self):
        for method, modes in [('current', (0, 0)), ('u8', (1, 0)), ('shared', (0, 1)), ('combined', (1, 1))]:
            job = job_for(self.case, method, 'pose_memory')
            self.assertEqual((job['pose_input_mode'], job['pose_storage_mode']), modes)
            self.assertEqual(job['gray_load_mode'], 'direct_gray8')

    def test_wrong_counts_and_hidden_warmups_rejected(self):
        original = copy.deepcopy(self.report)
        for mutate in [lambda raw: raw['runs'].pop(), lambda raw: raw['runs'].append(copy.deepcopy(raw['runs'][0])),
                       lambda raw: raw.update(warmups_ms=[1.])]:
            self.make_report('reproduction')
            self.mutate('current_direct', mutate)
            with self.assertRaises(ValueError):
                self.summary()
        self.report = original

    def test_job_hash_and_semantic_tamper_rejected(self):
        entry = self.report['execution_order'][0]
        path = self.base / entry['job']
        job = json.loads(path.read_text())
        job['pose_resident_mode'] = 2
        save(path, job)
        with self.assertRaises(ValueError):
            self.summary()
        entry['job_sha256'] = sha(path)
        with self.assertRaises(ValueError):
            self.summary()

    def test_raw_protocol_stderr_hashes_checked(self):
        entry = self.report['execution_order'][0]
        for filename in (entry['result'], entry['stderr'], 'protocol.json'):
            with self.subTest(filename=filename):
                path = self.base / filename
                original = path.read_bytes()
                path.write_text('{}')
                with self.assertRaises(ValueError):
                    self.summary()
                path.write_bytes(original)

    def test_duplicate_missing_reorder_failed_workers_rejected(self):
        for kind in ('duplicate', 'missing', 'reorder', 'failed'):
            with self.subTest(kind=kind):
                report = copy.deepcopy(self.report)
                if kind == 'duplicate': report['execution_order'][1] = report['execution_order'][0]
                elif kind == 'missing': report['execution_order'].pop()
                elif kind == 'reorder': report['execution_order'][:2] = report['execution_order'][1::-1]
                else: report['execution_order'][0]['returncode'] = 1
                with self.assertRaises(ValueError):
                    summarize(report, self.base)

    def test_changed_protocol_dimensions_and_phase_rejected(self):
        for field, value in [('phase', 'pose_memory'), ('repeats', 1), ('memory_page_bytes', 8192)]:
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report[field] = value
                with self.assertRaises(ValueError):
                    summarize(report, self.base)
        self.report['cases'][0]['source_shape'][0] += 1
        with self.assertRaises(ValueError):
            self.summary()

    def test_failure_retains_block_and_individual_repeat(self):
        self.mutate('current_legacy', lambda raw: raw['runs'][1].update(total_ms=30.), block=1)
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_performance_tolerances_passed'])
        self.assertFalse(result['uninstrumented_performance_tolerances_passed'])
        failure = result['performance_failures'][0]
        self.assertEqual((failure['block'], failure['failed_time_phases']), (1, ['repeat2']))
        pairs = result['cases'][0]['comparisons']['current_legacy_vs_frozen_legacy']
        self.assertTrue(pairs[0]['repeat_median']['time_tolerance_pass'])
        self.assertFalse(pairs[1]['calls'][2]['time_tolerance_pass'])
        self.assertTrue(pairs[1]['repeat_median']['time_tolerance_pass'])

    def test_probe_overhead_failure_is_diagnostic_only(self):
        self.mutate('probe_direct', lambda raw: raw['first'].update(total_ms=40.), block=0)
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_performance_tolerances_passed'])
        self.assertTrue(result['uninstrumented_performance_tolerances_passed'])
        self.assertTrue(all(f['diagnostic_only'] for f in result['performance_failures']))
        comparisons = result['cases'][0]['comparisons']
        self.assertIn('probe_direct_vs_current_direct', comparisons)
        self.assertTrue(result['cases'][0]['methods']['probe_direct']['diagnostic_only'])

    def test_rss_failure_retained(self):
        self.mutate('current_direct', lambda raw: raw['final_memory'].update(peak_rss_bytes=2 ** 32), block=0)
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        self.assertFalse(result['all_performance_tolerances_passed'])
        self.assertTrue(any(f['rss_failed'] for f in result['performance_failures']))

    def test_ordered_proposals_and_repeat_matches_strict(self):
        self.mutate('current_direct', lambda raw: raw['first']['proposals'].reverse())
        self.assertFalse(self.summary()['correctness_passed'])
        self.make_report('reproduction')
        self.mutate('current_legacy', lambda raw: raw['runs'][2]['matches'][0].update(score=.123))
        self.assertFalse(self.summary()['correctness_passed'])

    def test_ground_truth_failures_reported(self):
        self.mutate('current_direct', lambda raw: raw['first'].update(matches=[]))
        result = self.summary()
        self.assertFalse(result['correctness_passed'])
        self.assertGreater(result['cases'][0]['methods']['current_direct']['accuracy']['max_fn'], 0)

    def test_resource_enablement_missing_or_leaking_detected(self):
        self.mutate('probe_legacy', lambda raw: raw['first'].pop('resources'))
        self.assertFalse(self.summary()['correctness_passed'])
        self.make_report('reproduction')
        self.mutate('current_direct', lambda raw: raw['runs'][0].update(resources=self.resource_record()))
        self.assertFalse(self.summary()['correctness_passed'])

    def test_resource_unavailability_groups_are_independent(self):
        resource = self.resource_record()
        resource.update(available=False, process_user_ms=-1, process_system_ms=-1,
                        minor_faults=-1, major_faults=-1, voluntary_switches=-1, involuntary_switches=-1)
        self.assertTrue(resources_valid(dict(resources=resource), True))
        resource.update(host_load_available=False, host_busy_ticks=-1, host_idle_ticks=-1)
        self.assertTrue(resources_valid(dict(resources=resource), True))
        resource['minor_faults'] = 0
        self.assertFalse(resources_valid(dict(resources=resource), True))
        resource = self.resource_record()
        resource['host_load_available'] = False
        self.assertFalse(resources_valid(dict(resources=resource), True))

    def test_resource_negative_nonfinite_and_boolean_counters_invalid(self):
        for key, value in [('major_faults', -1), ('minor_faults', True), ('process_user_ms', float('nan')),
                           ('wall_ms', 0), ('available', 1)]:
            resource = self.resource_record()
            resource[key] = value
            self.assertFalse(resources_valid(dict(resources=resource), True))

    def test_resources_excluded_but_generation_work_capacities_exact(self):
        run = self.reference()
        probe = copy.deepcopy(run)
        probe['resources'] = self.resource_record()
        self.assertEqual(nonclock_state(probe), nonclock_state(run))
        for method, key in [('current_legacy', 'pose_gpu_bytes'), ('current_direct', 'generation')]:
            self.make_report('reproduction')
            self.mutate(method, lambda raw: raw['runs'][0].__setitem__(key, raw['runs'][0][key] + 1))
            self.assertFalse(self.summary()['correctness_passed'])

    def test_first_repeat_work_not_mistaken_for_drift(self):
        result = self.summary()
        self.assertTrue(result['correctness_passed'])
        blocks = result['cases'][0]['methods']['current_direct']['blocks']
        self.assertEqual(len(blocks[0]['calls']), 4)
        self.assertGreater(blocks[0]['calls'][0]['capacities']['pose_conversion_temporary_bytes'], 0)
        self.assertEqual(blocks[0]['calls'][1]['capacities']['pose_conversion_temporary_bytes'], 0)

    def test_gpu_unavailable_sentinel_is_not_summed(self):
        run = self.reference()
        run['work'].append(copy.deepcopy(run['work'][0]))
        run['fine_search'].append(copy.deepcopy(run['fine_search'][0]))
        run['fine_fusion'].append(copy.deepcopy(run['fine_fusion'][0]))
        clocks = stage_clocks(run)
        self.assertEqual(clocks['work']['sums']['fine_stats_device_ms'], -1)
        self.assertEqual(clocks['fine_search']['sums']['warp_device_ms'], -1)
        self.assertEqual(clocks['fine_fusion']['sums']['fused_device_ms'], 6.)
        run['fine_fusion'][0]['fused_device_ms'] = -1
        self.assertEqual(stage_clocks(run)['fine_fusion']['sums']['fused_device_ms'], -1)
        self.assertEqual(clocks['root']['total_ms'], 20.)
        self.assertEqual(clocks['root']['pose_create_ms'], 2.)
        self.assertNotIn('total', clocks)

    def test_pose_all_capacity_formulas_and_modes_checked(self):
        self.make_report('pose_memory')
        fields = ['pose_input_mode', 'pose_storage_mode', 'pose_host_image_bytes', 'pose_gpu_image_bytes',
                  'pose_unique_image_bytes', 'pose_shared_image_bytes', 'pose_host_bytes', 'pose_gpu_bytes',
                  'pose_conversion_temporary_bytes']
        for key in fields:
            self.make_report('pose_memory')
            self.mutate('combined', lambda raw: raw['first'].__setitem__(key, raw['first'][key] + 1), block=1)
            with self.subTest(key=key):
                self.assertFalse(self.summary()['correctness_passed'])

    def test_pose_page_rounding_separate_images(self):
        self.make_report('pose_memory')
        row = next(e for e in self.report['execution_order'] if e['method'] == 'shared')
        raw = json.loads((self.base / row['result']).read_text())
        self.assertEqual(raw['first']['pose_host_image_bytes'], 32768)
        self.assertTrue(pose_capacities_valid(raw['first'], self.reference(), 'shared', self.case, True, 16384))
        raw['first']['pose_unique_image_bytes'] = 16384
        self.assertFalse(pose_capacities_valid(raw['first'], self.reference(), 'shared', self.case, True, 16384))

    def test_pose_initialization_only_and_unchanged_frontend(self):
        for mutate in [lambda raw: raw['runs'][0].update(pose_conversion_temporary_bytes=2132),
                       lambda raw: raw['runs'][0].update(pose_create_ms=.1),
                       lambda raw: raw['first'].update(refinement_mode=2),
                       lambda raw: raw['first'].update(frame_fine_gpu_bytes=1),
                       lambda raw: raw['first']['work'][0].update(fine_commands=1)]:
            self.make_report('pose_memory')
            self.mutate('u8', mutate)
            self.assertFalse(self.summary()['correctness_passed'])

    def test_loader_metadata_missing_is_reportable(self):
        self.mutate('current_direct', lambda raw: raw.pop('load'))
        self.assertFalse(self.summary()['correctness_passed'])
        self.make_report('reproduction')
        self.mutate('current_legacy', lambda raw: raw['load']['source'].update(direct_gray8=True))
        self.assertFalse(self.summary()['correctness_passed'])

    def test_zero_time_or_negative_memory_rejected(self):
        self.mutate('frozen_legacy', lambda raw: raw['first'].update(total_ms=0.))
        with self.assertRaises(ValueError):
            self.summary()
        self.make_report('reproduction')
        self.mutate('current_direct', lambda raw: raw['final_memory'].update(current_rss_bytes=-1))
        with self.assertRaises(ValueError):
            self.summary()

    def test_frozen_binary_hashes_verify_stage31(self):
        build = self.base / 'baseline'
        protocol = self.base / 'baseline.json'
        records = {}
        for name in BINARY_NAMES:
            path = build / 'experiments' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            records['/prior/gray-loader/experiments/' + name] = sha(path)
        save(protocol, dict(binary_sha256=records))
        self.assertEqual(set(baseline(build, protocol)), set(BINARY_NAMES))
        (build / 'experiments' / BINARY_NAMES[0]).write_text('changed')
        with self.assertRaises(ValueError):
            baseline(build, protocol)

    def test_nonempty_output_refused(self):
        with self.assertRaises(ValueError): ensure_empty_output(self.base)
        empty = self.base / 'empty'
        ensure_empty_output(empty)
        empty.mkdir()
        ensure_empty_output(empty)
        file = self.base / 'file'
        file.write_text('existing')
        with self.assertRaises(ValueError): ensure_empty_output(file)

    def test_png_dimensions_read_without_decoding(self):
        path = self.base / 'input.png'
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 8192, 4096))
        self.assertEqual(png_shape(path), [8192, 4096])
        path.write_bytes(b'not an image')
        with self.assertRaises(ValueError): png_shape(path)


if __name__ == '__main__':
    unittest.main()
