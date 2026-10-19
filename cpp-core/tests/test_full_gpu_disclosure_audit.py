"""Archive-only audit behind the 2026-10-19 full-GPU disclosure."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import audit_full_gpu_disclosure as audit


def archive(path, members):
    with tarfile.open(path, 'w:gz') as tar:
        for name, value in members.items():
            data = json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def worker(first_pose, totals, formal_pose=0):
    return dict(first=dict(total_ms=500., pose_create_ms=first_pose),
                runs=[dict(total_ms=value, pose_create_ms=formal_pose) for value in totals])


class DisclosureAuditTest(unittest.TestCase):
    def pose(self, members):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'raw.tar.gz'
            archive(path, members)
            return audit.pose_context(path, 'formal')

    def test_first_call_pose_context_is_added_back(self):
        rows = self.pose({'formal/results/one-hybrid_best-b0.json': worker(20., [100., 102.]),
                          'formal/results/one-hybrid_best-b1.json': worker(30., [101., 103.]),
                          'formal/results/one-full_gpu_simd-b0.json': worker(0., [99., 101.]),
                          'formal/jobs/one-hybrid_best-b0.json': dict(ignored=True)})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['first_call_pose_create_ms'], [20., 30.])
        self.assertEqual((row['hybrid_median_ms'], row['first_call_pose_create_median_ms']), (101.5, 25.))
        simd = row['methods']['full_gpu_simd']
        self.assertAlmostEqual(simd['vs_hybrid_percent'], 100*(100/101.5-1))
        self.assertAlmostEqual(simd['vs_hybrid_with_pose_create_percent'], 100*(100/126.5-1))

    def test_formal_pose_rebuild_or_missing_field_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'rebuilt its pose context'):
            self.pose({'formal/results/one-hybrid_best-b0.json': worker(20., [100.], formal_pose=1.)})
        missing = worker(20., [100.])
        del missing['first']['pose_create_ms']
        with self.assertRaises(KeyError):
            self.pose({'formal/results/one-hybrid_best-b0.json': missing})

    def test_mode4_route_rule_and_stable_counts(self):
        def layer(width, height, angles, low=None):
            return dict(template_width=width, template_height=height, roi_cache_bytes=16384,
                        counts=dict(angles=dict(median=angles, min=angles if low is None else low, max=angles)))
        summary = dict(cases=[dict(name='mixed', methods=dict(full_gpu_simd=dict(effective_fine_modes=[4],
            fine_work=dict(layers=[layer(122, 122, 10), layer(123, 122, 3)]))))])
        row = audit.mode4_routes(summary)[0]
        self.assertEqual((row['simd32_angles'], row['fallback_angles'], row['simd32_windows']), (10, 3, 490))
        self.assertTrue(audit.simd32_route(264204, 1, 10**7))
        self.assertFalse(audit.simd32_route(264205, 1, 10**7))
        for change in (lambda case: case['methods']['full_gpu_simd'].update(effective_fine_modes=[2]),
                       lambda case: case['methods']['full_gpu_simd']['fine_work']['layers'].append(layer(10, 10, 5, low=4))):
            broken = json.loads(json.dumps(summary))
            change(broken['cases'][0])
            with self.assertRaises(ValueError):
                audit.mode4_routes(broken)

    def test_committed_archives_reproduce_disclosed_numbers(self):
        result = audit.audit()
        formal = {row['case']: row for row in result['pose_context']['40']['formal']}
        self.assertEqual([round(value, 2) for value in formal['many128_pose']['first_call_pose_create_ms']], [18.62, 19.36, 19.08])
        self.assertAlmostEqual(formal['many128_pose']['methods']['full_gpu_simd']['vs_hybrid_with_pose_create_percent'], -4.4, places=1)
        self.assertAlmostEqual(formal['many512_pose']['methods']['full_gpu_simd']['vs_hybrid_with_pose_create_percent'], 15.3, places=1)
        self.assertEqual(formal[audit.CASE_8K]['first_call_pose_create_median_ms'], 0)
        routes = {row['case']: row for row in result['mode4_routes']['profile-summary.json']}
        self.assertEqual((routes['many128_pose']['fallback_angles'], routes[audit.CASE_8K]['fallback_angles']), (0, 51))
        workload = result['workload_8k']
        self.assertEqual((workload['background_values'], workload['truth_angles'], workload['truth_scales']), ([127], [0], [1]))
        self.assertEqual(len(workload['truth']), 6)


if __name__ == '__main__':
    unittest.main()
