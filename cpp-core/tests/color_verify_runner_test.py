"""Small native worker contract controls; no downloaded data or performance claims."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


RUNNER = None
METAL = False


def ppm(path, width, height, pixels):
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + bytes(pixels))


class WorkerContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="color-verify-test-")
        self.root = Path(self.directory.name)
        width, height, tw, th = 96, 48, 16, 12
        template = []
        for y in range(th):
            for x in range(tw):
                template.extend((50 + (17*x + 19*y) % 120,
                                 65 + (7*x + 13*y) % 110,
                                 80 + (11*x + 5*y) % 100))
        source = bytearray([10] * width * height * 3)
        for instance, (left, top) in enumerate(((5, 4), (37, 23), (72, 8))):
            for y in range(th):
                for x in range(tw):
                    for c in range(3):
                        source[((y+top)*width+x+left)*3+c] = template[(y*tw+x)*3+c] + (45 if instance == 2 and c == 0 else 0)
        ppm(self.root / "source.ppm", width, height, source)
        ppm(self.root / "template.ppm", tw, th, template)
        ppm(self.root / "empty.ppm", width, height, [10]*width*height*3)
        self.job = dict(source=str(self.root / "source.ppm"),
                        template=str(self.root / "template.ppm"),
                        method="chroma", backend="cpu", threads=1, warmups=1, repeats=2,
                        search=dict(backend=0, refinement_backend=0, scales=[1], angle=0,
                                    limit=10, score=.95, proposal_score=.95, min_area=256,
                                    stop_layer=0, refine=False, overlap=0))

    def tearDown(self):
        self.directory.cleanup()

    def worker(self, job, success=True):
        request = self.root / "job.json"
        request.write_text(json.dumps(job))
        result = subprocess.run([str(RUNNER), str(request)], text=True,
                                capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_integrated_candidates_and_color_offset(self):
        baseline = self.worker(dict(self.job, method="geometry_only"))
        rgb = self.worker(dict(self.job, method="rgb_ncc", threads=2))
        chroma = self.worker(self.job)
        gray = self.worker(dict(self.job, method="gray_ncc"))
        expected = baseline["first"]["search_matches"]
        self.assertEqual(len(expected), 3)
        self.assertEqual(baseline["color_construct_ms"], 0)
        for result in (baseline, rgb, chroma, gray):
            self.assertTrue(result["exact_repeat"])
            self.assertEqual(result["pose_count"], 3)
            self.assertEqual(result["first"]["search_matches"], expected)
            for run in [result["first"], *result["runs"]]:
                self.assertEqual(len(run["scores"]), 3)
                self.assertGreaterEqual(run["pipeline_ms"], run["search_ms"])
                self.assertEqual(run["wall_ms"], run["pipeline_ms"])
                self.assertGreater(run["memory"]["peak_rss_bytes"], 0)
        self.assertTrue(all(value > .99999 for value in rgb["first"]["scores"]))
        by_x = dict(zip((round(value["pos_x"]) for value in expected), chroma["first"]["scores"]))
        self.assertEqual(by_x[13]["mean_distance"], 0)
        self.assertEqual(by_x[45]["mean_distance"], 0)
        self.assertGreater(by_x[80]["mean_distance"], .04)
        self.assertEqual(baseline["first"]["verification_ms"], 0)

    def test_empty_search_stays_empty(self):
        for method in ("geometry_only", "rgb_ncc", "chroma"):
            result = self.worker(dict(self.job, source=str(self.root/"empty.ppm"), method=method))
            self.assertEqual(result["pose_count"], 0)
            self.assertEqual(result["first"]["scores"], [])
            self.assertEqual(result["first"]["search_matches"], [])

    def test_pose_only_geometry_and_invalid_support(self):
        job = {key: value for key, value in self.job.items() if key != "search"}
        job["poses"] = [[13-48, 24-10, 0, 0], [10000, 0, 0, 0]]
        result = self.worker(job)
        self.assertFalse(result["integrated_search"])
        self.assertEqual(result["first"]["scores"][0]["mean_distance"], 0)
        self.assertEqual(result["first"]["scores"][1]["status"], 2)
        self.assertEqual(result["first"]["search_ms"], 0)

    def test_rejects_ambiguous_and_truth_inputs(self):
        self.worker(dict(self.job, poses=[[0, 0, 0, 0]]), success=False)
        self.worker(dict(self.job, truth=[]), success=False)
        self.worker(dict(self.job, search=dict(self.job["search"], truth=[])), success=False)
        self.worker(dict(self.job, search=dict(self.job["search"], unsupported=True)), success=False)
        self.worker(dict(self.job, method="rgb_ncc", mask="absent.png"), success=False)

    def test_metal_rgb_smoke(self):
        if not METAL:
            self.skipTest("pass --metal to exercise available Metal hardware")
        cpu = self.worker(dict(self.job, method="rgb_ncc"))
        metal = self.worker(dict(self.job, method="rgb_ncc", backend="metal"))
        self.assertEqual(cpu["first"]["search_matches"], metal["first"]["search_matches"])
        for reference, device in zip(cpu["first"]["scores"], metal["first"]["scores"]):
            self.assertLess(abs(reference-device), 3e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--metal", action="store_true")
    args, remaining = parser.parse_known_args()
    RUNNER = args.runner.resolve()
    METAL = args.metal
    unittest.main(argv=[__file__, *remaining])
