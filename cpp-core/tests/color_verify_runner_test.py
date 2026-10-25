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


def pgm(path, width, height, pixels):
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))


def linear(byte):
    value = byte / 255
    return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4


def encode(value):
    value = min(max(value, 0.), 1.)
    value = 12.92 * value if value <= .0031308 else 1.055 * value ** (1 / 2.4) - .055
    return int(round(255 * value))


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

    def illuminant_fixture(self):
        """16x12 label: colour columns 0..7, dark gap 8, off-white reference 9..15."""
        width, height, tw, th = 96, 48, 16, 12
        label = [(180 + 3*x, 70 + 2*y, 60 + x) if x < 8 else (40, 40, 40) if x == 8 else (230, 228, 220)
                 for y in range(th) for x in range(tw)]
        source = bytearray([10] * width * height * 3)
        def place(left, top, pixels):
            for k, rgb in enumerate(pixels):
                y, x = divmod(k, tw)
                source[((y+top)*width+x+left)*3:((y+top)*width+x+left)*3+3] = bytes(rgb)
        lit = lambda pixels, gain: [tuple(encode(linear(v)*g) for v, g in zip(rgb, gain)) for rgb in pixels]
        defect = [(r, g+40, b) if k % tw < 8 else (r, g, b) for k, (r, g, b) in enumerate(label)]
        bright = [(255, 255, 255) if k % tw > 8 else rgb for k, rgb in enumerate(label)]
        corners = ((5, 4), (37, 23), (72, 8), (72, 32))
        for (left, top), pixels in zip(corners, (label, lit(label, (1.2, 1, .8)), lit(defect, (1.2, 1, .8)), bright)):
            place(left, top, pixels)
        for y in range(0, 3):  # neutral card (linear .5) under the same white-balance shift
            for x in range(60, 66):
                source[(y*width+x)*3:(y*width+x)*3+3] = bytes(encode(.5*g) for g in (1.2, 1, .8))
        ppm(self.root / "label_source.ppm", width, height, source)
        ppm(self.root / "label.ppm", tw, th, [v for rgb in label for v in rgb])
        pgm(self.root / "colour.pgm", tw, th, [255 if k % tw < 8 else 0 for k in range(tw*th)])
        pgm(self.root / "white.pgm", tw, th, [255 if k % tw > 8 else 0 for k in range(tw*th)])
        poses = [[left + tw/2 - width/2, height/2 - top - th/2, 0, 0] for left, top in corners]
        return dict(source=str(self.root / "label_source.ppm"), template=str(self.root / "label.ppm"),
                    backend="cpu", threads=1, warmups=1, repeats=1, poses=poses,
                    mask=str(self.root / "colour.pgm"))

    def test_chroma_vk_white_balance_defect_and_saturation(self):
        job = self.illuminant_fixture()
        masked = self.worker(dict(job, method="chroma"))["first"]["scores"]
        result = self.worker(dict(job, method="chroma_vk", reference_mask=str(self.root / "white.pgm")))
        self.assertTrue(result["exact_repeat"])
        self.assertEqual(result["illuminant_config"]["clip_policy"], "rank")
        same, warm, defect, bright = result["first"]["scores"]
        self.assertEqual((same["status"], same["mean_distance"], same["gain"]), (0, 0, [1, 1, 1]))
        self.assertGreater(masked[1]["mean_distance"], .02)
        self.assertLess(warm["mean_distance"], .005)
        self.assertLess(warm["region_distance"], .003)
        self.assertGreater(warm["uncorrected_region_distance"], .02)
        self.assertGreater(defect["region_distance"], .04)
        self.assertEqual((bright["region_distance"], bright["uncorrected_region_distance"]), (-1, 0))
        self.assertAlmostEqual(warm["gain"][0] * 1.2, 1, delta=.01)
        self.assertAlmostEqual(warm["gain"][2] * .8, 1, delta=.01)
        self.assertGreater(defect["mean_distance"], .04)
        self.assertEqual((bright["status"], bright["mean_distance"], bright["gain"]), (3, -1, [0, 0, 0]))
        self.assertEqual(bright["uncorrected_mean_distance"], 0)
        self.assertEqual(bright["reference_clip_fraction"], [1, 1, 1])
        for record, chroma in zip(result["first"]["scores"], masked):
            self.assertEqual(record["uncorrected_mean_distance"], chroma["mean_distance"])
        none = self.worker(dict(job, method="chroma_vk", reference_mask=str(self.root / "white.pgm"),
                                reference_clip_policy="none"))["first"]["scores"]
        self.assertEqual(none[3]["status"], 0)
        card = self.worker(dict(job, method="chroma_vk", illuminant_reference="source_rect",
                                reference_rect=[60, 0, 66, 3]))
        self.assertEqual(card["illuminant_config"]["template_reference"], [1, 1, 1])
        scores = card["first"]["scores"]
        self.assertLess(scores[1]["mean_distance"], .005)
        self.assertEqual(scores[1]["reference_eligible"], 18)
        self.assertEqual(len({tuple(score["gain"]) for score in scores}), 1)

    def test_chroma_vk_integrated_keeps_search(self):
        job = {key: value for key, value in self.illuminant_fixture().items() if key not in ("poses", "mask")}
        job["search"] = self.job["search"]
        baseline = self.worker(dict(job, method="geometry_only"))
        result = self.worker(dict(job, method="chroma_vk", mask=str(self.root / "colour.pgm"),
                                  reference_mask=str(self.root / "white.pgm")))
        self.assertEqual(result["first"]["search_matches"], baseline["first"]["search_matches"])
        self.assertEqual(len(result["first"]["scores"]), len(baseline["first"]["search_matches"]))
        self.assertGreaterEqual(len(result["first"]["scores"]), 1)
        self.assertNotIn("illuminant_config", baseline)

    def test_chroma_vk_rejects_invalid_options(self):
        job = dict(self.illuminant_fixture(), method="chroma_vk")
        white = str(self.root / "white.pgm")
        self.worker(dict(job, method="chroma", reference_mask=white), success=False)
        self.worker(dict(job, method="chroma", reference_clip_policy="rank"), success=False)
        self.worker(job, success=False)  # object reference needs reference_mask
        self.worker(dict(job, reference_mask=white, reference_rect=[0, 0, 2, 2]), success=False)
        self.worker(dict(job, reference_mask=white, template_reference_rgb=[1, 1, 1]), success=False)
        self.worker(dict(job, reference_mask=white, reference_clip_policy="clip"), success=False)
        self.worker(dict(job, reference_mask=white, saturation_level=0), success=False)
        self.worker(dict(job, reference_mask=white, max_reference_clip_fraction=1), success=False)
        self.worker(dict(job, reference_mask=white, backend="metal"), success=False)
        self.worker(dict(job, reference_mask=white, threads=2), success=False)
        self.worker(dict(job, reference_mask=str(self.root / "empty.ppm")), success=False)
        self.worker(dict(job, illuminant_reference="source_rect"), success=False)
        self.worker(dict(job, illuminant_reference="source_rect", reference_rect=[0, 0, 97, 2]), success=False)
        self.worker(dict(job, illuminant_reference="source_rect", reference_rect=[0, 0, 2.5, 2]), success=False)
        self.worker(dict(job, illuminant_reference="card", reference_mask=white), success=False)

    def test_chroma_vk_linear_reference_model(self):
        job = dict(self.illuminant_fixture(), method="chroma_vk", reference_mask=str(self.root / "white.pgm"))
        constant = self.worker(job)
        linear = self.worker(dict(job, reference_model="linear"))
        self.assertEqual(constant["illuminant_config"]["reference_model"], "constant")
        self.assertEqual(linear["illuminant_config"]["reference_model"], "linear")
        for record in constant["first"]["scores"]:
            self.assertNotIn("reference_slope_u", record)
        same, warm, defect, bright = linear["first"]["scores"]
        self.assertEqual((same["status"], same["region_distance"], same["reference_slope_u"], same["reference_slope_v"]),
                         (0, 0, [0, 0, 0], [0, 0, 0]))
        self.assertLess(warm["region_distance"], .003)
        self.assertGreater(defect["region_distance"], .04)
        self.assertEqual((bright["status"], bright["region_distance"]), (3, -1))
        for a, b in zip(constant["first"]["scores"], linear["first"]["scores"]):
            self.assertEqual(a["uncorrected_region_distance"], b["uncorrected_region_distance"])
        self.worker(dict(job, reference_model="quadratic"), success=False)
        self.worker(dict(job, reference_model=1), success=False)
        self.worker(dict(job, method="chroma", reference_model="constant"), success=False)
        rect = {key: value for key, value in job.items() if key != "reference_mask"}
        self.worker(dict(rect, illuminant_reference="source_rect", reference_rect=[60, 0, 66, 3], reference_model="linear"),
                    success=False)

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
