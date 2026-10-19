"""Compare C++/Metal against independent NumPy two-pass float64 NCC and truth.

No OpenCV matching or GPU implementation is used as the reference. Include
fractional/border/odd-size/RGB poses, growth/reuse, low variance and invalid input.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pose_experiment import Scorer
from research_ncc_fourier import bilinear, channels, similarity_scene

ROOT = Path(__file__).resolve().parents[2]


def reference(source, template, poses):
    source, template = channels(source), channels(template)
    yy, xx = np.indices(template.shape[:2], dtype=np.float64)
    u, v = xx-template.shape[1]/2, template.shape[0]/2-yy
    centered = template-template.mean(axis=(0, 1), keepdims=True)
    energy_t = np.sum(centered**2)
    output = []
    for tx, ty, angle, alpha in poses:
        # Reproduce the public float32 geometry contract, but independently
        # calculate sampling and two-pass centering in float64.
        cs = float(np.float32(np.exp(float(alpha))*np.cos(np.deg2rad(float(angle)))))
        sn = float(np.float32(np.exp(float(alpha))*np.sin(np.deg2rad(float(angle)))))
        x = source.shape[1]/2+tx+cs*u-sn*v
        y = source.shape[0]/2-ty-(sn*u+cs*v)
        patch = np.stack([bilinear(source[..., c], x, y) for c in range(source.shape[2])], axis=2)
        patch -= patch.mean(axis=(0, 1), keepdims=True)
        energy = np.sum(patch**2)
        n = template.shape[0]*template.shape[1]
        output.append(-2 if min(energy_t, energy)/n <= 1e-12 else
                      float(np.sum(patch*centered)/np.sqrt(energy_t*energy)))
    return np.array(output)


def validate(build, metal=True):
    rng = np.random.default_rng(20261012)
    backends = [("cpu", 1), ("cpu", 8)] + ([("metal", 1)] if metal else [])
    results = []
    for shape, channel_count in [((17, 19), 1), ((65, 63), 3), ((128, 128), 1), ((257, 259), 3)]:
        h, w = shape
        source = rng.random((h+11, w+13, channel_count)).astype(np.float32)
        template = rng.random((h, w, channel_count)).astype(np.float32)
        poses = rng.uniform([-9, -8, -180, -.3], [9, 8, 180, .3], (37, 4)).astype(np.float32)
        expected = reference(source, template, poses)
        for backend, threads in backends:
            with Scorer(build, source, template, backend, threads) as scorer:
                # Exercise resize up/down then reuse (out-of-bounds or stale stats).
                for count in (1, 37, 3, 37):
                    scores, metrics = scorer.score(poses[:count])
                    error = float(np.max(np.abs(scores-expected[:count])))
                    assert error < 3e-5, (shape, channel_count, backend, error)
                results.append(dict(shape=shape, channels=channel_count, backend=backend,
                                    threads=threads, max_error=error, device=scorer.device))
    identities = []
    for rgb in (False, True):
        template = similarity_scene(128, rgb=rgb).astype(np.float32)
        source = similarity_scene(256, .87, 31.25, (5.5, -8.25), rgb=rgb).astype(np.float32)
        poses = np.array([[5.5, -8.25, 31.25, np.log(.87)], [0, 0, 0, 0],
                          [5.5, -8.25, 211.25, np.log(.87)]], np.float32)
        for backend, threads in backends:
            with Scorer(build, source, template, backend, threads) as scorer:
                scores, _ = scorer.score(poses)
                assert np.argmax(scores) == 0 and scores[0] > .9999, scores
                identities.append(dict(rgb=rgb, backend=backend, threads=threads, scores=scores.tolist()))
    degeneracy = []
    for backend, threads in backends:
        for image in (np.full((33, 35), .7, np.float32),
                      (.7+1e-4*rng.random((33, 35))).astype(np.float32)):
            with Scorer(build, image, image, backend, threads) as scorer:
                scores, _ = scorer.score(np.zeros((1, 4), np.float32))
                constant = np.ptp(image) == 0
                assert scores[0] == -2 if constant else abs(scores[0]-1) < 2e-5, scores
                degeneracy.append(dict(backend=backend, constant=bool(constant), score=float(scores[0])))
                try:
                    scorer.score([[0, 0, float('nan'), 0]])
                    raise AssertionError("non-finite pose accepted")
                except RuntimeError:
                    pass
        image = np.full((3, 3), np.nan, np.float32)
        try:
            Scorer(build, image, image, backend, threads)
            raise AssertionError("non-finite image accepted")
        except RuntimeError:
            pass
    flat_patches = []
    for backend, threads in backends:
        for size in (17, 65, 127):
            for amplitude in (.731, 1., 4.):
                image = np.zeros((256, 256), np.float32)
                start = (256-size)//2
                image[start:start+size, start:start+size] = amplitude
                template = rng.random((size, size)).astype(np.float32)
                pose = [[start+size/2-128, 128-(start+size/2), 0, 0]]
                with Scorer(build, image, template, backend, threads) as scorer:
                    scores, _ = scorer.score(pose)
                    assert scores[0] == -2, (backend, size, amplitude, scores)
                    assert scorer.memory()['host_buffer_bytes'] >= image.nbytes+template.nbytes
                    flat_patches.append(dict(backend=backend, size=size, amplitude=amplitude, score=float(scores[0])))
    return dict(tolerance=3e-5, random_pose_checks=results, independent_analytic_truth=identities,
                flat_patches_in_nonconstant_image=flat_patches,
                degeneracy=degeneracy, invalid_input_rejected=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "build/fourier-gpu")
    parser.add_argument("--no-metal", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.build, not args.no_metal)
    encoded = json.dumps(report, indent=2)+"\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
