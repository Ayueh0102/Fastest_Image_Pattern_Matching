"""Numerical research checks for NCC, rotation, scale and RGB.

NumPy FFT is a research backend, not part of the production C++ matcher.
Coordinates in the similarity experiment are x-right/y-up; positive angles are
counterclockwise. It uses independent analytic Gaussian scenes, not the matching
code's image warp. No GPU speedup or full-image detection claim is made here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def channels(image):
    image = np.asarray(image, dtype=np.float64)
    return image[..., None] if image.ndim == 2 else image


def ncc(a, b):
    a, b = channels(a), channels(b)
    a = a - a.mean(axis=(0, 1), keepdims=True)
    b = b - b.mean(axis=(0, 1), keepdims=True)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def fft_corr(image, kernel):
    shape = tuple(i + k - 1 for i, k in zip(image.shape, kernel.shape))
    spectrum = np.fft.rfft2(image, s=shape) * np.conj(np.fft.rfft2(kernel, s=shape))
    result = np.fft.irfft2(spectrum, s=shape)
    return result[:image.shape[0] - kernel.shape[0] + 1,
                  :image.shape[1] - kernel.shape[1] + 1]


def fft_ncc(image, template, mask):
    image, template = channels(image), channels(template)
    count = mask.sum()
    shape = (image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1)
    numerator, energy_i = np.zeros(shape), np.zeros(shape)
    energy_t = 0.0
    for channel in range(image.shape[2]):
        t, i = template[..., channel], image[..., channel]
        centered = (t - np.sum(mask * t) / count) * mask
        numerator += fft_corr(i, centered)
        sums, squares = fft_corr(i, mask), fft_corr(i * i, mask)
        energy_i += np.maximum(0, squares - sums * sums / count)
        energy_t += np.sum(centered * centered)
    return numerator / np.sqrt(energy_i * energy_t)


def direct_ncc_map(image, template, mask):
    image, template = channels(image), channels(template)
    select = mask.astype(bool)
    t = template[select]
    t = t - t.mean(axis=0)
    result = np.empty((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))
    for y, x in np.ndindex(result.shape):
        patch = image[y:y + template.shape[0], x:x + template.shape[1]][select]
        patch = patch - patch.mean(axis=0)
        result[y, x] = np.sum(patch * t) / np.sqrt(np.sum(patch * patch) * np.sum(t * t))
    return result


def bilinear(image, x, y):
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - ix, y - iy
    result = np.zeros(x.shape, np.float64)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        xx, yy = ix + ox, iy + oy
        valid = (xx >= 0) & (yy >= 0) & (xx < image.shape[1]) & (yy < image.shape[0])
        result[valid] += image[yy[valid], xx[valid]] * weight[valid]
    return result


def log_polar_magnitude(image, angular=720, radial=256):
    spectrum = np.fft.fftshift(np.fft.fft2(channels(image), axes=(0, 1)), axes=(0, 1))
    magnitude = np.sqrt(np.sum(np.abs(spectrum)**2, axis=2))
    # Real-image magnitude has a pi period. Preserve both spatial hypotheses.
    angles = np.arange(angular) * np.pi / angular
    log_radii = np.linspace(np.log(3), np.log(min(image.shape[:2]) / 3), radial)
    radius = np.exp(log_radii)
    x = image.shape[1] // 2 + np.cos(angles[:, None]) * radius[None, :]
    y = image.shape[0] // 2 - np.sin(angles[:, None]) * radius[None, :]
    polar = bilinear(magnitude, x, y)
    polar = np.log(np.maximum(polar, magnitude.max() * 1e-8))
    # Remove the shared isotropic radial profile; this is preprocessing, not an
    # assertion that finite sampled Fourier magnitudes obey an exact shift law.
    polar -= polar.mean(axis=0, keepdims=True)
    return polar, float(log_radii[1] - log_radii[0]), float(np.pi / angular)


def phase_shift(reference, moving, pad_radial=False):
    shape = (reference.shape[0], reference.shape[1] * (2 if pad_radial else 1))
    spectrum = np.sum(np.conj(np.fft.fft2(channels(reference), s=shape, axes=(0, 1))) *
                      np.fft.fft2(channels(moving), s=shape, axes=(0, 1)), axis=2)
    spectrum /= np.maximum(np.abs(spectrum), np.max(np.abs(spectrum)) * 1e-12)
    response = np.fft.ifft2(spectrum).real
    peak = np.unravel_index(np.argmax(response), response.shape)
    shift = []
    for axis, location in enumerate(peak):
        left, right = list(peak), list(peak)
        left[axis] = (location - 1) % shape[axis]
        right[axis] = (location + 1) % shape[axis]
        a, b, c = response[tuple(left)], response[peak], response[tuple(right)]
        offset = .5 * (a - c) / (a - 2*b + c) if abs(a - 2*b + c) > 1e-15 else 0
        value = location + float(np.clip(offset, -.5, .5))
        shift.append(value if value <= shape[axis]/2 else value - shape[axis])
    return shift


def analytic_pattern(x, y, gradient=False):
    output = np.zeros_like(x, dtype=float)
    gx, gy = np.zeros_like(output), np.zeros_like(output)
    for cx, cy, sx, sy, angle, amplitude in ((-30, 18, 7, 18, .3, 1), (22, 25, 15, 5, -.8, .8),
                                           (28, -23, 8, 12, 1.2, .7), (-12, -29, 6, 6, 0, .9),
                                           (4, 3, 3, 15, .8, .6)):
        u = np.cos(angle)*(x-cx) + np.sin(angle)*(y-cy)
        v = -np.sin(angle)*(x-cx) + np.cos(angle)*(y-cy)
        value = amplitude * np.exp(-.5*((u/sx)**2 + (v/sy)**2))
        output += value
        gx += value*(-u*np.cos(angle)/sx**2 + v*np.sin(angle)/sy**2)
        gy += value*(-u*np.sin(angle)/sx**2 - v*np.cos(angle)/sy**2)
    return (output, gx, gy) if gradient else output


def similarity_scene(size, scale=1, angle=0, shift=(0, 0), rgb=False):
    yy, xx = np.indices((size, size), dtype=float)
    x, y = xx - size/2 - shift[0], size/2 - yy - shift[1]
    theta = np.deg2rad(angle)
    u = (np.cos(theta)*x + np.sin(theta)*y)/scale
    v = (-np.sin(theta)*x + np.cos(theta)*y)/scale
    if rgb:
        return np.stack((analytic_pattern(u, v), .8*analytic_pattern(u+9, v-7),
                         .7*analytic_pattern(u-6, v+11)), axis=2)
    return analytic_pattern(u, v)


def transform_template(template, scale, angle, shift=(0, 0)):
    yy, xx = np.indices(template.shape[:2], dtype=float)
    x, y = xx-template.shape[1]/2-shift[0], template.shape[0]/2-yy-shift[1]
    theta = np.deg2rad(angle)
    u = (np.cos(theta)*x + np.sin(theta)*y)/scale + template.shape[1]/2
    v = template.shape[0]/2 - (-np.sin(theta)*x + np.cos(theta)*y)/scale
    result = np.stack([bilinear(channels(template)[..., c], u, v)
                       for c in range(channels(template).shape[2])], axis=2)
    return result[..., 0] if template.ndim == 2 else result


def estimate_similarity(template, source):
    reference, du, dphi = log_polar_magnitude(template)
    moving, _, _ = log_polar_magnitude(source)
    shifts = phase_shift(reference, moving, pad_radial=True)
    estimate_scale = float(np.exp(-shifts[1]*du))
    base_angle = float(np.rad2deg(shifts[0]*dphi))
    # Each magnitude solution has theta and theta+180 possibilities. Recover
    # translation with spatial correlation, then verify each with spatial NCC.
    hypotheses = []
    for candidate in (base_angle, base_angle + 180):
        transformed = transform_template(template, estimate_scale, candidate)
        # Sum channel cross spectra before normalizing their joint phase.
        delta_y, delta_x = phase_shift(transformed, source)
        shifted = transform_template(template, estimate_scale, candidate, (delta_x, -delta_y))
        hypotheses.append(dict(angle=(candidate+180)%360-180,
                               scale=estimate_scale, translation=[delta_x, -delta_y], score=ncc(source, shifted)))
    return max(hypotheses, key=lambda p: p['score']), hypotheses


def similarity_tests(rgb=False):
    template = similarity_scene(256, rgb=rgb)
    results = []
    for scale, angle, translation in ((1, 0, (7, -9)), (.8, 31, (8, -6)),
                                      (1.2, -47, (-5, 7)), (1.1, 143, (4, 3))):
        source = similarity_scene(256, scale, angle, translation, rgb=rgb)
        best, hypotheses = estimate_similarity(template, source)
        results.append(dict(truth=dict(scale=scale, angle=angle, translation=list(translation)),
                            estimate=best, hypotheses=hypotheses,
                            angle_error=abs((best['angle']-angle+180)%360-180),
                            relative_scale_error=abs(best['scale']/scale-1),
                            translation_error=float(np.linalg.norm(np.array(best['translation'])-translation))))
    return results


def assumption_stress_tests():
    template = similarity_scene(256)
    source = similarity_scene(256, .8, 31, (8, -6))
    rng = np.random.default_rng(193)
    occluded = source.copy(); occluded[:, 136:] = 0
    cases = [('additive_background', source + .4*rng.random(source.shape)),
             ('large_occlusion', occluded),
             ('two_different_poses', similarity_scene(256, .8, 31, (-38, 28)) +
              similarity_scene(256, 1.1, -47, (38, -30)))]
    results = []
    for name, image in cases:
        best, hypotheses = estimate_similarity(template, image)
        entry = dict(case=name, estimate=best, hypotheses=hypotheses)
        if name != 'two_different_poses':
            entry['underlying_pose'] = dict(angle=31, scale=.8, translation=[8, -6])
            entry['angle_error'] = abs((best['angle']-31+180)%360-180)
            entry['relative_scale_error'] = abs(best['scale']/.8-1)
        else:
            entry['underlying_poses'] = [dict(angle=31, scale=.8, translation=[-38, 28]),
                                        dict(angle=-47, scale=1.1, translation=[38, -30])]
            entry['limitation'] = 'one global similarity estimate cannot express both independently transformed objects'
        results.append(entry)
    return results


def gradient_test():
    yy, xx = np.indices((31, 33), dtype=float)
    qx, qy = (xx-16)*2, (15-yy)*2
    template = analytic_pattern(qx, qy).ravel()
    template -= template.mean()
    template /= np.linalg.norm(template)

    def evaluate(parameters, derivative=False):
        tx, ty, theta, alpha = parameters
        scale = np.exp(alpha)
        vx, vy = scale*(np.cos(theta)*qx-np.sin(theta)*qy), scale*(np.sin(theta)*qx+np.cos(theta)*qy)
        sample, gx, gy = analytic_pattern(tx+vx, ty+vy, gradient=True)
        centered = sample.ravel()-sample.mean()
        norm = np.linalg.norm(centered)
        z = centered/norm
        score = float(template@z)
        if not derivative:
            return score
        jacobian = np.stack((gx, gy, -gx*vy+gy*vx, gx*vx+gy*vy), axis=-1).reshape(-1, 4)
        jacobian -= jacobian.mean(axis=0)
        normalized_jacobian = (jacobian-z[:, None]*(z@jacobian)[None, :])/norm
        return template@normalized_jacobian

    parameters = np.array([.8, -.6, .07, .04])
    analytic = evaluate(parameters, derivative=True)
    numerical = []
    for axis in range(4):
        delta = np.zeros(4); delta[axis] = 1e-5
        numerical.append((evaluate(parameters+delta)-evaluate(parameters-delta))/(2e-5))
    error = float(np.max(np.abs(analytic-numerical)))
    assert error < 1e-7, error
    return dict(parameters=['tx', 'ty', 'theta_radians', 'log_scale'], analytic=analytic.tolist(),
                finite_difference=numerical, max_absolute_error=error)


def algebra_tests():
    rng = np.random.default_rng(714)
    fft_errors = []
    for count in (1, 3):
        source = rng.normal(size=(35, 41, count)) * 24 + 115
        template = rng.normal(size=(9, 11, count)) * 21 + 100
        for masked in (False, True):
            mask = (rng.random((9, 11)) > .25).astype(float) if masked else np.ones((9, 11))
            error = float(np.max(np.abs(fft_ncc(source, template, mask) - direct_ncc_map(source, template, mask))))
            assert error < 1e-10, error
            fft_errors.append(dict(channels=count, masked=masked, max_absolute_error=error))
    rgb = rng.normal(size=(20, 24, 3)) * np.array([10, 25, 40]) + np.array([80, 120, 150])
    global_gain = ncc(rgb, 1.4*rgb + [20, -7, 30])
    white_balance = ncc(rgb, rgb * [1.9, .6, 1.1])
    per_channel = float(np.mean([ncc(rgb[..., c], rgb[..., c]*[1.9, .6, 1.1][c]) for c in range(3)]))
    assert abs(global_gain - 1) < 1e-12 and abs(per_channel - 1) < 1e-12
    assert white_balance < .99
    # An orthonormal color change preserves the all-channel joint score.
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    other = rng.normal(size=rgb.shape)
    orthogonal_error = abs(ncc(rgb, other)-ncc(rgb@orthogonal, other@orthogonal))
    assert orthogonal_error < 1e-12
    gray_weights = np.array([77, 150, 29])/256
    invisible_color = np.array([150, -77, 0])/np.linalg.norm([150, -77, 0])
    other_color = rgb + 45*rng.uniform(-1, 1, rgb.shape[:2])[..., None]*invisible_color
    gray_alias_score = ncc(rgb@gray_weights, other_color@gray_weights)
    color_alias_score = ncc(rgb, other_color)
    assert abs(gray_alias_score-1) < 1e-12 and color_alias_score < .99
    # Unit-norm patch/template, projection NOT renormalized: Cauchy-Schwarz
    # bounds the NCC error by the norm of the residual template.
    t = rng.normal(size=128); t /= np.linalg.norm(t)
    basis, _ = np.linalg.qr(rng.normal(size=(128, 24)))
    approx = basis@(basis.T@t)
    epsilon = float(np.linalg.norm(t-approx))
    patches = rng.normal(size=(100, 128)); patches /= np.linalg.norm(patches, axis=1)[:, None]
    differences = np.abs(patches@t-patches@approx)
    assert differences.max() <= epsilon + 1e-12
    # Combined rotation/scale eigenfunction identity, pointwise away from r=0.
    r = np.exp(rng.uniform(-2, 3, 1000)); phi = rng.uniform(-np.pi, np.pi, 1000)
    scale, theta, nu, order = 1.37, .63, 2.7, 3
    def basis_value(radius, azimuth):
        return np.exp(1j*(nu*np.log(radius)+order*azimuth))/radius
    lhs = basis_value(r/scale, phi-theta)/scale
    rhs = np.exp(-1j*(nu*np.log(scale)+order*theta))*basis_value(r, phi)
    eigen_error = float(np.max(np.abs(lhs-rhs)))
    assert eigen_error < 1e-12
    return dict(fft_ncc=fft_errors, rgb=dict(common_gain_score=global_gain,
                independent_gain_joint_score=white_balance, independent_gain_per_channel_score=per_channel,
                orthogonal_color_score_error=orthogonal_error,
                same_grayscale_different_color_gray_score=gray_alias_score,
                same_grayscale_different_color_rgb_score=color_alias_score),
                approximation_bound=dict(residual_norm=epsilon, max_tested_score_error=float(differences.max())),
                pointwise_scale_rotation_basis_error=eigen_error)


def template_basis_diagnostic():
    # OpenCV is used only to decode this existing BMP fixture.
    import cv2
    path = ROOT/'Test Images/Dst7.bmp'
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError('missing template fixture')
    template = template.astype(float)-template.mean()
    angular, radial = 720, 512
    u = np.linspace(np.log(1), np.log(np.hypot(*template.shape)/2+2), radial)
    phi = np.arange(angular)*2*np.pi/angular
    radius = np.exp(u)
    q = bilinear(template, template.shape[1]/2+radius[None, :]*np.cos(phi[:, None]),
                 template.shape[0]/2-radius[None, :]*np.sin(phi[:, None]))*radius[None, :]
    # Multiplication by radius includes the square root of the log-polar area
    # Jacobian. Thus ordinary energy in q approximates Cartesian L2 energy.
    angular_energy = np.sum(np.abs(np.fft.fft(q, axis=0))**2, axis=1)
    frequencies = np.fft.fftfreq(angular)*angular
    rotation = [dict(max_order=k, modes=2*k+1,
                     relative_l2_error=float(np.sqrt(angular_energy[np.abs(frequencies)>k].sum()/angular_energy.sum())))
                for k in (4, 8, 16, 32, 64, 128)]
    joint_energy = np.abs(np.fft.fft2(q))**2
    radial_frequencies = np.fft.fftfreq(radial)*radial
    joint = []
    for k in (4, 8, 16, 32, 64):
        keep = (np.abs(frequencies)[:, None]<=k)&(np.abs(radial_frequencies)[None, :]<=k)
        joint.append(dict(max_order=k, modes=int(keep.sum()),
                          relative_l2_error=float(np.sqrt(joint_energy[~keep].sum()/joint_energy.sum()))))
    return dict(template=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                method='mean-centered rectangle, zero outside, q=r*T on 720x512 log-polar grid; radius>=1',
                limitation='finite-grid reconstruction proxy; not a certified Cartesian NCC bound or evidence against other bases',
                rotation=rotation, rotation_scale=joint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'cpp-core/docs/fourier-ncc-research-validation.json')
    args = parser.parse_args()
    report = dict(status='research prototype; no production matcher changes', numpy=np.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  conventions='x-right/y-up; angles CCW; I(x)=T(R(-theta)(x-translation)/scale)',
                  assumptions='single analytic object; no clutter/occlusion; finite sampled spectra are approximate',
                  algebra=algebra_tests(), ncc_similarity_gradient=gradient_test(),
                  similarity=similarity_tests(), similarity_rgb=similarity_tests(rgb=True),
                  assumption_stress=assumption_stress_tests(),
                  template_basis=template_basis_diagnostic())
    for test in report['similarity'] + report['similarity_rgb']:
        assert test['angle_error'] < .5 and test['relative_scale_error'] < .01, test
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
