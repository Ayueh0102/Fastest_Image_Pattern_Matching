"""Reproduce the heldout_16 local NCC failure with the report's frozen CPU binary.

This is an oracle-only diagnostic, not a deployable rescue or a new benchmark.
Truth-directed starts/line scans must never enter matching or acceptance gates.
No GPU calls, source edits, baseline rebuilds, or benchmark-gate changes occur.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys

import numpy as np

from benchmark_hybrid_refinement import pose_errors
from benchmark_pose_stress import ROOT, digest, fixture, reference_ncc
from pose_experiment import Scorer, library_path


CASE = 'heldout_16'
SEED_INDEX = 3
METHODS = ('legacy_cpu', 'legacy_hybrid', 'resident', 'resident_verified')
BUDGETS = (1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def read_report(path):
    data = path.read_bytes()
    decoded = gzip.decompress(data) if path.suffix == '.gz' else data
    return json.loads(decoded), hashlib.sha256(decoded).hexdigest()


def verify_baseline(report, runs, baseline):
    binary = library_path(baseline)
    actual = sha(binary)
    expected = {}
    manifest = report.get('baseline_provenance', {}).get('sha256', {})
    for name, value in manifest.items():
        if PurePosixPath(name.replace('\\', '/')).name == binary.name:
            expected['baseline_provenance.sha256:'+name] = value
    if not expected:
        # Old reports may expose only absolute frozen-build manifest entries.
        # Restrict lookup to their legacy CPU build, never the candidate build.
        origin = runs['legacy_cpu']['job']['build'].replace('\\', '/').rstrip('/')+'/'
        for name, value in report.get('binary_hashes', {}).items():
            normalized = name.replace('\\', '/')
            if normalized.startswith(origin) and PurePosixPath(normalized).name == binary.name:
                expected['binary_hashes:frozen/'+binary.name] = value
    if not expected or any(value != actual for value in expected.values()):
        raise ValueError('Supplied baseline binary does not match the report\'s frozen baseline SHA256')
    return binary, dict(path=display_path(binary), sha256=actual, matched_manifest_entries=expected)


def verify_inputs(report, runs, case):
    if case.get('kind') != 'heldout':
        raise ValueError('This diagnostic expects the original analytic held-out fixture')
    source, template = fixture(case)
    seeds = np.ascontiguousarray(case['initial_poses'], dtype=np.float32)
    arrays = dict(source=source, template=template, seeds=seeds)
    hashes = {name: digest(value) for name, value in arrays.items()}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    expected = report['fixtures'][CASE]
    if hashes != expected['input_sha256'] or shapes != expected['shapes']:
        raise ValueError('Regenerated source/template/seeds do not match ALL original fixture hashes/shapes')
    for method, run in runs.items():
        if run['job']['case'] != case or run['result']['input_sha256'] != hashes:
            raise ValueError('Case or original input hashes disagree for '+method)
    return source, template, seeds, hashes


def endpoint_hessian(cpu, endpoint):
    # NCC Hessian with respect to scaled coordinates, NOT the solver's GN H.
    # Finite sampling supports a local-maximum diagnosis; it is not a proof of
    # smoothness at every bilinear cell boundary or a global optimality claim.
    delta = np.array([.05, .05, .1, .001])
    probes = [endpoint.copy()]
    for axis in range(4):
        for sign in (-1, 1):
            value = endpoint.copy()
            value[axis] += sign*delta[axis]
            probes.append(value)
    cross_keys = []
    for a in range(4):
        for b in range(a+1, 4):
            for sa, sb in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                value = endpoint.copy()
                value[a] += sa*delta[a]
                value[b] += sb*delta[b]
                cross_keys.append((a, b, sa, sb))
                probes.append(value)
    scores, _ = cpu.score(np.asarray(probes))
    hessian = np.zeros((4, 4))
    for axis in range(4):
        hessian[axis, axis] = float(scores[1+2*axis])+float(scores[2+2*axis])-2*float(scores[0])
    for (a, b, sa, sb), value in zip(cross_keys, scores[9:]):
        hessian[a, b] += sa*sb*float(value)/4
        hessian[b, a] = hessian[a, b]
    return dict(scale=delta.tolist(), matrix=hessian.tolist(),
                eigenvalues=np.linalg.eigvalsh(hessian).tolist())


def geometry_evidence(source, template, seed, truth, lower, upper, trajectory, case):
    height, width = template.shape[:2]
    corners = np.array([[-width/2, height/2-height+1], [-width/2, height/2],
                        [width/2-1, height/2-height+1], [width/2-1, height/2]], np.float64)
    final30 = next(row['pose'] for row in trajectory if row['budget'] == 30)
    support = {}
    for name, pose in (('seed', seed), ('truth', truth), ('final30', final30),
                       ('final200', trajectory[-1]['pose'])):
        angle, scale = np.deg2rad(pose[2]), math.exp(pose[3])
        cs, sn = scale*math.cos(angle), scale*math.sin(angle)
        sx = source.shape[1]/2+pose[0]+cs*corners[:, 0]-sn*corners[:, 1]
        sy = source.shape[0]/2-pose[1]-(sn*corners[:, 0]+cs*corners[:, 1])
        support[name] = dict(x_min=float(sx.min()), x_max=float(sx.max()),
                            y_min=float(sy.min()), y_max=float(sy.max()),
                            all_bilinear_neighbors_inside=bool(sx.min() >= 0 and sx.max() < source.shape[1]-1
                                                              and sy.min() >= 0 and sy.max() < source.shape[0]-1))
    radius = math.exp(upper[3])*math.hypot(width/2, height/2)
    margin = min(source.shape[1]/2+lower[0]-radius,
                 source.shape[1]-1-(source.shape[1]/2+upper[0]+radius),
                 source.shape[0]/2-upper[1]-radius,
                 source.shape[0]-1-(source.shape[0]/2-lower[1]+radius))
    # Recreate the first/only channel's wave parameters from fixture.field().
    rng = np.random.default_rng(case['seed'])
    delta_theta, ratio = np.deg2rad(seed[2]-truth[2]), math.exp(seed[3]-truth[3])
    rotate = np.array([[math.cos(delta_theta), -math.sin(delta_theta)],
                       [math.sin(delta_theta), math.cos(delta_theta)]])
    theta = np.deg2rad(truth[2])
    inverse_rotation = np.array([[math.cos(theta), math.sin(theta)],
                                 [-math.sin(theta), math.cos(theta)]])
    displacement = corners@(ratio*rotate-np.eye(2)).T + inverse_rotation@(seed[:2]-truth[:2])/math.exp(truth[3])
    waves = []
    for _ in range(12):
        direction, _phase = rng.uniform(-math.pi, math.pi, 2)
        cycles = float(rng.uniform(1.3, 7.0))
        frequency = cycles*2*math.pi/case['size']
        wave = np.array([math.cos(direction), math.sin(direction)])*frequency
        shifts = displacement@wave
        waves.append(dict(cycles_across_template=cycles, wavelength_template_px=case['size']/cycles,
                          corner_phase_min_radians=float(shifts.min()), corner_phase_max_radians=float(shifts.max())))
    return dict(support=support, all_allowed_poses_conservative_source_margin_px=float(margin),
                initial_max_corner_displacement_in_template_px=float(np.linalg.norm(displacement, axis=1).max()),
                fixture_waves=waves)


def diagnose(report, runs, source, template, seeds, baseline, hashes):
    case = runs['legacy_cpu']['job']['case']
    config = dict(report['protocol']['configuration'], **case.get('config', {}))
    truth = np.asarray(case['truth'], np.float64)
    seed = seeds[SEED_INDEX].astype(np.float64)
    limits = np.array([config['max_translation']]*2 + [
        config['max_angle_degrees'] if config['optimize_angle'] else 0.,
        config['max_log_scale'] if config['optimize_scale'] else 0.])
    absolute_limits = np.array([1e6, 1e6, 36000., 3.])
    lower, upper = np.maximum(-absolute_limits, seed-limits), np.minimum(absolute_limits, seed+limits)
    result = dict(case=case, configuration=config, seed_index=SEED_INDEX, methods={},
                  source_shape=list(source.shape), template_shape=list(template.shape), input_sha256=hashes,
                  seed_pose=seed.tolist(), truth_pose=truth.tolist(), truth_minus_seed=(truth-seed).tolist(),
                  lower_bound=lower.tolist(), upper_bound=upper.tolist(),
                  truth_inside_seed_bounds=bool(np.all((truth >= lower) & (truth <= upper))),
                  template_std=float(template.std(dtype=np.float64)))
    for method, run in runs.items():
        value = run['result']['measured'][0]['result']
        result['methods'][method] = dict(pose=value['poses'][SEED_INDEX], score=value['scores'][SEED_INDEX],
                                         stats=value['stats'][SEED_INDEX])
    threads = runs['legacy_cpu']['job']['threads']
    with Scorer(baseline, source, template, 'cpu', threads) as cpu:
        initial_scores, _ = cpu.score(seeds)
        true_score, _ = cpu.score(truth[None, :])
        result['initial_scores'] = initial_scores.tolist()
        result['truth_score'] = float(true_score[0])
        result['truth_score_independent_float64'] = reference_ncc(source, template, truth)
        trajectory = []
        for budget in BUDGETS:
            poses, scores, stats, _ = cpu.refine(seeds[SEED_INDEX:SEED_INDEX+1], **dict(config, max_iterations=budget))
            trajectory.append(dict(budget=budget, pose=poses[0].tolist(), score=float(scores[0]), stats=stats[0],
                                   error=pose_errors(poses, truth, scores)[0]))
        result['iteration_budget_trajectory'] = trajectory
        # Verify that the replay reproduces the report before using its probes.
        replay = next(row for row in trajectory if row['budget'] == config['max_iterations'])
        old = result['methods']['legacy_cpu']
        if any(replay[key] != old[key] for key in ('pose', 'score', 'stats')):
            raise ValueError('Frozen CPU replay differs from the report endpoint or statistics')
        fractions = np.array([0, .025, .05, .075, .1, .125, .15, .2, .25, .3, .4, .5, .75, 1.])
        starts = (seed[None, :]+fractions[:, None]*(truth-seed)[None, :]).astype(np.float32)
        poses, scores, stats, _ = cpu.refine(starts, **dict(config, max_iterations=200))
        start_scores, _ = cpu.score(starts)
        result['basin_interpolation'] = [dict(fraction_toward_truth=float(f), initial_pose=start.tolist(),
            initial_score=float(a), pose=pose.tolist(), score=float(b), stats=stat, error=error)
            for f, start, a, pose, b, stat, error in zip(fractions, starts, start_scores, poses, scores, stats,
                                                      pose_errors(poses, truth, scores))]
        line = np.linspace(0., 1., 201)
        line_scores, _ = cpu.score(seed[None, :]+line[:, None]*(truth-seed)[None, :])
        result['straight_line_to_truth'] = [dict(fraction=float(f), score=float(s)) for f, s in zip(line, line_scores)]
        endpoint = np.array(trajectory[-1]['pose'], np.float64)
        nearby = []
        for axis, step in enumerate((.05, .05, .1, .001)):
            pair = np.repeat(endpoint[None, :], 2, axis=0)
            pair[0, axis] -= step
            pair[1, axis] += step
            values, _ = cpu.score(pair)
            nearby.append(dict(axis=axis, step=step, minus=float(values[0]), center=trajectory[-1]['score'], plus=float(values[1])))
        result['endpoint_axis_neighborhood'] = nearby
        result['endpoint_score_independent_float64'] = reference_ncc(source, template, endpoint)
        result['wrong_endpoint_scaled_ncc_hessian'] = endpoint_hessian(cpu, endpoint)
        result['straight_line_independent_float64'] = [dict(fraction=f,
            score=reference_ncc(source, template, seed+f*(truth-seed))) for f in (0., .005, .01, .02, .025, 1.)]
    result.update(geometry_evidence(source, template, seed, truth, lower, upper, trajectory, case))
    result['protocol'] = dict(backend='frozen CPU only', threads=threads, benchmark_inputs_unchanged=True,
        report_gates_unchanged=True, exact_report_cpu_endpoint_reproduced=True,
        truth_directed_probes='ORACLE-ONLY: basin_interpolation and straight_line probes use known truth; not a deployable rescue or detector.',
        basin_bounds='Each diagnostic restart keeps the original config, with bounds relative to that restart; none of these endpoints approach its bounds.',
        geometry='NCC is nonconvex. A finite-difference negative Hessian and nearby lower scores support, but do not prove, a local maximum.')
    result['report_gates'] = {key: report.get(key) for key in (
        'correctness_passed', 'local_accuracy_passed', 'full_accuracy_passed', 'cross_method_passed', 'acceptance_passed')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Original resident report JSON or JSON.gz')
    parser.add_argument('--baseline', type=Path, default=ROOT/'build/resident-baseline', help='Frozen baseline build directory')
    parser.add_argument('--output', type=Path, required=True, help='Diagnostic evidence JSON; original report is never overwritten')
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        parser.error('--output must not overwrite --report')
    report, decoded_hash = read_report(args.report)
    selected = [row for row in report['runs'] if row['job']['case']['name'] == CASE]
    runs = {row['job']['method']: row for row in selected}
    if len(selected) != len(METHODS) or set(runs) != set(METHODS):
        raise ValueError('Expected exactly one recorded heldout_16 run for each of the four methods')
    binary, binary_provenance = verify_baseline(report, runs, args.baseline)
    source, template, seeds, hashes = verify_inputs(report, runs, runs['legacy_cpu']['job']['case'])
    result = diagnose(report, runs, source, template, seeds, args.baseline, hashes)
    if sha(binary) != binary_provenance['sha256']:
        raise ValueError('Baseline binary changed during diagnosis')
    if {name: digest(value) for name, value in zip(('source', 'template', 'seeds'), (source, template, seeds))} != hashes:
        raise ValueError('Diagnostic input arrays changed during probing')
    tools = Path(__file__).resolve().parent
    dependencies = (Path(__file__).name, 'benchmark_pose_stress.py', 'benchmark_hybrid_refinement.py', 'pose_experiment.py')
    result['provenance'] = dict(script=display_path(__file__), script_sha256=sha(__file__),
        dependency_sha256={display_path(tools/name): sha(tools/name) for name in dependencies},
        report=display_path(args.report), report_file_sha256=sha(args.report), report_decoded_sha256=decoded_hash,
        report_git_head=report.get('git_head'), baseline_binary=binary_provenance,
        all_original_fixture_hashes_and_each_method_hashes_verified=True, input_hashes_after=hashes,
        numpy=np.__version__, python=sys.version, platform=platform.platform(),
        command=['python3', display_path(__file__), '--report', display_path(args.report),
                 '--baseline', display_path(args.baseline), '--output', display_path(args.output)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    endpoint = result['iteration_budget_trajectory'][-1]
    print(json.dumps(dict(output=display_path(args.output), verified_original_input_hashes=hashes,
                          verified_baseline_sha256=binary_provenance['sha256'],
                          endpoint_score=endpoint['score'], endpoint_iterations=endpoint['stats']['iterations'],
                          original_local_accuracy_passed=result['report_gates']['local_accuracy_passed'])))


if __name__ == '__main__':
    main()
