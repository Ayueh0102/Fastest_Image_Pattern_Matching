"""Independent one-to-one pose accuracy for crowded-scene experiments.

Call after the algorithm timing and RSS snapshots. Truth is never supplied to
the search. Each metric uses maximum-cardinality bipartite matching, so an
early nearest-neighbor choice cannot steal another object's only valid match.
The assignment is deterministic for a given input order, but does not minimize
total error among equally large matchings. All indices address input positions,
not optional ``index`` fields in the input dictionaries.
"""
from collections import deque

import numpy as np


def _poses(items, label):
    try:
        return np.asarray([[float(item[key]) for key in ('pos_x', 'pos_y', 'angle', 'scale')]
                           for item in items], dtype=np.float64).reshape(len(items), 4)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + ' require numeric pos_x, pos_y, angle and scale') from error


def _maximum_matching(adjacency, prediction_count):
    """Find augmenting paths iteratively, without recursion-depth limits."""
    truth_to_prediction = [-1] * len(adjacency)
    prediction_to_truth = [-1] * prediction_count
    for root in range(len(adjacency)):
        queue = deque([root])
        seen_truth = {root}
        predecessor = {}
        endpoint = None
        while queue and endpoint is None:
            truth = queue.popleft()
            for prediction in adjacency[truth]:
                if prediction in predecessor:
                    continue
                predecessor[prediction] = truth
                owner = prediction_to_truth[prediction]
                if owner == -1:
                    endpoint = prediction
                    break
                if owner not in seen_truth:
                    seen_truth.add(owner)
                    queue.append(owner)
        if endpoint is not None:
            prediction = endpoint
            while prediction != -1:
                truth = predecessor[prediction]
                previous = truth_to_prediction[truth]
                truth_to_prediction[truth] = prediction
                prediction_to_truth[prediction] = truth
                prediction = previous
    return truth_to_prediction


def _metric(allowed, position, angle, relative_scale):
    truth_count, prediction_count = allowed.shape
    adjacency = []
    for truth in range(truth_count):
        neighbors = np.flatnonzero(allowed[truth]).tolist()
        neighbors.sort(key=lambda prediction: (position[truth, prediction], angle[truth, prediction],
                                              relative_scale[truth, prediction], prediction))
        adjacency.append(neighbors)
    assignment = _maximum_matching(adjacency, prediction_count)
    pairs, details, used = [], [], set()
    for truth, prediction in enumerate(assignment):
        if prediction < 0:
            details.append(dict(truth_index=truth, matched=False, prediction_index=None, errors=None))
            continue
        errors = dict(position_px=float(position[truth, prediction]),
                      angle_deg=float(angle[truth, prediction]),
                      relative_scale=float(relative_scale[truth, prediction]))
        pairs.append(dict(truth_index=truth, prediction_index=prediction, **errors))
        details.append(dict(truth_index=truth, matched=True, prediction_index=prediction, errors=errors))
        used.add(prediction)
    tp = len(pairs)
    return dict(tp=tp, fp=prediction_count-tp, fn=truth_count-tp,
                precision=tp/prediction_count if prediction_count else 0.,
                recall=tp/truth_count if truth_count else 0.,
                pairs=pairs, details=details,
                max_errors={key: max(pair[key] for pair in pairs) if pairs else None
                            for key in ('position_px', 'angle_deg', 'relative_scale')},
                unmatched_truth_indices=[truth for truth, prediction in enumerate(assignment) if prediction < 0],
                unmatched_prediction_indices=[prediction for prediction in range(prediction_count) if prediction not in used])


def evaluate(matches, truths):
    """Return strict pose and independent localization metrics as JSON data.

    Strict edges require position <= 2 px, wrapped angle <= 1 degree and a
    predicted/truth scale ratio in [0.98, 1.02], inclusively. Localization edges
    require position <= 20 px and may have incorrect angle or scale. Nonfinite
    predictions and nonpositive predicted scales are unmatched false positives;
    invalid ground truth raises ValueError. Precision/recall use zero for empty
    denominators, and maximum errors are null when no pair is matched.
    """
    predicted, expected = _poses(matches, 'matches'), _poses(truths, 'truths')
    valid_truth = np.isfinite(expected).all(axis=1) & (expected[:, 3] > 0)
    if not valid_truth.all():
        raise ValueError('truths contain nonfinite poses or nonpositive scales at indices '
                         + str(np.flatnonzero(~valid_truth).tolist()))
    valid_prediction = np.isfinite(predicted).all(axis=1) & (predicted[:, 3] > 0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        position = np.hypot(predicted[None, :, 0]-expected[:, None, 0],
                            predicted[None, :, 1]-expected[:, None, 1])
        angle = np.abs((predicted[None, :, 2]-expected[:, None, 2]+180) % 360-180)
        scale_ratio = predicted[None, :, 3]/expected[:, None, 3]
        relative_scale = np.abs(scale_ratio-1)
        strict_edges = (valid_prediction[None, :] & (position <= 2.) & (angle <= 1.)
                        & (scale_ratio >= .98) & (scale_ratio <= 1.02))
        localization_edges = valid_prediction[None, :] & (position <= 20.)
    return dict(version=1, truth_count=len(truths), prediction_count=len(matches),
                thresholds=dict(strict=dict(position_px=2., angle_deg=1., relative_scale=.02),
                                localization=dict(position_px=20.)),
                matching='independent maximum-cardinality bipartite assignments; no minimum-cost claim',
                index_convention='zero-based positions in the input truths and matches arrays',
                zero_division=0., invalid_prediction_indices=np.flatnonzero(~valid_prediction).tolist(),
                strict=_metric(strict_edges, position, angle, relative_scale),
                localization=_metric(localization_edges, position, angle, relative_scale))
