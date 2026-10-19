"""Caller-side NCC rescue policy for an application's acceptance threshold.

These defaults mirror fpm_pose_rescue_default_config in pose_ncc.cpp. This
module performs no matching and needs no loaded library or benchmark modules.
Changing the detection threshold never changes the C++ library defaults.
"""
import math


DEFAULT_RESCUE_POLICY = dict(trigger_min_score=.1, trigger_score=.8, max_rescued_seeds=4,
    translation_radius=3., angle_radius_degrees=6., log_scale_radius=.06,
    short_iterations=8, max_retry_sample_iterations=512*1024*1024)


def rescue_policy_for_threshold(minimum_score, policy=None):
    """Return copied Scorer.refine_rescue kwargs bounded by caller acceptance.

    A score already accepted by the caller does not require low-score rescue.
    For thresholds below 0.1 the trigger interval becomes empty, rather than
    creating an invalid interval. Other policy fields are left unchanged.
    The returned policy is still validated by the C++ API when it is used.
    """
    minimum_score = float(minimum_score)
    if not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
        raise ValueError('minimum_score must be finite and between 0 and 1')
    result = dict(DEFAULT_RESCUE_POLICY, **({} if policy is None else policy))
    threshold = min(float(result['trigger_score']), minimum_score)
    result['trigger_score'] = threshold
    result['trigger_min_score'] = min(float(result['trigger_min_score']), threshold)
    return result
