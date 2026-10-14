// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include <fpm/matcher.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace fpm::research {

// Approximate, fixed-scale proposals on the original image. This score is not
// NCC and is not an upper bound on NCC. Every accepted proposal still requires
// independent full-template verification/refinement.
struct SparseOrientationSettings {
    int featureCount = 128;
    int positionStride = 4;
    double angleStepDegrees = 6;
    int spreadRadius = 2;
    double minGradient = 8; // Central-difference gradient magnitude, gray8 units.
    double minScore = .75;
    int candidateCap = 256;
    int cpuThreads = 8;
    int responseMode = 0; // 0: eight response planes; 1: exact mask + shared lookup.
};

struct SparseOrientationMetrics {
    double sourcePrepareMs = 0;
    double searchMs = 0;
    double selectMs = 0;
    std::uint64_t templateBytes = 0;
    std::uint64_t sourceBytes = 0; // Retained original-resolution response buffer.
    std::uint64_t lookupBytes = 0; // Shared read-only table, counted once per process.
    std::uint64_t workingBytes = 0; // Includes shared lookup; estimate, not peak RSS.
    std::uint64_t windows = 0;
    std::uint64_t featureEvaluations = 0;
    std::uint64_t thresholdSurvivors = 0;
    int featureCount = 0;
    int angleCount = 0;
    int responseMode = 0;
};

struct SparseOrientationResult {
    std::vector<Match> proposals;
    SparseOrientationMetrics metrics;
};

// Immutable derived template model, reusable across frames and threads. Does
// not retain the caller's pixels. No resized images or image pyramid are used.
class SparseOrientationPattern {
public:
    explicit SparseOrientationPattern(ImageView image,
        const SparseOrientationSettings& settings = {});
    ~SparseOrientationPattern();
    SparseOrientationResult search(ImageView source,
        const MatchConfig& config = {}) const;
    int featureCount() const;
    std::uint64_t templateBytes() const;
    int width() const;
    int height() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

// search uses MatchConfig angle limits only. Sparse scores use settings.minScore
// and the proposal budget uses settings.candidateCap, independently of the final
// NCC threshold/maxTargets/overlap. Unsigned gradients are inversion invariant.
// Full rotated rectangular support must fit the source. Flat/too-small templates or a
// flat source produce no proposals. Output ties are deterministic, also across
// cpuThreads settings. Match corners use the core width/2,height/2 convention.

} // namespace fpm::research
