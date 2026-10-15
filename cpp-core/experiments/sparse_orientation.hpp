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
    int searchBackend = 0; // 0: CPU; 1: Metal compact construction/scan/local peaks.
    int cpuGradientMode = 0; // 0: per-pixel atan2; 1: exact gray8 lookup. Metal always uses lookup.
};

struct SparseOrientationMetrics {
    double sourcePrepareMs = 0;
    double searchMs = 0;
    double selectMs = 0;
    std::uint64_t templateBytes = 0;
    std::uint64_t sourceBytes = 0; // Logical response bytes; included in GPU capacity on Metal.
    std::uint64_t lookupBytes = 0; // Shared read-only table, counted once per process.
    // CPU: retained gradient table capacity, included in templateBytes. Metal:
    // logical gradient table bytes, included in gpuBufferBytes. Do not add twice.
    std::uint64_t gradientLookupBytes = 0;
    std::uint64_t workingBytes = 0; // Partial host/device allocation estimate, not peak RSS.
    std::uint64_t windows = 0;
    std::uint64_t featureEvaluations = 0;
    std::uint64_t thresholdSurvivors = 0;
    int featureCount = 0;
    int angleCount = 0;
    int responseMode = 0;
    int selectedBackend = 0;
    int gradientMode = 0; // Effective mode: CPU setting, or 1 for Metal.
    double gpuSetupMs = 0, uploadMs = 0, encodeMs = 0, waitMs = 0;
    double deviceMs = 0, readbackMs = 0;
    std::uint64_t gpuBufferBytes = 0, commandBuffers = 0, dispatches = 0, cpuWaits = 0;
    std::uint64_t peakOverflowRetries = 0;
};

struct SparseOrientationResult {
    std::vector<Match> proposals;
    SparseOrientationMetrics metrics;
};

// Immutable derived template model, reusable across frames and threads. Does
// not retain the caller's pixels. Metal serializes searches and reuses owned
// device work buffers; CPU calls need no shared workspace. No pyramid is used.
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
