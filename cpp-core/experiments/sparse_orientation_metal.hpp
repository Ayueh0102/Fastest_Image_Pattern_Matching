// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include <fpm/matcher.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace fpm::research::detail {

// Coordinates, orientation bins and threshold lookup are prepared with the
// same host arithmetic as the CPU reference. All device scoring is integer.
struct SparseMetalFeature { std::int32_t offset; std::uint32_t bin; };
struct SparseMetalAngle {
    std::vector<std::int32_t> xs, ys;
    std::vector<SparseMetalFeature> features;
};
struct SparseMetalPeak { std::uint32_t xIndex, yIndex, score; };
struct SparseMetalAngleResult {
    std::vector<SparseMetalPeak> peaks;
    std::uint64_t windows = 0, evaluations = 0, survivors = 0;
};
struct SparseMetalMetrics {
    // setup/upload/encode/wait/readback partition host work; deviceMs is inside
    // waitMs and must not be added to it. sourcePrepareMs/searchMs are outer
    // wall intervals, also not additional terms in the same sum. With merged
    // submission, sourcePrepareMs excludes device preparation: that device work
    // executes in searchMs's first wait. Only the outer sum is mode-comparable.
    double setupMs = 0, uploadMs = 0, encodeMs = 0, waitMs = 0;
    double deviceMs = 0, readbackMs = 0, sourcePrepareMs = 0, searchMs = 0;
    std::uint64_t gpuBufferBytes = 0, commandBuffers = 0, dispatches = 0;
    std::uint64_t cpuWaits = 0, peakOverflowRetries = 0;
};
struct SparseMetalResult {
    std::vector<SparseMetalAngleResult> angles;
    SparseMetalMetrics metrics;
};
class SparseMetalBackend {
public:
    virtual ~SparseMetalBackend() = default;
    virtual SparseMetalResult search(ImageView source,
        const std::vector<SparseMetalAngle>& angles, int spreadRadius,
        std::uint32_t requiredScore, int submissionMode) = 0;
};

// gradientLut[(dy+255)*511+(dx+255)] is either zero or an orientation mask.
// responseLut[bin*256+mask] is the exact byte response of the CPU reference.
// Each backend serializes searches, so an immutable pattern remains safe for
// callers sharing it between threads. No fallback or approximate truncation.
std::shared_ptr<SparseMetalBackend> makeSparseMetalBackend(
    const std::vector<std::uint8_t>& gradientLut,
    const std::vector<std::uint8_t>& responseLut);

} // namespace fpm::research::detail
