// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fpm::research {

// Packed sRGB uint8, RGB channel order; stride is bytes. The constructor owns
// copies, so caller images and mask can be released after construction.
struct Rgb8View {
    const std::uint8_t* data = nullptr;
    int width = 0, height = 0;
    std::size_t stride = 0;
};
struct ColorVerifyConfig {
    double distanceThreshold = .04; // normalized RGB Euclidean distance / sqrt(2)
    double minLinearSum = .03;      // near-black chromaticity is unstable
    double minValidFraction = .95;  // missing/dark source support is NOT a match
};
struct ColorVerifyResult {
    // 0 valid, 1 template has no eligible pixels, 2 insufficient source support.
    // Invalid results retain diagnostic counts but distance/fraction stay -1.
    int status = 1;
    std::uint64_t eligiblePixels = 0, validPixels = 0, changedPixels = 0;
    double validFraction = 0, meanDistance = -1, changedFraction = -1;
};

// Candidate verification only: does not locate objects or assign anomaly labels.
// Decode sRGB with an explicit 256-value LUT, bilinearly sample linear RGB, then
// compare c=(r,g,b)/(r+g+b). This removes common linear-light gain (including
// neutral shading) while preserving chromaticity. It does NOT remove white-
// balance changes, additive lighting, clipping, geometry error or reflections.
// A threshold requires calibration on separate normal data; .04 is a research
// control value, not a universal defect threshold. No opaque color dependency.
class ColorVerifier {
public:
    ColorVerifier(Rgb8View source, Rgb8View templ, ColorVerifyConfig config = {},
        const std::uint8_t* mask = nullptr, std::size_t maskStride = 0);
    // Exactly the fpm_pose_score pose convention: [tx,ty,CCW degrees,log(scale)]
    // relative to (source_width/2,source_height/2), x-right/y-up.
    ColorVerifyResult score(const std::array<float, 4>& pose) const;
    std::size_t retainedBytes() const;
private:
    struct Sample { float u, v; std::array<double, 3> chroma; };
    int width_ = 0, height_ = 0;
    ColorVerifyConfig config_;
    std::array<double, 256> linear_{};
    std::vector<std::uint8_t> source_;
    std::vector<Sample> samples_;
};
} // namespace fpm::research
