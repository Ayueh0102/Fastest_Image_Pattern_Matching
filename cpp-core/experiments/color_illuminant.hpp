// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include "color_verify.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fpm::research {

// Where the source-side illuminant comes from.
enum class IlluminantSource {
    Object,     // template reference mask mapped through each pose (per object)
    SourceRect, // one fixed source rectangle, e.g. a calibration card (global)
};
// How clipped reference bytes enter the per-channel median.
enum class ReferenceClipPolicy {
    Rank, // keep clipped bytes in the rank statistic; the reference is unusable
          // when any channel's clipped share exceeds maxReferenceClipFraction
    Drop, // ablation: discard samples with any clipped channel, then take medians
    None, // ablation: no saturation handling anywhere
};

struct IlluminantChromaConfig {
    ColorVerifyConfig chroma;                 // colour-region threshold, dark limit, support
    IlluminantSource source = IlluminantSource::Object;
    ReferenceClipPolicy clipPolicy = ReferenceClipPolicy::Rank;
    int saturationLevel = 255;                // an sRGB byte >= this is clipped
    double maxReferenceClipFraction = .45;    // a median is unaffected below .5 clipped
    double minReferenceFraction = .5;         // in-bounds (and, for Drop, unclipped) share
    std::array<int, 4> referenceRect{};       // SourceRect: [x0,y0,x1,y1) source pixels
    std::array<double, 3> templateReference{1, 1, 1}; // SourceRect: linear RGB under template light
};

struct IlluminantChromaResult {
    // 0 valid, 1 no eligible template colour pixels, 2 insufficient colour support,
    // 3 reference unusable (outside, clipped or dark). Only status 0 has a corrected
    // distance; status 3 is an abstention, never an implicit acceptance.
    int status = 1;
    std::uint64_t eligiblePixels = 0, validPixels = 0, changedPixels = 0, clippedPixels = 0;
    // meanDistance averages per-pixel distances (as ColorVerifier); regionDistance
    // compares the chroma of the summed linear RGB of the valid samples with the
    // template's sum over the same samples, so zero-mean noise does not add a floor.
    double validFraction = 0, meanDistance = -1, changedFraction = -1, regionDistance = -1;
    // Same valid colour samples without correction; -1 without enough support.
    double uncorrectedMeanDistance = -1, uncorrectedRegionDistance = -1;
    std::uint64_t referenceEligible = 0, referenceSamples = 0;
    double referenceFraction = 0;
    std::array<double, 3> referenceClipFraction{};
    std::array<double, 3> gain{};             // template / source reference, 0 if unusable
};

// Illumination-normalized chroma verification for located candidates (opt-in
// research method; the existing ColorVerifier is unchanged). The source
// illuminant is estimated from a neutral reference: per-channel medians of the
// nearest source pixels under the template reference mask mapped through the
// pose, or of a fixed source rectangle. A von Kries diagonal gain maps source
// linear RGB into the template's illuminant before the ColorVerifier chroma
// distance is computed over the colour mask only:
//   gain_c = W_template,c / W_source,c,   d = ||c(gain*I) - c(T)|| / sqrt(2)
// Under Rank/Drop, colour samples with any clipped bilinear neighbour are missing
// support. A diagonal model cannot undo spatially varying light inside an
// object, inter-reflections or non-neutral references. Thresholds still need
// separate calibration data.
class IlluminantChromaVerifier {
public:
    // colourMask may be null (whole template). referenceMask is required for
    // Object and must be null for SourceRect. The constructor copies everything.
    IlluminantChromaVerifier(Rgb8View source, Rgb8View templ, const std::uint8_t* colourMask,
        std::size_t colourMaskStride, const std::uint8_t* referenceMask, std::size_t referenceMaskStride,
        IlluminantChromaConfig config = {});
    // fpm_pose_score pose convention, as ColorVerifier::score.
    IlluminantChromaResult score(const std::array<float, 4>& pose) const;
    const std::array<double, 3>& templateReference() const { return templateWhite_; }
    std::size_t retainedBytes() const;
private:
    struct Sample { float u, v; std::array<double, 3> chroma, linear; };
    struct Reference { bool usable = false; std::uint64_t eligible = 0, samples = 0; double fraction = 0;
        std::array<double, 3> clip{}, white{}; };
    Reference estimate(const std::vector<std::array<std::uint8_t, 3>>& bytes, std::uint64_t eligible) const;
    int width_ = 0, height_ = 0;
    IlluminantChromaConfig config_;
    std::array<double, 256> linear_{};
    std::array<double, 3> templateWhite_{};
    std::vector<std::uint8_t> source_;
    std::vector<Sample> samples_;
    std::vector<std::array<float, 2>> reference_;
    Reference global_;
};
} // namespace fpm::research
