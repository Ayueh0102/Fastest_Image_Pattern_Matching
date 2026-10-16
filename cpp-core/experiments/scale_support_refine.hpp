// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <cstdint>
#include <optional>

namespace fpm::research {
struct ScaleSupportCandidate {
    double scale = 1;
    double score = -2; // Full-template NCC, for unchanged caller acceptance/NMS.
    double interiorScore = -2;
};
struct ScaleSupportRefineResult {
    std::optional<ScaleSupportCandidate> candidate;
    bool eligible = false, boundaryOptimum = false;
    uint32_t evaluations = 0; // 33 interior scores plus one full score if changed.
    uint64_t samples = 0; // Template pixels actually evaluated, not allocated bytes.
    double originalInteriorScore = -2, bestInteriorScore = -2;
};
// Experimental CPU scale-only polish of an already-refined pose; no truth input.
// Fixed 33 linear factors [0.96,1.04], two-pixel template margin. Eligible gray
// templates have min dimension >=7 and area <=4096. No heap allocation.
// Invalid ImageViews throw invalid_argument. Nonfinite/invalid poses, degenerate
// scores, boundary optima and no improvement preserve the original (nullopt).
// Candidate changes neither center nor angle; corners must use ORIGINAL size.
// The caller retains its full-template score threshold; an interior score must
// never be used for final acceptance. Disabled by default in the pipeline.
ScaleSupportRefineResult refineScaleSupport(ImageView source, ImageView templ,
    Point center, double angleDegrees, double initialScale, bool invertSource = false);
} // namespace fpm::research
