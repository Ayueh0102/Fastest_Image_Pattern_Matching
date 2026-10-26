// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <cstdint>
#include <vector>

namespace fpm::research {
// Round 49 research rule (opt-in): choose a template's coarse top level from its
// own pixels. For each candidate top level L the template is rendered at eight
// fixed poses that sit a quarter coarse angle step and a quarter top-level pixel
// off the search grid, on two flat backgrounds two template standard
// deviations below and above its mean (clamped to 0..255). Each canvas is
// reduced L times with the matcher's pyrDown, rotated back on the coarse grid
// like the matcher, and the best NCC of the learned level-L template within one
// pixel of the object is kept.
// predictedScores[L] is the worse background's mean over the eight poses.
struct PyramidRuleDecision {
    int originalTop = 0;      // Top level that minReducedArea gives.
    int selectedTop = 0;      // Never deeper than originalTop.
    int effectiveMinArea = 0; // minReducedArea unchanged when selectedTop==originalTop.
    std::vector<double> predictedScores; // Index L = 0..originalTop.
    std::uint8_t lowBackground = 0, highBackground = 0;
    double ms = 0; // Host wall time of the decision; Session construction only.
};
constexpr int PYRAMID_RULE_POSES = 8;
// The deepest L <= originalTop whose predicted score reaches minScore; if none
// does, the deepest level with the highest predicted score.
PyramidRuleDecision decidePyramid(ImageView templ, int minReducedArea, double minScore);
// A minReducedArea for which fpm::Pattern stops exactly at `top` (0..its depth).
int minAreaForTop(int width, int height, int top);
// Mean over the eight poses on one background; 0 when the level is uniform.
double predictedTopScore(const Pattern& pattern, int level, std::uint8_t background);
} // namespace fpm::research
