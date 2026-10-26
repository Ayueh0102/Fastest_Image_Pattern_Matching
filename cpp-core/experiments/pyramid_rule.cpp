// SPDX-License-Identifier: BSD-2-Clause
#include "pyramid_rule.hpp"

namespace fpm::research {
// Placeholder before the round-49 implementation: keeps the original depth.
PyramidRuleDecision decidePyramid(ImageView templ, int minReducedArea, double)
{
    PyramidRuleDecision decision;
    decision.originalTop = int(Pattern(templ, minReducedArea).levels().size()) - 1;
    decision.selectedTop = decision.originalTop;
    decision.effectiveMinArea = minReducedArea;
    return decision;
}
int minAreaForTop(int, int, int) { return 0; }
double predictedTopScore(const Pattern&, int, std::uint8_t) { return 0; }
} // namespace fpm::research
