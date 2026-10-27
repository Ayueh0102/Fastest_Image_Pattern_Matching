// SPDX-License-Identifier: BSD-2-Clause
#include "match_verify.hpp"

namespace fpm::research {
// Red step: the checks are not implemented yet and keep every candidate.
MatchVerifyDecision verifyMatch(ImageView, ImageView, Point, double, double, double score, double minScore, bool,
    const MatchVerifyRules&)
{
    MatchVerifyDecision d;
    d.requiredScore = minScore;
    (void)score;
    return d;
}
} // namespace fpm::research
