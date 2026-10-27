// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <cstdint>

namespace fpm::research {
// Experimental (round 52) acceptance checks on final candidates; both off by default.
// Border check: a candidate whose template footprint leaves the source is kept
// only when at least borderMinInside of its template pixels map inside the source
// and the NCC over those pixels alone reaches minScore. Interior candidates are
// never rescored. Size check: a template covering area = w*h*scale^2 < smallTemplateArea
// source pixels needs score >= 1-(1-minScore)*sqrt(area/smallTemplateArea).
struct MatchVerifyRules {
    double borderMinInside = 0; // 0 off, otherwise (0,1]
    double smallTemplateArea = 0; // 0 off, otherwise source pixels > 0
    bool enabled() const { return borderMinInside > 0 || smallTemplateArea > 0; }
};
struct MatchVerifyDecision {
    bool keep = true;
    bool border = false; // footprint leaves the source (border check on)
    double insideFraction = 1; // template pixels mapped inside the source
    double insideScore = 2; // NCC over those pixels; 2 when not computed
    double requiredScore = 0; // threshold the size check applied (minScore when off or large)
    bool rejectedBorder = false, rejectedSize = false;
};
// Template pixel (x,y) maps to center + scale*R*(x - w/2, y - h/2) with the
// pipeline's makeMatch rotation (counter-clockwise on a y-down screen); source pixel
// centres are integers and samples are bilinear. Invalid views or rules throw.
MatchVerifyDecision verifyMatch(ImageView source, ImageView templ, Point center, double angleDegrees,
    double scale, double score, double minScore, bool invertSource, const MatchVerifyRules& rules);
} // namespace fpm::research
