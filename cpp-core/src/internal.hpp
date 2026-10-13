// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <array>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

namespace fpm::detail {
constexpr double PI = 3.14159265358979323846;
constexpr double D2R = PI / 180.0;
constexpr int PADDING = 3;
constexpr int EXTRA_CANDIDATES = 5;

void validateImage(ImageView image);
Image pyrDown(ImageView image);
TemplateLevel learnLevel(Image image);
Point rotate(Point p, Point center, double angleDegrees);
double normalizeAngle(double angle);
struct Canvas { Image image; Point translation; };
Canvas rotateCanvas(ImageView source, int templateWidth, int templateHeight,
	double angle, std::uint8_t border);
Image rotatedRoi(ImageView source, int width, int height, Point anchor, double angle);

struct ScoreMap {
	int width = 0;
	int height = 0;
	std::vector<double> values;
	double& at(int x, int y) { return values[std::size_t(y) * width + x]; }
	double at(int x, int y) const { return values[std::size_t(y) * width + x]; }
};
struct Peak { int x = 0; int y = 0; double score = -std::numeric_limits<double>::infinity(); };
// Optional experimental provider only at the dense pyramid-top stage. The
// portable CPU entry point never installs one; fine NCC remains unchanged.
class CoarseNccProvider {
public:
	virtual ~CoarseNccProvider() = default;
	virtual ScoreMap compute(ImageView source, const TemplateLevel& templ) = 0;
};
std::vector<Match> matchWithCoarseNcc(ImageView source, const Pattern& pattern,
	const MatchConfig& config, CoarseNccProvider* provider);
std::uint64_t dotProduct(const std::uint8_t* a, const std::uint8_t* b, int size, Backend backend);
struct NccStats {
	std::uint64_t windows = 0, projectionChecks = 0, partialChecks = 0;
	std::uint64_t rejected = 0, fullDotProducts = 0, dotPixels = 0;
};
// Shared normalization guards; retain legacy expression order. Integer GPU
// statistics and portable CPU statistics use these same binary64 operations.
inline double nccDenominator(double variance, double square, double templateNorm)
{
	return variance > std::min(0.5, 10 * FLT_EPSILON * square) ?
		std::sqrt(variance) * templateNorm : 0;
}
inline double nccNormalizeProduct(double product, double sum, double templateMean,
	double denominator)
{
	if (denominator == 0) return 0;
	const double numerator = product - sum * templateMean;
	if (std::abs(numerator) < denominator) return numerator / denominator;
	if (std::abs(numerator) < denominator * 1.125) return numerator > 0 ? 1 : -1;
	return 0;
}
// A finite positive rejectBelow enables conservative bounds. Values proven
// strictly below it may become -infinity; all other values are bit-identical
// to exhaustive NCC. Nonpositive thresholds bypass bounds (legacy saturation
// can map large negative numerators to zero). Stats are reset per invocation.
ScoreMap ncc(ImageView source, const TemplateLevel& templ, Backend backend,
	double rejectBelow = -std::numeric_limits<double>::infinity(), NccStats* stats = nullptr);
Peak maximum(const ScoreMap& map);
std::vector<Peak> selectPeaks(ScoreMap map, int width, int height,
	int count, double overlap, double threshold, bool blocked);
double overlap(const Match& a, const Match& b);
std::vector<Match> suppress(std::vector<Match> candidates, int maxTargets, double maxOverlap);
Match makeMatch(Point topLeft, int width, int height, double searchAngle, double score);

// Relative coordinates [-1,0,1]; index [angle][y][x]. Rejected fits leave zero
// offsets. A fit must be a finite local maximum inside the sampled cube.
bool refine3d(const std::array<double, 27>& samples, std::array<double, 3>& offset);
bool refine2d(const std::array<double, 9>& samples, Point& offset);
} // namespace fpm::detail
