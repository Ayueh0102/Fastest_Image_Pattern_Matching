// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <array>
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
std::uint64_t dotProduct(const std::uint8_t* a, const std::uint8_t* b, int size, Backend backend);
ScoreMap ncc(ImageView source, const TemplateLevel& templ, Backend backend);
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
