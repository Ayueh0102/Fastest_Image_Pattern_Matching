// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace fpm { namespace detail {
std::vector<Peak> selectPeaksFast(ScoreMap, int, int, int, double, double, bool);
} }

using namespace fpm::detail;
namespace {
void require(bool value, const char* text)
{
	if (!value) throw std::runtime_error(text);
}
void same(const std::vector<Peak>& actual, const std::vector<Peak>& expected,
	const std::string& label)
{
	if (actual.size() != expected.size()) throw std::runtime_error(label + ": peak count differs");
	for (std::size_t i = 0; i < actual.size(); ++i)
		if (actual[i].x != expected[i].x || actual[i].y != expected[i].y ||
			std::memcmp(&actual[i].score, &expected[i].score, sizeof(double)) != 0)
			throw std::runtime_error(label + ": peak/order/score bits differ at " + std::to_string(i));
}
void compare(const ScoreMap& map, int tw, int th, int count, double overlap,
	double threshold, const std::string& label)
{
	// selectPeaks remains the original implementation, with both original
	// block modes checked independently against the new implementation.
	const auto original = selectPeaks(map, tw, th, count, overlap, threshold, false);
	same(selectPeaks(map, tw, th, count, overlap, threshold, true), original, label + "/legacy blocks");
	same(selectPeaksFast(map, tw, th, count, overlap, threshold, false), original, label + "/scan flag");
	same(selectPeaksFast(map, tw, th, count, overlap, threshold, true), original, label + "/block flag");
}
void randomized()
{
	std::mt19937 rng(712689);
	const std::pair<int, int> shapes[] = {{1,1}, {1,67}, {97,1}, {2,3}, {31,33},
		{65,63}, {211,7}, {9,257}, {129,35}};
	const int counts[] = {0, 1, 2, 70, 1000};
	const double overlaps[] = {0, .13, .5, .63, .97, 1};
	const double inf = std::numeric_limits<double>::infinity();
	const double nan = std::numeric_limits<double>::quiet_NaN();
	const double thresholds[] = {-inf, -1, -.2, 0, .5, 1, inf, nan};
	for (int trial = 0; trial < 180; ++trial) {
		const auto shape = shapes[trial % 9];
		ScoreMap map{shape.first, shape.second, std::vector<double>(std::size_t(shape.first)*shape.second)};
		for (auto& score : map.values) {
			// Frequent exact ties, negative scores and signed zeros, together
			// with nonfinite values, exercise winner ordering and stop behavior.
			const unsigned draw = rng() % 128;
			if (draw < 3) score = nan;
			else if (draw < 6) score = -inf;
			else if (draw == 6 && trial % 11 == 0) score = inf;
			else if (draw < 10) score = -0.0;
			else score = double(int(rng() % 9)-4) / 4;
		}
		const int tw = 1 + int(rng() % unsigned(shape.first + 7));
		const int th = 1 + int(rng() % unsigned(shape.second + 5));
		compare(map, tw, th, counts[trial % 5], overlaps[(trial / 5) % 6],
			thresholds[(trial / 3) % 8], "random " + std::to_string(trial));
	}
}
void boundaries()
{
	const double inf = std::numeric_limits<double>::infinity();
	const double nan = std::numeric_limits<double>::quiet_NaN();
	ScoreMap edge{67, 71, std::vector<double>(67*71, -.8)};
	// Peaks straddle the 32x32 tile edges and map edges. Fractional suppression
	// extends differently to the left and right because legacy truncates.
	const std::pair<int, int> points[] = {{0,0}, {66,0}, {0,70}, {66,70},
		{31,31}, {32,31}, {31,32}, {32,32}, {63,63}, {64,64}, {6,5}, {7,5}};
	for (std::size_t i = 0; i < sizeof(points)/sizeof(points[0]); ++i)
		edge.at(points[i].first, points[i].second) = 1 - double(i)/32;
	for (const double overlap : {0., .17, .5, .63, .999999999, 1.})
		for (const auto shape : {std::pair<int,int>{1,1}, {7,3}, {3,11}, {89,83}})
			compare(edge, shape.first, shape.second, 70, overlap, -.5, "edge suppression");
	// For overlap=1 every pixel must be visited once in score/y/x order,
	// including signed-zero bits from the original winning pixel.
	ScoreMap ties{35, 33, std::vector<double>(35*33, .75)};
	for (std::size_t i = 0; i < ties.values.size(); ++i) ties.values[i] = i % 2 ? -0.0 : 0.0;
	compare(ties, 19, 7, 1000, 1, -inf, "many equal winners");
	for (const double threshold : {-inf, -.1, 0., std::nextafter(0., 1.), inf, nan})
		compare(ties, 7, 11, 70, .37, threshold, "threshold boundary");
	for (const double score : {-inf, nan, inf, -1., -.0, 1.}) {
		ScoreMap constant{33, 35, std::vector<double>(33*35, score)};
		for (const double threshold : {-inf, 0., 1., inf, nan})
			compare(constant, 13, 3, 70, .25, threshold, "constant/nonfinite");
	}
	ScoreMap nonfinite{3, 1, {nan, .9, inf}};
	require(selectPeaksFast(nonfinite, 1, 1, 70, 1, 0, true).empty(), "+infinity must stop selection");
	nonfinite.values = {nan, .9, -inf};
	const auto finite = selectPeaksFast(nonfinite, 1, 1, 70, 1, nan, true);
	require(finite.size() == 1 && finite.front().x == 1 && finite.front().score == .9,
		"NaN scores/threshold must preserve legacy behavior");
	for (const auto shape : {std::pair<int,int>{0,0}, {0,7}, {11,0}}) {
		ScoreMap empty{shape.first, shape.second, {}};
		require(selectPeaksFast(empty, 1, 1, 70, 0, 0, false).empty(), "empty map must be safe");
	}
	compare({2, 2, {1., .5, .25, 0.}}, 1, 1, -1, 1, -inf, "negative count");
}
} // namespace

int main()
{
	try {
		randomized(); boundaries();
		std::cout << "Tournament peak selection preserves legacy score bits, ties, nonfinite behavior and suppression\n";
		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n'; return 1;
	}
}
