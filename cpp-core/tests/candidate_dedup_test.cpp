// SPDX-License-Identifier: BSD-2-Clause
// Round 43 research filter: grid implementation against the quadratic
// definition, plus the symmetric-template and adjacent-object controls.
#include "internal.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm::detail;
namespace {
void require(bool value, const std::string& text)
{
	if (!value) throw std::runtime_error(text);
}
std::vector<std::uint8_t> quadratic(const std::vector<DedupPoint>& ranked, int topK, double radius, double window)
{
	std::vector<std::uint8_t> keep(ranked.size(), 1);
	if (!topK) return keep;
	for (std::size_t i = 0; i < ranked.size(); ++i) {
		int higher = 0;
		for (std::size_t j = 0; j < i; ++j) {
			const double x = ranked[i].center.x - ranked[j].center.x, y = ranked[i].center.y - ranked[j].center.y;
			higher += x * x + y * y <= radius * radius &&
				std::abs(normalizeAngle(ranked[i].angle - ranked[j].angle)) <= window + 1e-9;
		}
		keep[i] = higher < topK;
	}
	return keep;
}
std::vector<DedupPoint> cluster(double x, double y, double angle, int count, double spread, std::mt19937& rng)
{
	std::uniform_real_distribution<double> offset(-spread, spread);
	std::vector<DedupPoint> points;
	for (int i = 0; i < count; ++i) points.push_back({{x + offset(rng), y + offset(rng)}, angle + offset(rng) * .1});
	return points;
}
std::size_t kept(const std::vector<std::uint8_t>& keep, std::size_t first, std::size_t count)
{
	std::size_t total = 0;
	for (std::size_t i = first; i < first + count; ++i) total += keep[i];
	return total;
}
void randomized()
{
	std::mt19937 rng(43001);
	std::uniform_real_distribution<double> position(-50, 450), angle(-180, 180);
	std::uniform_int_distribution<int> centers(1, 12), members(1, 30);
	for (int trial = 0; trial < 300; ++trial) {
		std::vector<DedupPoint> ranked;
		const int clusters = centers(rng);
		for (int c = 0; c < clusters; ++c) {
			auto points = cluster(position(rng), position(rng), angle(rng), members(rng), 25, rng);
			ranked.insert(ranked.end(), points.begin(), points.end());
		}
		// Clusters near the +/-180 seam exercise wrapped angle differences.
		auto seam = cluster(position(rng), position(rng), 179.5, members(rng), 20, rng);
		ranked.insert(ranked.end(), seam.begin(), seam.end());
		std::shuffle(ranked.begin(), ranked.end(), rng); // rank order is the vector order
		const double radii[] = {0, .5, 1, 7.5, 40, 1000}, windows[] = {0, 3, 19, 180};
		const double radius = radii[trial % 6], window = windows[(trial / 6) % 4];
		const int topK = 1 + trial % 5;
		require(dedupKeep(ranked, topK, radius, window) == quadratic(ranked, topK, radius, window),
			"grid filter differs from quadratic definition in trial " + std::to_string(trial));
	}
}
void symmetricControl()
{
	// A 180-degree symmetric template produces two equally plausible poses at
	// one center. Position-only clustering lets the higher coarse alias erase
	// the true pose; clustering on angle keeps both for fine search to decide.
	std::mt19937 rng(43002);
	auto ranked = cluster(200, 150, 30, 8, 2, rng);
	const auto truePose = cluster(200, 150, -150, 8, 2, rng);
	ranked.insert(ranked.end(), truePose.begin(), truePose.end());
	const auto angular = dedupKeep(ranked, 3, 10, 20);
	require(kept(angular, 0, 8) == 3 && kept(angular, 8, 8) == 3, "angle clusters must each keep their top 3");
	const auto positionOnly = dedupKeep(ranked, 3, 10, 180);
	require(kept(positionOnly, 0, 8) == 3 && kept(positionOnly, 8, 8) == 0,
		"position-only control must reproduce the alias failure");
}
void adjacentControl()
{
	// Two objects whose centers are 50 px apart can both be valid outputs.
	// A radius below that separation keeps each object's own candidates; a
	// radius above it lets the stronger neighbor suppress the weaker object.
	std::mt19937 rng(43003);
	auto ranked = cluster(100, 100, 0, 6, 1, rng);
	const auto neighbor = cluster(150, 100, 0, 6, 1, rng);
	ranked.insert(ranked.end(), neighbor.begin(), neighbor.end());
	const auto small = dedupKeep(ranked, 3, 40, 20);
	require(kept(small, 0, 6) == 3 && kept(small, 6, 6) == 3, "radius below separation keeps both objects");
	const auto large = dedupKeep(ranked, 3, 60, 20);
	require(kept(large, 0, 6) == 3 && kept(large, 6, 6) == 0, "radius above separation must drop the weaker neighbor");
}
void edges()
{
	std::vector<DedupPoint> ranked(5, DedupPoint{{10, 10}, 5});
	require(dedupKeep(ranked, 0, 100, 180) == std::vector<std::uint8_t>(5, 1), "top_k=0 disables the filter");
	require(dedupKeep(ranked, 2, 0, 0) == std::vector<std::uint8_t>({1, 1, 0, 0, 0}), "exact duplicates count at zero radius");
	ranked[0].center.x = std::numeric_limits<double>::quiet_NaN();
	require(dedupKeep(ranked, 1, 1, 1) == std::vector<std::uint8_t>({1, 1, 0, 0, 0}), "nonfinite poses are kept and never counted");
	require(dedupKeep({}, 3, 1, 1).empty(), "empty input");
	for (auto bad : {-1., std::numeric_limits<double>::infinity()}) {
		bool threw = false;
		try { dedupKeep(ranked, 1, bad, 1); } catch (const std::invalid_argument&) { threw = true; }
		require(threw, "invalid radius must throw");
		threw = false;
		try { validateDedup({1, bad, 1}); } catch (const std::invalid_argument&) { threw = true; }
		require(threw, "invalid config must throw");
	}
	bool threw = false;
	try { validateDedup({-1, .5, 3}); } catch (const std::invalid_argument&) { threw = true; }
	require(threw, "negative top_k must throw");
}
}

int main()
{
	try {
		randomized(); symmetricControl(); adjacentControl(); edges();
		std::cout << "candidate dedup tests passed\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
