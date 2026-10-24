// SPDX-License-Identifier: BSD-2-Clause
// Experimental, semantics-changing research filter. Portable entry points never
// call it; see CandidateDedupConfig in internal.hpp for the exact rule.
#include "internal.hpp"
#include <cmath>
#include <unordered_map>
#include <utility>

namespace fpm::detail {
namespace {
struct CellHash {
	std::size_t operator()(const std::pair<std::int64_t, std::int64_t>& cell) const
	{
		const auto x = std::uint64_t(cell.first), y = std::uint64_t(cell.second);
		return std::size_t(x * 0x9E3779B97F4A7C15ull ^ (y + 0x632BE59BD9B4E019ull + (x << 6) + (x >> 2)));
	}
};
}

void validateDedup(const CandidateDedupConfig& config)
{
	if (config.topK < 0 || config.topK > (1 << 20) || !std::isfinite(config.radiusFraction) ||
		config.radiusFraction < 0 || !std::isfinite(config.angleSteps) || config.angleSteps < 0)
		throw std::invalid_argument("candidate dedup requires top_k 0..1048576 and finite nonnegative radius/angle window");
}

std::vector<std::uint8_t> dedupKeep(const std::vector<DedupPoint>& ranked, int topK,
	double radius, double angleWindow)
{
	if (topK < 0 || !std::isfinite(radius) || radius < 0 || !std::isfinite(angleWindow) || angleWindow < 0)
		throw std::invalid_argument("candidate dedup requires nonnegative top_k, radius and angle window");
	std::vector<std::uint8_t> keep(ranked.size(), 1);
	if (topK == 0) return keep;
	// Cells at least one radius wide: every neighbor lies in the 3x3 block.
	// Earlier candidates count whether or not they were themselves kept.
	const double cell = std::max(radius, 1.0), limit = radius * radius;
	constexpr double CELL_LIMIT = 4.0e15;
	std::unordered_map<std::pair<std::int64_t, std::int64_t>, std::vector<std::size_t>, CellHash> grid;
	for (std::size_t i = 0; i < ranked.size(); ++i) {
		const auto& p = ranked[i];
		const double fx = std::floor(p.center.x / cell), fy = std::floor(p.center.y / cell);
		// Nonfinite or absurd poses are kept and never suppress another pose.
		if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(p.angle) ||
			std::abs(fx) > CELL_LIMIT || std::abs(fy) > CELL_LIMIT) continue;
		const std::int64_t cx = std::int64_t(fx), cy = std::int64_t(fy);
		int higher = 0;
		for (std::int64_t dy = -1; dy <= 1 && higher < topK; ++dy)
			for (std::int64_t dx = -1; dx <= 1 && higher < topK; ++dx) {
				const auto found = grid.find({cx + dx, cy + dy});
				if (found == grid.end()) continue;
				for (const std::size_t j : found->second) {
					const auto& q = ranked[j];
					const double x = p.center.x - q.center.x, y = p.center.y - q.center.y;
					if (x * x + y * y <= limit && std::abs(normalizeAngle(p.angle - q.angle)) <= angleWindow + 1e-9 &&
						++higher >= topK) break;
				}
			}
		keep[i] = higher < topK;
		grid[{cx, cy}].push_back(i);
	}
	return keep;
}
} // namespace fpm::detail
