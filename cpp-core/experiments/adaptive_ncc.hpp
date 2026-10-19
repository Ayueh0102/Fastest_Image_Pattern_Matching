// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "internal.hpp"

namespace fpm::experimental {
// Research-only owning model: no change to public MatchConfig/C ABI or defaults.
// maxRectangles: 4, 8, 16, 32 or 64. blockBound adds per-leaf residual bounds.
class AdaptiveNccModel final : public detail::NccScreen {
public:
	AdaptiveNccModel(const TemplateLevel& templ, int maxRectangles = 16, bool blockBound = false);
	const TemplateLevel& level() const { return level_; }
	std::size_t retainedBytes() const;
	std::size_t buildWorkspaceBytes() const { return buildWorkspaceBytes_; }
	int rectangles() const { return rectangles_; }
	detail::ScoreMap score(ImageView source, double threshold, Backend backend = Backend::Auto,
		detail::NccStats* stats = nullptr) const;
	bool supports(ImageView source) const override;
	bool reject(const double* sums, const double* squares, std::size_t stride,
		std::size_t offset, double sum, double square, double denominator,
		double threshold, detail::NccStats* stats) const override;
private:
	struct Interval { double lo = 0, hi = 0; };
	struct Node {
		int x0, y0, x1, y1;
		double count, sum, square;
		Interval coefficient, reciprocal;
		double residualUpper;
	};
	struct Split { int parent, left, right; };
	TemplateLevel level_;
	std::vector<Node> nodes_;
	std::vector<Split> splits_;
	std::array<double, 65> templateResidual_{};
	int roots_ = 0, rectangles_ = 0;
	bool blockBound_ = false;
	std::size_t buildWorkspaceBytes_ = 0;
	static Interval add(Interval a, Interval b);
	static Interval subtract(Interval a, Interval b);
	static Interval multiplyPositive(Interval a, Interval b);
	static double up(double a);
	static double down(double a);
};
}
