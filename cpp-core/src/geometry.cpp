// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>

namespace fpm::detail {
Match makeMatch(Point p, int width, int height, double searchAngle, double score)
{
	const double c = std::cos(searchAngle * D2R), s = std::sin(searchAngle * D2R);
	Match result;
	result.corners = {{p, {p.x + width * c, p.y + width * s},
		{p.x + width * c - height * s, p.y + width * s + height * c},
		{p.x - height * s, p.y + height * c}}};
	result.center = {(result.corners[0].x + result.corners[2].x) * 0.5,
		(result.corners[0].y + result.corners[2].y) * 0.5};
	result.angle = normalizeAngle(-searchAngle);
	result.score = score;
	return result;
}

static double cross(Point a, Point b, Point p)
{
	return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

static double area(const std::vector<Point>& points)
{
	if (points.size() < 3) return 0;
	double sum = 0;
	// Triangulate relative to the first vertex to avoid large-coordinate cancellation.
	for (std::size_t i = 1; i + 1 < points.size(); ++i)
		sum += cross(points[0], points[i], points[i + 1]);
	return std::abs(sum) * 0.5;
}

double overlap(const Match& a, const Match& b)
{
	// Cheap AABB rejection before Sutherland-Hodgman convex polygon clipping.
	auto bounds = [](const Match& m) {
		std::array<double, 4> box{m.corners[0].x, m.corners[0].y, m.corners[0].x, m.corners[0].y};
		for (const auto& p : m.corners) {
			box[0] = std::min(box[0], p.x); box[1] = std::min(box[1], p.y);
			box[2] = std::max(box[2], p.x); box[3] = std::max(box[3], p.y);
		}
		return box;
	};
	const auto ba = bounds(a), bb = bounds(b);
	if (ba[2] <= bb[0] || bb[2] <= ba[0] || ba[3] <= bb[1] || bb[3] <= ba[1]) return 0;
	std::vector<Point> polygon(a.corners.begin(), a.corners.end());
	const double denominator = area(polygon);
	if (denominator == 0) return 0;
	for (int edge = 0; edge < 4 && !polygon.empty(); ++edge) {
		const Point v = b.corners[edge], w = b.corners[(edge + 1) % 4];
		std::vector<Point> next;
		Point previous = polygon.back();
		double previousSide = cross(v, w, previous);
		for (const auto& current : polygon) {
			const double currentSide = cross(v, w, current);
			const bool inside = currentSide >= 0, previousInside = previousSide >= 0;
			if (inside != previousInside) {
				const double t = previousSide / (previousSide - currentSide);
				next.push_back({previous.x + t * (current.x - previous.x), previous.y + t * (current.y - previous.y)});
			}
			if (inside) next.push_back(current);
			previous = current;
			previousSide = currentSide;
		}
		polygon = std::move(next);
	}
	return std::clamp(area(polygon) / denominator, 0.0, 1.0);
}

std::vector<Match> suppress(std::vector<Match> candidates, int maxTargets, double maxOverlap)
{
	std::stable_sort(candidates.begin(), candidates.end(), [](const Match& a, const Match& b) {
		if (a.score != b.score) return a.score > b.score;
		if (a.center.y != b.center.y) return a.center.y < b.center.y;
		if (a.center.x != b.center.x) return a.center.x < b.center.x;
		return a.angle < b.angle;
	});
	std::vector<Match> kept;
	for (const auto& candidate : candidates) {
		bool accept = true;
		for (const auto& previous : kept) {
			const double ratio = overlap(previous, candidate);
			// As in the legacy implementation, complete duplicates are removed
			// even when the user allows maximum overlap.
			if (ratio > maxOverlap + 1e-10 || ratio >= 1 - 1e-10) { accept = false; break; }
		}
		if (accept) kept.push_back(candidate);
		if (int(kept.size()) >= maxTargets) break;
	}
	return kept;
}
} // namespace fpm::detail
