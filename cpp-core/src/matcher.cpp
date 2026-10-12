// SPDX-License-Identifier: BSD-2-Clause
// Search strategy derived from DennisLiu1993's BSD-2-Clause MatchTool.
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace fpm {
namespace {
void validateConfig(const MatchConfig& config)
{
	if (config.maxTargets < 1 || config.maxTargets > INT_MAX - detail::EXTRA_CANDIDATES)
		throw std::invalid_argument("maxTargets must be positive and leave room for candidate padding");
	if (!std::isfinite(config.minScore) || config.minScore < 0 || config.minScore > 1)
		throw std::invalid_argument("minScore must be in [0,1]");
	if (!std::isfinite(config.maxOverlap) || config.maxOverlap < 0 || config.maxOverlap > 1)
		throw std::invalid_argument("maxOverlap must be in [0,1]");
	if (!std::isfinite(config.toleranceAngle) || config.toleranceAngle < 0 || config.toleranceAngle > 180)
		throw std::invalid_argument("toleranceAngle must be in [0,180]");
	if (config.backend != Backend::Auto && config.backend != Backend::Scalar)
		throw std::invalid_argument("unknown NCC backend");
	if (config.stopLayer != 0 && config.stopLayer != 1)
		throw std::invalid_argument("stopLayer must be 0 or 1");
	for (const auto& range : config.angleRanges)
		if (!std::isfinite(range.min) || !std::isfinite(range.max) || range.min < -180 || range.max > 180 || range.min > range.max)
			throw std::invalid_argument("angle range must satisfy -180 <= min <= max <= 180");
}

bool allowed(double searchAngle, const std::vector<AngleRange>& ranges)
{
	const double output = detail::normalizeAngle(-searchAngle);
	for (const auto& range : ranges)
		for (double value : {output, output - 360, output + 360})
			if (value >= range.min - 1e-9 && value <= range.max + 1e-9) return true;
	return false;
}

void appendAngle(std::vector<double>& angles, double angle)
{
	angle = detail::normalizeAngle(angle);
	for (double existing : angles)
		if (std::abs(detail::normalizeAngle(existing - angle)) < 1e-9) return;
	angles.push_back(angle);
}

std::vector<double> coarseAngles(const MatchConfig& config, const std::vector<AngleRange>& ranges, double step)
{
	std::vector<double> angles;
	if (config.angleRanges.empty()) {
		appendAngle(angles, 0);
		for (double a = step; a < config.toleranceAngle; a += step) appendAngle(angles, a);
		appendAngle(angles, config.toleranceAngle);
		for (double a = -step; a > -config.toleranceAngle; a -= step) appendAngle(angles, a);
		appendAngle(angles, -config.toleranceAngle);
	} else {
		for (const auto& range : ranges) {
			for (double a = range.min; a < range.max; a += step) appendAngle(angles, -a);
			appendAngle(angles, -range.max);
		}
	}
	return angles;
}

std::vector<double> localAngles(double center, double step, const std::vector<AngleRange>& ranges)
{
	std::vector<double> angles;
	for (int k = -1; k <= 1; ++k) {
		const double value = center + k * step;
		if (allowed(value, ranges)) appendAngle(angles, value);
	}
	// Include interval endpoints when refinement reaches a search boundary.
	for (const auto& range : ranges)
		for (double output : {range.min, range.max})
			if (std::abs(detail::normalizeAngle(-output - center)) <= step + 1e-9)
				appendAngle(angles, -output);
	std::sort(angles.begin(), angles.end(), [center](double a, double b) {
		return detail::normalizeAngle(a - center) < detail::normalizeAngle(b - center);
	});
	return angles;
}
}

std::vector<Match> match(ImageView source, const Pattern& pattern, const MatchConfig& config)
{
	using namespace detail;
	validateImage(source);
	validateConfig(config);
	const auto& levels = pattern.levels();
	if (levels.empty()) throw std::invalid_argument("pattern has no learned levels");
	const int tw = levels[0].image.width(), th = levels[0].image.height();
	if (tw > INT_MAX - 2 * PADDING || th > INT_MAX - 2 * PADDING)
		throw std::invalid_argument("template too large for ROI padding");
	if (tw > source.width || th > source.height) return {};
	const int top = int(levels.size()) - 1;
	const int stopLayer = std::min(config.stopLayer, top);
	std::vector<Image> sourcePyramid;
	sourcePyramid.emplace_back(source);
	if (config.invertSource)
		for (int y = 0; y < source.height; ++y)
			for (int x = 0; x < source.width; ++x)
				sourcePyramid[0].row(y)[x] = 255 - sourcePyramid[0].row(y)[x];
	for (int layer = 1; layer <= top; ++layer)
		sourcePyramid.push_back(pyrDown(sourcePyramid.back().view()));
	const auto ranges = config.angleRanges.empty() ?
		std::vector<AngleRange>{{-config.toleranceAngle, config.toleranceAngle}} : config.angleRanges;
	auto angleStep = [&](int layer) {
		return std::atan(2.0 / std::max(levels[layer].image.width(), levels[layer].image.height())) / D2R;
	};
	std::vector<double> thresholds(levels.size(), config.minScore);
	for (int layer = 1; layer <= top; ++layer) thresholds[layer] = thresholds[layer - 1] * 0.9;
	const auto& topSource = sourcePyramid[top];
	const Point center{(topSource.width() - 1) * 0.5, (topSource.height() - 1) * 0.5};
	const auto& topTemplate = levels[top];
	const int topW = topTemplate.image.width(), topH = topTemplate.image.height();
	const bool blocked = double(topSource.width()) * topSource.height() / (double(topW) * topH) > 500 && config.maxTargets > 10;
	struct Candidate { Point point; double angle; double score; };
	std::vector<Candidate> candidates;
	for (double angle : coarseAngles(config, ranges, angleStep(top))) {
		auto canvas = rotateCanvas(topSource.view(), topW, topH, angle, pattern.borderColor());
		if (canvas.image.width() < topW || canvas.image.height() < topH) continue;
		auto response = ncc(canvas.image.view(), topTemplate, config.backend,
			config.coarseNccPruning ? thresholds[top] : -std::numeric_limits<double>::infinity());
		for (const auto& peak : selectPeaks(std::move(response), topW, topH,
			config.maxTargets + EXTRA_CANDIDATES, config.maxOverlap, thresholds[top], blocked)) {
			const Point rotated{peak.x - canvas.translation.x, peak.y - canvas.translation.y};
			candidates.push_back({rotate(rotated, center, -angle), angle, peak.score});
		}
	}
	std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
	std::vector<Match> refined;
	// Every candidate is refined before NMS: coarse scores do not bound fine
	// scores, and many high-scoring candidates can belong to the same object.
	for (const auto& candidate : candidates) {
		Point point = candidate.point;
		double angle = candidate.angle;
		if (stopLayer > 0 && top == stopLayer) {
			if (candidate.score >= config.minScore)
				refined.push_back(makeMatch({point.x * 2, point.y * 2}, tw, th, angle, candidate.score));
			continue;
		}
		for (int layer = std::max(0, top - 1); layer >= stopLayer; --layer) {
			// Recenter on the fractional pose before fitting again. Otherwise
			// the integer translation grid can bias the angle maximum, even
			// after making angular samples finer (especially on small templates).
			const int passes = config.subpixel && layer == 0 ? 3 : 1;
			bool accepted = false;
			for (int pass = 0; pass < passes; ++pass) {
				const double scale = top == 0 || pass > 0 ? 1 : 2;
				const Point anchor{point.x * scale, point.y * scale};
				const auto& templ = levels[layer];
				double sampleStep = angleStep(layer);
				auto testAngles = localAngles(angle, sampleStep, ranges);
				std::vector<ScoreMap> maps;
				std::vector<Peak> peaks;
				int best = -1;
				auto evaluateAngles = [&] {
					maps.clear(); peaks.clear(); best = -1;
					for (double testAngle : testAngles) {
						auto roi = rotatedRoi(sourcePyramid[layer].view(), templ.image.width(), templ.image.height(), anchor, testAngle);
						maps.push_back(ncc(roi.view(), templ, config.backend));
						peaks.push_back(maximum(maps.back()));
						const int index = int(peaks.size()) - 1;
						if (best < 0 || peaks[index].score > peaks[best].score) best = index;
					}
				};
				evaluateAngles();
				// The coarse-to-fine winner can be the outer angle sample. Bracket
				// that maximum before fitting, otherwise "subpixel" silently becomes
				// position-only on precisely the candidates needing angle refinement.
				if (config.subpixel && layer == 0) {
					for (int resolution = 0; resolution < 3; ++resolution) {
						for (int attempt = 0; attempt < 3 && best >= 0 && testAngles.size() > 1; ++attempt) {
							if (best > 0 && best + 1 < int(testAngles.size())) break;
							auto next = localAngles(testAngles[best], sampleStep, ranges);
							if (next == testAngles) break;
							testAngles = std::move(next);
							evaluateAngles();
						}
						if (resolution == 2 || best < 0 || testAngles.size() < 2) break;
						// A two-pixel angular step is appropriate for finding a peak,
						// but too wide for a local quadratic fit on small templates.
						// Quarter that spacing in two stages before fitting x/y/angle.
						sampleStep *= 0.5;
						testAngles = localAngles(testAngles[best], sampleStep, ranges);
						evaluateAngles();
					}
				}
				if (best < 0 || peaks[best].score < thresholds[layer]) { accepted = false; break; }
				const Peak peak = peaks[best];
				Point location{double(peak.x), double(peak.y)};
				angle = testAngles[best];
				if (config.subpixel && layer == 0 && peak.x > 0 && peak.y > 0 && peak.x + 1 < maps[best].width && peak.y + 1 < maps[best].height) {
					bool refined3d = false;
					if (best > 0 && best + 1 < int(testAngles.size()) &&
						std::abs(normalizeAngle(angle - testAngles[best - 1]) - sampleStep) < 1e-8 &&
						std::abs(normalizeAngle(testAngles[best + 1] - angle) - sampleStep) < 1e-8) {
						std::array<double, 27> samples{};
						int index = 0;
						// Sample all angle planes at the SAME spatial coordinates.
						for (int k = best - 1; k <= best + 1; ++k)
							for (int y = -1; y <= 1; ++y)
								for (int x = -1; x <= 1; ++x) samples[index++] = maps[k].at(peak.x + x, peak.y + y);
						std::array<double, 3> offset{};
						if (refine3d(samples, offset) && allowed(angle + offset[2] * sampleStep, ranges)) {
							location.x += offset[0]; location.y += offset[1];
							angle += offset[2] * sampleStep;
							refined3d = true;
						}
					}
					if (!refined3d) {
						std::array<double, 9> samples{};
						int index = 0;
						for (int y = -1; y <= 1; ++y)
							for (int x = -1; x <= 1; ++x) samples[index++] = maps[best].at(peak.x + x, peak.y + y);
						Point offset;
						if (refine2d(samples, offset)) { location.x += offset.x; location.y += offset.y; }
					}
				}
				const Point delta = rotate({location.x - PADDING, location.y - PADDING}, {0, 0}, -angle);
				point = {anchor.x + delta.x, anchor.y + delta.y};
				accepted = true;
				if (pass + 1 == passes && layer == stopLayer && peak.score >= config.minScore && allowed(angle, ranges)) {
					const double outputScale = stopLayer == 0 ? 1 : 2;
					refined.push_back(makeMatch({point.x * outputScale, point.y * outputScale}, tw, th, angle, peak.score));
				}
			}
			if (!accepted) break;
		}
	}
	return suppress(std::move(refined), config.maxTargets, config.maxOverlap);
}
} // namespace fpm
