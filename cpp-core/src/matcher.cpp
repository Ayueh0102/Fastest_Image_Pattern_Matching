// SPDX-License-Identifier: BSD-2-Clause
// Search strategy derived from DennisLiu1993's BSD-2-Clause MatchTool.
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace fpm {
namespace {
enum class Stage { Unclassified, Source, Pyramid, CoarseRotation, CoarseNcc,
	PeakSelection, FineRotation, FineNcc, FineControl, Nms };
class StageClock {
	using Clock = std::chrono::steady_clock;
	detail::MatchStageProfile* profile_;
	Clock::time_point begin_{}, last_{};
	Stage current_ = Stage::Unclassified;
	double& destination(Stage stage) {
		switch (stage) {
		case Stage::Source: return profile_->source_copy_invert_ms;
		case Stage::Pyramid: return profile_->pyramid_ms;
		case Stage::CoarseRotation: return profile_->coarse_rotation_ms;
		case Stage::CoarseNcc: return profile_->coarse_ncc_ms;
		case Stage::PeakSelection: return profile_->peak_selection_ms;
		case Stage::FineRotation: return profile_->fine_rotation_ms;
		case Stage::FineNcc: return profile_->fine_ncc_ms;
		case Stage::FineControl: return profile_->fine_control_ms;
		case Stage::Nms: return profile_->nms_ms;
		default: return profile_->unclassified_ms;
		}
	}
public:
	explicit StageClock(detail::MatchStageProfile* profile) : profile_(profile) {
		if (profile_) { *profile_ = {}; begin_ = last_ = Clock::now(); }
	}
	Stage enter(Stage next) {
		const Stage previous = current_;
		if (profile_) {
			const auto now = Clock::now();
			destination(current_) += std::chrono::duration<double, std::milli>(now-last_).count();
			last_ = now; current_ = next; ++profile_->timed_transitions;
		}
		return previous;
	}
	~StageClock() {
		if (profile_) {
			const auto now = Clock::now();
			destination(current_) += std::chrono::duration<double, std::milli>(now-last_).count();
			profile_->core_wall_ms = std::chrono::duration<double, std::milli>(now-begin_).count();
		}
	}
};
class StageScope {
	StageClock& clock_;
	Stage previous_;
public:
	StageScope(StageClock& clock, Stage stage) : clock_(clock), previous_(clock.enter(stage)) {}
	~StageScope() { clock_.enter(previous_); }
};
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

std::vector<Match> detail::matchWithCoarseNcc(ImageView source, const Pattern& pattern,
	const MatchConfig& config, CoarseNccProvider* provider, MatchStageProfile* profile,
	const SourcePyramid* prepared, const CandidateDedupConfig* dedup, CandidateDedupStats* dedupStats)
{
	using namespace detail;
	StageClock stageClock(profile);
	validateImage(source);
	validateConfig(config);
	if (dedup) validateDedup(*dedup);
	if (dedupStats) *dedupStats = {};
	const auto& levels = pattern.levels();
	if (levels.empty()) throw std::invalid_argument("pattern has no learned levels");
	const int tw = levels[0].image.width(), th = levels[0].image.height();
	if (tw > INT_MAX - 2 * PADDING || th > INT_MAX - 2 * PADDING)
		throw std::invalid_argument("template too large for ROI padding");
	const int top = int(levels.size()) - 1;
	const int stopLayer = std::min(config.stopLayer, top);
	if (prepared) {
		if (prepared->inverted != config.invertSource || prepared->levels.size() < levels.size())
			throw std::invalid_argument("prepared source pyramid inversion or depth does not match the request");
		int width = source.width, height = source.height;
		for (int layer = 0; layer <= top; ++layer) {
			const auto& image = prepared->levels[layer];
			if (image.width() != width || image.height() != height)
				throw std::invalid_argument("prepared source pyramid dimensions do not match the request");
			width = width/2 + width%2; height = height/2 + height%2;
		}
	}
	if (tw > source.width || th > source.height) return {};
	std::vector<Image> ownedSourcePyramid;
	const auto& sourcePyramid = prepared ? prepared->levels : ownedSourcePyramid;
	if (!prepared) {
	if (provider) provider->invalidateFineSource();
	{
	StageScope stage(stageClock, Stage::Source);
	ownedSourcePyramid.emplace_back(source);
	if (config.invertSource)
		for (int y = 0; y < source.height; ++y)
			for (int x = 0; x < source.width; ++x)
				ownedSourcePyramid[0].row(y)[x] = 255 - ownedSourcePyramid[0].row(y)[x];
	}
	{
	StageScope stage(stageClock, Stage::Pyramid);
	if (!provider || !provider->buildPyramid(ownedSourcePyramid, top))
		for (int layer = 1; layer <= top; ++layer)
			ownedSourcePyramid.push_back(pyrDown(ownedSourcePyramid.back().view()));
	}
	}
	if (profile) {
		profile->pyramid_levels = levels.size();
		profile->source_copy_bytes = prepared ? 0 : std::uint64_t(source.width)*source.height;
		for (int layer = 0; layer <= top; ++layer) {
			const auto& level = sourcePyramid[layer];
			profile->pyramid_live_bytes += std::uint64_t(level.width())*level.height();
		}
		profile->pyramid_generated_bytes = prepared ? 0 : profile->pyramid_live_bytes-profile->source_copy_bytes;
	}
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
	const auto selectCoarsePeaks = provider && provider->fastCoarsePeaks() ? selectPeaksFast : selectPeaks;
	const bool blocked = double(topSource.width()) * topSource.height() / (double(topW) * topH) > 500 && config.maxTargets > 10;
	struct Candidate { Point point; double angle; double score; };
	std::vector<Candidate> candidates;
	const auto angles = coarseAngles(config, ranges, angleStep(top));
	const std::size_t batchAngles = provider ? provider->coarseBatchAngles() : 1;
	if (batchAngles < 1 || batchAngles > 4) throw std::runtime_error("coarse provider batch size must be 1..4");
	if (provider && provider->residentCoarsePeaks()) {
		if (provider->coarseOverlapEnabled())
			throw std::invalid_argument("resident coarse peaks do not support CPU overlap scheduling");
		for (std::size_t first = 0; first < angles.size(); first += batchAngles) {
			std::vector<Canvas> canvases;
			std::vector<double> batchAngleValues;
			for (std::size_t i = first; i < std::min(first + batchAngles, angles.size()); ++i) {
				if (profile) ++profile->coarse_angles;
				auto canvas = [&] {
					StageScope stage(stageClock, Stage::CoarseRotation);
					return rotateCanvas(topSource.view(), topW, topH, angles[i], pattern.borderColor());
				}();
				if (profile) {
					const auto bytes = std::uint64_t(canvas.image.width()) * canvas.image.height();
					profile->coarse_rotation_total_bytes += bytes;
					profile->coarse_rotation_peak_bytes = std::max(profile->coarse_rotation_peak_bytes, bytes);
				}
				if (canvas.image.width() < topW || canvas.image.height() < topH) continue;
				batchAngleValues.push_back(angles[i]); canvases.push_back(std::move(canvas));
			}
			if (canvases.empty()) continue;
			std::vector<ImageView> views;
			for (const auto& canvas : canvases) views.push_back(canvas.image.view());
			auto peaks = [&] {
				StageScope stage(stageClock, Stage::CoarseNcc);
				return provider->computePeakBatch(views, topTemplate,
					config.maxTargets + EXTRA_CANDIDATES, config.maxOverlap, thresholds[top]);
			}();
			if (peaks.size() != canvases.size()) throw std::runtime_error("invalid resident peak batch");
			for (std::size_t i = 0; i < peaks.size(); ++i) {
				if (profile) {
					++profile->coarse_maps;
					profile->coarse_windows += std::uint64_t(canvases[i].image.width()-topW+1) *
						(canvases[i].image.height()-topH+1);
				}
				for (const auto& peak : peaks[i]) {
					const Point rotated{peak.x-canvases[i].translation.x, peak.y-canvases[i].translation.y};
					candidates.push_back({rotate(rotated, center, -batchAngleValues[i]), batchAngleValues[i], peak.score});
				}
			}
		}
	} else if (provider && provider->coarseOverlapEnabled()) {
		// GPU slots own their input snapshots. CPU canvases can be released as
		// soon as begin returns; only angle/translation metadata stays live.
		// Submit the following batch before consuming the previous one, so
		// rotation and then exact CPU normalization/peaks can overlap GPU work.
		struct Drain {
			CoarseNccProvider* provider;
			~Drain() { provider->drainCoarseBatches(); }
		} drain{provider};
		struct Meta { Point translation; double angle; int width, height; };
		std::array<std::vector<Meta>, 2> metadata;
		auto consume = [&](std::size_t slot) {
			auto responses = [&] {
				StageScope stage(stageClock, Stage::CoarseNcc);
				return provider->finishCoarseBatch(slot);
			}();
			if (responses.size() != metadata[slot].size())
				throw std::runtime_error("coarse provider returned the wrong map count");
			for (std::size_t i = 0; i < responses.size(); ++i) {
				auto& response = responses[i]; const auto& meta = metadata[slot][i];
				if (response.width != meta.width || response.height != meta.height ||
					response.values.size() != std::size_t(meta.width)*meta.height)
					throw std::runtime_error("coarse provider returned an invalid score map");
				if (profile) {
					++profile->coarse_maps; profile->coarse_windows += response.values.size();
					const auto bytes = std::uint64_t(response.values.size())*sizeof(double);
					profile->scoremap_total_bytes += bytes;
					profile->scoremap_peak_bytes = std::max(profile->scoremap_peak_bytes, bytes);
				}
				StageScope peakStage(stageClock, Stage::PeakSelection);
				for (const auto& peak : selectCoarsePeaks(std::move(response), topW, topH,
					config.maxTargets + EXTRA_CANDIDATES, config.maxOverlap, thresholds[top], blocked)) {
					const Point rotated{peak.x - meta.translation.x, peak.y - meta.translation.y};
					candidates.push_back({rotate(rotated, center, -meta.angle), meta.angle, peak.score});
				}
			}
			metadata[slot].clear();
		};
		int previous = -1;
		std::size_t next = 0;
		for (std::size_t first = 0; first < angles.size(); first += batchAngles) {
			std::vector<Canvas> canvases;
			canvases.reserve(batchAngles); metadata[next].clear(); metadata[next].reserve(batchAngles);
			for (std::size_t i = first; i < std::min(first + batchAngles, angles.size()); ++i) {
				if (profile) ++profile->coarse_angles;
				auto canvas = [&] {
					StageScope stage(stageClock, Stage::CoarseRotation);
					return rotateCanvas(topSource.view(), topW, topH, angles[i], pattern.borderColor());
				}();
				if (profile) {
					const auto bytes = std::uint64_t(canvas.image.width())*canvas.image.height();
					profile->coarse_rotation_total_bytes += bytes;
					profile->coarse_rotation_peak_bytes = std::max(profile->coarse_rotation_peak_bytes, bytes);
				}
				if (canvas.image.width() < topW || canvas.image.height() < topH) continue;
				metadata[next].push_back({canvas.translation, angles[i],
					canvas.image.width()-topW+1, canvas.image.height()-topH+1});
				canvases.push_back(std::move(canvas));
			}
			if (canvases.empty()) continue;
			std::vector<ImageView> views; views.reserve(canvases.size());
			for (const auto& canvas : canvases) views.push_back(canvas.image.view());
			{
				StageScope stage(stageClock, Stage::CoarseNcc);
				provider->beginCoarseBatch(next, views, topTemplate);
			}
			canvases.clear();
			if (previous >= 0) consume(std::size_t(previous));
			previous = int(next); next = 1-next;
		}
		if (previous >= 0) consume(std::size_t(previous));
	} else if (batchAngles == 1) {
	for (double angle : angles) {
		if (profile) ++profile->coarse_angles;
		auto canvas = [&] {
			StageScope stage(stageClock, Stage::CoarseRotation);
			return rotateCanvas(topSource.view(), topW, topH, angle, pattern.borderColor());
		}();
		if (profile) {
			const auto bytes = std::uint64_t(canvas.image.width())*canvas.image.height();
			profile->coarse_rotation_total_bytes += bytes;
			profile->coarse_rotation_peak_bytes = std::max(profile->coarse_rotation_peak_bytes, bytes);
		}
		if (canvas.image.width() < topW || canvas.image.height() < topH) continue;
		auto response = [&] {
			StageScope stage(stageClock, Stage::CoarseNcc);
			return provider ? provider->compute(canvas.image.view(), topTemplate) :
				ncc(canvas.image.view(), topTemplate, config.backend,
				config.coarseNccPruning ? thresholds[top] : -std::numeric_limits<double>::infinity());
		}();
		if (profile) {
			++profile->coarse_maps; profile->coarse_windows += response.values.size();
			const auto bytes = std::uint64_t(response.values.size())*sizeof(double);
			profile->scoremap_total_bytes += bytes;
			profile->scoremap_peak_bytes = std::max(profile->scoremap_peak_bytes, bytes);
		}
		StageScope peakStage(stageClock, Stage::PeakSelection);
		for (const auto& peak : selectCoarsePeaks(std::move(response), topW, topH,
			config.maxTargets + EXTRA_CANDIDATES, config.maxOverlap, thresholds[top], blocked)) {
			const Point rotated{peak.x - canvas.translation.x, peak.y - canvas.translation.y};
			candidates.push_back({rotate(rotated, center, -angle), angle, peak.score});
		}
	}
	} else {
		// Keep at most four rotated canvases/maps alive. Submission order and
		// candidate insertion order remain identical to the single-angle path.
		for (std::size_t first = 0; first < angles.size(); first += batchAngles) {
			std::vector<Canvas> canvases;
			std::vector<double> batchAngleValues;
			canvases.reserve(batchAngles); batchAngleValues.reserve(batchAngles);
			for (std::size_t index = first; index < std::min(first + batchAngles, angles.size()); ++index) {
				if (profile) ++profile->coarse_angles;
				auto canvas = [&] {
					StageScope stage(stageClock, Stage::CoarseRotation);
					return rotateCanvas(topSource.view(), topW, topH, angles[index], pattern.borderColor());
				}();
				if (profile) {
					const auto bytes = std::uint64_t(canvas.image.width())*canvas.image.height();
					profile->coarse_rotation_total_bytes += bytes;
					profile->coarse_rotation_peak_bytes = std::max(profile->coarse_rotation_peak_bytes, bytes);
				}
				if (canvas.image.width() < topW || canvas.image.height() < topH) continue;
				canvases.push_back(std::move(canvas)); batchAngleValues.push_back(angles[index]);
			}
			if (canvases.empty()) continue;
			std::vector<ImageView> views; views.reserve(canvases.size());
			for (const auto& canvas : canvases) views.push_back(canvas.image.view());
			auto responses = [&] {
				StageScope stage(stageClock, Stage::CoarseNcc);
				return provider->computeBatch(views, topTemplate);
			}();
			if (responses.size() != canvases.size()) throw std::runtime_error("coarse provider returned the wrong map count");
			for (std::size_t index = 0; index < canvases.size(); ++index) {
				auto& response = responses[index]; const auto& canvas = canvases[index];
				const auto angle = batchAngleValues[index];
				const int width = canvas.image.width()-topW+1, height = canvas.image.height()-topH+1;
				if (response.width != width || response.height != height || response.values.size() != std::size_t(width)*height)
					throw std::runtime_error("coarse provider returned an invalid score map");
				if (profile) {
					++profile->coarse_maps; profile->coarse_windows += response.values.size();
					const auto bytes = std::uint64_t(response.values.size())*sizeof(double);
					profile->scoremap_total_bytes += bytes;
					profile->scoremap_peak_bytes = std::max(profile->scoremap_peak_bytes, bytes);
				}
				StageScope peakStage(stageClock, Stage::PeakSelection);
				for (const auto& peak : selectCoarsePeaks(std::move(response), topW, topH,
					config.maxTargets + EXTRA_CANDIDATES, config.maxOverlap, thresholds[top], blocked)) {
					const Point rotated{peak.x - canvas.translation.x, peak.y - canvas.translation.y};
					candidates.push_back({rotate(rotated, center, -angle), angle, peak.score});
				}
			}
		}
	}
	{
	StageScope stage(stageClock, Stage::PeakSelection);
	std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
	}
	if (profile) profile->coarse_candidates = candidates.size();
	if (dedupStats) dedupStats->input = dedupStats->kept = candidates.size();
	if (dedup && dedup->topK > 0) {
		// Research-only filter (CandidateDedupConfig); it changes which poses
		// are refined. Centers use the same corner convention as makeMatch.
		StageScope stage(stageClock, Stage::PeakSelection);
		const auto begin = std::chrono::steady_clock::now();
		const double radius = dedup->radiusFraction * (1 - config.maxOverlap) * std::min(tw, th);
		const double window = dedup->angleSteps * angleStep(top), toSource = std::ldexp(1.0, top);
		std::vector<DedupPoint> points;
		points.reserve(candidates.size());
		for (const auto& candidate : candidates) {
			const double c = std::cos(candidate.angle * D2R), s = std::sin(candidate.angle * D2R);
			points.push_back({{(candidate.point.x + 0.5 * (topW * c - topH * s)) * toSource,
				(candidate.point.y + 0.5 * (topW * s + topH * c)) * toSource}, candidate.angle});
		}
		const auto keep = dedupKeep(points, dedup->topK, radius, window);
		std::size_t kept = 0;
		for (std::size_t i = 0; i < candidates.size(); ++i)
			if (keep[i]) candidates[kept++] = candidates[i];
		candidates.resize(kept);
		if (dedupStats) {
			dedupStats->kept = kept; dedupStats->radius = radius; dedupStats->angleWindow = window;
			dedupStats->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		}
	}
	std::vector<Match> refined;
	const std::size_t fineBatchSize = provider ? provider->fineBatchSize() : 0;
	if (fineBatchSize > 1024) throw std::runtime_error("fine provider batch size must be at most 1024");
	// Every candidate is refined before NMS: coarse scores do not bound fine
	// scores, and many high-scoring candidates can belong to the same object.
	{
	StageScope fineStage(stageClock, Stage::FineControl);
	if (fineBatchSize == 0) {
	for (const auto& candidate : candidates) {
		Point point = candidate.point;
		double angle = candidate.angle;
		if (stopLayer > 0 && top == stopLayer) {
			if (candidate.score >= config.minScore)
				refined.push_back(makeMatch({point.x * 2, point.y * 2}, tw, th, angle, candidate.score));
			continue;
		}
		for (int layer = std::max(0, top - 1); layer >= stopLayer; --layer) {
			if (profile) ++profile->fine_layer_visits;
			// Recenter on the fractional pose before fitting again. Otherwise
			// the integer translation grid can bias the angle maximum, even
			// after making angular samples finer (especially on small templates).
			const int passes = config.subpixel && layer == 0 ? 3 : 1;
			bool accepted = false;
			for (int pass = 0; pass < passes; ++pass) {
				if (profile) ++profile->fine_passes;
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
						auto roi = [&] {
							StageScope stage(stageClock, Stage::FineRotation);
							return rotatedRoi(sourcePyramid[layer].view(), templ.image.width(), templ.image.height(), anchor, testAngle);
						}();
						if (profile) {
							++profile->fine_rotations;
							const auto bytes = std::uint64_t(roi.width())*roi.height();
							profile->fine_roi_total_bytes += bytes;
							profile->fine_roi_peak_bytes = std::max(profile->fine_roi_peak_bytes, bytes);
						}
						{
						StageScope stage(stageClock, Stage::FineNcc);
						maps.push_back(ncc(roi.view(), templ, config.backend));
						}
						if (profile) {
							++profile->fine_maps; profile->fine_windows += maps.back().values.size();
							const auto bytes = std::uint64_t(maps.back().values.size())*sizeof(double);
							profile->scoremap_total_bytes += bytes;
							profile->scoremap_peak_bytes = std::max(profile->scoremap_peak_bytes, bytes);
						}
						{
						StageScope stage(stageClock, Stage::PeakSelection);
						peaks.push_back(maximum(maps.back()));
						}
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
	} else {
		{
		StageScope stage(stageClock, Stage::FineNcc);
		provider->beginFineMatch(sourcePyramid);
		}
		struct FineState {
			Point point, anchor;
			double angle = 0, sampleStep = 0;
			bool active = true, evaluating = false, hasOutput = false;
			int resolution = 0, attempt = 0, best = -1;
			std::vector<double> testAngles;
			std::vector<ScoreMap> maps;
			std::vector<Peak> peaks;
			Match output;
		};
		const bool reuseAngles = provider->fineAngleReuseEnabled();
		// A bounded candidate group retains the angle planes needed by the
		// existing quadratic fit. GPU requests are bounded separately below.
		for (std::size_t first = 0; first < candidates.size(); first += fineBatchSize) {
			const std::size_t count = std::min(fineBatchSize, candidates.size()-first);
			std::vector<FineState> states(count);
			// Only the immediately preceding evaluation is retained. Identities
			// are separate from FineState so disabled reuse allocates no cache.
			std::vector<std::vector<double>> evaluatedAngles;
			if (reuseAngles) evaluatedAngles.resize(count);
			std::uint64_t angleCacheBytes = evaluatedAngles.capacity()*sizeof(std::vector<double>);
			for (std::size_t index = 0; index < count; ++index) {
				const auto& candidate = candidates[first+index]; auto& state = states[index];
				state.point = candidate.point; state.angle = candidate.angle;
				if (stopLayer > 0 && top == stopLayer) {
					state.active = false;
					if (candidate.score >= config.minScore) {
						state.output = makeMatch({state.point.x*2, state.point.y*2}, tw, th, state.angle, candidate.score);
						state.hasOutput = true;
					}
				}
			}
			for (int layer = std::max(0, top-1); layer >= stopLayer; --layer) {
				const auto& templ = levels[layer];
				const int passes = config.subpixel && layer == 0 ? 3 : 1;
				if (profile) for (const auto& state : states) if (state.active) ++profile->fine_layer_visits;
				for (int pass = 0; pass < passes; ++pass) {
					// Every pass can move the anchor; a layer also changes source and
					// template pixels. Never carry identities across either boundary.
					for (auto& angles : evaluatedAngles) angles.clear();
					for (auto& state : states) {
						state.evaluating = state.active;
						if (!state.active) continue;
						if (profile) ++profile->fine_passes;
						const double scale = top == 0 || pass > 0 ? 1 : 2;
						state.anchor = {state.point.x*scale, state.point.y*scale};
						state.sampleStep = angleStep(layer);
						state.testAngles = localAngles(state.angle, state.sampleStep, ranges);
						state.resolution = 0; state.attempt = 0;
					}
					while (std::any_of(states.begin(), states.end(), [](const FineState& state) { return state.evaluating; })) {
						std::uint64_t logicalRequests = 0, reusedMaps = 0, peakCacheBytes = angleCacheBytes;
						std::vector<FineRequest> requests;
						std::vector<std::pair<std::size_t, std::size_t>> destinations;
						requests.reserve(fineBatchSize); destinations.reserve(fineBatchSize);
						auto flush = [&] {
							if (requests.empty()) return;
							double rotationWallMs = 0;
							const double before = profile ? profile->fine_ncc_ms : 0;
							auto maps = [&] {
								StageScope stage(stageClock, Stage::FineNcc);
								return provider->computeFineBatch(sourcePyramid[layer].view(), templ, requests, &rotationWallMs);
							}();
							if (!std::isfinite(rotationWallMs) || rotationWallMs < 0)
								throw std::runtime_error("fine provider returned an invalid rotation wall time");
							if (profile) {
								const double rotation = std::min(rotationWallMs, profile->fine_ncc_ms-before);
								profile->fine_ncc_ms -= rotation; profile->fine_rotation_ms += rotation;
							}
							if (maps.size() != requests.size()) throw std::runtime_error("fine provider returned the wrong map count");
							for (std::size_t index = 0; index < maps.size(); ++index) {
								auto& map = maps[index];
								if (map.width != 2*PADDING+1 || map.height != 2*PADDING+1 ||
									map.values.size() != std::size_t(map.width)*map.height)
									throw std::runtime_error("fine provider returned an invalid score map");
								if (profile) {
									++profile->fine_rotations; ++profile->fine_maps;
									profile->fine_windows += map.values.size();
									const auto roiBytes = std::uint64_t(templ.image.width()+2*PADDING)*(templ.image.height()+2*PADDING);
									profile->fine_roi_total_bytes += roiBytes;
									profile->fine_roi_peak_bytes = std::max(profile->fine_roi_peak_bytes, roiBytes);
									const auto mapBytes = std::uint64_t(map.values.size())*sizeof(double);
									profile->scoremap_total_bytes += mapBytes;
									profile->scoremap_peak_bytes = std::max(profile->scoremap_peak_bytes, mapBytes);
								}
								const auto destination = destinations[index];
								states[destination.first].maps[destination.second] = std::move(map);
							}
							requests.clear(); destinations.clear();
						};
						for (std::size_t index = 0; index < count; ++index) {
							auto& state = states[index];
							if (!state.evaluating) continue;
							std::vector<ScoreMap> previousMaps;
							if (reuseAngles && !evaluatedAngles[index].empty())
								previousMaps = std::move(state.maps);
							state.maps.clear(); state.maps.resize(state.testAngles.size());
							state.peaks.clear(); state.best = -1;
							auto previousMapBytes = [&] {
								std::uint64_t bytes = previousMaps.capacity()*sizeof(ScoreMap);
								for (const auto& map : previousMaps) bytes += map.values.capacity()*sizeof(double);
								return bytes;
							};
							if (reuseAngles) {
								logicalRequests += state.testAngles.size();
								peakCacheBytes = std::max(peakCacheBytes, angleCacheBytes+previousMapBytes());
							}
							for (std::size_t angle = 0; angle < state.testAngles.size(); ++angle) {
								bool reused = false;
								if (reuseAngles) for (std::size_t old = 0; old < evaluatedAngles[index].size(); ++old) {
									// Angle quantization, epsilon equality, and modulo-360
									// folding can change legacy sampling near half integers.
									if (std::memcmp(&evaluatedAngles[index][old], &state.testAngles[angle], sizeof(double)) == 0) {
										state.maps[angle] = std::move(previousMaps[old]);
										++reusedMaps; reused = true; break;
									}
								}
								if (reused) continue;
								requests.push_back({state.anchor, state.testAngles[angle]}); destinations.push_back({index, angle});
								if (requests.size() == fineBatchSize) flush();
							}
							if (reuseAngles) {
								auto& angles = evaluatedAngles[index];
								angleCacheBytes -= angles.capacity()*sizeof(double);
								angles = state.testAngles;
								angleCacheBytes += angles.capacity()*sizeof(double);
								peakCacheBytes = std::max(peakCacheBytes, angleCacheBytes+previousMapBytes());
							}
						}
						flush();
						if (reuseAngles) provider->recordFineAngleReuse(logicalRequests, reusedMaps, peakCacheBytes);
						for (auto& state : states) {
							if (!state.evaluating) continue;
							{
							StageScope peakStage(stageClock, Stage::PeakSelection);
							for (const auto& map : state.maps) {
								state.peaks.push_back(maximum(map));
								const int index = int(state.peaks.size())-1;
								if (state.best < 0 || state.peaks[index].score > state.peaks[state.best].score) state.best = index;
							}
							}
							// Resume precisely where the original nested bracketing
							// loops would request their next evaluateAngles() call.
							if (config.subpixel && layer == 0) {
								if (state.attempt < 3 && state.best >= 0 && state.testAngles.size() > 1 &&
									!(state.best > 0 && state.best+1 < int(state.testAngles.size()))) {
									auto next = localAngles(state.testAngles[state.best], state.sampleStep, ranges);
									if (next != state.testAngles) {
										state.testAngles = std::move(next); ++state.attempt;
										continue;
									}
								}
								if (state.resolution < 2 && state.best >= 0 && state.testAngles.size() >= 2) {
									state.sampleStep *= 0.5;
									state.testAngles = localAngles(state.testAngles[state.best], state.sampleStep, ranges);
									++state.resolution; state.attempt = 0;
									continue;
								}
							}
							state.evaluating = false;
							const int best = state.best;
							if (best < 0 || state.peaks[best].score < thresholds[layer]) { state.active = false; continue; }
							const Peak peak = state.peaks[best];
							Point location{double(peak.x), double(peak.y)};
							auto& angle = state.angle; const auto& testAngles = state.testAngles;
							const auto& maps = state.maps; const double sampleStep = state.sampleStep;
							angle = testAngles[best];
							if (config.subpixel && layer == 0 && peak.x > 0 && peak.y > 0 && peak.x+1 < maps[best].width && peak.y+1 < maps[best].height) {
								bool refined3d = false;
								if (best > 0 && best+1 < int(testAngles.size()) &&
									std::abs(normalizeAngle(angle-testAngles[best-1])-sampleStep) < 1e-8 &&
									std::abs(normalizeAngle(testAngles[best+1]-angle)-sampleStep) < 1e-8) {
									std::array<double, 27> samples{}; int index = 0;
									for (int k = best-1; k <= best+1; ++k)
										for (int y = -1; y <= 1; ++y)
											for (int x = -1; x <= 1; ++x) samples[index++] = maps[k].at(peak.x+x, peak.y+y);
									std::array<double, 3> offset{};
									if (refine3d(samples, offset) && allowed(angle+offset[2]*sampleStep, ranges)) {
										location.x += offset[0]; location.y += offset[1]; angle += offset[2]*sampleStep; refined3d = true;
									}
								}
								if (!refined3d) {
									std::array<double, 9> samples{}; int index = 0;
									for (int y = -1; y <= 1; ++y)
										for (int x = -1; x <= 1; ++x) samples[index++] = maps[best].at(peak.x+x, peak.y+y);
									Point offset;
									if (refine2d(samples, offset)) { location.x += offset.x; location.y += offset.y; }
								}
							}
							const Point delta = rotate({location.x-PADDING, location.y-PADDING}, {0, 0}, -angle);
							state.point = {state.anchor.x+delta.x, state.anchor.y+delta.y};
							if (pass+1 == passes && layer == stopLayer && peak.score >= config.minScore && allowed(angle, ranges)) {
								const double outputScale = stopLayer == 0 ? 1 : 2;
								state.output = makeMatch({state.point.x*outputScale, state.point.y*outputScale}, tw, th, angle, peak.score);
								state.hasOutput = true;
							}
						}
					}
				}
			}
			// GPU completion order cannot change stable candidate ties or NMS.
			for (const auto& state : states) if (state.hasOutput) refined.push_back(state.output);
		}
	}
	}
	if (profile) profile->refined_candidates = refined.size();
	auto output = [&] {
		StageScope stage(stageClock, Stage::Nms);
		return suppress(std::move(refined), config.maxTargets, config.maxOverlap);
	}();
	if (profile) profile->output_matches = output.size();
	return output;
}
std::vector<Match> match(ImageView source, const Pattern& pattern, const MatchConfig& config)
{
	return detail::matchWithCoarseNcc(source, pattern, config, nullptr);
}
} // namespace fpm
