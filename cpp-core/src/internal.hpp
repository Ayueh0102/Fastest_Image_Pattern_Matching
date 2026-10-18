// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <array>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

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
struct FineRequest { Point anchor; double angle = 0; };
// An explicit immutable snapshot, reusable across template scales. Preparing a
// new frame must replace its pixels, even at the same input pointer/dimensions.
struct SourcePyramid { std::vector<Image> levels; bool inverted = false; };
// Experimental providers; portable entry points keep their original CPU path.
class CoarseNccProvider {
public:
	virtual ~CoarseNccProvider() = default;
	virtual void invalidateFineSource() {}
	virtual std::size_t fineBatchSize() const { return 0; }
	virtual bool fineAngleReuseEnabled() const { return false; }
	// Successful evaluation-round totals: logical requested maps and exact
	// previous-round reuse hits. cacheBytes is peak live cache-owned allocation
	// capacity (angle identities, previous maps), excluding current maps and
	// allocator overhead. It is not process RSS or a net allocation delta.
	virtual void recordFineAngleReuse(std::uint64_t requested, std::uint64_t reused,
		std::uint64_t cacheBytes) { (void)requested; (void)reused; (void)cacheBytes; }
	// Called once per match with immutable frame levels. Providers may cache
	// resources only within the explicitly prepared frame lifetime.
	virtual void beginFineMatch(const std::vector<Image>&) {}
	virtual std::vector<ScoreMap> computeFineBatch(ImageView, const TemplateLevel&,
		const std::vector<FineRequest>&, double* rotationWallMs) {
		(void)rotationWallMs; throw std::runtime_error("fine batching is unavailable");
	}
	virtual ScoreMap compute(ImageView source, const TemplateLevel& templ) = 0;
	// Optional experimental execution hooks. Level 0 has already been copied
	// and inverted on CPU. A requested GPU mode must throw on failure; false
	// means the provider has left the vector unchanged and requests CPU pyramid.
	virtual bool buildPyramid(std::vector<Image>&, int) { return false; }
	virtual std::size_t coarseBatchAngles() const { return 1; }
	virtual bool fastCoarsePeaks() const { return false; }
	// Experimental compact response: statistics, normalization and greedy
	// selection stay on the device. Scores need not be binary64-identical.
	virtual bool residentCoarsePeaks() const { return false; }
	virtual std::vector<std::vector<Peak>> computePeakBatch(const std::vector<ImageView>&,
		const TemplateLevel&, int, double, double) {
		throw std::runtime_error("resident coarse peaks are unavailable");
	}
	// A begin call must own all input bytes needed after it returns. Slots 0/1
	// are consumed in submission order; finish releases a slot for reuse. Drain
	// is mandatory on exceptional exits, and must not throw from destructors.
	virtual bool coarseOverlapEnabled() const { return false; }
	virtual void beginCoarseBatch(std::size_t, const std::vector<ImageView>&,
		const TemplateLevel&) { throw std::runtime_error("coarse overlap is unavailable"); }
	virtual std::vector<ScoreMap> finishCoarseBatch(std::size_t) {
		throw std::runtime_error("coarse overlap is unavailable");
	}
	virtual void drainCoarseBatches() noexcept {}
	virtual std::vector<ScoreMap> computeBatch(const std::vector<ImageView>& sources,
		const TemplateLevel& templ) {
		std::vector<ScoreMap> result;
		result.reserve(sources.size());
		for (const auto& source : sources) result.push_back(compute(source, templ));
		return result;
	}
};
// Optional exclusive host-wall profiling. A null pointer leaves all clocks and
// counters disabled; algorithm inputs, floating-point operations and order are
// unchanged. Byte counts describe nominal image/map buffers, not process RSS.
struct MatchStageProfile {
	double core_wall_ms = 0, source_copy_invert_ms = 0, pyramid_ms = 0;
	double coarse_rotation_ms = 0, coarse_ncc_ms = 0, peak_selection_ms = 0;
	double fine_rotation_ms = 0, fine_ncc_ms = 0, fine_control_ms = 0;
	double nms_ms = 0, unclassified_ms = 0;
	std::uint64_t pyramid_levels = 0, coarse_angles = 0, coarse_maps = 0, coarse_windows = 0;
	std::uint64_t coarse_candidates = 0, fine_layer_visits = 0, fine_passes = 0;
	std::uint64_t fine_rotations = 0, fine_maps = 0, fine_windows = 0;
	std::uint64_t refined_candidates = 0, output_matches = 0, source_copy_bytes = 0;
	std::uint64_t pyramid_generated_bytes = 0, pyramid_live_bytes = 0;
	std::uint64_t coarse_rotation_total_bytes = 0, coarse_rotation_peak_bytes = 0;
	std::uint64_t fine_roi_total_bytes = 0, fine_roi_peak_bytes = 0;
	std::uint64_t scoremap_peak_bytes = 0, scoremap_total_bytes = 0, timed_transitions = 0;
};
std::vector<Match> matchWithCoarseNcc(ImageView source, const Pattern& pattern,
	const MatchConfig& config, CoarseNccProvider* provider, MatchStageProfile* profile = nullptr,
	const SourcePyramid* prepared = nullptr);
std::uint64_t dotProduct(const std::uint8_t* a, const std::uint8_t* b, int size, Backend backend);
struct NccStats {
	std::uint64_t windows = 0, projectionChecks = 0, partialChecks = 0;
	std::uint64_t rejected = 0, fullDotProducts = 0, dotPixels = 0;
	std::uint64_t adaptiveChecks = 0, adaptiveRectangles = 0, adaptiveRejected = 0;
};
// Internal experimental screen. It can only remove a window proved strictly
// below the supplied threshold; surviving scores use the unchanged NCC path.
class NccScreen {
public:
	virtual ~NccScreen() = default;
	virtual bool supports(ImageView source) const = 0;
	virtual bool reject(const double* sums, const double* squares, std::size_t stride,
		std::size_t offset, double sum, double square, double denominator,
		double threshold, NccStats* stats) const = 0;
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
ScoreMap nccScreened(ImageView source, const TemplateLevel& templ, Backend backend,
	double rejectBelow, const NccScreen& screen, NccStats* stats = nullptr);
Peak maximum(const ScoreMap& map);
std::vector<Peak> selectPeaks(ScoreMap map, int width, int height,
	int count, double overlap, double threshold, bool blocked);
// Exact tournament selector; legacy selectPeaks remains the default/reference.
std::vector<Peak> selectPeaksFast(ScoreMap map, int width, int height,
	int count, double overlap, double threshold, bool blocked);
double overlap(const Match& a, const Match& b);
std::vector<Match> suppress(std::vector<Match> candidates, int maxTargets, double maxOverlap);
Match makeMatch(Point topLeft, int width, int height, double searchAngle, double score);

// Relative coordinates [-1,0,1]; index [angle][y][x]. Rejected fits leave zero
// offsets. A fit must be a finite local maximum inside the sampled cube.
bool refine3d(const std::array<double, 27>& samples, std::array<double, 3>& offset);
bool refine2d(const std::array<double, 9>& samples, Point& offset);
} // namespace fpm::detail
