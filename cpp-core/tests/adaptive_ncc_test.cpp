// SPDX-License-Identifier: BSD-2-Clause
#include "adaptive_ncc.hpp"
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm;
using namespace fpm::detail;
using fpm::experimental::AdaptiveNccModel;

namespace {
std::size_t checkedMaps = 0, checkedMatches = 0;
std::uint64_t adaptiveChecks = 0, adaptiveRejections = 0;

void require(bool value, const std::string& text)
{
	if (!value) throw std::runtime_error(text);
}
template<class Function> void rejects(Function function)
{
	try { function(); } catch (const std::invalid_argument&) { return; }
	throw std::runtime_error("invalid adaptive NCC input was accepted");
}
bool exact(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
Image noise(int width, int height, unsigned seed)
{
	Image image(width, height); std::mt19937 rng(seed);
	for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
		image.row(y)[x] = std::uint8_t(rng());
	return image;
}
void paste(Image& source, const Image& patch, int x, int y)
{
	for (int row = 0; row < patch.height(); ++row)
		std::copy_n(patch.row(row), patch.width(), source.row(y+row)+x);
}
void invert(Image& image)
{
	for (int y = 0; y < image.height(); ++y) for (int x = 0; x < image.width(); ++x)
		image.row(y)[x] = 255-image.row(y)[x];
}
void equalPeaks(const ScoreMap& actual, const ScoreMap& expected,
	const TemplateLevel& templ, double threshold, const std::string& label)
{
	for (double overlap : {0.0, .6, 1.0}) {
		const auto a = selectPeaks(actual, templ.image.width(), templ.image.height(), 12, overlap, threshold, true);
		const auto e = selectPeaks(expected, templ.image.width(), templ.image.height(), 12, overlap, threshold, true);
		require(a.size() == e.size(), label+": peak count changed");
		for (std::size_t i = 0; i < a.size(); ++i)
			require(a[i].x == e[i].x && a[i].y == e[i].y && exact(a[i].score, e[i].score),
				label+": peak position, score or tie order changed");
	}
}
void validateMap(const ScoreMap& actual, const ScoreMap& full,
	const TemplateLevel& templ, double threshold, const NccStats& stats, const std::string& label)
{
	require(actual.width == full.width && actual.height == full.height &&
		actual.values.size() == full.values.size(), label+": map dimensions changed");
	require(stats.windows == full.values.size(), label+": window statistics not reset");
	std::uint64_t rejected = 0;
	for (std::size_t i = 0; i < full.values.size(); ++i) {
		if (std::isfinite(actual.values[i]))
			require(exact(actual.values[i], full.values[i]), label+": surviving NCC changed at "+std::to_string(i));
		else {
			require(actual.values[i] == -std::numeric_limits<double>::infinity(), label+": invalid rejection sentinel");
			require(full.values[i] < threshold, label+": rejected score at/above threshold at "+std::to_string(i));
			++rejected;
		}
	}
	require(stats.rejected == rejected && stats.adaptiveRejected <= rejected,
		label+": rejection statistics disagree with output");
	if (!(std::isfinite(threshold) && threshold > 0 && threshold <= 1))
		require(rejected == 0 && stats.adaptiveChecks == 0, label+": unsupported threshold used bounds");
	equalPeaks(actual, full, templ, threshold, label);
	adaptiveChecks += stats.adaptiveChecks; adaptiveRejections += stats.adaptiveRejected;
	++checkedMaps;
}
void compareMaps(ImageView source, const TemplateLevel& templ, const std::string& label,
	bool bothBackends = true)
{
	const auto full = ncc(source, templ, Backend::Scalar);
	const auto simd = ncc(source, templ, Backend::Auto);
	require(simd.values.size() == full.values.size() &&
		std::memcmp(simd.values.data(), full.values.data(), full.values.size()*sizeof(double)) == 0,
		label+": exhaustive Scalar/Auto mismatch");
	std::vector<double> thresholds{-1, -.5, 0, .5, .8, .95, 1, 1.01};
	// At a nonperfect positive score, all three thresholds must follow the
	// exact binary64 comparison. A mathematically close upper bound is not
	// sufficient if it removes a window accepted by the existing evaluator.
	for (double score : full.values) if (score > .1 && score < 1) {
		thresholds.push_back(std::nextafter(score, -std::numeric_limits<double>::infinity()));
		thresholds.push_back(score);
		thresholds.push_back(std::nextafter(score, std::numeric_limits<double>::infinity()));
		break;
	}
	std::vector<std::unique_ptr<AdaptiveNccModel>> models;
	for (bool block : {false, true}) for (int cap : {4, 8, 16, 32, 64}) {
		models.push_back(std::make_unique<AdaptiveNccModel>(templ, cap, block));
		const auto& model = *models.back();
		require(model.rectangles() >= 0 && model.rectangles() <= cap, label+": rectangle budget exceeded");
		require(model.retainedBytes() >= std::size_t(templ.image.width())*templ.image.height(),
			label+": retained storage excludes owned template pixels");
	}
	for (Backend backend : {Backend::Scalar, Backend::Auto}) {
		if (!bothBackends && backend == Backend::Scalar) continue;
		for (double threshold : thresholds) {
			NccStats legacy; legacy.windows = std::numeric_limits<std::uint64_t>::max();
			validateMap(ncc(source, templ, backend, threshold, &legacy), full, templ, threshold,
				legacy, label+" legacy");
			for (std::size_t i = 0; i < models.size(); ++i) {
				NccStats stats; stats.windows = std::numeric_limits<std::uint64_t>::max();
				const auto actual = models[i]->score(source, threshold, backend, &stats);
				validateMap(actual, full, templ, threshold, stats, label+" model "+std::to_string(i));
			}
		}
	}
}

void maps()
{
	for (const auto shape : {std::pair<int,int>{1,17}, {17,1}, {7,9}, {17,19}, {33,35}}) {
		auto templ = noise(shape.first, shape.second, unsigned(117+shape.first));
		auto source = noise(shape.first+11, shape.second+9, unsigned(772+shape.second));
		paste(source, templ, 5, 3); source.row(3)[5] ^= 31;
		// Sentinel-filled source padding must neither enter integrals nor dot products.
		const std::size_t stride = source.width()+13;
		std::vector<std::uint8_t> padded(stride*source.height(), 251);
		for (int y = 0; y < source.height(); ++y)
			std::copy_n(source.row(y), source.width(), padded.data()+std::size_t(y)*stride);
		compareMaps({padded.data(), source.width(), source.height(), stride}, learnLevel(std::move(templ)),
			"padded "+std::to_string(shape.first)+"x"+std::to_string(shape.second));
	}
	// Uneven piecewise-constant structure rewards nonuniform splits, while
	// noise windows exercise real rejections rather than only fallback paths.
	Image structured(33, 35);
	for (int y = 0; y < structured.height(); ++y) for (int x = 0; x < structured.width(); ++x)
		structured.row(y)[x] = std::uint8_t(17+(x >= 7 ? 80 : 0)+(y >= 23 ? 70 : 0)+(x >= 27 ? 40 : 0));
	auto source = noise(49, 47, 209); paste(source, structured, 7, 5);
	compareMaps(source.view(), learnLevel(Image(structured.view())), "structured");
	Image negative(structured.view()); invert(negative);
	compareMaps(negative.view(), learnLevel(Image(structured.view())), "negative correlation");
	Image flat(49, 47);
	std::fill_n(flat.row(0), std::size_t(flat.width())*flat.height(), std::uint8_t(193));
	compareMaps(flat.view(), learnLevel(Image(structured.view())), "constant source", false);
	compareMaps(source.view(), learnLevel(Image(17, 19)), "constant template", false);
	compareMaps(Image(1, 1).view(), learnLevel(Image(1, 1)), "single constant pixel", false);
	for (int y = 0; y < source.height(); ++y) for (int x = 0; x < source.width(); ++x)
		source.row(y)[x] = 128+(source.row(y)[x]&1);
	auto low = noise(33, 35, 751);
	for (int y = 0; y < low.height(); ++y) for (int x = 0; x < low.width(); ++x)
		low.row(y)[x] = 128+(low.row(y)[x]&1);
	paste(source, low, 7, 5);
	compareMaps(source.view(), learnLevel(std::move(low)), "low contrast");
	// Raw products and sum-of-squares cross 32-bit range. Only four output
	// windows are needed to exercise large-sum centering and partial bounds.
	Image large(257, 258), largeSource(258, 259);
	for (int y = 0; y < large.height(); ++y) for (int x = 0; x < large.width(); ++x)
		large.row(y)[x] = std::uint8_t(254+((x+3*y)%17 != 0));
	std::fill_n(largeSource.row(0), std::size_t(largeSource.width())*largeSource.height(), std::uint8_t(255));
	paste(largeSource, large, 0, 0); largeSource.row(0)[0] = 255;
	std::uint64_t product = 0;
	for (int y = 0; y < large.height(); ++y) for (int x = 0; x < large.width(); ++x)
		product += unsigned(large.row(y)[x])*unsigned(largeSource.row(y)[x]);
	require(product > std::numeric_limits<std::uint32_t>::max(), "large-sum fixture did not cross uint32");
	compareMaps(largeSource.view(), learnLevel(std::move(large)), "large sums", false);
	require(adaptiveChecks > 0 && adaptiveRejections > 0, "adaptive screens were never exercised");
}

void ownershipAndValidation()
{
	auto image = noise(33, 35, 498); const auto snapshot = learnLevel(Image(image.view()));
	auto source = noise(39, 41, 399); paste(source, image, 3, 4);
	std::unique_ptr<AdaptiveNccModel> model;
	{
		auto temporary = learnLevel(std::move(image));
		model = std::make_unique<AdaptiveNccModel>(temporary, 64, true);
		std::fill_n(temporary.image.row(0), std::size_t(temporary.image.width())*temporary.image.height(), std::uint8_t(0));
		temporary.mean = -93; temporary.norm = 0;
	}
	require(exact(model->level().mean, snapshot.mean) && exact(model->level().norm, snapshot.norm),
		"model retained caller normalization metadata");
	require(std::memcmp(model->level().image.row(0), snapshot.image.row(0), 33*35) == 0,
		"model retained caller template pixels");
	const auto full = ncc(source.view(), snapshot, Backend::Scalar);
	for (double threshold : {0.0, .8, 1.0}) {
		NccStats stats; const auto actual = model->score(source.view(), threshold, Backend::Auto, &stats);
		validateMap(actual, full, snapshot, threshold, stats, "owned template");
	}
	for (int cap : {-1, 0, 1, 3, 5, 7, 15, 31, 63, 65})
		rejects([&] { AdaptiveNccModel invalid(snapshot, cap); });
	rejects([&] { model->score({}, .8); });
	rejects([&] { model->score({source.row(0), source.width(), source.height(), 1}, .8); });
	rejects([&] { model->score(Image(2, 2).view(), .8); });
	// Repeated calls must read changed source pixels, not retain a score map.
	invert(source); NccStats stats;
	const auto changed = ncc(source.view(), snapshot, Backend::Scalar);
	const auto actual = model->score(source.view(), .8, Backend::Auto, &stats);
	validateMap(actual, changed, snapshot, .8, stats, "changed source");
	for (double threshold : {std::numeric_limits<double>::infinity(),
		-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
		NccStats disabled;
		const auto map = model->score(source.view(), threshold, Backend::Auto, &disabled);
		require(map.values == changed.values && disabled.adaptiveChecks == 0,
			"nonfinite threshold did not bypass screening");
	}
	const int rounding = std::fegetround();
	require(std::fesetround(FE_DOWNWARD) == 0, "cannot set rounding mode");
	const bool supported = model->supports(source.view());
	const AdaptiveNccModel rounded(snapshot, 16);
	require(std::fesetround(rounding) == 0, "cannot restore rounding mode");
	require(!supported && rounded.rectangles() == 0, "non-nearest rounding must disable interval screening");
}

// Exercise just the new screen, without the older partial-dot pruning. Include
// the threshold exactly equal to the native score and its adjacent doubles.
void directBounds()
{
	std::size_t checks = 0, rejections = 0;
	for (int fixture = 0; fixture < 3; ++fixture) {
		Image templ(35, 33), source = noise(43, 40, 601+fixture);
		for (int y = 0; y < 33; ++y) for (int x = 0; x < 35; ++x)
			templ.row(y)[x] = fixture == 0 ? 20+70*(x>6)+90*(y>23) :
				fixture == 1 ? 200+((x+3*y)%7==0) : source.row(y)[x];
		paste(source, templ, 3, 4); source.row(4)[3] ^= 3;
		const auto level = learnLevel(std::move(templ));
		const auto full = ncc(source.view(), level, Backend::Scalar);
		const std::size_t stride = source.width()+1;
		std::vector<double> sums(stride*(source.height()+1)), squares(sums.size());
		for (int y = 0; y < source.height(); ++y) {
			double s=0, q=0;
			for (int x = 0; x < source.width(); ++x) {
				const double v=source.row(y)[x]; s+=v; q+=v*v;
				const auto i=(y+1)*stride+x+1;
				sums[i]=sums[i-stride]+s; squares[i]=squares[i-stride]+q;
			}
		}
		for (bool block : {false, true}) for (int cap : {4,8,16,32,64}) {
			const AdaptiveNccModel model(level, cap, block);
			for (int y = 0; y < full.height; ++y) for (int x = 0; x < full.width; ++x) {
				const auto a=y*stride+x, b=a+35, c=a+33*stride, d=c+35;
				const double sum=sums[d]-sums[b]-sums[c]+sums[a];
				const double square=squares[d]-squares[b]-squares[c]+squares[a];
				const double denominator=nccDenominator(std::max(0.,square-sum*sum/(35*33)),square,level.norm);
				const double score=full.at(x,y);
				for (double threshold : {.8, 1., score, std::nextafter(score, 0.),
					std::nextafter(score, std::numeric_limits<double>::infinity())}) {
					const bool rejected=model.reject(sums.data(),squares.data(),stride,a,sum,square,denominator,threshold,nullptr);
					require(!rejected || score < threshold, "direct adaptive screen falsely rejected native threshold");
					++checks; rejections+=rejected;
				}
			}
		}
	}
	require(rejections > 0, "direct screen test never rejected a window");
	std::cout << "Direct screen: " << checks << " boundaries, " << rejections << " safe rejections\n";
}

class ScreenProvider final : public CoarseNccProvider {
public:
	ScreenProvider(const Pattern& pattern, const MatchConfig& config, int cap, bool block)
		: model(pattern.levels().back(), cap, block), threshold(config.minScore), backend(config.backend)
	{
		for (std::size_t i = 1; i < pattern.levels().size(); ++i) threshold *= .9;
	}
	ScoreMap compute(ImageView source, const TemplateLevel& templ) override
	{
		require(templ.image.width() == model.level().image.width() &&
			templ.image.height() == model.level().image.height() &&
			exact(templ.mean, model.level().mean) && exact(templ.norm, model.level().norm),
			"adaptive provider received a different template level");
		++calls; return model.score(source, threshold, backend);
	}
	// fineBatchSize remains zero: production fine maps, subpixel fitting and
	// acceptance control run unchanged, including all below-threshold samples.
	AdaptiveNccModel model;
	double threshold;
	Backend backend;
	std::size_t calls = 0;
};
void equalMatches(const std::vector<Match>& actual, const std::vector<Match>& expected)
{
	require(actual.size() == expected.size(), "adaptive coarse changed full-match count");
	for (std::size_t i = 0; i < actual.size(); ++i) {
		const auto& a = actual[i]; const auto& e = expected[i];
		require(exact(a.score, e.score) && exact(a.angle, e.angle) &&
			exact(a.center.x, e.center.x) && exact(a.center.y, e.center.y),
			"adaptive coarse changed full-match score, pose or ordering");
		for (int corner = 0; corner < 4; ++corner)
			require(exact(a.corners[corner].x, e.corners[corner].x) && exact(a.corners[corner].y, e.corners[corner].y),
				"adaptive coarse changed full-match corners");
	}
}
void equalWork(const MatchStageProfile& a, const MatchStageProfile& e)
{
#define SAME(field) require(a.field == e.field, "adaptive coarse changed " #field)
	SAME(coarse_angles); SAME(coarse_maps); SAME(coarse_windows); SAME(coarse_candidates);
	SAME(fine_layer_visits); SAME(fine_passes); SAME(fine_rotations); SAME(fine_maps); SAME(fine_windows);
	SAME(refined_candidates); SAME(output_matches);
#undef SAME
}
void fullMatches()
{
	for (int mode = 0; mode < 6; ++mode) {
		const bool pyramid = mode == 1 || mode == 5;
		auto templ = noise(pyramid ? 41 : 21, pyramid ? 37 : 19, 913);
		auto source = noise(pyramid ? 127 : 83, pyramid ? 109 : 73, 881);
		paste(source, templ, 3, 5);
		paste(source, templ, source.width()-templ.width()-4, source.height()-templ.height()-3);
		const auto rotated = rotateCanvas(templ.view(), templ.width(), templ.height(), 7.25, 0);
		paste(source, rotated.image, (source.width()-rotated.image.width())/2, (source.height()-rotated.image.height())/2);
		MatchConfig config; config.minScore = .6; config.maxTargets = 5;
		config.toleranceAngle = mode == 0 ? 0 : 12;
		config.subpixel = mode == 2 || mode == 3 || mode == 5;
		config.maxOverlap = mode == 3 ? 1 : .2;
		config.stopLayer = mode == 5 ? 1 : 0;
		if (mode == 4) { config.invertSource = true; invert(source); config.angleRanges = {{-12, -2}, {0, 0}, {4, 12}}; }
		const Pattern pattern(templ.view(), pyramid ? 500 : 100000);
		MatchStageProfile fullProfile;
		const auto full = matchWithCoarseNcc(source.view(), pattern, config, nullptr, &fullProfile);
		require(!full.empty(), "full-match fixture did not retain a planted detection");
		config.coarseNccPruning = true;
		MatchStageProfile legacyProfile;
		equalMatches(matchWithCoarseNcc(source.view(), pattern, config, nullptr, &legacyProfile), full);
		equalWork(legacyProfile, fullProfile);
		for (bool block : {false, true}) for (int cap : {4, 16, 64}) {
			ScreenProvider provider(pattern, config, cap, block); MatchStageProfile profile;
			const auto actual = matchWithCoarseNcc(source.view(), pattern, config, &provider, &profile);
			equalMatches(actual, full); equalWork(profile, fullProfile);
			require(provider.calls == profile.coarse_maps && provider.calls > 0, "coarse-only provider was bypassed");
			++checkedMatches;
		}
	}
}
}

int main()
{
	try {
		maps(); ownershipAndValidation(); directBounds(); fullMatches();
		std::cout << "Adaptive NCC: " << checkedMaps << " exact/bounded maps, " << checkedMatches
			<< " full matches; " << adaptiveChecks << " adaptive checks, " << adaptiveRejections << " adaptive rejections passed\n";
		return 0;
	} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
