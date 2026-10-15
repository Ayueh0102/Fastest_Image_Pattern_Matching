// SPDX-License-Identifier: BSD-2-Clause
// CPU-only research worker. Timed calls never enable statistics or an oracle.
// A process measures exactly one input/method, then captures RSS BEFORE any
// exhaustive reference or profiling pass. Qt is used only for image/JSON I/O.
#include "adaptive_ncc.hpp"
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using fpm::detail::ScoreMap;
using fpm::detail::NccStats;
double elapsed(Clock::time_point start)
{ return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }

fpm::Image readImage(const QString& path)
{
	QImage image(path); image = image.convertToFormat(QImage::Format_RGB888);
	if (image.isNull()) throw std::invalid_argument("Cannot decode image");
	fpm::Image out(image.width(), image.height());
	for (int y = 0; y < image.height(); ++y)
		for (int x = 0; x < image.width(); ++x) {
			const auto* p = image.constScanLine(y)+3*x;
			out.row(y)[x] = fpm::grayFromRgb(p[0], p[1], p[2]);
		}
	return out;
}

QJsonObject memory()
{
	rusage usage{};
	if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("RSS unavailable");
	double peak = usage.ru_maxrss, resident = -1;
#ifndef __APPLE__
	peak *= 1024;
#else
	mach_task_basic_info info{}; mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
		reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) resident = info.resident_size;
#endif
	return {{"peak_rss_bytes", peak}, {"current_rss_bytes", resident}};
}

QJsonArray matches(const std::vector<fpm::Match>& values)
{
	QJsonArray out;
	for (const auto& m : values) {
		QJsonArray corners;
		for (const auto p : m.corners) corners.append(QJsonArray{p.x, p.y});
		out.append(QJsonObject{{"pos_x", m.center.x}, {"pos_y", m.center.y},
			{"angle", m.angle}, {"score", m.score}, {"scale", 1.}, {"corners", corners}});
	}
	return out;
}

QString mapHash(const ScoreMap& map)
{
	QCryptographicHash hash(QCryptographicHash::Sha256);
	// The hash is for repeatability within one architecture, not a portable
	// serialization contract. Cross-method comparisons use numerical oracles.
	hash.addData(QByteArrayView(reinterpret_cast<const char*>(&map.width), sizeof(map.width)));
	hash.addData(QByteArrayView(reinterpret_cast<const char*>(&map.height), sizeof(map.height)));
	for (std::size_t at = 0; at < map.values.size();) {
		const auto count = std::min<std::size_t>(map.values.size()-at, 1u<<20);
		hash.addData(QByteArrayView(reinterpret_cast<const char*>(map.values.data()+at), count*sizeof(double)));
		at += count;
	}
	return QString::fromLatin1(hash.result().toHex());
}

void accumulate(NccStats& a, const NccStats& b)
{
	a.windows += b.windows; a.projectionChecks += b.projectionChecks;
	a.partialChecks += b.partialChecks; a.rejected += b.rejected;
	a.fullDotProducts += b.fullDotProducts; a.dotPixels += b.dotPixels;
	a.adaptiveChecks += b.adaptiveChecks; a.adaptiveRectangles += b.adaptiveRectangles;
	a.adaptiveRejected += b.adaptiveRejected;
}

QJsonObject statsJson(const NccStats& s)
{
	return {{"windows", double(s.windows)}, {"projection_checks", double(s.projectionChecks)},
		{"partial_checks", double(s.partialChecks)}, {"rejected", double(s.rejected)},
		{"full_dot_products", double(s.fullDotProducts)}, {"dot_pixels", double(s.dotPixels)},
		{"adaptive_checks", double(s.adaptiveChecks)}, {"adaptive_rectangles", double(s.adaptiveRectangles)},
		{"adaptive_rejected", double(s.adaptiveRejected)}};
}

struct Validation {
	std::uint64_t maps = 0, windows = 0, survivors = 0, rejected = 0;
	std::uint64_t aboveThreshold = 0, falseRejects = 0, survivorMismatches = 0, invalidValues = 0;
	bool peakOrderExact = true;
	bool valid() const { return !falseRejects && !survivorMismatches && !invalidValues && peakOrderExact; }
	QJsonObject json() const {
		return {{"valid", valid()}, {"maps", double(maps)}, {"windows", double(windows)},
			{"survivors", double(survivors)}, {"rejected", double(rejected)},
			{"above_threshold", double(aboveThreshold)}, {"false_rejects", double(falseRejects)},
			{"survivor_bit_mismatches", double(survivorMismatches)},
			{"invalid_values", double(invalidValues)}, {"peak_order_exact", peakOrderExact}};
	}
};

bool samePeaks(const std::vector<fpm::detail::Peak>& a, const std::vector<fpm::detail::Peak>& b)
{
	if (a.size() != b.size()) return false;
	for (std::size_t i = 0; i < a.size(); ++i)
		if (a[i].x != b[i].x || a[i].y != b[i].y ||
			std::memcmp(&a[i].score, &b[i].score, sizeof(double))) return false;
	return true;
}

void validateMap(const ScoreMap& candidate, const ScoreMap& oracle,
	const fpm::TemplateLevel& templ, double threshold, const fpm::MatchConfig& config,
	bool blocked, Validation& v)
{
	if (candidate.width != oracle.width || candidate.height != oracle.height ||
		candidate.values.size() != oracle.values.size()) throw std::runtime_error("Map shape differs from oracle");
	++v.maps; v.windows += candidate.values.size();
	for (std::size_t i = 0; i < candidate.values.size(); ++i) {
		const double a = candidate.values[i], b = oracle.values[i];
		if (b >= threshold) ++v.aboveThreshold;
		if (a == -std::numeric_limits<double>::infinity()) {
			++v.rejected;
			if (b >= threshold) ++v.falseRejects;
		} else {
			++v.survivors;
			if (!std::isfinite(a)) ++v.invalidValues;
			if (std::memcmp(&a, &b, sizeof(double))) ++v.survivorMismatches;
		}
	}
	const int count = config.maxTargets+fpm::detail::EXTRA_CANDIDATES;
	const auto a = fpm::detail::selectPeaks(candidate, templ.image.width(), templ.image.height(),
		count, config.maxOverlap, threshold, blocked);
	const auto b = fpm::detail::selectPeaks(oracle, templ.image.width(), templ.image.height(),
		count, config.maxOverlap, threshold, blocked);
	v.peakOrderExact &= samePeaks(a, b);
}

class Provider final : public fpm::detail::CoarseNccProvider {
public:
	Provider(const fpm::TemplateLevel& level, QString method, int cap, double threshold,
		const fpm::MatchConfig& config, bool blocked)
		: level_(level), method_(std::move(method)), threshold_(threshold), config_(config), blocked_(blocked)
	{
		if (method_ == "global" || method_ == "block") {
			const auto start = Clock::now();
			model_ = std::make_unique<fpm::experimental::AdaptiveNccModel>(level_, cap, method_ == "block");
			buildMs = elapsed(start);
		}
	}
	ScoreMap compute(fpm::ImageView source, const fpm::TemplateLevel& templ) override {
		if (&templ != &level_) throw std::runtime_error("Provider received a different template level");
		NccStats current;
		auto* count = diagnostics ? &current : nullptr;
		auto out = model_ ? model_->score(source, threshold_, config_.backend, count) :
			fpm::detail::ncc(source, templ, config_.backend,
				method_ == "legacy" ? threshold_ : -std::numeric_limits<double>::infinity(), count);
		if (diagnostics) {
			accumulate(stats, current);
			const auto oracle = fpm::detail::ncc(source, templ, config_.backend);
			validateMap(out, oracle, templ, threshold_, config_, blocked_, validation);
		}
		return out;
	}
	std::size_t retainedBytes() const { return model_ ? model_->retainedBytes() : 0; }
	std::size_t workspaceBytes() const { return model_ ? model_->buildWorkspaceBytes() : 0; }
	int rectangles() const { return model_ ? model_->rectangles() : 0; }
	double buildMs = 0;
	bool diagnostics = false;
	NccStats stats;
	Validation validation;
private:
	const fpm::TemplateLevel& level_;
	QString method_;
	double threshold_;
	fpm::MatchConfig config_;
	bool blocked_;
	std::unique_ptr<fpm::experimental::AdaptiveNccModel> model_;
};

fpm::MatchConfig matchConfig(const QJsonObject& j)
{
	fpm::MatchConfig c;
	c.minScore = j["score"].toDouble(.8);
	c.maxTargets = j["limit"].toInt(70);
	c.maxOverlap = j["overlap"].toDouble(.2);
	c.toleranceAngle = j["angle"].toDouble(180);
	c.stopLayer = j["stop_layer"].toInt(0);
	c.subpixel = j["subpixel"].toBool(false);
	c.invertSource = j["invert_source"].toBool(false);
	const auto backend = j["backend"].toString("auto");
	if (backend != "auto" && backend != "scalar") throw std::invalid_argument("Backend must be auto or scalar");
	c.backend = backend == "auto" ? fpm::Backend::Auto : fpm::Backend::Scalar;
	if (!std::isfinite(c.minScore) || c.minScore <= 0 || c.minScore > 1 ||
		c.maxTargets < 1 || c.maxTargets > 100000 || !std::isfinite(c.maxOverlap) ||
		c.maxOverlap < 0 || c.maxOverlap > 1 || !std::isfinite(c.toleranceAngle) ||
		c.toleranceAngle < 0 || c.toleranceAngle > 180 || c.stopLayer < 0)
		throw std::invalid_argument("Invalid matching settings");
	return c;
}

QJsonObject profileJson(const fpm::detail::MatchStageProfile& p)
{
	return {{"core_wall_ms", p.core_wall_ms}, {"source_copy_invert_ms", p.source_copy_invert_ms},
		{"pyramid_ms", p.pyramid_ms}, {"coarse_rotation_ms", p.coarse_rotation_ms},
		{"coarse_ncc_ms", p.coarse_ncc_ms}, {"peak_selection_ms", p.peak_selection_ms},
		{"fine_rotation_ms", p.fine_rotation_ms}, {"fine_ncc_ms", p.fine_ncc_ms},
		{"fine_control_ms", p.fine_control_ms}, {"nms_ms", p.nms_ms},
		{"coarse_maps", double(p.coarse_maps)}, {"coarse_windows", double(p.coarse_windows)},
		{"coarse_candidates", double(p.coarse_candidates)}, {"fine_maps", double(p.fine_maps)},
		{"fine_windows", double(p.fine_windows)}, {"output_matches", double(p.output_matches)}};
}
}

int main(int argc, char** argv)
{
	QCoreApplication app(argc, argv);
	try {
		if (argc != 2) throw std::invalid_argument("Usage: adaptive_ncc_runner job.json");
		QFile file(QString::fromLocal8Bit(argv[1]));
		if (!file.open(QIODevice::ReadOnly)) throw std::invalid_argument("Cannot open job");
		const auto j = QJsonDocument::fromJson(file.readAll()).object();
		if (j.isEmpty() || j.contains("truth")) throw std::invalid_argument("Expected job without ground truth");
		const auto mode = j["mode"].toString("map"), method = j["method"].toString("full");
		if (mode != "map" && mode != "full") throw std::invalid_argument("Mode must be map or full");
		if (method != "full" && method != "legacy" && method != "global" && method != "block")
			throw std::invalid_argument("Unknown NCC method");
		const int cap = j["cap"].toInt(16), repeats = j["repeats"].toInt(5);
		const double warmup = j["warmup_seconds"].toDouble(.2);
		if ((cap != 4 && cap != 8 && cap != 16 && cap != 32 && cap != 64) || repeats < 1 ||
			repeats > 10000 || !std::isfinite(warmup) || warmup < .2 || warmup > 60)
			throw std::invalid_argument("Invalid cap, repetitions, or warmup (minimum 0.2 seconds)");
		auto source = readImage(j["source"].toString());
		auto templ = readImage(j["template"].toString());
		const int originalWidth = source.width(), originalHeight = source.height();
		QJsonArray crop{0, 0, originalWidth, originalHeight};
		if (j.contains("source_crop")) {
			if (mode != "map") throw std::invalid_argument("Source crops are allowed only for map microbenchmarks");
			crop = j["source_crop"].toArray();
			if (crop.size() != 4) throw std::invalid_argument("source_crop must be [x,y,width,height]");
			for (const auto value : crop)
				if (!value.isDouble() || value.toDouble() != value.toInt(-1)) throw std::invalid_argument("Non-integer crop");
			const int x = crop[0].toInt(), y = crop[1].toInt(), w = crop[2].toInt(), h = crop[3].toInt();
			if (x < 0 || y < 0 || w <= 0 || h <= 0 || x > source.width()-w || y > source.height()-h)
				throw std::invalid_argument("Invalid source crop");
			source = fpm::Image({source.row(y)+x, w, h, std::size_t(source.width())});
		}
		const auto config = matchConfig(j);
		const auto patternStart = Clock::now();
		fpm::Pattern pattern(templ.view(), j["min_area"].toInt(256));
		const double patternBuildMs = elapsed(patternStart);
		const int top = int(pattern.levels().size())-1;
		const int layer = mode == "full" ? top : j["level"].toInt(0);
		if (layer < 0 || layer > top) throw std::invalid_argument("Requested map level is outside template pyramid");
		const auto prepareStart = Clock::now();
		if (mode == "map") {
			if (config.invertSource)
				for (int y = 0; y < source.height(); ++y)
					for (int x = 0; x < source.width(); ++x) source.row(y)[x] = 255-source.row(y)[x];
			for (int i = 0; i < layer; ++i) source = fpm::detail::pyrDown(source.view());
		}
		const double sourcePrepareMs = elapsed(prepareStart);
		const auto& level = pattern.levels()[layer];
		if (source.width() < level.image.width() || source.height() < level.image.height())
			throw std::invalid_argument("Source does not fit template");
		double threshold = config.minScore;
		for (int i = 0; i < layer; ++i) threshold *= .9;
		if (mode == "map" && j.contains("threshold")) threshold = j["threshold"].toDouble(-1);
		if (!std::isfinite(threshold) || threshold <= 0 || threshold > 1) throw std::invalid_argument("Invalid threshold");
		int coarseWidth = source.width(), coarseHeight = source.height();
		if (mode == "full") for (int i = 0; i < top; ++i) {
			coarseWidth = coarseWidth/2+coarseWidth%2; coarseHeight = coarseHeight/2+coarseHeight%2;
		}
		const bool blocked = double(coarseWidth)*coarseHeight /
			(double(level.image.width())*level.image.height()) > 500 && config.maxTargets > 10;
		const auto beforeModelMemory = memory();
		Provider provider(level, method, cap, threshold, config, blocked);
		const auto afterModelMemory = memory();
		QString referenceHash; QJsonArray referenceMatches;
		bool exactRepeat = true;
		auto run = [&](bool first) {
			double ms = 0;
			if (mode == "map") {
				const auto start = Clock::now();
				const auto map = provider.compute(source.view(), level);
				ms = elapsed(start);
				const auto hash = mapHash(map);
				if (first) referenceHash = hash;
				else exactRepeat &= hash == referenceHash;
			} else {
				const auto start = Clock::now();
				const auto result = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, &provider);
				ms = elapsed(start);
				const auto output = matches(result);
				if (first) referenceMatches = output;
				else exactRepeat &= output == referenceMatches;
			}
			return ms;
		};
		const double firstMs = run(true);
		QJsonArray warmups, samples;
		const auto warmStart = Clock::now();
		do { warmups.append(run(false)); } while (elapsed(warmStart) < warmup*1000);
		for (int i = 0; i < repeats; ++i) samples.append(run(false));
		// Capture before stats, oracle maps, peak copies, or exhaustive matching.
		const auto measuredMemory = memory();
		provider.diagnostics = true;
		bool exactFinal = true;
		QJsonObject profile;
		if (mode == "map") {
			const auto diagnosed = provider.compute(source.view(), level);
			exactFinal = mapHash(diagnosed) == referenceHash;
		} else {
			const auto diagnosed = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, &provider);
			const auto oracle = fpm::match(source.view(), pattern, config);
			exactFinal = matches(diagnosed) == referenceMatches && matches(oracle) == referenceMatches;
			provider.diagnostics = false;
			fpm::detail::MatchStageProfile measured;
			const auto profiled = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, &provider, &measured);
			exactFinal &= matches(profiled) == referenceMatches;
			profile = profileJson(measured);
		}
		const bool valid = exactRepeat && exactFinal && provider.validation.valid();
		QJsonObject result{{"schema", 1}, {"mode", mode}, {"method", method}, {"cap", cap},
			{"rectangles", provider.rectangles()}, {"backend", j["backend"].toString("auto")},
			{"compiled_backend", fpm::compiledBackend()}, {"level", layer}, {"pyramid_top", top},
			{"threshold", threshold}, {"source_original_shape", QJsonArray{originalHeight, originalWidth}},
			{"source_crop", crop}, {"source_shape", QJsonArray{source.height(), source.width()}},
			{"template_shape", QJsonArray{level.image.height(), level.image.width()}},
			{"pattern_build_ms", patternBuildMs}, {"source_prepare_ms", sourcePrepareMs},
			{"model_build_ms", provider.buildMs}, {"model_retained_bytes", double(provider.retainedBytes())},
			{"model_build_workspace_bytes", double(provider.workspaceBytes())},
			{"memory_before_model", beforeModelMemory}, {"memory_after_model", afterModelMemory},
			{"first_ms", firstMs}, {"warmups_ms", warmups}, {"samples_ms", samples},
			{"measured_memory", measuredMemory}, {"statistics", statsJson(provider.stats)},
			{"validation", provider.validation.json()}, {"exact_repeat", exactRepeat},
			{"exact_final", exactFinal}, {"valid", valid}, {"profile_after_timing", profile},
			{"full_search", mode == "full"}, {"fine_path_changed", false},
			{"rss_includes_oracle", false}, {"timed_calls_collect_stats", false}};
		if (mode == "map") result["map_sha256_native"] = referenceHash;
		else result["matches"] = referenceMatches;
		std::cout << QJsonDocument(result).toJson(QJsonDocument::Compact).constData() << '\n';
		return valid ? 0 : 1;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n'; return 1;
	}
}
