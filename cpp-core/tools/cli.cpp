// SPDX-License-Identifier: BSD-2-Clause
#include "cli.hpp"
#include "fpm/chroma.hpp"
#include "fpm/matcher.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef FPM_WITH_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif

namespace {
std::string token(std::istream& stream)
{
	std::string text;
	while (stream >> text) {
		if (text[0] != '#') return text;
		stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	throw std::runtime_error("incomplete PGM header");
}
int integer(const std::string& s)
{
	size_t end = 0;
	const int v = std::stoi(s, &end);
	if (end != s.size()) throw std::invalid_argument("invalid integer: " + s);
	return v;
}
double number(const std::string& s)
{
	size_t end = 0;
	const double v = std::stod(s, &end);
	if (end != s.size() || !std::isfinite(v)) throw std::invalid_argument("invalid number: " + s);
	return v;
}
#ifdef FPM_WITH_OPENCV
cv::Mat readColour(const std::string& path)
{
	const auto image = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
	if (image.empty()) throw std::runtime_error("cannot load image: " + path);
	return image;
}
fpm::ColourImageView bgrView(const cv::Mat& image)
{
	return {image.data,image.cols,image.rows,image.step[0],true};
}
#endif
fpm::Image load(const std::string& path)
{
#ifdef FPM_WITH_OPENCV
	const auto image=readColour(path);
	fpm::Image gray(image.cols, image.rows);
	for (int y = 0; y < image.rows; ++y)
		for (int x = 0; x < image.cols; ++x) {
			const auto& p = image.at<cv::Vec3b>(y, x);
			gray.row(y)[x] = fpm::grayFromRgb(p[2], p[1], p[0]);
		}
	return gray;
#else
	std::ifstream stream(path, std::ios::binary);
	if (!stream) throw std::runtime_error("cannot open image: " + path);
	if (token(stream) != "P5") throw std::runtime_error("this build reads P5 PGM only; enable FPM_WITH_OPENCV for BMP/PNG/JPEG");
	const int width = integer(token(stream)), height = integer(token(stream));
	if (integer(token(stream)) != 255) throw std::runtime_error("PGM max value must be 255");
	const int separator = stream.get();
	if (separator == '\r' && stream.peek() == '\n') stream.get();
	else if (separator != ' ' && separator != '\t' && separator != '\n' && separator != '\r') throw std::runtime_error("invalid PGM separator");
	fpm::Image image(width, height);
	for (int y = 0; y < height; ++y)
		if (!stream.read(reinterpret_cast<char*>(image.row(y)), width)) throw std::runtime_error("truncated PGM pixels");
	return image;
#endif
}
}

int runPortableCli(int argc, char** argv)
{
	try {
		std::string sourcePath, templatePath;
		fpm::MatchConfig config;
		int minArea = 256;
		bool useChroma = false;
		for (int i = 1; i < argc; ++i) {
			const std::string arg = argv[i];
			auto value = [&]() -> std::string {
				if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
				return argv[++i];
			};
			if (arg == "--help" || arg == "-h") {
				std::cout << "Usage: fpm_cli --src image --tmpl template [options]\n"
					"  --target-num N  --score 0..1  --overlap 0..1  --angle 0..180\n"
					"  --angle-range MIN MAX (repeatable; OUTPUT angles in degrees CCW)\n"
					"  --min-area N  --subpixel  --invert  --no-simd  --fast\n"
#ifdef FPM_WITH_OPENCV
					"  --chroma  Locate by template colour contrast (Co/Cg)\n"
#endif
					"Output: JSON; match_ms excludes file loading and learning.\n";
				return 0;
			}
			if (arg == "--src") sourcePath = value();
			else if (arg == "--tmpl") templatePath = value();
			else if (arg == "--target-num") config.maxTargets = integer(value());
			else if (arg == "--score") config.minScore = number(value());
			else if (arg == "--overlap") config.maxOverlap = number(value());
			else if (arg == "--angle") config.toleranceAngle = number(value());
			else if (arg == "--min-area") minArea = integer(value());
			else if (arg == "--angle-range") {
				const double low = number(value()), high = number(value());
				config.angleRanges.push_back({low, high});
			}
			else if (arg == "--subpixel") config.subpixel = true;
			else if (arg == "--chroma") useChroma = true;
			else if (arg == "--invert") config.invertSource = true;
			else if (arg == "--no-simd") config.backend = fpm::Backend::Scalar;
			else if (arg == "--fast") config.stopLayer = 1;
			else if (arg != "--no-display") throw std::invalid_argument("unknown option: " + arg);
		}
		if (sourcePath.empty() || templatePath.empty()) throw std::invalid_argument("--src and --tmpl are required");
		using Clock = std::chrono::steady_clock;
		fpm::Image source, grayTemplate;
		fpm::ImageView templateView;
		double featureMs = 0;
		std::unique_ptr<fpm::ChromaTemplate> colourTemplate;
		if (useChroma) {
#ifdef FPM_WITH_OPENCV
			const auto rgbTemplate = readColour(templatePath), rgbSource = readColour(sourcePath);
			const auto start = Clock::now();
			colourTemplate = std::make_unique<fpm::ChromaTemplate>(bgrView(rgbTemplate));
			templateView = colourTemplate->feature().view();
			source = colourTemplate->project(bgrView(rgbSource));
			featureMs = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
#else
			throw std::invalid_argument("--chroma requires an image-codec build with FPM_WITH_OPENCV");
#endif
		} else {
			source = load(sourcePath); grayTemplate = load(templatePath); templateView = grayTemplate.view();
		}
		const auto t0 = Clock::now();
		const fpm::Pattern pattern(templateView, minArea);
		const auto t1 = Clock::now();
		const auto matches = fpm::match(source.view(), pattern, config);
		const auto t2 = Clock::now();
		std::cout << std::setprecision(12) << '{';
		if (useChroma) std::cout << "\"search_feature\":\"chroma_cocg\",\"feature_ms\":" << featureMs << ',';
		std::cout << "\"backend\":\"" << (config.backend == fpm::Backend::Scalar ? "scalar" : fpm::compiledBackend())
			<< "\",\"learn_ms\":" << std::chrono::duration<double, std::milli>(t1 - t0).count()
			<< ",\"match_ms\":" << std::chrono::duration<double, std::milli>(t2 - t1).count() << ",\"matches\":[";
		for (size_t i = 0; i < matches.size(); ++i) {
			const auto& r = matches[i];
			if (i) std::cout << ',';
			std::cout << "{\"index\":" << i << ",\"score\":" << r.score << ",\"angle\":" << r.angle
				<< ",\"pos_x\":" << r.center.x << ",\"pos_y\":" << r.center.y << ",\"corners\":[";
			for (int k = 0; k < 4; ++k) {
				if (k) std::cout << ',';
				std::cout << '[' << r.corners[k].x << ',' << r.corners[k].y << ']';
			}
			std::cout << "]}";
		}
		std::cout << "]}\n";
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "fpm: " << error.what() << '\n';
		return 1;
	}
}
