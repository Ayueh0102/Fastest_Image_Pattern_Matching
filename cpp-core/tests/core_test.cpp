// SPDX-License-Identifier: BSD-2-Clause
// Self-contained test runner so the reference core builds/tests offline.
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm;
using namespace fpm::detail;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(double value, double expected, double tolerance = 1e-9)
{
	if (!std::isfinite(value) || std::abs(value - expected) > tolerance)
		throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(value));
}
template<class F> void invalid(F f)
{
	try { f(); } catch (const std::invalid_argument&) { return; }
	throw std::runtime_error("expected invalid_argument");
}
Image noise(int w, int h, unsigned seed)
{
	Image out(w, h);
	std::mt19937 rng(seed);
	for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out.row(y)[x] = std::uint8_t(rng() % 256);
	return out;
}
void paste(Image& out, const Image& patch, int x, int y)
{
	for (int row = 0; row < patch.height(); ++row) std::copy_n(patch.row(row), patch.width(), out.row(y + row) + x);
}
void imageAndPyramid()
{
	near(grayFromRgb(255,0,0),77); near(grayFromRgb(0,255,0),149);
	near(grayFromRgb(0,0,255),29); near(grayFromRgb(255,255,255),255);
	std::vector<std::uint8_t> storage(5 * 9, 77);
	ImageView view{storage.data(), 5, 5, 9};
	Image owned(view);
	storage[0] = 0;
	require(owned.row(0)[0] == 77, "image must own a copy");
	auto small = pyrDown(owned.view());
	require(small.width() == 3 && small.height() == 3, "odd dimensions use ceil half");
	for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x) near(small.row(y)[x], 77);
	Image ramp(5, 1);
	for (int x = 0; x < 5; ++x) ramp.row(0)[x] = x * 16;
	const auto reduced = pyrDown(ramp.view());
	near(reduced.row(0)[0], 12); near(reduced.row(0)[1], 32); near(reduced.row(0)[2], 52);
	Pattern p(owned.view(), 4);
	require(p.levels().size() == 3, "legacy area-based pyramid depth");
	near(p.levels()[0].norm, 0);
	invalid([&] { Pattern bad(view, 0); });
	invalid([&] { Pattern bad({storage.data(), 5, 5, 4}); });
	invalid([] { Pattern bad({}); });
}
void nccReference()
{
	const auto image = noise(19, 17, 10), patch = noise(7, 5, 20);
	const auto templ = learnLevel(Image(patch.view()));
	std::vector<std::uint8_t> padded(17 * 25);
	for (int y = 0; y < 17; ++y) std::copy_n(image.row(y), 19, padded.data() + y * 25);
	const auto scores = ncc({padded.data(), 19, 17, 25}, templ, Backend::Scalar);
	const auto simd = ncc(image.view(), templ, Backend::Auto);
	for (int y = 0; y < scores.height; ++y) for (int x = 0; x < scores.width; ++x) {
		double mean = 0;
		for (int j = 0; j < 5; ++j) for (int i = 0; i < 7; ++i) mean += image.row(y+j)[x+i];
		mean /= 35;
		double cross = 0, variance = 0;
		for (int j = 0; j < 5; ++j) for (int i = 0; i < 7; ++i) {
			const double d = image.row(y+j)[x+i] - mean;
			cross += d * (patch.row(j)[i] - templ.mean); variance += d*d;
		}
		near(scores.at(x,y), cross / std::sqrt(variance) / templ.norm);
		near(simd.at(x,y), scores.at(x,y), 0);
	}
	Image flat(7,5);
	const auto constantTemplate = ncc(image.view(), learnLevel(flat), Backend::Auto);
	for (double v : constantTemplate.values) near(v, 1);
	const auto constantSource = ncc(Image(19,17).view(), templ, Backend::Auto);
	for (double v : constantSource.values) near(v, 0);
	const auto exact = ncc(patch.view(), templ, Backend::Auto);
	near(exact.at(0,0), 1);
	Image inverted(patch.view());
	for (int y=0;y<5;++y) for (int x=0;x<7;++x) inverted.row(y)[x] = 255-inverted.row(y)[x];
	near(ncc(inverted.view(), templ, Backend::Auto).at(0,0), -1);
}
void nccPruning()
{
	auto verify = [](ImageView source, const TemplateLevel& templ, double threshold, Backend backend) {
		const auto full = ncc(source, templ, backend);
		NccStats stats;
		const auto bounded = ncc(source, templ, backend, threshold, &stats);
		require(stats.windows == full.values.size(), "NCC stats window count");
		for (std::size_t i = 0; i < full.values.size(); ++i) {
			if (std::isfinite(bounded.values[i])) near(bounded.values[i], full.values[i], 0);
			else require(full.values[i] < threshold, "bound must not reject a score at or above threshold");
		}
		const auto a = selectPeaks(full, templ.image.width(), templ.image.height(), 50, .6, threshold, true);
		const auto b = selectPeaks(bounded, templ.image.width(), templ.image.height(), 50, .6, threshold, true);
		require(a.size() == b.size(), "pruned peak count must match exhaustive");
		for (std::size_t i = 0; i < a.size(); ++i)
			require(a[i].x == b[i].x && a[i].y == b[i].y && a[i].score == b[i].score,
				"pruning must preserve peak order and scores");
		return stats;
	};
	for (int size : {16, 31, 32, 65}) {
		auto patch = noise(size, size + 2, 741 + size);
		auto source = noise(130, 121, 502 + size);
		paste(source, patch, 5, 8);
		const auto templ = learnLevel(patch);
		for (Backend backend : {Backend::Scalar, Backend::Auto})
			for (double threshold : {-1.0, -.5, 0.0, .5, .8, .99, 1.0}) {
				const auto stats = verify(source.view(), templ, threshold, backend);
				if (size >= 32 && threshold >= .8) require(stats.rejected > 0 && stats.partialChecks > 0,
					"large random template should exercise partial-dot bounds");
			}
		// Equality and adjacent representable thresholds at a nonperfect peak.
		source.row(8)[5] ^= 63;
		const double equality = ncc(source.view(), templ, Backend::Auto).at(5, 8);
		for (double threshold : {std::nextafter(equality, 0.0), equality, std::nextafter(equality, 1.0)})
			verify(source.view(), templ, threshold, Backend::Auto);
	}
	Image blockPatch(32, 34);
	for (int y = 0; y < 34; ++y) for (int x = 0; x < 32; ++x)
		blockPatch.row(y)[x] = std::uint8_t(20 + (x >= 16 ? 100 : 0) + (y >= 17 ? 70 : 0));
	auto source = noise(111, 103, 2001);
	paste(source, blockPatch, 17, 31);
	const auto structured = learnLevel(blockPatch);
	const auto projected = verify(source.view(), structured, .8, Backend::Auto);
	require(projected.projectionChecks > 0 && projected.rejected > 0, "rectangular projection bound exercised");
	// Odd support, padded stride, low variance and negative legacy saturation.
	for (int y = 0; y < source.height(); ++y) for (int x = 0; x < source.width(); ++x)
		source.row(y)[x] = 128 + (source.row(y)[x] % 2);
	auto low = noise(35, 33, 17);
	for (int y = 0; y < low.height(); ++y) for (int x = 0; x < low.width(); ++x)
		low.row(y)[x] = 128 + (low.row(y)[x] % 2);
	paste(source, low, 23, 17);
	std::vector<std::uint8_t> padded(120 * 103, 231);
	for (int y = 0; y < 103; ++y) std::copy_n(source.row(y), 111, padded.data() + y * 120);
	verify({padded.data(), 111, 103, 120}, learnLevel(low), .95, Backend::Auto);
	verify(Image(111, 103).view(), structured, .8, Backend::Auto);
	verify(source.view(), learnLevel(Image(32, 34)), 1, Backend::Auto);
	Image negative(blockPatch.view());
	for (int y = 0; y < 34; ++y) for (int x = 0; x < 32; ++x) negative.row(y)[x] = 255 - negative.row(y)[x];
	verify(negative.view(), structured, .8, Backend::Auto);
	verify(negative.view(), structured, -1, Backend::Auto);
	MatchConfig config; config.minScore = .8; config.maxTargets = 3; config.subpixel = true;
	// No reduction ensures a useful coarse bound, then fine maps are reused
	// by subpixel fitting. Full end-to-end results must remain identical.
	const Pattern pattern(low.view(), 10000);
	const auto full = fpm::match(source.view(), pattern, config);
	config.coarseNccPruning = true;
	const auto bounded = fpm::match(source.view(), pattern, config);
	require(full.size() == bounded.size() && !full.empty(), "bounded matcher count");
	for (std::size_t i = 0; i < full.size(); ++i) {
		near(bounded[i].score, full[i].score, 0); near(bounded[i].angle, full[i].angle, 0);
		near(bounded[i].center.x, full[i].center.x, 0); near(bounded[i].center.y, full[i].center.y, 0);
	}
}
void simdOverflow()
{
	std::vector<std::uint8_t> a(200003,255), b(a);
	for (int length : {1,15,16,17,31,32,33,47,48,4095,4096,4097,4111,4112,4127,4128,33025,65536,66052,200003}) {
		const std::uint64_t expected = std::uint64_t(length)*255*255;
		require(dotProduct(a.data(),b.data(),length,Backend::Scalar)==expected,"scalar overflow");
		require(dotProduct(a.data(),b.data(),length,Backend::Auto)==expected,"SIMD overflow/tail");
	}
	// Every SIMD block/tail split and misalignment. Each operand ends exactly
	// at its allocation, so sanitizer builds also catch a read past the end.
	std::mt19937 rng(2024);
	for (int length = 0; length <= 300; ++length)
		for (int offset = 0; offset < 4; ++offset) {
			std::vector<std::uint8_t> x(length + offset), y(length + offset);
			for (auto& v : x) v = std::uint8_t(rng());
			for (auto& v : y) v = std::uint8_t(rng());
			if (length % 7 == 0) std::fill(x.begin(), x.end(), 255);
			std::uint64_t expected = 0;
			for (int i = 0; i < length; ++i) expected += unsigned(x[offset + i]) * y[offset + i];
			require(dotProduct(x.data()+offset,y.data()+offset,length,Backend::Scalar)==expected,"scalar random dot");
			require(dotProduct(x.data()+offset,y.data()+offset,length,Backend::Auto)==expected,"SIMD random dot");
		}
}
// Verbatim copies of the original rotation and pyramid loops. Library paths
// (SIMD rotation pairs, direct pyramid interiors) must match them byte for byte.
std::uint8_t referencePixel(ImageView src, int x, int y, std::uint8_t border)
{
	return x < 0 || y < 0 || x >= src.width || y >= src.height ? border :
		src.data[std::size_t(y) * src.stride + x];
}
Image referenceWarp(ImageView src, int width, int height, Point anchor, double angle, std::uint8_t border)
{
	Image result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			const double sx = anchor.x + c * x - s * y;
			const double sy = anchor.y + s * x + c * y;
			if (sx < -1 || sy < -1 || sx >= src.width || sy >= src.height) {
				result.row(y)[x] = border;
				continue;
			}
			const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
			const double dx = sx - ix, dy = sy - iy;
			const double top = referencePixel(src, ix, iy, border) * (1 - dx) + referencePixel(src, ix + 1, iy, border) * dx;
			const double bottom = referencePixel(src, ix, iy + 1, border) * (1 - dx) + referencePixel(src, ix + 1, iy + 1, border) * dx;
			result.row(y)[x] = std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
		}
	return result;
}
int referenceReflect(int p, int length)
{
	if (length == 1) return 0;
	while (p < 0 || p >= length)
		p = p < 0 ? -p : 2 * length - p - 2;
	return p;
}
Image referencePyrDown(ImageView image)
{
	Image result(image.width / 2 + image.width % 2, image.height / 2 + image.height % 2);
	constexpr int kernel[] = {1, 4, 6, 4, 1};
	std::vector<int> horizontal(std::size_t(image.height) * result.width());
	for (int y = 0; y < image.height; ++y) {
		const auto* row = image.data + std::size_t(y) * image.stride;
		for (int x = 0; x < result.width(); ++x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * row[referenceReflect(x * 2 + k, image.width)];
			horizontal[std::size_t(y) * result.width() + x] = sum;
		}
	}
	for (int y = 0; y < result.height(); ++y)
		for (int x = 0; x < result.width(); ++x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * horizontal[std::size_t(referenceReflect(y * 2 + k, image.height)) * result.width() + x];
			result.row(y)[x] = std::uint8_t((sum + 128) / 256);
		}
	return result;
}
bool samePixels(const Image& a, const Image& b)
{
	if (a.width() != b.width() || a.height() != b.height()) return false;
	for (int y = 0; y < a.height(); ++y)
		if (!std::equal(a.row(y), a.row(y) + a.width(), b.row(y))) return false;
	return true;
}
void exactRotationAndPyramid()
{
	std::mt19937 rng(77);
	for (int trial = 0; trial < 3000; ++trial) {
		// Small sizes, padded strides and anchors far outside reach every
		// border/interior mix; the last trials use a larger random image.
		const bool large = trial >= 2900;
		const int w = large ? 180 + int(rng() % 60) : 1 + int(rng() % 40);
		const int h = large ? 150 + int(rng() % 60) : 1 + int(rng() % 40);
		const int pad = int(rng() % 5);
		std::vector<std::uint8_t> pixels(std::size_t(w + pad) * h);
		for (auto& v : pixels) v = std::uint8_t(rng());
		const ImageView src{pixels.data(), w, h, std::size_t(w + pad)};
		require(samePixels(pyrDown(src), referencePyrDown(src)), "pyrDown must match the original loop");
		const double angles[] = {0, 90, -90, 180, 45, 1e-9, -37.25, double(rng() % 36000) / 100 - 180};
		const double angle = angles[rng() % 8];
		const int tw = 1 + int(rng() % (large ? 120 : 30)), th = 1 + int(rng() % (large ? 100 : 30));
		const Point anchor{double(int(rng() % 160) - 50) / 3, double(int(rng() % 160) - 50) / 7};
		const Point origin = rotate({-PADDING, -PADDING}, {0, 0}, -angle);
		require(samePixels(rotatedRoi(src, tw, th, anchor, angle), referenceWarp(src, tw + 2 * PADDING,
			th + 2 * PADDING, {anchor.x + origin.x, anchor.y + origin.y}, angle, 0)), "rotated ROI must match the original loop");
		const std::uint8_t border = rng() % 2 ? 255 : 0;
		const auto canvas = rotateCanvas(src, std::min(tw, w), std::min(th, h), angle, border);
		const Point center{(w - 1) * 0.5, (h - 1) * 0.5};
		const Point canvasAnchor = rotate({-canvas.translation.x, -canvas.translation.y}, center, -angle);
		require(samePixels(canvas.image, referenceWarp(src, canvas.image.width(), canvas.image.height(),
			canvasAnchor, angle, border)), "rotation canvas must match the original loop");
	}
}
// Verbatim exhaustive NCC of the original: binary64 integral images and a
// row-by-row binary64 sum of exact row dot products.
ScoreMap referenceNcc(ImageView src, const TemplateLevel& templ)
{
	const auto& t = templ.image;
	const int width = src.width - t.width() + 1, height = src.height - t.height() + 1;
	ScoreMap out{width, height, std::vector<double>(std::size_t(width) * height)};
	const double area = double(t.width()) * t.height();
	if (templ.norm * templ.norm / area < DBL_EPSILON) {
		std::fill(out.values.begin(), out.values.end(), 1.0);
		return out;
	}
	const std::size_t stride = std::size_t(src.width) + 1;
	std::vector<double> sums(stride * (std::size_t(src.height) + 1), 0);
	std::vector<double> squares(sums.size(), 0);
	for (int y = 0; y < src.height; ++y) {
		double sum = 0, square = 0;
		for (int x = 0; x < src.width; ++x) {
			const double v = src.data[std::size_t(y) * src.stride + x];
			sum += v;
			square += v * v;
			const auto i = (std::size_t(y) + 1) * stride + x + 1;
			sums[i] = sums[i - stride] + sum;
			squares[i] = squares[i - stride] + square;
		}
	}
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			const std::size_t a = std::size_t(y) * stride + x;
			const std::size_t b = a + t.width(), c = a + std::size_t(t.height()) * stride, d = c + t.width();
			const double sum = sums[d] - sums[b] - sums[c] + sums[a];
			const double square = squares[d] - squares[b] - squares[c] + squares[a];
			const double variance = std::max(0.0, square - sum * sum / area);
			const double denominator = nccDenominator(variance, square, templ.norm);
			if (denominator == 0) { out.at(x, y) = 0; continue; }
			double product = 0;
			for (int ty = 0; ty < t.height(); ++ty) {
				std::uint64_t row = 0;
				const auto* s = src.data + std::size_t(y + ty) * src.stride + x;
				for (int tx = 0; tx < t.width(); ++tx) row += unsigned(t.row(ty)[tx]) * s[tx];
				product += double(row);
			}
			out.at(x, y) = nccNormalizeProduct(product, sum, templ.mean, denominator);
		}
	return out;
}
void exactExhaustiveNcc()
{
	std::mt19937 rng(91);
	auto check = [](ImageView src, const TemplateLevel& templ, const char* message) {
		const auto expected = referenceNcc(src, templ);
		for (Backend backend : {Backend::Scalar, Backend::Auto}) {
			const auto actual = ncc(src, templ, backend);
			require(actual.width == expected.width && actual.height == expected.height &&
				std::memcmp(actual.values.data(), expected.values.data(), expected.values.size() * sizeof(double)) == 0, message);
		}
	};
	for (int trial = 0; trial < 1500; ++trial) {
		// Narrow, block-aligned, multi-block and odd widths; padded and tight
		// strides so windows near the end use the row-by-row fallback.
		const int tw = trial < 1200 ? 1 + int(rng() % 40) : 40 + int(rng() % 120);
		const int th = trial < 1200 ? 1 + int(rng() % 12) : 1 + int(rng() % 75);
		const int w = tw + int(rng() % 24), h = th + int(rng() % 10), pad = rng() % 3 ? 0 : int(rng() % 20);
		std::vector<std::uint8_t> pixels(std::size_t(w + pad) * h);
		const int mode = int(rng() % 4);
		for (auto& v : pixels) v = mode == 0 ? std::uint8_t(250 + rng() % 6) : mode == 1 ? std::uint8_t(rng() % 4) : std::uint8_t(rng());
		Image patch(tw, th);
		for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x)
			patch.row(y)[x] = mode == 0 ? std::uint8_t(250 + rng() % 6) : std::uint8_t(rng());
		check({pixels.data(), w, h, std::size_t(w + pad)}, learnLevel(patch), "exhaustive NCC must match the original binary64 path");
	}
	// Near-saturated wide rows: 4096 bytes uses the grouped kernel, 4097 does
	// not. 4096x20 products exceed one uint32 sum and 4096x160 exceed one uint32
	// accumulator lane of every grouped kernel, so the flushes must work.
	for (int th : {20, 160})
		for (int tw : {4096, 4097}) {
			Image patch(tw, th);
			for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x) patch.row(y)[x] = std::uint8_t(x == y ? 0 : 255);
			Image source(tw + 3, th + 4);
			for (int y = 0; y < th + 4; ++y) for (int x = 0; x < tw + 3; ++x) source.row(y)[x] = std::uint8_t((x + 3 * y) % 11 ? 255 : 254);
			check(source.view(), learnLevel(patch), "saturated wide NCC must match the original binary64 path");
		}
}
void peakExtraction()
{
	ScoreMap map{100,30,std::vector<double>(3000,0)};
	for(int y=0;y<20;++y) for(int x=0;x<20;++x) map.at(x,y)=.99;
	map.at(90,25)=.90;
	for (bool block : {false,true}) {
		const auto peaks=selectPeaks(map,25,25,2,0,.8,block);
		require(peaks.size()==2,"broad first peak must not hide second target");
		require(peaks[1].x==90 && peaks[1].y==25,"second target location");
	}
	std::mt19937 rng(100);
	for (auto& v : map.values) v=double(rng()%100)/100;
	const auto scan=selectPeaks(map,13,7,60,.63,.3,false);
	const auto block=selectPeaks(map,13,7,60,.63,.3,true);
	require(scan.size()==block.size(),"BlockMax count");
	for(size_t i=0;i<scan.size();++i) require(scan[i].x==block[i].x && scan[i].y==block[i].y && scan[i].score==block[i].score,"BlockMax exact tie order");
	ScoreMap tiny{2,1,{1,.9}};
	const auto peaks=selectPeaks(tiny,1,1,5,1,0,true);
	require(peaks.size()==2,"overlap=1 must always consume a peak");
}
void geometry()
{
	const auto a=makeMatch({0,0},10,10,0,.99), b=makeMatch({4,0},10,10,0,.95);
	near(overlap(a,b),.6);
	require(suppress({a,b},5,.5).size()==1,"intersection/template-area NMS");
	require(suppress({a,b},5,.7).size()==2,"allowed overlap");
	require(suppress({a,a},5,1).size()==1,"full duplicates");
	near(overlap(a,makeMatch({10,0},10,10,0,1)),0);
	near(overlap(a,makeMatch({10,0},10,10,90,1)),1);
	const double root=std::sqrt(2.0);
	near(overlap(a,makeMatch({5,5-5*root},10,10,45,1)),2*root-2);
	const auto p=rotate({12,35},{8,9},-64);
	const auto original=rotate(p,{8,9},64);
	near(original.x,12); near(original.y,35);
	near(normalizeAngle(-180),180);
}
void subpixel()
{
	std::array<double,27> cube{}; int i=0;
	for(int t=-1;t<=1;++t) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x)
		cube[i++]=1-.2*(x-.2)*(x-.2)-.3*(y+.3)*(y+.3)-.1*(t-.4)*(t-.4);
	std::array<double,3> offset{};
	require(refine3d(cube,offset),"3D fit");
	near(offset[0],.2); near(offset[1],-.3); near(offset[2],.4);
	std::array<double,9> patch{}; i=0;
	for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) patch[i++]=1-.2*(x-.25)*(x-.25)-.3*(y+.4)*(y+.4);
	Point p;
	require(refine2d(patch,p),"2D fit"); near(p.x,.25); near(p.y,-.4);
	cube.fill(1); require(!refine3d(cube,offset),"flat fit rejected");
	i=0;
	for(int t=-1;t<=1;++t) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) cube[i++]=x*x-y*y-t*t;
	require(!refine3d(cube,offset),"saddle rejected");
	cube[0]=std::numeric_limits<double>::quiet_NaN();
	require(!refine3d(cube,offset),"nonfinite fit rejected");
}
void matching()
{
	Image source=noise(161,123,1), templ=noise(31,23,2);
	paste(source,templ,20,30); paste(source,templ,100,80);
	Pattern pattern(templ.view(),64);
	MatchConfig config; config.minScore=.99; config.maxTargets=5;
	const auto matches=fpm::match(source.view(),pattern,config);
	require(matches.size()==2,"two planted targets");
	near(matches[0].center.x,35.5); near(matches[0].center.y,41.5);
	near(matches[1].center.x,115.5); near(matches[1].center.y,91.5);
	for (const auto& m:matches) { near(m.angle,0); near(m.score,1); }
	config.stopLayer=1; config.minScore=.5; config.maxTargets=2;
	const auto fast=fpm::match(source.view(),pattern,config);
	require(fast.size()==2,"FastMode two targets");
	for(const auto& m:fast) {
		require(std::min(std::hypot(m.center.x-35.5,m.center.y-41.5),std::hypot(m.center.x-115.5,m.center.y-91.5))<=2,"FastMode full-size output coordinates");
		near(std::hypot(m.corners[1].x-m.corners[0].x,m.corners[1].y-m.corners[0].y),31);
	}
	config.stopLayer=2;
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.stopLayer=0; config.minScore=.99; config.maxTargets=5;
	for(int y=0;y<source.height();++y) for(int x=0;x<source.width();++x) source.row(y)[x]=255-source.row(y)[x];
	config.invertSource=true;
	require(fpm::match(source.view(),pattern,config).size()==2,"source inversion");
	config.minScore=std::numeric_limits<double>::quiet_NaN();
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.minScore=.99; config.maxTargets=0;
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.maxTargets=1; config.angleRanges={{20,10}};
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.angleRanges.clear();
	require(fpm::match(Image(2,2).view(),pattern,config).empty(),"oversized template");
}
void rotatedMatching()
{
	Image templ=noise(23,17,30), source=noise(100,90,31);
	for(int y=0;y<17;++y) for(int x=0;x<23;++x) source.row(30+22-x)[40+y]=templ.row(y)[x];
	Pattern pattern(templ.view(),256);
	MatchConfig c; c.minScore=.95; c.maxTargets=1; c.angleRanges={{90,90},{-30,-20}};
	const auto matches=fpm::match(source.view(),pattern,c);
	require(matches.size()==1,"two angle ranges find +90 object");
	near(matches[0].angle,90); near(matches[0].center.x,48.5); near(matches[0].center.y,40.5);
	c.angleRanges={{-90,-90}};
	require(fpm::match(source.view(),pattern,c).empty(),"excluded angle");
	// No-pyramid path and subpixel at fixed angle must also be safe.
	Pattern flat(templ.view(),10000); c.angleRanges={{90,90}}; c.subpixel=true;
	const auto fine=fpm::match(source.view(),flat,c);
	require(fine.size()==1,"layer-zero refinement");
	near(fine[0].center.x,48.5,.2); near(fine[0].center.y,40.5,.2);
}

void sixRotatedCopies()
{
	// Independently render six actual copies. This catches angle-sign, pyramid
	// propagation and candidate loss without assuming every rotor arm is equal.
	auto intensity=[](double x,double y) {
		return 20 + 150*std::exp(-((x-16)*(x-16)/50+(y-13)*(y-13)/24)) +
			110*std::exp(-((x-34)*(x-34)/22+(y-24)*(y-24)/40)) +
			75*std::exp(-((x-14)*(x-14)/16+(y-30)*(y-30)/12));
	};
	Image templ(51,39), source(380,260);
	for (int y=0;y<39;++y) for (int x=0;x<51;++x)
		templ.row(y)[x]=std::uint8_t(std::lround(intensity(x,y)));
	const double angles[]={0,60,120,180,-120,-60};
	const Point centers[]={{70.3,70.4},{180.3,70.4},{290.3,70.4},{70.3,180.4},{180.3,180.4},{290.3,180.4}};
	for (int y=0;y<source.height();++y) for (int x=0;x<source.width();++x) {
		double value=20;
		for (int i=0;i<6;++i) {
			const double a=angles[i]*3.14159265358979323846/180;
			const double dx=x-centers[i].x, dy=y-centers[i].y;
			// Inverse of the image-space CCW pose; no production rotate/warp call.
			const double tx=std::cos(a)*dx-std::sin(a)*dy+25.5;
			const double ty=std::sin(a)*dx+std::cos(a)*dy+19.5;
			value+=intensity(tx,ty)-20;
		}
		source.row(y)[x]=std::uint8_t(std::clamp(std::lround(value),0L,255L));
	}
	for (int area : {64,256,1024}) for (bool subpixel : {false,true}) {
		Pattern pattern(templ.view(),area);
		MatchConfig config; config.maxTargets=10; config.minScore=.95; config.toleranceAngle=180;
		config.subpixel=subpixel;
		const auto matches=fpm::match(source.view(),pattern,config);
		require(matches.size()==6,"six true rotated copies must survive coarse-to-fine search at score .95");
		bool seen[6]{};
		double maxPositionError=0, maxAngleError=0, minScore=1;
		for (const auto& m : matches) {
			int found=-1;
			// Integer angular sampling at this template width is atan(2/51)
			// = 2.246 degrees. This test verifies recall/distinct poses at that
			// resolution; a quadratic fit is not an accuracy guarantee for every
			// texture. The separate fractionalPose test checks known fit accuracy.
			const bool precise=subpixel && area>=256;
			const double angleTolerance=precise ? .1 : std::atan(2./51)*180/3.14159265358979323846;
			for (int i=0;i<6;++i) if (!seen[i] && std::hypot(m.center.x-centers[i].x,m.center.y-centers[i].y)<(precise?.02:1.5) &&
				std::abs(std::remainder(m.angle-angles[i],360.0))<angleTolerance) { found=i; break; }
			if (found<0) std::cerr<<"six-copy rejected area="<<area<<" subpixel="<<subpixel<<" x="<<m.center.x<<" y="<<m.center.y<<" angle="<<m.angle<<" score="<<m.score<<'\n';
			require(found>=0,"unexpected six-copy pose or duplicate"); seen[found]=true;
			maxPositionError=std::max(maxPositionError,std::hypot(m.center.x-centers[found].x,m.center.y-centers[found].y));
			maxAngleError=std::max(maxAngleError,std::abs(std::remainder(m.angle-angles[found],360.0)));
			minScore=std::min(minScore,m.score);
		}
		std::cout<<"six-copy area "<<area<<", subpixel "<<subpixel<<": max position error "<<maxPositionError
			<<", max angle error "<<maxAngleError<<", minimum score "<<minScore<<'\n';
	}
}

void fractionalPose()
{
	// Independently render a continuous asymmetric texture at a known pose.
	// This checks end-to-end coordinate/angle precision, not just the fitter.
	auto intensity=[](double x,double y) {
		return std::uint8_t(std::clamp(std::lround(20 +
			150*std::exp(-((x-16)*(x-16)/50+(y-13)*(y-13)/24)) +
			110*std::exp(-((x-34)*(x-34)/22+(y-24)*(y-24)/40)) +
			75*std::exp(-((x-14)*(x-14)/16+(y-30)*(y-30)/12))),0L,255L));
	};
	Image templ(51,39);
	for(int y=0;y<39;++y) for(int x=0;x<51;++x) templ.row(y)[x]=intensity(x,y);
	for(double angle : {0.0,21.3}) {
		const Point origin{50.3,70.4};
		Image source(150,130);
		for(int y=0;y<130;++y) for(int x=0;x<150;++x) {
			const auto p=rotate({x-origin.x,y-origin.y},{0,0},-angle);
			source.row(y)[x]=intensity(p.x,p.y);
		}
		Pattern pattern(templ.view(),256);
		MatchConfig c; c.maxTargets=1; c.minScore=.9; c.toleranceAngle=angle==0?0:40;
		const auto integer=fpm::match(source.view(),pattern,c);
		c.subpixel=true;
		const auto fractional=fpm::match(source.view(),pattern,c);
		require(integer.size()==1 && fractional.size()==1,"fractional pose detected");
		const auto truth=makeMatch(origin,51,39,-angle,1);
		auto error=[&](const Match& m) {return std::hypot(m.center.x-truth.center.x,m.center.y-truth.center.y);};
		std::cout << "fractional pose " << angle << ": integer error " << error(integer[0]) << ", refined error " << error(fractional[0]) << ", angle " << fractional[0].angle << '\n';
		require(error(fractional[0])<error(integer[0]),"subpixel improves known position");
		require(error(fractional[0])<.25,"subpixel position within a quarter pixel");
		near(fractional[0].angle,angle,.5);
	}
}
}
int main()
{
	const std::pair<const char*,std::function<void()>> tests[]={{"image/pyramid",imageAndPyramid},{"NCC/reference/stride",nccReference},{"NCC/conservative-pruning",nccPruning},{"SIMD/overflow/tails",simdOverflow},{"rotation/pyramid/exact-reference",exactRotationAndPyramid},{"NCC/exhaustive/exact-reference",exactExhaustiveNcc},{"peaks/BlockMax",peakExtraction},{"geometry/NMS",geometry},{"match/six-rotated-copies",sixRotatedCopies},{"subpixel/fits",subpixel},{"match/validation/invert",matching},{"rotation/ranges/layer0",rotatedMatching},{"subpixel/known-fractional-pose",fractionalPose}};
	int failed=0;
	for(const auto& test:tests) {
		try { test.second(); std::cout<<"PASS "<<test.first<<'\n'; }
		catch(const std::exception& e) { ++failed; std::cerr<<"FAIL "<<test.first<<": "<<e.what()<<'\n'; }
	}
	return failed ? 1 : 0;
}
