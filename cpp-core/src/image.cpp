// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <stdexcept>

// The NEON rotation path mirrors each multiply-add that clang's default
// -ffp-contract=on fuses in warpPixel's expressions; core_test compares it
// with a copy of the original loop. Other compilers keep the scalar loop.
#if !defined(FPM_DISABLE_SIMD) && defined(__clang__) && defined(__aarch64__) && \
	(defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define FPM_NEON_WARP 1
#endif

namespace fpm {
namespace detail {
void validateImage(ImageView image)
{
	if (!image.data || image.width <= 0 || image.height <= 0 ||
		image.stride < std::size_t(image.width) ||
		image.stride > (std::size_t(PTRDIFF_MAX) - image.width) / std::size_t(image.height))
		throw std::invalid_argument("expected a nonempty gray8 image with a valid positive stride");
}
}

Image::Image(int width, int height) : width_(width), height_(height)
{
	if (width <= 0 || height <= 0 ||
		std::size_t(width) > std::size_t(PTRDIFF_MAX) / std::size_t(height))
		throw std::invalid_argument("invalid image dimensions");
	pixels_.resize(std::size_t(width) * height);
}

Image::Image(ImageView source) : Image(source.width, source.height)
{
	detail::validateImage(source);
	for (int y = 0; y < height_; ++y)
		std::copy_n(source.data + std::size_t(y) * source.stride, width_, row(y));
}

Pattern::Pattern(ImageView image, int minReducedArea)
{
	detail::validateImage(image);
	if (minReducedArea < 1)
		throw std::invalid_argument("minReducedArea must be positive");
	levels_.push_back(detail::learnLevel(Image(image)));
	borderColor_ = levels_.front().mean < 128 ? 255 : 0;
	// Preserve GetTopLayer's area/4 rule and squared floor(sqrt(area)).
	const auto side = std::uint64_t(std::sqrt(double(minReducedArea)));
	std::uint64_t area = std::uint64_t(image.width) * image.height;
	while (area > side * side) {
		const auto& previous = levels_.back().image;
		if (previous.width() == 1 && previous.height() == 1) break;
		levels_.push_back(detail::learnLevel(detail::pyrDown(previous.view())));
		area /= 4;
	}
}

namespace detail {
static int reflect101(int p, int length)
{
	if (length == 1) return 0;
	while (p < 0 || p >= length)
		p = p < 0 ? -p : 2 * length - p - 2;
	return p;
}

Image pyrDown(ImageView image)
{
	validateImage(image);
	Image result(image.width / 2 + image.width % 2, image.height / 2 + image.height % 2);
	const int width = result.width();
	constexpr int kernel[] = {1, 4, 6, 4, 1};
	// Separable [1,4,6,4,1]/16 filter, REFLECT_101, rounded once after both axes.
	// Integer sums are exact: columns whose taps 2x-2..2x+2 lie inside the row
	// skip reflect101, and rows are reflected once per output row.
	std::vector<int> horizontal(std::size_t(image.height) * width);
	const int interior = std::max(1, std::min(width, (image.width - 1) / 2));
	for (int y = 0; y < image.height; ++y) {
		const auto* row = image.data + std::size_t(y) * image.stride;
		auto* out = horizontal.data() + std::size_t(y) * width;
		auto reflected = [&](int x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * row[reflect101(x * 2 + k, image.width)];
			return sum;
		};
		out[0] = reflected(0);
		for (int x = 1; x < interior; ++x) {
			const auto* p = row + 2 * x;
			out[x] = p[-2] + 4 * p[-1] + 6 * p[0] + 4 * p[1] + p[2];
		}
		for (int x = interior; x < width; ++x) out[x] = reflected(x);
	}
	for (int y = 0; y < result.height(); ++y) {
		const int* rows[5];
		for (int k = 0; k < 5; ++k)
			rows[k] = horizontal.data() + std::size_t(reflect101(y * 2 + k - 2, image.height)) * width;
		auto* out = result.row(y);
		for (int x = 0; x < width; ++x)
			out[x] = std::uint8_t((rows[0][x] + 4 * rows[1][x] + 6 * rows[2][x] + 4 * rows[3][x] + rows[4][x] + 128) / 256);
	}
	return result;
}

TemplateLevel learnLevel(Image image)
{
	double sum = 0;
	const double area = double(image.width()) * image.height();
	for (int y = 0; y < image.height(); ++y)
		for (int x = 0; x < image.width(); ++x) sum += image.row(y)[x];
	const double mean = sum / area;
	double varianceSum = 0;
	for (int y = 0; y < image.height(); ++y)
		for (int x = 0; x < image.width(); ++x) {
			const double delta = image.row(y)[x] - mean;
			varianceSum += delta * delta;
		}
	return {std::move(image), mean, std::sqrt(varianceSum)};
}

Point rotate(Point p, Point center, double angle)
{
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	const double x = p.x - center.x, y = p.y - center.y;
	return {center.x + c * x + s * y, center.y - s * x + c * y};
}

double normalizeAngle(double angle)
{
	angle = std::remainder(angle, 360.0);
	return angle <= -180 ? angle + 360 : angle;
}

static std::uint8_t pixel(ImageView src, int x, int y, std::uint8_t border)
{
	return x < 0 || y < 0 || x >= src.width || y >= src.height ? border :
		src.data[std::size_t(y) * src.stride + x];
}

// Inverse affine sampling: output (x,y) maps to anchor + R(-angle)*(x,y).
// Bilinear interpolation uses full double precision, rounded to uint8. Unlike
// OpenCV INTER_LINEAR, coordinates are not quantized to a 1/32-pixel table.
static std::uint8_t warpPixel(ImageView src, int x, int y, Point anchor, double c, double s, std::uint8_t border)
{
	const double sx = anchor.x + c * x - s * y;
	const double sy = anchor.y + s * x + c * y;
	if (sx < -1 || sy < -1 || sx >= src.width || sy >= src.height) return border;
	const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
	const double dx = sx - ix, dy = sy - iy;
	const double top = pixel(src, ix, iy, border) * (1 - dx) + pixel(src, ix + 1, iy, border) * dx;
	const double bottom = pixel(src, ix, iy + 1, border) * (1 - dx) + pixel(src, ix + 1, iy + 1, border) * dx;
	return std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
}

#if FPM_NEON_WARP
static std::uint64_t bytePair(const std::uint8_t* p)
{
	std::uint16_t value;
	std::memcpy(&value, p, 2);
	return value;
}

// Two pixels per step with warpPixel's operations: sx=(ax+c*x)-s*y and
// sy=(ay+s*x)+c*y fused twice, top=p00*(1-dx)+round(p10*dx) fused, and so on.
// A pair with any pixel outside 0<=sx<w-1, 0<=sy<h-1 uses warpPixel.
static int warpRowNeon(ImageView src, std::uint8_t* out, int width, int y, Point anchor,
	double c, double s, std::uint8_t border)
{
	const float64x2_t vc = vdupq_n_f64(c), vs = vdupq_n_f64(s), vy = vdupq_n_f64(y);
	const float64x2_t ax = vdupq_n_f64(anchor.x), ay = vdupq_n_f64(anchor.y);
	const float64x2_t one = vdupq_n_f64(1), zero = vdupq_n_f64(0), maximum = vdupq_n_f64(255);
	const float64x2_t innerW = vdupq_n_f64(src.width - 1), innerH = vdupq_n_f64(src.height - 1);
	const float64x2_t stride = vdupq_n_f64(double(src.stride)), two = vdupq_n_f64(2);
	const uint64x2_t low = vdupq_n_u64(255);
	const double first[2] = {0, 1};
	float64x2_t vx = vld1q_f64(first);
	int x = 0;
	for (; x + 2 <= width; x += 2, vx = vaddq_f64(vx, two)) {
		const float64x2_t sx = vfmsq_f64(vfmaq_f64(ax, vc, vx), vs, vy);
		const float64x2_t sy = vfmaq_f64(vfmaq_f64(ay, vs, vx), vc, vy);
		const uint64x2_t inside = vandq_u64(vandq_u64(vcgeq_f64(sx, zero), vcltq_f64(sx, innerW)),
			vandq_u64(vcgeq_f64(sy, zero), vcltq_f64(sy, innerH)));
		if (!(vgetq_lane_u64(inside, 0) & vgetq_lane_u64(inside, 1))) {
			out[x] = warpPixel(src, x, y, anchor, c, s, border);
			out[x + 1] = warpPixel(src, x + 1, y, anchor, c, s, border);
			continue;
		}
		const float64x2_t fx = vrndmq_f64(sx), fy = vrndmq_f64(sy);
		const float64x2_t dx = vsubq_f64(sx, fx), dy = vsubq_f64(sy, fy);
		// Integer-valued and far below 2^53, so the offsets are exact.
		const int64x2_t offset = vcvtq_s64_f64(vfmaq_f64(fx, fy, stride));
		const auto* p = src.data + vgetq_lane_s64(offset, 0);
		const auto* q = src.data + vgetq_lane_s64(offset, 1);
		const uint64x2_t upper = vcombine_u64(vcreate_u64(bytePair(p)), vcreate_u64(bytePair(q)));
		const uint64x2_t lower = vcombine_u64(vcreate_u64(bytePair(p + src.stride)), vcreate_u64(bytePair(q + src.stride)));
		const float64x2_t p00 = vcvtq_f64_u64(vandq_u64(upper, low)), p10 = vcvtq_f64_u64(vshrq_n_u64(upper, 8));
		const float64x2_t p01 = vcvtq_f64_u64(vandq_u64(lower, low)), p11 = vcvtq_f64_u64(vshrq_n_u64(lower, 8));
		const float64x2_t rx = vsubq_f64(one, dx), ry = vsubq_f64(one, dy);
		const float64x2_t top = vfmaq_f64(vmulq_f64(p10, dx), p00, rx);
		const float64x2_t bottom = vfmaq_f64(vmulq_f64(p11, dx), p01, rx);
		const float64x2_t value = vaddq_f64(vfmaq_f64(vmulq_f64(bottom, dy), top, ry), vdupq_n_f64(0.5));
		const uint64x2_t result = vcvtq_u64_f64(vminq_f64(vmaxq_f64(vrndmq_f64(value), zero), maximum));
		out[x] = std::uint8_t(vgetq_lane_u64(result, 0));
		out[x + 1] = std::uint8_t(vgetq_lane_u64(result, 1));
	}
	return x;
}
#endif

static Image warp(ImageView src, int width, int height, Point anchor, double angle, std::uint8_t border)
{
	Image result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	for (int y = 0; y < height; ++y) {
		auto* out = result.row(y);
		int x = 0;
#if FPM_NEON_WARP
		x = warpRowNeon(src, out, width, y, anchor, c, s, border);
#endif
		for (; x < width; ++x) out[x] = warpPixel(src, x, y, anchor, c, s, border);
	}
	return result;
}

Canvas rotateCanvas(ImageView src, int tw, int th, double angle, std::uint8_t border)
{
	const Point center{(src.width - 1) * 0.5, (src.height - 1) * 0.5};
	const double c = std::abs(std::cos(angle * D2R)), s = std::abs(std::sin(angle * D2R));
	// Retain the original reduced rotation canvas, with a full-box fallback.
	double w = 2 * std::ceil(((src.width - 1) * c + (src.height - 1) * s) * 0.5 - th * s * c);
	double h = 2 * std::ceil(((src.width - 1) * s + (src.height - 1) * c) * 0.5 - tw * s * c);
	if (s < 1e-12) { w = src.width; h = src.height; }
	else if (c < 1e-12) { w = src.height; h = src.width; }
	if (w < tw || h < th) {
		w = std::ceil(src.width * c + src.height * s);
		h = std::ceil(src.width * s + src.height * c);
	}
	if (w > INT_MAX - 1 || h > INT_MAX - 1) throw std::invalid_argument("rotation canvas is too large");
	const Point translation{(w - 1) * 0.5 - center.x, (h - 1) * 0.5 - center.y};
	const Point anchor = rotate({-translation.x, -translation.y}, center, -angle);
	return {warp(src, int(w), int(h), anchor, angle, border), translation};
}

Image rotatedRoi(ImageView src, int width, int height, Point anchor, double angle)
{
	const Point origin = rotate({-PADDING, -PADDING}, {0, 0}, -angle);
	return warp(src, width + 2 * PADDING, height + 2 * PADDING,
		{anchor.x + origin.x, anchor.y + origin.y}, angle, 0);
}
} // namespace detail
} // namespace fpm
