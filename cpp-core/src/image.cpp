// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

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
	constexpr int kernel[] = {1, 4, 6, 4, 1};
	// Separable [1,4,6,4,1]/16 filter, REFLECT_101, rounded once after both axes.
	std::vector<int> horizontal(std::size_t(image.height) * result.width());
	for (int y = 0; y < image.height; ++y) {
		const auto* row = image.data + std::size_t(y) * image.stride;
		for (int x = 0; x < result.width(); ++x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * row[reflect101(x * 2 + k, image.width)];
			horizontal[std::size_t(y) * result.width() + x] = sum;
		}
	}
	for (int y = 0; y < result.height(); ++y)
		for (int x = 0; x < result.width(); ++x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * horizontal[std::size_t(reflect101(y * 2 + k, image.height)) * result.width() + x];
			result.row(y)[x] = std::uint8_t((sum + 128) / 256);
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
static Image warp(ImageView src, int width, int height, Point anchor, double angle, std::uint8_t border)
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
			const double top = pixel(src, ix, iy, border) * (1 - dx) + pixel(src, ix + 1, iy, border) * dx;
			const double bottom = pixel(src, ix, iy + 1, border) * (1 - dx) + pixel(src, ix + 1, iy + 1, border) * dx;
			result.row(y)[x] = std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
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
