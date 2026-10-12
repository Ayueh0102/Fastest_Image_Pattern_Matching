// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fpm {

// Shared 8-bit RGB conversion for file-loading adapters. Equal channels are
// preserved exactly; alpha is ignored. The matching API itself takes gray8.
inline std::uint8_t grayFromRgb(std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
	return std::uint8_t((77u * r + 150u * g + 29u * b + 128u) >> 8);
}

// Single-channel unsigned 8-bit pixels; stride is in bytes and may exceed width.
// The caller owns the buffer for the duration of the call. Negative strides
// are not supported. Public entry points throw std::invalid_argument on misuse.
struct ImageView {
	const std::uint8_t* data = nullptr;
	int width = 0;
	int height = 0;
	std::size_t stride = 0;
};

class Image {
public:
	Image() = default;
	Image(int width, int height);
	explicit Image(ImageView source);
	int width() const { return width_; }
	int height() const { return height_; }
	std::uint8_t* row(int y) { return pixels_.data() + std::size_t(y) * width_; }
	const std::uint8_t* row(int y) const { return pixels_.data() + std::size_t(y) * width_; }
	ImageView view() const { return {pixels_.data(), width_, height_, std::size_t(width_)}; }
private:
	int width_ = 0;
	int height_ = 0;
	std::vector<std::uint8_t> pixels_;
};

struct Point { double x = 0; double y = 0; };
struct AngleRange { double min = 0; double max = 0; };
enum class Backend { Scalar, Auto }; // Auto uses NEON or SSE2 when compiled in.

struct MatchConfig {
	int maxTargets = 70;
	double minScore = 0.5;
	double maxOverlap = 0; // Intersection area / template area, NOT IoU.
	double toleranceAngle = 0; // [0,180] degrees; ignored when angleRanges is nonempty.
	// Output angles (CCW in x-right/y-down images), each in [-180,180].
	// Split a range crossing +/-180 into two ranges. Endpoints are inclusive.
	std::vector<AngleRange> angleRanges;
	bool subpixel = false;
	bool invertSource = false;
	int stopLayer = 0; // 0 = full precision, 1 = stop at half resolution (legacy FastMode).
	Backend backend = Backend::Auto;
	// Opt-in exact threshold bounds at the coarse layer. Fine score maps and
	// accepted coarse scores are unchanged. Small/unprofitable cases bypass it.
	bool coarseNccPruning = false;
};

struct Match {
	std::array<Point, 4> corners{}; // LT, RT, RB, LB; extents use width/height.
	Point center;
	double angle = 0; // Output template orientation in (-180,180], degrees CCW.
	double score = 0; // NCC at the sampled integer maximum, before subpixel fitting.
};

struct TemplateLevel {
	Image image;
	double mean = 0;
	double norm = 0; // sqrt(sum((T - mean)^2))
};

// Immutable, owning learned template. Reusable across frames and threads.
// Pyramid depth belongs to the pattern, so match cannot accidentally use a
// different depth. Learning copies the pixels, including level zero.
class Pattern {
public:
	explicit Pattern(ImageView image, int minReducedArea = 256);
	const std::vector<TemplateLevel>& levels() const { return levels_; }
	std::uint8_t borderColor() const { return borderColor_; }
private:
	std::vector<TemplateLevel> levels_;
	std::uint8_t borderColor_ = 0;
};

// Results are sorted by descending score, with deterministic tie breaking.
// A template that does not fit the unrotated source returns an empty result,
// matching the legacy project's input-size contract. No scale search is done.
std::vector<Match> match(ImageView source, const Pattern& pattern,
	const MatchConfig& config = {});
const char* compiledBackend();

} // namespace fpm
