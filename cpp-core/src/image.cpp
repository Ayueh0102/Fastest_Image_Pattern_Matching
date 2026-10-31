// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <stdexcept>

// The NEON rotation path mirrors each multiply-add that clang's default
// -ffp-contract=on fuses in warpPixel's expressions; core_test compares it
// with a copy of the original loop. Other ARM compilers keep the scalar loop.
#if !defined(FPM_DISABLE_SIMD) && defined(__clang__) && defined(__aarch64__) && \
	(defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define FPM_NEON_WARP 1
// On x86-64 without FMA nothing is contracted, so the SSE2 and AVX2 rows round
// every multiply and add separately in warpPixel's order. Builds where the
// compiler may fuse (__FMA__, MSVC /fp:contract or /fp:fast) would fail the same
// core_test; GCC and clang -mfma builds keep the scalar loop.
#elif FPM_X86 && !defined(__FMA__)
#include <immintrin.h>
#define FPM_X86_WARP 1
#endif
#if FPM_X86
#include <emmintrin.h>
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
#if FPM_X86
// pyrDown with SSE2 integer sums; the same integers as the scalar loops. Eight
// outputs x..x+7 need bytes 2x-2..2x+16; reads stop at 2x+29 < source width.
static int pyrDownRowSse2(const std::uint8_t* row, int* out, int x, int interior, int sourceWidth)
{
	const __m128i lowByte = _mm_set1_epi16(0x00ff), zero = _mm_setzero_si128();
	for (; x + 8 <= interior && 2 * x + 30 <= sourceWidth; x += 8) {
		// e[j] = row[2(x+j-1)], o[j] = row[2(x+j-1)+1]; sums stay <= 16*255.
		const auto* p = row + 2 * x - 2;
		const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
		const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
		const __m128i e0 = _mm_and_si128(a, lowByte), o0 = _mm_srli_epi16(a, 8);
		const __m128i e8 = _mm_and_si128(b, lowByte), o8 = _mm_srli_epi16(b, 8);
		const __m128i e1 = _mm_or_si128(_mm_srli_si128(e0, 2), _mm_slli_si128(e8, 14));
		const __m128i e2 = _mm_or_si128(_mm_srli_si128(e0, 4), _mm_slli_si128(e8, 12));
		const __m128i o1 = _mm_or_si128(_mm_srli_si128(o0, 2), _mm_slli_si128(o8, 14));
		// out[x+k] = e[k] + 4*o[k] + 6*e[k+1] + 4*o[k+1] + e[k+2]
		const __m128i sum = _mm_add_epi16(_mm_add_epi16(e0, e2), _mm_add_epi16(_mm_slli_epi16(_mm_add_epi16(o0, o1), 2),
			_mm_add_epi16(_mm_slli_epi16(e1, 2), _mm_slli_epi16(e1, 1))));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), _mm_unpacklo_epi16(sum, zero));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out + x + 4), _mm_unpackhi_epi16(sum, zero));
	}
	return x;
}

// Vertical taps; the nonnegative (sum + 128) / 256 is a shift and stays <= 255.
static int pyrDownColumnsSse2(const int* const* rows, std::uint8_t* out, int width)
{
	const __m128i bias = _mm_set1_epi32(128);
	int x = 0;
	for (; x + 8 <= width; x += 8) {
		__m128i halves[2];
		for (int h = 0; h < 2; ++h) {
			const auto load = [&](int k) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[k] + x + 4 * h)); };
			const __m128i center = load(2);
			const __m128i sum = _mm_add_epi32(_mm_add_epi32(load(0), load(4)), _mm_add_epi32(
				_mm_slli_epi32(_mm_add_epi32(load(1), load(3)), 2), _mm_add_epi32(_mm_slli_epi32(center, 2), _mm_slli_epi32(center, 1))));
			halves[h] = _mm_srli_epi32(_mm_add_epi32(sum, bias), 8);
		}
		const __m128i words = _mm_packs_epi32(halves[0], halves[1]);
		_mm_storel_epi64(reinterpret_cast<__m128i*>(out + x), _mm_packus_epi16(words, words));
	}
	return x;
}
#endif

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
		int x = 1;
#if FPM_X86
		x = pyrDownRowSse2(row, out, x, interior, image.width);
#endif
		for (; x < interior; ++x) {
			const auto* p = row + 2 * x;
			out[x] = p[-2] + 4 * p[-1] + 6 * p[0] + 4 * p[1] + p[2];
		}
		for (; x < width; ++x) out[x] = reflected(x);
	}
	for (int y = 0; y < result.height(); ++y) {
		const int* rows[5];
		for (int k = 0; k < 5; ++k)
			rows[k] = horizontal.data() + std::size_t(reflect101(y * 2 + k - 2, image.height)) * width;
		auto* out = result.row(y);
		int x = 0;
#if FPM_X86
		x = pyrDownColumnsSse2(rows, out, width);
#endif
		for (; x < width; ++x)
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

#if FPM_NEON_WARP || FPM_X86_WARP
static std::uint64_t bytePair(const std::uint8_t* p)
{
	std::uint16_t value;
	std::memcpy(&value, p, 2);
	return value;
}
#endif

#if FPM_NEON_WARP
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
#elif FPM_X86_WARP
// Two pixels per step. sx=(ax+c*x)-s*y, sy=(ay+s*x)+c*y, top=p00*(1-dx)+p10*dx
// and the final value are evaluated with warpPixel's roundings. Only pairs with
// both pixels inside 0<=sx<w-1, 0<=sy<h-1 come here, so floor is truncation and
// value>=0.5; the rest use warpPixel.
static int warpRowSse2(ImageView src, std::uint8_t* out, int width, int y, Point anchor,
	double c, double s, std::uint8_t border)
{
	const __m128d vc = _mm_set1_pd(c), vs = _mm_set1_pd(s), vy = _mm_set1_pd(y);
	const __m128d ax = _mm_set1_pd(anchor.x), ay = _mm_set1_pd(anchor.y);
	const __m128d one = _mm_set1_pd(1), zero = _mm_setzero_pd(), half = _mm_set1_pd(0.5), two = _mm_set1_pd(2);
	const __m128d innerW = _mm_set1_pd(src.width - 1), innerH = _mm_set1_pd(src.height - 1);
	const __m128d sy = _mm_mul_pd(vs, vy), cy = _mm_mul_pd(vc, vy);
	__m128d vx = _mm_set_pd(1, 0);
	int x = 0;
	for (; x + 2 <= width; x += 2, vx = _mm_add_pd(vx, two)) {
		const __m128d px = _mm_sub_pd(_mm_add_pd(ax, _mm_mul_pd(vc, vx)), sy);
		const __m128d py = _mm_add_pd(_mm_add_pd(ay, _mm_mul_pd(vs, vx)), cy);
		const __m128d inside = _mm_and_pd(_mm_and_pd(_mm_cmpge_pd(px, zero), _mm_cmplt_pd(px, innerW)),
			_mm_and_pd(_mm_cmpge_pd(py, zero), _mm_cmplt_pd(py, innerH)));
		if (_mm_movemask_pd(inside) != 3) {
			out[x] = warpPixel(src, x, y, anchor, c, s, border);
			out[x + 1] = warpPixel(src, x + 1, y, anchor, c, s, border);
			continue;
		}
		const __m128i ix = _mm_cvttpd_epi32(px), iy = _mm_cvttpd_epi32(py);
		const __m128d dx = _mm_sub_pd(px, _mm_cvtepi32_pd(ix)), dy = _mm_sub_pd(py, _mm_cvtepi32_pd(iy));
		int xs[4], ys[4];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(xs), ix);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(ys), iy);
		const auto* p = src.data + std::size_t(ys[0]) * src.stride + xs[0];
		const auto* q = src.data + std::size_t(ys[1]) * src.stride + xs[1];
		const auto pu = int(bytePair(p)), qu = int(bytePair(q));
		const auto pl = int(bytePair(p + src.stride)), ql = int(bytePair(q + src.stride));
		const __m128d p00 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, qu & 255, pu & 255));
		const __m128d p10 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, qu >> 8, pu >> 8));
		const __m128d p01 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, ql & 255, pl & 255));
		const __m128d p11 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, ql >> 8, pl >> 8));
		const __m128d rx = _mm_sub_pd(one, dx), ry = _mm_sub_pd(one, dy);
		const __m128d top = _mm_add_pd(_mm_mul_pd(p00, rx), _mm_mul_pd(p10, dx));
		const __m128d bottom = _mm_add_pd(_mm_mul_pd(p01, rx), _mm_mul_pd(p11, dx));
		const __m128d value = _mm_add_pd(_mm_add_pd(_mm_mul_pd(top, ry), _mm_mul_pd(bottom, dy)), half);
		int result[4];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(result), _mm_cvttpd_epi32(value));
		out[x] = std::uint8_t(std::min(result[0], 255));
		out[x + 1] = std::uint8_t(std::min(result[1], 255));
	}
	return x;
}

// The same with four pixels per step. Pixels are loaded as byte pairs; a gather
// was 8% faster on the 285K pilot but is slow on other x86 cores.
FPM_TARGET_AVX2 static int warpRowAvx2(ImageView src, std::uint8_t* out, int width, int y, Point anchor,
	double c, double s, std::uint8_t border)
{
	const __m256d vc = _mm256_set1_pd(c), vs = _mm256_set1_pd(s), vy = _mm256_set1_pd(y);
	const __m256d ax = _mm256_set1_pd(anchor.x), ay = _mm256_set1_pd(anchor.y);
	const __m256d one = _mm256_set1_pd(1), zero = _mm256_setzero_pd(), half = _mm256_set1_pd(0.5), four = _mm256_set1_pd(4);
	const __m256d innerW = _mm256_set1_pd(src.width - 1), innerH = _mm256_set1_pd(src.height - 1);
	const __m256d sy = _mm256_mul_pd(vs, vy), cy = _mm256_mul_pd(vc, vy);
	const __m128i low = _mm_set1_epi32(255);
	__m256d vx = _mm256_set_pd(3, 2, 1, 0);
	int x = 0;
	for (; x + 4 <= width; x += 4, vx = _mm256_add_pd(vx, four)) {
		const __m256d px = _mm256_sub_pd(_mm256_add_pd(ax, _mm256_mul_pd(vc, vx)), sy);
		const __m256d py = _mm256_add_pd(_mm256_add_pd(ay, _mm256_mul_pd(vs, vx)), cy);
		const __m256d inside = _mm256_and_pd(
			_mm256_and_pd(_mm256_cmp_pd(px, zero, _CMP_GE_OQ), _mm256_cmp_pd(px, innerW, _CMP_LT_OQ)),
			_mm256_and_pd(_mm256_cmp_pd(py, zero, _CMP_GE_OQ), _mm256_cmp_pd(py, innerH, _CMP_LT_OQ)));
		if (_mm256_movemask_pd(inside) != 15) {
			for (int k = 0; k < 4; ++k) out[x + k] = warpPixel(src, x + k, y, anchor, c, s, border);
			continue;
		}
		const __m128i ix = _mm256_cvttpd_epi32(px), iy = _mm256_cvttpd_epi32(py);
		const __m256d dx = _mm256_sub_pd(px, _mm256_cvtepi32_pd(ix)), dy = _mm256_sub_pd(py, _mm256_cvtepi32_pd(iy));
		int xs[4], ys[4], upperPairs[4], lowerPairs[4];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(xs), ix);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(ys), iy);
		for (int k = 0; k < 4; ++k) {
			const auto* p = src.data + std::size_t(ys[k]) * src.stride + xs[k];
			upperPairs[k] = int(bytePair(p));
			lowerPairs[k] = int(bytePair(p + src.stride));
		}
		const __m128i upper = _mm_loadu_si128(reinterpret_cast<const __m128i*>(upperPairs));
		const __m128i lower = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lowerPairs));
		const __m256d p00 = _mm256_cvtepi32_pd(_mm_and_si128(upper, low)), p10 = _mm256_cvtepi32_pd(_mm_srli_epi32(upper, 8));
		const __m256d p01 = _mm256_cvtepi32_pd(_mm_and_si128(lower, low)), p11 = _mm256_cvtepi32_pd(_mm_srli_epi32(lower, 8));
		const __m256d rx = _mm256_sub_pd(one, dx), ry = _mm256_sub_pd(one, dy);
		const __m256d top = _mm256_add_pd(_mm256_mul_pd(p00, rx), _mm256_mul_pd(p10, dx));
		const __m256d bottom = _mm256_add_pd(_mm256_mul_pd(p01, rx), _mm256_mul_pd(p11, dx));
		const __m256d value = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(top, ry), _mm256_mul_pd(bottom, dy)), half);
		const __m128i result = _mm_min_epi32(_mm256_cvttpd_epi32(value), low);
		const int packed = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(result, result), _mm_setzero_si128()));
		std::memcpy(out + x, &packed, 4);
	}
	_mm256_zeroupper();
	return x;
}
#endif

static Image warp(ImageView src, int width, int height, Point anchor, double angle, std::uint8_t border)
{
	Image result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
#if FPM_X86_WARP
	const auto warpRow = x86Isa() >= X86Isa::Avx2 ? warpRowAvx2 : warpRowSse2;
#endif
	for (int y = 0; y < height; ++y) {
		auto* out = result.row(y);
		int x = 0;
#if FPM_NEON_WARP
		x = warpRowNeon(src, out, width, y, anchor, c, s, border);
#elif FPM_X86_WARP
		x = warpRow(src, out, width, y, anchor, c, s, border);
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
