// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <stdexcept>

#if !defined(FPM_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define FPM_NEON 1
#elif !defined(FPM_DISABLE_SIMD) && (defined(__SSE2__) || defined(_M_X64))
#include <emmintrin.h>
#define FPM_SSE2 1
#endif

namespace fpm {
const char* compiledBackend()
{
#if FPM_NEON
	return "NEON";
#elif FPM_SSE2
	return "SSE2";
#else
	return "scalar";
#endif
}

namespace detail {
std::uint64_t dotProduct(const std::uint8_t* a, const std::uint8_t* b, int size, Backend backend)
{
	std::uint64_t total = 0;
	int x = 0;
	if (backend == Backend::Auto) {
		// Each SIMD accumulator is flushed every 4096 bytes. Neither vector
		// lanes nor their horizontal sum can overflow, even for all-255 rows.
		while (size - x >= 16) {
			const int end = x + std::min(4096, (size - x) / 16 * 16);
#if FPM_NEON
			uint32x4_t sum = vdupq_n_u32(0);
			for (; x < end; x += 16) {
				const auto va = vld1q_u8(a + x), vb = vld1q_u8(b + x);
				sum = vaddq_u32(sum, vpaddlq_u16(vmull_u8(vget_low_u8(va), vget_low_u8(vb))));
				sum = vaddq_u32(sum, vpaddlq_u16(vmull_u8(vget_high_u8(va), vget_high_u8(vb))));
			}
			std::uint32_t lanes[4];
			vst1q_u32(lanes, sum);
			for (auto lane : lanes) total += lane;
#elif FPM_SSE2
			__m128i sum = _mm_setzero_si128();
			const auto zero = _mm_setzero_si128();
			for (; x < end; x += 16) {
				const auto va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + x));
				const auto vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + x));
				sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpacklo_epi8(va, zero), _mm_unpacklo_epi8(vb, zero)));
				sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpackhi_epi8(va, zero), _mm_unpackhi_epi8(vb, zero)));
			}
			std::uint32_t lanes[4];
			_mm_storeu_si128(reinterpret_cast<__m128i*>(lanes), sum);
			for (auto lane : lanes) total += lane;
#else
			(void)end;
			break;
#endif
		}
	}
	for (; x < size; ++x) total += unsigned(a[x]) * unsigned(b[x]);
	return total;
}

ScoreMap ncc(ImageView src, const TemplateLevel& templ, Backend backend)
{
	validateImage(src);
	const auto& t = templ.image;
	const int width = src.width - t.width() + 1, height = src.height - t.height() + 1;
	if (t.width() < 1 || t.height() < 1 || width <= 0 || height <= 0)
		throw std::invalid_argument("NCC template must fit the source");
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
			double denominator = 0;
			if (variance > std::min(0.5, 10 * FLT_EPSILON * square))
				denominator = std::sqrt(variance) * templ.norm;
			if (denominator == 0) { out.at(x, y) = 0; continue; }
			double product = 0;
			for (int ty = 0; ty < t.height(); ++ty)
				product += double(dotProduct(t.row(ty), src.data + std::size_t(y + ty) * src.stride + x, t.width(), backend));
			const double numerator = product - sum * templ.mean;
			// Original CCOEFF_Denominator's three-way numerical guard.
			if (std::abs(numerator) < denominator) out.at(x, y) = numerator / denominator;
			else if (std::abs(numerator) < denominator * 1.125) out.at(x, y) = numerator > 0 ? 1 : -1;
			else out.at(x, y) = 0;
		}
	return out;
}

static bool better(Peak a, Peak b)
{
	return a.score > b.score || (a.score == b.score && (a.y < b.y || (a.y == b.y && a.x < b.x)));
}

Peak maximum(const ScoreMap& map)
{
	Peak best;
	for (int y = 0; y < map.height; ++y)
		for (int x = 0; x < map.width; ++x) {
			Peak p{x, y, map.at(x, y)};
			if (better(p, best)) best = p;
		}
	return best;
}

std::vector<Peak> selectPeaks(ScoreMap map, int tw, int th, int count,
	double overlapRatio, double threshold, bool blocked)
{
	struct Block { int x0, y0, x1, y1; Peak best; };
	std::vector<Block> blocks;
	const int bw = blocked ? std::max(1, std::min(tw, map.width / 2) * 2) : map.width;
	const int bh = blocked ? std::max(1, std::min(th, map.height / 2) * 2) : map.height;
	for (int y = 0; y < map.height; y += bh)
		for (int x = 0; x < map.width; x += bw)
			blocks.push_back({x, y, std::min(x + bw, map.width), std::min(y + bh, map.height), {}});
	auto update = [&](Block& b) {
		b.best = {};
		for (int y = b.y0; y < b.y1; ++y)
			for (int x = b.x0; x < b.x1; ++x) {
				Peak p{x, y, map.at(x, y)};
				if (better(p, b.best)) b.best = p;
			}
	};
	for (auto& b : blocks) update(b);
	std::vector<Peak> peaks;
	for (int i = 0; i < count; ++i) {
		Peak p;
		for (const auto& b : blocks) if (better(b.best, p)) p = b.best;
		if (!std::isfinite(p.score) || p.score < threshold) break;
		peaks.push_back(p);
		// Legacy suppression rectangle, plus the selected pixel even at overlap=1.
		const double rx = tw * (1 - overlapRatio), ry = th * (1 - overlapRatio);
		const int left = int(p.x - rx), top = int(p.y - ry);
		const int x0 = std::max(0, std::min(p.x, left));
		const int y0 = std::max(0, std::min(p.y, top));
		const int x1 = std::min(map.width, std::max(p.x + 1, left + int(2 * rx)));
		const int y1 = std::min(map.height, std::max(p.y + 1, top + int(2 * ry)));
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x) map.at(x, y) = -std::numeric_limits<double>::infinity();
		for (auto& b : blocks)
			if (b.x0 < x1 && b.x1 > x0 && b.y0 < y1 && b.y1 > y0) update(b);
	}
	return peaks;
}
} // namespace detail
} // namespace fpm
