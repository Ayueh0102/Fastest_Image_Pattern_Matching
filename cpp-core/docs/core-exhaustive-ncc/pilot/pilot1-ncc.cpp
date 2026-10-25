// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <stdexcept>

#if !defined(FPM_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define FPM_NEON 1
// Compile-time selection: e.g. Apple clang's default arm64 CPU or GCC with
// -march=armv8.2-a+dotprod. Other ARM builds keep the widening NEON path.
#if defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
#define FPM_DOTPROD 1
#endif
#elif !defined(FPM_DISABLE_SIMD) && (defined(__SSE2__) || defined(_M_X64))
#include <emmintrin.h>
#define FPM_SSE2 1
#endif

namespace fpm {
const char* compiledBackend()
{
#if FPM_DOTPROD
	return "NEON+DOTPROD";
#elif FPM_NEON
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
#if FPM_DOTPROD
			// UDOT adds four products (each <= 65025) to a u32 lane. A chunk
			// has at most 4096 products, so even the sum of all four lanes is
			// <= 266342400 < 2^32 and vaddvq_u32 is exact.
			uint32x4_t sum = vdupq_n_u32(0);
			for (; x < end; x += 16) sum = vdotq_u32(sum, vld1q_u8(a + x), vld1q_u8(b + x));
			total += vaddvq_u32(sum);
#elif FPM_NEON
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
#if FPM_DOTPROD
		// A 1..15 byte tail of a row of at least 16 bytes: reload the last 16
		// bytes, which stay in bounds, and zero the lanes already counted.
		if (x >= 16 && x < size) {
			static const std::uint8_t tail[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
			const auto va = vandq_u8(vld1q_u8(a + size - 16), vld1q_u8(tail + (size - x)));
			total += vaddvq_u32(vdotq_u32(vdupq_n_u32(0), va, vld1q_u8(b + size - 16)));
			x = size;
		}
#endif
	}
	for (; x < size; ++x) total += unsigned(a[x]) * unsigned(b[x]);
	return total;
}

#if FPM_DOTPROD
// Exact u8 dot product of a whole window. Template rows are zero-padded to
// 16*blocks bytes, so each source row may be read up to 15 bytes past the
// window; the caller keeps the last read inside the source buffer. Four rows
// feed four accumulators. Between flushes the lanes receive fewer than
// 256 + 4*blocks UDOTs (blocks <= 256): <= 20480 products, < 2^32 in total.
static std::uint64_t windowDot(const std::uint8_t* padded, int blocks, int rows,
	const std::uint8_t* src, std::size_t stride)
{
	const std::size_t row = std::size_t(16) * blocks;
	std::uint64_t total = 0;
	uint32x4_t s0 = vdupq_n_u32(0), s1 = s0, s2 = s0, s3 = s0;
	int pending = 0, y = 0;
	for (; y + 4 <= rows; y += 4) {
		const auto* t = padded + y * row;
		const auto* p = src + y * stride;
		for (std::size_t b = 0; b < row; b += 16) {
			s0 = vdotq_u32(s0, vld1q_u8(t + b), vld1q_u8(p + b));
			s1 = vdotq_u32(s1, vld1q_u8(t + row + b), vld1q_u8(p + stride + b));
			s2 = vdotq_u32(s2, vld1q_u8(t + 2 * row + b), vld1q_u8(p + 2 * stride + b));
			s3 = vdotq_u32(s3, vld1q_u8(t + 3 * row + b), vld1q_u8(p + 3 * stride + b));
		}
		if ((pending += 4 * blocks) >= 256) {
			total += vaddvq_u32(vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3)));
			s0 = s1 = s2 = s3 = vdupq_n_u32(0);
			pending = 0;
		}
	}
	for (; y < rows; ++y)
		for (std::size_t b = 0; b < row; b += 16)
			s0 = vdotq_u32(s0, vld1q_u8(padded + y * row + b), vld1q_u8(src + y * stride + b));
	return total + vaddvq_u32(vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3)));
}
#endif

static ScoreMap computeNcc(ImageView src, const TemplateLevel& templ, Backend backend,
	double rejectBelow, NccStats* stats, const NccScreen* screen)
{
	validateImage(src);
	const auto& t = templ.image;
	const int width = src.width - t.width() + 1, height = src.height - t.height() + 1;
	if (t.width() < 1 || t.height() < 1 || width <= 0 || height <= 0)
		throw std::invalid_argument("NCC template must fit the source");
	ScoreMap out{width, height, std::vector<double>(std::size_t(width) * height)};
	NccStats counters;
	counters.windows = out.values.size();
	if (stats) *stats = counters;
	const double area = double(t.width()) * t.height();
	if (templ.norm * templ.norm / area < DBL_EPSILON) {
		std::fill(out.values.begin(), out.values.end(), 1.0);
		return out;
	}
	const std::size_t stride = std::size_t(src.width) + 1;
	// Integral entries are exact integers within binary64's integer range.
	// Beyond it, keep the original evaluation path instead of relying on a
	// cancellation-sensitive bound. The guard is far above practical images.
	const bool integerRange = double(src.width) * src.height <= 9007199254740992.0 / 65025;
	const bool useBounds = std::isfinite(rejectBelow) && rejectBelow > 0 && rejectBelow <= 1 &&
		area >= 256 && integerRange;
	const bool screened = screen && std::isfinite(rejectBelow) && rejectBelow > 0 &&
		rejectBelow <= 1 && screen->supports(src);
	// Exhaustive maps keep the integral images in uint64. Each binary64 entry
	// and each partial sum of row dot products is an exact integer below 2^53
	// here, so window sums, squares and products are the same values.
	const bool exact = integerRange && !useBounds && !screened;
	std::vector<double> sums, squares;
	std::vector<std::uint64_t> exactSums, exactSquares;
	if (exact) {
		exactSums.resize(stride * (std::size_t(src.height) + 1));
		exactSquares.resize(exactSums.size());
		for (int y = 0; y < src.height; ++y) {
			std::uint64_t sum = 0, square = 0;
			const auto* row = src.data + std::size_t(y) * src.stride;
			const auto i = (std::size_t(y) + 1) * stride + 1;
			for (int x = 0; x < src.width; ++x) {
				const unsigned v = row[x];
				sum += v;
				square += v * v;
				exactSums[i + x] = exactSums[i + x - stride] + sum;
				exactSquares[i + x] = exactSquares[i + x - stride] + square;
			}
		}
	} else {
		sums.assign(stride * (std::size_t(src.height) + 1), 0);
		squares.assign(sums.size(), 0);
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
	}
#if FPM_DOTPROD
	const int padBlocks = (t.width() + 15) / 16;
	const std::size_t padRow = std::size_t(16) * padBlocks;
	std::vector<std::uint8_t> padded;
	if (exact && backend == Backend::Auto && padBlocks <= 256) {
		padded.assign(padRow * t.height(), 0);
		for (int y = 0; y < t.height(); ++y) std::copy_n(t.row(y), t.width(), padded.data() + y * padRow);
	}
	// Last byte a padded window may read, relative to its last source row.
	const std::size_t readLimit = std::size_t(src.height - 1) * src.stride + src.width;
#endif
	struct Block { int x0, y0, x1, y1; double count, centeredSum; };
	std::array<Block, 4> blocks{};
	double templateProjection = 0;
	bool project = false;
	// Roundoff allowance covers centering, projection and the learned norm's
	// area-term sum. Inflate residual energies as well as the final numerator;
	// candidates on/near the threshold therefore fall back to exact evaluation.
	const double roundoff = 64 * DBL_EPSILON * area * area * 65025;
	if (!screened && useBounds && t.width() >= 2 && t.height() >= 2) {
		int index = 0;
		for (int by = 0; by < 2; ++by) for (int bx = 0; bx < 2; ++bx) {
			auto& b = blocks[index++];
			b.x0 = bx * t.width() / 2; b.x1 = (bx + 1) * t.width() / 2;
			b.y0 = by * t.height() / 2; b.y1 = (by + 1) * t.height() / 2;
			b.count = double(b.x1 - b.x0) * (b.y1 - b.y0);
			double sum = 0;
			for (int y = b.y0; y < b.y1; ++y)
				for (int x = b.x0; x < b.x1; ++x) sum += t.row(y)[x];
			b.centeredSum = sum - b.count * templ.mean;
			templateProjection += b.centeredSum * b.centeredSum / b.count;
		}
		// For unstructured patches this bound is at least approximately
		// sqrt(1 - projected_fraction). Skip it when it is unlikely to reject.
		project = templateProjection / (templ.norm * templ.norm) > 1 - rejectBelow * rejectBelow + .05;
	}
	const bool partial = useBounds && area >= 1024 && rejectBelow >= .5 && t.height() >= 4;
	std::array<int, 2> checkRows{};
	std::array<double, 2> prefixTemplateSum{}, tailTemplateEnergy{};
	if (partial) {
		checkRows[0] = std::clamp(int(std::ceil(t.height() * (1 - rejectBelow + .1))), 1, t.height() - 2);
		checkRows[1] = (checkRows[0] + t.height()) / 2;
		for (int k = 0; k < 2; ++k)
			for (int y = 0; y < t.height(); ++y)
				for (int x = 0; x < t.width(); ++x) {
					if (y < checkRows[k]) prefixTemplateSum[k] += t.row(y)[x];
					else {
						const double v = t.row(y)[x] - templ.mean;
						tailTemplateEnergy[k] += v * v;
					}
				}
	}
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			const std::size_t a = std::size_t(y) * stride + x;
			const std::size_t b = a + t.width(), c = a + std::size_t(t.height()) * stride, d = c + t.width();
			const double sum = exact ? double(exactSums[d] - exactSums[b] - (exactSums[c] - exactSums[a])) :
				sums[d] - sums[b] - sums[c] + sums[a];
			const double square = exact ? double(exactSquares[d] - exactSquares[b] - (exactSquares[c] - exactSquares[a])) :
				squares[d] - squares[b] - squares[c] + squares[a];
			const double variance = std::max(0.0, square - sum * sum / area);
			const double denominator = nccDenominator(variance, square, templ.norm);
			if (denominator == 0) { out.at(x, y) = 0; continue; }
			const double sourceMean = sum / area;
			if (screened && screen->reject(sums.data(), squares.data(), stride, a,
				sum, square, denominator, rejectBelow, stats ? &counters : nullptr)) {
				out.at(x, y) = -std::numeric_limits<double>::infinity();
				if (stats) { ++counters.rejected; ++counters.adaptiveRejected; }
				continue;
			}
			if (project) {
				if (stats) ++counters.projectionChecks;
				double product = 0, projection = 0;
				for (const auto& block : blocks) {
					const auto i = a + std::size_t(block.y0) * stride + block.x0;
					const auto j = a + std::size_t(block.y0) * stride + block.x1;
					const auto k = a + std::size_t(block.y1) * stride + block.x0;
					const auto l = a + std::size_t(block.y1) * stride + block.x1;
					const double centeredSum = sums[l] - sums[j] - sums[k] + sums[i] - block.count * sourceMean;
					product += centeredSum * block.centeredSum / block.count;
					projection += centeredSum * centeredSum / block.count;
				}
				// Orthogonal projection onto the four rectangular means:
				// <x,t> <= <Px,Pt> + ||(I-P)x|| ||(I-P)t||.
				const double sourceResidual = std::max(0.0, variance - projection) + roundoff;
				const double templateResidual = std::max(0.0, templ.norm * templ.norm - templateProjection) + roundoff;
				const double upper = product + std::sqrt(sourceResidual * templateResidual) + roundoff;
				if (upper < rejectBelow * denominator) {
					out.at(x, y) = -std::numeric_limits<double>::infinity();
					if (stats) ++counters.rejected;
					continue;
				}
			}
			double product = 0;
			int rows = 0, checkpoint = 0;
			bool rejected = false;
			if (!partial) {
#if FPM_DOTPROD
				// The same exact integer as the row-by-row binary64 sum below.
				if (!padded.empty() && std::size_t(y + t.height() - 1) * src.stride + x + padRow <= readLimit)
					product = double(windowDot(padded.data(), padBlocks, t.height(),
						src.data + std::size_t(y) * src.stride + x, src.stride));
				else
#endif
				for (int ty = 0; ty < t.height(); ++ty)
					product += double(dotProduct(t.row(ty), src.data + std::size_t(y + ty) * src.stride + x, t.width(), backend));
				rows = t.height();
			} else for (int ty = 0; ty < t.height(); ++ty) {
				product += double(dotProduct(t.row(ty), src.data + std::size_t(y + ty) * src.stride + x, t.width(), backend));
				rows = ty + 1;
				if (checkpoint >= 2 || rows != checkRows[checkpoint]) continue;
				if (stats) ++counters.partialChecks;
				const auto e = a + std::size_t(rows) * stride, f = e + t.width();
				const double prefixSum = sums[f] - sums[b] - sums[e] + sums[a];
				const double prefixSquare = squares[f] - squares[b] - squares[e] + squares[a];
				const double prefixCount = double(rows) * t.width();
				const double centeredProduct = product - templ.mean * prefixSum -
					sourceMean * (prefixTemplateSum[checkpoint] - prefixCount * templ.mean);
				const double tailEnergy = std::max(0.0, square - prefixSquare - 2 * sourceMean * (sum - prefixSum) +
					(area - prefixCount) * sourceMean * sourceMean) + roundoff;
				const double upper = centeredProduct +
					std::sqrt(tailEnergy * (tailTemplateEnergy[checkpoint] + roundoff)) + roundoff;
				++checkpoint;
				if (upper < rejectBelow * denominator) { rejected = true; break; }
			}
			if (stats) counters.dotPixels += std::uint64_t(rows) * t.width();
			if (rejected) {
				out.at(x, y) = -std::numeric_limits<double>::infinity();
				if (stats) ++counters.rejected;
				continue;
			}
			if (stats) ++counters.fullDotProducts;
			out.at(x, y) = nccNormalizeProduct(product, sum, templ.mean, denominator);
		}
	if (stats) *stats = counters;
	return out;
}

ScoreMap ncc(ImageView source, const TemplateLevel& templ, Backend backend,
	double rejectBelow, NccStats* stats)
{
	return computeNcc(source, templ, backend, rejectBelow, stats, nullptr);
}

ScoreMap nccScreened(ImageView source, const TemplateLevel& templ, Backend backend,
	double rejectBelow, const NccScreen& screen, NccStats* stats)
{
	return computeNcc(source, templ, backend, rejectBelow, stats, &screen);
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

std::vector<Peak> selectPeaksFast(ScoreMap map, int tw, int th, int count,
	double overlapRatio, double threshold, bool blocked)
{
	// `blocked` chooses only an implementation in the legacy selector. Both
	// choices have the same score / y / x ordering and suppression rectangle.
	(void)blocked;
	std::vector<Peak> peaks;
	if (count <= 0 || map.width <= 0 || map.height <= 0) return peaks;
	if (count == 1) {
		const auto p = maximum(map);
		if (std::isfinite(p.score) && !(p.score < threshold)) peaks.push_back(p);
		return peaks;
	}

	// A bounded tournament tree stores one winner per 32x32 tile. Suppression
	// only decreases scores: a tile whose winner survives cannot have a new
	// winner, even if other pixels in that tile have been suppressed. This
	// avoids rescanning every intersecting block after every selected peak.
	// The tree has fewer than 4 * tileCount Peak entries, independent of count.
	constexpr int tileSide = 32;
	const int columns = (map.width - 1) / tileSide + 1;
	const int rows = (map.height - 1) / tileSide + 1;
	const std::size_t tileCount = std::size_t(columns) * rows;
	std::size_t leafBase = 1;
	while (leafBase < tileCount) leafBase *= 2;
	std::vector<Peak> tree(leafBase * 2);
	auto winner = [](Peak a, Peak b) { return better(b, a) ? b : a; };
	auto scan = [&](int tx, int ty) {
		Peak best;
		const int x0 = tx * tileSide, y0 = ty * tileSide;
		const int x1 = x0 + std::min(tileSide, map.width - x0);
		const int y1 = y0 + std::min(tileSide, map.height - y0);
		for (int y = y0; y < y1; ++y)
			for (int x = x0; x < x1; ++x) {
				const Peak p{x, y, map.at(x, y)};
				if (better(p, best)) best = p;
			}
		return best;
	};
	for (int ty = 0; ty < rows; ++ty)
		for (int tx = 0; tx < columns; ++tx)
			tree[leafBase + std::size_t(ty) * columns + tx] = scan(tx, ty);
	for (std::size_t node = leafBase; node-- > 1;)
		tree[node] = winner(tree[node * 2], tree[node * 2 + 1]);
	for (int i = 0; i < count; ++i) {
		const auto p = tree[1];
		// Keep the legacy NaN-threshold behavior and its stop at +infinity.
		// NaN map values never become winners because better() discards them.
		if (!std::isfinite(p.score) || p.score < threshold) break;
		peaks.push_back(p);
		if (i + 1 == count) break;
		// Deliberately retain truncation toward zero and the asymmetric legacy
		// endpoint calculation, including suppression of the selected pixel
		// when overlapRatio == 1.
		const double rx = tw * (1 - overlapRatio), ry = th * (1 - overlapRatio);
		const int left = int(p.x - rx), top = int(p.y - ry);
		const int x0 = std::max(0, std::min(p.x, left));
		const int y0 = std::max(0, std::min(p.y, top));
		const int x1 = std::min(map.width, std::max(p.x + 1, left + int(2 * rx)));
		const int y1 = std::min(map.height, std::max(p.y + 1, top + int(2 * ry)));
		for (int y = y0; y < y1; ++y)
			std::fill_n(map.values.data() + std::size_t(y) * map.width + x0,
				x1 - x0, -std::numeric_limits<double>::infinity());
		for (int ty = y0 / tileSide; ty <= (y1 - 1) / tileSide; ++ty)
			for (int tx = x0 / tileSide; tx <= (x1 - 1) / tileSide; ++tx) {
				auto node = leafBase + std::size_t(ty) * columns + tx;
				const auto old = tree[node];
				if (old.x < x0 || old.x >= x1 || old.y < y0 || old.y >= y1) continue;
				tree[node] = scan(tx, ty);
				while (node > 1) {
					node /= 2;
					const auto next = winner(tree[node * 2], tree[node * 2 + 1]);
					const auto previous = tree[node];
					if (next.score == previous.score && next.x == previous.x && next.y == previous.y) break;
					tree[node] = next;
				}
			}
	}
	return peaks;
}
} // namespace detail
} // namespace fpm
