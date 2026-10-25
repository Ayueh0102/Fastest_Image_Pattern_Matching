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
// Exact u8 dot products of eight horizontally adjacent windows; each template
// block is loaded once for all of them. Template rows are zero-padded to
// 16*blocks bytes, so source rows are read up to 22 bytes past the first
// window. Between flushes each accumulator receives fewer than 256 + blocks
// UDOTs (blocks <= 256): < 8192 products, below 2^32 even summed over lanes.
static void windowDot8(const std::uint8_t* padded, int blocks, int rows,
	const std::uint8_t* src, std::size_t stride, std::uint64_t* out)
{
	const std::size_t row = std::size_t(16) * blocks;
	uint32x4_t a0 = vdupq_n_u32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
	std::uint64_t total[8] = {};
	int pending = 0;
	for (int y = 0; y < rows; ++y) {
		const auto* t = padded + y * row;
		const auto* p = src + y * stride;
		for (std::size_t b = 0; b < row; b += 16) {
			const uint8x16_t tb = vld1q_u8(t + b);
			a0 = vdotq_u32(a0, tb, vld1q_u8(p + b));
			a1 = vdotq_u32(a1, tb, vld1q_u8(p + b + 1));
			a2 = vdotq_u32(a2, tb, vld1q_u8(p + b + 2));
			a3 = vdotq_u32(a3, tb, vld1q_u8(p + b + 3));
			a4 = vdotq_u32(a4, tb, vld1q_u8(p + b + 4));
			a5 = vdotq_u32(a5, tb, vld1q_u8(p + b + 5));
			a6 = vdotq_u32(a6, tb, vld1q_u8(p + b + 6));
			a7 = vdotq_u32(a7, tb, vld1q_u8(p + b + 7));
		}
		if ((pending += blocks) >= 256) {
			total[0] += vaddvq_u32(a0); total[1] += vaddvq_u32(a1);
			total[2] += vaddvq_u32(a2); total[3] += vaddvq_u32(a3);
			total[4] += vaddvq_u32(a4); total[5] += vaddvq_u32(a5);
			total[6] += vaddvq_u32(a6); total[7] += vaddvq_u32(a7);
			a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = vdupq_n_u32(0);
			pending = 0;
		}
	}
	out[0] = total[0] + vaddvq_u32(a0); out[1] = total[1] + vaddvq_u32(a1);
	out[2] = total[2] + vaddvq_u32(a2); out[3] = total[3] + vaddvq_u32(a3);
	out[4] = total[4] + vaddvq_u32(a4); out[5] = total[5] + vaddvq_u32(a5);
	out[6] = total[6] + vaddvq_u32(a6); out[7] = total[7] + vaddvq_u32(a7);
}
#endif

// Default exhaustive map (no bounds or screens). Window sums and squares slide
// over exact integer column sums instead of binary64 integral images, and UDOT
// builds take products eight windows at a time. Every sum, square and product
// is an exact integer below 2^53, the same value as the original integral
// difference or row-by-row binary64 sum, so all scores are bit-identical.
static void exhaustiveNcc(ImageView src, const TemplateLevel& templ, Backend backend,
	ScoreMap& out, NccStats* stats, NccStats& counters)
{
	const auto& t = templ.image;
	const int tw = t.width(), th = t.height();
	const double area = double(tw) * th;
	// Column sums of th (+1 while sliding) rows: <= 65537*65025 < 2^32, as the
	// caller limits th to 65536.
	std::vector<std::uint32_t> columns(src.width, 0), columnSquares(src.width, 0);
	for (int y = 0; y < th; ++y) {
		const auto* p = src.data + std::size_t(y) * src.stride;
		for (int x = 0; x < src.width; ++x) {
			columns[x] += p[x];
			columnSquares[x] += unsigned(p[x]) * p[x];
		}
	}
	std::vector<double> sums(out.width), squares(out.width), denominators(out.width);
#if FPM_DOTPROD
	const int blocks = (tw + 15) / 16;
	const bool grouped = backend == Backend::Auto && blocks <= 256;
	std::vector<std::uint8_t> padded, source;
	std::vector<std::uint64_t> products;
	if (grouped) {
		const std::size_t row = std::size_t(16) * blocks;
		padded.assign(row * th, 0);
		for (int y = 0; y < th; ++y) std::copy_n(t.row(y), tw, padded.data() + y * row);
		// Tight rows plus a zero tail that covers the last group's reads.
		source.assign(std::size_t(src.width) * src.height + row + 8, 0);
		for (int y = 0; y < src.height; ++y)
			std::copy_n(src.data + std::size_t(y) * src.stride, src.width, source.data() + std::size_t(y) * src.width);
		products.resize(std::size_t(out.width) + 7);
	}
#endif
	for (int y = 0; y < out.height; ++y) {
		if (y) {
			const auto* removed = src.data + std::size_t(y - 1) * src.stride;
			const auto* added = src.data + std::size_t(y + th - 1) * src.stride;
			for (int x = 0; x < src.width; ++x) {
				columns[x] = columns[x] + added[x] - removed[x];
				columnSquares[x] = columnSquares[x] + unsigned(added[x]) * added[x] - unsigned(removed[x]) * removed[x];
			}
		}
		std::uint64_t sum = 0, square = 0;
		for (int x = 0; x < tw; ++x) { sum += columns[x]; square += columnSquares[x]; }
		for (int x = 0; x < out.width; ++x) {
			if (x) {
				sum = sum + columns[x + tw - 1] - columns[x - 1];
				square = square + columnSquares[x + tw - 1] - columnSquares[x - 1];
			}
			sums[x] = double(sum);
			squares[x] = double(square);
			const double variance = std::max(0.0, squares[x] - sums[x] * sums[x] / area);
			denominators[x] = nccDenominator(variance, squares[x], templ.norm);
		}
#if FPM_DOTPROD
		// Like the original, products are only needed where denominator != 0.
		if (grouped)
			for (int x = 0; x < out.width; x += 8) {
				bool needed = false;
				for (int k = x; k < std::min(x + 8, out.width); ++k) needed |= denominators[k] != 0;
				if (needed)
					windowDot8(padded.data(), blocks, th, source.data() + std::size_t(y) * src.width + x,
						src.width, products.data() + x);
			}
#endif
		for (int x = 0; x < out.width; ++x) {
			if (denominators[x] == 0) { out.at(x, y) = 0; continue; }
			double product = 0;
#if FPM_DOTPROD
			if (grouped) product = double(products[x]);
			else
#endif
			for (int ty = 0; ty < th; ++ty)
				product += double(dotProduct(t.row(ty), src.data + std::size_t(y + ty) * src.stride + x, tw, backend));
			if (stats) { counters.dotPixels += std::uint64_t(th) * tw; ++counters.fullDotProducts; }
			out.at(x, y) = nccNormalizeProduct(product, sums[x], templ.mean, denominators[x]);
		}
	}
}

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
	// Integral entries are exact integers within binary64's integer range.
	// Beyond it, keep the original evaluation path instead of relying on a
	// cancellation-sensitive bound. The guard is far above practical images.
	const bool integerRange = double(src.width) * src.height <= 9007199254740992.0 / 65025;
	const bool useBounds = std::isfinite(rejectBelow) && rejectBelow > 0 && rejectBelow <= 1 &&
		area >= 256 && integerRange;
	const bool screened = screen && std::isfinite(rejectBelow) && rejectBelow > 0 &&
		rejectBelow <= 1 && screen->supports(src);
	if (integerRange && !useBounds && !screened && t.height() <= 65536) {
		exhaustiveNcc(src, templ, backend, out, stats, counters);
		if (stats) *stats = counters;
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
			const double sum = sums[d] - sums[b] - sums[c] + sums[a];
			const double square = squares[d] - squares[b] - squares[c] + squares[a];
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
