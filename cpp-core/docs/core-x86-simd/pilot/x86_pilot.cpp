// x86 pilot (selection only): exact warp rows and 8-window u8 dot products.
// Every variant is compared byte for byte (warp) or exactly (dot) with a copy of
// the scalar reference before it is timed. Build: cl /O2 /EHsc x86_pilot.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define T_AVX2
#define T_VNNI8
#else
#include <cpuid.h>
#define T_AVX2 __attribute__((target("avx2")))
#define T_VNNI8 __attribute__((target("avx2,avxvnniint8")))
#endif

struct Img { int w, h; std::size_t stride; std::vector<std::uint8_t> px; const std::uint8_t* data() const { return px.data(); } };

static bool hasAvx2 = false, hasVnni8 = false;
static void detect()
{
	int r[4];
#if defined(_MSC_VER)
	__cpuid(r, 1);
#else
	__cpuid(1, r[0], r[1], r[2], r[3]);
#endif
	const bool osxsave = r[2] >> 27 & 1, avx = r[2] >> 28 & 1;
	unsigned long long xcr0 = 0;
	if (osxsave) {
#if defined(_MSC_VER)
		xcr0 = _xgetbv(0);
#else
		unsigned lo, hi; __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0)); xcr0 = (unsigned long long)hi << 32 | lo;
#endif
	}
	const bool ymm = avx && (xcr0 & 6) == 6;
#if defined(_MSC_VER)
	__cpuidex(r, 7, 0);
#else
	__cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
#endif
	hasAvx2 = ymm && (r[1] >> 5 & 1);
#if defined(_MSC_VER)
	__cpuidex(r, 7, 1);
#else
	__cpuid_count(7, 1, r[0], r[1], r[2], r[3]);
#endif
	hasVnni8 = hasAvx2 && (r[3] >> 4 & 1);
}

// ---------------- warp: verbatim scalar reference ----------------
static std::uint8_t pixel(const Img& s, int x, int y, std::uint8_t border)
{
	return x < 0 || y < 0 || x >= s.w || y >= s.h ? border : s.px[std::size_t(y) * s.stride + x];
}
static std::uint8_t warpPixel(const Img& src, int x, int y, double ax, double ay, double c, double s, std::uint8_t border)
{
	const double sx = ax + c * x - s * y;
	const double sy = ay + s * x + c * y;
	if (sx < -1 || sy < -1 || sx >= src.w || sy >= src.h) return border;
	const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
	const double dx = sx - ix, dy = sy - iy;
	const double top = pixel(src, ix, iy, border) * (1 - dx) + pixel(src, ix + 1, iy, border) * dx;
	const double bottom = pixel(src, ix, iy + 1, border) * (1 - dx) + pixel(src, ix + 1, iy + 1, border) * dx;
	return std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
}
static int rowScalar(const Img&, std::uint8_t*, int, int, double, double, double, double, std::uint8_t) { return 0; }

static std::uint32_t pair16(const std::uint8_t* p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }

// SSE2, 2 px: unfused ops in the scalar order; floor == trunc on the inside path (sx, sy >= 0).
static int rowSse2(const Img& src, std::uint8_t* out, int width, int y, double ax, double ay, double c, double s, std::uint8_t border)
{
	const __m128d vc = _mm_set1_pd(c), vs = _mm_set1_pd(s), vy = _mm_set1_pd(y), vax = _mm_set1_pd(ax), vay = _mm_set1_pd(ay);
	const __m128d zero = _mm_setzero_pd(), one = _mm_set1_pd(1), half = _mm_set1_pd(.5), two = _mm_set1_pd(2);
	const __m128d innerW = _mm_set1_pd(src.w - 1), innerH = _mm_set1_pd(src.h - 1);
	const __m128d sy0 = _mm_mul_pd(vs, vy), cy0 = _mm_mul_pd(vc, vy);
	__m128d vx = _mm_set_pd(1, 0);
	int x = 0;
	for (; x + 2 <= width; x += 2, vx = _mm_add_pd(vx, two)) {
		const __m128d sx = _mm_sub_pd(_mm_add_pd(vax, _mm_mul_pd(vc, vx)), sy0);
		const __m128d syv = _mm_add_pd(_mm_add_pd(vay, _mm_mul_pd(vs, vx)), cy0);
		const __m128d in = _mm_and_pd(_mm_and_pd(_mm_cmpge_pd(sx, zero), _mm_cmplt_pd(sx, innerW)),
			_mm_and_pd(_mm_cmpge_pd(syv, zero), _mm_cmplt_pd(syv, innerH)));
		if (_mm_movemask_pd(in) != 3) {
			out[x] = warpPixel(src, x, y, ax, ay, c, s, border);
			out[x + 1] = warpPixel(src, x + 1, y, ax, ay, c, s, border);
			continue;
		}
		const __m128i ix = _mm_cvttpd_epi32(sx), iy = _mm_cvttpd_epi32(syv);
		const __m128d dx = _mm_sub_pd(sx, _mm_cvtepi32_pd(ix)), dy = _mm_sub_pd(syv, _mm_cvtepi32_pd(iy));
		alignas(16) int xi[4], yi[4];
		_mm_store_si128((__m128i*)xi, ix); _mm_store_si128((__m128i*)yi, iy);
		const std::uint8_t* p = src.data() + std::size_t(yi[0]) * src.stride + xi[0];
		const std::uint8_t* q = src.data() + std::size_t(yi[1]) * src.stride + xi[1];
		const std::uint32_t pu = pair16(p), qu = pair16(q), pl = pair16(p + src.stride), ql = pair16(q + src.stride);
		const __m128d p00 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, int(qu & 255), int(pu & 255)));
		const __m128d p10 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, int(qu >> 8), int(pu >> 8)));
		const __m128d p01 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, int(ql & 255), int(pl & 255)));
		const __m128d p11 = _mm_cvtepi32_pd(_mm_set_epi32(0, 0, int(ql >> 8), int(pl >> 8)));
		const __m128d rx = _mm_sub_pd(one, dx), ry = _mm_sub_pd(one, dy);
		const __m128d top = _mm_add_pd(_mm_mul_pd(p00, rx), _mm_mul_pd(p10, dx));
		const __m128d bottom = _mm_add_pd(_mm_mul_pd(p01, rx), _mm_mul_pd(p11, dx));
		const __m128d value = _mm_add_pd(_mm_add_pd(_mm_mul_pd(top, ry), _mm_mul_pd(bottom, dy)), half);
		alignas(16) int v[4];
		_mm_store_si128((__m128i*)v, _mm_cvttpd_epi32(value));
		out[x] = std::uint8_t(std::min(v[0], 255));
		out[x + 1] = std::uint8_t(std::min(v[1], 255));
	}
	return x;
}

// AVX2, 4 px. Gather variant reads 4 bytes per row at each pixel, so the inside
// test also requires sx < w - 3 (stricter tests only send more pixels to warpPixel).
template <bool Gather>
T_AVX2 static int rowAvx2(const Img& src, std::uint8_t* out, int width, int y, double ax, double ay, double c, double s, std::uint8_t border)
{
	const __m256d vc = _mm256_set1_pd(c), vs = _mm256_set1_pd(s), vy = _mm256_set1_pd(y), vax = _mm256_set1_pd(ax), vay = _mm256_set1_pd(ay);
	const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1), half = _mm256_set1_pd(.5), four = _mm256_set1_pd(4);
	const __m256d innerW = _mm256_set1_pd(Gather ? src.w - 3 : src.w - 1), innerH = _mm256_set1_pd(src.h - 1);
	const __m256d sy0 = _mm256_mul_pd(vs, vy), cy0 = _mm256_mul_pd(vc, vy);
	const __m128i stride = _mm_set1_epi32(int(src.stride)), low = _mm_set1_epi32(255);
	__m256d vx = _mm256_set_pd(3, 2, 1, 0);
	int x = 0;
	for (; x + 4 <= width; x += 4, vx = _mm256_add_pd(vx, four)) {
		const __m256d sx = _mm256_sub_pd(_mm256_add_pd(vax, _mm256_mul_pd(vc, vx)), sy0);
		const __m256d syv = _mm256_add_pd(_mm256_add_pd(vay, _mm256_mul_pd(vs, vx)), cy0);
		const __m256d in = _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(sx, zero, _CMP_GE_OQ), _mm256_cmp_pd(sx, innerW, _CMP_LT_OQ)),
			_mm256_and_pd(_mm256_cmp_pd(syv, zero, _CMP_GE_OQ), _mm256_cmp_pd(syv, innerH, _CMP_LT_OQ)));
		if (_mm256_movemask_pd(in) != 15) {
			for (int k = 0; k < 4; ++k) out[x + k] = warpPixel(src, x + k, y, ax, ay, c, s, border);
			continue;
		}
		const __m128i ix = _mm256_cvttpd_epi32(sx), iy = _mm256_cvttpd_epi32(syv);
		const __m256d dx = _mm256_sub_pd(sx, _mm256_cvtepi32_pd(ix)), dy = _mm256_sub_pd(syv, _mm256_cvtepi32_pd(iy));
		__m128i upper, lower;
		if (Gather) {
			const __m128i offset = _mm_add_epi32(_mm_mullo_epi32(iy, stride), ix);
			upper = _mm_i32gather_epi32((const int*)src.data(), offset, 1);
			lower = _mm_i32gather_epi32((const int*)(src.data() + src.stride), offset, 1);
		} else {
			alignas(16) int xi[4], yi[4];
			_mm_store_si128((__m128i*)xi, ix); _mm_store_si128((__m128i*)yi, iy);
			std::uint32_t u[4], l[4];
			for (int k = 0; k < 4; ++k) {
				const std::uint8_t* p = src.data() + std::size_t(yi[k]) * src.stride + xi[k];
				u[k] = pair16(p); l[k] = pair16(p + src.stride);
			}
			upper = _mm_set_epi32(int(u[3]), int(u[2]), int(u[1]), int(u[0]));
			lower = _mm_set_epi32(int(l[3]), int(l[2]), int(l[1]), int(l[0]));
		}
		const __m256d p00 = _mm256_cvtepi32_pd(_mm_and_si128(upper, low)), p10 = _mm256_cvtepi32_pd(_mm_and_si128(_mm_srli_epi32(upper, 8), low));
		const __m256d p01 = _mm256_cvtepi32_pd(_mm_and_si128(lower, low)), p11 = _mm256_cvtepi32_pd(_mm_and_si128(_mm_srli_epi32(lower, 8), low));
		const __m256d rx = _mm256_sub_pd(one, dx), ry = _mm256_sub_pd(one, dy);
		const __m256d top = _mm256_add_pd(_mm256_mul_pd(p00, rx), _mm256_mul_pd(p10, dx));
		const __m256d bottom = _mm256_add_pd(_mm256_mul_pd(p01, rx), _mm256_mul_pd(p11, dx));
		const __m256d value = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(top, ry), _mm256_mul_pd(bottom, dy)), half);
		const __m128i v = _mm_min_epi32(_mm256_cvttpd_epi32(value), low);
		const __m128i packed = _mm_packus_epi16(_mm_packs_epi32(v, v), _mm_setzero_si128());
		const int word = _mm_cvtsi128_si32(packed);
		std::memcpy(out + x, &word, 4);
	}
	_mm256_zeroupper();
	return x;
}

using RowFn = int (*)(const Img&, std::uint8_t*, int, int, double, double, double, double, std::uint8_t);
static void warp(RowFn row, const Img& src, std::vector<std::uint8_t>& out, int width, int height, double ax, double ay, double angle, std::uint8_t border)
{
	const double c = std::cos(angle * 3.14159265358979323846 / 180), s = std::sin(angle * 3.14159265358979323846 / 180);
	for (int y = 0; y < height; ++y) {
		auto* o = out.data() + std::size_t(y) * width;
		int x = row(src, o, width, y, ax, ay, c, s, border);
		for (; x < width; ++x) o[x] = warpPixel(src, x, y, ax, ay, c, s, border);
	}
}

// ---------------- 8-window exact dot products ----------------
static void dotScalar(const std::uint8_t* t, int tw, int th, const std::uint8_t* src, std::size_t stride, std::uint64_t* out)
{
	for (int k = 0; k < 8; ++k) {
		std::uint64_t total = 0;
		for (int y = 0; y < th; ++y)
			for (int x = 0; x < tw; ++x) total += unsigned(t[y * tw + x]) * src[y * stride + x + k];
		out[k] = total;
	}
}
static std::uint64_t hsum128(__m128i v) { alignas(16) std::uint32_t l[4]; _mm_store_si128((__m128i*)l, v); return std::uint64_t(l[0]) + l[1] + l[2] + l[3]; }

// SSE2: template block widened once; per window 2 unpacks, 2 madd, 2 adds (4 products per lane per block).
static void dot8Sse2(const std::uint8_t* padded, int blocks, int rows, const std::uint8_t* src, std::size_t stride, std::uint64_t* out)
{
	const std::size_t row = std::size_t(16) * blocks;
	const __m128i zero = _mm_setzero_si128();
	__m128i a[8];
	for (auto& v : a) v = zero;
	std::uint64_t total[8] = {};
	int pending = 0;
	for (int y = 0; y < rows; ++y) {
		const auto* t = padded + y * row;
		const auto* p = src + y * stride;
		for (std::size_t b = 0; b < row; b += 16) {
			const __m128i tb = _mm_loadu_si128((const __m128i*)(t + b));
			const __m128i tl = _mm_unpacklo_epi8(tb, zero), th = _mm_unpackhi_epi8(tb, zero);
			for (int k = 0; k < 8; ++k) {
				const __m128i sb = _mm_loadu_si128((const __m128i*)(p + b + k));
				a[k] = _mm_add_epi32(a[k], _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi8(sb, zero), tl), _mm_madd_epi16(_mm_unpackhi_epi8(sb, zero), th)));
			}
		}
		if ((pending += blocks) >= 256) {
			for (int k = 0; k < 8; ++k) { total[k] += hsum128(a[k]); a[k] = zero; }
			pending = 0;
		}
	}
	for (int k = 0; k < 8; ++k) out[k] = total[k] + hsum128(a[k]);
}

T_AVX2 static std::uint64_t hsum256(__m256i v) { alignas(32) std::uint32_t l[8]; _mm256_store_si256((__m256i*)l, v); std::uint64_t s = 0; for (auto x : l) s += x; return s; }

// AVX2: per window one zero-extending load and one 16-lane madd per block (2 products per lane).
T_AVX2 static void dot8Avx2(const std::uint8_t* padded, int blocks, int rows, const std::uint8_t* src, std::size_t stride, std::uint64_t* out)
{
	const std::size_t row = std::size_t(16) * blocks;
	__m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
	std::uint64_t total[8] = {};
	int pending = 0;
	for (int y = 0; y < rows; ++y) {
		const auto* t = padded + y * row;
		const auto* p = src + y * stride;
		for (std::size_t b = 0; b < row; b += 16) {
			const __m256i tb = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(t + b)));
			const auto* q = p + b;
#define W(k, acc) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(q + k))), tb))
			W(0, a0); W(1, a1); W(2, a2); W(3, a3); W(4, a4); W(5, a5); W(6, a6); W(7, a7);
#undef W
		}
		if ((pending += blocks) >= 256) {
			total[0] += hsum256(a0); total[1] += hsum256(a1); total[2] += hsum256(a2); total[3] += hsum256(a3);
			total[4] += hsum256(a4); total[5] += hsum256(a5); total[6] += hsum256(a6); total[7] += hsum256(a7);
			a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = _mm256_setzero_si256();
			pending = 0;
		}
	}
	out[0] = total[0] + hsum256(a0); out[1] = total[1] + hsum256(a1); out[2] = total[2] + hsum256(a2); out[3] = total[3] + hsum256(a3);
	out[4] = total[4] + hsum256(a4); out[5] = total[5] + hsum256(a5); out[6] = total[6] + hsum256(a6); out[7] = total[7] + hsum256(a7);
	_mm256_zeroupper();
}

// AVX-VNNI-INT8 vpdpbuud on 16-byte blocks (xmm, same accumulation as UDOT).
T_VNNI8 static void dot8Vnni128(const std::uint8_t* padded, int blocks, int rows, const std::uint8_t* src, std::size_t stride, std::uint64_t* out)
{
	const std::size_t row = std::size_t(16) * blocks;
	__m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
	std::uint64_t total[8] = {};
	int pending = 0;
	for (int y = 0; y < rows; ++y) {
		const auto* t = padded + y * row;
		const auto* p = src + y * stride;
		for (std::size_t b = 0; b < row; b += 16) {
			const __m128i tb = _mm_loadu_si128((const __m128i*)(t + b));
			const auto* q = p + b;
#define W(k, acc) acc = _mm_dpbuud_epi32(acc, _mm_loadu_si128((const __m128i*)(q + k)), tb)
			W(0, a0); W(1, a1); W(2, a2); W(3, a3); W(4, a4); W(5, a5); W(6, a6); W(7, a7);
#undef W
		}
		if ((pending += blocks) >= 256) {
			total[0] += hsum128(a0); total[1] += hsum128(a1); total[2] += hsum128(a2); total[3] += hsum128(a3);
			total[4] += hsum128(a4); total[5] += hsum128(a5); total[6] += hsum128(a6); total[7] += hsum128(a7);
			a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = _mm_setzero_si128();
			pending = 0;
		}
	}
	out[0] = total[0] + hsum128(a0); out[1] = total[1] + hsum128(a1); out[2] = total[2] + hsum128(a2); out[3] = total[3] + hsum128(a3);
	out[4] = total[4] + hsum128(a4); out[5] = total[5] + hsum128(a5); out[6] = total[6] + hsum128(a6); out[7] = total[7] + hsum128(a7);
}

// AVX-VNNI-INT8 on 32-byte pairs of blocks (ymm), with an xmm step for an odd last block.
T_VNNI8 static void dot8Vnni256(const std::uint8_t* padded, int blocks, int rows, const std::uint8_t* src, std::size_t stride, std::uint64_t* out)
{
	const std::size_t row = std::size_t(16) * blocks, pairs = row / 32 * 32;
	__m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
	__m128i b0 = _mm_setzero_si128(), b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0, b6 = b0, b7 = b0;
	std::uint64_t total[8] = {};
	int pending = 0;
	for (int y = 0; y < rows; ++y) {
		const auto* t = padded + y * row;
		const auto* p = src + y * stride;
		std::size_t b = 0;
		for (; b < pairs; b += 32) {
			const __m256i tb = _mm256_loadu_si256((const __m256i*)(t + b));
			const auto* q = p + b;
#define W(k, acc) acc = _mm256_dpbuud_epi32(acc, _mm256_loadu_si256((const __m256i*)(q + k)), tb)
			W(0, a0); W(1, a1); W(2, a2); W(3, a3); W(4, a4); W(5, a5); W(6, a6); W(7, a7);
#undef W
		}
		if (b < row) {
			const __m128i tb = _mm_loadu_si128((const __m128i*)(t + b));
			const auto* q = p + b;
#define W(k, acc) acc = _mm_dpbuud_epi32(acc, _mm_loadu_si128((const __m128i*)(q + k)), tb)
			W(0, b0); W(1, b1); W(2, b2); W(3, b3); W(4, b4); W(5, b5); W(6, b6); W(7, b7);
#undef W
		}
		if ((pending += blocks) >= 256) {
			total[0] += hsum256(a0) + hsum128(b0); total[1] += hsum256(a1) + hsum128(b1);
			total[2] += hsum256(a2) + hsum128(b2); total[3] += hsum256(a3) + hsum128(b3);
			total[4] += hsum256(a4) + hsum128(b4); total[5] += hsum256(a5) + hsum128(b5);
			total[6] += hsum256(a6) + hsum128(b6); total[7] += hsum256(a7) + hsum128(b7);
			a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = _mm256_setzero_si256();
			b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = _mm_setzero_si128();
			pending = 0;
		}
	}
	out[0] = total[0] + hsum256(a0) + hsum128(b0); out[1] = total[1] + hsum256(a1) + hsum128(b1);
	out[2] = total[2] + hsum256(a2) + hsum128(b2); out[3] = total[3] + hsum256(a3) + hsum128(b3);
	out[4] = total[4] + hsum256(a4) + hsum128(b4); out[5] = total[5] + hsum256(a5) + hsum128(b5);
	out[6] = total[6] + hsum256(a6) + hsum128(b6); out[7] = total[7] + hsum256(a7) + hsum128(b7);
	_mm256_zeroupper();
}

// Per-window SSE2 row dot product as the current core does it (reference for timing only).
static std::uint64_t rowDotSse2(const std::uint8_t* a, const std::uint8_t* b, int size)
{
	std::uint64_t total = 0;
	int x = 0;
	const __m128i zero = _mm_setzero_si128();
	__m128i sum = zero;
	for (; x + 16 <= size; x += 16) {
		const auto va = _mm_loadu_si128((const __m128i*)(a + x)), vb = _mm_loadu_si128((const __m128i*)(b + x));
		sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpacklo_epi8(va, zero), _mm_unpacklo_epi8(vb, zero)));
		sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpackhi_epi8(va, zero), _mm_unpackhi_epi8(vb, zero)));
	}
	total = hsum128(sum);
	for (; x < size; ++x) total += unsigned(a[x]) * b[x];
	return total;
}

template <class F> static double medianMs(F&& f, int reps)
{
	std::vector<double> t;
	for (int r = 0; r < reps; ++r) {
		const auto a = std::chrono::steady_clock::now();
		f();
		t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
	}
	std::sort(t.begin(), t.end());
	return t[t.size() / 2];
}

int main()
{
	detect();
	std::printf("{\"avx2\":%d,\"vnni_int8\":%d,\n", hasAvx2, hasVnni8);
	std::mt19937 rng(20261031);
	// Warp: 1100x800 source (stride +13), 768x527 output, 41 angles incl. 0, +-90, 180, 45, 1e-9, -37.25.
	Img src{1100, 800, 1113, {}};
	src.px.resize(src.stride * src.h);
	for (auto& v : src.px) v = std::uint8_t(rng());
	std::vector<double> angles = {0, 90, -90, 180, 45, 1e-9, -37.25};
	std::uniform_real_distribution<double> ang(-180, 180);
	while (angles.size() < 41) angles.push_back(ang(rng));
	const int W = 768, H = 527;
	struct V { const char* name; RowFn fn; bool ok; };
	std::vector<V> variants = {{"scalar", rowScalar, true}, {"sse2_2px", rowSse2, true},
		{"avx2_4px_loads", rowAvx2<false>, hasAvx2}, {"avx2_4px_gather", rowAvx2<true>, hasAvx2}};
	std::vector<std::uint8_t> ref(std::size_t(W) * H), got(ref.size());
	std::printf("\"warp\":[");
	for (std::size_t v = 0; v < variants.size(); ++v) {
		if (!variants[v].ok) continue;
		std::size_t mismatches = 0;
		for (std::size_t i = 0; i < angles.size(); ++i) {
			const double ax = 150 + 0.37 * i, ay = 120 - 0.21 * i;
			for (std::uint8_t border : {std::uint8_t(0), std::uint8_t(255)}) {
				warp(rowScalar, src, ref, W, H, ax, ay, angles[i], border);
				warp(variants[v].fn, src, got, W, H, ax, ay, angles[i], border);
				for (std::size_t k = 0; k < ref.size(); ++k) mismatches += ref[k] != got[k];
			}
			// Far outside anchors exercise the border-only rows.
			warp(rowScalar, src, ref, W, H, -5000, 9000, angles[i], 0);
			warp(variants[v].fn, src, got, W, H, -5000, 9000, angles[i], 0);
			for (std::size_t k = 0; k < ref.size(); ++k) mismatches += ref[k] != got[k];
		}
		const double ms = medianMs([&] { for (int i = 0; i < 8; ++i) warp(variants[v].fn, src, got, W, H, 160, 110, angles[8 + i], 0); }, 9) / 8;
		std::printf("%s{\"variant\":\"%s\",\"mismatched_bytes\":%zu,\"ms_per_768x527\":%.4f}", v ? "," : "", variants[v].name, mismatches, ms);
	}
	{
		// Fresh zero-initialised output per call, as Image(width, height) does in the core.
		volatile unsigned sink = 0;
		const double alloc = medianMs([&] { for (int i = 0; i < 64; ++i) { std::vector<std::uint8_t> fresh(std::size_t(W) * H); sink = sink + fresh[i * 97]; } }, 9) / 64;
		const double allocWarp = medianMs([&] { for (int i = 0; i < 8; ++i) { std::vector<std::uint8_t> fresh(std::size_t(W) * H); warp(rowScalar, src, fresh, W, H, 160, 110, angles[8 + i], 0); sink = sink + fresh[i]; } }, 9) / 8;
		std::printf(",{\"variant\":\"alloc_zero_only\",\"ms_per_768x527\":%.4f},{\"variant\":\"scalar_with_fresh_alloc\",\"ms_per_768x527\":%.4f}", alloc, allocWarp);
	}
	std::printf("],\n\"dot8\":[");
	// Dot: (a) level-0 ROI shape 762x521 template, 7x7 windows; (b) top-level 48x33 template over 1000x800 canvas.
	struct Shape { const char* name; int tw, th, sw, sh; };
	const Shape shapes[] = {{"roi_762x521", 762, 521, 768, 527}, {"top_48x33", 48, 33, 1000, 800}, {"mid_190x130", 190, 130, 196, 136}};
	bool first = true;
	for (const auto& sh : shapes) {
		std::vector<std::uint8_t> t(std::size_t(sh.tw) * sh.th);
		for (auto& v : t) v = std::uint8_t(rng());
		const int blocks = (sh.tw + 15) / 16;
		const std::size_t row = std::size_t(16) * blocks;
		std::vector<std::uint8_t> padded(row * sh.th, 0);
		for (int y = 0; y < sh.th; ++y) std::copy_n(t.data() + std::size_t(y) * sh.tw, sh.tw, padded.data() + y * row);
		std::vector<std::uint8_t> s(std::size_t(sh.sw) * sh.sh + row + 8, 0);
		for (std::size_t k = 0; k < std::size_t(sh.sw) * sh.sh; ++k) s[k] = std::uint8_t(rng());
		// Worst-case magnitudes on a second pair: all 255.
		std::vector<std::uint8_t> padded255(padded), s255(s);
		for (int y = 0; y < sh.th; ++y) std::fill_n(padded255.data() + y * row, sh.tw, std::uint8_t(255));
		std::fill_n(s255.data(), std::size_t(sh.sw) * sh.sh, std::uint8_t(255));
		const int ow = sh.sw - sh.tw + 1, oh = sh.sh - sh.th + 1;
		using DotFn = void (*)(const std::uint8_t*, int, int, const std::uint8_t*, std::size_t, std::uint64_t*);
		struct D { const char* name; DotFn fn; bool ok; };
		const D dots[] = {{"sse2", dot8Sse2, true}, {"avx2", dot8Avx2, hasAvx2}, {"vnni8_128", dot8Vnni128, hasVnni8}, {"vnni8_256", dot8Vnni256, hasVnni8}};
		// Exactness: every group start of the first two output rows plus random ones, both data sets.
		for (const auto& d : dots) {
			if (!d.ok) continue;
			std::size_t bad = 0;
			for (int pass = 0; pass < 2; ++pass) {
				const auto& pp = pass ? padded255 : padded;
				const auto& ss = pass ? s255 : s;
				const auto& tt = pass ? std::vector<std::uint8_t>(t.size(), 255) : t;
				for (int y = 0; y < std::min(oh, 2); ++y)
					for (int x = 0; x + 8 <= std::max(ow, 8) && x < ow; x += std::max(1, ow / 5)) {
						std::uint64_t r[8], g[8];
						dotScalar(tt.data(), sh.tw, sh.th, ss.data() + std::size_t(y) * sh.sw + x, sh.sw, r);
						d.fn(pp.data(), blocks, sh.th, ss.data() + std::size_t(y) * sh.sw + x, sh.sw, g);
						for (int k = 0; k < 8; ++k) if (x + k < ow) bad += r[k] != g[k];
					}
			}
			// Time a full exhaustive product pass (all windows, 8 per call).
			std::vector<std::uint64_t> prod(std::size_t(ow) + 8);
			const double ms = medianMs([&] {
				for (int y = 0; y < oh; ++y)
					for (int x = 0; x < ow; x += 8) d.fn(padded.data(), blocks, sh.th, s.data() + std::size_t(y) * sh.sw + x, sh.sw, prod.data() + x);
			}, 5);
			std::printf("%s{\"shape\":\"%s\",\"variant\":\"%s\",\"mismatches\":%zu,\"ms_all_windows\":%.3f}", first ? "" : ",", sh.name, d.name, bad, ms);
			first = false;
		}
		// Current core on x86: per window, per row SSE2 dot products.
		volatile std::uint64_t sink = 0;
		const double ms = medianMs([&] {
			for (int y = 0; y < oh; ++y)
				for (int x = 0; x < ow; ++x) {
					std::uint64_t total = 0;
					for (int ty = 0; ty < sh.th; ++ty) total += rowDotSse2(t.data() + std::size_t(ty) * sh.tw, s.data() + std::size_t(y + ty) * sh.sw + x, sh.tw);
					sink = sink + total;
				}
		}, 5);
		std::printf(",{\"shape\":\"%s\",\"variant\":\"current_per_window_sse2\",\"ms_all_windows\":%.3f}", sh.name, ms);
	}
	std::printf("]}\n");
}
