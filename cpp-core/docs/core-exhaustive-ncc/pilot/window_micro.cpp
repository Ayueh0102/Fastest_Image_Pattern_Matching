// Scratch: whole-window UDOT kernels vs the 65b7dad per-row path (selection only).
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using u8 = std::uint8_t;
using u64 = std::uint64_t;

static inline u64 rowDot(const u8* a, const u8* b, int size)  // 65b7dad Backend::Auto
{
	u64 total = 0;
	int x = 0;
	while (size - x >= 16) {
		const int end = x + std::min(4096, (size - x) / 16 * 16);
		uint32x4_t sum = vdupq_n_u32(0);
		for (; x < end; x += 16) sum = vdotq_u32(sum, vld1q_u8(a + x), vld1q_u8(b + x));
		total += vaddvq_u32(sum);
	}
	if (x >= 16 && x < size) {
		static const u8 tail[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
		const auto va = vandq_u8(vld1q_u8(a + size - 16), vld1q_u8(tail + (size - x)));
		total += vaddvq_u32(vdotq_u32(vdupq_n_u32(0), va, vld1q_u8(b + size - 16)));
		x = size;
	}
	for (; x < size; ++x) total += unsigned(a[x]) * unsigned(b[x]);
	return total;
}

// Window kernels over a zero-padded template (rows of 16*blocks bytes).
template <int R>
static inline u64 windowDot(const u8* padded, int blocks, int rows, const u8* src, std::size_t stride)
{
	u64 total = 0;
	uint32x4_t s[R];
	for (int k = 0; k < R; ++k) s[k] = vdupq_n_u32(0);
	int pending = 0, ty = 0;
	const std::size_t rowBytes = std::size_t(16) * blocks;
	for (; ty + R <= rows; ty += R) {
		for (int b = 0; b < blocks; ++b)
			for (int k = 0; k < R; ++k)
				s[k] = vdotq_u32(s[k], vld1q_u8(padded + (ty + k) * rowBytes + 16 * b), vld1q_u8(src + (ty + k) * stride + 16 * b));
		if ((pending += R * blocks) >= 256) {
			for (int k = 1; k < R; ++k) s[0] = vaddq_u32(s[0], s[k]);
			total += vaddvq_u32(s[0]);
			for (int k = 0; k < R; ++k) s[k] = vdupq_n_u32(0);
			pending = 0;
		}
	}
	for (; ty < rows; ++ty)
		for (int b = 0; b < blocks; ++b)
			s[0] = vdotq_u32(s[0], vld1q_u8(padded + ty * rowBytes + 16 * b), vld1q_u8(src + ty * stride + 16 * b));
	for (int k = 1; k < R; ++k) s[0] = vaddq_u32(s[0], s[k]);
	return total + vaddvq_u32(s[0]);
}


// Eight horizontally adjacent windows share each template block load.
static inline void windowDot8(const u8* padded, int blocks, int rows, const u8* src, std::size_t stride, u64* out)
{
	const std::size_t row = std::size_t(16) * blocks;
	uint32x4_t a0 = vdupq_n_u32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
	u64 t[8] = {};
	int pending = 0;
	for (int y = 0; y < rows; ++y) {
		const u8* tp = padded + y * row;
		const u8* p = src + y * stride;
		for (std::size_t b = 0; b < row; b += 16) {
			const uint8x16_t tb = vld1q_u8(tp + b);
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
			t[0] += vaddvq_u32(a0); t[1] += vaddvq_u32(a1); t[2] += vaddvq_u32(a2); t[3] += vaddvq_u32(a3);
			t[4] += vaddvq_u32(a4); t[5] += vaddvq_u32(a5); t[6] += vaddvq_u32(a6); t[7] += vaddvq_u32(a7);
			a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = vdupq_n_u32(0);
			pending = 0;
		}
	}
	out[0] = t[0] + vaddvq_u32(a0); out[1] = t[1] + vaddvq_u32(a1); out[2] = t[2] + vaddvq_u32(a2); out[3] = t[3] + vaddvq_u32(a3);
	out[4] = t[4] + vaddvq_u32(a4); out[5] = t[5] + vaddvq_u32(a5); out[6] = t[6] + vaddvq_u32(a6); out[7] = t[7] + vaddvq_u32(a7);
}

struct Case { const char* name; int tw, th, sw, sh; };

int main()
{
	std::mt19937 rng(5);
	const Case cases[] = {{"Src5 top 10x10", 10, 10, 160, 120}, {"Src10 top 14x14", 14, 14, 400, 300},
		{"Src1 top 30x9", 30, 9, 200, 150}, {"Src4 33x17", 33, 17, 260, 260},
		{"Src7 fine 143x98", 143, 98, 149, 104}, {"Src6 fine 202x106", 202, 106, 208, 112},
		{"Src1 L0 466x135", 466, 135, 472, 141}, {"Src6 L0 848x446", 848, 446, 854, 452}, {"Src7 L0 762x521", 762, 521, 768, 527}};
	double sink = 0;
	for (const auto& c : cases) {
		std::vector<u8> t(std::size_t(c.tw) * c.th), s(std::size_t(c.sw) * c.sh + 64 + 16 * ((c.tw + 15) / 16));
		for (auto& v : t) v = u8(rng());
		for (auto& v : s) v = u8(rng());
		const int blocks = (c.tw + 15) / 16;
		std::vector<u8> padded(std::size_t(16) * blocks * c.th, 0);
		for (int y = 0; y < c.th; ++y) std::copy_n(t.data() + std::size_t(y) * c.tw, c.tw, padded.data() + std::size_t(y) * 16 * blocks);
		const int ww = c.sw - c.tw + 1, wh = c.sh - c.th + 1;
		auto run = [&](int mode) {
			double acc = 0;
			for (int y = 0; y < wh; ++y)
				for (int x = 0; x < ww; ++x) {
					const u8* src = s.data() + std::size_t(y) * c.sw + x;
					if (mode == 4) continue;
					if (mode == 0) {
						double product = 0;
						for (int ty = 0; ty < c.th; ++ty) product += double(rowDot(t.data() + std::size_t(ty) * c.tw, src + std::size_t(ty) * c.sw, c.tw));
						acc += product;
					} else if (mode == 1) acc += double(windowDot<1>(padded.data(), blocks, c.th, src, c.sw));
					else if (mode == 2) acc += double(windowDot<2>(padded.data(), blocks, c.th, src, c.sw));
					else if (mode == 3) acc += double(windowDot<4>(padded.data(), blocks, c.th, src, c.sw));
				}
			if (mode == 4) {
				acc = 0;
				std::vector<u64> prod(ww + 8);
				for (int y = 0; y < wh; ++y) {
					for (int x = 0; x < ww; x += 8) windowDot8(padded.data(), blocks, c.th, s.data() + std::size_t(y) * c.sw + x, c.sw, prod.data() + x);
					for (int x = 0; x < ww; ++x) acc += double(prod[x]);
				}
			}
			return acc;
		};
		const double ref = run(0);
		for (int m = 1; m < 5; ++m) if (run(m) != ref) { std::printf("MISMATCH %s mode %d\n", c.name, m); return 1; }
		double best[5] = {1e30, 1e30, 1e30, 1e30, 1e30};
		for (int rep = 0; rep < 5; ++rep)
			for (int m = 0; m < 5; ++m) {
				const auto t0 = std::chrono::steady_clock::now();
				sink += run(m);
				best[m] = std::min(best[m], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
			}
		const double windows = double(ww) * wh;
		std::printf("%-20s windows %8.0f  ns/window: per-row %8.1f  win1 %8.1f  win2 %8.1f  win4 %8.1f  group8 %8.1f\n", c.name, windows,
			1e6 * best[0] / windows, 1e6 * best[1] / windows, 1e6 * best[2] / windows, 1e6 * best[3] / windows, 1e6 * best[4] / windows);
	}
	std::printf("sink %.0f\n", sink);
}
