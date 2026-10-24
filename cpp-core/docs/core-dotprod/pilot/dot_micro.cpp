// Scratch microbenchmark: candidate exact u8 dot-product kernels (not committed).
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using u8 = std::uint8_t;
using u64 = std::uint64_t;

__attribute__((noinline)) u64 scalarDot(const u8* a, const u8* b, int n)
{
	u64 t = 0;
	for (int x = 0; x < n; ++x) t += unsigned(a[x]) * unsigned(b[x]);
	return t;
}

__attribute__((noinline)) u64 neonDot(const u8* a, const u8* b, int size)
{
	u64 total = 0;
	int x = 0;
	while (size - x >= 16) {
		const int end = x + std::min(4096, (size - x) / 16 * 16);
		uint32x4_t sum = vdupq_n_u32(0);
		for (; x < end; x += 16) {
			const auto va = vld1q_u8(a + x), vb = vld1q_u8(b + x);
			sum = vaddq_u32(sum, vpaddlq_u16(vmull_u8(vget_low_u8(va), vget_low_u8(vb))));
			sum = vaddq_u32(sum, vpaddlq_u16(vmull_u8(vget_high_u8(va), vget_high_u8(vb))));
		}
		std::uint32_t lanes[4];
		vst1q_u32(lanes, sum);
		for (auto lane : lanes) total += lane;
	}
	for (; x < size; ++x) total += unsigned(a[x]) * unsigned(b[x]);
	return total;
}

template <int ACC, bool MASK>
__attribute__((noinline)) u64 udotDot(const u8* a, const u8* b, int size)
{
	static const u8 tailMask[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
	u64 total = 0;
	int x = 0;
	while (size - x >= 16) {
		const int end = x + std::min(4096, (size - x) / 16 * 16);
		uint32x4_t s[ACC];
		for (int k = 0; k < ACC; ++k) s[k] = vdupq_n_u32(0);
		for (; x + 16 * ACC <= end; x += 16 * ACC)
			for (int k = 0; k < ACC; ++k) s[k] = vdotq_u32(s[k], vld1q_u8(a + x + 16 * k), vld1q_u8(b + x + 16 * k));
		for (; x < end; x += 16) s[0] = vdotq_u32(s[0], vld1q_u8(a + x), vld1q_u8(b + x));
		for (int k = 1; k < ACC; ++k) s[0] = vaddq_u32(s[0], s[k]);
		total += vaddvq_u32(s[0]);
	}
	if (MASK && x < size && x >= 16) {
		const int r = size - x;
		const auto va = vandq_u8(vld1q_u8(a + size - 16), vld1q_u8(tailMask + r));
		total += vaddvq_u32(vdotq_u32(vdupq_n_u32(0), va, vld1q_u8(b + size - 16)));
		x = size;
	}
	for (; x < size; ++x) total += unsigned(a[x]) * unsigned(b[x]);
	return total;
}

template <class F>
double timeIt(F f, const std::vector<u8>& t, const std::vector<u8>& s, int w, int rows, int stride, int reps, u64& sink)
{
	auto best = 1e30;
	for (int rep = 0; rep < 5; ++rep) {
		const auto t0 = std::chrono::steady_clock::now();
		for (int r = 0; r < reps; ++r)
			for (int x = 0; x < 64; ++x)
				for (int y = 0; y < rows; ++y) sink += f(t.data() + std::size_t(y) * w, s.data() + std::size_t(y) * stride + x, w);
		const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
		best = std::min(best, ns / (double(reps) * 64 * rows));
	}
	return best;
}

int main()
{
	std::mt19937 rng(7);
	const int widths[] = {11, 16, 23, 27, 29, 47, 58, 95, 116, 190, 233, 381, 466, 762, 848};
	u64 sink = 0;
	std::printf("%5s %8s %8s %8s %8s %8s %8s %8s  (ns per row call)\n", "w", "scalar", "neon", "udot1", "udot2", "udot4", "udot2m", "udot4m");
	for (int w : widths) {
		const int rows = 64, stride = w + 80;
		std::vector<u8> t(std::size_t(w) * rows), s(std::size_t(stride) * rows);
		for (auto& v : t) v = u8(rng());
		for (auto& v : s) v = u8(rng());
		// exactness spot-check
		for (int x = 0; x < 64; ++x) {
			const auto ref = scalarDot(t.data(), s.data() + x, w);
			if (neonDot(t.data(), s.data() + x, w) != ref || udotDot<1, false>(t.data(), s.data() + x, w) != ref ||
				udotDot<2, true>(t.data(), s.data() + x, w) != ref || udotDot<4, true>(t.data(), s.data() + x, w) != ref) {
				std::printf("MISMATCH w=%d\n", w); return 1;
			}
		}
		const int reps = std::max(1, 200000 / (w * 64));
		std::printf("%5d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", w,
			timeIt(scalarDot, t, s, w, rows, stride, reps, sink),
			timeIt(neonDot, t, s, w, rows, stride, reps, sink),
			timeIt(udotDot<1, false>, t, s, w, rows, stride, reps, sink),
			timeIt(udotDot<2, false>, t, s, w, rows, stride, reps, sink),
			timeIt(udotDot<4, false>, t, s, w, rows, stride, reps, sink),
			timeIt(udotDot<2, true>, t, s, w, rows, stride, reps, sink),
			timeIt(udotDot<4, true>, t, s, w, rows, stride, reps, sink));
	}
	std::printf("sink %llu\n", (unsigned long long)sink);
}
