// Scratch: exactness + speed of warp/pyrDown candidates vs the 51d95b1 originals.
#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr double PI = 3.14159265358979323846;
constexpr double D2R = PI / 180.0;
struct View { const std::uint8_t* data; int width, height; std::size_t stride; };
struct Img {
	int w = 0, h = 0; std::vector<std::uint8_t> px;
	Img(int w_, int h_) : w(w_), h(h_), px(std::size_t(w_) * h_) {}
	std::uint8_t* row(int y) { return px.data() + std::size_t(y) * w; }
	View view() const { return {px.data(), w, h, std::size_t(w)}; }
};

static std::uint8_t pixel(View src, int x, int y, std::uint8_t border)
{
	return x < 0 || y < 0 || x >= src.width || y >= src.height ? border :
		src.data[std::size_t(y) * src.stride + x];
}

__attribute__((noinline)) Img warpOld(View src, int width, int height, double ax, double ay, double angle, std::uint8_t border)
{
	Img result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			const double sx = ax + c * x - s * y;
			const double sy = ay + s * x + c * y;
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

static inline std::uint8_t blend(int p00, int p10, int p01, int p11, double dx, double dy)
{
	const double top = p00 * (1 - dx) + p10 * dx;
	const double bottom = p01 * (1 - dx) + p11 * dx;
	return std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
}

// A: same per-pixel expressions; interior pixels skip neighbor border checks.
__attribute__((noinline)) Img warpA(View src, int width, int height, double ax, double ay, double angle, std::uint8_t border)
{
	Img result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	const unsigned innerW = unsigned(src.width - 1), innerH = unsigned(src.height - 1);
	for (int y = 0; y < height; ++y) {
		auto* out = result.row(y);
		for (int x = 0; x < width; ++x) {
			const double sx = ax + c * x - s * y;
			const double sy = ay + s * x + c * y;
			if (sx < -1 || sy < -1 || sx >= src.width || sy >= src.height) { out[x] = border; continue; }
			const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
			const double dx = sx - ix, dy = sy - iy;
			if (unsigned(ix) < innerW && unsigned(iy) < innerH) {
				const auto* p = src.data + std::size_t(iy) * src.stride + ix;
				out[x] = blend(p[0], p[1], p[src.stride], p[src.stride + 1], dx, dy);
			} else
				out[x] = blend(pixel(src, ix, iy, border), pixel(src, ix + 1, iy, border),
					pixel(src, ix, iy + 1, border), pixel(src, ix + 1, iy + 1, border), dx, dy);
		}
	}
	return result;
}

// B: two pixels per float64x2 step, mirroring every fused/rounded operation.
__attribute__((noinline)) Img warpB(View src, int width, int height, double ax, double ay, double angle, std::uint8_t border)
{
	Img result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	const unsigned innerW = unsigned(src.width - 1), innerH = unsigned(src.height - 1);
	const float64x2_t vc = vdupq_n_f64(c), vs = vdupq_n_f64(s), vax = vdupq_n_f64(ax), vay = vdupq_n_f64(ay);
	const float64x2_t one = vdupq_n_f64(1), half = vdupq_n_f64(0.5), lo = vdupq_n_f64(0), hi = vdupq_n_f64(255);
	const float64x2_t minusOne = vdupq_n_f64(-1), sw = vdupq_n_f64(src.width), sh = vdupq_n_f64(src.height);
	for (int y = 0; y < height; ++y) {
		auto* out = result.row(y);
		const float64x2_t vy = vdupq_n_f64(y);
		int x = 0;
		for (; x + 2 <= width; x += 2) {
			const double xs[2] = {double(x), double(x + 1)};
			const float64x2_t vx = vld1q_f64(xs);
			// sx = (ax + c*x) - s*y and sy = (ay + s*x) + c*y, both fused like the scalar code.
			const float64x2_t sx = vfmsq_f64(vfmaq_f64(vax, vc, vx), vs, vy);
			const float64x2_t sy = vfmaq_f64(vfmaq_f64(vay, vs, vx), vc, vy);
			const uint64x2_t outside = vorrq_u64(vorrq_u64(vcltq_f64(sx, minusOne), vcltq_f64(sy, minusOne)),
				vorrq_u64(vcgeq_f64(sx, sw), vcgeq_f64(sy, sh)));
			// NaN compares false in both versions, exactly like the scalar test.
			if (vgetq_lane_u64(outside, 0) | vgetq_lane_u64(outside, 1)) {
				for (int k = 0; k < 2; ++k) {
					const double px = vgetq_lane_f64(sx, 0), py = vgetq_lane_f64(sy, 0);
					(void)px; (void)py;
				}
				// fall back to scalar for this pair
				for (int k = x; k < x + 2; ++k) {
					const double qx = ax + c * k - s * y;
					const double qy = ay + s * k + c * y;
					if (qx < -1 || qy < -1 || qx >= src.width || qy >= src.height) { out[k] = border; continue; }
					const int ix = int(std::floor(qx)), iy = int(std::floor(qy));
					const double dx = qx - ix, dy = qy - iy;
					out[k] = blend(pixel(src, ix, iy, border), pixel(src, ix + 1, iy, border),
						pixel(src, ix, iy + 1, border), pixel(src, ix + 1, iy + 1, border), dx, dy);
				}
				continue;
			}
			const float64x2_t fx = vrndmq_f64(sx), fy = vrndmq_f64(sy);
			const int64x2_t ix = vcvtq_s64_f64(fx), iy = vcvtq_s64_f64(fy);
			const std::int64_t ix0 = vgetq_lane_s64(ix, 0), ix1 = vgetq_lane_s64(ix, 1);
			const std::int64_t iy0 = vgetq_lane_s64(iy, 0), iy1 = vgetq_lane_s64(iy, 1);
			const float64x2_t dx = vsubq_f64(sx, fx), dy = vsubq_f64(sy, fy);
			double q[4][2];
			bool inner = true;
			const std::int64_t ixs[2] = {ix0, ix1}, iys[2] = {iy0, iy1};
			for (int k = 0; k < 2; ++k) {
				const int px = int(ixs[k]), py = int(iys[k]);
				if (unsigned(px) < innerW && unsigned(py) < innerH) {
					const auto* p = src.data + std::size_t(py) * src.stride + px;
					q[0][k] = p[0]; q[1][k] = p[1]; q[2][k] = p[src.stride]; q[3][k] = p[src.stride + 1];
				} else {
					q[0][k] = pixel(src, px, py, border); q[1][k] = pixel(src, px + 1, py, border);
					q[2][k] = pixel(src, px, py + 1, border); q[3][k] = pixel(src, px + 1, py + 1, border);
				}
			}
			(void)inner;
			const float64x2_t p00 = vld1q_f64(q[0]), p10 = vld1q_f64(q[1]), p01 = vld1q_f64(q[2]), p11 = vld1q_f64(q[3]);
			const float64x2_t omdx = vsubq_f64(one, dx), omdy = vsubq_f64(one, dy);
			const float64x2_t top = vfmaq_f64(vmulq_f64(p10, dx), p00, omdx);
			const float64x2_t bottom = vfmaq_f64(vmulq_f64(p11, dx), p01, omdx);
			const float64x2_t v = vaddq_f64(vfmaq_f64(vmulq_f64(bottom, dy), top, omdy), half);
			const float64x2_t r = vminq_f64(vmaxq_f64(vrndmq_f64(v), lo), hi);
			const int64x2_t ri = vcvtq_s64_f64(r);
			out[x] = std::uint8_t(vgetq_lane_s64(ri, 0));
			out[x + 1] = std::uint8_t(vgetq_lane_s64(ri, 1));
		}
		for (; x < width; ++x) {
			const double sx = ax + c * x - s * y;
			const double sy = ay + s * x + c * y;
			if (sx < -1 || sy < -1 || sx >= src.width || sy >= src.height) { out[x] = border; continue; }
			const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
			const double dx = sx - ix, dy = sy - iy;
			out[x] = blend(pixel(src, ix, iy, border), pixel(src, ix + 1, iy, border),
				pixel(src, ix, iy + 1, border), pixel(src, ix + 1, iy + 1, border), dx, dy);
		}
	}
	return result;
}


// C: vector interior test; integer gathers of adjacent byte pairs.
static inline void warpScalarPixel(View src, std::uint8_t* out, int x, int y, double ax, double ay, double c, double s, std::uint8_t border)
{
	const double sx = ax + c * x - s * y;
	const double sy = ay + s * x + c * y;
	if (sx < -1 || sy < -1 || sx >= src.width || sy >= src.height) { out[x] = border; return; }
	const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
	const double dx = sx - ix, dy = sy - iy;
	out[x] = blend(pixel(src, ix, iy, border), pixel(src, ix + 1, iy, border),
		pixel(src, ix, iy + 1, border), pixel(src, ix + 1, iy + 1, border), dx, dy);
}
static inline std::uint64_t pair(const std::uint8_t* p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
__attribute__((noinline)) Img warpC(View src, int width, int height, double ax, double ay, double angle, std::uint8_t border)
{
	Img result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	const float64x2_t vc = vdupq_n_f64(c), vs = vdupq_n_f64(s), vax = vdupq_n_f64(ax), vay = vdupq_n_f64(ay);
	const float64x2_t one = vdupq_n_f64(1), half = vdupq_n_f64(0.5), lo = vdupq_n_f64(0), hi = vdupq_n_f64(255);
	const float64x2_t zero = vdupq_n_f64(0), innerW = vdupq_n_f64(src.width - 1), innerH = vdupq_n_f64(src.height - 1);
	const float64x2_t vstride = vdupq_n_f64(double(src.stride)), step = vdupq_n_f64(2);
	const uint64x2_t byte = vdupq_n_u64(255);
	const double first[2] = {0, 1};
	for (int y = 0; y < height; ++y) {
		auto* out = result.row(y);
		const float64x2_t vy = vdupq_n_f64(y);
		float64x2_t vx = vld1q_f64(first);
		int x = 0;
		for (; x + 2 <= width; x += 2, vx = vaddq_f64(vx, step)) {
			const float64x2_t sx = vfmsq_f64(vfmaq_f64(vax, vc, vx), vs, vy);
			const float64x2_t sy = vfmaq_f64(vfmaq_f64(vay, vs, vx), vc, vy);
			// 0 <= sx < width-1 and 0 <= sy < height-1 <=> all four neighbors exist.
			const uint64x2_t inside = vandq_u64(vandq_u64(vcgeq_f64(sx, zero), vcltq_f64(sx, innerW)),
				vandq_u64(vcgeq_f64(sy, zero), vcltq_f64(sy, innerH)));
			if (!(vgetq_lane_u64(inside, 0) & vgetq_lane_u64(inside, 1))) {
				warpScalarPixel(src, out, x, y, ax, ay, c, s, border);
				warpScalarPixel(src, out, x + 1, y, ax, ay, c, s, border);
				continue;
			}
			const float64x2_t fx = vrndmq_f64(sx), fy = vrndmq_f64(sy);
			const float64x2_t dx = vsubq_f64(sx, fx), dy = vsubq_f64(sy, fy);
			const int64x2_t offset = vcvtq_s64_f64(vfmaq_f64(fx, fy, vstride));
			const auto* p0 = src.data + vgetq_lane_s64(offset, 0);
			const auto* p1 = src.data + vgetq_lane_s64(offset, 1);
			const uint64x2_t top = vcombine_u64(vcreate_u64(pair(p0)), vcreate_u64(pair(p1)));
			const uint64x2_t bot = vcombine_u64(vcreate_u64(pair(p0 + src.stride)), vcreate_u64(pair(p1 + src.stride)));
			const float64x2_t p00 = vcvtq_f64_u64(vandq_u64(top, byte)), p10 = vcvtq_f64_u64(vshrq_n_u64(top, 8));
			const float64x2_t p01 = vcvtq_f64_u64(vandq_u64(bot, byte)), p11 = vcvtq_f64_u64(vshrq_n_u64(bot, 8));
			const float64x2_t omdx = vsubq_f64(one, dx), omdy = vsubq_f64(one, dy);
			const float64x2_t t = vfmaq_f64(vmulq_f64(p10, dx), p00, omdx);
			const float64x2_t b = vfmaq_f64(vmulq_f64(p11, dx), p01, omdx);
			const float64x2_t v = vaddq_f64(vfmaq_f64(vmulq_f64(b, dy), t, omdy), half);
			const uint64x2_t r = vcvtq_u64_f64(vminq_f64(vmaxq_f64(vrndmq_f64(v), lo), hi));
			out[x] = std::uint8_t(vgetq_lane_u64(r, 0));
			out[x + 1] = std::uint8_t(vgetq_lane_u64(r, 1));
		}
		for (; x < width; ++x) warpScalarPixel(src, out, x, y, ax, ay, c, s, border);
	}
	return result;
}

__attribute__((noinline)) Img warpD(View src, int width, int height, double ax, double ay, double angle, std::uint8_t border)
{
	Img result(width, height);
	const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
	const float64x2_t vc = vdupq_n_f64(c), vs = vdupq_n_f64(s), vax = vdupq_n_f64(ax), vay = vdupq_n_f64(ay);
	const float64x2_t one = vdupq_n_f64(1), half = vdupq_n_f64(0.5), lo = vdupq_n_f64(0), hi = vdupq_n_f64(255);
	const float64x2_t zero = vdupq_n_f64(0), innerW = vdupq_n_f64(src.width - 1), innerH = vdupq_n_f64(src.height - 1);
	const float64x2_t vstride = vdupq_n_f64(double(src.stride)), step = vdupq_n_f64(2), step4 = vdupq_n_f64(4);
	const uint64x2_t byte = vdupq_n_u64(255);
	const double first[2] = {0, 1};
	for (int y = 0; y < height; ++y) {
		auto* out = result.row(y);
		const float64x2_t vy = vdupq_n_f64(y);
		float64x2_t vx = vld1q_f64(first);
		int x = 0;
		auto two = [&](int x, float64x2_t vx) {
			const float64x2_t sx = vfmsq_f64(vfmaq_f64(vax, vc, vx), vs, vy);
			const float64x2_t sy = vfmaq_f64(vfmaq_f64(vay, vs, vx), vc, vy);
			// 0 <= sx < width-1 and 0 <= sy < height-1 <=> all four neighbors exist.
			const uint64x2_t inside = vandq_u64(vandq_u64(vcgeq_f64(sx, zero), vcltq_f64(sx, innerW)),
				vandq_u64(vcgeq_f64(sy, zero), vcltq_f64(sy, innerH)));
			if (!(vgetq_lane_u64(inside, 0) & vgetq_lane_u64(inside, 1))) {
				warpScalarPixel(src, out, x, y, ax, ay, c, s, border);
				warpScalarPixel(src, out, x + 1, y, ax, ay, c, s, border);
				return;
			}
			const float64x2_t fx = vrndmq_f64(sx), fy = vrndmq_f64(sy);
			const float64x2_t dx = vsubq_f64(sx, fx), dy = vsubq_f64(sy, fy);
			const int64x2_t offset = vcvtq_s64_f64(vfmaq_f64(fx, fy, vstride));
			const auto* p0 = src.data + vgetq_lane_s64(offset, 0);
			const auto* p1 = src.data + vgetq_lane_s64(offset, 1);
			const uint64x2_t top = vcombine_u64(vcreate_u64(pair(p0)), vcreate_u64(pair(p1)));
			const uint64x2_t bot = vcombine_u64(vcreate_u64(pair(p0 + src.stride)), vcreate_u64(pair(p1 + src.stride)));
			const float64x2_t p00 = vcvtq_f64_u64(vandq_u64(top, byte)), p10 = vcvtq_f64_u64(vshrq_n_u64(top, 8));
			const float64x2_t p01 = vcvtq_f64_u64(vandq_u64(bot, byte)), p11 = vcvtq_f64_u64(vshrq_n_u64(bot, 8));
			const float64x2_t omdx = vsubq_f64(one, dx), omdy = vsubq_f64(one, dy);
			const float64x2_t t = vfmaq_f64(vmulq_f64(p10, dx), p00, omdx);
			const float64x2_t b = vfmaq_f64(vmulq_f64(p11, dx), p01, omdx);
			const float64x2_t v = vaddq_f64(vfmaq_f64(vmulq_f64(b, dy), t, omdy), half);
			const uint64x2_t r = vcvtq_u64_f64(vminq_f64(vmaxq_f64(vrndmq_f64(v), lo), hi));
			out[x] = std::uint8_t(vgetq_lane_u64(r, 0));
			out[x + 1] = std::uint8_t(vgetq_lane_u64(r, 1));
				};
		for (; x + 4 <= width; x += 4, vx = vaddq_f64(vx, step4)) { two(x, vx); two(x + 2, vaddq_f64(vx, step)); }
		for (; x + 2 <= width; x += 2, vx = vaddq_f64(vx, step)) two(x, vx);
		for (; x < width; ++x) warpScalarPixel(src, out, x, y, ax, ay, c, s, border);
	}
	return result;
}

static int reflect101(int p, int length)
{
	if (length == 1) return 0;
	while (p < 0 || p >= length) p = p < 0 ? -p : 2 * length - p - 2;
	return p;
}

__attribute__((noinline)) Img pyrOld(View image)
{
	Img result(image.width / 2 + image.width % 2, image.height / 2 + image.height % 2);
	constexpr int kernel[] = {1, 4, 6, 4, 1};
	std::vector<int> horizontal(std::size_t(image.height) * result.w);
	for (int y = 0; y < image.height; ++y) {
		const auto* row = image.data + std::size_t(y) * image.stride;
		for (int x = 0; x < result.w; ++x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k) sum += kernel[k + 2] * row[reflect101(x * 2 + k, image.width)];
			horizontal[std::size_t(y) * result.w + x] = sum;
		}
	}
	for (int y = 0; y < result.h; ++y)
		for (int x = 0; x < result.w; ++x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * horizontal[std::size_t(reflect101(y * 2 + k, image.height)) * result.w + x];
			result.row(y)[x] = std::uint8_t((sum + 128) / 256);
		}
	return result;
}

__attribute__((noinline)) Img pyrNew(View image)
{
	Img result(image.width / 2 + image.width % 2, image.height / 2 + image.height % 2);
	const int width = result.w;
	constexpr int kernel[] = {1, 4, 6, 4, 1};
	std::vector<int> horizontal(std::size_t(image.height) * width);
	// Columns 1..interior-1 read 2x-2..2x+2 inside the row; only the edges reflect.
	const int interior = std::max(1, std::min(width, (image.width - 1) / 2));
	for (int y = 0; y < image.height; ++y) {
		const auto* row = image.data + std::size_t(y) * image.stride;
		auto* out = horizontal.data() + std::size_t(y) * width;
		auto edge = [&](int x) {
			int sum = 0;
			for (int k = -2; k <= 2; ++k) sum += kernel[k + 2] * row[reflect101(x * 2 + k, image.width)];
			return sum;
		};
		out[0] = edge(0);
		for (int x = 1; x < interior; ++x) {
			const auto* p = row + 2 * x;
			out[x] = p[-2] + 4 * p[-1] + 6 * p[0] + 4 * p[1] + p[2];
		}
		for (int x = interior; x < width; ++x) out[x] = edge(x);
	}
	for (int y = 0; y < result.h; ++y) {
		const int* r[5];
		for (int k = 0; k < 5; ++k) r[k] = horizontal.data() + std::size_t(reflect101(y * 2 + k - 2, image.height)) * width;
		auto* out = result.row(y);
		for (int x = 0; x < width; ++x)
			out[x] = std::uint8_t((r[0][x] + 4 * r[1][x] + 6 * r[2][x] + 4 * r[3][x] + r[4][x] + 128) / 256);
	}
	return result;
}

template <class F> double best(F f, int reps)
{
	double b = 1e30;
	for (int i = 0; i < reps; ++i) {
		const auto t0 = std::chrono::steady_clock::now();
		f();
		b = std::min(b, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
	}
	return b;
}

int main()
{
	std::mt19937 rng(11);
	// Exactness: random sizes, strides, anchors (inside and far outside) and angles.
	long checked = 0;
	for (int t = 0; t < 4000; ++t) {
		const int w = 1 + rng() % 40, h = 1 + rng() % 40, pad = rng() % 5;
		std::vector<std::uint8_t> buf(std::size_t(w + pad) * h + 1);
		for (auto& v : buf) v = std::uint8_t(rng());
		const View src{buf.data(), w, h, std::size_t(w + pad)};
		const double angles[] = {0, 90, -90, 180, 45, 1e-9, 37.25, double(rng() % 36000) / 100 - 180};
		const double angle = angles[rng() % 8];
		const double ax = double(int(rng() % 120) - 60) / 3, ay = double(int(rng() % 120) - 60) / 7;
		const int ow = 1 + rng() % 50, oh = 1 + rng() % 50;
		const std::uint8_t border = rng() % 2 ? 255 : 0;
		const auto a = warpOld(src, ow, oh, ax, ay, angle, border);
		const auto b = warpA(src, ow, oh, ax, ay, angle, border);
		const auto cc = warpB(src, ow, oh, ax, ay, angle, border);
		const auto dd = warpC(src, ow, oh, ax, ay, angle, border);
		const auto ee = warpD(src, ow, oh, ax, ay, angle, border);
		if (a.px != b.px || a.px != cc.px || a.px != dd.px || a.px != ee.px) { std::printf("WARP MISMATCH t=%d A=%d B=%d\n", t, a.px != b.px, a.px != cc.px); return 1; }
		const auto p = pyrOld(src), q = pyrNew(src);
		if (p.px != q.px) { std::printf("PYR MISMATCH t=%d w=%d h=%d\n", t, w, h); return 1; }
		checked += long(ow) * oh;
	}
	std::printf("exact: %ld warp pixels, 4000 pyr images\n", checked);
	// Speed: Src7-like level-0 ROI (768x527) and a 4024x3036 pyrDown.
	Img big(4024, 3036);
	for (auto& v : big.px) v = std::uint8_t(rng());
	const auto v = big.view();
	double sink = 0;
	for (double angle : {0.0, 13.7, 120.1}) {
		const double ax = 1500, ay = 1000;
		std::printf("warp 768x527 angle %6.1f  old %.3f ms  A %.3f ms  B %.3f ms  C %.3f ms  D %.3f ms\n", angle,
			best([&] { sink += warpOld(v, 768, 527, ax, ay, angle, 0).px[5]; }, 15),
			best([&] { sink += warpA(v, 768, 527, ax, ay, angle, 0).px[5]; }, 15),
			best([&] { sink += warpB(v, 768, 527, ax, ay, angle, 0).px[5]; }, 15),
			best([&] { sink += warpC(v, 768, 527, ax, ay, angle, 0).px[5]; }, 15),
			best([&] { sink += warpD(v, 768, 527, ax, ay, angle, 0).px[5]; }, 15));
		if (warpOld(v, 768, 527, ax, ay, angle, 0).px != warpC(v, 768, 527, ax, ay, angle, 0).px) { std::puts("BIG MISMATCH C"); return 1; }
		if (warpOld(v, 768, 527, ax, ay, angle, 0).px != warpB(v, 768, 527, ax, ay, angle, 0).px) { std::puts("BIG MISMATCH"); return 1; }
	}
	std::printf("canvas 200x180 rotate 37deg old %.3f A %.3f B %.3f C %.3f\n",
		best([&] { sink += warpOld(v, 200, 180, -20, -30, 37, 0).px[5]; }, 50),
		best([&] { sink += warpA(v, 200, 180, -20, -30, 37, 0).px[5]; }, 50),
		best([&] { sink += warpB(v, 200, 180, -20, -30, 37, 0).px[5]; }, 50),
		best([&] { sink += warpC(v, 200, 180, -20, -30, 37, 0).px[5]; }, 50));
	std::printf("pyrDown 4024x3036 old %.3f ms new %.3f ms\n",
		best([&] { sink += pyrOld(v).px[3]; }, 7), best([&] { sink += pyrNew(v).px[3]; }, 7));
	if (pyrOld(v).px != pyrNew(v).px) { std::puts("BIG PYR MISMATCH"); return 1; }
	std::printf("sink %.0f\n", sink);
}
