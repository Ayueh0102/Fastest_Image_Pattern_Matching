// SPDX-License-Identifier: BSD-2-Clause
// Experimental CUDA backend (backend 2) of the coarse/frame research API.
//
// Every score is binary64-identical to the CPU path of the same build. The host
// computes each warp map with fpm_core's canvasWarp()/roiWarp(); the device
// samples it in FP64 with every multiply and add rounded separately, which is
// the x86 core's (unfused) warpPixel order. Window statistics are exact
// integers, and normalization repeats nccDenominator() and
// nccNormalizeProduct() with the same binary64 operations.
//
// Coarse search is one fused launch over all angles: a block warps its canvas
// tile into shared memory, scores its windows and appends only the entries at
// or above the threshold, so neither canvases nor score maps are written to
// device memory. The host then selects peaks with selectPeaksSparse(), which
// equals selectPeaks() on the full map.
//
// Fine search warps ROI row stripes into shared memory and accumulates the 49
// window statistics across stripes. Its pixels use the Metal path's exact Q28
// integer sampling (fine_sampling.hpp); a pixel whose value lies within the
// host's rounding guard, or a request the guard cannot cover, is sampled with
// the FP64 warpPixel() instead, so every ROI byte equals rotatedRoi()'s.
//
// Opt-in transfer modes (fpm_frame_set_transfer_mode) change only how work
// reaches the device and how the host waits: lean prepare reuses a page-locked
// host level 0 and overlaps its chunked copy with the uploads; lean calls drop
// timing events and memsets and poll a completion value in mapped memory that
// the last kernel of each call writes. The kernels' arithmetic is unchanged.
#include "coarse_internal.hpp"
#include "fine_sampling.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using u8 = unsigned char;
using u64 = unsigned long long;
using Clock = std::chrono::steady_clock;
constexpr int PAD = fpm::detail::PADDING, FINE_SIDE = 2 * PAD + 1, FINE_WINDOWS = FINE_SIDE * FINE_SIDE;
constexpr int TILE_X = 32, TILE_Y = 8, COARSE_THREADS = TILE_X * TILE_Y, FINE_THREADS = 256;
// Per-row and per-lane uint32 statistics stay exact while count * 65025 < 2^32.
constexpr u64 MAX_U32_PIXELS = 66051;
// Binary32 scores (screening and the FP32 ablation) use exact uint64/int64 centered
// sums, which hold while the template has at most this many pixels.
constexpr u64 SCORE32_MAX_PIXELS = 1000000;
// Screened windows are dropped only when their binary32 score is below the threshold
// by this margin, far above the binary32 and binary64 score errors (see screenTemplate()).
constexpr double SCREEN_MARGIN = 1e-4;
// Fine NCC mode 4 uses screenRow() from this ROI width (selected on a pilot of widths 0-512).
// Mode 5 also takes more warps per block from there: shared memory keeps two blocks per SM
// whatever a block's size, so its occupancy is 14 warps' 58% against 8 warps' 33%; 16 warps
// fit only one block per SM (selected on a pilot of 8 to 24 warps).
constexpr int FINE_SCREEN_MIN_WIDTH = 256, FINE_WIDE_THREADS = 448;

// NVTX range for Nsight Systems timelines; inert without an attached tool.
struct Range {
    explicit Range(const char* name) { nvtxRangePushA(name); }
    ~Range() { nvtxRangePop(); }
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
};

double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(status));
}

struct DeviceWarp { double ax, ay, c, s; int width, height; };
DeviceWarp deviceWarp(const fpm::detail::WarpMap& m) { return {m.anchor.x, m.anchor.y, m.c, m.s, m.width, m.height}; }
// Q28 fast path (fine_sampling.hpp) plus the binary64 map it must reproduce.
struct FineWarp { long long ax, ay, c, s; u64 guard; int repairAll, reserved; DeviceWarp exact; };
FineWarp fineWarp(const fpm::detail::WarpMap& map)
{
    fpm_coarse_detail::FineWarpPlan plan{map.anchor, map.c, map.s, {}};
    fpm_coarse_detail::describeFineWarp(plan, map.width, map.height);
    const auto& d = plan.descriptor;
    return {d.anchorX, d.anchorY, d.cosine, d.sine, d.roundingGuard, int(d.repairAll), 0, deviceWarp(map)};
}
struct CoarseJob { DeviceWarp warp; int mapWidth, mapHeight; u64 entryOffset; unsigned capacity; };
// Coarse NCC modes 2-4: the canvas map with its Q28 descriptor (q.exact is the map).
struct CoarseQJob { FineWarp q; int mapWidth, mapHeight; u64 entryOffset; unsigned capacity; };
struct Entry { int x, y; double score; };

// ---- device arithmetic -----------------------------------------------------
__device__ __forceinline__ u8 pixelAt(const u8* s, int w, int h, int x, int y, u8 border)
{
    return x < 0 || y < 0 || x >= w || y >= h ? border : s[size_t(y) * w + x];
}

// warpPixel(): sx=(ax+c*x)-s*y, sy=(ay+s*x)+c*y, bilinear, floor(v+0.5), clamp.
__device__ u8 warpPixel(const u8* s, int w, int h, int x, int y, const DeviceWarp& m, u8 border)
{
    const double fx = x, fy = y;
    const double sx = __dsub_rn(__dadd_rn(m.ax, __dmul_rn(m.c, fx)), __dmul_rn(m.s, fy));
    const double sy = __dadd_rn(__dadd_rn(m.ay, __dmul_rn(m.s, fx)), __dmul_rn(m.c, fy));
    if (sx < -1 || sy < -1 || sx >= w || sy >= h) return border;
    const int ix = int(floor(sx)), iy = int(floor(sy));
    const double dx = __dsub_rn(sx, double(ix)), dy = __dsub_rn(sy, double(iy));
    const double rx = __dsub_rn(1.0, dx), ry = __dsub_rn(1.0, dy);
    const double top = __dadd_rn(__dmul_rn(double(pixelAt(s, w, h, ix, iy, border)), rx),
        __dmul_rn(double(pixelAt(s, w, h, ix + 1, iy, border)), dx));
    const double bottom = __dadd_rn(__dmul_rn(double(pixelAt(s, w, h, ix, iy + 1, border)), rx),
        __dmul_rn(double(pixelAt(s, w, h, ix + 1, iy + 1, border)), dx));
    const double v = floor(__dadd_rn(__dadd_rn(__dmul_rn(top, ry), __dmul_rn(bottom, dy)), 0.5));
    return u8(v < 0.0 ? 0.0 : 255.0 < v ? 255.0 : v);
}

// nccDenominator() then nccNormalizeProduct() on exact integer statistics.
__device__ double normalizeScore(u64 sum, u64 square, u64 product, double mean, double norm, double area)
{
    const double s = __ull2double_rn(sum), q = __ull2double_rn(square);
    const double v0 = __dsub_rn(q, __ddiv_rn(__dmul_rn(s, s), area));
    const double variance = 0.0 < v0 ? v0 : 0.0;
    const double bound = __dmul_rn(double(10 * FLT_EPSILON), q);
    const double limit = bound < 0.5 ? bound : 0.5;
    const double denominator = variance > limit ? __dmul_rn(__dsqrt_rn(variance), norm) : 0.0;
    if (denominator == 0) return 0;
    const double numerator = __dsub_rn(__ull2double_rn(product), __dmul_rn(s, mean));
    if (fabs(numerator) < denominator) return __ddiv_rn(numerator, denominator);
    if (fabs(numerator) < __dmul_rn(denominator, 1.125)) return numerator > 0 ? 1 : -1;
    return 0;
}

__device__ __forceinline__ u64 warpSum(u64 v)
{
    for (int offset = 16; offset; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Sum, square and template product of one window whose top-left is p.
__device__ __forceinline__ void windowStats(const u8* p, int stride, const u8* t, int tw, int th,
    u64& sum, u64& square, u64& product)
{
    sum = square = product = 0;
    for (int r = 0; r < th; ++r) {
        unsigned rs = 0, rq = 0, rd = 0;
        const u8* row = p + size_t(r) * stride;
        const u8* tr = t + size_t(r) * tw;
        for (int k = 0; k < tw; ++k) { const unsigned v = row[k]; rs += v; rq += v * v; rd += v * tr[k]; }
        sum += rs; square += rq; product += rd;
    }
}

// Warp-aggregated append of the windows whose scores reach the threshold.
__device__ __forceinline__ void appendEntry(bool keep, int x, int y, double score, unsigned* count, Entry* entries, unsigned capacity)
{
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    if (!mask) return;
    const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
    unsigned base = 0;
    if (lane == leader) base = atomicAdd(count, unsigned(__popc(mask)));
    base = __shfl_sync(0xffffffffu, base, leader);
    const unsigned index = base + __popc(mask & ((1u << lane) - 1));
    if (keep && index < capacity) entries[index] = {x, y, score};
}

// ---- kernels ------------------------------------------------------------------
// pyrDown(): [1,4,6,4,1] on both axes, REFLECT_101, (sum+128)/256, all integer.
__device__ __forceinline__ int reflect101(int p, int n)
{
    if (n == 1) return 0;
    while (p < 0 || p >= n) p = p < 0 ? -p : 2 * n - p - 2;
    return p;
}
__global__ void pyrDownKernel(const u8* in, int w, int h, u8* out, int ow, int oh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    constexpr int k[5] = {1, 4, 6, 4, 1};
    int total = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        const u8* row = in + size_t(reflect101(2 * y + dy, h)) * w;
        int horizontal = 0;
        for (int dx = -2; dx <= 2; ++dx) horizontal += k[dx + 2] * row[reflect101(2 * x + dx, w)];
        total += k[dy + 2] * horizontal;
    }
    out[size_t(y) * ow + x] = u8((total + 128) / 256);
}

// Fused coarse search: one TILE_X x TILE_Y block of windows of one angle.
// Shared memory: template, then the canvas tile the block's windows cover.
__global__ void coarseFused(const u8* source, int sw, int sh, const CoarseJob* jobs, const u8* templ,
    int tw, int th, double mean, double norm, double threshold, int uniform, u8 border,
    Entry* entries, unsigned* counts)
{
    extern __shared__ u8 shared[];
    const CoarseJob job = jobs[blockIdx.z];
    const int x0 = blockIdx.x * TILE_X, y0 = blockIdx.y * TILE_Y;
    if (x0 >= job.mapWidth || y0 >= job.mapHeight) return; // uniform per block
    const int tileW = TILE_X + tw - 1, tileH = TILE_Y + th - 1, tid = threadIdx.y * TILE_X + threadIdx.x;
    u8* t = shared;
    u8* tile = shared + tw * th;
    for (int i = tid; i < tw * th; i += COARSE_THREADS) t[i] = templ[i];
    for (int i = tid; i < tileW * tileH; i += COARSE_THREADS) {
        const int cx = x0 + i % tileW, cy = y0 + i / tileW;
        tile[i] = cx < job.warp.width && cy < job.warp.height ? warpPixel(source, sw, sh, cx, cy, job.warp, border) : 0;
    }
    __syncthreads();
    const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    bool keep = false;
    double score = 0;
    if (x < job.mapWidth && y < job.mapHeight) {
        if (uniform) score = 1;
        else {
            u64 sum, square, product;
            windowStats(tile + threadIdx.y * tileW + threadIdx.x, tileW, t, tw, th, sum, square, product);
            score = normalizeScore(sum, square, product, mean, norm, double(tw) * th);
        }
        keep = !(score < threshold);
    }
    appendEntry(keep, x, y, score, counts + blockIdx.z, entries + job.entryOffset, job.capacity);
}

// Unfused coarse search (ablation and large templates): warp every canvas to
// device memory, then score windows from it with the template in global memory.
__global__ void canvasWarpKernel(const u8* source, int sw, int sh, const CoarseJob* jobs, const u64* canvasOffsets,
    u8* canvases, u8 border)
{
    const CoarseJob job = jobs[blockIdx.z];
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= job.warp.width || y >= job.warp.height) return;
    canvases[canvasOffsets[blockIdx.z] + size_t(y) * job.warp.width + x] = warpPixel(source, sw, sh, x, y, job.warp, border);
}
__global__ void coarseFromCanvas(const u8* canvases, const u64* canvasOffsets, const CoarseJob* jobs, const u8* templ,
    int tw, int th, double mean, double norm, double threshold, int uniform, Entry* entries, unsigned* counts)
{
    const CoarseJob job = jobs[blockIdx.z];
    const int x = blockIdx.x * TILE_X + threadIdx.x, y = blockIdx.y * TILE_Y + threadIdx.y;
    bool keep = false;
    double score = 0;
    if (x < job.mapWidth && y < job.mapHeight) {
        if (uniform) score = 1;
        else {
            u64 sum, square, product;
            windowStats(canvases + canvasOffsets[blockIdx.z] + size_t(y) * job.warp.width + x, job.warp.width,
                templ, tw, th, sum, square, product);
            score = normalizeScore(sum, square, product, mean, norm, double(tw) * th);
        }
        keep = !(score < threshold);
    }
    appendEntry(keep, x, y, score, counts + blockIdx.z, entries + job.entryOffset, job.capacity);
}

// Lean calls: the last block of the grid to finish writes `value` to the
// mapped completion flag the host polls. Every thread fences its own writes at
// system scope before the barrier, so the flag follows all of the grid's
// output writes; `done` is left zero for the next call.
__device__ void signalLast(unsigned* done, unsigned* flag, unsigned value)
{
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(done, 1u) == gridDim.x * gridDim.y * gridDim.z - 1) {
        *done = 0;
        __threadfence_system();
        *static_cast<volatile unsigned*>(flag) = value;
    }
}

// Copies counts and each angle's kept entries to mapped host memory (one block
// per angle); block 0 also copies the two kernel counters that follow the
// counts (FP64-sampled pixels and binary64-scored windows). With a flag (lean
// calls) it also clears the counts for the next search and signals completion.
template <typename Job>
__global__ void packEntries(const Entry* entries, unsigned* counts, const Job* jobs,
    Entry* hostEntries, unsigned* hostCounts, unsigned* done, unsigned* flag, unsigned value)
{
    const unsigned n = counts[blockIdx.x];
    const u64 offset = jobs[blockIdx.x].entryOffset;
    if (threadIdx.x == 0) hostCounts[blockIdx.x] = n;
    const bool counter = blockIdx.x == 0 && threadIdx.x < 2;
    if (counter) hostCounts[gridDim.x + threadIdx.x] = counts[gridDim.x + threadIdx.x];
    for (unsigned i = threadIdx.x; i < min(n, jobs[blockIdx.x].capacity); i += blockDim.x) hostEntries[offset + i] = entries[offset + i];
    if (flag) {
        __syncthreads();
        if (threadIdx.x == 0) counts[blockIdx.x] = 0;
        if (counter) counts[gridDim.x + threadIdx.x] = 0;
        signalLast(done, flag, value);
    }
}

// Dense score map of one image (fpm_coarse_scoremap diagnostics).
__global__ void denseScores(const u8* image, int width, int mapWidth, int mapHeight, const u8* templ,
    int tw, int th, double mean, double norm, int uniform, double* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mapWidth || y >= mapHeight) return;
    double score = 1;
    if (!uniform) {
        u64 sum, square, product;
        windowStats(image + size_t(y) * width + x, width, templ, tw, th, sum, square, product);
        score = normalizeScore(sum, square, product, mean, norm, double(tw) * th);
    }
    out[size_t(y) * mapWidth + x] = score;
}

// fine_guarded_warp(): exact Q28 bilinear numerator over zero borders; within the
// guard of a rounding threshold (or for repairAll requests) use warpPixel().
__device__ __forceinline__ u8 finePixel(const u8* s, int w, int h, int x, int y, const FineWarp& m, unsigned& repaired)
{
    if (!m.repairAll) {
        constexpr long long unit = 1ll << 28;
        const long long sx = m.ax + m.c * x - m.s * y, sy = m.ay + m.s * x + m.c * y;
        u64 numerator = 0;
        if (sx >= -unit && sy >= -unit && sx < (long long)w * unit && sy < (long long)h * unit) {
            const int ix = int(sx >> 28), iy = int(sy >> 28);
            const u64 dx = u64(sx & (unit - 1)), dy = u64(sy & (unit - 1)), one = u64(unit);
            const u64 top = u64(pixelAt(s, w, h, ix, iy, 0)) * (one - dx) + u64(pixelAt(s, w, h, ix + 1, iy, 0)) * dx;
            const u64 bottom = u64(pixelAt(s, w, h, ix, iy + 1, 0)) * (one - dx) + u64(pixelAt(s, w, h, ix + 1, iy + 1, 0)) * dx;
            numerator = top * (one - dy) + bottom * dy;
        }
        constexpr u64 half = u64(1) << 55;
        const u64 remainder = numerator & ((u64(1) << 56) - 1);
        const u64 distance = remainder > half ? remainder - half : half - remainder;
        if (distance > m.guard) return u8((numerator + half) >> 56);
    }
    ++repaired;
    return warpPixel(s, w, h, x, y, m.exact, 0);
}

// The same Q28 rule at given Q28 coordinates (sx, sy), over an image extended by
// the constant `border` (still continuous and 255-Lipschitz, so the guard holds).
// The bilinear numerator top*(1-dy)+bottom*dy is formed as top*2^28+(bottom-top)*dy
// in wrapping uint64 arithmetic: its true value lies in [0, 255*2^56], so the
// wrapped result is exact. (x, y) are the map coordinates for the FP64 fallback.
__device__ __forceinline__ u8 q28Sample(const u8* s, int w, int h, long long sx, long long sy, u64 guard,
    int x, int y, const DeviceWarp& exact, u8 border, unsigned& repaired)
{
    const long long ix = sx >> 28, iy = sy >> 28;
    u64 numerator = u64(border) << 56;
    if (ix >= -1 && iy >= -1 && ix < w && iy < h) {
        u8 p00, p10, p01, p11;
        if (ix >= 0 && iy >= 0 && ix + 1 < w && iy + 1 < h) {
            const u8* p = s + size_t(iy) * w + size_t(ix);
            p00 = p[0]; p10 = p[1]; p01 = p[w]; p11 = p[w + 1];
        } else {
            p00 = pixelAt(s, w, h, int(ix), int(iy), border); p10 = pixelAt(s, w, h, int(ix) + 1, int(iy), border);
            p01 = pixelAt(s, w, h, int(ix), int(iy) + 1, border); p11 = pixelAt(s, w, h, int(ix) + 1, int(iy) + 1, border);
        }
        const long long dx = sx & ((1ll << 28) - 1), dy = sy & ((1ll << 28) - 1);
        const long long top = ((long long)p00 << 28) + (long long)(int(p10) - int(p00)) * dx;
        const long long bottom = ((long long)p01 << 28) + (long long)(int(p11) - int(p01)) * dx;
        numerator = (u64(top) << 28) + u64(bottom - top) * u64(dy);
    }
    constexpr u64 half = u64(1) << 55;
    const u64 remainder = numerator & ((u64(1) << 56) - 1);
    const u64 distance = remainder > half ? remainder - half : half - remainder;
    if (distance > guard) return u8((numerator + half) >> 56);
    ++repaired;
    return warpPixel(s, w, h, x, y, exact, border);
}

// q28Sample()'s byte from 32-bit arithmetic without branches (Nsight: per-pixel control
// flow took about 40% of the fine kernel's stall samples). With the x weights cut to Q23
// (each corner difference then moves the value by under 255 Q23 units) and one high
// multiply for y (under 2 more), the Q23 value is within 257 units of the exact one, so
// a value farther than that plus the guard from a rounding boundary rounds like
// q28Sample() and needs neither its numerator nor its fallback. Pixels whose four
// neighbours are all outside get the border, as there. `unsure` is set for the rest (the
// margin of a rounding boundary, the image's edge band): the caller asks q28Sample().
// The image has at least two columns and two rows.
__device__ __forceinline__ int screenMargin(u64 guard)
{
    const u64 units = guard >> 33; // Q56 to Q23; the 258 rounds it up
    return 258 + (units < (1u << 23) ? int(units) : 1 << 23);
}
__device__ __forceinline__ u8 q28Screen(const u8* s, int w, int h, long long sx, long long sy, int margin, u8 border, bool& unsure)
{
    constexpr int HALF = 1 << 22;
    const int ix = int(sx >> 28), iy = int(sy >> 28);
    const bool inner = (unsigned(ix) < unsigned(w - 1)) & (unsigned(iy) < unsigned(h - 1));
    const bool outside = (unsigned(ix + 1) > unsigned(w)) | (unsigned(iy + 1) > unsigned(h));
    const u8* p = s + (inner ? size_t(iy) * w + ix : size_t(0));
    const int p00 = p[0], p10 = p[1], p01 = p[w], p11 = p[w + 1];
    const int dx = int((unsigned(sx) << 4) >> 9), dy = int((unsigned(sy) << 4) >> 1);
    const int top = (p00 << 23) + (p10 - p00) * dx, bottom = (p01 << 23) + (p11 - p01) * dx;
    const int value = top + 2 * __mulhi(bottom - top, dy);
    const int fraction = (value & (2 * HALF - 1)) - HALF, distance = fraction < 0 ? -fraction : fraction;
    unsure = inner ? distance <= margin : (!outside) | (margin >= HALF);
    return inner ? u8((value + HALF) >> 23) : border;
}
// One row of a warp's lane: out[x] for x = first, first + 32, ... below end, at map
// column column0 + x and map row y. Unsure pixels are asked of q28Sample() per chunk of
// 32, after the branch-free loop; about 0.1% of pixels are (the guard is that wide).
__device__ __forceinline__ void screenRow(const u8* s, int w, int h, const FineWarp& q, int margin, u8 border,
    int column0, int y, int first, int end, u8* out, unsigned& repaired)
{
    const long long stepX = 32 * q.c, stepY = 32 * q.s;
    long long sx = q.ax + q.c * (column0 + first) - q.s * y, sy = q.ay + q.s * (column0 + first) + q.c * y;
    for (int x = first; x < end;) {
        const int chunk = x;
        unsigned unsureBits = 0;
        for (int bit = 0; bit < 32 && x < end; ++bit, x += 32, sx += stepX, sy += stepY) {
            bool unsure;
            out[x] = q28Screen(s, w, h, sx, sy, margin, border, unsure);
            unsureBits |= unsigned(unsure) << bit;
        }
        while (unsureBits) {
            const int tx = chunk + 32 * (__ffs(unsureBits) - 1), cx = column0 + tx;
            unsureBits &= unsureBits - 1;
            out[tx] = q28Sample(s, w, h, q.ax + q.c * cx - q.s * y, q.ay + q.s * cx + q.c * y, q.guard, cx, y, q.exact, border, repaired);
        }
    }
}

// FP32 sampling (inexact ablation): warpPixel() with every quantity in binary32.
__device__ u8 warpPixel32(const u8* s, int w, int h, int x, int y, const DeviceWarp& m, u8 border)
{
    const float fx = float(x), fy = float(y), c = float(m.c), sn = float(m.s);
    const float sx = __fsub_rn(__fadd_rn(float(m.ax), __fmul_rn(c, fx)), __fmul_rn(sn, fy));
    const float sy = __fadd_rn(__fadd_rn(float(m.ay), __fmul_rn(sn, fx)), __fmul_rn(c, fy));
    if (sx < -1 || sy < -1 || sx >= w || sy >= h) return border;
    const int ix = int(floorf(sx)), iy = int(floorf(sy));
    const float dx = __fsub_rn(sx, float(ix)), dy = __fsub_rn(sy, float(iy));
    const float rx = __fsub_rn(1.0f, dx), ry = __fsub_rn(1.0f, dy);
    const float top = __fadd_rn(__fmul_rn(float(pixelAt(s, w, h, ix, iy, border)), rx),
        __fmul_rn(float(pixelAt(s, w, h, ix + 1, iy, border)), dx));
    const float bottom = __fadd_rn(__fmul_rn(float(pixelAt(s, w, h, ix, iy + 1, border)), rx),
        __fmul_rn(float(pixelAt(s, w, h, ix + 1, iy + 1, border)), dx));
    const float v = floorf(__fadd_rn(__fadd_rn(__fmul_rn(top, ry), __fmul_rn(bottom, dy)), 0.5f));
    return u8(v < 0.0f ? 0.0f : 255.0f < v ? 255.0f : v);
}

// Exact integer template statistics for FP32 scores; dropBelow is the screening
// cut, a binary32 value at or below threshold - SCREEN_MARGIN.
struct ScoreTemplate { u64 sum; float sqrtVariance, dropBelow; int enabled, reserved; };
// NCC in binary32 from exact centered integers, n*sum(xt)-sum(x)*sum(t) over
// sqrt((n*sum(x^2)-sum(x)^2)*(n*sum(t^2)-sum(t)^2)), the integers exact while
// n <= SCORE32_MAX_PIXELS. Returns false (no score) unless the window's variance
// exceeds `factor` times normalizeScore()'s zero-variance limit, 10*FLT_EPSILON*sum(x^2)
// capped at 0.5 (taken here as 1.2e-6, slightly larger).
__device__ __forceinline__ bool score32(u64 sum, u64 square, u64 product, u64 n, const ScoreTemplate& t,
    float factor, float& score)
{
    const float variance = __ull2float_rn(n * square - sum * sum);
    if (!(variance > __fmul_rn(factor * float(n), fminf(__fmul_rn(1.2e-6f, __ull2float_rn(square)), 0.5f)))) return false;
    const long long numerator = (long long)(n * product) - (long long)(sum * t.sum);
    score = __fdiv_rn(__ll2float_rn(numerator), __fmul_rn(__fsqrt_rn(variance), t.sqrtVariance));
    return true;
}

__device__ __forceinline__ unsigned warpSum32(unsigned v)
{
    for (int offset = 16; offset; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Fused coarse search, NCC modes 2-4 (one TILE_X x TILE_Y block of windows of one
// angle, like coarseFused). Each warp samples tile rows with Q28 coordinates that
// step by 32 columns; mode 2 and 3 use the exact Q28 rule with its FP64 fallback,
// mode 4 the binary32 warpPixel32() (inexact ablation). Mode 2 scores every window
// in binary64; mode 3 first scores it in binary32 and computes the binary64 score
// only when it may reach the threshold (the kept set and every kept score equal
// mode 0's); mode 4 keeps the binary32 scores. counts[jobCount] and
// counts[jobCount + 1] accumulate FP64-sampled pixels and binary64-scored windows.
template <int MODE>
__global__ void coarseQ28(const u8* source, int sw, int sh, const CoarseQJob* jobs, const u8* templ,
    int tw, int th, double mean, double norm, double threshold, int uniform, u8 border, ScoreTemplate stats,
    Entry* entries, unsigned* counts, int jobCount)
{
    extern __shared__ u8 shared[];
    const CoarseQJob job = jobs[blockIdx.z];
    const int x0 = blockIdx.x * TILE_X, y0 = blockIdx.y * TILE_Y;
    if (x0 >= job.mapWidth || y0 >= job.mapHeight) return; // uniform per block
    const int tileW = TILE_X + tw - 1, tileH = TILE_Y + th - 1, tid = threadIdx.y * TILE_X + threadIdx.x;
    u8* t = shared;
    u8* tile = shared + tw * th;
    for (int i = tid; i < tw * th; i += COARSE_THREADS) t[i] = templ[i];
    const FineWarp& q = job.q;
    const long long stepX = 32 * q.c, stepY = 32 * q.s;
    unsigned repaired = 0;
    for (int ty = threadIdx.y; ty < tileH; ty += TILE_Y) {
        const int cy = y0 + ty, first = x0 + threadIdx.x;
        long long sx = q.ax + q.c * first - q.s * cy, sy = q.ay + q.s * first + q.c * cy;
        for (int tx = threadIdx.x; tx < tileW; tx += TILE_X, sx += stepX, sy += stepY) {
            const int cx = x0 + tx;
            u8 value = 0;
            if (cx < q.exact.width && cy < q.exact.height) {
                if (MODE == 4) value = warpPixel32(source, sw, sh, cx, cy, q.exact, border);
                else if (q.repairAll) { ++repaired; value = warpPixel(source, sw, sh, cx, cy, q.exact, border); }
                else value = q28Sample(source, sw, sh, sx, sy, q.guard, cx, cy, q.exact, border, repaired);
            }
            tile[ty * tileW + tx] = value;
        }
    }
    if (repaired) atomicAdd(counts + jobCount, repaired);
    __syncthreads();
    const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    bool keep = false;
    double score = 0;
    if (x < job.mapWidth && y < job.mapHeight) {
        if (uniform) { score = 1; keep = !(score < threshold); }
        else {
            u64 sum, square, product;
            windowStats(tile + threadIdx.y * tileW + threadIdx.x, tileW, t, tw, th, sum, square, product);
            const u64 n = u64(tw) * th;
            float s32;
            if (MODE == 4) {
                score = score32(sum, square, product, n, stats, 1.0f, s32) ? fminf(fmaxf(s32, -1.0f), 1.0f) : 0.0f;
                keep = !(score < threshold);
            } else if (MODE == 3 && stats.enabled && n * square == sum * sum) {
                // Zero variance (a constant window, e.g. canvas border): normalizeScore()
                // computes q - s*s/area within a few ulps of 0, below its limit, and returns +0.
                score = 0;
                keep = !(score < threshold);
            } else if (MODE == 3 && stats.enabled && score32(sum, square, product, n, stats, 4.0f, s32) &&
                s32 < stats.dropBelow) {
                keep = false;
            } else {
                if (MODE == 3) atomicAdd(counts + jobCount + 1, 1u);
                score = normalizeScore(sum, square, product, mean, norm, double(tw) * th);
                keep = !(score < threshold);
            }
        }
    }
    appendEntry(keep, x, y, score, counts + blockIdx.z, entries + job.entryOffset, job.capacity);
}

// Coarse NCC modes 5 and 6: a block of `warps` warps covers 16 * warps x TILE_Y windows
// of one angle, and each warp computes 16 x 8 of them with registers, not per-window
// loops. Each thread owns four windows, m in {g, g + 8}, n in {2t, 2t + 1} for lane
// g * 4 + t: the layout of the m16n8k32 accumulator.
//
// Sums and squares slide: per-column totals over the first th rows give the first
// window row (lanes scan over dx), and each later window row adds one tile row and
// removes another. Products come from one of two equal computations. Mode 6 uses the
// int8 Tensor core: for tile row rho, A[m][k] is the row at column dx + m + k and
// B[k][n] is template row rho - n, so A * B adds to each window (m, n) that row's
// contribution. Mode 5 uses __dp4a: every lane walks the template for its four windows.
// Rows and columns past the canvas are read only for windows past the map, which are
// never scored. Scores follow mode 3: binary32 screening, binary64 for the rest. Mode 7 is
// mode 6 in binary32 only (inexact ablation): warpPixel32() and binary32 scores, as mode 4.
constexpr int COARSE_WARP_X = 16, COARSE_WINDOW_WARPS = 4;
__device__ __forceinline__ void mmaU8(unsigned& c0, unsigned& c1, unsigned& c2, unsigned& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#endif
}
// Lane dx gets the sum of lanes 0..dx, for dx < 16.
__device__ __forceinline__ unsigned laneScan16(unsigned v, int lane)
{
    for (int offset = 1; offset < COARSE_WARP_X; offset <<= 1) {
        const unsigned below = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) v += below;
    }
    return v;
}
// Change of each dx's window sum and square when tile row `in` enters and `out`
// leaves (both at this warp's first column, word aligned). Exact modulo 2^32.
__device__ __forceinline__ void slideWindows(const unsigned* in, const unsigned* out, int tw, int lane, unsigned& sum, unsigned& square)
{
    constexpr unsigned ONES = 0x01010101u;
    const int full = tw >> 2, words = (tw + 3) >> 2;
    const unsigned tail = (1u << (8 * (tw & 3))) - 1;
    unsigned s = 0, q = 0;
    for (int j = lane; j < words; j += 32) {
        unsigned a = in[j], b = out[j];
        if (j == full) { a &= tail; b &= tail; }
        s += __dp4a(a, ONES, 0u) - __dp4a(b, ONES, 0u);
        q += __dp4a(a, a, 0u) - __dp4a(b, b, 0u);
    }
    s = warpSum32(s); q = warpSum32(q);
    if (lane > 0) {
        s = q = 0;
        if (lane < COARSE_WARP_X) {
            const auto* inBytes = reinterpret_cast<const u8*>(in);
            const auto* outBytes = reinterpret_cast<const u8*>(out);
            const unsigned a = inBytes[tw + lane - 1], b = inBytes[lane - 1], c = outBytes[tw + lane - 1], d = outBytes[lane - 1];
            s = (a - b) - (c - d);
            q = (a * a - b * b) - (c * c - d * d);
        }
    }
    sum = laneScan16(s, lane); square = laneScan16(q, lane);
}
template <int MODE>
__global__ void coarseWarpWindows(const u8* source, int sw, int sh, const CoarseQJob* jobs, const unsigned* templ,
    int tw, int th, double mean, double norm, double threshold, u8 border, ScoreTemplate stats,
    Entry* entries, unsigned* counts, int jobCount)
{
    extern __shared__ __align__(16) u8 shared[];
    const CoarseQJob job = jobs[blockIdx.z];
    const int warps = blockDim.y, tileX = COARSE_WARP_X * warps;
    const int x0 = blockIdx.x * tileX, y0 = blockIdx.y * TILE_Y;
    if (x0 >= job.mapWidth || y0 >= job.mapHeight) return; // uniform per block
    const int lane = threadIdx.x, warp = threadIdx.y, threads = warps * 32, tid = warp * 32 + lane;
    const int templateWords = (tw + 3) / 4, pitchWords = (tileX + tw + 2) / 4, pitch = pitchWords * 4;
    const int validY = min(job.mapHeight - y0, TILE_Y), rows = validY + th - 1, columns = min(job.mapWidth - x0, tileX) + tw - 1;
    auto* t = reinterpret_cast<unsigned*>(shared);
    auto* tile = t + size_t(templateWords) * th;
    auto* tileBytes = reinterpret_cast<u8*>(tile);
    auto* column = tile + size_t(pitchWords) * (TILE_Y + th - 1); // sum and square per tile column
    for (int i = tid; i < templateWords * th; i += threads) t[i] = templ[i];
    const FineWarp& q = job.q;
    const int margin = screenMargin(q.guard);
    unsigned repaired = 0;
    if (MODE == 7) {
        for (int ty = warp; ty < rows; ty += warps) {
            u8* out = tileBytes + size_t(ty) * pitch;
            for (int tx = lane; tx < columns; tx += 32) out[tx] = warpPixel32(source, sw, sh, x0 + tx, y0 + ty, q.exact, border);
        }
    } else if (q.repairAll) {
        for (int ty = warp; ty < rows; ty += warps) {
            u8* out = tileBytes + size_t(ty) * pitch;
            for (int tx = lane; tx < columns; tx += 32) { ++repaired; out[tx] = warpPixel(source, sw, sh, x0 + tx, y0 + ty, q.exact, border); }
        }
    } else {
        for (int ty = warp; ty < rows; ty += warps)
            screenRow(source, sw, sh, q, margin, border, x0, y0 + ty, lane, columns, tileBytes + size_t(ty) * pitch, repaired);
    }
    if (repaired) atomicAdd(counts + jobCount, repaired);
    __syncthreads();
    for (int c = tid; c < columns; c += threads) {
        unsigned sum = 0, square = 0;
        for (int r = 0; r < th; ++r) { const unsigned v = tileBytes[size_t(r) * pitch + c]; sum += v; square += v * v; }
        column[2 * c] = sum; column[2 * c + 1] = square;
    }
    __syncthreads();
    const int first = COARSE_WARP_X * warp;
    if (x0 + first >= job.mapWidth) return; // uniform per warp

    // Sums and squares of this thread's four windows.
    const int group = lane >> 2, inGroup = lane & 3;
    unsigned windowSum[4] = {}, windowSquare[4] = {};
    {
        unsigned sum = 0, square = 0;
        for (int c = lane; c < tw; c += 32) { sum += column[2 * (first + c)]; square += column[2 * (first + c) + 1]; }
        sum = warpSum32(sum); square = warpSum32(square);
        if (lane > 0) {
            sum = square = 0;
            if (lane < COARSE_WARP_X) {
                const int gone = first + lane - 1, comes = gone + tw;
                sum = column[2 * comes] - column[2 * gone]; square = column[2 * comes + 1] - column[2 * gone + 1];
            }
        }
        sum = laneScan16(sum, lane); square = laneScan16(square, lane);
        for (int dy = 0; dy < validY; ++dy) {
            const unsigned s0 = __shfl_sync(0xffffffffu, sum, group), s1 = __shfl_sync(0xffffffffu, sum, group + 8);
            const unsigned q0 = __shfl_sync(0xffffffffu, square, group), q1 = __shfl_sync(0xffffffffu, square, group + 8);
            if (inGroup == dy >> 1) {
                const int i = dy & 1;
                windowSum[i] = s0; windowSquare[i] = q0; windowSum[2 + i] = s1; windowSquare[2 + i] = q1;
            }
            if (dy + 1 < validY) {
                unsigned ds, dq;
                slideWindows(tile + size_t(th + dy) * pitchWords + first / 4, tile + size_t(dy) * pitchWords + first / 4, tw, lane, ds, dq);
                sum += ds; square += dq;
            }
        }
    }

    // Products of the four windows, accumulated exactly while n * 65025 < 2^32.
    unsigned p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    const int shift = (group & 3) * 8;
    if (MODE >= 6) {
        // Measured m16n8k32 layout (lane = group * 4 + inGroup, bytes low to high): A registers
        // hold rows group, group + 8 at columns 4 * inGroup, then the same at + 16; B registers
        // hold column n = group at rows 4 * inGroup, then + 16; the accumulator holds rows
        // group, group + 8 at columns 2 * inGroup, 2 * inGroup + 1 (four registers, row first).
        for (int rho = 0; rho < rows; ++rho) {
            const int tr = rho - group;
            const unsigned rowMask = tr >= 0 && tr < th ? ~0u : 0u;
            const unsigned* tRow = t + size_t(tr < 0 ? 0 : tr >= th ? th - 1 : tr) * templateWords;
            const unsigned* row = tile + size_t(rho) * pitchWords + (first + group) / 4 + inGroup;
            for (int k = 0; k < templateWords; k += 8) {
                const unsigned w0 = row[k], w1 = row[k + 1], w2 = row[k + 2], w3 = row[k + 3];
                const unsigned w4 = row[k + 4], w5 = row[k + 5], w6 = row[k + 6], w7 = row[k + 7];
                const int b = k + inGroup;
                const unsigned b0 = b < templateWords ? tRow[b] & rowMask : 0u, b1 = b + 4 < templateWords ? tRow[b + 4] & rowMask : 0u;
                mmaU8(p0, p1, p2, p3, __funnelshift_r(w0, w1, shift), __funnelshift_r(w2, w3, shift),
                    __funnelshift_r(w4, w5, shift), __funnelshift_r(w6, w7, shift), b0, b1);
            }
        }
    } else {
        const int n0 = 2 * inGroup, n1 = n0 + 1, word0 = (first + group) / 4, word1 = word0 + 2;
        for (int r = 0; r < th; ++r) {
            const unsigned* tRow = t + size_t(r) * templateWords;
            const unsigned* r00 = tile + size_t(r + n0) * pitchWords + word0;
            const unsigned* r01 = tile + size_t(r + n1) * pitchWords + word0;
            const unsigned* r10 = tile + size_t(r + n0) * pitchWords + word1;
            const unsigned* r11 = tile + size_t(r + n1) * pitchWords + word1;
            unsigned a0 = r00[0], a1 = r01[0], a2 = r10[0], a3 = r11[0];
            for (int j = 0; j < templateWords; ++j) {
                const unsigned tv = tRow[j];
                const unsigned b0 = r00[j + 1], b1 = r01[j + 1], b2 = r10[j + 1], b3 = r11[j + 1];
                p0 = __dp4a(__funnelshift_r(a0, b0, shift), tv, p0);
                p1 = __dp4a(__funnelshift_r(a1, b1, shift), tv, p1);
                p2 = __dp4a(__funnelshift_r(a2, b2, shift), tv, p2);
                p3 = __dp4a(__funnelshift_r(a3, b3, shift), tv, p3);
                a0 = b0; a1 = b1; a2 = b2; a3 = b3;
            }
        }
    }

    // Scores, as coarse NCC mode 3.
    const unsigned products[4] = {p0, p1, p2, p3};
    const u64 pixels = u64(tw) * th;
    unsigned exact = 0;
    for (int i = 0; i < 4; ++i) {
        const int x = x0 + first + (i & 2 ? group + 8 : group), y = y0 + 2 * inGroup + (i & 1);
        bool keep = false;
        double score = 0;
        if (x < job.mapWidth && y < job.mapHeight) {
            const u64 sum = windowSum[i], square = windowSquare[i], product = products[i];
            float s32;
            if (MODE == 7) {
                score = score32(sum, square, product, pixels, stats, 1.0f, s32) ? fminf(fmaxf(s32, -1.0f), 1.0f) : 0.0f;
                keep = !(score < threshold);
            } else if (stats.enabled && pixels * square == sum * sum) {
                keep = !(score < threshold); // zero variance: normalizeScore() returns +0
            } else if (stats.enabled && score32(sum, square, product, pixels, stats, 4.0f, s32) && s32 < stats.dropBelow) {
                keep = false;
            } else {
                ++exact;
                score = normalizeScore(sum, square, product, mean, norm, double(tw) * th);
                keep = !(score < threshold);
            }
        }
        appendEntry(keep, x, y, score, counts + blockIdx.z, entries + job.entryOffset, job.capacity);
    }
    if (exact) atomicAdd(counts + jobCount + 1, exact);
}

// Fine search, one (template row stripe of at most 32 rows, request) per block of
// 8 warps. The stripe's ROI rows are sampled into shared memory (rows of `pitch`
// bytes, template rows of `tpitch`, both odd word counts). Warp dy < 7 computes
// the template products of window row dy: lane r takes template row r and keeps
// the seven dx sums, four pixels per __dp4a. Warp 7 computes each ROI row's sums
// and squares for the seven dx offsets. Per-lane uint32 sums are exact while
// tw <= 66051; block totals are added to acc.
//
// SAMPLING 0 is the original sampling loop (fine NCC modes 0 and 1); 2 samples
// each ROI row with one warp, its Q28 coordinates stepping by 32 columns, through
// q28Sample() (same bytes, mode 2); 3 samples in binary32 (inexact ablation, mode 3);
// 4 is 2 through screenRow() (same bytes, mode 4; narrow ROIs, where a row is only a
// few pixels per lane and its setup outweighs the cheaper pixels, keep SAMPLING 2);
// 5 is 4 and 6 is 2 with the products by int8 Tensor core (mode 5): one m16n8k32 takes two
// ROI rows. For the pair (rho, rho + 1), rho even, A[m][k] is row rho at column dx + k for
// m = dx and row rho + 1 for m = dx + 8, and B[k][n] is template row rho + 1 - n, so the
// accumulator (dx, n) belongs to window (dx, n - 1) and (dx + 8, n) to window (dx, n).
// Of `warps` warps, warp w takes the pairs and the rows of the row statistics w mod warps
// (one warp for those would now be the slowest). Template rows are padded to 32 bytes, so the loop needs
// no bounds, and a zero row after them stands in for rows outside the stripe. The
// accumulators of one warp are exact while its windows hold at most 66051 pixels (host
// check); the last stage adds the warps' totals. 7 is 3 with those products (inexact
// ablation, mode 6).
template <int SAMPLING, bool DISTRIBUTED_DP4A = false>
__global__ void fineStripes(const u8* source, int sw, int sh, const FineWarp* requests, const u8* templ,
    int tw, int th, int stripeRows, int pitch, int tpitch, u64* acc, unsigned* repairedTotal)
{
    extern __shared__ __align__(16) u8 shared[];
    __shared__ unsigned blockRepaired;
    const int rw = tw + 2 * PAD, r0 = blockIdx.x * stripeRows, rows = min(stripeRows, th - r0), roiRows = rows + 2 * PAD;
    u8* roi = shared;
    u8* t = shared + size_t(roiRows) * pitch;
    const int templateRows = SAMPLING >= 5 ? rows + 1 : rows;
    unsigned* rowStats = reinterpret_cast<unsigned*>(t + size_t(templateRows) * tpitch);
    unsigned* products = rowStats + size_t(roiRows) * FINE_SIDE * 2;
    // One load per block: lean calls read requests from mapped host memory.
    __shared__ FineWarp request;
    if (threadIdx.x == 0) { blockRepaired = 0; request = requests[blockIdx.y]; }
    __syncthreads();
    const FineWarp m = request;
    unsigned repaired = 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x / 32;
    if (SAMPLING == 0) {
        for (int i = threadIdx.x; i < roiRows * rw; i += blockDim.x) {
            const int x = i % rw, y = i / rw;
            roi[size_t(y) * pitch + x] = finePixel(source, sw, sh, x, r0 + y, m, repaired);
        }
        for (int i = threadIdx.x; i < rows * tpitch; i += blockDim.x) {
            const int y = i / tpitch, x = i % tpitch;
            t[i] = x < tw ? templ[size_t(r0 + y) * tw + x] : 0;
        }
    } else {
        const long long stepX = 32 * m.c, stepY = 32 * m.s;
        if ((SAMPLING == 4 || SAMPLING == 5) && !m.repairAll && sw > 1 && sh > 1) {
            const int margin = screenMargin(m.guard);
            for (int y = warp; y < roiRows; y += warps)
                screenRow(source, sw, sh, m, margin, 0, 0, r0 + y, lane, rw, roi + size_t(y) * pitch, repaired);
        } else {
            for (int y = warp; y < roiRows; y += warps) {
                const int ry = r0 + y;
                long long sx = m.ax + m.c * lane - m.s * ry, sy = m.ay + m.s * lane + m.c * ry;
                u8* out = roi + size_t(y) * pitch;
                for (int x = lane; x < rw; x += 32, sx += stepX, sy += stepY) {
                    if (SAMPLING == 3 || SAMPLING == 7) out[x] = warpPixel32(source, sw, sh, x, ry, m.exact, 0);
                    else if (m.repairAll) { ++repaired; out[x] = warpPixel(source, sw, sh, x, ry, m.exact, 0); }
                    else out[x] = q28Sample(source, sw, sh, sx, sy, m.guard, x, ry, m.exact, 0, repaired);
                }
            }
        }
        for (int y = warp; y < templateRows; y += warps) {
            const u8* in = templ + size_t(r0 + y) * tw;
            u8* out = t + size_t(y) * tpitch;
            for (int x = lane; x < tpitch; x += 32) out[x] = y < rows && x < tw ? in[x] : 0;
        }
    }
    if (repaired) atomicAdd(&blockRepaired, repaired);
    __syncthreads();
    u64* out = acc + size_t(blockIdx.y) * FINE_WINDOWS * 3;
    if (SAMPLING >= 5 && !DISTRIBUTED_DP4A) {
        const int group = lane >> 2, inGroup = lane & 3, shift = (group & 3) * 8, words = tpitch / 4;
        const int offset = group / 4 + inGroup, rowWords = pitch / 4, tWords = tpitch / 4;
        const auto* roiWords = reinterpret_cast<const unsigned*>(roi);
        const auto* tBase = reinterpret_cast<const unsigned*>(t);
        unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int rho = 2 * warp; rho < roiRows; rho += 2 * warps) {
            const int tr = rho + 1 - group;
            const unsigned* b = tBase + size_t(tr >= 0 && tr < rows ? tr : rows) * tWords + inGroup;
            const unsigned* r0Row = roiWords + size_t(rho) * rowWords + offset;
            const unsigned* r1Row = r0Row + rowWords;
            for (int k = 0; k < words; k += 8, b += 8, r0Row += 8, r1Row += 8) {
                mmaU8(c0, c1, c2, c3, __funnelshift_r(r0Row[0], r0Row[1], shift), __funnelshift_r(r1Row[0], r1Row[1], shift),
                    __funnelshift_r(r0Row[4], r0Row[5], shift), __funnelshift_r(r1Row[4], r1Row[5], shift), b[0], b[4]);
            }
        }
        const unsigned even = c1 + c2, odd = c3 + __shfl_down_sync(0xffffffffu, c0, 1);
        if (group < FINE_SIDE) {
            const int base = warp * FINE_WINDOWS + group + 2 * inGroup * FINE_SIDE;
            products[base] = even;
            if (inGroup < 3) products[base + FINE_SIDE] = odd;
        }
    } else if (DISTRIBUTED_DP4A || warp < FINE_SIDE) {
        const int dy = warp % FINE_SIDE;
        const int groups = (warps + FINE_SIDE - 1 - dy) / FINE_SIDE;
        const int row = DISTRIBUTED_DP4A ? lane * groups + warp / FINE_SIDE : lane;
        unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
        if (row < rows) {
            const auto* rowWords = reinterpret_cast<const unsigned*>(roi + size_t(row + dy) * pitch);
            const auto* templateWords = reinterpret_cast<const unsigned*>(t + size_t(row) * tpitch);
            unsigned w0 = rowWords[0], w1 = rowWords[1];
            for (int j = 0, words = DISTRIBUTED_DP4A ? (tw + 3) / 4 : tpitch / 4; j < words; ++j) {
                const unsigned w2 = rowWords[j + 2], tv = templateWords[j];
                a0 = __dp4a(w0, tv, a0);
                a1 = __dp4a(__funnelshift_r(w0, w1, 8), tv, a1);
                a2 = __dp4a(__funnelshift_r(w0, w1, 16), tv, a2);
                a3 = __dp4a(__funnelshift_r(w0, w1, 24), tv, a3);
                a4 = __dp4a(w1, tv, a4);
                a5 = __dp4a(__funnelshift_r(w1, w2, 8), tv, a5);
                a6 = __dp4a(__funnelshift_r(w1, w2, 16), tv, a6);
                w0 = w1; w1 = w2;
            }
        }
        const unsigned values[FINE_SIDE] = {a0, a1, a2, a3, a4, a5, a6};
        for (int dx = 0; dx < FINE_SIDE; ++dx) {
            const u64 product = warpSum(values[dx]);
            if (lane == 0) atomicAdd(out + (dy * FINE_SIDE + dx) * 3 + 2, product);
        }
    }
    if (SAMPLING >= 5 || warp == FINE_SIDE) {
        const int full = tw / 4;
        for (int rho = SAMPLING >= 5 ? warp : 0; rho < roiRows; rho += SAMPLING >= 5 ? warps : 1) {
            const u8* row = roi + size_t(rho) * pitch;
            const auto* words = reinterpret_cast<const unsigned*>(row);
            unsigned sum = 0, square = 0;
            for (int j = lane; j < full; j += 32) {
                const unsigned w = words[j];
                sum = __dp4a(w, 0x01010101u, sum); square = __dp4a(w, w, square);
            }
            if (lane == 0)
                for (int k = 4 * full; k < tw; ++k) { const unsigned v = row[k]; sum += v; square += v * v; }
            sum = warpSum32(sum); square = warpSum32(square);
            if (lane == 0) {
                // Slide the tw-pixel window right; unsigned wraparound cancels exactly.
                for (int dx = 0; dx < FINE_SIDE; ++dx) {
                    rowStats[(rho * FINE_SIDE + dx) * 2] = sum; rowStats[(rho * FINE_SIDE + dx) * 2 + 1] = square;
                    if (dx + 1 < FINE_SIDE) {
                        const unsigned in = row[tw + dx], gone = row[dx];
                        sum += in - gone; square += in * in - gone * gone;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < FINE_WINDOWS) {
        const int dx = threadIdx.x % FINE_SIDE, dy = threadIdx.x / FINE_SIDE;
        u64 sum = 0, square = 0;
        for (int r = 0; r < rows; ++r) {
            sum += rowStats[((r + dy) * FINE_SIDE + dx) * 2]; square += rowStats[((r + dy) * FINE_SIDE + dx) * 2 + 1];
        }
        atomicAdd(out + threadIdx.x * 3, sum); atomicAdd(out + threadIdx.x * 3 + 1, square);
        if (SAMPLING >= 5 && !DISTRIBUTED_DP4A) {
            u64 product = 0;
            for (int w = 0; w < warps; ++w) product += products[w * FINE_WINDOWS + threadIdx.x];
            atomicAdd(out + threadIdx.x * 3 + 2, product);
        }
    }
    if (threadIdx.x == 0 && blockRepaired) atomicAdd(repairedTotal, blockRepaired);
}
// Binary32 fine score (inexact ablation, fine NCC mode 3), 0 for near-zero variance.
__device__ __forceinline__ double fineScore32(const u64* stats, u64 n, const ScoreTemplate& t)
{
    float s;
    return score32(stats[0], stats[1], stats[2], n, t, 1.0f, s) ? fminf(fmaxf(s, -1.0f), 1.0f) : 0.0f;
}
// Scores to mapped host memory, followed by the count of FP64-sampled pixels.
// With a flag (lean calls) each thread also clears the statistics it read, so
// the accumulators are zero for the next batch, and the grid signals completion.
template <bool FP32>
__global__ void fineNormalize(u64* acc, int windows, double mean, double norm, double area, int uniform,
    unsigned* repaired, double* out, unsigned* done, unsigned* flag, unsigned value, ScoreTemplate stats)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < windows) out[i] = uniform ? 1.0 : FP32 ? fineScore32(acc + 3 * i, u64(area), stats) :
        normalizeScore(acc[3 * i], acc[3 * i + 1], acc[3 * i + 2], mean, norm, area);
    if (i == 0) *reinterpret_cast<unsigned*>(out + windows) = uniform ? 0 : *repaired;
    if (flag) {
        if (i < windows) acc[3 * i] = acc[3 * i + 1] = acc[3 * i + 2] = 0;
        if (i == 0) *repaired = 0;
        signalLast(done, flag, value);
    }
}

// ---- host buffers -------------------------------------------------------------
class DeviceBytes {
public:
    ~DeviceBytes() { if (data_) cudaFree(data_); }
    void* reserve(size_t bytes) {
        if (bytes <= capacity_) return data_;
        if (data_) check(cudaFree(data_), "free");
        data_ = nullptr; capacity_ = 0;
        check(cudaMalloc(&data_, bytes), "malloc");
        capacity_ = bytes;
        return data_;
    }
    size_t capacity() const { return capacity_; }
private:
    void* data_ = nullptr;
    size_t capacity_ = 0;
};
// Page-locked host memory, also mapped into the device address space.
class HostBytes {
public:
    ~HostBytes() { if (data_) cudaFreeHost(data_); }
    void* reserve(size_t bytes) {
        if (bytes <= capacity_) return data_;
        if (data_) check(cudaFreeHost(data_), "free host");
        data_ = nullptr; capacity_ = 0;
        check(cudaHostAlloc(&data_, bytes, cudaHostAllocMapped), "host alloc");
        capacity_ = bytes;
        return data_;
    }
    void* device() const {
        void* p = nullptr;
        check(cudaHostGetDevicePointer(&p, data_, 0), "mapped pointer");
        return p;
    }
    void* host() const { return data_; }
    size_t capacity() const { return capacity_; }
private:
    void* data_ = nullptr;
    size_t capacity_ = 0;
};

size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

// Lean prepare's source copy with several threads (fpm_frame_set_transfer_threads).
// A task is one prepare: share k of every chunk (rows [k*rows/T, (k+1)*rows/T) of
// it) is copied by worker k, share 0 by the caller, chunk after chunk; done(c)
// counts the shares of chunk c that are copied, so the caller can queue chunk c's
// upload while the workers copy chunk c + 1.
class CopyPool {
public:
    struct Task { const u8* source; size_t stride; u8* host; int width, height, chunkRows, chunks; bool invert; };
    ~CopyPool() { resize(1); }
    int threads() const { return int(workers_.size()) + 1; }
    void resize(int threads) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) worker.join();
        workers_.clear();
        stop_ = false;
        for (int share = 1; share < threads; ++share) workers_.emplace_back([this, share] { run(share); });
    }
    void start(const Task& task) {
        if (task.chunks > int(capacity_)) {
            done_.reset(new std::atomic<int>[size_t(task.chunks)]);
            capacity_ = size_t(task.chunks);
        }
        for (int c = 0; c < task.chunks; ++c) done_[c].store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            ++generation_;
        }
        wake_.notify_all();
    }
    void copyShare(int share, int chunk) {
        const int y0 = chunk * task_.chunkRows, rows = std::min(task_.chunkRows, task_.height - y0), n = threads();
        copyRows(task_, y0 + share * rows / n, y0 + (share + 1) * rows / n);
        done_[chunk].fetch_add(1, std::memory_order_release);
    }
    bool chunkDone(int chunk) const { return done_[chunk].load(std::memory_order_acquire) == threads(); }
    // Every worker has left the task, so the next start() may replace it.
    void waitFinished() const {
        while (finished_.load(std::memory_order_acquire) != int(workers_.size())) {}
    }
    static void copyRows(const Task& task, int begin, int end) {
        for (int y = begin; y < end; ++y) {
            const u8* in = task.source + size_t(y) * task.stride;
            u8* out = task.host + size_t(y) * task.width;
            if (task.invert) for (int x = 0; x < task.width; ++x) out[x] = u8(255 - in[x]);
            else std::memcpy(out, in, size_t(task.width));
        }
    }
private:
    void run(int share) {
        unsigned long long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            for (int c = 0; c < task_.chunks; ++c) copyShare(share, c);
            finished_.fetch_add(1, std::memory_order_release);
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
    unsigned long long generation_ = 0;
    Task task_{};
    std::unique_ptr<std::atomic<int>[]> done_;
    size_t capacity_ = 0;
    std::atomic<int> finished_{0};
};
// Lean prepare copies the source in chunks of at least 2 MiB, at most four per image.
constexpr size_t MIN_UPLOAD_CHUNK = size_t(2) << 20, MAX_UPLOAD_CHUNKS = 4;

class CudaBackend final : public fpm_coarse_detail::Backend {
public:
    CudaBackend() {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
            cudaGetLastError();
            throw std::runtime_error("no CUDA device is available");
        }
        check(cudaSetDevice(0), "set device");
        check(cudaDeviceGetAttribute(&sharedOptin_, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0), "shared limit");
        check(cudaFuncSetAttribute(coarseFused, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedOptin_), "coarse shared");
        // coarseQ28 also has static shared counters; its dynamic limit is what remains.
        cudaFuncAttributes q28{};
        check(cudaFuncGetAttributes(&q28, coarseQ28<2>), "coarse attributes");
        coarseQ28Limit_ = sharedOptin_ - int(q28.sharedSizeBytes);
        check(cudaFuncSetAttribute(coarseQ28<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, coarseQ28Limit_), "coarse shared");
        check(cudaFuncSetAttribute(coarseQ28<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, coarseQ28Limit_), "coarse shared");
        check(cudaFuncSetAttribute(coarseQ28<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, coarseQ28Limit_), "coarse shared");
        cudaFuncAttributes windowsAttributes{};
        check(cudaFuncGetAttributes(&windowsAttributes, coarseWarpWindows<5>), "coarse attributes");
        coarseWindowsLimit_ = sharedOptin_ - int(windowsAttributes.sharedSizeBytes);
        check(cudaFuncSetAttribute(coarseWarpWindows<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, coarseWindowsLimit_), "coarse shared");
        check(cudaFuncSetAttribute(coarseWarpWindows<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, coarseWindowsLimit_), "coarse shared");
        check(cudaFuncSetAttribute(coarseWarpWindows<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, coarseWindowsLimit_), "coarse shared");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "device properties");
        tensorCores_ = properties.major >= 8;
        // The fine sampling variants may differ in static shared memory; budget for the largest.
        cudaFuncAttributes fine{}, variant{};
        check(cudaFuncGetAttributes(&fine, fineStripes<0>), "fine attributes");
        check(cudaFuncGetAttributes(&variant, fineStripes<2>), "fine attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<3>), "fine attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<4>), "fine attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<5>), "fine attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<6>), "fine attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<7>), "fine attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<5, true>), "dp4a wide attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        check(cudaFuncGetAttributes(&variant, fineStripes<6, true>), "dp4a narrow attributes");
        fine.sharedSizeBytes = std::max(fine.sharedSizeBytes, variant.sharedSizeBytes);
        fineSharedLimit_ = sharedOptin_ - int(fine.sharedSizeBytes);
        // Prefer stripes small enough for two resident blocks per SM.
        int perSm = 0;
        check(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0), "SM shared");
        fineSharedTarget_ = std::min(fineSharedLimit_, perSm / 2 - 1024 - int(fine.sharedSizeBytes));
        check(cudaFuncSetAttribute(fineStripes<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaFuncSetAttribute(fineStripes<5, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "dp4a wide shared");
        check(cudaFuncSetAttribute(fineStripes<6, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "dp4a narrow shared");
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
        check(cudaEventCreate(&begin_), "event"); check(cudaEventCreate(&end_), "event");
        check(cudaEventCreateWithFlags(&sourceRead_, cudaEventDisableTiming), "event");
        // Lean-call completion flag (mapped) and grid counters [coarse, fine] (device).
        flag_ = new (signal_.reserve(64)) std::atomic<unsigned>(0);
        flagDevice_ = static_cast<unsigned*>(signal_.device());
        check(cudaMemsetAsync(leanDone(), 0, 16, stream_), "lean counters");
        resetMetrics();
        resetTransferMetrics();
    }
    ~CudaBackend() override {
        releaseHostSource();
        cudaEventDestroy(begin_); cudaEventDestroy(end_); cudaEventDestroy(sourceRead_);
        cudaStreamDestroy(stream_);
    }

    // ---- configuration ----
    void setExecution(const FpmCoarseExecutionConfigV1& config) override {
        if (config.pyramid_mode < 0 || config.pyramid_mode > 1 || config.batch_angles != 1)
            throw std::invalid_argument("CUDA supports pyramid modes 0 (CPU) and 1 (exact GPU) with batch_angles 1");
        pyramidMode_ = config.pyramid_mode;
    }
    void setFineBatchSize(std::size_t count) override { fineBatchSize_ = count; }
    std::size_t fineBatchSize() const override { return fineBatchSize_; }
    // Coarse NCC modes: 0 fused, 1 canvases in device memory (unfused ablation),
    // 2 fused with exact Q28 tile sampling, 3 that plus binary32 screening of the
    // scores, 4 binary32 sampling and scores (inexact ablation), 5 mode 3 with window
    // products by __dp4a in warp-level tiles, 6 the same with int8 Tensor cores,
    // 7 mode 6 in binary32 only (inexact ablation; outputs of all but 4 and 7 are identical).
    void setCoarseOptimization(int peakMode, int nccMode) override {
        if (peakMode != 0 || nccMode < 0 || nccMode > 8)
            throw std::invalid_argument("CUDA coarse optimization takes peak mode 0 and NCC mode 0..8");
        if (nccMode >= 6 && !tensorCores_) throw std::invalid_argument("CUDA coarse NCC modes 6 through 8 need compute capability 8.0");
        fusedCoarse_ = nccMode != 1;
        coarseMode_ = nccMode;
    }
    bool deviceCoarse() const override { return true; }
    // Fine NCC modes: 0 the Q28 fast path with the guard; 1 the FP64 warpPixel()
    // for every fine pixel (ablation and tests of the FP64 path); 2 mode 0's bytes
    // from a row-wise loop with stepped Q28 coordinates; 3 binary32 sampling and
    // scores (inexact ablation); 4 mode 2 with the rounding screened in 32-bit arithmetic
    // on ROIs of at least FINE_SCREEN_MIN_WIDTH; 5 mode 4 with the products by int8
    // Tensor core (needs compute capability 8.0); 6 mode 3 with those products (inexact
    // ablation).
    void setFineNccMode(int mode) override {
        if (mode < 0 || mode > 7) throw std::invalid_argument("CUDA fine NCC mode is 0..7");
        if (mode >= 5 && !tensorCores_) throw std::invalid_argument("CUDA fine NCC modes 5 through 7 need compute capability 8.0");
        exactFine_ = mode == 1;
        fineMode_ = mode;
    }
    // Transfer mode bit 1: lean prepare; bit 2: lean calls; bit 4: lean calls upload
    // the fine requests instead of reading them from mapped memory
    // (fpm_frame_set_transfer_mode).
    void setTransferMode(int mode) override {
        if (mode < 0 || mode > 7) throw std::invalid_argument("CUDA transfer mode is 0..7");
        releaseHostSource();
        leanPrepare_ = (mode & 1) != 0; leanCalls_ = (mode & 2) != 0; uploadRequests_ = (mode & 4) != 0;
        // Mode 0 leaves counters dirty; lean calls clear them once before relying on that.
        leanCountsZeroed_ = leanAccZeroed_ = 0;
        transfer_.mode = mode;
        resetMetrics();
    }
    bool reusesSourceLevels() const override { return leanPrepare_ && pyramidMode_; }
    void setTransferThreads(int threads) override {
        if (threads < 1 || threads > 16) throw std::invalid_argument("CUDA transfer threads are 1..16");
        if (threads != copyPool_.threads()) copyPool_.resize(threads);
        copyThreads_ = threads;
    }

    // ---- metrics ----
    void resetMetrics() override {
        // Lean modes record no timing events: their device times are unavailable (-1).
        metrics_ = {}; metrics_.device_ms = leanCalls_ ? -1 : 0;
        fine_ = {}; fine_.version = 1; fine_.struct_bytes = sizeof(fine_); fine_.device_ms = leanCalls_ ? -1 : 0;
        execution_ = {}; execution_.version = 1; execution_.struct_bytes = sizeof(execution_);
        execution_.pyramid_mode = pyramidMode_; execution_.batch_angles = 1; execution_.pyramid_device_ms = 0;
        kernel_ = {}; kernel_.version = 1; kernel_.struct_bytes = sizeof(kernel_);
    }
    FpmCudaKernelMetricsV1 cudaKernelMetrics() const override {
        auto m = kernel_;
        m.coarse_ncc_mode = coarseMode_; m.fine_ncc_mode = fineMode_; m.copy_threads = copyThreads_;
        return m;
    }
    void resetTransferMetrics() override {
        const int mode = transfer_.mode, registered = transfer_.source_registered;
        const u64 bytes = transfer_.registered_bytes;
        transfer_ = {}; transfer_.version = 1; transfer_.struct_bytes = sizeof(transfer_);
        transfer_.mode = mode; transfer_.source_registered = registered; transfer_.registered_bytes = bytes;
    }
    FpmTransferMetricsV1 transferMetrics() const override { return transfer_; }
    FpmCoarseMetrics metrics() const override {
        auto m = metrics_;
        m.gpu_bytes = retainedDeviceBytes();
        return m;
    }
    FpmFineMetricsV1 fineMetrics() const override {
        auto m = fine_;
        for (const auto& buffer : templateBuffers_) m.template_buffer_bytes += buffer.capacity();
        m.retained_gpu_bytes = fineDevice_.capacity() + fineAcc_.capacity() + m.template_buffer_bytes;
        return m;
    }
    FpmCoarseExecutionMetricsV1 executionMetrics() const override {
        auto m = execution_;
        m.pyramid_mode = pyramidMode_; m.batch_angles = 1;
        m.pyramid_buffer_bytes = pyramid_.capacity();
        return m;
    }

    void invalidateFineSource() override { resident_.clear(); uploaded_.clear(); uploadedBytes_ = 0; }

    // ---- pyramid ----
    bool buildPyramid(std::vector<fpm::Image>& pyramid, int top) override {
        if (!pyramidMode_) return false;
        const Range range("fpm pyramid");
        const auto start = Clock::now();
        if (pyramid.size() != 1) throw std::invalid_argument("CUDA pyramid expects level 0 only");
        struct Level { int width, height; size_t offset; };
        std::vector<Level> levels{{pyramid[0].width(), pyramid[0].height(), 0}};
        size_t total = size_t(levels[0].width) * levels[0].height;
        for (int layer = 1; layer <= top; ++layer) {
            const auto& p = levels.back();
            levels.push_back({p.width / 2 + p.width % 2, p.height / 2 + p.height % 2, align16(total)});
            total = levels.back().offset + size_t(levels.back().width) * levels.back().height;
        }
        auto* device = static_cast<u8*>(pyramid_.reserve(total));
        execution_.pyramid_input_bytes += size_t(levels[0].width) * levels[0].height;
        const auto upload = Clock::now();
        check(cudaEventRecord(begin_, stream_), "event");
        check(cudaMemcpyAsync(device, pyramid[0].row(0), size_t(levels[0].width) * levels[0].height,
            cudaMemcpyHostToDevice, stream_), "pyramid upload");
        for (int layer = 1; layer <= top; ++layer) {
            const auto& in = levels[layer - 1]; const auto& out = levels[layer];
            const dim3 block(32, 8), grid((out.width + 31) / 32, (out.height + 7) / 8);
            pyrDownKernel<<<grid, block, 0, stream_>>>(device + in.offset, in.width, in.height,
                device + out.offset, out.width, out.height);
            check(cudaGetLastError(), "pyramid kernel");
        }
        pyramid.reserve(levels.size());
        for (int layer = 1; layer <= top; ++layer) {
            pyramid.emplace_back(levels[layer].width, levels[layer].height);
            check(cudaMemcpyAsync(pyramid.back().row(0), device + levels[layer].offset,
                size_t(levels[layer].width) * levels[layer].height, cudaMemcpyDeviceToHost, stream_), "pyramid copyout");
            execution_.pyramid_output_bytes += size_t(levels[layer].width) * levels[layer].height;
        }
        check(cudaEventRecord(end_, stream_), "event");
        execution_.pyramid_encode_ms += elapsed(upload);
        const auto wait = Clock::now();
        check(cudaStreamSynchronize(stream_), "pyramid");
        execution_.pyramid_wait_ms += elapsed(wait);
        execution_.pyramid_device_ms += eventMs();
        ++execution_.pyramid_command_buffers;
        execution_.pyramid_levels += top;
        for (int layer = 0; layer <= top; ++layer)
            resident_[pyramid[layer].row(0)] = {device + levels[layer].offset, levels[layer].width, levels[layer].height};
        execution_.pyramid_wall_ms += elapsed(start);
        return true;
    }
    // Lean prepare (transfer mode bit 1). The frame's host levels persist across
    // prepares of one size, so level 0 stays page-locked and its pages resident.
    // The source is copied into level 0 in chunks; each chunk's upload is queued
    // as soon as it is copied and overlaps the next copy. The device pyramid is
    // not waited for: later calls are ordered after it on the stream. Levels
    // 1..top reach the host only for the CPU fine path (no fine batch).
    bool prepareSource(fpm::ImageView source, bool invert, int top, std::vector<fpm::Image>& levels,
        double* copyMs) override
    {
        if (!leanPrepare_ || !pyramidMode_) return false;
        const Range range("fpm lean prepare");
        const auto start = Clock::now();
        waitSourceRead();
        struct Level { int width, height; size_t offset; };
        std::vector<Level> shape{{source.width, source.height, 0}};
        size_t total = size_t(source.width) * source.height;
        for (int layer = 1; layer <= top; ++layer) {
            const auto& p = shape.back();
            shape.push_back({p.width / 2 + p.width % 2, p.height / 2 + p.height % 2, align16(total)});
            total = shape.back().offset + size_t(shape.back().width) * shape.back().height;
        }
        const int width = source.width, height = source.height;
        const size_t bytes = size_t(width) * height;
        bool same = levels.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i)
            same = levels[i].width() == shape[i].width && levels[i].height() == shape[i].height;
        if (!same) {
            releaseHostSource();
            levels.clear();
            levels.reserve(shape.size());
            for (const auto& level : shape) levels.emplace_back(level.width, level.height);
            ++transfer_.host_level_rebuilds;
            if (cudaHostRegister(levels[0].row(0), bytes, cudaHostRegisterDefault) == cudaSuccess) {
                registered_ = levels[0].row(0); transfer_.source_registered = 1; transfer_.registered_bytes = bytes;
            } else {
                cudaGetLastError(); // e.g. a page shared with another registered range: plain uploads
            }
        }
        auto* device = static_cast<u8*>(pyramid_.reserve(total));
        const size_t chunk = std::max(MIN_UPLOAD_CHUNK, (bytes + MAX_UPLOAD_CHUNKS - 1) / MAX_UPLOAD_CHUNKS);
        const int chunkRows = int(std::min<size_t>(size_t(height), std::max<size_t>(1, chunk / size_t(width))));
        u8* host = levels[0].row(0);
        double copy = 0;
        const int chunks = (height + chunkRows - 1) / chunkRows;
        const CopyPool::Task task{source.data, source.stride, host, width, height, chunkRows, chunks, invert};
        if (copyPool_.threads() > 1) copyPool_.start(task);
        for (int c = 0; c < chunks; ++c) {
            const int y0 = c * chunkRows, rows = std::min(chunkRows, height - y0);
            const auto copyStart = Clock::now();
            if (copyPool_.threads() > 1) {
                copyPool_.copyShare(0, c);
                while (!copyPool_.chunkDone(c)) {}
            } else {
                CopyPool::copyRows(task, y0, y0 + rows);
            }
            copy += elapsed(copyStart);
            const size_t offset = size_t(y0) * width;
            check(cudaMemcpyAsync(device + offset, host + offset, size_t(rows) * width, cudaMemcpyHostToDevice, stream_),
                "lean upload");
            ++transfer_.upload_chunks;
        }
        if (copyPool_.threads() > 1) copyPool_.waitFinished();
        check(cudaEventRecord(sourceRead_, stream_), "event");
        sourceReadPending_ = true;
        for (int layer = 1; layer <= top; ++layer) {
            const auto& in = shape[layer - 1]; const auto& out = shape[layer];
            const dim3 block(32, 8), grid((out.width + 31) / 32, (out.height + 7) / 8);
            pyrDownKernel<<<grid, block, 0, stream_>>>(device + in.offset, in.width, in.height,
                device + out.offset, out.width, out.height);
            check(cudaGetLastError(), "pyramid kernel");
        }
        execution_.pyramid_input_bytes += bytes; execution_.pyramid_levels += top;
        ++execution_.pyramid_command_buffers; execution_.pyramid_device_ms = -1;
        hostless_.clear();
        if (!fineBatchSize_) {
            for (int layer = 1; layer <= top; ++layer) {
                const size_t levelBytes = size_t(shape[layer].width) * shape[layer].height;
                check(cudaMemcpyAsync(levels[layer].row(0), device + shape[layer].offset, levelBytes,
                    cudaMemcpyDeviceToHost, stream_), "pyramid copyout");
                execution_.pyramid_output_bytes += levelBytes;
            }
            const auto wait = Clock::now();
            check(cudaStreamSynchronize(stream_), "pyramid");
            execution_.pyramid_wait_ms += elapsed(wait);
            sourceReadPending_ = false;
        } else {
            for (int layer = 1; layer <= top; ++layer) {
                hostless_.insert(levels[layer].row(0));
                transfer_.host_levels_skipped_bytes += size_t(shape[layer].width) * shape[layer].height;
            }
            flush();
        }
        for (int layer = 0; layer <= top; ++layer)
            resident_[levels[layer].row(0)] = {device + shape[layer].offset, shape[layer].width, shape[layer].height};
        ++transfer_.lean_prepares;
        transfer_.prepare_copy_ms += copy; transfer_.prepare_enqueue_ms += elapsed(start) - copy;
        execution_.pyramid_wall_ms += elapsed(start);
        *copyMs = copy;
        return true;
    }
    void releaseHostSource() noexcept override {
        if (sourceReadPending_) { cudaEventSynchronize(sourceRead_); sourceReadPending_ = false; }
        if (registered_ && cudaHostUnregister(registered_) != cudaSuccess) cudaGetLastError();
        registered_ = nullptr;
        hostless_.clear();
        transfer_.source_registered = 0; transfer_.registered_bytes = 0;
    }

    // ---- coarse ----
    std::vector<CoarseAngle> computeCoarseDevice(fpm::ImageView top, const fpm::TemplateLevel& templ,
        const std::vector<double>& angles, std::uint8_t border, int count, double overlap, double threshold) override
    {
        return coarseWithCapacity(top, templ, angles, border, count, overlap, threshold,
            coarseMode_ == 8 ? 1024u : UINT_MAX);
    }

    // Sparse mode first reserves 1024 entries per angle. Counts still include
    // every passing window. Replay overflowing angles with the measured capacity.
    std::vector<CoarseAngle> coarseWithCapacity(fpm::ImageView top, const fpm::TemplateLevel& templ,
        const std::vector<double>& angles, std::uint8_t border, int count, double overlap,
        double threshold, unsigned angleCapacity)
    {
        const Range range("fpm coarse");
        const auto start = Clock::now();
        const int tw = templ.image.width(), th = templ.image.height();
        checkTemplate(tw, th);
        if (angles.size() > 65535) throw std::invalid_argument("CUDA coarse search takes at most 65535 angles");
        std::vector<CoarseAngle> results(angles.size());
        std::vector<CoarseJob> jobs;
        std::vector<size_t> jobAngle;
        std::vector<u64> canvasOffsets;
        u64 windows = 0, entrySlots = 0, canvasBytes = 0;
        int maxMapW = 0, maxMapH = 0, maxCanvasW = 0, maxCanvasH = 0;
        for (size_t i = 0; i < angles.size(); ++i) {
            auto& r = results[i];
            const auto map = fpm::detail::canvasWarp(top.width, top.height, tw, th, angles[i], &r.translation);
            r.canvasWidth = map.width; r.canvasHeight = map.height;
            if (map.width < tw || map.height < th) continue;
            checkImage(map.width, map.height);
            const int mw = map.width - tw + 1, mh = map.height - th + 1;
            const auto slots = unsigned(std::min<u64>(u64(mw) * mh, angleCapacity));
            jobs.push_back({deviceWarp(map), mw, mh, entrySlots, slots});
            entrySlots += slots;
            jobAngle.push_back(i);
            canvasOffsets.push_back(canvasBytes);
            windows += u64(mw) * mh; canvasBytes += align16(u64(map.width) * map.height);
            maxMapW = std::max(maxMapW, mw); maxMapH = std::max(maxMapH, mh);
            maxCanvasW = std::max(maxCanvasW, map.width); maxCanvasH = std::max(maxCanvasH, map.height);
        }
        metrics_.score_maps += jobs.size(); metrics_.windows += windows;
        if (jobs.empty() || count <= 0) { metrics_.wall_ms += elapsed(start); return results; }
        const u8* source = residentLevel(top);
        const bool uniform = templ.norm * templ.norm / (double(tw) * th) < DBL_EPSILON;
        const size_t templateBytes = size_t(tw) * th, sharedBytes = templateBytes + size_t(TILE_X + tw - 1) * (TILE_Y + th - 1);
        const bool fused = fusedCoarse_ && sharedBytes <= size_t(sharedOptin_);
        // NCC modes 2-4 run coarseQ28 when its tile fits shared memory; otherwise modes
        // 2 and 3 fall back to mode 0's (or mode 1's) path, which is exact as well.
        const bool q28 = fused && coarseMode_ >= 2 && sharedBytes <= size_t(coarseQ28Limit_);
        const bool coarse32 = coarseMode_ == 4 || coarseMode_ == 7;
        if (coarse32 && !q28) throw std::invalid_argument("CUDA coarse NCC modes 4 and 7 need a tile that fits shared memory");
        std::vector<CoarseQJob> qJobs;
        if (q28) {
            qJobs.reserve(jobs.size());
            for (const auto& job : jobs) {
                const auto& w = job.warp;
                qJobs.push_back({fineWarp({{w.ax, w.ay}, w.c, w.s, w.width, w.height}), job.mapWidth, job.mapHeight, job.entryOffset, job.capacity});
            }
        }
        const ScoreTemplate stats = q28 ? scoreTemplate(templ, threshold, uniform, coarse32) : ScoreTemplate{};
        // Modes 5 and 6 fall back to mode 3, and mode 7 to mode 4, for a uniform template,
        // or when their tile does not fit shared memory or uint32 products could overflow.
        const int windowTileX = COARSE_WARP_X * COARSE_WINDOW_WARPS;
        const size_t paddedWords = size_t((tw + 3) / 4) * th, pitchWords = (windowTileX + tw + 2) / 4;
        const size_t windowShared = 4 * (paddedWords + pitchWords * (TILE_Y + th - 1) + 2 * size_t(windowTileX + tw - 1));
        const bool warpWindows = q28 && coarseMode_ >= 5 && !uniform && u64(tw) * th <= MAX_U32_PIXELS &&
            windowShared <= size_t(coarseWindowsLimit_) && top.width > 1 && top.height > 1;

        // One staging upload: jobs, canvas offsets, template.
        const size_t jobBytes = q28 ? sizeof(CoarseQJob) : sizeof(CoarseJob);
        const size_t jobsBytes = align16(jobs.size() * jobBytes), offsetsBytes = align16(canvasOffsets.size() * sizeof(u64));
        // Modes 5 and 6 read the template as zero-padded words.
        const size_t stagedTemplate = warpWindows ? 4 * paddedWords : templateBytes;
        const size_t stageBytes = jobsBytes + offsetsBytes + align16(stagedTemplate);
        auto* hostStage = static_cast<u8*>(coarseStage_.reserve(stageBytes));
        if (q28) std::memcpy(hostStage, qJobs.data(), qJobs.size() * jobBytes);
        else std::memcpy(hostStage, jobs.data(), jobs.size() * jobBytes);
        std::memcpy(hostStage + jobsBytes, canvasOffsets.data(), canvasOffsets.size() * sizeof(u64));
        u8* stagedRows = hostStage + jobsBytes + offsetsBytes;
        if (warpWindows) {
            const int paddedWidth = 4 * ((tw + 3) / 4);
            std::memset(stagedRows, 0, stagedTemplate);
            for (int y = 0; y < th; ++y) std::memcpy(stagedRows + size_t(y) * paddedWidth, templ.image.row(y), size_t(tw));
        } else {
            std::memcpy(stagedRows, templ.image.row(0), templateBytes);
        }
        // Per-angle counts, then the FP64-pixel and binary64-score counters.
        const size_t countsBytes = align16((jobs.size() + 2) * sizeof(unsigned));
        // Lean calls keep the counts in their own buffer, which every search leaves zero.
        const size_t inlineCounts = leanCalls_ ? 0 : countsBytes;
        auto* device = static_cast<u8*>(coarseDevice_.reserve(stageBytes + inlineCounts + entrySlots * sizeof(Entry) +
            (fused ? 0 : canvasBytes)));
        const auto* dJobs = reinterpret_cast<const CoarseJob*>(device);
        const auto* dOffsets = reinterpret_cast<const u64*>(device + jobsBytes);
        const u8* dTemplate = device + jobsBytes + offsetsBytes;
        auto* dCounts = leanCalls_ ? leanCounts(countsBytes) : reinterpret_cast<unsigned*>(device + stageBytes);
        auto* dEntries = reinterpret_cast<Entry*>(device + stageBytes + inlineCounts);
        u8* dCanvases = device + stageBytes + inlineCounts + entrySlots * sizeof(Entry);
        auto* hostOut = static_cast<u8*>(coarseOut_.reserve(countsBytes + entrySlots * sizeof(Entry)));
        auto* mapped = static_cast<u8*>(coarseOut_.device());
        LeanGuard guard(*this);
        const unsigned signal = leanCalls_ ? nextSignal() : 0;
        const auto encode = Clock::now();
        if (!leanCalls_) check(cudaEventRecord(begin_, stream_), "event");
        check(cudaMemcpyAsync(device, hostStage, stageBytes, cudaMemcpyHostToDevice, stream_), "coarse upload");
        if (!leanCalls_) check(cudaMemsetAsync(dCounts, 0, countsBytes, stream_), "coarse counts");
        const dim3 block(TILE_X, TILE_Y), grid((maxMapW + TILE_X - 1) / TILE_X, (maxMapH + TILE_Y - 1) / TILE_Y, unsigned(jobs.size()));
        const auto* dQJobs = reinterpret_cast<const CoarseQJob*>(device);
        const int jobCount = int(jobs.size());
        if (warpWindows) {
            const dim3 windowBlock(32, COARSE_WINDOW_WARPS);
            const dim3 windowGrid((maxMapW + windowTileX - 1) / windowTileX, (maxMapH + TILE_Y - 1) / TILE_Y, unsigned(jobs.size()));
            const auto* dWords = reinterpret_cast<const unsigned*>(dTemplate);
            if (coarseMode_ == 5) coarseWarpWindows<5><<<windowGrid, windowBlock, windowShared, stream_>>>(source, top.width, top.height,
                dQJobs, dWords, tw, th, templ.mean, templ.norm, threshold, border, stats, dEntries, dCounts, jobCount);
            else if (coarseMode_ == 6 || coarseMode_ == 8) coarseWarpWindows<6><<<windowGrid, windowBlock, windowShared, stream_>>>(source, top.width, top.height,
                dQJobs, dWords, tw, th, templ.mean, templ.norm, threshold, border, stats, dEntries, dCounts, jobCount);
            else coarseWarpWindows<7><<<windowGrid, windowBlock, windowShared, stream_>>>(source, top.width, top.height,
                dQJobs, dWords, tw, th, templ.mean, templ.norm, threshold, border, stats, dEntries, dCounts, jobCount);
        } else if (q28 && coarseMode_ == 2) {
            coarseQ28<2><<<grid, block, sharedBytes, stream_>>>(source, top.width, top.height, dQJobs, dTemplate,
                tw, th, templ.mean, templ.norm, threshold, uniform, border, stats, dEntries, dCounts, jobCount);
        } else if (q28 && !coarse32) {
            coarseQ28<3><<<grid, block, sharedBytes, stream_>>>(source, top.width, top.height, dQJobs, dTemplate,
                tw, th, templ.mean, templ.norm, threshold, uniform, border, stats, dEntries, dCounts, jobCount);
        } else if (q28) {
            coarseQ28<4><<<grid, block, sharedBytes, stream_>>>(source, top.width, top.height, dQJobs, dTemplate,
                tw, th, templ.mean, templ.norm, threshold, uniform, border, stats, dEntries, dCounts, jobCount);
        } else if (fused) {
            coarseFused<<<grid, block, sharedBytes, stream_>>>(source, top.width, top.height, dJobs, dTemplate,
                tw, th, templ.mean, templ.norm, threshold, uniform, border, dEntries, dCounts);
        } else {
            const dim3 canvasGrid((maxCanvasW + 31) / 32, (maxCanvasH + 7) / 8, unsigned(jobs.size()));
            canvasWarpKernel<<<canvasGrid, dim3(32, 8), 0, stream_>>>(source, top.width, top.height, dJobs, dOffsets, dCanvases, border);
            coarseFromCanvas<<<grid, block, 0, stream_>>>(dCanvases, dOffsets, dJobs, dTemplate, tw, th,
                templ.mean, templ.norm, threshold, uniform, dEntries, dCounts);
        }
        check(cudaGetLastError(), "coarse kernel");
        auto* hostEntriesOut = reinterpret_cast<Entry*>(mapped + countsBytes);
        auto* hostCountsOut = reinterpret_cast<unsigned*>(mapped);
        unsigned* done = leanCalls_ ? leanDone() : nullptr;
        unsigned* flag = leanCalls_ ? flagDevice() : nullptr;
        if (q28) packEntries<<<unsigned(jobs.size()), 256, 0, stream_>>>(dEntries, dCounts, dQJobs, hostEntriesOut,
            hostCountsOut, done, flag, signal);
        else packEntries<<<unsigned(jobs.size()), 256, 0, stream_>>>(dEntries, dCounts, dJobs, hostEntriesOut,
            hostCountsOut, done, flag, signal);
        check(cudaGetLastError(), "coarse pack");
        if (!leanCalls_) check(cudaEventRecord(end_, stream_), "event");
        metrics_.upload_ms += elapsed(encode);
        const auto wait = Clock::now();
        if (leanCalls_) waitSignal(signal);
        else check(cudaStreamSynchronize(stream_), "coarse");
        guard.done();
        metrics_.wait_ms += elapsed(wait);
        if (!leanCalls_) metrics_.device_ms += eventMs();
        ++metrics_.command_buffers;
        const auto select = Clock::now();
        const auto* hostCounts = reinterpret_cast<const unsigned*>(hostOut);
        const auto* hostEntries = reinterpret_cast<const Entry*>(hostOut + countsBytes);
        kernel_.coarse_fp64_pixels += hostCounts[jobs.size()];
        kernel_.coarse_fp64_scores += hostCounts[jobs.size() + 1];
        kernel_.coarse_windows += windows;
        std::vector<std::pair<size_t, unsigned>> replay;
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (hostCounts[j] > jobs[j].capacity) {
                replay.emplace_back(jobAngle[j], hostCounts[j]);
                continue;
            }
            std::vector<fpm::detail::Peak> kept(hostCounts[j]);
            const Entry* e = hostEntries + jobs[j].entryOffset;
            for (unsigned k = 0; k < hostCounts[j]; ++k) kept[k] = {e[k].x, e[k].y, e[k].score};
            results[jobAngle[j]].peaks = fpm::detail::selectPeaksSparse(std::move(kept), jobs[j].mapWidth,
                jobs[j].mapHeight, tw, th, count, overlap);
        }
        metrics_.normalization_ms += elapsed(select);
        metrics_.ncc_ms += elapsed(start);
        metrics_.wall_ms += elapsed(start);
        // Replays reuse the mapped buffers. Finish reading all parent results first.
        for (const auto& [index, required] : replay) {
            auto one = coarseWithCapacity(top, templ, {angles[index]}, border, count, overlap, threshold, required);
            results[index] = std::move(one.front());
        }
        return results;
    }

    fpm::detail::ScoreMap compute(fpm::ImageView source, const fpm::TemplateLevel& templ) override {
        const auto start = Clock::now();
        fpm::detail::validateImage(source);
        const int tw = templ.image.width(), th = templ.image.height();
        checkTemplate(tw, th);
        const int mw = source.width - tw + 1, mh = source.height - th + 1;
        if (mw <= 0 || mh <= 0) throw std::invalid_argument("NCC template must fit the source");
        checkImage(source.width, source.height);
        fpm::detail::ScoreMap map{mw, mh, std::vector<double>(size_t(mw) * mh)};
        const size_t imageBytes = size_t(source.width) * source.height, templateBytes = size_t(tw) * th;
        auto* host = static_cast<u8*>(coarseStage_.reserve(align16(imageBytes) + templateBytes));
        for (int y = 0; y < source.height; ++y) std::memcpy(host + size_t(y) * source.width, source.data + y * source.stride, source.width);
        std::memcpy(host + align16(imageBytes), templ.image.row(0), templateBytes);
        auto* device = static_cast<u8*>(coarseDevice_.reserve(align16(imageBytes) + align16(templateBytes) + map.values.size() * sizeof(double)));
        auto* dOut = reinterpret_cast<double*>(device + align16(imageBytes) + align16(templateBytes));
        check(cudaMemcpyAsync(device, host, align16(imageBytes) + templateBytes, cudaMemcpyHostToDevice, stream_), "map upload");
        const bool uniform = templ.norm * templ.norm / (double(tw) * th) < DBL_EPSILON;
        denseScores<<<dim3((mw + 31) / 32, (mh + 7) / 8), dim3(32, 8), 0, stream_>>>(device, source.width, mw, mh,
            device + align16(imageBytes), tw, th, templ.mean, templ.norm, uniform, dOut);
        check(cudaGetLastError(), "map kernel");
        check(cudaMemcpyAsync(map.values.data(), dOut, map.values.size() * sizeof(double), cudaMemcpyDeviceToHost, stream_), "map download");
        check(cudaStreamSynchronize(stream_), "map");
        ++metrics_.score_maps; metrics_.windows += map.values.size(); ++metrics_.command_buffers;
        metrics_.wall_ms += elapsed(start); metrics_.ncc_ms += elapsed(start);
        return map;
    }

    // ---- fine ----
    // Template levels are immutable during one match; their device copies are
    // reused by later fine calls of the same match only.
    void beginFineMatch(const std::vector<fpm::Image>&) override { templates_.clear(); fineStats_.clear(); }
    std::vector<fpm::detail::ScoreMap> computeFineBatch(fpm::ImageView source, const fpm::TemplateLevel& templ,
        const std::vector<fpm::detail::FineRequest>& requests, double* rotationWallMs) override
    {
        if (rotationWallMs) *rotationWallMs = 0;
        if (!fineBatchSize_) throw std::runtime_error("fine CUDA batching was not enabled");
        if (requests.empty()) return {};
        const Range range("fpm fine batch");
        const auto start = Clock::now();
        fpm::detail::validateImage(source);
        const int tw = templ.image.width(), th = templ.image.height();
        checkTemplate(tw, th);
        if (tw > INT_MAX - 2 * PAD || th > INT_MAX - 2 * PAD) throw std::invalid_argument("fine template dimensions exceed ROI limits");
        const int rw = tw + 2 * PAD;
        checkImage(rw, th + 2 * PAD);
        // Odd word counts keep the 32 lanes (rows) of a warp in distinct banks;
        // ROI rows hold two extra words for the dx funnel shifts.
        const bool tensorWanted = fineMode_ >= 5;
        const int tensorThreads = rw >= FINE_SCREEN_MIN_WIDTH ? FINE_WIDE_THREADS : FINE_THREADS;
        int tpitch = (tw + 3) & ~3;
        if ((tpitch / 4) % 2 == 0) tpitch += 4;
        const int pitch = tpitch + 8;
        if (tensorWanted) tpitch = (tw + 31) & ~31;
        auto sharedFor = [&](int rows) {
            return size_t(rows + 2 * PAD) * pitch + size_t(rows) * tpitch + size_t(rows + 2 * PAD) * FINE_SIDE * 2 * sizeof(unsigned) +
                (tensorWanted ? size_t(tpitch) + size_t(tensorThreads / 32) * FINE_WINDOWS * sizeof(unsigned) : 0);
        };
        int stripe = std::min(th, 32);
        while (stripe > 16 && sharedFor(stripe) > size_t(fineSharedTarget_)) --stripe;
        while (stripe > 1 && sharedFor(stripe) > size_t(fineSharedLimit_)) --stripe;
        if (sharedFor(stripe) > size_t(fineSharedLimit_)) throw std::invalid_argument("fine template is too wide for CUDA shared memory");
        // Each pair of a warp adds at most two rows to one of its windows.
        const u64 pairsPerWarp = (u64(stripe + 2 * PAD + 1) / 2 + tensorThreads / 32 - 1) / (tensorThreads / 32);
        const bool tensor = tensorWanted && 2 * pairsPerWarp * tw <= MAX_U32_PIXELS;
        const u8* level = residentLevel(source);
        std::vector<FineWarp> warps;
        warps.reserve(requests.size());
        u64 repairAll = 0;
        for (const auto& r : requests) {
            warps.push_back(fineWarp(fpm::detail::roiWarp(tw, th, r.anchor, r.angle)));
            if (exactFine_) warps.back().repairAll = 1;
            repairAll += warps.back().repairAll != 0;
        }
        const size_t warpsBytes = align16(warps.size() * sizeof(FineWarp));
        auto* host = static_cast<u8*>(fineStage_.reserve(warpsBytes));
        std::memcpy(host, warps.data(), warps.size() * sizeof(FineWarp));
        // Lean calls let the kernel read the requests from the mapped staging buffer,
        // unless transfer bit 4 asks for the upload.
        const bool mappedRequests = leanCalls_ && !uploadRequests_;
        auto* device = mappedRequests ? static_cast<u8*>(fineStage_.device()) : static_cast<u8*>(fineDevice_.reserve(warpsBytes));
        const u8* deviceTemplate = fineTemplate(templ);
        const size_t windows = requests.size() * FINE_WINDOWS;
        auto* acc = static_cast<u64*>(fineAcc_.reserve(windows * 3 * sizeof(u64) + 16));
        auto* repairedCount = reinterpret_cast<unsigned*>(acc + windows * 3);
        auto* out = static_cast<double*>(fineOut_.reserve(windows * sizeof(double) + 16));
        auto* mappedOut = static_cast<double*>(fineOut_.device());
        const bool uniform = templ.norm * templ.norm / (double(tw) * th) < DBL_EPSILON;
        LeanGuard guard(*this);
        const unsigned signal = leanCalls_ ? nextSignal() : 0;
        if (leanCalls_ && leanAccZeroed_ != fineAcc_.capacity()) {
            check(cudaMemsetAsync(acc, 0, fineAcc_.capacity(), stream_), "fine accumulators");
            leanAccZeroed_ = fineAcc_.capacity();
        }
        const auto encode = Clock::now();
        if (!leanCalls_) {
            check(cudaEventRecord(begin_, stream_), "event");
            check(cudaMemcpyAsync(device, host, warpsBytes, cudaMemcpyHostToDevice, stream_), "fine upload");
            check(cudaMemsetAsync(acc, 0, windows * 3 * sizeof(u64) + 16, stream_), "fine accumulators");
        } else if (!mappedRequests) {
            check(cudaMemcpyAsync(device, host, warpsBytes, cudaMemcpyHostToDevice, stream_), "fine upload");
        }
        if (!uniform) {
            const dim3 grid((th + stripe - 1) / stripe, unsigned(requests.size()));
            const auto* dWarps = reinterpret_cast<const FineWarp*>(device);
            if (tensor && fineMode_ == 7 && rw >= FINE_SCREEN_MIN_WIDTH) fineStripes<5, true><<<grid, tensorThreads, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if (tensor && fineMode_ == 7) fineStripes<6, true><<<grid, tensorThreads, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if (tensor && fineMode_ == 6) fineStripes<7><<<grid, tensorThreads, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if (tensor && rw >= FINE_SCREEN_MIN_WIDTH) fineStripes<5><<<grid, tensorThreads, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if (tensor) fineStripes<6><<<grid, tensorThreads, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if ((fineMode_ == 4 || fineMode_ == 5 || fineMode_ == 7) && rw >= FINE_SCREEN_MIN_WIDTH) fineStripes<4><<<grid, FINE_THREADS, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if (fineMode_ == 2 || fineMode_ == 4 || fineMode_ == 5 || fineMode_ == 7) fineStripes<2><<<grid, FINE_THREADS, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else if (fineMode_ == 3 || fineMode_ == 6) fineStripes<3><<<grid, FINE_THREADS, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            else fineStripes<0><<<grid, FINE_THREADS, sharedFor(stripe), stream_>>>(level, source.width,
                source.height, dWarps, deviceTemplate, tw, th, stripe, pitch, tpitch, acc, repairedCount);
            check(cudaGetLastError(), "fine kernel");
        }
        unsigned* done = leanCalls_ ? leanDone() + 1 : nullptr;
        unsigned* flag = leanCalls_ ? flagDevice() : nullptr;
        const unsigned normalizeBlocks = unsigned((windows + 255) / 256);
        if (fineMode_ == 3 || fineMode_ == 6) fineNormalize<true><<<normalizeBlocks, 256, 0, stream_>>>(acc, int(windows), templ.mean,
            templ.norm, double(tw) * th, uniform, repairedCount, mappedOut, done, flag, signal, fineScoreTemplate(templ, uniform));
        else fineNormalize<false><<<normalizeBlocks, 256, 0, stream_>>>(acc, int(windows), templ.mean, templ.norm,
            double(tw) * th, uniform, repairedCount, mappedOut, done, flag, signal, ScoreTemplate{});
        check(cudaGetLastError(), "fine normalize");
        if (!leanCalls_) check(cudaEventRecord(end_, stream_), "event");
        fine_.encode_ms += elapsed(encode);
        const auto wait = Clock::now();
        if (leanCalls_) waitSignal(signal);
        else check(cudaStreamSynchronize(stream_), "fine");
        guard.done();
        fine_.wait_ms += elapsed(wait);
        if (!leanCalls_) fine_.device_ms += eventMs();
        ++fine_.calls; ++fine_.command_buffers;
        fine_.maps += requests.size();
        fine_.roi_pixels += u64(requests.size()) * rw * (th + 2 * PAD);
        fine_.peak_batch_maps = std::max<u64>(fine_.peak_batch_maps, requests.size());
        fine_.repair_all_maps += repairAll;
        fine_.repair_pixels += *reinterpret_cast<const unsigned*>(out + windows);
        std::vector<fpm::detail::ScoreMap> maps;
        maps.reserve(requests.size());
        for (size_t i = 0; i < requests.size(); ++i)
            maps.push_back({FINE_SIDE, FINE_SIDE, std::vector<double>(out + i * FINE_WINDOWS, out + (i + 1) * FINE_WINDOWS)});
        fine_.wall_ms += elapsed(start);
        return maps;
    }

private:
    struct Resident { const u8* data; int width, height; };
    static void checkTemplate(int tw, int th) {
        if (tw < 1 || th < 1) throw std::invalid_argument("template is empty");
        if (u64(tw) > MAX_U32_PIXELS) throw std::invalid_argument("template rows exceed exact CUDA row statistics");
        if (u64(tw) * th > (u64(1) << 53) / 65025) throw std::invalid_argument("template exceeds exact NCC statistics range");
        if (th > 65536) throw std::invalid_argument("template is taller than the exact CPU NCC path");
    }
    // Images the CPU scores with its exact integer path (computeNcc's integerRange).
    static void checkImage(int width, int height) {
        if (double(width) * height > 9007199254740992.0 / 65025)
            throw std::invalid_argument("image exceeds the exact NCC statistics range");
    }
    double eventMs() {
        float ms = 0;
        check(cudaEventElapsedTime(&ms, begin_, end_), "event time");
        return ms;
    }
    // Device copy of a host image: a pyramid level from buildPyramid(), or a
    // cached upload, valid until invalidateFineSource().
    const u8* residentLevel(fpm::ImageView image) {
        auto it = resident_.find(image.data);
        if (it != resident_.end() && it->second.width == image.width && it->second.height == image.height &&
            image.stride == size_t(image.width)) return it->second.data;
        if (hostless_.count(image.data)) throw std::logic_error("CUDA lean prepare left this pyramid level on the device only");
        const Range range("fpm source upload");
        auto& buffer = uploaded_[image.data];
        const size_t bytes = size_t(image.width) * image.height;
        auto* device = static_cast<u8*>(buffer.reserve(bytes));
        check(cudaMemcpy2DAsync(device, image.width, image.data, image.stride, image.width, image.height,
            cudaMemcpyHostToDevice, stream_), "source upload");
        check(cudaStreamSynchronize(stream_), "source upload");
        uploadedBytes_ += bytes;
        ++fine_.source_uploads; fine_.source_upload_bytes += bytes;
        resident_[image.data] = {device, image.width, image.height};
        return device;
    }
    const u8* fineTemplate(const fpm::TemplateLevel& templ) {
        const u8* key = templ.image.row(0);
        auto it = templates_.find(key);
        if (it != templates_.end()) return it->second.data;
        const size_t bytes = size_t(templ.image.width()) * templ.image.height();
        auto& buffer = templateBuffers_[templates_.size() % TEMPLATE_SLOTS];
        auto* device = static_cast<u8*>(buffer.reserve(bytes));
        check(cudaMemcpyAsync(device, key, bytes, cudaMemcpyHostToDevice, stream_), "fine template upload");
        if (templates_.size() >= TEMPLATE_SLOTS) templates_.clear();
        templates_[key] = {device, templ.image.width(), templ.image.height()};
        return device;
    }
    size_t retainedDeviceBytes() const {
        size_t bytes = pyramid_.capacity() + coarseDevice_.capacity() + fineDevice_.capacity() + fineAcc_.capacity();
        bytes += leanCounts_.capacity() + leanDone_.capacity();
        for (const auto& item : uploaded_) bytes += item.second.capacity();
        for (const auto& buffer : templateBuffers_) bytes += buffer.capacity();
        return bytes;
    }

    // ---- binary32 scores ----
    // Exact template sums for binary32 scores. Screening (coarse NCC mode 3) is
    // enabled only where its margin is proven: a template that is not uniform (then
    // norm >= sqrt(1/2) for integer pixels; required >= 0.5) with n <= SCORE32_MAX_PIXELS
    // pixels. In the windows it screens (variance above 4x normalizeScore()'s limit)
    // the binary64 score is within 2^-53*255^2*n/norm (the S*mean rounding) plus
    // 2^-52*sum(x^2)/variance (the variance cancellation) of the exact NCC, below
    // 1.9e-5, and the binary32 score within 6*2^-24 of it, so a window below
    // threshold - SCREEN_MARGIN in binary32 is below the threshold in binary64.
    static ScoreTemplate scoreTemplate(const fpm::TemplateLevel& templ, double threshold, bool uniform, bool required) {
        const int tw = templ.image.width(), th = templ.image.height();
        const u64 n = u64(tw) * th;
        ScoreTemplate s{};
        if (n > SCORE32_MAX_PIXELS) {
            if (required) throw std::invalid_argument("binary32 CUDA scores need a template of at most 1e6 pixels");
            return s;
        }
        u64 sum = 0, square = 0;
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) { const u64 v = templ.image.row(y)[x]; sum += v; square += v * v; }
        s.sum = sum;
        s.sqrtVariance = float(std::sqrt(double(n * square - sum * sum)));
        const double cut = threshold - SCREEN_MARGIN;
        float drop = float(cut);
        if (double(drop) > cut) drop = std::nextafter(drop, -std::numeric_limits<float>::infinity());
        s.dropBelow = drop;
        s.enabled = !uniform && templ.norm >= 0.5;
        return s;
    }
    const ScoreTemplate& fineScoreTemplate(const fpm::TemplateLevel& templ, bool uniform) {
        auto it = fineStats_.find(templ.image.row(0));
        if (it == fineStats_.end()) it = fineStats_.emplace(templ.image.row(0), scoreTemplate(templ, 0, uniform, true)).first;
        return it->second;
    }

    // ---- lean calls ----
    // A lean call that fails may leave its counters dirty; the next one clears them.
    struct LeanGuard {
        explicit LeanGuard(CudaBackend& owner) : backend(owner) {}
        ~LeanGuard() {
            if (finished || !backend.leanCalls_) return;
            backend.leanCountsZeroed_ = backend.leanAccZeroed_ = 0;
            backend.leanDoneDirty_ = true;
        }
        void done() { finished = true; }
        CudaBackend& backend;
        bool finished = false;
    };
    unsigned nextSignal() {
        if (leanDoneDirty_) {
            check(cudaMemsetAsync(leanDone(), 0, leanDone_.capacity(), stream_), "lean counters");
            leanDoneDirty_ = false;
        }
        if (++signalValue_ == 0) ++signalValue_;
        return signalValue_;
    }
    unsigned* leanDone() { return static_cast<unsigned*>(leanDone_.reserve(16)); }
    unsigned* flagDevice() const { return flagDevice_; }
    unsigned* leanCounts(size_t bytes) {
        auto* counts = static_cast<unsigned*>(leanCounts_.reserve(bytes));
        if (leanCountsZeroed_ != leanCounts_.capacity()) {
            check(cudaMemsetAsync(counts, 0, leanCounts_.capacity(), stream_), "coarse counts");
            leanCountsZeroed_ = leanCounts_.capacity();
        }
        return counts;
    }
    // Submits the queued commands (WDDM batches them until a query or a wait).
    void flush() {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaErrorNotReady) cudaGetLastError();
        else check(status, "stream");
    }
    // Spins on the mapped flag the call's last kernel writes; an occasional
    // stream query reports asynchronous errors.
    void waitSignal(unsigned value) {
        const auto start = Clock::now();
        flush();
        for (unsigned spins = 1; flag_->load(std::memory_order_acquire) != value; ++spins) {
            if (spins % 4096) continue;
            ++transfer_.signal_queries;
            const cudaError_t status = cudaStreamQuery(stream_);
            if (status == cudaSuccess) {
                if (flag_->load(std::memory_order_acquire) == value) break;
                throw std::runtime_error("CUDA lean call finished without its completion signal");
            }
            if (status != cudaErrorNotReady) check(status, "lean wait");
            cudaGetLastError();
        }
        ++transfer_.signal_waits;
        transfer_.signal_wait_ms += elapsed(start);
    }
    void waitSourceRead() {
        if (!sourceReadPending_) return;
        check(cudaEventSynchronize(sourceRead_), "source read");
        sourceReadPending_ = false;
    }

    int sharedOptin_ = 0, fineSharedLimit_ = 0, fineSharedTarget_ = 0, pyramidMode_ = 0, coarseMode_ = 0, fineMode_ = 0;
    int coarseQ28Limit_ = 0, coarseWindowsLimit_ = 0;
    bool tensorCores_ = false;
    bool fusedCoarse_ = true, exactFine_ = false;
    bool leanPrepare_ = false, leanCalls_ = false, uploadRequests_ = false, sourceReadPending_ = false, leanDoneDirty_ = false;
    std::unordered_map<const void*, ScoreTemplate> fineStats_;
    FpmCudaKernelMetricsV1 kernel_{};
    CopyPool copyPool_;
    int copyThreads_ = 1;
    size_t fineBatchSize_ = 0, uploadedBytes_ = 0, leanCountsZeroed_ = 0, leanAccZeroed_ = 0;
    unsigned signalValue_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr, sourceRead_ = nullptr;
    DeviceBytes pyramid_, coarseDevice_, fineDevice_, fineAcc_, leanCounts_, leanDone_;
    HostBytes coarseStage_, coarseOut_, fineStage_, fineOut_, signal_;
    std::atomic<unsigned>* flag_ = nullptr;
    unsigned* flagDevice_ = nullptr;
    void* registered_ = nullptr;
    // Host levels lean prepare did not fill; residentLevel() never uploads them.
    std::unordered_set<const void*> hostless_;
    FpmTransferMetricsV1 transfer_{};
    std::unordered_map<const void*, Resident> resident_;
    std::unordered_map<const void*, DeviceBytes> uploaded_;
    static constexpr size_t TEMPLATE_SLOTS = 16;
    DeviceBytes templateBuffers_[TEMPLATE_SLOTS];
    std::unordered_map<const void*, Resident> templates_;
    FpmCoarseMetrics metrics_{};
    FpmFineMetricsV1 fine_{};
    FpmCoarseExecutionMetricsV1 execution_{};
};
}

std::unique_ptr<fpm_coarse_detail::Backend> fpm_coarse_detail::makeCudaBackend()
{
    return std::make_unique<CudaBackend>();
}
