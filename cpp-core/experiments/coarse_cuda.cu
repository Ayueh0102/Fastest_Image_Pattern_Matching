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
#include "coarse_internal.hpp"
#include "fine_sampling.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using u8 = unsigned char;
using u64 = unsigned long long;
using Clock = std::chrono::steady_clock;
constexpr int PAD = fpm::detail::PADDING, FINE_SIDE = 2 * PAD + 1, FINE_WINDOWS = FINE_SIDE * FINE_SIDE;
constexpr int TILE_X = 32, TILE_Y = 8, COARSE_THREADS = TILE_X * TILE_Y, FINE_THREADS = 256;
// Per-row and per-lane uint32 statistics stay exact while count * 65025 < 2^32.
constexpr u64 MAX_U32_PIXELS = 66051;

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
struct CoarseJob { DeviceWarp warp; int mapWidth, mapHeight; u64 entryOffset; };
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
__device__ __forceinline__ void appendEntry(bool keep, int x, int y, double score, unsigned* count, Entry* entries)
{
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    if (!mask) return;
    const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
    unsigned base = 0;
    if (lane == leader) base = atomicAdd(count, unsigned(__popc(mask)));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (keep) entries[base + __popc(mask & ((1u << lane) - 1))] = {x, y, score};
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
    appendEntry(keep, x, y, score, counts + blockIdx.z, entries + job.entryOffset);
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
    appendEntry(keep, x, y, score, counts + blockIdx.z, entries + job.entryOffset);
}

// Copies counts and each angle's kept entries to mapped host memory (one block per angle).
__global__ void packEntries(const Entry* entries, const unsigned* counts, const CoarseJob* jobs,
    Entry* hostEntries, unsigned* hostCounts)
{
    const unsigned n = counts[blockIdx.x];
    const u64 offset = jobs[blockIdx.x].entryOffset;
    if (threadIdx.x == 0) hostCounts[blockIdx.x] = n;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) hostEntries[offset + i] = entries[offset + i];
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

__device__ __forceinline__ unsigned warpSum32(unsigned v)
{
    for (int offset = 16; offset; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Fine search, one (template row stripe of at most 32 rows, request) per block of
// 8 warps. The stripe's ROI rows are sampled into shared memory (rows of `pitch`
// bytes, template rows of `tpitch`, both odd word counts). Warp dy < 7 computes
// the template products of window row dy: lane r takes template row r and keeps
// the seven dx sums, four pixels per __dp4a. Warp 7 computes each ROI row's sums
// and squares for the seven dx offsets. Per-lane uint32 sums are exact while
// tw <= 66051; block totals are added to acc.
__global__ void fineStripes(const u8* source, int sw, int sh, const FineWarp* requests, const u8* templ,
    int tw, int th, int stripeRows, int pitch, int tpitch, u64* acc, unsigned* repairedTotal)
{
    extern __shared__ __align__(16) u8 shared[];
    __shared__ unsigned blockRepaired;
    const int rw = tw + 2 * PAD, r0 = blockIdx.x * stripeRows, rows = min(stripeRows, th - r0), roiRows = rows + 2 * PAD;
    u8* roi = shared;
    u8* t = shared + size_t(roiRows) * pitch;
    unsigned* rowStats = reinterpret_cast<unsigned*>(t + size_t(rows) * tpitch);
    if (threadIdx.x == 0) blockRepaired = 0;
    __syncthreads();
    const FineWarp m = requests[blockIdx.y];
    unsigned repaired = 0;
    for (int i = threadIdx.x; i < roiRows * rw; i += blockDim.x) {
        const int x = i % rw, y = i / rw;
        roi[size_t(y) * pitch + x] = finePixel(source, sw, sh, x, r0 + y, m, repaired);
    }
    for (int i = threadIdx.x; i < rows * tpitch; i += blockDim.x) {
        const int y = i / tpitch, x = i % tpitch;
        t[i] = x < tw ? templ[size_t(r0 + y) * tw + x] : 0;
    }
    if (repaired) atomicAdd(&blockRepaired, repaired);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u64* out = acc + size_t(blockIdx.y) * FINE_WINDOWS * 3;
    if (warp < FINE_SIDE) {
        unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
        if (lane < rows) {
            const auto* rowWords = reinterpret_cast<const unsigned*>(roi + size_t(lane + warp) * pitch);
            const auto* templateWords = reinterpret_cast<const unsigned*>(t + size_t(lane) * tpitch);
            unsigned w0 = rowWords[0], w1 = rowWords[1];
            for (int j = 0, words = tpitch / 4; j < words; ++j) {
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
            if (lane == 0) atomicAdd(out + (warp * FINE_SIDE + dx) * 3 + 2, product);
        }
    } else if (warp == FINE_SIDE) {
        const int full = tw / 4;
        for (int rho = 0; rho < roiRows; ++rho) {
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
    }
    if (threadIdx.x == 0 && blockRepaired) atomicAdd(repairedTotal, blockRepaired);
}
// Scores to mapped host memory, followed by the count of FP64-sampled pixels.
__global__ void fineNormalize(const u64* acc, int windows, double mean, double norm, double area, int uniform,
    const unsigned* repaired, double* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < windows) out[i] = uniform ? 1.0 : normalizeScore(acc[3 * i], acc[3 * i + 1], acc[3 * i + 2], mean, norm, area);
    if (i == 0) *reinterpret_cast<unsigned*>(out + windows) = uniform ? 0 : *repaired;
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
    size_t capacity() const { return capacity_; }
private:
    void* data_ = nullptr;
    size_t capacity_ = 0;
};

size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

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
        cudaFuncAttributes fine{};
        check(cudaFuncGetAttributes(&fine, fineStripes), "fine attributes");
        fineSharedLimit_ = sharedOptin_ - int(fine.sharedSizeBytes);
        // Prefer stripes small enough for two resident blocks per SM.
        int perSm = 0;
        check(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0), "SM shared");
        fineSharedTarget_ = std::min(fineSharedLimit_, perSm / 2 - 1024 - int(fine.sharedSizeBytes));
        check(cudaFuncSetAttribute(fineStripes, cudaFuncAttributeMaxDynamicSharedMemorySize, fineSharedLimit_), "fine shared");
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
        check(cudaEventCreate(&begin_), "event"); check(cudaEventCreate(&end_), "event");
        resetMetrics();
    }
    ~CudaBackend() override {
        cudaEventDestroy(begin_); cudaEventDestroy(end_);
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
    void setCoarseOptimization(int peakMode, int nccMode) override {
        if (peakMode != 0 || nccMode < 0 || nccMode > 1)
            throw std::invalid_argument("CUDA coarse optimization takes peak mode 0 and NCC mode 0 (fused) or 1 (canvas)");
        fusedCoarse_ = nccMode == 0;
    }
    bool deviceCoarse() const override { return true; }

    // ---- metrics ----
    void resetMetrics() override {
        metrics_ = {}; metrics_.device_ms = 0;
        fine_ = {}; fine_.version = 1; fine_.struct_bytes = sizeof(fine_); fine_.device_ms = 0;
        execution_ = {}; execution_.version = 1; execution_.struct_bytes = sizeof(execution_);
        execution_.pyramid_mode = pyramidMode_; execution_.batch_angles = 1; execution_.pyramid_device_ms = 0;
    }
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

    // ---- coarse ----
    std::vector<CoarseAngle> computeCoarseDevice(fpm::ImageView top, const fpm::TemplateLevel& templ,
        const std::vector<double>& angles, std::uint8_t border, int count, double overlap, double threshold) override
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
        u64 windows = 0, canvasBytes = 0;
        int maxMapW = 0, maxMapH = 0, maxCanvasW = 0, maxCanvasH = 0;
        for (size_t i = 0; i < angles.size(); ++i) {
            auto& r = results[i];
            const auto map = fpm::detail::canvasWarp(top.width, top.height, tw, th, angles[i], &r.translation);
            r.canvasWidth = map.width; r.canvasHeight = map.height;
            if (map.width < tw || map.height < th) continue;
            checkImage(map.width, map.height);
            const int mw = map.width - tw + 1, mh = map.height - th + 1;
            jobs.push_back({deviceWarp(map), mw, mh, windows});
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

        // One staging upload: jobs, canvas offsets, template.
        const size_t jobsBytes = align16(jobs.size() * sizeof(CoarseJob)), offsetsBytes = align16(canvasOffsets.size() * sizeof(u64));
        const size_t stageBytes = jobsBytes + offsetsBytes + align16(templateBytes);
        auto* hostStage = static_cast<u8*>(coarseStage_.reserve(stageBytes));
        std::memcpy(hostStage, jobs.data(), jobs.size() * sizeof(CoarseJob));
        std::memcpy(hostStage + jobsBytes, canvasOffsets.data(), canvasOffsets.size() * sizeof(u64));
        std::memcpy(hostStage + jobsBytes + offsetsBytes, templ.image.row(0), templateBytes);
        const size_t countsBytes = align16(jobs.size() * sizeof(unsigned));
        auto* device = static_cast<u8*>(coarseDevice_.reserve(stageBytes + countsBytes + windows * sizeof(Entry) +
            (fused ? 0 : canvasBytes)));
        const auto* dJobs = reinterpret_cast<const CoarseJob*>(device);
        const auto* dOffsets = reinterpret_cast<const u64*>(device + jobsBytes);
        const u8* dTemplate = device + jobsBytes + offsetsBytes;
        auto* dCounts = reinterpret_cast<unsigned*>(device + stageBytes);
        auto* dEntries = reinterpret_cast<Entry*>(device + stageBytes + countsBytes);
        u8* dCanvases = device + stageBytes + countsBytes + windows * sizeof(Entry);
        auto* hostOut = static_cast<u8*>(coarseOut_.reserve(countsBytes + windows * sizeof(Entry)));
        auto* mapped = static_cast<u8*>(coarseOut_.device());
        const auto encode = Clock::now();
        check(cudaEventRecord(begin_, stream_), "event");
        check(cudaMemcpyAsync(device, hostStage, stageBytes, cudaMemcpyHostToDevice, stream_), "coarse upload");
        check(cudaMemsetAsync(dCounts, 0, countsBytes, stream_), "coarse counts");
        const dim3 block(TILE_X, TILE_Y), grid((maxMapW + TILE_X - 1) / TILE_X, (maxMapH + TILE_Y - 1) / TILE_Y, unsigned(jobs.size()));
        if (fused) {
            coarseFused<<<grid, block, sharedBytes, stream_>>>(source, top.width, top.height, dJobs, dTemplate,
                tw, th, templ.mean, templ.norm, threshold, uniform, border, dEntries, dCounts);
        } else {
            const dim3 canvasGrid((maxCanvasW + 31) / 32, (maxCanvasH + 7) / 8, unsigned(jobs.size()));
            canvasWarpKernel<<<canvasGrid, dim3(32, 8), 0, stream_>>>(source, top.width, top.height, dJobs, dOffsets, dCanvases, border);
            coarseFromCanvas<<<grid, block, 0, stream_>>>(dCanvases, dOffsets, dJobs, dTemplate, tw, th,
                templ.mean, templ.norm, threshold, uniform, dEntries, dCounts);
        }
        check(cudaGetLastError(), "coarse kernel");
        packEntries<<<unsigned(jobs.size()), 256, 0, stream_>>>(dEntries, dCounts, dJobs,
            reinterpret_cast<Entry*>(mapped + countsBytes), reinterpret_cast<unsigned*>(mapped));
        check(cudaGetLastError(), "coarse pack");
        check(cudaEventRecord(end_, stream_), "event");
        metrics_.upload_ms += elapsed(encode);
        const auto wait = Clock::now();
        check(cudaStreamSynchronize(stream_), "coarse");
        metrics_.wait_ms += elapsed(wait);
        metrics_.device_ms += eventMs();
        ++metrics_.command_buffers;
        const auto select = Clock::now();
        const auto* hostCounts = reinterpret_cast<const unsigned*>(hostOut);
        const auto* hostEntries = reinterpret_cast<const Entry*>(hostOut + countsBytes);
        for (size_t j = 0; j < jobs.size(); ++j) {
            std::vector<fpm::detail::Peak> kept(hostCounts[j]);
            const Entry* e = hostEntries + jobs[j].entryOffset;
            for (unsigned k = 0; k < hostCounts[j]; ++k) kept[k] = {e[k].x, e[k].y, e[k].score};
            results[jobAngle[j]].peaks = fpm::detail::selectPeaksSparse(std::move(kept), jobs[j].mapWidth,
                jobs[j].mapHeight, tw, th, count, overlap);
        }
        metrics_.normalization_ms += elapsed(select);
        metrics_.ncc_ms += elapsed(start);
        metrics_.wall_ms += elapsed(start);
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
    void beginFineMatch(const std::vector<fpm::Image>&) override { templates_.clear(); }
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
        int tpitch = (tw + 3) & ~3;
        if ((tpitch / 4) % 2 == 0) tpitch += 4;
        const int pitch = tpitch + 8;
        auto sharedFor = [&](int rows) {
            return size_t(rows + 2 * PAD) * pitch + size_t(rows) * tpitch + size_t(rows + 2 * PAD) * FINE_SIDE * 2 * sizeof(unsigned);
        };
        int stripe = std::min(th, 32);
        while (stripe > 16 && sharedFor(stripe) > size_t(fineSharedTarget_)) --stripe;
        while (stripe > 1 && sharedFor(stripe) > size_t(fineSharedLimit_)) --stripe;
        if (sharedFor(stripe) > size_t(fineSharedLimit_)) throw std::invalid_argument("fine template is too wide for CUDA shared memory");
        const u8* level = residentLevel(source);
        std::vector<FineWarp> warps;
        warps.reserve(requests.size());
        u64 repairAll = 0;
        for (const auto& r : requests) {
            warps.push_back(fineWarp(fpm::detail::roiWarp(tw, th, r.anchor, r.angle)));
            repairAll += warps.back().repairAll != 0;
        }
        const size_t warpsBytes = align16(warps.size() * sizeof(FineWarp));
        auto* host = static_cast<u8*>(fineStage_.reserve(warpsBytes));
        std::memcpy(host, warps.data(), warps.size() * sizeof(FineWarp));
        auto* device = static_cast<u8*>(fineDevice_.reserve(warpsBytes));
        const u8* deviceTemplate = fineTemplate(templ);
        const size_t windows = requests.size() * FINE_WINDOWS;
        auto* acc = static_cast<u64*>(fineAcc_.reserve(windows * 3 * sizeof(u64) + 16));
        auto* repairedCount = reinterpret_cast<unsigned*>(acc + windows * 3);
        auto* out = static_cast<double*>(fineOut_.reserve(windows * sizeof(double) + 16));
        auto* mappedOut = static_cast<double*>(fineOut_.device());
        const bool uniform = templ.norm * templ.norm / (double(tw) * th) < DBL_EPSILON;
        const auto encode = Clock::now();
        check(cudaEventRecord(begin_, stream_), "event");
        check(cudaMemcpyAsync(device, host, warpsBytes, cudaMemcpyHostToDevice, stream_), "fine upload");
        check(cudaMemsetAsync(acc, 0, windows * 3 * sizeof(u64) + 16, stream_), "fine accumulators");
        if (!uniform) {
            fineStripes<<<dim3((th + stripe - 1) / stripe, unsigned(requests.size())), FINE_THREADS, sharedFor(stripe), stream_>>>(
                level, source.width, source.height, reinterpret_cast<const FineWarp*>(device), deviceTemplate,
                tw, th, stripe, pitch, tpitch, acc, repairedCount);
            check(cudaGetLastError(), "fine kernel");
        }
        fineNormalize<<<unsigned((windows + 255) / 256), 256, 0, stream_>>>(acc, int(windows), templ.mean, templ.norm,
            double(tw) * th, uniform, repairedCount, mappedOut);
        check(cudaGetLastError(), "fine normalize");
        check(cudaEventRecord(end_, stream_), "event");
        fine_.encode_ms += elapsed(encode);
        const auto wait = Clock::now();
        check(cudaStreamSynchronize(stream_), "fine");
        fine_.wait_ms += elapsed(wait);
        fine_.device_ms += eventMs();
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
        for (const auto& item : uploaded_) bytes += item.second.capacity();
        return bytes;
    }

    int sharedOptin_ = 0, fineSharedLimit_ = 0, fineSharedTarget_ = 0, pyramidMode_ = 0;
    bool fusedCoarse_ = true;
    size_t fineBatchSize_ = 0, uploadedBytes_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
    DeviceBytes pyramid_, coarseDevice_, fineDevice_, fineAcc_;
    HostBytes coarseStage_, coarseOut_, fineStage_, fineOut_;
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
