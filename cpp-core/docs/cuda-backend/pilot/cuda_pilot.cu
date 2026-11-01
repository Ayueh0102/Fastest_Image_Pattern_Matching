// SPDX-License-Identifier: BSD-2-Clause
// Exploratory CUDA pilot, not a formal timing. For one fixture it checks that
// FP64 warps, exact integer NCC statistics and the exact pyrDown on the GPU
// reproduce fpm_core's rotateCanvas/rotatedRoi/ncc/pyrDown byte for byte, and
// times kernels (cudaEvent), transfers and host round trips on this PC.
//
// usage: cuda_pilot SRC.pgm TMPL.pgm MIN_AREA TOLERANCE [ROIS]
// prints one JSON object.
#include "pilot.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

using u8 = unsigned char;
using u64 = unsigned long long;
constexpr int PADDING = 3, FINE = 2 * PADDING + 1;

// ---- device arithmetic in the order of the x86 (unfused) core ------------
__device__ __forceinline__ u8 pixelAt(const u8* s, int w, int h, int x, int y, u8 border)
{
    return x < 0 || y < 0 || x >= w || y >= h ? border : s[size_t(y) * w + x];
}

// warpPixel(): sx=(ax+c*x)-s*y, sy=(ay+s*x)+c*y, bilinear, floor(v+0.5), clamp.
__device__ u8 warpPixel(const u8* s, int w, int h, int x, int y, const WarpJob& j, u8 border)
{
    const double fx = x, fy = y;
    const double sx = __dsub_rn(__dadd_rn(j.ax, __dmul_rn(j.c, fx)), __dmul_rn(j.s, fy));
    const double sy = __dadd_rn(__dadd_rn(j.ay, __dmul_rn(j.s, fx)), __dmul_rn(j.c, fy));
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

// nccDenominator() + nccNormalizeProduct() on exact integer statistics.
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

// ---- kernels --------------------------------------------------------------
__global__ void warpKernel(const u8* src, int sw, int sh, const WarpJob* jobs, u8* out, u8 border)
{
    const WarpJob j = jobs[blockIdx.z];
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= j.width || y >= j.height) return;
    out[j.offset + size_t(y) * j.width + x] = warpPixel(src, sw, sh, x, y, j, border);
}

// One thread per window, template in shared memory; row sums stay below 2^32.
__global__ void nccNaive(const u8* images, const WarpJob* jobs, const u64* mapOffsets, const u8* templ,
    int tw, int th, double mean, double norm, double* maps)
{
    extern __shared__ u8 t[];
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tw * th; i += blockDim.x * blockDim.y) t[i] = templ[i];
    __syncthreads();
    const WarpJob j = jobs[blockIdx.z];
    const int mw = j.width - tw + 1, mh = j.height - th + 1;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mw || y >= mh) return;
    const u8* p = images + j.offset + size_t(y) * j.width + x;
    u64 sum = 0, square = 0, product = 0;
    for (int r = 0; r < th; ++r) {
        unsigned rs = 0, rq = 0, rd = 0;
        const u8* row = p + size_t(r) * j.width;
        const u8* tr = t + r * tw;
        for (int k = 0; k < tw; ++k) { const unsigned v = row[k]; rs += v; rq += v * v; rd += v * tr[k]; }
        sum += rs; square += rq; product += rd;
    }
    maps[mapOffsets[blockIdx.z] + size_t(y) * mw + x] =
        normalizeScore(sum, square, product, mean, norm, double(tw) * th);
}

__device__ __forceinline__ u64 warpSum(u64 v)
{
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Fine maps: one block per (window, ROI); threads stride over template pixels.
__global__ void fineNaive(const u8* rois, const WarpJob* jobs, const u8* templ, int tw, int th,
    double mean, double norm, double* maps)
{
    __shared__ u64 partial[3][32];
    const WarpJob j = jobs[blockIdx.y];
    const int window = blockIdx.x, wx = window % FINE, wy = window / FINE;
    const u8* p = rois + j.offset + size_t(wy) * j.width + wx;
    u64 sum = 0, square = 0, product = 0;
    for (int r = threadIdx.y; r < th; r += blockDim.y) {
        unsigned rs = 0, rq = 0, rd = 0;
        const u8* row = p + size_t(r) * j.width;
        const u8* tr = templ + size_t(r) * tw;
        for (int k = threadIdx.x; k < tw; k += blockDim.x) { const unsigned v = row[k]; rs += v; rq += v * v; rd += v * tr[k]; }
        sum += rs; square += rq; product += rd;
    }
    sum = warpSum(sum); square = warpSum(square); product = warpSum(product);
    const int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31, warp = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) { partial[0][warp] = sum; partial[1][warp] = square; partial[2][warp] = product; }
    __syncthreads();
    if (warp == 0) {
        const int warps = (blockDim.x * blockDim.y + 31) / 32;
        sum = lane < warps ? partial[0][lane] : 0; square = lane < warps ? partial[1][lane] : 0;
        product = lane < warps ? partial[2][lane] : 0;
        sum = warpSum(sum); square = warpSum(square); product = warpSum(product);
        if (lane == 0) maps[size_t(blockIdx.y) * FINE * FINE + window] =
            normalizeScore(sum, square, product, mean, norm, double(tw) * th);
    }
}

__device__ __forceinline__ int reflect101(int p, int n)
{
    if (n == 1) return 0;
    while (p < 0 || p >= n) p = p < 0 ? -p : 2 * n - p - 2;
    return p;
}

// pyrDown(): [1,4,6,4,1] both axes, REFLECT_101, (sum+128)/256, all integers.
__global__ void pyrDownKernel(const u8* in, int w, int h, u8* out, int ow, int oh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    const int k[5] = {1, 4, 6, 4, 1};
    int total = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        const u8* row = in + size_t(reflect101(2 * y + dy, h)) * w;
        int horizontal = 0;
        for (int dx = -2; dx <= 2; ++dx) horizontal += k[dx + 2] * row[reflect101(2 * x + dx, w)];
        total += k[dy + 2] * horizontal;
    }
    out[size_t(y) * ow + x] = u8((total + 128) / 256);
}

__global__ void emptyKernel(int* p) { if (p && threadIdx.x == 1000000) *p = 0; }
__global__ void touchKernel(int* p) { if (threadIdx.x == 0) *p += 1; }

// ---- host -----------------------------------------------------------------
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

struct Timer {
    cudaEvent_t a, b;
    Timer() { CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b)); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { CHECK(cudaEventRecord(a)); }
    double stop() { float ms = 0; CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b)); CHECK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};

template <class T> T* deviceCopy(const std::vector<T>& v)
{
    T* p = nullptr;
    CHECK(cudaMalloc(&p, std::max<size_t>(1, v.size() * sizeof(T))));
    if (!v.empty()) CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

std::string latencyJson()
{
    int* d = nullptr; int* pinned = nullptr;
    CHECK(cudaMalloc(&d, sizeof(int))); CHECK(cudaMemset(d, 0, sizeof(int)));
    CHECK(cudaMallocHost(&pinned, sizeof(int)));
    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
    std::vector<double> launch, roundtrip, graph;
    for (int i = 0; i < 2000; ++i) {
        auto t = Clock::now();
        emptyKernel<<<1, 32, 0, stream>>>(nullptr); CHECK(cudaStreamSynchronize(stream));
        if (i >= 200) launch.push_back(since(t) * 1000);
    }
    for (int i = 0; i < 2000; ++i) {
        auto t = Clock::now();
        CHECK(cudaMemcpyAsync(d, pinned, sizeof(int), cudaMemcpyHostToDevice, stream));
        touchKernel<<<1, 32, 0, stream>>>(d);
        CHECK(cudaMemcpyAsync(pinned, d, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        if (i >= 200) roundtrip.push_back(since(t) * 1000);
    }
    cudaGraph_t g; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMemcpyAsync(d, pinned, sizeof(int), cudaMemcpyHostToDevice, stream));
    touchKernel<<<1, 32, 0, stream>>>(d);
    CHECK(cudaMemcpyAsync(pinned, d, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamEndCapture(stream, &g));
    CHECK(cudaGraphInstantiate(&exec, g, 0));
    for (int i = 0; i < 2000; ++i) {
        auto t = Clock::now();
        CHECK(cudaGraphLaunch(exec, stream)); CHECK(cudaStreamSynchronize(stream));
        if (i >= 200) graph.push_back(since(t) * 1000);
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(g); cudaStreamDestroy(stream);
    cudaFreeHost(pinned); cudaFree(d);
    char buf[256];
    std::snprintf(buf, sizeof buf, "{\"empty_launch_sync_us\":%.2f,\"pinned_roundtrip_us\":%.2f,\"graph_roundtrip_us\":%.2f}",
        median(launch), median(roundtrip), median(graph));
    return buf;
}
}

int main(int argc, char** argv)
{
    if (argc < 5) { std::fprintf(stderr, "usage: cuda_pilot SRC.pgm TMPL.pgm MIN_AREA TOLERANCE [ROIS|latency]\n"); return 1; }
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, 0));
    CHECK(cudaFree(nullptr));
    if (argc > 5 && std::string(argv[5]) == "latency") {
        std::printf("{\"device\":\"%s\",\"latency\":%s}\n", prop.name, latencyJson().c_str());
        return 0;
    }
    const int roiCount = argc > 5 ? std::atoi(argv[5]) : 64;
    const Workload w = loadWorkload(argv[1], argv[2], std::atoi(argv[3]), std::atof(argv[4]));
    const int top = int(w.templ.size()) - 1;
    Timer timer;

    // 1. pyramid: GPU levels from level 0, compared with fpm_core pyrDown.
    bool pyramidSame = true; double pyramidCpu = 0, pyramidGpu = 0;
    {
        Plane level = w.source[0];
        for (int l = 1; l <= top; ++l) {
            const Plane ref = referencePyrDown(level, &pyramidCpu);
            u8 *din = deviceCopy(level.pixels), *dout = nullptr;
            CHECK(cudaMalloc(&dout, ref.pixels.size()));
            dim3 b(32, 8), g((ref.width + 31) / 32, (ref.height + 7) / 8);
            timer.start(); pyrDownKernel<<<g, b>>>(din, level.width, level.height, dout, ref.width, ref.height);
            pyramidGpu += timer.stop(); CHECK(cudaGetLastError());
            std::vector<u8> got(ref.pixels.size());
            CHECK(cudaMemcpy(got.data(), dout, got.size(), cudaMemcpyDeviceToHost));
            pyramidSame &= got == ref.pixels && ref.pixels == w.source[l].pixels;
            cudaFree(din); cudaFree(dout);
            level = ref;
        }
    }

    // 2. coarse: every angle's canvas and score map in one launch each.
    const Plane& topSource = w.source[top];
    const TemplatePlane& topT = w.templ[top];
    std::vector<WarpJob> jobs; std::vector<Plane> canvases;
    std::vector<u64> mapOffsets; u64 pixels = 0, windows = 0; double rotateCpu = 0, nccCpu = 0;
    int maxW = 0, maxH = 0;
    for (std::size_t i = 0; i < w.angles.size(); ++i) {
        Plane canvas;
        WarpJob j = canvasJob(topSource, topT.image.width, topT.image.height, w.angles[i], w.border, &canvas, &rotateCpu);
        if (j.width != canvas.width || j.height != canvas.height) { std::fprintf(stderr, "canvas size mismatch\n"); return 3; }
        // matchWithCoarseNcc skips canvases smaller than the template.
        if (j.width < topT.image.width || j.height < topT.image.height) continue;
        canvases.push_back(std::move(canvas));
        j.offset = pixels; pixels += u64(j.width) * j.height;
        mapOffsets.push_back(windows);
        windows += u64(j.width - topT.image.width + 1) * (j.height - topT.image.height + 1);
        maxW = std::max(maxW, j.width); maxH = std::max(maxH, j.height);
        jobs.push_back(j);
    }
    std::vector<double> refMaps; refMaps.reserve(windows);
    for (const auto& c : canvases) { auto m = referenceNcc(c, topT, &nccCpu); refMaps.insert(refMaps.end(), m.begin(), m.end()); }

    u8* dTop = deviceCopy(topSource.pixels);
    WarpJob* dJobs = deviceCopy(jobs);
    u64* dOffsets = deviceCopy(mapOffsets);
    u8* dTempl = deviceCopy(topT.image.pixels);
    u8* dCanvas = nullptr; double* dMaps = nullptr;
    CHECK(cudaMalloc(&dCanvas, pixels)); CHECK(cudaMalloc(&dMaps, windows * sizeof(double)));
    std::vector<double> warpMs, nccMs, mapDownMs;
    std::vector<u8> gotCanvas(pixels); std::vector<double> gotMaps(windows);
    for (int rep = 0; rep < 11; ++rep) {
        dim3 b(32, 8), g((maxW + 31) / 32, (maxH + 7) / 8, unsigned(jobs.size()));
        timer.start(); warpKernel<<<g, b>>>(dTop, topSource.width, topSource.height, dJobs, dCanvas, w.border);
        warpMs.push_back(timer.stop()); CHECK(cudaGetLastError());
        dim3 gm((maxW - topT.image.width + 32) / 32, (maxH - topT.image.height + 8) / 8, unsigned(jobs.size()));
        timer.start();
        nccNaive<<<gm, b, topT.image.pixels.size()>>>(dCanvas, dJobs, dOffsets, dTempl,
            topT.image.width, topT.image.height, topT.mean, topT.norm, dMaps);
        nccMs.push_back(timer.stop()); CHECK(cudaGetLastError());
        auto t = Clock::now();
        CHECK(cudaMemcpy(gotMaps.data(), dMaps, windows * sizeof(double), cudaMemcpyDeviceToHost));
        mapDownMs.push_back(since(t));
    }
    CHECK(cudaMemcpy(gotCanvas.data(), dCanvas, pixels, cudaMemcpyDeviceToHost));
    u64 canvasDiff = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i)
        canvasDiff += std::memcmp(gotCanvas.data() + jobs[i].offset, canvases[i].pixels.data(), canvases[i].pixels.size()) != 0;
    u64 mapDiff = 0;
    for (u64 i = 0; i < windows; ++i) mapDiff += std::memcmp(&gotMaps[i], &refMaps[i], sizeof(double)) != 0;
    cudaFree(dTop); cudaFree(dJobs); cudaFree(dOffsets); cudaFree(dTempl); cudaFree(dCanvas); cudaFree(dMaps);

    // 3. fine: level-0 ROIs at deterministic pseudo-random anchors and angles.
    const Plane& base = w.source[0];
    const TemplatePlane& t0 = w.templ[0];
    std::vector<WarpJob> roiJobs; std::vector<Plane> rois(roiCount);
    std::vector<double> refFine; double roiCpu = 0, fineNccCpu = 0; u64 roiPixels = 0;
    unsigned state = 12345u;
    auto next = [&] { state = state * 1664525u + 1013904223u; return (state >> 8) / double(1u << 24); };
    for (int i = 0; i < roiCount; ++i) {
        const double ax = next() * base.width, ay = next() * base.height, angle = (next() * 2 - 1) * 180;
        WarpJob j = roiJob(base, t0.image.width, t0.image.height, ax, ay, angle, &rois[i], &roiCpu);
        j.offset = roiPixels; roiPixels += u64(j.width) * j.height;
        roiJobs.push_back(j);
        auto m = referenceNcc(rois[i], t0, &fineNccCpu);
        refFine.insert(refFine.end(), m.begin(), m.end());
    }
    u8* dBase = deviceCopy(base.pixels);
    WarpJob* dRoiJobs = deviceCopy(roiJobs);
    u8* dT0 = deviceCopy(t0.image.pixels);
    u8* dRois = nullptr; double* dFine = nullptr;
    CHECK(cudaMalloc(&dRois, roiPixels)); CHECK(cudaMalloc(&dFine, refFine.size() * sizeof(double)));
    std::vector<double> roiMs, fineMs;
    for (int rep = 0; rep < 11; ++rep) {
        dim3 b(32, 8), g((t0.image.width + 2 * PADDING + 31) / 32, (t0.image.height + 2 * PADDING + 7) / 8, unsigned(roiCount));
        timer.start(); warpKernel<<<g, b>>>(dBase, base.width, base.height, dRoiJobs, dRois, 0);
        roiMs.push_back(timer.stop()); CHECK(cudaGetLastError());
        timer.start(); fineNaive<<<dim3(FINE * FINE, roiCount), dim3(32, 8)>>>(dRois, dRoiJobs, dT0,
            t0.image.width, t0.image.height, t0.mean, t0.norm, dFine);
        fineMs.push_back(timer.stop()); CHECK(cudaGetLastError());
    }
    std::vector<u8> gotRois(roiPixels); std::vector<double> gotFine(refFine.size());
    CHECK(cudaMemcpy(gotRois.data(), dRois, roiPixels, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gotFine.data(), dFine, gotFine.size() * sizeof(double), cudaMemcpyDeviceToHost));
    u64 roiDiff = 0, fineDiff = 0;
    for (int i = 0; i < roiCount; ++i)
        roiDiff += std::memcmp(gotRois.data() + roiJobs[i].offset, rois[i].pixels.data(), rois[i].pixels.size()) != 0;
    for (std::size_t i = 0; i < refFine.size(); ++i) fineDiff += std::memcmp(&gotFine[i], &refFine[i], sizeof(double)) != 0;
    cudaFree(dBase); cudaFree(dRoiJobs); cudaFree(dT0); cudaFree(dRois); cudaFree(dFine);

    std::printf("{\"device\":\"%s\",\"core_backend\":\"%s\",\"source\":[%d,%d],\"template\":[%d,%d],\"levels\":%d,"
        "\"pyramid\":{\"identical\":%s,\"cpu_ms\":%.3f,\"gpu_kernel_ms\":%.3f},"
        "\"coarse\":{\"top_source\":[%d,%d],\"top_template\":[%d,%d],\"angles\":%zu,\"canvas_pixels\":%llu,\"windows\":%llu,"
        "\"canvas_mismatch\":%llu,\"map_mismatch\":%llu,\"cpu_rotate_ms\":%.3f,\"cpu_ncc_ms\":%.3f,"
        "\"gpu_warp_ms\":%.3f,\"gpu_ncc_naive_ms\":%.3f,\"map_download_ms\":%.3f},"
        "\"fine\":{\"rois\":%d,\"roi\":[%d,%d],\"roi_mismatch\":%llu,\"map_mismatch\":%llu,\"cpu_rotate_ms\":%.3f,"
        "\"cpu_ncc_ms\":%.3f,\"gpu_warp_ms\":%.3f,\"gpu_ncc_naive_ms\":%.3f}}\n",
        prop.name, coreBackend(), base.width, base.height, t0.image.width, t0.image.height, top + 1,
        pyramidSame ? "true" : "false", pyramidCpu, pyramidGpu,
        topSource.width, topSource.height, topT.image.width, topT.image.height, w.angles.size(),
        (unsigned long long)pixels, (unsigned long long)windows, (unsigned long long)canvasDiff, (unsigned long long)mapDiff,
        rotateCpu, nccCpu, median(warpMs), median(nccMs), median(mapDownMs),
        roiCount, t0.image.width + 2 * PADDING, t0.image.height + 2 * PADDING,
        (unsigned long long)roiDiff, (unsigned long long)fineDiff, roiCpu, fineNccCpu, median(roiMs), median(fineMs));
    return canvasDiff || mapDiff || roiDiff || fineDiff || !pyramidSame ? 4 : 0;
}
