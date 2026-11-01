// SPDX-License-Identifier: BSD-2-Clause
// Transfer and synchronization pilot for the second CUDA round (exploratory, not a gate).
// xfer_pilot [spin]   spin: cudaSetDeviceFlags(cudaDeviceScheduleSpin) before the context.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); std::exit(1); } } while (0)

__global__ void tiny(unsigned* out, unsigned v) { if (threadIdx.x == 0) out[0] = v; }
__global__ void tinyFlag(volatile unsigned* flag, unsigned v) { if (threadIdx.x == 0) { __threadfence_system(); flag[0] = v; } }
__global__ void busy(float* p, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i];
    for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 0.5f;
    p[i] = v;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static void report(const char* name, std::vector<double> v) {
    std::sort(v.begin(), v.end());
    std::printf("{\"name\":\"%s\",\"median_ms\":%.5f,\"p10_ms\":%.5f,\"p90_ms\":%.5f,\"n\":%zu}\n", name,
        v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], v.size());
}

int main(int argc, char** argv) {
    const bool spin = argc > 1 && std::strcmp(argv[1], "spin") == 0;
    if (spin) CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaStream_t s2; CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    const int W = 4024, H = 3036; const size_t N = size_t(W) * H;  // Src7 size
    std::vector<unsigned char> src(N), pageDst(N);
    for (size_t i = 0; i < N; ++i) src[i] = (unsigned char)(i * 2654435761u >> 24);
    unsigned char* dev; CK(cudaMalloc(&dev, N + (1 << 20)));
    float* fbuf; CK(cudaMalloc(&fbuf, 1 << 24));
    unsigned* dflag; CK(cudaMalloc(&dflag, 64));
    // warm the GPU clocks
    {
        const auto t = Clock::now();
        while (ms(t) < 600) { busy<<<4096, 256, 0, s>>>(fbuf, 1 << 20, 2000); CK(cudaStreamSynchronize(s)); }
    }
    const int R = 200;
    std::printf("{\"spin\":%d}\n", spin ? 1 : 0);

    // ---- CPU copies ----
    { std::vector<double> t; for (int r = 0; r < 40; ++r) { auto a = Clock::now(); std::memcpy(pageDst.data(), src.data(), N); t.push_back(ms(a)); } report("cpu memcpy warm pageable 12MB", t); }
    { std::vector<double> t; for (int r = 0; r < 20; ++r) { auto a = Clock::now(); std::vector<unsigned char> f(N); std::memcpy(f.data(), src.data(), N); t.push_back(ms(a)); } report("cpu fresh vector(N)+memcpy 12MB", t); }
    // registration cost
    {
        std::vector<double> t;
        for (int r = 0; r < 10; ++r) {
            std::vector<unsigned char> f(N); auto a = Clock::now();
            CK(cudaHostRegister(f.data(), N, cudaHostRegisterDefault)); t.push_back(ms(a));
            CK(cudaHostUnregister(f.data()));
        }
        report("cudaHostRegister 12MB", t);
    }
    std::vector<unsigned char> reg(N); CK(cudaHostRegister(reg.data(), N, cudaHostRegisterDefault));
    unsigned char* pin; CK(cudaHostAlloc(&pin, N, cudaHostAllocDefault));
    { std::vector<double> t; for (int r = 0; r < 40; ++r) { auto a = Clock::now(); std::memcpy(reg.data(), src.data(), N); t.push_back(ms(a)); } report("cpu memcpy into registered 12MB", t); }

    // ---- whole-image H2D ----
    auto h2d = [&](const char* name, const unsigned char* host) {
        std::vector<double> t;
        for (int r = 0; r < 60; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(dev, host, N, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); t.push_back(ms(a)); }
        report(name, t);
    };
    h2d("h2d pageable 12MB", src.data());
    h2d("h2d registered 12MB", reg.data());
    h2d("h2d hostalloc 12MB", pin);
    // pageable copy + h2d (the round-1 path: host copy into a warm Image then pageable upload)
    { std::vector<double> t; for (int r = 0; r < 60; ++r) { auto a = Clock::now(); std::memcpy(pageDst.data(), src.data(), N);
        CK(cudaMemcpyAsync(dev, pageDst.data(), N, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); t.push_back(ms(a)); } report("warm copy + pageable h2d 12MB", t); }
    // chunked copy into registered memory with overlapped DMA
    for (int chunks : {1, 4, 8, 16, 32}) for (int flush = 0; flush < 2; ++flush) {
        std::vector<double> t;
        const size_t step = (N + chunks - 1) / chunks;
        for (int r = 0; r < 60; ++r) {
            auto a = Clock::now();
            for (size_t off = 0; off < N; off += step) {
                const size_t n = std::min(step, N - off);
                std::memcpy(reg.data() + off, src.data() + off, n);
                CK(cudaMemcpyAsync(dev + off, reg.data() + off, n, cudaMemcpyHostToDevice, s));
                if (flush) (void)cudaStreamQuery(s);
            }
            CK(cudaStreamSynchronize(s)); t.push_back(ms(a));
        }
        char name[96]; std::snprintf(name, sizeof name, "chunked copy+h2d registered chunks=%d flush=%d", chunks, flush);
        report(name, t);
    }
    // D2H 4 MB registered vs pageable
    {
        const size_t M = N / 3;
        std::vector<double> a1, a2;
        for (int r = 0; r < 60; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(pageDst.data(), dev, M, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); a1.push_back(ms(a)); }
        for (int r = 0; r < 60; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(reg.data(), dev, M, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); a2.push_back(ms(a)); }
        report("d2h pageable 4MB", a1); report("d2h registered 4MB", a2);
    }

    // ---- API costs and roundtrips ----
    unsigned* hflag; CK(cudaHostAlloc(&hflag, 64, cudaHostAllocMapped));
    unsigned* dmapped; CK(cudaHostGetDevicePointer((void**)&dmapped, hflag, 0));
    cudaEvent_t e1, e2; CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
    cudaEvent_t en; CK(cudaEventCreateWithFlags(&en, cudaEventDisableTiming));
    unsigned char small[512]; std::memset(small, 1, sizeof small);
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); tiny<<<1, 32, 0, s>>>(dflag, r); t.push_back(ms(a)); CK(cudaStreamSynchronize(s)); } report("api launch", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaEventRecord(e1, s)); t.push_back(ms(a)); CK(cudaStreamSynchronize(s)); } report("api eventRecord timing", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaEventRecord(en, s)); t.push_back(ms(a)); CK(cudaStreamSynchronize(s)); } report("api eventRecord disableTiming", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaMemsetAsync(dflag, 0, 64, s)); t.push_back(ms(a)); CK(cudaStreamSynchronize(s)); } report("api memset 64B", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(dev, small, sizeof small, cudaMemcpyHostToDevice, s)); t.push_back(ms(a)); CK(cudaStreamSynchronize(s)); } report("api memcpy 512B pageable", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(dev, pin, 512, cudaMemcpyHostToDevice, s)); t.push_back(ms(a)); CK(cudaStreamSynchronize(s)); } report("api memcpy 512B pinned", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); (void)cudaStreamQuery(s); t.push_back(ms(a)); } report("api streamQuery idle", t); }
    // roundtrips: small pinned upload + kernel, then wait
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(dev, pin, 512, cudaMemcpyHostToDevice, s)); tiny<<<1, 32, 0, s>>>(dflag, r); CK(cudaStreamSynchronize(s)); t.push_back(ms(a)); } report("roundtrip upload+kernel streamSync", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(dev, pin, 512, cudaMemcpyHostToDevice, s)); tiny<<<1, 32, 0, s>>>(dflag, r); CK(cudaEventRecord(en, s)); CK(cudaEventSynchronize(en)); t.push_back(ms(a)); } report("roundtrip upload+kernel eventSync", t); }
    for (int flush = 0; flush < 2; ++flush) {
        std::vector<double> t; int timeouts = 0;
        for (int r = 0; r < R; ++r) {
            hflag[0] = 0; const unsigned v = r + 1;
            auto a = Clock::now();
            CK(cudaMemcpyAsync(dev, pin, 512, cudaMemcpyHostToDevice, s));
            tinyFlag<<<1, 32, 0, s>>>(dmapped, v);
            if (flush) (void)cudaStreamQuery(s);
            while (*(volatile unsigned*)hflag != v) { if (ms(a) > 50) { ++timeouts; break; } }
            t.push_back(ms(a));
            CK(cudaStreamSynchronize(s));
        }
        char name[96]; std::snprintf(name, sizeof name, "roundtrip upload+kernel mapped-flag poll flush=%d timeouts=%d", flush, timeouts);
        report(name, t);
    }
    // a realistic fine-call shape: event, pinned upload, memset, two kernels, event, sync
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaEventRecord(e1, s)); CK(cudaMemcpyAsync(dev, pin, 512, cudaMemcpyHostToDevice, s));
        CK(cudaMemsetAsync(dflag, 0, 64, s)); tiny<<<1, 32, 0, s>>>(dflag, r); tiny<<<1, 32, 0, s>>>(dflag, r); CK(cudaEventRecord(e2, s)); CK(cudaStreamSynchronize(s)); t.push_back(ms(a)); } report("fine-shape round1 (ev,cpy,memset,k,k,ev,sync)", t); }
    { std::vector<double> t; for (int r = 0; r < R; ++r) { auto a = Clock::now(); CK(cudaMemcpyAsync(dev, pin, 512, cudaMemcpyHostToDevice, s));
        tiny<<<1, 32, 0, s>>>(dflag, r); tiny<<<1, 32, 0, s>>>(dflag, r); CK(cudaStreamSynchronize(s)); t.push_back(ms(a)); } report("fine-shape lean (cpy,k,k,sync)", t); }
    std::printf("{\"done\":1}\n");
    return 0;
}
