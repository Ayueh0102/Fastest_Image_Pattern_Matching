// SPDX-License-Identifier: BSD-2-Clause
#include "pose_internal.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fpm_exp {
using Clock = std::chrono::steady_clock;
static float value(const Data& d, int x, int y, int c)
{
    if (x < 0 || y < 0 || x >= d.sw || y >= d.sh) return -d.sourceMean[c];
    return d.source[(size_t(y)*d.sw+x)*d.channels+c];
}

// Independent CPU double accumulation with the same packed float geometry as
// Metal. Bilinear math stays float32, avoiding a hidden precision/warp mismatch.
static float scoreOne(const Data& d, const PackedPose& p)
{
    double sums[3]{}, squares[3]{}, products[3]{};
    for (int y = 0; y < d.th; ++y) for (int x = 0; x < d.tw; ++x) {
        const float u = x - d.tw*.5f, v = d.th*.5f-y;
        const float sx = d.sw*.5f+p.tx+p.cs*u-p.sn*v;
        const float sy = d.sh*.5f-p.ty-(p.sn*u+p.cs*v);
        const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
        const float dx = sx-ix, dy = sy-iy;
        for (int c = 0; c < d.channels; ++c) {
            const float a = value(d, ix, iy, c), b = value(d, ix+1, iy, c);
            const float e = value(d, ix, iy+1, c), f = value(d, ix+1, iy+1, c);
            const double sample = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
            sums[c] += sample;
            squares[c] += sample*sample;
            products[c] += sample*d.templ[(size_t(y)*d.tw+x)*d.channels+c];
        }
    }
    const double n = double(d.tw)*d.th;
    double energy = 0, product = 0;
    for (int c = 0; c < d.channels; ++c) {
        energy += std::max(0.0, squares[c]-sums[c]*sums[c]/n);
        product += products[c]-sums[c]*d.templateSum[c]/n;
    }
    if (energy/n <= 1e-12 || d.templateEnergy/n <= 1e-12) return -2;
    return float(std::clamp(product/std::sqrt(energy*d.templateEnergy), -1.0, 1.0));
}

class Cpu final : public Backend {
    const Data& data_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    bool stop_ = false;
    size_t epoch_ = 0, remaining_ = 0;
    std::atomic<size_t> next_{0};
    const std::vector<PackedPose>* poses_ = nullptr;
    float* scores_ = nullptr;
    std::string name_;
    void work() {
        for (size_t i = next_.fetch_add(1); i < poses_->size(); i = next_.fetch_add(1))
            scores_[i] = scoreOne(data_, (*poses_)[i]);
    }
    void shutdown() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        start_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
public:
    Cpu(const Data& data, int threads) : data_(data), name_("C++ float sampling / double NCC / " + std::to_string(threads) + " threads") {
        try {
            for (int i = 1; i < threads; ++i) workers_.emplace_back([this] {
                size_t epoch = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    start_.wait(lock, [&] { return stop_ || epoch_ != epoch; });
                    if (stop_) return;
                    epoch = epoch_;
                    lock.unlock(); work(); lock.lock();
                    if (--remaining_ == 0) done_.notify_one();
                }
            });
        } catch (...) { shutdown(); throw; }
    }
    ~Cpu() override { shutdown(); }
    const char* name() const override { return name_.c_str(); }
    void score(const std::vector<PackedPose>& poses, float* out, FpmPoseMetrics&) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            poses_ = &poses; scores_ = out; next_ = 0;
            remaining_ = workers_.size(); ++epoch_;
        }
        start_.notify_all(); work();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }
};

struct Context {
    Data data;
    std::vector<PackedPose> packed;
    std::unique_ptr<Backend> backend; // destroyed before data
};
static void errorMessage(char* error, int cap, const char* message) {
    if (error && cap > 0) std::snprintf(error, size_t(cap), "%s", message);
}
static size_t imageSize(int w, int h, int c) {
    if (w < 1 || h < 1 || w > 32768 || h > 32768 || (c != 1 && c != 3))
        throw std::invalid_argument("invalid image dimensions/channels");
    const size_t size = size_t(w)*h*c;
    if (size > 128*1024*1024) throw std::invalid_argument("experiment image size limit exceeded");
    return size;
}
}

extern "C" {
void* fpm_pose_create(int backend, int threads, const float* source, int sw, int sh,
    const float* templ, int tw, int th, int channels, char* error, int error_capacity)
{
    using namespace fpm_exp;
    try {
        if (!source || !templ || threads < 1 || threads > 64 || (backend != 0 && backend != 1))
            throw std::invalid_argument("invalid pointer/backend/thread count");
        auto ctx = std::make_unique<Context>();
        auto& d = ctx->data;
        d.sw = sw; d.sh = sh; d.tw = tw; d.th = th; d.channels = channels;
        d.source.assign(source, source+imageSize(sw, sh, channels));
        d.templ.assign(templ, templ+imageSize(tw, th, channels));
        for (auto* image : {&d.source, &d.templ})
            for (float x : *image) if (!std::isfinite(x) || std::abs(x) > 16)
                throw std::invalid_argument("inputs must be finite with abs(value) <= 16");
        for (int c = 0; c < channels; ++c) {
            double sum = 0;
            for (size_t i = c; i < d.source.size(); i += channels) sum += d.source[i];
            d.sourceMean[c] = float(sum/(size_t(sw)*sh));
            for (size_t i = c; i < d.source.size(); i += channels) d.source[i] -= d.sourceMean[c];
            sum = 0;
            for (size_t i = c; i < d.templ.size(); i += channels) sum += d.templ[i];
            const float mean = float(sum/(size_t(tw)*th));
            double squares = 0;
            for (size_t i = c; i < d.templ.size(); i += channels) {
                d.templ[i] -= mean;
                d.templateSum[c] += d.templ[i];
                squares += double(d.templ[i])*d.templ[i];
            }
            d.templateEnergy += squares-d.templateSum[c]*d.templateSum[c]/(double(tw)*th);
        }
        if (backend == 0) ctx->backend = std::make_unique<Cpu>(d, threads);
        else {
#ifdef FPM_EXPERIMENT_METAL
            ctx->backend = makeMetal(d);
#else
            throw std::runtime_error("Metal backend was not compiled on this platform");
#endif
        }
        errorMessage(error, error_capacity, "");
        return ctx.release();
    } catch (const std::exception& e) { errorMessage(error, error_capacity, e.what()); return nullptr; }
}
const char* fpm_pose_device(void* handle)
{
    return handle ? static_cast<fpm_exp::Context*>(handle)->backend->name() : "invalid handle";
}
int fpm_pose_memory(void* handle, FpmPoseMetrics* metrics)
{
    if (!handle || !metrics) return -1;
    auto& ctx = *static_cast<fpm_exp::Context*>(handle);
    *metrics = {}; metrics->device_ms = -1;
    metrics->host_buffer_bytes = (ctx.data.source.capacity()+ctx.data.templ.capacity())*sizeof(float)
        + ctx.packed.capacity()*sizeof(fpm_exp::PackedPose);
    ctx.backend->memory(*metrics);
    return 0;
}
int fpm_pose_score(void* handle, const float* poses, int count, float* scores,
    FpmPoseMetrics* metrics, char* error, int error_capacity)
{
    using namespace fpm_exp;
    try {
        if (!handle || !poses || !scores || !metrics || count < 1 || count > 65536)
            throw std::invalid_argument("invalid score arguments (1 <= count <= 65536)");
        const auto begin = Clock::now();
        auto& ctx = *static_cast<Context*>(handle);
        *metrics = {}; metrics->device_ms = -1;
        ctx.packed.resize(size_t(count));
        for (int i = 0; i < count; ++i) {
            const float* p = poses+size_t(i)*4;
            for (int j = 0; j < 4; ++j) if (!std::isfinite(p[j])) throw std::invalid_argument("non-finite pose");
            if (std::abs(p[0]) > 1e6 || std::abs(p[1]) > 1e6 || std::abs(p[2]) > 36000 || std::abs(p[3]) > 3)
                throw std::invalid_argument("pose outside experiment limits");
            const double theta = p[2]*3.14159265358979323846/180, scale = std::exp(double(p[3]));
            ctx.packed[i] = {p[0], p[1], float(scale*std::cos(theta)), float(scale*std::sin(theta))};
        }
        ctx.backend->score(ctx.packed, scores, *metrics);
        metrics->host_buffer_bytes = (ctx.data.source.capacity()+ctx.data.templ.capacity())*sizeof(float)
            + ctx.packed.capacity()*sizeof(PackedPose);
        metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
        errorMessage(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) { errorMessage(error, error_capacity, e.what()); return -1; }
}
void fpm_pose_destroy(void* handle) { delete static_cast<fpm_exp::Context*>(handle); }
}
