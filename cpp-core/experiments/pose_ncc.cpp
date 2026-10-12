// SPDX-License-Identifier: BSD-2-Clause
#include "pose_internal.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fpm_exp {
using Clock = std::chrono::steady_clock;
constexpr double radians = 3.14159265358979323846/180;
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

static PackedPose pack(const double* p)
{
    // Match the public scorer's float input and packed geometry exactly.
    const double theta = double(float(p[2]))*3.14159265358979323846/180;
    const double scale = std::exp(double(float(p[3])));
    return {float(p[0]), float(p[1]), float(scale*std::cos(theta)), float(scale*std::sin(theta))};
}

struct Linearization {
    double score = -2;
    double h[4][4]{}, g[4]{};
};

// Let c = P I(W(p)), z = c/||c||, tau = P T/||P T||, where P removes
// the mean independently in each channel. E = 1/2 ||z-tau||^2 = 1-NCC.
// Jz = (I-zz') P J / ||c||. Sufficient statistics below form Jz'Jz and
// Jz'(tau-z), without storing any per-pixel residual or Jacobian arrays.
// Internal angle increments are radians; public pose angles remain degrees.
static Linearization linearize(const Data& d, const PackedPose& p,
    const FpmPoseRefineConfig& config)
{
    double sum[3]{}, square[3]{}, product[3]{};
    double jsum[3][4]{}, jx[3][4]{}, jt[3][4]{}, jj[4][4]{};
    for (int y = 0; y < d.th; ++y) for (int x = 0; x < d.tw; ++x) {
        const float u = x-d.tw*.5f, v = d.th*.5f-y;
        const float rx = p.cs*u-p.sn*v, ry = p.sn*u+p.cs*v;
        // Keep operations in the same order as scoreOne (float rounding matters).
        const float sx = d.sw*.5f+p.tx+p.cs*u-p.sn*v;
        const float sy = d.sh*.5f-p.ty-(p.sn*u+p.cs*v);
        const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
        const float dx = sx-ix, dy = sy-iy;
        for (int c = 0; c < d.channels; ++c) {
            const float a = value(d, ix, iy, c), b = value(d, ix+1, iy, c);
            const float e = value(d, ix, iy+1, c), f = value(d, ix+1, iy+1, c);
            const double sample = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
            const double gx = (b-a)*(1-dy)+(f-e)*dy;
            const double gy = (e-a)*(1-dx)+(f-b)*dx;
            const double t = d.templ[(size_t(y)*d.tw+x)*d.channels+c];
            double j[4] = {gx, -gy, -gx*ry-gy*rx, gx*rx-gy*ry};
            if (!config.optimize_angle || config.max_angle_degrees == 0) j[2] = 0;
            if (!config.optimize_scale || config.max_log_scale == 0) j[3] = 0;
            if (config.max_translation == 0) j[0] = j[1] = 0;
            sum[c] += sample; square[c] += sample*sample; product[c] += sample*t;
            for (int k = 0; k < 4; ++k) {
                jsum[c][k] += j[k]; jx[c][k] += j[k]*sample; jt[c][k] += j[k]*t;
                for (int l = k; l < 4; ++l) jj[k][l] += j[k]*j[l];
            }
        }
    }
    const double n = double(d.tw)*d.th;
    double energy = 0, dot = 0, centeredJx[4]{}, centeredJt[4]{};
    for (int c = 0; c < d.channels; ++c) {
        energy += std::max(0.0, square[c]-sum[c]*sum[c]/n);
        dot += product[c]-sum[c]*d.templateSum[c]/n;
        for (int k = 0; k < 4; ++k) {
            centeredJx[k] += jx[c][k]-jsum[c][k]*sum[c]/n;
            centeredJt[k] += jt[c][k]-jsum[c][k]*d.templateSum[c]/n;
            for (int l = k; l < 4; ++l) jj[k][l] -= jsum[c][k]*jsum[c][l]/n;
        }
    }
    Linearization result;
    if (energy/n <= 1e-12 || d.templateEnergy/n <= 1e-12) return result;
    const double denom = std::sqrt(energy*d.templateEnergy);
    result.score = std::clamp(dot/denom, -1.0, 1.0);
    for (int k = 0; k < 4; ++k) {
        result.g[k] = centeredJt[k]/denom-result.score*centeredJx[k]/energy;
        for (int l = k; l < 4; ++l) {
            result.h[k][l] = jj[k][l]/energy-centeredJx[k]*centeredJx[l]/(energy*energy);
            result.h[l][k] = result.h[k][l];
        }
    }
    return result;
}

static bool solveDamped(const Linearization& a, double damping, double* step)
{
    double m[4][5]{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) m[i][j] = a.h[i][j];
        m[i][i] += damping*std::max(a.h[i][i], 1e-12);
        m[i][4] = a.g[i];
    }
    // Tiny dense system, partial pivoting; no dependency on a matrix package.
    for (int i = 0; i < 4; ++i) {
        int pivot = i;
        for (int j = i+1; j < 4; ++j) if (std::abs(m[j][i]) > std::abs(m[pivot][i])) pivot = j;
        if (!std::isfinite(m[pivot][i]) || std::abs(m[pivot][i]) < 1e-24) return false;
        for (int j = i; j <= 4; ++j) std::swap(m[i][j], m[pivot][j]);
        const double divisor = m[i][i];
        for (int j = i; j <= 4; ++j) m[i][j] /= divisor;
        for (int row = 0; row < 4; ++row) if (row != i) {
            const double factor = m[row][i];
            for (int j = i; j <= 4; ++j) m[row][j] -= factor*m[i][j];
        }
    }
    for (int i = 0; i < 4; ++i) {
        step[i] = m[i][4];
        if (!std::isfinite(step[i])) return false;
    }
    return true;
}

static void refineOne(const Data& d, const float* seed, const FpmPoseRefineConfig& config,
    float* output, float& score, FpmPoseRefineStats& stats)
{
    double p[4] = {seed[0], seed[1], seed[2], seed[3]};
    const double bound[4] = {config.max_translation, config.max_translation,
        config.optimize_angle ? config.max_angle_degrees : 0,
        config.optimize_scale ? config.max_log_scale : 0};
    const double limit[4] = {1e6, 1e6, 36000, 3};
    stats = {}; stats.status = 1;
    score = scoreOne(d, pack(p)); ++stats.evaluations;
    double damping = 1e-3;
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        if (score == -2) { stats.status = 2; break; }
        const auto a = linearize(d, pack(p), config);
        ++stats.evaluations; ++stats.iterations;
        if (a.score == -2) { stats.status = 2; break; }
        bool accepted = false, small = false;
        for (int trial = 0; trial < 10; ++trial) {
            double step[4]{}, q[4]{};
            if (!solveDamped(a, damping, step)) { damping *= 10; continue; }
            step[2] /= radians;
            // Preserve direction while keeping one iteration inside a local
            // linearization basin. These are step limits, not total-travel limits.
            const double ratio = std::max({1.0, std::abs(step[0])/2,
                std::abs(step[1])/2, std::abs(step[2])/5, std::abs(step[3])/.1});
            for (int k = 0; k < 4; ++k) {
                const double lower = std::max(-limit[k], double(seed[k])-bound[k]);
                const double upper = std::min(limit[k], double(seed[k])+bound[k]);
                q[k] = float(std::clamp(p[k]+step[k]/ratio, lower, upper));
                step[k] = q[k]-p[k];
            }
            small = std::max(std::abs(step[0]), std::abs(step[1])) < 1e-4
                && std::abs(step[2]) < 1e-4 && std::abs(step[3]) < 1e-6;
            if (small) break;
            const float next = scoreOne(d, pack(q)); ++stats.evaluations;
            if (next > score) {
                std::copy(q, q+4, p); score = next;
                damping = std::max(1e-8, damping*.3); accepted = true; break;
            }
            damping *= 10;
        }
        if (!accepted) { stats.status = small ? 0 : 3; break; }
    }
    for (int k = 0; k < 4; ++k) output[k] = float(p[k]);
}

class Cpu final : public Backend {
    const Data& data_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    bool stop_ = false;
    size_t epoch_ = 0, remaining_ = 0;
    std::atomic<size_t> next_{0};
    size_t count_ = 0;
    std::function<void(size_t)> job_;
    std::string name_;
    void work() {
        for (size_t i = next_.fetch_add(1); i < count_; i = next_.fetch_add(1)) job_(i);
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
        parallel(poses.size(), [&](size_t i) { out[i] = scoreOne(data_, poses[i]); });
    }
    void parallel(size_t count, std::function<void(size_t)> job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            count_ = count; job_ = std::move(job); next_ = 0;
            remaining_ = workers_.size(); ++epoch_;
        }
        start_.notify_all(); work();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        job_ = {};
    }
};

struct Context {
    Data data;
    std::vector<PackedPose> packed;
    std::unique_ptr<Backend> backend; // destroyed before data
    std::unique_ptr<Cpu> refiner;
    int threads = 1;
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
        ctx->threads = threads;
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
void fpm_pose_refine_default_config(FpmPoseRefineConfig* config)
{
    if (config) *config = {30, 1, 1, 12, 15, std::log(1.3)};
}
int fpm_pose_refine(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, float* refined, float* scores,
    FpmPoseRefineStats* stats, FpmPoseMetrics* metrics, char* error, int error_capacity)
{
    using namespace fpm_exp;
    try {
        if (!handle || !seeds || !refined || !scores || !metrics || count < 1 || count > 65536)
            throw std::invalid_argument("invalid refinement arguments (1 <= count <= 65536)");
        FpmPoseRefineConfig options;
        fpm_pose_refine_default_config(&options);
        if (config) options = *config;
        if (options.max_iterations < 1 || options.max_iterations > 200
            || (options.optimize_scale != 0 && options.optimize_scale != 1)
            || (options.optimize_angle != 0 && options.optimize_angle != 1)
            || !std::isfinite(options.max_translation) || options.max_translation < 0 || options.max_translation > 1024
            || !std::isfinite(options.max_angle_degrees) || options.max_angle_degrees < 0 || options.max_angle_degrees > 180
            || !std::isfinite(options.max_log_scale) || options.max_log_scale < 0 || options.max_log_scale > 3)
            throw std::invalid_argument("invalid refinement configuration");
        for (int i = 0; i < count; ++i) {
            const float* p = seeds+size_t(i)*4;
            for (int j = 0; j < 4; ++j) if (!std::isfinite(p[j])) throw std::invalid_argument("non-finite refinement seed");
            if (std::abs(p[0]) > 1e6 || std::abs(p[1]) > 1e6 || std::abs(p[2]) > 36000 || std::abs(p[3]) > 3)
                throw std::invalid_argument("refinement seed outside experiment limits");
        }
        const auto begin = Clock::now();
        auto& ctx = *static_cast<Context*>(handle);
        auto* cpu = dynamic_cast<Cpu*>(ctx.backend.get());
        if (!cpu) {
            if (!ctx.refiner) ctx.refiner = std::make_unique<Cpu>(ctx.data, ctx.threads);
            cpu = ctx.refiner.get();
        }
        cpu->parallel(size_t(count), [&](size_t i) {
            FpmPoseRefineStats localStats;
            refineOne(ctx.data, seeds+4*i, options, refined+4*i, scores[i], localStats);
            if (stats) stats[i] = localStats;
        });
        fpm_pose_memory(handle, metrics);
        metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
        errorMessage(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) { errorMessage(error, error_capacity, e.what()); return -1; }
}
void fpm_pose_destroy(void* handle) { delete static_cast<fpm_exp::Context*>(handle); }
}
