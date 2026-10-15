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
#include <limits>
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

static float boundedFloat(double value, double lower, double upper)
{
    // Bounds contain the original float seed, so at least one representable
    // result exists. Casting a double-clamped endpoint can round OUTSIDE its
    // interval; move that result inward by one float ULP when necessary.
    float result = float(std::clamp(value, lower, upper));
    if (double(result) < lower) result = std::nextafter(result, std::numeric_limits<float>::infinity());
    if (double(result) > upper) result = std::nextafter(result, -std::numeric_limits<float>::infinity());
    return result;
}

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
    float* output, float& score, FpmPoseRefineStats& stats, const float* boundOrigin = nullptr)
{
    double p[4] = {seed[0], seed[1], seed[2], seed[3]};
    if (!boundOrigin) boundOrigin = seed;
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
                const double lower = std::max(-limit[k], double(boundOrigin[k])-bound[k]);
                const double upper = std::min(limit[k], double(boundOrigin[k])+bound[k]);
                q[k] = boundedFloat(p[k]+step[k]/ratio, lower, upper);
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

struct AcceleratedState {
    double pose[4]{}, candidate[4]{};
    double damping = 1e-3;
    float initialScore = -2, deviceScore = -2;
    FpmPoseRefineStats stats{};
    bool active = false, pending = false;
};
struct AcceleratedWorkspace {
    std::vector<AcceleratedState> states;
    std::vector<size_t> active, pending, owners;
    std::vector<PackedPose> packed;
    std::vector<Linearization> linearizations;
    std::vector<float> values;
    uint64_t bytes() const {
        return states.capacity()*sizeof(AcceleratedState)
            + (active.capacity()+pending.capacity()+owners.capacity())*sizeof(size_t)
            + packed.capacity()*sizeof(PackedPose)
            + linearizations.capacity()*sizeof(Linearization) + values.capacity()*sizeof(float);
    }
};
struct RescueWorkspace {
    std::vector<size_t> eligible, parents, owners, continuationOwners;
    std::vector<ResidentSeed> trials, continuation;
    std::vector<float> output, values, baseScores;
    std::vector<FpmPoseRefineStats> trialStats, totalStats;
    uint64_t bytes() const {
        return (eligible.capacity()+parents.capacity()+owners.capacity()+continuationOwners.capacity())*sizeof(size_t)
            + (trials.capacity()+continuation.capacity())*sizeof(ResidentSeed)
            + (output.capacity()+values.capacity()+baseScores.capacity())*sizeof(float)
            + (trialStats.capacity()+totalStats.capacity())*sizeof(FpmPoseRefineStats);
    }
};
struct Context {
    Data data;
    std::vector<PackedPose> packed;
    std::unique_ptr<Backend> backend; // destroyed before data
    std::unique_ptr<Backend> lazyMetal;
    std::unique_ptr<Cpu> refiner;
    AcceleratedWorkspace accelerated;
    RescueWorkspace rescue;
    std::vector<ResidentSeed> residentSeeds;
    FpmPoseResidentMetrics lastResident{};
    int residentMode = 0;
    int threads = 1, creationBackend = 0;
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

struct DeviceTiming {
    double milliseconds = 0;
    bool used = false, complete = true;
    void add(const FpmPoseMetrics& metrics) {
        used = true;
        if (!std::isfinite(metrics.device_ms) || metrics.device_ms < 0) complete = false;
        else milliseconds += metrics.device_ms;
    }
    double result() const { return used && complete ? milliseconds : -1; }
};

// Returns false only if the backend has no derivative implementation. All
// expensive device operations are batches; the CPU only solves small systems
// between dispatches, then verifies and optionally polishes the final poses.
static bool refineAccelerated(Context& ctx, Backend& backend, const float* seeds, int count,
    const FpmPoseRefineConfig& config, int polishIterations, float* output,
    float* scores, FpmPoseRefineStats* stats, double& deviceMilliseconds)
{
    auto* cpu = dynamic_cast<Cpu*>(ctx.backend.get());
    if (!cpu) {
        if (!ctx.refiner) ctx.refiner = std::make_unique<Cpu>(ctx.data, ctx.threads);
        cpu = ctx.refiner.get();
    }
    auto& w = ctx.accelerated;
    w.states.resize(size_t(count));
    cpu->parallel(size_t(count), [&](size_t i) {
        auto& s = w.states[i]; s = {};
        for (int k = 0; k < 4; ++k) s.pose[k] = seeds[4*i+k];
        s.initialScore = scoreOne(ctx.data, pack(s.pose));
        s.stats.evaluations = 1;
        s.stats.status = s.initialScore == -2 ? 2 : (s.initialScore == 1 ? 0 : 1);
        s.active = s.initialScore > -2 && s.initialScore < 1;
    });
    DeviceTiming timing;
    const double bound[4] = {config.max_translation, config.max_translation,
        config.optimize_angle ? config.max_angle_degrees : 0,
        config.optimize_scale ? config.max_log_scale : 0};
    const double limit[4] = {1e6, 1e6, 36000, 3};
    // GPU float reductions can change the last few score bits. Comparisons use
    // the SAME score kernel on both sides and ignore gains below two float eps.
    // This tolerance is not an accuracy guarantee; CPU verification below is.
    constexpr double scoreTolerance = 2*std::numeric_limits<float>::epsilon();
    for (int iteration = 0; iteration < config.max_iterations; ++iteration) {
        w.active.clear(); w.packed.clear();
        for (size_t i = 0; i < w.states.size(); ++i) if (w.states[i].active) {
            w.active.push_back(i); w.packed.push_back(pack(w.states[i].pose));
        }
        if (w.active.empty()) break;
        FpmPoseMetrics derivativeMetrics{}; derivativeMetrics.device_ms = -1;
        if (!backend.linearize(w.packed, config, w.linearizations, derivativeMetrics)) {
            if (iteration != 0) throw std::runtime_error("derivative backend became unavailable during refinement");
            return false;
        }
        if (w.linearizations.size() != w.active.size()) throw std::runtime_error("invalid derivative batch size");
        timing.add(derivativeMetrics);
        if (iteration == 0) {
            w.values.resize(w.active.size());
            FpmPoseMetrics scoreMetrics{}; scoreMetrics.device_ms = -1;
            backend.score(w.packed, w.values.data(), scoreMetrics);
            timing.add(scoreMetrics);
            for (size_t j = 0; j < w.active.size(); ++j) {
                auto& s = w.states[w.active[j]];
                s.deviceScore = w.values[j]; ++s.stats.evaluations;
            }
        }
        w.pending.clear();
        for (size_t j = 0; j < w.active.size(); ++j) {
            auto& s = w.states[w.active[j]];
            ++s.stats.iterations; ++s.stats.evaluations;
            if (w.linearizations[j].score == -2 || s.deviceScore == -2) {
                s.active = false; s.stats.status = 2;
            } else if (!std::isfinite(w.linearizations[j].score) || !std::isfinite(s.deviceScore)) {
                s.active = false; s.stats.status = 3;
            } else { s.pending = true; w.pending.push_back(j); }
        }
        for (int trial = 0; trial < 10 && !w.pending.empty(); ++trial) {
            w.owners.clear(); w.packed.clear();
            for (size_t j : w.pending) {
                const size_t index = w.active[j];
                auto& s = w.states[index];
                double step[4]{};
                if (!solveDamped(w.linearizations[j], s.damping, step)) { s.damping *= 10; continue; }
                step[2] /= radians;
                const double ratio = std::max({1.0, std::abs(step[0])/2,
                    std::abs(step[1])/2, std::abs(step[2])/5, std::abs(step[3])/.1});
                for (int k = 0; k < 4; ++k) {
                    const double lower = std::max(-limit[k], double(seeds[4*index+k])-bound[k]);
                    const double upper = std::min(limit[k], double(seeds[4*index+k])+bound[k]);
                    s.candidate[k] = boundedFloat(s.pose[k]+step[k]/ratio, lower, upper);
                    step[k] = s.candidate[k]-s.pose[k];
                }
                const bool small = std::max(std::abs(step[0]), std::abs(step[1])) < 1e-4
                    && std::abs(step[2]) < 1e-4 && std::abs(step[3]) < 1e-6;
                if (small) { s.active = s.pending = false; s.stats.status = 0; continue; }
                w.owners.push_back(j); w.packed.push_back(pack(s.candidate));
            }
            if (!w.owners.empty()) {
                w.values.resize(w.owners.size());
                FpmPoseMetrics scoreMetrics{}; scoreMetrics.device_ms = -1;
                backend.score(w.packed, w.values.data(), scoreMetrics);
                timing.add(scoreMetrics);
                for (size_t j = 0; j < w.owners.size(); ++j) {
                    auto& s = w.states[w.active[w.owners[j]]];
                    ++s.stats.evaluations;
                    const float next = w.values[j];
                    if (std::isfinite(next) && double(next) > double(s.deviceScore)+scoreTolerance) {
                        std::copy(s.candidate, s.candidate+4, s.pose); s.deviceScore = next;
                        s.damping = std::max(1e-8, s.damping*.3); s.pending = false;
                    } else s.damping *= 10;
                }
            }
            w.pending.erase(std::remove_if(w.pending.begin(), w.pending.end(),
                [&](size_t j) { return !w.states[w.active[j]].pending; }), w.pending.end());
        }
        for (size_t j : w.pending) {
            auto& s = w.states[w.active[j]];
            s.active = s.pending = false; s.stats.status = 3;
        }
    }
    FpmPoseRefineConfig polishConfig = config;
    polishConfig.max_iterations = polishIterations;
    cpu->parallel(size_t(count), [&](size_t i) {
        auto& s = w.states[i];
        float verified = scoreOne(ctx.data, pack(s.pose)); ++s.stats.evaluations;
        if (!std::isfinite(verified) || verified < s.initialScore) {
            for (int k = 0; k < 4; ++k) s.pose[k] = seeds[4*i+k];
            verified = s.initialScore; s.stats.status = 3;
        }
        if (polishIterations > 0) {
            float start[4];
            for (int k = 0; k < 4; ++k) start[k] = float(s.pose[k]);
            FpmPoseRefineStats polished;
            refineOne(ctx.data, start, polishConfig, output+4*i, scores[i], polished, seeds+4*i);
            s.stats.iterations += polished.iterations; s.stats.evaluations += polished.evaluations;
            s.stats.status = polished.status;
        } else {
            for (int k = 0; k < 4; ++k) output[4*i+k] = float(s.pose[k]);
            scores[i] = verified;
        }
        if (stats) stats[i] = s.stats;
    });
    deviceMilliseconds = timing.result();
    return true;
}
}

extern "C" {
void* fpm_pose_create(int backend, int threads, const float* source, int sw, int sh,
    const float* templ, int tw, int th, int channels, char* error, int error_capacity)
{
    using namespace fpm_exp;
    try {
        if (!source || !templ || threads < 1 || threads > 64 || (backend != 0 && backend != 1 && backend != 2))
            throw std::invalid_argument("invalid pointer/backend/thread count");
        auto ctx = std::make_unique<Context>();
        ctx->threads = threads;
        ctx->creationBackend = backend;
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
        if (backend != 1) ctx->backend = std::make_unique<Cpu>(d, threads);
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
int fpm_pose_backend(void* handle)
{
    return handle ? static_cast<fpm_exp::Context*>(handle)->creationBackend : -1;
}
int fpm_pose_set_resident_mode(void* handle, int mode, char* error, int error_capacity)
{
    using namespace fpm_exp;
    try {
        if (!handle || mode < 0 || mode > 1) throw std::invalid_argument("invalid resident implementation mode");
        auto& ctx = *static_cast<Context*>(handle);
        if (mode && (ctx.creationBackend == 0 || ctx.data.channels != 1
            || uint64_t(ctx.data.tw)*ctx.data.th > 16384))
            throw std::invalid_argument("fused resident requires a Metal/auto grayscale handle and <=16384 template pixels");
        ctx.residentMode = mode;
        errorMessage(error, error_capacity, ""); return 0;
    } catch (const std::exception& e) { errorMessage(error, error_capacity, e.what()); return -1; }
}
int fpm_pose_resident_mode(void* handle)
{
    return handle ? static_cast<fpm_exp::Context*>(handle)->residentMode : -1;
}
int fpm_pose_last_resident_metrics(void* handle, FpmPoseResidentMetrics* metrics)
{
    if (metrics) *metrics = {};
    if (!handle || !metrics) return -1;
    *metrics = static_cast<fpm_exp::Context*>(handle)->lastResident;
    return 0;
}
int fpm_pose_memory(void* handle, FpmPoseMetrics* metrics)
{
    if (!handle || !metrics) return -1;
    auto& ctx = *static_cast<fpm_exp::Context*>(handle);
    *metrics = {}; metrics->device_ms = -1;
    metrics->host_buffer_bytes = (ctx.data.source.capacity()+ctx.data.templ.capacity())*sizeof(float)
        + ctx.packed.capacity()*sizeof(fpm_exp::PackedPose) + ctx.accelerated.bytes()
        + ctx.residentSeeds.capacity()*sizeof(fpm_exp::ResidentSeed) + ctx.rescue.bytes();
    ctx.backend->memory(*metrics);
    if (ctx.lazyMetal) {
        FpmPoseMetrics lazy{};
        ctx.lazyMetal->memory(lazy);
        metrics->gpu_buffer_bytes += lazy.gpu_buffer_bytes;
        metrics->gpu_allocated_bytes = std::max(metrics->gpu_allocated_bytes, lazy.gpu_allocated_bytes);
    }
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
        if (ctx.lazyMetal) {
            FpmPoseMetrics lazy{};
            ctx.lazyMetal->memory(lazy);
            metrics->gpu_buffer_bytes += lazy.gpu_buffer_bytes;
            metrics->gpu_allocated_bytes = std::max(metrics->gpu_allocated_bytes, lazy.gpu_allocated_bytes);
        }
        metrics->host_buffer_bytes = (ctx.data.source.capacity()+ctx.data.templ.capacity())*sizeof(float)
            + ctx.packed.capacity()*sizeof(PackedPose) + ctx.accelerated.bytes()
            + ctx.residentSeeds.capacity()*sizeof(ResidentSeed) + ctx.rescue.bytes();
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
int fpm_pose_refine_accelerated(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, int cpu_polish_iterations, float* refined,
    float* scores, FpmPoseRefineStats* stats, FpmPoseMetrics* metrics,
    char* error, int error_capacity)
{
    using namespace fpm_exp;
    try {
        if (!handle || !seeds || !refined || !scores || !metrics || count < 1 || count > 65536)
            throw std::invalid_argument("invalid refinement arguments (1 <= count <= 65536)");
        if (cpu_polish_iterations < 0 || cpu_polish_iterations > 30)
            throw std::invalid_argument("CPU polish iterations must be between 0 and 30");
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
        auto& ctx = *static_cast<Context*>(handle);
        if (dynamic_cast<Cpu*>(ctx.backend.get()))
            return fpm_pose_refine(handle, seeds, count, &options, refined, scores, stats, metrics, error, error_capacity);
        const auto begin = Clock::now();
        double deviceTime = -1;
        if (!refineAccelerated(ctx, *ctx.backend, seeds, count, options, cpu_polish_iterations,
            refined, scores, stats, deviceTime)) {
            const int result = fpm_pose_refine(handle, seeds, count, &options,
                refined, scores, stats, metrics, error, error_capacity);
            if (result == 0) metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
            return result;
        }
        fpm_pose_memory(handle, metrics);
        metrics->device_ms = deviceTime;
        metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
        errorMessage(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) { errorMessage(error, error_capacity, e.what()); return -1; }
}
int fpm_pose_refine_resident(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, int cpu_verify, float* refined, float* scores,
    FpmPoseRefineStats* stats, FpmPoseResidentMetrics* resident_metrics,
    FpmPoseMetrics* metrics, char* error, int error_capacity)
{
    using namespace fpm_exp;
    const auto begin = Clock::now();
    if (handle) static_cast<Context*>(handle)->lastResident = {};
    if (resident_metrics) *resident_metrics = {};
    if (metrics) { *metrics = {}; metrics->device_ms = -1; }
    try {
        if (!handle || !seeds || !refined || !scores || !resident_metrics || !metrics
            || count < 1 || count > 65536)
            throw std::invalid_argument("invalid resident refinement arguments (1 <= count <= 65536)");
        if (cpu_verify != 0 && cpu_verify != 1)
            throw std::invalid_argument("CPU verification must be 0 or 1");
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
            for (int k = 0; k < 4; ++k) if (!std::isfinite(p[k]))
                throw std::invalid_argument("non-finite refinement seed");
            if (std::abs(p[0]) > 1e6 || std::abs(p[1]) > 1e6 || std::abs(p[2]) > 36000 || std::abs(p[3]) > 3)
                throw std::invalid_argument("refinement seed outside experiment limits");
        }
        auto& ctx = *static_cast<Context*>(handle);
        if (ctx.creationBackend == 0)
            throw std::invalid_argument("GPU-resident refinement requires a Metal or auto handle");
        ctx.residentSeeds.resize(size_t(count));
        const double bound[4] = {options.max_translation, options.max_translation,
            options.optimize_angle ? options.max_angle_degrees : 0,
            options.optimize_scale ? options.max_log_scale : 0};
        const double limit[4] = {1e6, 1e6, 36000, 3};
        for (int i = 0; i < count; ++i) for (int k = 0; k < 4; ++k) {
            auto& seed = ctx.residentSeeds[size_t(i)];
            seed.pose[k] = seeds[size_t(i)*4+k];
            const double lower = std::max(-limit[k], double(seed.pose[k])-bound[k]);
            const double upper = std::min(limit[k], double(seed.pose[k])+bound[k]);
            seed.lower[k] = boundedFloat(lower, lower, upper);
            seed.upper[k] = boundedFloat(upper, lower, upper);
        }
        Backend* selected = ctx.backend.get();
        if (ctx.creationBackend == 2) {
#ifdef FPM_EXPERIMENT_METAL
            if (!ctx.lazyMetal) {
                const auto setupBegin = Clock::now();
                ctx.lazyMetal = makeMetal(ctx.data);
                resident_metrics->setup_ms = std::chrono::duration<double, std::milli>(Clock::now()-setupBegin).count();
            }
            selected = ctx.lazyMetal.get();
#else
            throw std::runtime_error("Metal backend was not compiled on this platform");
#endif
        }
        if (!selected->refineResident(ctx.residentSeeds, options, refined, scores, stats, *resident_metrics, *metrics, ctx.residentMode))
            throw std::runtime_error("GPU-resident refinement is unsupported by this backend");
        // Cheap validation remains enabled even when CPU pixel verification is
        // disabled. Device failures are errors, never silently repaired poses.
        for (int i = 0; i < count; ++i) {
            if (!std::isfinite(scores[i]) || (scores[i] != -2.f && (scores[i] < -1.f || scores[i] > 1.f)))
                throw std::runtime_error("invalid score returned by GPU-resident refinement");
            for (int k = 0; k < 4; ++k) {
                const float value = refined[size_t(i)*4+k];
                const auto& seed = ctx.residentSeeds[size_t(i)];
                if (!std::isfinite(value) || value < seed.lower[k] || value > seed.upper[k])
                    throw std::runtime_error("GPU-resident refinement returned a pose outside its bounds");
            }
        }
        resident_metrics->cpu_verified = 0;
        resident_metrics->cpu_verify_ms = 0;
        if (cpu_verify) {
            const auto verifyBegin = Clock::now();
            auto* cpu = dynamic_cast<Cpu*>(ctx.backend.get());
            if (!cpu) {
                if (!ctx.refiner) ctx.refiner = std::make_unique<Cpu>(ctx.data, ctx.threads);
                cpu = ctx.refiner.get();
            }
            cpu->parallel(size_t(count), [&](size_t i) {
                const auto& seed = ctx.residentSeeds[i];
                double original[4], candidate[4];
                for (int k = 0; k < 4; ++k) {
                    original[k] = seed.pose[k]; candidate[k] = refined[4*i+k];
                }
                const float initialScore = scoreOne(ctx.data, pack(original));
                float finalScore = scoreOne(ctx.data, pack(candidate));
                if (stats) stats[i].evaluations += 2;
                if (!std::isfinite(finalScore) || finalScore < initialScore) {
                    for (int k = 0; k < 4; ++k) refined[4*i+k] = seed.pose[k];
                    finalScore = initialScore;
                    if (stats) stats[i].status = 3;
                }
                scores[i] = finalScore;
            });
            resident_metrics->cpu_verify_ms = std::chrono::duration<double, std::milli>(Clock::now()-verifyBegin).count();
            resident_metrics->cpu_verified = 1;
        }
        const double deviceTime = metrics->device_ms;
        fpm_pose_memory(handle, metrics);
        metrics->device_ms = deviceTime;
        metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
        ctx.lastResident = *resident_metrics;
        errorMessage(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) {
        if (resident_metrics) *resident_metrics = {};
        if (metrics) { *metrics = {}; metrics->device_ms = -1; }
        errorMessage(error, error_capacity, e.what());
        return -1;
    }
}
void fpm_pose_rescue_default_config(FpmPoseRescueConfig* config)
{
    if (config) *config = {0, 4, 8, 0, .1, .8, 3., 6., .06, uint64_t(512)*1024*1024};
}
int fpm_pose_refine_rescue(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, const FpmPoseRescueConfig* rescue_config,
    float* refined, float* scores, FpmPoseRefineStats* stats,
    FpmPoseRescueMetrics* rescue_metrics, FpmPoseMetrics* metrics,
    char* error, int error_capacity)
{
    using namespace fpm_exp;
    static_assert(sizeof(FpmPoseRescueConfig) == 64);
    static_assert(sizeof(FpmPoseRescueMetrics) == 136);
    const auto begin = Clock::now();
    if (handle) static_cast<Context*>(handle)->lastResident = {};
    if (rescue_metrics) *rescue_metrics = {};
    if (metrics) { *metrics = {}; metrics->device_ms = -1; }
    try {
        if (!handle || !seeds || !refined || !scores || !rescue_metrics || !metrics
            || count < 1 || count > 65536)
            throw std::invalid_argument("invalid rescue refinement arguments (1 <= count <= 65536)");
        FpmPoseRefineConfig options;
        fpm_pose_refine_default_config(&options);
        if (config) options = *config;
        FpmPoseRescueConfig policy;
        fpm_pose_rescue_default_config(&policy);
        if (rescue_config) policy = *rescue_config;
        if ((policy.backend != 0 && policy.backend != 1)
            || policy.max_rescued_seeds < 0 || policy.max_rescued_seeds > 64
            || policy.short_iterations < 1 || policy.short_iterations > 200 || policy.reserved != 0
            || !std::isfinite(policy.trigger_min_score) || policy.trigger_min_score < -1 || policy.trigger_min_score > 1
            || !std::isfinite(policy.trigger_score) || policy.trigger_score < policy.trigger_min_score || policy.trigger_score > 1
            || !std::isfinite(policy.translation_radius) || policy.translation_radius < 0 || policy.translation_radius > 1024
            || !std::isfinite(policy.angle_radius_degrees) || policy.angle_radius_degrees < 0 || policy.angle_radius_degrees > 180
            || !std::isfinite(policy.log_scale_radius) || policy.log_scale_radius < 0 || policy.log_scale_radius > 3)
            throw std::invalid_argument("invalid rescue configuration");
        // Validate every option and seed before dispatching or allocating Metal.
        if (options.max_iterations < 1 || options.max_iterations > 200
            || (options.optimize_scale != 0 && options.optimize_scale != 1)
            || (options.optimize_angle != 0 && options.optimize_angle != 1)
            || !std::isfinite(options.max_translation) || options.max_translation < 0 || options.max_translation > 1024
            || !std::isfinite(options.max_angle_degrees) || options.max_angle_degrees < 0 || options.max_angle_degrees > 180
            || !std::isfinite(options.max_log_scale) || options.max_log_scale < 0 || options.max_log_scale > 3)
            throw std::invalid_argument("invalid refinement configuration");
        for (int i = 0; i < count; ++i) {
            const float* p = seeds+size_t(i)*4;
            for (int k = 0; k < 4; ++k) if (!std::isfinite(p[k]))
                throw std::invalid_argument("non-finite refinement seed");
            if (std::abs(p[0]) > 1e6 || std::abs(p[1]) > 1e6 || std::abs(p[2]) > 36000 || std::abs(p[3]) > 3)
                throw std::invalid_argument("refinement seed outside experiment limits");
        }
        auto& ctx = *static_cast<Context*>(handle);
        if (policy.backend == 1 && ctx.creationBackend == 0)
            throw std::invalid_argument("resident rescue requires a Metal or auto handle");
        auto& w = ctx.rescue;
        w.totalStats.resize(size_t(count));
        w.eligible.clear(); w.parents.clear(); w.owners.clear(); w.trials.clear();
        w.continuation.clear(); w.continuationOwners.clear(); w.baseScores.clear();
        rescue_metrics->selected_backend = policy.backend;
        DeviceTiming deviceTiming;
        FpmPoseResidentMetrics aggregate{};
        auto recordResident = [&](const FpmPoseResidentMetrics& diagnostic, const FpmPoseMetrics& measured) {
            aggregate.command_buffers += diagnostic.command_buffers;
            aggregate.cpu_waits += diagnostic.cpu_waits;
            aggregate.encoded_dispatches += diagnostic.encoded_dispatches;
            aggregate.auxiliary_buffer_bytes = std::max(aggregate.auxiliary_buffer_bytes, diagnostic.auxiliary_buffer_bytes);
            aggregate.setup_ms += diagnostic.setup_ms;
            aggregate.encode_ms += diagnostic.encode_ms;
            aggregate.wait_ms += diagnostic.wait_ms;
            rescue_metrics->command_buffers += diagnostic.command_buffers;
            rescue_metrics->cpu_waits += diagnostic.cpu_waits;
            rescue_metrics->auxiliary_buffer_bytes = std::max(rescue_metrics->auxiliary_buffer_bytes,
                diagnostic.auxiliary_buffer_bytes);
            rescue_metrics->cpu_verify_ms += diagnostic.cpu_verify_ms;
            deviceTiming.add(measured);
        };
        const auto baselineBegin = Clock::now();
        FpmPoseMetrics baselineMetrics{};
        char phaseError[2048]{};
        if (policy.backend == 1) {
            FpmPoseResidentMetrics resident{};
            if (fpm_pose_refine_resident(handle, seeds, count, &options, 1, refined, scores,
                w.totalStats.data(), &resident, &baselineMetrics, phaseError, sizeof(phaseError)))
                throw std::runtime_error(phaseError);
            recordResident(resident, baselineMetrics);
        } else if (fpm_pose_refine(handle, seeds, count, &options, refined, scores,
            w.totalStats.data(), &baselineMetrics, phaseError, sizeof(phaseError)))
            throw std::runtime_error(phaseError);
        rescue_metrics->baseline_ms = std::chrono::duration<double, std::milli>(Clock::now()-baselineBegin).count();
        const auto retryBegin = Clock::now();
        for (int i = 0; i < count; ++i)
            if (std::isfinite(scores[i]) && scores[i] >= policy.trigger_min_score && scores[i] < policy.trigger_score)
                w.eligible.push_back(size_t(i));
        // Stable score priority is deterministic and independent of the truth.
        std::stable_sort(w.eligible.begin(), w.eligible.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });
        rescue_metrics->eligible_seeds = w.eligible.size();
        const size_t parentLimit = std::min(w.eligible.size(), size_t(policy.max_rescued_seeds));
        rescue_metrics->parent_cap_skipped_seeds = w.eligible.size()-parentLimit;
        const uint64_t perParent = uint64_t(ctx.data.tw)*ctx.data.th*ctx.data.channels
            *(uint64_t(16)*policy.short_iterations+options.max_iterations);
        uint64_t remaining = policy.max_retry_sample_iterations;
        const double bound[4] = {options.max_translation, options.max_translation,
            options.optimize_angle ? options.max_angle_degrees : 0,
            options.optimize_scale ? options.max_log_scale : 0};
        const double radii[4] = {policy.translation_radius, policy.translation_radius,
            options.optimize_angle ? policy.angle_radius_degrees : 0,
            options.optimize_scale ? policy.log_scale_radius : 0};
        const double limit[4] = {1e6, 1e6, 36000, 3};
        const unsigned bases[4] = {2, 3, 5, 7};
        auto halton = [](unsigned index, unsigned base) {
            double value = 0, factor = 1;
            while (index) { factor /= base; value += factor*(index%base); index /= base; }
            return value;
        };
        for (size_t rank = 0; rank < parentLimit; ++rank) {
            if (perParent > remaining) { ++rescue_metrics->budget_skipped_seeds; continue; }
            const size_t parent = w.eligible[rank], first = w.trials.size();
            for (unsigned index = 1; index <= 16; ++index) {
                ResidentSeed trial{};
                bool unchanged = true;
                for (int k = 0; k < 4; ++k) {
                    const double origin = seeds[4*parent+k];
                    const double lower = std::max(-limit[k], origin-bound[k]);
                    const double upper = std::min(limit[k], origin+bound[k]);
                    trial.lower[k] = boundedFloat(lower, lower, upper);
                    trial.upper[k] = boundedFloat(upper, lower, upper);
                    trial.pose[k] = boundedFloat(origin+(2*halton(index, bases[k])-1)*radii[k], lower, upper);
                    unchanged &= trial.pose[k] == seeds[4*parent+k];
                }
                if (unchanged) continue;
                bool duplicate = false;
                for (size_t previous = first; previous < w.trials.size(); ++previous)
                    duplicate |= std::equal(trial.pose, trial.pose+4, w.trials[previous].pose);
                if (duplicate) continue;
                w.trials.push_back(trial); w.owners.push_back(parent);
            }
            if (w.trials.size() == first) continue;
            w.parents.push_back(parent); w.baseScores.push_back(scores[parent]);
            remaining -= perParent;
            rescue_metrics->reserved_retry_sample_iterations += perParent;
        }
        rescue_metrics->selected_seeds = w.parents.size();
        rescue_metrics->retry_starts = w.trials.size();
        if (!w.trials.empty()) {
            auto* cpu = dynamic_cast<Cpu*>(ctx.backend.get());
            if (!cpu) {
                if (!ctx.refiner) ctx.refiner = std::make_unique<Cpu>(ctx.data, ctx.threads);
                cpu = ctx.refiner.get();
            }
            Backend* residentBackend = ctx.creationBackend == 2 ? ctx.lazyMetal.get() : ctx.backend.get();
            auto runTrials = [&](const std::vector<ResidentSeed>& trials, const std::vector<size_t>& owners,
                const FpmPoseRefineConfig& trialConfig) {
                w.output.resize(trials.size()*4); w.values.resize(trials.size()); w.trialStats.resize(trials.size());
                if (policy.backend == 1) {
                    FpmPoseResidentMetrics resident{}; FpmPoseMetrics measured{}; measured.device_ms = -1;
                    if (!residentBackend->refineResident(trials, trialConfig, w.output.data(), w.values.data(),
                        w.trialStats.data(), resident, measured, ctx.residentMode))
                        throw std::runtime_error("GPU-resident rescue unsupported by this backend");
                    recordResident(resident, measured);
                } else cpu->parallel(trials.size(), [&](size_t i) {
                    refineOne(ctx.data, trials[i].pose, trialConfig, w.output.data()+4*i,
                        w.values[i], w.trialStats[i], seeds+4*owners[i]);
                });
                for (size_t i = 0; i < trials.size(); ++i) for (int k = 0; k < 4; ++k) {
                    const float value = w.output[4*i+k];
                    if (!std::isfinite(value) || value < trials[i].lower[k] || value > trials[i].upper[k])
                        throw std::runtime_error("rescue returned a pose outside ORIGINAL seed bounds");
                }
                // Re-score all retries before ranking/acceptance. A GPU score
                // never replaces the exact baseline on its own. No CPU polish.
                const auto verifyBegin = Clock::now();
                cpu->parallel(trials.size(), [&](size_t i) {
                    double pose[4];
                    for (int k = 0; k < 4; ++k) pose[k] = w.output[4*i+k];
                    w.values[i] = scoreOne(ctx.data, pack(pose));
                    ++w.trialStats[i].evaluations;
                });
                rescue_metrics->cpu_verify_ms += std::chrono::duration<double, std::milli>(Clock::now()-verifyBegin).count();
                for (size_t i = 0; i < trials.size(); ++i) {
                    const size_t owner = owners[i];
                    w.totalStats[owner].iterations += w.trialStats[i].iterations;
                    w.totalStats[owner].evaluations += w.trialStats[i].evaluations;
                    rescue_metrics->retry_evaluations += w.trialStats[i].evaluations;
                    if (w.values[i] > scores[owner]) {
                        std::copy(w.output.data()+4*i, w.output.data()+4*i+4, refined+4*owner);
                        scores[owner] = w.values[i]; w.totalStats[owner].status = w.trialStats[i].status;
                    }
                }
            };
            auto shortConfig = options; shortConfig.max_iterations = policy.short_iterations;
            runTrials(w.trials, w.owners, shortConfig);
            for (size_t i = 0; i < w.parents.size(); ++i) {
                const size_t parent = w.parents[i];
                if (!(scores[parent] > w.baseScores[i])) continue;
                ResidentSeed trial{};
                for (int k = 0; k < 4; ++k) {
                    const double lower = std::max(-limit[k], double(seeds[4*parent+k])-bound[k]);
                    const double upper = std::min(limit[k], double(seeds[4*parent+k])+bound[k]);
                    trial.pose[k] = refined[4*parent+k];
                    trial.lower[k] = boundedFloat(lower, lower, upper);
                    trial.upper[k] = boundedFloat(upper, lower, upper);
                }
                w.continuation.push_back(trial); w.continuationOwners.push_back(parent);
            }
            rescue_metrics->continuations = w.continuation.size();
            if (!w.continuation.empty()) runTrials(w.continuation, w.continuationOwners, options);
            for (size_t i = 0; i < w.parents.size(); ++i)
                rescue_metrics->accepted_seeds += scores[w.parents[i]] > w.baseScores[i];
        }
        if (stats) std::copy(w.totalStats.begin(), w.totalStats.end(), stats);
        rescue_metrics->retry_ms = std::chrono::duration<double, std::milli>(Clock::now()-retryBegin).count();
        rescue_metrics->host_workspace_bytes = w.bytes();
        fpm_pose_memory(handle, metrics);
        metrics->device_ms = deviceTiming.result();
        metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
        if (policy.backend == 1) {
            aggregate.cpu_verify_ms = rescue_metrics->cpu_verify_ms;
            aggregate.cpu_verified = 1;
        }
        ctx.lastResident = aggregate;
        errorMessage(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) {
        if (rescue_metrics) *rescue_metrics = {};
        if (handle) static_cast<Context*>(handle)->lastResident = {};
        if (metrics) { *metrics = {}; metrics->device_ms = -1; }
        errorMessage(error, error_capacity, e.what());
        return -1;
    }
}
void fpm_pose_auto_default_config(FpmPoseAutoConfig* config)
{
    if (config) *config = {131072, 0, 4, 0};
}
const char* fpm_pose_route_reason(int reason)
{
    switch (reason) {
    case FPM_POSE_ROUTE_CPU_HANDLE: return "CPU handle";
    case FPM_POSE_ROUTE_BELOW_POSE_WORK: return "below per-pose sample threshold";
    case FPM_POSE_ROUTE_BELOW_BATCH_WORK: return "below batch sample threshold";
    case FPM_POSE_ROUTE_BELOW_ITERATIONS: return "below iteration threshold";
    case FPM_POSE_ROUTE_METAL_SELECTED: return "Metal workload thresholds met";
    case FPM_POSE_ROUTE_METAL_UNAVAILABLE: return "Metal unavailable";
    case FPM_POSE_ROUTE_CPU_NO_ACTIVE_PARAMETERS: return "no active parameters";
    default: return "unknown routing reason";
    }
}
int fpm_pose_refine_auto(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, const FpmPoseAutoConfig* auto_config,
    float* refined, float* scores, FpmPoseRefineStats* stats, FpmPoseRoute* route,
    FpmPoseMetrics* metrics, char* error, int error_capacity)
{
    using namespace fpm_exp;
    const auto begin = Clock::now();
    try {
        if (!handle || !seeds || !refined || !scores || !route || !metrics || count < 1 || count > 65536)
            throw std::invalid_argument("invalid auto refinement arguments (1 <= count <= 65536)");
        *route = {}; route->selected_backend = -1; route->reason = -1;
        FpmPoseRefineConfig options;
        fpm_pose_refine_default_config(&options);
        if (config) options = *config;
        FpmPoseAutoConfig routing;
        fpm_pose_auto_default_config(&routing);
        if (auto_config) routing = *auto_config;
        // Validate every input before considering lazy Metal initialization,
        // including options that a selected CPU route will not otherwise use.
        if (options.max_iterations < 1 || options.max_iterations > 200
            || (options.optimize_scale != 0 && options.optimize_scale != 1)
            || (options.optimize_angle != 0 && options.optimize_angle != 1)
            || !std::isfinite(options.max_translation) || options.max_translation < 0 || options.max_translation > 1024
            || !std::isfinite(options.max_angle_degrees) || options.max_angle_degrees < 0 || options.max_angle_degrees > 180
            || !std::isfinite(options.max_log_scale) || options.max_log_scale < 0 || options.max_log_scale > 3)
            throw std::invalid_argument("invalid refinement configuration");
        if (routing.min_iterations < 1 || routing.min_iterations > 200
            || routing.cpu_polish_iterations < 0 || routing.cpu_polish_iterations > 30)
            throw std::invalid_argument("invalid auto routing configuration");
        for (int i = 0; i < count; ++i) {
            const float* p = seeds+size_t(i)*4;
            for (int j = 0; j < 4; ++j) if (!std::isfinite(p[j])) throw std::invalid_argument("non-finite refinement seed");
            if (std::abs(p[0]) > 1e6 || std::abs(p[1]) > 1e6 || std::abs(p[2]) > 36000 || std::abs(p[3]) > 3)
                throw std::invalid_argument("refinement seed outside experiment limits");
        }
        auto& ctx = *static_cast<Context*>(handle);
        route->pose_samples = uint64_t(ctx.data.tw)*uint64_t(ctx.data.th)*uint64_t(ctx.data.channels);
        route->batch_samples = route->pose_samples*uint64_t(count);
        route->selected_backend = 0;
        const bool activeParameters = options.max_translation > 0
            || (options.optimize_angle && options.max_angle_degrees > 0)
            || (options.optimize_scale && options.max_log_scale > 0);
        if (ctx.creationBackend == 0) route->reason = FPM_POSE_ROUTE_CPU_HANDLE;
        else if (!activeParameters) route->reason = FPM_POSE_ROUTE_CPU_NO_ACTIVE_PARAMETERS;
        else if (route->pose_samples < routing.min_pose_samples) route->reason = FPM_POSE_ROUTE_BELOW_POSE_WORK;
        else if (route->batch_samples < routing.min_batch_samples) route->reason = FPM_POSE_ROUTE_BELOW_BATCH_WORK;
        else if (options.max_iterations < routing.min_iterations) route->reason = FPM_POSE_ROUTE_BELOW_ITERATIONS;
        else route->reason = FPM_POSE_ROUTE_METAL_SELECTED;

        Backend* selected = nullptr;
        if (route->reason == FPM_POSE_ROUTE_METAL_SELECTED) {
            if (ctx.creationBackend == 1) selected = ctx.backend.get();
            else {
#ifdef FPM_EXPERIMENT_METAL
                if (!ctx.lazyMetal) {
                    const auto setupBegin = Clock::now();
                    try { ctx.lazyMetal = makeMetal(ctx.data); }
                    catch (const MetalUnavailable&) { route->reason = FPM_POSE_ROUTE_METAL_UNAVAILABLE; }
                    // Only unavailable hardware is a fallback. Shader, memory,
                    // queue or dispatch errors propagate to the outer handler.
                    route->metal_setup_ms = std::chrono::duration<double, std::milli>(Clock::now()-setupBegin).count();
                }
                selected = ctx.lazyMetal.get();
#else
                route->reason = FPM_POSE_ROUTE_METAL_UNAVAILABLE;
#endif
            }
        }
        if (selected) {
            double deviceTime = -1;
            if (refineAccelerated(ctx, *selected, seeds, count, options,
                routing.cpu_polish_iterations, refined, scores, stats, deviceTime)) {
                route->selected_backend = 1;
                fpm_pose_memory(handle, metrics);
                metrics->device_ms = deviceTime;
                metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
                errorMessage(error, error_capacity, "");
                return 0;
            }
            route->reason = FPM_POSE_ROUTE_METAL_UNAVAILABLE;
        }
        const int result = fpm_pose_refine(handle, seeds, count, &options,
            refined, scores, stats, metrics, error, error_capacity);
        if (result == 0) metrics->wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
        return result;
    } catch (const std::exception& e) { errorMessage(error, error_capacity, e.what()); return -1; }
}
void fpm_pose_destroy(void* handle) { delete static_cast<fpm_exp::Context*>(handle); }
}
