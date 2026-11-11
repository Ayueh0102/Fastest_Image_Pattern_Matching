// SPDX-License-Identifier: BSD-2-Clause
// The CUDA backend (2) against the CPU backend (0) of the same build: every
// result field must be bit-identical. Covered paths: the handle API (device
// coarse search, CPU pyramid and fine search), frames with the device pyramid
// and device fine search (several batch sizes, multiple models, the unfused
// coarse ablation), dense score maps, and option validation; and the transfer
// modes (lean prepare and lean calls) over repeated prepares, size and inversion
// changes, and their metrics. Exit 77 when no CUDA device is available.
#include "coarse_ncc.h"
#include "fpm/matcher.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const std::string& text) { if (!value) throw std::runtime_error(text); }
fpm_image view(const fpm::Image& image) { return {image.row(0), image.width(), image.height(), std::size_t(image.width())}; }

// A textured template: a few random blobs and ramps plus noise; or a flat one.
fpm::Image makeTemplate(int width, int height, unsigned seed, bool flat = false)
{
    fpm::Image image(width, height);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    struct Blob { double x, y, r, a; };
    std::vector<Blob> blobs;
    for (int i = 0; i < 6; ++i) blobs.push_back({u(rng) * width, u(rng) * height, 2 + u(rng) * width / 3, u(rng) * 160 - 80});
    const double gx = u(rng) * 60 - 30, gy = u(rng) * 60 - 30;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double v = 128 + gx * x / width + gy * y / height + (u(rng) - 0.5) * 24;
            for (const auto& b : blobs) v += b.a * std::exp(-((x - b.x) * (x - b.x) + (y - b.y) * (y - b.y)) / (2 * b.r * b.r));
            image.row(y)[x] = flat ? 90 : std::uint8_t(std::clamp(v, 0.0, 255.0));
        }
    return image;
}

// Background noise with rotated copies of the template pasted in (bilinear).
fpm::Image makeScene(int width, int height, const fpm::Image& templ, int copies, double maxAngle, unsigned seed)
{
    fpm::Image image(width, height);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    const int base = int(u(rng) * 160) + 40;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) image.row(y)[x] = std::uint8_t(std::clamp(base + (u(rng) - 0.5) * 70, 0.0, 255.0));
    const int tw = templ.width(), th = templ.height();
    for (int k = 0; k < copies; ++k) {
        const double angle = (u(rng) * 2 - 1) * maxAngle * 3.14159265358979323846 / 180;
        const double cx = tw / 2.0 + u(rng) * std::max(1, width - tw), cy = th / 2.0 + u(rng) * std::max(1, height - th);
        const double c = std::cos(angle), s = std::sin(angle), radius = std::hypot(tw, th) / 2 + 2;
        for (int y = std::max(0, int(cy - radius)); y < std::min(height, int(cy + radius) + 1); ++y)
            for (int x = std::max(0, int(cx - radius)); x < std::min(width, int(cx + radius) + 1); ++x) {
                const double dx = x - cx, dy = y - cy;
                const double tx = c * dx + s * dy + (tw - 1) / 2.0, ty = -s * dx + c * dy + (th - 1) / 2.0;
                const int ix = int(std::floor(tx)), iy = int(std::floor(ty));
                if (ix < 0 || iy < 0 || ix + 1 >= tw || iy + 1 >= th) continue;
                const double fx = tx - ix, fy = ty - iy;
                const double v = (templ.row(iy)[ix] * (1 - fx) + templ.row(iy)[ix + 1] * fx) * (1 - fy) +
                    (templ.row(iy + 1)[ix] * (1 - fx) + templ.row(iy + 1)[ix + 1] * fx) * fy;
                image.row(y)[x] = std::uint8_t(std::clamp(v + (u(rng) - 0.5) * 10, 0.0, 255.0));
            }
    }
    return image;
}

struct Handle {
    fpm_coarse_handle* value = nullptr;
    Handle(const fpm::Image& templ, int area, int backend) {
        FpmCoarseMetrics metrics{}; char error[1024]{};
        require(fpm_coarse_create_backend(view(templ), area, backend, &value, &metrics, error, sizeof(error)) == FPM_OK, error);
    }
    ~Handle() { fpm_coarse_destroy(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
struct Frame {
    fpm_frame_handle* value = nullptr;
    char error[1024]{};
    Frame(int backend, int pyramid, int fineBatch, int nccMode = 0, int transfer = 0) {
        FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
        config.backend = backend; config.pyramid_mode = pyramid; config.batch_angles = 1; config.fine_batch_size = fineBatch;
        require(fpm_frame_create(&config, sizeof(config), &value, error, sizeof(error)) == FPM_OK, error);
        if (nccMode) {
            FpmCoarseOptimizationConfigV1 optimization{1, sizeof(FpmCoarseOptimizationConfigV1), 0, nccMode, 0, 0};
            require(fpm_frame_set_coarse_optimization(value, &optimization, sizeof(optimization), error, sizeof(error)) == FPM_OK, error);
        }
        if (transfer) setTransfer(transfer);
    }
    void setTransfer(int mode) {
        require(fpm_frame_set_transfer_mode(value, mode, error, sizeof(error)) == FPM_OK, error);
    }
    void setFineMode(int mode) {
        require(fpm_frame_set_fine_ncc_mode(value, mode, error, sizeof(error)) == FPM_OK, error);
    }
    void setThreads(int threads) {
        require(fpm_frame_set_transfer_threads(value, threads, error, sizeof(error)) == FPM_OK, error);
    }
    FpmCudaKernelMetricsV1 kernel() {
        FpmCudaKernelMetricsV1 metrics{};
        require(fpm_frame_get_cuda_kernel_metrics(value, &metrics, sizeof(metrics), error, sizeof(error)) == FPM_OK, error);
        return metrics;
    }
    FpmTransferMetricsV1 transfer() {
        FpmTransferMetricsV1 metrics{};
        require(fpm_frame_get_transfer_metrics(value, &metrics, sizeof(metrics), error, sizeof(error)) == FPM_OK, error);
        return metrics;
    }
    ~Frame() { fpm_frame_destroy(value); }
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    void prepare(const fpm::Image& source, std::vector<fpm_coarse_handle*> models, int invert) {
        FpmFrameMetricsV1 metrics{};
        require(fpm_frame_prepare(value, view(source), models.data(), models.size(), invert, &metrics, error, sizeof(error)) == FPM_OK, error);
    }
    std::vector<fpm_result> match(fpm_coarse_handle* model, const fpm_config& config) {
        std::vector<fpm_result> results(256); std::size_t count = 0;
        require(fpm_frame_match(value, model, &config, results.data(), results.size(), &count, &lastCoarse, nullptr, 0,
            &lastExecution, sizeof(lastExecution), &lastFine, sizeof(lastFine), error, sizeof(error)) == FPM_OK, error);
        results.resize(count);
        return results;
    }
    FpmCoarseMetrics lastCoarse{};
    FpmCoarseExecutionMetricsV1 lastExecution{};
    FpmFineMetricsV1 lastFine{};
};
std::vector<fpm_result> handleMatch(Handle& handle, const fpm::Image& source, const fpm_config& config)
{
    std::vector<fpm_result> results(256); std::size_t count = 0;
    FpmCoarseMetrics metrics{}; char error[1024]{};
    require(fpm_coarse_match(handle.value, view(source), &config, results.data(), results.size(), &count,
        &metrics, error, sizeof(error)) == FPM_OK, error);
    results.resize(count);
    return results;
}
void same(const std::vector<fpm_result>& expected, const std::vector<fpm_result>& actual, const std::string& what)
{
    require(expected.size() == actual.size(), what + ": match count " + std::to_string(expected.size()) +
        " vs " + std::to_string(actual.size()));
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(std::memcmp(&expected[i], &actual[i], sizeof(fpm_result)) == 0, what + ": match " + std::to_string(i) + " differs");
}

struct Case {
    int sw, sh, tw, th, copies, area;
    double maxAngle;
    fpm_config config;
    bool flat = false;
    std::vector<fpm_angle_range> ranges{};
};
fpm_config config(int targets, double score, double overlap, double tolerance, int subpixel, int stop, int invert)
{
    fpm_config c{};
    c.max_targets = targets; c.min_score = score; c.max_overlap = overlap; c.tolerance_angle = tolerance;
    c.subpixel = subpixel; c.invert_source = invert; c.stop_layer = stop;
    return c;
}

int matchedTotal = 0, approximateRuns = 0, approximateIdentical = 0, tensorRuns = 0, tensorIdentical = 0;
void runCase(Case& item, unsigned seed)
{
    const auto templ = makeTemplate(item.tw, item.th, seed, item.flat);
    auto source = makeScene(item.sw, item.sh, templ, item.copies, item.maxAngle, seed + 7);
    if (item.config.invert_source)
        for (int y = 0; y < source.height(); ++y) for (int x = 0; x < source.width(); ++x) source.row(y)[x] = 255 - source.row(y)[x];
    if (!item.ranges.empty()) { item.config.ranges = item.ranges.data(); item.config.range_count = item.ranges.size(); }
    const std::string name = std::to_string(item.sw) + "x" + std::to_string(item.sh) + " template " +
        std::to_string(item.tw) + "x" + std::to_string(item.th) + " seed " + std::to_string(seed);
    Handle cpuModel(templ, item.area, 0);
    Frame cpu(0, 0, 0);
    cpu.prepare(source, {cpuModel.value}, item.config.invert_source);
    const auto expected = cpu.match(cpuModel.value, item.config);
    matchedTotal += int(expected.size());

    Handle cudaHandle(templ, item.area, 2);
    same(expected, handleMatch(cudaHandle, source, item.config), name + " handle");
    for (int batch : {64, 3}) {
        Frame gpu(2, 1, batch);
        gpu.prepare(source, {cpuModel.value}, item.config.invert_source);
        same(expected, gpu.match(cpuModel.value, item.config), name + " frame batch " + std::to_string(batch));
    }
    Frame unfused(2, 0, 16, 1);
    unfused.prepare(source, {cpuModel.value}, item.config.invert_source);
    same(expected, unfused.match(cpuModel.value, item.config), name + " unfused coarse, CPU pyramid");
    // Transfer modes, each prepared twice: the second prepare reuses the host
    // levels and the second search starts from the counters the first cleared.
    for (int mode : {1, 2, 3}) {
        for (int batch : {64, 3, 0}) {
            if (batch == 0 && mode != 1) continue; // fine batch 0: the CPU fine path reads lean host levels
            Frame lean(2, 1, batch, 0, mode);
            for (int repeat = 0; repeat < 2; ++repeat) {
                lean.prepare(source, {cpuModel.value}, item.config.invert_source);
                same(expected, lean.match(cpuModel.value, item.config), name + " transfer " + std::to_string(mode) +
                    " batch " + std::to_string(batch) + " repeat " + std::to_string(repeat));
            }
        }
    }
    // Exact kernel modes: coarse NCC 2/3/5/6, fine NCC 2, uploaded fine requests
    // (transfer bit 4) and four copy threads, each prepared twice.
    struct Variant { int coarse, fine, transfer, threads, batch; };
    for (const Variant v : {Variant{2, 0, 0, 1, 64}, Variant{3, 2, 0, 1, 64}, Variant{3, 2, 3, 4, 3},
             Variant{0, 0, 7, 1, 64}, Variant{3, 2, 7, 4, 64}, Variant{3, 2, 1, 4, 0},
             Variant{5, 0, 0, 1, 64}, Variant{5, 2, 3, 1, 3}, Variant{6, 0, 0, 1, 64}, Variant{6, 2, 3, 1, 64},
             Variant{6, 2, 3, 4, 3}, Variant{6, 2, 1, 1, 0}, Variant{6, 4, 3, 1, 64}, Variant{3, 4, 0, 1, 3},
             Variant{6, 5, 3, 1, 64}, Variant{5, 5, 3, 4, 3}, Variant{3, 5, 0, 1, 64}, Variant{6, 5, 1, 1, 3}}) {
        Frame frame(2, 1, v.batch, v.coarse, v.transfer);
        if (v.fine) frame.setFineMode(v.fine);
        if (v.threads != 1) frame.setThreads(v.threads);
        const std::string tag = name + " coarse " + std::to_string(v.coarse) + " fine " + std::to_string(v.fine) +
            " transfer " + std::to_string(v.transfer) + " threads " + std::to_string(v.threads) + " batch " + std::to_string(v.batch);
        for (int repeat = 0; repeat < 2; ++repeat) {
            frame.prepare(source, {cpuModel.value}, item.config.invert_source);
            same(expected, frame.match(cpuModel.value, item.config), tag + " repeat " + std::to_string(repeat));
        }
    }
    // The binary32 ablation (coarse 4, fine 3) must run; its outputs may differ.
    Frame approximate(2, 1, 64, 4, 3);
    approximate.setFineMode(3);
    approximate.prepare(source, {cpuModel.value}, item.config.invert_source);
    const auto approx = approximate.match(cpuModel.value, item.config);
    approximateRuns += 1; approximateIdentical += approx.size() == expected.size() &&
        (expected.empty() || std::memcmp(approx.data(), expected.data(), expected.size() * sizeof(fpm_result)) == 0);
    // So must the same ablation on the Tensor core kernels (coarse 7, fine 6).
    Frame approximateTensor(2, 1, 64, 7, 3);
    approximateTensor.setFineMode(6);
    approximateTensor.prepare(source, {cpuModel.value}, item.config.invert_source);
    const auto approxTensor = approximateTensor.match(cpuModel.value, item.config);
    tensorRuns += 1; tensorIdentical += approxTensor.size() == expected.size() &&
        (expected.empty() || std::memcmp(approxTensor.data(), expected.data(), expected.size() * sizeof(fpm_result)) == 0);
}

// One mode-3 frame (and one mode-1 frame without a fine batch) through a
// sequence of sources: repeated, resized (also in height only), inverted, prepared without a match,
// models of different depths, and a switch to mode 0 and back. Every match
// must equal a fresh CPU frame's.
void transferSequence()
{
    // ma has three levels and mb two, so a model change also changes the pyramid depth.
    const auto a = makeTemplate(40, 30, 21), b = makeTemplate(70, 52, 22);
    const auto big = makeScene(960, 720, a, 6, 30, 23), small = makeScene(320, 240, a, 3, 30, 24);
    const auto other = makeScene(960, 720, b, 4, 30, 25), shorter = makeScene(960, 600, a, 5, 30, 26);
    Handle ma(a, 64, 0), mb(b, 512, 0);
    const auto c = config(8, 0.6, 0.2, 30, 1, 0, 0), ci = config(8, 0.6, 0.2, 30, 1, 0, 1);
    struct Step { const fpm::Image* source; fpm_coarse_handle* model; bool invert; bool matchIt; int setMode; };
    const Step steps[] = {
        {&big, ma.value, false, true, -1}, {&big, ma.value, false, true, -1}, {&other, ma.value, false, false, -1},
        {&other, mb.value, false, true, -1}, {&small, ma.value, false, true, -1}, {&big, ma.value, true, true, -1},
        {&big, mb.value, false, true, -1}, {&big, ma.value, false, true, 0}, {&small, ma.value, true, true, 3},
        {&big, ma.value, false, true, -1}, {&shorter, ma.value, false, true, -1}, {&other, mb.value, true, true, -1},
    };
    Frame lean(2, 1, 64, 0, 3), hostLevels(2, 1, 0, 0, 1);
    for (std::size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i) {
        const auto& step = steps[i];
        if (step.setMode >= 0) lean.setTransfer(step.setMode);
        const auto& cfg = step.invert ? ci : c;
        Frame cpu(0, 0, 0);
        cpu.prepare(*step.source, {step.model}, step.invert);
        lean.prepare(*step.source, {step.model}, step.invert);
        hostLevels.prepare(*step.source, {step.model}, step.invert);
        if (!step.matchIt) continue;
        const auto expected = cpu.match(step.model, cfg);
        matchedTotal += int(expected.size());
        same(expected, lean.match(step.model, cfg), "transfer sequence step " + std::to_string(i));
        same(expected, hostLevels.match(step.model, cfg), "transfer sequence (fine batch 0) step " + std::to_string(i));
    }
}

// The lean paths must actually run: their metrics show it. The 2400x1800
// source is uploaded in three chunks of at most 2 MiB.
void transferMetrics()
{
    const auto templ = makeTemplate(40, 30, 31);
    const auto source = makeScene(2400, 1800, templ, 4, 180, 32);
    Handle model(templ, 64, 0);
    const auto c = config(6, 0.6, 0.2, 180, 0, 0, 0);
    Frame cpu(0, 0, 0);
    cpu.prepare(source, {model.value}, 0);
    const auto expected = cpu.match(model.value, c);
    require(!expected.empty(), "transfer metrics scene has no match");
    matchedTotal += int(expected.size());
    FpmFineMetricsV1 reference{};
    for (int mode : {0, 1, 2, 3}) {
        Frame frame(2, 1, 64, 0, mode);
        frame.prepare(source, {model.value}, 0);
        frame.prepare(source, {model.value}, 0);
        same(expected, frame.match(model.value, c), "transfer metrics mode " + std::to_string(mode));
        if (mode == 0) {
            reference = frame.lastFine;
            require(reference.repair_pixels > 0 && reference.calls > 1, "transfer metrics scene needs FP64 pixels and several fine calls");
        }
        // Lean calls clear the FP64-pixel count between fine calls too.
        require(frame.lastFine.repair_pixels == reference.repair_pixels && frame.lastFine.roi_pixels == reference.roi_pixels &&
            frame.lastFine.calls == reference.calls, "transfer mode " + std::to_string(mode) + ": fine counts differ from mode 0");
        const auto m = frame.transfer();
        const std::string tag = "transfer mode " + std::to_string(mode) + ": ";
        require(m.version == 1 && m.struct_bytes == sizeof(m) && m.mode == mode, tag + "metrics header");
        const bool leanPrepare = mode & 1, leanCalls = mode & 2;
        require((m.lean_prepares == 1) == leanPrepare, tag + "lean prepare count " + std::to_string(m.lean_prepares));
        if (leanPrepare) {
            require(m.source_registered == 1 && m.registered_bytes == 2400u * 1800u, tag + "level 0 not page-locked");
            require(m.upload_chunks == 3 && m.host_levels_skipped_bytes > 0 && m.host_level_rebuilds == 0,
                tag + "chunks " + std::to_string(m.upload_chunks) + ", skipped " + std::to_string(m.host_levels_skipped_bytes) +
                ", rebuilds " + std::to_string(m.host_level_rebuilds));
        }
        require((m.signal_waits >= 2) == leanCalls, tag + "signal waits " + std::to_string(m.signal_waits));
        require((frame.lastFine.device_ms < 0) == leanCalls && (frame.lastCoarse.device_ms < 0) == leanCalls,
            tag + "device times");
        require(frame.lastFine.calls >= 1, tag + "no fine call");
    }
    // Copy threads, the Q28 coarse tiles and the binary32 screen are visible in the metrics.
    Frame threaded(2, 1, 64, 3, 3);
    threaded.setThreads(4);
    threaded.setFineMode(2);
    threaded.prepare(source, {model.value}, 0);
    same(expected, threaded.match(model.value, c), "four copy threads, coarse 3, fine 2");
    const auto k = threaded.kernel();
    require(k.version == 1 && k.coarse_ncc_mode == 3 && k.fine_ncc_mode == 2 && k.copy_threads == 4, "kernel metrics header");
    require(k.coarse_fp64_scores > 0 && k.coarse_fp64_scores < k.coarse_windows / 4,
        "binary32 screen: " + std::to_string(k.coarse_fp64_scores) + " of " + std::to_string(k.coarse_windows));
    require(threaded.transfer().upload_chunks == 3, "four copy threads changed the chunks");
    // Warp-level window products screen the same way.
    for (int mode : {5, 6}) {
        Frame windows(2, 1, 64, mode, 3);
        windows.setFineMode(2);
        windows.prepare(source, {model.value}, 0);
        same(expected, windows.match(model.value, c), "coarse " + std::to_string(mode) + ", fine 2");
        const auto w = windows.kernel();
        require(w.coarse_ncc_mode == mode && w.coarse_fp64_scores > 0 && w.coarse_fp64_scores < w.coarse_windows / 4,
            "coarse " + std::to_string(mode) + " screen: " + std::to_string(w.coarse_fp64_scores) + " of " + std::to_string(w.coarse_windows));
    }
}

void denseMaps()
{
    for (unsigned seed : {3u, 4u}) {
        const auto templ = makeTemplate(31 + seed, 17 + 3 * seed, seed);
        const auto source = makeScene(140, 90, templ, 2, 180, seed);
        std::vector<double> maps[2];
        for (int backend : {0, 2}) {
            Handle handle(templ, 1u << 30, backend);
            std::size_t count = 0; FpmCoarseMetrics metrics{}; char error[1024]{};
            require(fpm_coarse_scoremap(handle.value, view(source), nullptr, 0, &count, &metrics, error, sizeof(error)) == FPM_BUFFER_TOO_SMALL, error);
            auto& map = maps[backend / 2];
            map.resize(count);
            require(fpm_coarse_scoremap(handle.value, view(source), map.data(), map.size(), &count, &metrics, error, sizeof(error)) == FPM_OK, error);
        }
        require(maps[0].size() == maps[1].size() && std::memcmp(maps[0].data(), maps[1].data(), maps[0].size() * sizeof(double)) == 0,
            "dense CUDA score map differs from CPU");
    }
}

void multipleModels()
{
    const auto a = makeTemplate(48, 36, 11), b = makeTemplate(30, 70, 12);
    auto source = makeScene(640, 480, a, 3, 180, 13);
    const auto other = makeScene(640, 480, b, 3, 180, 14);
    for (int y = 240; y < 480; ++y) std::memcpy(source.row(y), other.row(y), 640);
    Handle ma(a, 64, 0), mb(b, 128, 0);
    const auto c = config(6, 0.6, 0.2, 180, 1, 0, 0);
    Frame cpu(0, 0, 0), gpu(2, 1, 32);
    cpu.prepare(source, {ma.value, mb.value}, 0);
    gpu.prepare(source, {ma.value, mb.value}, 0);
    for (int repeat = 0; repeat < 2; ++repeat) {
        same(cpu.match(ma.value, c), gpu.match(ma.value, c), "two models: first");
        same(cpu.match(mb.value, c), gpu.match(mb.value, c), "two models: second");
    }
}

void validation()
{
    FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
    config.backend = 2; config.pyramid_mode = 2; config.batch_angles = 1;
    fpm_frame_handle* frame = nullptr;
    require(fpm_frame_create(&config, sizeof(config), &frame, nullptr, 0) == FPM_ERROR && !frame, "separable CUDA pyramid accepted");
    config.pyramid_mode = 1; config.batch_angles = 4;
    require(fpm_frame_create(&config, sizeof(config), &frame, nullptr, 0) == FPM_ERROR && !frame, "CUDA batch_angles 4 accepted");
    Frame valid(2, 1, 8);
    FpmCoarseOptimizationConfigV1 optimization{1, sizeof(FpmCoarseOptimizationConfigV1), 1, 0, 0, 0};
    require(fpm_frame_set_coarse_optimization(valid.value, &optimization, sizeof(optimization), nullptr, 0) == FPM_ERROR,
        "CUDA peak mode 1 accepted");
    optimization.peak_mode = 0; optimization.ncc_mode = 9;
    require(fpm_frame_set_coarse_optimization(valid.value, &optimization, sizeof(optimization), nullptr, 0) == FPM_ERROR,
        "CUDA coarse NCC mode 9 accepted");
    require(fpm_frame_set_transfer_mode(valid.value, 8, nullptr, 0) == FPM_ERROR, "CUDA transfer mode 8 accepted");
    require(fpm_frame_set_transfer_threads(valid.value, 17, nullptr, 0) == FPM_ERROR, "17 transfer threads accepted");
    require(fpm_frame_set_fine_ncc_mode(valid.value, 8, nullptr, 0) == FPM_ERROR, "CUDA fine NCC mode 8 accepted");
    Frame cpu(0, 0, 0);
    require(fpm_frame_set_transfer_mode(cpu.value, 1, nullptr, 0) == FPM_ERROR, "CPU transfer mode 1 accepted");
    require(fpm_frame_set_transfer_mode(cpu.value, 0, nullptr, 0) == FPM_OK, "CPU transfer mode 0 rejected");
    // Changing the mode invalidates the prepared frame.
    const auto templ = makeTemplate(24, 20, 41);
    const auto source = makeScene(200, 150, templ, 1, 0, 42);
    Handle model(templ, 64, 0);
    valid.prepare(source, {model.value}, 0);
    valid.setTransfer(3);
    std::vector<fpm_result> results(8); std::size_t count = 0;
    FpmCoarseMetrics metrics{}; FpmCoarseExecutionMetricsV1 execution{}; FpmFineMetricsV1 fine{};
    fpm_config c{};
    c.max_targets = 2; c.min_score = 0.5;
    require(fpm_frame_match(valid.value, model.value, &c, results.data(), results.size(), &count, &metrics, nullptr, 0,
        &execution, sizeof(execution), &fine, sizeof(fine), nullptr, 0) == FPM_ERROR, "match after a mode change accepted");
    // Lean prepare needs the device pyramid; with the CPU pyramid only lean calls apply.
    Frame cpuPyramid(2, 0, 16, 0, 3), reference(0, 0, 0);
    cpuPyramid.prepare(source, {model.value}, 0); reference.prepare(source, {model.value}, 0);
    same(reference.match(model.value, c), cpuPyramid.match(model.value, c), "transfer mode 3 with the CPU pyramid");
    require(cpuPyramid.transfer().lean_prepares == 0, "lean prepare ran with the CPU pyramid");
}
}

int main()
{
    {
        FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
        config.backend = 2; config.pyramid_mode = 1; config.batch_angles = 1;
        fpm_frame_handle* frame = nullptr; char error[1024]{};
        if (fpm_frame_create(&config, sizeof(config), &frame, error, sizeof(error)) != FPM_OK) {
            if (std::strstr(error, "no CUDA device")) { std::cout << "SKIP: " << error << "\n"; return 77; }
            std::cerr << "FAIL CUDA frame creation: " << error << "\n";
            return 1;
        }
        fpm_frame_destroy(frame);
    }
    std::vector<Case> cases = {
        {320, 240, 40, 30, 3, 64, 180, config(5, 0.7, 0, 180, 0, 0, 0)},
        {320, 240, 40, 30, 3, 64, 180, config(5, 0.7, 0, 180, 1, 0, 0)},
        {640, 480, 64, 64, 6, 256, 180, config(10, 0.6, 0.5, 180, 0, 0, 0)},
        {640, 480, 64, 64, 6, 256, 180, config(10, 0.6, 0.5, 180, 0, 1, 0)},
        {800, 600, 120, 80, 4, 256, 30, config(8, 0.75, 0.2, 30, 1, 0, 0)},
        {500, 400, 23, 57, 8, 64, 180, config(30, 0.5, 0.3, 180, 0, 0, 1)},
        {700, 500, 300, 14, 3, 128, 10, config(4, 0.6, 0, 10, 1, 0, 0)},
        {420, 300, 9, 9, 10, 16, 180, config(20, 0.8, 0, 180, 0, 0, 0)},
        {600, 450, 50, 50, 5, 1024, 0, config(12, 0.8, 0.8, 0, 0, 0, 0)},
        {300, 300, 32, 32, 2, 64, 180, config(5, 0.5, 0, 180, 0, 0, 0), true},
        {1200, 900, 150, 110, 5, 256, 180, config(6, 0.7, 0.1, 180, 1, 0, 0)},
        {640, 480, 60, 45, 5, 128, 180, config(8, 0.6, 0, 0, 1, 0, 0), false, {{-30, 10}, {150, 180}, {-180, -170}}},
        {200, 160, 180, 150, 1, 256, 5, config(2, 0.5, 0, 8, 1, 0, 0)},
        {60, 50, 80, 40, 1, 64, 0, config(2, 0.5, 0, 180, 0, 0, 0)},
    };
    const std::pair<const char*, void (*)()> named[] = {{"dense maps", denseMaps}, {"two models", multipleModels},
        {"validation", validation}, {"transfer sequence", transferSequence}, {"transfer metrics", transferMetrics}};
    try {
        for (std::size_t i = 0; i < cases.size(); ++i) runCase(cases[i], unsigned(101 + 17 * i));
        for (const auto& test : named) test.second();
        require(matchedTotal >= 30, "too few CPU matches to exercise the CUDA paths: " + std::to_string(matchedTotal));
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << "\n";
        return 1;
    }
    std::cout << "CUDA backend bit-identical on " << cases.size() << " scenes, " << matchedTotal << " CPU matches; "
              << "binary32 ablation identical on " << approximateIdentical << " of " << approximateRuns << " scenes, on the Tensor core kernels "
              << tensorIdentical << " of " << tensorRuns << "\n";
    return 0;
}
