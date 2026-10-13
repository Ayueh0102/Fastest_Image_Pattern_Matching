// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_ncc.h"
#include "fpm/matcher.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
fpm_image view(fpm::ImageView value) { return {value.data, value.width, value.height, value.stride}; }
fpm::Image noise(int width, int height, unsigned seed)
{
    fpm::Image result(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) result.row(y)[x] = uint8_t(rng());
    return result;
}
void paste(fpm::Image& image, const fpm::Image& patch, int x, int y)
{
    for (int row = 0; row < patch.height(); ++row) std::copy_n(patch.row(row), patch.width(), image.row(y+row)+x);
}
struct Pattern {
    fpm_coarse_handle* value = nullptr;
    std::size_t depth = 0;
    Pattern(const fpm::Image& image, int area = 64) : depth(fpm::Pattern(image.view(), area).levels().size()) {
        FpmCoarseMetrics metrics{}; char error[1024]{};
        require(fpm_coarse_create_backend(view(image.view()), area, 0, &value, &metrics, error, sizeof(error)) == FPM_OK, error);
    }
    ~Pattern() { fpm_coarse_destroy(value); }
    Pattern(const Pattern&) = delete;
    Pattern& operator=(const Pattern&) = delete;
};
std::vector<fpm_result> baseline(Pattern& pattern, fpm::ImageView source, const fpm_config& config)
{
    std::vector<fpm_result> results(size_t(config.max_targets)); std::size_t count = 0;
    FpmCoarseMetrics metrics{}; char error[1024]{};
    require(fpm_coarse_match(pattern.value, view(source), &config, results.data(), results.size(), &count,
        &metrics, error, sizeof(error)) == FPM_OK, error);
    results.resize(count); return results;
}
void equal(const std::vector<fpm_result>& actual, const std::vector<fpm_result>& expected)
{
    require(actual.size() == expected.size() && (actual.empty() ||
        std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(fpm_result)) == 0),
        "shared frame changed score, geometry, output count or stable order");
}
struct Frame {
    fpm_frame_handle* value = nullptr;
    char error[1024]{};
    FpmFrameMetricsV1 metrics{};
    FpmCoarseExecutionMetricsV1 preparedExecution{}, execution{};
    FpmCoarseMetrics coarse{};
    FpmCoarseProfileV1 profile{};
    FpmFineMetricsV1 fine{};
    uint64_t generation = 0, matched = 0;
    bool gpuFine;
    Frame(int backend, bool fineEnabled) : gpuFine(fineEnabled) {
        FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
        if (!backend) { config.backend = 0; config.pyramid_mode = 0; config.batch_angles = 1; }
        config.fine_batch_size = fineEnabled ? 128 : 0;
        require(fpm_frame_create(&config, sizeof(config), &value, error, sizeof(error)) == FPM_OK, error);
    }
    ~Frame() { fpm_frame_destroy(value); }
    void prepare(fpm::ImageView source, const std::vector<fpm_coarse_handle*>& models, bool inverted = false) {
        require(fpm_frame_prepare(value, view(source), models.data(), models.size(), inverted, &metrics, error, sizeof(error)) == FPM_OK, error);
        ++generation; matched = 0; inspect();
        require(metrics.pyramid_levels > 0 && metrics.host_pyramid_bytes >= uint64_t(source.width)*source.height,
            "prepared source storage is missing from frame metrics");
    }
    void inspect() {
        require(fpm_frame_get_metrics(value, &metrics, sizeof(metrics), &preparedExecution, sizeof(preparedExecution), error, sizeof(error)) == FPM_OK, error);
        require(metrics.version == 1 && metrics.struct_bytes == sizeof(metrics) && metrics.generation == generation &&
            metrics.models_matched == matched, "frame generation/model counters are inconsistent");
    }
    int raw(Pattern& pattern, const fpm_config& config, fpm_result* results, std::size_t capacity,
        std::size_t& count, bool profiled = true) {
        const int status = fpm_frame_match(value, pattern.value, &config, results, capacity, &count, &coarse,
            profiled ? &profile : nullptr, profiled ? sizeof(profile) : 0,
            &execution, sizeof(execution), &fine, sizeof(fine), error, sizeof(error));
        if (status == FPM_OK || status == FPM_BUFFER_TOO_SMALL) ++matched;
        return status;
    }
    std::vector<fpm_result> match(Pattern& pattern, const fpm_config& config, bool profiled = true) {
        std::vector<fpm_result> results(size_t(config.max_targets)); std::size_t count = 0;
        require(raw(pattern, config, results.data(), results.size(), count, profiled) == FPM_OK, error);
        results.resize(count);
        require(execution.version == 1 && execution.struct_bytes == sizeof(execution) &&
            fine.version == 1 && fine.struct_bytes == sizeof(fine), "per-model diagnostic ABI metadata incorrect");
        require(execution.pyramid_levels == 0 && execution.pyramid_command_buffers == 0,
            "per-model call rebuilt the shared pyramid");
        if (profiled) {
            require(profile.source_copy_invert_ms == 0 && profile.pyramid_ms == 0 && profile.source_copy_bytes == 0 &&
                profile.pyramid_generated_bytes == 0 && profile.pyramid_levels == pattern.depth,
                "profile charged shared preparation to each model");
            if (gpuFine) require(fine.maps == profile.fine_maps && fine.peak_batch_maps <= 128,
                "fine provider work or configured batch size changed");
        }
        inspect();
        require(metrics.coarse_gpu_bytes == coarse.gpu_bytes && metrics.fine_gpu_bytes == fine.retained_gpu_bytes,
            "frame retained allocation totals disagree with model diagnostics");
        return results;
    }
};
void lifecycle(int backend, bool fine)
{
    auto source = noise(139, 117, 6182), small = noise(13, 11, 8189), large = noise(25, 21, 8881);
    paste(source, small, 9, 11); paste(source, small, 109, 91);
    paste(source, large, 48, 32); paste(source, large, 84, 58);
    Pattern shallow(small, 1024), deep(large, 256);
    const std::vector<fpm_coarse_handle*> models{shallow.value, deep.value};
    Frame frame(backend, fine);
    fpm_config config{8, .6, .25, 24, 1, 0, 0, 0, nullptr, 0};
    const auto expectedSmall = baseline(shallow, source.view(), config), expectedLarge = baseline(deep, source.view(), config);
    if (expectedSmall.size() < 2 || expectedLarge.size() < 2)
        throw std::runtime_error("fixture must contain real nonempty matches for both templates: small="+
            std::to_string(expectedSmall.size())+", large="+std::to_string(expectedLarge.size()));
    std::size_t count = 123;
    require(frame.raw(deep, config, nullptr, 0, count) == FPM_ERROR && count == 0, "unprepared frame accepted matching");
    frame.prepare(source.view(), models);
    require(frame.metrics.pyramid_levels == deep.depth && deep.depth > shallow.depth, "shared pyramid did not use deepest model");
    equal(frame.match(shallow, config), expectedSmall);
    equal(frame.match(deep, config), expectedLarge);
    equal(frame.match(shallow, config, false), expectedSmall);
    const auto preparation = frame.preparedExecution;
    equal(frame.match(deep, config), expectedLarge);
    require(std::memcmp(&preparation, &frame.preparedExecution, sizeof(preparation)) == 0,
        "per-model metrics reset destroyed preparation diagnostics");

    // Too-small capacity must leave all output bytes untouched, then a retry
    // recomputes the model and is counted as another completed model call.
    fpm_result guard{}; std::memset(&guard, 0xC7, sizeof(guard)); const auto originalGuard = guard;
    require(frame.raw(deep, config, &guard, 1, count) == FPM_BUFFER_TOO_SMALL && count == expectedLarge.size() &&
        std::memcmp(&guard, &originalGuard, sizeof(guard)) == 0, "capacity query wrote partial results");
    equal(frame.match(deep, config), expectedLarge);
    require(frame.raw(deep, config, nullptr, 0, count) == FPM_BUFFER_TOO_SMALL && count == expectedLarge.size(),
        "null output capacity query failed");
    frame.inspect();

    // Invalid output structure lengths are rejected without writing that buffer.
    FpmCoarseProfileV1 profile{}; profile.version = 901;
    FpmCoarseExecutionMetricsV1 execution{}; execution.version = 902;
    FpmFineMetricsV1 diagnostics{}; diagnostics.version = 903;
    FpmCoarseMetrics metrics{};
    require(fpm_frame_match(frame.value, deep.value, &config, &guard, 1, &count, &metrics,
        &profile, sizeof(profile)-1, &execution, sizeof(execution), &diagnostics, sizeof(diagnostics), frame.error, sizeof(frame.error)) == FPM_ERROR &&
        count == 0 && profile.version == 901 && std::memcmp(&guard, &originalGuard, sizeof(guard)) == 0, "invalid profile size accepted/written");
    require(fpm_frame_match(frame.value, deep.value, &config, &guard, 1, &count, &metrics,
        nullptr, 0, &execution, sizeof(execution)-1, &diagnostics, sizeof(diagnostics), frame.error, sizeof(frame.error)) == FPM_ERROR &&
        execution.version == 902, "invalid execution size accepted/written");
    require(fpm_frame_match(frame.value, deep.value, &config, &guard, 1, &count, &metrics,
        nullptr, 0, &execution, sizeof(execution), &diagnostics, sizeof(diagnostics)-1, frame.error, sizeof(frame.error)) == FPM_ERROR &&
        diagnostics.version == 903, "invalid fine metrics size accepted/written");
    FpmFrameMetricsV1 invalidMetrics{}; invalidMetrics.version = 904;
    require(fpm_frame_get_metrics(frame.value, &invalidMetrics, sizeof(invalidMetrics)-1, &execution, sizeof(execution), frame.error, sizeof(frame.error)) == FPM_ERROR &&
        invalidMetrics.version == 904, "invalid frame metrics size accepted/written");
    frame.inspect();

    // Mutation preserves the exact input address but must not change an explicit
    // snapshot until prepare is called again. Refresh moves real objects.
    auto changed = noise(source.width(), source.height(), 8872);
    paste(changed, small, 25, 77); paste(changed, small, 95, 13);
    paste(changed, large, 4, 9); paste(changed, large, 65, 55);
    auto* sameAddress = source.row(0);
    std::copy_n(changed.row(0), size_t(source.width())*source.height(), source.row(0));
    require(source.row(0) == sameAddress, "test did not preserve source address");
    equal(frame.match(deep, config), expectedLarge);
    frame.prepare(source.view(), models);
    const auto changedLarge = baseline(deep, source.view(), config);
    require(changedLarge.size() >= 2, "refreshed fixture lost real detections");
    equal(frame.match(deep, config), changedLarge);

    config.invert_source = 1;
    require(frame.raw(deep, config, nullptr, 0, count) == FPM_ERROR && count == 0, "snapshot inversion mismatch was accepted");
    for (int y = 0; y < source.height(); ++y) for (int x = 0; x < source.width(); ++x) source.row(y)[x] = 255-source.row(y)[x];
    frame.prepare(source.view(), models, true);
    equal(frame.match(deep, config), baseline(deep, source.view(), config));

    // Failure invalidates prior contents; no stale result is available afterward.
    const fpm_image invalid{nullptr, source.width(), source.height(), size_t(source.width())};
    require(fpm_frame_prepare(frame.value, invalid, models.data(), models.size(), 1, &frame.metrics, frame.error, sizeof(frame.error)) == FPM_ERROR,
        "invalid source prepare succeeded");
    frame.matched = 0; frame.inspect();
    require(frame.metrics.pyramid_levels == 0 && frame.metrics.host_pyramid_bytes == 0, "failed prepare retained an accessible old snapshot");
    require(frame.raw(deep, config, nullptr, 0, count) == FPM_ERROR && count == 0, "matching used snapshot after failed prepare");
    frame.prepare(source.view(), {shallow.value}, true);
    require(frame.raw(deep, config, nullptr, 0, count) == FPM_ERROR && count == 0, "insufficient prepared depth silently rebuilt levels");
    frame.prepare(source.view(), models, true);
    equal(frame.match(deep, config), baseline(deep, source.view(), config));
}
void templateLifetime(int backend, bool fine)
{
    auto source = noise(159, 103, 4901); std::vector<fpm::Image> templates;
    for (unsigned index = 0; index < 5; ++index) {
        templates.push_back(noise(17, 15, 6800+index)); paste(source, templates.back(), 5+int(index)*29, 31);
    }
    Frame frame(backend, fine);
    fpm_config config{4, .75, 0, 0, 1, 0, 0, 0, nullptr, 0};
    for (int iteration = 0; iteration < 15; ++iteration) {
        // Allocators commonly reuse the just-destroyed pattern address here.
        // The frame must never infer template identity solely from that address.
        Pattern pattern(templates[size_t(iteration)%templates.size()], 256);
        if (iteration == 0) frame.prepare(source.view(), {pattern.value});
        const auto expected = baseline(pattern, source.view(), config);
        require(!expected.empty(), "lifetime fixture has no real match");
        equal(frame.match(pattern, config), expected);
    }
    require(frame.generation == 1 && frame.matched == 15, "template lifetime caused implicit source refresh");
}
void configurationErrors()
{
    FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
    require(config.version == 1 && config.struct_bytes == sizeof(config) && config.backend == 1 &&
        config.pyramid_mode == 1 && config.batch_angles == 4 && config.fine_batch_size == 0 && config.reserved == 0,
        "frame defaults changed unexpectedly");
    for (int kind = 0; kind < 8; ++kind) {
        fpm_frame_default_config(&config);
        if (kind == 0) config.version = 2;
        if (kind == 1) config.struct_bytes = 1;
        if (kind == 2) config.reserved = 1;
        if (kind == 3) config.backend = 2;
        if (kind == 4) config.pyramid_mode = 3;
        if (kind == 5) config.batch_angles = 0;
        if (kind == 6) config.fine_batch_size = 1025;
        if (kind == 7) config.backend = 0; // CPU cannot silently accept GPU options.
        fpm_frame_handle* output = reinterpret_cast<fpm_frame_handle*>(uintptr_t(1));
        require(fpm_frame_create(&config, sizeof(config), &output, nullptr, 0) == FPM_ERROR && output == nullptr,
            "invalid frame creation config accepted or retained output handle");
    }
    fpm_frame_default_config(&config); fpm_frame_handle* output = nullptr;
    require(fpm_frame_create(&config, sizeof(config)-1, &output, nullptr, 0) == FPM_ERROR &&
        fpm_frame_create(nullptr, sizeof(config), &output, nullptr, 0) == FPM_ERROR &&
        fpm_frame_create(&config, sizeof(config), nullptr, nullptr, 0) == FPM_ERROR,
        "invalid frame creation pointer/size accepted");
}
}
int main(int argc, char** argv)
{
    static_assert(sizeof(FpmFrameConfigV1) == 32 && sizeof(FpmFrameMetricsV1) == 80 && sizeof(FpmFineMetricsV1) == 216,
        "shared-frame/fine diagnostics ABI changed");
    try {
        const bool metal = argc == 2 && std::string(argv[1]) == "--metal";
        const bool fine = argc == 2 && std::string(argv[1]) == "--fine";
        require(argc == 1 || metal || fine, "Usage: shared_frame_test [--metal|--fine]");
        configurationErrors(); lifecycle(metal || fine, fine); templateLifetime(metal || fine, fine);
        std::cout << "Shared source lifecycle, refresh, model identity, exact matching and ABI errors passed\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string text(error.what());
        if (text.find("is unavailable") != std::string::npos) { std::cout << "SKIP: " << text << '\n'; return 77; }
        std::cerr << text << '\n'; return 1;
    }
}
