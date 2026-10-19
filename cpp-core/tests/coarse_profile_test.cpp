// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_ncc.h"
#include "fpm/matcher.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
fpm_image cview(fpm::ImageView image) { return {image.data, image.width, image.height, image.stride}; }
fpm::Image noise(int width, int height, unsigned seed)
{
    fpm::Image result(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) result.row(y)[x] = uint8_t(rng());
    return result;
}
struct Handle {
    fpm_coarse_handle* value = nullptr;
    FpmCoarseMetrics metrics{};
    char error[1024]{};
    Handle(fpm::ImageView image, int backend, int area = 64) {
        if (fpm_coarse_create_backend(cview(image), area, backend, &value, &metrics, error, sizeof(error)) != FPM_OK)
            throw std::runtime_error(error);
        if (backend == 0) require(metrics.gpu_bytes == 0 && metrics.device_ms == -1, "CPU constructor initialized GPU resources");
    }
    ~Handle() { fpm_coarse_destroy(value); }
};

void checkProfile(const FpmCoarseProfileV1& profile, const FpmCoarseMetrics& metrics,
    int backend, size_t outputs, const fpm::Pattern& pattern, fpm::ImageView source)
{
    require(profile.version == 1 && profile.struct_bytes == sizeof(profile) && profile.backend == backend,
        "profile version/size/backend incorrect");
    const double stages[] = {profile.source_copy_invert_ms, profile.pyramid_ms, profile.coarse_rotation_ms,
        profile.coarse_ncc_ms, profile.peak_selection_ms, profile.fine_rotation_ms,
        profile.fine_ncc_ms, profile.fine_control_ms, profile.nms_ms, profile.unclassified_ms};
    double sum = 0;
    for (double value : stages) { require(std::isfinite(value) && value >= 0, "invalid exclusive stage time"); sum += value; }
    require(std::abs(sum-profile.core_wall_ms) <= 1e-6*std::max(1., profile.core_wall_ms),
        "exclusive stage times do not add to core wall");
    require(profile.api_wall_ms == metrics.wall_ms && profile.api_residual_ms >= 0 &&
        std::abs(profile.api_wall_ms-profile.core_wall_ms-profile.api_residual_ms) < 1e-7,
        "API residual does not account for outer work");
    require(profile.pyramid_levels == pattern.levels().size(), "incorrect pyramid layer count");
    require(profile.source_copy_bytes == uint64_t(source.width)*source.height, "source byte counter incorrect");
    require(profile.pyramid_live_bytes == profile.source_copy_bytes+profile.pyramid_generated_bytes,
        "pyramid byte counters inconsistent");
    uint64_t liveBytes = 0;
    int width = source.width, height = source.height;
    for (size_t layer = 0; layer < pattern.levels().size(); ++layer) {
        liveBytes += uint64_t(width)*height; width = (width+1)/2; height = (height+1)/2;
    }
    require(profile.pyramid_live_bytes == liveBytes, "pyramid byte counter is not nominal image output bytes");
    require(profile.coarse_angles >= profile.coarse_maps && profile.coarse_maps > 0 && profile.coarse_windows > 0,
        "missing coarse stage counts");
    require(profile.fine_maps == profile.fine_rotations && profile.fine_passes >= profile.fine_layer_visits,
        "fine stage counters inconsistent");
    require(profile.output_matches == outputs && profile.refined_candidates >= outputs,
        "candidate/output counts inconsistent");
    require(profile.coarse_rotation_peak_bytes <= profile.coarse_rotation_total_bytes &&
        profile.fine_roi_peak_bytes <= profile.fine_roi_total_bytes && profile.scoremap_peak_bytes <= profile.scoremap_total_bytes,
        "nominal single-output peaks exceed generated totals");
    require(profile.timed_transitions > 0, "profile did not enable clocks");
    if (backend == 0) require(metrics.device_ms == -1 && metrics.gpu_bytes == 0 && metrics.command_buffers == 0,
        "CPU profiling initialized or dispatched Metal");
}

void tests(int backend)
{
    fpm::Image source = noise(127, 97, 4301), templ = noise(25, 21, 4302);
    for (const fpm::Point origin : {fpm::Point{17, 29}, fpm::Point{76, 51}})
        for (int y = 0; y < templ.height(); ++y)
            std::copy_n(templ.row(y), templ.width(), source.row(int(origin.y)+y)+int(origin.x));
    std::vector<uint8_t> paddedSource(size_t(source.width()+7)*source.height(), 0xD3);
    for (int y = 0; y < source.height(); ++y)
        std::copy_n(source.row(y), source.width(), paddedSource.data()+size_t(y)*(source.width()+7));
    const fpm_image input{paddedSource.data(), source.width(), source.height(), size_t(source.width()+7)};
    fpm::Pattern pattern(templ.view(), 64);
    Handle handle(templ.view(), backend);
    FpmCoarseExecutionConfigV1 execution{};
    fpm_coarse_execution_default_config(&execution);
    require(execution.version == 1 && execution.struct_bytes == sizeof(execution) &&
        execution.pyramid_mode == 0 && execution.batch_angles == 1 && !execution.reserved0 && !execution.reserved1,
        "invalid execution defaults");
    auto set = [&] { return fpm_coarse_set_execution(handle.value, &execution, sizeof(execution), handle.error, sizeof(handle.error)); };
    require(set() == FPM_OK, handle.error);
    for (int invalid = 0; invalid < 7; ++invalid) {
        fpm_coarse_execution_default_config(&execution);
        if (invalid == 0) execution.version = 2;
        if (invalid == 1) execution.struct_bytes = 1;
        if (invalid == 2) execution.reserved0 = 1;
        if (invalid == 3) execution.reserved1 = 1;
        if (invalid == 4) execution.pyramid_mode = 3;
        if (invalid == 5) execution.batch_angles = 0;
        if (invalid == 6) execution.batch_angles = 5;
        require(set() == FPM_ERROR, "invalid execution configuration accepted");
    }
    fpm_coarse_execution_default_config(&execution);
    if (!backend) {
        execution.pyramid_mode = 1; require(set() == FPM_ERROR, "CPU accepted Metal pyramid");
        execution.pyramid_mode = 0; execution.batch_angles = 2;
        require(set() == FPM_ERROR, "CPU accepted Metal batching");
        fpm_coarse_execution_default_config(&execution);
    }
    require(fpm_coarse_set_execution(handle.value, &execution, sizeof(execution)-1, nullptr, 0) == FPM_ERROR &&
        fpm_coarse_set_execution(handle.value, nullptr, sizeof(execution), nullptr, 0) == FPM_ERROR &&
        fpm_coarse_set_execution(nullptr, &execution, sizeof(execution), nullptr, 0) == FPM_ERROR,
        "invalid execution pointer/size accepted");
    FpmCoarseExecutionMetricsV1 extra{};
    extra.version = 991;
    require(fpm_coarse_get_execution_metrics(handle.value, &extra, sizeof(extra)-1, nullptr, 0) == FPM_ERROR &&
        extra.version == 991 && fpm_coarse_get_execution_metrics(handle.value, nullptr, sizeof(extra), nullptr, 0) == FPM_ERROR,
        "invalid execution metrics pointer/size written or accepted");
    require(fpm_coarse_get_execution_metrics(handle.value, &extra, sizeof(extra), handle.error, sizeof(handle.error)) == FPM_OK &&
        extra.version == 1 && extra.struct_bytes == sizeof(extra) && extra.pyramid_mode == 0 && extra.batch_angles == 1,
        "failed setters changed effective execution options");
    for (int mode = 0; mode < 5; ++mode) {
        fpm_config config{5, .65, 0., mode == 0 ? 0. : 35., mode == 2, mode == 3, 0, mode == 1, nullptr, 0};
        const fpm_angle_range ranges[] = {{-35, -5}, {5, 35}};
        if (mode == 4) { config.ranges = ranges; config.range_count = 2; }
        std::vector<fpm_result> expected(5), actual(5);
        size_t count = 0, actualCount = 0;
        require(fpm_coarse_match(handle.value, input, &config, expected.data(), expected.size(), &count,
            &handle.metrics, handle.error, sizeof(handle.error)) == FPM_OK, handle.error);
        FpmCoarseProfileV1 profile{};
        require(fpm_coarse_match_profile(handle.value, input, &config, actual.data(), actual.size(), &actualCount,
            &handle.metrics, &profile, sizeof(profile), handle.error, sizeof(handle.error)) == FPM_OK, handle.error);
        require(actualCount == count && std::memcmp(expected.data(), actual.data(), count*sizeof(fpm_result)) == 0,
            "profiling changed match order, geometry or scores");
        checkProfile(profile, handle.metrics, backend, count, pattern, source.view());
        if (backend) {
            // Switch modes on this handle and compare every result bit, including
            // inverted-source, subpixel and half-resolution contracts.
            for (int pyramidMode : {1, 2, 0}) for (int batch : {4, 1}) {
                execution.pyramid_mode = pyramidMode; execution.batch_angles = batch;
                require(set() == FPM_OK, handle.error);
                require(fpm_coarse_match_profile(handle.value, input, &config, actual.data(), actual.size(), &actualCount,
                    &handle.metrics, &profile, sizeof(profile), handle.error, sizeof(handle.error)) == FPM_OK, handle.error);
                require(actualCount == count && std::memcmp(expected.data(), actual.data(), count*sizeof(fpm_result)) == 0,
                    "execution options changed complete matching results");
                checkProfile(profile, handle.metrics, backend, count, pattern, source.view());
                require(fpm_coarse_get_execution_metrics(handle.value, &extra, sizeof(extra), handle.error, sizeof(handle.error)) == FPM_OK,
                    handle.error);
                require(extra.pyramid_mode == pyramidMode && extra.batch_angles == batch &&
                    extra.pyramid_levels == (pyramidMode ? pattern.levels().size()-1 : 0) &&
                    extra.pyramid_command_buffers == (pyramidMode ? 1u : 0u), "execution counters/options not reset per match");
                require(extra.max_batch_maps <= uint64_t(batch) && extra.batch_source_arena_bytes <= 64*1024*1024 &&
                    extra.batch_statistics_bytes <= 16*1024*1024 && extra.pyramid_buffer_bytes <= handle.metrics.gpu_bytes,
                    "execution memory or group cap violated");
                if (batch == 1) require(extra.batch_calls == 0 && extra.batch_maps == 0, "unbatched path retained batch counters");
                else require(extra.batch_maps == profile.coarse_maps, "batch map count disagrees with core profile");
            }
        }
        // Profiled mode must not retain counters in a following unprofiled call.
        require(fpm_coarse_match(handle.value, input, &config, actual.data(), actual.size(), &actualCount,
            &handle.metrics, handle.error, sizeof(handle.error)) == FPM_OK && actualCount == count &&
            std::memcmp(expected.data(), actual.data(), count*sizeof(fpm_result)) == 0, "profile changed later matching state");
    }
    fpm_config config{5, .65, 0., 0., 0, 0, 0, 0, nullptr, 0};
    FpmCoarseProfileV1 profile{};
    size_t count = 0;
    require(fpm_coarse_match_profile(handle.value, input, &config, nullptr, 0, &count,
        &handle.metrics, &profile, sizeof(profile), handle.error, sizeof(handle.error)) == FPM_BUFFER_TOO_SMALL && count == 2,
        "profiled capacity query must run complete matching without partial output");
    profile.version = 991;
    require(fpm_coarse_match_profile(handle.value, input, &config, nullptr, 0, &count,
        &handle.metrics, &profile, sizeof(profile)-1, handle.error, sizeof(handle.error)) == FPM_ERROR &&
        count == 0 && profile.version == 991, "wrong profile buffer size was accepted or written");
    require(fpm_coarse_match_profile(handle.value, input, &config, nullptr, 0, &count,
        &handle.metrics, nullptr, sizeof(profile), handle.error, sizeof(handle.error)) == FPM_ERROR && count == 0,
        "missing profile pointer accepted");
    // CPU scoremap supports the additional CPU handle instead of dereferencing
    // the absent Metal provider. Exact map agreement is covered by NCC tests.
    require(fpm_coarse_scoremap(handle.value, input, nullptr, 0, &count,
        &handle.metrics, handle.error, sizeof(handle.error)) == FPM_BUFFER_TOO_SMALL && count > 0,
        "CPU/Metal handle scoremap size query failed");
    std::vector<double> map(count);
    require(fpm_coarse_scoremap(handle.value, input, map.data(), map.size(), &count,
        &handle.metrics, handle.error, sizeof(handle.error)) == FPM_OK, handle.error);
}
}

int main(int argc, char** argv)
{
    static_assert(sizeof(FpmCoarseMetrics) == 160 && sizeof(FpmCoarseProfileV1) == 296 &&
        sizeof(FpmCoarseExecutionConfigV1) == 32 && sizeof(FpmCoarseExecutionMetricsV1) == 192, "coarse ABI changed");
    try {
        const bool metal = argc == 2 && std::string(argv[1]) == "--metal";
        require(argc == 1 || metal, "Usage: coarse_profile_test [--metal]");
        tests(metal ? 1 : 0);
        std::cout << "Coarse profiling preserves outputs, exclusive stages, byte/count semantics and capacity behavior\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string text(error.what());
        if (text.find("is unavailable") != std::string::npos) { std::cout << "SKIP: " << text << '\n'; return 77; }
        std::cerr << text << '\n'; return 1;
    }
}
