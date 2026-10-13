// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void message(char* error, size_t capacity, const char* text)
{
    if (!error || !capacity) return;
    const size_t length = std::min(capacity - 1, std::strlen(text));
    std::memcpy(error, text, length); error[length] = '\0';
}
fpm::ImageView view(fpm_image image)
{
    return {image.data, image.width, image.height, image.stride};
}
template<class F> int protect(FpmCoarseMetrics* metrics, char* error, size_t capacity, F f)
{
    const auto start = Clock::now();
    message(error, capacity, "");
    if (metrics) *metrics = {};
    try {
        if (!metrics) throw std::invalid_argument("metrics pointer is required");
        const int result = f();
        metrics->wall_ms = elapsed(start);
        return result;
    } catch (const std::exception& e) { message(error, capacity, e.what()); }
    catch (...) { message(error, capacity, "unknown coarse NCC exception"); }
    if (metrics) metrics->wall_ms = elapsed(start);
    return FPM_ERROR;
}
fpm::MatchConfig convert(const fpm_config* config)
{
    fpm::MatchConfig c;
    if (!config) return c;
    c.maxTargets = config->max_targets; c.minScore = config->min_score;
    c.maxOverlap = config->max_overlap; c.toleranceAngle = config->tolerance_angle;
    c.subpixel = config->subpixel != 0; c.invertSource = config->invert_source != 0;
    c.stopLayer = config->stop_layer;
    c.backend = config->scalar ? fpm::Backend::Scalar : fpm::Backend::Auto;
    if (config->range_count && !config->ranges) throw std::invalid_argument("range buffer is null");
    for (size_t i = 0; i < config->range_count; ++i)
        c.angleRanges.push_back({config->ranges[i].min, config->ranges[i].max});
    return c;
}
}

struct fpm_coarse_handle {
    fpm::Pattern pattern;
    std::unique_ptr<fpm_coarse_detail::Backend> backend;
    fpm_coarse_handle(fpm_image image, int area) : pattern(view(image), area) {
#ifdef FPM_EXPERIMENT_METAL
        backend = fpm_coarse_detail::makeMetalBackend();
#else
        throw std::runtime_error("Metal coarse NCC is unavailable on this build");
#endif
    }
    FpmCoarseMetrics metrics() const {
        auto m = backend->metrics();
        for (const auto& level : pattern.levels())
            m.host_bytes += uint64_t(level.image.width()) * level.image.height();
        return m;
    }
};

extern "C" {
int fpm_coarse_create(fpm_image templ, int minArea, fpm_coarse_handle** out,
    FpmCoarseMetrics* metrics, char* error, size_t capacity)
{
    if (out) *out = nullptr;
    return protect(metrics, error, capacity, [&] {
        if (!out) throw std::invalid_argument("out handle pointer is required");
        const auto start = Clock::now();
        auto handle = std::make_unique<fpm_coarse_handle>(templ, minArea);
        *metrics = handle->metrics(); metrics->setup_ms = elapsed(start);
        *out = handle.release();
        return FPM_OK;
    });
}
void fpm_coarse_destroy(fpm_coarse_handle* handle) { delete handle; }
int fpm_coarse_match(fpm_coarse_handle* handle, fpm_image source, const fpm_config* config,
    fpm_result* results, size_t capacity, size_t* count, FpmCoarseMetrics* metrics,
    char* error, size_t errorCapacity)
{
    if (count) *count = 0;
    return protect(metrics, error, errorCapacity, [&] {
        if (!handle || !count || (!results && capacity))
            throw std::invalid_argument("handle, count and output buffer must be valid");
        handle->backend->resetMetrics();
        const auto matches = fpm::detail::matchWithCoarseNcc(view(source), handle->pattern,
            convert(config), handle->backend.get());
        *metrics = handle->metrics(); *count = matches.size();
        if (matches.size() > capacity) return FPM_BUFFER_TOO_SMALL;
        for (size_t i = 0; i < matches.size(); ++i) {
            for (int k = 0; k < 4; ++k)
                results[i].corners[k] = {matches[i].corners[k].x, matches[i].corners[k].y};
            results[i].center = {matches[i].center.x, matches[i].center.y};
            results[i].angle = matches[i].angle; results[i].score = matches[i].score;
        }
        return FPM_OK;
    });
}
int fpm_coarse_scoremap(fpm_coarse_handle* handle, fpm_image source, double* results,
    size_t capacity, size_t* count, FpmCoarseMetrics* metrics, char* error, size_t errorCapacity)
{
    if (count) *count = 0;
    return protect(metrics, error, errorCapacity, [&] {
        if (!handle || !count || (!results && capacity))
            throw std::invalid_argument("handle, count and output buffer must be valid");
        fpm::detail::validateImage(view(source));
        const auto& templ = handle->pattern.levels().front();
        const int width = source.width - templ.image.width() + 1;
        const int height = source.height - templ.image.height() + 1;
        if (width < 1 || height < 1) throw std::invalid_argument("NCC template must fit the source");
        *count = size_t(width) * height;
        handle->backend->resetMetrics();
        if (*count > capacity) { *metrics = handle->metrics(); return FPM_BUFFER_TOO_SMALL; }
        const auto map = handle->backend->compute(view(source), templ);
        std::copy(map.values.begin(), map.values.end(), results);
        *metrics = handle->metrics();
        return FPM_OK;
    });
}
}
