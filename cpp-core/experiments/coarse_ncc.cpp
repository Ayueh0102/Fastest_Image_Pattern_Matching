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
    FpmCoarseMetrics cpuMetrics{};
    FpmCoarseExecutionConfigV1 executionConfig{};
    fpm_coarse_handle(fpm_image image, int area, int backendMode) : pattern(view(image), area) {
        fpm_coarse_execution_default_config(&executionConfig);
        cpuMetrics.device_ms = -1;
        if (backendMode == 0) return;
#ifdef FPM_EXPERIMENT_METAL
        backend = fpm_coarse_detail::makeMetalBackend();
#else
        throw std::runtime_error("Metal coarse NCC is unavailable on this build");
#endif
    }
    void resetMetrics() {
        if (backend) backend->resetMetrics();
        else { cpuMetrics = {}; cpuMetrics.device_ms = -1; }
    }
    FpmCoarseMetrics metrics() const {
        auto m = backend ? backend->metrics() : cpuMetrics;
        for (const auto& level : pattern.levels())
            m.host_bytes += uint64_t(level.image.width()) * level.image.height();
        return m;
    }
};

struct fpm_frame_handle {
    FpmFrameConfigV1 config{};
    std::unique_ptr<fpm_coarse_detail::Backend> backend;
    fpm::detail::SourcePyramid pyramid;
    FpmFrameMetricsV1 frameMetrics{};
    FpmCoarseExecutionMetricsV1 prepareExecution{};
    uint64_t generation = 0;
    bool ready = false;
    FpmFrameMetricsV1 metrics() const {
        auto value = frameMetrics;
        value.version = 1; value.struct_bytes = sizeof(value); value.generation = generation;
        value.pyramid_levels = pyramid.levels.size(); value.host_pyramid_bytes = 0;
        for (const auto& level : pyramid.levels) value.host_pyramid_bytes += uint64_t(level.width())*level.height();
        if (backend) {
            value.coarse_gpu_bytes = backend->metrics().gpu_bytes;
            value.fine_gpu_bytes = backend->fineMetrics().retained_gpu_bytes;
        }
        return value;
    }
};

namespace {
int exportMatches(const std::vector<fpm::Match>& matches, fpm_result* results,
    size_t capacity, size_t* count)
{
    *count = matches.size();
    if (matches.size() > capacity) return FPM_BUFFER_TOO_SMALL;
    for (size_t i = 0; i < matches.size(); ++i) {
        for (int k = 0; k < 4; ++k)
            results[i].corners[k] = {matches[i].corners[k].x, matches[i].corners[k].y};
        results[i].center = {matches[i].center.x, matches[i].center.y};
        results[i].angle = matches[i].angle; results[i].score = matches[i].score;
    }
    return FPM_OK;
}
void copyProfile(FpmCoarseProfileV1& output, const fpm::detail::MatchStageProfile& input, int backend)
{
    output.version = 1; output.struct_bytes = sizeof(output); output.backend = backend;
#define COPY_PROFILE(field) output.field = input.field
    COPY_PROFILE(core_wall_ms); COPY_PROFILE(source_copy_invert_ms); COPY_PROFILE(pyramid_ms);
    COPY_PROFILE(coarse_rotation_ms); COPY_PROFILE(coarse_ncc_ms); COPY_PROFILE(peak_selection_ms);
    COPY_PROFILE(fine_rotation_ms); COPY_PROFILE(fine_ncc_ms); COPY_PROFILE(fine_control_ms);
    COPY_PROFILE(nms_ms); COPY_PROFILE(unclassified_ms);
    COPY_PROFILE(pyramid_levels); COPY_PROFILE(coarse_angles); COPY_PROFILE(coarse_maps); COPY_PROFILE(coarse_windows);
    COPY_PROFILE(coarse_candidates); COPY_PROFILE(fine_layer_visits); COPY_PROFILE(fine_passes);
    COPY_PROFILE(fine_rotations); COPY_PROFILE(fine_maps); COPY_PROFILE(fine_windows);
    COPY_PROFILE(refined_candidates); COPY_PROFILE(output_matches); COPY_PROFILE(source_copy_bytes);
    COPY_PROFILE(pyramid_generated_bytes); COPY_PROFILE(pyramid_live_bytes);
    COPY_PROFILE(coarse_rotation_total_bytes); COPY_PROFILE(coarse_rotation_peak_bytes);
    COPY_PROFILE(fine_roi_total_bytes); COPY_PROFILE(fine_roi_peak_bytes);
    COPY_PROFILE(scoremap_peak_bytes); COPY_PROFILE(scoremap_total_bytes); COPY_PROFILE(timed_transitions);
#undef COPY_PROFILE
}
}

extern "C" {
void fpm_frame_default_config(FpmFrameConfigV1* config)
{
    static_assert(sizeof(FpmFrameConfigV1) == 32, "frame config v1 ABI changed");
    static_assert(sizeof(FpmFineMetricsV1) == 216, "fine metrics v1 ABI changed");
    static_assert(sizeof(FpmFrameMetricsV1) == 80, "frame metrics v1 ABI changed");
    if (!config) return;
    *config = {}; config->version = 1; config->struct_bytes = sizeof(*config);
    config->backend = 1; config->pyramid_mode = 1; config->batch_angles = 4;
}
int fpm_frame_create(const FpmFrameConfigV1* config, size_t bytes,
    fpm_frame_handle** out, char* error, size_t capacity)
{
    if (out) *out = nullptr;
    message(error, capacity, "");
    try {
        if (!out || !config || bytes != sizeof(*config) || config->version != 1 ||
            config->struct_bytes != sizeof(*config) || config->reserved)
            throw std::invalid_argument("valid frame config v1, exact size and output are required");
        if (config->backend < 0 || config->backend > 1 || config->pyramid_mode < 0 ||
            config->pyramid_mode > 2 || config->batch_angles < 1 || config->batch_angles > 4 ||
            config->fine_batch_size < 0 || config->fine_batch_size > 1024)
            throw std::invalid_argument("invalid frame backend/pyramid/batch configuration");
        if (config->backend == 0 && (config->pyramid_mode || config->batch_angles != 1 || config->fine_batch_size))
            throw std::invalid_argument("CPU frame supports only CPU pyramid, single coarse angle and CPU fine search");
        auto frame = std::make_unique<fpm_frame_handle>();
        frame->config = *config;
        frame->prepareExecution.version = 1;
        frame->prepareExecution.struct_bytes = sizeof(frame->prepareExecution);
        frame->prepareExecution.pyramid_mode = config->pyramid_mode;
        frame->prepareExecution.batch_angles = config->batch_angles;
        if (config->backend) {
#ifdef FPM_EXPERIMENT_METAL
            frame->backend = fpm_coarse_detail::makeMetalBackend();
            FpmCoarseExecutionConfigV1 execution{};
            fpm_coarse_execution_default_config(&execution);
            execution.pyramid_mode = config->pyramid_mode;
            execution.batch_angles = config->batch_angles;
            frame->backend->setExecution(execution);
            frame->backend->setFineBatchSize(size_t(config->fine_batch_size));
#else
            throw std::runtime_error("Metal shared frame is unavailable on this build");
#endif
        }
        *out = frame.release();
        return FPM_OK;
    } catch (const std::exception& e) { message(error, capacity, e.what()); }
    catch (...) { message(error, capacity, "unknown frame creation exception"); }
    return FPM_ERROR;
}
void fpm_frame_destroy(fpm_frame_handle* frame) { delete frame; }
int fpm_frame_prepare(fpm_frame_handle* frame, fpm_image source,
    fpm_coarse_handle* const* patterns, size_t patternCount, int invert,
    FpmFrameMetricsV1* metrics, char* error, size_t capacity)
{
    const auto start = Clock::now();
    message(error, capacity, "");
    if (metrics) *metrics = {};
    try {
        if (frame) {
            frame->ready = false;
            if (frame->backend) { frame->backend->invalidateFineSource(); frame->backend->resetMetrics(); }
            frame->pyramid = {}; frame->frameMetrics = {}; frame->prepareExecution = {};
        }
        if (!frame || !patterns || !patternCount || !metrics || (invert != 0 && invert != 1))
            throw std::invalid_argument("frame, nonempty patterns, metrics and boolean invert are required");
        fpm::detail::validateImage(view(source));
        int top = 0;
        for (size_t i = 0; i < patternCount; ++i) {
            if (!patterns[i]) throw std::invalid_argument("pattern handle is null");
            top = std::max(top, int(patterns[i]->pattern.levels().size())-1);
        }
        const auto copy = Clock::now();
        frame->pyramid.inverted = invert != 0;
        frame->pyramid.levels.emplace_back(view(source));
        if (invert) {
            auto& level = frame->pyramid.levels.front();
            for (int y = 0; y < level.height(); ++y)
                for (int x = 0; x < level.width(); ++x) level.row(y)[x] = uint8_t(255-level.row(y)[x]);
        }
        frame->frameMetrics.source_copy_ms = elapsed(copy);
        const auto pyramid = Clock::now();
        if (!frame->backend || !frame->backend->buildPyramid(frame->pyramid.levels, top))
            for (int layer = 1; layer <= top; ++layer)
                frame->pyramid.levels.push_back(fpm::detail::pyrDown(frame->pyramid.levels.back().view()));
        frame->frameMetrics.pyramid_ms = elapsed(pyramid);
        if (frame->backend) frame->prepareExecution = frame->backend->executionMetrics();
        else {
            frame->prepareExecution.version = 1; frame->prepareExecution.struct_bytes = sizeof(frame->prepareExecution);
            frame->prepareExecution.batch_angles = 1; frame->prepareExecution.pyramid_device_ms = -1;
        }
        frame->frameMetrics.prepare_wall_ms = elapsed(start);
        ++frame->generation;
        frame->ready = true;
        *metrics = frame->metrics();
        return FPM_OK;
    } catch (const std::exception& e) { message(error, capacity, e.what()); }
    catch (...) { message(error, capacity, "unknown frame preparation exception"); }
    if (frame) {
        if (frame->backend) frame->backend->invalidateFineSource();
        frame->pyramid = {};
    }
    return FPM_ERROR;
}
int fpm_frame_get_metrics(fpm_frame_handle* frame, FpmFrameMetricsV1* metrics,
    size_t metricsBytes, FpmCoarseExecutionMetricsV1* execution, size_t executionBytes,
    char* error, size_t capacity)
{
    message(error, capacity, "");
    try {
        if (!frame || !metrics || metricsBytes != sizeof(*metrics) ||
            !execution || executionBytes != sizeof(*execution))
            throw std::invalid_argument("frame metrics/execution pointers and exact v1 sizes are required");
        *metrics = frame->metrics(); *execution = frame->prepareExecution;
        return FPM_OK;
    } catch (const std::exception& e) { message(error, capacity, e.what()); }
    catch (...) { message(error, capacity, "unknown frame metrics exception"); }
    return FPM_ERROR;
}
int fpm_frame_match(fpm_frame_handle* frame, fpm_coarse_handle* pattern,
    const fpm_config* config, fpm_result* results, size_t capacity, size_t* count,
    FpmCoarseMetrics* metrics, FpmCoarseProfileV1* profile, size_t profileBytes,
    FpmCoarseExecutionMetricsV1* execution, size_t executionBytes,
    FpmFineMetricsV1* fine, size_t fineBytes, char* error, size_t errorCapacity)
{
    const auto start = Clock::now();
    if (count) *count = 0;
    const bool writableProfile = profile && profileBytes == sizeof(*profile);
    if (writableProfile) *profile = {};
    fpm::detail::MatchStageProfile stages;
    const int status = protect(metrics, error, errorCapacity, [&] {
        if (!frame || !frame->ready || !pattern || !count || (!results && capacity) ||
            (!writableProfile && (profile || profileBytes)) || !execution || executionBytes != sizeof(*execution) ||
            !fine || fineBytes != sizeof(*fine))
            throw std::invalid_argument("ready frame, pattern, valid outputs and exact optional profile sizes are required");
        *execution = {}; *fine = {};
        const auto settings = convert(config);
        if (settings.invertSource != frame->pyramid.inverted)
            throw std::invalid_argument("config inversion differs from prepared frame");
        if (frame->backend) frame->backend->resetMetrics();
        const auto found = fpm::detail::matchWithCoarseNcc(frame->pyramid.levels.front().view(), pattern->pattern,
            settings, frame->backend.get(), writableProfile ? &stages : nullptr, &frame->pyramid);
        *metrics = frame->backend ? frame->backend->metrics() : FpmCoarseMetrics{};
        for (const auto& level : pattern->pattern.levels()) metrics->host_bytes += uint64_t(level.image.width())*level.image.height();
        if (frame->backend) {
            *execution = frame->backend->executionMetrics(); *fine = frame->backend->fineMetrics();
        } else {
            metrics->device_ms = -1;
            execution->version = 1; execution->struct_bytes = sizeof(*execution); execution->batch_angles = 1;
            execution->pyramid_device_ms = -1;
            fine->version = 1; fine->struct_bytes = sizeof(*fine); fine->device_ms = -1;
            if (writableProfile) {
                metrics->ncc_ms = stages.coarse_ncc_ms;
                metrics->score_maps = stages.coarse_maps; metrics->windows = stages.coarse_windows;
            }
        }
        ++frame->frameMetrics.models_matched;
        return exportMatches(found, results, capacity, count);
    });
    if (writableProfile) {
        copyProfile(*profile, stages, frame && frame->backend ? 1 : 0);
        if (metrics) {
            metrics->wall_ms = elapsed(start);
            profile->api_wall_ms = metrics->wall_ms;
            profile->api_residual_ms = metrics->wall_ms-stages.core_wall_ms;
        }
    }
    return status;
}

void fpm_coarse_execution_default_config(FpmCoarseExecutionConfigV1* config)
{
    static_assert(sizeof(FpmCoarseExecutionConfigV1) == 32, "execution config v1 ABI changed");
    if (!config) return;
    *config = {}; config->version = 1; config->struct_bytes = sizeof(*config); config->batch_angles = 1;
}
int fpm_coarse_set_execution(fpm_coarse_handle* handle,
    const FpmCoarseExecutionConfigV1* config, size_t configBytes, char* error, size_t capacity)
{
    message(error, capacity, "");
    try {
        if (!handle || !config || configBytes != sizeof(*config))
            throw std::invalid_argument("handle, execution config v1 pointer and exact structure size are required");
        if (config->version != 1 || config->struct_bytes != sizeof(*config) ||
            config->reserved0 || config->reserved1)
            throw std::invalid_argument("execution config version/size/reserved fields are invalid");
        if (config->pyramid_mode < 0 || config->pyramid_mode > 2 || config->batch_angles < 1 || config->batch_angles > 4)
            throw std::invalid_argument("pyramid_mode must be 0..2 and batch_angles must be 1..4");
        if (handle->backend) handle->backend->setExecution(*config);
        else if (config->pyramid_mode || config->batch_angles != 1)
            throw std::invalid_argument("CPU handles support only CPU pyramid and batch_angles=1");
        handle->executionConfig = *config;
        return FPM_OK;
    } catch (const std::exception& e) { message(error, capacity, e.what()); }
    catch (...) { message(error, capacity, "unknown execution configuration exception"); }
    return FPM_ERROR;
}
int fpm_coarse_get_execution_metrics(fpm_coarse_handle* handle,
    FpmCoarseExecutionMetricsV1* metrics, size_t metricsBytes, char* error, size_t capacity)
{
    static_assert(sizeof(FpmCoarseExecutionMetricsV1) == 192, "execution metrics v1 ABI changed");
    message(error, capacity, "");
    try {
        if (!handle || !metrics || metricsBytes != sizeof(*metrics))
            throw std::invalid_argument("handle, execution metrics v1 pointer and exact structure size are required");
        auto value = handle->backend ? handle->backend->executionMetrics() : FpmCoarseExecutionMetricsV1{};
        value.version = 1; value.struct_bytes = sizeof(value);
        value.pyramid_mode = handle->executionConfig.pyramid_mode;
        value.batch_angles = handle->executionConfig.batch_angles;
        if (!handle->backend) value.pyramid_device_ms = -1;
        *metrics = value;
        return FPM_OK;
    } catch (const std::exception& e) { message(error, capacity, e.what()); }
    catch (...) { message(error, capacity, "unknown execution metrics exception"); }
    return FPM_ERROR;
}
int fpm_coarse_create(fpm_image templ, int minArea, fpm_coarse_handle** out,
    FpmCoarseMetrics* metrics, char* error, size_t capacity)
{
    return fpm_coarse_create_backend(templ, minArea, 1, out, metrics, error, capacity);
}
int fpm_coarse_create_backend(fpm_image templ, int minArea, int backend,
    fpm_coarse_handle** out, FpmCoarseMetrics* metrics, char* error, size_t capacity)
{
    if (out) *out = nullptr;
    return protect(metrics, error, capacity, [&] {
        if (!out) throw std::invalid_argument("out handle pointer is required");
        if (backend != 0 && backend != 1) throw std::invalid_argument("backend must be 0 (CPU) or 1 (Metal)");
        const auto start = Clock::now();
        auto handle = std::make_unique<fpm_coarse_handle>(templ, minArea, backend);
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
        handle->resetMetrics();
        const auto matches = fpm::detail::matchWithCoarseNcc(view(source), handle->pattern,
            convert(config), handle->backend.get());
        *metrics = handle->metrics();
        return exportMatches(matches, results, capacity, count);
    });
}
int fpm_coarse_match_profile(fpm_coarse_handle* handle, fpm_image source, const fpm_config* config,
    fpm_result* results, size_t capacity, size_t* count, FpmCoarseMetrics* metrics,
    FpmCoarseProfileV1* profile, size_t profileBytes, char* error, size_t errorCapacity)
{
    static_assert(sizeof(FpmCoarseProfileV1) == 296, "profile v1 ABI changed");
    const auto apiBegin = Clock::now();
    if (count) *count = 0;
    const bool writableProfile = profile && profileBytes == sizeof(FpmCoarseProfileV1);
    if (writableProfile) *profile = {};
    fpm::detail::MatchStageProfile stages;
    const int result = protect(metrics, error, errorCapacity, [&] {
        if (!writableProfile) throw std::invalid_argument("profile v1 pointer and exact structure size are required");
        if (!handle || !count || (!results && capacity))
            throw std::invalid_argument("handle, count and output buffer must be valid");
        handle->resetMetrics();
        const auto matches = fpm::detail::matchWithCoarseNcc(view(source), handle->pattern,
            convert(config), handle->backend.get(), &stages);
        if (!handle->backend) {
            handle->cpuMetrics.ncc_ms = stages.coarse_ncc_ms;
            handle->cpuMetrics.score_maps = stages.coarse_maps;
            handle->cpuMetrics.windows = stages.coarse_windows;
        }
        *metrics = handle->metrics();
        return exportMatches(matches, results, capacity, count);
    });
    if (writableProfile) {
        copyProfile(*profile, stages, handle && handle->backend ? 1 : 0);
        if (metrics) {
            metrics->wall_ms = elapsed(apiBegin);
            profile->api_wall_ms = metrics->wall_ms;
            profile->api_residual_ms = metrics->wall_ms-stages.core_wall_ms;
        }
    }
    return result;
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
        handle->resetMetrics();
        if (*count > capacity) { *metrics = handle->metrics(); return FPM_BUFFER_TOO_SMALL; }
        const auto start = Clock::now();
        const auto map = handle->backend ? handle->backend->compute(view(source), templ) :
            fpm::detail::ncc(view(source), templ, fpm::Backend::Auto);
        if (!handle->backend) {
            handle->cpuMetrics.ncc_ms = elapsed(start); handle->cpuMetrics.score_maps = 1;
            handle->cpuMetrics.windows = map.values.size();
            handle->cpuMetrics.scoremap_peak_bytes = map.values.size()*sizeof(double);
        }
        std::copy(map.values.begin(), map.values.end(), results);
        *metrics = handle->metrics();
        return FPM_OK;
    });
}
}
