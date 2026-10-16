/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef FPM_COARSE_NCC_H
#define FPM_COARSE_NCC_H
#include "fpm/c_api.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fpm_coarse_handle fpm_coarse_handle;
/* All times are milliseconds. wall_ms covers the entire synchronous API;
 * ncc_ms is the sum of top-layer NCC calls, including upload/normalization.
 * device_ms is command-buffer GPU duration, -1 when unavailable. Intervals
 * overlap and must not be added. upload_ms measures CPU copies into shared
 * buffers (not discrete-GPU/PCIe transfer). host_bytes counts retained template pixels;
 * gpu_bytes counts retained shared buffers, including workspace_bytes. RSS
 * and these buffers overlap on Apple unified memory and must not be added.
 */
typedef struct FpmCoarseMetrics {
    double wall_ms, setup_ms, upload_ms, device_ms;
    double wait_ms, normalization_ms, ncc_ms, reserved_ms;
    uint64_t score_maps, windows, command_buffers, integer32_maps;
    uint64_t integer64_maps, host_bytes, gpu_bytes, workspace_bytes;
    uint64_t source_buffer_bytes, template_buffer_bytes, scoremap_peak_bytes, reserved_bytes;
} FpmCoarseMetrics;

/* Version 1 opt-in exclusive host-wall profile. Each core stage plus
 * unclassified_ms sums to core_wall_ms (roundoff aside); api_residual_ms is
 * API validation/result packing outside that core interval. Coarse NCC GPU wait time is
 * INCLUDED in coarse_ncc_ms; GPU pyramid wait is included in pyramid_ms.
 * Neither wait is added again to these exclusive stage times.
 * peak_selection_ms includes coarse candidate extraction/sort and fine maxima.
 * fine_control_ms includes local-angle control, subpixel fitting and temporary
 * cleanup outside measured fine ROI/NCC calls. All profiling bookkeeping and
 * clock overhead are part of the reported wall intervals; measure profile-off
 * separately before interpreting small performance deltas.
 * Byte counters are nominal image/map output bytes, not allocations or RSS:
 * pyramid_live_bytes includes source level 0; generated excludes it; neither
 * counts CPU horizontal-filter scratch (e.g. 128 MiB on the first 8K level).
 * Process peak RSS must be measured separately. Total
 * rotation/ROI/map bytes sum across calls, and peak is largest SINGLE output.
 * Maps retained together for subpixel fitting are not a combined memory peak.
 */
typedef struct FpmCoarseProfileV1 {
    uint32_t version, struct_bytes;
    int32_t backend, reserved;
    double api_wall_ms, core_wall_ms, source_copy_invert_ms, pyramid_ms;
    double coarse_rotation_ms, coarse_ncc_ms, peak_selection_ms;
    double fine_rotation_ms, fine_ncc_ms, fine_control_ms, nms_ms;
    double unclassified_ms, api_residual_ms;
    uint64_t pyramid_levels, coarse_angles, coarse_maps, coarse_windows;
    uint64_t coarse_candidates, fine_layer_visits, fine_passes;
    uint64_t fine_rotations, fine_maps, fine_windows, refined_candidates, output_matches;
    uint64_t source_copy_bytes, pyramid_generated_bytes, pyramid_live_bytes;
    uint64_t coarse_rotation_total_bytes, coarse_rotation_peak_bytes;
    uint64_t fine_roi_total_bytes, fine_roi_peak_bytes;
    uint64_t scoremap_peak_bytes, scoremap_total_bytes, timed_transitions;
} FpmCoarseProfileV1;

/* Explicit execution options. Defaults retain CPU pyramid and one coarse
 * angle per submission. GPU modes must fail explicitly if unsupported.
 * pyramid_mode: 0 CPU, 1 direct exact integer GPU filter, 2 separable exact
 * integer GPU filter. batch_angles: 1..4, a bounded CPU-rotation/GPU-NCC batch.
 * Batch source arenas are capped at 64 MiB and statistics at 16 MiB;
 * actual batches may be smaller to obey these limits. A single source larger
 * than 64 MiB retains the original single-map path, whose source buffer may
 * exceed that batch-arena cap. Fine search stays CPU.
 */
typedef struct FpmCoarseExecutionConfigV1 {
    uint32_t version, struct_bytes;
    int32_t pyramid_mode, batch_angles;
    uint64_t reserved0, reserved1;
} FpmCoarseExecutionConfigV1;
/* Additional diagnostics; original 160B metrics and 296B profile are unchanged.
 * Pyramid intervals are nested in profile.pyramid_ms. Existing metrics' NCC,
 * upload, wait and device intervals still describe coarse NCC only. GPU bytes
 * in original metrics count ALL retained allocations; the following byte fields
 * are breakdowns, not additional allocations to add to that total or to RSS.
 * Pyramid output is copied back for unchanged CPU fine search. Counters/times
 * reset per match; retained capacities and selected options remain queryable.
 * pyramid_levels counts generated levels (excluding level 0). Pyramid buffer
 * byte counters describe retained capacities including alignment padding.
 * batch_calls counts actual provider groups, including groups of one;
 * batch_maps counts their maps. batch_encode_ms and source_upload_bytes count
 * only multi-map batches; source upload bytes exclude alignment padding.
 */
typedef struct FpmCoarseExecutionMetricsV1 {
    uint32_t version, struct_bytes;
    int32_t pyramid_mode, batch_angles;
    double pyramid_wall_ms, pyramid_setup_ms, pyramid_upload_ms, pyramid_encode_ms;
    double pyramid_wait_ms, pyramid_device_ms, pyramid_copyout_ms, batch_encode_ms;
    uint64_t pyramid_command_buffers, pyramid_levels, pyramid_input_bytes, pyramid_output_bytes;
    uint64_t pyramid_scratch_bytes, pyramid_buffer_bytes, batch_calls, batch_maps;
    uint64_t max_batch_maps, batch_source_arena_bytes, batch_source_upload_bytes;
    uint64_t batch_template_buffer_bytes, batch_statistics_bytes, reserved_bytes;
} FpmCoarseExecutionMetricsV1;
/* Batched fine search diagnostics. All host times are ms and include waits;
 * device_ms overlaps wait_ms. Retained bytes are capacities, overlap RSS, and
 * are NOT included in the legacy coarse gpu_bytes. Reset per pattern match;
 * cached allocations survive until frame destruction. */
typedef struct FpmFineMetricsV1 {
    uint32_t version, struct_bytes;
    double wall_ms, setup_ms, upload_ms, rotation_ms, repair_ms;
    double ncc_ms, normalization_ms, encode_ms, wait_ms, device_ms;
    uint64_t calls, maps, roi_pixels, repair_pixels, repair_all_maps;
    uint64_t command_buffers, source_uploads, source_upload_bytes;
    uint64_t source_buffer_bytes, roi_buffer_bytes, guard_buffer_bytes;
    uint64_t descriptor_buffer_bytes, statistics_buffer_bytes, template_buffer_bytes;
    uint64_t retained_gpu_bytes, peak_batch_maps;
} FpmFineMetricsV1;
/* Opt-in fine normalization reuse diagnostics. row_buffer_bytes is INCLUDED
 * in FineMetricsV1.statistics_buffer_bytes, not additional to it. The new row
 * and dot dispatches share one command buffer; no extra CPU wait is introduced.
 * stats_device_ms measures that full command (not an isolated kernel). */
typedef struct FpmFineReuseMetricsV1 {
    uint32_t version, struct_bytes;
    int32_t ncc_mode, reserved;
    uint64_t row_buffer_bytes, row_dispatches, statistics_dispatches;
    double stats_device_ms;
} FpmFineReuseMetricsV1;
/* A frame owns one immutable source pyramid and execution backend, reused by
 * any number of template/scale handles. Explicit prepare ALWAYS refreshes pixels;
 * input pointer/dimensions alone never imply content identity. One caller at a
 * time. No asynchronous access or ownership of pattern handles is retained. */
typedef struct fpm_frame_handle fpm_frame_handle;
/* 0 original SIMD/window; 1 shared horizontal sum/square statistics. Idle
 * frame only. CPU accepts only 0. Public structs/defaults remain unchanged. */
FPM_API int fpm_frame_set_fine_ncc_mode(fpm_frame_handle* frame, int32_t mode,
    char* error, size_t error_capacity);
FPM_API int fpm_frame_get_fine_reuse_metrics(fpm_frame_handle* frame,
    FpmFineReuseMetricsV1* metrics, size_t metrics_bytes, char* error, size_t error_capacity);
typedef struct FpmFrameConfigV1 {
    uint32_t version, struct_bytes;
    int32_t backend, pyramid_mode, batch_angles, fine_batch_size;
    uint64_t reserved;
} FpmFrameConfigV1;
typedef struct FpmFrameMetricsV1 {
    uint32_t version, struct_bytes;
    double prepare_wall_ms, source_copy_ms, pyramid_ms;
    uint64_t generation, models_matched, pyramid_levels, host_pyramid_bytes;
    uint64_t coarse_gpu_bytes, fine_gpu_bytes;
} FpmFrameMetricsV1;
/* Optional coarse execution changes. peak_mode: 0 legacy, 1 exact tile tree.
 * ncc_mode: 0 scalar-window, 1 SIMD-group/window, 2 four adjacent outputs with
 * register strip reuse, 3 8x8 outputs with 16x16 template/source staging.
 * All NCC modes retain exact integer statistics and CPU double normalization.
 * Explicit opt-in; existing layouts/reserved fields and defaults unchanged. */
typedef struct FpmCoarseOptimizationConfigV1 {
    uint32_t version, struct_bytes;
    int32_t peak_mode, ncc_mode;
    uint64_t reserved0, reserved1;
} FpmCoarseOptimizationConfigV1;
FPM_API int fpm_frame_set_coarse_optimization(fpm_frame_handle* frame,
    const FpmCoarseOptimizationConfigV1* config, size_t config_bytes,
    char* error, size_t error_capacity);
/* Opt-in two-slot CPU rotation / GPU coarse NCC pipeline. Public match remains
 * synchronous. Each slot owns up to four source canvases, exact integer stats
 * and template pixels. Caps per slot: 64 MiB sources, 16 MiB statistics;
 * oversized/single/degenerate batches use the original synchronous path.
 * Reserved fields must be zero. Changing options requires an idle frame. */
typedef struct FpmCoarseOverlapConfigV1 {
    uint32_t version, struct_bytes;
    int32_t enabled, reserved0;
    uint64_t reserved1, reserved2;
} FpmCoarseOverlapConfigV1;
/* Per-model diagnostics, reset on match/prepare. Times are ms. submit_ms
 * includes copy/encode/commit; finish_wait_ms measures only finish waits.
 * device_ms is summed command GPU duration; device_span_ms includes gaps
 * between first GPU start and last GPU end. They overlap host times and must
 * not be added. -1 means unavailable GPU timestamps. Pending counts submitted
 * commands not yet consumed, not proof of concurrent device execution.
 * Fallback work is excluded from async timings and counted in fallback_ms.
 * Retained slot capacities are included in legacy coarse gpu_bytes, overlap
 * RSS on unified memory, and remain allocated when the option is disabled.
 * No raw caller image, model or template pointers survive begin. */
typedef struct FpmCoarseOverlapMetricsV1 {
    uint32_t version, struct_bytes;
    int32_t enabled, reserved;
    double submit_ms, copy_ms, encode_ms, finish_wait_ms;
    double normalization_ms, device_ms, device_span_ms, fallback_ms;
    uint64_t submitted_batches, submitted_maps, submissions_with_pending, max_pending;
    uint64_t completed_batches, fallback_batches, fallback_maps, fallback_single;
    uint64_t fallback_degenerate, fallback_oversized, source_upload_bytes, template_upload_bytes;
    uint64_t slot0_source_bytes, slot0_statistics_bytes, slot0_template_bytes;
    uint64_t slot1_source_bytes, slot1_statistics_bytes, slot1_template_bytes;
    uint64_t retained_gpu_bytes, drained_batches;
} FpmCoarseOverlapMetricsV1;
FPM_API int fpm_frame_set_coarse_overlap(fpm_frame_handle* frame,
    const FpmCoarseOverlapConfigV1* config, size_t config_bytes,
    char* error, size_t error_capacity);
FPM_API int fpm_frame_coarse_overlap_metrics(fpm_frame_handle* frame,
    FpmCoarseOverlapMetricsV1* metrics, size_t metrics_bytes,
    char* error, size_t error_capacity);
FPM_API void fpm_frame_default_config(FpmFrameConfigV1* config);
FPM_API int fpm_frame_create(const FpmFrameConfigV1* config, size_t config_bytes,
    fpm_frame_handle** out, char* error, size_t error_capacity);
FPM_API void fpm_frame_destroy(fpm_frame_handle* frame);
/* Build to the deepest provided model. Patterns are consulted only during this
 * call. Failed prepare invalidates the old frame to prevent stale use. */
FPM_API int fpm_frame_prepare(fpm_frame_handle* frame, fpm_image source,
    fpm_coarse_handle* const* patterns, size_t pattern_count, int invert_source,
    FpmFrameMetricsV1* metrics, char* error, size_t error_capacity);
FPM_API int fpm_frame_get_metrics(fpm_frame_handle* frame, FpmFrameMetricsV1* metrics,
    size_t metrics_bytes, FpmCoarseExecutionMetricsV1* prepare_execution,
    size_t execution_bytes, char* error, size_t error_capacity);
/* config invert_source must agree with the prepared snapshot. Profile optional:
 * use NULL/0 to disable clocks. Result capacity semantics match coarse_match.
 * Coarse/execution/fine diagnostics are per model; prepare costs are separate
 * and must be counted ONCE in a timed multiscale workflow. */
FPM_API int fpm_frame_match(fpm_frame_handle* frame, fpm_coarse_handle* pattern,
    const fpm_config* config, fpm_result* results, size_t capacity, size_t* count,
    FpmCoarseMetrics* metrics, FpmCoarseProfileV1* profile, size_t profile_bytes,
    FpmCoarseExecutionMetricsV1* execution, size_t execution_bytes,
    FpmFineMetricsV1* fine, size_t fine_bytes, char* error, size_t error_capacity);

FPM_API void fpm_coarse_execution_default_config(FpmCoarseExecutionConfigV1* config);
FPM_API int fpm_coarse_set_execution(fpm_coarse_handle* handle,
    const FpmCoarseExecutionConfigV1* config, size_t config_bytes, char* error, size_t error_capacity);
FPM_API int fpm_coarse_get_execution_metrics(fpm_coarse_handle* handle,
    FpmCoarseExecutionMetricsV1* metrics, size_t metrics_bytes, char* error, size_t error_capacity);

/* Experimental, explicitly selected Metal coarse NCC. Each handle is owned by
 * one caller at a time. Existing CPU entry points and ABI 1 are unchanged.
 * Shader compilation occurs at create; no silent CPU fallback on failure.
 * Metrics and out/count pointers are required. Error strings are optional.
 */
FPM_API int fpm_coarse_create(fpm_image templ, int min_area,
    fpm_coarse_handle** out, FpmCoarseMetrics* metrics, char* error, size_t error_capacity);
/* Additional explicit backend constructor: 0 portable CPU, 1 existing Metal
 * dense coarse NCC. CPU works without Metal and never creates a GPU context.
 * It shares the exact matching algorithm with the Metal constructor.
 * Original fpm_coarse_create remains equivalent to backend=1.
 */
FPM_API int fpm_coarse_create_backend(fpm_image templ, int min_area, int backend,
    fpm_coarse_handle** out, FpmCoarseMetrics* metrics, char* error, size_t error_capacity);
FPM_API void fpm_coarse_destroy(fpm_coarse_handle* handle);
/* GPU computes exact integer sum/square/product for each top-layer window.
 * CPU keeps double normalization, rotation, peak selection, fine search and
 * suppression. The source pyramid defaults to CPU; execution config can select
 * an exact GPU pyramid. This is not a fully GPU-resident coarse search.
 * Config/result capacity semantics match fpm_match, including recomputation
 * after BUFFER_TOO_SMALL. Config may be NULL; no partial results are written.
 */
FPM_API int fpm_coarse_match(fpm_coarse_handle* handle, fpm_image source,
    const fpm_config* config, fpm_result* results, size_t capacity, size_t* count,
    FpmCoarseMetrics* metrics, char* error, size_t error_capacity);
/* Profiling is enabled only through this additional entry point; old match()
 * does not read profiling clocks or maintain profiling counters. Profile and
 * profile_bytes==sizeof(FpmCoarseProfileV1) are required. Version/size are output
 * fields. BUFFER_TOO_SMALL still profiles the completed match without writing
 * partial results. On errors, any available profile is partial diagnostic data.
 */
FPM_API int fpm_coarse_match_profile(fpm_coarse_handle* handle, fpm_image source,
    const fpm_config* config, fpm_result* results, size_t capacity, size_t* count,
    FpmCoarseMetrics* metrics, FpmCoarseProfileV1* profile, size_t profile_bytes,
    char* error, size_t error_capacity);
/* Diagnostic dense NCC against the ORIGINAL template (not pyramid top).
 * Output is row-major (source.w-template.w+1)*(source.h-template.h+1).
 * A too-small buffer receives no partial output and matching is not run.
 */
FPM_API int fpm_coarse_scoremap(fpm_coarse_handle* handle, fpm_image source,
    double* results, size_t capacity, size_t* count, FpmCoarseMetrics* metrics,
    char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
