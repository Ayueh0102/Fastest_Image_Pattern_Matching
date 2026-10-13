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

/* Experimental, explicitly selected Metal coarse NCC. Each handle is owned by
 * one caller at a time. Existing CPU entry points and ABI 1 are unchanged.
 * Shader compilation occurs at create; no silent CPU fallback on failure.
 * Metrics and out/count pointers are required. Error strings are optional.
 */
FPM_API int fpm_coarse_create(fpm_image templ, int min_area,
    fpm_coarse_handle** out, FpmCoarseMetrics* metrics, char* error, size_t error_capacity);
FPM_API void fpm_coarse_destroy(fpm_coarse_handle* handle);
/* GPU computes exact integer sum/square/product for each top-layer window.
 * CPU keeps double normalization, pyramid, rotation, peak selection, fine
 * search and suppression. This is not a fully GPU-resident coarse search.
 * Config/result capacity semantics match fpm_match, including recomputation
 * after BUFFER_TOO_SMALL. Config may be NULL; no partial results are written.
 */
FPM_API int fpm_coarse_match(fpm_coarse_handle* handle, fpm_image source,
    const fpm_config* config, fpm_result* results, size_t capacity, size_t* count,
    FpmCoarseMetrics* metrics, char* error, size_t error_capacity);
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
