// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include <stdint.h>

#if defined(_WIN32)
# ifdef FPM_EXPERIMENT_BUILD
#  define FPM_EXP_API __declspec(dllexport)
# else
#  define FPM_EXP_API __declspec(dllimport)
# endif
#else
# define FPM_EXP_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

// Experimental API 1, independent of the stable fpm_c ABI. Dense float32 HWC,
// channels = 1 or 3. Inputs must be finite and abs(value) <= 16.
// backend: 0 = C++ (threads >= 1), 1 = Metal. Handles are NOT thread safe.
// The handle owns copies; source/template preparation is measured separately.
typedef struct FpmPoseMetrics {
    double wall_ms;                 // packing, allocations, dispatch, wait, output copy
    double device_ms;               // Metal command-buffer GPU interval; -1 if unavailable
    uint64_t host_buffer_bytes;     // owned vector capacities, NOT whole-process RSS
    uint64_t gpu_buffer_bytes;      // requested Metal buffer lengths (shared memory)
    uint64_t gpu_allocated_bytes;   // MTLDevice currentAllocatedSize snapshot, NOT peak RSS
} FpmPoseMetrics;

FPM_EXP_API void* fpm_pose_create(int backend, int threads, const float* source,
    int sw, int sh, const float* templ, int tw, int th, int channels,
    char* error, int error_capacity);
FPM_EXP_API const char* fpm_pose_device(void* handle);
// Current owned buffers even before the first score; no work is dispatched.
FPM_EXP_API int fpm_pose_memory(void* handle, FpmPoseMetrics* metrics);
// Each pose = [tx, ty, theta_degrees, log_scale]. x-right/y-up, origin at
// (width/2,height/2), positive rotation CCW. Sample source at t + s R theta u.
// Outside source is zero, included in fixed rectangular template support.
// NCC centers each channel then jointly normalizes. No mask or antialias filter.
// A degenerate template/patch returns -2 (invalid), never a perfect match.
FPM_EXP_API int fpm_pose_score(void* handle, const float* poses, int count,
    float* scores, FpmPoseMetrics* metrics, char* error, int error_capacity);

// Continuous local refinement; this path runs portable C++ on either kind of
// handle. It minimizes 1-NCC with an analytic bilinear-sampling Jacobian and a
// damped Gauss-Newton step. Accepted scores never decrease. This is a local
// solver, not a guarantee that the best rotation/scale has been found.
typedef struct FpmPoseRefineConfig {
    int max_iterations;
    int optimize_scale;
    int optimize_angle;
    double max_translation;         // independent x/y bounds from each input seed
    double max_angle_degrees;       // angle bound from each input seed
    double max_log_scale;           // log-scale bound from each input seed
} FpmPoseRefineConfig;
typedef struct FpmPoseRefineStats {
    int iterations;
    int evaluations;
    int status;                    // 0: small step, 1: iteration limit,
                                   // 2: degenerate patch/template, 3: stalled
} FpmPoseRefineStats;
// Defaults: 30 iterations, angle/scale enabled, +/-12 px, +/-15 deg,
// +/-log(1.3). A zero bound freezes that coordinate. Config may be null below.
FPM_EXP_API void fpm_pose_refine_default_config(FpmPoseRefineConfig* config);
// Input/output poses have the same layout as score(). Stats may be null.
// Inputs/outputs must not overlap. Uses the thread count supplied at creation.
// Refinement reports CPU wall time; device_ms is -1 even for a Metal handle.
FPM_EXP_API int fpm_pose_refine(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, float* refined, float* scores,
    FpmPoseRefineStats* stats, FpmPoseMetrics* metrics, char* error, int error_capacity);
// Optional device-accelerated refinement: batched derivative reductions and
// candidate scores on a capable backend, with small CPU solves. GPU acceptance
// has a float noise tolerance; the trajectory need not equal CPU refinement.
// Exact CPU verification restores the input seed on final score regression.
// Optional CPU polish (0..30 iterations) stays inside the ORIGINAL seed bounds.
// Stats include device iterations, CPU polish and verification evaluations;
// iterations <= config.max_iterations + cpu_polish_iterations on a device.
// Status describes the LAST phase (CPU polish, when enabled), not the sum:
// 1 may mean only the polish budget was exhausted; 0 also covers a seed already
// scoring 1. A status never certifies geometric accuracy or a global optimum.
// CPU/unsupported handles explicitly fall back to fpm_pose_refine unchanged:
// cpu_polish_iterations is validated but ignored, and device_ms remains -1.
// Device time sums all dispatches, or is -1 if any timestamp is unavailable.
FPM_EXP_API int fpm_pose_refine_accelerated(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, int cpu_polish_iterations, float* refined,
    float* scores, FpmPoseRefineStats* stats, FpmPoseMetrics* metrics,
    char* error, int error_capacity);
FPM_EXP_API void fpm_pose_destroy(void* handle);
#ifdef __cplusplus
}
#endif
