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
// backend: 0 = C++ (threads >= 1), 1 = Metal, 2 = C++ with lazy Metal auto
// refinement. Mode 2 does not initialize Metal at creation. Handles are NOT
// thread safe. score(), refine(), and refine_accelerated() on mode 2 remain
// CPU operations; only refine_auto() can select and initialize Metal.
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
// Creation mode (0/1/2), or -1 for null. device() names the default score backend.
FPM_EXP_API int fpm_pose_backend(void* handle);
// Current owned buffers even before the first score; no work is dispatched.
FPM_EXP_API int fpm_pose_memory(void* handle, FpmPoseMetrics* metrics);
// Each pose = [tx, ty, theta_degrees, log_scale]. x-right/y-up, origin at
// (width/2,height/2), positive rotation CCW. Sample source at t + s R theta u.
// Outside source is zero, included in fixed rectangular template support.
// NCC centers each channel then jointly normalizes. No mask or antialias filter.
// A degenerate template/patch returns -2 (invalid), never a perfect match.
FPM_EXP_API int fpm_pose_score(void* handle, const float* poses, int count,
    float* scores, FpmPoseMetrics* metrics, char* error, int error_capacity);

// Continuous local refinement; this path runs portable C++ on any kind of
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

// Opt-in workload routing; thresholds are tunable heuristics, not a guarantee
// that the selected backend is fastest. Samples count scalar channel values:
// pose_samples = template_width * template_height * channels;
// batch_samples = pose_samples * seed_count (one nominal evaluation, not the
// measured total iterations). Routing never inspects seed scores or truth.
typedef struct FpmPoseAutoConfig {
    uint64_t min_pose_samples;
    uint64_t min_batch_samples;
    int min_iterations;             // 1..200; compared with refine max_iterations
    int cpu_polish_iterations;      // 0..30; used only for a selected Metal route
} FpmPoseAutoConfig;
typedef enum FpmPoseRouteReason {
    FPM_POSE_ROUTE_CPU_HANDLE = 0,
    FPM_POSE_ROUTE_BELOW_POSE_WORK = 1,
    FPM_POSE_ROUTE_BELOW_BATCH_WORK = 2,
    FPM_POSE_ROUTE_BELOW_ITERATIONS = 3,
    FPM_POSE_ROUTE_METAL_SELECTED = 4,
    FPM_POSE_ROUTE_METAL_UNAVAILABLE = 5,
    FPM_POSE_ROUTE_CPU_NO_ACTIVE_PARAMETERS = 6
} FpmPoseRouteReason;
typedef struct FpmPoseRoute {
    int selected_backend;           // 0 = CPU, 1 = Metal; meaningful on success
    int reason;                     // FpmPoseRouteReason
    uint64_t pose_samples;
    uint64_t batch_samples;
    double metal_setup_ms;          // this call's lazy setup attempt, 0 if reused
} FpmPoseRoute;
// Defaults calibrated for warm M4 Pro versus 8-thread CPU: 131072 pose samples,
// no batch minimum, 4 iterations, no CPU polish. This is a workload heuristic,
// not a universal hardware threshold. Zero sample thresholds disable sample gates.
FPM_EXP_API void fpm_pose_auto_default_config(FpmPoseAutoConfig* config);
FPM_EXP_API const char* fpm_pose_route_reason(int reason);
// route and metrics are required; both configs and stats may be null.
// CPU handles stay CPU. Existing Metal handles can select CPU while retaining
// their buffers. Auto handles initialize Metal only for an eligible call.
// Read route.selected_backend for this call; device() still names the handle's
// default score backend, including after an auto call has selected Metal.
// Missing/uncompiled Metal falls back with METAL_UNAVAILABLE; shader, allocation
// and dispatch failures remain errors. Wall time includes routing and lazy
// setup; memory includes all retained CPU, Metal and solver buffers.
FPM_EXP_API int fpm_pose_refine_auto(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, const FpmPoseAutoConfig* auto_config,
    float* refined, float* scores, FpmPoseRefineStats* stats, FpmPoseRoute* route,
    FpmPoseMetrics* metrics, char* error, int error_capacity);
FPM_EXP_API void fpm_pose_destroy(void* handle);
#ifdef __cplusplus
}
#endif
