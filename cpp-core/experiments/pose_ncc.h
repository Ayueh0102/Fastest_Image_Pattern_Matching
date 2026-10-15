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
// CPU operations; refine_auto() and refine_resident() can initialize Metal.
// The handle owns copies; source/template preparation is measured separately.
typedef struct FpmPoseMetrics {
    double wall_ms;                 // packing, allocations, dispatch, wait, output copy
    double device_ms;               // Metal command-buffer GPU interval; -1 if unavailable
    uint64_t host_buffer_bytes;     // CPU-accessible owned capacities, includes shared pages; NOT RSS
    uint64_t gpu_buffer_bytes;      // requested Metal buffer lengths (shared memory)
    uint64_t gpu_allocated_bytes;   // MTLDevice currentAllocatedSize snapshot, NOT peak RSS
} FpmPoseMetrics;

FPM_EXP_API void* fpm_pose_create(int backend, int threads, const float* source,
    int sw, int sh, const float* templ, int tw, int th, int channels,
    char* error, int error_capacity);
// Opt-in storage: 0 preserves separate CPU vectors and Metal buffers; 1 owns
// page-aligned images and wraps the same pages with Metal no-copy buffers.
// Mode 1 requires a Metal/auto handle and a unified-memory Metal device on use.
// Auto remains lazy; unavailable auto may still use its CPU path. No caller
// memory is borrowed; both modes copy inputs and preserve their lifetime contract.
// No fallback to duplicate storage on unsupported devices or allocation errors.
FPM_EXP_API void* fpm_pose_create_with_storage(int backend, int threads, const float* source,
    int sw, int sh, const float* templ, int tw, int th, int channels, int storage_mode,
    char* error, int error_capacity);
typedef struct FpmPoseStorageInfo {
    int mode;
    int reserved;
    uint64_t host_image_bytes;
    uint64_t gpu_image_bytes;
    uint64_t shared_image_bytes; // overlap of host/gpu image capacities, not additional memory
    uint64_t unique_image_bytes; // host + gpu - shared; excludes workspaces/driver/RSS
} FpmPoseStorageInfo;
// Current capacity accounting without allocation/dispatch; null returns -1.
FPM_EXP_API int fpm_pose_storage_info(void* handle, FpmPoseStorageInfo* info);
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

// Opt-in GPU-resident refinement diagnostics; all times are milliseconds.
// wait_ms includes GPU execution and must NOT be added to metrics.device_ms.
// Auxiliary bytes exclude resident source/template buffers; memory() reports
// all retained buffers. Dispatch counts include predicated inactive work.
typedef struct FpmPoseResidentMetrics {
    uint64_t command_buffers;
    uint64_t cpu_waits;
    uint64_t encoded_dispatches;
    uint64_t auxiliary_buffer_bytes;
    double setup_ms;                // this call's lazy Metal/resident pipeline setup
    double encode_ms;
    double wait_ms;
    double cpu_verify_ms;
    int cpu_verified;
    int reserved;
} FpmPoseResidentMetrics;
// Opt-in implementation selection; creation defaults to 0 (original resident
// kernels). Mode 1 fuses one grayscale candidate into one 256-thread GPU group;
// supports <=16384 template pixels, chunks of <=512 poses. No automatic fallback.
// Mode 2 uses 128 threads with 32-wide SIMD shuffles and compact gray statistics.
// SIMD width is checked on first dispatch; unsupported devices error, never fall back.
// Setting mode 1/2 rejects CPU handles, RGB and larger templates without changing
// the previous setting. Selection does not initialize lazy Metal resources.
// Existing ABI structs/functions and ordinary score/CPU refine stay unchanged.
FPM_EXP_API int fpm_pose_set_resident_mode(void* handle, int mode, char* error, int error_capacity);
FPM_EXP_API int fpm_pose_resident_mode(void* handle); // 0/1/2; -1 for null
// Diagnostics from the most recent explicit resident/rescue call. Rescue sums
// baseline and retries; auxiliary bytes are their maximum, not their sum.
// Reset at entry/error; CPU rescue returns zeros. Does not dispatch any work.
FPM_EXP_API int fpm_pose_last_resident_metrics(void* handle, FpmPoseResidentMetrics* metrics);
// Actual compiled fused pipeline from the last successful resident/rescue call;
// zeros for CPU, original multi-kernel mode 0, no call, or failed call. Changing
// the selected mode alone does not change this record. No compilation or work.
typedef struct FpmPoseResidentKernelInfo {
    uint32_t threads;
    uint32_t simd_width;
    uint64_t threadgroup_bytes;
} FpmPoseResidentKernelInfo;
FPM_EXP_API int fpm_pose_last_resident_kernel(void* handle, FpmPoseResidentKernelInfo* info);
// GPU owns the local solve, acceptance and stopping decisions. Only Metal or
// auto handles are supported; CPU handles and unavailable Metal return errors.
// Auto handles initialize Metal lazily, without applying workload thresholds.
// Resident auxiliary storage is capped at 16 MiB; a single pose that cannot
// fit returns an error rather than taking the older paths' oversized exception.
// cpu_verify must be 0 or 1 (recommended: 1). When 1, AFTER device work the CPU
// recomputes both original/final NCC scores and restores the seed on regression.
// These two evaluations are included in stats; there is no CPU polish. When 0,
// scores are device scores and CPU NCC monotonicity is not guaranteed; no CPU
// pixel scans occur. Both modes reject nonfinite/out-of-bounds device outputs.
// Config and stats may be null; both metrics outputs are required. Inputs and
// outputs must not overlap. wall_ms includes validation, setup and verification.
// On failure, provided metrics are reset (device_ms=-1); use memory() to query
// allocations retained by the handle after a failed call.
// GPU float arithmetic may follow a different path from either existing solver.
FPM_EXP_API int fpm_pose_refine_resident(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, int cpu_verify, float* refined, float* scores,
    FpmPoseRefineStats* stats, FpmPoseResidentMetrics* resident_metrics,
    FpmPoseMetrics* metrics, char* error, int error_capacity);

// Optional bounded local-basin rescue. The baseline runs once, then up to
// max_rescued_seeds low-scoring parents receive a fixed 16-point Halton bank
// (bases 2,3,5,7; indices 1..16), short refinement and at most one continuation.
// Every retry keeps the ORIGINAL input seed's bounds. No ground truth enters
// selection. The default score window is [0.1,0.8), descending score/stable tie.
typedef struct FpmPoseRescueConfig {
    int backend;                   // 0 = CPU, 1 = resident Metal + CPU verification
    int max_rescued_seeds;          // 0..64; 0 disables retries
    int short_iterations;          // 1..200
    int reserved;                  // must be zero
    double trigger_min_score;
    double trigger_score;
    double translation_radius;
    double angle_radius_degrees;
    double log_scale_radius;
    uint64_t max_retry_sample_iterations;
} FpmPoseRescueConfig;
typedef struct FpmPoseRescueMetrics {
    uint64_t eligible_seeds;
    uint64_t selected_seeds;
    uint64_t accepted_seeds;
    uint64_t budget_skipped_seeds;
    uint64_t parent_cap_skipped_seeds;
    uint64_t retry_starts;
    uint64_t continuations;
    uint64_t reserved_retry_sample_iterations;
    uint64_t retry_evaluations;
    uint64_t command_buffers;
    uint64_t cpu_waits;
    uint64_t auxiliary_buffer_bytes; // maximum resident GPU workspace, images excluded
    uint64_t host_workspace_bytes;  // retained rescue vector capacities
    double baseline_ms;
    double retry_ms;
    double cpu_verify_ms;           // baseline/retry exact CPU score checks
    int selected_backend;
    int reserved;
} FpmPoseRescueMetrics;
// Defaults: CPU, at most 4 parents, 8 short iterations, score window [0.1,0.8),
// Halton radii [3 px,3 px,6 deg,0.06 log-scale], 512*1024*1024 sample-iterations.
// The work proxy reserves template_width*height*channels*(16*short_iterations
// + max_iterations) per parent, including a full continuation even if unused.
// It is not a measured pixel-operation count: LM line-search trials and exact
// CPU verification add work. No parent starts unless its whole proxy fits.
FPM_EXP_API void fpm_pose_rescue_default_config(FpmPoseRescueConfig* config);
// CPU handles support only backend 0; backend 1 requires Metal/auto and never
// silently falls back. Every retry result is re-scored exactly on CPU before
// comparison; returned NCC never decreases from the selected baseline path.
// Stats sum baseline + ALL short/continuation iterations/evaluations, while
// status describes the chosen endpoint. Configs/stats may be null; both metrics
// outputs are required. Error resets metrics. Inputs/outputs must not overlap.
FPM_EXP_API int fpm_pose_refine_rescue(void* handle, const float* seeds, int count,
    const FpmPoseRefineConfig* config, const FpmPoseRescueConfig* rescue_config,
    float* refined, float* scores, FpmPoseRefineStats* stats,
    FpmPoseRescueMetrics* rescue_metrics, FpmPoseMetrics* metrics,
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
