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
FPM_EXP_API void fpm_pose_destroy(void* handle);
#ifdef __cplusplus
}
#endif
