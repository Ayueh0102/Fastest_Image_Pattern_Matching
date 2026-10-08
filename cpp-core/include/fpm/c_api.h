/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef FPM_C_API_H
#define FPM_C_API_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(FPM_SHARED)
# if defined(FPM_BUILDING_LIBRARY)
#  define FPM_API __declspec(dllexport)
# else
#  define FPM_API __declspec(dllimport)
# endif
#else
# define FPM_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fpm_pattern fpm_pattern;
typedef struct { const uint8_t* data; int width, height; size_t stride; } fpm_image;
typedef struct { double x, y; } fpm_point;
typedef struct { double min, max; } fpm_angle_range;
typedef struct {
	int max_targets;
	double min_score, max_overlap, tolerance_angle;
	int subpixel, invert_source, scalar, stop_layer;
	const fpm_angle_range* ranges;
	size_t range_count;
} fpm_config;
typedef struct {
	fpm_point corners[4]; /* LT, RT, RB, LB */
	fpm_point center;
	double angle, score;
} fpm_result;
enum { FPM_OK = 0, FPM_ERROR = 1, FPM_BUFFER_TOO_SMALL = 2 };

FPM_API unsigned fpm_abi_version(void); /* ABI 1 */
FPM_API fpm_config fpm_default_config(void);
/* Pixels are copied. Destroy the returned handle with fpm_destroy. */
FPM_API int fpm_learn(fpm_image image, int min_area, fpm_pattern** out,
	char* error, size_t error_capacity);
FPM_API void fpm_destroy(fpm_pattern* pattern);
/* Call with results=NULL,capacity=0 to query count. Too-small buffers receive
 * no partial results. Retrying recomputes the match. No C++ exception escapes.
 * Config is optional (NULL uses defaults). Pattern supports concurrent reads.
 * Error storage belongs to the caller, is NUL-terminated when capacity > 0.
 * Valid memory for all pointers and image rows remains the caller's obligation.
 */
FPM_API int fpm_match(fpm_image source, const fpm_pattern* pattern,
	const fpm_config* config, fpm_result* results, size_t capacity,
	size_t* count, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
