// SPDX-License-Identifier: BSD-2-Clause
#include "fpm/c_api.h"
#include "fpm/matcher.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>

struct fpm_pattern {
	fpm::Pattern value;
	fpm_pattern(fpm_image image, int area) : value({image.data, image.width, image.height, image.stride}, area) {}
};

namespace {
void message(char* error, size_t capacity, const char* text)
{
	if (!error || !capacity) return;
	const size_t length = std::min(capacity - 1, std::strlen(text));
	std::memcpy(error, text, length);
	error[length] = '\0';
}
template<class Function> int protect(char* error, size_t capacity, Function function)
{
	message(error, capacity, "");
	try { return function(); }
	catch (const std::exception& e) { message(error, capacity, e.what()); }
	catch (...) { message(error, capacity, "unknown C++ exception"); }
	return FPM_ERROR;
}
}

extern "C" {
unsigned fpm_abi_version(void) { return 1; }
fpm_config fpm_default_config(void) { return {70, 0.5, 0, 0, 0, 0, 0, 0, nullptr, 0}; }
int fpm_learn(fpm_image image, int area, fpm_pattern** out, char* error, size_t capacity)
{
	if (out) *out = nullptr;
	return protect(error, capacity, [&]() -> int {
		if (!out) throw std::invalid_argument("out pattern pointer is required");
		*out = new fpm_pattern(image, area);
		return FPM_OK;
	});
}
void fpm_destroy(fpm_pattern* pattern) { delete pattern; }
int fpm_match(fpm_image source, const fpm_pattern* pattern, const fpm_config* config,
	fpm_result* results, size_t capacity, size_t* count, char* error, size_t errorCapacity)
{
	if (count) *count = 0;
	return protect(error, errorCapacity, [&]() -> int {
		if (!pattern || !count || (!results && capacity))
			throw std::invalid_argument("pattern, count and output buffer must be valid");
		fpm::MatchConfig c;
		if (config) {
			c.maxTargets = config->max_targets;
			c.minScore = config->min_score;
			c.maxOverlap = config->max_overlap;
			c.toleranceAngle = config->tolerance_angle;
			c.subpixel = config->subpixel != 0;
			c.invertSource = config->invert_source != 0;
			c.stopLayer = config->stop_layer;
			c.backend = config->scalar ? fpm::Backend::Scalar : fpm::Backend::Auto;
			if (config->range_count && !config->ranges) throw std::invalid_argument("range buffer is null");
			for (size_t i = 0; i < config->range_count; ++i)
				c.angleRanges.push_back({config->ranges[i].min, config->ranges[i].max});
		}
		const auto matches = fpm::match({source.data, source.width, source.height, source.stride}, pattern->value, c);
		*count = matches.size();
		if (matches.size() > capacity) return FPM_BUFFER_TOO_SMALL;
		for (size_t i = 0; i < matches.size(); ++i) {
			for (int k = 0; k < 4; ++k) results[i].corners[k] = {matches[i].corners[k].x, matches[i].corners[k].y};
			results[i].center = {matches[i].center.x, matches[i].center.y};
			results[i].angle = matches[i].angle;
			results[i].score = matches[i].score;
		}
		return FPM_OK;
	});
}
} // extern C
