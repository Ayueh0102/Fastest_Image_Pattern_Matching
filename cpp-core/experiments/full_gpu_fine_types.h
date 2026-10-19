// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#ifdef __METAL_VERSION__
#define FG_FINE_UINT uint
#define FG_FINE_COUNTER atomic_uint
#else
#include <cstdint>
#define FG_FINE_UINT std::uint32_t
#define FG_FINE_COUNTER std::uint32_t
#endif

// Mode 0 retains the original full-ROI-cache/direct-sampling evaluator.
// Mode 1 tiles only ROIs that exceed the full-ROI cache capacity.
// Mode 2 adds proven-bounded uint32 per-lane moments, widened before reduction;
// larger templates retain the original uint64 arithmetic.
// Mode 3 also reuses source moments across overlapping cached-ROI windows;
// noncached ROIs follow mode 2. Neither mode adds threadgroup/device scratch.
// Mode 4 maps each cached window to a 32-thread SIMD group, reconstructing
// the same four exact stripe moments. The host uses mode 2 when the fine
// pipeline's SIMD width is not 32; noncached ROIs also follow mode 2.
// Profiling counts which mode-4 angle evaluations took each of those routes.
struct FGFineOptions {
    FG_FINE_UINT mode;
    FG_FINE_UINT diagnosticsEnabled;
};

// One record per global FGLevel. Zero before the command; read only after its
// final completion. Nonprofiling calls execute no diagnostic atomics.
// A reached candidate contributes <= 3 passes * (1 initial + 9 boundary
// reevaluations + 2 finer resolutions) * 35 angles = 1260 angle evaluations
// at one level. With the host's <= 2,000,000 candidate bound, each counter
// remains <= 2,520,000,000, below UINT32_MAX.
// In effective mode 4, each angle evaluation's 49 windows used either the
// SIMD32 kernel or the mode-2 stripe loop (uncached ROI or uint64 moments);
// both route counters stay zero in every other mode.
struct FGFineCounters {
    FG_FINE_COUNTER reachedCandidates;
    FG_FINE_COUNTER cachedAngles;
    FG_FINE_COUNTER directAngles;
    FG_FINE_COUNTER tiledAngles;
    FG_FINE_COUNTER simd32Angles;
    FG_FINE_COUNTER fallbackAngles;
};

#undef FG_FINE_UINT
#undef FG_FINE_COUNTER
