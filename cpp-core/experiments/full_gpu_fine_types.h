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
struct FGFineCounters {
    FG_FINE_COUNTER reachedCandidates;
    FG_FINE_COUNTER cachedAngles;
    FG_FINE_COUNTER directAngles;
    FG_FINE_COUNTER tiledAngles;
};

#undef FG_FINE_UINT
#undef FG_FINE_COUNTER
