// SPDX-License-Identifier: BSD-2-Clause
#pragma once
// Every score-map entry is retained. Padding is bounded to 255 records.
#define FG_PEAK_TILE_SIZE 256
#ifdef __METAL_VERSION__
struct FGPeakEntry { float score; uint index; };
#else
#include <cstdint>
struct FGPeakEntry { float score; std::uint32_t index; };
static_assert(sizeof(FGPeakEntry)==8,"Full GPU tiled peak ABI");
#endif
