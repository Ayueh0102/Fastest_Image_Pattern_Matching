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
// Peak mode 2: one threadgroup per map, at most this many lanes and maps per
// dispatch. Each map owns scoreStride floats of scores, 2*scoreStride state
// bytes, scoreStride/32 removal words, keyStride ordered output keys and three
// listCapacity point lists in its ring slot; scoreStride is a multiple of 64.
#define FG_PEAK_PARALLEL_LANES 1024
#define FG_PEAK_PARALLEL_MAPS 16
#define FG_PEAK_PARALLEL_DEFAULT_MAPS 8
#define FG_PEAK_LIST_CAPACITY 65536
struct FGPeakBatch {
#ifdef __METAL_VERSION__
    uint scoreStride,keyStride,listCapacity,reserved;
#else
    std::uint32_t scoreStride,keyStride,listCapacity,reserved;
#endif
};
