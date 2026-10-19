// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#ifdef __METAL_VERSION__
#define FG_UINT uint
#define FG_ULONG ulong
#else
#include <cstdint>
#define FG_UINT std::uint32_t
#define FG_ULONG std::uint64_t
#endif
// Scalar-only ABI; byte offsets refer to packed gray8 pyramid arenas.
struct FGLevel {
    FG_UINT sw, sh, tw, th;
    FG_UINT sourceOffset, templateOffset, statisticsIndex, reserved;
};
struct FGStats { FG_ULONG sum, square, variance, area; };
// Coarse: x/y top-left and angle is source rotation. Fine/finish: x/y center
// and angle is public counter-clockwise object angle. Invalid slots have valid=0.
struct FGCandidate {
    float x,y,angle,score,scale,coarseScore;
    FG_UINT valid,order,model,reserved0,reserved1,reserved2;
};
struct FGFineConfig {
    FG_UINT first,count,levelBase,top;
    FG_UINT stop,subpixel,rangeCount,model;
    float minScore,maxOverlap,scale,reserved;
    FG_UINT templateWidth,templateHeight,maxTargets,reserved2;
};
struct FGRange { float minimum,maximum; };
struct FGNmsConfig {
    FG_UINT first,count,outputFirst,limit;
    FG_UINT templateWidth,templateHeight,globalStage,rangeCount;
    float minScore,maxOverlap; FG_UINT reserved0,reserved1;
};
struct FGFinishConfig {
    FG_UINT sw,sh,tw,th;
    FG_UINT modelCount,proposalStride,capacity,optimizeMask;
    float minScore,maxOverlap; FG_UINT rangeCount,refine;
};
// Status counters are GPU-written. No host read is allowed before final wait.
struct FGStatus {
    FG_UINT proposals,matches,rescueParents,rescueStarts;
    FG_UINT rescueContinuations,overflow,reserved0,reserved1;
};
// GPU-formatted final/audit output. Geometry is never recomputed on the CPU.
struct FGResult {
    float x,y,angle,score,scale;
    float corners[8];
    FG_UINT reserved[3];
};
#undef FG_UINT
#undef FG_ULONG
