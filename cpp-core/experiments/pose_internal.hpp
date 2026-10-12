// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "pose_ncc.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace fpm_exp {
struct PackedPose { float tx, ty, cs, sn; };
struct Data {
    int sw, sh, tw, th, channels;
    std::vector<float> source, templ;
    std::array<float, 3> sourceMean{};
    std::array<double, 3> templateSum{};
    double templateEnergy = 0;
};
// Normalized-NCC Gauss-Newton statistics. Translation uses pixels, rotation
// uses radians, and scale uses log(scale), matching the portable CPU solver.
struct Linearization {
    double score = -2;
    double h[4][4]{}, g[4]{};
};
struct Backend {
    virtual ~Backend() = default;
    virtual void score(const std::vector<PackedPose>&, float*, FpmPoseMetrics&) = 0;
    virtual const char* name() const = 0;
    virtual void memory(FpmPoseMetrics&) const {}
    // Optional batched derivative path. False means unsupported; callers must
    // explicitly select a CPU fallback rather than mislabeling device work.
    virtual bool linearize(const std::vector<PackedPose>&, const FpmPoseRefineConfig&,
        std::vector<Linearization>&, FpmPoseMetrics&) { return false; }
};
#ifdef FPM_EXPERIMENT_METAL
std::unique_ptr<Backend> makeMetal(const Data&);
#endif
}
