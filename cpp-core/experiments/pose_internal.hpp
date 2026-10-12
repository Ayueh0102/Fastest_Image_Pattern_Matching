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
struct Backend {
    virtual ~Backend() = default;
    virtual void score(const std::vector<PackedPose>&, float*, FpmPoseMetrics&) = 0;
    virtual const char* name() const = 0;
    virtual void memory(FpmPoseMetrics&) const {}
};
#ifdef FPM_EXPERIMENT_METAL
std::unique_ptr<Backend> makeMetal(const Data&);
#endif
}
