// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "pose_ncc.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fpm_exp {
// Absence is a supported auto-routing fallback. Compilation, allocation and
// dispatch errors remain ordinary failures rather than silently selecting CPU.
struct MetalUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct PackedPose { float tx, ty, cs, sn; };
// Public pose units. Endpoints are rounded INWARD from double seed-relative
// bounds on the host; device code must not recompute seed +/- float(bound).
struct ResidentSeed { float pose[4], lower[4], upper[4]; };
static_assert(sizeof(ResidentSeed) == 48);
static_assert(sizeof(FpmPoseResidentMetrics) == 72);
// Immutable after preparation. Shared pages can be retained independently by
// Metal's no-copy buffer, so CPU views and device resources share one owner.
struct ImageStorage {
    std::vector<float> values;
    std::shared_ptr<float> pages;
    size_t count = 0, pageBytes = 0;
    void allocate(size_t n, bool shared);
    void assign(const float* first, const float* last, bool shared);
    float* data() { return pages ? pages.get() : values.data(); }
    const float* data() const { return pages ? pages.get() : values.data(); }
    size_t size() const { return pages ? count : values.size(); }
    size_t capacity() const { return pages ? pageBytes/sizeof(float) : values.capacity(); }
    float& operator[](size_t i) { return data()[i]; }
    const float& operator[](size_t i) const { return data()[i]; }
    const float* begin() const { return data(); }
    const float* end() const { return data()+size(); }
};
struct Data {
    int sw, sh, tw, th, channels;
    int storageMode = 0;
    ImageStorage source, templ;
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
    virtual uint64_t imageBufferBytes() const { return 0; }
    // Optional batched derivative path. False means unsupported; callers must
    // explicitly select a CPU fallback rather than mislabeling device work.
    virtual bool linearize(const std::vector<PackedPose>&, const FpmPoseRefineConfig&,
        std::vector<Linearization>&, FpmPoseMetrics&) { return false; }
    virtual FpmPoseResidentKernelInfo residentKernelInfo(int) const { return {}; }
    virtual bool refineResident(const std::vector<ResidentSeed>&, const FpmPoseRefineConfig&,
        float*, float*, FpmPoseRefineStats*, FpmPoseResidentMetrics&, FpmPoseMetrics&, int = 0) { return false; }
};
#ifdef FPM_EXPERIMENT_METAL
std::unique_ptr<Backend> makeMetal(const Data&);
#endif
}
