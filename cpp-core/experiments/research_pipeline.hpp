// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include "coarse_ncc.h"
#include "pose_ncc.h"
#include "sparse_orientation.hpp"
#include <memory>

namespace fpm::research {
// Native orchestration of the previously Python-driven gray8 experiment.
// One caller at a time. Session owns immutable source/template snapshots.
struct Settings {
    int backend = 1; // 0 CPU, 1 Metal (explicit failure if unavailable)
    int minReducedArea = 256;
    std::vector<double> scales{1.}; // proposal bank, not final refinement bounds
    int cpuThreads = 8;
    int residentMode = 0; // 0 original Metal resident; 1 fused256, 2 SIMD128; fused gray templates <=16384 pixels.
    int poseStorageMode = 0; // 0 separate CPU/Metal images; 1 shared pages on unified-memory Metal.
    int poseInputMode = 0; // 0 Session float conversion; 1 convert gray8 directly into owned pose images.
    // Experimental fixed-scale full-image proposals. No reduced image levels
    // or dense NCC maps are constructed; final acceptance still uses full NCC.
    bool sparseOrientation = false;
    SparseOrientationSettings sparse;
};
struct Options {
    MatchConfig match;
    int nccMode = 0;
    bool overlap = false;
    bool refine = true;
    double proposalScore = .5; // capped at final minScore; ignored without refine
};
struct Result {
    std::vector<Match> matches, proposals;
    std::vector<double> scales, proposalScales;
    double prepareMs = 0, proposalMs = 0, refineMs = 0, nmsMs = 0, totalMs = 0;
    FpmFrameMetricsV1 frame{};
    FpmPoseMetrics pose{};
    FpmPoseRescueMetrics rescue{};
    FpmPoseResidentKernelInfo residentKernel{};
    FpmPoseStorageInfo poseStorage{};
    double poseCreateMs = 0; // New pose context initialization only, nested in refineMs.
    uint64_t poseConversionTemporaryBytes = 0; // Initialization-only float conversion capacity, not retained or RSS.
    int poseInputMode = 0; // Actual context input format, or configured selection before creation.
    FpmPoseResidentMetrics resident{}; // Aggregated baseline/retry diagnostics.
    int refinementMode = 0; // Selected mode; zero dispatches means no GPU refinement ran.
    bool sparseEnabled = false;
    SparseOrientationMetrics sparse{};
    std::vector<FpmCoarseMetrics> coarse;
    std::vector<FpmFineMetricsV1> fine;
    std::vector<FpmCoarseOverlapMetricsV1> overlapMetrics;
};
// Public for cross-language parity checks; zero extension, width/2 center,
// ties-to-even dimensions and floor(value+.5) gray8 rounding. No antialiasing.
Image scaledTemplate(ImageView source, double scale);
class Session {
public:
    Session(ImageView source, ImageView templ, const Settings& settings = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Result run(const Options& options = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fpm::research
