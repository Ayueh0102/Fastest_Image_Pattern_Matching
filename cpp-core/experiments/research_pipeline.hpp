// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include "coarse_ncc.h"
#include "pose_ncc.h"
#include "sparse_orientation.hpp"
#include <memory>
#include <string>

namespace fpm::research {
// Native orchestration of the previously Python-driven gray8 experiment.
// One caller at a time. Session owns image snapshots; updateSource replaces only the source.
struct Settings {
    bool fullGpu = false; // Experimental complete gray8 Metal data path; no intermediate host decisions.
    int backend = 1; // Proposal backend: 0 CPU, 1 Metal (explicit failure if unavailable)
    int refinementBackend = -1; // -1 inherits backend; 0 CPU, 1 Metal. Defaults unchanged.
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
    int fullGpuPeakMode = 0; // Full resident path: 0 original stripes; 1 exact sorted-tile merge; 2 exact parallel rounds over batched maps.
    int fullGpuPeakBatch = 0; // Peak mode 2 only: maps selected per dispatch, 1..16 (1 keeps one score map); 0 uses 8.
    int fullGpuFineMode = 0; // 0 original; 1 row cache; 2 bounded uint32; 3 reused moments; 4 SIMD32 windows.
    bool fullGpuProfiling = false; // Encoder GPU timestamps; diagnostic overhead, no intermediate waits.
    int nccMode = 0;
    int coarsePeakMode = 1; // 1 exact CPU tree; 2 experimental resident Metal FP32 peaks.
    int fineNccMode = 0; // Metal: 0 original; 1 shared rows; 2/3 strip with chunks 32/16.
    bool fineProfiling = false; // Bounded exact request observation; adds host overhead.
    bool fineAngleReuse = false; // Previous angle planes within one state/layer/pass only.
    bool fineFused = false; // One warp+NCC submission with exact CPU statistics repair.
    bool overlap = false;
    bool refine = true;
    bool scaleSupportRefine = false; // Experimental small-template interior-support scale correction.
    double proposalScore = .5; // capped at final minScore; ignored without refine
};
struct SourceUpdate {
    double updateMs = 0; // Validation, source snapshot copy and pose teardown; excludes next run.
    uint64_t copiedBytes = 0, generation = 0;
    size_t retainedModels = 0; // Learned coarse models, or one sparse template model.
    bool dimensionsChanged = false, poseInvalidated = false;
};
struct FullGpuStageTiming {
    std::string name;
    double gpuMs = 0;
    uint64_t encoders = 0;
};
struct FullGpuFineWork {
    uint32_t model = 0, layer = 0, templateWidth = 0, templateHeight = 0, roiCacheBytes = 0;
    uint64_t reachedCandidates = 0, angles = 0, cachedAngles = 0, directAngles = 0, tiledAngles = 0;
    uint64_t sourceSamples = 0, tileLoads = 0; // Exact sampler-call counts derived from measured angle counts, not memory traffic.
    // Effective mode 4 only: windows (49 per angle) scored by the SIMD32 kernel or by its mode-2 fallback.
    uint64_t simd32Windows = 0, fallbackWindows = 0;
};
struct FullGpuMetrics {
    bool enabled = false;
    int peakMode = 0;
    int fineMode = 0;
    int fineEffectiveMode = 0; // Kernel option after the SIMD-width check; per-window routes are below.
    uint32_t fineSimdWidth = 0;
    bool fineWorkAvailable = false;
    uint64_t fineDiagnosticBytes = 0, fineDiagnosticReadbackBytes = 0, fineThreadgroupBytes = 0;
    std::vector<FullGpuFineWork> fineWork;
    uint64_t fineSimd32Windows = 0, fineFallbackWindows = 0; // Sums of fineWork; profiling runs only.
    uint64_t peakScratchBytes = 0;
    // Maps selected per peak dispatch (1 unless peak mode 2) and the retained score storage they use.
    uint64_t peakBatchMaps = 0, coarseScoreBytes = 0;
    // Counted where run() submits, waits and reads GPU output: submitted command
    // buffers, host completion waits, GPU-output reads followed by another
    // submission, and those later submissions (host decisions between GPU
    // stages). Host work after the final wait is outside the last two counters.
    uint64_t commandBuffers = 0, cpuWaits = 0, intermediateReadbacks = 0, cpuAlgorithmSteps = 0;
    uint64_t gpuBytes = 0, uploadBytes = 0, readbackBytes = 0, dispatches = 0, fineRoiCacheBytes = 0;
    double encodeMs = 0, waitMs = 0, deviceMs = 0, uploadMs = 0, readbackMs = 0;
    bool profileRequested = false, profileAvailable = false;
    std::string profileStatus = "disabled";
    uint64_t profileSamples = 0, profileCounterBytes = 0, profileResolveBytes = 0;
    double profileHostMs = 0, profileGpuMs = 0;
    std::vector<FullGpuStageTiming> stages;
};
struct Result {
    FullGpuMetrics fullGpu;
    std::vector<Match> matches, proposals;
    std::vector<double> scales, proposalScales;
    double prepareMs = 0, proposalMs = 0, refineMs = 0, nmsMs = 0, totalMs = 0;
    FpmFrameMetricsV1 frame{};
    FpmPoseMetrics pose{};
    FpmPoseRescueMetrics rescue{};
    FpmPoseResidentKernelInfo residentKernel{};
    FpmPoseStorageInfo poseStorage{};
    bool scaleSupportEnabled = false;
    double scaleSupportMs = 0; // CPU correction, nested in refineMs; default disabled.
    uint64_t scaleSupportEligible = 0, scaleSupportChanged = 0, scaleSupportBoundary = 0;
    uint64_t scaleSupportEvaluations = 0, scaleSupportSamples = 0;
    double poseCreateMs = 0; // New pose context initialization only, nested in refineMs.
    uint64_t poseConversionTemporaryBytes = 0; // Initialization-only float conversion capacity, not retained or RSS.
    int poseInputMode = 0; // Actual context input format, or configured selection before creation.
    FpmPoseResidentMetrics resident{}; // Aggregated baseline/retry diagnostics.
    int refinementMode = 0; // Selected mode; zero dispatches means no GPU refinement ran.
    int configuredRefinementBackend = 0; // Resolved requested backend, even with no candidates.
    uint64_t sourceGeneration = 0; // Incremented only by successful updateSource calls.
    uint64_t refinementCandidates = 0; // Zero means no refinement route ran.
    bool sparseEnabled = false;
    SparseOrientationMetrics sparse{};
    std::vector<FpmCoarseMetrics> coarse;
    std::vector<FpmFineMetricsV1> fine;
    std::vector<FpmFineReuseMetricsV1> fineReuse;
    std::vector<FpmFineSearchMetricsV1> fineSearch;
    std::vector<FpmFineFusionMetricsV1> fineFusion;
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
    // Strong validation/copy guarantee: invalid input leaves the previous source usable.
    // Template models and frame allocation capacity are retained; source-bound pose
    // images are released and lazily recreated on the next refinement.
    SourceUpdate updateSource(ImageView source);
    Result run(const Options& options = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fpm::research
