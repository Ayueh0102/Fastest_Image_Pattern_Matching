// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static double sampleReference(const std::vector<float>& source, int w, int h, int channels,
    double x, double y, int c)
{
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double dx = x-ix, dy = y-iy;
    double result = 0;
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
        if (ix+i >= 0 && ix+i < w && iy+j >= 0 && iy+j < h)
            result += source[(size_t(iy+j)*w+ix+i)*channels+c]
                * (i ? dx : 1-dx)*(j ? dy : 1-dy);
    return result;
}

static void testRefinement(int backend, int channels, int threads)
{
    const int sw = 96, sh = 88, tw = 41, th = 37;
    const double pi = std::acos(-1.0), angle = 31.25, scale = 1.08, tx = 4.3, ty = -2.6;
    std::vector<float> source(size_t(sw)*sh*channels), templ(size_t(tw)*th*channels);
    // Non-periodic combinations and independently varying RGB channels exercise
    // the geometric Jacobian and per-channel centering, not merely self-matching.
    for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x)
        for (int c = 0; c < channels; ++c)
            source[(size_t(y)*sw+x)*channels+c] = float(.4+.08*c
                +.10*std::sin(.13*x+.07*y+.7*c)+.12*std::cos(.04*x-.19*y+.5*c)
                +.08*std::sin(.0009*x*x+.0018*x*y-.003*y*y+.2*c));
    const double cs = scale*std::cos(angle*pi/180), sn = scale*std::sin(angle*pi/180);
    for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x)
        for (int c = 0; c < channels; ++c) {
            const double u = x-tw*.5, v = th*.5-y;
            templ[(size_t(y)*tw+x)*channels+c] = float(sampleReference(source, sw, sh, channels,
                sw*.5+tx+cs*u-sn*v, sh*.5-ty-sn*u-cs*v, c));
        }
    char error[1024]{};
    void* handle = fpm_pose_create(backend, threads, source.data(), sw, sh,
        templ.data(), tw, th, channels, error, sizeof(error));
    require(handle, error);
    struct Cleanup { void* h; ~Cleanup() { fpm_pose_destroy(h); } } cleanup{handle};
    std::vector<float> seeds = {float(tx-2.3), float(ty+1.7), float(angle-7), float(std::log(scale)-.09),
        float(tx+1.8), float(ty-2.1), float(angle+5), float(std::log(scale)+.07)};
    std::vector<float> output(8), scores(2), initial(2), rescored(2);
    FpmPoseMetrics metrics{};
    FpmPoseRefineStats stats[2]{};
    FpmPoseRefineConfig config;
    fpm_pose_refine_default_config(&config);
    require(fpm_pose_score(handle, seeds.data(), 2, initial.data(), &metrics, error, sizeof(error)) == 0, error);
    require(fpm_pose_refine(handle, seeds.data(), 2, &config, output.data(), scores.data(),
        stats, &metrics, error, sizeof(error)) == 0, error);
    require(metrics.device_ms == -1, "CPU refinement mislabeled as GPU work");
    require(fpm_pose_score(handle, output.data(), 2, rescored.data(), &metrics, error, sizeof(error)) == 0, error);
    for (int i = 0; i < 2; ++i) {
        require(std::abs(output[4*i]-tx) < .02 && std::abs(output[4*i+1]-ty) < .02, "refined subpixel translation");
        require(std::abs(output[4*i+2]-angle) < .04, "refined rotation beyond two degrees");
        require(std::abs(std::exp(output[4*i+3])/scale-1) < .001, "refined continuous scale");
        require(scores[i] > .99999 && scores[i] >= initial[i]-1e-6, "refinement score regression");
        require(std::abs(scores[i]-rescored[i]) < 1e-6, "refinement changed NCC semantics");
        require(stats[i].evaluations >= stats[i].iterations && stats[i].iterations > 0, "refinement stats");
    }
    const auto referencePoses = output, referenceScores = scores;
    const FpmPoseRefineStats referenceStats[2] = {stats[0], stats[1]};
    void* verifier = fpm_pose_create(0, 1, source.data(), sw, sh, templ.data(), tw, th,
        channels, error, sizeof(error));
    require(verifier, error);
    Cleanup verificationCleanup{verifier};
    std::vector<float> exactInitial(2), exactFinal(2);
    require(fpm_pose_score(verifier, seeds.data(), 2, exactInitial.data(), &metrics, error, sizeof(error)) == 0, error);
    for (int polish : {0, 2}) {
        require(fpm_pose_refine_accelerated(handle, seeds.data(), 2, &config, polish,
            output.data(), scores.data(), stats, &metrics, error, sizeof(error)) == 0, error);
        require(metrics.wall_ms >= 0, "accelerated wall time");
        if (backend == 0) {
            require(metrics.device_ms == -1, "CPU accelerated fallback mislabeled as GPU");
            require(output == referencePoses && scores == referenceScores, "CPU accelerated fallback changed results");
            for (int i = 0; i < 2; ++i)
                require(stats[i].iterations == referenceStats[i].iterations
                    && stats[i].evaluations == referenceStats[i].evaluations
                    && stats[i].status == referenceStats[i].status, "CPU fallback applied extra polish");
        }
        require(fpm_pose_score(verifier, output.data(), 2, exactFinal.data(), &metrics, error, sizeof(error)) == 0, error);
        for (int i = 0; i < 2; ++i) {
            require(scores[i] == exactFinal[i] && scores[i] >= exactInitial[i], "accelerated CPU-verified score regression");
            require(std::abs(output[4*i]-tx) < .03 && std::abs(output[4*i+1]-ty) < .03, "accelerated subpixel position");
            require(std::abs(output[4*i+2]-angle) < .05
                && std::abs(std::exp(output[4*i+3])/scale-1) < .001, "accelerated rotation/scale");
            require(stats[i].iterations <= config.max_iterations+polish, "accelerated iteration limit");
        }
    }
    config.optimize_scale = config.optimize_angle = 0;
    config.max_translation = .1;
    require(fpm_pose_refine(handle, seeds.data(), 2, &config, output.data(), scores.data(),
        nullptr, &metrics, error, sizeof(error)) == 0, error);
    for (int i = 0; i < 2; ++i) {
        require(output[4*i+2] == seeds[4*i+2] && output[4*i+3] == seeds[4*i+3], "disabled degrees of freedom changed");
        require(std::abs(output[4*i]-seeds[4*i]) <= .100001f
            && std::abs(output[4*i+1]-seeds[4*i+1]) <= .100001f, "seed-relative bound exceeded");
    }
    config.max_iterations = 1;
    require(fpm_pose_refine_accelerated(handle, seeds.data(), 2, &config, 2,
        output.data(), scores.data(), stats, &metrics, error, sizeof(error)) == 0, error);
    for (int i = 0; i < 2; ++i) {
        require(output[4*i+2] == seeds[4*i+2] && output[4*i+3] == seeds[4*i+3], "accelerated frozen parameters changed");
        require(std::abs(output[4*i]-seeds[4*i]) <= .100001f
            && std::abs(output[4*i+1]-seeds[4*i+1]) <= .100001f, "accelerated polish escaped original bounds");
        require(stats[i].iterations <= (backend == 0 ? 1 : 3), "accelerated combined iteration budget");
    }
    for (int polish : {-1, 31})
        require(fpm_pose_refine_accelerated(handle, seeds.data(), 2, &config, polish,
            output.data(), scores.data(), stats, &metrics, error, sizeof(error)) != 0, "invalid polish iterations accepted");
    config.max_translation = -1;
    require(fpm_pose_refine(handle, seeds.data(), 2, &config, output.data(), scores.data(),
        nullptr, &metrics, error, sizeof(error)) != 0, "invalid refine config accepted");
    require(fpm_pose_refine_accelerated(handle, seeds.data(), 2, &config, 2, output.data(), scores.data(),
        nullptr, &metrics, error, sizeof(error)) != 0, "invalid accelerated config accepted");
    fpm_pose_refine_default_config(&config);
    for (int iterations : {0, 201}) {
        config.max_iterations = iterations;
        require(fpm_pose_refine_accelerated(handle, seeds.data(), 2, &config, 2, output.data(), scores.data(),
            nullptr, &metrics, error, sizeof(error)) != 0, "invalid accelerated iteration bound accepted");
    }
    const float finiteSeed = seeds[0];
    seeds[0] = std::numeric_limits<float>::quiet_NaN();
    require(fpm_pose_refine_accelerated(handle, seeds.data(), 2, nullptr, 2, output.data(), scores.data(),
        nullptr, &metrics, error, sizeof(error)) != 0, "non-finite accelerated seed accepted");
    seeds[0] = finiteSeed;
    seeds[0] = 5000; seeds[1] = 5000;
    require(fpm_pose_refine(handle, seeds.data(), 1, nullptr, output.data(), scores.data(),
        stats, &metrics, error, sizeof(error)) == 0, error);
    require(scores[0] == -2 && stats[0].status == 2, "outside zero-support patch accepted");
    require(fpm_pose_refine_accelerated(handle, seeds.data(), 1, nullptr, 0, output.data(), scores.data(),
        stats, &metrics, error, sizeof(error)) == 0, error);
    require(scores[0] == -2 && stats[0].status == 2, "accelerated outside-support patch accepted");
}

static void testNarrowRepresentableBounds(int backend)
{
    // At tx=10000 the adjacent floats are 0.0009765625 apart. A +/-0.0006
    // interval therefore permits ONLY the original tx. Double-clamping then
    // casting naively used to round outward and accept an illegal better pose.
    const int sw = 32768, sh = 24, tw = 17, th = 13;
    const float seed[4] = {10000, 0, 0, 0};
    std::vector<float> source(size_t(sw)*sh, .2f), templ(size_t(tw)*th);
    const int centerX = sw/2+10000;
    for (int y = 0; y < sh; ++y) for (int x = centerX-20; x <= centerX+20; ++x)
        source[size_t(y)*sw+x] = float(.4+.2*std::sin(.43*(x-centerX)+.16*y)
            +.1*std::cos(.33*(x-centerX)-.29*y));
    FpmPoseRefineConfig config;
    fpm_pose_refine_default_config(&config);
    config.max_iterations = 3;
    config.optimize_angle = config.optimize_scale = 0;
    config.max_translation = .0006;
    for (double direction : {-1., 1.}) {
        for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x)
            templ[size_t(y)*tw+x] = float(sampleReference(source, sw, sh, 1,
                centerX+(x-tw*.5)+direction*.05, sh*.5+(y-th*.5)+direction*.05, 0));
        char error[1024]{};
        void* handle = fpm_pose_create(backend, 1, source.data(), sw, sh,
            templ.data(), tw, th, 1, error, sizeof(error));
        require(handle, error);
        struct Cleanup { void* h; ~Cleanup() { fpm_pose_destroy(h); } } cleanup{handle};
        FpmPoseMetrics metrics{};
        FpmPoseRefineStats stats{};
        float output[4]{}, score = 0;
        for (int accelerated : {0, 1}) {
            const int result = accelerated
                ? fpm_pose_refine_accelerated(handle, seed, 1, &config, 2, output, &score, &stats, &metrics, error, sizeof(error))
                : fpm_pose_refine(handle, seed, 1, &config, output, &score, &stats, &metrics, error, sizeof(error));
            require(result == 0, error);
            require(output[0] == seed[0], "float rounding escaped sub-ULP translation bound");
            require(std::abs(double(output[1])-seed[1]) <= config.max_translation, "float rounding escaped y bound");
            require(output[2] == seed[2] && output[3] == seed[3], "narrow-bounds frozen parameters changed");
        }
    }
}

static void testAutoRouting(int backend)
{
    const int w = 41, h = 39, channels = 3, count = 2;
    std::vector<float> image(size_t(w)*h*channels);
    for (size_t i = 0; i < image.size(); ++i)
        image[i] = float(.4+.2*std::sin(.13*i)+.1*std::cos(.027*i));
    char error[1024]{};
    void* cpu = fpm_pose_create(0, 2, image.data(), w, h, image.data(), w, h,
        channels, error, sizeof(error));
    require(cpu, error);
    struct Cleanup { void* h; ~Cleanup() { fpm_pose_destroy(h); } } cleanup{cpu};
    void* automatic = fpm_pose_create(2, 2, image.data(), w, h, image.data(), w, h,
        channels, error, sizeof(error));
    require(automatic, error);
    Cleanup automaticCleanup{automatic};
    require(fpm_pose_backend(cpu) == 0 && fpm_pose_backend(automatic) == 2
        && fpm_pose_backend(nullptr) == -1, "creation backend query");
    FpmPoseMetrics metrics{};
    require(fpm_pose_memory(automatic, &metrics) == 0 && metrics.gpu_buffer_bytes == 0
        && metrics.gpu_allocated_bytes == 0, "auto handle initialized Metal eagerly");
    const float seeds[count*4] = {.25f, -.5f, 2, .04f, -.4f, .6f, -1, .03f};
    float reference[count*4]{}, output[count*4]{}, referenceScores[count]{}, scores[count]{};
    FpmPoseRefineStats stats[count]{}, referenceStats[count]{};
    FpmPoseRefineConfig config;
    fpm_pose_refine_default_config(&config);
    config.max_iterations = 4;
    config.max_translation = 1;
    config.max_angle_degrees = 3;
    config.max_log_scale = .1;
    FpmPoseAutoConfig routing;
    fpm_pose_auto_default_config(&routing);
    require(routing.min_pose_samples == 131072 && routing.min_batch_samples == 0
        && routing.min_iterations == 4 && routing.cpu_polish_iterations == 0, "auto defaults");
    FpmPoseRoute route{};
    require(fpm_pose_refine(cpu, seeds, count, &config, reference, referenceScores,
        referenceStats, &metrics, error, sizeof(error)) == 0, error);
    const auto cpuEquivalent = [&] {
        require(std::equal(output, output+count*4, reference)
            && std::equal(scores, scores+count, referenceScores), "CPU auto route changed results");
        for (int i = 0; i < count; ++i)
            require(stats[i].iterations == referenceStats[i].iterations
                && stats[i].evaluations == referenceStats[i].evaluations
                && stats[i].status == referenceStats[i].status, "CPU auto route changed stats");
    };
    const auto runAuto = [&](void* handle, int expectedReason) {
        require(fpm_pose_refine_auto(handle, seeds, count, &config, &routing, output, scores,
            stats, &route, &metrics, error, sizeof(error)) == 0, error);
        require(route.reason == expectedReason, "unexpected auto route reason");
        require(route.pose_samples == uint64_t(w)*h*channels
            && route.batch_samples == uint64_t(w)*h*channels*count, "auto sample accounting");
        require(route.metal_setup_ms >= 0 && metrics.wall_ms >= route.metal_setup_ms, "auto setup/wall accounting");
    };
    runAuto(automatic, FPM_POSE_ROUTE_BELOW_POSE_WORK); cpuEquivalent();
    require(route.selected_backend == 0 && route.metal_setup_ms == 0
        && metrics.device_ms == -1 && metrics.gpu_buffer_bytes == 0, "small auto route allocated Metal");
    routing.min_pose_samples = 0;
    routing.min_batch_samples = uint64_t(w)*h*channels*count+1;
    runAuto(automatic, FPM_POSE_ROUTE_BELOW_BATCH_WORK); cpuEquivalent();
    routing.min_batch_samples = 0;
    routing.min_iterations = config.max_iterations+1;
    runAuto(automatic, FPM_POSE_ROUTE_BELOW_ITERATIONS); cpuEquivalent();
    routing.min_iterations = 1;
    runAuto(cpu, FPM_POSE_ROUTE_CPU_HANDLE); cpuEquivalent();
    require(route.selected_backend == 0 && metrics.gpu_buffer_bytes == 0, "CPU handle auto initialized GPU");

    const auto activeConfig = config;
    config.max_translation = 0; config.optimize_angle = config.optimize_scale = 0;
    runAuto(automatic, FPM_POSE_ROUTE_CPU_NO_ACTIVE_PARAMETERS);
    require(std::equal(output, output+count*4, seeds) && route.selected_backend == 0
        && metrics.gpu_buffer_bytes == 0, "frozen auto pose initialized Metal or moved");
    config = activeConfig;
    // Invalid inputs must be rejected before any eligible lazy Metal setup.
    for (int minimum : {0, 201}) {
        routing.min_iterations = minimum;
        require(fpm_pose_refine_auto(automatic, seeds, count, &config, &routing, output, scores,
            stats, &route, &metrics, error, sizeof(error)) != 0, "invalid auto minimum iterations accepted");
    }
    routing.min_iterations = 1;
    for (int polish : {-1, 31}) {
        routing.cpu_polish_iterations = polish;
        require(fpm_pose_refine_auto(automatic, seeds, count, &config, &routing, output, scores,
            stats, &route, &metrics, error, sizeof(error)) != 0, "invalid auto polish accepted");
    }
    routing.cpu_polish_iterations = 0;
    config.max_iterations = 0;
    require(fpm_pose_refine_auto(automatic, seeds, count, &config, &routing, output, scores,
        stats, &route, &metrics, error, sizeof(error)) != 0, "invalid auto refinement config accepted");
    config = activeConfig;
    float invalidSeeds[count*4]; std::copy(seeds, seeds+count*4, invalidSeeds);
    invalidSeeds[0] = std::numeric_limits<float>::quiet_NaN();
    require(fpm_pose_refine_auto(automatic, invalidSeeds, count, &config, &routing, output, scores,
        stats, &route, &metrics, error, sizeof(error)) != 0, "invalid auto seed accepted");
    require(fpm_pose_refine_auto(automatic, seeds, 0, &config, &routing, output, scores,
        stats, &route, &metrics, error, sizeof(error)) != 0, "empty auto batch accepted");
    require(fpm_pose_refine_auto(automatic, seeds, count, &config, &routing, output, scores,
        stats, nullptr, &metrics, error, sizeof(error)) != 0, "missing auto route output accepted");
    require(fpm_pose_memory(automatic, &metrics) == 0 && metrics.gpu_buffer_bytes == 0,
        "invalid auto input caused lazy allocation");
    require(fpm_pose_refine_accelerated(automatic, seeds, count, &config, 2, output, scores,
        stats, &metrics, error, sizeof(error)) == 0, error);
    cpuEquivalent();
    require(metrics.device_ms == -1 && metrics.gpu_buffer_bytes == 0, "accelerated auto handle initialized Metal");

    if (backend == 1) {
        void* metal = fpm_pose_create(1, 2, image.data(), w, h, image.data(), w, h,
            channels, error, sizeof(error));
        require(metal, error);
        Cleanup metalCleanup{metal};
        require(fpm_pose_backend(metal) == 1, "Metal creation backend query");
        float metalPoses[count*4]{}, metalScores[count]{};
        require(fpm_pose_refine_accelerated(metal, seeds, count, &config, 0, metalPoses, metalScores,
            nullptr, &metrics, error, sizeof(error)) == 0, error);
        runAuto(automatic, FPM_POSE_ROUTE_METAL_SELECTED);
        require(route.selected_backend == 1 && route.metal_setup_ms > 0
            && metrics.gpu_buffer_bytes > 0, "eligible auto route did not initialize Metal");
        require(std::equal(output, output+count*4, metalPoses)
            && std::equal(scores, scores+count, metalScores), "forced auto Metal differs from accelerated Metal");
        for (int i = 0; i < count; ++i) {
            require(std::abs(double(output[4*i])-seeds[4*i]) <= config.max_translation
                && std::abs(double(output[4*i+1])-seeds[4*i+1]) <= config.max_translation
                && std::abs(double(output[4*i+2])-seeds[4*i+2]) <= config.max_angle_degrees
                && std::abs(double(output[4*i+3])-seeds[4*i+3]) <= config.max_log_scale, "auto Metal route escaped bounds");
        }
        runAuto(automatic, FPM_POSE_ROUTE_METAL_SELECTED);
        require(route.metal_setup_ms == 0, "auto route failed to reuse Metal");
        const uint64_t retainedBytes = metrics.gpu_buffer_bytes;
        routing.min_pose_samples = std::numeric_limits<uint64_t>::max();
        runAuto(automatic, FPM_POSE_ROUTE_BELOW_POSE_WORK); cpuEquivalent();
        require(route.selected_backend == 0 && metrics.device_ms == -1
            && metrics.gpu_buffer_bytes == retainedBytes, "CPU route dropped retained Metal accounting");
        runAuto(metal, FPM_POSE_ROUTE_BELOW_POSE_WORK); cpuEquivalent();
        require(route.selected_backend == 0 && metrics.device_ms == -1
            && metrics.gpu_buffer_bytes > 0, "Metal handle CPU route changed GPU ownership");
        require(fpm_pose_refine_accelerated(automatic, seeds, count, &config, 2, output, scores,
            stats, &metrics, error, sizeof(error)) == 0, error);
        cpuEquivalent();
        require(metrics.device_ms == -1 && metrics.gpu_buffer_bytes == retainedBytes,
            "accelerated() auto handle silently reused Metal");
        require(fpm_pose_refine(automatic, seeds, count, &config, output, scores,
            stats, &metrics, error, sizeof(error)) == 0, error);
        cpuEquivalent();
        require(metrics.device_ms == -1 && metrics.gpu_buffer_bytes == retainedBytes,
            "refine() auto handle silently reused Metal");
        require(std::strcmp(fpm_pose_device(automatic), fpm_pose_device(cpu)) == 0,
            "auto device name stopped describing default score backend");
        float cpuSeedScores[count]{};
        require(fpm_pose_score(cpu, seeds, count, cpuSeedScores, &metrics, error, sizeof(error)) == 0, error);
        require(fpm_pose_score(automatic, seeds, count, scores, &metrics, error, sizeof(error)) == 0, error);
        require(std::equal(scores, scores+count, cpuSeedScores) && metrics.device_ms == -1
            && metrics.gpu_buffer_bytes == retainedBytes, "auto score backend or memory accounting changed");
    }
#if !defined(__APPLE__) || defined(FPM_TEST_NO_METAL)
    else {
        runAuto(automatic, FPM_POSE_ROUTE_METAL_UNAVAILABLE); cpuEquivalent();
        require(route.selected_backend == 0 && route.metal_setup_ms == 0
            && metrics.device_ms == -1 && metrics.gpu_buffer_bytes == 0, "uncompiled Metal fallback");
    }
#endif
    for (int reason = 0; reason <= 6; ++reason)
        require(std::strlen(fpm_pose_route_reason(reason)) > 0, "missing route reason text");
    require(std::strcmp(fpm_pose_route_reason(-1), "unknown routing reason") == 0, "unknown route reason");
}
int main(int argc, char**)
{
    try {
        const int backend = argc > 1 ? 1 : 0; // optional manual GPU run
        char error[1024]{};
        const int w = 35, h = 33;
        for (int channels : {1, 3}) for (int threads : {1, 8}) {
            testRefinement(backend, channels, threads);
            std::vector<float> image(size_t(w)*h*channels);
            for (size_t i = 0; i < image.size(); ++i) image[i] = float(.3+.2*std::sin(double(i)*.33));
            void* handle = fpm_pose_create(backend, threads, image.data(), w, h, image.data(), w, h, channels, error, sizeof(error));
            require(handle, error);
            struct Cleanup { void* h; ~Cleanup() { fpm_pose_destroy(h); } } cleanup{handle};
            FpmPoseMetrics metrics{};
            require(fpm_pose_memory(handle, &metrics) == 0, "memory query");
            require(metrics.host_buffer_bytes >= 2*image.size()*sizeof(float), "owned bytes before first score");
            std::vector<float> poses(37*4, 0), scores(37);
            for (int count : {1, 37, 3, 37}) {
                require(fpm_pose_score(handle, poses.data(), count, scores.data(), &metrics, error, sizeof(error)) == 0, error);
                for (int i = 0; i < count; ++i) require(std::abs(scores[i]-1) < 1e-5f, "identity NCC or stale result");
                require(metrics.wall_ms >= 0, "wall time");
            }
            poses[0] = std::numeric_limits<float>::quiet_NaN();
            require(fpm_pose_score(handle, poses.data(), 1, scores.data(), &metrics, error, sizeof(error)) != 0, "NaN pose accepted");
            poses[0] = 0;
            require(fpm_pose_score(handle, poses.data(), 1, scores.data(), &metrics, error, sizeof(error)) == 0, "handle unusable after invalid pose");
        }
        testNarrowRepresentableBounds(backend);
        testAutoRouting(backend);
        float x = 0;
        require(!fpm_pose_create(0, 0, &x, 1, 1, &x, 1, 1, 1, error, sizeof(error)), "zero threads accepted");
        require(!fpm_pose_create(0, 1, &x, 1, 1, &x, 1, 1, 2, error, sizeof(error)), "two channels accepted");
        require(!fpm_pose_create(0, 1, nullptr, 1, 1, &x, 1, 1, 1, error, sizeof(error)), "null source accepted");
        require(fpm_pose_memory(nullptr, nullptr) == -1, "null memory query accepted");
        FpmPoseMetrics metrics{};
        FpmPoseRefineStats stats{};
        float seed[4]{}, refined[4]{}, score = 0;
        void* constant = fpm_pose_create(0, 1, &x, 1, 1, &x, 1, 1, 1, error, sizeof(error));
        require(constant, error);
        require(fpm_pose_refine(constant, seed, 1, nullptr, refined, &score, &stats, &metrics,
            error, sizeof(error)) == 0 && score == -2 && stats.status == 2, "constant template refinement");
        fpm_pose_destroy(constant);
        fpm_pose_destroy(nullptr);
        std::cout << "Experimental pose ABI, lifetime, NCC identity and continuous RGB/gray refinement passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
