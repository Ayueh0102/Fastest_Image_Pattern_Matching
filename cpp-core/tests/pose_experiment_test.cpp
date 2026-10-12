// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <algorithm>
#include <cmath>
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
    config.optimize_scale = config.optimize_angle = 0;
    config.max_translation = .1;
    require(fpm_pose_refine(handle, seeds.data(), 2, &config, output.data(), scores.data(),
        nullptr, &metrics, error, sizeof(error)) == 0, error);
    for (int i = 0; i < 2; ++i) {
        require(output[4*i+2] == seeds[4*i+2] && output[4*i+3] == seeds[4*i+3], "disabled degrees of freedom changed");
        require(std::abs(output[4*i]-seeds[4*i]) <= .100001f
            && std::abs(output[4*i+1]-seeds[4*i+1]) <= .100001f, "seed-relative bound exceeded");
    }
    config.max_translation = -1;
    require(fpm_pose_refine(handle, seeds.data(), 2, &config, output.data(), scores.data(),
        nullptr, &metrics, error, sizeof(error)) != 0, "invalid refine config accepted");
    seeds[0] = 5000; seeds[1] = 5000;
    require(fpm_pose_refine(handle, seeds.data(), 1, nullptr, output.data(), scores.data(),
        stats, &metrics, error, sizeof(error)) == 0, error);
    require(scores[0] == -2 && stats[0].status == 2, "outside zero-support patch accepted");
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
