// SPDX-License-Identifier: BSD-2-Clause
// Link the portable solver with a controlled backend factory, without Metal.
// Hardware absence is a valid fallback; real initialization errors must surface.
#include "pose_internal.hpp"
#include <cstring>
#include <iostream>

namespace {
bool unavailable = true;
int setupAttempts = 0;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}
namespace fpm_exp {
std::unique_ptr<Backend> makeMetal(const Data&) {
    ++setupAttempts;
    if (unavailable) throw MetalUnavailable("forced no device");
    throw std::runtime_error("forced shader compilation failure");
}
}
int main() {
    try {
        float image[16];
        for (int i = 0; i < 16; ++i) image[i] = float(i)/16;
        char error[256]{};
        void* handle = fpm_pose_create(2, 1, image, 4, 4, image, 4, 4, 1, error, sizeof(error));
        require(handle, error);
        struct Cleanup { void* handle; ~Cleanup() { fpm_pose_destroy(handle); } } cleanup{handle};
        float seed[4] = {.2f, .3f, 1, .02f}, output[4], score;
        FpmPoseMetrics metrics{};
        FpmPoseRoute route{};
        FpmPoseAutoConfig automatic{0, 0, 1, 0};
        FpmPoseRefineConfig config;
        fpm_pose_refine_default_config(&config);
        config.max_iterations = 0;
        require(fpm_pose_refine_auto(handle, seed, 1, &config, &automatic, output, &score,
            nullptr, &route, &metrics, error, sizeof(error)) == -1 && setupAttempts == 0,
            "invalid input reached the Metal factory");
        require(fpm_pose_refine_auto(handle, seed, 1, nullptr, &automatic, output, &score,
            nullptr, &route, &metrics, error, sizeof(error)) == 0, error);
        require(setupAttempts == 1 && route.selected_backend == 0
            && route.reason == FPM_POSE_ROUTE_METAL_UNAVAILABLE
            && metrics.device_ms == -1 && metrics.gpu_buffer_bytes == 0,
            "hardware absence did not produce an explicit CPU fallback");
        unavailable = false;
        require(fpm_pose_refine_auto(handle, seed, 1, nullptr, &automatic, output, &score,
            nullptr, &route, &metrics, error, sizeof(error)) == -1
            && std::strstr(error, "forced shader compilation failure") && setupAttempts == 2,
            "initialization failure was hidden by automatic fallback");
        require(fpm_pose_memory(handle, &metrics) == 0 && metrics.gpu_buffer_bytes == 0,
            "failed initialization retained a Metal buffer");
        std::cout << "Auto validation, typed absence fallback and initialization error propagation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
