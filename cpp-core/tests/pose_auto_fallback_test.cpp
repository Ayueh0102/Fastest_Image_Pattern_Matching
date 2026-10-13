// SPDX-License-Identifier: BSD-2-Clause
// Link the portable solver with a controlled backend factory, without Metal.
// Auto permits absence fallback; resident refinement requires a real device.
#include "pose_internal.hpp"
#include <cstring>
#include <iostream>
#include <limits>

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

        FpmPoseResidentMetrics resident{};
        const auto poisonMetrics = [&] {
            resident = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
            metrics = {11, 12, 13, 14, 15};
        };
        const auto requireResetMetrics = [&] {
            require(resident.command_buffers == 0 && resident.cpu_waits == 0
                && resident.encoded_dispatches == 0 && resident.auxiliary_buffer_bytes == 0
                && resident.setup_ms == 0 && resident.encode_ms == 0 && resident.wait_ms == 0
                && resident.cpu_verify_ms == 0 && resident.cpu_verified == 0 && resident.reserved == 0,
                "resident failure retained stale diagnostics");
            require(metrics.wall_ms == 0 && metrics.device_ms == -1 && metrics.host_buffer_bytes == 0
                && metrics.gpu_buffer_bytes == 0 && metrics.gpu_allocated_bytes == 0,
                "resident failure retained stale metrics");
        };
        const auto invalidResident = [&](const float* poses, int count,
            const FpmPoseRefineConfig* options, int verify) {
            const int attempts = setupAttempts;
            poisonMetrics();
            require(fpm_pose_refine_resident(handle, poses, count, options, verify,
                output, &score, nullptr, &resident, &metrics, error, sizeof(error)) == -1
                && setupAttempts == attempts, "invalid resident input reached Metal setup");
            requireResetMetrics();
        };
        invalidResident(seed, 1, &config, 1); // max_iterations remains zero
        invalidResident(seed, 1, nullptr, -1);
        invalidResident(seed, 1, nullptr, 2);
        invalidResident(seed, 0, nullptr, 1);
        invalidResident(seed, 65537, nullptr, 1); // reject before reading pose storage
        invalidResident(nullptr, 1, nullptr, 1);
        float invalidSeed[4] = {std::numeric_limits<float>::quiet_NaN(), 0, 0, 0};
        invalidResident(invalidSeed, 1, nullptr, 1);

        void* cpu = fpm_pose_create(0, 1, image, 4, 4, image, 4, 4, 1, error, sizeof(error));
        require(cpu, error);
        Cleanup cpuCleanup{cpu};
        const int beforeCpu = setupAttempts;
        poisonMetrics();
        require(fpm_pose_refine_resident(cpu, seed, 1, nullptr, 1, output, &score,
            nullptr, &resident, &metrics, error, sizeof(error)) == -1
            && std::strstr(error, "requires a Metal or auto handle") && setupAttempts == beforeCpu,
            "resident CPU handle silently fell back or attempted Metal setup");
        requireResetMetrics();

        unavailable = true;
        for (int verify : {0, 1}) {
            const int attempts = setupAttempts;
            poisonMetrics();
            score = 99;
            require(fpm_pose_refine_resident(handle, seed, 1, nullptr, verify,
                output, &score, nullptr, &resident, &metrics, error, sizeof(error)) == -1
                && std::strstr(error, "forced no device") && setupAttempts == attempts+1
                && score == 99, "resident hardware absence was hidden by CPU fallback");
            requireResetMetrics();
        }
        unavailable = false;
        const int beforeShaderFailure = setupAttempts;
        poisonMetrics();
        require(fpm_pose_refine_resident(handle, seed, 1, nullptr, 1,
            output, &score, nullptr, &resident, &metrics, error, sizeof(error)) == -1
            && std::strstr(error, "forced shader compilation failure")
            && setupAttempts == beforeShaderFailure+1,
            "resident initialization failure was hidden or rewritten");
        requireResetMetrics();
        require(fpm_pose_memory(handle, &metrics) == 0 && metrics.gpu_buffer_bytes == 0,
            "failed resident initialization retained a Metal buffer");
        std::cout << "Auto fallback and resident validation/error propagation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
