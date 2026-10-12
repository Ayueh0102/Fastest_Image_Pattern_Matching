// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char**)
{
    try {
        const int backend = argc > 1 ? 1 : 0; // optional manual GPU run
        char error[1024]{};
        const int w = 35, h = 33;
        for (int channels : {1, 3}) for (int threads : {1, 8}) {
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
        fpm_pose_destroy(nullptr);
        std::cout << "Experimental pose ABI, lifetime, buffer reuse and NCC identity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
