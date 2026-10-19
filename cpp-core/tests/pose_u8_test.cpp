// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Handle = std::unique_ptr<void, decltype(&fpm_pose_destroy)>;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    int sw = 65, sh = 67, tw = 17, th = 19, channels = 1;
    int left = 23, top = 29;
    uint64_t sourceStride = 0, templateStride = 0;
    std::vector<uint8_t> source, templ;
    std::vector<float> sourceFloat, templateFloat;
};

Fixture fixture(int channels, int invert, bool padded)
{
    Fixture f;
    f.channels = channels;
    f.sourceStride = uint64_t(f.sw)*channels+(padded ? 13 : 0);
    f.templateStride = uint64_t(f.tw)*channels+(padded ? 7 : 0);
    f.source.assign(size_t(f.sourceStride)*f.sh, 219);
    f.templ.assign(size_t(f.templateStride)*f.th, 37);
    f.sourceFloat.resize(size_t(f.sw)*f.sh*channels);
    f.templateFloat.resize(size_t(f.tw)*f.th*channels);
    std::array<std::array<bool, 256>, 3> seen{};
    for (int y = 0; y < f.sh; ++y) for (int x = 0; x < f.sw; ++x)
        for (int c = 0; c < channels; ++c) {
            const uint32_t pixel = uint32_t(y*f.sw+x);
            uint32_t hash = pixel*747796405u+uint32_t(c)*2891336453u+277803737u;
            hash = ((hash >> ((hash >> 28)+4))^hash)*277803737u;
            // The first 256 pixels cover every byte value in every channel.
            const uint8_t value = uint8_t(pixel < 256 ? pixel+83u*c : (hash >> 22)^hash);
            f.source[size_t(y*f.sourceStride+x*channels+c)] = value;
            seen[c][value] = true;
            f.sourceFloat[(size_t(y)*f.sw+x)*channels+c] = (invert ? 255-value : value)/255.f;
        }
    for (int c = 0; c < channels; ++c)
        require(std::all_of(seen[c].begin(), seen[c].end(), [](bool value) { return value; }),
            "fixture does not cover all byte values");
    for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
        for (int c = 0; c < channels; ++c) {
            const uint8_t source = f.source[size_t((y+f.top)*f.sourceStride+(x+f.left)*channels+c)];
            const uint8_t value = uint8_t(invert ? 255-source : source);
            f.templ[size_t(y*f.templateStride+x*channels+c)] = value;
            f.templateFloat[(size_t(y)*f.tw+x)*channels+c] = value/255.f;
        }
    return f;
}

void sameStats(const std::vector<FpmPoseRefineStats>& a, const std::vector<FpmPoseRefineStats>& b)
{
    require(a.size() == b.size(), "refinement statistics length differs");
    for (size_t i = 0; i < a.size(); ++i)
        require(a[i].iterations == b[i].iterations && a[i].evaluations == b[i].evaluations
            && a[i].status == b[i].status, "direct-u8 refinement statistics differ");
}

void sameStorage(void* a, void* b, int mode, bool hasMetal)
{
    FpmPoseStorageInfo first{}, second{};
    require(fpm_pose_storage_info(a, &first) == 0 && fpm_pose_storage_info(b, &second) == 0,
        "storage query failed");
    require(first.mode == mode && second.mode == mode && first.host_image_bytes > 0
        && first.host_image_bytes == second.host_image_bytes
        && first.gpu_image_bytes == second.gpu_image_bytes
        && first.shared_image_bytes == second.shared_image_bytes
        && first.unique_image_bytes == second.unique_image_bytes,
        "direct-u8 image allocation accounting differs");
    require((second.gpu_image_bytes > 0) == hasMetal, "unexpected Metal initialization state");
    require(second.unique_image_bytes == second.host_image_bytes+second.gpu_image_bytes-second.shared_image_bytes,
        "invalid unique image allocation accounting");
    if (mode && hasMetal)
        require(second.host_image_bytes == second.gpu_image_bytes
            && second.host_image_bytes == second.shared_image_bytes,
            "direct-u8 shared image pages are not aliased");
}

void equivalent(int backend, int storage, int channels, int invert, bool padded, bool resident)
{
    Fixture f = fixture(channels, invert, padded);
    char error[1024]{};
    Handle old(fpm_pose_create_with_storage(backend, 2, f.sourceFloat.data(), f.sw, f.sh,
        f.templateFloat.data(), f.tw, f.th, channels, storage, error, sizeof(error)), fpm_pose_destroy);
    require(bool(old), error);
    Handle direct(fpm_pose_create_u8(backend, 2, f.source.data(), f.sw, f.sh, f.sourceStride,
        f.templ.data(), f.tw, f.th, f.templateStride, channels, invert, storage, error, sizeof(error)), fpm_pose_destroy);
    require(bool(direct), error);
    require(fpm_pose_input_format(old.get()) == 0 && fpm_pose_input_format(direct.get()) == 1,
        "input format diagnostic differs");
    require(fpm_pose_backend(old.get()) == backend && fpm_pose_backend(direct.get()) == backend,
        "direct-u8 creation backend differs");
    sameStorage(old.get(), direct.get(), storage, backend == 1);
    // Mutate then release every caller buffer. Handles must own independent snapshots.
    std::fill(f.source.begin(), f.source.end(), 0);
    std::fill(f.templ.begin(), f.templ.end(), 255);
    std::fill(f.sourceFloat.begin(), f.sourceFloat.end(), std::numeric_limits<float>::quiet_NaN());
    std::fill(f.templateFloat.begin(), f.templateFloat.end(), std::numeric_limits<float>::quiet_NaN());
    std::vector<uint8_t>().swap(f.source); std::vector<uint8_t>().swap(f.templ);
    std::vector<float>().swap(f.sourceFloat); std::vector<float>().swap(f.templateFloat);
    const float tx = float(f.left+f.tw*.5-f.sw*.5), ty = float(f.sh*.5-f.top-f.th*.5);
    const std::vector<float> seeds{tx, ty, 0, 0, tx+.35f, ty-.2f, .4f, .002f,
        tx-.6f, ty+.5f, -1.f, -.01f, 10000, -10000, 90, .1f};
    const int count = int(seeds.size()/4);
    std::vector<float> oldScores(count), directScores(count), oldPoses(seeds.size()), directPoses(seeds.size());
    std::vector<FpmPoseRefineStats> oldStats(count), directStats(count);
    FpmPoseMetrics oldMetrics{}, directMetrics{};
    FpmPoseRefineConfig config{};
    fpm_pose_refine_default_config(&config);
    config.max_iterations = 3;
    require(fpm_pose_score(old.get(), seeds.data(), count, oldScores.data(), &oldMetrics, error, sizeof(error)) == 0, error);
    require(fpm_pose_score(direct.get(), seeds.data(), count, directScores.data(), &directMetrics, error, sizeof(error)) == 0, error);
    require(oldScores == directScores, "direct-u8 scores differ from explicit conversion");
    require(directScores.front() > .9999f && directScores.back() == -2,
        "direct-u8 fixture exact match or outside-source score is invalid");
    require(fpm_pose_refine(old.get(), seeds.data(), count, &config, oldPoses.data(), oldScores.data(),
        oldStats.data(), &oldMetrics, error, sizeof(error)) == 0, error);
    require(fpm_pose_refine(direct.get(), seeds.data(), count, &config, directPoses.data(), directScores.data(),
        directStats.data(), &directMetrics, error, sizeof(error)) == 0, error);
    require(oldScores == directScores && oldPoses == directPoses, "direct-u8 CPU refinement differs");
    sameStats(oldStats, directStats);
    require(fpm_pose_refine_accelerated(old.get(), seeds.data(), count, &config, 1, oldPoses.data(), oldScores.data(),
        oldStats.data(), &oldMetrics, error, sizeof(error)) == 0, error);
    require(fpm_pose_refine_accelerated(direct.get(), seeds.data(), count, &config, 1, directPoses.data(), directScores.data(),
        directStats.data(), &directMetrics, error, sizeof(error)) == 0, error);
    require(oldScores == directScores && oldPoses == directPoses, "direct-u8 accelerated refinement differs");
    sameStats(oldStats, directStats);
    if (backend == 2) {
        sameStorage(old.get(), direct.get(), storage, false);
        FpmPoseAutoConfig automatic{};
        fpm_pose_auto_default_config(&automatic);
        automatic.min_pose_samples = std::numeric_limits<uint64_t>::max();
        FpmPoseRoute oldRoute{}, directRoute{};
        require(fpm_pose_refine_auto(old.get(), seeds.data(), count, &config, &automatic, oldPoses.data(),
            oldScores.data(), oldStats.data(), &oldRoute, &oldMetrics, error, sizeof(error)) == 0, error);
        require(fpm_pose_refine_auto(direct.get(), seeds.data(), count, &config, &automatic, directPoses.data(),
            directScores.data(), directStats.data(), &directRoute, &directMetrics, error, sizeof(error)) == 0, error);
        require(oldRoute.selected_backend == 0 && directRoute.selected_backend == 0
            && oldRoute.reason == directRoute.reason, "direct-u8 lazy CPU route differs");
        require(oldScores == directScores && oldPoses == directPoses, "direct-u8 lazy CPU result differs");
        sameStats(oldStats, directStats);
        sameStorage(old.get(), direct.get(), storage, false);
    }
    if (!resident) return;
    for (int mode : {0, 2}) {
        if (mode && channels != 1) continue;
        require(fpm_pose_set_resident_mode(old.get(), mode, error, sizeof(error)) == 0, error);
        require(fpm_pose_set_resident_mode(direct.get(), mode, error, sizeof(error)) == 0, error);
        for (int verify : {0, 1}) {
            FpmPoseResidentMetrics oldResident{}, directResident{};
            require(fpm_pose_refine_resident(old.get(), seeds.data(), count, &config, verify, oldPoses.data(),
                oldScores.data(), oldStats.data(), &oldResident, &oldMetrics, error, sizeof(error)) == 0, error);
            require(fpm_pose_refine_resident(direct.get(), seeds.data(), count, &config, verify, directPoses.data(),
                directScores.data(), directStats.data(), &directResident, &directMetrics, error, sizeof(error)) == 0, error);
            require(oldScores == directScores && oldPoses == directPoses, "direct-u8 resident refinement differs");
            sameStats(oldStats, directStats);
            require(oldResident.command_buffers > 0 && oldResident.command_buffers == directResident.command_buffers
                && oldResident.cpu_waits == directResident.cpu_waits
                && oldResident.encoded_dispatches == directResident.encoded_dispatches
                && oldResident.auxiliary_buffer_bytes == directResident.auxiliary_buffer_bytes
                && oldResident.cpu_verified == directResident.cpu_verified,
                "direct-u8 resident work accounting differs");
            FpmPoseResidentKernelInfo oldKernel{}, directKernel{};
            require(fpm_pose_last_resident_kernel(old.get(), &oldKernel) == 0
                && fpm_pose_last_resident_kernel(direct.get(), &directKernel) == 0, "resident kernel query failed");
            require(oldKernel.threads == directKernel.threads && oldKernel.simd_width == directKernel.simd_width
                && oldKernel.threadgroup_bytes == directKernel.threadgroup_bytes,
                "direct-u8 resident kernel geometry differs");
            sameStorage(old.get(), direct.get(), storage, true);
        }
    }
}

struct Arguments {
    int backend = 0, threads = 1, sw = 4, sh = 4, tw = 4, th = 4, channels = 1, invert = 0, storage = 0;
    uint64_t sourceStride = 4, templateStride = 4;
    const uint8_t* source = nullptr;
    const uint8_t* templ = nullptr;
};

void* create(const Arguments& a, char* error, int capacity)
{
    return fpm_pose_create_u8(a.backend, a.threads, a.source, a.sw, a.sh, a.sourceStride,
        a.templ, a.tw, a.th, a.templateStride, a.channels, a.invert, a.storage, error, capacity);
}

void invalidArguments()
{
    std::array<uint8_t, 48> image{};
    Arguments valid;
    valid.source = valid.templ = image.data();
    auto invalid = [&](const Arguments& args) {
        char error[256]{};
        Handle handle(create(args, error, sizeof(error)), fpm_pose_destroy);
        require(!handle && error[0], "invalid direct-u8 arguments accepted or missing error");
    };
    for (int value : {-1, 3}) {auto a = valid; a.backend = value; invalid(a);}
    for (int value : {0, 65}) {auto a = valid; a.threads = value; invalid(a);}
    for (int value : {0, 2, 4}) {auto a = valid; a.channels = value; invalid(a);}
    for (int value : {-1, 0, 32769}) {
        auto a = valid; a.sw = value; invalid(a);
        a = valid; a.sh = value; invalid(a);
        a = valid; a.tw = value; invalid(a);
        a = valid; a.th = value; invalid(a);
    }
    {auto a = valid; a.source = nullptr; invalid(a);}
    {auto a = valid; a.templ = nullptr; invalid(a);}
    for (int value : {-1, 2}) {auto a = valid; a.invert = value; invalid(a);}
    for (int value : {-1, 1, 2}) {auto a = valid; a.storage = value; invalid(a);}
    for (uint64_t value : {uint64_t(0), uint64_t(3), std::numeric_limits<uint64_t>::max(),
        uint64_t(std::numeric_limits<ptrdiff_t>::max())+1,
        uint64_t(std::numeric_limits<ptrdiff_t>::max())/2}) {
        auto a = valid; a.sourceStride = value; invalid(a);
        a = valid; a.templateStride = value; invalid(a);
    }
    {auto a = valid; a.channels = 3; a.sourceStride = 11; a.templateStride = 12; invalid(a);}
    {auto a = valid; a.channels = 3; a.sourceStride = 12; a.templateStride = 11; invalid(a);}
    {auto a = valid; a.sw = a.sh = 32768; a.sourceStride = 32768; invalid(a);}
    {auto a = valid; a.tw = a.th = 32768; a.templateStride = 32768; invalid(a);}
    const auto* wrapped = reinterpret_cast<const uint8_t*>(std::numeric_limits<uintptr_t>::max()-8);
    {auto a = valid; a.source = wrapped; invalid(a);}
    {auto a = valid; a.templ = wrapped; invalid(a);}
    auto a = valid; a.source = nullptr;
    require(!create(a, nullptr, 0), "null optional error destination is not supported");
    char tiny[]{'x', 'y'};
    require(!create(a, tiny, 1) && tiny[0] == '\0' && tiny[1] == 'y', "small error destination overrun");
    require(fpm_pose_input_format(nullptr) == -1, "null input format query must return -1");
}
}

int main(int argc, char** argv)
{
    try {
        require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--metal"), "usage: pose_u8_test [--metal]");
        const bool metal = argc == 2;
        invalidArguments();
        if (metal) {
            uint8_t image[16]{}; char error[1024]{};
            Handle probe(fpm_pose_create_u8(1, 1, image, 4, 4, 4, image, 4, 4, 4, 1, 0, 0,
                error, sizeof(error)), fpm_pose_destroy);
            if (!probe) {
                const std::string message(error);
                if (message == "No Metal GPU available" || message == "Metal backend was not compiled on this platform") {
                    std::cout << "SKIP: " << message << '\n'; return 77;
                }
                throw std::runtime_error(message);
            }
            for (int storage : {0, 1}) for (int channels : {1, 3}) for (int invert : {0, 1})
                equivalent(1, storage, channels, invert, true, true);
            for (int storage : {0, 1}) equivalent(2, storage, 1, 1, true, true);
            std::cout << "Direct-u8 Metal: 10 float-reference pairs, score/CPU/accelerated/resident 0/2, "
                "exact poses/scores/statistics, shared/copy ownership and lazy initialization passed\n";
        } else {
            for (int channels : {1, 3}) for (int invert : {0, 1}) for (bool padded : {false, true})
                equivalent(0, 0, channels, invert, padded, false);
            for (int channels : {1, 3}) equivalent(2, 0, channels, 1, true, false);
            std::cout << "Direct-u8 CPU: 10 float-reference pairs, gray/RGB all 256 values, dense/padded strides, "
                "inversion, caller lifetime, exact refinement/statistics, lazy auto and guards passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
