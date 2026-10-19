// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t AUXILIARY_LIMIT = 16u*1024u*1024u;
constexpr double SCORE_TOLERANCE = 3e-5;
const double PI = std::acos(-1.0);

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    int sw = 192, sh = 176, tw = 65, th = 67, channels = 1;
    std::array<double, 4> truth{3.125, -2.375, -43.25, std::log(1.13)};
    std::vector<float> source, templ;
};

double texture(double x, double y, int c)
{
    return .4+.03*c+.11*std::sin(.093*x+.067*y+.4*c)
        +.10*std::cos(-.041*x+.127*y+.8*c)
        +.07*std::sin(.0013*x*x+.0017*x*y-.0009*y*y+.6*c)
        +.10*std::exp(-((x-11)*(x-11)+(y+8)*(y+8))/180.);
}

Fixture fixture(int channels, double angle = -43.25, double scale = 1.13)
{
    Fixture f;
    f.channels = channels; f.truth[2] = angle; f.truth[3] = std::log(scale);
    f.source.resize(size_t(f.sw)*f.sh*channels);
    f.templ.resize(size_t(f.tw)*f.th*channels);
    // Independently evaluate a continuous scene and template. Neither calls the
    // production warp/scorer; bilinear source sampling retains its real error.
    const double a = angle*PI/180, cs = std::cos(a), sn = std::sin(a);
    for (int y = 0; y < f.sh; ++y) for (int x = 0; x < f.sw; ++x) {
        const double dx = x-f.sw*.5-f.truth[0], dy = f.sh*.5-y-f.truth[1];
        for (int c = 0; c < channels; ++c)
            f.source[(size_t(y)*f.sw+x)*channels+c] = float(texture((cs*dx+sn*dy)/scale, (-sn*dx+cs*dy)/scale, c));
    }
    for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
        for (int c = 0; c < channels; ++c)
            f.templ[(size_t(y)*f.tw+x)*channels+c] = float(texture(x-f.tw*.5, f.th*.5-y, c));
    return f;
}

struct Handle {
    void* value = nullptr;
    Handle(const Fixture& f, int backend)
    {
        char error[1024]{};
        value = fpm_pose_create(backend, 8, f.source.data(), f.sw, f.sh,
            f.templ.data(), f.tw, f.th, f.channels, error, sizeof(error));
        if (!value) throw std::runtime_error(error);
    }
    ~Handle() { fpm_pose_destroy(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

double sample(const Fixture& f, double x, double y, int c)
{
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double fx = x-ix, fy = y-iy;
    double value = 0;
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
        if (ix+i >= 0 && ix+i < f.sw && iy+j >= 0 && iy+j < f.sh)
            value += f.source[(size_t(iy+j)*f.sw+ix+i)*f.channels+c]
                *(i ? fx : 1-fx)*(j ? fy : 1-fy);
    return value;
}

double reference(const Fixture& f, const float* pose)
{
    const double a = double(pose[2])*PI/180, scale = std::exp(double(pose[3]));
    const double cs = scale*std::cos(a), sn = scale*std::sin(a);
    std::vector<double> values(f.templ.size());
    double means[3]{}, templateMeans[3]{}, cross = 0, energy = 0, templateEnergy = 0;
    const double n = double(f.tw)*f.th;
    for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
        for (int c = 0; c < f.channels; ++c) {
            const double u = x-f.tw*.5, v = f.th*.5-y;
            const size_t i = (size_t(y)*f.tw+x)*f.channels+c;
            values[i] = sample(f, f.sw*.5+pose[0]+cs*u-sn*v, f.sh*.5-pose[1]-sn*u-cs*v, c);
            means[c] += values[i]/n; templateMeans[c] += double(f.templ[i])/n;
        }
    for (size_t i = 0; i < values.size(); ++i) {
        const int c = int(i%f.channels);
        const double v = values[i]-means[c], t = double(f.templ[i])-templateMeans[c];
        cross += v*t; energy += v*v; templateEnergy += t*t;
    }
    if (energy/n <= 1e-12 || templateEnergy/n <= 1e-12) return -2;
    return std::max(-1., std::min(1., cross/std::sqrt(energy*templateEnergy)));
}

FpmPoseRefineConfig config()
{
    FpmPoseRefineConfig options;
    fpm_pose_refine_default_config(&options);
    return options;
}

std::vector<float> seeds(const Fixture& f)
{
    std::vector<float> poses;
    for (const auto& offset : {std::array<double, 4>{1.2, -1.3, 2.5, -.03},
                              std::array<double, 4>{-2.1, 1.7, -4., .055}})
        for (int k = 0; k < 4; ++k) poses.push_back(float(f.truth[k]+offset[k]));
    return poses;
}

struct Result {
    std::vector<float> poses, scores;
    std::vector<FpmPoseRefineStats> stats;
    FpmPoseResidentMetrics resident{};
    FpmPoseMetrics metrics{};
};

Result run(Handle& handle, const std::vector<float>& initial, const FpmPoseRefineConfig& options, int verify)
{
    const int count = int(initial.size()/4);
    Result result;
    result.poses.resize(initial.size()); result.scores.resize(count); result.stats.resize(count);
    char error[1024]{};
    require(fpm_pose_refine_resident(handle.value, initial.data(), count, &options, verify,
        result.poses.data(), result.scores.data(), result.stats.data(), &result.resident,
        &result.metrics, error, sizeof(error)) == 0, error);
    const auto& d = result.resident;
    require(d.command_buffers > 0 && d.command_buffers == d.cpu_waits, "resident command/wait accounting");
    require(d.encoded_dispatches >= d.command_buffers, "resident dispatch accounting");
    require(d.auxiliary_buffer_bytes > 0 && d.auxiliary_buffer_bytes <= AUXILIARY_LIMIT, "resident auxiliary cap");
    require(d.cpu_verified == verify && (verify || d.cpu_verify_ms == 0), "resident CPU verification flag/time");
    require(d.setup_ms >= 0 && d.encode_ms >= 0 && d.wait_ms >= 0 && d.cpu_verify_ms >= 0,
        "resident negative timing");
    require(result.metrics.wall_ms >= d.cpu_verify_ms && result.metrics.wall_ms >= d.setup_ms,
        "resident wall missing verification/setup");
    require(result.metrics.gpu_buffer_bytes >= d.auxiliary_buffer_bytes, "resident memory accounting");
    const double bounds[] = {options.max_translation, options.max_translation,
        options.optimize_angle ? options.max_angle_degrees : 0,
        options.optimize_scale ? options.max_log_scale : 0};
    for (int i = 0; i < count; ++i) {
        require(std::isfinite(result.scores[i]), "resident non-finite score");
        require(result.stats[i].iterations >= 0 && result.stats[i].iterations <= options.max_iterations,
            "resident iteration budget");
        require(result.stats[i].evaluations >= 1+2*verify && result.stats[i].status >= 0
            && result.stats[i].status <= 3, "resident evaluation/status accounting");
        for (int k = 0; k < 4; ++k) {
            const double value = result.poses[4*i+k];
            require(std::isfinite(value) && std::abs(value-double(initial[4*i+k])) <= bounds[k],
                "resident escaped exact seed-relative bound");
            if (bounds[k] == 0) require(result.poses[4*i+k] == initial[4*i+k], "resident frozen parameter changed");
        }
    }
    return result;
}

void checkScores(const Fixture& f, const std::vector<float>& initial, const Result& result, int verify)
{
    Handle cpu(f, 0);
    const int count = int(result.scores.size());
    std::vector<float> before(count), after(count);
    FpmPoseMetrics metrics{};
    char error[1024]{};
    require(fpm_pose_score(cpu.value, initial.data(), count, before.data(), &metrics, error, sizeof(error)) == 0, error);
    require(fpm_pose_score(cpu.value, result.poses.data(), count, after.data(), &metrics, error, sizeof(error)) == 0, error);
    for (int i = 0; i < count; ++i) {
        require(std::abs(double(result.scores[i])-after[i]) <= SCORE_TOLERANCE, "resident NCC differs from CPU");
        if (verify) require(result.scores[i] == after[i] && after[i] >= before[i], "verified resident score regression");
        // GPU-only monotonicity is intentionally NOT a contract. The benchmark
        // reports CPU-rechecked regressions separately instead of hiding them.
        if (count <= 16 || i == 0 || i == count/2 || i == count-1)
            require(std::abs(double(result.scores[i])-reference(f, result.poses.data()+4*i)) <= SCORE_TOLERANCE,
                "resident NCC differs from independent float64 oracle");
    }
}

void testAccuracy()
{
    for (int channels : {1, 3}) for (const auto& angleScale : {
        std::array<double, 2>{31.25, 1.08}, std::array<double, 2>{-74.25, .83},
        std::array<double, 2>{173.4, 1.16}}) {
        const Fixture f = fixture(channels, angleScale[0], angleScale[1]);
        Handle metal(f, 1);
        const auto initial = seeds(f);
        for (int verify : {0, 1}) {
            const auto result = run(metal, initial, config(), verify);
            checkScores(f, initial, result, verify);
            for (size_t i = 0; i < result.scores.size(); ++i) {
                require(result.scores[i] >= .8, "resident true pose score");
                require(std::hypot(result.poses[4*i]-f.truth[0], result.poses[4*i+1]-f.truth[1]) <= .5,
                    "resident true subpixel accuracy");
                require(std::abs(std::remainder(result.poses[4*i+2]-f.truth[2], 360.)) <= .5,
                    "resident true rotation accuracy");
                require(std::abs(std::exp(result.poses[4*i+3]-f.truth[3])-1) <= .01,
                    "resident true scale accuracy");
            }
        }
    }
}

void testDifficultInputs()
{
    for (int channels : {1, 3}) for (int kind = 0; kind < 6; ++kind) {
        Fixture f = fixture(channels);
        auto initial = seeds(f);
        if (kind == 0 || kind == 1) std::fill(f.source.begin(), f.source.end(), .5f);
        if (kind == 0 || kind == 2) std::fill(f.templ.begin(), f.templ.end(), .5f);
        if (kind == 3) {
            for (auto* values : {&f.source, &f.templ})
                for (float& value : *values) value = .5f+.0002f*value;
        }
        if (kind == 4) {
            for (int y = 0; y < f.sh; ++y) for (int x = 0; x < f.sw; ++x)
                for (int c = 0; c < channels; ++c)
                    f.source[(size_t(y)*f.sw+x)*channels+c] = float(.4+.1*std::sin(.17*x));
            for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
                for (int c = 0; c < channels; ++c)
                    f.templ[(size_t(y)*f.tw+x)*channels+c] = float(.4+.1*std::sin(.17*x));
        }
        if (kind == 5) initial = {82.25f, 63.75f, 0, 0, 100000, 100000, 0, 0};
        Handle metal(f, 1);
        for (int verify : {0, 1}) {
            const auto result = run(metal, initial, config(), verify);
            checkScores(f, initial, result, verify);
            for (int i = 0; i < 2; ++i) if (kind < 3 || (kind == 5 && i == 1)) {
                require(result.scores[i] == -2, "resident accepted degenerate/empty-support patch");
                require(std::equal(initial.begin()+4*i, initial.begin()+4*i+4, result.poses.begin()+4*i),
                    "resident moved degenerate/empty-support seed");
            }
        }
    }
}

void testFrozenAndLargeAngles()
{
    const Fixture f = fixture(3, 35998, .96);
    auto initial = seeds(f);
    // Keep all initial angles inside the public +/-36000-degree domain.
    initial[2] = 35996; initial[6] = 35999;
    Handle metal(f, 1);
    for (int mode = 0; mode < 3; ++mode) {
        auto options = config();
        options.max_iterations = mode == 2 ? 1 : 6;
        if (mode < 2) options.optimize_angle = options.optimize_scale = 0;
        if (mode == 1) options.max_translation = 0;
        for (int verify : {0, 1}) {
            const auto result = run(metal, initial, options, verify);
            checkScores(f, initial, result, verify);
        }
    }
}

void testNarrowBounds()
{
    Fixture f;
    f.sw = 32768; f.sh = 24; f.tw = 17; f.th = 13;
    f.source.assign(size_t(f.sw)*f.sh, .2f); f.templ.resize(size_t(f.tw)*f.th);
    const int center = f.sw/2+10000;
    for (int y = 0; y < f.sh; ++y) for (int x = center-20; x <= center+20; ++x)
        f.source[size_t(y)*f.sw+x] = float(.4+.2*std::sin(.43*(x-center)+.16*y)+.1*std::cos(.33*(x-center)-.29*y));
    const std::vector<float> initial{10000, 0, 0, 0};
    auto options = config();
    options.max_iterations = 3; options.optimize_angle = options.optimize_scale = 0;
    options.max_translation = .0006;
    for (double direction : {-1., 1.}) {
        for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
            f.templ[size_t(y)*f.tw+x] = float(sample(f, center+(x-f.tw*.5)+direction*.05,
                f.sh*.5+y-f.th*.5+direction*.05, 0));
        Handle metal(f, 1);
        for (int verify : {0, 1}) {
            const auto result = run(metal, initial, options, verify);
            require(result.poses[0] == initial[0], "resident rounded outside sub-ULP bound");
            // The float spacing is 0.0009765625 at tx=10000. There is no
            // representable neighbor within +/-0.0006: no epsilon is allowed.
        }
    }
}

void testLazyAndChunking()
{
    Fixture f = fixture(1);
    Handle automatic(f, 2);
    FpmPoseMetrics before{}, after{};
    require(fpm_pose_memory(automatic.value, &before) == 0 && before.gpu_buffer_bytes == 0,
        "resident auto handle initialized eagerly");
    auto options = config(); options.max_iterations = 1;
    auto initial = seeds(f);
    auto first = run(automatic, initial, options, 1);
    require(fpm_pose_memory(automatic.value, &after) == 0 && after.gpu_buffer_bytes > 0,
        "resident lazy Metal not accounted");
    auto second = run(automatic, initial, options, 0);
    require(second.resident.setup_ms == 0, "resident rebuilt cached pipeline");
    checkScores(f, initial, first, 1); checkScores(f, initial, second, 0);

    // A one-pixel constant template makes all 65536 seeds immediately invalid.
    // State/args/reduction storage still exceeds 16 MiB and MUST be chunked,
    // without the hundreds of millions of pixels of a throughput benchmark.
    f.sw = f.sh = 17; f.tw = f.th = 1;
    f.source.assign(17*17, .5f); f.templ.assign(1, .5f);
    Handle chunked(f, 1);
    initial.assign(65536*4, 0);
    for (int count : {65536, 3}) {
        initial.resize(size_t(count)*4);
        const auto result = run(chunked, initial, options, 0);
        require(count == 3 || result.resident.command_buffers > 1, "resident oversized batch was not chunked");
        require(result.poses == initial && std::all_of(result.scores.begin(), result.scores.end(),
            [](float score) { return score == -2; }), "resident chunked result indexing/stale state");
        checkScores(f, initial, result, 0);
    }
}

void testValidation(int backend)
{
    const Fixture f = fixture(1);
    Handle handle(f, backend);
    const float initial[4]{0, 0, 0, 0};
    float output[4]{}, score = 0;
    FpmPoseRefineStats stats{};
    FpmPoseResidentMetrics resident{};
    FpmPoseMetrics metrics{};
    char error[1024]{};
    const auto invalid = [&](const float* input, int count, const FpmPoseRefineConfig* options, int verify) {
        resident.command_buffers = 99; metrics.wall_ms = 99; metrics.device_ms = 99;
        require(fpm_pose_refine_resident(handle.value, input, count, options, verify, output, &score,
            &stats, &resident, &metrics, error, sizeof(error)) != 0, "resident accepted invalid request");
        require(resident.command_buffers == 0 && metrics.wall_ms == 0 && metrics.device_ms == -1,
            "resident failure returned stale metrics");
    };
    if (backend == 0) { invalid(initial, 1, nullptr, 1); return; }
    for (int count : {0, 65537}) invalid(initial, count, nullptr, 1);
    for (int verify : {-1, 2}) invalid(initial, 1, nullptr, verify);
    invalid(nullptr, 1, nullptr, 0);
    float nonfinite[4]{0, 0, std::numeric_limits<float>::infinity(), 0};
    invalid(nonfinite, 1, nullptr, 0);
    nonfinite[2] = 36001; invalid(nonfinite, 1, nullptr, 0);
    for (int iterations : {0, 201}) {
        auto options = config(); options.max_iterations = iterations;
        invalid(initial, 1, &options, 0);
    }
    auto options = config(); options.max_translation = -.1;
    invalid(initial, 1, &options, 1);
    require(fpm_pose_refine_resident(handle.value, initial, 1, nullptr, 1, output, &score,
        nullptr, &resident, &metrics, error, sizeof(error)) == 0, "resident null optional config/stats or recovery failed");
}
} // namespace

int main()
{
    try {
        testValidation(0);
        const Fixture probe = fixture(1);
        char error[1024]{};
        void* metal = fpm_pose_create(1, 1, probe.source.data(), probe.sw, probe.sh,
            probe.templ.data(), probe.tw, probe.th, 1, error, sizeof(error));
        if (!metal) {
            const std::string message(error);
            if (message == "No Metal GPU available" || message == "Metal backend was not compiled on this platform") {
                std::cout << "SKIP: " << message << '\n'; return 77;
            }
            throw std::runtime_error(message);
        }
        fpm_pose_destroy(metal);
        testValidation(1);
        testAccuracy();
        testDifficultInputs();
        testFrozenAndLargeAngles();
        testNarrowBounds();
        testLazyAndChunking();
        std::cout << "Resident gray/RGB accuracy, independent NCC, degeneracy, strict bounds, lazy setup and chunk cap passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
