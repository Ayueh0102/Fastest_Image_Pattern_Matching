// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
constexpr uint64_t AUXILIARY_LIMIT = 16u*1024u*1024u;
constexpr double PI = 3.14159265358979323846;

double texture(double x, double y, int channel)
{
    return .4+.03*channel+.10*std::sin(.17*x+.11*y+.4*channel)
        +.08*std::cos(-.071*x+.21*y+.8*channel)
        +.05*std::sin(.0023*x*x+.0037*x*y-.0019*y*y+.6*channel)
        +.12*std::exp(-((x-7)*(x-7)+(y+5)*(y+5))/60.);
}

struct Fixture {
    int sw = 160, sh = 144, tw = 49, th = 47, channels;
    std::array<double, 4> truth{2.125, -1.375, -37.25, std::log(1.07)};
    std::vector<float> source, templ;
    explicit Fixture(int c) : channels(c), source(size_t(sw)*sh*c), templ(size_t(tw)*th*c)
    {
        const double theta = truth[2]*PI/180, scale = std::exp(truth[3]);
        for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x) {
            const double dx = x-sw*.5-truth[0], dy = sh*.5-y-truth[1];
            for (int k = 0; k < c; ++k)
                source[(size_t(y)*sw+x)*c+k] = float(texture((std::cos(theta)*dx+std::sin(theta)*dy)/scale,
                    (-std::sin(theta)*dx+std::cos(theta)*dy)/scale, k));
        }
        for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x) for (int k = 0; k < c; ++k)
            templ[(size_t(y)*tw+x)*c+k] = float(texture(x-tw*.5, th*.5-y, k));
    }
};

struct Handle {
    void* value = nullptr;
    Handle(const Fixture& fixture, int backend)
    {
        char error[1024]{};
        value = fpm_pose_create(backend, 4, fixture.source.data(), fixture.sw, fixture.sh,
            fixture.templ.data(), fixture.tw, fixture.th, fixture.channels, error, sizeof(error));
        if (!value) throw std::runtime_error(error);
    }
    ~Handle() { fpm_pose_destroy(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

struct Result {
    std::vector<float> poses, scores;
    std::vector<FpmPoseRefineStats> stats;
    FpmPoseRescueMetrics rescue{};
    FpmPoseMetrics metrics{};
};

FpmPoseRefineConfig refineConfig()
{
    FpmPoseRefineConfig result{};
    fpm_pose_refine_default_config(&result);
    return result;
}

FpmPoseRescueConfig rescueConfig(int backend)
{
    FpmPoseRescueConfig result{};
    fpm_pose_rescue_default_config(&result);
    result.backend = backend;
    return result;
}

std::vector<float> seeds(const Fixture& fixture, int count)
{
    std::vector<float> result(size_t(count)*4);
    for (int i = 0; i < count; ++i) {
        const std::array<double, 4> error{i%2 ? -1.2 : 1.5, i%2 ? 1.4 : -1.1,
            i%2 ? -3.1 : 2.5, i%2 ? .025 : -.035};
        for (int j = 0; j < 4; ++j) result[size_t(i)*4+j] = float(fixture.truth[j]+error[j]);
    }
    return result;
}

Result run(Handle& handle, const std::vector<float>& initial, const FpmPoseRefineConfig& config,
           const FpmPoseRescueConfig& rescue)
{
    Result result;
    result.poses.resize(initial.size()); result.scores.resize(initial.size()/4); result.stats.resize(initial.size()/4);
    char error[1024]{};
    if (fpm_pose_refine_rescue(handle.value, initial.data(), int(initial.size()/4), &config, &rescue,
        result.poses.data(), result.scores.data(), result.stats.data(), &result.rescue, &result.metrics,
        error, sizeof(error))) throw std::runtime_error(error);
    return result;
}

void check(const Fixture& fixture, const std::vector<float>& initial, const Result& result,
           const FpmPoseRefineConfig& config, const FpmPoseRescueConfig& rescue)
{
    Handle oracle(fixture, 0);
    std::vector<float> checked(result.scores.size()), seedScores(result.scores.size());
    FpmPoseMetrics ignored{}; char error[1024]{};
    for (const auto& pair : {std::make_pair(result.poses.data(), checked.data()), std::make_pair(initial.data(), seedScores.data())})
        require(fpm_pose_score(oracle.value, pair.first, int(checked.size()), pair.second,
            &ignored, error, sizeof(error)) == 0, "CPU oracle scoring failed");
    require(checked == result.scores, "rescue must return exact CPU scores");
    const std::array<double, 4> bounds{config.max_translation, config.max_translation,
        config.optimize_angle ? config.max_angle_degrees : 0., config.optimize_scale ? config.max_log_scale : 0.};
    for (size_t i = 0; i < initial.size(); ++i) {
        require(std::isfinite(result.poses[i]), "nonfinite rescue output");
        require(std::abs(double(result.poses[i])-initial[i]) <= bounds[i%4], "retry escaped original seed bounds");
        if (bounds[i%4] == 0) require(result.poses[i] == initial[i], "frozen coordinate changed");
    }
    for (size_t i = 0; i < checked.size(); ++i) {
        require(checked[i] >= seedScores[i], "rescue regressed original CPU score");
        require(result.stats[i].iterations <= 2*config.max_iterations+16*rescue.short_iterations,
            "rescue iterations exceeded declared per-parent work");
        require(result.stats[i].status >= 0 && result.stats[i].status <= 3, "invalid winning status");
    }
    const auto& diagnostic = result.rescue;
    const uint64_t perParent = fixture.templ.size()*uint64_t(16*rescue.short_iterations+config.max_iterations);
    require(diagnostic.accepted_seeds <= diagnostic.selected_seeds && diagnostic.selected_seeds <= diagnostic.eligible_seeds,
        "invalid rescue parent accounting");
    require(diagnostic.selected_seeds <= uint64_t(rescue.max_rescued_seeds), "rescue parent cap exceeded");
    require(diagnostic.retry_starts <= 16*diagnostic.selected_seeds && diagnostic.continuations <= diagnostic.selected_seeds,
        "rescue restart/continuation cap exceeded");
    require(diagnostic.reserved_retry_sample_iterations == diagnostic.selected_seeds*perParent &&
        diagnostic.reserved_retry_sample_iterations <= rescue.max_retry_sample_iterations, "rescue proxy budget exceeded");
    require(diagnostic.command_buffers == diagnostic.cpu_waits, "unexpected host wait accounting");
    require(diagnostic.auxiliary_buffer_bytes <= AUXILIARY_LIMIT, "GPU rescue workspace exceeded cap");
    require(diagnostic.selected_backend == rescue.backend, "rescue silently changed backend");
}

void testBackend(int backend)
{
    for (int channels : {1, 3}) {
        Fixture fixture(channels);
        Handle handle(fixture, backend);
        auto initial = seeds(fixture, 8);
        auto config = refineConfig();
        auto options = rescueConfig(backend);
        auto disabled = options; disabled.max_retry_sample_iterations = 0;
        const Result baseline = run(handle, initial, config, disabled);
        check(fixture, initial, baseline, config, disabled);
        auto noParents = options; noParents.max_rescued_seeds = 0;
        const Result cappedOff = run(handle, initial, config, noParents);
        require(cappedOff.poses == baseline.poses && cappedOff.scores == baseline.scores &&
            cappedOff.rescue.selected_seeds == 0, "zero parent cap changed baseline");
        const Result easy = run(handle, initial, config, options);
        check(fixture, initial, easy, config, options);
        require(easy.rescue.selected_seeds == 0 && easy.poses == baseline.poses && easy.scores == baseline.scores,
            "easy-case rescue changed the baseline or did unnecessary retries");
        for (size_t i = 0; i < easy.scores.size(); ++i) {
            require(easy.scores[i] >= .8f && std::hypot(easy.poses[4*i]-fixture.truth[0], easy.poses[4*i+1]-fixture.truth[1]) < .5,
                "identifiable analytic fixture lost precision");
        }
        // Force the policy window for budget tests, without consulting truth.
        config.max_iterations = 2;
        options.trigger_min_score = -1.; options.trigger_score = 1.; options.max_rescued_seeds = 2;
        options.short_iterations = 3;
        const auto forced = run(handle, initial, config, options);
        check(fixture, initial, forced, config, options);
        require(forced.rescue.selected_seeds == 2 && forced.rescue.parent_cap_skipped_seeds > 0,
            "forced rescue did not exercise parent cap");
        const uint64_t perParent = fixture.templ.size()*uint64_t(16*options.short_iterations+config.max_iterations);
        options.max_retry_sample_iterations = perParent-1;
        const auto budgeted = run(handle, initial, config, options);
        check(fixture, initial, budgeted, config, options);
        require(budgeted.rescue.selected_seeds == 0 && budgeted.rescue.budget_skipped_seeds > 0,
            "partial parent budget incorrectly admitted retries");
        options.max_retry_sample_iterations = 2*perParent;
        config.optimize_angle = 0; config.optimize_scale = 0;
        const auto frozen = run(handle, initial, config, options);
        check(fixture, initial, frozen, config, options);
        config.optimize_angle = 1; config.optimize_scale = 1;
        config.max_translation = 1e-10; config.max_angle_degrees = 1e-10; config.max_log_scale = 1e-10;
        const auto narrow = run(handle, initial, config, options);
        check(fixture, initial, narrow, config, options);
        require(narrow.poses == initial, "sub-ULP original bounds changed pose");
        std::fill(fixture.templ.begin(), fixture.templ.end(), .5f);
        Handle degenerate(fixture, backend);
        const auto invalid = run(degenerate, initial, refineConfig(), rescueConfig(backend));
        require(invalid.poses == initial && invalid.rescue.selected_seeds == 0 &&
            std::all_of(invalid.scores.begin(), invalid.scores.end(), [](float x) { return x == -2; }),
            "degenerate templates entered retry search");
    }
}

void testValidation()
{
    Fixture fixture(1); Handle handle(fixture, 0);
    auto initial = seeds(fixture, 1);
    float poses[4]{}, score{}; FpmPoseMetrics metrics{}; FpmPoseRescueMetrics diagnostic{};
    char error[1024]{};
    const auto reject = [&](FpmPoseRescueConfig options) {
        metrics.wall_ms = 99; diagnostic.selected_seeds = 99;
        require(fpm_pose_refine_rescue(handle.value, initial.data(), 1, nullptr, &options,
            poses, &score, nullptr, &diagnostic, &metrics, error, sizeof(error)) != 0, "invalid rescue configuration accepted");
        require(metrics.wall_ms == 0 && metrics.device_ms == -1 && diagnostic.selected_seeds == 0,
            "rescue failed request returned stale diagnostics");
    };
    auto options = rescueConfig(0); options.backend = 2; reject(options);
    options = rescueConfig(0); options.reserved = 1; reject(options);
    options = rescueConfig(0); options.max_rescued_seeds = 65; reject(options);
    options = rescueConfig(0); options.short_iterations = 0; reject(options);
    options = rescueConfig(0); options.translation_radius = -1; reject(options);
    options = rescueConfig(0); options.trigger_score = std::numeric_limits<double>::quiet_NaN(); reject(options);
    options = rescueConfig(0); options.trigger_min_score = .9; reject(options);
    require(fpm_pose_refine_rescue(handle.value, initial.data(), 1, nullptr, nullptr,
        poses, &score, nullptr, nullptr, &metrics, error, sizeof(error)) != 0,
        "null required rescue diagnostics accepted");
    require(fpm_pose_refine_rescue(handle.value, initial.data(), 1, nullptr, nullptr,
        poses, &score, nullptr, &diagnostic, nullptr, error, sizeof(error)) != 0,
        "null required general metrics accepted");
    require(fpm_pose_refine_rescue(handle.value, initial.data(), 1, nullptr, nullptr,
        poses, &score, nullptr, &diagnostic, &metrics, error, sizeof(error)) == 0,
        "optional configs/stats or recovery after invalid request failed");
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const bool useMetal = argc == 2 && std::string(argv[1]) == "--metal";
        if (argc != 1 && !useMetal) throw std::runtime_error("Usage: pose_rescue_test [--metal]");
        if (useMetal) {
            Fixture probe(1); char error[1024]{};
            void* metal = fpm_pose_create(1, 1, probe.source.data(), probe.sw, probe.sh,
                probe.templ.data(), probe.tw, probe.th, 1, error, sizeof(error));
            if (!metal) {
                if (std::string(error) == "No Metal GPU available" || std::string(error) == "Metal backend was not compiled on this platform") {
                    std::cout << "SKIP: " << error << '\n'; return 77;
                }
                throw std::runtime_error(error);
            }
            fpm_pose_destroy(metal);
            testBackend(1);
        } else {
            testValidation();
            testBackend(0);
        }
        std::cout << "Bounded rescue: CPU scores, original bounds, budget/parent caps, no-op fast path and API validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
