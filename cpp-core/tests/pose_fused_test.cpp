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
constexpr double PI = 3.14159265358979323846;
constexpr double SCORE_TOLERANCE = 3e-5;
constexpr uint64_t AUXILIARY_LIMIT = 16u*1024u*1024u;
int checkedCalls = 0;

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

// This continuous fixture differs from the resident/rescue test textures.
// Generate both images independently: no production warp or NCC is used.
double texture(double x, double y, int channel)
{
    return .37+.025*channel+.13*std::sin(.137*x+.083*y+.7*channel)
        +.09*std::cos(.061*x-.193*y+.3*channel)
        +.08*std::sin(.0018*x*x-.0021*x*y+.0011*y*y+.9*channel)
        +.14*std::exp(-((x+6)*(x+6)+(y-4)*(y-4))/83.);
}

struct Fixture {
    int sw = 204, sh = 198, tw = 61, th = 59, channels = 1;
    std::array<double, 4> truth{2.375, -3.125, 27.75, std::log(1.06)};
    std::vector<float> source, templ;
};

Fixture fixture(int tw = 61, int th = 59, double angle = 27.75,
                double scale = 1.06, int channels = 1)
{
    Fixture f;
    f.tw = tw; f.th = th; f.channels = channels;
    f.truth[2] = angle; f.truth[3] = std::log(scale);
    f.source.resize(size_t(f.sw)*f.sh*channels);
    f.templ.resize(size_t(tw)*th*channels);
    const double radians = angle*PI/180, cs = std::cos(radians), sn = std::sin(radians);
    for (int y = 0; y < f.sh; ++y) for (int x = 0; x < f.sw; ++x) {
        const double dx = x-f.sw*.5-f.truth[0], dy = f.sh*.5-y-f.truth[1];
        for (int c = 0; c < channels; ++c)
            f.source[(size_t(y)*f.sw+x)*channels+c] = float(texture(
                (cs*dx+sn*dy)/scale, (-sn*dx+cs*dy)/scale, c));
    }
    for (int y = 0; y < th; ++y) for (int x = 0; x < tw; ++x)
        for (int c = 0; c < channels; ++c)
            f.templ[(size_t(y)*tw+x)*channels+c] = float(texture(x-tw*.5, th*.5-y, c));
    return f;
}

struct Handle {
    void* value = nullptr;
    explicit Handle(const Fixture& f, int backend)
    {
        char error[1024]{};
        value = fpm_pose_create(backend, 4, f.source.data(), f.sw, f.sh,
            f.templ.data(), f.tw, f.th, f.channels, error, sizeof(error));
        if (!value) throw std::runtime_error(error);
    }
    ~Handle() { fpm_pose_destroy(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

void setMode(Handle& handle, int mode)
{
    char error[1024]{};
    require(fpm_pose_set_resident_mode(handle.value, mode, error, sizeof(error)) == 0, error);
    require(fpm_pose_resident_mode(handle.value) == mode, "resident mode was not retained");
}

FpmPoseRefineConfig config()
{
    FpmPoseRefineConfig c{};
    fpm_pose_refine_default_config(&c);
    return c;
}

std::vector<float> seeds(const Fixture& f, int count = 2)
{
    std::vector<float> result(size_t(count)*4);
    for (int i = 0; i < count; ++i) {
        const std::array<double, 4> offset{i%2 ? -1.45 : 1.1, i%2 ? .95 : -1.65,
            i%2 ? -2.8 : 3.3, i%2 ? .035 : -.027};
        for (int k = 0; k < 4; ++k) result[size_t(i)*4+k] = float(f.truth[k]+offset[k]);
    }
    return result;
}

double sample(const Fixture& f, double x, double y, int channel)
{
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double fx = x-ix, fy = y-iy;
    double value = 0;
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
        if (ix+i >= 0 && ix+i < f.sw && iy+j >= 0 && iy+j < f.sh)
            value += f.source[(size_t(iy+j)*f.sw+ix+i)*f.channels+channel]
                *(i ? fx : 1-fx)*(j ? fy : 1-fy);
    return value;
}

// Two-pass centered float64 oracle, deliberately distinct from the production
// sum-of-squares scorer. Includes zero samples outside the full source image.
double reference(const Fixture& f, const float* pose)
{
    const double radians = double(pose[2])*PI/180, scale = std::exp(double(pose[3]));
    const double cs = scale*std::cos(radians), sn = scale*std::sin(radians);
    const double n = double(f.tw)*f.th;
    double patchMeans[3]{}, templateMeans[3]{};
    std::vector<double> patch(f.templ.size());
    for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
        for (int c = 0; c < f.channels; ++c) {
            const double u = x-f.tw*.5, v = f.th*.5-y;
            const size_t i = (size_t(y)*f.tw+x)*f.channels+c;
            patch[i] = sample(f, f.sw*.5+pose[0]+cs*u-sn*v,
                f.sh*.5-pose[1]-sn*u-cs*v, c);
            patchMeans[c] += patch[i]/n; templateMeans[c] += double(f.templ[i])/n;
        }
    double cross = 0, patchEnergy = 0, templateEnergy = 0;
    for (size_t i = 0; i < patch.size(); ++i) {
        const int c = int(i%f.channels);
        const double p = patch[i]-patchMeans[c], t = double(f.templ[i])-templateMeans[c];
        cross += p*t; patchEnergy += p*p; templateEnergy += t*t;
    }
    if (patchEnergy/n <= 1e-12 || templateEnergy/n <= 1e-12) return -2;
    return std::clamp(cross/std::sqrt(patchEnergy*templateEnergy), -1., 1.);
}

struct Result {
    std::vector<float> poses, scores;
    std::vector<FpmPoseRefineStats> stats;
    FpmPoseResidentMetrics resident{};
    FpmPoseMetrics metrics{};
};

void checkBounds(const std::vector<float>& initial, const Result& result,
                 const FpmPoseRefineConfig& options)
{
    const double bounds[]{options.max_translation, options.max_translation,
        options.optimize_angle ? options.max_angle_degrees : 0.,
        options.optimize_scale ? options.max_log_scale : 0.};
    for (size_t i = 0; i < initial.size(); ++i) {
        require(std::isfinite(result.poses[i]), "nonfinite fused pose");
        require(std::abs(double(result.poses[i])-initial[i]) <= bounds[i%4],
            "fused output escaped exact original seed bounds");
        if (bounds[i%4] == 0) require(result.poses[i] == initial[i], "fused changed frozen parameter");
    }
}

Result run(Handle& handle, const std::vector<float>& initial,
           const FpmPoseRefineConfig& options, int verify)
{
    Result result;
    const int count = int(initial.size()/4);
    result.poses.resize(initial.size()); result.scores.resize(count); result.stats.resize(count);
    char error[1024]{};
    require(fpm_pose_refine_resident(handle.value, initial.data(), count, &options, verify,
        result.poses.data(), result.scores.data(), result.stats.data(), &result.resident,
        &result.metrics, error, sizeof(error)) == 0, error);
    ++checkedCalls;
    const auto& d = result.resident;
    require(d.command_buffers > 0 && d.command_buffers == d.cpu_waits,
        "resident command/wait accounting");
    require(d.encoded_dispatches >= d.command_buffers, "resident dispatch accounting");
    if (fpm_pose_resident_mode(handle.value) == 1) {
        require(d.command_buffers == uint64_t((count+511)/512), "fused chunk boundary changed");
        require(d.encoded_dispatches == d.command_buffers, "fused path did not use one dispatch per chunk");
    }
    require(d.auxiliary_buffer_bytes > 0 && d.auxiliary_buffer_bytes <= AUXILIARY_LIMIT,
        "fused auxiliary memory cap");
    require(result.metrics.gpu_buffer_bytes >= d.auxiliary_buffer_bytes, "resident memory excludes workspace");
    require(d.cpu_verified == verify && (verify || d.cpu_verify_ms == 0), "resident verification flag/time");
    require(d.setup_ms >= 0 && d.encode_ms >= 0 && d.wait_ms >= 0 && d.cpu_verify_ms >= 0,
        "resident negative timing");
    require(result.metrics.wall_ms >= d.cpu_verify_ms && result.metrics.wall_ms >= d.setup_ms,
        "resident wall missing setup/verification");
    FpmPoseResidentMetrics last{};
    require(fpm_pose_last_resident_metrics(handle.value, &last) == 0, "last resident metrics unavailable");
    require(last.encoded_dispatches == d.encoded_dispatches && last.command_buffers == d.command_buffers
        && last.cpu_waits == d.cpu_waits && last.auxiliary_buffer_bytes == d.auxiliary_buffer_bytes
        && last.cpu_verified == d.cpu_verified, "last resident metrics stale");
    for (int i = 0; i < count; ++i) {
        require(std::isfinite(result.scores[i]), "nonfinite fused NCC score");
        require(result.stats[i].iterations >= 0 && result.stats[i].iterations <= options.max_iterations,
            "fused iteration budget exceeded");
        require(result.stats[i].evaluations >= 1+2*verify && result.stats[i].status >= 0
            && result.stats[i].status <= 3, "fused evaluation/status accounting");
    }
    checkBounds(initial, result, options);
    return result;
}

void checkScores(const Fixture& f, const std::vector<float>& initial,
                 const Result& result, int verify)
{
    Handle cpu(f, 0);
    const int count = int(result.scores.size());
    std::vector<float> before(count), after(count);
    FpmPoseMetrics metrics{}; char error[1024]{};
    require(fpm_pose_score(cpu.value, initial.data(), count, before.data(), &metrics, error, sizeof(error)) == 0, error);
    require(fpm_pose_score(cpu.value, result.poses.data(), count, after.data(), &metrics, error, sizeof(error)) == 0, error);
    for (int i = 0; i < count; ++i) {
        require(std::abs(double(result.scores[i])-after[i]) <= SCORE_TOLERANCE,
            "resident NCC differs from exact CPU score");
        if (verify) require(result.scores[i] == after[i] && after[i] >= before[i],
            "verified resident score regressed or was not exact CPU score");
        // Check the independent oracle for every ordinary pose and on both
        // sides of the 512-pose chunk boundary for the large batch.
        if (count <= 16 || i == 0 || i == 255 || i == 511 || i == count-1)
            require(std::abs(double(result.scores[i])-reference(f, result.poses.data()+size_t(i)*4)) <= SCORE_TOLERANCE,
                "resident NCC differs from independent centered oracle");
    }
}

void checkTruth(const Fixture& f, const Result& result)
{
    for (size_t i = 0; i < result.scores.size(); ++i) {
        require(result.scores[i] >= .98, "fused identifiable fixture NCC accuracy");
        require(std::hypot(result.poses[4*i]-f.truth[0], result.poses[4*i+1]-f.truth[1]) <= .5,
            "fused identifiable fixture position error");
        require(std::abs(std::remainder(result.poses[4*i+2]-f.truth[2], 360.)) <= .5,
            "fused identifiable fixture rotation error");
        require(std::abs(std::exp(result.poses[4*i+3]-f.truth[3])-1) <= .01,
            "fused identifiable fixture scale error");
    }
}

void testAccuracyAndSwitching()
{
    for (const auto& parameters : {std::array<double, 4>{31, 29, -68.25, .91},
        std::array<double, 4>{61, 59, 27.75, 1.06}, std::array<double, 4>{127, 129, 172.3, .94},
        std::array<double, 4>{128, 128, -132.75, 1.03}}) {
        const Fixture f = fixture(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3]);
        const auto initial = seeds(f);
        Handle metal(f, 1);
        require(fpm_pose_resident_mode(metal.value) == 0, "default resident mode changed");
        for (int verify : {0, 1}) {
            setMode(metal, 0);
            const auto baseline = run(metal, initial, config(), verify);
            checkScores(f, initial, baseline, verify); checkTruth(f, baseline);
            setMode(metal, 1);
            const auto fused = run(metal, initial, config(), verify);
            checkScores(f, initial, fused, verify); checkTruth(f, fused);
            for (size_t i = 0; i < fused.scores.size(); ++i) {
                require(std::abs(double(fused.scores[i])-baseline.scores[i]) <= 1e-4,
                    "fused score differs materially on identifiable fixture");
                require(std::hypot(fused.poses[4*i]-baseline.poses[4*i], fused.poses[4*i+1]-baseline.poses[4*i+1]) <= .1,
                    "fused position differs materially on identifiable fixture");
            }
            setMode(metal, 0);
            const auto restored = run(metal, initial, config(), verify);
            require(restored.poses == baseline.poses && restored.scores == baseline.scores,
                "switching mode back changed existing resident output");
        }
    }
}

void testDifficultInputs()
{
    for (int kind = 0; kind < 7; ++kind) {
        Fixture f = fixture(35, 33);
        auto initial = seeds(f);
        if (kind == 0 || kind == 1) std::fill(f.source.begin(), f.source.end(), .45f);
        if (kind == 0 || kind == 2) std::fill(f.templ.begin(), f.templ.end(), .45f);
        if (kind == 3)
            for (auto* values : {&f.source, &f.templ})
                for (float& value : *values) value = .5f+.0002f*value;
        if (kind == 4) {
            for (int y = 0; y < f.sh; ++y) for (int x = 0; x < f.sw; ++x)
                f.source[size_t(y)*f.sw+x] = float(.3+.12*std::cos(.21*x));
            for (int y = 0; y < f.th; ++y) for (int x = 0; x < f.tw; ++x)
                f.templ[size_t(y)*f.tw+x] = float(.3+.12*std::cos(.21*x));
        }
        if (kind == 5) initial = {95.25f, 85.75f, 11.5f, 0, 100000, -100000, 0, 0};
        if (kind == 6) f = fixture(1, 1);
        Handle metal(f, 1); setMode(metal, 1);
        for (int verify : {0, 1}) {
            const auto result = run(metal, initial, config(), verify);
            checkScores(f, initial, result, verify);
            for (int i = 0; i < 2; ++i) if (kind < 3 || kind == 6 || (kind == 5 && i == 1)) {
                require(result.scores[i] == -2, "fused accepted constant or empty support patch");
                require(std::equal(initial.begin()+4*i, initial.begin()+4*i+4, result.poses.begin()+4*i),
                    "fused moved degenerate pose");
            }
        }
    }
}

void testFrozenAndInwardBounds()
{
    const Fixture f = fixture(45, 37, 35998.25, .98);
    auto initial = seeds(f); initial[2] = 35996; initial[6] = 35999;
    Handle metal(f, 1); setMode(metal, 1);
    for (int frozen = 0; frozen < 4; ++frozen) {
        auto options = config(); options.max_iterations = frozen == 3 ? 1 : 5;
        if (frozen < 3) options.optimize_angle = 0;
        if (frozen < 2) options.optimize_scale = 0;
        if (frozen == 0) options.max_translation = 0;
        for (int verify : {0, 1}) checkScores(f, initial, run(metal, initial, options, verify), verify);
    }

    Fixture narrow;
    narrow.sw = 32768; narrow.sh = 26; narrow.tw = 19; narrow.th = 17;
    narrow.source.assign(size_t(narrow.sw)*narrow.sh, .2f);
    narrow.templ.resize(size_t(narrow.tw)*narrow.th);
    const int center = narrow.sw/2+10000;
    for (int y = 0; y < narrow.sh; ++y) for (int x = center-24; x <= center+24; ++x)
        narrow.source[size_t(y)*narrow.sw+x] = float(texture(x-center, y-narrow.sh*.5, 0));
    const std::vector<float> narrowInitial{10000, 0, 0, 0};
    auto options = config(); options.max_iterations = 4;
    options.optimize_angle = options.optimize_scale = 0; options.max_translation = .0006;
    for (double direction : {-1., 1.}) {
        for (int y = 0; y < narrow.th; ++y) for (int x = 0; x < narrow.tw; ++x)
            narrow.templ[size_t(y)*narrow.tw+x] = float(sample(narrow,
                center+x-narrow.tw*.5+direction*.04, narrow.sh*.5+y-narrow.th*.5+direction*.04, 0));
        Handle handle(narrow, 1); setMode(handle, 1);
        for (int verify : {0, 1}) {
            const auto result = run(handle, narrowInitial, options, verify);
            require(result.poses[0] == narrowInitial[0], "fused rounded outside sub-ULP inward bound");
            checkScores(narrow, narrowInitial, result, verify);
        }
    }
}

void testModeGuardsWithoutGpu()
{
    char error[1024]{};
    require(fpm_pose_resident_mode(nullptr) == -1, "null mode query accepted");
    require(fpm_pose_set_resident_mode(nullptr, 0, error, sizeof(error)) == -1, "null mode setter accepted");
    FpmPoseResidentMetrics diagnostic{};
    require(fpm_pose_last_resident_metrics(nullptr, &diagnostic) == -1, "null diagnostics handle accepted");
    for (int kind = 0; kind < 3; ++kind) {
        const Fixture f = kind == 1 ? fixture(31, 29, 0, 1, 3)
            : kind == 2 ? fixture(129, 129) : fixture();
        Handle handle(f, kind == 0 ? 0 : 2);
        require(fpm_pose_resident_mode(handle.value) == 0, "initial resident mode was not zero");
        FpmPoseMetrics before{}, after{};
        require(fpm_pose_memory(handle.value, &before) == 0 && before.gpu_buffer_bytes == 0,
            "CPU/auto mode guard initialized Metal eagerly");
        for (int rejected : {-1, 1, 2}) {
            error[0] = 0;
            require(fpm_pose_set_resident_mode(handle.value, rejected, error, sizeof(error)) == -1 && error[0],
                "unsupported fused mode accepted or missing error");
            require(fpm_pose_resident_mode(handle.value) == 0, "failed mode setter mutated mode");
        }
        setMode(handle, 0);
        require(fpm_pose_memory(handle.value, &after) == 0 && after.gpu_buffer_bytes == 0,
            "mode guard/reset initialized lazy GPU");
        require(fpm_pose_last_resident_metrics(handle.value, nullptr) == -1, "null diagnostic output accepted");
    }
}

void testLazyRecoveryAndChunking()
{
    const Fixture f = fixture(17, 19);
    Handle automatic(f, 2);
    FpmPoseMetrics memory{};
    setMode(automatic, 1);
    require(fpm_pose_memory(automatic.value, &memory) == 0 && memory.gpu_buffer_bytes == 0,
        "enabling fused mode eagerly initialized lazy GPU");
    char error[1024]{};
    for (int invalid : {-1, 2}) {
        require(fpm_pose_set_resident_mode(automatic.value, invalid, error, sizeof(error)) == -1,
            "invalid mode accepted after enabling fused");
        require(fpm_pose_resident_mode(automatic.value) == 1, "invalid mode reset valid selection");
    }
    auto options = config(); options.max_iterations = 2;
    auto initial = seeds(f, 513);
    for (int i = 0; i < 513; ++i) {
        initial[size_t(i)*4] += float((i%19)*.013);
        initial[size_t(i)*4+1] -= float((i%23)*.009);
    }
    const auto first = run(automatic, initial, options, 1);
    require(first.resident.command_buffers == 2, "513 fused poses did not split into two chunks");
    checkScores(f, initial, first, 1);
    require(fpm_pose_memory(automatic.value, &memory) == 0 && memory.gpu_buffer_bytes > 0,
        "lazy fused GPU memory was not reported");
    std::vector<float> boundary(initial.begin()+512*4, initial.end());
    const auto singleton = run(automatic, boundary, options, 1);
    require(singleton.scores[0] == first.scores.back()
        && std::equal(singleton.poses.begin(), singleton.poses.end(), first.poses.end()-4),
        "fused second chunk differs from isolated pose");
    require(singleton.resident.setup_ms == 0, "fused cached pipeline rebuilt");

    // An invalid call after successful GPU work must reset exposed diagnostics
    // while preserving the selected mode, then allow the next valid request.
    Result output; output.poses.resize(4); output.scores.resize(1); output.stats.resize(1);
    float invalid[4]{0, 0, std::numeric_limits<float>::infinity(), 0};
    output.resident.command_buffers = 99; output.metrics.wall_ms = 99;
    require(fpm_pose_refine_resident(automatic.value, invalid, 1, &options, 1,
        output.poses.data(), output.scores.data(), output.stats.data(), &output.resident,
        &output.metrics, error, sizeof(error)) == -1, "fused accepted nonfinite seed");
    require(output.resident.command_buffers == 0 && output.metrics.wall_ms == 0 && output.metrics.device_ms == -1,
        "fused failure retained output diagnostics");
    FpmPoseResidentMetrics last{};
    require(fpm_pose_last_resident_metrics(automatic.value, &last) == 0
        && last.command_buffers == 0 && last.encoded_dispatches == 0, "fused failure retained cached diagnostics");
    require(fpm_pose_resident_mode(automatic.value) == 1, "fused failure reset mode");
    initial.resize(3*4);
    const auto recovered = run(automatic, initial, options, 0);
    checkScores(f, initial, recovered, 0);
    setMode(automatic, 0);
    checkScores(f, initial, run(automatic, initial, options, 1), 1);
    setMode(automatic, 1);
    checkScores(f, initial, run(automatic, initial, options, 1), 1);
}

void testRescue()
{
    const Fixture f = fixture(27, 25);
    Handle metal(f, 1); setMode(metal, 1);
    const auto initial = seeds(f, 4);
    auto options = config(); options.max_iterations = 2;
    const auto baseline = run(metal, initial, options, 1);
    FpmPoseRescueConfig rescue{}; fpm_pose_rescue_default_config(&rescue);
    rescue.backend = 1; rescue.short_iterations = 3; rescue.max_rescued_seeds = 2;
    rescue.trigger_min_score = -1; rescue.trigger_score = 1;
    const uint64_t perParent = f.templ.size()*uint64_t(16*rescue.short_iterations+options.max_iterations);
    for (uint64_t budget : {uint64_t(0), perParent-1, 2*perParent}) {
        rescue.max_retry_sample_iterations = budget;
        Result result;
        result.poses.resize(initial.size()); result.scores.resize(initial.size()/4); result.stats.resize(initial.size()/4);
        FpmPoseRescueMetrics diagnostic{}; char error[1024]{};
        require(fpm_pose_refine_rescue(metal.value, initial.data(), int(result.scores.size()), &options, &rescue,
            result.poses.data(), result.scores.data(), result.stats.data(), &diagnostic, &result.metrics,
            error, sizeof(error)) == 0, error);
        ++checkedCalls;
        checkBounds(initial, result, options); checkScores(f, initial, result, 1);
        FpmPoseResidentMetrics last{};
        require(fpm_pose_last_resident_metrics(metal.value, &last) == 0, "rescue resident diagnostics unavailable");
        require(last.command_buffers == diagnostic.command_buffers && last.cpu_waits == diagnostic.cpu_waits
            && last.encoded_dispatches == last.command_buffers && last.command_buffers > 0,
            "rescue ignored fused mode or did not aggregate diagnostics");
        require(last.cpu_verified == 1 && diagnostic.selected_backend == 1, "rescue omitted CPU verification or changed backend");
        require(diagnostic.auxiliary_buffer_bytes > 0 && diagnostic.auxiliary_buffer_bytes <= AUXILIARY_LIMIT,
            "fused rescue memory cap");
        require(diagnostic.reserved_retry_sample_iterations == diagnostic.selected_seeds*perParent
            && diagnostic.reserved_retry_sample_iterations <= budget, "fused rescue changed full-parent budget reservation");
        require(diagnostic.selected_seeds <= 2 && diagnostic.accepted_seeds <= diagnostic.selected_seeds
            && diagnostic.retry_starts <= 16*diagnostic.selected_seeds
            && diagnostic.continuations <= diagnostic.selected_seeds, "fused rescue cap accounting");
        for (size_t i = 0; i < result.scores.size(); ++i) {
            require(result.scores[i] >= baseline.scores[i], "fused rescue regressed fused baseline");
            require(result.stats[i].iterations <= 2*options.max_iterations+16*rescue.short_iterations,
                "fused rescue exceeded iteration budget");
        }
        if (budget < perParent) {
            require(diagnostic.selected_seeds == 0 && diagnostic.budget_skipped_seeds > 0,
                "fused rescue admitted partial-parent budget");
            require(result.poses == baseline.poses && result.scores == baseline.scores,
                "disabled rescue changed fused baseline");
        } else require(diagnostic.selected_seeds == 2 && diagnostic.retry_starts > 0,
            "forced fused rescue did not exercise retry bank");
    }
    // The same handle can still run CPU rescue explicitly. Its last resident
    // diagnostics must not misleadingly describe the previous GPU call.
    rescue.backend = 0; rescue.max_rescued_seeds = 0;
    std::vector<float> poses(initial.size()), scores(initial.size()/4);
    FpmPoseRescueMetrics diagnostic{}; FpmPoseMetrics metrics{}; char error[1024]{};
    require(fpm_pose_refine_rescue(metal.value, initial.data(), int(scores.size()), &options, &rescue,
        poses.data(), scores.data(), nullptr, &diagnostic, &metrics, error, sizeof(error)) == 0, error);
    FpmPoseResidentMetrics last{};
    require(fpm_pose_last_resident_metrics(metal.value, &last) == 0
        && last.command_buffers == 0 && last.encoded_dispatches == 0, "CPU rescue retained stale GPU diagnostics");
}
} // namespace

int main()
{
    try {
        testModeGuardsWithoutGpu();
        const Fixture probe = fixture(); char error[1024]{};
        void* available = fpm_pose_create(1, 1, probe.source.data(), probe.sw, probe.sh,
            probe.templ.data(), probe.tw, probe.th, 1, error, sizeof(error));
        if (!available) {
            const std::string message(error);
            if (message == "No Metal GPU available" || message == "Metal backend was not compiled on this platform") {
                std::cout << "SKIP: " << message << '\n'; return 77;
            }
            throw std::runtime_error(message);
        }
        fpm_pose_destroy(available);
        testAccuracyAndSwitching(); testDifficultInputs(); testFrozenAndInwardBounds();
        testLazyRecoveryAndChunking(); testRescue();
        std::cout << "Fused gray resident: " << checkedCalls << " GPU calls; independent NCC/GT, baseline switching, "
            "strict bounds, degeneracy, lazy guards, 513-pose chunks and bounded rescue passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
