// SPDX-License-Identifier: BSD-2-Clause
// The compact CPU path is the oracle for Metal construction, sparse scanning,
// early rejection and local peaks. Known quarter turns copy integer pixels;
// the fixtures do not call the production warp or orientation quantizer.
#include "sparse_orientation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace fpm;
using namespace fpm::research;
std::size_t comparisons = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void invalid(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

void fill(Image& image, std::uint8_t value) {
    for (int y = 0; y < image.height(); ++y)
        std::fill(image.row(y), image.row(y) + image.width(), value);
}

Image texture(int width = 24, int height = 20, bool lowContrast = false) {
    Image image(width, height);
    fill(image, 128);
    std::uint32_t state = 0x43c629b1u;
    for (int y = 2; y < height - 2; ++y)
        for (int x = 2; x < width - 2; ++x) {
            state = state * 1664525u + 1013904223u;
            image.row(y)[x] = lowContrast ? std::uint8_t(125 + ((state >> 24) % 7)) :
                std::uint8_t(32 + ((state >> 24) % 192));
        }
    return image;
}

void paste(Image& source, const Image& templ, int cx, int cy, int quarterTurns = 0) {
    const int q = ((quarterTurns % 4) + 4) % 4;
    for (int y = 0; y < templ.height(); ++y)
        for (int x = 0; x < templ.width(); ++x) {
            const int dx = x - templ.width() / 2, dy = y - templ.height() / 2;
            const int rx = q == 0 ? dx : q == 1 ? dy : q == 2 ? -dx : -dy;
            const int ry = q == 0 ? dy : q == 1 ? -dx : q == 2 ? -dy : dx;
            require(cx + rx >= 0 && cx + rx < source.width() &&
                cy + ry >= 0 && cy + ry < source.height(), "Quarter-turn fixture does not fit");
            source.row(cy + ry)[cx + rx] = templ.row(y)[x];
        }
}

void sameMatches(const std::vector<Match>& cpu, const std::vector<Match>& metal) {
    require(cpu.size() == metal.size(), "CPU/Metal sparse proposal counts differ");
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        const auto& a = cpu[i]; const auto& b = metal[i];
        require(a.center.x == b.center.x && a.center.y == b.center.y &&
            a.angle == b.angle && a.score == b.score,
            "CPU/Metal ordered sparse pose or score differs");
        for (std::size_t j = 0; j < a.corners.size(); ++j)
            require(a.corners[j].x == b.corners[j].x && a.corners[j].y == b.corners[j].y,
                "CPU/Metal sparse proposal corners differ");
    }
}

void sameWork(const SparseOrientationMetrics& cpu, const SparseOrientationMetrics& metal) {
    require(cpu.featureCount == metal.featureCount && cpu.angleCount == metal.angleCount,
        "Metal changed the feature model or angle bank");
    require(cpu.windows == metal.windows && cpu.featureEvaluations == metal.featureEvaluations &&
        cpu.thresholdSurvivors == metal.thresholdSurvivors,
        "CPU/Metal searched windows, early-bound work or survivors differ");
}

void routes(const SparseOrientationResult& cpu, const SparseOrientationResult& metal) {
    require(cpu.metrics.gradientMode == 0 && cpu.metrics.gradientLookupBytes == 0,
        "CPU reference unexpectedly used a gradient lookup");
    require(metal.metrics.gradientMode == 1 && metal.metrics.gradientLookupBytes ==
        (metal.metrics.featureCount > 0 ? 261121u : 0u),
        "Metal did not report its effective gradient lookup route");
    require(cpu.metrics.selectedBackend == 0 && cpu.metrics.gpuBufferBytes == 0 &&
        cpu.metrics.commandBuffers == 0 && cpu.metrics.cpuWaits == 0,
        "CPU sparse route reported GPU allocations or execution");
    require(cpu.metrics.responseMode == 1 && metal.metrics.responseMode == 1,
        "Backend parity test did not use compact responses");
    if (metal.metrics.featureCount > 0 && metal.metrics.windows > 0) {
        require(metal.metrics.selectedBackend == 1, "Explicit Metal sparse route silently fell back");
        require(metal.metrics.gpuBufferBytes > 0 && metal.metrics.commandBuffers > 0 &&
            metal.metrics.cpuWaits > 0 && metal.metrics.cpuWaits <= metal.metrics.commandBuffers,
            "Active Metal sparse route did not report valid GPU execution counters");
    }
}

SparseOrientationSettings defaults() {
    SparseOrientationSettings settings;
    settings.featureCount = 64;
    settings.positionStride = 1;
    settings.angleStepDegrees = 90.;
    settings.spreadRadius = 0;
    settings.minScore = .96;
    settings.candidateCap = 32;
    settings.cpuThreads = 1;
    settings.responseMode = 1;
    return settings;
}

SparseOrientationResult compare(ImageView source, ImageView templ,
    SparseOrientationSettings settings, const MatchConfig& config = {}) {
    settings.searchBackend = 0;
    SparseOrientationPattern cpuPattern(templ, settings);
    const auto cpu = cpuPattern.search(source, config);
    settings.searchBackend = 1;
    SparseOrientationPattern metalPattern(templ, settings);
    const auto metal = metalPattern.search(source, config);
    sameMatches(cpu.proposals, metal.proposals);
    sameWork(cpu.metrics, metal.metrics);
    routes(cpu, metal);
    ++comparisons;
    return metal;
}

bool contains(const std::vector<Match>& matches, double x, double y, double angle) {
    return std::any_of(matches.begin(), matches.end(), [&](const Match& match) {
        return std::hypot(match.center.x - x, match.center.y - y) < .1 &&
            std::abs(std::remainder(match.angle - angle, 360.)) < .1;
    });
}

void rotationAndStrideContracts() {
    const auto templ = texture();
    Image source(128, 112);
    fill(source, 128);
    paste(source, templ, 12, 10);
    paste(source, templ, 96, 30, -1);
    paste(source, templ, 86, 86, 2);
    auto settings = defaults();
    MatchConfig config;
    config.toleranceAngle = 180.;
    config.maxTargets = 1; // Raw proposal budget is independent of final NCC targets.
    const auto all = compare(source.view(), templ.view(), settings, config);
    require(contains(all.proposals, 12, 10, 0) && contains(all.proposals, 96, 30, -90) &&
        contains(all.proposals, 86, 86, 180), "Metal missed a known boundary or quarter-turn pose");
    require(all.proposals.size() > std::size_t(config.maxTargets),
        "Final NCC target limit truncated Metal raw proposals");

    config.angleRanges = {{-90., -90.}};
    require(contains(compare(source.view(), templ.view(), settings, config).proposals, 96, 30, -90),
        "Metal missed a single negative-angle range");
    config.angleRanges = {{170., 180.}, {-180., -170.}};
    require(contains(compare(source.view(), templ.view(), settings, config).proposals, 86, 86, 180),
        "Metal missed a wrapped range endpoint");
    config.angleRanges = {{-37.5, 42.5}, {170., 180.}, {-180., -170.}};
    settings.angleStepDegrees = 17.;
    compare(source.view(), templ.view(), settings, config);

    const auto wide = texture(32, 20);
    Image narrow(24, 40);
    fill(narrow, 128);
    paste(narrow, wide, 12, 20, 1);
    config.angleRanges = {{90., 90.}};
    require(contains(compare(narrow.view(), wide.view(), defaults(), config).proposals, 12, 20, 90),
        "Metal rejected a rotated template that fits a narrow source");

    // The maximum coordinate must be included even when not on the stride grid.
    Image edges(83, 71);
    fill(edges, 128);
    paste(edges, templ, 71, 61);
    config.angleRanges = {{0., 0.}};
    for (int stride : {4, 7, 64}) {
        settings = defaults(); settings.positionStride = stride;
        const auto result = compare(edges.view(), templ.view(), settings, config);
        require(contains(result.proposals, 71, 61, 0), "Metal omitted the nondivisible right/bottom edge");
    }

    // Odd template dimensions exercise half-pixel feature coordinates. Poison
    // padding in both input views must never enter the Metal gradient stencil.
    const auto oddTemplate = texture(25, 21);
    const auto oddSource = texture(83, 71);
    settings = defaults(); settings.spreadRadius = 2; settings.minScore = .7;
    config.angleRanges = {{-33.5, 42.5}}; settings.angleStepDegrees = 19.;
    const auto packed = compare(oddSource.view(), oddTemplate.view(), settings, config);
    const std::size_t sourceStride = oddSource.width() + 17;
    const std::size_t templateStride = oddTemplate.width() + 11;
    std::vector<std::uint8_t> sourceStorage(sourceStride * oddSource.height(), 255);
    std::vector<std::uint8_t> templateStorage(templateStride * oddTemplate.height(), 0);
    for (int y = 0; y < oddSource.height(); ++y)
        std::copy(oddSource.row(y), oddSource.row(y) + oddSource.width(), sourceStorage.data() + y * sourceStride);
    for (int y = 0; y < oddTemplate.height(); ++y)
        std::copy(oddTemplate.row(y), oddTemplate.row(y) + oddTemplate.width(), templateStorage.data() + y * templateStride);
    const auto padded = compare({sourceStorage.data(), oddSource.width(), oddSource.height(), sourceStride},
        {templateStorage.data(), oddTemplate.width(), oddTemplate.height(), templateStride}, settings, config);
    sameMatches(packed.proposals, padded.proposals);
    sameWork(packed.metrics, padded.metrics);
}

void parametersAndThresholds() {
    const auto templ = texture();
    Image source(83, 71);
    std::uint32_t state = 0x7f561ab3u;
    for (int y = 0; y < source.height(); ++y)
        for (int x = 0; x < source.width(); ++x) {
            state = state * 1664525u + 1013904223u;
            source.row(y)[x] = x < 38 ? std::uint8_t(state >> 24) :
                std::uint8_t(125 + ((state >> 24) % 7));
        }
    paste(source, templ, 60, 20, -1);
    paste(source, templ, 60, 52, 2);
    int index = 0;
    for (int features : {17, 64, 128})
        for (int stride : {1, 4})
            for (int spread : {0, 2}) {
                auto settings = defaults();
                settings.featureCount = features; settings.positionStride = stride;
                settings.spreadRadius = spread; settings.cpuThreads = index % 2 ? 3 : 1;
                settings.candidateCap = index % 3 ? 32 : 7;
                settings.minScore = index % 2 ? .96 : .74;
                MatchConfig config;
                config.toleranceAngle = 180.;
                if (index % 3 == 0) {
                    config.angleRanges = {{-37.5, 42.5}, {170., 180.}, {-180., -170.}};
                    settings.angleStepDegrees = 17.;
                }
                compare(source.view(), templ.view(), settings, config);
                ++index;
            }

    // Nontrivial quantized score neighbors detect an incorrect float threshold
    // or GPU early-bound rejection while retaining the CPU integer contract.
    auto settings = defaults(); settings.featureCount = 17; settings.minScore = .5;
    const auto clutter = texture(65, 57);
    const auto initial = compare(clutter.view(), templ.view(), settings);
    require(!initial.proposals.empty(), "Score boundary fixture has no proposal");
    const double score = initial.proposals.front().score;
    require(score > 0 && score < 1., "Score boundary fixture is not a nonunit score");
    for (double threshold : {std::nextafter(score, 0.), score, std::nextafter(score, 1.), 1.}) {
        settings.minScore = threshold;
        compare(clutter.view(), templ.view(), settings);
    }

    const auto low = texture(24, 20, true);
    Image lowSource(73, 63);
    fill(lowSource, 128); paste(lowSource, low, 36, 30);
    for (double threshold : {.5, std::nextafter(2.5, 0.), 2.5,
            std::nextafter(2.5, 3.), 12., 1000.}) {
        settings = defaults(); settings.minGradient = threshold;
        const auto result = compare(lowSource.view(), low.view(), settings);
        if (threshold == .5)
            require(contains(result.proposals, 36, 30, 0), "Low-contrast Metal fixture lost its known pose");
        if (threshold >= 12.) require(result.proposals.empty(), "Below-threshold gradients generated proposals");
    }
}

void repeatedFramesAndTies() {
    const auto templ = texture();
    Image source(128, 112);
    fill(source, 128);
    for (int y : {22, 78}) for (int x : {22, 62, 102}) paste(source, templ, x, y);
    auto settings = defaults(); settings.candidateCap = 3;
    settings.searchBackend = 0;
    SparseOrientationPattern cpu(templ.view(), settings);
    settings.searchBackend = 1;
    SparseOrientationPattern metal(templ.view(), settings);
    const auto first = cpu.search(source.view());
    require(first.proposals.size() == 3, "Tie fixture did not reach the candidate cap");
    require(first.proposals.front().score == first.proposals.back().score,
        "Candidate cap fixture did not exercise tied scores");
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto actual = metal.search(source.view());
        sameMatches(first.proposals, actual.proposals); sameWork(first.metrics, actual.metrics);
        routes(first, actual); ++comparisons;
    }
    // Identical addresses, different pixel contents: a cached GPU source must
    // be refreshed. Unsigned orientations must also preserve contrast polarity.
    for (int y = 0; y < source.height(); ++y)
        for (int x = 0; x < source.width(); ++x) source.row(y)[x] = std::uint8_t(255 - source.row(y)[x]);
    auto expected = cpu.search(source.view()), actual = metal.search(source.view());
    sameMatches(first.proposals, expected.proposals); sameWork(first.metrics, expected.metrics);
    sameMatches(expected.proposals, actual.proposals); sameWork(expected.metrics, actual.metrics);
    routes(expected, actual); ++comparisons;
    for (int y = 0; y < source.height(); ++y)
        std::reverse(source.row(y), source.row(y) + source.width());
    expected = cpu.search(source.view()); actual = metal.search(source.view());
    sameMatches(expected.proposals, actual.proposals); sameWork(expected.metrics, actual.metrics);
    routes(expected, actual); ++comparisons;
    fill(source, 128);
    expected = cpu.search(source.view()); actual = metal.search(source.view());
    require(actual.proposals.empty(), "Metal reused stale proposals after source became flat");
    sameMatches(expected.proposals, actual.proposals); sameWork(expected.metrics, actual.metrics);
    routes(expected, actual); ++comparisons;
}

void finiteGradientDomain() {
    // Every gray8 central difference is an integer pair in [-255,255]^2.
    // Independent 3x3 cells cover all 261121 pairs, at the sampled locations of
    // a single-feature 3x3 template. Eight template directions exercise every
    // unsigned bin. Compare observable peaks AND uncapped threshold survivors;
    // this does not claim direct access to all intermediate GPU response pixels.
    constexpr int values = 511, width = 3 * values + 1;
    Image source(width, width); fill(source, 128);
    auto cross = [](Image& image, int x, int y, int dx, int dy) {
        image.row(y)[x - 1] = std::uint8_t(std::max(0, -dx));
        image.row(y)[x + 1] = std::uint8_t(std::max(0, dx));
        image.row(y - 1)[x] = std::uint8_t(std::max(0, -dy));
        image.row(y + 1)[x] = std::uint8_t(std::max(0, dy));
    };
    for (int dy = -255; dy <= 255; ++dy)
        for (int dx = -255; dx <= 255; ++dx)
            cross(source, 1 + 3 * (dx + 255), 1 + 3 * (dy + 255), dx, dy);
    const std::array<std::array<int, 2>, 8> directions{{
        {{200, 0}}, {{200, 83}}, {{200, 200}}, {{83, 200}},
        {{0, 200}}, {{-83, 200}}, {{-200, 200}}, {{-200, 83}}
    }};
    std::uint64_t survivors = 0;
    for (const auto& direction : directions) {
        Image templ(3, 3); fill(templ, 128);
        cross(templ, 1, 1, direction[0], direction[1]);
        auto settings = defaults(); settings.featureCount = 1;
        settings.positionStride = 3; settings.minGradient = .25;
        settings.minScore = 1.; settings.candidateCap = 64;
        const auto result = compare(source.view(), templ.view(), settings);
        require(result.metrics.featureCount == 1 && result.metrics.angleCount == 1 &&
            result.metrics.windows == std::uint64_t(values) * values &&
            result.metrics.featureEvaluations == result.metrics.windows,
            "Finite-gradient fixture did not sample every signed difference exactly once");
        survivors += result.metrics.thresholdSurvivors;
    }
    require(survivors == std::uint64_t(values) * values - 1,
        "Gradient corpus was not partitioned into eight nonzero orientation bins");
}

void emptyAndInvalidContracts() {
    const auto templ = texture();
    Image source(65, 57), flat(24, 20), tiny(2, 2), tooSmall(8, 8);
    fill(source, 128); fill(flat, 128); fill(tiny, 128); fill(tooSmall, 128);
    auto settings = defaults();
    require(compare(source.view(), flat.view(), settings).proposals.empty(), "Flat Metal template generated proposals");
    require(compare(source.view(), tiny.view(), settings).proposals.empty(), "Tiny Metal template generated proposals");
    require(compare(tooSmall.view(), templ.view(), settings).proposals.empty(), "Metal searched a template that cannot fit");
    for (int backend : {-1, 2, std::numeric_limits<int>::max()}) {
        settings.searchBackend = backend;
        invalid([&] { SparseOrientationPattern bad(templ.view(), settings); },
            "Invalid sparse search backend was accepted");
    }
    settings = defaults(); settings.searchBackend = 1; settings.responseMode = 0;
    invalid([&] { SparseOrientationPattern bad(templ.view(), settings); },
        "Metal search silently accepted incompatible response planes");
    settings.responseMode = 1;
    SparseOrientationPattern metal(templ.view(), settings);
    MatchConfig config; config.angleRanges = {{20., -20.}};
    invalid([&] { metal.search(source.view(), config); }, "Metal accepted a reversed angle interval");
    invalid([&] { metal.search({source.row(0), source.width(), source.height(), 1}); },
        "Metal accepted an invalid source stride");
}

void capacityAndSharingContracts() {
    Image templ(3,3);fill(templ,128);templ.row(1)[0]=0;templ.row(1)[2]=255;
    Image many(330,330);fill(many,128);
    for(int y=4;y<326;y+=6) for(int x=4;x<326;x+=6) {
        many.row(y)[x-1]=0;many.row(y)[x+1]=255;
    }
    auto settings=defaults();settings.featureCount=1;settings.minScore=1.;settings.candidateCap=50;
    auto result=compare(many.view(),templ.view(),settings);
    require(result.metrics.peakOverflowRetries>0,"Peak overflow fixture did not exercise exact retry");

    auto model=texture();settings=defaults();settings.positionStride=4;
    SparseOrientationPattern cpu(model.view(),settings);settings.searchBackend=1;
    SparseOrientationPattern gpu(model.view(),settings);
    Image small(65,57),large(200,170);fill(small,128);fill(large,128);
    paste(small,model,32,28);paste(large,model,100,80);
    const auto a=cpu.search(small.view()),b=cpu.search(large.view());
    for(auto source:{small.view(),large.view(),small.view()}) {
        const auto expected=cpu.search(source),actual=gpu.search(source);
        sameMatches(expected.proposals,actual.proposals);sameWork(expected.metrics,actual.metrics);++comparisons;
    }
    auto fa=std::async(std::launch::async,[&]{return gpu.search(small.view());});
    auto fb=std::async(std::launch::async,[&]{return gpu.search(large.view());});
    const auto ga=fa.get(),gb=fb.get();
    sameMatches(a.proposals,ga.proposals);sameWork(a.metrics,ga.metrics);
    sameMatches(b.proposals,gb.proposals);sameWork(b.metrics,gb.metrics);comparisons+=2;

    settings=defaults();settings.positionStride=4;settings.angleStepDegrees=4;
    MatchConfig angles;angles.toleranceAngle=180;
    result=compare(small.view(),model.view(),settings,angles);
    require(result.metrics.angleCount>64 && result.metrics.commandBuffers>=3,"More than 64 angles did not split batches");
    // 3x3 support at stride1 gives 2049^2 windows, exceeding the explicit
    // 4M single-angle limit. Rejection must occur before score allocation.
    Image oversized(2052,2052);fill(oversized,128);
    settings=defaults();settings.featureCount=1;settings.searchBackend=1;
    SparseOrientationPattern limited(templ.view(),settings);
    invalid([&]{limited.search(oversized.view());},"Oversized single-angle grid was not rejected");
}
}

int main() {
    try {
        rotationAndStrideContracts();
        parametersAndThresholds();
        repeatedFramesAndTies();
        finiteGradientDomain();
        emptyAndInvalidContracts();
        capacityAndSharingContracts();
        std::cout << "Sparse Metal: " << comparisons << " exact CPU proposal/work parity cases; "
            << "rotation endpoints, stride edges, odd/padded views, tied caps, mutable frames, "
            << "score/gradient boundaries, all 261121 gray8 difference pairs, and invalid routes passed\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        // A library compilation, shader validation or execution error is a
        // failure. Only the precise device-unavailable condition may skip.
        if (message == "Metal device is unavailable") {
            std::cout << "SKIP: " << message << '\n'; return 77;
        }
        std::cerr << message << '\n'; return 1;
    }
}
