// SPDX-License-Identifier: BSD-2-Clause
// The original CPU arithmetic is the exact regression oracle for the retained
// gray8 gradient lookup. Independent pixel fixtures also check observable bin
// populations, known poses and threshold boundaries; no GPU is needed.
#include "sparse_orientation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace fpm;
using namespace fpm::research;
constexpr std::uint64_t gradientTableBytes = 511u * 511u;
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

void fill(Image& image, std::uint8_t value = 128) {
    for (int y = 0; y < image.height(); ++y)
        std::fill(image.row(y), image.row(y) + image.width(), value);
}

Image texture(int width = 24, int height = 20) {
    Image image(width, height);
    fill(image);
    std::uint32_t state = 0x43c629b1u;
    for (int y = 2; y < height - 2; ++y)
        for (int x = 2; x < width - 2; ++x) {
            state = state * 1664525u + 1013904223u;
            image.row(y)[x] = std::uint8_t(32 + ((state >> 24) % 192));
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

SparseOrientationSettings defaults(int responseMode) {
    SparseOrientationSettings settings;
    settings.featureCount = 64;
    settings.positionStride = 1;
    settings.angleStepDegrees = 90.;
    settings.spreadRadius = 0;
    settings.minScore = .96;
    settings.candidateCap = 32;
    settings.cpuThreads = 1;
    settings.responseMode = responseMode;
    return settings;
}

void sameResult(const SparseOrientationResult& expected, const SparseOrientationResult& actual) {
    require(expected.proposals.size() == actual.proposals.size(), "Gradient modes changed proposal count");
    for (std::size_t i = 0; i < expected.proposals.size(); ++i) {
        const auto& a = expected.proposals[i]; const auto& b = actual.proposals[i];
        require(a.center.x == b.center.x && a.center.y == b.center.y &&
            a.angle == b.angle && a.score == b.score,
            "Gradient modes changed ordered proposal pose or score");
        for (std::size_t j = 0; j < a.corners.size(); ++j)
            require(a.corners[j].x == b.corners[j].x && a.corners[j].y == b.corners[j].y,
                "Gradient modes changed proposal corners");
    }
    const auto& a = expected.metrics; const auto& b = actual.metrics;
    require(a.featureCount == b.featureCount && a.angleCount == b.angleCount &&
        a.windows == b.windows && a.featureEvaluations == b.featureEvaluations &&
        a.thresholdSurvivors == b.thresholdSurvivors,
        "Gradient modes changed model, angle bank, searched windows or exact-bound work");
}

void cpuMetrics(const SparseOrientationMetrics& metrics, int mode) {
    require(metrics.gradientMode == mode && metrics.selectedBackend == 0,
        "CPU gradient mode or backend was reported incorrectly");
    require(metrics.gpuBufferBytes == 0 && metrics.commandBuffers == 0 &&
        metrics.dispatches == 0 && metrics.cpuWaits == 0 && metrics.peakOverflowRetries == 0 &&
        metrics.gpuSetupMs == 0 && metrics.uploadMs == 0 && metrics.encodeMs == 0 &&
        metrics.waitMs == 0 && metrics.deviceMs == 0 && metrics.readbackMs == 0,
        "CPU gradient path reported GPU allocations or execution");
    const auto retained = mode == 1 && metrics.featureCount > 0 ? gradientTableBytes : 0;
    require(metrics.gradientLookupBytes == retained, "Retained CPU gradient lookup bytes differ");
}

void compareResults(const SparseOrientationResult& reference, const SparseOrientationResult& lookup,
    ImageView source, int responseMode) {
    sameResult(reference, lookup);
    cpuMetrics(reference.metrics, 0); cpuMetrics(lookup.metrics, 1);
    const auto& a = reference.metrics; const auto& b = lookup.metrics;
    require(a.responseMode == responseMode && b.responseMode == responseMode,
        "CPU gradient test used a different response layout");
    const auto logicalBytes = a.featureCount > 0 ?
        std::uint64_t(source.width) * source.height * (responseMode == 0 ? 8 : 1) : 0;
    require(a.sourceBytes == logicalBytes && b.sourceBytes == logicalBytes,
        "CPU gradient mode changed logical source-response storage");
    require(b.templateBytes == a.templateBytes + b.gradientLookupBytes &&
        a.workingBytes == b.workingBytes && a.lookupBytes == b.lookupBytes,
        "CPU retained lookup was not accounted exactly once as model memory");
    ++comparisons;
}

SparseOrientationResult compare(ImageView source, ImageView templ,
    SparseOrientationSettings settings, const MatchConfig& config = {}) {
    settings.searchBackend = 0; settings.cpuGradientMode = 0;
    const SparseOrientationPattern reference(templ, settings);
    settings.cpuGradientMode = 1;
    const SparseOrientationPattern lookup(templ, settings);
    const auto a = reference.search(source, config), b = lookup.search(source, config);
    require(a.metrics.templateBytes == reference.templateBytes() &&
        b.metrics.templateBytes == lookup.templateBytes(), "Model and search memory accounting differ");
    compareResults(a, b, source, settings.responseMode);
    return b;
}

bool contains(const std::vector<Match>& matches, double x, double y, double angle) {
    return std::any_of(matches.begin(), matches.end(), [&](const Match& match) {
        return std::hypot(match.center.x - x, match.center.y - y) < .1 &&
            std::abs(std::remainder(match.angle - angle, 360.)) < .1;
    });
}

void rotationStrideAndPadding(int responseMode) {
    const auto templ = texture();
    Image source(128, 112); fill(source);
    paste(source, templ, 12, 10);
    paste(source, templ, 96, 30, -1);
    paste(source, templ, 86, 86, 2);
    auto settings = defaults(responseMode);
    MatchConfig config; config.toleranceAngle = 180.;
    const auto result = compare(source.view(), templ.view(), settings, config);
    require(contains(result.proposals, 12, 10, 0) && contains(result.proposals, 96, 30, -90) &&
        contains(result.proposals, 86, 86, 180), "CPU lookup lost a known boundary/rotation pose");
    config.angleRanges = {{170., 180.}, {-180., -170.}};
    require(contains(compare(source.view(), templ.view(), settings, config).proposals, 86, 86, 180),
        "CPU lookup lost a wrapped angle endpoint");
    settings.featureCount = 17; settings.spreadRadius = 2; settings.cpuThreads = 3;
    settings.minScore = .74; settings.angleStepDegrees = 17.;
    config.angleRanges = {{-37.5, 42.5}};
    const auto tail = compare(source.view(), templ.view(), settings, config);
    require(tail.metrics.featureCount == 17, "Feature-tail fixture did not exercise 17 features");

    Image edge(83, 71); fill(edge); paste(edge, templ, 71, 61);
    settings = defaults(responseMode); settings.positionStride = 7;
    require(contains(compare(edge.view(), templ.view(), settings).proposals, 71, 61, 0),
        "CPU lookup omitted a nondivisible last row or column");

    const auto oddTemplate = texture(25, 21), oddSource = texture(83, 71);
    settings.featureCount = 17; settings.spreadRadius = 2; settings.minScore = .7;
    config.angleRanges = {{-33.5, 42.5}}; settings.angleStepDegrees = 19.;
    const auto packed = compare(oddSource.view(), oddTemplate.view(), settings, config);
    const std::size_t sourceStride = oddSource.width() + 17, templateStride = oddTemplate.width() + 11;
    std::vector<std::uint8_t> sourceStorage(sourceStride * oddSource.height(), 255);
    std::vector<std::uint8_t> templateStorage(templateStride * oddTemplate.height(), 0);
    for (int y = 0; y < oddSource.height(); ++y)
        std::copy(oddSource.row(y), oddSource.row(y) + oddSource.width(), sourceStorage.data() + y * sourceStride);
    for (int y = 0; y < oddTemplate.height(); ++y)
        std::copy(oddTemplate.row(y), oddTemplate.row(y) + oddTemplate.width(), templateStorage.data() + y * templateStride);
    const auto padded = compare({sourceStorage.data(), oddSource.width(), oddSource.height(), sourceStride},
        {templateStorage.data(), oddTemplate.width(), oddTemplate.height(), templateStride}, settings, config);
    sameResult(packed, padded);
}

void cross(Image& image, int x, int y, int dx, int dy) {
    image.row(y)[x - 1] = std::uint8_t(std::max(0, -dx));
    image.row(y)[x + 1] = std::uint8_t(std::max(0, dx));
    image.row(y - 1)[x] = std::uint8_t(std::max(0, -dy));
    image.row(y + 1)[x] = std::uint8_t(std::max(0, dy));
}

void thresholdBoundaries(int responseMode) {
    Image templ(3, 3), source(4, 4); fill(templ); fill(source);
    cross(templ, 1, 1, 30, 40); cross(source, 1, 1, 3, 4);
    auto settings = defaults(responseMode); settings.featureCount = 1; settings.minScore = 1.;
    for (double threshold : {std::nextafter(2.5, 0.), 2.5, std::nextafter(2.5, 3.)}) {
        settings.minGradient = threshold;
        const auto result = compare(source.view(), templ.view(), settings);
        require(result.metrics.windows == 1 && result.metrics.featureEvaluations == 1,
            "Gradient boundary fixture did not isolate one source sample");
        require(result.metrics.thresholdSurvivors == std::uint64_t(threshold <= 2.5) &&
            result.proposals.size() == std::size_t(threshold <= 2.5),
            "Exact gradient-magnitude boundary changed inclusion");
    }

    const auto clutter = texture(65, 57), texturedTemplate = texture();
    settings = defaults(responseMode); settings.featureCount = 17; settings.minScore = .5;
    const auto first = compare(clutter.view(), texturedTemplate.view(), settings);
    require(!first.proposals.empty(), "Quantized score-boundary fixture has no proposal");
    const auto score = first.proposals.front().score;
    require(score > 0 && score < 1., "Quantized score-boundary fixture has a unit score");
    for (double threshold : {std::nextafter(score, 0.), score, std::nextafter(score, 1.)}) {
        settings.minScore = threshold;
        compare(clutter.view(), texturedTemplate.view(), settings);
    }
}

void finiteGradientDomain() {
    // Independent 3x3 crosses expose every signed gray8 difference exactly once
    // on a stride-three grid. The oracle chooses the closest unsigned unit
    // direction by absolute dot product, without atan2/modulo/bin rounding.
    // Each bin's uncapped survivor count is checked; selected peaks and work
    // are also exactly compared, without claiming access to intermediate maps.
    constexpr int values = 511, width = 3 * values + 1;
    Image source(width, width); fill(source);
    std::array<std::array<long double, 2>, 8> unit{};
    const long double pi = std::acos(-1.L);
    for (int bin = 0; bin < 8; ++bin)
        unit[bin] = {std::cos(bin * pi / 8), std::sin(bin * pi / 8)};
    std::array<std::uint64_t, 8> expected{};
    for (int dy = -255; dy <= 255; ++dy)
        for (int dx = -255; dx <= 255; ++dx) {
            cross(source, 1 + 3 * (dx + 255), 1 + 3 * (dy + 255), dx, dy);
            if (dx == 0 && dy == 0) continue;
            int best = 0; long double projection = -1;
            for (int bin = 0; bin < 8; ++bin) {
                const auto dot = std::abs(dx * unit[bin][0] + dy * unit[bin][1]);
                if (dot > projection) { projection = dot; best = bin; }
            }
            ++expected[best];
        }
    const std::array<std::array<int, 2>, 8> directions{{
        {{200, 0}}, {{200, 83}}, {{200, 200}}, {{83, 200}},
        {{0, 200}}, {{-83, 200}}, {{-200, 200}}, {{-200, 83}}
    }};
    for (int responseMode : {0, 1}) {
        std::uint64_t total = 0;
        for (int bin = 0; bin < 8; ++bin) {
            Image templ(3, 3); fill(templ);
            cross(templ, 1, 1, directions[bin][0], directions[bin][1]);
            auto settings = defaults(responseMode); settings.featureCount = 1;
            settings.positionStride = 3; settings.minGradient = .25;
            settings.minScore = 1.; settings.candidateCap = 64;
            const auto result = compare(source.view(), templ.view(), settings);
            require(result.metrics.featureCount == 1 && result.metrics.angleCount == 1 &&
                result.metrics.windows == gradientTableBytes &&
                result.metrics.featureEvaluations == gradientTableBytes,
                "Finite-gradient corpus did not sample every signed difference exactly once");
            require(result.metrics.thresholdSurvivors == expected[bin],
                "Gradient-bin population disagrees with independent nearest-direction oracle");
            total += result.metrics.thresholdSurvivors;
        }
        require(total == gradientTableBytes - 1, "Gradient corpus did not partition every nonzero difference");
    }
}

void mutableFramesAndSharing(int responseMode) {
    const auto templ = texture();
    auto settings = defaults(responseMode); settings.candidateCap = 3;
    settings.cpuGradientMode = 0; const SparseOrientationPattern reference(templ.view(), settings);
    settings.cpuGradientMode = 1; const SparseOrientationPattern lookup(templ.view(), settings);
    Image source(128, 112); fill(source);
    for (int y : {22, 78}) for (int x : {22, 62, 102}) paste(source, templ, x, y);
    const auto first = reference.search(source.view());
    require(first.proposals.size() == 3 && first.proposals.front().score == first.proposals.back().score,
        "Candidate-cap fixture did not exercise tied scores");
    auto check = [&](ImageView view) {
        const auto a = reference.search(view), b = lookup.search(view);
        compareResults(a, b, view, responseMode);
        return b;
    };
    check(source.view());
    for (int y = 0; y < source.height(); ++y)
        for (int x = 0; x < source.width(); ++x) source.row(y)[x] = std::uint8_t(255 - source.row(y)[x]);
    sameResult(first, check(source.view()));
    fill(source);
    require(check(source.view()).proposals.empty(), "CPU lookup reused stale pixels or proposals");
    Image small(65, 57), large(200, 170); fill(small); fill(large);
    paste(small, templ, 32, 28); paste(large, templ, 100, 80);
    for (auto view : {small.view(), large.view(), small.view()}) check(view);
    const auto expectedSmall = reference.search(small.view()), expectedLarge = reference.search(large.view());
    auto a = std::async(std::launch::async, [&] { return lookup.search(small.view()); });
    auto b = std::async(std::launch::async, [&] { return lookup.search(large.view()); });
    compareResults(expectedSmall, a.get(), small.view(), responseMode);
    compareResults(expectedLarge, b.get(), large.view(), responseMode);
}

void emptyAndInvalid(int responseMode) {
    const auto templ = texture();
    Image source(65, 57), flat(24, 20); fill(source); fill(flat);
    auto settings = defaults(responseMode);
    require(compare(source.view(), flat.view(), settings).proposals.empty(), "Flat template produced proposals");
    for (const auto dimensions : {std::array<int, 2>{1, 1}, {2, 2}, {2, 5}, {5, 1}}) {
        Image tiny(dimensions[0], dimensions[1]); fill(tiny);
        require(compare(source.view(), tiny.view(), settings).proposals.empty(), "Tiny template produced proposals");
        require(compare(tiny.view(), templ.view(), settings).proposals.empty(), "Tiny source produced proposals");
    }
    settings.minGradient = 1000.;
    require(compare(source.view(), templ.view(), settings).proposals.empty(), "Above-range threshold produced a model");
    settings = defaults(responseMode);
    for (int mode : {-1, 2, std::numeric_limits<int>::max()}) {
        settings.cpuGradientMode = mode;
        invalid([&] { SparseOrientationPattern bad(templ.view(), settings); },
            "Invalid CPU gradient mode was accepted");
        invalid([&] { SparseOrientationPattern bad(flat.view(), settings); },
            "Empty model bypassed gradient-mode validation");
    }
    settings.cpuGradientMode = 1;
    SparseOrientationPattern lookup(templ.view(), settings);
    invalid([&] { lookup.search({source.row(0), source.width(), source.height(), 1}); },
        "CPU lookup accepted invalid source stride");
}
}

int main() {
    try {
        for (int responseMode : {0, 1}) {
            rotationStrideAndPadding(responseMode);
            thresholdBoundaries(responseMode);
            mutableFramesAndSharing(responseMode);
            emptyAndInvalid(responseMode);
        }
        finiteGradientDomain();
        std::cout << "CPU gradient lookup: " << comparisons << " exact proposal/work parity cases; "
            << "both response layouts, all 261121 gray8 difference pairs with independent bin counts, "
            << "threshold boundaries, padded/mutable/tiny inputs, shared-model searches and memory accounting passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
