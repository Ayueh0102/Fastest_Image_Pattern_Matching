// SPDX-License-Identifier: BSD-2-Clause
// The original eight-plane representation is the independent regression oracle
// for the one-byte orientation-mask representation. Integer quarter-turn pixels
// supply known poses without using the production warp or score functions.
#include "research_pipeline.hpp"
#include "sparse_orientation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

void fill(Image& image, std::uint8_t value) {
    for (int y = 0; y < image.height(); ++y)
        std::fill(image.row(y), image.row(y) + image.width(), value);
}

Image texture(int width = 24, int height = 20, bool thin = false) {
    Image image(width, height);
    fill(image, 128);
    std::uint32_t state = 0x43c629b1u;
    for (int y = 2; y < height - 2; ++y)
        for (int x = 2; x < width - 2; ++x) {
            state = state * 1664525u + 1013904223u;
            if (!thin) image.row(y)[x] = std::uint8_t(32 + ((state >> 24) % 192));
            else if (x == 4 || y == 5 || x + y == width - 3 ||
                (y == height - 4 && x > width / 2))
                image.row(y)[x] = std::uint8_t(24 + ((state >> 24) % 32));
        }
    return image;
}

void paste(Image& source, const Image& templ, int cx, int cy, int quarterTurns,
    bool inverted = false) {
    const int q = ((quarterTurns % 4) + 4) % 4;
    for (int y = 0; y < templ.height(); ++y)
        for (int x = 0; x < templ.width(); ++x) {
            const int dx = x - templ.width() / 2;
            const int dy = y - templ.height() / 2;
            const int rx = q == 0 ? dx : q == 1 ? dy : q == 2 ? -dx : -dy;
            const int ry = q == 0 ? dy : q == 1 ? -dx : q == 2 ? -dy : dx;
            require(cx + rx >= 0 && cx + rx < source.width() &&
                cy + ry >= 0 && cy + ry < source.height(), "Quarter-turn fixture is out of bounds");
            const auto pixel = templ.row(y)[x];
            source.row(cy + ry)[cx + rx] = inverted ? std::uint8_t(255 - pixel) : pixel;
        }
}

void sameMatches(const std::vector<Match>& reference, const std::vector<Match>& compact) {
    require(reference.size() == compact.size(), "Compact/reference output counts differ");
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const auto& a = reference[i]; const auto& b = compact[i];
        require(a.center.x == b.center.x && a.center.y == b.center.y &&
            a.angle == b.angle && a.score == b.score,
            "Compact/reference ordered pose or score differs");
        for (std::size_t j = 0; j < a.corners.size(); ++j)
            require(a.corners[j].x == b.corners[j].x && a.corners[j].y == b.corners[j].y,
                "Compact/reference corners differ");
    }
}

void sameWork(const SparseOrientationMetrics& a, const SparseOrientationMetrics& b) {
    require(a.featureCount == b.featureCount && a.angleCount == b.angleCount,
        "Compact mode changed the template features or angle bank");
    require(a.windows == b.windows && a.featureEvaluations == b.featureEvaluations &&
        a.thresholdSurvivors == b.thresholdSurvivors,
        "Compact mode changed searched windows, exact-bound work, or threshold survivors");
}

bool contains(const std::vector<Match>& matches, double x, double y, double angle,
    double positionTolerance = .1, double angleTolerance = .1) {
    return std::any_of(matches.begin(), matches.end(), [&](const Match& match) {
        return std::hypot(match.center.x - x, match.center.y - y) <= positionTolerance &&
            std::abs(std::remainder(match.angle - angle, 360.)) <= angleTolerance;
    });
}

SparseOrientationResult compare(ImageView source, ImageView templ,
    SparseOrientationSettings settings, const MatchConfig& config) {
    settings.responseMode = 0;
    SparseOrientationPattern referencePattern(templ, settings);
    const auto reference = referencePattern.search(source, config);
    settings.responseMode = 1;
    SparseOrientationPattern compactPattern(templ, settings);
    const auto compact = compactPattern.search(source, config);
    sameMatches(reference.proposals, compact.proposals);
    sameWork(reference.metrics, compact.metrics);
    require(reference.metrics.responseMode == 0 && compact.metrics.responseMode == 1,
        "Response representation was not reported correctly");
    const auto pixels = std::uint64_t(source.width) * source.height;
    if (reference.metrics.featureCount) {
        require(reference.metrics.sourceBytes == 8 * pixels && compact.metrics.sourceBytes == pixels,
            "Active response allocation is not eight bytes versus one byte per pixel");
        require(compact.metrics.lookupBytes == 2048,
            "Compact response lookup table is not reported as 2048 bytes");
    } else {
        require(reference.proposals.empty() && compact.proposals.empty() &&
            reference.metrics.sourceBytes == 0 && compact.metrics.sourceBytes == 0,
            "Empty feature model allocated or searched source responses");
    }
    ++comparisons;
    return compact;
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
    return settings;
}

void knownPoseAndBoundaryContracts() {
    const auto templ = texture();
    Image source(128, 112);
    fill(source, 128);
    paste(source, templ, 12, 10, 0);
    paste(source, templ, 96, 30, -1);
    paste(source, templ, 86, 86, 2);
    auto settings = defaults();
    MatchConfig config;
    config.toleranceAngle = 180.;
    const auto all = compare(source.view(), templ.view(), settings, config);
    require(contains(all.proposals, 12, 10, 0), "Compact route missed the source-boundary pose");
    require(contains(all.proposals, 96, 30, -90), "Compact route missed a negative rotation");
    require(contains(all.proposals, 86, 86, 180), "Compact route missed the wrapped half turn");
    auto threaded = settings;
    threaded.cpuThreads = 3;
    const auto parallel = compare(source.view(), templ.view(), threaded, config);
    sameMatches(all.proposals, parallel.proposals);
    sameWork(all.metrics, parallel.metrics);

    config.angleRanges = {{-90., -90.}};
    require(contains(compare(source.view(), templ.view(), settings, config).proposals, 96, 30, -90),
        "Compact route missed the restricted negative angle");
    config.angleRanges = {{170., 180.}, {-180., -170.}};
    require(contains(compare(source.view(), templ.view(), settings, config).proposals, 86, 86, 180),
        "Compact route missed the split wrap-around angle range");

    const auto wide = texture(32, 20);
    Image narrow(24, 40);
    fill(narrow, 128);
    paste(narrow, wide, 12, 20, 1);
    config.angleRanges = {{90., 90.}};
    require(contains(compare(narrow.view(), wide.view(), settings, config).proposals, 12, 20, 90),
        "Compact route rejected a rotated template that fits a narrow source");

    // Unused row padding must have no influence on the mask or response lookup.
    config.angleRanges.clear();
    const auto stride = std::size_t(source.width()) + 17;
    std::vector<std::uint8_t> padded(stride * source.height(), 255);
    for (int y = 0; y < source.height(); ++y)
        std::copy(source.row(y), source.row(y) + source.width(), padded.data() + std::size_t(y) * stride);
    const auto paddedResult = compare({padded.data(), source.width(), source.height(), stride},
        templ.view(), settings, config);
    sameMatches(all.proposals, paddedResult.proposals);
    sameWork(all.metrics, paddedResult.metrics);

    Image flat(24, 20), blank(128, 112), tooSmall(8, 8);
    fill(flat, 128); fill(blank, 128); fill(tooSmall, 128);
    require(compare(source.view(), flat.view(), settings, config).proposals.empty(),
        "Flat compact model generated proposals");
    require(compare(blank.view(), templ.view(), settings, config).proposals.empty(),
        "Flat compact source generated proposals");
    require(compare(tooSmall.view(), templ.view(), settings, config).proposals.empty(),
        "Compact route searched a source too small for every angle");
}

void parameterMatrix() {
    int index = 0;
    for (bool thin : {false, true}) {
        const auto templ = texture(24, 20, thin);
        Image source(83, 71);
        std::uint32_t state = thin ? 0x234be12u : 0x738ab41u;
        for (int y = 0; y < source.height(); ++y)
            for (int x = 0; x < source.width(); ++x) {
                state = state * 1664525u + 1013904223u;
                // Alternate strong clutter and low-gradient background. This
                // exercises both mask unions and early threshold rejection.
                source.row(y)[x] = x < 38 ? std::uint8_t(state >> 24) :
                    std::uint8_t(125 + ((state >> 24) % 7));
            }
        paste(source, templ, 60, 20, -1);
        paste(source, templ, 60, 52, 2);
        for (int features : {17, 64, 128})
            for (int stride : {1, 4})
                for (int spread : {0, 2, 4})
                    for (int threads : {1, 3}) {
                        auto settings = defaults();
                        settings.featureCount = features;
                        settings.positionStride = stride;
                        settings.spreadRadius = spread;
                        settings.cpuThreads = threads;
                        settings.candidateCap = index % 3 == 0 ? 7 : 32;
                        settings.minScore = index % 2 ? .96 : .74;
                        MatchConfig config;
                        if (index % 3 == 0) config.toleranceAngle = 180.;
                        else if (index % 3 == 1) config.angleRanges = {{-90., -90.}, {90., 90.}};
                        else {
                            config.angleRanges = {{-37.5, 42.5}, {170., 180.}, {-180., -170.}};
                            settings.angleStepDegrees = 17.;
                        }
                        compare(source.view(), templ.view(), settings, config);
                        ++index;
                    }
    }
    require(index == 72, "Parameter matrix lost coverage");
}

void scoreBoundaryContracts() {
    const auto templ = texture();
    Image source(65, 57);
    // An independently seeded clutter image gives nonunit sparse scores.
    // Testing both neighbors of one observed quantized score detects
    // rounding-dependent early exits without assuming a particular score.
    std::uint32_t state = 0x7f561ab3u;
    for (int y = 0; y < source.height(); ++y)
        for (int x = 0; x < source.width(); ++x) {
            state = state * 1664525u + 1013904223u;
            source.row(y)[x] = std::uint8_t(state >> 24);
        }
    auto settings = defaults();
    settings.featureCount = 17;
    settings.minScore = .5;
    MatchConfig config;
    const auto initial = compare(source.view(), templ.view(), settings, config);
    require(!initial.proposals.empty(), "Score-boundary fixture has no proposal");
    const double score = initial.proposals.front().score;
    require(score > 0 && score < 1., "Score-boundary fixture did not produce a nontrivial score");
    const double towardZero = std::nextafter(score, 0.);
    const double towardOne = std::nextafter(score, 1.);
    for (double threshold : {towardZero, score, towardOne}) {
        settings.minScore = threshold;
        compare(source.view(), templ.view(), settings, config);
    }
}

void pipelineContracts() {
    const auto templ = texture();
    Image source(128, 80);
    fill(source, 128);
    paste(source, templ, 30, 34, -1);
    paste(source, templ, 98, 34, -1, true);
    Settings settings;
    settings.backend = 0;
    settings.cpuThreads = 1;
    settings.scales = {1.};
    settings.sparseOrientation = true;
    settings.sparse = defaults();
    Options options;
    options.match.toleranceAngle = 180.;
    options.match.minScore = .97;
    options.match.maxTargets = 4;
    options.match.maxOverlap = .2;
    settings.sparse.responseMode = 0;
    Session referenceSession(source.view(), templ.view(), settings);
    const auto reference = referenceSession.run(options);
    settings.sparse.responseMode = 1;
    Session compactSession(source.view(), templ.view(), settings);
    const auto compact = compactSession.run(options);
    sameMatches(reference.proposals, compact.proposals);
    sameMatches(reference.matches, compact.matches);
    sameWork(reference.sparse, compact.sparse);
    require(reference.scales == compact.scales && reference.proposalScales == compact.proposalScales,
        "Compact pipeline changed scale outputs");
    require(compact.matches.size() == 1 && contains(compact.matches, 30, 34, -90, .35, .5),
        "Full NCC did not accept the real target and reject the inverted target");
    for (const auto* result : {&reference, &compact}) {
        require(result->sparseEnabled && result->frame.host_pyramid_bytes == 0 &&
            result->frame.models_matched == 0 && result->coarse.empty() && result->fine.empty(),
            "Sparse proposal route performed dense NCC or built a pyramid");
        require(result->rescue.selected_backend == 0, "Parity probe did not use portable CPU refinement");
        for (double scale : result->scales) require(scale == 1., "Compact refinement changed the fixed scale");
    }
    sameMatches(compact.matches, compactSession.run(options).matches);
    options.match.invertSource = true;
    const auto invertedReference = referenceSession.run(options);
    const auto invertedCompact = compactSession.run(options);
    sameMatches(invertedReference.proposals, invertedCompact.proposals);
    sameMatches(invertedReference.matches, invertedCompact.matches);
    require(invertedCompact.matches.size() == 1 &&
        contains(invertedCompact.matches, 98, 34, -90, .35, .5),
        "Compact pipeline reused stale full-NCC source pixels after inversion");
}

void invalidModeContracts() {
    const auto templ = texture();
    for (int mode : {-1, 2, std::numeric_limits<int>::max()}) {
        auto settings = defaults();
        settings.responseMode = mode;
        bool rejected = false;
        try { SparseOrientationPattern pattern(templ.view(), settings); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid response representation mode was accepted");
    }
}
}

int main() {
    try {
        knownPoseAndBoundaryContracts();
        parameterMatrix();
        scoreBoundaryContracts();
        pipelineContracts();
        invalidModeContracts();
        std::cout << "Compact sparse responses: " << comparisons
            << " exact proposal/work parity cases, eight-to-one active image bytes, "
            << "CPU NCC integration, fixed scale and invalid-mode rejection passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
