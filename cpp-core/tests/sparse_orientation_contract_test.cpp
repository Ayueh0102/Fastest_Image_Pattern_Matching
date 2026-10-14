// SPDX-License-Identifier: BSD-2-Clause
// Portable contracts for the experimental original-resolution proposal path.
// Quarter-turn fixtures copy integer pixels directly; no production warp or
// scoring helper is used to manufacture the expected poses.
#include "research_pipeline.hpp"
#include "sparse_orientation.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace fpm;
using namespace fpm::research;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void invalid(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

void fill(Image& image, uint8_t value) {
    for (int y = 0; y < image.height(); ++y)
        std::fill(image.row(y), image.row(y) + image.width(), value);
}

Image texture(int width = 24, int height = 20) {
    Image templ(width, height);
    fill(templ, 128);
    uint32_t state = 0x43c629b1u;
    for (int y = 2; y < templ.height() - 2; ++y)
        for (int x = 2; x < templ.width() - 2; ++x) {
            state = state * 1664525u + 1013904223u;
            templ.row(y)[x] = uint8_t(32 + ((state >> 24) % 192));
        }
    return templ;
}

// Positive angles rotate CCW in an x-right/y-down image. Using integer
// coordinates around width/2 avoids both interpolation and a half-pixel shift.
void paste(Image& source, const Image& templ, int cx, int cy,
    int quarterTurns, bool invert = false) {
    const int q = ((quarterTurns % 4) + 4) % 4;
    for (int y = 0; y < templ.height(); ++y)
        for (int x = 0; x < templ.width(); ++x) {
            const int dx = x - templ.width() / 2;
            const int dy = y - templ.height() / 2;
            const int rx = q == 0 ? dx : q == 1 ? dy : q == 2 ? -dx : -dy;
            const int ry = q == 0 ? dy : q == 1 ? -dx : q == 2 ? -dy : dx;
            require(cx + rx >= 0 && cx + rx < source.width() &&
                cy + ry >= 0 && cy + ry < source.height(), "Fixture does not fit");
            const auto value = templ.row(y)[x];
            source.row(cy + ry)[cx + rx] = invert ? uint8_t(255 - value) : value;
        }
}

double angleDistance(double a, double b) {
    return std::abs(std::remainder(a - b, 360.));
}

bool contains(const std::vector<Match>& matches, double x, double y,
    double angle, double positionTolerance = .1, double angleTolerance = .1) {
    return std::any_of(matches.begin(), matches.end(), [&](const Match& match) {
        return std::hypot(match.center.x - x, match.center.y - y) <= positionTolerance &&
            angleDistance(match.angle, angle) <= angleTolerance;
    });
}

void same(const std::vector<Match>& a, const std::vector<Match>& b) {
    require(a.size() == b.size(), "Repeated proposal count changed");
    for (size_t i = 0; i < a.size(); ++i) {
        require(a[i].center.x == b[i].center.x && a[i].center.y == b[i].center.y &&
            a[i].angle == b[i].angle && a[i].score == b[i].score,
            "Repeated ordered proposal pose or score changed");
        for (size_t j = 0; j < a[i].corners.size(); ++j)
            require(a[i].corners[j].x == b[i].corners[j].x &&
                a[i].corners[j].y == b[i].corners[j].y,
                "Repeated proposal corners changed");
    }
}

SparseOrientationSettings settings() {
    SparseOrientationSettings s;
    s.featureCount = 64;
    s.positionStride = 1;
    s.spreadRadius = 0;
    s.candidateCap = 32;
    s.cpuThreads = 1;
    s.angleStepDegrees = 90.;
    s.minGradient = 8.;
    s.minScore = .96;
    return s;
}

void proposalContracts() {
    auto templ = texture();
    Image source(128, 112);
    fill(source, 128);
    // The border instance also guards against silently treating the cropped
    // template as a search ROI. All locations are searched in the full image.
    paste(source, templ, 12, 10, 0);
    paste(source, templ, 96, 30, -1);
    paste(source, templ, 86, 86, 2);
    const auto s = settings();
    MatchConfig config;
    config.toleranceAngle = 180.;
    config.maxTargets = 1; // The proposal budget must remain independent.
    SparseOrientationPattern pattern(templ.view(), s);
    require(pattern.featureCount() > 0 && pattern.featureCount() <= s.featureCount,
        "Feature count is empty or exceeds its budget");
    const auto first = pattern.search(source.view(), config);
    require(contains(first.proposals, 12, 10, 0), "Top-left full-image proposal missing");
    require(contains(first.proposals, 96, 30, -90), "Negative quarter-turn proposal missing");
    require(contains(first.proposals, 86, 86, 180), "Wrapped 180-degree proposal missing");
    require(first.proposals.size() <= size_t(s.candidateCap), "Proposal cap exceeded");
    require(first.proposals.size() > size_t(config.maxTargets),
        "Final target limit prematurely truncated raw proposals");
    for (const auto& match : first.proposals)
        require(std::isfinite(match.score) && match.score >= 0 && match.score <= 1.,
            "Orientation score is outside its documented range");
    same(first.proposals, pattern.search(source.view(), config).proposals);

    auto parallel = s;
    parallel.cpuThreads = 2;
    SparseOrientationPattern threaded(templ.view(), parallel);
    same(first.proposals, threaded.search(source.view(), config).proposals);

    // Strided image views must ignore poison padding, and template learning
    // must own its data after construction.
    const size_t stride = size_t(source.width()) + 13;
    std::vector<uint8_t> padded(stride * source.height(), 255);
    for (int y = 0; y < source.height(); ++y)
        std::copy(source.row(y), source.row(y) + source.width(), padded.data() + size_t(y) * stride);
    same(first.proposals, pattern.search({padded.data(), source.width(), source.height(), stride}, config).proposals);
    fill(templ, 0);
    same(first.proposals, pattern.search(source.view(), config).proposals);

    auto capped = s;
    capped.candidateCap = 2;
    const auto cleanTemplate = texture();
    SparseOrientationPattern limited(cleanTemplate.view(), capped);
    const auto limitedResult = limited.search(source.view(), config);
    require(limitedResult.proposals.size() == 2, "Explicit proposal cap was not honored");
    same(limitedResult.proposals, limited.search(source.view(), config).proposals);

    config.angleRanges = {{-90., -90.}};
    const auto negativeOnly = pattern.search(source.view(), config);
    require(contains(negativeOnly.proposals, 96, 30, -90), "Restricted negative angle was skipped");
    for (const auto& match : negativeOnly.proposals)
        require(angleDistance(match.angle, -90.) < 1e-8, "Proposal escaped an explicit angle range");
    config.angleRanges = {{170., 180.}, {-180., -170.}};
    const auto wrap = pattern.search(source.view(), config);
    require(contains(wrap.proposals, 86, 86, 180), "Split wrap-around range missed its endpoint");
    for (const auto& match : wrap.proposals)
        require(std::abs(match.angle) >= 170. - 1e-8, "Proposal escaped the wrap-around angle range");

    // Unlike the legacy unrotated-size gate, rotation can make a wider
    // template fit a narrow source. This route must search that valid pose.
    const auto wide = texture(32, 20);
    Image narrowSource(24, 40);
    fill(narrowSource, 128);
    paste(narrowSource, wide, 12, 20, 1);
    SparseOrientationPattern widePattern(wide.view(), s);
    MatchConfig quarterOnly;
    quarterOnly.angleRanges = {{90., 90.}};
    require(contains(widePattern.search(narrowSource.view(), quarterOnly).proposals, 12, 20, 90),
        "Unrotated template size incorrectly rejected a fitting rotated pose");

    Image flat(24, 20), blank(128, 112), tooSmall(8, 8);
    fill(flat, 128); fill(blank, 128); fill(tooSmall, 128);
    SparseOrientationPattern flatPattern(flat.view(), s);
    require(flatPattern.featureCount() == 0 && flatPattern.search(source.view()).proposals.empty(),
        "A flat template produced orientation features or proposals");
    config.angleRanges.clear();
    require(pattern.search(blank.view(), config).proposals.empty(), "Flat source produced false proposals");
    require(pattern.search(tooSmall.view(), config).proposals.empty(), "Oversized template was searched");
    invalid([&] { SparseOrientationPattern bad({}, s); }, "Invalid template view accepted");
    auto badSettings = s; badSettings.positionStride = 0;
    invalid([&] { SparseOrientationPattern bad(cleanTemplate.view(), badSettings); }, "Zero position stride accepted");
    config.angleRanges = {{20., -20.}};
    invalid([&] { pattern.search(source.view(), config); }, "Reversed angle range accepted");
}

void pipelineContracts() {
    auto templ = texture();
    Image source(128, 80);
    fill(source, 128);
    paste(source, templ, 30, 34, -1);
    paste(source, templ, 98, 34, -1, true);
    Settings s;
    s.backend = 0;
    s.cpuThreads = 1;
    s.sparseOrientation = true;
    s.sparse = settings();
    s.scales = {1.};
    Options options;
    options.match.toleranceAngle = 180.;
    options.match.minScore = .97;
    options.match.maxTargets = 4;
    options.match.maxOverlap = .2;
    options.refine = true;
    Session session(source.view(), templ.view(), s);
    const auto first = session.run(options);
    require(first.sparseEnabled, "Session did not report its sparse route");
    require(contains(first.proposals, 30, 34, -90) && contains(first.proposals, 98, 34, -90),
        "Unsigned orientation proposal fixture did not include both contrast polarities");
    require(first.matches.size() == 1 && contains(first.matches, 30, 34, -90, .35, .5),
        "Full NCC did not accept the true target and reject contrast inversion");
    require(first.frame.host_pyramid_bytes == 0 && first.frame.models_matched == 0 &&
        first.coarse.empty() && first.fine.empty(), "Sparse route constructed or searched a pyramid");
    require(first.rescue.selected_backend == 0, "Portable probe did not use CPU refinement");
    require(first.scales.size() == first.matches.size() && first.proposalScales.size() == first.proposals.size(),
        "Scale/output counts differ");
    for (double scale : first.scales) require(scale == 1., "Refinement changed fixed scale");
    for (double scale : first.proposalScales) require(scale == 1., "Proposal changed fixed scale");
    for (const auto& match : first.matches)
        require(match.score >= options.match.minScore, "Final score bypassed NCC acceptance");
    const auto second = session.run(options);
    same(first.proposals, second.proposals);
    same(first.matches, second.matches);
    // Session owns both original-resolution input snapshots.
    fill(source, 0); fill(templ, 0);
    same(first.matches, session.run(options).matches);
    options.match.invertSource = true;
    const auto inverted = session.run(options);
    require(inverted.matches.size() == 1 && contains(inverted.matches, 98, 34, -90, .35, .5),
        "Inverted source used stale full-NCC pixels");
    options.match.invertSource = false;
    same(first.matches, session.run(options).matches);
    options.refine = false;
    invalid([&] { session.run(options); }, "Sparse orientation was accepted without full NCC refinement");
    options.refine = true; options.match.stopLayer = 1;
    invalid([&] { session.run(options); }, "Sparse route accepted a pyramid stop layer");
    options.match.stopLayer = 0; options.overlap = true;
    invalid([&] { session.run(options); }, "Sparse route silently accepted coarse overlap mode");
    s.scales = {.9, 1.};
    invalid([&] { Session invalidSession(source.view(), templ.view(), s); }, "Sparse route accepted a scale bank");
    s.scales = {.9};
    invalid([&] { Session invalidSession(source.view(), templ.view(), s); }, "Sparse route accepted a nonunit fixed scale");
}
}

int main() {
    try {
        proposalContracts();
        pipelineContracts();
        std::cout << "Sparse orientation contracts: full-image quarter turns, border, "
            "stride, ownership, deterministic caps, empty/invalid inputs, full NCC "
            "polarity gate, scale lock and no-pyramid routing passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
