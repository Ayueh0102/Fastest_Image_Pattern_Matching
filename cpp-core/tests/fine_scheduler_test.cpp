// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm;
using namespace fpm::detail;
namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
template<class Exception, class Function> void rejects(Function function)
{
    try { function(); } catch (const Exception&) { return; }
    throw std::runtime_error("invalid scheduler/prepared input was accepted");
}
Image noise(int width, int height, unsigned seed)
{
    Image result(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) result.row(y)[x] = uint8_t(rng());
    return result;
}
void paste(Image& image, const Image& patch, int x, int y)
{
    for (int row = 0; row < patch.height(); ++row)
        std::copy_n(patch.row(row), patch.width(), image.row(y+row)+x);
}
void invert(Image& image)
{
    for (int y = 0; y < image.height(); ++y) for (int x = 0; x < image.width(); ++x) image.row(y)[x] = 255-image.row(y)[x];
}
SourcePyramid prepare(ImageView source, std::size_t depth, bool inverted)
{
    SourcePyramid result; result.inverted = inverted; result.levels.emplace_back(source);
    if (inverted) invert(result.levels.front());
    while (result.levels.size() < depth) result.levels.push_back(pyrDown(result.levels.back().view()));
    return result;
}
class ExactProvider : public CoarseNccProvider {
public:
    std::size_t batch, maxRequests = 0, requests = 0, calls = 0, begins = 0, invalidations = 0, builds = 0;
    int malformed = 0;
    explicit ExactProvider(std::size_t size) : batch(size) {}
    ScoreMap compute(ImageView source, const TemplateLevel& templ) override { return ncc(source, templ, Backend::Auto); }
    std::size_t fineBatchSize() const override { return batch; }
    void invalidateFineSource() override { ++invalidations; }
    void beginFineMatch(const std::vector<Image>& levels) override { require(!levels.empty(), "beginFineMatch has no source"); ++begins; }
    bool buildPyramid(std::vector<Image>&, int) override { ++builds; return false; }
    std::vector<ScoreMap> computeFineBatch(ImageView source, const TemplateLevel& templ,
        const std::vector<FineRequest>& work, double* rotationWallMs) override
    {
        require(!work.empty() && work.size() <= batch, "scheduler exceeded request batch bound");
        ++calls; requests += work.size(); maxRequests = std::max(maxRequests, work.size());
        std::vector<ScoreMap> maps; maps.reserve(work.size()); *rotationWallMs = 0;
        for (const auto& request : work) {
            const auto start = std::chrono::steady_clock::now();
            auto roi = rotatedRoi(source, templ.image.width(), templ.image.height(), request.anchor, request.angle);
            *rotationWallMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            maps.push_back(ncc(roi.view(), templ, Backend::Auto));
        }
        if (malformed == 1) maps.pop_back();
        if (malformed == 2) maps.front().width = 0;
        if (malformed == 3) *rotationWallMs = std::numeric_limits<double>::quiet_NaN();
        return maps;
    }
};
void equalMatches(const std::vector<Match>& actual, const std::vector<Match>& expected)
{
    require(actual.size() == expected.size(), "wavefront changed final detection count");
    require(actual.empty() || std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(Match)) == 0,
        "wavefront changed result geometry, score or stable order");
}
void equalWork(const MatchStageProfile& actual, const MatchStageProfile& expected)
{
#define SAME(field) require(actual.field == expected.field, "wavefront changed " #field)
    SAME(pyramid_levels); SAME(coarse_angles); SAME(coarse_maps); SAME(coarse_windows); SAME(coarse_candidates);
    SAME(fine_layer_visits); SAME(fine_passes); SAME(fine_rotations); SAME(fine_maps); SAME(fine_windows);
    SAME(refined_candidates); SAME(output_matches); SAME(pyramid_live_bytes);
    SAME(coarse_rotation_total_bytes); SAME(coarse_rotation_peak_bytes);
    SAME(fine_roi_total_bytes); SAME(fine_roi_peak_bytes); SAME(scoremap_peak_bytes); SAME(scoremap_total_bytes);
#undef SAME
}
void profileAccounting(const MatchStageProfile& profile)
{
    const double times[] = {profile.source_copy_invert_ms, profile.pyramid_ms, profile.coarse_rotation_ms,
        profile.coarse_ncc_ms, profile.peak_selection_ms, profile.fine_rotation_ms,
        profile.fine_ncc_ms, profile.fine_control_ms, profile.nms_ms, profile.unclassified_ms};
    double sum = 0;
    for (const auto time : times) { require(std::isfinite(time) && time >= 0, "invalid exclusive time after rotation transfer"); sum += time; }
    require(std::abs(sum-profile.core_wall_ms) <= 1e-6*std::max(1., profile.core_wall_ms), "exclusive times no longer sum to core wall");
}
void compare(ImageView source, const Pattern& pattern, const MatchConfig& config, bool requireFound)
{
    MatchStageProfile referenceProfile;
    const auto reference = matchWithCoarseNcc(source, pattern, config, nullptr, &referenceProfile);
    if (requireFound) require(!reference.empty(), "test fixture did not find planted template");
    profileAccounting(referenceProfile);
    auto prepared = prepare(source, pattern.levels().size()+2, config.invertSource);
    // Prepared storage and its pixels must remain immutable throughout matching.
    std::vector<std::vector<uint8_t>> pixels;
    for (const auto& level : prepared.levels) pixels.emplace_back(level.row(0), level.row(0)+size_t(level.width())*level.height());
    for (std::size_t batch : {0u, 1u, 2u, 7u, 32u, 1024u}) {
        for (bool shared : {false, true}) {
            ExactProvider provider(batch); MatchStageProfile profile;
            const auto actual = matchWithCoarseNcc(source, pattern, config, &provider, &profile, shared ? &prepared : nullptr);
            equalMatches(actual, reference); equalWork(profile, referenceProfile); profileAccounting(profile);
            require(provider.invalidations == (shared ? 0u : 1u) && provider.builds == (shared ? 0u : 1u),
                "shared frame rebuilt/invalidated or fresh source retained stale resources");
            require(provider.begins == (batch ? 1u : 0u), "fine lifecycle hook called in wrong mode");
            require(provider.requests == (batch ? profile.fine_maps : 0u) && provider.maxRequests <= batch,
                "provider work counters differ from fine requests");
            if (shared) require(profile.source_copy_bytes == 0 && profile.pyramid_generated_bytes == 0 &&
                profile.source_copy_invert_ms == 0 && profile.pyramid_ms == 0, "prepared match repeated source preparation");
            else require(profile.source_copy_bytes == referenceProfile.source_copy_bytes &&
                profile.pyramid_generated_bytes == referenceProfile.pyramid_generated_bytes, "ordinary preparation accounting changed");
            for (size_t layer = 0; layer < prepared.levels.size(); ++layer)
                require(std::memcmp(pixels[layer].data(), prepared.levels[layer].row(0), pixels[layer].size()) == 0,
                    "matching modified immutable source pyramid");
        }
    }
}
void equivalence()
{
    for (unsigned seed = 0; seed < 4; ++seed) {
        auto templ = noise(17+int(seed)*2, 15+int(seed), 1781+seed);
        auto source = noise(123, 111, 8200+seed);
        paste(source, templ, 5, 7); paste(source, templ, 79, 72);
        const auto rotated = rotateCanvas(templ.view(), templ.width(), templ.height(), 13.25+seed*3.7, 0);
        paste(source, rotated.image, 42, 34);
        for (int variant = 0; variant < 6; ++variant) {
            MatchConfig config; config.maxTargets = 6; config.minScore = .6; config.maxOverlap = .3;
            config.toleranceAngle = variant == 0 ? 0 : 40; config.subpixel = variant != 0 && variant != 5;
            config.stopLayer = variant == 2 ? 1 : 0;
            if (variant == 3) config.angleRanges = {{-40, -4}, {0, 0}, {9, 37}};
            if (variant == 4) config.angleRanges = {{-180, -172}, {-2, 3}, {174, 180}};
            Image input(source.view());
            config.invertSource = (seed+unsigned(variant))%2 != 0;
            if (config.invertSource) invert(input);
            Pattern pattern(templ.view(), variant == 4 ? 100000 : 64);
            // Padding exercises source stride without changing snapshot pixels.
            std::vector<uint8_t> padded(size_t(input.width()+9)*input.height(), 219);
            for (int y = 0; y < input.height(); ++y) std::copy_n(input.row(y), input.width(), padded.data()+size_t(y)*(input.width()+9));
            compare({padded.data(), input.width(), input.height(), size_t(input.width()+9)}, pattern, config, variant == 0);
        }
        if (seed == 0) {
            MatchConfig stopped; stopped.maxTargets = 6; stopped.minScore = .6; stopped.stopLayer = 1; stopped.subpixel = true;
            compare(source.view(), Pattern(templ.view(), 100), stopped, false);
        }
    }
    // All-equal scores exercise stable candidate insertion order through NMS.
    Image flatTemplate(3, 3), flatSource(17, 19);
    MatchConfig ties; ties.maxTargets = 7; ties.minScore = .9; ties.maxOverlap = .5; ties.subpixel = true;
    compare(flatSource.view(), Pattern(flatTemplate.view(), 64), ties, true);
    // Empty coarse results do not manufacture work or a match.
    const auto source = noise(71, 79, 9081), templ = noise(21, 19, 7122);
    MatchConfig empty; empty.minScore = 1; empty.toleranceAngle = 25; empty.subpixel = true;
    compare(source.view(), Pattern(templ.view(), 64), empty, false);
}
void validationAndSnapshot()
{
    auto source = noise(79, 83, 483), templ = noise(17, 19, 921); paste(source, templ, 21, 23);
    Pattern pattern(templ.view(), 64); MatchConfig config; config.maxTargets = 3; config.minScore = .7;
    auto prepared = prepare(source.view(), pattern.levels().size(), false);
    const auto original = match(source.view(), pattern, config);
    invert(source); // Same input address/dimensions; explicit snapshot is unchanged.
    ExactProvider provider(7);
    equalMatches(matchWithCoarseNcc(source.view(), pattern, config, &provider, nullptr, &prepared), original);
    auto wrong = prepared; wrong.inverted = true;
    rejects<std::invalid_argument>([&] { matchWithCoarseNcc(source.view(), pattern, config, &provider, nullptr, &wrong); });
    wrong = prepared; wrong.levels.pop_back();
    rejects<std::invalid_argument>([&] { matchWithCoarseNcc(source.view(), pattern, config, &provider, nullptr, &wrong); });
    wrong = prepared; wrong.levels.front() = Image(78, 83);
    rejects<std::invalid_argument>([&] { matchWithCoarseNcc(source.view(), pattern, config, &provider, nullptr, &wrong); });
    wrong = prepared; wrong.levels.back() = Image(5, 5);
    rejects<std::invalid_argument>([&] { matchWithCoarseNcc(source.view(), pattern, config, &provider, nullptr, &wrong); });
    ExactProvider oversized(1025);
    rejects<std::runtime_error>([&] { matchWithCoarseNcc(templ.view(), Pattern(templ.view()), config, &oversized); });
    for (int kind : {1, 2, 3}) {
        ExactProvider malformed(7); malformed.malformed = kind;
        rejects<std::runtime_error>([&] { matchWithCoarseNcc(templ.view(), Pattern(templ.view()), config, &malformed); });
    }
}
}
int main()
{
    try {
        equivalence(); validationAndSnapshot();
        std::cout << "Fine wavefront/prepared source preserve exact outputs, work counts, ordering and exclusive profiles\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
