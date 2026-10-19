// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using fpm::detail::ScoreMap;
using Backend = fpm_coarse_detail::Backend;
void require(bool value, const std::string& text)
{
    if (!value) throw std::runtime_error(text);
}
template<class Function> void rejects(Function function)
{
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid coarse tiling operation accepted");
}
fpm::Image raster(int width, int height, unsigned seed, int kind = 0)
{
    fpm::Image image(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        image.row(y)[x] = kind == 2 ? uint8_t(193) : kind == 1 ?
            uint8_t(255-((x == 0 && y == 0) || (x+3*y+int(seed%131))%131 == 0)) : uint8_t(rng());
    return image;
}
std::unique_ptr<Backend> makeBackend()
{
    auto backend = fpm_coarse_detail::makeMetalBackend();
    FpmCoarseExecutionConfigV1 execution{}; fpm_coarse_execution_default_config(&execution);
    execution.batch_angles = 4; backend->setExecution(execution);
    return backend;
}
void requireEqual(const ScoreMap& actual, const ScoreMap& expected, const std::string& label)
{
    require(actual.width == expected.width && actual.height == expected.height &&
        actual.values.size() == expected.values.size(), label+": NCC map shape changed");
    for (size_t i = 0; i < actual.values.size(); ++i)
        require(std::memcmp(&actual.values[i], &expected.values[i], sizeof(double)) == 0,
            label+": exact NCC score changed at window "+std::to_string(i));
}
void requireEqual(const std::vector<ScoreMap>& actual, const std::vector<ScoreMap>& expected,
    const std::string& label)
{
    require(actual.size() == expected.size(), label+": maps lost or reordered");
    for (size_t i = 0; i < actual.size(); ++i)
        requireEqual(actual[i], expected[i], label+" map "+std::to_string(i));
}
std::vector<ScoreMap> reference(const std::vector<fpm::ImageView>& sources,
    const fpm::TemplateLevel& templ)
{
    std::vector<ScoreMap> result;
    for (const auto source : sources) result.push_back(fpm::detail::ncc(source, templ, fpm::Backend::Scalar));
    return result;
}
void bounded(const Backend& backend)
{
    const auto m = backend.coarseOverlapMetrics();
    require(m.max_pending <= 2 && m.slot0_source_bytes <= 64u*1024*1024 &&
        m.slot1_source_bytes <= 64u*1024*1024 && m.slot0_statistics_bytes <= 16u*1024*1024 &&
        m.slot1_statistics_bytes <= 16u*1024*1024, "tiling exceeded two-slot allocation/pending limits");
    require(m.retained_gpu_bytes == m.slot0_source_bytes+m.slot0_statistics_bytes+m.slot0_template_bytes+
        m.slot1_source_bytes+m.slot1_statistics_bytes+m.slot1_template_bytes,
        "tiling overlap memory breakdown disagrees with retained capacity");
}

void mapsAndBatchBoundaries()
{
    auto backend = makeBackend();
    // 257x257 is still uint32: 66049*65025 = 4294836225 < UINT32_MAX.
    // 257x258 crosses that boundary. Sparse 254 pixels retain large integer
    // sums/products while keeping the template and source nonconstant.
    const std::pair<int,int> shapes[] = {{1,1}, {1,17}, {17,1}, {3,3}, {7,9},
        {15,16}, {16,16}, {17,15}, {31,33}, {257,257}, {257,258}};
    size_t checked = 0;
    for (const auto shape : shapes) for (int kind : {0,1,2}) {
        auto templ = fpm::detail::learnLevel(raster(shape.first, shape.second, 113, kind));
        std::vector<fpm::Image> owners;
        std::vector<fpm::ImageView> sources;
        // Output width 1/2/7/8/13 and height 1/5/8/9/3 exercise horizontal
        // four-window tails, 8x8 output tile tails, and exact one-pixel maps.
        const int widths[] = {1,2,7,8,13}, heights[] = {1,5,8,9,3};
        for (int i = 0; i < 5; ++i)
            owners.push_back(raster(shape.first+widths[i]-1+5, shape.second+heights[i]-1,
                unsigned(731+i), kind == 2 ? 0 : kind));
        for (auto& owner : owners) {
            auto view = owner.view(); view.width -= 5; sources.push_back(view);
        }
        const auto expected = reference(sources, templ);
        for (int mode : {0,2,3,0}) {
            backend->setCoarseOptimization(1, mode); backend->resetMetrics();
            const std::string label = "mode "+std::to_string(mode)+", template "+
                std::to_string(shape.first)+"x"+std::to_string(shape.second)+", kind "+std::to_string(kind);
            requireEqual(backend->computeBatch(sources, templ), expected, label+" batch");
            requireEqual(backend->compute(sources[2], templ), expected[2], label+" single");
            const auto metrics = backend->metrics();
            require(metrics.score_maps == 6, label+": map accounting changed");
            if (shape.first > 1 || shape.second > 1) if (kind != 2) {
                const bool use32 = uint64_t(shape.first)*shape.second <=
                    std::numeric_limits<uint32_t>::max()/uint64_t(65025);
                require((use32 ? metrics.integer32_maps : metrics.integer64_maps) == 6,
                    label+": wrong integer statistics width");
            }
            checked += 6;
        }
    }
    // A flat source with a nonconstant template must use the exact zero-
    // denominator normalization, not manufacture a high correlation score.
    const auto templ = fpm::detail::learnLevel(raster(17, 19, 983));
    const auto flat = raster(37, 31, 0, 2);
    const auto expected = fpm::detail::ncc(flat.view(), templ, fpm::Backend::Scalar);
    for (int mode : {0,2,3}) {
        backend->setCoarseOptimization(0, mode); backend->resetMetrics();
        requireEqual(backend->compute(flat.view(), templ), expected, "constant source"); ++checked;
    }
    std::cout << "Exact tiling NCC maps: " << checked << '\n';
}

void twoPendingSlots()
{
    auto backend = makeBackend(); backend->setCoarseOverlap(true);
    for (int mode : {2,3}) for (int extent : {17,258}) {
        backend->setCoarseOptimization(1, mode); backend->resetMetrics();
        std::vector<fpm::Image> owners;
        for (int i = 0; i < 4; ++i) owners.push_back(raster(extent+7+i, extent+9+i, unsigned(223+i), 1));
        std::vector<fpm::ImageView> first, second;
        for (int i = 0; i < 4; ++i) {
            auto v = owners[i].view(); v.width -= 3;
            (i < 2 ? first : second).push_back(v);
        }
        auto templ = fpm::detail::learnLevel(raster(extent, extent, 303, 1));
        const auto expectedFirst = reference(first, templ);
        backend->beginCoarseBatch(0, first, templ);
        templ = fpm::detail::learnLevel(raster(extent, extent, 304, 1));
        const auto expectedSecond = reference(second, templ);
        backend->beginCoarseBatch(1, second, templ);
        rejects([&] { backend->setCoarseOptimization(1, mode == 2 ? 3 : 2); });
        // Queued work owns both source pixels and normalization metadata.
        for (auto& owner : owners)
            std::fill_n(owner.row(0), size_t(owner.width())*owner.height(), uint8_t(0));
        templ = fpm::detail::learnLevel(raster(extent, extent, 305));
        requireEqual(backend->finishCoarseBatch(0), expectedFirst, "tiling slot 0 ownership");
        requireEqual(backend->finishCoarseBatch(1), expectedSecond, "tiling slot 1 ownership");
        const auto m = backend->coarseOverlapMetrics();
        require(m.submitted_batches == 2 && m.completed_batches == 2 && m.max_pending == 2 &&
            m.submissions_with_pending == 1 && m.fallback_batches == 0,
            "tiling did not execute two actual pending GPU batches");
        bounded(*backend);
    }
}

void chunkOffsetsAndAsyncFallback()
{
    auto backend = makeBackend();
    auto source = raster(1537, 1001, 719), small = raster(19, 17, 720);
    auto templ = fpm::detail::learnLevel(raster(3, 3, 721));
    const auto expected = fpm::detail::ncc(source.view(), templ, fpm::Backend::Scalar);
    const auto expectedSmall = fpm::detail::ncc(small.view(), templ, fpm::Backend::Scalar);
    // More than 16 MiB/12 windows forces a nonzero firstWindow that is not
    // aligned to an output row, four-window group, or 8x8 output tile.
    require(expected.values.size() > (16u*1024*1024)/12, "chunk test does not cross statistics workspace");
    for (int mode : {0,2,3}) {
        backend->setCoarseOptimization(1, mode); backend->resetMetrics();
        requireEqual(backend->compute(source.view(), templ), expected, "chunked firstWindow mode "+std::to_string(mode));
        require(backend->metrics().command_buffers > 1, "large tiling map was not chunked");
        if (!mode) continue;
        backend->setCoarseOverlap(true); backend->resetMetrics();
        // Slot 0 is still pending when slot 1 takes an oversized synchronous
        // fallback. Its workspace and template must remain independent.
        backend->beginCoarseBatch(0, {small.view(), small.view()}, templ);
        backend->beginCoarseBatch(1, {source.view(), small.view()}, templ);
        requireEqual(backend->finishCoarseBatch(0), {expectedSmall, expectedSmall}, "pending tile batch");
        const auto actual = backend->finishCoarseBatch(1);
        require(actual.size() == 2, "oversized tiling fallback lost maps");
        requireEqual(actual[0], expected, "oversized tile fallback firstWindow");
        requireEqual(actual[1], expectedSmall, "oversized tile fallback tail");
        const auto m = backend->coarseOverlapMetrics();
        require(m.submitted_batches == 1 && m.completed_batches == 1 && m.fallback_batches == 1 &&
            m.fallback_oversized == 1 && m.fallback_maps == 2,
            "oversized tiling fallback was not recorded correctly");
        bounded(*backend); backend->setCoarseOverlap(false);
    }
}

void paste(fpm::Image& source, const fpm::Image& templ, int x, int y)
{
    for (int row = 0; row < templ.height(); ++row)
        std::copy_n(templ.row(row), templ.width(), source.row(y+row)+x);
}
void sameWork(const fpm::detail::MatchStageProfile& actual, const fpm::detail::MatchStageProfile& expected)
{
#define SAME(field) require(actual.field == expected.field, "tiling changed matcher work: " #field)
    SAME(coarse_angles); SAME(coarse_maps); SAME(coarse_windows); SAME(coarse_candidates);
    SAME(fine_layer_visits); SAME(fine_passes); SAME(fine_rotations); SAME(fine_maps); SAME(fine_windows);
    SAME(refined_candidates); SAME(output_matches);
#undef SAME
}
void matchingOrder()
{
    auto backend = makeBackend();
    for (int variant = 0; variant < 3; ++variant) {
        const auto templ = raster(19+variant, 17+variant, unsigned(911+variant));
        auto source = raster(173, 151, unsigned(615+variant));
        paste(source, templ, 11, 13); paste(source, templ, 121, 97);
        const auto turned = fpm::detail::rotateCanvas(templ.view(), templ.width(), templ.height(), 17.25, 0);
        paste(source, turned.image, 61, 55);
        fpm::Pattern pattern(templ.view(), 64);
        fpm::MatchConfig config; config.maxTargets = 8; config.minScore = .5; config.maxOverlap = .25;
        config.toleranceAngle = variant == 2 ? 0 : 90; config.subpixel = true; config.stopLayer = variant%2;
        if (variant == 1) config.angleRanges = {{-90,-5}, {0,0}, {7,90}};
        fpm::detail::MatchStageProfile expectedProfile{};
        const auto expected = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, nullptr, &expectedProfile);
        require(expected.size() >= 2, "tiling matching fixture did not detect planted copies");
        for (int mode : {0,2,3}) for (bool async : {false,true}) {
            backend->setCoarseOptimization(1, mode); backend->setCoarseOverlap(async); backend->resetMetrics();
            fpm::detail::MatchStageProfile actualProfile{};
            const auto actual = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, backend.get(), &actualProfile);
            require(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(fpm::Match)) == 0,
                "tiling changed ordered full matches, scores or geometry");
            sameWork(actualProfile, expectedProfile); bounded(*backend);
            const auto m = backend->coarseOverlapMetrics();
            if (async && variant != 2)
                require(m.submitted_batches >= 2 && m.max_pending == 2,
                    "tiling full matcher did not exercise two pending angle batches");
            require(m.submitted_batches == m.completed_batches && m.drained_batches == 0,
                "tiling matcher discarded work or left a batch pending");
        }
    }
}

struct Frame {
    fpm_frame_handle* handle = nullptr;
    char error[1024]{};
    explicit Frame(bool metal) {
        FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
        if (!metal) { config.backend = 0; config.pyramid_mode = 0; config.batch_angles = 1; }
        require(fpm_frame_create(&config, sizeof(config), &handle, error, sizeof(error)) == FPM_OK, error);
    }
    ~Frame() { fpm_frame_destroy(handle); }
    int set(const FpmCoarseOptimizationConfigV1& config) {
        return fpm_frame_set_coarse_optimization(handle, &config, sizeof(config), error, sizeof(error));
    }
};
void configuration()
{
    static_assert(sizeof(FpmCoarseOptimizationConfigV1) == 32, "tiling changed coarse optimization ABI");
    for (bool metal : {false,true}) {
        Frame frame(metal);
        FpmCoarseOptimizationConfigV1 valid{1, sizeof(FpmCoarseOptimizationConfigV1), 0, 0, 0, 0};
        require(frame.set(valid) == FPM_OK, "default coarse optimization rejected");
        for (int peak : {0,1}) for (int mode : {2,3,2}) {
            valid.peak_mode = peak; valid.ncc_mode = mode;
            require(frame.set(valid) == (metal ? FPM_OK : FPM_ERROR),
                "Metal rejected tiling route or CPU accepted unavailable tiling route");
        }
        valid.peak_mode = 0; valid.ncc_mode = 0;
        for (int kind = 0; kind < 8; ++kind) {
            auto invalid = valid;
            if (kind == 0) invalid.version = 2;
            if (kind == 1) invalid.struct_bytes = sizeof(invalid)-1;
            if (kind == 2) invalid.peak_mode = 3;
            if (kind == 3) invalid.peak_mode = -1;
            if (kind == 4) invalid.ncc_mode = -1;
            if (kind == 5) invalid.ncc_mode = 4;
            if (kind == 6) invalid.reserved0 = 1;
            if (kind == 7) invalid.reserved1 = 1;
            require(frame.set(invalid) == FPM_ERROR, "invalid tiling config accepted");
        }
        require(fpm_frame_set_coarse_optimization(frame.handle, nullptr, sizeof(valid), nullptr, 0) == FPM_ERROR &&
            fpm_frame_set_coarse_optimization(frame.handle, &valid, sizeof(valid)-1, nullptr, 0) == FPM_ERROR &&
            fpm_frame_set_coarse_optimization(nullptr, &valid, sizeof(valid), nullptr, 0) == FPM_ERROR,
            "invalid tiling setter pointers/size accepted");
        require(frame.set(valid) == FPM_OK, "legacy route cannot be restored after tiling configuration");
    }
}
} // namespace

int main()
{
    try {
        configuration(); mapsAndBatchBoundaries(); twoPendingSlots();
        chunkOffsetsAndAsyncFallback(); matchingOrder();
        std::cout << "Coarse tiling preserves exact maps, boundaries, chunk offsets, async ownership and full matching\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (message.find("is unavailable") != std::string::npos) {
            std::cout << "SKIP: " << message << '\n'; return 77;
        }
        std::cerr << message << '\n'; return 1;
    }
}
