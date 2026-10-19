// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
fpm::Image noise(int width, int height, unsigned seed)
{
    fpm::Image result(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) result.row(y)[x] = uint8_t(rng());
    return result;
}
void equalImage(const fpm::Image& left, const fpm::Image& right)
{
    require(left.width() == right.width() && left.height() == right.height(), "pyramid dimensions differ");
    require(std::memcmp(left.row(0), right.row(0), size_t(left.width())*left.height()) == 0,
        "Metal pyramid differs from CPU integer Gaussian filter");
}
void memory(const fpm_coarse_detail::Backend& backend)
{
    const auto extra = backend.executionMetrics(); const auto legacy = backend.metrics();
    require(extra.version == 1 && extra.struct_bytes == sizeof(extra), "execution metrics metadata incorrect");
    require(extra.batch_source_arena_bytes <= 64u*1024*1024 && extra.batch_statistics_bytes <= 16u*1024*1024 &&
        extra.max_batch_maps <= uint64_t(extra.batch_angles), "Metal batch allocation/group cap exceeded");
    require(extra.pyramid_buffer_bytes == extra.pyramid_input_bytes+extra.pyramid_output_bytes+extra.pyramid_scratch_bytes,
        "pyramid retained memory accounting inconsistent");
    require(legacy.gpu_bytes == legacy.source_buffer_bytes+legacy.template_buffer_bytes+legacy.workspace_bytes+
        extra.pyramid_buffer_bytes, "retained GPU allocations omitted or counted twice");
    require(extra.batch_source_arena_bytes <= legacy.source_buffer_bytes &&
        extra.batch_template_buffer_bytes == legacy.template_buffer_bytes && extra.batch_statistics_bytes == legacy.workspace_bytes,
        "execution allocation breakdown disagrees with original metrics");
}
void pyramidTests()
{
    auto backend = fpm_coarse_detail::makeMetalBackend();
    FpmCoarseExecutionConfigV1 config{}; fpm_coarse_execution_default_config(&config);
    for (const auto dimensions : {std::pair<int,int>{1, 1}, {1, 17}, {23, 1}, {2, 3}, {17, 31}, {128, 97}, {511, 513}}) {
        std::vector<fpm::Image> actual; actual.push_back(noise(dimensions.first, dimensions.second, 9871));
        auto* samePointer = actual.front().row(0);
        for (int top : {0, 1, 4}) for (int mode : {0, 1, 2, 0, 2, 1}) {
            actual.resize(1);
            require(actual.front().row(0) == samePointer, "test failed to preserve source allocation");
            // Mutate at the same address/dimensions between every call. Cached
            // allocations must not imply cached source content.
            for (int y = 0; y < actual.front().height(); ++y) for (int x = 0; x < actual.front().width(); ++x)
                actual.front().row(y)[x] = uint8_t(255-actual.front().row(y)[x]+x+top+mode);
            std::vector<fpm::Image> expected{fpm::Image(actual.front().view())};
            for (int layer = 1; layer <= top; ++layer) expected.push_back(fpm::detail::pyrDown(expected.back().view()));
            config.pyramid_mode = mode; backend->setExecution(config); backend->resetMetrics();
            const bool handled = backend->buildPyramid(actual, top);
            require(handled == (mode != 0), "pyramid opt-in/fallback contract changed");
            if (mode == 0) {
                require(actual.size() == 1, "disabled GPU pyramid modified input vector");
                equalImage(actual.front(), expected.front());
            } else {
                require(actual.size() == expected.size(), "GPU pyramid returned wrong level count");
                for (size_t layer = 0; layer < expected.size(); ++layer) equalImage(actual[layer], expected[layer]);
            }
            const auto m = backend->executionMetrics();
            require(m.pyramid_levels == uint64_t(mode ? top : 0) &&
                m.pyramid_command_buffers == uint64_t(mode && top ? 1 : 0), "pyramid counters not reset per call");
            require(m.batch_calls == 0 && m.batch_maps == 0 && backend->metrics().command_buffers == 0,
                "pyramid dispatch mixed with coarse NCC metrics");
            memory(*backend);
        }
    }
}
struct Raster {
    int width, height; size_t stride; std::vector<uint8_t> bytes;
    Raster(int w, int h, unsigned seed, int padding = 0) : width(w), height(h), stride(size_t(w)+padding), bytes(stride*h) {
        std::mt19937 rng(seed); for (auto& byte : bytes) byte = uint8_t(rng());
    }
    fpm::ImageView view() const { return {bytes.data(), width, height, stride}; }
};
void batchCase(int width, int height, int tw, int th, int count, int batchSize,
    int expectedGroups, int expectedMax, bool degenerate = false)
{
    auto backend = fpm_coarse_detail::makeMetalBackend();
    auto image = noise(tw, th, 321);
    if (degenerate) std::fill_n(image.row(0), size_t(tw)*th, uint8_t(91));
    const auto templ = fpm::detail::learnLevel(std::move(image));
    std::vector<Raster> sources; sources.reserve(size_t(count));
    for (int index = 0; index < count; ++index) sources.emplace_back(width, height, unsigned(654+index), index%3);
    std::vector<fpm::ImageView> views; for (const auto& source : sources) views.push_back(source.view());
    FpmCoarseExecutionConfigV1 config{}; fpm_coarse_execution_default_config(&config); config.batch_angles = batchSize;
    backend->setExecution(config);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) for (auto& source : sources) for (auto& byte : source.bytes) byte = uint8_t(255-byte);
        backend->resetMetrics();
        const auto actual = backend->computeBatch(views, templ);
        require(actual.size() == views.size(), "batch map count changed");
        for (size_t index = 0; index < views.size(); ++index) {
            const auto expected = fpm::detail::ncc(views[index], templ, fpm::Backend::Scalar);
            require(actual[index].width == expected.width && actual[index].height == expected.height &&
                actual[index].values == expected.values, "ordered batched scoremap differs from exact CPU NCC");
        }
        const auto m = backend->executionMetrics();
        require(m.batch_calls == uint64_t(expectedGroups) && m.max_batch_maps == uint64_t(expectedMax) &&
            m.batch_maps == uint64_t(count), "batch group/cap/tail accounting incorrect");
        require(backend->metrics().score_maps == uint64_t(count), "legacy score map count changed");
        if (!degenerate) require((uint64_t(tw)*th <= uint64_t(UINT32_MAX)/65025 ? backend->metrics().integer32_maps :
            backend->metrics().integer64_maps) == uint64_t(count), "wrong integer statistics width");
        memory(*backend);
    }
}
void stressSourceCap()
{
    // All sources share one allocation to bound test RSS. A near-full template
    // makes only one NCC window, allowing source-arena limits (rather than the
    // statistics limit) to decide grouping. These long per-window kernels are
    // deliberately excluded from the regular quick suite.
    for (int size : {4097, 8193}) {
        auto backend = fpm_coarse_detail::makeMetalBackend();
        auto source = noise(size, size, 5501);
        const auto templ = fpm::detail::learnLevel(fpm::Image(source.view()));
        FpmCoarseExecutionConfigV1 config{}; fpm_coarse_execution_default_config(&config); config.batch_angles = 4;
        backend->setExecution(config); backend->resetMetrics();
        const int count = size == 4097 ? 4 : 1;
        const auto actual = backend->computeBatch(std::vector<fpm::ImageView>(size_t(count), source.view()), templ);
        const auto expected = fpm::detail::ncc(source.view(), templ, fpm::Backend::Scalar);
        for (const auto& map : actual) require(map.values == expected.values, "source-cap fallback changed NCC");
        const auto m = backend->executionMetrics();
        require(m.batch_maps == uint64_t(count) && m.batch_calls == uint64_t(size == 4097 ? 2 : 1) &&
            m.max_batch_maps == uint64_t(size == 4097 ? 3 : 1), "64 MiB source cap did not split/fallback as specified");
        if (size == 8193) require(backend->metrics().source_buffer_bytes > 64u*1024*1024 &&
            m.batch_source_arena_bytes == 0, "oversized singleton did not use original source path");
        memory(*backend);
    }
}
}
int main(int argc, char** argv)
{
    try {
        const bool stress = argc == 2 && std::string(argv[1]) == "--stress";
        require(argc == 1 || stress, "Usage: coarse_execution_test [--stress]");
        if (stress) stressSourceCap();
        else {
            pyramidTests();
            batchCase(59, 53, 23, 17, 9, 4, 3, 4);
            batchCase(261, 261, 258, 258, 4, 4, 1, 4);
            batchCase(59, 53, 23, 17, 4, 4, 4, 1, true);
            batchCase(700, 700, 3, 3, 4, 4, 2, 2);
            batchCase(1200, 1200, 3, 3, 2, 4, 2, 1);
        }
        std::cout << "Exact Metal pyramid/batch outputs, stale-content protection, ordering and bounded allocations passed\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string text(error.what());
        if (text.find("is unavailable") != std::string::npos) { std::cout << "SKIP: " << text << '\n'; return 77; }
        std::cerr << text << '\n'; return 1;
    }
}
