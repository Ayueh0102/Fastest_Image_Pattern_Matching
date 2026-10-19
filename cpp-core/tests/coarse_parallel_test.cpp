// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
fpm::Image raster(int width, int height, unsigned seed, bool saturated = false)
{
    fpm::Image result(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        result.row(y)[x] = saturated ? uint8_t(255-((x+3*y)%131 == 0)) : uint8_t(rng());
    return result;
}
void equal(const fpm::detail::ScoreMap& a, const fpm::detail::ScoreMap& b)
{
    require(a.width == b.width && a.height == b.height && a.values == b.values,
        "parallel coarse NCC differs from scalar integer/double reference");
}
}
int main()
{
    try {
        auto backend = fpm_coarse_detail::makeMetalBackend();
        FpmCoarseExecutionConfigV1 execution{}; fpm_coarse_execution_default_config(&execution);
        execution.batch_angles = 4; backend->setExecution(execution);
        size_t maps = 0;
        for (int mode : {0, 1, 0, 1}) {
            backend->setCoarseOptimization(mode, mode);
            require(backend->fastCoarsePeaks() == bool(mode), "peak optimization toggle ignored");
            for (auto size : {std::pair<int,int>{1,1}, {3,5}, {15,17}, {31,29}, {257,257}, {258,258}}) {
                for (bool saturated : {false, true}) {
                    backend->resetMetrics();
                    auto templ = fpm::detail::learnLevel(raster(size.first, size.second, 834, saturated));
                    std::vector<fpm::Image> owners;
                    std::vector<fpm::ImageView> sources;
                    for (int i = 0; i < 5; ++i) owners.push_back(raster(size.first+11+i, size.second+7+i, 918+i, saturated));
                    // Include a strided view: padding columns must never be sampled.
                    for (auto& image : owners) { auto v = image.view(); v.width -= 3; sources.push_back(v); }
                    const auto actual = backend->computeBatch(sources, templ);
                    require(actual.size() == sources.size(), "coarse batch lost maps");
                    for (size_t i = 0; i < sources.size(); ++i) {
                        equal(actual[i], fpm::detail::ncc(sources[i], templ, fpm::Backend::Scalar)); ++maps;
                    }
                    equal(backend->compute(sources.front(), templ), actual.front()); ++maps;
                }
            }
        }
        // Force a nonzero-window-offset chunk with a tiny template, cheaply.
        backend->setCoarseOptimization(1, 1); backend->resetMetrics();
        auto large = raster(1537, 1001, 511);
        auto tiny = fpm::detail::learnLevel(raster(3, 3, 77));
        equal(backend->compute(large.view(), tiny), fpm::detail::ncc(large.view(), tiny, fpm::Backend::Scalar));
        require(backend->metrics().command_buffers > 1, "chunked nonzero first-window path not covered");
        std::cout << "Exact coarse SIMD NCC: " << maps+1 << " maps, modes toggled, uint64 overflow values, stride and chunk offsets\n";
        return 0;
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("is unavailable") != std::string::npos) { std::cout << "SKIP: " << message << '\n'; return 77; }
        std::cerr << message << '\n'; return 1;
    }
}
