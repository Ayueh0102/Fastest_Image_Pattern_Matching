// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_ncc.h"
#include "internal.hpp"
#include "metal_device.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
fpm_image cview(ImageView image) { return {image.data, image.width, image.height, image.stride}; }
Image noise(int w, int h, unsigned seed) {
    Image result(w, h); std::mt19937 rng(seed);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) result.row(y)[x] = uint8_t(rng());
    return result;
}
struct Handle {
    fpm_coarse_handle* value = nullptr;
    FpmCoarseMetrics metrics{};
    char error[1024]{};
    Handle(ImageView templ, int minArea = 256) {
        if (fpm_coarse_create(cview(templ), minArea, &value, &metrics, error, sizeof(error)) != FPM_OK)
            throw std::runtime_error(error);
    }
    ~Handle() { fpm_coarse_destroy(value); }
};
void compareMap(ImageView source, const Image& templ, int expectedBits = 32, bool chunked = false) {
    Handle h(templ.view());
    const auto expected = detail::ncc(source, detail::learnLevel(Image(templ.view())), Backend::Auto);
    size_t count = 0;
    require(fpm_coarse_scoremap(h.value, cview(source), nullptr, 0, &count, &h.metrics,
        h.error, sizeof(h.error)) == FPM_BUFFER_TOO_SMALL, "map size query");
    require(count == expected.values.size() && h.metrics.command_buffers == 0, "map query must not run GPU");
    std::vector<double> actual(count, -17);
    require(fpm_coarse_scoremap(h.value, cview(source), actual.data(), actual.size(), &count,
        &h.metrics, h.error, sizeof(h.error)) == FPM_OK, h.error);
    if (std::memcmp(actual.data(), expected.values.data(), count * sizeof(double)) != 0) {
        for (size_t i = 0; i < count; ++i)
            if (std::memcmp(&actual[i], &expected.values[i], sizeof(double)) != 0)
                throw std::runtime_error("score map differs at " + std::to_string(i) +
                    ": CPU " + std::to_string(expected.values[i]) + " GPU " + std::to_string(actual[i]));
    }
    require(h.metrics.workspace_bytes <= 16 * 1024 * 1024, "statistics workspace cap");
    require(h.metrics.gpu_bytes == h.metrics.source_buffer_bytes + h.metrics.template_buffer_bytes +
        h.metrics.workspace_bytes, "buffer accounting");
    require(h.metrics.scoremap_peak_bytes == count * sizeof(double), "CPU scoremap accounting");
    if (expectedBits == 32) require(h.metrics.integer32_maps == 1, "uint32 path");
    if (expectedBits == 64) require(h.metrics.integer64_maps == 1, "uint64 path");
    if (expectedBits == 0) require(h.metrics.command_buffers == 0, "constant template shortcut");
    if (chunked) require(h.metrics.command_buffers > 1, "large map must chunk statistics");
    require(h.metrics.wall_ms >= h.metrics.ncc_ms && h.metrics.ncc_ms >= h.metrics.normalization_ms,
        "wall intervals contain component intervals");
    // Reuse the exact same handle and ensure source pixels are uploaded anew.
    Image changed(source);
    changed.row(0)[0] ^= 127;
    const auto second = detail::ncc(changed.view(), detail::learnLevel(Image(templ.view())), Backend::Auto);
    require(fpm_coarse_scoremap(h.value, cview(changed.view()), actual.data(), actual.size(), &count,
        &h.metrics, h.error, sizeof(h.error)) == FPM_OK, h.error);
    require(std::memcmp(actual.data(), second.values.data(), count * sizeof(double)) == 0,
        "reused handle must not cache stale source");
}
void maps() {
    for (unsigned seed = 1; seed <= 7; ++seed) {
        auto src = noise(31 + seed, 24 + seed, seed), templ = noise(3 + seed, 2 + seed, seed + 30);
        std::vector<uint8_t> padded(size_t(src.height()) * (src.width() + 7), 239);
        for (int y = 0; y < src.height(); ++y)
            std::copy_n(src.row(y), src.width(), padded.data() + size_t(y) * (src.width() + 7));
        compareMap({padded.data(), src.width(), src.height(), size_t(src.width() + 7)}, templ);
    }
    Image constantSource(29, 21), constantTemplate(7, 5);
    for (int y = 0; y < 21; ++y) std::fill_n(constantSource.row(y), 29, 193);
    compareMap(constantSource.view(), noise(7, 5, 81));
    compareMap(noise(29, 21, 82).view(), constantTemplate, 0);
    Image low = noise(29, 21, 91), templ = noise(7, 5, 92);
    for (int y = 0; y < low.height(); ++y) for (int x = 0; x < low.width(); ++x) low.row(y)[x] = 128 + low.row(y)[x] % 2;
    for (int y = 0; y < templ.height(); ++y) for (int x = 0; x < templ.width(); ++x) templ.row(y)[x] = 128 + templ.row(y)[x] % 2;
    compareMap(low.view(), templ);
    Image inverse(templ.view());
    for (int y = 0; y < inverse.height(); ++y) for (int x = 0; x < inverse.width(); ++x) inverse.row(y)[x] = 255 - inverse.row(y)[x];
    compareMap(inverse.view(), templ);
    // Max product 65025 * area straddles UINT32_MAX at area 66051/66052.
    for (int bits : {32, 64}) {
        const int w = bits == 32 ? 3 : 4, h = bits == 32 ? 22017 : 16513;
        Image large(w, h), source(w + 1, h + 1);
        for (int y = 0; y < h; ++y) std::fill_n(large.row(y), w, 255);
        for (int y = 0; y < source.height(); ++y) std::fill_n(source.row(y), source.width(), 255);
        large.row(1)[0] = 254; source.row(1)[0] = 254;
        uint64_t actualProduct = 0, actualSquare = 0;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            actualProduct += unsigned(large.row(y)[x]) * unsigned(source.row(y)[x]);
            actualSquare += unsigned(source.row(y)[x]) * unsigned(source.row(y)[x]);
        }
        require((actualProduct > UINT32_MAX) == (bits == 64) &&
            (actualSquare > UINT32_MAX) == (bits == 64), "overflow fixture must straddle actual uint32 sum limits");
        compareMap(source.view(), large, bits);
    }
    compareMap(noise(1200, 1200, 188).view(), noise(2, 2, 199), 32, true);
}
void fullMatches() {
    Image source = noise(127, 97, 411), templ = noise(21, 17, 412);
    for (const Point p : {Point{21, 27}, Point{78, 53}})
        for (int y = 0; y < templ.height(); ++y)
            std::copy_n(templ.row(y), templ.width(), source.row(int(p.y) + y) + int(p.x));
    Pattern pattern(templ.view(), 64); Handle handle(templ.view(), 64);
    for (int mode = 0; mode < 6; ++mode) {
        MatchConfig config; config.minScore = .65; config.maxTargets = 5;
        config.toleranceAngle = mode == 0 ? 0 : 180;
        config.stopLayer = mode == 1; config.subpixel = mode == 2;
        config.maxOverlap = mode == 3 ? 1 : 0;
        config.invertSource = mode == 4;
        if (mode == 5) config.angleRanges = {{-15, 0}, {165, 180}};
        fpm_config cc{config.maxTargets, config.minScore, config.maxOverlap, config.toleranceAngle,
            config.subpixel, config.invertSource, 0, config.stopLayer, nullptr, 0};
        std::vector<fpm_angle_range> ranges;
        for (const auto& range : config.angleRanges) ranges.push_back({range.min, range.max});
        cc.ranges = ranges.data(); cc.range_count = ranges.size();
        const auto expected = match(source.view(), pattern, config);
        std::vector<fpm_result> actual(32); size_t count = 0;
        require(fpm_coarse_match(handle.value, cview(source.view()), &cc, actual.data(), actual.size(),
            &count, &handle.metrics, handle.error, sizeof(handle.error)) == FPM_OK, handle.error);
        require(count == expected.size(), "full match candidate count");
        for (size_t i = 0; i < count; ++i) {
            require(actual[i].score == expected[i].score && actual[i].angle == expected[i].angle &&
                actual[i].center.x == expected[i].center.x && actual[i].center.y == expected[i].center.y,
                "full match exact output order/pose/score");
            for (int k = 0; k < 4; ++k)
                require(actual[i].corners[k].x == expected[i].corners[k].x &&
                    actual[i].corners[k].y == expected[i].corners[k].y, "full match exact corners");
        }
    }
    size_t count = 123;
    require(fpm_coarse_match(handle.value, cview(source.view()), nullptr, nullptr, 1, &count,
        &handle.metrics, handle.error, sizeof(handle.error)) == FPM_ERROR && count == 0, "bad output pointer");
    require(fpm_coarse_scoremap(handle.value, cview(source.view()), nullptr, 0, &count, nullptr,
        handle.error, sizeof(handle.error)) == FPM_ERROR, "metrics required");
}
}
int main() {
    static_assert(sizeof(FpmCoarseMetrics) == 160, "coarse metrics ABI changed");
    if (!fpm_test::metalDeviceAvailable()) { std::cout << "SKIP: no Metal device\n"; return 77; }
    try { maps(); fullMatches(); std::cout << "Metal coarse NCC exact maps / matches passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
