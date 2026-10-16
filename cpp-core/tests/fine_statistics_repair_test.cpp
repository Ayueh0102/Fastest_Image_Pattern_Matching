// SPDX-License-Identifier: BSD-2-Clause
#include "fine_statistics_repair.hpp"
#include <array>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
using fpm_coarse_detail::FineStatisticsRepair;
template<class Value> struct Stats { Value sum = 0, square = 0, product = 0; };
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function call)
{
    try { call(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid fine statistics repair input was accepted");
}
template<class Value>
std::array<Stats<Value>,49> reference(const std::vector<std::uint8_t>& roi, std::size_t rs,
    const std::vector<std::uint8_t>& templ, std::size_t ts, int width, int height)
{
    std::array<Stats<Value>,49> result{};
    // Independent full-window enumeration, intentionally without the helper's
    // pixel-to-window range formulas or its delta correction arithmetic.
    for (int window = 0; window < 49; ++window) {
        auto& s = result[window];
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const std::uint64_t a = roi[std::size_t(window/7+y)*rs+window%7+x];
            const std::uint64_t t = templ[std::size_t(y)*ts+x];
            s.sum += Value(a); s.square += Value(a*a); s.product += Value(a*t);
        }
    }
    return result;
}
template<class Value>
void equal(const std::array<Stats<Value>,49>& actual, const std::array<Stats<Value>,49>& expected)
{
    for (int i = 0; i < 49; ++i) require(actual[i].sum == expected[i].sum &&
        actual[i].square == expected[i].square && actual[i].product == expected[i].product,
        "corrected integer statistics differ from independent full recomputation");
}
template<class Value> void exercise(int width, int height, unsigned seed)
{
    const std::size_t ts = std::size_t(width)+5, rs = std::size_t(width)+13;
    std::vector<std::uint8_t> templ(ts*height), roi(rs*(height+6));
    std::mt19937 rng(seed);
    for (auto& v : templ) v = std::uint8_t(rng());
    for (auto& v : roi) v = std::uint8_t(rng());
    const FineStatisticsRepair repair(templ.data(),width,height,ts);
    auto stats = reference<Value>(roi,rs,templ,ts,width,height);
    const auto initial = stats;
    std::array<Stats<Value>,49> recomputed{};
    repair.recompute(recomputed.data(),roi.data(),rs); equal(recomputed,initial);
    require(repair.correctPixel(stats.data(),0,0,roi[0],roi[0]) == 0,
        "unchanged guarded pixel manufactured statistic updates");
    equal(stats,initial);
    // Every ROI pixel, including thin templates, all four edges/corners and
    // interior pixels, is changed. This also tests arbitrary signed deltas.
    for (int y = 0; y < height+6; ++y) for (int x = 0; x < width+6; ++x) {
        const auto index = std::size_t(y)*rs+x;
        const auto before = roi[index], after = std::uint8_t(255-before);
        const auto updates = repair.correctPixel(stats.data(),x,y,before,after);
        std::uint32_t expectedUpdates = 0;
        for (int wy = 0; wy < 7; ++wy) for (int wx = 0; wx < 7; ++wx)
            if (x >= wx && x < wx+width && y >= wy && y < wy+height) ++expectedUpdates;
        require(updates == expectedUpdates, "pixel correction touched the wrong windows");
        roi[index] = after;
        // Check every step for small templates; checkpoint larger cases to
        // keep the test focused rather than becoming a CPU benchmark.
        if (width*height <= 21 || x == width+5)
            equal(stats,reference<Value>(roi,rs,templ,ts,width,height));
    }
    equal(stats,reference<Value>(roi,rs,templ,ts,width,height));
    repair.recompute(recomputed.data(),roi.data(),rs); equal(stats,recomputed);
    // Repeatedly repair the same pixel, using its CURRENT value each time.
    const int x = (width+6)/2, y = (height+6)/2;
    for (std::uint8_t after : {std::uint8_t(0),std::uint8_t(255),std::uint8_t(0)}) {
        auto& before = roi[std::size_t(y)*rs+x];
        repair.correctPixel(stats.data(),x,y,before,after); before = after;
        equal(stats,reference<Value>(roi,rs,templ,ts,width,height));
    }
}
template<class Value> void saturation(int height)
{
    constexpr int width = 257;
    const std::size_t ts = width, rs = width+6;
    std::vector<std::uint8_t> templ(ts*height,255), roi(rs*(height+6),255);
    const FineStatisticsRepair repair(templ.data(),width,height,ts);
    auto stats = reference<Value>(roi,rs,templ,ts,width,height);
    const std::uint64_t expected = std::uint64_t(width)*height*65025;
    require(std::uint64_t(stats[0].square) == expected && std::uint64_t(stats[0].product) == expected,
        "saturation fixture did not reach expected statistic magnitude");
    if (height == 258) require(expected > std::numeric_limits<std::uint32_t>::max(),
        "64-bit fixture did not cross the uint32 boundary");
    // Maximal negative then positive deltas affect all 49 windows.
    require(repair.correctPixel(stats.data(),6,6,255,0) == 49,"interior update omitted windows");
    roi[6*rs+6] = 0; equal(stats,reference<Value>(roi,rs,templ,ts,width,height));
    repair.correctPixel(stats.data(),6,6,0,255); roi[6*rs+6] = 255;
    equal(stats,reference<Value>(roi,rs,templ,ts,width,height));
    // Full repairAll-style fallback starts from zeroed provisional statistics.
    std::array<Stats<Value>,49> rebuilt{};
    repair.recompute(rebuilt.data(),roi.data(),rs); equal(stats,rebuilt);
}
void invalidInputs()
{
    std::uint8_t pixel = 0;
    rejects([&] { FineStatisticsRepair(nullptr,1,1,1); });
    rejects([&] { FineStatisticsRepair(&pixel,0,1,1); });
    rejects([&] { FineStatisticsRepair(&pixel,1,-1,1); });
    rejects([&] { FineStatisticsRepair(&pixel,std::numeric_limits<int>::max(),1,1); });
    rejects([&] { FineStatisticsRepair(&pixel,2,1,1); });
    rejects([&] { FineStatisticsRepair(&pixel,1,2,std::numeric_limits<std::size_t>::max()); });
    rejects([&] { FineStatisticsRepair(&pixel,1000000,1000000,1000000); });
    const FineStatisticsRepair small(&pixel,1,1,1);
    std::array<Stats<std::uint32_t>,49> stats{};
    std::vector<std::uint8_t> roi(49);
    rejects([&] { small.correctPixel(static_cast<Stats<std::uint32_t>*>(nullptr),0,0,0,1); });
    for (const auto& xy : {std::pair<int,int>{-1,0},{0,-1},{7,0},{0,7}})
        rejects([&] { small.correctPixel(stats.data(),xy.first,xy.second,0,1); });
    rejects([&] { small.recompute(stats.data(),nullptr,7); });
    rejects([&] { small.recompute(stats.data(),roi.data(),6); });
    rejects([&] { small.recompute(stats.data(),roi.data(),std::numeric_limits<std::size_t>::max()); });
    const FineStatisticsRepair large(&pixel,257,258,257);
    rejects([&] { large.correctPixel(stats.data(),0,0,0,1); });
    rejects([&] { large.recompute(stats.data(),roi.data(),263); });
}
}
int main()
{
    try {
        for (const auto& shape : {std::pair<int,int>{1,1},{1,17},{19,1},{3,2},{7,7},{17,19},{63,47}}) {
            exercise<std::uint32_t>(shape.first,shape.second,718);
            exercise<std::uint64_t>(shape.first,shape.second,814);
            exercise<std::int64_t>(shape.first,shape.second,912);
        }
        saturation<std::uint32_t>(257); saturation<std::uint64_t>(258); saturation<std::int64_t>(258);
        invalidInputs();
        std::cout << "Fine statistic corrections equal full integer NCC at borders, thin templates, signed deltas and uint32/64 limits\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
