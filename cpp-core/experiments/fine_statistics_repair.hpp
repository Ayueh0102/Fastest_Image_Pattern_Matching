// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fpm_coarse_detail {
// Exact post-NCC correction of a gray8 (tw+6)x(th+6) ROI's 7x7 statistics.
// Template storage is borrowed and must remain immutable for this object's
// lifetime. Stats has sum/square/product fields of the same integer type.
// No heap storage, floating-point arithmetic, or GPU functionality is used.
class FineStatisticsRepair {
public:
    static constexpr int outputWidth = 7;
    static constexpr int windowCount = outputWidth * outputWidth;

    FineStatisticsRepair(const std::uint8_t* templatePixels, int width, int height,
        std::size_t templateStride)
        : template_(templatePixels), width_(width), height_(height), stride_(templateStride)
    {
        if (!template_ || width_ < 1 || height_ < 1 ||
            width_ > std::numeric_limits<int>::max()-6 ||
            height_ > std::numeric_limits<int>::max()-6)
            throw std::invalid_argument("fine repair requires a finite positive template shape");
        validateLayout(width_, height_, stride_);
        area_ = std::uint64_t(width_) * std::uint64_t(height_);
        // The existing NCC normalization converts exact integers to binary64.
        // This also leaves ample room for every signed64 correction operation.
        if (area_ > (std::uint64_t(1) << 53) / 65025)
            throw std::invalid_argument("fine repair exceeds exact NCC statistics range");
    }

    // windows contains exactly 49 row-major statistics of the current ROI.
    // before MUST be that ROI pixel's current byte, not its already repaired
    // value. Each correction replaces one byte: at every intermediate step the
    // statistics therefore describe another valid gray8 ROI and remain in
    // [0,65025*area]. Apply each pending replacement once; repeated updates are
    // valid only when before reflects the most recently applied replacement.
    // Returns the number of window-statistic triples updated (0..49).
    template<class Stats>
    std::uint32_t correctPixel(Stats* windows, int x, int y,
        std::uint8_t before, std::uint8_t after) const
    {
        checkStats<Stats>(windows);
        if (x < 0 || y < 0 || x >= width_+6 || y >= height_+6)
            throw std::invalid_argument("fine repair pixel lies outside the ROI");
        if (before == after) return 0;
        using Value = decltype(std::declval<Stats&>().sum);
        const std::int64_t delta = std::int64_t(after)-std::int64_t(before);
        const std::int64_t deltaSquare = delta*(std::int64_t(after)+std::int64_t(before));
        const int firstX = std::max(0, x-width_+1), lastX = std::min(outputWidth-1, x);
        const int firstY = std::max(0, y-height_+1), lastY = std::min(outputWidth-1, y);
        std::uint32_t updated = 0;
        for (int wy = firstY; wy <= lastY; ++wy) {
            const auto* templateRow = template_+std::size_t(y-wy)*stride_;
            for (int wx = firstX; wx <= lastX; ++wx) {
                auto& s = windows[wy*outputWidth+wx];
                // Cast BEFORE subtraction/addition: unsigned arithmetic would
                // wrap when a bright provisional pixel becomes a dark pixel.
                s.sum = Value(std::int64_t(s.sum)+delta);
                s.square = Value(std::int64_t(s.square)+deltaSquare);
                s.product = Value(std::int64_t(s.product)+delta*std::int64_t(templateRow[x-wx]));
                ++updated;
            }
        }
        return updated;
    }

    // Complete scalar fallback for a repaired ROI. Useful when dense repair
    // would otherwise require many scattered corrections; this is O(49*area),
    // so the host must measure it rather than assume it beats another dispatch.
    template<class Stats>
    void recompute(Stats* windows, const std::uint8_t* roi, std::size_t roiStride) const
    {
        checkStats<Stats>(windows);
        if (!roi) throw std::invalid_argument("fine repair requires ROI pixels");
        validateLayout(width_+6, height_+6, roiStride);
        using Value = decltype(std::declval<Stats&>().sum);
        for (int wy = 0; wy < outputWidth; ++wy) for (int wx = 0; wx < outputWidth; ++wx) {
            std::int64_t sum = 0, square = 0, product = 0;
            for (int ty = 0; ty < height_; ++ty) {
                const auto* sourceRow = roi+std::size_t(wy+ty)*roiStride+wx;
                const auto* templateRow = template_+std::size_t(ty)*stride_;
                for (int tx = 0; tx < width_; ++tx) {
                    const std::int64_t pixel = sourceRow[tx];
                    sum += pixel; square += pixel*pixel; product += pixel*templateRow[tx];
                }
            }
            auto& s = windows[wy*outputWidth+wx];
            s.sum = Value(sum); s.square = Value(square); s.product = Value(product);
        }
    }

private:
    static void validateLayout(int width, int height, std::size_t stride)
    {
        if (stride < std::size_t(width) || std::size_t(height-1) >
            (std::numeric_limits<std::size_t>::max()-std::size_t(width))/stride)
            throw std::invalid_argument("fine repair has an invalid stride or address span");
    }
    template<class Stats>
    void checkStats(const Stats* windows) const
    {
        using Value = decltype(std::declval<Stats&>().sum);
        static_assert(std::is_same<Value, decltype(std::declval<Stats&>().square)>::value &&
            std::is_same<Value, decltype(std::declval<Stats&>().product)>::value,
            "fine repair statistics fields must have one integer type");
        static_assert(std::is_same<Value, std::uint32_t>::value ||
            std::is_same<Value, std::uint64_t>::value || std::is_same<Value, std::int64_t>::value,
            "fine repair supports uint32, uint64 or signed64 statistics");
        if (!windows) throw std::invalid_argument("fine repair requires 49 statistics");
        if (area_*65025 > std::uint64_t(std::numeric_limits<Value>::max()))
            throw std::invalid_argument("fine repair statistics type is too narrow");
    }
    const std::uint8_t* template_;
    int width_, height_;
    std::size_t stride_;
    std::uint64_t area_ = 0;
};
} // namespace fpm_coarse_detail
