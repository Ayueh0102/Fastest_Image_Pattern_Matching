// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fpm_coarse_detail {
// Shared host/shader ABI. Coordinates are signed Q28. The two-dimensional
// interpolation numerator has denominator 2^56 and maximum 255*2^56 < 2^64.
struct FineWarpDescriptor {
    std::int64_t anchorX, anchorY, cosine, sine;
    std::uint64_t roundingGuard;
    std::uint32_t repairAll, reserved;
};
struct FineWarpConfig {
    std::uint32_t sourceWidth, sourceHeight, roiWidth, roiHeight;
    std::uint32_t roiPixels, roiStride, count, reserved;
};
struct FineStatsConfig {
    std::uint32_t roiWidth, templateWidth, templateHeight, outputWidth;
    std::uint32_t roiStride, windowsPerMap, count, reserved;
};
static_assert(sizeof(FineWarpDescriptor) == 48 && sizeof(FineWarpConfig) == 32 &&
    sizeof(FineStatsConfig) == 32, "Fine Metal ABI changed");

struct FineWarpPlan {
    fpm::Point anchor;
    double cosine, sine;
    FineWarpDescriptor descriptor{};
};

inline FineWarpPlan prepareFineWarp(const fpm::detail::FineRequest& request, int width, int height)
{
    if (!std::isfinite(request.anchor.x) || !std::isfinite(request.anchor.y) ||
        !std::isfinite(request.angle) || width < 1 || height < 1)
        throw std::invalid_argument("fine sampling requires finite pose and positive ROI dimensions");
    // Match rotatedRoi -> rotate -> warp exactly; do not algebraically merge
    // these expressions or derive sine/cosine from the padded-anchor rotation.
    const auto origin = fpm::detail::rotate({-fpm::detail::PADDING, -fpm::detail::PADDING},
        {0, 0}, -request.angle);
    FineWarpPlan plan{{request.anchor.x + origin.x, request.anchor.y + origin.y},
        std::cos(request.angle * fpm::detail::D2R), std::sin(request.angle * fpm::detail::D2R), {}};
    constexpr double scale = 268435456.0; // 2^28, exactly representable.
    const double extent = double(width - 1) + double(height - 1);
    // This conservative range also proves that all Q28 affine products/adds
    // fit signed 64-bit. Out-of-range requests use counted CPU pixel repair.
    if (!std::isfinite(plan.anchor.x) || !std::isfinite(plan.anchor.y) ||
        std::abs(plan.anchor.x) + extent + 1 > 2147483648.0 ||
        std::abs(plan.anchor.y) + extent + 1 > 2147483648.0) {
        plan.descriptor.repairAll = 1;
        return plan;
    }
    auto& descriptor = plan.descriptor;
    descriptor.anchorX = std::llround(plan.anchor.x * scale);
    descriptor.anchorY = std::llround(plan.anchor.y * scale);
    descriptor.cosine = std::llround(plan.cosine * scale);
    descriptor.sine = std::llround(plan.sine * scale);

    // Quantizing anchor/c/s contributes at most (1+x+y)/(2*2^28)
    // coordinate error. 32*epsilon*M conservatively bounds the CPU binary64
    // products/additions, whether contracted into FMA or evaluated separately.
    // Bilinear interpolation over gray8 with zero-extended borders is globally
    // continuous and 255-Lipschitz in EACH coordinate, including cell/border
    // crossings. Thus no assumption about keeping the same floor(sx/sy) is
    // needed. Add 8192*epsilon for the legacy interpolation/rounding arithmetic.
    const double epsilon = std::numeric_limits<double>::epsilon();
    const double quantization = (1 + extent) / (2 * scale);
    const double errorX = quantization + 32 * epsilon * (std::abs(plan.anchor.x) + extent + 1);
    const double errorY = quantization + 32 * epsilon * (std::abs(plan.anchor.y) + extent + 1);
    const double error = 255 * (errorX + errorY) + 8192 * epsilon;
    if (!std::isfinite(error) || error >= 0.25) {
        descriptor.repairAll = 1;
        return plan;
    }
    // Compare on the GPU using integers, not a rounded FP32 confidence test.
    // Outward rounding and 256 numerator units cover host guard conversion.
    descriptor.roundingGuard = std::uint64_t(std::ceil(std::ldexp(
        std::nextafter(error, std::numeric_limits<double>::infinity()), 56))) + 256;
    return plan;
}

inline std::uint8_t sampleFineLegacyPixel(fpm::ImageView source, const FineWarpPlan& plan, int x, int y)
{
    const double c = plan.cosine, s = plan.sine;
    const auto anchor = plan.anchor;
    const double sx = anchor.x + c * x - s * y;
    const double sy = anchor.y + s * x + c * y;
    if (sx < -1 || sy < -1 || sx >= source.width || sy >= source.height) return 0;
    const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
    const double dx = sx - ix, dy = sy - iy;
    auto pixel = [&](int px, int py) -> std::uint8_t {
        return px < 0 || py < 0 || px >= source.width || py >= source.height ? 0 :
            source.data[std::size_t(py) * source.stride + px];
    };
    const double top = pixel(ix, iy) * (1 - dx) + pixel(ix + 1, iy) * dx;
    const double bottom = pixel(ix, iy + 1) * (1 - dx) + pixel(ix + 1, iy + 1) * dx;
    return std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
}
} // namespace fpm_coarse_detail
