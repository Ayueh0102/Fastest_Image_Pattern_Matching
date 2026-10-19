// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include "fpm/matcher.hpp"
#include <QImage>
#include <QString>
#include <cstring>
#include <stdexcept>

namespace fpm::io {

enum class GrayLoadMode { DirectGray8, LegacyRgb };

struct GrayLoadInfo {
    bool directGray8 = false;
    // Logical RGB888 conversion buffer size, including row padding; not RSS.
    std::size_t rgbTemporaryBytes = 0;
};

// Accept by value so a temporary decoded image can be released by the RGB
// conversion before allocating the gray output, as in the original loader.
inline Image grayFromQImage(QImage image, GrayLoadMode mode = GrayLoadMode::DirectGray8,
                           GrayLoadInfo* info = nullptr)
{
    if (info) *info = {};
    if (image.isNull()) throw std::invalid_argument("Cannot decode image");
    if (mode != GrayLoadMode::DirectGray8 && mode != GrayLoadMode::LegacyRgb)
        throw std::invalid_argument("Invalid gray load mode");
    if (mode == GrayLoadMode::DirectGray8 && image.format() == QImage::Format_Grayscale8) {
        // 77+150+29=256, so grayFromRgb(v,v,v)==v for every gray8 value.
        // Only this explicit format bypasses conversion: palettes, alpha and
        // 16-bit gray continue through the exact existing Qt RGB888 path.
        Image out(image.width(), image.height());
        for (int y = 0; y < image.height(); ++y)
            std::memcpy(out.row(y), image.constScanLine(y), std::size_t(image.width()));
        if (info) info->directGray8 = true;
        return out;
    }
    image = image.convertToFormat(QImage::Format_RGB888);
    if (image.isNull()) throw std::invalid_argument("Cannot decode image");
    if (info) info->rgbTemporaryBytes = std::size_t(image.sizeInBytes());
    Image out(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        const auto* row = image.constScanLine(y);
        for (int x = 0; x < image.width(); ++x) {
            const auto* p = row + 3 * x;
            out.row(y)[x] = grayFromRgb(p[0], p[1], p[2]);
        }
    }
    return out;
}

inline Image readGrayImage(const QString& path, GrayLoadMode mode = GrayLoadMode::DirectGray8,
                           GrayLoadInfo* info = nullptr)
{
    return grayFromQImage(QImage(path), mode, info);
}

} // namespace fpm::io
