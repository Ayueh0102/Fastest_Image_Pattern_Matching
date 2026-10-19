// SPDX-License-Identifier: BSD-2-Clause
#include "color_verify.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fpm::research {
namespace {
std::size_t validate(Rgb8View image)
{
    if (!image.data || image.width <= 0 || image.height <= 0)
        throw std::invalid_argument("RGB image must be nonempty");
    const auto row = std::size_t(image.width)*3;
    const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (row > limit || image.stride < row || image.stride > limit ||
        std::size_t(image.height-1) > (limit-row)/image.stride)
        throw std::invalid_argument("Invalid RGB image stride or extent");
    return row*std::size_t(image.height);
}
}

ColorVerifier::ColorVerifier(Rgb8View source, Rgb8View templ, ColorVerifyConfig config,
    const std::uint8_t* mask, std::size_t maskStride)
    : width_(source.width), height_(source.height), config_(config)
{
    const auto sourceBytes = validate(source);
    validate(templ);
    if (!std::isfinite(config.distanceThreshold) || config.distanceThreshold < 0 || config.distanceThreshold > 1 ||
        !std::isfinite(config.minLinearSum) || config.minLinearSum <= 0 || config.minLinearSum > 3 ||
        !std::isfinite(config.minValidFraction) || config.minValidFraction <= 0 || config.minValidFraction > 1)
        throw std::invalid_argument("Invalid color verification configuration");
    if (mask) {
        const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        if (maskStride < std::size_t(templ.width) || maskStride > limit ||
            std::size_t(templ.height-1) > (limit-std::size_t(templ.width))/maskStride)
            throw std::invalid_argument("Invalid template mask stride or extent");
    } else if (maskStride) throw std::invalid_argument("Mask stride requires mask pixels");
    for (int i = 0; i < 256; ++i) {
        const double value = i/255.;
        linear_[i] = value <= .04045 ? value/12.92 : std::pow((value+.055)/1.055, 2.4);
    }
    source_.resize(sourceBytes);
    for (int y = 0; y < source.height; ++y)
        std::memcpy(source_.data()+std::size_t(y)*width_*3,
            source.data+std::size_t(y)*source.stride, std::size_t(width_)*3);
    for (int y = 0; y < templ.height; ++y) for (int x = 0; x < templ.width; ++x) {
        if (mask && !mask[std::size_t(y)*maskStride+x]) continue;
        const auto* p = templ.data+std::size_t(y)*templ.stride+std::size_t(x)*3;
        std::array<double, 3> values{linear_[p[0]],linear_[p[1]],linear_[p[2]]};
        const double sum = values[0]+values[1]+values[2];
        if (sum < config.minLinearSum) continue;
        for (auto& value : values) value /= sum;
        samples_.push_back({x-templ.width*.5f, templ.height*.5f-y, values});
    }
}

ColorVerifyResult ColorVerifier::score(const std::array<float, 4>& pose) const
{
    for (auto value : pose) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite color pose");
    const double angle = double(pose[2])*3.14159265358979323846/180.;
    const double scale = std::exp(double(pose[3]));
    const float cs = float(scale*std::cos(angle)), sn = float(scale*std::sin(angle));
    if (!std::isfinite(cs) || !std::isfinite(sn) || scale <= 0 || (cs == 0 && sn == 0))
        throw std::invalid_argument("Unrepresentable color pose scale");
    ColorVerifyResult out;
    out.eligiblePixels = samples_.size();
    if (samples_.empty()) return out;
    double total = 0;
    for (const auto& sample : samples_) {
        // Same packed float geometry as the existing CPU/Metal NCC API.
        const float sx = width_*.5f+pose[0]+cs*sample.u-sn*sample.v;
        const float sy = height_*.5f-pose[1]-(sn*sample.u+cs*sample.v);
        // Check bounds before any float-to-int conversion. At the final row or
        // column, clamping is exact for the integer coordinate's zero weight.
        if (!(sx >= 0 && sy >= 0 && double(sx) <= width_-1 && double(sy) <= height_-1)) continue;
        const int x = int(std::floor(sx)), y = int(std::floor(sy));
        const int x1 = std::min(x+1,width_-1), y1 = std::min(y+1,height_-1);
        const double dx = sx-x, dy = sy-y;
        std::array<double, 3> values{};
        for (int c = 0; c < 3; ++c) {
            const double a = linear_[source_[(std::size_t(y)*width_+x)*3+c]];
            const double b = linear_[source_[(std::size_t(y)*width_+x1)*3+c]];
            const double e = linear_[source_[(std::size_t(y1)*width_+x)*3+c]];
            const double f = linear_[source_[(std::size_t(y1)*width_+x1)*3+c]];
            values[c] = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
        }
        const double sum = values[0]+values[1]+values[2];
        if (sum < config_.minLinearSum) continue;
        double squared = 0;
        for (int c = 0; c < 3; ++c) {
            const double difference = values[c]/sum-sample.chroma[c];
            squared += difference*difference;
        }
        const double distance = std::sqrt(squared*.5);
        total += distance;
        ++out.validPixels;
        if (distance > config_.distanceThreshold) ++out.changedPixels;
    }
    out.validFraction = double(out.validPixels)/out.eligiblePixels;
    out.status = out.validFraction >= config_.minValidFraction ? 0 : 2;
    if (out.status == 0) {
        out.meanDistance = total/out.validPixels;
        out.changedFraction = double(out.changedPixels)/out.validPixels;
    }
    return out;
}

std::size_t ColorVerifier::retainedBytes() const
{
    return sizeof(*this)+source_.capacity()*sizeof(std::uint8_t)+samples_.capacity()*sizeof(Sample);
}
} // namespace fpm::research
