// SPDX-License-Identifier: BSD-2-Clause
#include "color_illuminant.hpp"
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
void validateMask(const std::uint8_t* mask, std::size_t stride, Rgb8View templ, const char* what)
{
    if (mask) {
        const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        if (stride < std::size_t(templ.width) || stride > limit ||
            std::size_t(templ.height-1) > (limit-std::size_t(templ.width))/stride)
            throw std::invalid_argument(std::string("Invalid ")+what+" stride or extent");
    } else if (stride) throw std::invalid_argument(std::string(what)+" stride requires mask pixels");
}
double chromaDistance(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
    const double sa = a[0]+a[1]+a[2], sb = b[0]+b[1]+b[2];
    double squared = 0;
    for (int c = 0; c < 3; ++c) {
        const double difference = a[c]/sa-b[c]/sb;
        squared += difference*difference;
    }
    return std::sqrt(squared*.5);
}
// Exact median; an even count averages the two middle order statistics.
double median(std::vector<double>& values)
{
    const auto middle = values.begin()+std::ptrdiff_t(values.size()/2);
    std::nth_element(values.begin(), middle, values.end());
    const double upper = *middle;
    if (values.size()%2) return upper;
    return (*std::max_element(values.begin(), middle)+upper)*.5;
}
}

IlluminantChromaVerifier::Reference IlluminantChromaVerifier::estimate(
    const std::vector<std::array<std::uint8_t, 3>>& bytes, std::uint64_t eligible) const
{
    Reference out;
    out.eligible = eligible;
    std::array<std::uint64_t, 3> clipped{};
    std::array<std::vector<double>, 3> channels;
    for (auto& channel : channels) channel.reserve(bytes.size());
    for (const auto& pixel : bytes) {
        bool any = false;
        for (int c = 0; c < 3; ++c) if (pixel[c] >= config_.saturationLevel) { ++clipped[c]; any = true; }
        if (any && config_.clipPolicy == ReferenceClipPolicy::Drop) continue;
        for (int c = 0; c < 3; ++c) channels[c].push_back(linear_[pixel[c]]);
    }
    out.samples = channels[0].size();
    out.fraction = eligible ? double(out.samples)/double(eligible) : 0;
    for (int c = 0; c < 3; ++c) out.clip[c] = bytes.empty() ? 0 : double(clipped[c])/double(bytes.size());
    out.usable = out.samples > 0 && out.fraction >= config_.minReferenceFraction;
    if (config_.clipPolicy == ReferenceClipPolicy::Rank)
        out.usable = out.usable && *std::max_element(out.clip.begin(), out.clip.end()) <= config_.maxReferenceClipFraction;
    if (!out.usable) return out;
    for (int c = 0; c < 3; ++c) out.white[c] = median(channels[c]);
    out.usable = out.white[0] > 0 && out.white[1] > 0 && out.white[2] > 0 &&
        out.white[0]+out.white[1]+out.white[2] >= config_.chroma.minLinearSum;
    return out;
}

IlluminantChromaVerifier::IlluminantChromaVerifier(Rgb8View source, Rgb8View templ, const std::uint8_t* colourMask,
    std::size_t colourMaskStride, const std::uint8_t* referenceMask, std::size_t referenceMaskStride,
    IlluminantChromaConfig config)
    : width_(source.width), height_(source.height), config_(config)
{
    const auto sourceBytes = validate(source);
    validate(templ);
    const auto& chroma = config.chroma;
    if (!std::isfinite(chroma.distanceThreshold) || chroma.distanceThreshold < 0 || chroma.distanceThreshold > 1 ||
        !std::isfinite(chroma.minLinearSum) || chroma.minLinearSum <= 0 || chroma.minLinearSum > 3 ||
        !std::isfinite(chroma.minValidFraction) || chroma.minValidFraction <= 0 || chroma.minValidFraction > 1 ||
        config.saturationLevel < 1 || config.saturationLevel > 255 ||
        !(config.maxReferenceClipFraction >= 0 && config.maxReferenceClipFraction < 1) ||
        !(config.minReferenceFraction > 0 && config.minReferenceFraction <= 1))
        throw std::invalid_argument("Invalid illuminant chroma configuration");
    validateMask(colourMask, colourMaskStride, templ, "colour mask");
    if (config.model == ReferenceModel::Linear && config.source != IlluminantSource::Object)
        throw std::invalid_argument("Linear reference model requires an object reference");
    if (config.source == IlluminantSource::Object) {
        if (!referenceMask) throw std::invalid_argument("Object illuminant requires a reference mask");
        validateMask(referenceMask, referenceMaskStride, templ, "reference mask");
    } else {
        if (referenceMask || referenceMaskStride) throw std::invalid_argument("Source rectangle illuminant takes no reference mask");
        const auto& r = config.referenceRect;
        if (r[0] < 0 || r[1] < 0 || r[0] >= r[2] || r[1] >= r[3] || r[2] > source.width || r[3] > source.height)
            throw std::invalid_argument("Reference rectangle must be a nonempty source region");
        for (auto value : config.templateReference)
            if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Invalid template reference RGB");
    }
    for (int i = 0; i < 256; ++i) {
        const double value = i/255.;
        linear_[i] = value <= .04045 ? value/12.92 : std::pow((value+.055)/1.055, 2.4);
    }
    source_.resize(sourceBytes);
    for (int y = 0; y < source.height; ++y)
        std::memcpy(source_.data()+std::size_t(y)*width_*3,
            source.data+std::size_t(y)*source.stride, std::size_t(width_)*3);
    std::vector<std::array<std::uint8_t, 3>> bytes;
    for (int y = 0; y < templ.height; ++y) for (int x = 0; x < templ.width; ++x) {
        const auto* p = templ.data+std::size_t(y)*templ.stride+std::size_t(x)*3;
        if (referenceMask && referenceMask[std::size_t(y)*referenceMaskStride+x]) {
            reference_.push_back({x-templ.width*.5f, templ.height*.5f-y});
            bytes.push_back({p[0],p[1],p[2]});
        }
        if (colourMask && !colourMask[std::size_t(y)*colourMaskStride+x]) continue;
        const std::array<double, 3> linear{linear_[p[0]],linear_[p[1]],linear_[p[2]]};
        const double sum = linear[0]+linear[1]+linear[2];
        if (sum < chroma.minLinearSum) continue;
        auto values = linear;
        for (auto& value : values) value /= sum;
        samples_.push_back({x-templ.width*.5f, templ.height*.5f-y, values, linear});
    }
    if (config.source == IlluminantSource::Object) {
        const auto white = estimate(bytes, bytes.size());
        if (!white.usable) throw std::invalid_argument("Template reference is empty, clipped or dark");
        templateWhite_ = white.white;
        if (config.model == ReferenceModel::Linear) {
            // Outer thirds by rank of u (ends 0, 1) and v (ends 2, 3); ties keep raster order.
            const std::size_t n = reference_.size(), third = n/3;
            if (third == 0) throw std::invalid_argument("Linear reference model needs at least 3 reference pixels");
            ends_.assign(n, 0);
            std::vector<std::size_t> order(n);
            for (int axis = 0; axis < 2; ++axis) {
                for (std::size_t k = 0; k < n; ++k) order[k] = k;
                std::stable_sort(order.begin(), order.end(),
                    [&](std::size_t a, std::size_t b) { return reference_[a][axis] < reference_[b][axis]; });
                for (std::size_t k = 0; k < third; ++k) {
                    ends_[order[k]] |= std::uint8_t(1u << (2*axis));
                    ends_[order[n-1-k]] |= std::uint8_t(2u << (2*axis));
                }
                if (reference_[order[third-1]][axis] == reference_[order[n-third]][axis])
                    throw std::invalid_argument("Linear reference model needs a reference extended along both template axes");
            }
            for (int e = 0; e < 4; ++e) endEligible_[e] = third;
        }
    } else {
        templateWhite_ = config.templateReference;
        bytes.clear();
        const auto& r = config.referenceRect;
        for (int y = r[1]; y < r[3]; ++y) for (int x = r[0]; x < r[2]; ++x) {
            const auto* p = source_.data()+(std::size_t(y)*width_+x)*3;
            bytes.push_back({p[0],p[1],p[2]});
        }
        global_ = estimate(bytes, bytes.size());
    }
}

IlluminantChromaResult IlluminantChromaVerifier::score(const std::array<float, 4>& pose) const
{
    for (auto value : pose) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite color pose");
    const double angle = double(pose[2])*3.14159265358979323846/180.;
    const double scale = std::exp(double(pose[3]));
    const float cs = float(scale*std::cos(angle)), sn = float(scale*std::sin(angle));
    if (!std::isfinite(cs) || !std::isfinite(sn) || scale <= 0 || (cs == 0 && sn == 0))
        throw std::invalid_argument("Unrepresentable color pose scale");
    IlluminantChromaResult out;
    out.eligiblePixels = samples_.size();
    if (samples_.empty()) return out;
    Reference reference = global_;
    const bool linear = config_.model == ReferenceModel::Linear;
    std::array<double, 3> slopeU{}, slopeV{};
    double centreU = 0, centreV = 0;
    if (config_.source == IlluminantSource::Object) {
        std::vector<std::array<std::uint8_t, 3>> bytes;
        std::array<std::vector<std::array<std::uint8_t, 3>>, 4> endBytes;
        // Linear: template u and v of the samples that enter each median (ends and whole reference).
        std::array<std::array<std::vector<double>, 2>, 4> endCoordinates;
        std::array<std::vector<double>, 2> coordinates;
        bytes.reserve(reference_.size());
        for (std::size_t k = 0; k < reference_.size(); ++k) {
            const auto& sample = reference_[k];
            const float sx = width_*.5f+pose[0]+cs*sample[0]-sn*sample[1];
            const float sy = height_*.5f-pose[1]-(sn*sample[0]+cs*sample[1]);
            // Nearest pixel: the bounds keep floor(s+.5) inside the image before
            // any conversion; double addition of .5 to a float is exact.
            if (!(sx >= -.5f && sy >= -.5f && double(sx) < width_-.5 && double(sy) < height_-.5)) continue;
            const int x = int(std::floor(double(sx)+.5)), y = int(std::floor(double(sy)+.5));
            const auto* p = source_.data()+(std::size_t(y)*width_+x)*3;
            bytes.push_back({p[0],p[1],p[2]});
            if (!linear) continue;
            bool kept = config_.clipPolicy != ReferenceClipPolicy::Drop; // estimate() drops samples with any clipped byte
            if (!kept) kept = p[0] < config_.saturationLevel && p[1] < config_.saturationLevel && p[2] < config_.saturationLevel;
            if (kept) for (int axis = 0; axis < 2; ++axis) coordinates[axis].push_back(sample[axis]);
            for (int e = 0; e < 4; ++e) if (ends_[k] >> e & 1u) {
                endBytes[e].push_back(bytes.back());
                if (kept) for (int axis = 0; axis < 2; ++axis) endCoordinates[e][axis].push_back(sample[axis]);
            }
        }
        reference = estimate(bytes, reference_.size());
        if (linear && reference.usable) {
            // Each end is estimated like the whole reference (same clip policy and limits) and
            // placed at the median (u, v) of the samples behind its median. The two slopes solve
            // both end pairs jointly, so an end pair that is unbalanced in the other coordinate
            // (rank ties) does not leak that axis's gradient into its own slope.
            std::array<Reference, 4> end;
            for (int e = 0; e < 4; ++e) {
                end[e] = estimate(endBytes[e], endEligible_[e]);
                reference.usable = reference.usable && end[e].usable;
            }
            if (reference.usable) {
                centreU = median(coordinates[0]);
                centreV = median(coordinates[1]);
                std::array<std::array<double, 2>, 4> at{};
                for (int e = 0; e < 4; ++e) for (int axis = 0; axis < 2; ++axis) at[e][axis] = median(endCoordinates[e][axis]);
                const double du1 = at[1][0]-at[0][0], dv1 = at[1][1]-at[0][1]; // u pair: low -> high u
                const double du2 = at[3][0]-at[2][0], dv2 = at[3][1]-at[2][1]; // v pair: low -> high v
                const double determinant = du1*dv2-dv1*du2;
                reference.usable = du1 > 0 && dv2 > 0 && determinant > 0;
                if (reference.usable) for (int c = 0; c < 3; ++c) {
                    const double dw1 = end[1].white[c]-end[0].white[c], dw2 = end[3].white[c]-end[2].white[c];
                    slopeU[c] = (dw1*dv2-dw2*dv1)/determinant;
                    slopeV[c] = (du1*dw2-du2*dw1)/determinant;
                }
            }
        }
    }
    out.referenceEligible = reference.eligible;
    out.referenceSamples = reference.samples;
    out.referenceFraction = reference.fraction;
    out.referenceClipFraction = reference.clip;
    if (reference.usable) for (int c = 0; c < 3; ++c) out.gain[c] = templateWhite_[c]/reference.white[c];
    bool positive = true; // Linear: the extrapolated illuminant stays positive at every valid colour sample
    std::array<double, 3> correctedSum{};
    const bool clipCheck = config_.clipPolicy != ReferenceClipPolicy::None;
    double total = 0, uncorrected = 0;
    std::array<double, 3> sourceSum{}, templateSum{};
    for (const auto& sample : samples_) {
        // Same packed float geometry and bilinear interpolation as ColorVerifier.
        const float sx = width_*.5f+pose[0]+cs*sample.u-sn*sample.v;
        const float sy = height_*.5f-pose[1]-(sn*sample.u+cs*sample.v);
        if (!(sx >= 0 && sy >= 0 && double(sx) <= width_-1 && double(sy) <= height_-1)) continue;
        const int x = int(std::floor(sx)), y = int(std::floor(sy));
        const int x1 = std::min(x+1,width_-1), y1 = std::min(y+1,height_-1);
        const std::size_t corners[4]{(std::size_t(y)*width_+x)*3, (std::size_t(y)*width_+x1)*3,
            (std::size_t(y1)*width_+x)*3, (std::size_t(y1)*width_+x1)*3};
        if (clipCheck) {
            bool clipped = false;
            for (auto index : corners) for (int c = 0; c < 3; ++c) clipped |= source_[index+c] >= config_.saturationLevel;
            if (clipped) { ++out.clippedPixels; continue; }
        }
        const double dx = sx-x, dy = sy-y;
        std::array<double, 3> values{};
        for (int c = 0; c < 3; ++c) {
            const double a = linear_[source_[corners[0]+c]], b = linear_[source_[corners[1]+c]];
            const double e = linear_[source_[corners[2]+c]], f = linear_[source_[corners[3]+c]];
            values[c] = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
        }
        const double sum = values[0]+values[1]+values[2];
        if (sum < config_.chroma.minLinearSum) continue;
        double squared = 0;
        for (int c = 0; c < 3; ++c) {
            const double difference = values[c]/sum-sample.chroma[c];
            squared += difference*difference;
        }
        uncorrected += std::sqrt(squared*.5);
        ++out.validPixels;
        for (int c = 0; c < 3; ++c) { sourceSum[c] += values[c]; templateSum[c] += sample.linear[c]; }
        if (!reference.usable) continue;
        std::array<double, 3> corrected{};
        if (linear) {
            std::array<double, 3> white{};
            for (int c = 0; c < 3; ++c) {
                white[c] = reference.white[c]+slopeU[c]*(double(sample.u)-centreU)+slopeV[c]*(double(sample.v)-centreV);
                positive = positive && white[c] > 0;
            }
            if (!positive) continue;
            for (int c = 0; c < 3; ++c) {
                corrected[c] = values[c]*(templateWhite_[c]/white[c]);
                correctedSum[c] += corrected[c];
            }
        } else for (int c = 0; c < 3; ++c) corrected[c] = values[c]*out.gain[c];
        const double correctedTotal = corrected[0]+corrected[1]+corrected[2];
        squared = 0;
        for (int c = 0; c < 3; ++c) {
            const double difference = corrected[c]/correctedTotal-sample.chroma[c];
            squared += difference*difference;
        }
        const double distance = std::sqrt(squared*.5);
        total += distance;
        if (distance > config_.chroma.distanceThreshold) ++out.changedPixels;
    }
    out.validFraction = double(out.validPixels)/out.eligiblePixels;
    const bool support = out.validFraction >= config_.chroma.minValidFraction;
    if (support) {
        out.uncorrectedMeanDistance = uncorrected/out.validPixels;
        out.uncorrectedRegionDistance = chromaDistance(sourceSum, templateSum);
    }
    if (!positive) out.gain = {};
    out.status = !support ? 2 : reference.usable && positive ? 0 : 3;
    if (out.status == 0) {
        out.meanDistance = total/out.validPixels;
        out.changedFraction = double(out.changedPixels)/out.validPixels;
        if (linear) {
            out.regionDistance = chromaDistance(correctedSum, templateSum);
            out.referenceSlopeU = slopeU;
            out.referenceSlopeV = slopeV;
        } else {
            std::array<double, 3> corrected{};
            for (int c = 0; c < 3; ++c) corrected[c] = sourceSum[c]*out.gain[c];
            out.regionDistance = chromaDistance(corrected, templateSum);
        }
    }
    return out;
}

std::size_t IlluminantChromaVerifier::retainedBytes() const
{
    return sizeof(*this)+source_.capacity()*sizeof(std::uint8_t)+samples_.capacity()*sizeof(Sample)+
        reference_.capacity()*sizeof(reference_[0])+ends_.capacity();
}
} // namespace fpm::research
