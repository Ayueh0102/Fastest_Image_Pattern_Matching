// SPDX-License-Identifier: BSD-2-Clause
#include "scale_support_refine.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fpm::research {
namespace {
void validate(ImageView image)
{
    if (!image.data || image.width <= 0 || image.height <= 0 ||
        image.stride < size_t(image.width) ||
        image.stride > (std::numeric_limits<size_t>::max() - size_t(image.width)) / size_t(image.height))
        throw std::invalid_argument("Invalid scale support gray8 image");
}
double pixel(ImageView image, int x, int y, bool invert)
{
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) return 0;
    const double value = image.data[size_t(y)*image.stride+size_t(x)];
    return invert ? 255.-value : value;
}
double bilinear(ImageView image, double x, double y, bool invert)
{
    // These checks precede floor/int conversion, including overflowed poses.
    if (!std::isfinite(x) || !std::isfinite(y) || x <= -1. || y <= -1. ||
        x >= image.width || y >= image.height) return 0;
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double fx = x-ix, fy = y-iy;
    const double a = pixel(image,ix,iy,invert), b = pixel(image,ix+1,iy,invert);
    const double c = pixel(image,ix,iy+1,invert), d = pixel(image,ix+1,iy+1,invert);
    return (a+(b-a)*fx)*(1.-fy)+(c+(d-c)*fx)*fy;
}
double score(ImageView source, ImageView templ, Point center,
    double cs, double sn, int margin, bool invert, ScaleSupportRefineResult& result)
{
    ++result.evaluations;
    double sumI=0, sumT=0, squareI=0, squareT=0, product=0;
    const uint64_t n = uint64_t(templ.width-2*margin)*uint64_t(templ.height-2*margin);
    result.samples += n;
    for (int y=margin; y<templ.height-margin; ++y) for (int x=margin; x<templ.width-margin; ++x) {
        const double u=x-templ.width/2., v=y-templ.height/2.;
        const double sampled=bilinear(source,center.x+cs*u+sn*v,center.y-sn*u+cs*v,invert);
        const double target=templ.data[size_t(y)*templ.stride+size_t(x)];
        sumI+=sampled;sumT+=target;squareI+=sampled*sampled;squareT+=target*target;product+=sampled*target;
    }
    const double energyI=std::max(0.,squareI-sumI*sumI/double(n));
    const double energyT=std::max(0.,squareT-sumT*sumT/double(n));
    if (energyI/double(n) <= 1e-12 || energyT/double(n) <= 1e-12) return -2;
    const double value=(product-sumI*sumT/double(n))/std::sqrt(energyI*energyT);
    return std::isfinite(value) ? std::clamp(value,-1.,1.) : -2.;
}
} // namespace

ScaleSupportRefineResult refineScaleSupport(ImageView source, ImageView templ,
    Point center, double angleDegrees, double initialScale, bool invertSource)
{
    validate(source);validate(templ);
    ScaleSupportRefineResult result;
    if (std::min(templ.width,templ.height)<7 || uint64_t(templ.width)*uint64_t(templ.height)>4096 ||
        !std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(angleDegrees) ||
        !std::isfinite(initialScale) || initialScale<=0 || initialScale>std::numeric_limits<double>::max()/1.04)
        return result;
    result.eligible=true;
    const double radians=std::remainder(angleDegrees,360.)*(3.14159265358979323846/180.);
    const double cs=std::cos(radians), sn=std::sin(radians);
    result.originalInteriorScore=score(source,templ,center,initialScale*cs,initialScale*sn,2,invertSource,result);
    result.bestInteriorScore=result.originalInteriorScore;
    if (result.originalInteriorScore < -1.) return result;
    int best=16;
    for (int index=0;index<33;++index) {
        if (index==16) continue;
        const double scale=initialScale*(.96+.0025*index);
        const double value=score(source,templ,center,scale*cs,scale*sn,2,invertSource,result);
        // Strict improvement preserves the center index for equal/flat scores.
        if (value>result.bestInteriorScore+1e-12) { result.bestInteriorScore=value;best=index; }
    }
    result.boundaryOptimum=best==0 || best==32;
    if (best==16 || result.boundaryOptimum) return result;
    const double scale=initialScale*(.96+.0025*best);
    const double full=score(source,templ,center,scale*cs,scale*sn,0,invertSource,result);
    if (full < -1. || !std::isfinite(full)) return result;
    result.candidate=ScaleSupportCandidate{scale,full,result.bestInteriorScore};
    return result;
}
} // namespace fpm::research
