// SPDX-License-Identifier: BSD-2-Clause
#include "fpm/chroma.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fpm {
namespace {
struct Chroma { int co, cg; };
Chroma chroma(const std::uint8_t* p, bool bgr)
{
	const int r = bgr ? p[2] : p[0], g = p[1], b = bgr ? p[0] : p[2];
	// Offset and quantize (R-B)/2 and (2G-R-B)/4 into unsigned bytes.
	return {(r-b+256)/2, (2*g-r-b+512)/4};
}
void validate(ColourImageView rgb)
{
	const auto max = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
	if (!rgb.data || rgb.width <= 0 || rgb.height <= 0 || std::size_t(rgb.width) > max/3 ||
		rgb.stride < std::size_t(rgb.width)*3 || rgb.stride > max/std::size_t(rgb.height))
		throw std::invalid_argument("Expected an RGB8 image with a valid stride.");
}
struct Moments {
	double n = 0, co = 0, cg = 0, co2 = 0, cg2 = 0;
	void add(Chroma c) {
		++n; co += c.co; cg += c.cg;
		co2 += c.co*c.co; cg2 += c.cg*c.cg;
	}
};
}

ChromaTemplate::ChromaTemplate(ColourImageView rgb)
{
	validate(rgb);
	Moments all;
	for (int y=0; y<rgb.height; ++y) {
		const auto* p = rgb.data + std::size_t(y)*rgb.stride;
		for (int x=0; x<rgb.width; ++x, p+=3) all.add(chroma(p,rgb.bgr));
	}
	const double meanCo = all.co/all.n, meanCg = all.cg/all.n;
	const bool splitCo = all.co2/all.n - meanCo*meanCo >= all.cg2/all.n - meanCg*meanCg;
	Moments low, high;
	for (int y=0; y<rgb.height; ++y) {
		const auto* p = rgb.data + std::size_t(y)*rgb.stride;
		for (int x=0; x<rgb.width; ++x, p+=3) {
			const auto c = chroma(p,rgb.bgr);
			const bool upper = splitCo ? c.co > meanCo : c.cg > meanCg;
			(upper ? high : low).add(c);
		}
	}
	auto flat = [] { return std::domain_error("Chroma search needs two contrasting colours. Include the part edge and background in the template, or use grayscale."); };
	if (!low.n || !high.n) throw flat();
	const double loCo=low.co/low.n, loCg=low.cg/low.n, hiCo=high.co/high.n, hiCg=high.cg/high.n;
	const double radius2=(hiCo-loCo)*(hiCo-loCo)+(hiCg-loCg)*(hiCg-loCg);
	if (radius2 < 1.) throw flat();
	// Centre the affinity on the more saturated group. The second group's
	// distance defines the support, so distant hues map to zero.
	const bool useHigh = std::hypot(hiCo-128.,hiCg-128.) >= std::hypot(loCo-128.,loCg-128.);
	const double co0=useHigh?hiCo:loCo, cg0=useHigh?hiCg:loCg;
	for (int co=0; co<256; ++co) for (int cg=0; cg<256; ++cg) {
		const double d2=(co-co0)*(co-co0)+(cg-cg0)*(cg-cg0);
		affinity_[co*256+cg]=std::uint8_t(std::lround(255.*std::max(0.,1.-d2/radius2)));
	}
	feature_=project(rgb);
	const auto first=feature_.row(0)[0];
	for (int y=0; y<rgb.height; ++y)
		if (std::any_of(feature_.row(y),feature_.row(y)+rgb.width,[first](auto v){return v!=first;})) return;
	throw flat();
}

Image ChromaTemplate::project(ColourImageView rgb) const
{
	validate(rgb);
	Image result(rgb.width,rgb.height);
	for (int y=0; y<rgb.height; ++y) {
		const auto* p=rgb.data+std::size_t(y)*rgb.stride;
		auto* out=result.row(y);
		for (int x=0; x<rgb.width; ++x,p+=3) {
			const auto c=chroma(p,rgb.bgr);
			out[x]=affinity_[c.co*256+c.cg];
		}
	}
	return result;
}
} // namespace fpm
