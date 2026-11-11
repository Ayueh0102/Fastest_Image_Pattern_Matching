// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "matcher.hpp"

namespace fpm {

// Interleaved RGB or BGR, eight bits per component.
struct ColourImageView {
	const std::uint8_t* data;
	int width, height;
	std::size_t stride;
	bool bgr = false;
};

// A template-specific backprojection in the two chroma dimensions of YCoCg.
// The same mapping is applied to every source before any candidate is selected.
class ChromaTemplate {
public:
	explicit ChromaTemplate(ColourImageView rgb);
	Image project(ColourImageView rgb) const;
	const Image& feature() const { return feature_; }
private:
	std::array<std::uint8_t, 256 * 256> affinity_;
	Image feature_;
};

} // namespace fpm
