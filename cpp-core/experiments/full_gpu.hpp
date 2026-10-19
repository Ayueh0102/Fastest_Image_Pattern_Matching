// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <memory>
namespace fpm::research {
struct Settings; struct Options; struct Result;
class FullGpuPipeline {
public:
    virtual ~FullGpuPipeline() = default;
    virtual void updateSource(ImageView source) = 0;
    virtual Result run(const Options&) = 0;
};
std::unique_ptr<FullGpuPipeline> makeFullGpuPipeline(ImageView, ImageView, const Settings&);
}
