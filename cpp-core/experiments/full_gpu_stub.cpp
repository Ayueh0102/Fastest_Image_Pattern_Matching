// SPDX-License-Identifier: BSD-2-Clause
#include "full_gpu.hpp"
#include <stdexcept>
#ifndef __APPLE__
namespace fpm::research {
std::unique_ptr<FullGpuPipeline> makeFullGpuPipeline(ImageView,ImageView,const Settings&) {
    throw std::runtime_error("Complete GPU pipeline requires Apple Metal");
}
}
#endif
