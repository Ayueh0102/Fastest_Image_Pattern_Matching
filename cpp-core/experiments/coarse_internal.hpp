// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "coarse_ncc.h"
#include "internal.hpp"
#include <memory>

namespace fpm_coarse_detail {
class Backend : public fpm::detail::CoarseNccProvider {
public:
    virtual void resetMetrics() = 0;
    virtual FpmCoarseMetrics metrics() const = 0;
};
std::unique_ptr<Backend> makeMetalBackend();
}
