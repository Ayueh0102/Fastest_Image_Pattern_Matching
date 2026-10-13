// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "coarse_ncc.h"
#include "internal.hpp"
#include <memory>
#include <stdexcept>

namespace fpm_coarse_detail {
class Backend : public fpm::detail::CoarseNccProvider {
public:
    virtual void resetMetrics() = 0;
    virtual void invalidateFineSource() {}
    virtual void setFineBatchSize(std::size_t count) {
        if (count) throw std::invalid_argument("fine GPU batching is unavailable");
    }
    virtual FpmFineMetricsV1 fineMetrics() const {
        FpmFineMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value); return value;
    }
    virtual FpmCoarseMetrics metrics() const = 0;
    virtual void setExecution(const FpmCoarseExecutionConfigV1& config) {
        if (config.pyramid_mode != 0 || config.batch_angles != 1)
            throw std::invalid_argument("GPU execution options are unsupported by this backend");
    }
    virtual FpmCoarseExecutionMetricsV1 executionMetrics() const {
        FpmCoarseExecutionMetricsV1 value{};
        value.version = 1; value.struct_bytes = sizeof(value); value.batch_angles = 1;
        return value;
    }
};
std::unique_ptr<Backend> makeMetalBackend();
}
