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
    virtual void setFineNccMode(int mode) {
        if (mode) throw std::invalid_argument("fine NCC reuse requires Metal");
    }
    virtual FpmFineReuseMetricsV1 fineReuseMetrics() const {
        FpmFineReuseMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value);
        value.stats_device_ms = -1; return value;
    }
    virtual void setFineFused(bool enabled) {
        if (enabled) throw std::invalid_argument("fine fusion requires Metal");
    }
    virtual FpmFineFusionMetricsV1 fineFusionMetrics() const {
        FpmFineFusionMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value);
        value.fused_device_ms = -1; return value;
    }
    virtual void setFineSearchOptions(bool profiling, bool reuse) {
        if (profiling || reuse) throw std::invalid_argument("fine search options require Metal");
    }
    virtual FpmFineSearchMetricsV1 fineSearchMetrics() const {
        FpmFineSearchMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value);
        value.warp_device_ms = -1; return value;
    }
    virtual void setCoarseOptimization(int peakMode, int nccMode) {
        if (peakMode || nccMode) throw std::invalid_argument("coarse optimization is unavailable");
    }
    virtual FpmFineMetricsV1 fineMetrics() const {
        FpmFineMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value); return value;
    }
    virtual FpmCoarseMetrics metrics() const = 0;
    virtual void setCoarseOverlap(bool enabled) {
        if (enabled) throw std::invalid_argument("coarse overlap is unavailable");
    }
    virtual FpmCoarseOverlapMetricsV1 coarseOverlapMetrics() const {
        FpmCoarseOverlapMetricsV1 value{};
        value.version = 1; value.struct_bytes = sizeof(value);
        value.device_ms = value.device_span_ms = -1;
        return value;
    }
    virtual void setExecution(const FpmCoarseExecutionConfigV1& config) {
        if (config.pyramid_mode != 0 || config.batch_angles != 1)
            throw std::invalid_argument("GPU execution options are unsupported by this backend");
    }
    virtual FpmCoarseExecutionMetricsV1 executionMetrics() const {
        FpmCoarseExecutionMetricsV1 value{};
        value.version = 1; value.struct_bytes = sizeof(value); value.batch_angles = 1;
        return value;
    }
    // Frame transfer mode (fpm_frame_set_transfer_mode). A backend that reuses
    // source levels keeps the frame's host levels across prepares and fills
    // them in prepareSource(); releaseHostSource() must run before the frame
    // frees or replaces those levels.
    virtual void setTransferMode(int mode) {
        if (mode) throw std::invalid_argument("transfer modes require CUDA");
    }
    virtual bool reusesSourceLevels() const { return false; }
    virtual bool prepareSource(fpm::ImageView, bool, int, std::vector<fpm::Image>&, double*) { return false; }
    virtual void releaseHostSource() noexcept {}
    virtual void resetTransferMetrics() {}
    virtual FpmTransferMetricsV1 transferMetrics() const {
        FpmTransferMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value); return value;
    }
    virtual void setTransferThreads(int threads) {
        if (threads != 1) throw std::invalid_argument("transfer threads require CUDA");
    }
    virtual FpmCudaKernelMetricsV1 cudaKernelMetrics() const {
        FpmCudaKernelMetricsV1 value{}; value.version = 1; value.struct_bytes = sizeof(value); return value;
    }
};
std::unique_ptr<Backend> makeMetalBackend();
std::unique_ptr<Backend> makeCudaBackend();
}
