// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include "coarse_metal_source.hpp"
#include "fine_sampling.hpp"
#include "fine_request_profile.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fpm_coarse_detail {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
constexpr size_t WORKSPACE_LIMIT = 16 * 1024 * 1024;
constexpr size_t SOURCE_ARENA_LIMIT = 64 * 1024 * 1024;
constexpr size_t BUFFER_ALIGNMENT = 256;
size_t alignedBytes(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - (BUFFER_ALIGNMENT - 1))
        throw std::invalid_argument("Metal buffer alignment overflow");
    return (bytes + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
}
struct Config {
    uint32_t sw, tw, th, outputWidth;
    uint64_t firstWindow;
    uint32_t count, reserved;
};
struct Stats32 { uint32_t sum, square, product; };
struct Stats64 { uint64_t sum, square, product; };
struct PyramidConfig { uint32_t width, height, outputWidth, outputHeight; };
static_assert(sizeof(Config) == 32 && sizeof(Stats32) == 12 && sizeof(Stats64) == 24,
    "Metal statistics ABI changed");
static_assert(sizeof(PyramidConfig) == 16, "Metal pyramid ABI changed");
std::runtime_error metalError(const char* stage, NSError* error) {
    return std::runtime_error(std::string(stage) + ": " +
        (error ? error.localizedDescription.UTF8String : "unknown Metal failure"));
}
// The operation order intentionally matches src/ncc.cpp exactly. Metal only
// computes exact integer statistics; no FP32 normalization or thresholding.
double normalize(double sum, double square, double product,
    const fpm::TemplateLevel& templ, double area)
{
    const double variance = std::max(0.0, square - sum * sum / area);
    return fpm::detail::nccNormalizeProduct(product, sum, templ.mean,
        fpm::detail::nccDenominator(variance, square, templ.norm));
}

class MetalBackend final : public Backend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> stats32_, stats64_;
    id<MTLComputePipelineState> parallelStats32_, parallelStats64_;
    id<MTLComputePipelineState> stripStats32_, stripStats64_, stagedStats32_, stagedStats64_;
    int coarsePeakMode_ = 0, coarseNccMode_ = 0;
    id<MTLComputePipelineState> pyramidDirect_, pyramidHorizontal_, pyramidVertical_;
    id<MTLComputePipelineState> fineWarp_, fineStats32_, fineStats64_;
    id<MTLComputePipelineState> fineRows32_, fineRows64_, fineShared32_, fineShared64_;
    std::array<id<MTLComputePipelineState>, 2> fineStrip32_{}, fineStrip64_{};
    int fineNccMode_ = 0;
    FpmFineReuseMetricsV1 fineReuse_{};
    bool fineProfiling_ = false, fineAngleReuse_ = false;
    FpmFineSearchMetricsV1 fineSearch_{};
    FineRequestProfiler fineRequestProfiler_;
    id<MTLBuffer> fineRows_;
    id<MTLBuffer> source_, batchSource_, template_, workspace_;
    id<MTLBuffer> pyramidInput_, pyramidOutput_, pyramidScratch_;
    const fpm::TemplateLevel* uploadedTemplate_ = nullptr;
    FpmCoarseMetrics metrics_{};
    FpmCoarseExecutionConfigV1 execution_{1, sizeof(FpmCoarseExecutionConfigV1), 0, 1, 0, 0};
    FpmCoarseExecutionMetricsV1 executionMetrics_{};
    FpmFineMetricsV1 fineMetrics_{};
    size_t fineBatchSize_ = 0, nextFineSourceBuffer_ = 0;
    id<MTLBuffer> fineRois_, fineGuards_, fineDescriptors_, fineStatistics_, fineTemplate_;
    const fpm::TemplateLevel* uploadedFineTemplate_ = nullptr;
    struct FineSource {
        fpm::ImageView view;
        id<MTLBuffer> buffer;
        size_t offset;
    };
    std::vector<FineSource> fineSources_;
    // Only these source buffers are owned exclusively by fine search. Aliases
    // of the already-accounted pyramid input/output must not be counted again.
    std::vector<id<MTLBuffer>> fineSourceBuffers_;
    struct CoarseSlot {
        id<MTLBuffer> source = nil, statistics = nil, templ = nil;
        id<MTLCommandBuffer> command = nil;
        struct Item { int width, height; size_t windows, statisticsOffset; };
        std::vector<Item> items;
        std::vector<fpm::detail::ScoreMap> fallback;
        // Only the two learned scalars are used; image stays empty. The GPU
        // template pixels live in this slot's owned Metal buffer.
        fpm::TemplateLevel normalizationTemplate;
        double area = 0;
        uint64_t sequence = 0;
        bool occupied = false, use32 = false;
    };
    std::array<CoarseSlot, 2> coarseSlots_;
    bool coarseOverlap_ = false;
    uint64_t coarseSequence_ = 0;
    double firstCoarseGpuStart_ = 0, lastCoarseGpuEnd_ = 0;
    FpmCoarseOverlapMetricsV1 overlapMetrics_{};

    void requireCoarseIdle() const {
        for (const auto& slot : coarseSlots_)
            if (slot.occupied) throw std::runtime_error("coarse overlap slot is still pending");
    }

    void ensure(id<MTLBuffer> __strong& buffer, size_t bytes, double* setup = nullptr) {
        if (buffer && buffer.length >= bytes) return;
        if (bytes > device_.maxBufferLength) throw std::runtime_error("Metal buffer exceeds device capacity");
        const auto start = Clock::now();
        id<MTLBuffer> replacement = [device_ newBufferWithLength:bytes
            options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
        if (!replacement) throw std::runtime_error("Metal buffer allocation failed");
        buffer = replacement;
        (setup ? *setup : metrics_.setup_ms) += elapsed(start);
    }
    FineSource fineSource(fpm::ImageView source) {
        for (const auto& existing : fineSources_)
            if (existing.view.data == source.data && existing.view.width == source.width &&
                existing.view.height == source.height && existing.view.stride == source.stride)
                return existing;
        if (nextFineSourceBuffer_ == fineSourceBuffers_.size()) fineSourceBuffers_.push_back(nil);
        auto& buffer = fineSourceBuffers_[nextFineSourceBuffer_++];
        ensure(buffer, size_t(source.width) * source.height, &fineMetrics_.setup_ms);
        const auto upload = Clock::now();
        for (int y = 0; y < source.height; ++y)
            std::memcpy(static_cast<uint8_t*>(buffer.contents) + size_t(y) * source.width,
                source.data + size_t(y) * source.stride, source.width);
        fineMetrics_.upload_ms += elapsed(upload);
        ++fineMetrics_.source_uploads;
        fineMetrics_.source_upload_bytes += uint64_t(source.width) * source.height;
        fineSources_.push_back({source, buffer, 0});
        return fineSources_.back();
    }
    void finishFineCommand(id<MTLCommandBuffer> command, const char* stage) {
        const auto wait = Clock::now();
        [command commit]; [command waitUntilCompleted];
        fineMetrics_.wait_ms += elapsed(wait); ++fineMetrics_.command_buffers;
        if (command.status != MTLCommandBufferStatusCompleted) throw metalError(stage, command.error);
        if (command.GPUEndTime > command.GPUStartTime && command.GPUStartTime > 0) {
            if (fineMetrics_.device_ms >= 0)
                fineMetrics_.device_ms += 1000 * (command.GPUEndTime - command.GPUStartTime);
        } else fineMetrics_.device_ms = -1;
    }
    id<MTLComputePipelineState> coarsePipeline(bool use32) const {
        if (coarseNccMode_ == 1) return use32 ? parallelStats32_ : parallelStats64_;
        if (coarseNccMode_ == 2) return use32 ? stripStats32_ : stripStats64_;
        if (coarseNccMode_ == 3) return use32 ? stagedStats32_ : stagedStats64_;
        return use32 ? stats32_ : stats64_;
    }
    void dispatchCoarse(id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline, const Config& config) const
    {
        const size_t windows = config.count;
        if (coarseNccMode_ >= 2) {
            const uint64_t firstRow = config.firstWindow / config.outputWidth;
            const uint64_t lastRow = (config.firstWindow + windows - 1) / config.outputWidth;
            const size_t rows = size_t(lastRow - firstRow + 1);
            if (coarseNccMode_ == 2) {
                [encoder dispatchThreads:MTLSizeMake((size_t(config.outputWidth) + 3) / 4, rows, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
            } else {
                // Full groups are required: no edge thread may skip a barrier.
                [encoder dispatchThreadgroups:MTLSizeMake((size_t(config.outputWidth) + 7) / 8,
                    (rows + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            }
            return;
        }
        if (coarseNccMode_ == 0) {
            const NSUInteger threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(windows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            return;
        }
        const NSUInteger simdWidth = pipeline.threadExecutionWidth;
        if (!simdWidth || simdWidth > 1024 || pipeline.maxTotalThreadsPerThreadgroup < simdWidth)
            throw std::runtime_error("coarse parallel NCC requires complete SIMD groups of at most 1024 lanes");
        const NSUInteger groups = std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / simdWidth);
        const NSUInteger threadgroups = (windows + groups - 1) / groups;
        // Shader grid indices use uint. The 16 MiB statistics limit normally
        // makes this automatic; retain an explicit guard if limits change.
        if (threadgroups > std::numeric_limits<uint32_t>::max() / (groups * simdWidth))
            throw std::invalid_argument("coarse parallel NCC grid exceeds uint32 indices");
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(groups * simdWidth, 1, 1)];
    }
public:
    ~MetalBackend() override { drainCoarseBatches(); }
    MetalBackend() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("Metal device is unavailable");
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Metal command queue creation failed");
            MTLCompileOptions* options = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device_ newLibraryWithSource:
                [NSString stringWithUTF8String:fpmCoarseMetalSource] options:options error:&error];
            if (!library) throw metalError("coarse shader compilation", error);
            library_ = library;
            stats32_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"coarse_stats32"] error:&error];
            if (!stats32_) throw metalError("coarse uint32 pipeline", error);
            stats64_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"coarse_stats64"] error:&error];
            if (!stats64_) throw metalError("coarse uint64 pipeline", error);
            pyramidDirect_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pyramid_direct"] error:&error];
            if (!pyramidDirect_) throw metalError("pyramid direct pipeline", error);
            pyramidHorizontal_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pyramid_horizontal"] error:&error];
            if (!pyramidHorizontal_) throw metalError("pyramid horizontal pipeline", error);
            pyramidVertical_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pyramid_vertical"] error:&error];
            if (!pyramidVertical_) throw metalError("pyramid vertical pipeline", error);
            fineWarp_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"fine_guarded_warp"] error:&error];
            if (!fineWarp_) throw metalError("fine guarded warp pipeline", error);
            fineStats32_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"fine_stats32"] error:&error];
            if (!fineStats32_) throw metalError("fine uint32 pipeline", error);
            fineStats64_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"fine_stats64"] error:&error];
            if (!fineStats64_) throw metalError("fine uint64 pipeline", error);
        }
    }
    void resetMetrics() override {
        drainCoarseBatches();
        metrics_ = {}; executionMetrics_ = {}; fineMetrics_ = {}; fineReuse_ = {};
        fineSearch_ = {}; fineRequestProfiler_.reset();
        overlapMetrics_ = {}; firstCoarseGpuStart_ = lastCoarseGpuEnd_ = 0;
        // A shared frame may visit new pattern objects at recycled addresses.
        // Revalidate template content once per model; source-frame identities
        // remain valid until explicit invalidateFineSource/prepare.
        uploadedTemplate_ = nullptr; uploadedFineTemplate_ = nullptr;
    }
    void invalidateFineSource() override {
        drainCoarseBatches();
        fineSources_.clear(); nextFineSourceBuffer_ = 0;
        fineSearch_ = {}; fineRequestProfiler_.reset();
    }
    void setCoarseOptimization(int peakMode, int nccMode) override {
        requireCoarseIdle();
        if (peakMode < 0 || peakMode > 1 || nccMode < 0 || nccMode > 3)
            throw std::invalid_argument("coarse peak mode must be 0..1 and NCC mode 0..3");
        if (nccMode == 1 && !parallelStats32_) {
            @autoreleasepool {
                NSError* error = nil;
                const id<MTLComputePipelineState> stats32 = [device_ newComputePipelineStateWithFunction:
                    [library_ newFunctionWithName:@"coarse_parallel_stats32"] error:&error];
                if (!stats32) throw metalError("coarse parallel uint32 pipeline", error);
                const id<MTLComputePipelineState> stats64 = [device_ newComputePipelineStateWithFunction:
                    [library_ newFunctionWithName:@"coarse_parallel_stats64"] error:&error];
                if (!stats64) throw metalError("coarse parallel uint64 pipeline", error);
                parallelStats32_ = stats32; parallelStats64_ = stats64;
            }
        }
        if (nccMode >= 2 && !(nccMode == 2 ? stripStats32_ : stagedStats32_)) {
            @autoreleasepool {
                NSError* error = nil;
                NSString* name32 = nccMode == 2 ? @"coarse_strip4_32" : @"coarse_staged8_32";
                NSString* name64 = nccMode == 2 ? @"coarse_strip4_64" : @"coarse_staged8_64";
                id<MTLComputePipelineState> p32 = [device_ newComputePipelineStateWithFunction:
                    [library_ newFunctionWithName:name32] error:&error];
                if (!p32) throw metalError("coarse tiled uint32 pipeline", error);
                id<MTLComputePipelineState> p64 = [device_ newComputePipelineStateWithFunction:
                    [library_ newFunctionWithName:name64] error:&error];
                if (!p64) throw metalError("coarse tiled uint64 pipeline", error);
                for (id<MTLComputePipelineState> pipeline : {p32, p64}) {
                    if (pipeline.maxTotalThreadsPerThreadgroup < 64 ||
                        pipeline.staticThreadgroupMemoryLength > device_.maxThreadgroupMemoryLength)
                        throw std::runtime_error("coarse tiled NCC exceeds device threadgroup limits");
                }
                if (nccMode == 2) { stripStats32_ = p32; stripStats64_ = p64; }
                else { stagedStats32_ = p32; stagedStats64_ = p64; }
            }
        }
        coarsePeakMode_ = peakMode; coarseNccMode_ = nccMode;
    }
    bool fastCoarsePeaks() const override { return coarsePeakMode_ == 1; }
    void setCoarseOverlap(bool enabled) override {
        requireCoarseIdle(); coarseOverlap_ = enabled;
    }
    bool coarseOverlapEnabled() const override { return coarseOverlap_; }
    FpmCoarseOverlapMetricsV1 coarseOverlapMetrics() const override {
        auto m = overlapMetrics_;
        m.version = 1; m.struct_bytes = sizeof(m); m.enabled = coarseOverlap_ ? 1 : 0;
        const auto& a = coarseSlots_[0]; const auto& b = coarseSlots_[1];
        m.slot0_source_bytes = a.source ? a.source.length : 0;
        m.slot0_statistics_bytes = a.statistics ? a.statistics.length : 0;
        m.slot0_template_bytes = a.templ ? a.templ.length : 0;
        m.slot1_source_bytes = b.source ? b.source.length : 0;
        m.slot1_statistics_bytes = b.statistics ? b.statistics.length : 0;
        m.slot1_template_bytes = b.templ ? b.templ.length : 0;
        m.retained_gpu_bytes = m.slot0_source_bytes + m.slot0_statistics_bytes + m.slot0_template_bytes +
            m.slot1_source_bytes + m.slot1_statistics_bytes + m.slot1_template_bytes;
        if (m.device_ms < 0) m.device_span_ms = -1;
        else m.device_span_ms = firstCoarseGpuStart_ > 0 ? 1000*(lastCoarseGpuEnd_-firstCoarseGpuStart_) : 0;
        return m;
    }
    void drainCoarseBatches() noexcept override {
        for (auto& slot : coarseSlots_) {
            if (slot.command) {
                if (slot.command.status != MTLCommandBufferStatusCompleted &&
                    slot.command.status != MTLCommandBufferStatusError)
                    [slot.command waitUntilCompleted];
                ++overlapMetrics_.drained_batches;
            }
            slot.command = nil; slot.occupied = false;
            slot.items.clear(); slot.fallback.clear();
        }
    }
    void setFineBatchSize(size_t count) override {
        if (count > 1024) throw std::invalid_argument("fine batch size must be in 0..1024");
        fineBatchSize_ = count;
    }
    size_t fineBatchSize() const override { return fineBatchSize_; }
    void setFineNccMode(int mode) override {
        requireCoarseIdle();
        if (mode < 0 || mode > 3) throw std::invalid_argument("fine NCC mode must be in 0..3");
        if (mode == 1 && !fineRows32_) {
            @autoreleasepool {
                NSError* error = nil;
                auto make = [&](NSString* name) {
                    id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:
                        [library_ newFunctionWithName:name] error:&error];
                    if (!pipeline) throw metalError("fine shared statistics pipeline", error);
                    const auto width = pipeline.threadExecutionWidth;
                    if (!width || width > 1024 || pipeline.maxTotalThreadsPerThreadgroup < width)
                        throw std::runtime_error("fine shared statistics SIMD width exceeds exact reduction limits");
                    return pipeline;
                };
                // Publish only after all four pipelines are available.
                auto row32 = make(@"fine_rows32"), row64 = make(@"fine_rows64");
                auto stats32 = make(@"fine_shared_stats32"), stats64 = make(@"fine_shared_stats64");
                fineRows32_ = row32; fineRows64_ = row64;
                fineShared32_ = stats32; fineShared64_ = stats64;
            }
        }
        if (mode >= 2 && !fineStrip32_[mode - 2]) {
            @autoreleasepool {
                auto make = [&](NSString* name) {
                    NSError* error = nil;
                    id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:
                        [library_ newFunctionWithName:name] error:&error];
                    if (!pipeline) throw metalError("fine strip statistics pipeline", error);
                    const auto width = pipeline.threadExecutionWidth;
                    if (!width || width > 1024 || pipeline.maxTotalThreadsPerThreadgroup < width)
                        throw std::runtime_error("fine strip SIMD width exceeds exact reduction limits");
                    return pipeline;
                };
                auto stats32 = make(mode == 2 ? @"fine_strip_chunk32_stats32" : @"fine_strip_chunk16_stats32");
                auto stats64 = make(mode == 2 ? @"fine_strip_chunk32_stats64" : @"fine_strip_chunk16_stats64");
                fineStrip32_[mode - 2] = stats32; fineStrip64_[mode - 2] = stats64;
            }
        }
        fineNccMode_ = mode;
        if (mode != 1) fineRows_ = nil;
    }
    void setFineSearchOptions(bool profiling, bool reuse) override {
        requireCoarseIdle();
        if (profiling != fineProfiling_ || reuse != fineAngleReuse_) {
            fineSearch_ = {}; fineRequestProfiler_.reset();
        }
        fineProfiling_ = profiling; fineAngleReuse_ = reuse;
    }
    bool fineAngleReuseEnabled() const override { return fineAngleReuse_; }
    void recordFineAngleReuse(uint64_t requested, uint64_t reused, uint64_t cacheBytes) override {
        fineSearch_.logical_requests += requested; fineSearch_.reused_maps += reused;
        fineSearch_.cache_peak_bytes = std::max(fineSearch_.cache_peak_bytes, cacheBytes);
    }
    FpmFineSearchMetricsV1 fineSearchMetrics() const override {
        auto m = fineSearch_; m.version = 1; m.struct_bytes = sizeof(m);
        m.profiling_enabled = fineProfiling_; m.angle_reuse_enabled = fineAngleReuse_;
        if (fineProfiling_) {
            const auto s = fineRequestProfiler_.stats();
            m.calls = s.calls; m.requests = s.requests;
            m.within_call_duplicates = s.within_call_duplicates; m.cross_call_duplicates = s.cross_call_duplicates;
            m.untracked_call_keys = s.untracked_call_keys; m.tracked_unique_keys = s.tracked_unique_keys;
            m.history_limit_keys = s.history_limit_keys; m.history_key_payload_bytes = s.history_key_payload_bytes;
            m.history_bucket_count = s.history_bucket_count; m.peak_call_key_capacity_bytes = s.peak_call_key_capacity_bytes;
        } else m.warp_device_ms = -1;
        return m;
    }
    FpmFineReuseMetricsV1 fineReuseMetrics() const override {
        auto m = fineReuse_; m.version = 1; m.struct_bytes = sizeof(m); m.ncc_mode = fineNccMode_;
        m.row_buffer_bytes = fineRows_ ? fineRows_.length : 0;
        return m;
    }
    void beginFineMatch(const std::vector<fpm::Image>&) override {
        // Each new model must upload its template even if its address was reused.
        // Do not invalidate immutable source levels shared by other scale models.
        uploadedFineTemplate_ = nullptr;
    }
    FpmFineMetricsV1 fineMetrics() const override {
        auto m = fineMetrics_;
        m.version = 1; m.struct_bytes = sizeof(m);
        m.source_buffer_bytes = 0;
        for (const auto& buffer : fineSourceBuffers_) if (buffer) m.source_buffer_bytes += buffer.length;
        m.roi_buffer_bytes = fineRois_ ? fineRois_.length : 0;
        m.guard_buffer_bytes = fineGuards_ ? fineGuards_.length : 0;
        m.descriptor_buffer_bytes = fineDescriptors_ ? fineDescriptors_.length : 0;
        m.statistics_buffer_bytes = (fineStatistics_ ? fineStatistics_.length : 0) +
            (fineRows_ ? fineRows_.length : 0);
        m.template_buffer_bytes = fineTemplate_ ? fineTemplate_.length : 0;
        m.retained_gpu_bytes = m.source_buffer_bytes + m.roi_buffer_bytes + m.guard_buffer_bytes +
            m.descriptor_buffer_bytes + m.statistics_buffer_bytes + m.template_buffer_bytes;
        return m;
    }
    void setExecution(const FpmCoarseExecutionConfigV1& config) override {
        requireCoarseIdle();
        if (config.version != 1 || config.struct_bytes != sizeof(config) ||
            config.pyramid_mode < 0 || config.pyramid_mode > 2 ||
            config.batch_angles < 1 || config.batch_angles > 4 ||
            config.reserved0 || config.reserved1)
            throw std::invalid_argument("invalid Metal execution configuration v1");
        execution_ = config;
    }
    size_t coarseBatchAngles() const override { return size_t(execution_.batch_angles); }
    FpmCoarseExecutionMetricsV1 executionMetrics() const override {
        auto m = executionMetrics_;
        m.version = 1; m.struct_bytes = sizeof(m);
        m.pyramid_mode = execution_.pyramid_mode; m.batch_angles = execution_.batch_angles;
        m.pyramid_input_bytes = pyramidInput_ ? pyramidInput_.length : 0;
        m.pyramid_output_bytes = pyramidOutput_ ? pyramidOutput_.length : 0;
        m.pyramid_scratch_bytes = pyramidScratch_ ? pyramidScratch_.length : 0;
        m.pyramid_buffer_bytes = m.pyramid_input_bytes + m.pyramid_output_bytes + m.pyramid_scratch_bytes;
        m.batch_source_arena_bytes = batchSource_ ? batchSource_.length : 0;
        m.batch_template_buffer_bytes = template_ ? template_.length : 0;
        m.batch_statistics_bytes = workspace_ ? workspace_.length : 0;
        return m;
    }
    FpmCoarseMetrics metrics() const override {
        auto m = metrics_;
        m.workspace_bytes = workspace_ ? workspace_.length : 0;
        m.gpu_bytes = (source_ ? source_.length : 0) + (batchSource_ ? batchSource_.length : 0) +
            (template_ ? template_.length : 0) + m.workspace_bytes + executionMetrics().pyramid_buffer_bytes;
        m.source_buffer_bytes = (source_ ? source_.length : 0) + (batchSource_ ? batchSource_.length : 0);
        m.template_buffer_bytes = template_ ? template_.length : 0;
        const auto overlap = coarseOverlapMetrics();
        m.workspace_bytes += overlap.slot0_statistics_bytes + overlap.slot1_statistics_bytes;
        m.gpu_bytes += overlap.retained_gpu_bytes;
        m.source_buffer_bytes += overlap.slot0_source_bytes + overlap.slot1_source_bytes;
        m.template_buffer_bytes += overlap.slot0_template_bytes + overlap.slot1_template_bytes;
        return m;
    }
    bool buildPyramid(std::vector<fpm::Image>& pyramid, int top) override {
        if (execution_.pyramid_mode == 0) return false;
        invalidateFineSource();
        @autoreleasepool {
            const auto start = Clock::now();
            if (pyramid.size() != 1 || top < 0)
                throw std::invalid_argument("Metal pyramid requires exactly level zero and nonnegative depth");
            const auto source = pyramid.front().view();
            fpm::detail::validateImage(source);
            if (top == 0) {
                executionMetrics_.pyramid_wall_ms += elapsed(start);
                return true;
            }
            struct Level { PyramidConfig config; size_t offset, bytes; };
            std::vector<Level> levels;
            levels.reserve(size_t(top));
            size_t outputBytes = 0, scratchBytes = 0;
            uint32_t width = uint32_t(source.width), height = uint32_t(source.height);
            for (int layer = 1; layer <= top; ++layer) {
                const uint32_t outWidth = width / 2 + width % 2;
                const uint32_t outHeight = height / 2 + height % 2;
                const size_t bytes = size_t(outWidth) * outHeight;
                const size_t offset = alignedBytes(outputBytes);
                if (bytes > std::numeric_limits<size_t>::max() - offset)
                    throw std::invalid_argument("Metal pyramid allocation overflow");
                levels.push_back({{width, height, outWidth, outHeight}, offset, bytes});
                outputBytes = offset + bytes;
                scratchBytes = std::max(scratchBytes, size_t(height) * outWidth * sizeof(uint16_t));
                width = outWidth; height = outHeight;
            }
            ensure(pyramidInput_, size_t(source.width) * source.height, &executionMetrics_.pyramid_setup_ms);
            ensure(pyramidOutput_, outputBytes, &executionMetrics_.pyramid_setup_ms);
            if (execution_.pyramid_mode == 2)
                ensure(pyramidScratch_, scratchBytes, &executionMetrics_.pyramid_setup_ms);
            const auto upload = Clock::now();
            for (int y = 0; y < source.height; ++y)
                std::memcpy(static_cast<uint8_t*>(pyramidInput_.contents) + size_t(y) * source.width,
                    source.data + size_t(y) * source.stride, source.width);
            executionMetrics_.pyramid_upload_ms += elapsed(upload);
            const auto encode = Clock::now();
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (!command) throw std::runtime_error("Metal pyramid command buffer creation failed");
            auto dispatch = [&](id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> input, size_t inputOffset, id<MTLBuffer> output, size_t outputOffset,
                const PyramidConfig& config, uint32_t gridWidth, uint32_t gridHeight) {
                // Separate encoders plus tracked resources preserve dependencies
                // between levels and between the two separable-filter passes.
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!encoder) throw std::runtime_error("Metal pyramid encoder creation failed");
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:input offset:inputOffset atIndex:0];
                [encoder setBuffer:output offset:outputOffset atIndex:1];
                [encoder setBytes:&config length:sizeof(config) atIndex:2];
                const NSUInteger tx = std::min<NSUInteger>(16, pipeline.maxTotalThreadsPerThreadgroup);
                const NSUInteger ty = std::min<NSUInteger>(16, pipeline.maxTotalThreadsPerThreadgroup / tx);
                [encoder dispatchThreads:MTLSizeMake(gridWidth, gridHeight, 1)
                    threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
                [encoder endEncoding];
            };
            for (size_t layer = 0; layer < levels.size(); ++layer) {
                const auto& level = levels[layer];
                const id<MTLBuffer> input = layer == 0 ? pyramidInput_ : pyramidOutput_;
                const size_t inputOffset = layer == 0 ? 0 : levels[layer - 1].offset;
                if (execution_.pyramid_mode == 1) {
                    dispatch(pyramidDirect_, input, inputOffset, pyramidOutput_, level.offset,
                        level.config, level.config.outputWidth, level.config.outputHeight);
                } else {
                    dispatch(pyramidHorizontal_, input, inputOffset, pyramidScratch_, 0,
                        level.config, level.config.outputWidth, level.config.height);
                    dispatch(pyramidVertical_, pyramidScratch_, 0, pyramidOutput_, level.offset,
                        level.config, level.config.outputWidth, level.config.outputHeight);
                }
            }
            executionMetrics_.pyramid_encode_ms += elapsed(encode);
            const auto wait = Clock::now();
            [command commit]; [command waitUntilCompleted];
            executionMetrics_.pyramid_wait_ms += elapsed(wait);
            ++executionMetrics_.pyramid_command_buffers;
            if (command.status != MTLCommandBufferStatusCompleted)
                throw metalError("Metal pyramid execution", command.error);
            if (command.GPUEndTime > command.GPUStartTime && command.GPUStartTime > 0) {
                if (executionMetrics_.pyramid_device_ms >= 0)
                    executionMetrics_.pyramid_device_ms += 1000 * (command.GPUEndTime - command.GPUStartTime);
            } else executionMetrics_.pyramid_device_ms = -1;
            const auto copyout = Clock::now();
            pyramid.reserve(levels.size() + 1);
            for (const auto& level : levels) {
                pyramid.emplace_back(int(level.config.outputWidth), int(level.config.outputHeight));
                std::memcpy(pyramid.back().row(0),
                    static_cast<const uint8_t*>(pyramidOutput_.contents) + level.offset, level.bytes);
            }
            // The GPU already owns byte-identical frame levels. Fine search can
            // read these directly; only metadata is cached, never stale pixels.
            fineSources_.push_back({pyramid.front().view(), pyramidInput_, 0});
            for (size_t layer = 0; layer < levels.size(); ++layer)
                fineSources_.push_back({pyramid[layer + 1].view(), pyramidOutput_, levels[layer].offset});
            executionMetrics_.pyramid_copyout_ms += elapsed(copyout);
            executionMetrics_.pyramid_levels += levels.size();
            executionMetrics_.pyramid_wall_ms += elapsed(start);
            return true;
        }
    }
    std::vector<fpm::detail::ScoreMap> computeFineBatch(fpm::ImageView source,
        const fpm::TemplateLevel& templ, const std::vector<fpm::detail::FineRequest>& requests,
        double* rotationWallMs) override
    {
        if (rotationWallMs) *rotationWallMs = 0;
        if (!fineBatchSize_) throw std::runtime_error("fine GPU batching was not enabled");
        @autoreleasepool {
            const auto start = Clock::now();
            ++fineMetrics_.calls;
            fpm::detail::validateImage(source);
            const int tw = templ.image.width(), th = templ.image.height();
            if (tw < 1 || th < 1 || tw > INT_MAX - 2 * fpm::detail::PADDING ||
                th > INT_MAX - 2 * fpm::detail::PADDING)
                throw std::invalid_argument("fine template dimensions exceed ROI limits");
            const int rw = tw + 2 * fpm::detail::PADDING, rh = th + 2 * fpm::detail::PADDING;
            const size_t roiPixels = size_t(rw) * rh, roiStride = alignedBytes(roiPixels);
            // A hard arena limit is explicit, rather than an unreported CPU
            // fallback or an unbounded allocation for an oversized template.
            if (roiStride > SOURCE_ARENA_LIMIT)
                throw std::invalid_argument("fine ROI exceeds the 64 MiB Metal arena limit");
            const uint64_t area = uint64_t(tw) * th;
            if (area > (uint64_t(1) << 53) / 65025)
                throw std::invalid_argument("template exceeds exact NCC statistics range");
            constexpr int outputWidth = 2 * fpm::detail::PADDING + 1;
            constexpr size_t windowsPerMap = size_t(outputWidth) * outputWidth;
            const bool use32 = area <= std::numeric_limits<uint32_t>::max() / uint64_t(65025);
            const size_t itemBytes = use32 ? sizeof(Stats32) : sizeof(Stats64);
            const size_t rowBytesPerMap = size_t(rh) * outputWidth * (use32 ? 8 : 16);
            if (fineNccMode_ == 1 && rowBytesPerMap > WORKSPACE_LIMIT)
                throw std::invalid_argument("fine row statistics exceed the 16 MiB arena limit");
            const size_t maxMaps = std::min({fineBatchSize_, SOURCE_ARENA_LIMIT / roiStride,
                WORKSPACE_LIMIT / (windowsPerMap * itemBytes),
                fineNccMode_ == 1 ? WORKSPACE_LIMIT / rowBytesPerMap : fineBatchSize_});
            std::vector<fpm::detail::ScoreMap> maps;
            maps.reserve(requests.size());
            if (requests.empty()) {
                fineMetrics_.wall_ms += elapsed(start);
                return maps;
            }
            const auto resident = fineSource(source);
            ensure(fineTemplate_, size_t(area), &fineMetrics_.setup_ms);
            if (uploadedFineTemplate_ != &templ) {
                const auto upload = Clock::now();
                std::memcpy(fineTemplate_.contents, templ.image.view().data, size_t(area));
                fineMetrics_.upload_ms += elapsed(upload);
                uploadedFineTemplate_ = &templ;
            }
            if (fineProfiling_) {
                const auto observation = Clock::now();
                fineRequestProfiler_.observe(source, templ, requests);
                fineSearch_.observe_ms += elapsed(observation);
            }
            double rotationTotal = 0;
            for (size_t first = 0; first < requests.size(); first += maxMaps) {
                const size_t count = std::min(maxMaps, requests.size() - first);
                const size_t pixels = count * roiPixels, guardWords = (pixels + 31) / 32;
                ensure(fineRois_, count * roiStride, &fineMetrics_.setup_ms);
                ensure(fineGuards_, guardWords * sizeof(uint32_t), &fineMetrics_.setup_ms);
                ensure(fineDescriptors_, count * sizeof(FineWarpDescriptor), &fineMetrics_.setup_ms);
                ensure(fineStatistics_, count * windowsPerMap * itemBytes, &fineMetrics_.setup_ms);
                if (fineNccMode_ == 1) ensure(fineRows_, count * rowBytesPerMap, &fineMetrics_.setup_ms);
                const auto rotation = Clock::now();
                std::vector<FineWarpPlan> plans;
                plans.reserve(count);
                for (size_t i = 0; i < count; ++i) plans.push_back(prepareFineWarp(requests[first + i], rw, rh));
                if (fineProfiling_) fineSearch_.plan_ms += elapsed(rotation);
                const auto upload = Clock::now();
                auto* descriptors = static_cast<FineWarpDescriptor*>(fineDescriptors_.contents);
                for (size_t i = 0; i < count; ++i) descriptors[i] = plans[i].descriptor;
                std::memset(fineGuards_.contents, 0, guardWords * sizeof(uint32_t));
                const double descriptorMs = elapsed(upload);
                fineMetrics_.upload_ms += descriptorMs;
                if (fineProfiling_) fineSearch_.descriptor_ms += descriptorMs;
                const FineWarpConfig warpConfig{uint32_t(source.width), uint32_t(source.height),
                    uint32_t(rw), uint32_t(rh), uint32_t(roiPixels), uint32_t(roiStride), uint32_t(count), 0};
                const auto warpEncode = Clock::now();
                id<MTLCommandBuffer> warpCommand = [queue_ commandBuffer];
                if (!warpCommand) throw std::runtime_error("fine warp command buffer creation failed");
                id<MTLComputeCommandEncoder> warpEncoder = [warpCommand computeCommandEncoder];
                if (!warpEncoder) throw std::runtime_error("fine warp encoder creation failed");
                [warpEncoder setComputePipelineState:fineWarp_];
                [warpEncoder setBuffer:resident.buffer offset:resident.offset atIndex:0];
                [warpEncoder setBuffer:fineRois_ offset:0 atIndex:1];
                [warpEncoder setBuffer:fineGuards_ offset:0 atIndex:2];
                [warpEncoder setBuffer:fineDescriptors_ offset:0 atIndex:3];
                [warpEncoder setBytes:&warpConfig length:sizeof(warpConfig) atIndex:4];
                [warpEncoder dispatchThreads:MTLSizeMake(pixels, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,
                        fineWarp_.maxTotalThreadsPerThreadgroup), 1, 1)];
                [warpEncoder endEncoding];
                const double warpEncodeMs = elapsed(warpEncode);
                fineMetrics_.encode_ms += warpEncodeMs;
                const double beforeWarpWait = fineMetrics_.wait_ms;
                finishFineCommand(warpCommand, "fine guarded warp execution");
                if (fineProfiling_) {
                    fineSearch_.warp_encode_ms += warpEncodeMs;
                    fineSearch_.warp_wait_ms += fineMetrics_.wait_ms - beforeWarpWait;
                    ++fineSearch_.warp_batches;
                    if (warpCommand.GPUEndTime > warpCommand.GPUStartTime && warpCommand.GPUStartTime > 0) {
                        if (fineSearch_.warp_device_ms >= 0)
                            fineSearch_.warp_device_ms += 1000 * (warpCommand.GPUEndTime - warpCommand.GPUStartTime);
                    } else fineSearch_.warp_device_ms = -1;
                }

                const auto repair = Clock::now();
                const uint64_t beforeRepairPixels = fineMetrics_.repair_pixels;
                auto* roiData = static_cast<uint8_t*>(fineRois_.contents);
                for (size_t i = 0; i < count; ++i) if (plans[i].descriptor.repairAll) {
                    ++fineMetrics_.repair_all_maps;
                    auto* output = roiData + i * roiStride;
                    for (int y = 0; y < rh; ++y) for (int x = 0; x < rw; ++x)
                        output[size_t(y) * rw + x] = sampleFineLegacyPixel(source, plans[i], x, y);
                    fineMetrics_.repair_pixels += roiPixels;
                }
                const auto scan = fineProfiling_ ? Clock::now() : Clock::time_point{};
                if (fineProfiling_) {
                    fineSearch_.repair_all_ms += std::chrono::duration<double, std::milli>(scan - repair).count();
                    fineSearch_.guard_words += guardWords;
                }
                const auto* guards = static_cast<const uint32_t*>(fineGuards_.contents);
                for (size_t word = 0; word < guardWords; ++word) {
                    uint32_t bits = guards[word];
                    if (fineProfiling_ && bits) ++fineSearch_.nonzero_guard_words;
                    while (bits) {
                        const unsigned bit = unsigned(__builtin_ctz(bits));
                        const size_t index = word * 32 + bit;
                        const size_t map = index / roiPixels, pixel = index % roiPixels;
                        // Padding bits are initialized to zero and no shader
                        // invocation writes beyond count*roiPixels.
                        roiData[map * roiStride + pixel] = sampleFineLegacyPixel(source, plans[map],
                            int(pixel % rw), int(pixel / rw));
                        ++fineMetrics_.repair_pixels;
                        bits &= bits - 1;
                    }
                }
                if (fineProfiling_) {
                    fineSearch_.guard_scan_ms += elapsed(scan);
                    if (fineMetrics_.repair_pixels == beforeRepairPixels) ++fineSearch_.zero_repair_batches;
                }
                fineMetrics_.repair_ms += elapsed(repair);
                const double rotationElapsed = elapsed(rotation);
                rotationTotal += rotationElapsed; fineMetrics_.rotation_ms += rotationElapsed;
                fineMetrics_.roi_pixels += pixels;

                const auto ncc = Clock::now();
                const auto nccEncode = Clock::now();
                const id<MTLComputePipelineState> pipeline = fineNccMode_ >= 2 ?
                    (use32 ? fineStrip32_[fineNccMode_ - 2] : fineStrip64_[fineNccMode_ - 2]) :
                    (fineNccMode_ == 1 ? (use32 ? fineShared32_ : fineShared64_) : (use32 ? fineStats32_ : fineStats64_));
                const NSUInteger simdWidth = pipeline.threadExecutionWidth;
                if (!simdWidth || simdWidth > 1024 || pipeline.maxTotalThreadsPerThreadgroup < simdWidth)
                    throw std::runtime_error("fine NCC SIMD width exceeds exact reduction limits");
                const NSUInteger groupsPerThreadgroup = std::min<NSUInteger>(8,
                    pipeline.maxTotalThreadsPerThreadgroup / simdWidth);
                const FineStatsConfig statsConfig{uint32_t(rw), uint32_t(tw), uint32_t(th), uint32_t(outputWidth),
                    uint32_t(roiStride), uint32_t(windowsPerMap), uint32_t(count), 0};
                id<MTLCommandBuffer> statsCommand = [queue_ commandBuffer];
                if (!statsCommand) throw std::runtime_error("fine NCC command buffer creation failed");
                if (fineNccMode_ == 1) {
                    const auto rowPipeline = use32 ? fineRows32_ : fineRows64_;
                    const auto rowWidth = rowPipeline.threadExecutionWidth;
                    const auto rowGroups = std::min<NSUInteger>(8, rowPipeline.maxTotalThreadsPerThreadgroup / rowWidth);
                    const size_t groups = (count * size_t(rh) + rowGroups - 1) / rowGroups;
                    if (groups > std::numeric_limits<uint32_t>::max() / (rowGroups * rowWidth))
                        throw std::invalid_argument("fine row dispatch exceeds uint32 indices");
                    id<MTLComputeCommandEncoder> rowEncoder = [statsCommand computeCommandEncoder];
                    if (!rowEncoder) throw std::runtime_error("fine row encoder creation failed");
                    [rowEncoder setComputePipelineState:rowPipeline];
                    [rowEncoder setBuffer:fineRois_ offset:0 atIndex:0];
                    [rowEncoder setBuffer:fineRows_ offset:0 atIndex:1];
                    [rowEncoder setBytes:&statsConfig length:sizeof(statsConfig) atIndex:2];
                    [rowEncoder dispatchThreadgroups:MTLSizeMake(groups,1,1)
                        threadsPerThreadgroup:MTLSizeMake(rowGroups * rowWidth,1,1)];
                    [rowEncoder endEncoding];
                    ++fineReuse_.row_dispatches;
                    // Shared buffers use tracked hazards. A following encoder
                    // in the SAME command buffer observes the completed rows;
                    // neither a second commit nor a CPU wait is needed.
                }
                id<MTLComputeCommandEncoder> statsEncoder = [statsCommand computeCommandEncoder];
                if (!statsEncoder) throw std::runtime_error("fine NCC encoder creation failed");
                [statsEncoder setComputePipelineState:pipeline];
                [statsEncoder setBuffer:fineRois_ offset:0 atIndex:0];
                [statsEncoder setBuffer:fineTemplate_ offset:0 atIndex:1];
                [statsEncoder setBuffer:fineStatistics_ offset:0 atIndex:2];
                [statsEncoder setBytes:&statsConfig length:sizeof(statsConfig) atIndex:3];
                if (fineNccMode_ == 1) [statsEncoder setBuffer:fineRows_ offset:0 atIndex:4];
                // Strip mode owns 4+3 windows per row: 14 complete groups/map.
                // Original modes own one window/group. Full threadgroups keep
                // reductions complete; padded groups return uniformly.
                static_assert(outputWidth == 7, "fine strip kernels require a 7x7 map");
                const size_t groupsPerMap = fineNccMode_ >= 2 ? size_t(outputWidth) * 2 : windowsPerMap;
                [statsEncoder dispatchThreadgroups:MTLSizeMake(
                    (count * groupsPerMap + groupsPerThreadgroup - 1) / groupsPerThreadgroup, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(groupsPerThreadgroup * simdWidth, 1, 1)];
                [statsEncoder endEncoding];
                const double nccEncodeMs = elapsed(nccEncode);
                fineMetrics_.encode_ms += nccEncodeMs;
                if (fineProfiling_) fineSearch_.ncc_encode_ms += nccEncodeMs;
                const double beforeNccWait = fineMetrics_.wait_ms;
                // CPU repairs are complete before this commit, so shared ROI
                // writes are visible to the following GPU statistics dispatch.
                finishFineCommand(statsCommand, "fine NCC execution");
                if (fineProfiling_) fineSearch_.ncc_wait_ms += fineMetrics_.wait_ms - beforeNccWait;
                ++fineReuse_.statistics_dispatches;
                if (statsCommand.GPUEndTime > statsCommand.GPUStartTime && statsCommand.GPUStartTime > 0) {
                    if (fineReuse_.stats_device_ms >= 0)
                        fineReuse_.stats_device_ms += 1000 * (statsCommand.GPUEndTime - statsCommand.GPUStartTime);
                } else fineReuse_.stats_device_ms = -1;
                const auto normalization = Clock::now();
                const bool degenerate = templ.norm * templ.norm / double(area) < DBL_EPSILON;
                for (size_t i = 0; i < count; ++i) {
                    fpm::detail::ScoreMap map{outputWidth, outputWidth, std::vector<double>(windowsPerMap)};
                    if (degenerate) std::fill(map.values.begin(), map.values.end(), 1.0);
                    else if (use32) {
                        const auto* stats = static_cast<const Stats32*>(fineStatistics_.contents) + i * windowsPerMap;
                        for (size_t j = 0; j < windowsPerMap; ++j)
                            map.values[j] = normalize(stats[j].sum, stats[j].square, stats[j].product, templ, double(area));
                    } else {
                        const auto* stats = static_cast<const Stats64*>(fineStatistics_.contents) + i * windowsPerMap;
                        for (size_t j = 0; j < windowsPerMap; ++j)
                            map.values[j] = normalize(double(stats[j].sum), double(stats[j].square),
                                double(stats[j].product), templ, double(area));
                    }
                    maps.push_back(std::move(map));
                }
                fineMetrics_.normalization_ms += elapsed(normalization);
                fineMetrics_.ncc_ms += elapsed(ncc);
                fineMetrics_.maps += count;
                fineMetrics_.peak_batch_maps = std::max<uint64_t>(fineMetrics_.peak_batch_maps, count);
            }
            if (rotationWallMs) *rotationWallMs = rotationTotal;
            fineMetrics_.wall_ms += elapsed(start);
            return maps;
        }
    }
    void beginCoarseBatch(size_t index, const std::vector<fpm::ImageView>& sources,
        const fpm::TemplateLevel& templ) override
    {
        @autoreleasepool {
            const auto start = Clock::now();
            if (!coarseOverlap_) throw std::runtime_error("coarse overlap is disabled");
            if (index >= coarseSlots_.size()) throw std::invalid_argument("coarse overlap slot must be 0 or 1");
            auto& slot = coarseSlots_[index];
            if (slot.occupied) throw std::runtime_error("coarse overlap slot is still pending");
            if (sources.empty() || sources.size() > size_t(execution_.batch_angles))
                throw std::invalid_argument("coarse overlap batch must contain 1..batch_angles sources");
            const int tw = templ.image.width(), th = templ.image.height();
            if (tw < 1 || th < 1) throw std::invalid_argument("NCC template dimensions must be positive");
            const uint64_t area = uint64_t(tw)*th;
            if (area > (uint64_t(1) << 53)/65025)
                throw std::invalid_argument("template exceeds exact NCC statistics range");
            const bool use32 = area <= std::numeric_limits<uint32_t>::max()/uint64_t(65025);
            const size_t itemBytes = use32 ? sizeof(Stats32) : sizeof(Stats64);
            const bool degenerate = templ.norm*templ.norm/double(area) < DBL_EPSILON;
            size_t sourceBytes = 0, statisticsBytes = 0;
            bool oversized = false;
            std::array<size_t, 4> sourceOffsets{};
            std::vector<CoarseSlot::Item> items;
            items.reserve(sources.size());
            for (size_t i = 0; i < sources.size(); ++i) {
                const auto& source = sources[i];
                fpm::detail::validateImage(source);
                const int width = source.width-tw+1, height = source.height-th+1;
                if (width < 1 || height < 1) throw std::invalid_argument("NCC template must fit the source");
                const size_t windows = size_t(width)*height;
                const size_t pixels = size_t(source.width)*source.height;
                const size_t sourceOffset = alignedBytes(sourceBytes), statisticsOffset = alignedBytes(statisticsBytes);
                // Keep even partially accumulated arenas bounded; once one
                // item does not fit, validate the rest without growing them.
                if (!oversized && (sourceOffset > SOURCE_ARENA_LIMIT || statisticsOffset > WORKSPACE_LIMIT ||
                    pixels > SOURCE_ARENA_LIMIT-sourceOffset || windows > (WORKSPACE_LIMIT-statisticsOffset)/itemBytes))
                    oversized = true;
                if (!oversized) {
                    sourceOffsets[i] = sourceOffset;
                    sourceBytes = sourceOffset+pixels; statisticsBytes = statisticsOffset+windows*itemBytes;
                }
                items.push_back({width, height, windows, statisticsOffset});
            }
            if (sources.size() == 1 || degenerate || oversized) {
                // Preserve the original subdivision/chunking and exact scalar
                // normalization. An outstanding other slot owns distinct buffers.
                const auto fallbackStart = Clock::now();
                slot.fallback = computeBatch(sources, templ);
                overlapMetrics_.fallback_ms += elapsed(fallbackStart);
                ++overlapMetrics_.fallback_batches; overlapMetrics_.fallback_maps += sources.size();
                if (degenerate) ++overlapMetrics_.fallback_degenerate;
                else if (oversized) ++overlapMetrics_.fallback_oversized;
                else ++overlapMetrics_.fallback_single;
                slot.sequence = ++coarseSequence_; slot.occupied = true;
                return;
            }
            ensure(slot.source, sourceBytes); ensure(slot.statistics, statisticsBytes); ensure(slot.templ, size_t(area));
            const auto copy = Clock::now();
            for (size_t i = 0; i < sources.size(); ++i) {
                const auto& source = sources[i];
                for (int y = 0; y < source.height; ++y)
                    std::memcpy(static_cast<uint8_t*>(slot.source.contents)+sourceOffsets[i]+size_t(y)*source.width,
                        source.data+size_t(y)*source.stride, source.width);
                const auto pixels = uint64_t(source.width)*source.height;
                overlapMetrics_.source_upload_bytes += pixels;
                executionMetrics_.batch_source_upload_bytes += pixels;
            }
            std::memcpy(slot.templ.contents, templ.image.view().data, size_t(area));
            overlapMetrics_.template_upload_bytes += area;
            const double copyMs = elapsed(copy);
            overlapMetrics_.copy_ms += copyMs; metrics_.upload_ms += copyMs;
            slot.normalizationTemplate.mean = templ.mean; slot.normalizationTemplate.norm = templ.norm;
            slot.area = double(area); slot.use32 = use32; slot.items = std::move(items);
            const auto encode = Clock::now();
            const id<MTLComputePipelineState> pipeline = coarsePipeline(use32);
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (!command) throw std::runtime_error("Metal asynchronous NCC command buffer creation failed");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Metal asynchronous NCC encoder creation failed");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:slot.templ offset:0 atIndex:1];
            for (size_t i = 0; i < sources.size(); ++i) {
                const auto& item = slot.items[i];
                const Config config{uint32_t(sources[i].width), uint32_t(tw), uint32_t(th),
                    uint32_t(item.width), 0, uint32_t(item.windows), 0};
                [encoder setBuffer:slot.source offset:sourceOffsets[i] atIndex:0];
                [encoder setBuffer:slot.statistics offset:item.statisticsOffset atIndex:2];
                [encoder setBytes:&config length:sizeof(config) atIndex:3];
                dispatchCoarse(encoder, pipeline, config);
            }
            [encoder endEncoding];
            const double encodeMs = elapsed(encode);
            overlapMetrics_.encode_ms += encodeMs; executionMetrics_.batch_encode_ms += encodeMs;
            size_t pending = 0;
            for (const auto& other : coarseSlots_) if (other.command) ++pending;
            if (pending) ++overlapMetrics_.submissions_with_pending;
            overlapMetrics_.max_pending = std::max<uint64_t>(overlapMetrics_.max_pending, pending+1);
            slot.command = command; slot.sequence = ++coarseSequence_; slot.occupied = true;
            [command commit];
            ++metrics_.command_buffers;
            ++overlapMetrics_.submitted_batches; overlapMetrics_.submitted_maps += sources.size();
            ++executionMetrics_.batch_calls; executionMetrics_.batch_maps += sources.size();
            executionMetrics_.max_batch_maps = std::max<uint64_t>(executionMetrics_.max_batch_maps, sources.size());
            for (const auto& item : slot.items) {
                ++metrics_.score_maps; metrics_.windows += item.windows;
                if (use32) ++metrics_.integer32_maps; else ++metrics_.integer64_maps;
                metrics_.scoremap_peak_bytes = std::max<uint64_t>(metrics_.scoremap_peak_bytes, item.windows*sizeof(double));
            }
            const double submitMs = elapsed(start);
            overlapMetrics_.submit_ms += submitMs; metrics_.ncc_ms += submitMs;
        }
    }
    std::vector<fpm::detail::ScoreMap> finishCoarseBatch(size_t index) override {
        @autoreleasepool {
            const auto start = Clock::now();
            if (index >= coarseSlots_.size()) throw std::invalid_argument("coarse overlap slot must be 0 or 1");
            auto& slot = coarseSlots_[index];
            if (!slot.occupied) throw std::runtime_error("coarse overlap slot is empty");
            for (const auto& other : coarseSlots_)
                if (other.occupied && other.sequence < slot.sequence)
                    throw std::runtime_error("coarse overlap slots must finish in submission order");
            if (!slot.command) {
                slot.occupied = false;
                return std::move(slot.fallback);
            }
            const auto wait = Clock::now();
            [slot.command waitUntilCompleted];
            const double waitMs = elapsed(wait);
            overlapMetrics_.finish_wait_ms += waitMs; metrics_.wait_ms += waitMs;
            if (slot.command.status != MTLCommandBufferStatusCompleted)
                throw metalError("Metal asynchronous NCC execution", slot.command.error);
            if (slot.command.GPUEndTime > slot.command.GPUStartTime && slot.command.GPUStartTime > 0) {
                const double gpuMs = 1000*(slot.command.GPUEndTime-slot.command.GPUStartTime);
                if (metrics_.device_ms >= 0) metrics_.device_ms += gpuMs;
                if (overlapMetrics_.device_ms >= 0) overlapMetrics_.device_ms += gpuMs;
                if (!firstCoarseGpuStart_) firstCoarseGpuStart_ = slot.command.GPUStartTime;
                lastCoarseGpuEnd_ = slot.command.GPUEndTime;
            } else metrics_.device_ms = overlapMetrics_.device_ms = -1;
            const auto normalization = Clock::now();
            std::vector<fpm::detail::ScoreMap> maps;
            maps.reserve(slot.items.size());
            for (const auto& item : slot.items) {
                fpm::detail::ScoreMap map{item.width, item.height, std::vector<double>(item.windows)};
                const auto* bytes = static_cast<const uint8_t*>(slot.statistics.contents)+item.statisticsOffset;
                if (slot.use32) {
                    const auto* stats = reinterpret_cast<const Stats32*>(bytes);
                    for (size_t i = 0; i < item.windows; ++i)
                        map.values[i] = normalize(stats[i].sum, stats[i].square, stats[i].product,
                            slot.normalizationTemplate, slot.area);
                } else {
                    const auto* stats = reinterpret_cast<const Stats64*>(bytes);
                    for (size_t i = 0; i < item.windows; ++i)
                        map.values[i] = normalize(double(stats[i].sum), double(stats[i].square), double(stats[i].product),
                            slot.normalizationTemplate, slot.area);
                }
                maps.push_back(std::move(map));
            }
            const double normalizationMs = elapsed(normalization);
            overlapMetrics_.normalization_ms += normalizationMs; metrics_.normalization_ms += normalizationMs;
            ++overlapMetrics_.completed_batches;
            slot.command = nil; slot.occupied = false; slot.items.clear();
            // Count only host work done here, excluding CPU rotation and peak
            // extraction between begin and finish. GPU duration is separate.
            metrics_.ncc_ms += elapsed(start);
            return maps;
        }
    }
    fpm::detail::ScoreMap compute(fpm::ImageView source, const fpm::TemplateLevel& templ) override {
        @autoreleasepool {
            const auto start = Clock::now();
            fpm::detail::validateImage(source);
            const int tw = templ.image.width(), th = templ.image.height();
            const int width = source.width - tw + 1, height = source.height - th + 1;
            if (tw < 1 || th < 1 || width < 1 || height < 1)
                throw std::invalid_argument("NCC template must fit the source");
            const size_t windows = size_t(width) * height;
            const uint64_t area = uint64_t(tw) * th;
            // Exact uint64 -> binary64 statistics under all practical Metal
            // buffer limits; explicit guard documents the equality contract.
            if (area > (uint64_t(1) << 53) / 65025)
                throw std::invalid_argument("template exceeds exact NCC statistics range");
            fpm::detail::ScoreMap map{width, height, std::vector<double>(windows)};
            metrics_.scoremap_peak_bytes = std::max<uint64_t>(metrics_.scoremap_peak_bytes, windows * sizeof(double));
            ++metrics_.score_maps; metrics_.windows += windows;
            if (templ.norm * templ.norm / double(area) < DBL_EPSILON) {
                std::fill(map.values.begin(), map.values.end(), 1.0);
                metrics_.ncc_ms += elapsed(start);
                return map;
            }
            const bool use32 = area <= std::numeric_limits<uint32_t>::max() / uint64_t(65025);
            if (use32) ++metrics_.integer32_maps; else ++metrics_.integer64_maps;
            const size_t itemBytes = use32 ? sizeof(Stats32) : sizeof(Stats64);
            const size_t batch = std::min(windows, WORKSPACE_LIMIT / itemBytes);
            ensure(source_, size_t(source.width) * source.height);
            ensure(template_, size_t(area));
            ensure(workspace_, batch * itemBytes);
            const auto upload = Clock::now();
            for (int y = 0; y < source.height; ++y)
                std::memcpy(static_cast<uint8_t*>(source_.contents) + size_t(y) * source.width,
                    source.data + size_t(y) * source.stride, source.width);
            if (uploadedTemplate_ != &templ) {
                std::memcpy(template_.contents, templ.image.view().data, size_t(area));
                uploadedTemplate_ = &templ;
            }
            metrics_.upload_ms += elapsed(upload);
            const id<MTLComputePipelineState> pipeline = coarsePipeline(use32);
            for (size_t first = 0; first < windows; first += batch) {
                const size_t count = std::min(batch, windows - first);
                const Config config{uint32_t(source.width), uint32_t(tw), uint32_t(th),
                    uint32_t(width), uint64_t(first), uint32_t(count), 0};
                id<MTLCommandBuffer> command = [queue_ commandBuffer];
                if (!command) throw std::runtime_error("Metal command buffer creation failed");
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!encoder) throw std::runtime_error("Metal compute encoder creation failed");
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:source_ offset:0 atIndex:0];
                [encoder setBuffer:template_ offset:0 atIndex:1];
                [encoder setBuffer:workspace_ offset:0 atIndex:2];
                [encoder setBytes:&config length:sizeof(config) atIndex:3];
                dispatchCoarse(encoder, pipeline, config);
                [encoder endEncoding];
                const auto wait = Clock::now();
                [command commit]; [command waitUntilCompleted];
                metrics_.wait_ms += elapsed(wait); ++metrics_.command_buffers;
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw metalError("coarse GPU execution", command.error);
                if (command.GPUEndTime > command.GPUStartTime && command.GPUStartTime > 0) {
                    if (metrics_.device_ms >= 0)
                        metrics_.device_ms += 1000 * (command.GPUEndTime - command.GPUStartTime);
                } else metrics_.device_ms = -1;
                const auto normalization = Clock::now();
                if (use32) {
                    const auto* stats = static_cast<const Stats32*>(workspace_.contents);
                    for (size_t i = 0; i < count; ++i)
                        map.values[first + i] = normalize(stats[i].sum, stats[i].square, stats[i].product, templ, double(area));
                } else {
                    const auto* stats = static_cast<const Stats64*>(workspace_.contents);
                    for (size_t i = 0; i < count; ++i)
                        map.values[first + i] = normalize(double(stats[i].sum), double(stats[i].square),
                            double(stats[i].product), templ, double(area));
                }
                metrics_.normalization_ms += elapsed(normalization);
            }
            metrics_.ncc_ms += elapsed(start);
            return map;
        }
    }
    std::vector<fpm::detail::ScoreMap> computeBatch(const std::vector<fpm::ImageView>& sources,
        const fpm::TemplateLevel& templ) override
    {
        @autoreleasepool {
            std::vector<fpm::detail::ScoreMap> maps(sources.size());
            if (sources.empty()) return maps;
            const int tw = templ.image.width(), th = templ.image.height();
            if (tw < 1 || th < 1) throw std::invalid_argument("NCC template dimensions must be positive");
            const uint64_t area = uint64_t(tw) * th;
            if (area > (uint64_t(1) << 53) / 65025)
                throw std::invalid_argument("template exceeds exact NCC statistics range");
            const bool use32 = area <= std::numeric_limits<uint32_t>::max() / uint64_t(65025);
            const size_t itemBytes = use32 ? sizeof(Stats32) : sizeof(Stats64);
            const bool degenerate = templ.norm * templ.norm / double(area) < DBL_EPSILON;
            struct Item { int width, height; size_t windows, sourceBytes; };
            std::vector<Item> items;
            items.reserve(sources.size());
            for (const auto& source : sources) {
                fpm::detail::validateImage(source);
                const int width = source.width - tw + 1, height = source.height - th + 1;
                if (width < 1 || height < 1)
                    throw std::invalid_argument("NCC template must fit the source");
                items.push_back({width, height, size_t(width) * height,
                    size_t(source.width) * source.height});
            }
            size_t first = 0;
            while (first < sources.size()) {
                const auto start = Clock::now();
                size_t count = 0, sourceBytes = 0, statisticsBytes = 0;
                std::array<size_t, 4> sourceOffsets{}, statisticsOffsets{};
                if (!degenerate) {
                    while (count < size_t(execution_.batch_angles) && first + count < sources.size()) {
                        const auto& item = items[first + count];
                        const size_t sourceOffset = alignedBytes(sourceBytes);
                        const size_t statisticsOffset = alignedBytes(statisticsBytes);
                        // Division-based checks avoid overflowing byte products.
                        // A single oversized map uses the existing chunked path.
                        if (item.sourceBytes > SOURCE_ARENA_LIMIT - sourceOffset ||
                            item.windows > (WORKSPACE_LIMIT - statisticsOffset) / itemBytes)
                            break;
                        sourceOffsets[count] = sourceOffset;
                        statisticsOffsets[count] = statisticsOffset;
                        sourceBytes = sourceOffset + item.sourceBytes;
                        statisticsBytes = statisticsOffset + item.windows * itemBytes;
                        ++count;
                    }
                }
                if (count < 2) {
                    // This is a Metal single-map path, not a CPU fallback.
                    maps[first] = compute(sources[first], templ);
                    ++executionMetrics_.batch_calls; ++executionMetrics_.batch_maps;
                    executionMetrics_.max_batch_maps = std::max<uint64_t>(executionMetrics_.max_batch_maps, 1);
                    ++first;
                    continue;
                }
                for (size_t i = 0; i < count; ++i) {
                    const auto& item = items[first + i];
                    maps[first + i] = {item.width, item.height, std::vector<double>(item.windows)};
                    metrics_.scoremap_peak_bytes = std::max<uint64_t>(metrics_.scoremap_peak_bytes,
                        item.windows * sizeof(double));
                    ++metrics_.score_maps; metrics_.windows += item.windows;
                    if (use32) ++metrics_.integer32_maps; else ++metrics_.integer64_maps;
                }
                ensure(batchSource_, sourceBytes);
                ensure(template_, size_t(area));
                ensure(workspace_, statisticsBytes);
                const auto upload = Clock::now();
                for (size_t i = 0; i < count; ++i) {
                    const auto& source = sources[first + i];
                    for (int y = 0; y < source.height; ++y)
                        std::memcpy(static_cast<uint8_t*>(batchSource_.contents) + sourceOffsets[i] +
                            size_t(y) * source.width, source.data + size_t(y) * source.stride, source.width);
                    executionMetrics_.batch_source_upload_bytes += items[first + i].sourceBytes;
                }
                if (uploadedTemplate_ != &templ) {
                    std::memcpy(template_.contents, templ.image.view().data, size_t(area));
                    uploadedTemplate_ = &templ;
                }
                metrics_.upload_ms += elapsed(upload);
                const auto encode = Clock::now();
                const id<MTLComputePipelineState> pipeline = coarsePipeline(use32);
                id<MTLCommandBuffer> command = [queue_ commandBuffer];
                if (!command) throw std::runtime_error("Metal batched NCC command buffer creation failed");
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!encoder) throw std::runtime_error("Metal batched NCC encoder creation failed");
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:template_ offset:0 atIndex:1];
                for (size_t i = 0; i < count; ++i) {
                    const auto& source = sources[first + i];
                    const auto& item = items[first + i];
                    const Config config{uint32_t(source.width), uint32_t(tw), uint32_t(th),
                        uint32_t(item.width), 0, uint32_t(item.windows), 0};
                    [encoder setBuffer:batchSource_ offset:sourceOffsets[i] atIndex:0];
                    [encoder setBuffer:workspace_ offset:statisticsOffsets[i] atIndex:2];
                    [encoder setBytes:&config length:sizeof(config) atIndex:3];
                    dispatchCoarse(encoder, pipeline, config);
                }
                [encoder endEncoding];
                executionMetrics_.batch_encode_ms += elapsed(encode);
                const auto wait = Clock::now();
                [command commit]; [command waitUntilCompleted];
                metrics_.wait_ms += elapsed(wait); ++metrics_.command_buffers;
                ++executionMetrics_.batch_calls; executionMetrics_.batch_maps += count;
                executionMetrics_.max_batch_maps = std::max<uint64_t>(executionMetrics_.max_batch_maps, count);
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw metalError("Metal batched NCC execution", command.error);
                if (command.GPUEndTime > command.GPUStartTime && command.GPUStartTime > 0) {
                    if (metrics_.device_ms >= 0)
                        metrics_.device_ms += 1000 * (command.GPUEndTime - command.GPUStartTime);
                } else metrics_.device_ms = -1;
                const auto normalization = Clock::now();
                for (size_t i = 0; i < count; ++i) {
                    auto& map = maps[first + i];
                    const auto* bytes = static_cast<const uint8_t*>(workspace_.contents) + statisticsOffsets[i];
                    if (use32) {
                        const auto* stats = reinterpret_cast<const Stats32*>(bytes);
                        for (size_t j = 0; j < map.values.size(); ++j)
                            map.values[j] = normalize(stats[j].sum, stats[j].square, stats[j].product, templ, double(area));
                    } else {
                        const auto* stats = reinterpret_cast<const Stats64*>(bytes);
                        for (size_t j = 0; j < map.values.size(); ++j)
                            map.values[j] = normalize(double(stats[j].sum), double(stats[j].square),
                                double(stats[j].product), templ, double(area));
                    }
                }
                metrics_.normalization_ms += elapsed(normalization);
                metrics_.ncc_ms += elapsed(start);
                first += count;
            }
            return maps;
        }
    }
};
}
std::unique_ptr<Backend> makeMetalBackend() { return std::make_unique<MetalBackend>(); }
}
