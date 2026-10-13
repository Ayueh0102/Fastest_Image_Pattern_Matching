// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include "coarse_metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cfloat>
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
struct Config {
    uint32_t sw, tw, th, outputWidth;
    uint64_t firstWindow;
    uint32_t count, reserved;
};
struct Stats32 { uint32_t sum, square, product; };
struct Stats64 { uint64_t sum, square, product; };
static_assert(sizeof(Config) == 32 && sizeof(Stats32) == 12 && sizeof(Stats64) == 24,
    "Metal statistics ABI changed");
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
    id<MTLComputePipelineState> stats32_, stats64_;
    id<MTLBuffer> source_, template_, workspace_;
    const fpm::TemplateLevel* uploadedTemplate_ = nullptr;
    FpmCoarseMetrics metrics_{};

    void ensure(id<MTLBuffer> __strong& buffer, size_t bytes) {
        if (buffer && buffer.length >= bytes) return;
        if (bytes > device_.maxBufferLength) throw std::runtime_error("Metal buffer exceeds device capacity");
        const auto start = Clock::now();
        id<MTLBuffer> replacement = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!replacement) throw std::runtime_error("Metal buffer allocation failed");
        buffer = replacement;
        metrics_.setup_ms += elapsed(start);
    }
public:
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
            stats32_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"coarse_stats32"] error:&error];
            if (!stats32_) throw metalError("coarse uint32 pipeline", error);
            stats64_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"coarse_stats64"] error:&error];
            if (!stats64_) throw metalError("coarse uint64 pipeline", error);
        }
    }
    void resetMetrics() override { metrics_ = {}; }
    FpmCoarseMetrics metrics() const override {
        auto m = metrics_;
        m.workspace_bytes = workspace_ ? workspace_.length : 0;
        m.gpu_bytes = (source_ ? source_.length : 0) +
            (template_ ? template_.length : 0) + m.workspace_bytes;
        m.source_buffer_bytes = source_ ? source_.length : 0;
        m.template_buffer_bytes = template_ ? template_.length : 0;
        return m;
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
            const id<MTLComputePipelineState> pipeline = use32 ? stats32_ : stats64_;
            const NSUInteger threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
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
                [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
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
};
}
std::unique_ptr<Backend> makeMetalBackend() { return std::make_unique<MetalBackend>(); }
}
