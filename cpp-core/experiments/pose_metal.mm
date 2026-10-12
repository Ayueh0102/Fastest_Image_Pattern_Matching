// SPDX-License-Identifier: BSD-2-Clause
#include "pose_internal.hpp"
#include "metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace fpm_exp {
struct alignas(16) MetalConfig {
    uint32_t sw, sh, tw, th, channels, tiles, count, unused;
    float sourceMean[4], templateSum[4];
    float templateEnergy, padding[3];
};
static_assert(sizeof(MetalConfig) == 80);
class Metal final : public Backend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> partialPipeline_, finishPipeline_;
    id<MTLBuffer> source_, template_, poses_, partials_, scores_;
    MetalConfig config_{};
    std::string name_;
    id<MTLBuffer> buffer(size_t bytes) {
        if (bytes > device_.maxBufferLength) throw std::runtime_error("Metal buffer exceeds maxBufferLength");
        id<MTLBuffer> out = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!out) throw std::runtime_error("Metal buffer allocation failed");
        return out;
    }
public:
    explicit Metal(const Data& d) {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("No Metal GPU available");
            name_ = [device_.name UTF8String];
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Could not create Metal command queue");
            NSError* error = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else
#endif
            {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:fpmMetalSource]
                options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            partialPipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pose_partials"] error:&error];
            if (!partialPipeline_) throw std::runtime_error([[error localizedDescription] UTF8String]);
            finishPipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pose_finish"] error:&error];
            if (!finishPipeline_) throw std::runtime_error([[error localizedDescription] UTF8String]);
            if (partialPipeline_.maxTotalThreadsPerThreadgroup < 256)
                throw std::runtime_error("Metal device cannot run 256-thread workgroups");
            source_ = buffer(d.source.size()*sizeof(float)); template_ = buffer(d.templ.size()*sizeof(float));
            std::memcpy(source_.contents, d.source.data(), source_.length);
            std::memcpy(template_.contents, d.templ.data(), template_.length);
            config_.sw = d.sw; config_.sh = d.sh; config_.tw = d.tw; config_.th = d.th;
            config_.channels = d.channels; config_.tiles = (uint32_t(d.tw)*d.th+1023)/1024;
            config_.templateEnergy = float(d.templateEnergy);
            for (int c = 0; c < d.channels; ++c) {
                config_.sourceMean[c] = d.sourceMean[c]; config_.templateSum[c] = float(d.templateSum[c]);
            }
        }
    }
    const char* name() const override { return name_.c_str(); }
    void memory(FpmPoseMetrics& metrics) const override {
        metrics.gpu_buffer_bytes = source_.length+template_.length+poses_.length+partials_.length+scores_.length;
        metrics.gpu_allocated_bytes = device_.currentAllocatedSize;
    }
    void score(const std::vector<PackedPose>& poses, float* out, FpmPoseMetrics& metrics) override {
        @autoreleasepool {
            // Bound retained auxiliary storage independently of batch size.
            // Source/template remain resident. One oversized pose is permitted
            // when its reduction alone exceeds the budget. Buffer replacement
            // can transiently retain the previous allocation as well.
            constexpr size_t workspaceBytes = 16 * 1024 * 1024;
            const size_t partialBytes = size_t(config_.tiles) * 64;
            const size_t perPose = sizeof(PackedPose) + partialBytes + sizeof(float);
            const size_t capacity = std::min(poses.size(), std::max(size_t(1),
                std::min(workspaceBytes / perPose, size_t(device_.maxBufferLength) / partialBytes)));
            const size_t count = capacity;
            if (!poses_ || poses_.length < count*sizeof(PackedPose)) {
                // Allocate a complete replacement first; failed growth keeps the old buffers valid.
                id<MTLBuffer> p = buffer(count*sizeof(PackedPose));
                id<MTLBuffer> partial = buffer(count*config_.tiles*64);
                id<MTLBuffer> score = buffer(count*sizeof(float));
                poses_ = p; partials_ = partial; scores_ = score;
            }
            metrics.device_ms = 0;
            bool allChunksTimed = true;
            for (size_t offset = 0; offset < poses.size(); offset += capacity) {
              @autoreleasepool {
            const size_t count = std::min(capacity, poses.size() - offset);
            std::memcpy(poses_.contents, poses.data() + offset, count*sizeof(PackedPose));
            config_.count = uint32_t(count);
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (!command) throw std::runtime_error("Could not create Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Could not create Metal compute encoder");
            [encoder setComputePipelineState:partialPipeline_];
            [encoder setBuffer:source_ offset:0 atIndex:0]; [encoder setBuffer:template_ offset:0 atIndex:1];
            [encoder setBuffer:poses_ offset:0 atIndex:2]; [encoder setBuffer:partials_ offset:0 atIndex:3];
            [encoder setBytes:&config_ length:sizeof(config_) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(config_.tiles, count, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            // Sequential encoders with tracked resources make partial writes visible to finish.
            encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Could not create Metal finish encoder");
            [encoder setComputePipelineState:finishPipeline_];
            [encoder setBuffer:partials_ offset:0 atIndex:0]; [encoder setBuffer:scores_ offset:0 atIndex:1];
            [encoder setBytes:&config_ length:sizeof(config_) atIndex:2];
            const NSUInteger group = std::min(NSUInteger(64), finishPipeline_.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
            [encoder endEncoding];
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error(command.error ? [[command.error localizedDescription] UTF8String] : "Metal command failed");
            std::memcpy(out + offset, scores_.contents, count*sizeof(float));
            if (command.GPUStartTime > 0 && command.GPUEndTime >= command.GPUStartTime) {
                metrics.device_ms += (command.GPUEndTime-command.GPUStartTime)*1000;
            } else allChunksTimed = false;
              }
            }
            if (!allChunksTimed) metrics.device_ms = -1;
            memory(metrics);
        }
    }
};
std::unique_ptr<Backend> makeMetal(const Data& d) { return std::make_unique<Metal>(d); }
}
