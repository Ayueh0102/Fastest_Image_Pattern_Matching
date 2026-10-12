// SPDX-License-Identifier: BSD-2-Clause
#include "pose_internal.hpp"
#include "metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace fpm_exp {
struct alignas(16) MetalConfig {
    uint32_t sw, sh, tw, th, channels, tiles, count, unused;
    float sourceMean[4], templateSum[4];
    float templateEnergy, padding[3];
};
static_assert(sizeof(MetalConfig) == 80);
// Packed scalar arrays match Metal's layout (no float3/float4 padding).
// Per-channel means; summed centered covariance/template products of
// [sample,Jx,Jy,Jtheta,Jlogscale]. Normalized NCC needs no cross-channel terms.
struct DerivativePartial {
    float mean[3][5], covariance[15], product[5];
    uint32_t count;
};
static_assert(sizeof(DerivativePartial) == 144);
static int covarianceIndex(int i, int j) { return i*5-i*(i-1)/2+j-i; }
class Metal final : public Backend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> partialPipeline_, finishPipeline_, derivativePipeline_, derivativeFinishPipeline_;
    id<MTLBuffer> source_, template_, workspace_;
    MetalConfig config_{};
    double templateEnergy_ = 0;
    std::array<double, 3> templateSum_{};
    std::string name_;
    id<MTLBuffer> buffer(size_t bytes) {
        if (bytes > device_.maxBufferLength) throw std::runtime_error("Metal buffer exceeds maxBufferLength");
        id<MTLBuffer> out = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!out) throw std::runtime_error("Metal buffer allocation failed");
        return out;
    }
    struct Workspace {
        size_t capacity, partialOffset, resultOffset, bytes;
    };
    Workspace prepareWorkspace(size_t count, size_t partialBytes, size_t resultBytes) {
        // One grow-only raw allocation serves BOTH score and derivative modes.
        // Changing their layouts does not reallocate once sufficient capacity
        // exists. All offsets satisfy Metal's constant/device buffer alignment.
        constexpr size_t workspaceBytes = 16*1024*1024;
        const size_t budget = std::min(workspaceBytes, size_t(device_.maxBufferLength));
        const size_t perPose = sizeof(PackedPose)+partialBytes+resultBytes;
        const size_t capacity = std::min(count, std::max(size_t(1),
            (budget > 512 ? budget-512 : 0)/perPose));
        auto aligned = [](size_t n) { return (n+255)/256*256; };
        const size_t partialOffset = aligned(capacity*sizeof(PackedPose));
        const size_t resultOffset = aligned(partialOffset+capacity*partialBytes);
        const size_t bytes = resultOffset+capacity*resultBytes;
        // A single oversized pose is allowed; buffer() still enforces the
        // device limit. Replacement can transiently retain the previous buffer.
        if (!workspace_ || workspace_.length < bytes) workspace_ = buffer(bytes);
        return {capacity, partialOffset, resultOffset, bytes};
    }
    static bool accumulateTime(id<MTLCommandBuffer> command, FpmPoseMetrics& metrics) {
        const double start = command.GPUStartTime, end = command.GPUEndTime;
        if (!std::isfinite(start) || !std::isfinite(end) || start <= 0 || end < start) return false;
        metrics.device_ms += (end-start)*1000;
        return true;
    }
    static void complete(id<MTLCommandBuffer> command) {
        [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(command.error ? [[command.error localizedDescription] UTF8String] : "Metal command failed");
    }
    Linearization assemble(const DerivativePartial& p) const {
        Linearization result;
        const double n = double(config_.tw)*config_.th;
        const double energy = std::max(0.0, double(p.covariance[0]));
        double product = p.product[0], jx[4]{}, jt[4]{};
        for (int k = 0; k < 4; ++k) {
            jx[k] = p.covariance[covarianceIndex(0, k+1)];
            jt[k] = p.product[k+1];
        }
        for (uint32_t c = 0; c < config_.channels; ++c) {
            product -= double(p.mean[c][0])*templateSum_[c];
            for (int k = 0; k < 4; ++k) jt[k] -= double(p.mean[c][k+1])*templateSum_[c];
        }
        if (energy/n <= 1e-12 || templateEnergy_/n <= 1e-12) return result;
        const double denominator = std::sqrt(energy*templateEnergy_);
        result.score = std::clamp(product/denominator, -1.0, 1.0);
        for (int k = 0; k < 4; ++k) {
            result.g[k] = jt[k]/denominator-result.score*jx[k]/energy;
            for (int l = k; l < 4; ++l) {
                result.h[k][l] = double(p.covariance[covarianceIndex(k+1, l+1)])/energy-jx[k]*jx[l]/(energy*energy);
                result.h[l][k] = result.h[k][l];
            }
        }
        return result;
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
            derivativePipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pose_derivative_partials"] error:&error];
            if (!derivativePipeline_) throw std::runtime_error([[error localizedDescription] UTF8String]);
            derivativeFinishPipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"pose_derivative_finish"] error:&error];
            if (!derivativeFinishPipeline_) throw std::runtime_error([[error localizedDescription] UTF8String]);
            if (partialPipeline_.maxTotalThreadsPerThreadgroup < 256)
                throw std::runtime_error("Metal device cannot run 256-thread workgroups");
            if (derivativePipeline_.maxTotalThreadsPerThreadgroup < 128)
                throw std::runtime_error("Metal device cannot run 128-thread derivative workgroups");
            source_ = buffer(d.source.size()*sizeof(float)); template_ = buffer(d.templ.size()*sizeof(float));
            std::memcpy(source_.contents, d.source.data(), source_.length);
            std::memcpy(template_.contents, d.templ.data(), template_.length);
            config_.sw = d.sw; config_.sh = d.sh; config_.tw = d.tw; config_.th = d.th;
            config_.channels = d.channels; config_.tiles = (uint32_t(d.tw)*d.th+1023)/1024;
            config_.templateEnergy = float(d.templateEnergy);
            templateEnergy_ = d.templateEnergy; templateSum_ = d.templateSum;
            for (int c = 0; c < d.channels; ++c) {
                config_.sourceMean[c] = d.sourceMean[c]; config_.templateSum[c] = float(d.templateSum[c]);
            }
        }
    }
    const char* name() const override { return name_.c_str(); }
    void memory(FpmPoseMetrics& metrics) const override {
        metrics.gpu_buffer_bytes = source_.length+template_.length+workspace_.length;
        metrics.gpu_allocated_bytes = device_.currentAllocatedSize;
    }
    void score(const std::vector<PackedPose>& poses, float* out, FpmPoseMetrics& metrics) override {
        @autoreleasepool {
            const auto workspace = prepareWorkspace(poses.size(), size_t(config_.tiles)*64, sizeof(float));
            const size_t capacity = workspace.capacity;
            metrics.device_ms = 0;
            bool allChunksTimed = true;
            for (size_t offset = 0; offset < poses.size(); offset += capacity) {
              @autoreleasepool {
            const size_t count = std::min(capacity, poses.size() - offset);
            std::memcpy(workspace_.contents, poses.data() + offset, count*sizeof(PackedPose));
            config_.count = uint32_t(count);
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (!command) throw std::runtime_error("Could not create Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Could not create Metal compute encoder");
            [encoder setComputePipelineState:partialPipeline_];
            [encoder setBuffer:source_ offset:0 atIndex:0]; [encoder setBuffer:template_ offset:0 atIndex:1];
            [encoder setBuffer:workspace_ offset:0 atIndex:2]; [encoder setBuffer:workspace_ offset:workspace.partialOffset atIndex:3];
            [encoder setBytes:&config_ length:sizeof(config_) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(config_.tiles, count, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            // Sequential encoders with tracked resources make partial writes visible to finish.
            encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Could not create Metal finish encoder");
            [encoder setComputePipelineState:finishPipeline_];
            [encoder setBuffer:workspace_ offset:workspace.partialOffset atIndex:0]; [encoder setBuffer:workspace_ offset:workspace.resultOffset atIndex:1];
            [encoder setBytes:&config_ length:sizeof(config_) atIndex:2];
            const NSUInteger group = std::min(NSUInteger(64), finishPipeline_.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
            [encoder endEncoding];
            complete(command);
            std::memcpy(out + offset, static_cast<const char*>(workspace_.contents)+workspace.resultOffset, count*sizeof(float));
            if (!accumulateTime(command, metrics)) allChunksTimed = false;
              }
            }
            if (!allChunksTimed) metrics.device_ms = -1;
            memory(metrics);
        }
    }
    bool linearize(const std::vector<PackedPose>& poses, const FpmPoseRefineConfig& options,
        std::vector<Linearization>& output, FpmPoseMetrics& metrics) override {
        const auto begin = std::chrono::steady_clock::now();
        metrics = {}; metrics.device_ms = 0;
        output.resize(poses.size());
        if (poses.empty()) { memory(metrics); return true; }
        @autoreleasepool {
            const auto workspace = prepareWorkspace(poses.size(), size_t(config_.tiles)*sizeof(DerivativePartial), sizeof(DerivativePartial));
            config_.unused = (options.max_translation != 0 ? 1u : 0u)
                | (options.optimize_angle && options.max_angle_degrees != 0 ? 2u : 0u)
                | (options.optimize_scale && options.max_log_scale != 0 ? 4u : 0u);
            bool allChunksTimed = true;
            for (size_t offset = 0; offset < poses.size(); offset += workspace.capacity) {
                @autoreleasepool {
                    const size_t count = std::min(workspace.capacity, poses.size()-offset);
                    std::memcpy(workspace_.contents, poses.data()+offset, count*sizeof(PackedPose));
                    config_.count = uint32_t(count);
                    id<MTLCommandBuffer> command = [queue_ commandBuffer];
                    if (!command) throw std::runtime_error("Could not create Metal derivative command buffer");
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    if (!encoder) throw std::runtime_error("Could not create Metal derivative encoder");
                    [encoder setComputePipelineState:derivativePipeline_];
                    [encoder setBuffer:source_ offset:0 atIndex:0]; [encoder setBuffer:template_ offset:0 atIndex:1];
                    [encoder setBuffer:workspace_ offset:0 atIndex:2];
                    [encoder setBuffer:workspace_ offset:workspace.partialOffset atIndex:3];
                    [encoder setBytes:&config_ length:sizeof(config_) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(config_.tiles, count, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];
                    encoder = [command computeCommandEncoder];
                    if (!encoder) throw std::runtime_error("Could not create Metal derivative finish encoder");
                    [encoder setComputePipelineState:derivativeFinishPipeline_];
                    [encoder setBuffer:workspace_ offset:workspace.partialOffset atIndex:0];
                    [encoder setBuffer:workspace_ offset:workspace.resultOffset atIndex:1];
                    [encoder setBytes:&config_ length:sizeof(config_) atIndex:2];
                    const NSUInteger group = std::min(NSUInteger(32), derivativeFinishPipeline_.maxTotalThreadsPerThreadgroup);
                    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
                    [encoder endEncoding];
                    complete(command);
                    const auto* result = reinterpret_cast<const DerivativePartial*>(
                        static_cast<const char*>(workspace_.contents)+workspace.resultOffset);
                    for (size_t i = 0; i < count; ++i) output[offset+i] = assemble(result[i]);
                    if (!accumulateTime(command, metrics)) allChunksTimed = false;
                }
            }
            if (!allChunksTimed) metrics.device_ms = -1;
            memory(metrics);
        }
        metrics.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
        return true;
    }
};
std::unique_ptr<Backend> makeMetal(const Data& d) { return std::make_unique<Metal>(d); }
}
