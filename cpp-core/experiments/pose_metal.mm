// SPDX-License-Identifier: BSD-2-Clause
#include "pose_internal.hpp"
#include "metal_source.hpp"
#include "resident_metal_source.hpp"
#include "fused_metal_source.hpp"
#include "simd_metal_source.hpp"
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
// Resident-only layouts mirror pose_resident.metal. Scalar host arrays have
// the same offsets as six Metal float4 fields; the complete stride is 208 B.
struct alignas(16) ResidentState {
    float pose[4], lower[4], upper[4], packed[4], candidate[4], candidatePacked[4];
    float h[16], g[4], damping, score;
    uint32_t iterations, evaluations, status, active, pending, proposed;
};
struct ResidentControl { uint32_t activeCount, candidateCount, reserved[2]; };
struct ResidentDispatch { uint32_t x, y, z, padding; };
struct ResidentOptions { uint32_t count, finishThreads, reserved[2]; };
struct ResidentTrial { uint32_t first, final; };
static_assert(sizeof(ResidentState) == 208);
static_assert(sizeof(ResidentControl) == 16 && sizeof(ResidentDispatch) == 16);
static_assert(sizeof(ResidentOptions) == 16 && sizeof(ResidentTrial) == 8);
enum ResidentGrid { ResidentOuter, ResidentDerivativeTiles, ResidentDerivativeFinish,
    ResidentTrialControl, ResidentScoreTiles, ResidentScoreFinish };
static int covarianceIndex(int i, int j) { return i*5-i*(i-1)/2+j-i; }
class Metal final : public Backend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> partialPipeline_, finishPipeline_, derivativePipeline_, derivativeFinishPipeline_;
    id<MTLComputePipelineState> residentInitializePipeline_, residentPreparePipeline_, residentTrialPipeline_;
    id<MTLComputePipelineState> fusedPipeline_, simdPipeline_;
    bool residentReady_ = false;
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
    id<MTLBuffer> imageBuffer(const ImageStorage& image, bool shared) {
        if (!shared) {
            id<MTLBuffer> out = buffer(image.size()*sizeof(float));
            std::memcpy(out.contents,image.data(),out.length);
            return out;
        }
        if (!image.pages || image.pageBytes > device_.maxBufferLength)
            throw std::runtime_error("Invalid shared pose image allocation or Metal buffer size");
        const auto owner = image.pages;
        id<MTLBuffer> out = [device_ newBufferWithBytesNoCopy:owner.get() length:image.pageBytes
            options:MTLResourceStorageModeShared deallocator:^(void*, NSUInteger) { (void)owner; }];
        if (!out || out.contents != image.data())
            throw std::runtime_error("Metal could not wrap shared pose image without copying");
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
    struct ResidentWorkspace {
        size_t capacity, configOffset, controlOffset, indirectOffset, stateOffset;
        size_t activeOffset, candidateOffset, packedOffset, partialOffset, resultOffset, bytes;
    };
    ResidentWorkspace prepareResidentWorkspace(size_t count) {
        constexpr size_t maximumBytes = 16*1024*1024;
        const size_t budget = std::min(maximumBytes, size_t(device_.maxBufferLength));
        const size_t partialBytes = size_t(config_.tiles)*sizeof(DerivativePartial);
        const size_t perPose = sizeof(ResidentState)+2*sizeof(uint32_t)+sizeof(PackedPose)
            +partialBytes+sizeof(DerivativePartial);
        // Reserve enough for all fixed headers and 256-byte alignment gaps.
        const size_t capacity = std::min(count, std::max(size_t(1),
            (budget > 2048 ? budget-2048 : 0)/perPose));
        ResidentWorkspace result{}; result.capacity = capacity;
        auto reserve = [&](size_t bytes) {
            const size_t offset = (result.bytes+255)/256*256;
            result.bytes = offset+bytes;
            return offset;
        };
        result.configOffset = reserve(sizeof(MetalConfig));
        result.controlOffset = reserve(sizeof(ResidentControl));
        result.indirectOffset = reserve(6*sizeof(ResidentDispatch));
        result.stateOffset = reserve(capacity*sizeof(ResidentState));
        result.activeOffset = reserve(capacity*sizeof(uint32_t));
        result.candidateOffset = reserve(capacity*sizeof(uint32_t));
        result.packedOffset = reserve(capacity*sizeof(PackedPose));
        result.partialOffset = reserve(capacity*partialBytes);
        result.resultOffset = reserve(capacity*sizeof(DerivativePartial));
        if (result.bytes > budget)
            throw std::runtime_error("one resident pose exceeds the 16 MiB auxiliary workspace cap or Metal device buffer limit");
        // Reuse the legacy raw allocation; there is no second retained solver
        // buffer. If an earlier legacy call used its oversized-pose exception,
        // replace it so this opt-in path still observes its strict 16 MiB cap.
        // As in prepareWorkspace(), replacement can transiently retain both.
        if (!workspace_ || workspace_.length < result.bytes || workspace_.length > budget)
            workspace_ = buffer(result.bytes);
        return result;
    }
    void prepareResidentPipelines() {
        if (residentReady_) return;
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
        // The original construction path compiles only its original source.
        // Additional solver code and pipelines are created on first opt-in use.
        const std::string program = std::string(fpmMetalSource)+"\n"+fpmResidentMetalSource;
        id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:program.c_str()]
            options:options error:&error];
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        auto pipeline = [&](NSString* name) {
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> result = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&pipelineError];
            if (!result) throw std::runtime_error([[pipelineError localizedDescription] UTF8String]);
            if (result.maxTotalThreadsPerThreadgroup < 64)
                throw std::runtime_error("Metal device cannot run 64-thread resident control workgroups");
            return result;
        };
        residentInitializePipeline_ = pipeline(@"resident_initialize");
        residentPreparePipeline_ = pipeline(@"resident_prepare");
        residentTrialPipeline_ = pipeline(@"resident_trial");
        residentReady_ = true;
    }
    void prepareFusedPipeline(int mode) {
        if (mode == 2 ? simdPipeline_ != nil : fusedPipeline_ != nil) return;
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
        const std::string program = std::string(fpmMetalSource)+"\n"+fpmResidentMetalSource+"\n"+fpmFusedMetalSource
            +(mode == 2 ? std::string("\n")+fpmSimdMetalSource : "");
        id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:program.c_str()]
            options:options error:&error];
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:
            [library newFunctionWithName:(mode == 2 ? @"pose_refine_simd" : @"pose_refine_fused")] error:&error];
        if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
        if (mode == 2 && pipeline.threadExecutionWidth != 32)
            throw std::runtime_error("SIMD resident requires a Metal SIMD width of 32");
        if (pipeline.maxTotalThreadsPerThreadgroup < (mode == 2 ? 128u : 256u) ||
            pipeline.staticThreadgroupMemoryLength > device_.maxThreadgroupMemoryLength)
            throw std::runtime_error("Metal device cannot run fused resident workgroup");
        if (mode == 2) simdPipeline_ = pipeline; else fusedPipeline_ = pipeline;
    }
    bool refineFused(const std::vector<ResidentSeed>& seeds, const FpmPoseRefineConfig& options,
        float* poses, float* scores, FpmPoseRefineStats* stats,
        FpmPoseResidentMetrics& diagnostic, FpmPoseMetrics& metrics, int mode) {
        const auto begin = std::chrono::steady_clock::now();
        auto elapsed = [](std::chrono::steady_clock::time_point start) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        };
        if (config_.channels != 1 || uint64_t(config_.tw)*config_.th > 16384)
            throw std::invalid_argument("fused resident supports grayscale templates with <=16384 pixels");
        metrics = {}; metrics.device_ms = 0;
        @autoreleasepool {
            if (mode == 2 ? simdPipeline_ == nil : fusedPipeline_ == nil) {
                const auto setup = std::chrono::steady_clock::now();
                prepareFusedPipeline(mode); diagnostic.setup_ms += elapsed(setup);
            }
            // Bound the duration of a single submission without changing any
            // candidate's iteration/trial budget. No cross-workgroup barrier.
            const size_t capacity = std::min<size_t>(512, seeds.size());
            constexpr size_t stateOffset = 256, budget = 16*1024*1024;
            const size_t bytes = stateOffset+capacity*sizeof(ResidentState);
            if (!workspace_ || workspace_.length < bytes || workspace_.length > budget)
                workspace_ = buffer(bytes);
            diagnostic.auxiliary_buffer_bytes = workspace_.length;
            bool timed = true;
            for (size_t offset = 0; offset < seeds.size(); offset += capacity) {
              @autoreleasepool {
                const size_t count = std::min(capacity, seeds.size()-offset);
                auto* data = static_cast<char*>(workspace_.contents);
                auto* states = reinterpret_cast<ResidentState*>(data+stateOffset);
                for (size_t i = 0; i < count; ++i) {
                    auto& state = states[i]; state = {};
                    const auto& seed = seeds[offset+i];
                    std::copy(seed.pose, seed.pose+4, state.pose);
                    std::copy(seed.lower, seed.lower+4, state.lower);
                    std::copy(seed.upper, seed.upper+4, state.upper);
                    const double angle = double(seed.pose[2])*3.14159265358979323846/180;
                    const double scale = std::exp(double(seed.pose[3]));
                    const PackedPose packed{seed.pose[0], seed.pose[1], float(scale*std::cos(angle)), float(scale*std::sin(angle))};
                    std::memcpy(state.packed, &packed, sizeof(packed));
                }
                MetalConfig dynamicConfig = config_; dynamicConfig.count = uint32_t(count);
                dynamicConfig.unused = (options.max_translation != 0 ? 1u : 0u)
                    | (options.optimize_angle && options.max_angle_degrees != 0 ? 2u : 0u)
                    | (options.optimize_scale && options.max_log_scale != 0 ? 4u : 0u);
                std::memcpy(data, &dynamicConfig, sizeof(dynamicConfig));
                const auto encodeBegin = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> command = [queue_ commandBuffer];
                if (!command) throw std::runtime_error("Could not create fused resident command buffer");
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!encoder) throw std::runtime_error("Could not create fused resident encoder");
                [encoder setComputePipelineState:(mode == 2 ? simdPipeline_ : fusedPipeline_)];
                [encoder setBuffer:source_ offset:0 atIndex:0];
                [encoder setBuffer:template_ offset:0 atIndex:1];
                [encoder setBuffer:workspace_ offset:stateOffset atIndex:2];
                [encoder setBuffer:workspace_ offset:0 atIndex:3];
                const uint32_t iterations = uint32_t(options.max_iterations);
                [encoder setBytes:&iterations length:sizeof(iterations) atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(count,1,1)
                    threadsPerThreadgroup:MTLSizeMake(mode == 2 ? 128 : 256,1,1)];
                [encoder endEncoding];
                ++diagnostic.encoded_dispatches; ++diagnostic.command_buffers;
                diagnostic.encode_ms += elapsed(encodeBegin);
                const auto waitBegin = std::chrono::steady_clock::now();
                ++diagnostic.cpu_waits; complete(command);
                diagnostic.wait_ms += elapsed(waitBegin);
                if (!accumulateTime(command, metrics)) timed = false;
                for (size_t i = 0; i < count; ++i) {
                    const auto& state = states[i];
                    std::copy(state.pose, state.pose+4, poses+(offset+i)*4);
                    scores[offset+i] = state.score;
                    if (stats) stats[offset+i] = {int(state.iterations), int(state.evaluations), int(state.status)};
                }
              }
            }
            if (!timed) metrics.device_ms = -1;
            memory(metrics);
        }
        metrics.wall_ms = elapsed(begin);
        return true;
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
            if (!device_) throw MetalUnavailable("No Metal GPU available");
            if (d.storageMode && !device_.hasUnifiedMemory)
                throw std::runtime_error("Shared pose storage requires a unified-memory Metal device");
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
            for (id<MTLComputePipelineState> pipeline : {partialPipeline_, finishPipeline_, derivativePipeline_, derivativeFinishPipeline_}) {
                if (pipeline.staticThreadgroupMemoryLength > device_.maxThreadgroupMemoryLength)
                    throw std::runtime_error("Metal pose pipeline exceeds device threadgroup memory limit");
            }
            if (derivativePipeline_.maxTotalThreadsPerThreadgroup < 128)
                throw std::runtime_error("Metal device cannot run 128-thread derivative workgroups");
            source_ = imageBuffer(d.source,d.storageMode != 0);
            template_ = imageBuffer(d.templ,d.storageMode != 0);
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
    uint64_t imageBufferBytes() const override { return source_.length+template_.length; }
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
    FpmPoseResidentKernelInfo residentKernelInfo(int mode) const override {
        id<MTLComputePipelineState> pipeline = mode == 2 ? simdPipeline_ : (mode == 1 ? fusedPipeline_ : nil);
        if (!pipeline) return {};
        return {mode == 2 ? 128u : 256u, uint32_t(pipeline.threadExecutionWidth), uint64_t(pipeline.staticThreadgroupMemoryLength)};
    }
    bool refineResident(const std::vector<ResidentSeed>& seeds, const FpmPoseRefineConfig& options,
        float* poses, float* scores, FpmPoseRefineStats* stats,
        FpmPoseResidentMetrics& residentMetrics, FpmPoseMetrics& metrics, int mode = 0) override {
        const auto begin = std::chrono::steady_clock::now();
        auto elapsed = [](std::chrono::steady_clock::time_point start) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        };
        metrics = {}; metrics.device_ms = -1;
        if (seeds.empty()) { memory(metrics); return true; }
        if (mode == 1 || mode == 2) return refineFused(seeds, options, poses, scores, stats, residentMetrics, metrics, mode);
        if (mode != 0) throw std::invalid_argument("invalid Metal resident mode");
        @autoreleasepool {
            const auto workspace = prepareResidentWorkspace(seeds.size());
            residentMetrics.auxiliary_buffer_bytes = workspace_.length;
            if (!residentReady_) {
                const auto setupBegin = std::chrono::steady_clock::now();
                prepareResidentPipelines();
                residentMetrics.setup_ms += elapsed(setupBegin);
            }
            const NSUInteger finishThreads = std::min({NSUInteger(32), finishPipeline_.maxTotalThreadsPerThreadgroup,
                derivativeFinishPipeline_.maxTotalThreadsPerThreadgroup});
            if (!finishThreads) throw std::runtime_error("Metal finish pipeline has no supported threads");
            metrics.device_ms = 0;
            bool allChunksTimed = true;
            for (size_t offset = 0; offset < seeds.size(); offset += workspace.capacity) {
              @autoreleasepool {
                const size_t count = std::min(workspace.capacity, seeds.size()-offset);
                auto* bytes = static_cast<char*>(workspace_.contents);
                auto* states = reinterpret_cast<ResidentState*>(bytes+workspace.stateOffset);
                auto* packed = reinterpret_cast<PackedPose*>(bytes+workspace.packedOffset);
                for (size_t i = 0; i < count; ++i) {
                    auto& state = states[i]; state = {};
                    const auto& seed = seeds[offset+i];
                    std::copy(seed.pose, seed.pose+4, state.pose);
                    std::copy(seed.lower, seed.lower+4, state.lower);
                    std::copy(seed.upper, seed.upper+4, state.upper);
                    // Initial packing matches the portable scorer exactly.
                    // Subsequent proposal packing is performed entirely on GPU.
                    const double angle = double(seed.pose[2])*3.14159265358979323846/180;
                    const double scale = std::exp(double(seed.pose[3]));
                    packed[i] = {seed.pose[0], seed.pose[1], float(scale*std::cos(angle)), float(scale*std::sin(angle))};
                    std::memcpy(state.packed, &packed[i], sizeof(PackedPose));
                }
                MetalConfig dynamicConfig = config_;
                dynamicConfig.count = uint32_t(count);
                dynamicConfig.unused = (options.max_translation != 0 ? 1u : 0u)
                    | (options.optimize_angle && options.max_angle_degrees != 0 ? 2u : 0u)
                    | (options.optimize_scale && options.max_log_scale != 0 ? 4u : 0u);
                std::memcpy(bytes+workspace.configOffset, &dynamicConfig, sizeof(dynamicConfig));
                const ResidentOptions residentOptions{uint32_t(count), uint32_t(finishThreads), {0, 0}};
                const auto encodeBegin = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> command = [queue_ commandBuffer];
                if (!command) throw std::runtime_error("Could not create Metal resident command buffer");
                ++residentMetrics.command_buffers;
                auto encoderFor = [&](id<MTLComputePipelineState> pipeline) {
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    if (!encoder) throw std::runtime_error("Could not create Metal resident compute encoder");
                    [encoder setComputePipelineState:pipeline];
                    return encoder;
                };
                auto bind = [&](id<MTLComputeCommandEncoder> encoder, size_t bufferOffset, NSUInteger index) {
                    [encoder setBuffer:workspace_ offset:bufferOffset atIndex:index];
                };
                auto dispatch = [&](id<MTLComputeCommandEncoder> encoder, int grid, NSUInteger threads, MTLSize direct) {
                    if (grid < 0) [encoder dispatchThreadgroups:direct threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                    else [encoder dispatchThreadgroupsWithIndirectBuffer:workspace_
                        indirectBufferOffset:workspace.indirectOffset+size_t(grid)*sizeof(ResidentDispatch)
                        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                    [encoder endEncoding];
                    ++residentMetrics.encoded_dispatches;
                };
                auto partials = [&](bool derivative, bool initial) {
                    id<MTLComputeCommandEncoder> encoder = encoderFor(derivative ? derivativePipeline_ : partialPipeline_);
                    [encoder setBuffer:source_ offset:0 atIndex:0];
                    [encoder setBuffer:template_ offset:0 atIndex:1];
                    bind(encoder, workspace.packedOffset, 2); bind(encoder, workspace.partialOffset, 3);
                    bind(encoder, workspace.configOffset, 4);
                    dispatch(encoder, initial ? -1 : (derivative ? ResidentDerivativeTiles : ResidentScoreTiles),
                        derivative ? 128 : 256, MTLSizeMake(config_.tiles, count, 1));
                };
                auto finish = [&](bool derivative, bool initial) {
                    id<MTLComputeCommandEncoder> encoder = encoderFor(derivative ? derivativeFinishPipeline_ : finishPipeline_);
                    bind(encoder, workspace.partialOffset, 0); bind(encoder, workspace.resultOffset, 1);
                    bind(encoder, workspace.configOffset, 2);
                    dispatch(encoder, initial ? -1 : (derivative ? ResidentDerivativeFinish : ResidentScoreFinish),
                        finishThreads, MTLSizeMake((count+finishThreads-1)/finishThreads, 1, 1));
                };
                auto trial = [&](bool first, bool final) {
                    id<MTLComputeCommandEncoder> encoder = encoderFor(residentTrialPipeline_);
                    bind(encoder, workspace.stateOffset, 0); bind(encoder, workspace.controlOffset, 1);
                    bind(encoder, workspace.activeOffset, 2); bind(encoder, workspace.candidateOffset, 3);
                    bind(encoder, workspace.packedOffset, 4); bind(encoder, workspace.resultOffset, 5);
                    bind(encoder, workspace.configOffset, 6); bind(encoder, workspace.indirectOffset, 7);
                    [encoder setBytes:&residentOptions length:sizeof(residentOptions) atIndex:8];
                    const ResidentTrial step{uint32_t(first), uint32_t(final)};
                    [encoder setBytes:&step length:sizeof(step) atIndex:9];
                    dispatch(encoder, ResidentTrialControl, 64, MTLSizeMake(1, 1, 1));
                };
                // Original score kernels evaluate both sides of every GPU
                // acceptance comparison. No initial CPU pixel scan is needed.
                partials(false, true); finish(false, true);
                {
                    id<MTLComputeCommandEncoder> encoder = encoderFor(residentInitializePipeline_);
                    bind(encoder, workspace.stateOffset, 0); bind(encoder, workspace.resultOffset, 1);
                    bind(encoder, workspace.controlOffset, 2); bind(encoder, workspace.indirectOffset, 3);
                    [encoder setBytes:&residentOptions length:sizeof(residentOptions) atIndex:4];
                    dispatch(encoder, -1, 64, MTLSizeMake(1, 1, 1));
                }
                for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
                    id<MTLComputeCommandEncoder> encoder = encoderFor(residentPreparePipeline_);
                    bind(encoder, workspace.stateOffset, 0); bind(encoder, workspace.controlOffset, 1);
                    bind(encoder, workspace.activeOffset, 2); bind(encoder, workspace.packedOffset, 3);
                    bind(encoder, workspace.configOffset, 4); bind(encoder, workspace.indirectOffset, 5);
                    [encoder setBytes:&residentOptions length:sizeof(residentOptions) atIndex:6];
                    dispatch(encoder, ResidentOuter, 64, MTLSizeMake(1, 1, 1));
                    partials(true, false); finish(true, false);
                    trial(true, false);
                    for (int attempt = 0; attempt < 10; ++attempt) {
                        partials(false, false); finish(false, false);
                        trial(false, attempt == 9);
                    }
                }
                // Separate encoders + tracked shared resources preserve every
                // shader write before subsequent indirect dispatch/config reads.
                // All state transitions are GPU-only until this single wait.
                residentMetrics.encode_ms += elapsed(encodeBegin);
                const auto waitBegin = std::chrono::steady_clock::now();
                ++residentMetrics.cpu_waits;
                complete(command);
                residentMetrics.wait_ms += elapsed(waitBegin);
                if (!accumulateTime(command, metrics)) allChunksTimed = false;
                for (size_t i = 0; i < count; ++i) {
                    const auto& state = states[i];
                    std::copy(state.pose, state.pose+4, poses+4*(offset+i));
                    scores[offset+i] = state.score;
                    if (stats) stats[offset+i] = {int(state.iterations), int(state.evaluations), int(state.status)};
                }
              }
            }
            if (!allChunksTimed) metrics.device_ms = -1;
            memory(metrics);
        }
        metrics.wall_ms = elapsed(begin);
        return true;
    }
};
std::unique_ptr<Backend> makeMetal(const Data& d) { return std::make_unique<Metal>(d); }
}
