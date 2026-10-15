// SPDX-License-Identifier: BSD-2-Clause
#include "sparse_orientation_metal.hpp"
#include "sparse_orientation_metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fpm::research::detail {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
// The first backend batches complete angle grids. Its score arena is bounded
// to 16 MiB; even an adversarial all-peak result fits in a 48 MiB peak arena.
// A larger *single* grid needs future row tiling and is rejected explicitly.
constexpr std::size_t windowBudget=4*1024*1024;
constexpr std::size_t angleBudget=64;
constexpr std::size_t initialPeakCapacity=1024;
struct ImageConfig { std::uint32_t width,height,radius,reserved; };
struct ScoreConfig {
    std::uint32_t sourceWidth,gridWidth,gridHeight,featureCount;
    std::uint32_t requiredScore,scoreOffset,groupOffset,featureOffset;
    std::uint32_t xOffset,yOffset,peakOffset,peakCapacity;
    std::uint32_t counterOffset,groupThreads,reserved0,reserved1;
};
struct GroupStats { std::uint32_t evaluations,survivors; };
static_assert(sizeof(ImageConfig)==16 && sizeof(ScoreConfig)==64 &&
    sizeof(GroupStats)==8 && sizeof(SparseMetalFeature)==8 && sizeof(SparseMetalPeak)==12,
    "Sparse orientation Metal ABI changed");
std::runtime_error metalError(const char* stage,NSError* error) {
    return std::runtime_error(std::string(stage)+": "+
        (error ? error.localizedDescription.UTF8String : "unknown Metal failure"));
}
std::uint32_t checked32(std::size_t size) {
    if(size>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Metal sparse descriptor exceeds uint32 range");
    return std::uint32_t(size);
}

class MetalBackend final : public SparseMetalBackend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> gradient_,horizontal_,vertical_,score_,peaks_;
    id<MTLBuffer> gradientLut_,responseLut_,source_,spread_,horizontalMask_;
    id<MTLBuffer> xs_,ys_,features_,scores_,statistics_,peakBuffer_,counters_;
    std::size_t groupThreads_=0;
    std::mutex mutex_;

    void ensure(id<MTLBuffer> __strong& buffer,std::size_t bytes,bool gpuOnly,SparseMetalMetrics& m) {
        if(buffer && buffer.length>=bytes) return;
        if(!bytes || bytes>device_.maxBufferLength)
            throw std::runtime_error("Metal sparse buffer exceeds device capacity");
        const auto start=Clock::now();
        const auto storage=gpuOnly ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared;
        id<MTLBuffer> replacement=[device_ newBufferWithLength:bytes
            options:storage | MTLResourceHazardTrackingModeTracked];
        if(!replacement) throw std::runtime_error("Metal sparse buffer allocation failed");
        buffer=replacement;
        m.setupMs+=elapsed(start);
    }
    std::uint64_t capacity() const {
        return gradientLut_.length+responseLut_.length+source_.length+spread_.length+
            horizontalMask_.length+xs_.length+ys_.length+features_.length+scores_.length+
            statistics_.length+peakBuffer_.length+counters_.length;
    }
    void finish(id<MTLCommandBuffer> command,SparseMetalMetrics& m) {
        const auto start=Clock::now();
        [command commit];[command waitUntilCompleted];
        m.waitMs+=elapsed(start);++m.commandBuffers;++m.cpuWaits;
        if(command.status!=MTLCommandBufferStatusCompleted)
            throw metalError("sparse orientation GPU command",command.error);
        if(command.GPUEndTime>command.GPUStartTime && command.GPUStartTime>0) {
            if(m.deviceMs>=0) m.deviceMs+=1000*(command.GPUEndTime-command.GPUStartTime);
        } else m.deviceMs=-1;
    }
    id<MTLCommandBuffer> command() {
        id<MTLCommandBuffer> result=[queue_ commandBuffer];
        if(!result) throw std::runtime_error("Metal sparse command buffer creation failed");
        return result;
    }
    id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> command) {
        id<MTLComputeCommandEncoder> result=[command computeCommandEncoder];
        if(!result) throw std::runtime_error("Metal sparse command encoder creation failed");
        return result;
    }
    void dispatch(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,
        std::size_t count,SparseMetalMetrics& m) {
        const auto threads=std::min<NSUInteger>(256,pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count,1,1)
            threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ++m.dispatches;
    }
    id<MTLCommandBuffer> prepareSource(ImageView source,int spreadRadius,bool merge,
        SparseMetalMetrics& m) {
        const auto start=Clock::now();
        const std::size_t count=std::size_t(source.width)*source.height;
        ensure(source_,count,false,m);ensure(spread_,count,true,m);ensure(horizontalMask_,count,true,m);
        auto stage=Clock::now();
        for(int y=0;y<source.height;++y)
            std::memcpy(static_cast<std::uint8_t*>(source_.contents)+std::size_t(y)*source.width,
                source.data+std::size_t(y)*source.stride,source.width);
        m.uploadMs+=elapsed(stage);
        stage=Clock::now();
        const ImageConfig config{std::uint32_t(source.width),std::uint32_t(source.height),
            std::uint32_t(spreadRadius),0};
        id<MTLCommandBuffer> cmd=command();
        id<MTLComputeCommandEncoder> enc=encoder(cmd);
        [enc setComputePipelineState:gradient_];
        [enc setBuffer:source_ offset:0 atIndex:0];[enc setBuffer:gradientLut_ offset:0 atIndex:1];
        [enc setBuffer:spread_ offset:0 atIndex:2];[enc setBytes:&config length:sizeof(config) atIndex:3];
        dispatch(enc,gradient_,count,m);[enc endEncoding];
        // Separate encoders establish tracked-buffer dependencies without CPU
        // intervention. The gradient mask can be overwritten after horizontal.
        enc=encoder(cmd);[enc setComputePipelineState:horizontal_];
        [enc setBuffer:spread_ offset:0 atIndex:0];[enc setBuffer:horizontalMask_ offset:0 atIndex:1];
        [enc setBytes:&config length:sizeof(config) atIndex:2];
        dispatch(enc,horizontal_,count,m);[enc endEncoding];
        enc=encoder(cmd);[enc setComputePipelineState:vertical_];
        [enc setBuffer:horizontalMask_ offset:0 atIndex:0];[enc setBuffer:spread_ offset:0 atIndex:1];
        [enc setBytes:&config length:sizeof(config) atIndex:2];
        dispatch(enc,vertical_,count,m);[enc endEncoding];
        m.encodeMs+=elapsed(stage);
        if(!merge) {finish(cmd,m);cmd=nil;}
        m.sourcePrepareMs=elapsed(start);
        // The merged command remains uncommitted and local to this search.
        // First-batch scoring consumes spread_ through tracked dependencies;
        // its completion wait includes device source preparation.
        return cmd;
    }
    struct BatchItem { std::size_t angleIndex,windows,groups;ScoreConfig config; };
    void encodePeaks(id<MTLCommandBuffer> cmd,const std::vector<BatchItem>& items,SparseMetalMetrics& m) {
        id<MTLComputeCommandEncoder> enc=encoder(cmd);
        [enc setComputePipelineState:peaks_];[enc setBuffer:scores_ offset:0 atIndex:0];
        [enc setBuffer:peakBuffer_ offset:0 atIndex:1];[enc setBuffer:counters_ offset:0 atIndex:2];
        for(const auto& item:items) {
            [enc setBytes:&item.config length:sizeof(item.config) atIndex:3];
            dispatch(enc,peaks_,item.windows,m);
        }
        [enc endEncoding];
    }
    void scanBatch(const std::vector<SparseMetalAngle>& angles,std::vector<BatchItem>& items,
        std::size_t windows,std::size_t groups,std::size_t xCount,std::size_t yCount,
        std::size_t featureCount,std::size_t peakCount,id<MTLCommandBuffer> __strong& prepared,
        SparseMetalResult& result) {
        auto& m=result.metrics;
        ensure(xs_,xCount*sizeof(std::int32_t),false,m);ensure(ys_,yCount*sizeof(std::int32_t),false,m);
        ensure(features_,featureCount*sizeof(SparseMetalFeature),false,m);
        ensure(scores_,windows*sizeof(std::uint32_t),true,m);ensure(statistics_,groups*sizeof(GroupStats),false,m);
        ensure(peakBuffer_,peakCount*sizeof(SparseMetalPeak),false,m);
        ensure(counters_,items.size()*sizeof(std::uint32_t),false,m);
        auto stage=Clock::now();
        for(const auto& item:items) {
            const auto& angle=angles[item.angleIndex];const auto& c=item.config;
            std::memcpy(static_cast<std::int32_t*>(xs_.contents)+c.xOffset,angle.xs.data(),angle.xs.size()*sizeof(std::int32_t));
            std::memcpy(static_cast<std::int32_t*>(ys_.contents)+c.yOffset,angle.ys.data(),angle.ys.size()*sizeof(std::int32_t));
            std::memcpy(static_cast<SparseMetalFeature*>(features_.contents)+c.featureOffset,
                angle.features.data(),angle.features.size()*sizeof(SparseMetalFeature));
        }
        std::memset(counters_.contents,0,items.size()*sizeof(std::uint32_t));
        m.uploadMs+=elapsed(stage);
        stage=Clock::now();
        id<MTLCommandBuffer> cmd=prepared ? prepared : command();
        prepared=nil; // Transfer ownership so completed buffers can be released before overflow growth.
        id<MTLComputeCommandEncoder> enc=encoder(cmd);
        [enc setComputePipelineState:score_];
        [enc setBuffer:spread_ offset:0 atIndex:0];[enc setBuffer:responseLut_ offset:0 atIndex:1];
        [enc setBuffer:xs_ offset:0 atIndex:2];[enc setBuffer:ys_ offset:0 atIndex:3];
        [enc setBuffer:features_ offset:0 atIndex:4];[enc setBuffer:scores_ offset:0 atIndex:5];
        [enc setBuffer:statistics_ offset:0 atIndex:6];
        for(const auto& item:items) {
            [enc setBytes:&item.config length:sizeof(item.config) atIndex:7];
            // Full groups are required for the reductions and barriers. The
            // padded lanes contribute zero and never access the score arena.
            [enc dispatchThreadgroups:MTLSizeMake(item.groups,1,1)
                threadsPerThreadgroup:MTLSizeMake(groupThreads_,1,1)];
            ++m.dispatches;
        }
        [enc endEncoding];encodePeaks(cmd,items,m);m.encodeMs+=elapsed(stage);finish(cmd,m);
        // Completed commands can be released before growing any output buffer.
        cmd=nil;enc=nil;
        stage=Clock::now();
        const auto* stats=static_cast<const GroupStats*>(statistics_.contents);
        const auto* counts=static_cast<const std::uint32_t*>(counters_.contents);
        bool overflow=false;
        std::vector<std::uint32_t> actualCounts;actualCounts.reserve(items.size());
        for(const auto& item:items) {
            auto& out=result.angles[item.angleIndex];out.windows=item.windows;
            for(std::size_t k=0;k<item.groups;++k) {
                out.evaluations+=stats[item.config.groupOffset+k].evaluations;
                out.survivors+=stats[item.config.groupOffset+k].survivors;
            }
            const auto count=counts[item.config.counterOffset];
            if(count>item.windows) throw std::runtime_error("Metal sparse peak count exceeds grid");
            actualCounts.push_back(count);overflow|=count>item.config.peakCapacity;
        }
        m.readbackMs+=elapsed(stage);
        if(overflow) {
            // Retain every peak. The first pass provides the exact capacity;
            // only peak extraction repeats, so work counters remain comparable.
            ++m.peakOverflowRetries;peakCount=0;
            for(std::size_t i=0;i<items.size();++i) {
                items[i].config.peakOffset=checked32(peakCount);
                items[i].config.peakCapacity=actualCounts[i];peakCount+=actualCounts[i];
            }
            ensure(peakBuffer_,std::max<std::size_t>(1,peakCount)*sizeof(SparseMetalPeak),false,m);
            stage=Clock::now();std::memset(counters_.contents,0,items.size()*sizeof(std::uint32_t));
            m.uploadMs+=elapsed(stage);
            stage=Clock::now();cmd=command();encodePeaks(cmd,items,m);m.encodeMs+=elapsed(stage);finish(cmd,m);cmd=nil;
        }
        stage=Clock::now();counts=static_cast<const std::uint32_t*>(counters_.contents);
        const auto* peaks=static_cast<const SparseMetalPeak*>(peakBuffer_.contents);
        for(std::size_t i=0;i<items.size();++i) {
            const auto& item=items[i];const auto count=counts[item.config.counterOffset];
            if(count!=actualCounts[i] || count>item.config.peakCapacity)
                throw std::runtime_error("Metal sparse peak extraction count changed on retry");
            auto& out=result.angles[item.angleIndex];
            out.peaks.assign(peaks+item.config.peakOffset,peaks+item.config.peakOffset+count);
            // Atomic append order is nondeterministic; restore CPU row order.
            std::sort(out.peaks.begin(),out.peaks.end(),[](const auto& a,const auto& b) {
                return a.yIndex!=b.yIndex ? a.yIndex<b.yIndex : a.xIndex<b.xIndex;
            });
        }
        m.readbackMs+=elapsed(stage);
    }
public:
    MetalBackend(const std::vector<std::uint8_t>& gradientLut,const std::vector<std::uint8_t>& responseLut) {
        if(gradientLut.size()!=511*511 || responseLut.size()!=8*256)
            throw std::invalid_argument("Metal sparse lookup table size mismatch");
        @autoreleasepool {
            device_=MTLCreateSystemDefaultDevice();
            if(!device_) throw std::runtime_error("Metal device is unavailable");
            queue_=[device_ newCommandQueue];
            if(!queue_) throw std::runtime_error("Metal sparse command queue creation failed");
            NSError* error=nil;
            MTLCompileOptions* options=[MTLCompileOptions new];
            library_=[device_ newLibraryWithSource:[NSString stringWithUTF8String:fpmSparseOrientationMetalSource]
                options:options error:&error];
            if(!library_) throw metalError("sparse orientation shader compilation",error);
            auto pipeline=[&](NSString* name) {
                id<MTLFunction> function=[library_ newFunctionWithName:name];
                if(!function) throw std::runtime_error(std::string("Metal sparse function missing: ")+name.UTF8String);
                id<MTLComputePipelineState> result=[device_ newComputePipelineStateWithFunction:function error:&error];
                if(!result) throw metalError("sparse orientation pipeline creation",error);
                return result;
            };
            gradient_=pipeline(@"sparse_gradient");horizontal_=pipeline(@"sparse_spread_horizontal");
            vertical_=pipeline(@"sparse_spread_vertical");score_=pipeline(@"sparse_score");peaks_=pipeline(@"sparse_peaks");
            const std::size_t width=score_.threadExecutionWidth;
            groupThreads_=std::min<std::size_t>(256,score_.maxTotalThreadsPerThreadgroup);
            if(!width || width>groupThreads_)
                throw std::runtime_error("Metal sparse score SIMD width is unsupported");
            groupThreads_=(groupThreads_/width)*width;
            SparseMetalMetrics ignored;
            ensure(gradientLut_,gradientLut.size(),false,ignored);ensure(responseLut_,responseLut.size(),false,ignored);
            std::memcpy(gradientLut_.contents,gradientLut.data(),gradientLut.size());
            std::memcpy(responseLut_.contents,responseLut.data(),responseLut.size());
        }
    }
    SparseMetalResult search(ImageView source,const std::vector<SparseMetalAngle>& angles,
        int spreadRadius,std::uint32_t requiredScore,int submissionMode) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if(submissionMode<0 || submissionMode>1)
            throw std::invalid_argument("Metal sparse submission mode must be 0 or 1");
        if(!source.data || source.width<=0 || source.height<=0 || source.stride<std::size_t(source.width) ||
            source.stride>(std::size_t(PTRDIFF_MAX)-source.width)/std::size_t(source.height) ||
            std::uint64_t(source.width)*source.height>INT32_MAX || spreadRadius<0 || spreadRadius>16)
            throw std::invalid_argument("Metal sparse image or spread radius exceeds supported indexing");
        // Validate all plans before allocating or dispatching. Offsets are
        // signed, while addresses are proven inside the gray8 source arena.
        const std::int64_t imageSize=std::int64_t(source.width)*source.height;
        for(const auto& angle:angles) {
            if(angle.xs.empty() || angle.ys.empty()) continue;
            if(angle.xs.size()>windowBudget/angle.ys.size())
                throw std::invalid_argument("Metal sparse single-angle grid exceeds 4194304 windows; use a larger position stride");
            if(angle.features.empty() || angle.features.size()>4096 || !requiredScore ||
                requiredScore>angle.features.size()*255)
                throw std::invalid_argument("Metal sparse feature count or threshold is invalid");
            const auto xx=std::minmax_element(angle.xs.begin(),angle.xs.end());
            const auto yy=std::minmax_element(angle.ys.begin(),angle.ys.end());
            if(*xx.first<0 || *xx.second>=source.width || *yy.first<0 || *yy.second>=source.height)
                throw std::invalid_argument("Metal sparse grid coordinates exceed source");
            const std::int64_t first=std::int64_t(*yy.first)*source.width+*xx.first;
            const std::int64_t last=std::int64_t(*yy.second)*source.width+*xx.second;
            for(const auto& feature:angle.features)
                if(feature.bin>=8 || first+feature.offset<0 || last+feature.offset>=imageSize)
                    throw std::invalid_argument("Metal sparse feature address exceeds source");
        }
        @autoreleasepool {
            SparseMetalResult result;result.angles.resize(angles.size());
            bool any=false;for(const auto& angle:angles) any|=!angle.xs.empty() && !angle.ys.empty();
            if(!any) {result.metrics.gpuBufferBytes=capacity();return result;}
            id<MTLCommandBuffer> prepared=prepareSource(source,spreadRadius,submissionMode==1,result.metrics);
            const auto start=Clock::now();
            std::size_t next=0;
            while(next<angles.size()) {
                std::vector<BatchItem> items;
                std::size_t windows=0,groups=0,xCount=0,yCount=0,featureCount=0,peakCount=0;
                while(next<angles.size() && items.size()<angleBudget) {
                    const auto& angle=angles[next];
                    if(angle.xs.empty() || angle.ys.empty()) {++next;continue;}
                    const auto count=angle.xs.size()*angle.ys.size();
                    if(windows && windows+count>windowBudget) break;
                    const auto groupCount=(count+groupThreads_-1)/groupThreads_;
                    const auto peakCapacity=std::min(count,initialPeakCapacity);
                    ScoreConfig config{std::uint32_t(source.width),checked32(angle.xs.size()),checked32(angle.ys.size()),
                        checked32(angle.features.size()),requiredScore,checked32(windows),checked32(groups),checked32(featureCount),
                        checked32(xCount),checked32(yCount),checked32(peakCount),checked32(peakCapacity),
                        checked32(items.size()),checked32(groupThreads_),0,0};
                    items.push_back({next++,count,groupCount,config});windows+=count;groups+=groupCount;
                    xCount+=angle.xs.size();yCount+=angle.ys.size();featureCount+=angle.features.size();peakCount+=peakCapacity;
                }
                if(!items.empty()) {
                    scanBatch(angles,items,windows,groups,xCount,yCount,featureCount,peakCount,prepared,result);
                }
            }
            result.metrics.searchMs=elapsed(start);result.metrics.gpuBufferBytes=capacity();
            return result;
        }
    }
};
} // namespace
std::shared_ptr<SparseMetalBackend> makeSparseMetalBackend(
    const std::vector<std::uint8_t>& gradientLut,const std::vector<std::uint8_t>& responseLut) {
    return std::make_shared<MetalBackend>(gradientLut,responseLut);
}
} // namespace fpm::research::detail
