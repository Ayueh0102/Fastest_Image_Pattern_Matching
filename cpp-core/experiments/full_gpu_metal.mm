// SPDX-License-Identifier: BSD-2-Clause
#include "full_gpu.hpp"
#include "research_pipeline.hpp"
#include "full_gpu_types.h"
#include "full_gpu_metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fpm::research {
namespace {
using Clock=std::chrono::steady_clock;
constexpr double pi=3.14159265358979323846;
constexpr size_t arenaLimit=64*1024*1024, statisticsLimit=16*1024*1024, totalLimit=size_t(1024)*1024*1024;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
float scoreCutoff(double value) {
    const float rounded=float(value);
    return double(rounded)<value ? std::nextafter(rounded,std::numeric_limits<float>::infinity()):rounded;
}
size_t aligned(size_t n){return (n+255)&~size_t(255);}
struct Scale {uint32_t sw,sh,w,h;float inverse;uint32_t reserved[3];};
struct Coarse {
    uint32_t sw,sh,w,h,tw,th,first,limit,statsIndex,borderIndex,windows,model;
    float cosine,sine,anchorX,anchorY,translationX,translationY,angle,threshold,overlap,scale;
    uint32_t reserved[2];
};
struct StatsConfig {uint32_t sw,tw,th,width;uint64_t first;uint32_t count,reserved;};
static_assert(sizeof(FGCandidate)==48 && sizeof(FGFineConfig)==64 && sizeof(FGResult)==64 && sizeof(Coarse)==96 && sizeof(StatsConfig)==32,"Full GPU ABI");
std::runtime_error failure(const char* what,NSError* error) {
    return std::runtime_error(std::string(what)+(error?std::string(": ")+[[error localizedDescription] UTF8String]:""));
}
uint32_t dimension(double value) {
    if(!std::isfinite(value)||value<.5||value>32768)throw std::invalid_argument("Full GPU scaled dimension must be 1..32768");
    const double lower=std::floor(value),part=value-lower;
    const auto result=uint32_t(lower)+(part>.5 || (part==.5 && (uint32_t(lower)&1)));
    if(!result)throw std::invalid_argument("Full GPU scale rounds dimension to zero");return result;
}
double angle(double a){a=std::remainder(a,360.);return a<=-180?a+360:a;}
std::vector<double> angles(const MatchConfig& c,double step) {
    std::vector<double> result;
    auto add=[&](double a){a=angle(a);for(double b:result)if(std::abs(angle(a-b))<1e-9)return;result.push_back(a);};
    if(c.angleRanges.empty()) {
        add(0);for(double a=step;a<c.toleranceAngle;a+=step)add(a);add(c.toleranceAngle);
        for(double a=-step;a>-c.toleranceAngle;a-=step)add(a);add(-c.toleranceAngle);
    } else for(auto r:c.angleRanges){for(double a=r.min;a<r.max;a+=step)add(-a);add(-r.max);}
    return result;
}
struct Model {uint32_t levelBase,top,first,count;std::vector<Coarse> maps;};
struct Plan {
    std::vector<FGLevel> levels;std::vector<Model> models;
    std::vector<std::array<uint32_t,3>> source; // width,height,byte offset
    size_t sourceBytes=0,templateBytes=0,canvasBytes=0,scoreBytes=0,statsBytes=0;
    uint32_t candidates=0,capacity=0;
};
class MetalFull final:public FullGpuPipeline {
    Image source_,templ_;Settings settings_;
    id<MTLDevice> device_;id<MTLCommandQueue> queue_;id<MTLLibrary> library_;
    std::map<std::string,id<MTLComputePipelineState>> pipelines_;
    std::map<std::string,id<MTLBuffer>> buffers_;
    FullGpuMetrics metrics_{};
    uint32_t fineRoiCacheBytes_=16384;
    id<MTLBuffer> buffer(const char* name,size_t bytes,bool shared=false) {
        bytes=std::max<size_t>(256,aligned(bytes));
        if(bytes>device_.maxBufferLength || bytes>totalLimit)throw std::invalid_argument("Full GPU buffer exceeds explicit 1 GiB/device limit");
        auto& b=buffers_[name];
        if(!b || b.length<bytes) {
            size_t total=bytes;for(const auto& p:buffers_)if(p.first!=name)total+=p.second.length;
            if(total>totalLimit)throw std::invalid_argument("Full GPU retained buffers exceed explicit 1 GiB budget");
            auto replacement=[device_ newBufferWithLength:bytes options:shared?MTLResourceStorageModeShared:MTLResourceStorageModePrivate];
            if(!replacement)throw std::runtime_error("Full GPU buffer allocation failed");b=replacement;b.label=[NSString stringWithUTF8String:name];
        }
        return b;
    }
    id<MTLComputePipelineState> pipeline(const char* name) {
        auto it=pipelines_.find(name);if(it!=pipelines_.end())return it->second;
        NSError* error=nil;auto function=[library_ newFunctionWithName:[NSString stringWithUTF8String:name]];
        if(!function)throw std::runtime_error(std::string("Missing complete GPU kernel: ")+name);
        auto p=[device_ newComputePipelineStateWithFunction:function error:&error];if(!p)throw failure(name,error);
        if(p.maxTotalThreadsPerThreadgroup<256 || p.staticThreadgroupMemoryLength>device_.maxThreadgroupMemoryLength)
            throw std::runtime_error(std::string("Unsupported full GPU workgroup: ")+name+
                " max_threads="+std::to_string(p.maxTotalThreadsPerThreadgroup)+
                " group_bytes="+std::to_string(p.staticThreadgroupMemoryLength)+
                " device_group_bytes="+std::to_string(device_.maxThreadgroupMemoryLength));
        pipelines_[name]=p;return p;
    }
    id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> command,const char* name) {
        auto e=[command computeCommandEncoder];if(!e)throw std::runtime_error("Full GPU encoder creation failed");
        e.label=[NSString stringWithUTF8String:name];[e setComputePipelineState:pipeline(name)];return e;
    }
    void linear(id<MTLComputeCommandEncoder> e,size_t n) {
        [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];++metrics_.dispatches;
    }
    void groups(id<MTLComputeCommandEncoder> e,size_t n,size_t lanes=256) {
        [e dispatchThreadgroups:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(lanes,1,1)];[e endEncoding];++metrics_.dispatches;
    }
    void zero(id<MTLCommandBuffer> command,id<MTLBuffer> b) {
        auto e=[command blitCommandEncoder];if(!e)throw std::runtime_error("Full GPU blit encoder creation failed");
        [e fillBuffer:b range:NSMakeRange(0,b.length) value:0];[e endEncoding];
    }
    void copy(id<MTLCommandBuffer> command,id<MTLBuffer> src,size_t from,id<MTLBuffer> dst,size_t to,size_t bytes) {
        auto e=[command blitCommandEncoder];if(!e)throw std::runtime_error("Full GPU blit encoder creation failed");
        [e copyFromBuffer:src sourceOffset:from toBuffer:dst destinationOffset:to size:bytes];[e endEncoding];
    }
    Plan plan(const Options& o) {
        Plan p;uint32_t maxTop=0;
        const uint64_t side=uint64_t(std::sqrt(double(settings_.minReducedArea)));
        for(size_t i=0;i<settings_.scales.size();++i) {
            double scale=settings_.scales[i];if(!std::isfinite(scale)||scale<=0)throw std::invalid_argument("Full GPU scale must be finite and positive");
            if(o.refine && std::abs(std::log(scale))>3.)throw std::invalid_argument("Full GPU refinement requires seed log scale in [-3,3]");
            uint32_t w=dimension(templ_.width()*scale),h=dimension(templ_.height()*scale);
            uint64_t area=uint64_t(w)*h;if(area>UINT64_MAX/65025/area)throw std::invalid_argument("Full GPU template centered moments exceed uint64");
            Model m{uint32_t(p.levels.size()),0,0,0,{}};
            do {
                const auto offset=aligned(p.templateBytes);if(offset+size_t(w)*h>totalLimit)throw std::invalid_argument("Full GPU model bank too large");
                p.levels.push_back({0,0,w,h,0,uint32_t(offset),uint32_t(p.levels.size()),0});p.templateBytes=offset+size_t(w)*h;
                if(area<=side*side || (w==1&&h==1))break;
                area/=4;w=(w+1)/2;h=(h+1)/2;++m.top;
            }while(true);
            maxTop=std::max(maxTop,m.top);p.models.push_back(m);
        }
        uint32_t sw=uint32_t(source_.width()),sh=uint32_t(source_.height());
        if(uint64_t(sw)*sh>128ull*1024*1024)throw std::invalid_argument("Full GPU source exceeds 128M gray pixels");
        for(uint32_t i=0;i<=maxTop;++i) {
            size_t offset=aligned(p.sourceBytes);p.source.push_back({sw,sh,uint32_t(offset)});p.sourceBytes=offset+size_t(sw)*sh;sw=(sw+1)/2;sh=(sh+1)/2;
        }
        for(size_t i=0;i<p.models.size();++i) {
            auto& m=p.models[i];m.first=p.candidates;
            for(uint32_t layer=0;layer<=m.top;++layer) {
                auto& l=p.levels[m.levelBase+layer];l.sw=p.source[layer][0];l.sh=p.source[layer][1];l.sourceOffset=p.source[layer][2];
            }
            const auto base=p.levels[m.levelBase],top=p.levels[m.levelBase+m.top];
            if(base.tw>uint32_t(source_.width()) || base.th>uint32_t(source_.height()))continue;
            const double step=std::atan(2./std::max(top.tw,top.th))*180/pi;
            for(double a:angles(o.match,step)) {
                const double c=std::cos(a*pi/180),s=std::sin(a*pi/180),ac=std::abs(c),as=std::abs(s);
                double w=2*std::ceil(((top.sw-1)*ac+(top.sh-1)*as)*.5-top.th*as*ac);
                double h=2*std::ceil(((top.sw-1)*as+(top.sh-1)*ac)*.5-top.tw*as*ac);
                if(as<1e-12){w=top.sw;h=top.sh;}else if(ac<1e-12){w=top.sh;h=top.sw;}
                if(w<top.tw || h<top.th){w=std::ceil(top.sw*ac+top.sh*as);h=std::ceil(top.sw*as+top.sh*ac);}
                if(w<top.tw || h<top.th)continue;
                if(w*h>arenaLimit || (w-top.tw+1)*(h-top.th+1)>arenaLimit/sizeof(float))throw std::invalid_argument("Full GPU coarse canvas/map exceeds 64 MiB arena");
                const double cx=(top.sw-1)*.5,cy=(top.sh-1)*.5,tx=(w-1)*.5-cx,ty=(h-1)*.5-cy;
                const double ax=cx+c*(-tx-cx)-s*(-ty-cy),ay=cy+s*(-tx-cx)+c*(-ty-cy);
                const uint32_t count=uint32_t(o.match.maxTargets+5);
                double threshold=(o.refine?std::min(o.proposalScore,o.match.minScore):o.match.minScore)*std::pow(.9,m.top);
                const float cutoff=scoreCutoff(threshold);
                Coarse map{top.sw,top.sh,uint32_t(w),uint32_t(h),top.tw,top.th,p.candidates,count,top.statisticsIndex,base.statisticsIndex,
                    uint32_t((w-top.tw+1)*(h-top.th+1)),uint32_t(i),float(c),float(s),float(ax),float(ay),float(tx),float(ty),float(a),cutoff,float(o.match.maxOverlap),float(settings_.scales[i]),{0,0}};
                m.maps.push_back(map);p.candidates+=count;m.count+=count;
                if(p.candidates>2000000)throw std::invalid_argument("Full GPU candidate arena exceeds 2M explicit slots");
                p.canvasBytes=std::max(p.canvasBytes,size_t(w*h));p.scoreBytes=std::max(p.scoreBytes,size_t(map.windows)*4);
                const size_t bytes=uint64_t(top.tw)*top.th<=UINT32_MAX/65025 ? 12:24;
                p.statsBytes=std::max(p.statsBytes,std::min(statisticsLimit,size_t(map.windows)*bytes));
            }
        }
        const size_t capacity=settings_.scales.size()*size_t(o.match.maxTargets);
        if(capacity>65536)throw std::invalid_argument("Full GPU proposal capacity exceeds 65536; reduce scale/target bank");
        p.capacity=uint32_t(capacity);return p;
    }
public:
    MetalFull(ImageView source,ImageView templ,const Settings& settings):source_(source),templ_(templ),settings_(settings) {
        const uint64_t area=uint64_t(templ_.width())*templ_.height();
        if(area>UINT64_MAX/65025/area)throw std::invalid_argument("Full GPU original template centered moments exceed uint64");
        for(double scale:settings_.scales) {
            if(!std::isfinite(scale)||scale<=0)throw std::invalid_argument("Full GPU scale must be finite and positive");
            dimension(templ_.width()*scale);dimension(templ_.height()*scale);
        }
        @autoreleasepool {
            device_=MTLCreateSystemDefaultDevice();if(!device_)throw std::runtime_error("Metal unavailable for complete GPU pipeline");
            queue_=[device_ newCommandQueue];if(!queue_)throw std::runtime_error("Metal queue unavailable");
            MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
            // Shader Validation duplicates threadgroup allocations. Keep its
            // instrumented fine kernel within the same device's 32 KiB limit.
            // Only cache capacity changes: larger ROIs use identical GPU samples.
            const char* validation=std::getenv("MTL_SHADER_VALIDATION");
            if(validation && std::strcmp(validation,"1")==0)fineRoiCacheBytes_=4096;
            options.preprocessorMacros=@{@"FG_FINE_ROI_CAPACITY":@(fineRoiCacheBytes_)};
            NSError* error=nil;library_=[device_ newLibraryWithSource:[NSString stringWithUTF8String:fpmFullGpuMetalSource] options:options error:&error];
            if(!library_)throw failure("Full GPU shader compilation",error);
            // Compile every entry before timed calls; unsupported devices fail
            // explicitly without partially running a host replacement.
            for(const char* name:{"fg_input","fg_scale_template","pyramid_direct","fg_learn","fg_rotate","coarse_strip4_32","coarse_strip4_64",
                "fg_normalize32","fg_normalize64","fg_coarse_peaks","fg_fine_search","fg_model_nms","fg_global_nms","fg_source_moments",
                "fg_pose_config","fg_center_source","fg_center_template","fg_compact_prepare","pose_refine_fused","fg_prepare_rescue",
                "fg_select_continuations","fg_accept_continuations","fg_score_states","fg_publish_states","fg_write_results"})pipeline(name);
        }
    }
    void updateSource(ImageView source) override {Image replacement(source);source_=std::move(replacement);}
    Result run(const Options& o) override {
      @autoreleasepool {
        const auto started=Clock::now();metrics_={};metrics_.enabled=true;metrics_.fineRoiCacheBytes=fineRoiCacheBytes_;
        if(o.overlap || o.fineProfiling || o.scaleSupportRefine || o.match.coarseNccPruning)
            throw std::invalid_argument("Complete GPU pipeline does not support host overlap, fine profiling, CPU scale-support correction or pruning");
        if(o.match.angleRanges.size()>16)throw std::invalid_argument("Complete GPU pipeline supports at most 16 angle intervals");
        Plan p=plan(o);
        std::vector<FGRange> ranges;for(auto r:o.match.angleRanges)ranges.push_back({float(r.min),float(r.max)});
        if(ranges.empty())ranges.push_back({-float(o.match.toleranceAngle),float(o.match.toleranceAngle)});
        const uint32_t sw=source_.width(),sh=source_.height(),tw=templ_.width(),th=templ_.height();
        const FGFinishConfig finish{sw,sh,tw,th,uint32_t(p.models.size()),uint32_t(o.match.maxTargets),p.capacity,
            1u|((o.match.toleranceAngle!=0 || !o.match.angleRanges.empty())?2u:0u)|(settings_.scales.size()>1?4u:0u),
            scoreCutoff(o.match.minScore),float(o.match.maxOverlap),uint32_t(ranges.size()),uint32_t(o.refine)};
        const size_t proposalResultOffset=256,matchResultOffset=aligned(proposalResultOffset+size_t(p.capacity)*sizeof(FGResult));
        auto rawSource=buffer("input-source",size_t(sw)*sh,true),rawTemplate=buffer("input-template",size_t(tw)*th,true);
        auto source=buffer("source-pyramid",p.sourceBytes),templ=buffer("template-pyramids",p.templateBytes),stats=buffer("template-moments",(p.levels.size()+1)*sizeof(FGStats));
        auto levels=buffer("level-descriptors",p.levels.size()*sizeof(FGLevel),true),rangeBuffer=buffer("angle-ranges",ranges.size()*sizeof(FGRange),true);
        auto canvas=buffer("rotated-canvas",p.canvasBytes),scoremap=buffer("coarse-scores",p.scoreBytes),statistics=buffer("coarse-statistics",p.statsBytes);
        auto candidates=buffer("fine-candidates",size_t(p.candidates)*sizeof(FGCandidate));
        auto modelOutput=buffer("model-proposals",size_t(p.capacity)*sizeof(FGCandidate)),counts=buffer("model-counts",p.models.size()*4);
        auto compact=buffer("compact-proposals",size_t(p.capacity)*sizeof(FGCandidate)),snapshot=buffer("proposal-snapshot",size_t(p.capacity)*sizeof(FGCandidate));
        auto matches=buffer("final-matches",size_t(o.match.maxTargets)*sizeof(FGCandidate)),status=buffer("status",sizeof(FGStatus));
        auto readback=buffer("final-readback",matchResultOffset+size_t(o.match.maxTargets)*sizeof(FGResult),true);
        auto config=buffer("pose-config",80),seeds=buffer("original-seed-states",size_t(p.capacity)*208),states=buffer("pose-states",size_t(p.capacity)*208);
        auto upload=Clock::now();
        std::memcpy(rawSource.contents,source_.view().data,size_t(sw)*sh);std::memcpy(rawTemplate.contents,templ_.view().data,size_t(tw)*th);
        std::memcpy(levels.contents,p.levels.data(),p.levels.size()*sizeof(FGLevel));std::memcpy(rangeBuffer.contents,ranges.data(),ranges.size()*sizeof(FGRange));
        metrics_.uploadBytes=size_t(sw)*sh+size_t(tw)*th+p.levels.size()*sizeof(FGLevel)+ranges.size()*sizeof(FGRange);metrics_.uploadMs=elapsed(upload);
        auto command=[queue_ commandBuffer];if(!command)throw std::runtime_error("Full GPU command creation failed");command.label=@"Full resident template matching";
        const auto encodeStart=Clock::now();
        zero(command,candidates);zero(command,modelOutput);zero(command,counts);zero(command,status);zero(command,config);
        auto e=encoder(command,"fg_input");std::array<uint32_t,2> input{sw*sh,uint32_t(o.match.invertSource)};
        [e setBuffer:rawSource offset:0 atIndex:0];[e setBuffer:source offset:0 atIndex:1];[e setBytes:input.data() length:sizeof(input) atIndex:2];linear(e,sw*sh);
        for(size_t l=1;l<p.source.size();++l) {
            const auto a=p.source[l-1],b=p.source[l];std::array<uint32_t,4> c{a[0],a[1],b[0],b[1]};
            e=encoder(command,"pyramid_direct");[e setBuffer:source offset:a[2] atIndex:0];[e setBuffer:source offset:b[2] atIndex:1];[e setBytes:c.data() length:sizeof(c) atIndex:2];
            [e dispatchThreads:MTLSizeMake(b[0],b[1],1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];++metrics_.dispatches;
        }
        auto learn=[&](id<MTLBuffer> image,size_t offset,uint32_t area,uint32_t index){
            std::array<uint32_t,2> c{area,index};auto en=encoder(command,"fg_learn");[en setBuffer:image offset:offset atIndex:0];[en setBuffer:stats offset:0 atIndex:1];[en setBytes:c.data() length:sizeof(c) atIndex:2];groups(en,1);
        };
        const uint32_t originalStats=uint32_t(p.levels.size());learn(rawTemplate,0,tw*th,originalStats);
        for(size_t m=0;m<p.models.size();++m) {
            const auto& model=p.models[m];const auto base=p.levels[model.levelBase];
            Scale c{tw,th,base.tw,base.th,float(1./settings_.scales[m]),{0,0,0}};
            e=encoder(command,"fg_scale_template");[e setBuffer:rawTemplate offset:0 atIndex:0];[e setBuffer:templ offset:base.templateOffset atIndex:1];[e setBytes:&c length:sizeof(c) atIndex:2];linear(e,size_t(base.tw)*base.th);
            learn(templ,base.templateOffset,base.tw*base.th,base.statisticsIndex);
            for(uint32_t l=1;l<=model.top;++l) {
                const auto a=p.levels[model.levelBase+l-1],b=p.levels[model.levelBase+l];std::array<uint32_t,4> cfg{a.tw,a.th,b.tw,b.th};
                e=encoder(command,"pyramid_direct");[e setBuffer:templ offset:a.templateOffset atIndex:0];[e setBuffer:templ offset:b.templateOffset atIndex:1];[e setBytes:cfg.data() length:sizeof(cfg) atIndex:2];
                [e dispatchThreads:MTLSizeMake(b.tw,b.th,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];++metrics_.dispatches;learn(templ,b.templateOffset,b.tw*b.th,b.statisticsIndex);
            }
            const auto top=p.levels[model.levelBase+model.top];
            for(const auto& map:model.maps) {
                e=encoder(command,"fg_rotate");[e setBuffer:source offset:top.sourceOffset atIndex:0];[e setBuffer:canvas offset:0 atIndex:1];[e setBuffer:stats offset:0 atIndex:2];[e setBytes:&map length:sizeof(map) atIndex:3];linear(e,size_t(map.w)*map.h);
                bool u32=uint64_t(top.tw)*top.th<=UINT32_MAX/65025;const uint32_t chunk=uint32_t(statisticsLimit/(u32?12:24));
                for(uint32_t first=0;first<map.windows;first+=chunk) {
                    uint32_t n=std::min(chunk,map.windows-first),width=map.w-map.tw+1;
                    StatsConfig cfg{map.w,map.tw,map.th,width,first,n,0};
                    e=encoder(command,u32?"coarse_strip4_32":"coarse_strip4_64");[e setBuffer:canvas offset:0 atIndex:0];[e setBuffer:templ offset:top.templateOffset atIndex:1];[e setBuffer:statistics offset:0 atIndex:2];[e setBytes:&cfg length:sizeof(cfg) atIndex:3];
                    uint32_t rows=(first+n-1)/width-first/width+1;[e dispatchThreads:MTLSizeMake((width+3)/4,rows,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];++metrics_.dispatches;
                    std::array<uint32_t,4> norm{first,n,top.statisticsIndex,0};e=encoder(command,u32?"fg_normalize32":"fg_normalize64");[e setBuffer:statistics offset:0 atIndex:0];[e setBuffer:scoremap offset:0 atIndex:1];[e setBuffer:stats offset:0 atIndex:2];[e setBytes:norm.data() length:sizeof(norm) atIndex:3];linear(e,n);
                }
                e=encoder(command,"fg_coarse_peaks");[e setBuffer:scoremap offset:0 atIndex:0];[e setBuffer:candidates offset:0 atIndex:1];[e setBytes:&map length:sizeof(map) atIndex:2];groups(e,1);
            }
            if(model.count) {
                FGFineConfig fine{model.first,model.count,model.levelBase,model.top,std::min(uint32_t(o.match.stopLayer),model.top),uint32_t(o.match.subpixel),uint32_t(ranges.size()),uint32_t(m),
                    scoreCutoff(o.refine?std::min(o.proposalScore,o.match.minScore):o.match.minScore),float(o.match.maxOverlap),float(settings_.scales[m]),0,base.tw,base.th,uint32_t(o.match.maxTargets),0};
                e=encoder(command,"fg_fine_search");[e setBuffer:source offset:0 atIndex:0];[e setBuffer:templ offset:0 atIndex:1];[e setBuffer:levels offset:0 atIndex:2];[e setBuffer:stats offset:0 atIndex:3];[e setBuffer:candidates offset:0 atIndex:4];[e setBytes:&fine length:sizeof(fine) atIndex:5];[e setBuffer:rangeBuffer offset:0 atIndex:6];groups(e,model.count);
                FGNmsConfig nms{model.first,model.count,uint32_t(m)*uint32_t(o.match.maxTargets),uint32_t(o.match.maxTargets),base.tw,base.th,0,uint32_t(ranges.size()),fine.minScore,fine.maxOverlap,uint32_t(m),0};
                e=encoder(command,"fg_model_nms");[e setBuffer:candidates offset:0 atIndex:0];[e setBuffer:modelOutput offset:0 atIndex:1];[e setBuffer:counts offset:0 atIndex:2];[e setBuffer:status offset:0 atIndex:3];[e setBytes:&nms length:sizeof(nms) atIndex:4];[e setBuffer:rangeBuffer offset:0 atIndex:5];groups(e,1);
            }
        }
        id<MTLBuffer> floatSource=nil,floatTemplate=nil;
        if(o.refine) {
            floatSource=buffer("pose-source",size_t(sw)*sh*4);floatTemplate=buffer("pose-template",size_t(tw)*th*4);
            auto partials=buffer("source-moment-partials",size_t((uint64_t(sw)*sh+4095)/4096)*sizeof(FGStats));
            uint32_t count=sw*sh;e=encoder(command,"fg_source_moments");[e setBuffer:source offset:0 atIndex:0];[e setBuffer:partials offset:0 atIndex:1];[e setBytes:&count length:4 atIndex:2];groups(e,(uint64_t(count)+4095)/4096);
            e=encoder(command,"fg_pose_config");[e setBuffer:partials offset:0 atIndex:0];[e setBuffer:stats offset:0 atIndex:1];[e setBuffer:config offset:0 atIndex:2];[e setBytes:&finish length:sizeof(finish) atIndex:3];[e setBytes:&originalStats length:4 atIndex:4];groups(e,1);
            e=encoder(command,"fg_center_source");[e setBuffer:source offset:0 atIndex:0];[e setBuffer:floatSource offset:0 atIndex:1];[e setBuffer:config offset:0 atIndex:2];linear(e,size_t(sw)*sh);
            e=encoder(command,"fg_center_template");[e setBuffer:rawTemplate offset:0 atIndex:0];[e setBuffer:floatTemplate offset:0 atIndex:1];[e setBuffer:config offset:0 atIndex:2];groups(e,1);
        }
        e=encoder(command,"fg_compact_prepare");[e setBuffer:modelOutput offset:0 atIndex:0];[e setBuffer:counts offset:0 atIndex:1];[e setBuffer:compact offset:0 atIndex:2];[e setBuffer:seeds offset:0 atIndex:3];[e setBuffer:states offset:0 atIndex:4];[e setBuffer:config offset:0 atIndex:5];[e setBuffer:status offset:0 atIndex:6];[e setBytes:&finish length:sizeof(finish) atIndex:7];groups(e,1,1);
        copy(command,compact,0,snapshot,0,size_t(p.capacity)*sizeof(FGCandidate));
        if(o.refine) {
            auto refine=[&](id<MTLBuffer> state,id<MTLBuffer> cfg,uint32_t cap,uint32_t iterations){auto en=encoder(command,"pose_refine_fused");[en setBuffer:floatSource offset:0 atIndex:0];[en setBuffer:floatTemplate offset:0 atIndex:1];[en setBuffer:state offset:0 atIndex:2];[en setBuffer:cfg offset:0 atIndex:3];[en setBytes:&iterations length:4 atIndex:4];groups(en,cap);};
            refine(states,config,p.capacity,30);
            auto retries=buffer("rescue-states",64*208),retryOwners=buffer("rescue-owners",64*4),parents=buffer("rescue-parents",4*4),baseScores=buffer("rescue-base-scores",4*4),retryConfig=buffer("rescue-config",80);
            auto continuations=buffer("continuation-states",4*208),continuationOwners=buffer("continuation-owners",4*4),continuationConfig=buffer("continuation-config",80);
            e=encoder(command,"fg_prepare_rescue");[e setBuffer:seeds offset:0 atIndex:0];[e setBuffer:states offset:0 atIndex:1];[e setBuffer:retries offset:0 atIndex:2];[e setBuffer:retryOwners offset:0 atIndex:3];[e setBuffer:parents offset:0 atIndex:4];[e setBuffer:baseScores offset:0 atIndex:5];[e setBuffer:retryConfig offset:0 atIndex:6];[e setBuffer:status offset:0 atIndex:7];[e setBytes:&finish length:sizeof(finish) atIndex:8];[e setBuffer:config offset:0 atIndex:9];groups(e,1,1);
            refine(retries,retryConfig,64,8);
            e=encoder(command,"fg_select_continuations");[e setBuffer:states offset:0 atIndex:0];[e setBuffer:retries offset:0 atIndex:1];[e setBuffer:retryOwners offset:0 atIndex:2];[e setBuffer:parents offset:0 atIndex:3];[e setBuffer:baseScores offset:0 atIndex:4];[e setBuffer:continuations offset:0 atIndex:5];[e setBuffer:continuationOwners offset:0 atIndex:6];[e setBuffer:continuationConfig offset:0 atIndex:7];[e setBuffer:status offset:0 atIndex:8];[e setBuffer:retryConfig offset:0 atIndex:9];groups(e,1,1);
            refine(continuations,continuationConfig,4,30);
            e=encoder(command,"fg_accept_continuations");[e setBuffer:states offset:0 atIndex:0];[e setBuffer:continuations offset:0 atIndex:1];[e setBuffer:continuationOwners offset:0 atIndex:2];[e setBuffer:status offset:0 atIndex:3];groups(e,1,1);
            e=encoder(command,"fg_score_states");[e setBuffer:floatSource offset:0 atIndex:0];[e setBuffer:floatTemplate offset:0 atIndex:1];[e setBuffer:states offset:0 atIndex:2];[e setBuffer:config offset:0 atIndex:3];groups(e,p.capacity);
            e=encoder(command,"fg_publish_states");[e setBuffer:compact offset:0 atIndex:0];[e setBuffer:states offset:0 atIndex:1];[e setBuffer:status offset:0 atIndex:2];[e setBytes:&finish length:sizeof(finish) atIndex:3];linear(e,p.capacity);
        }
        FGNmsConfig finalNms{0,p.capacity,0,uint32_t(o.match.maxTargets),tw,th,1,uint32_t(ranges.size()),scoreCutoff(o.match.minScore),float(o.match.maxOverlap),0,0};
        e=encoder(command,"fg_global_nms");[e setBuffer:compact offset:0 atIndex:0];[e setBuffer:matches offset:0 atIndex:1];[e setBuffer:counts offset:0 atIndex:2];[e setBuffer:status offset:0 atIndex:3];[e setBytes:&finalNms length:sizeof(finalNms) atIndex:4];[e setBuffer:rangeBuffer offset:0 atIndex:5];groups(e,1);
        for(uint32_t kind=0;kind<2;++kind) {
            e=encoder(command,"fg_write_results");[e setBuffer:kind?matches:snapshot offset:0 atIndex:0];[e setBuffer:readback offset:kind?matchResultOffset:proposalResultOffset atIndex:1];[e setBuffer:status offset:0 atIndex:2];[e setBytes:&finish length:sizeof(finish) atIndex:3];[e setBytes:&kind length:4 atIndex:4];linear(e,kind?size_t(o.match.maxTargets):p.capacity);
        }
        copy(command,status,0,readback,0,sizeof(FGStatus));metrics_.encodeMs=elapsed(encodeStart);
        // The sole synchronization boundary. All intermediate allocations are
        // MTLStorageModePrivate and never expose .contents to host code.
        const auto wait=Clock::now();[command commit];++metrics_.commandBuffers;[command waitUntilCompleted];++metrics_.cpuWaits;metrics_.waitMs=elapsed(wait);
        if(command.status!=MTLCommandBufferStatusCompleted)throw failure("Complete GPU execution",command.error);
        metrics_.deviceMs=command.GPUStartTime>0 && command.GPUEndTime>=command.GPUStartTime ? 1000*(command.GPUEndTime-command.GPUStartTime):-1;
        const auto read=Clock::now();const auto* bytes=static_cast<const uint8_t*>(readback.contents);FGStatus resultStatus{};std::memcpy(&resultStatus,bytes,sizeof(resultStatus));
        if(resultStatus.overflow || resultStatus.proposals>p.capacity || resultStatus.matches>uint32_t(o.match.maxTargets))throw std::runtime_error("Full GPU overflow/capacity flag: "+std::to_string(resultStatus.overflow));
        Result r;
        auto collect=[&](size_t offset,uint32_t count,std::vector<Match>& output,std::vector<double>& scales){
            auto* values=reinterpret_cast<const FGResult*>(bytes+offset);output.reserve(count);scales.reserve(count);
            for(uint32_t i=0;i<count;++i){const auto& v=values[i];Match m;m.center={v.x,v.y};m.angle=v.angle;m.score=v.score;for(size_t k=0;k<4;++k)m.corners[k]={v.corners[2*k],v.corners[2*k+1]};output.push_back(m);scales.push_back(v.scale);}
        };
        collect(proposalResultOffset,resultStatus.proposals,r.proposals,r.proposalScales);collect(matchResultOffset,resultStatus.matches,r.matches,r.scales);
        metrics_.readbackBytes=sizeof(FGStatus)+size_t(resultStatus.proposals+resultStatus.matches)*sizeof(FGResult);metrics_.readbackMs=elapsed(read);
        for(const auto& entry:buffers_)metrics_.gpuBytes+=entry.second.length;
        r.fullGpu=metrics_;r.refinementCandidates=o.refine?resultStatus.proposals:0;r.rescue.selected_seeds=resultStatus.rescueParents;r.rescue.retry_starts=resultStatus.rescueStarts;r.rescue.continuations=resultStatus.rescueContinuations;r.configuredRefinementBackend=1;
        r.refinementMode=1;r.poseInputMode=1;r.rescue.selected_backend=r.refinementCandidates?1:0;
        if(r.refinementCandidates) {
            const auto solver=pipeline("pose_refine_fused");
            r.residentKernel={256,uint32_t(solver.threadExecutionWidth),solver.staticThreadgroupMemoryLength};
        }
        r.frame.models_matched=p.models.size();r.totalMs=elapsed(started);return r;
      }
    }
};
}
std::unique_ptr<FullGpuPipeline> makeFullGpuPipeline(ImageView source,ImageView templ,const Settings& settings){return std::make_unique<MetalFull>(source,templ,settings);}
}
