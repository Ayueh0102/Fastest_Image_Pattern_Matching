// SPDX-License-Identifier: BSD-2-Clause
// Peak mode 2 differential test. Ordered candidate bytes of the batched
// parallel-round kernel must equal the tiled baseline (mode 1) byte for byte
// and an independent scalar greedy reference, for single maps, batches of maps
// in separate ring slots, and every supported lane count. Ring scores must stay
// unmodified.
#include "full_gpu_types.h"
#include "full_gpu_peaks_types.h"
#include "full_gpu_metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Coarse {
    uint32_t sw,sh,w,h,tw,th,first,limit,statsIndex,borderIndex,windows,model;
    float cosine,sine,anchorX,anchorY,translationX,translationY,angle,threshold,overlap,scale;
    uint32_t reserved[2];
};
static_assert(sizeof(Coarse)==96 && sizeof(FGCandidate)==48 && sizeof(FGPeakBatch)==16);
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
bool better(float a,uint32_t ai,float b,uint32_t bi) {return a>b || (a==b && ai<bi);}
// Sequential greedy selection, written independently of every kernel.
std::vector<uint32_t> reference(std::vector<float> scores,const Coarse& c) {
    std::vector<uint32_t> result;
    const uint32_t width=c.w-c.tw+1,height=c.h-c.th+1;
    for(uint32_t k=0;k<c.limit;++k) {
        float best=-std::numeric_limits<float>::infinity();uint32_t bi=UINT32_MAX;
        for(uint32_t i=0;i<c.windows;++i)if(better(scores[i],i,best,bi)) {best=scores[i];bi=i;}
        if(!std::isfinite(best)||best<c.threshold)break;
        result.push_back(bi);if(k+1==c.limit)break;
        const uint32_t x=bi%width,y=bi/width;
        const float rw=c.tw*(1-c.overlap),rh=c.th*(1-c.overlap);
        const int left=int(float(x)-rw),top=int(float(y)-rh);
        const uint32_t x0=uint32_t(std::max(0,std::min(int(x),left)));
        const uint32_t x1=uint32_t(std::min(int(width),std::max(int(x)+1,left+int(2*rw))));
        const uint32_t y0=uint32_t(std::max(0,std::min(int(y),top)));
        const uint32_t y1=uint32_t(std::min(int(height),std::max(int(y)+1,top+int(2*rh))));
        for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x)
            scores[y*width+x]=-std::numeric_limits<float>::infinity();
    }
    return result;
}
struct Fixture {std::string label;std::vector<float> scores;Coarse config;};
size_t slots(const Coarse& c) {return size_t(c.first)+c.limit+3;}
struct GPU {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> sort,tiled,parallel;
    GPU() {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
        queue=[device newCommandQueue];NSError* error=nil;
        MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:fpmFullGpuMetalSource] options:options error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto compile=[&](NSString* name) {
            auto function=[library newFunctionWithName:name];require(function!=nil,"Missing peak kernel");
            auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
            require(pipeline.maxTotalThreadsPerThreadgroup>=256,"Unsupported peak workgroup");return pipeline;
        };
        sort=compile(@"fg_coarse_sort_tiles");tiled=compile(@"fg_coarse_peaks_tiled");parallel=compile(@"fg_coarse_peaks_parallel");
    }
    id<MTLBuffer> buffer(size_t size,const void* data=nullptr) {
        auto b=[device newBufferWithLength:std::max<size_t>(256,size) options:MTLResourceStorageModeShared];
        require(b!=nil,"Metal allocation failed");std::memset(b.contents,0,b.length);
        if(data&&size)std::memcpy(b.contents,data,size);return b;
    }
    void run(id<MTLCommandBuffer> command) {
        [command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
    // Maps of one dispatch get consecutive candidate ranges, as in the pipeline.
    // listCapacity 0 forces the whole-map path; the default takes the list path
    // for every map with at most that many candidates.
    void check(const std::string& label,const std::vector<const Fixture*>& batch,NSUInteger lanes,uint32_t listCapacity) {
        require(!batch.empty() && batch.size()<=FG_PEAK_PARALLEL_MAPS,label+": invalid batch size");
        std::vector<Coarse> configs;size_t total=0,stride=0;uint32_t keyStride=1;
        for(const auto* fixture:batch) {
            Coarse c=fixture->config;c.windows=uint32_t(fixture->scores.size());c.first=uint32_t(total)+3;
            configs.push_back(c);total+=slots(c);
            stride=std::max(stride,(fixture->scores.size()+63)/64*64);keyStride=std::max(keyStride,c.limit);
        }
        auto baseline=buffer(total*sizeof(FGCandidate)),output=buffer(total*sizeof(FGCandidate));
        // Baseline: one sort and tiled merge per map on a private copy of its scores.
        auto command=[queue commandBuffer];
        std::vector<id<MTLBuffer>> retained;
        for(size_t j=0;j<batch.size();++j) {
            const auto& input=batch[j]->scores;const auto& c=configs[j];
            const uint32_t tiles=(c.windows+255)/256;
            auto scores=buffer(input.size()*4,input.data());
            auto entries=buffer(size_t(tiles)*256*sizeof(FGPeakEntry)),heads=buffer(size_t(tiles)*4);
            retained.insert(retained.end(),{scores,entries,heads});
            if(tiles) {
                auto e=[command computeCommandEncoder];[e setComputePipelineState:sort];
                [e setBuffer:scores offset:0 atIndex:0];[e setBuffer:entries offset:0 atIndex:1];[e setBytes:&c length:sizeof(c) atIndex:2];[e setBuffer:heads offset:0 atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(tiles,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
            }
            auto e=[command computeCommandEncoder];[e setComputePipelineState:tiled];
            [e setBuffer:scores offset:0 atIndex:0];[e setBuffer:baseline offset:0 atIndex:1];[e setBytes:&c length:sizeof(c) atIndex:2];
            [e setBuffer:entries offset:0 atIndex:3];[e setBuffer:heads offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
        }
        // Mode 2: every map in its own ring slot, one threadgroup each, one dispatch.
        std::vector<float> ring(stride*batch.size(),std::numeric_limits<float>::quiet_NaN());
        for(size_t j=0;j<batch.size();++j)std::copy(batch[j]->scores.begin(),batch[j]->scores.end(),ring.begin()+j*stride);
        auto scoreRing=buffer(ring.size()*4,ring.data()),states=buffer(2*stride*batch.size()),keys=buffer(size_t(keyStride)*batch.size()*8);
        auto removal=buffer(stride/8*batch.size()),lists=buffer(3*size_t(listCapacity)*4*batch.size());
        const FGPeakBatch config{uint32_t(stride),keyStride,listCapacity,0};
        auto e=[command computeCommandEncoder];[e setComputePipelineState:parallel];
        [e setBuffer:scoreRing offset:0 atIndex:0];[e setBuffer:output offset:0 atIndex:1];
        [e setBytes:configs.data() length:configs.size()*sizeof(Coarse) atIndex:2];[e setBuffer:states offset:0 atIndex:3];
        [e setBuffer:keys offset:0 atIndex:4];[e setBytes:&config length:sizeof(config) atIndex:5];[e setBuffer:removal offset:0 atIndex:6];[e setBuffer:lists offset:0 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(batch.size(),1,1) threadsPerThreadgroup:MTLSizeMake(lanes,1,1)];[e endEncoding];
        run(command);
        require(std::memcmp(baseline.contents,output.contents,total*sizeof(FGCandidate))==0,label+": ordered candidate bytes differ from mode 1");
        require(std::memcmp(scoreRing.contents,ring.data(),ring.size()*4)==0,label+": parallel selection modified its scores");
        const auto* candidates=static_cast<const FGCandidate*>(output.contents);
        for(size_t j=0;j<batch.size();++j) {
            const auto& input=batch[j]->scores;const auto& c=configs[j];
            const auto expected=reference(input,c);const std::string name=label+"/"+batch[j]->label;
            const size_t begin=c.first-3;
            for(size_t k=begin;k<begin+slots(c);++k) {
                const bool valid=k>=c.first && k<c.first+expected.size();
                require(candidates[k].valid==uint32_t(valid),name+": invalid output slot/guard was modified");
                if(!valid)continue;
                const uint32_t index=expected[k-c.first];
                require(candidates[k].score==input[index] && candidates[k].coarseScore==input[index] && candidates[k].order==k,
                    name+": scalar score/order mismatch");
                if(c.cosine==1 && c.sine==0 && c.translationX==0 && c.translationY==0)
                    require(candidates[k].x==index%(c.w-c.tw+1) && candidates[k].y==index/(c.w-c.tw+1),name+": scalar peak location mismatch");
            }
        }
    }
};
Coarse config(uint32_t width,uint32_t height,uint32_t tw=9,uint32_t th=7,uint32_t limit=64,float overlap=.2f,float threshold=.4f) {
    Coarse c{};c.w=c.sw=width+tw-1;c.h=c.sh=height+th-1;c.tw=tw;c.th=th;c.first=3;c.limit=limit;
    c.cosine=1;c.scale=1;c.overlap=overlap;c.threshold=threshold;c.model=7;return c;
}
// Smooth blobs above a noise floor, shaped like top-level many-object maps.
std::vector<float> blobs(std::mt19937& random,uint32_t width,uint32_t height,uint32_t count,float radius) {
    std::vector<float> scores(size_t(width)*height);
    std::uniform_real_distribution<float> unit(0,1);
    for(auto& score:scores)score=.3f*unit(random)-.1f;
    for(uint32_t n=0;n<count;++n) {
        const float cx=unit(random)*width,cy=unit(random)*height,peak=.55f+.45f*unit(random);
        const int r=int(std::ceil(3*radius));
        for(int y=std::max(0,int(cy)-r);y<std::min(int(height),int(cy)+r+1);++y)
            for(int x=std::max(0,int(cx)-r);x<std::min(int(width),int(cx)+r+1);++x) {
                const float d2=(x-cx)*(x-cx)+(y-cy)*(y-cy);auto& score=scores[size_t(y)*width+x];
                score=std::max(score,peak*std::exp(-d2/(2*radius*radius)));
            }
    }
    // Quantize to create exact ties inside the blobs.
    for(auto& score:scores)score=std::round(score*4096)/4096;
    return scores;
}
std::vector<Fixture> fixtures() {
    const float inf=std::numeric_limits<float>::infinity(),nan=std::numeric_limits<float>::quiet_NaN();
    std::vector<Fixture> f;
    auto add=[&](std::string label,std::vector<float> scores,Coarse c){f.push_back({std::move(label),std::move(scores),c});};
    // The round-38 suite: same fixtures and seed.
    add("empty",{},config(1,1));
    add("one",{.9f},config(1,1,1,1,1));
    add("zero limit",{.9f},config(1,1,1,1,0));
    add("all NaN",std::vector<float>(257,nan),config(257,1));
    add("all negative infinity",std::vector<float>(257,-inf),config(257,1));
    add("all below threshold",std::vector<float>(257,.1f),config(257,1));
    add("threshold equality",std::vector<float>(257,.4f),config(257,1,1,1,512,1));
    add("positive infinity stops",{nan,.9f,inf,-inf,1.f},config(5,1));
    add("NaN mixed",{nan,.9f,-inf,nan,.95f,.4f,nan,.7f},config(8,1,1,1,8,1));
    add("negative accepted",{-.8f,-.2f,nan,-.5f},config(4,1,1,1,8,1,-.75f));
    add("signed zero ties",{-0.f,+0.f,-0.f,+0.f},config(4,1,1,1,8,1,0));
    add("ties and padded final tile",std::vector<float>(65537,.75f),config(65537,1,1,1,517,1));
    add("ties broad suppression",std::vector<float>(259*37,.75f),config(259,37,17,13,129,0));
    std::vector<float> freed(1025,.05f);freed[255]=.95f;freed[256]=.99f;freed[254]=.90f;freed[261]=.85f;
    add("cross tile released neighbour",freed,config(1025,1,2,1,16,.5f));
    std::mt19937 random(0x38c0a5e);
    for(uint32_t seed=0;seed<48;++seed) {
        const uint32_t width=31+random()%293,height=1+random()%93;
        std::vector<float> scores(size_t(width)*height);
        for(auto& score:scores)score=float(int(random()%41)-10)/32.f;
        if(seed%4==0)scores[random()%scores.size()]=nan;
        if(seed%7==0)scores[random()%scores.size()]=-inf;
        const uint32_t tw=1+random()%21,th=1+random()%19;
        auto c=config(width,height,tw,th,1+random()%160,float(seed%5)*.25f,.4f);
        if(seed%3==0) {c.cosine=.8f;c.sine=.6f;c.translationX=.25f;c.translationY=-.75f;c.angle=36.8699f;c.scale=1.2f;}
        add("random "+std::to_string(seed),scores,c);
    }
    for(uint32_t limit:{128u,512u}) {
        const uint32_t width=769,height=193;std::vector<float> scores(size_t(width)*height);
        for(auto& score:scores)score=float(random()%65536)/65536.f;
        add("many "+std::to_string(limit),scores,config(width,height,5,3,limit,.5f,.4f));
    }
    // Mode 2 additions. With tw=9 and overlap .2 a rectangle spans [x-8,x+6):
    // h is covered by the lower-priority s, while its own blocker b is removed
    // by a. Removing higher-priority points inside a selected rectangle would
    // drop h, which the baseline selects in its second step.
    std::vector<float> survivor(40,.1f);survivor[10]=.99f;survivor[15]=.95f;survivor[16]=.9f;survivor[22]=.85f;
    add("asymmetric higher-priority survivor",survivor,config(40,1,9,1,8,.2f));
    // Left/top borders: trunc toward zero shifts rectangles one pixel right/down.
    std::vector<float> border(24*24,.1f);
    for(uint32_t y=0;y<6;++y)for(uint32_t x=0;x<6;++x)border[y*24+x]=.5f+.01f*float((x*7+y*3)%11);
    add("border truncation",border,config(24,24,9,7,32,.2f));
    add("NaN threshold",{nan,-inf,-.5f,.25f,-2.f,.25f},config(6,1,2,1,8,.5f,nan));
    add("negative infinity threshold",{-inf,-1e30f,nan,-3.f},config(4,1,1,1,8,1,-inf));
    add("positive infinity late",{.9f,.8f,.7f,.6f,inf},config(5,1,1,1,8,1));
    add("limit one",std::vector<float>(300,.6f),config(30,10,5,4,1,.2f));
    std::vector<float> ramp(97*61);
    for(size_t i=0;i<ramp.size();++i)ramp[i]=float((i*7919)%257)/256.f;
    add("dense distinct limit above accepted",ramp,config(97,61,6,5,1000,.35f,0));
    std::mt19937 asymmetric(0x42a5);
    for(uint32_t seed=0;seed<96;++seed) {
        const uint32_t width=8+asymmetric()%120,height=1+asymmetric()%60;
        std::vector<float> scores(size_t(width)*height);
        // Few distinct values: ties everywhere, and many chains of covers.
        for(auto& score:scores)score=float(int(asymmetric()%9))/8.f;
        if(seed%5==0)scores[asymmetric()%scores.size()]=nan;
        if(seed%11==0)scores[asymmetric()%scores.size()]=-inf;
        if(seed%31==0)scores[asymmetric()%scores.size()]=inf;
        const uint32_t tw=1+asymmetric()%25,th=1+asymmetric()%17;
        const float overlaps[]={0,.1f,.2f,.3f,.35f,.45f,.6f,.75f,.9f,1};
        auto c=config(width,height,tw,th,1+asymmetric()%200,overlaps[seed%10],.3f);
        if(seed%4==1) {c.cosine=-.6f;c.sine=.8f;c.translationX=-1.5f;c.translationY=2.25f;c.angle=126.8699f;c.scale=.75f;c.model=2;}
        add("asymmetric random "+std::to_string(seed),scores,c);
    }
    // Shaped like the formal many128/many512 top levels (18x13 and 14x10 templates).
    std::mt19937 shaped(0x512);
    add("many-object 18x13",blobs(shaped,1007,1009,600,2.5f),config(1007,1009,18,13,773,.2f,.3645f));
    add("many-object 14x10",blobs(shaped,1100,1048,700,2.f),config(1100,1048,14,10,773,.2f,.3645f));
    add("many-object 11x8",blobs(shaped,567,459,300,1.6f),config(567,459,11,8,133,.2f,.3645f));
    return f;
}
void tests(GPU& gpu) {
    const auto all=fixtures();
    std::vector<NSUInteger> lanes;
    for(NSUInteger n=256;n<=std::min<NSUInteger>(FG_PEAK_PARALLEL_LANES,gpu.parallel.maxTotalThreadsPerThreadgroup);n*=2)lanes.push_back(n);
    // Both paths at every lane count; small capacities split fixtures between them.
    for(const auto& fixture:all)for(auto n:lanes)for(uint32_t capacity:{0u,7u,uint32_t(FG_PEAK_LIST_CAPACITY)})
        gpu.check("single lanes "+std::to_string(n)+" capacity "+std::to_string(capacity),{&fixture},n,capacity);
    // Batches mix sizes, limits, thresholds and transforms in separate slots.
    std::mt19937 random(0x42b);
    std::vector<const Fixture*> order;for(const auto& fixture:all)order.push_back(&fixture);
    for(uint32_t pass=0;pass<3;++pass) {
        std::shuffle(order.begin(),order.end(),random);
        for(size_t i=0;i<order.size();) {
            const size_t count=std::min<size_t>(1+random()%FG_PEAK_PARALLEL_MAPS,order.size()-i);
            gpu.check("batch "+std::to_string(pass)+"/"+std::to_string(i),std::vector<const Fixture*>(order.begin()+i,order.begin()+i+count),
                lanes[random()%lanes.size()],pass ? uint32_t(FG_PEAK_LIST_CAPACITY):0u);
            i+=count;
        }
    }
    std::cout<<"Full GPU parallel peaks: "<<all.size()<<" fixtures x "<<lanes.size()<<" lane counts x 3 list capacities, and 3 batched passes, match mode 1 and the scalar reference\n";
}
}
int main() {
    @autoreleasepool {
        try {GPU gpu;tests(gpu);return 0;}
        catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
