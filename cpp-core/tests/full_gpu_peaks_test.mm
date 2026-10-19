// SPDX-License-Identifier: BSD-2-Clause
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
static_assert(sizeof(Coarse)==96 && sizeof(FGCandidate)==48);
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
bool better(float a,uint32_t ai,float b,uint32_t bi) {return a>b || (a==b && ai<bi);}
std::vector<uint32_t> reference(std::vector<float>& scores,const Coarse& c) {
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
struct GPU {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> original,sort,tiled;
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
        original=compile(@"fg_coarse_peaks");sort=compile(@"fg_coarse_sort_tiles");tiled=compile(@"fg_coarse_peaks_tiled");
    }
    id<MTLBuffer> buffer(size_t size,const void* data=nullptr) {
        auto b=[device newBufferWithLength:std::max<size_t>(256,size) options:MTLResourceStorageModeShared];
        require(b!=nil,"Metal allocation failed");std::memset(b.contents,0,b.length);
        if(data&&size)std::memcpy(b.contents,data,size);return b;
    }
    void check(const std::string& label,const std::vector<float>& input,Coarse c) {
        c.windows=uint32_t(input.size());const uint32_t tiles=(c.windows+255)/256;
        const size_t count=size_t(c.first)+c.limit+3;
        auto a=buffer(input.size()*4,input.data()),b=buffer(input.size()*4,input.data());
        auto outA=buffer(count*sizeof(FGCandidate)),outB=buffer(count*sizeof(FGCandidate));
        auto entries=buffer(size_t(tiles)*256*sizeof(FGPeakEntry)),heads=buffer(size_t(tiles)*4);
        auto command=[queue commandBuffer];
        auto e=[command computeCommandEncoder];[e setComputePipelineState:original];
        [e setBuffer:a offset:0 atIndex:0];[e setBuffer:outA offset:0 atIndex:1];[e setBytes:&c length:sizeof(c) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
        if(tiles) {
            e=[command computeCommandEncoder];[e setComputePipelineState:sort];
            [e setBuffer:b offset:0 atIndex:0];[e setBuffer:entries offset:0 atIndex:1];[e setBytes:&c length:sizeof(c) atIndex:2];[e setBuffer:heads offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(tiles,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
        }
        e=[command computeCommandEncoder];[e setComputePipelineState:tiled];
        [e setBuffer:b offset:0 atIndex:0];[e setBuffer:outB offset:0 atIndex:1];[e setBytes:&c length:sizeof(c) atIndex:2];
        [e setBuffer:entries offset:0 atIndex:3];[e setBuffer:heads offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
        [command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        require(std::memcmp(outA.contents,outB.contents,count*sizeof(FGCandidate))==0,label+": ordered candidate bytes differ");
        require(std::memcmp(a.contents,b.contents,input.size()*4)==0,label+": suppression map bytes differ");
        auto cpu=input;const auto expected=reference(cpu,c);
        require(std::memcmp(cpu.data(),a.contents,input.size()*4)==0,label+": GPU suppression differs from scalar reference");
        const auto* output=static_cast<const FGCandidate*>(outB.contents);
        for(size_t k=0;k<count;++k) {
            const bool valid=k>=c.first && k<c.first+expected.size();
            require(output[k].valid==uint32_t(valid),label+": invalid output slot/guard was modified");
            if(!valid)continue;
            const uint32_t index=expected[k-c.first];
            require(output[k].score==input[index] && output[k].coarseScore==input[index] && output[k].order==k,
                label+": scalar score/order mismatch");
            if(c.cosine==1 && c.sine==0 && c.translationX==0 && c.translationY==0)
                require(output[k].x==index%(c.w-c.tw+1) && output[k].y==index/(c.w-c.tw+1),
                    label+": scalar peak location mismatch");
        }
    }
};
Coarse config(uint32_t width,uint32_t height,uint32_t tw=9,uint32_t th=7,uint32_t limit=64,float overlap=.2f,float threshold=.4f) {
    Coarse c{};c.w=c.sw=width+tw-1;c.h=c.sh=height+th-1;c.tw=tw;c.th=th;c.first=3;c.limit=limit;
    c.cosine=1;c.scale=1;c.overlap=overlap;c.threshold=threshold;c.model=7;return c;
}
void tests(GPU& gpu) {
    const float inf=std::numeric_limits<float>::infinity(),nan=std::numeric_limits<float>::quiet_NaN();
    gpu.check("empty",{},config(1,1));
    gpu.check("one",{.9f},config(1,1,1,1,1));
    gpu.check("zero limit",{.9f},config(1,1,1,1,0));
    gpu.check("all NaN",std::vector<float>(257,nan),config(257,1));
    gpu.check("all negative infinity",std::vector<float>(257,-inf),config(257,1));
    gpu.check("all below threshold",std::vector<float>(257,.1f),config(257,1));
    gpu.check("threshold equality",std::vector<float>(257,.4f),config(257,1,1,1,512,1));
    gpu.check("positive infinity stops",{nan,.9f,inf,-inf,1.f},config(5,1));
    gpu.check("NaN mixed",{nan,.9f,-inf,nan,.95f,.4f,nan,.7f},config(8,1,1,1,8,1));
    gpu.check("negative accepted",{-.8f,-.2f,nan,-.5f},config(4,1,1,1,8,1,-.75f));
    gpu.check("signed zero ties",{-0.f,+0.f,-0.f,+0.f},config(4,1,1,1,8,1,0));
    // All ties cross tile boundaries. overlap=1 removes exactly the chosen pixel.
    gpu.check("ties and padded final tile",std::vector<float>(65537,.75f),config(65537,1,1,1,517,1));
    gpu.check("ties broad suppression",std::vector<float>(259*37,.75f),config(259,37,17,13,129,0));
    // A large early local value can be suppressed across a tile boundary while
    // its lower-ranked neighbour must remain eligible. Keeping only tile-local
    // greedy maxima/top-K would fail this fixture.
    std::vector<float> freed(1025,.05f);freed[255]=.95f;freed[256]=.99f;freed[254]=.90f;freed[261]=.85f;
    gpu.check("cross tile released neighbour",freed,config(1025,1,2,1,16,.5f));
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
        gpu.check("random "+std::to_string(seed),scores,c);
    }
    for(uint32_t limit:{128u,512u}) {
        const uint32_t width=769,height=193;std::vector<float> scores(size_t(width)*height);
        for(auto& score:scores)score=float(random()%65536)/65536.f;
        gpu.check("many "+std::to_string(limit),scores,config(width,height,5,3,limit,.5f,.4f));
    }
}
}
int main() {
    @autoreleasepool {
        try {GPU gpu;tests(gpu);std::cout<<"Full GPU tiled peaks: 64 scalar/original/tiled differential fixtures passed\n";return 0;}
        catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
