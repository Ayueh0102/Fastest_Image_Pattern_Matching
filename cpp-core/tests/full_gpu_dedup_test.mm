// SPDX-License-Identifier: BSD-2-Clause
// Round 43: the Metal dedup kernels against the portable dedupKeep rule on
// identical FP32 slot layouts (map-major prefixes, ties, +/-180 seam).
#include "full_gpu_types.h"
#include "full_gpu_metal_source.hpp"
#include "internal.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
struct Layout {std::vector<FGCandidate> slots;FGDedupConfig config;};
Layout make(std::mt19937& rng,uint32_t maps,uint32_t limit,uint32_t topK) {
    Layout l;l.slots.assign(size_t(maps)*limit+7,FGCandidate{});
    const float step=360.f/float(maps);
    l.config={7,limit,maps,topK,14,10,0,0,30.f,2.5f*step,8.f,0};
    std::uniform_real_distribution<float> object(5,60),jitter(-1.5f,1.5f),score(.3f,1.f);
    std::uniform_int_distribution<uint32_t> fill(0,limit);
    std::vector<std::pair<float,float>> objects(9);
    for(auto& o:objects)o={object(rng),object(rng)};
    for(uint32_t map=0;map<maps;++map) {
        // Map angles straddle the seam: -180+step/2 ... 180-step/2.
        const float angle=-180.f+step*(float(map)+.5f);
        std::vector<float> scores(fill(rng));
        for(auto& s:scores)s=score(rng);
        std::sort(scores.begin(),scores.end(),std::greater<float>());
        for(size_t j=1;j<scores.size();j+=5)scores[j]=scores[j-1]; // exact ties rank by slot order
        for(uint32_t j=0;j<scores.size();++j) {
            const auto& o=objects[(map*7+j)%objects.size()];
            const uint32_t slot=l.config.first+map*limit+j;
            l.slots[slot]={o.first+jitter(rng),o.second+jitter(rng),angle,scores[j],1,scores[j],1,slot,0,0,0,0};
        }
    }
    return l;
}
bool borderline(const Layout& l,const std::vector<fpm::detail::DedupPoint>& points) {
    const double r2=double(l.config.radius)*l.config.radius,window=l.config.angleWindow;
    for(size_t i=0;i<points.size();++i)for(size_t j=0;j<i;++j) {
        const double x=points[i].center.x-points[j].center.x,y=points[i].center.y-points[j].center.y;
        const double a=std::abs(fpm::detail::normalizeAngle(points[i].angle-points[j].angle));
        if(std::abs(x*x+y*y-r2)<1e-5*r2 || std::abs(a-window)<1e-4)return true;
    }
    return false;
}
}

int main() {
    try {
        @autoreleasepool {
            auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
            auto queue=[device newCommandQueue];NSError* error=nil;
            MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
            auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:fpmFullGpuMetalSource] options:options error:&error];
            if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
            auto compile=[&](NSString* name){
                auto function=[library newFunctionWithName:name];require(function!=nil,"Missing dedup kernel");
                auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];
                if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);return pipeline;
            };
            auto mark=compile(@"fg_dedup_mark"),apply=compile(@"fg_dedup_apply");
            std::mt19937 rng(43004);int checked=0,skipped=0;
            for(int trial=0;trial<100 && checked<40;++trial) {
                auto l=make(rng,uint32_t(12+trial%25),uint32_t(8+trial%17),uint32_t(1+trial%4));
                const auto& k=l.config;
                std::vector<uint32_t> valid;
                for(uint32_t i=0;i<k.maps*k.limit;++i)if(l.slots[k.first+i].valid)valid.push_back(k.first+i);
                std::stable_sort(valid.begin(),valid.end(),[&](uint32_t a,uint32_t b){
                    const auto& x=l.slots[a];const auto& y=l.slots[b];return x.score>y.score || (x.score==y.score && x.order<y.order);});
                std::vector<fpm::detail::DedupPoint> points;
                for(uint32_t slot:valid) {
                    const auto& c=l.slots[slot];const double r=double(c.angle)*fpm::detail::D2R;
                    points.push_back({{(c.x+.5*(k.templateWidth*std::cos(r)-k.templateHeight*std::sin(r)))*k.toSource,
                        (c.y+.5*(k.templateWidth*std::sin(r)+k.templateHeight*std::cos(r)))*k.toSource},c.angle});
                }
                if(borderline(l,points)){++skipped;continue;}
                const auto keep=fpm::detail::dedupKeep(points,int(k.topK),k.radius,k.angleWindow);
                const size_t bytes=l.slots.size()*sizeof(FGCandidate);
                auto slots=[device newBufferWithBytes:l.slots.data() length:bytes options:MTLResourceStorageModeShared];
                auto drop=[device newBufferWithLength:l.slots.size()*4 options:MTLResourceStorageModeShared];
                auto counts=[device newBufferWithLength:8 options:MTLResourceStorageModeShared];
                std::memset(drop.contents,0xff,drop.length);std::memset(counts.contents,0,counts.length);
                auto command=[queue commandBuffer];
                auto e=[command computeCommandEncoder];[e setComputePipelineState:mark];
                [e setBuffer:slots offset:0 atIndex:0];[e setBuffer:drop offset:0 atIndex:1];[e setBuffer:counts offset:0 atIndex:2];[e setBytes:&k length:sizeof(k) atIndex:3];
                [e dispatchThreads:MTLSizeMake(k.maps*k.limit,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
                e=[command computeCommandEncoder];[e setComputePipelineState:apply];
                [e setBuffer:slots offset:0 atIndex:0];[e setBuffer:drop offset:0 atIndex:1];[e setBytes:&k length:sizeof(k) atIndex:2];
                [e dispatchThreads:MTLSizeMake(k.maps*k.limit,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
                [command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"dedup command failed");
                const auto* out=static_cast<const FGCandidate*>(slots.contents);const auto* flags=static_cast<const uint32_t*>(drop.contents);
                const auto* totals=static_cast<const uint32_t*>(counts.contents);
                const size_t expectedKept=size_t(std::accumulate(keep.begin(),keep.end(),0));
                require(totals[0]==valid.size() && totals[1]==expectedKept,"dedup counters differ in trial "+std::to_string(trial));
                for(size_t r=0;r<valid.size();++r) {
                    const uint32_t slot=valid[r];
                    require(flags[slot]==uint32_t(!keep[r]) && out[slot].valid==keep[r],
                        "GPU dedup decision differs from dedupKeep in trial "+std::to_string(trial)+" rank "+std::to_string(r));
                }
                for(uint32_t i=0;i<k.first;++i)require(flags[i]==0xffffffffu,"dedup wrote outside its model range");
                for(uint32_t i=0;i<k.maps*k.limit;++i)if(!l.slots[k.first+i].valid)require(flags[k.first+i]==0 && !out[k.first+i].valid,"invalid slot changed");
                ++checked;
            }
            require(checked>=40,"too few non-borderline dedup layouts: "+std::to_string(checked)+" checked, "+std::to_string(skipped)+" skipped");
            std::cout<<"full GPU dedup kernels match dedupKeep on "<<checked<<" layouts ("<<skipped<<" borderline skipped)\n";
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
