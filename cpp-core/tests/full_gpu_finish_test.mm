// SPDX-License-Identifier: BSD-2-Clause
#include "full_gpu_types.h"
#include "full_gpu_metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
struct alignas(16) PoseConfig {
    uint32_t sw,sh,tw,th,channels,tiles,count,unused;
    float sourceMean[4],templateSum[4],templateEnergy,padding[3];
};
struct alignas(16) PoseState {
    float pose[4],lower[4],upper[4],packed[4],candidate[4],candidatePacked[4];
    float h[16],g[4],damping,score;
    uint32_t iterations,evaluations,status,active,pending,proposed;
};
static_assert(sizeof(PoseConfig)==80 && sizeof(PoseState)==208 && sizeof(FGResult)==64);
struct GPU {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString*,id<MTLComputePipelineState>>* pipelines;
    GPU() {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
        queue=[device newCommandQueue];NSError* error=nil;
        MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        library=[device newLibraryWithSource:[NSString stringWithUTF8String:fpmFullGpuMetalSource]
            options:options error:&error];
        if(!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        pipelines=[NSMutableDictionary new];
    }
    id<MTLBuffer> buffer(size_t bytes,const void* input=nullptr) {
        // Even a zero-count dispatch binds at least one complete ABI record;
        // Metal API validation checks argument size before the kernel guard.
        auto b=[device newBufferWithLength:std::max<size_t>(bytes,256) options:MTLResourceStorageModeShared];
        require(b!=nil,"Metal buffer allocation failed");std::memset(b.contents,0,b.length);
        if(input && bytes) std::memcpy(b.contents,input,bytes);return b;
    }
    template<class T> id<MTLBuffer> value(const T& input) {return buffer(sizeof(T),&input);}
    void run(const char* name,std::initializer_list<id<MTLBuffer>> buffers,uint32_t groups=1,uint32_t threads=1) {
        auto key=[NSString stringWithUTF8String:name];auto pipeline=pipelines[key];
        if(!pipeline) {
            NSError* error=nil;auto function=[library newFunctionWithName:key];
            require(function!=nil,"Missing finish kernel");
            pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            if(!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
            pipelines[key]=pipeline;
        }
        require(threads<=pipeline.maxTotalThreadsPerThreadgroup,"Unsupported finish threadgroup size");
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];uint32_t index=0;
        for(auto b:buffers) [encoder setBuffer:b offset:0 atIndex:index++];
        [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
};
template<class T> T* data(id<MTLBuffer> b) {return static_cast<T*>(b.contents);}
FGCandidate candidate(float x,float y,float angle,float score,float scale,uint32_t order) {
    return {x,y,angle,score,scale,score,1,order,0,0,0,0};
}
std::vector<FGCandidate> nms(GPU& gpu,const std::vector<FGCandidate>& input,bool global,
    float overlap=1,uint32_t width=20,uint32_t height=10,std::vector<FGRange> ranges={}) {
    auto in=gpu.buffer(input.size()*sizeof(FGCandidate),input.data());
    auto out=gpu.buffer(std::max<size_t>(1,input.size())*sizeof(FGCandidate));
    auto counts=gpu.buffer(sizeof(uint32_t));FGStatus initial{};initial.proposals=uint32_t(input.size());
    auto status=gpu.value(initial);
    FGNmsConfig config{0,uint32_t(input.size()),0,uint32_t(input.size()+1),width,height,uint32_t(global),
        uint32_t(ranges.size()),.5f,overlap,0,0};
    // The output holds input.size() records; selection can never exceed input count.
    auto cb=gpu.value(config);auto rb=gpu.buffer(ranges.size()*sizeof(FGRange),ranges.data());
    gpu.run(global?"fg_global_nms":"fg_model_nms",{in,out,counts,status,cb,rb},1,256);
    const auto s=*data<FGStatus>(status);require(s.overflow==0,"NMS geometry overflow");
    const uint32_t count=global?s.matches:*data<uint32_t>(counts);
    require(count<=input.size(),"NMS count exceeds input");
    return {data<FGCandidate>(out),data<FGCandidate>(out)+count};
}

void checkNms(GPU& gpu) {
    const std::vector<FGCandidate> duplicate{candidate(100,100,23,.9f,1,0),candidate(100,100,23,.8f,1,1)};
    require(nms(gpu,duplicate,false).size()==1,"Per-model overlap=1 must remove full duplicates");
    require(nms(gpu,duplicate,true).size()==2,"Global overlap=1 must retain equal geometry");
    require(nms(gpu,{candidate(100,100,31,.9f,2,0),candidate(100,100,31,.8f,1,1)},true,.5f).size()==1,
        "Global NMS must divide by the smaller cross-scale polygon area");
    // Two centered 40x10 rectangles at right angles overlap in a 10x10 square.
    const std::vector<FGCandidate> crossing{candidate(100,100,0,.9f,1,0),candidate(100,100,90,.8f,1,1)};
    require(nms(gpu,crossing,false,.3f,40,10).size()==2,"Rotated intersection overestimated");
    require(nms(gpu,crossing,false,.2f,40,10).size()==1,"Rotated intersection underestimated");
    std::vector<FGCandidate> ties{
        candidate(20,10,0,.8f,1,1),candidate(0,20,0,.8f,1,0),candidate(10,10,5,.8f,1,2),
        candidate(10,10,-5,.8f,1,7),candidate(10,10,-5,.8f,1,3),candidate(200,200,0,.9f,1,9)};
    auto ordered=nms(gpu,ties,true);const uint32_t expected[]={9,3,7,2,1,0};
    require(ordered.size()==6,"Tie fixture count changed");
    for(size_t i=0;i<ordered.size();++i) require(ordered[i].order==expected[i],"NMS stable score/y/x/angle/order mismatch");
    auto filtered=nms(gpu,{candidate(0,0,179,.9f,1,0),candidate(100,0,-179,.8f,1,1),
        candidate(200,0,0,.7f,1,2),candidate(300,0,179,.499f,1,3),
        candidate(400,0,179,std::numeric_limits<float>::quiet_NaN(),1,4)},true,1,20,10,{{178,180},{-180,-178}});
    require(filtered.size()==2 && filtered[0].order==0 && filtered[1].order==1,"NMS threshold/range/nonfinite filtering failed");
    require(nms(gpu,{},false).empty() && nms(gpu,{},true).empty(),"Empty NMS should produce zero output");
}

void checkCompactAndOutput(GPU& gpu) {
    std::vector<FGCandidate> perModel(8);
    perModel[0]=candidate(100,200,90,.9f,2,3);perModel[1]=candidate(20,30,-10,.8f,1,9);
    perModel[4]=candidate(50,60,0,.7f,.5f,1);
    const uint32_t counts[2]={2,1};auto input=gpu.buffer(perModel.size()*sizeof(FGCandidate),perModel.data());
    auto count=gpu.buffer(sizeof(counts),counts);auto compact=gpu.buffer(8*sizeof(FGCandidate));
    auto seeds=gpu.buffer(8*sizeof(PoseState)),states=gpu.buffer(8*sizeof(PoseState));
    PoseConfig pc{};pc.sourceMean[0]=.25f;pc.templateEnergy=5;auto config=gpu.value(pc);
    auto status=gpu.value(FGStatus{});FGFinishConfig finish{640,512,20,10,2,4,8,7,.75f,.2f,0,1};
    auto fc=gpu.value(finish);
    gpu.run("fg_compact_prepare",{input,count,compact,seeds,states,config,status,fc});
    require(data<FGStatus>(status)->proposals==3 && !data<FGStatus>(status)->overflow,"Proposal compaction count failed");
    require(data<PoseConfig>(config)->count==3 && data<PoseConfig>(config)->sourceMean[0]==.25f
        && data<PoseConfig>(config)->templateEnergy==5,"Compaction modified prepared pose statistics");
    require(data<FGCandidate>(compact)[2].x==50 && data<FGCandidate>(compact)[2].order==2,"Compaction model/stable order failed");
    const auto& s=data<PoseState>(seeds)[0];
    require(s.pose[0]==-220 && s.pose[1]==56 && s.pose[2]==90,"Public-to-resident pose conversion failed");
    for(uint32_t d=0;d<4;++d) require(s.lower[d]<=s.pose[d] && s.upper[d]>=s.pose[d],"Seed outside original bounds");
    auto result=gpu.buffer(8*sizeof(FGResult));auto kind=gpu.value(uint32_t(0));
    gpu.run("fg_write_results",{compact,result,status,fc,kind},8);
    const auto& r=data<FGResult>(result)[0];const float expected[]={90,220,90,180,110,180,110,220};
    require(r.x==100 && r.y==200 && r.scale==2 && r.score==.9f,"Final GPU record changed values");
    for(uint32_t i=0;i<8;++i) require(std::abs(r.corners[i]-expected[i])<1e-4,"GPU output corners mismatch");
    finish.capacity=2;*data<FGFinishConfig>(fc)=finish;
    gpu.run("fg_compact_prepare",{input,count,compact,seeds,states,config,status,fc});
    require(data<FGStatus>(status)->proposals==2 && (data<FGStatus>(status)->overflow&2),"Compaction must flag capacity loss");
    finish.capacity=8;*data<FGFinishConfig>(fc)=finish;*data<FGStatus>(status)={};
    data<FGCandidate>(input)[0].x=2000000.f;
    gpu.run("fg_compact_prepare",{input,count,compact,seeds,states,config,status,fc});
    require(data<FGStatus>(status)->proposals==2 && (data<FGStatus>(status)->overflow&8),
        "Out-of-domain seed must flag an error before solver dispatch");
}

void checkRescue(GPU& gpu) {
    constexpr uint32_t n=8;
    std::vector<FGCandidate> proposals;
    for(uint32_t i=0;i<n;++i) proposals.push_back(candidate(100+20*i,100,0,.7f,1,i));
    auto input=gpu.buffer(n*sizeof(FGCandidate),proposals.data()),counts=gpu.value(n);
    auto compact=gpu.buffer(n*sizeof(FGCandidate)),seeds=gpu.buffer(n*sizeof(PoseState)),baseline=gpu.buffer(n*sizeof(PoseState));
    auto config=gpu.value(PoseConfig{}),status=gpu.value(FGStatus{});
    FGFinishConfig finish{640,512,100,100,1,n,n,7,.75f,.2f,0,1};auto fc=gpu.value(finish);
    gpu.run("fg_compact_prepare",{input,counts,compact,seeds,baseline,config,status,fc});
    const float scores[]={.8f,.7f,.7f,.1f,.099f,.6f,.5f,std::numeric_limits<float>::quiet_NaN()};
    for(uint32_t i=0;i<n;++i) {data<PoseState>(baseline)[i].score=scores[i];data<PoseState>(baseline)[i].pose[0]+=1;}
    auto retries=gpu.buffer(64*sizeof(PoseState)),owners=gpu.buffer(64*sizeof(uint32_t));
    auto parents=gpu.buffer(4*sizeof(uint32_t)),baseScores=gpu.buffer(4*sizeof(float));
    auto retryConfig=gpu.value(PoseConfig{});
    auto prepare=[&] {gpu.run("fg_prepare_rescue",{seeds,baseline,retries,owners,parents,baseScores,retryConfig,status,fc,config});};
    prepare();require(data<FGStatus>(status)->rescueParents==4 && data<FGStatus>(status)->rescueStarts==64
        && data<PoseConfig>(retryConfig)->count==64,"Rescue top-four/16-start counts failed");
    const uint32_t expected[]={1,2,5,6};
    for(uint32_t i=0;i<4;++i) require(data<uint32_t>(parents)[i]==expected[i],"Rescue descending/stable score priority failed");
    for(uint32_t i=0;i<64;++i) {
        const auto& trial=data<PoseState>(retries)[i];const auto& seed=data<PoseState>(seeds)[data<uint32_t>(owners)[i]];
        for(uint32_t d=0;d<4;++d) {
            require(trial.lower[d]==seed.lower[d] && trial.upper[d]==seed.upper[d],"Retry lost original seed bounds");
            require(trial.pose[d]>=seed.lower[d] && trial.pose[d]<=seed.upper[d],"Retry escaped original bounds");
        }
    }
    const auto& first=data<PoseState>(retries)[0];const auto& origin=data<PoseState>(seeds)[1];
    require(first.pose[0]==origin.pose[0] && std::abs(first.pose[1]-origin.pose[1]+1)<1e-5
        && std::abs(first.pose[2]+3.6f)<1e-5,"Halton start must use the original seed, not baseline endpoint");
    // Fake solver endpoints isolate GPU selection and continuation decisions.
    for(uint32_t i=0;i<64;++i) data<PoseState>(retries)[i].score=-2;
    for(uint32_t i=0;i<4;++i) {
        auto& a=data<PoseState>(retries)[16*i];auto& b=data<PoseState>(retries)[16*i+1];
        a.score=b.score=scores[expected[i]]+.01f;a.pose[0]+=1;b.pose[0]+=2;
    }
    auto continuations=gpu.buffer(4*sizeof(PoseState)),continuationOwners=gpu.buffer(4*sizeof(uint32_t));
    auto continuationConfig=gpu.value(PoseConfig{});
    gpu.run("fg_select_continuations",{baseline,retries,owners,parents,baseScores,continuations,continuationOwners,continuationConfig,status,retryConfig});
    require(data<FGStatus>(status)->rescueContinuations==4 && data<PoseConfig>(continuationConfig)->count==4,"Improved parents must continue once");
    for(uint32_t i=0;i<4;++i) {
        require(data<uint32_t>(continuationOwners)[i]==expected[i],"Continuation ownership changed");
        require(data<PoseState>(continuations)[i].pose[0]==data<PoseState>(retries)[16*i].pose[0],"Equal-score retry replaced first winner");
        for(uint32_t d=0;d<4;++d) require(data<PoseState>(continuations)[i].lower[d]==data<PoseState>(seeds)[expected[i]].lower[d]
            && data<PoseState>(continuations)[i].upper[d]==data<PoseState>(seeds)[expected[i]].upper[d],"Continuation changed original bounds");
    }
    const float before=data<PoseState>(baseline)[expected[0]].score;
    data<PoseState>(continuations)[0].score+=.01f;
    data<PoseState>(continuations)[1].pose[0]+=5; // Equal score must not replace.
    data<PoseState>(continuations)[2].score=std::numeric_limits<float>::quiet_NaN();
    data<PoseState>(continuations)[3].score=-2;
    const float stableX=data<PoseState>(baseline)[expected[1]].pose[0];
    gpu.run("fg_accept_continuations",{baseline,continuations,continuationOwners,status});
    require(data<PoseState>(baseline)[expected[0]].score>before && data<PoseState>(baseline)[expected[1]].pose[0]==stableX,
        "Continuation must accept strictly improved finite scores only");
    // Policy boundaries and whole-parent work budgeting.
    finish.tw=finish.th=1400;*data<FGFinishConfig>(fc)=finish;prepare();
    require(data<FGStatus>(status)->rescueParents==1,"Rescue sample budget must admit only one complete parent");
    finish.tw=finish.th=2000;*data<FGFinishConfig>(fc)=finish;prepare();
    require(data<FGStatus>(status)->rescueStarts==0 && data<PoseConfig>(retryConfig)->count==0,"Oversized parent must not start partially");
    finish.tw=finish.th=100;finish.minScore=.1f;*data<FGFinishConfig>(fc)=finish;prepare();
    require(data<FGStatus>(status)->rescueStarts==0,"Upper trigger boundary must be exclusive");
    finish.minScore=.75f;*data<FGFinishConfig>(fc)=finish;
    for(uint32_t i=0;i<n;++i) data<PoseState>(baseline)[i].score=i==3?.1f:.9f;
    prepare();require(data<FGStatus>(status)->rescueParents==1 && data<uint32_t>(parents)[0]==3,"Lower trigger boundary must be inclusive");
    finish.optimizeMask=0;*data<FGFinishConfig>(fc)=finish;prepare();
    require(data<FGStatus>(status)->rescueStarts==0,"Unchanged Halton starts must be removed");
    finish.optimizeMask=7;finish.refine=0;*data<FGFinishConfig>(fc)=finish;prepare();
    require(data<FGStatus>(status)->rescueStarts==0,"Disabled refinement must not schedule rescue");
}
}
int main() {
    @autoreleasepool {
        try {
            GPU gpu;checkNms(gpu);checkCompactAndOutput(gpu);checkRescue(gpu);
            std::cout<<"Full GPU finish: NMS geometry/order/thresholds, compaction/overflow, GPU result corners, rescue budgets/original bounds/continuations passed\n";
            return 0;
        } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
