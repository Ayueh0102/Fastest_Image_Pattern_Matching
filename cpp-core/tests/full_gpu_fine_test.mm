// SPDX-License-Identifier: BSD-2-Clause
#include "full_gpu_types.h"
#include "full_gpu_fine_types.h"
#include "full_gpu_metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
struct ProbeConfig {
    float anchorX,anchorY,angle,threshold;
    uint32_t mode,reserved[3];
};
static_assert(sizeof(ProbeConfig)==32 && sizeof(FGFineOptions)==8 && sizeof(FGFineCounters)==24);
// Independent copy of Stage 38's original moment loops. It intentionally does
// not call the production path selector or new moment helper. Thus mode 0's
// refactoring is checked as well as modes 1/2, down to each lane's integer
// sums. Mode 3 shares source moments and changes their stripe ownership, so
// it checks all aggregate moments and unchanged per-lane dot products.
const char* probeSource=R"METAL(
struct FGFineProbeConfig {
    float anchorX,anchorY,angle,threshold;
    uint mode; uint reserved[3];
};
inline void fg_fine_legacy_moments(device const uchar* source, device const uchar* templ,
    FGLevel level, float2 anchor, float c, float s, threadgroup uchar* roi,
    threadgroup ulong* partial, uint lane)
{
    const uint roiW = level.tw + 6, roiH = level.th + 6;
    const ulong roiArea = ulong(roiW) * roiH;
    const bool cached = roiArea <= FG_FINE_ROI_BYTES;
    const uint window = lane / 4, strip = lane % 4;
    if (cached) {
        for (uint p = lane; p < uint(roiArea); p += 256)
            roi[p] = fg_fine_pixel(source, level, anchor, c, s, p % roiW, p / roiW);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (window < FG_FINE_WINDOWS) {
        ulong sum = 0, square = 0, dot = 0;
        const uint wx = window % 7, wy = window / 7;
        const uint area = level.tw * level.th;
        for (uint p = strip; p < area; p += 4) {
            const uint x = p % level.tw, y = p / level.tw;
            const ulong pixel = cached ? roi[(wy + y) * roiW + wx + x] :
                fg_fine_pixel(source, level, anchor, c, s, wx + x, wy + y);
            const ulong t = templ[level.templateOffset + p];
            sum += pixel; square += pixel * pixel; dot += pixel * t;
        }
        partial[lane * 3] = sum;
        partial[lane * 3 + 1] = square;
        partial[lane * 3 + 2] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
kernel void fg_fine_moments_probe(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], constant FGLevel& level [[buffer(2)]],
    constant FGStats& stats [[buffer(3)]], constant FGFineProbeConfig& config [[buffer(4)]],
    device float* output [[buffer(5)]], device ulong* moments [[buffer(6)]],
    device uint* metadata [[buffer(7)]], uint lane [[thread_index_in_threadgroup]])
{
    threadgroup uchar roi[FG_FINE_ROI_CAPACITY];
    threadgroup ulong partial[196 * 3];
    threadgroup float maps[49];
    const float radians = config.angle * FG_FINE_D2R;
    const float c = cos(radians), s = sin(radians);
    const float2 anchor = float2(config.anchorX, config.anchorY);
    if (config.mode == 9)
        fg_fine_legacy_moments(source, templ, level, anchor, c, s, roi, partial, lane);
    else
        fg_fine_angle_moments(source, templ, level, anchor, c, s, roi, partial, lane,
            fg_fine_path(level, config.mode), config.mode);
    if (lane < 196) {
        for (uint j = 0; j < 3; ++j) moments[lane * 3 + j] = partial[lane * 3 + j];
    }
    if (lane < 49) {
        ulong sum = 0, square = 0, dot = 0;
        for (uint k = 0; k < 4; ++k) {
            const uint p = (lane * 4 + k) * 3;
            sum += partial[p]; square += partial[p + 1]; dot += partial[p + 2];
        }
        moments[588 + lane * 3] = sum;
        moments[588 + lane * 3 + 1] = square;
        moments[588 + lane * 3 + 2] = dot;
        maps[lane] = fg_ncc(sum, square, dot, stats);
        output[lane] = maps[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint peak = 0;
        for (uint p = 1; p < 49; ++p) if (maps[p] > maps[peak]) peak = p;
        metadata[0] = peak;
        metadata[1] = maps[peak] >= config.threshold;
        metadata[2] = fg_fine_path(level, config.mode == 9 ? 0 : config.mode);
        metadata[3] = config.mode != 9 && config.mode >= 2 && fg_fine_narrow_moments(level);
        metadata[4] = config.mode == 3 && metadata[2] == 0 && metadata[3];
        FGLevel boundary = level;
        boundary.tw = 264204; boundary.th = 1;
        metadata[5] = fg_fine_narrow_moments(boundary);
        boundary.tw = 264205;
        metadata[6] = fg_fine_narrow_moments(boundary);
        metadata[7] = config.mode == 4 && metadata[2] == 0 && metadata[3];
    }
}
)METAL";
struct Fixture {
    std::string name;
    uint32_t tw,th;
    float x=11,y=13,angle=0,threshold=.5f;
    int sourceConstant=-1,templateConstant=-1;
};
struct Result {
    std::vector<float> maps;
    std::vector<uint64_t> moments;
    std::vector<uint32_t> metadata;
};
struct GPU {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    uint32_t capacity;
    uint32_t pathChecks[3]={},narrowChecks=0,wideFallbackChecks=0,reuseChecks=0,simdChecks=0;
    explicit GPU(uint32_t bytes):capacity(bytes) {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
        queue=[device newCommandQueue];NSError* error=nil;
        MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        options.preprocessorMacros=@{@"FG_FINE_ROI_CAPACITY":@(capacity)};
        const std::string source=std::string(fpmFullGpuMetalSource)+probeSource;
        auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        auto function=[library newFunctionWithName:@"fg_fine_moments_probe"];
        pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
        require(pipeline.maxTotalThreadsPerThreadgroup>=256,"Unsupported fine workgroup");
    }
    id<MTLBuffer> buffer(size_t size,const void* input=nullptr) {
        auto b=[device newBufferWithLength:std::max<size_t>(size,256) options:MTLResourceStorageModeShared];
        require(b!=nil,"Metal allocation failed");std::memset(b.contents,0xa5,b.length);
        if(input&&size)std::memcpy(b.contents,input,size);return b;
    }
    Result run(id<MTLBuffer> source,id<MTLBuffer> templ,const FGLevel& level,const FGStats& stats,ProbeConfig config) {
        auto maps=buffer(53*sizeof(float)),moments=buffer((735+4)*sizeof(uint64_t)),metadata=buffer(12*sizeof(uint32_t));
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:source offset:0 atIndex:0];[encoder setBuffer:templ offset:0 atIndex:1];
        [encoder setBytes:&level length:sizeof(level) atIndex:2];[encoder setBytes:&stats length:sizeof(stats) atIndex:3];
        [encoder setBytes:&config length:sizeof(config) atIndex:4];
        [encoder setBuffer:maps offset:0 atIndex:5];[encoder setBuffer:moments offset:0 atIndex:6];
        [encoder setBuffer:metadata offset:0 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        for(auto pair:{std::make_pair(maps,49*sizeof(float)),std::make_pair(moments,735*sizeof(uint64_t)),
            std::make_pair(metadata,8*sizeof(uint32_t))}) {
            const auto* bytes=static_cast<const unsigned char*>(pair.first.contents);
            for(size_t i=pair.second;i<pair.first.length;++i)require(bytes[i]==0xa5,"Fine probe output guard overwritten");
        }
        Result result;
        auto* m=static_cast<float*>(maps.contents);result.maps.assign(m,m+49);
        auto* p=static_cast<uint64_t*>(moments.contents);result.moments.assign(p,p+735);
        auto* d=static_cast<uint32_t*>(metadata.contents);result.metadata.assign(d,d+8);
        return result;
    }
    void check(const Fixture& f,std::mt19937& random) {
        FGLevel level{std::max(400u,f.tw+40),std::max(280u,f.th+40),f.tw,f.th,13,19,0,0};
        std::vector<unsigned char> source(size_t(level.sw)*level.sh+level.sourceOffset+11,0x79);
        std::vector<unsigned char> templ(size_t(level.tw)*level.th+level.templateOffset+17,0x37);
        for(size_t i=0;i<size_t(level.sw)*level.sh;++i)
            source[level.sourceOffset+i]=f.sourceConstant<0?random()%256:unsigned(f.sourceConstant);
        FGStats stats{};stats.area=uint64_t(level.tw)*level.th;
        for(size_t i=0;i<stats.area;++i) {
            const uint64_t v=f.templateConstant<0?random()%256:unsigned(f.templateConstant);
            templ[level.templateOffset+i]=static_cast<unsigned char>(v);stats.sum+=v;stats.square+=v*v;
        }
        stats.variance=stats.area*stats.square-stats.sum*stats.sum;
        auto src=buffer(source.size(),source.data()),tpl=buffer(templ.size(),templ.data());
        const auto legacy=run(src,tpl,level,stats,{f.x,f.y,f.angle,f.threshold,9,{0,0,0}});
        const uint64_t roiArea=uint64_t(f.tw+6)*(f.th+6);
        for(uint32_t mode:{0u,1u,2u,3u,4u}) {
            const uint32_t effectiveMode=mode==4&&pipeline.threadExecutionWidth!=32?2:mode;
            const auto result=run(src,tpl,level,stats,{f.x,f.y,f.angle,f.threshold,effectiveMode,{0,0,0}});
            const auto label=f.name+"/cache"+std::to_string(capacity)+"/mode"+std::to_string(mode);
            require(std::memcmp(result.maps.data(),legacy.maps.data(),49*sizeof(float))==0,label+": 49 NCC score bytes differ");
            const bool reuse=mode==3&&roiArea<=capacity;
            if (!reuse) require(result.moments==legacy.moments,label+": per-lane or aggregated uint64 moments differ");
            else {
                require(std::equal(result.moments.begin()+588,result.moments.end(),legacy.moments.begin()+588),
                    label+": aggregated uint64 source/dot moments differ");
                for(size_t lane=0;lane<196;++lane)
                    require(result.moments[lane*3+2]==legacy.moments[lane*3+2],label+": per-lane uint64 dots differ");
            }
            require(result.metadata[0]==legacy.metadata[0]&&result.metadata[1]==legacy.metadata[1],label+": peak/threshold differs");
            const uint32_t path=roiArea<=capacity?0:(mode>=1&&f.tw+6<=capacity/7?2:1);
            require(result.metadata[2]==path,label+": wrong cache/direct/tiled path");++pathChecks[path];
            const bool narrow=mode>=2&&(stats.area+3)/4<=66051;
            require(result.metadata[3]==uint32_t(narrow),label+": wrong uint32/uint64 arithmetic path");
            require(result.metadata[4]==uint32_t(reuse&&narrow),label+": wrong overlapping-moment path");
            require(result.metadata[5]==1&&result.metadata[6]==0,label+": uint32 boundary guard differs");
            const bool simd=effectiveMode==4&&path==0&&narrow;
            require(result.metadata[7]==uint32_t(simd),label+": wrong SIMD moment path");
            narrowChecks+=narrow;wideFallbackChecks+=mode>=2&&!narrow;reuseChecks+=reuse&&narrow;simdChecks+=simd;
            if(f.templateConstant>=0) {
                require(std::all_of(result.maps.begin(),result.maps.end(),[](float v){return v==1;}),label+": constant template scores");
                require(result.metadata[0]==0&&result.metadata[1]==uint32_t(1.f>=f.threshold),label+": tie/threshold contract");
            }
        }
        // Angle zero and integer anchors give exact integer source coordinates,
        // permitting an independent CPU oracle with no trig/interpolation drift.
        if(f.angle==0&&std::floor(f.x)==f.x&&std::floor(f.y)==f.y) {
            std::vector<uint64_t> expected(735,0);
            for(uint32_t window=0;window<49;++window)for(uint32_t p=0;p<stats.area;++p) {
                const int x=int(f.x)-3+int(window%7+p%f.tw),y=int(f.y)-3+int(window/7+p/f.tw);
                const uint64_t v=x<0||y<0||x>=int(level.sw)||y>=int(level.sh)?0:
                    source[level.sourceOffset+size_t(y)*level.sw+size_t(x)];
                const uint64_t t=templ[level.templateOffset+p];
                const size_t lane=(window*4+(p%4))*3,aggregate=588+window*3;
                expected[lane]+=v;expected[lane+1]+=v*v;expected[lane+2]+=v*t;
                expected[aggregate]+=v;expected[aggregate+1]+=v*v;expected[aggregate+2]+=v*t;
            }
            require(legacy.moments==expected,f.name+": moments differ from scalar integer-coordinate oracle");
        }
    }
};
size_t tests(GPU& gpu) {
    std::mt19937 random(0x39f17e5);
    const uint32_t side=uint32_t(std::sqrt(gpu.capacity))-6,wide=gpu.capacity/7-6,row16=gpu.capacity/22-6;
    std::vector<Fixture> fixtures={
        {"one",1,1},{"small even",2,2},{"small odd",7,9},
        {"exact ROI capacity",side,side},{"one row over capacity",side,side+1},
        {"many-object scaled template",175,128},{"odd width and partial final tile",175,131},
        {"even width and partial final tile",176,130},
        {"wide one-row full cache",wide,1},{"wide one-row tile body",wide,2},
        {"one pixel too wide fallback",wide+1,2},
        {"sixteen row fit",row16,49},{"fifteen row fit",row16+1,49},
        {"rotated negative border",175,131,-2.25f,-.75f,23.1f},
        {"rotated far outside",175,131,10000,10000,-46.3f},
        {"half turn",175,131,190,190,180},
        {"negative rotation",175,131,40.125f,29.75f,-63.37f},
        {"constant template threshold tie",175,131,11,13,0,1,-1,103},
        {"constant template threshold above",175,131,11,13,0,std::nextafter(1.f,2.f),-1,103},
        {"zero source threshold tie",175,131,11,13,0,0,0,-1},
        {"constant both",175,131,11,13,0,1,255,255},
        {"integer zero border",175,131,-3,-5,0},
        {"cached white narrow moments",61,59,11,13,0,1,255,255},
        {"exact uint32 four-stripe bound",492,537,11,13,0,1,255,255},
        {"uint32 overflow fallback white",493,536,11,13,0,1,255,255},
        {"uint32 overflow fallback random",492,538,11,13,0},
    };
    for(uint32_t k=0;k<26;++k) {
        Fixture f{"random "+std::to_string(k),uint32_t(1+random()%201),uint32_t(1+random()%181)};
        if(k%3) {f.x=float(int(random()%500)-50)/4;f.y=float(int(random()%500)-50)/4;f.angle=float(int(random()%36000)-18000)/100;}
        fixtures.push_back(f);
    }
    for(const auto& f:fixtures)gpu.check(f,random);
    for(uint32_t path=0;path<3;++path)require(gpu.pathChecks[path]>0,"A cache/direct/tiled path was not exercised");
    require(gpu.narrowChecks&&gpu.wideFallbackChecks&&gpu.reuseChecks,"Narrow, overflow fallback or reuse path not exercised");
    require(gpu.pipeline.threadExecutionWidth!=32||gpu.simdChecks,"32-thread SIMD path not exercised");
    std::cout<<"cache="<<gpu.capacity<<" fixtures="<<fixtures.size()<<" path comparisons cached/direct/tiled="
        <<gpu.pathChecks[0]<<'/'<<gpu.pathChecks[1]<<'/'<<gpu.pathChecks[2]
        <<" narrow/wide fallback/reuse/SIMD="<<gpu.narrowChecks<<'/'<<gpu.wideFallbackChecks<<'/'<<gpu.reuseChecks<<'/'<<gpu.simdChecks
        <<" SIMD width="<<gpu.pipeline.threadExecutionWidth<<'\n';
    return fixtures.size();
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            bool validation=false;
            const char* setting=std::getenv("MTL_SHADER_VALIDATION");
            if(setting&&std::strcmp(setting,"1")==0)validation=true;
            if(argc==2&&std::strcmp(argv[1],"--validation-cache")==0)validation=true;
            else require(argc==1,"Usage: fpm_full_gpu_fine_test [--validation-cache]");
            size_t count=0;
            if(!validation) {GPU normal(16384);count+=tests(normal);}
            GPU reduced(4096);count+=tests(reduced);
            std::cout<<"Full GPU fine NCC: "<<count<<" frozen-legacy/original/tiled/narrow/reuse/SIMD fixtures passed; exact 49 maps, aggregate moments, unchanged dot stripes, overflow fallback, scalar oracle and threshold ties\n";
            return 0;
        } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
