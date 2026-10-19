// SPDX-License-Identifier: BSD-2-Clause
// Independent residency check for the complete Metal pipeline. The Metal
// command-queue, command-buffer and buffer methods are replaced in this
// process, so a second command buffer, an additional host wait, a completion
// handler, or host access to buffer memory between GPU submissions is seen
// even when FullGpuMetrics is not updated. Compiled without ARC so that the
// replacement functions return the original objects unchanged.
#include "research_pipeline.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fpm;
using namespace fpm::research;
namespace {
void require(bool value,const std::string& message) { if(!value) throw std::runtime_error(message); }

enum class Kind { Create, Commit, Wait, ScheduledWait, Handler, Contents };
std::mutex eventLock;
std::vector<Kind> events;
bool recording=false;
thread_local int depth=0;
// Metal may call its own replaced methods internally; only the outermost call
// on a thread is a use by the caller.
struct Scope {
    bool outer;
    Scope():outer(depth++==0) {}
    ~Scope() { --depth; }
};
void record(Kind kind) {
    std::lock_guard<std::mutex> lock(eventLock);
    if(recording) events.push_back(kind);
}
void begin() { std::lock_guard<std::mutex> lock(eventLock); events.clear(); recording=true; }
std::vector<Kind> end() {
    std::lock_guard<std::mutex> lock(eventLock);
    recording=false; std::vector<Kind> trace; trace.swap(events); return trace;
}

IMP commandBufferImp,unretainedImp,descriptorImp,commitImp,waitImp,scheduledImp,completedHandlerImp,scheduledHandlerImp,contentsImp;
id hookCommandBuffer(id self,SEL selector) {
    Scope scope; id result=reinterpret_cast<id(*)(id,SEL)>(commandBufferImp)(self,selector);
    if(scope.outer) record(Kind::Create);
    return result;
}
id hookUnretained(id self,SEL selector) {
    Scope scope; id result=reinterpret_cast<id(*)(id,SEL)>(unretainedImp)(self,selector);
    if(scope.outer) record(Kind::Create);
    return result;
}
id hookDescriptor(id self,SEL selector,id descriptor) {
    Scope scope; id result=reinterpret_cast<id(*)(id,SEL,id)>(descriptorImp)(self,selector,descriptor);
    if(scope.outer) record(Kind::Create);
    return result;
}
void hookCommit(id self,SEL selector) {
    Scope scope; if(scope.outer) record(Kind::Commit);
    reinterpret_cast<void(*)(id,SEL)>(commitImp)(self,selector);
}
void hookWait(id self,SEL selector) {
    Scope scope; reinterpret_cast<void(*)(id,SEL)>(waitImp)(self,selector);
    if(scope.outer) record(Kind::Wait);
}
void hookScheduled(id self,SEL selector) {
    Scope scope; reinterpret_cast<void(*)(id,SEL)>(scheduledImp)(self,selector);
    if(scope.outer) record(Kind::ScheduledWait);
}
void hookCompletedHandler(id self,SEL selector,id handler) {
    Scope scope; if(scope.outer) record(Kind::Handler);
    reinterpret_cast<void(*)(id,SEL,id)>(completedHandlerImp)(self,selector,handler);
}
void hookScheduledHandler(id self,SEL selector,id handler) {
    Scope scope; if(scope.outer) record(Kind::Handler);
    reinterpret_cast<void(*)(id,SEL,id)>(scheduledHandlerImp)(self,selector,handler);
}
void* hookContents(id self,SEL selector) {
    Scope scope; if(scope.outer) record(Kind::Contents);
    return reinterpret_cast<void*(*)(id,SEL)>(contentsImp)(self,selector);
}
void replace(Class type,SEL selector,IMP replacement,IMP& original,std::set<Method>& replaced) {
    Method method=class_getInstanceMethod(type,selector);
    require(method!=nullptr,std::string(class_getName(type))+" lacks "+sel_getName(selector));
    require(replaced.insert(method).second,std::string("Metal method replaced twice: ")+sel_getName(selector));
    original=method_setImplementation(method,replacement);
}
void install(id<MTLDevice> device) {
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> buffer=[device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    require(queue!=nil && buffer!=nil,"Metal probe objects unavailable");
    // The pipeline creates its queue, command buffers and shared buffers in
    // the same way, so they resolve to these classes' method implementations.
    Class queueType=object_getClass(queue),commandType=object_getClass([queue commandBuffer]),bufferType=object_getClass(buffer);
    std::set<Method> replaced;
    replace(queueType,@selector(commandBuffer),IMP(hookCommandBuffer),commandBufferImp,replaced);
    replace(queueType,@selector(commandBufferWithUnretainedReferences),IMP(hookUnretained),unretainedImp,replaced);
    replace(queueType,@selector(commandBufferWithDescriptor:),IMP(hookDescriptor),descriptorImp,replaced);
    replace(commandType,@selector(commit),IMP(hookCommit),commitImp,replaced);
    replace(commandType,@selector(waitUntilCompleted),IMP(hookWait),waitImp,replaced);
    replace(commandType,@selector(waitUntilScheduled),IMP(hookScheduled),scheduledImp,replaced);
    replace(commandType,@selector(addCompletedHandler:),IMP(hookCompletedHandler),completedHandlerImp,replaced);
    replace(commandType,@selector(addScheduledHandler:),IMP(hookScheduledHandler),scheduledHandlerImp,replaced);
    replace(bufferType,@selector(contents),IMP(hookContents),contentsImp,replaced);
    [buffer release];[queue release];
}

struct Observed {
    uint64_t creates=0,commits=0,waits=0,scheduledWaits=0,handlers=0;
    uint64_t staging=0,encoding=0,inFlight=0,intermediateReads=0,decisionPoints=0,finalReads=0;
};
// Same definitions as FullGpuMetrics, from the intercepted calls alone.
Observed observe(const std::vector<Kind>& trace) {
    Observed o;uint64_t pending=0,afterWait=0;
    for(const Kind kind:trace) switch(kind) {
    case Kind::Create: ++o.creates; break;
    case Kind::Commit:
        if(afterWait) { o.intermediateReads+=afterWait; ++o.decisionPoints; afterWait=0; }
        ++o.commits; ++pending; break;
    case Kind::Wait: ++o.waits; if(pending) --pending; break;
    case Kind::ScheduledWait: ++o.scheduledWaits; break;
    case Kind::Handler: ++o.handlers; break;
    case Kind::Contents:
        if(pending) { ++o.inFlight; ++o.intermediateReads; }
        else if(o.waits) ++afterWait;
        else if(o.creates) ++o.encoding;
        else ++o.staging;
        break;
    }
    o.finalReads=afterWait;
    return o;
}
std::string describe(const Observed& o) {
    return " (command buffers "+std::to_string(o.creates)+", commits "+std::to_string(o.commits)+", waits "+
        std::to_string(o.waits)+", scheduled waits "+std::to_string(o.scheduledWaits)+", handlers "+std::to_string(o.handlers)+
        ", host views staging/encoding/in-flight/intermediate/final "+std::to_string(o.staging)+"/"+std::to_string(o.encoding)+"/"+
        std::to_string(o.inFlight)+"/"+std::to_string(o.intermediateReads)+"/"+std::to_string(o.finalReads)+")";
}
void check(const Observed& o,const FullGpuMetrics& m,const std::string& label) {
    require(o.creates==1 && o.commits==1 && o.waits==1,label+": expected one command buffer, one commit and one host wait"+describe(o));
    require(o.scheduledWaits==0 && o.handlers==0,label+": scheduled wait or completion handler"+describe(o));
    require(o.encoding==0 && o.inFlight==0 && o.intermediateReads==0 && o.decisionPoints==0,
        label+": host buffer view after command creation and before the final wait"+describe(o));
    require(o.staging>=2 && o.finalReads>=1,label+": interception missed the uploads or final readback"+describe(o));
    require(m.enabled && m.commandBuffers==o.commits && m.cpuWaits==o.waits+o.scheduledWaits &&
        m.intermediateReadbacks==o.intermediateReads && m.cpuAlgorithmSteps==o.decisionPoints,
        label+": reported residency counters differ from intercepted Metal calls"+describe(o));
}
// The replaced methods must observe direct API use, and observe() must flag a
// two-submission sequence with a host read between the submissions.
void selfTest(id<MTLDevice> device) {
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> buffer=[device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    require(queue!=nil && buffer!=nil,"Metal self-test objects unavailable");
    begin();
    std::memset(buffer.contents,0,16);
    id<MTLCommandBuffer> first=[queue commandBuffer];[first commit];[first waitUntilCompleted];
    const volatile unsigned char seen=static_cast<const unsigned char*>(buffer.contents)[0];(void)seen;
    id<MTLCommandBuffer> second=[queue commandBufferWithUnretainedReferences];[second commit];[second waitUntilCompleted];
    (void)buffer.contents;
    auto o=observe(end());
    require(o.creates==2 && o.commits==2 && o.waits==2 && o.staging==1 && o.intermediateReads==1 &&
        o.decisionPoints==1 && o.finalReads==1,"interception self-test missed a two-submission sequence"+describe(o));
    begin();
    id<MTLCommandBuffer> third=[queue commandBufferWithDescriptor:[[MTLCommandBufferDescriptor new] autorelease]];
    [third addCompletedHandler:^(id<MTLCommandBuffer>){}];[third commit];[third waitUntilScheduled];[third waitUntilCompleted];
    o=observe(end());
    require(o.creates==1 && o.handlers==1 && o.scheduledWaits==1 && o.waits==1,"interception self-test missed a handler or scheduled wait"+describe(o));
    [buffer release];[queue release];
}
Image scene(ImageView templ,int width,int height,int left,int top,bool empty=false) {
    Image source(width,height);
    for(int y=0;y<height;++y) std::fill_n(source.row(y),width,uint8_t(23));
    if(!empty) for(int y=0;y<templ.height;++y)
        std::copy_n(templ.data+size_t(y)*templ.stride,templ.width,source.row(top+y)+left);
    return source;
}
}
int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"Metal device unavailable");
            install(device);selfTest(device);
            Image templ(32,24);std::mt19937 rng(370019);
            for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) templ.row(y)[x]=uint8_t(rng());
            const auto source=scene(templ.view(),192,128,50,40);
            Settings settings;settings.fullGpu=true;settings.backend=1;settings.refinementBackend=1;
            settings.minReducedArea=256;settings.scales={1.};settings.poseInputMode=1;settings.poseStorageMode=1;
            begin();
            Session session(source.view(),templ.view(),settings);
            const auto construction=observe(end());
            require(construction.creates==0 && construction.commits==0 && construction.waits==0,
                "complete GPU construction submitted work"+describe(construction));
            Options base;base.nccMode=2;base.fineNccMode=2;base.fineFused=true;base.fineAngleReuse=true;base.proposalScore=.6;
            base.match.minScore=.9;base.match.maxTargets=4;base.match.maxOverlap=.2;base.match.toleranceAngle=0;base.match.stopLayer=0;
            size_t calls=0;
            auto run=[&](Session& target,const Options& options,const std::string& label,size_t objects) {
                begin();Result result;
                try { result=target.run(options); } catch(...) { end(); throw; }
                check(observe(end()),result.fullGpu,label);++calls;
                require(result.matches.size()==objects,label+": unexpected match count");
                return result;
            };
            for(bool refine:{false,true}) for(int peak:{0,1}) for(int fine:{0,1,2,3,4}) for(bool profile:{false,true}) {
                auto options=base;options.refine=refine;options.fullGpuPeakMode=peak;options.fullGpuFineMode=fine;options.fullGpuProfiling=profile;
                run(session,options,"refine"+std::to_string(refine)+"/peak"+std::to_string(peak)+"/fine"+std::to_string(fine)+
                    "/profile"+std::to_string(profile),1);
            }
            auto tiled=base;tiled.refine=true;tiled.fullGpuPeakMode=1;tiled.fullGpuFineMode=4;
            const auto moved=scene(templ.view(),224,160,100,80);
            begin();session.updateSource(moved.view());
            const auto update=observe(end());
            require(update.creates==0 && update.commits==0 && update.waits==0 && update.staging==0,
                "source update used command buffers or host buffer views"+describe(update));
            run(session,tiled,"updated source",1);
            session.updateSource(scene(templ.view(),192,128,0,0,true).view());
            run(session,tiled,"empty source",0);
            auto bank=settings;bank.scales={.9,1.,1.1};
            Session continuous(source.view(),templ.view(),bank);
            for(int fine:{0,4}) for(bool profile:{false,true}) {
                auto options=tiled;options.match.subpixel=true;options.fullGpuFineMode=fine;options.fullGpuProfiling=profile;
                run(continuous,options,"scale bank/fine"+std::to_string(fine)+"/profile"+std::to_string(profile),1);
            }
            [device release];
            std::cout<<"Full Metal residency: interception self-test plus "<<calls<<" pipeline calls with one command buffer, one commit, "
                "one host wait, no handlers and no host buffer view between submission and final wait; reported counters match\n";
            return 0;
        } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
    }
}
