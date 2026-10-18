// SPDX-License-Identifier: BSD-2-Clause
#include "research_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fpm;
using namespace fpm::research;
namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void invalid(F&& operation,const char* message) {
    bool rejected=false;
    try { operation(); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,message);
}
Image scene(ImageView templ,int width,int height,int left,int top,bool empty=false) {
    Image source(width,height);
    for(int y=0;y<height;++y) std::fill_n(source.row(y),width,uint8_t(23));
    if(!empty) for(int y=0;y<templ.height;++y)
        std::copy_n(templ.data+size_t(y)*templ.stride,templ.width,source.row(top+y)+left);
    return source;
}
void resident(const Result& result) {
    const auto& metrics=result.fullGpu;
    require(metrics.enabled,"full GPU route not enabled");
    require(metrics.commandBuffers==1 && metrics.cpuWaits==1,"full GPU path submitted or waited more than once");
    require(metrics.intermediateReadbacks==0 && metrics.cpuAlgorithmSteps==0,"intermediate CPU algorithm/readback occurred");
    require(metrics.gpuBytes>0 && metrics.uploadBytes>0 && metrics.readbackBytes>0,"resident allocation/transfer accounting missing");
    require(std::isfinite(result.totalMs) && result.totalMs>0,"invalid pipeline elapsed time");
}
void one(const Result& result,double x,double y,double angle=0,double scale=1.,double score=.85) {
    resident(result);
    std::cerr<<"truth "<<x<<","<<y<<","<<angle<<","<<scale<<" proposals="<<result.proposals.size()<<" matches="<<result.matches.size()<<'\n';
    for(size_t i=0;i<result.matches.size();++i) std::cerr<<"  got "<<result.matches[i].center.x<<","<<result.matches[i].center.y<<","<<result.matches[i].angle<<","<<result.scales[i]<<" score="<<result.matches[i].score<<'\n';
    require(result.matches.size()==1 && result.scales.size()==1,"independent truth expects exactly one object");
    const auto& match=result.matches[0];
    require(std::hypot(match.center.x-x,match.center.y-y)<=2.,"ground-truth position error exceeds 2 pixels");
    require(std::abs(std::remainder(match.angle-angle,360.))<=1.,"ground-truth angle error exceeds 1 degree");
    require(std::abs(result.scales[0]/scale-1.)<=.02,"ground-truth scale error exceeds 2 percent");
    require(std::isfinite(match.score) && match.score>=score && match.score<=1.00001,"invalid or insufficient final NCC");
}
void same(const Result& first,const Result& second) {
    require(first.matches.size()==second.matches.size() && first.proposals.size()==second.proposals.size(),"repeated result count differs");
    require(first.scales==second.scales && first.proposalScales==second.proposalScales,"repeated scales differ");
    auto compare=[](const auto& left,const auto& right) {
        for(size_t i=0;i<left.size();++i) {
            require(left[i].center.x==right[i].center.x && left[i].center.y==right[i].center.y &&
                left[i].angle==right[i].angle && left[i].score==right[i].score,"repeated ordered pose/score differs");
            for(size_t corner=0;corner<4;++corner)
                require(left[i].corners[corner].x==right[i].corners[corner].x &&
                    left[i].corners[corner].y==right[i].corners[corner].y,"repeated corners differ");
        }
    };
    compare(first.matches,second.matches);compare(first.proposals,second.proposals);
    resident(second);
}
}
int main(int argc,char** argv) {
    try {
        bool tiled=false,profile=false;
        for(int i=1;i<argc;++i) {
            if(std::string(argv[i])=="--tiled") tiled=true;
            else if(std::string(argv[i])=="--profile") profile=true;
            else throw std::invalid_argument("Unknown full GPU test argument");
        }
        Image templ(32,24);std::mt19937 rng(370019);
        for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) templ.row(y)[x]=uint8_t(rng());
        auto source=scene(templ.view(),192,128,50,40);
        const size_t stride=size_t(source.width())+13;
        std::vector<uint8_t> padded(stride*source.height(),197);
        for(int y=0;y<source.height();++y) std::copy_n(source.row(y),source.width(),padded.data()+size_t(y)*stride);
        Settings settings;settings.fullGpu=true;settings.backend=1;settings.refinementBackend=1;
        settings.minReducedArea=256;settings.scales={1.};settings.poseInputMode=1;settings.poseStorageMode=1;
        Session session({padded.data(),source.width(),source.height(),stride},templ.view(),settings);
        Options options;options.refine=false;options.nccMode=2;options.fineNccMode=2;
        options.fullGpuPeakMode=tiled?1:0;options.fullGpuProfiling=profile;
        options.fineFused=true;options.fineAngleReuse=true;
        options.match.minScore=.9;options.match.maxTargets=4;options.match.maxOverlap=.2;
        options.match.toleranceAngle=0;options.match.stopLayer=0;
        const auto first=session.run(options);one(first,66,52);
        require(first.fullGpu.peakMode==options.fullGpuPeakMode,"Peak strategy reporting mismatch");
        require(first.fullGpu.profileRequested==profile,"Profiling request reporting mismatch");
        if(profile && first.fullGpu.profileAvailable) {
            double sum=0;uint64_t encoders=0;
            require(!first.fullGpu.stages.empty(),"Available GPU profile lacks stages");
            for(const auto& stage:first.fullGpu.stages) {
                require(!stage.name.empty() && stage.encoders>0 && std::isfinite(stage.gpuMs) && stage.gpuMs>=0,
                    "Invalid GPU stage timing");
                sum+=stage.gpuMs;encoders+=stage.encoders;
            }
            require(std::abs(sum-first.fullGpu.profileGpuMs)<1e-6 && first.fullGpu.profileSamples==2*encoders,
                "GPU profile sample accounting mismatch");
        } else require(first.fullGpu.stages.empty(),"Unavailable GPU profile must not report stages");
        auto control=options;control.fullGpuProfiling=false;control.fullGpuPeakMode=0;
        same(first,session.run(control));
        same(first,session.run(options));
        require(first.sourceGeneration==0,"initial source generation differs");
        std::fill(padded.begin(),padded.end(),0);templ.row(0)[0]^=255;
        same(first,session.run(options));templ.row(0)[0]^=255;

        const auto empty=scene(templ.view(),192,128,0,0,true);
        auto updated=session.updateSource(empty.view());
        require(updated.generation==1,"empty update did not advance generation");
        const auto nothing=session.run(options);resident(nothing);
        require(nothing.matches.empty() && nothing.proposals.empty(),"constant image produced a textured-template match");
        auto moved=scene(templ.view(),224,160,100,80);
        updated=session.updateSource(moved.view());
        require(updated.generation==2 && updated.dimensionsChanged,"dimension-changing update accounting wrong");
        moved.row(80)[100]^=255;
        const auto after=session.run(options);one(after,116,92);
        require(after.sourceGeneration==2,"source generation missing from output");
        for(const ImageView bad:std::vector<ImageView>{{nullptr,2,2,2},{source.view().data,2,2,1},{source.view().data,0,2,2}}) {
            invalid([&]{session.updateSource(bad);},"invalid source update accepted");
            const auto kept=session.run(options);same(after,kept);
            require(kept.sourceGeneration==2,"invalid update changed source generation");
        }

        // Exact quarter turn rendered independently by pixel permutation.
        Image quarterTemplate(23,17);
        for(int y=0;y<17;++y) for(int x=0;x<23;++x) quarterTemplate.row(y)[x]=uint8_t(rng());
        auto quarter=scene(quarterTemplate.view(),128,96,0,0,true);
        for(int y=0;y<17;++y) for(int x=0;x<23;++x) quarter.row(30+22-x)[40+y]=quarterTemplate.row(y)[x];
        auto rotationSettings=settings;rotationSettings.minReducedArea=100000;
        Session rotation(quarter.view(),quarterTemplate.view(),rotationSettings);
        auto turn=options;turn.match.angleRanges={{90.,90.}};
        one(rotation.run(turn),48.5,40.5,90.);
        turn.match.angleRanges={{-90.,-90.}};
        const auto excluded=rotation.run(turn);resident(excluded);
        require(excluded.matches.empty(),"angle restriction admitted an excluded quarter turn");

        // Physical image edge, plus the original-resolution/no-pyramid route.
        auto edge=scene(templ.view(),192,128,0,0);
        auto flat=settings;flat.minReducedArea=100000;
        Session boundary(edge.view(),templ.view(),flat);
        one(boundary.run(options),16,12);
        auto stop=options;stop.match.stopLayer=1;stop.match.minScore=.7;stop.match.maxTargets=1;
        one(session.run(stop),116,92,0,1,.7);
        // Narrow intervals make the subpixel angle list shrink from two to one
        // when its resolution halves; all SIMD groups must agree on the branch.
        Session shrinking(source.view(),templ.view(),flat);
        auto shrink=options;shrink.match.subpixel=true;shrink.match.angleRanges={{0.,0.},{3.,3.}};
        const auto shrinkingResult=shrinking.run(shrink);one(shrinkingResult,66,52);
        for(int repeat=0;repeat<5;++repeat) same(shrinkingResult,shrinking.run(shrink));
        auto noisy=source;noisy.row(43)[54]^=127;
        Session threshold(noisy.view(),templ.view(),flat);
        auto cutoff=options;cutoff.match.minScore=.8;
        const auto thresholdResult=threshold.run(cutoff);one(thresholdResult,66,52,0,1,.8);
        require(thresholdResult.matches[0].score<1.,"Threshold fixture needs an imperfect match");
        cutoff.match.minScore=std::nextafter(thresholdResult.matches[0].score,1.);
        const auto belowCutoff=threshold.run(cutoff);resident(belowCutoff);
        require(belowCutoff.matches.empty(),"FP32 rounding admitted a score below the double cutoff");
        auto inverted=source;
        for(int y=0;y<inverted.height();++y)for(int x=0;x<inverted.width();++x)inverted.row(y)[x]=255-inverted.row(y)[x];
        Session inversion(inverted.view(),templ.view(),settings);
        auto invert=options;invert.match.invertSource=true;one(inversion.run(invert),66,52);

        // The legacy constant-template contract assigns NCC=1, rather than
        // dividing by zero. Its many equivalent positions are not pose truth.
        Image constantTemplate(8,8);
        for(int y=0;y<8;++y) std::fill_n(constantTemplate.row(y),8,uint8_t(73));
        auto constantImage=scene(constantTemplate.view(),48,40,0,0,true);
        Session constant(constantImage.view(),constantTemplate.view(),flat);
        auto constantOptions=options;constantOptions.match.minScore=.99;
        const auto ambiguous=constant.run(constantOptions);resident(ambiguous);
        require(!ambiguous.matches.empty() && ambiguous.matches.size()<=size_t(constantOptions.match.maxTargets),
            "constant-template NCC contract lost all candidates or exceeded the limit");
        for(const auto& match:ambiguous.matches)
            require(std::isfinite(match.score) && std::abs(match.score-1.)<1e-6,"constant-template NCC is not finite one");
        same(ambiguous,constant.run(constantOptions));

        auto bank=settings;bank.scales={.9,1.,1.1};
        Session continuous(source.view(),templ.view(),bank);
        auto refine=options;refine.refine=true;refine.proposalScore=.6;refine.match.subpixel=true;
        const auto refined=continuous.run(refine);one(refined,66,52);
        same(refined,continuous.run(refine));
        // More than 16K pixels exercises multiple fused-solver tiles without
        // relying on the older host wrapper's small-template dispatch limit.
        Image largeTemplate(192,96);
        for(int y=0;y<largeTemplate.height();++y)for(int x=0;x<largeTemplate.width();++x)largeTemplate.row(y)[x]=uint8_t(rng());
        auto largeSource=scene(largeTemplate.view(),224,128,16,16);
        Session large(largeSource.view(),largeTemplate.view(),flat);
        one(large.run(refine),112,64);
        Image mediumTemplate(96,64);
        for(int y=0;y<mediumTemplate.height();++y)for(int x=0;x<mediumTemplate.width();++x)mediumTemplate.row(y)[x]=uint8_t(rng());
        auto oddSource=scene(mediumTemplate.view(),129,97,16,16);
        Session cacheBoundary(oddSource.view(),mediumTemplate.view(),flat);
        // Cached at 16 KiB, direct-sampled at 4 KiB under Shader Validation.
        one(cacheBoundary.run(options),64,48);

        auto bad=settings;bad.backend=0;
        invalid([&]{Session rejected(source.view(),templ.view(),bad);},"full GPU accepted CPU proposal backend");
        bad=settings;bad.refinementBackend=0;
        invalid([&]{Session rejected(source.view(),templ.view(),bad);},"full GPU accepted CPU refinement backend");
        bad=settings;bad.sparseOrientation=true;
        invalid([&]{Session rejected(source.view(),templ.view(),bad);},"full GPU accepted unsupported sparse route");
        bad=settings;bad.scales={0.};
        invalid([&]{Session rejected(source.view(),templ.view(),bad);},"full GPU accepted zero scale");
        bad=settings;bad.scales={std::exp(3.1)};
        Session excessiveScale(source.view(),templ.view(),bad);
        invalid([&]{excessiveScale.run(refine);},"full GPU accepted a seed scale outside solver domain");
        auto unsupported=refine;unsupported.scaleSupportRefine=true;
        invalid([&]{continuous.run(unsupported);},"full GPU silently accepted CPU scale-support correction");
        unsupported=options;unsupported.overlap=true;
        invalid([&]{session.run(unsupported);},"full GPU accepted unsupported CPU/GPU overlap control");
        unsupported=options;unsupported.fineProfiling=true;
        invalid([&]{session.run(unsupported);},"full GPU accepted CPU fine-request profiling");
        unsupported=options;unsupported.match.maxTargets=0;
        invalid([&]{session.run(unsupported);},"full GPU accepted zero target limit");
        unsupported=options;unsupported.match.angleRanges={{30.,20.}};
        invalid([&]{session.run(unsupported);},"full GPU accepted inverted angle range");
        unsupported=options;unsupported.fullGpuPeakMode=2;
        invalid([&]{session.run(unsupported);},"full GPU accepted unknown peak mode");
        auto hybridSettings=settings;hybridSettings.fullGpu=false;
        Session hybrid(source.view(),templ.view(),hybridSettings);
        unsupported=options;unsupported.fullGpuPeakMode=1;
        invalid([&]{hybrid.run(unsupported);},"hybrid silently accepted full GPU peak mode");
        unsupported=options;unsupported.fullGpuPeakMode=0;unsupported.fullGpuProfiling=true;
        invalid([&]{hybrid.run(unsupported);},"hybrid silently accepted full GPU profiling");
        same(after,session.run(options));
        std::cout<<"Full Metal pipeline: independent upright/quarter-turn/edge truth, multiscale refinement, empty image, snapshots, updates, residency and invalid options passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
