// SPDX-License-Identifier: BSD-2-Clause
#include "research_pipeline.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fpm;
using namespace fpm::research;
namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void sameMatches(const std::vector<Match>& a,const std::vector<Match>& b) {
    require(a.size()==b.size(),"different output count");
    for(size_t i=0;i<a.size();++i) {
        require(a[i].center.x==b[i].center.x && a[i].center.y==b[i].center.y &&
            a[i].angle==b[i].angle && a[i].score==b[i].score,"different ordered pose or score");
        for(size_t k=0;k<4;++k) require(a[i].corners[k].x==b[i].corners[k].x &&
            a[i].corners[k].y==b[i].corners[k].y,"different corners");
    }
}
void same(const Result& a,const Result& b) {
    sameMatches(a.matches,b.matches);sameMatches(a.proposals,b.proposals);
    require(a.scales==b.scales && a.proposalScales==b.proposalScales,"different scale");
}
Image scene(ImageView templ,int w,int h,int x,int y,bool empty=false) {
    Image image(w,h);
    for(int iy=0;iy<h;++iy) for(int ix=0;ix<w;++ix) image.row(iy)[ix]=23;
    if(!empty) for(int iy=0;iy<templ.height;++iy) for(int ix=0;ix<templ.width;++ix)
        image.row(y+iy)[x+ix]=templ.data[size_t(iy)*templ.stride+ix];
    return image;
}
void validateBadSettings(ImageView source,ImageView templ,const Settings& base) {
    for(int backend:{-2,2}) {
        auto invalid=base;invalid.refinementBackend=backend;bool caught=false;
        try { Session s(source,templ,invalid); } catch(const std::invalid_argument&) { caught=true; }
        require(caught,"invalid refinement backend accepted");
    }
    for(bool storage:{false,true}) {
        auto invalid=base;invalid.refinementBackend=0;
        if(storage) invalid.poseStorageMode=1;else invalid.residentMode=1;
        bool caught=false;try { Session s(source,templ,invalid); } catch(const std::invalid_argument&) { caught=true; }
        require(caught,"CPU refinement accepted Metal-only settings");
    }
}
}
int main(int argc,char**) {
    try {
        const bool metal=argc>1;Image templ(32,24);std::mt19937 rng(8031);
        for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) templ.row(y)[x]=uint8_t(rng());
        auto initial=scene(templ.view(),192,128,39,35);
        Settings settings;settings.backend=metal?1:0;settings.poseInputMode=1;
        settings.poseStorageMode=metal?1:0;settings.minReducedArea=10000;settings.scales={.9,1.,1.1};
        Options options;options.match.minScore=.85;options.match.maxTargets=3;
        options.match.toleranceAngle=0;options.match.stopLayer=0;options.proposalScore=.6;
        options.fineNccMode=metal?2:0;options.fineAngleReuse=metal;options.fineFused=metal;
        Session reuse(initial.view(),templ.view(),settings);
        const auto original=reuse.run(options);if(original.matches.size()!=1) throw std::runtime_error("initial target count="+std::to_string(original.matches.size())+" proposals="+std::to_string(original.proposals.size()));
        require(original.sourceGeneration==0 && original.configuredRefinementBackend==(metal?1:0),"initial route/generation wrong");
        auto explicitSettings=settings;explicitSettings.refinementBackend=settings.backend;
        Session explicitSession(initial.view(),templ.view(),explicitSettings);same(original,explicitSession.run(options));
        const std::array<std::array<int,4>,6> frames{{{{192,128,89,68}},{{256,192,170,120}},{{128,96,20,47}},
            {{192,128,39,35}},{{224,160,144,70}},{{192,128,39,35}}}};
        uint64_t generation=0;int previousW=192,previousH=128;
        for(size_t i=0;i<frames.size();++i) {
            const auto spec=frames[i];auto input=scene(templ.view(),spec[0],spec[1],spec[2],spec[3],i==4);
            const auto update=reuse.updateSource(input.view());++generation;
            require(update.generation==generation && update.retainedModels==3 &&
                update.copiedBytes==uint64_t(spec[0])*spec[1] && update.updateMs>=0,"update accounting wrong");
            require(update.dimensionsChanged==(spec[0]!=previousW || spec[1]!=previousH),"dimension invalidation wrong");
            require(update.poseInvalidated==(i!=5),"source-bound pose was not invalidated exactly");
            Session fresh(input.view(),templ.view(),settings);const auto expected=fresh.run(options);
            // Caller memory can change immediately after updateSource returns.
            input.row(0)[0]^=255;
            const auto actual=reuse.run(options);same(expected,actual);
            require(actual.sourceGeneration==generation,"source generation not surfaced");
            require(actual.poseCreateMs>0 || actual.refinementCandidates==0,"updated source reused stale pose images");
            const auto repeated=reuse.run(options);same(actual,repeated);require(repeated.poseCreateMs==0,"repeat rebuilt pose context");
            if(i==4) require(actual.matches.empty() && actual.refinementCandidates==0 &&
                actual.poseStorage.host_image_bytes==0,"empty frame retained stale pose image");
            else require(actual.matches.size()==1 && std::hypot(actual.matches[0].center.x-(spec[2]+16),
                actual.matches[0].center.y-(spec[3]+12))<.1,"updated target incorrect");
            previousW=spec[0];previousH=spec[1];
        }
        const auto beforeInvalid=reuse.run(options);
        for(const auto bad:std::array<ImageView,3>{{{nullptr,2,2,2},{initial.view().data,2,2,1},{initial.view().data,0,2,2}}}) {
            bool caught=false;try { reuse.updateSource(bad); } catch(const std::invalid_argument&) {caught=true;}
            require(caught,"invalid update accepted");const auto after=reuse.run(options);same(beforeInvalid,after);
            require(after.sourceGeneration==generation && after.poseCreateMs==0,"invalid update mutated usable session");
        }
        require(!original.scaleSupportEnabled && original.scaleSupportEvaluations==0 && original.scaleSupportMs==0,
            "default path ran experimental scale correction");
        Options polish=options;polish.scaleSupportRefine=true;
        const auto polished=reuse.run(polish);
        require(polished.scaleSupportEnabled && polished.scaleSupportEligible==polished.refinementCandidates &&
            polished.scaleSupportEvaluations>=polished.scaleSupportEligible && polished.scaleSupportMs>=0 &&
            polished.refineMs>=polished.scaleSupportMs && polished.matches.size()==1,"scale support integration missing");
        require(polished.matches[0].center.x==beforeInvalid.matches[0].center.x &&
            polished.matches[0].center.y==beforeInvalid.matches[0].center.y &&
            polished.matches[0].angle==beforeInvalid.matches[0].angle,"scale polish changed position or angle");
        auto fixedSettings=settings;fixedSettings.scales={1.};
        Session fixed(initial.view(),templ.view(),fixedSettings);
        const auto fixedResult=fixed.run(polish);
        require(fixedResult.scaleSupportEvaluations==0 && fixedResult.scaleSupportMs==0,"fixed scale ran scale correction");
        polish.refine=false;const auto discrete=reuse.run(polish);
        require(discrete.scaleSupportEvaluations==0 && discrete.scaleSupportMs==0,"discrete route ran scale correction");
        validateBadSettings(initial.view(),templ.view(),settings);
        if(metal) {
            auto cpuRefinement=settings;cpuRefinement.refinementBackend=0;cpuRefinement.poseStorageMode=0;
            Session mixed(initial.view(),templ.view(),cpuRefinement);auto result=mixed.run(options);
            sameMatches(original.proposals,result.proposals);require(original.proposalScales==result.proposalScales,"backend override changed proposals");
            require(result.configuredRefinementBackend==0 && result.rescue.selected_backend==0 &&
                result.refinementCandidates>0 && result.pose.gpu_buffer_bytes==0 && result.resident.encoded_dispatches==0,
                "CPU refinement override ignored");
            require(result.frame.coarse_gpu_bytes>0 && result.matches.size()==1,"CPU override changed Metal frontend or lost target");
            require(std::hypot(result.matches[0].center.x-original.matches[0].center.x,
                result.matches[0].center.y-original.matches[0].center.y)<.25 &&
                std::abs(result.matches[0].score-original.matches[0].score)<1e-4,"mixed backend accuracy mismatch");
            auto cpuFrontend=settings;cpuFrontend.backend=0;cpuFrontend.refinementBackend=1;
            Options cpuOptions=options;cpuOptions.fineNccMode=0;cpuOptions.fineAngleReuse=false;cpuOptions.fineFused=false;
            Session reverse(initial.view(),templ.view(),cpuFrontend);auto reversed=reverse.run(cpuOptions);
            require(reversed.configuredRefinementBackend==1 && reversed.rescue.selected_backend==1 &&
                reversed.pose.gpu_buffer_bytes>0 && reversed.frame.coarse_gpu_bytes==0,"Metal refinement override ignored");
        }
        std::cout<<(metal?"Metal":"CPU")<<" source updates: changed size/pixels, empty/recovery, exact fresh parity, owned snapshots, invalid updates, backend split passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
