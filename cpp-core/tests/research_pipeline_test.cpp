// SPDX-License-Identifier: BSD-2-Clause
#include "research_pipeline.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fpm;
using namespace fpm::research;
void require(bool v,const char* why) { if(!v) throw std::runtime_error(why); }
void same(const Result& a,const Result& b) {
    require(a.matches.size()==b.matches.size() && a.scales==b.scales,"result count/scale differs");
    require(a.proposals.size()==b.proposals.size(),"proposal count differs");
    auto compare=[](const auto& aa,const auto& bb) {
        for(size_t i=0;i<aa.size();++i) {
            require(aa[i].score==bb[i].score && aa[i].angle==bb[i].angle && aa[i].center.x==bb[i].center.x && aa[i].center.y==bb[i].center.y,"ordered pose/score differs");
            for(int j=0;j<4;++j) require(aa[i].corners[j].x==bb[i].corners[j].x && aa[i].corners[j].y==bb[i].corners[j].y,"corners differ");
        }
    };
    compare(a.matches,b.matches);compare(a.proposals,b.proposals);
}
int main(int argc,char**) {
    try {
        const bool metal=argc>1;
        Image templ(32,24),source(256,192);std::mt19937 random(17);
        for(int y=0;y<24;++y) for(int x=0;x<32;++x) templ.row(y)[x]=uint8_t(random());
        auto small=scaledTemplate(templ.view(),.75);
        for(int y=0;y<192;++y) for(int x=0;x<256;++x) source.row(y)[x]=20;
        for(int y=0;y<24;++y) for(int x=0;x<32;++x) source.row(30+y)[40+x]=templ.row(y)[x];
        for(int y=0;y<small.height();++y) for(int x=0;x<small.width();++x) source.row(110+y)[160+x]=small.row(y)[x];
        for(int y=0;y<24;++y) for(int x=0;x<32;++x) source.row(80+y)[100+x]=255-templ.row(y)[x];
        Settings s;s.backend=metal?1:0;s.scales={.75,1.};s.minReducedArea=10000;
        Session session(source.view(),templ.view(),s);Options o;o.refine=false;o.match.minScore=.9;o.match.maxTargets=5;o.match.maxOverlap=.2;o.match.toleranceAngle=180;
        auto baseline=session.run(o);require(baseline.matches.size()==2,"scaled instances missing");
        require(baseline.frame.models_matched==2 && baseline.frame.generation==1,"frame not shared/prepared");
        for(auto m:baseline.matches) require((std::hypot(m.center.x-56,m.center.y-42)<.1) || (std::hypot(m.center.x-172,m.center.y-119)<.1),"wrong position");
        // Source/template snapshots are owned; caller changes must not leak in.
        source.row(30)[40]^=255;templ.row(0)[0]^=255;same(baseline,session.run(o));
        if(metal) for(int mode:{0,2,3}) for(bool overlap:{false,true}) {
            o.nccMode=mode;o.overlap=overlap;const auto r=session.run(o);same(baseline,r);
            for(auto m:r.overlapMetrics) require(m.submitted_batches==m.completed_batches,"undrained async batches");
        }
        o.refine=true;auto refined=session.run(o);require(!refined.matches.empty(),"refine failed");
        require(refined.rescue.selected_backend==(metal?1:0),"wrong solver backend");
        require(refined.refinementMode==0,"default refinement mode changed");
        require(metal ? refined.resident.encoded_dispatches>refined.resident.command_buffers : refined.resident.encoded_dispatches==0,"wrong default refinement diagnostics");
        same(refined,session.run(o));
        if(metal) for(int fusedMode:{1,2}) for(int storageMode:{0,1}) {
            // Recreate from the original input, not the caller-mutated buffers.
            if(fusedMode==1 && storageMode==0) {source.row(30)[40]^=255;templ.row(0)[0]^=255;}
            Settings fusedSettings=s;fusedSettings.residentMode=fusedMode;fusedSettings.poseStorageMode=storageMode;
            Session fused(source.view(),templ.view(),fusedSettings);
            Options discrete=o;discrete.refine=false;const auto noRefinement=fused.run(discrete);
            require(noRefinement.refinementMode==fusedMode && noRefinement.resident.encoded_dispatches==0,"selected fused mode lost when refinement disabled");
            const auto result=fused.run(o);
            require(result.poseStorage.mode==storageMode && result.poseCreateMs>0,"wrong storage route or missing initialization time");
            require(storageMode ? result.poseStorage.shared_image_bytes==result.poseStorage.host_image_bytes && result.poseStorage.shared_image_bytes>0 : result.poseStorage.shared_image_bytes==0,"wrong shared image accounting");
            require(result.refinementMode==fusedMode && result.resident.command_buffers>0 &&
                result.resident.encoded_dispatches==result.resident.command_buffers &&
                result.resident.cpu_waits==result.resident.command_buffers && result.resident.cpu_verified==1,"fused pipeline route ignored");
            // The .75 bilinear-resized random patch passes the discrete matcher
            // but both refiners reject it at .9 full-resolution NCC. Preserve
            // that control outcome; continuous scale GT is tested separately.
            require(result.matches.size()==refined.matches.size() && result.matches.size()==1 &&
                result.proposals.size()==refined.proposals.size(),"fused pipeline changed reference detections");
            for(const auto& m:result.matches) require(m.score>.99 &&
                std::hypot(m.center.x-56,m.center.y-42)<.1,"fused pipeline GT mismatch");
            const auto repeated=fused.run(o);same(result,repeated);require(repeated.poseCreateMs==0,"cached pose context rebuilt");
            const auto retained=fused.run(discrete);
            require(retained.poseCreateMs==0 && retained.resident.encoded_dispatches==0 &&
                retained.poseStorage.unique_image_bytes==result.poseStorage.unique_image_bytes &&
                retained.poseStorage.shared_image_bytes==result.poseStorage.shared_image_bytes,
                "discrete-only call hid retained pose storage or dispatched refinement");
            Options invertedOptions=o;invertedOptions.match.invertSource=true;
            const auto invertedFused=fused.run(invertedOptions);
            require(invertedFused.matches.size()==1 && std::hypot(invertedFused.matches[0].center.x-116,invertedFused.matches[0].center.y-92)<.1 && invertedFused.matches[0].score>.99,"fused pipeline stale inverted source");
            same(result,fused.run(o));
        }
        o.match.invertSource=true;auto inverted=session.run(o);require(inverted.matches.size()==1 && std::hypot(inverted.matches[0].center.x-116,inverted.matches[0].center.y-92)<.1 && inverted.matches[0].score>.99,"stale noninverted pose/image");
        o.match.invertSource=false;same(refined,session.run(o));
        o.match.angleRanges={{70,40}};bool rejected=false;try { session.run(o); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid range accepted");o.match.angleRanges.clear();same(refined,session.run(o));
        o.match.angleRanges={{20,30}};auto restricted=session.run(o);for(auto m:restricted.matches) require(m.angle>=20-1e-5 && m.angle<=30+1e-5,"refinement escaped angle range");
        Settings invalid=s;invalid.scales={0};rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"zero scale accepted");
        for(int mode:{-1,3}) { invalid=s;invalid.residentMode=mode;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid resident mode accepted"); }
        invalid=s;invalid.backend=0;invalid.residentMode=1;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"CPU pipeline accepted fused mode");
        for(int mode:{-1,2}) {invalid=s;invalid.poseStorageMode=mode;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid storage mode accepted");}
        invalid=s;invalid.backend=0;invalid.poseStorageMode=1;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"CPU pipeline accepted shared pose storage");
        Image odd(5,7);auto half=scaledTemplate(odd.view(),.5);require(half.width()==2 && half.height()==4,"ties-even resize");
        std::cout<<(metal?"Metal":"CPU")<<" native pipeline: scale, source ownership, options, repeated calls, inversion and refinement passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
