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
        source.row(30)[40]^=255;templ.row(0)[0]^=255;
        if(metal) {
            // Reduced levels are required to actually visit the fine scheduler.
            auto fineSettings=s;fineSettings.minReducedArea=64;
            Session fineSession(source.view(),templ.view(),fineSettings);
            Options fineOptions=o;fineOptions.match.subpixel=true;
            const auto original=fineSession.run(fineOptions);
            fineOptions.fineNccMode=1;
            const auto shared=fineSession.run(fineOptions);same(original,shared);
            bool ran=false;
            for(size_t i=0;i<shared.fineReuse.size();++i) {
                const auto& reuse=shared.fineReuse[i];
                require(reuse.ncc_mode==1 && reuse.row_dispatches==reuse.statistics_dispatches,"shared fine route ignored");
                require(shared.fine[i].command_buffers==original.fine[i].command_buffers,"extra fine waits introduced");
                ran|=reuse.row_dispatches>0;
            }
            require(ran,"full pipeline test omitted fine dispatches");
            fineOptions.fineNccMode=0;same(original,fineSession.run(fineOptions));
            for(int mode:{2,3}) {
                // Independent sessions start with the same empty arenas as
                // the original run, so compare high-water capacities fairly.
                Session stripSession(source.view(),templ.view(),fineSettings);
                fineOptions.fineNccMode=mode;
                const auto strip=stripSession.run(fineOptions);same(original,strip);
                require(strip.fineReuse.size()==original.fineReuse.size(),"fine scale metrics missing");
                bool stripRan=false;
                for(size_t i=0;i<strip.fineReuse.size();++i) {
                    const auto& reuse=strip.fineReuse[i];
                    const auto& actual=strip.fine[i];const auto& expected=original.fine[i];
                    require(reuse.ncc_mode==mode && !reuse.row_dispatches && !reuse.row_buffer_bytes,
                        "strip fine route ignored or retained unused row storage");
                    require(reuse.statistics_dispatches==original.fineReuse[i].statistics_dispatches &&
                        actual.command_buffers==expected.command_buffers && actual.maps==expected.maps &&
                        actual.peak_batch_maps==expected.peak_batch_maps,"strip route changed fine work or waits");
                    require(actual.source_buffer_bytes==expected.source_buffer_bytes &&
                        actual.roi_buffer_bytes==expected.roi_buffer_bytes && actual.guard_buffer_bytes==expected.guard_buffer_bytes &&
                        actual.descriptor_buffer_bytes==expected.descriptor_buffer_bytes &&
                        actual.statistics_buffer_bytes==expected.statistics_buffer_bytes &&
                        actual.template_buffer_bytes==expected.template_buffer_bytes &&
                        actual.retained_gpu_bytes==expected.retained_gpu_bytes,"strip route changed arena capacity");
                    require(actual.roi_pixels==expected.roi_pixels && actual.repair_pixels==expected.repair_pixels,
                        "strip route changed guarded rotation");
                    stripRan|=reuse.statistics_dispatches>0;
                }
                require(stripRan,"full pipeline strip test omitted fine dispatches");
                same(strip,stripSession.run(fineOptions));
                fineOptions.fineNccMode=1;same(original,stripSession.run(fineOptions));
                fineOptions.fineNccMode=mode;
                const auto switched=stripSession.run(fineOptions);same(original,switched);
                for(const auto& reuse:switched.fineReuse)
                    require(!reuse.row_buffer_bytes && !reuse.row_dispatches,"pipeline strip switch retained row work/storage");
            }
        }
        if(metal) {
            auto settings=s;settings.minReducedArea=64;
            Session measured(source.view(),templ.view(),settings);
            Options options=o;options.match.subpixel=true;options.match.minScore=.6;options.fineNccMode=2;
            const auto reference=measured.run(options);
            for(bool fusion:{false,true}) for(bool profiling:{false,true}) for(bool reuse:{false,true}) {
                options.fineFused=fusion;options.fineProfiling=profiling;options.fineAngleReuse=reuse;
                const auto actual=measured.run(options);same(reference,actual);
                require(actual.fineSearch.size()==reference.fine.size(),"fine search metrics missing");
                uint64_t hits=0;
                for(size_t i=0;i<actual.fineSearch.size();++i) {
                    const auto& d=actual.fineSearch[i];const auto& f=actual.fine[i];
                    const auto& g=actual.fineFusion[i];
                    require(g.enabled==fusion && f.command_buffers==actual.fineReuse[i].statistics_dispatches*(fusion?1:2)+g.fallback_batches,
                        "fusion submission contract failed");
                    require(!fusion || g.fused_batches>0,"integration omitted fused work");
                    require(d.profiling_enabled==profiling && d.angle_reuse_enabled==reuse,"fine search option ignored");
                    if(reuse) {
                        require(d.logical_requests==reference.fine[i].maps && d.reused_maps+f.maps==d.logical_requests,
                            "reuse changed logical angle work");
                        hits+=d.reused_maps;
                    }
                    if(profiling) {
                        require(d.requests==f.maps && d.requests==d.within_call_duplicates+d.cross_call_duplicates+
                            d.tracked_unique_keys+d.untracked_call_keys,"request observation accounting differs");
                        require(d.tracked_unique_keys<=65536 && d.warp_batches==actual.fineReuse[i].statistics_dispatches,
                            "fine diagnostic limit or batches differ");
                        require(std::abs(d.warp_wait_ms+d.ncc_wait_ms+g.fused_wait_ms-f.wait_ms)<1e-7,"split waits do not sum to measured wait");
                    }
                }
                require(!reuse || hits>0,"reuse integration omitted hits");same(actual,measured.run(options));
            }
            options.fineFused=false;options.fineProfiling=false;options.fineAngleReuse=false;same(reference,measured.run(options));
        } else {
            Options fusion=o; fusion.fineFused=true; bool fusionRejected=false;
            try { session.run(fusion); } catch(const std::invalid_argument&) { fusionRejected=true; }
            require(fusionRejected,"CPU accepted Metal fine fusion");
            for(bool profiling:{false,true}) {
                Options invalid=o;invalid.fineProfiling=profiling;invalid.fineAngleReuse=!profiling;bool caught=false;
                try{session.run(invalid);}catch(const std::invalid_argument&){caught=true;}
                require(caught,"CPU accepted Metal fine search options");
            }
        }
        for(int mode:{-1,4}) {
            Options invalid=o;invalid.fineNccMode=mode;bool caught=false;
            try{session.run(invalid);}catch(const std::invalid_argument&){caught=true;}
            require(caught,"invalid fine mode accepted");
        }
        if(!metal) for(int mode:{1,2,3}) {
            Options invalid=o;invalid.fineNccMode=mode;bool caught=false;
            try{session.run(invalid);}catch(const std::invalid_argument&){caught=true;}
            require(caught,"CPU accepted Metal fine mode");
        }
        if(metal) for(int mode:{0,2,3}) for(bool overlap:{false,true}) {
            o.nccMode=mode;o.overlap=overlap;const auto r=session.run(o);same(baseline,r);
            for(auto m:r.overlapMetrics) require(m.submitted_batches==m.completed_batches,"undrained async batches");
        }
        o.refine=true;auto refined=session.run(o);require(!refined.matches.empty(),"refine failed");
        require(refined.rescue.selected_backend==(metal?1:0),"wrong solver backend");
        require(refined.refinementMode==0,"default refinement mode changed");
        require(metal ? refined.resident.encoded_dispatches>refined.resident.command_buffers : refined.resident.encoded_dispatches==0,"wrong default refinement diagnostics");
        same(refined,session.run(o));
        require(refined.poseInputMode==0 && refined.poseConversionTemporaryBytes>0,"float conversion allocation missing");
        Settings directSettings=s;directSettings.poseInputMode=1;
        Session direct(source.view(),templ.view(),directSettings);
        const auto directResult=direct.run(o);same(refined,directResult);
        require(directResult.poseInputMode==1 && directResult.poseConversionTemporaryBytes==0 && directResult.poseCreateMs>0,"direct input route ignored");
        const auto directRepeated=direct.run(o);same(directResult,directRepeated);
        require(directRepeated.poseCreateMs==0 && directRepeated.poseConversionTemporaryBytes==0,"direct cached context recreated");
        Result floatFused,floatInverted;
        if(metal) for(int fusedMode:{1,2}) for(int storageMode:{0,1}) for(int inputMode:{0,1}) {
            Settings fusedSettings=s;fusedSettings.residentMode=fusedMode;fusedSettings.poseStorageMode=storageMode;fusedSettings.poseInputMode=inputMode;
            Session fused(source.view(),templ.view(),fusedSettings);
            Options discrete=o;discrete.refine=false;const auto noRefinement=fused.run(discrete);
            require(noRefinement.refinementMode==fusedMode && noRefinement.resident.encoded_dispatches==0,"selected fused mode lost when refinement disabled");
            const auto result=fused.run(o);
            require(result.poseStorage.mode==storageMode && result.poseCreateMs>0,"wrong storage route or missing initialization time");
            require(result.poseInputMode==inputMode && (inputMode ? result.poseConversionTemporaryBytes==0 : result.poseConversionTemporaryBytes>0),"wrong input route or temporary capacity");
            if(inputMode==0) floatFused=result;else same(floatFused,result);
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
            const auto repeated=fused.run(o);same(result,repeated);require(repeated.poseCreateMs==0 && repeated.poseConversionTemporaryBytes==0,"cached pose context rebuilt");
            const auto retained=fused.run(discrete);
            require(retained.poseCreateMs==0 && retained.poseConversionTemporaryBytes==0 && retained.poseInputMode==inputMode && retained.resident.encoded_dispatches==0 &&
                retained.poseStorage.unique_image_bytes==result.poseStorage.unique_image_bytes &&
                retained.poseStorage.shared_image_bytes==result.poseStorage.shared_image_bytes,
                "discrete-only call hid retained pose storage or dispatched refinement");
            Options invertedOptions=o;invertedOptions.match.invertSource=true;
            const auto invertedFused=fused.run(invertedOptions);
            if(inputMode==0) floatInverted=invertedFused;else same(floatInverted,invertedFused);
            require(invertedFused.matches.size()==1 && std::hypot(invertedFused.matches[0].center.x-116,invertedFused.matches[0].center.y-92)<.1 && invertedFused.matches[0].score>.99,"fused pipeline stale inverted source");
            same(result,fused.run(o));
        }
        o.match.invertSource=true;auto inverted=session.run(o);require(inverted.matches.size()==1 && std::hypot(inverted.matches[0].center.x-116,inverted.matches[0].center.y-92)<.1 && inverted.matches[0].score>.99,"stale noninverted pose/image");
        const auto directInverted=direct.run(o);same(inverted,directInverted);
        require(directInverted.poseCreateMs>0 && directInverted.poseConversionTemporaryBytes==0,"direct inversion failed to rebuild without temporary");
        o.match.invertSource=false;same(refined,session.run(o));
        same(refined,direct.run(o));
        o.match.angleRanges={{70,40}};bool rejected=false;try { session.run(o); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid range accepted");o.match.angleRanges.clear();same(refined,session.run(o));
        o.match.angleRanges={{20,30}};auto restricted=session.run(o);for(auto m:restricted.matches) require(m.angle>=20-1e-5 && m.angle<=30+1e-5,"refinement escaped angle range");
        Settings invalid=s;invalid.scales={0};rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"zero scale accepted");
        for(int mode:{-1,3}) { invalid=s;invalid.residentMode=mode;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid resident mode accepted"); }
        invalid=s;invalid.backend=0;invalid.residentMode=1;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"CPU pipeline accepted fused mode");
        for(int mode:{-1,2}) {invalid=s;invalid.poseStorageMode=mode;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid storage mode accepted");}
        invalid=s;invalid.backend=0;invalid.poseStorageMode=1;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"CPU pipeline accepted shared pose storage");
        for(int mode:{-1,2}) {invalid=s;invalid.poseInputMode=mode;rejected=false;try{Session broken(source.view(),templ.view(),invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid input mode accepted");}
        Image odd(5,7);auto half=scaledTemplate(odd.view(),.5);require(half.width()==2 && half.height()==4,"ties-even resize");
        std::cout<<(metal?"Metal":"CPU")<<" native pipeline: scale, source ownership, options, repeated calls, inversion and refinement passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
