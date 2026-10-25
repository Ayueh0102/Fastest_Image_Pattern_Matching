// SPDX-License-Identifier: BSD-2-Clause
// Round 43 Session contract for the semantics-changing candidate dedup option:
// off is the unchanged path, on only filters coarse candidates, and both the
// hybrid and complete GPU routes report their filter counts.
#include "research_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
using namespace fpm;
using namespace fpm::research;
namespace {
void require(bool v,const std::string& why) { if(!v) throw std::runtime_error(why); }
template<class F> void invalid(F&& operation,const char* why) {
    bool rejected=false;
    try { operation(); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,why);
}
void same(const Result& a,const Result& b,const char* why) {
    auto compare=[&](const std::vector<Match>& x,const std::vector<Match>& y) {
        require(x.size()==y.size(),why);
        for(size_t i=0;i<x.size();++i) {
            require(x[i].score==y[i].score && x[i].angle==y[i].angle && x[i].center.x==y[i].center.x && x[i].center.y==y[i].center.y,why);
            for(int k=0;k<4;++k) require(x[i].corners[k].x==y[i].corners[k].x && x[i].corners[k].y==y[i].corners[k].y,why);
        }
    };
    compare(a.matches,b.matches);compare(a.proposals,b.proposals);require(a.scales==b.scales,why);
}
struct Truth { double x,y,angle; };
void truth(const Result& r,const std::vector<Truth>& expected,const char* why) {
    require(r.matches.size()==expected.size(),std::string(why)+": match count");
    for(const auto& t:expected) {
        bool found=false;
        for(const auto& m:r.matches)
            found|=std::hypot(m.center.x-t.x,m.center.y-t.y)<=2 && std::abs(std::remainder(m.angle-t.angle,360.))<=1;
        require(found,std::string(why)+": ground-truth object missing");
    }
}
// Dedup measures distances between object centers, not the coarse corner
// points. Scene 1: the rotated corners of an upright and a 180-degree object
// coincide. Scene 2: their corners offset by twice the half size coincide.
// The centers are 66 px apart in both, so a position-only one-candidate
// filter must keep both objects.
void centerConvention(const Settings& base) {
    Image templ(64,16);
    for(int y=0;y<16;++y) for(int x=0;x<64;++x)
        templ.row(y)[x]=uint8_t(std::clamp(128+60*std::sin(x/5.+y/7.)+40*std::cos(x/3.-y/9.)+1.5*x,0.,255.));
    for(const auto& b:std::vector<Truth>{{36,84,180},{164,116,180}}) {
        Image source(260,200);std::mt19937 random(43006);
        for(int y=0;y<200;++y) for(int x=0;x<260;++x) source.row(y)[x]=uint8_t(90+random()%5);
        for(int y=0;y<16;++y) for(int x=0;x<64;++x) {
            source.row(92+y)[68+x]=templ.row(y)[x];
            source.row(int(b.y)-8+y)[int(b.x)-32+x]=templ.row(15-y)[63-x];
        }
        auto s=base;s.minReducedArea=256;
        Session session(source.view(),templ.view(),s);
        Options o;o.refine=false;o.match.minScore=.8;o.match.maxTargets=4;o.match.maxOverlap=.2;o.match.toleranceAngle=180;
        o.match.stopLayer=0;o.match.subpixel=true;
        const std::vector<Truth> expected{{100,100,0},b};
        truth(session.run(o),expected,"center scene baseline");
        o.candidateDedupTopK=1;o.candidateDedupAngleSteps=1000;
        truth(session.run(o),expected,"center convention");
    }
}
}
int main(int argc,char** argv) {
    try {
        const std::string mode=argc>1?argv[1]:"--cpu";
        const bool metal=mode!="--cpu",full=mode=="--full-gpu";
        // Four upright copies and one 180-degree copy; centers use width/2.
        Image templ(48,32),source(384,288);std::mt19937 random(43005);
        // Smooth asymmetric texture: it must survive the reduced pyramid levels.
        for(int y=0;y<32;++y) for(int x=0;x<48;++x)
            templ.row(y)[x]=uint8_t(std::clamp(128+60*std::sin(x/5.+y/7.)+40*std::cos(x/3.-y/9.)+1.5*x-random()%7,0.,255.));
        for(int y=0;y<288;++y) for(int x=0;x<384;++x) source.row(y)[x]=uint8_t(90+random()%5);
        std::vector<Truth> expected;
        for(auto [left,top]:std::vector<std::pair<int,int>>{{20,20},{150,30},{260,40},{40,180}}) {
            for(int y=0;y<32;++y) for(int x=0;x<48;++x) source.row(top+y)[left+x]=templ.row(y)[x];
            expected.push_back({left+24.,top+16.,0});
        }
        for(int y=0;y<32;++y) for(int x=0;x<48;++x) source.row(200+y)[250+x]=templ.row(31-y)[47-x];
        expected.push_back({274.,216.,180});
        Settings s;s.backend=metal?1:0;s.refinementBackend=s.backend;s.scales={1.};s.minReducedArea=64;s.fullGpu=full;
        Session session(source.view(),templ.view(),s);
        Options o;o.refine=false;o.match.minScore=.8;o.match.maxTargets=8;o.match.maxOverlap=.2;o.match.toleranceAngle=180;o.match.stopLayer=0;o.match.subpixel=true;
        const auto baseline=session.run(o);truth(baseline,expected,"baseline");
        auto off=o;off.candidateDedupRadius=3.5;off.candidateDedupAngleSteps=999; // ignored while top_k is 0
        const auto disabled=session.run(off);same(baseline,disabled,"top_k=0 changed the default path");
        require(disabled.candidateDedupTopK==0,"disabled option not reported");
        if(full) require(disabled.candidateDedup.empty(),"full GPU must not count candidates while disabled");
        else {
            require(disabled.candidateDedup.size()==1,"one dedup record per model");
            const auto& d=disabled.candidateDedup[0];
            require(d.input>0 && d.input==d.kept && d.filterMs==0,"disabled filter must keep every candidate");
        }
        auto on=o;on.candidateDedupTopK=3;
        const auto deduped=session.run(on);
        require(deduped.candidateDedupTopK==3 && deduped.candidateDedup.size()==1,"enabled option not reported");
        const auto& d=deduped.candidateDedup[0];
        require(d.kept>0 && d.kept<d.input,"scene must exercise both kept and dropped candidates");
        require(std::abs(d.radiusPx-.5*.8*32)<1e-9,"radius is radius*(1-maxOverlap)*min(template side)");
        // Three top-level steps of atan(2/max side); the top size depends on the pyramid.
        bool steps=false;
        for(int side=1;side<=48;++side) steps|=std::abs(d.angleWindowDeg-3*std::atan(2./side)*180/3.141592653589793)<1e-6;
        require(steps,"angle window is not 3 top-level coarse steps");
        truth(deduped,expected,"k=3");
        if(metal && !full) require(deduped.fine[0].maps<baseline.fine[0].maps,"dedup did not reduce fine-search maps");
        centerConvention(s);
        same(deduped,session.run(on),"dedup output must repeat exactly");
        if(full) {
            const auto& g=deduped.fullGpu;
            require(g.commandBuffers==1 && g.cpuWaits==1 && g.intermediateReadbacks==0 && g.cpuAlgorithmSteps==0,
                "dedup added a host decision to the complete GPU path");
            require(g.readbackBytes>baseline.fullGpu.readbackBytes,"dedup counters must be part of the final readback");
        } else if(metal) {
            // Hybrid Metal coarse scores are exact CPU-normalized values, so the
            // CPU frame must make the same filter decisions.
            auto cpu=s;cpu.backend=0;cpu.refinementBackend=0;
            Session reference(source.view(),templ.view(),cpu);
            const auto expectedRun=reference.run(on);same(expectedRun,deduped,"CPU and Metal hybrid frames differ with dedup");
            require(expectedRun.candidateDedup[0].input==d.input && expectedRun.candidateDedup[0].kept==d.kept,"CPU and Metal kept counts differ");
        }
        auto refine=on;refine.refine=true;truth(session.run(refine),expected,"k=3 with pose refinement");
        auto bad=on;bad.candidateDedupTopK=-1;invalid([&]{session.run(bad);},"negative top_k accepted");
        bad=on;bad.candidateDedupRadius=std::nan("");invalid([&]{session.run(bad);},"NaN radius accepted");
        bad=on;bad.candidateDedupRadius=4.5;invalid([&]{session.run(bad);},"radius above 4 accepted");
        bad=on;bad.candidateDedupAngleSteps=-1;invalid([&]{session.run(bad);},"negative angle window accepted");
        if(!metal) {
            Settings sparse;sparse.backend=0;sparse.sparseOrientation=true;
            Session orientation(source.view(),templ.view(),sparse);Options so;so.refine=true;so.match.stopLayer=0;so.candidateDedupTopK=3;
            invalid([&]{orientation.run(so);},"sparse orientation accepted candidate dedup");
        }
        std::cout<<"candidate dedup pipeline "<<mode<<": "<<d.input<<" -> "<<d.kept<<" candidates\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
