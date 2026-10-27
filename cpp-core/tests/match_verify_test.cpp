// SPDX-License-Identifier: BSD-2-Clause
// Round 52 contract for the opt-in match verification: interior candidates are
// never rescored, a footprint that leaves the source keeps only an in-image NCC
// that reaches minScore with enough of the template inside, small templates need
// the size-scaled threshold, invalid rules throw, and a Session with the rules
// off (or not triggered) is unchanged. A smooth ramp on a flat noisy scene makes
// the border false positives of round 50 (proc50105) on every backend.
#include "research_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
Image texture(int w,int h,unsigned seed) {
    Image t(w,h);std::mt19937 random(seed);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        t.row(y)[x]=uint8_t(std::clamp(128+60*std::sin(x/3.+y/5.)+40*std::cos(x/2.-y/4.)+int(random()%11)-5.,0.,255.));
    return t;
}
Image noise(int w,int h,int level,unsigned seed) {
    Image s(w,h);std::mt19937 random(seed);std::normal_distribution<double> n(0,6);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) s.row(y)[x]=uint8_t(std::clamp(level+n(random),0.,255.));
    return s;
}
// Writes templ into source with its centre at (cx,cy) and the verifier's mapping:
// template pixel (tx,ty) sits at centre + R(angle)*(tx-w/2, ty-h/2); pixels off the source are dropped.
void paste(Image& source,const Image& templ,double cx,double cy,double angle,bool invert=false) {
    const double a=angle*3.14159265358979323846/180,c=std::cos(a),s=std::sin(a);
    const int w=templ.width(),h=templ.height();
    for(int y=0;y<source.height();++y) for(int x=0;x<source.width();++x) {
        const double dx=x-cx,dy=y-cy,u=c*dx-s*dy+w/2.,v=s*dx+c*dy+h/2.;
        if(u<0 || v<0 || u>w-1 || v>h-1) continue;
        const int iu=std::min(int(u),w-2),iv=std::min(int(v),h-2);const double fu=u-iu,fv=v-iv;
        const double p=(templ.row(iv)[iu]*(1-fu)+templ.row(iv)[iu+1]*fu)*(1-fv)+(templ.row(iv+1)[iu]*(1-fu)+templ.row(iv+1)[iu+1]*fu)*fv;
        source.row(y)[x]=uint8_t(std::clamp(std::floor((invert?255-p:p)+.5),0.,255.));
    }
}
int touching(const Result& r,int w,int h) {
    int n=0;
    for(const auto& m:r.matches) {
        bool out=false;for(const auto& p:m.corners) out=out || p.x<0 || p.y<0 || p.x>w || p.y>h;
        n+=out;
    }
    return n;
}
void unitChecks() {
    const Image templ=texture(40,30,52001);
    Image source=noise(256,192,128,52002);
    paste(source,templ,100,96,0);paste(source,templ,8,40,0);paste(source,templ,-4,150,0);paste(source,templ,200,5,30);
    MatchVerifyRules off,border,size;border.borderMinInside=.5;size.smallTemplateArea=784;
    // Off: everything is kept, nothing is computed.
    auto d=verifyMatch(source.view(),templ.view(),{-4,150},0,1,.81,.8,false,off);
    require(d.keep && !d.border && d.insideScore==2 && d.requiredScore==.8,"off rules changed a decision");
    // Interior candidates are never rescored.
    d=verifyMatch(source.view(),templ.view(),{100,96},0,1,.99,.8,false,border);
    require(d.keep && !d.border && d.insideFraction==1 && d.insideScore==2,"interior candidate rescored");
    // A true object 30% off the left edge keeps an in-image score near 1.
    d=verifyMatch(source.view(),templ.view(),{8,40},0,1,.81,.8,false,border);
    require(d.border && std::abs(d.insideFraction-28./40)<1e-12 && d.insideScore>.999 && d.keep,"partial true object rejected");
    // Rotated partial copy across the top edge uses the same mapping.
    d=verifyMatch(source.view(),templ.view(),{200,5},30,1,.81,.8,false,border);
    require(d.border && d.insideFraction>.5 && d.insideScore>.98 && d.keep,"rotated partial object rejected");
    // Inverted source: the sign of the in-image NCC follows invertSource.
    Image inverted=noise(256,192,128,52003);paste(inverted,templ,8,40,0,true);
    require(verifyMatch(inverted.view(),templ.view(),{8,40},0,1,.81,.8,true,border).keep,"inverted partial object rejected");
    d=verifyMatch(inverted.view(),templ.view(),{8,40},0,1,.81,.8,false,border);
    require(!d.keep && d.rejectedBorder && d.insideScore< -.99,"uninverted negative copy kept");
    // Same footprint over flat noise: the in-image part does not resemble the template.
    d=verifyMatch(source.view(),templ.view(),{8,120},0,1,.9,.8,false,border);
    require(d.border && !d.keep && d.rejectedBorder && d.insideScore<.5,"border false positive kept");
    // Less than half inside is rejected even when the visible part matches.
    d=verifyMatch(source.view(),templ.view(),{-4,150},0,1,.9,.8,false,border);
    require(std::abs(d.insideFraction-16./40)<1e-12 && d.insideScore>.999 && !d.keep && d.rejectedBorder,"mostly outside candidate kept");
    // Size rule: 12x8 at scale 1 needs 1-0.2*sqrt(96/784); larger areas keep minScore.
    const Image tiny=texture(12,8,52004),big=texture(40,40,52005);
    const double need=1-.2*std::sqrt(96./784);
    d=verifyMatch(source.view(),tiny.view(),{100,96},0,1,need-1e-6,.8,false,size);
    require(!d.keep && d.rejectedSize && std::abs(d.requiredScore-need)<1e-15,"size threshold not applied");
    require(verifyMatch(source.view(),tiny.view(),{100,96},0,1,need+1e-6,.8,false,size).keep,"score above the size threshold rejected");
    d=verifyMatch(source.view(),tiny.view(),{100,96},0,2,.87,.8,false,size);
    require(std::abs(d.requiredScore-(1-.2*std::sqrt(384./784)))<1e-15 && d.keep,"size threshold ignores the match scale");
    d=verifyMatch(source.view(),big.view(),{100,96},0,1,.8,.8,false,size);
    require(d.requiredScore==.8 && d.keep,"large template threshold changed");
    // Invalid rules and views throw.
    for(double v:{-.1,1.5,std::numeric_limits<double>::quiet_NaN()}) {
        MatchVerifyRules r;r.borderMinInside=v;
        invalid([&]{verifyMatch(source.view(),templ.view(),{100,96},0,1,.9,.8,false,r);},"invalid border fraction accepted");
    }
    for(double v:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        MatchVerifyRules r;r.smallTemplateArea=v;
        invalid([&]{verifyMatch(source.view(),templ.view(),{100,96},0,1,.9,.8,false,r);},"invalid area accepted");
    }
    invalid([&]{verifyMatch(ImageView{},templ.view(),{100,96},0,1,.9,.8,false,border);},"empty source accepted");
}
void sessionChecks(bool metal,bool full) {
    // A smooth ramp on flat noise: every match of the plain pipeline crosses the border.
    Image ramp(20,14);std::mt19937 random(52006);std::normal_distribution<double> n(0,2);
    for(int y=0;y<14;++y) for(int x=0;x<20;++x) ramp.row(y)[x]=uint8_t(std::clamp(40+180.*x/19+n(random),0.,255.));
    const Image flat=noise(256,256,128,52007);
    Settings s;s.backend=metal?1:0;s.refinementBackend=s.backend;s.scales={1.};s.minReducedArea=256;s.fullGpu=full;
    Options o;o.match.minScore=.8;o.match.maxTargets=100;o.match.maxOverlap=.2;o.match.toleranceAngle=180;o.match.stopLayer=1;
    if(full) {o.fullGpuPeakMode=2;o.fullGpuFineMode=4;}
    Session rampSession(flat.view(),ramp.view(),s);
    const auto plain=rampSession.run(o);
    require(touching(plain,256,256)>0,"ramp fixture lost its border false positives");
    require(!plain.verifyEnabled && plain.verifyChecked==0,"disabled verification reported work");
    auto on=o;on.verify.borderMinInside=.5;
    const auto checked=rampSession.run(on);
    require(checked.verifyEnabled && checked.verifyAfterNms==full,"verification mode not reported");
    require(touching(checked,256,256)==0 && checked.verifyRejectedBorder>0 && checked.verifyBorder>=checked.verifyRejectedBorder,"border false positives survived");
    // Textured interior copies: neither rule triggers, so outputs are identical.
    const Image templ=texture(48,32,52008);Image source=noise(384,288,96,52009);
    for(auto [x,y,a]:std::vector<std::tuple<double,double,double>>{{60,60,0},{200,80,35},{300,200,-120},{100,220,90}}) paste(source,templ,x,y,a);
    Session textured(source.view(),templ.view(),s);
    const auto base=textured.run(o);
    require(base.matches.size()==4,"textured fixture changed");
    auto quiet=o;quiet.verify.borderMinInside=.5;quiet.verify.smallTemplateArea=1000; // 48*32 >= 1000
    same(base,textured.run(quiet),"untriggered rules changed the outputs");
    // A large area makes the size rule bite; every kept match meets its threshold.
    auto strict=o;strict.verify.smallTemplateArea=1e6;
    const auto tight=textured.run(strict);const double need=1-.2*std::sqrt(48.*32/1e6);
    for(const auto& m:tight.matches) require(m.score>=need,"size threshold not enforced");
    std::vector<Match> expected;for(const auto& m:base.matches) if(m.score>=need) expected.push_back(m);
    require(tight.matches.size()==expected.size(),"size rule removed the wrong matches");
    for(size_t i=0;i<expected.size();++i) require(tight.matches[i].score==expected[i].score && tight.matches[i].center.x==expected[i].center.x,"size rule reordered matches");
    // Before NMS (CPU/hybrid) rejected duplicates are counted too; after NMS only final matches are.
    const uint64_t dropped=base.matches.size()-expected.size();
    require(full?tight.verifyRejectedSize==dropped:tight.verifyRejectedSize>=dropped,"size rejections miscounted");
    invalid([&]{auto bad=o;bad.verify.borderMinInside=2;rampSession.run(bad);},"invalid option accepted by the Session");
}
}
int main(int argc,char** argv) {
    try {
        const std::string mode=argc>1?argv[1]:"--cpu";
        unitChecks();
        sessionChecks(mode!="--cpu",mode=="--full-gpu");
        std::cout<<"match verify "<<mode<<" passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"match verify failed: "<<e.what()<<"\n";
        return 1;
    }
}
