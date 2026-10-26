// SPDX-License-Identifier: BSD-2-Clause
// Round 49 contract for the opt-in pyramid rule: the area mapping lands on the
// chosen top level, the decision never deepens the pyramid, fine texture that
// vanishes in the reduced levels is searched higher up, coarse structure keeps
// its depth, and a Session with the rule off (or unreachable) is unchanged.
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
int depth(const Image& templ,int minArea) { return int(Pattern(templ.view(),minArea).levels().size())-1; }
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
// Near-Nyquist stripes on a mid-gray plate: pyrDown removes them, so the 11x8
// top level of this 44x30 template is almost flat while 22x15 still has them.
Image fineTexture() {
    Image t(44,30);std::mt19937 random(49001);
    for(int y=0;y<30;++y) for(int x=0;x<44;++x)
        t.row(y)[x]=uint8_t(std::clamp(128+70*std::sin(2.6*x+1.9*y)+30*std::cos(1.3*x-2.4*y)+int(random()%9)-4.,0.,255.));
    return t;
}
// One bright off-center disk and a gradient: survives every reduction.
Image coarseStructure() {
    Image t(64,64);
    for(int y=0;y<64;++y) for(int x=0;x<64;++x)
        t.row(y)[x]=uint8_t(std::clamp(60+1.2*x+(std::hypot(x-40.,y-22.)<15?110.:0.)+(std::hypot(x-18.,y-46.)<9?-40.:0.),0.,255.));
    return t;
}
void areaMapping() {
    for(auto [w,h]:std::vector<std::pair<int,int>>{{43,30},{58,71},{17,33},{160,120},{1,1},{3,500},{255,1},{102,140},{33,33}}) {
        Image t(w,h);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) t.row(y)[x]=uint8_t((7*x+13*y)%251);
        for(int minArea:{1,64,256,512,1024}) {
            const int top=depth(t,minArea);
            for(int level=0;level<=top;++level) {
                const int area=minAreaForTop(w,h,level);
                require(area>=1,"minAreaForTop must be positive");
                require(depth(t,area)==level,"minAreaForTop "+std::to_string(w)+"x"+std::to_string(h)+
                    " level "+std::to_string(level)+" gives depth "+std::to_string(depth(t,area)));
            }
        }
    }
    invalid([] { minAreaForTop(0,4,0); },"zero width accepted");
    invalid([] { minAreaForTop(4,4,-1); },"negative level accepted");
}
void decisions() {
    const auto fine=fineTexture(),coarse=coarseStructure();
    const auto f=decidePyramid(fine.view(),256,.5);
    require(f.originalTop==2 && f.predictedScores.size()==3,"fine texture keeps three candidate levels");
    require(f.predictedScores[2]<.5 && f.predictedScores[1]>f.predictedScores[2],"fine texture top level should predict a low score");
    require(f.selectedTop<2 && f.predictedScores[f.selectedTop]>=.5,"fine texture should start higher up");
    require(f.effectiveMinArea==minAreaForTop(44,30,f.selectedTop) && depth(fine,f.effectiveMinArea)==f.selectedTop,
        "effective area must give the selected depth");
    const auto c=decidePyramid(coarse.view(),256,.5);
    require(c.originalTop==2 && c.selectedTop==2 && c.effectiveMinArea==256,"coarse structure keeps its depth and area");
    require(c.predictedScores[0]>.9 && c.predictedScores[2]>=.5,"coarse structure predicts high scores");
    // Unreachable threshold: deepest level with the highest predicted score.
    const auto top=decidePyramid(fine.view(),256,1);
    const double best=*std::max_element(top.predictedScores.begin(),top.predictedScores.end());
    require(top.predictedScores[top.selectedTop]==best,"fallback picks the highest score");
    for(int level=top.selectedTop+1;level<=top.originalTop;++level)
        require(top.predictedScores[level]<best,"fallback picks the deepest highest score");
    // Always reachable threshold: the original depth and area.
    const auto keep=decidePyramid(fine.view(),256,-1);
    require(keep.selectedTop==2 && keep.effectiveMinArea==256,"threshold -1 keeps the depth");
    // Uniform templates have no predicted score and keep the depth.
    Image flat(40,40);for(int y=0;y<40;++y) std::fill_n(flat.row(y),40,uint8_t(77));
    const auto u=decidePyramid(flat.view(),256,.5);
    require(u.selectedTop==u.originalTop && u.originalTop==depth(flat,256),"uniform template keeps its depth");
    for(double s:u.predictedScores) require(s==0,"uniform levels predict 0");
    require(u.lowBackground==77 && u.highBackground==77,"uniform backgrounds");
    // Percentile backgrounds, nearest rank.
    std::vector<uint8_t> values;
    for(int y=0;y<30;++y) values.insert(values.end(),fine.row(y),fine.row(y)+44);
    std::sort(values.begin(),values.end());
    require(f.lowBackground==values[size_t(.1*(values.size()-1))] &&
        f.highBackground==values[size_t(std::ceil(.9*(values.size()-1)))],"backgrounds are the 10th/90th percentiles");
    require(f.lowBackground<f.highBackground,"textured template backgrounds differ");
    // Predicted score: worse background's mean, deterministic.
    const Pattern pattern(fine.view(),256);
    for(int level=0;level<=2;++level) {
        const double low=predictedTopScore(pattern,level,f.lowBackground),high=predictedTopScore(pattern,level,f.highBackground);
        require(f.predictedScores[level]==std::min(low,high),"score is the worse background");
        require(low==predictedTopScore(pattern,level,f.lowBackground),"score is deterministic");
        require(low>=-1 && low<=1 && high>=-1 && high<=1,"score is an NCC");
    }
    const auto again=decidePyramid(fine.view(),256,.5);
    require(again.predictedScores==f.predictedScores && again.selectedTop==f.selectedTop,"decision is deterministic");
    invalid([&] { predictedTopScore(pattern,3,0); },"level beyond the pattern accepted");
    invalid([&] { predictedTopScore(pattern,-1,0); },"negative level accepted");
    for(double bad:{std::numeric_limits<double>::quiet_NaN(),1.5,-1.5})
        invalid([&] { decidePyramid(fine.view(),256,bad); },"invalid threshold accepted");
    // Never deeper, on random textures at several smoothness levels.
    std::mt19937 random(49002);
    for(int trial=0;trial<12;++trial) {
        const int w=12+int(random()%70),h=12+int(random()%70),period=1+trial%4;
        Image t(w,h);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) t.row(y)[x]=uint8_t(random()%256);
        Image smooth(w,h);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            int sum=0,n=0;
            for(int dy=-period;dy<=period;++dy) for(int dx=-period;dx<=period;++dx)
                if(x+dx>=0 && x+dx<w && y+dy>=0 && y+dy<h) { sum+=t.row(y+dy)[x+dx];++n; }
            smooth.row(y)[x]=uint8_t(sum/n);
        }
        for(double threshold:{0.,.5,.9}) {
            const auto d=decidePyramid(smooth.view(),256,threshold);
            require(d.originalTop==depth(smooth,256) && d.selectedTop<=d.originalTop,"rule must never deepen the pyramid");
            require(depth(smooth,d.effectiveMinArea)==d.selectedTop,"effective area must give the selected depth");
            require(d.selectedTop!=d.originalTop || d.effectiveMinArea==256,"unchanged depth keeps the area");
        }
    }
}
struct Truth { double x,y,angle; };
// Five rotated copies of the fine-texture template, rendered by inverse bilinear
// sampling on a flat background (independent of the matcher's warp).
Image scene(const Image& t,const std::vector<Truth>& objects) {
    Image s(360,260);for(int y=0;y<260;++y) std::fill_n(s.row(y),360,uint8_t(118));
    const double cx=(t.width()-1)/2.,cy=(t.height()-1)/2.;
    for(const auto& o:objects) {
        const double c=std::cos(o.angle*M_PI/180),sn=std::sin(o.angle*M_PI/180);
        for(int y=0;y<260;++y) for(int x=0;x<360;++x) {
            const double dx=x-o.x,dy=y-o.y,u=c*dx-sn*dy+cx,v=sn*dx+c*dy+cy;
            if(u<0 || v<0 || u>t.width()-1 || v>t.height()-1) continue;
            const int iu=std::min(int(u),t.width()-2),iv=std::min(int(v),t.height()-2);const double fu=u-iu,fv=v-iv;
            const double value=(t.row(iv)[iu]*(1-fu)+t.row(iv)[iu+1]*fu)*(1-fv)+(t.row(iv+1)[iu]*(1-fu)+t.row(iv+1)[iu+1]*fu)*fv;
            s.row(y)[x]=uint8_t(std::floor(value+.5));
        }
    }
    return s;
}
void session(bool metal) {
    const auto fine=fineTexture();
    const std::vector<Truth> objects{{60,60,17},{180,62,-64},{300,70,103},{90,190,-151},{250,185,48}};
    const auto source=scene(fine,objects);
    Settings base;base.backend=metal?1:0;base.refinementBackend=base.backend;base.minReducedArea=256;base.scales={1.};
    Options o;o.match.minScore=.75;o.match.maxTargets=10;o.match.maxOverlap=.2;o.match.toleranceAngle=180;o.match.stopLayer=0;
    o.proposalScore=.5;
    Session off(source.view(),fine.view(),base);
    require(off.pyramidRule().empty(),"rule off reports no decisions");
    auto unreachable=base;unreachable.pyramidRule=1;unreachable.pyramidRuleMinScore=-1;
    Session keep(source.view(),fine.view(),unreachable);
    require(keep.pyramidRule().size()==1 && keep.pyramidRule()[0].selectedTop==keep.pyramidRule()[0].originalTop,
        "threshold -1 keeps the depth");
    const auto offResult=off.run(o);
    same(offResult,keep.run(o),"rule at the original depth must not change the output");
    auto on=base;on.pyramidRule=1;on.pyramidRuleMinScore=.5;
    Session adaptive(source.view(),fine.view(),on);
    const auto& decision=adaptive.pyramidRule().at(0);
    const auto direct=decidePyramid(fine.view(),256,.5);
    require(decision.selectedTop==direct.selectedTop && decision.predictedScores==direct.predictedScores,
        "Session decision differs from decidePyramid");
    require(decision.selectedTop<decision.originalTop,"fine texture should be searched higher up");
    const auto r=adaptive.run(o);
    int found=0;
    for(const auto& t:objects) {
        bool hit=false;
        for(const auto& m:r.matches) hit|=std::hypot(m.center.x-t.x,m.center.y-t.y)<=2 && std::abs(std::remainder(m.angle-t.angle,360.))<=1;
        found+=hit;
    }
    require(found==int(objects.size()),"adaptive depth lost objects: "+std::to_string(found)+"/5");
    // One decision per scale model, on the scaled template.
    auto multi=on;multi.scales={.9,1.,1.1};
    Session scales(source.view(),fine.view(),multi);
    require(scales.pyramidRule().size()==3,"one decision per scale");
    for(size_t i=0;i<3;++i) {
        const auto scaled=scaledTemplate(fine.view(),multi.scales[i]);
        require(scales.pyramidRule()[i].predictedScores==decidePyramid(scaled.view(),256,.5).predictedScores,
            "scale decision must use the scaled template");
    }
    // Updating the source keeps the template decisions.
    scales.updateSource(source.view());
    require(scales.pyramidRule().size()==3,"updateSource kept decisions");
    // Invalid settings.
    for(int bad:{-1,2}) { auto s=base;s.pyramidRule=bad;invalid([&] { Session x(source.view(),fine.view(),s); },"invalid rule mode accepted"); }
    for(double bad:{std::numeric_limits<double>::quiet_NaN(),1.01,-1.01}) {
        auto s=on;s.pyramidRuleMinScore=bad;invalid([&] { Session x(source.view(),fine.view(),s); },"invalid rule threshold accepted");
    }
    auto sparse=on;sparse.sparseOrientation=true;sparse.backend=0;sparse.refinementBackend=0;
    invalid([&] { Session x(source.view(),fine.view(),sparse); },"rule with sparse orientation accepted");
    auto full=on;full.fullGpu=true;full.backend=1;full.refinementBackend=1;
    invalid([&] { Session x(source.view(),fine.view(),full); },"rule with the complete GPU path accepted");
}
}
int main(int argc,char** argv) {
    try {
        const bool metal=argc>1 && std::string(argv[1])=="--metal";
        areaMapping();
        decisions();
        session(metal);
        std::cout<<"Pyramid rule: exact area mapping, never deeper, fine texture higher up, unchanged when off\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
