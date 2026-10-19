// SPDX-License-Identifier: BSD-2-Clause
#include "scale_support_refine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
struct Fixture {
    int sw=80,sh=64,tw=33,th=17,left=21,top=22;
    size_t ss=89,ts=40;
    std::vector<uint8_t> source=std::vector<uint8_t>(ss*sh,150),templ=std::vector<uint8_t>(ts*th,99);
    Fixture() {
        for(int y=0;y<th;++y) for(int x=0;x<tw;++x) {
            // Smooth asymmetric independent integer raster; exact known scale1.
            const auto value=uint8_t(std::lround(120.+35.*std::sin(x*.31+y*.17)+30.*std::cos(y*.48-x*.13)));
            templ[size_t(y)*ts+x]=value;source[size_t(y+top)*ss+x+left]=value;
        }
    }
    fpm::ImageView s() const {return {source.data(),sw,sh,ss};}
    fpm::ImageView t() const {return {templ.data(),tw,th,ts};}
    fpm::Point center() const {return {left+tw/2.,top+th/2.};}
};
} // namespace
int main()
{
    try {
        Fixture f;
        const auto exact=fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),0.,1.);
        require(exact.eligible && !exact.candidate && !exact.boundaryOptimum,"exact copy changed");
        require(exact.evaluations==33 && exact.samples==uint64_t(33*29*13),"exact work accounting");
        require(std::abs(exact.originalInteriorScore-1.)<1e-12,"integer center convention wrong");
        const auto refined=fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),0.,.98);
        require(refined.candidate.has_value(),"nearby scale not corrected");
        require(std::abs(refined.candidate->scale-1.)<.0025,"incorrect recovered scale");
        require(refined.evaluations==34 && refined.samples==uint64_t(33*29*13+33*17),"changed work accounting");
        require(refined.candidate->score>.999,"candidate not verified on full template");
        const auto outside=fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),0.,.90);
        require(outside.boundaryOptimum && !outside.candidate,"bounded optimum must preserve original");
        require(outside.evaluations==33,"boundary should not compute full verification");
        Fixture rotated;
        std::fill(rotated.source.begin(),rotated.source.end(),150);
        for(int y=0;y<rotated.th;++y) for(int x=0;x<rotated.tw;++x)
            rotated.source[size_t(49-x)*rotated.ss+size_t(32+y)]=rotated.templ[size_t(y)*rotated.ts+x];
        const auto quarter=fpm::research::refineScaleSupport(rotated.s(),rotated.t(),{40.5,32.5},90.,.98);
        require(quarter.candidate && std::abs(quarter.candidate->scale-1.)<.0025 && quarter.candidate->score>.999,
            "quarter-turn geometry differs");
        const auto far=fpm::research::refineScaleSupport(f.s(),f.t(),{std::numeric_limits<double>::max(),0},0.,1.);
        require(far.eligible && !far.candidate && far.evaluations==1,"far coordinates must safely score zero");
        for(auto& value:f.source)value=uint8_t(255-value);
        const auto inverted=fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),0.,.98,true);
        require(inverted.candidate && inverted.candidate->scale==refined.candidate->scale &&
            std::abs(inverted.candidate->score-refined.candidate->score)<1e-12,"invert source differs");
        for(auto& value:f.templ)value=120;
        const auto constant=fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),0.,1.);
        require(constant.eligible && !constant.candidate && constant.evaluations==1,"constant template accepted");
        require(!fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),0.,-1.).eligible,"negative scale accepted");
        require(!fpm::research::refineScaleSupport(f.s(),f.t(),f.center(),std::numeric_limits<double>::quiet_NaN(),1.).eligible,"NaN accepted");
        auto small=f.t();small.width=6;
        require(!fpm::research::refineScaleSupport(f.s(),small,f.center(),0.,1.).eligible,"small unsupported template accepted");
        auto huge=f.s();huge.width=65;huge.height=64;
        require(!fpm::research::refineScaleSupport(f.s(),huge,f.center(),0.,1.).eligible,"large unsupported template accepted");
        bool threw=false;auto invalid=f.s();invalid.stride=1;
        try {fpm::research::refineScaleSupport(invalid,f.t(),f.center(),0.,1.);}catch(const std::invalid_argument&){threw=true;}
        require(threw,"invalid image stride accepted");
        std::cout<<"Scale support refinement controls passed\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
