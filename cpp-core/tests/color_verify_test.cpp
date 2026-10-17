// SPDX-License-Identifier: BSD-2-Clause
#include "color_verify.hpp"
#include "pose_ncc.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
struct Fixture {
    int sw=12,sh=12,tw=6,th=4;
    std::size_t ss=39,ts=21; // exercise padded rows
    std::vector<std::uint8_t> source=std::vector<std::uint8_t>(ss*sh,0);
    std::vector<std::uint8_t> templ=std::vector<std::uint8_t>(ts*th,255);
    Fixture() {
        for(int y=0;y<th;++y)for(int x=0;x<tw;++x) {
            auto* p=templ.data()+std::size_t(y)*ts+3*x;
            p[0]=std::uint8_t(80+9*x+5*y);p[1]=std::uint8_t(110+3*x+7*y);p[2]=std::uint8_t(150-7*x+4*y);
        }
        copy();
    }
    void copy() {
        for(int y=0;y<th;++y)for(int x=0;x<tw;++x)for(int c=0;c<3;++c)
            source[std::size_t(y+4)*ss+3*(x+3)+c]=templ[std::size_t(y)*ts+3*x+c];
    }
    fpm::research::Rgb8View s()const{return{source.data(),sw,sh,ss};}
    fpm::research::Rgb8View t()const{return{templ.data(),tw,th,ts};}
    float ncc() const {
        char error[512]{};
        std::unique_ptr<void,decltype(&fpm_pose_destroy)> handle(fpm_pose_create_u8(0,1,source.data(),sw,sh,ss,
            templ.data(),tw,th,ts,3,0,0,error,sizeof(error)),&fpm_pose_destroy);
        if(!handle)throw std::runtime_error(error);
        const float pose[4]{};float score=0;FpmPoseMetrics metrics{};
        if(fpm_pose_score(handle.get(),pose,1,&score,&metrics,error,sizeof(error)))throw std::runtime_error(error);
        return score;
    }
};
double linear(double encoded){return encoded<=.04045?encoded/12.92:std::pow((encoded+.055)/1.055,2.4);}
std::uint8_t expose(std::uint8_t byte,double gain) {
    const double value=linear(byte/255.)*gain;
    const double encoded=value<=.0031308?12.92*value:1.055*std::pow(value,1./2.4)-.055;
    return std::uint8_t(std::lround(255*encoded));
}
}
int main()
{
    try {
        using fpm::research::ColorVerifier;
        Fixture exact;
        ColorVerifier original(exact.s(),exact.t());
        auto result=original.score({0,0,0,0});
        require(result.status==0 && result.validPixels==24 && result.changedPixels==0 && result.meanDistance==0,"Exact RGB crop changed");
        require(exact.ncc()>.999999,"Exact RGB NCC differs");
        const auto full=ColorVerifier(exact.t(),exact.t()).score({0,0,0,0});
        require(full.status==0 && full.validPixels==24 && full.meanDistance==0,"Integer last row/column incorrectly excluded");
        std::fill(exact.source.begin(),exact.source.end(),0);
        require(original.score({0,0,0,0}).meanDistance==0,"Constructor borrowed source memory");
        const auto dark=ColorVerifier(exact.s(),exact.t()).score({0,0,0,0});
        require(dark.status==2 && dark.meanDistance==-1 && dark.validPixels==0,"Dark patch was accepted");
        const auto outside=original.score({100,0,0,0});
        require(outside.status==2 && outside.validPixels==0 && outside.changedFraction==-1,"Outside source was accepted");
        const auto edge=original.score({std::numeric_limits<float>::max(),0,0,0});
        require(edge.status==2,"Huge finite pose unsafe");

        Fixture shifted;
        for(int y=0;y<shifted.th;++y)for(int x=0;x<shifted.tw;++x)
            shifted.source[std::size_t(y+4)*shifted.ss+3*(x+3)]+=45;
        const auto shift=ColorVerifier(shifted.s(),shifted.t()).score({0,0,0,0});
        require(shift.status==0 && shift.changedFraction==1 && shift.meanDistance>.09,"Chroma lost a constant channel-offset color change");
        require(shifted.ncc()>.999999,"Existing RGB NCC should cancel per-channel offsets");

        Fixture exposure;
        for(int y=0;y<exposure.th;++y)for(int x=0;x<exposure.tw;++x)for(int c=0;c<3;++c) {
            auto& value=exposure.source[std::size_t(y+4)*exposure.ss+3*(x+3)+c];value=expose(value,.5);
        }
        const auto gain=ColorVerifier(exposure.s(),exposure.t()).score({0,0,0,0});
        require(gain.status==0 && gain.meanDistance<.004 && gain.changedPixels==0,"Neutral linear exposure was treated as a color defect");

        Fixture rotated;std::fill(rotated.source.begin(),rotated.source.end(),0);
        for(int y=0;y<rotated.th;++y)for(int x=0;x<rotated.tw;++x)for(int c=0;c<3;++c)
            rotated.source[std::size_t(9-x)*rotated.ss+3*(4+y)+c]=rotated.templ[std::size_t(y)*rotated.ts+3*x+c];
        const auto quarter=ColorVerifier(rotated.s(),rotated.t()).score({0,0,90,0});
        require(quarter.status==0 && quarter.meanDistance<1e-12,"Quarter-turn pose convention differs");

        Fixture masked;
        std::vector<std::uint8_t> mask(8*masked.th,0);
        for(int y=0;y<masked.th;++y)for(int x=0;x<masked.tw;++x) {
            if(x<3) mask[y*8+x]=1;
            else masked.source[std::size_t(y+4)*masked.ss+3*(x+3)]+=45;
        }
        const auto limited=ColorVerifier(masked.s(),masked.t(),{},mask.data(),8).score({0,0,0,0});
        require(limited.eligiblePixels==12 && limited.meanDistance==0,"Template mask ignored");
        require(ColorVerifier(masked.s(),masked.t()).score({0,0,0,0}).changedPixels==12,"Unmasked color defect lost");
        std::fill(mask.begin(),mask.end(),0);
        require(ColorVerifier(masked.s(),masked.t(),{},mask.data(),8).score({0,0,0,0}).status==1,"Empty support accepted");
        bool threw=false;
        try{original.score({0,0,std::numeric_limits<float>::quiet_NaN(),0});}catch(const std::invalid_argument&){threw=true;}
        require(threw,"NaN pose accepted");threw=false;
        try{original.score({0,0,0,1000});}catch(const std::invalid_argument&){threw=true;}
        require(threw,"Unrepresentable scale accepted");threw=false;
        try{original.score({0,0,0,-1000});}catch(const std::invalid_argument&){threw=true;}
        require(threw,"Underflowing scale accepted");threw=false;
        auto invalid=masked.s();invalid.stride=1;
        try{ColorVerifier bad(invalid,masked.t());}catch(const std::invalid_argument&){threw=true;}
        require(threw,"Invalid RGB stride accepted");threw=false;
        fpm::research::ColorVerifyConfig config;config.minValidFraction=0;
        try{ColorVerifier bad(masked.s(),masked.t(),config);}catch(const std::invalid_argument&){threw=true;}
        require(threw,"Invalid config accepted");
        std::cout<<"Color geometry, exposure, NCC-offset, mask, ownership and invalid-support controls passed\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
