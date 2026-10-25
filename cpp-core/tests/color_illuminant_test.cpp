// SPDX-License-Identifier: BSD-2-Clause
#include "color_illuminant.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using fpm::research::ColorVerifier;
using fpm::research::IlluminantChromaConfig;
using fpm::research::IlluminantChromaVerifier;
using fpm::research::IlluminantSource;
using fpm::research::ReferenceClipPolicy;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
double linear(std::uint8_t byte)
{
    const double encoded=byte/255.;
    return encoded<=.04045?encoded/12.92:std::pow((encoded+.055)/1.055,2.4);
}
std::uint8_t encode(double value)
{
    value=std::clamp(value,0.,1.);
    const double encoded=value<=.0031308?12.92*value:1.055*std::pow(value,1./2.4)-.055;
    return std::uint8_t(std::lround(255*encoded));
}
// 9x6 template: colour columns 0..3, a dark gap, off-white reference columns 5..8.
// The gap keeps colour samples' bilinear neighbours off the reference.
struct Fixture {
    int sw=21,sh=16,tw=9,th=6,left=6,top=5;
    std::size_t ss=66,ts=30,ms=11; // padded rows
    std::vector<std::uint8_t> source=std::vector<std::uint8_t>(ss*sh,30);
    std::vector<std::uint8_t> templ=std::vector<std::uint8_t>(ts*th,0);
    std::vector<std::uint8_t> colour=std::vector<std::uint8_t>(ms*th,0), reference=std::vector<std::uint8_t>(ms*th,0);
    Fixture() {
        for(int y=0;y<th;++y)for(int x=0;x<tw;++x) {
            auto* p=templ.data()+std::size_t(y)*ts+3*x;
            if(x<4) {p[0]=std::uint8_t(150+9*x+4*y);p[1]=std::uint8_t(60+5*x+3*y);p[2]=std::uint8_t(50+3*x+6*y);colour[y*ms+x]=1;}
            else if(x==4) {p[0]=p[1]=p[2]=40;}
            else {p[0]=236;p[1]=234;p[2]=226;reference[y*ms+x]=1;}
        }
        paste({1,1,1});
    }
    std::uint8_t* at(int x,int y){return source.data()+std::size_t(y+top)*ss+3*(x+left);}
    // Template under a diagonal linear-light gain, re-encoded to sRGB bytes.
    void paste(std::array<double,3> gain) {
        for(int y=0;y<th;++y)for(int x=0;x<tw;++x)for(int c=0;c<3;++c) {
            const auto byte=templ[std::size_t(y)*ts+3*x+c];
            at(x,y)[c]=gain[c]==1?byte:encode(linear(byte)*gain[c]);
        }
    }
    fpm::research::Rgb8View s()const{return{source.data(),sw,sh,ss};}
    fpm::research::Rgb8View t()const{return{templ.data(),tw,th,ts};}
    IlluminantChromaVerifier object(IlluminantChromaConfig config={})const
    {return IlluminantChromaVerifier(s(),t(),colour.data(),ms,reference.data(),ms,config);}
    double masked(std::array<float,4> pose)const
    {return ColorVerifier(s(),t(),{},colour.data(),ms).score(pose).meanDistance;}
};
template<class Function> bool throws(Function function)
{try{function();}catch(const std::invalid_argument&){return true;}return false;}
}

int main()
{
    try {
        Fixture exact;
        const auto same=exact.object().score({0,0,0,0});
        require(same.status==0 && same.validPixels==24 && same.referenceSamples==24 && same.referenceFraction==1,"Exact crop support differs");
        require(same.gain==std::array<double,3>{1,1,1} && same.meanDistance==0 && same.uncorrectedMeanDistance==0 &&
            same.regionDistance==0 && same.uncorrectedRegionDistance==0,"Exact crop needs unit gain");
        std::fill(exact.source.begin(),exact.source.end(),0);
        Fixture copy;const auto owned=copy.object();
        std::fill(copy.source.begin(),copy.source.end(),0);
        require(owned.score({0,0,0,0}).meanDistance==0,"Constructor borrowed source memory");

        // Shared geometry: without saturation handling the uncorrected distance is
        // ColorVerifier's masked chroma, bit for bit, on an off-grid noisy pose.
        Fixture noisy;
        for(int y=0;y<noisy.th;++y)for(int x=0;x<noisy.tw;++x)noisy.at(x,y)[(x+y)%3]+=std::uint8_t(7*((x*y)%3));
        IlluminantChromaConfig none;none.clipPolicy=ReferenceClipPolicy::None;none.chroma.minValidFraction=.5;
        const std::array<float,4> offgrid{.3f,-.2f,1.7f,.01f};
        const auto shared=noisy.object(none).score(offgrid);
        fpm::research::ColorVerifyConfig loose;loose.minValidFraction=.5;
        const auto reference=ColorVerifier(noisy.s(),noisy.t(),loose,noisy.colour.data(),noisy.ms).score(offgrid);
        require(shared.validPixels==reference.validPixels && shared.uncorrectedMeanDistance==reference.meanDistance,
            "Uncorrected distance is not the existing masked chroma");

        // White-balance shift: the diagonal gain removes it, masked chroma does not.
        Fixture warm;warm.paste({1.18,1,.78});
        const auto balanced=warm.object().score({0,0,0,0});
        require(balanced.status==0 && balanced.meanDistance<.002 && balanced.regionDistance<.001,"von Kries correction did not remove a white-balance shift");
        require(balanced.uncorrectedRegionDistance>.03,"Uncorrected region distance missed the white-balance shift");
        require(std::abs(balanced.gain[0]*1.18-1)<.01 && std::abs(balanced.gain[1]-1)<.01 && std::abs(balanced.gain[2]*.78-1)<.01,
            "Gain is not template/source reference");
        require(warm.masked({0,0,0,0})>.03 && balanced.uncorrectedMeanDistance>.03,"White-balance control is too weak");
        Fixture defect=warm;
        for(int y=0;y<defect.th;++y)for(int x=0;x<4;++x)defect.at(x,y)[1]=std::uint8_t(defect.at(x,y)[1]+40);
        require(defect.object().score({0,0,0,0}).meanDistance>.05 && defect.object().score({0,0,0,0}).regionDistance>.05,
            "Correction hid a real colour change");
        // Zero-mean pixel noise: the per-pixel mean keeps a floor, the region mean does not.
        Fixture grain;
        for(int y=0;y<grain.th;++y)for(int x=0;x<4;++x){auto* p=grain.at(x,y);const int sign=(x+y)%2?6:-6;p[1]=std::uint8_t(p[1]+sign);}
        const auto grained=grain.object().score({0,0,0,0});
        require(grained.meanDistance>4*grained.regionDistance,"Region distance should average zero-mean noise");

        // Same scene at a quarter turn: the reference follows the pose.
        Fixture turned;std::fill(turned.source.begin(),turned.source.end(),30);
        const Fixture shifted=warm;
        for(int y=0;y<turned.th;++y)for(int x=0;x<turned.tw;++x)for(int c=0;c<3;++c)
            turned.source[std::size_t(12-x)*turned.ss+3*(8+y)+c]=shifted.source[std::size_t(y+5)*shifted.ss+3*(x+6)+c];
        const auto quarter=turned.object().score({.5f,.5f,90,0});
        require(quarter.status==0 && quarter.referenceSamples==24 && quarter.gain==balanced.gain && quarter.meanDistance==balanced.meanDistance,
            "Rotated reference not mapped through the pose");

        // Saturation. A fully clipped reference abstains; the uncorrected distance stays.
        Fixture bright;
        for(int y=0;y<bright.th;++y)for(int x=5;x<bright.tw;++x)for(int c=0;c<3;++c)bright.at(x,y)[c]=255;
        const auto clipped=bright.object().score({0,0,0,0});
        require(clipped.status==3 && clipped.meanDistance==-1 && clipped.changedFraction==-1 && clipped.regionDistance==-1 &&
            clipped.uncorrectedMeanDistance==0 && clipped.uncorrectedRegionDistance==0,
            "Clipped reference was not an abstention");
        require(clipped.gain==std::array<double,3>{0,0,0} && clipped.referenceClipFraction[0]==1,"Clipped reference diagnostics");
        require(bright.object(none).score({0,0,0,0}).status==0,"Policy none must not check saturation");
        // Red reference 9 x 244, 5 x 250, 10 x 255: 10/24 clipped keeps the rank median
        // (12th/13th of 24) at 250; dropping clipped samples biases it to 244.
        Fixture partial;
        for(int k=0;k<24;++k)partial.at(5+k%4,k/4)[0]=k<9?244:k<14?250:255;
        const auto ranked=partial.object().score({0,0,0,0});
        require(ranked.status==0 && ranked.referenceClipFraction[0]==10/24.,"Partial clipping fraction");
        require(ranked.gain[0]==linear(236)/linear(250),"Rank median must keep clipped bytes in the order statistic");
        IlluminantChromaConfig drop;drop.clipPolicy=ReferenceClipPolicy::Drop;
        const auto dropped=partial.object(drop).score({0,0,0,0});
        require(dropped.status==0 && dropped.referenceSamples==14 && dropped.gain[0]==linear(236)/linear(244),"Drop ablation differs");
        Fixture half;
        for(int k=0;k<24;++k)half.at(5+k%4,k/4)[2]=k<10?255:226;
        require(half.object().score({0,0,0,0}).status==0,"10/24 clipped is within .45");
        half.at(5+10%4,10/4)[2]=255;
        require(half.object().score({0,0,0,0}).status==3,"11/24 clipped accepted by rank policy");
        IlluminantChromaConfig level;level.saturationLevel=250;
        Fixture edge;
        for(int y=0;y<edge.th;++y)for(int x=5;x<edge.tw;++x)edge.at(x,y)[0]=250;
        require(edge.object(level).score({0,0,0,0}).status==3,"Byte equal to the saturation level is clipped");
        level.saturationLevel=251;
        require(edge.object(level).score({0,0,0,0}).status==0,"Byte below the saturation level is not clipped");
        // An even count averages the two middle order statistics.
        Fixture even;
        for(int k=0;k<24;++k)even.at(5+k%4,k/4)[1]=k<12?230:238;
        require(even.object().score({0,0,0,0}).gain[1]==linear(234)/((linear(230)+linear(238))*.5),"Even-count median");
        // Nearest source pixel, not floor: one reference column, the next source column differs.
        Fixture nearest;
        std::fill(nearest.reference.begin(),nearest.reference.end(),0);
        for(int y=0;y<nearest.th;++y){nearest.reference[y*nearest.ms+5]=1;nearest.at(6,y)[0]=200;}
        require(nearest.object().score({.6f,0,0,0}).gain[0]==linear(236)/linear(200),"Reference must use the nearest source pixel");
        require(nearest.object().score({.4f,0,0,0}).gain[0]==1,"Reference rounding below one half");
        // Outliers inside the reference mask: the median ignores them, a mean would not.
        Fixture marks;
        for(int k=0;k<5;++k)for(int c=0;c<3;++c)marks.at(5+k%4,k/4)[c]=52;
        const auto marked=marks.object().score({0,0,0,0});
        require(marked.status==0 && marked.gain==std::array<double,3>{1,1,1} && marked.meanDistance==0,"Reference estimate is not a median");

        // Clipped colour samples are missing support under rank/drop only.
        Fixture hot;
        for(int y=0;y<hot.th;++y)hot.at(0,y)[0]=255;
        const auto hotResult=hot.object().score({0,0,0,0});
        require(hotResult.clippedPixels>0 && hotResult.validPixels+hotResult.clippedPixels==24 && hotResult.status==2,
            "Clipped colour samples were not excluded");
        require(hot.object(none).score({0,0,0,0}).clippedPixels==0,"Policy none excluded clipped colour samples");

        // Reference mostly outside the image while the colour region is inside.
        Fixture outside;
        const auto out=outside.object().score({7.5f,0,0,0});  // reference columns map to 18.5..21.5
        require(out.referenceSamples==2*6 && out.status==0,"Half-inside reference should remain usable");
        const auto gone=outside.object().score({8.5f,0,0,0}); // 1 of 4 columns inside
        require(gone.referenceFraction<.5 && gone.status==3 && gone.uncorrectedMeanDistance>=0,"Reference fraction not enforced");
        const auto missing=outside.object().score({100,0,0,0});
        require(missing.status==2 && missing.uncorrectedMeanDistance==-1 && missing.uncorrectedRegionDistance==-1 && missing.regionDistance==-1,
            "Missing colour support must be status 2");

        // Global source rectangle (calibration card) under the same white-balance shift.
        Fixture card;card.paste({1.18,1,.78});
        for(int y=0;y<3;++y)for(int x=0;x<4;++x){auto* p=card.source.data()+std::size_t(y)*card.ss+3*x;
            p[0]=encode(.7*1.18);p[1]=encode(.7);p[2]=encode(.7*.78);}
        IlluminantChromaConfig global;global.source=IlluminantSource::SourceRect;global.referenceRect={0,0,4,3};
        const IlluminantChromaVerifier cardVerifier(card.s(),card.t(),card.colour.data(),card.ms,nullptr,0,global);
        const auto cardResult=cardVerifier.score({0,0,0,0});
        require(cardResult.status==0 && cardResult.referenceEligible==12 && cardResult.meanDistance<.002,"Card correction failed");
        require(cardVerifier.templateReference()==std::array<double,3>{1,1,1},"Default template reference is neutral");
        require(cardVerifier.score({3,2,0,0}).gain==cardResult.gain,"Global reference must not depend on the pose");

        // Invalid inputs.
        require(throws([&]{IlluminantChromaVerifier(exact.s(),exact.t(),exact.colour.data(),exact.ms,nullptr,0);}),"Missing reference mask");
        require(throws([&]{IlluminantChromaVerifier(card.s(),card.t(),nullptr,0,card.reference.data(),card.ms,global);}),"Rect with mask");
        global.referenceRect={0,0,22,3};
        require(throws([&]{IlluminantChromaVerifier(card.s(),card.t(),nullptr,0,nullptr,0,global);}),"Rect outside source");
        global.referenceRect={2,0,2,3};
        require(throws([&]{IlluminantChromaVerifier(card.s(),card.t(),nullptr,0,nullptr,0,global);}),"Empty rect");
        global.referenceRect={0,0,4,3};global.templateReference={1,0,1};
        require(throws([&]{IlluminantChromaVerifier(card.s(),card.t(),nullptr,0,nullptr,0,global);}),"Zero template reference");
        Fixture white;
        for(int y=0;y<white.th;++y)for(int x=5;x<white.tw;++x)for(int c=0;c<3;++c)white.templ[std::size_t(y)*white.ts+3*x+c]=255;
        require(throws([&]{white.object();}),"Clipped template reference accepted");
        IlluminantChromaConfig bad;bad.saturationLevel=0;
        require(throws([&]{exact.object(bad);}),"Saturation level 0 accepted");
        bad=IlluminantChromaConfig{};bad.maxReferenceClipFraction=1;
        require(throws([&]{exact.object(bad);}),"Clip fraction 1 accepted");
        bad=IlluminantChromaConfig{};bad.minReferenceFraction=0;
        require(throws([&]{exact.object(bad);}),"Reference fraction 0 accepted");
        require(throws([&]{exact.object().score({0,0,std::numeric_limits<float>::quiet_NaN(),0});}),"NaN pose accepted");
        std::vector<std::uint8_t> empty(exact.ms*exact.th,0);
        require(IlluminantChromaVerifier(exact.s(),exact.t(),empty.data(),exact.ms,exact.reference.data(),exact.ms).score({0,0,0,0}).status==1,
            "Empty colour support accepted");
        std::cout<<"Illuminant chroma gain, geometry, saturation, median, support and card controls passed\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
