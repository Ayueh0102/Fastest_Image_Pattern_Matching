// SPDX-License-Identifier: BSD-2-Clause
// Independent finite-difference check of Metal's normalized-image Jacobian.
#include "pose_internal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fpm_exp;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float pixel(const Data& d, int x, int y, int c)
{
    if (x < 0 || y < 0 || x >= d.sw || y >= d.sh) return -d.sourceMean[c];
    return d.source[(size_t(y)*d.sw+x)*d.channels+c];
}
std::vector<double> samples(const Data& d, const PackedPose& p)
{
    std::vector<double> result(size_t(d.tw)*d.th*d.channels);
    for (int y = 0; y < d.th; ++y) for (int x = 0; x < d.tw; ++x) {
        const float u = x-d.tw*.5f, v = d.th*.5f-y;
        const float sx = d.sw*.5f+p.tx+p.cs*u-p.sn*v;
        const float sy = d.sh*.5f-p.ty-(p.sn*u+p.cs*v);
        const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
        const float dx = sx-ix, dy = sy-iy;
        for (int c = 0; c < d.channels; ++c) {
            const float a = pixel(d,ix,iy,c), b = pixel(d,ix+1,iy,c);
            const float e = pixel(d,ix,iy+1,c), f = pixel(d,ix+1,iy+1,c);
            result[(size_t(y)*d.tw+x)*d.channels+c] = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
        }
    }
    return result;
}
void normalize(std::vector<double>& values, int channels)
{
    const double n = double(values.size())/channels;
    for (int c = 0; c < channels; ++c) {
        double sum = 0;
        for (size_t i = c; i < values.size(); i += channels) sum += values[i];
        for (size_t i = c; i < values.size(); i += channels) values[i] -= sum/n;
    }
    double norm = std::sqrt(std::inner_product(values.begin(),values.end(),values.begin(),0.0));
    require(norm > 0, "invalid finite-difference fixture");
    for (auto& v : values) v /= norm;
}
Data fixture(int channels, float contrast)
{
    Data d{}; d.sw=112; d.sh=104; d.tw=43; d.th=39; d.channels=channels;
    d.source.resize(size_t(d.sw)*d.sh*channels);
    for (int c=0;c<channels;++c) {
        double sum=0;
        for (int y=0;y<d.sh;++y) for (int x=0;x<d.sw;++x) {
            float v=float(.4+contrast*(std::sin(.13*x+.057*y+.6*c)+.7*std::cos(.041*x-.18*y+.8*c)));
            d.source[(size_t(y)*d.sw+x)*channels+c]=v; sum+=v;
        }
        d.sourceMean[c]=float(sum/(d.sw*d.sh));
        for (size_t i=c;i<d.source.size();i+=channels) d.source[i]-=d.sourceMean[c];
    }
    auto raw=samples(d,{1.12f,-.73f,.977f,.194f});
    d.templ.resize(raw.size());
    const double n=double(d.tw)*d.th;
    for (int c=0;c<channels;++c) {
        double sum=0, square=0;
        for (size_t i=c;i<raw.size();i+=channels) sum+=raw[i];
        for (size_t i=c;i<raw.size();i+=channels) {
            d.templ[i]=float(raw[i]-sum/n);
            d.templateSum[c]+=d.templ[i]; square+=double(d.templ[i])*d.templ[i];
        }
        d.templateEnergy+=square-d.templateSum[c]*d.templateSum[c]/n;
    }
    return d;
}
void check(int channels, float contrast)
{
    auto d=fixture(channels,contrast);
    auto backend=makeMetal(d);
    const PackedPose p{.317f,-.243f,.981f,.187f};
    FpmPoseRefineConfig config{30,1,1,12,15,.26};
    std::vector<Linearization> result;
    FpmPoseMetrics metrics{};
    require(backend->linearize({p},config,result,metrics),"Metal derivative path unavailable");
    require(result.size()==1,"incorrect derivative result count");
    auto z=samples(d,p); normalize(z,channels);
    std::vector<double> t(d.templ.begin(),d.templ.end()); normalize(t,channels);
    const double score=std::inner_product(z.begin(),z.end(),t.begin(),0.0);
    require(std::abs(result[0].score-score)<3e-5,"GPU normalized score mismatch");
    std::vector<double> j[4];
    const double steps[4]={.003,.003,.00015,.00015};
    for (int k=0;k<4;++k) {
        PackedPose plus=p,minus=p;
        if(k==0) {plus.tx+=float(steps[k]);minus.tx-=float(steps[k]);}
        if(k==1) {plus.ty+=float(steps[k]);minus.ty-=float(steps[k]);}
        if(k==2) {
            const double c=std::cos(steps[k]),s=std::sin(steps[k]);
            plus.cs=float(c*p.cs-s*p.sn);plus.sn=float(s*p.cs+c*p.sn);
            minus.cs=float(c*p.cs+s*p.sn);minus.sn=float(-s*p.cs+c*p.sn);
        }
        if(k==3) {plus.cs*=float(std::exp(steps[k]));plus.sn*=float(std::exp(steps[k]));
                  minus.cs*=float(std::exp(-steps[k]));minus.sn*=float(std::exp(-steps[k]));}
        auto a=samples(d,plus),b=samples(d,minus);normalize(a,channels);normalize(b,channels);
        j[k].resize(z.size());
        for(size_t i=0;i<z.size();++i) j[k][i]=(a[i]-b[i])/(2*steps[k]);
        double g=0;
        for(size_t i=0;i<z.size();++i) g+=j[k][i]*(t[i]-z[i]);
        require(std::abs(result[0].g[k]-g)<.015*std::max(.01,std::abs(g)),"GPU gradient finite-difference mismatch");
    }
    for(int k=0;k<4;++k) for(int l=0;l<4;++l) {
        const double h=std::inner_product(j[k].begin(),j[k].end(),j[l].begin(),0.0);
        const double tolerance=.015*std::sqrt(result[0].h[k][k]*result[0].h[l][l])+1e-6;
        require(std::isfinite(result[0].h[k][l]) && std::abs(result[0].h[k][l]-h)<tolerance,
                "GPU Hessian finite-difference mismatch");
    }
    config.optimize_angle=config.optimize_scale=0;
    require(backend->linearize({p,p,p},config,result,metrics),"second derivative dispatch failed");
    for(const auto& a:result) for(int k=2;k<4;++k) {
        require(a.g[k]==0,"frozen derivative gradient changed");
        for(int l=0;l<4;++l) require(a.h[k][l]==0 && a.h[l][k]==0,"frozen derivative Hessian changed");
    }
}
}
int main()
{
    try {
        for(int channels:{1,3}) for(float contrast:{.08f,.00002f}) check(channels,contrast);
        std::cout<<"Metal normalized NCC Jacobian/Hessian finite differences passed\n";
        return 0;
    } catch(const std::exception& e) {
        if(std::string(e.what())=="No Metal GPU available") return 77;
        std::cerr<<e.what()<<'\n';return 1;
    }
}
