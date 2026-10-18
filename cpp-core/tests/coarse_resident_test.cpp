// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
fpm::Image raster(int w, int h, unsigned seed, int kind=0) {
    fpm::Image image(w,h); std::mt19937 rng(seed);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        image.row(y)[x]=kind==1 ? uint8_t(254+rng()%2) : kind==2 ? 127 : uint8_t(rng());
    return image;
}
template<class F> void rejects(F fn) {
    bool failed=false; try {fn();} catch(const std::invalid_argument&) {failed=true;}
    require(failed,"invalid resident option accepted");
}
}
int main() {
    try {
        auto gpu=fpm_coarse_detail::makeMetalBackend();
        FpmCoarseExecutionConfigV1 execution{};fpm_coarse_execution_default_config(&execution);
        execution.batch_angles=4;gpu->setExecution(execution);gpu->setCoarseOptimization(2,0);
        require(gpu->residentCoarsePeaks(),"resident mode not enabled");
        rejects([&]{gpu->setCoarseOverlap(true);});
        gpu->setCoarseOptimization(1,0);gpu->setCoarseOverlap(true);
        rejects([&]{gpu->setCoarseOptimization(2,0);});
        require(!gpu->residentCoarsePeaks(),"failed setter changed selected mode");
        gpu->setCoarseOverlap(false);gpu->setCoarseOptimization(2,0);
        size_t maps=0, changedOrders=0;double maxError=0;
        for(int mode:{0,2}) {
            gpu->setCoarseOptimization(2,mode);
            for(int kind:{0,1,2}) for(auto size:{std::pair<int,int>{1,1},{3,5},{17,13},{258,258}}) {
                auto templ=fpm::detail::learnLevel(raster(size.first,size.second,841,kind));
                std::vector<fpm::Image> owners;
                for(int k=0;k<5;++k) owners.push_back(raster(size.first+8+k,size.second+7+k,901+k,kind));
                std::vector<fpm::ImageView> views;
                for(auto& owner:owners) {auto v=owner.view();v.width-=2;views.push_back(v);}
                for(double overlap:{0.,.2,1.}) {
                    gpu->resetMetrics();
                    const auto actual=gpu->computePeakBatch(views,templ,32,overlap,-1.);
                    require(actual.size()==views.size(),"map count mismatch");
                    for(size_t k=0;k<views.size();++k) {
                        auto remaining=fpm::detail::ncc(views[k],templ,fpm::Backend::Scalar);
                        const auto reference=fpm::detail::selectPeaksFast(remaining,size.first,size.second,32,overlap,-1.,false);
                        require(actual[k].size()==reference.size(),"peak count mismatch");
                        for(size_t j=0;j<reference.size();++j) {
                            const auto p=actual[k][j];
                            if(p.x!=reference[j].x || p.y!=reference[j].y) ++changedOrders;
                            require(p.x>=0 && p.y>=0 && p.x<remaining.width && p.y<remaining.height,"peak outside map");
                            const double expected=remaining.at(p.x,p.y);
                            maxError=std::max(maxError,std::abs(p.score-expected));
                            require(std::abs(p.score-expected)<3e-6,"FP32 NCC score error exceeds tolerance or duplicate suppressed peak");
                            require(fpm::detail::maximum(remaining).score-expected<3e-6,"greedy winner exceeds numerical error tolerance");
                            const double rx=size.first*(1-overlap),ry=size.second*(1-overlap);
                            const int left=int(p.x-rx),top=int(p.y-ry);
                            const int x0=std::max(0,std::min(p.x,left)),y0=std::max(0,std::min(p.y,top));
                            const int x1=std::min(remaining.width,std::max(p.x+1,left+int(2*rx)));
                            const int y1=std::min(remaining.height,std::max(p.y+1,top+int(2*ry)));
                            for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) remaining.at(x,y)=-INFINITY;
                        }
                        ++maps;
                    }
                    require(gpu->metrics().normalization_ms==0,"resident route normalized dense maps on CPU");
                    require(gpu->metrics().scoremap_peak_bytes==0,"resident route materialized CPU score map");
                    require(gpu->metrics().command_buffers==2,"five maps should use two batches without iterative waits");
                }
            }
        }
        auto source=raster(1537,1001,811);
        auto templ=fpm::detail::learnLevel(raster(3,3,92));
        gpu->resetMetrics();
        const auto actual=gpu->computePeakBatch({source.view()},templ,8,.2,.9).front();
        const auto reference=fpm::detail::selectPeaksFast(fpm::detail::ncc(source.view(),templ,fpm::Backend::Scalar),3,3,8,.2,.9,false);
        require(actual.size()==reference.size(),"chunked count mismatch");
        for(size_t i=0;i<actual.size();++i)
            require(actual[i].x==reference[i].x && actual[i].y==reference[i].y,"chunked position mismatch");
        require(gpu->metrics().command_buffers==1,"chunked resident route waited between chunks");
        // Many identical peaks exercise deterministic ties and overlap=1.
        auto constant=fpm::detail::learnLevel(raster(1,1,0,2));
        const auto ties=gpu->computePeakBatch({source.view()},constant,1005,1,1).front();
        require(ties.size()==1005,"candidate budget was truncated");
        for(size_t i=0;i<ties.size();++i) require(ties[i].x==int(i) && ties[i].y==0 && ties[i].score==1,"tie ordering changed");
        const auto empty=gpu->computePeakBatch({source.view()},templ,8,.2,1).front();
        require(empty.empty(),"threshold=1 accepted a random imperfect patch");
        rejects([&]{gpu->computePeakBatch({source.view()},templ,0,.2,.5);});
        rejects([&]{gpu->computePeakBatch({source.view()},templ,1,NAN,.5);});
        rejects([&]{gpu->computePeakBatch({source.view()},templ,1,.2,NAN);});
        gpu->setCoarseOptimization(1,0);
        require(!gpu->residentCoarsePeaks(),"mode failed to reset");
        std::cout<<"Resident coarse: "<<maps<<" maps; max score error "<<maxError
            <<"; "<<changedOrders<<" ordered peak entries differ from double reference (reported, not bit-exact)"
            <<"; strided/constant/low-contrast/uint64/chunk/tie/threshold/mode checks passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
