// SPDX-License-Identifier: BSD-2-Clause
// Isolated fine batch, including guarded warp, host repair and normalization.
// CPU reference and full-map comparisons are outside timed intervals.
#include "coarse_internal.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/resource.h>

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
using Maps=std::vector<fpm::detail::ScoreMap>;
bool equal(const Maps& a,const Maps& b) {
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i) if(a[i].width!=b[i].width || a[i].height!=b[i].height ||
        a[i].values.size()!=b[i].values.size() || std::memcmp(a[i].values.data(),b[i].values.data(),a[i].values.size()*sizeof(double))) return false;
    return true;
}
std::string digest(const Maps& maps) {
    uint64_t hash=14695981039346656037ull;
    for(const auto& map:maps) {
        const auto* p=reinterpret_cast<const unsigned char*>(map.values.data());
        for(size_t i=0;i<map.values.size()*sizeof(double);++i) { hash^=p[i];hash*=1099511628211ull; }
    }
    std::ostringstream out;out<<std::hex<<hash;return out.str();
}
fpm::Image raster(int w,int h,unsigned seed) {
    fpm::Image image(w,h);std::mt19937 random(seed);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) image.row(y)[x]=uint8_t(random());
    return image;
}
struct Record { double wall;FpmFineMetricsV1 fine;FpmFineReuseMetricsV1 reuse;bool exact; };
void print(const Record& r) {
    const auto& m=r.fine;const auto& n=r.reuse;
    std::cout<<"{\"wall_ms\":"<<r.wall<<",\"rotation_ms\":"<<m.rotation_ms<<",\"repair_ms\":"<<m.repair_ms
        <<",\"ncc_ms\":"<<m.ncc_ms<<",\"normalization_ms\":"<<m.normalization_ms<<",\"wait_ms\":"<<m.wait_ms
        <<",\"device_ms\":"<<m.device_ms<<",\"stats_device_ms\":"<<n.stats_device_ms<<",\"upload_ms\":"<<m.upload_ms
        <<",\"command_buffers\":"<<m.command_buffers<<",\"maps\":"<<m.maps<<",\"roi_pixels\":"<<m.roi_pixels
        <<",\"repair_pixels\":"<<m.repair_pixels<<",\"gpu_bytes\":"<<m.retained_gpu_bytes
        <<",\"statistics_bytes\":"<<m.statistics_buffer_bytes<<",\"row_bytes\":"<<n.row_buffer_bytes
        <<",\"row_dispatches\":"<<n.row_dispatches<<",\"stats_dispatches\":"<<n.statistics_dispatches
        <<",\"peak_batch\":"<<m.peak_batch_maps<<",\"exact_to_first\":"<<(r.exact?"true":"false")<<'}';
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=8) throw std::invalid_argument("Usage: fine_probe mode tw th count batch repeats warm_ms");
        const int mode=std::stoi(argv[1]),tw=std::stoi(argv[2]),th=std::stoi(argv[3]),count=std::stoi(argv[4]),batch=std::stoi(argv[5]),repeats=std::stoi(argv[6]);
        const double warm=std::stod(argv[7]);
        if(mode<0 || mode>3 || tw<1 || th<1 || tw>4096 || th>4096 || count<1 || count>4096 || batch<1 || batch>1024 || repeats<1 || repeats>1000 || !std::isfinite(warm) || warm<0 || warm>60000)
            throw std::invalid_argument("Invalid probe settings");
        const auto setup=Clock::now();
        auto source=raster(std::max(640,tw+20),std::max(512,th+20),731);
        const auto templ=fpm::detail::learnLevel(raster(tw,th,881));
        std::vector<fpm::detail::FineRequest> poses;
        for(int i=0;i<count;++i) poses.push_back({{double((i*23+17)%source.width())+.35,double((i*37+29)%source.height())+.65},double((i*13)%47)-23.});
        auto backend=fpm_coarse_detail::makeMetalBackend();backend->setFineBatchSize(batch);backend->setFineNccMode(mode);
        const double setupMs=ms(setup);
        backend->resetMetrics();auto t=Clock::now();auto firstMaps=backend->computeFineBatch(source.view(),templ,poses,nullptr);
        Record first{ms(t),backend->fineMetrics(),backend->fineReuseMetrics(),true};
        auto run=[&](){backend->resetMetrics();const auto start=Clock::now();auto maps=backend->computeFineBatch(source.view(),templ,poses,nullptr);
            const double wall=ms(start);return Record{wall,backend->fineMetrics(),backend->fineReuseMetrics(),equal(maps,firstMaps)};};
        // Fixed-size warmup bookkeeping: a faster mode must not accumulate
        // more telemetry and inflate its process RSS measurement.
        std::vector<Record> runs;runs.reserve(repeats);
        size_t warmupCalls=0;bool warmupsExact=true;const auto start=Clock::now();
        do {const auto r=run();warmupsExact &= r.exact;++warmupCalls;} while(ms(start)<warm);
        const double warmupMs=ms(start);
        for(int i=0;i<repeats;++i) runs.push_back(run());
        rusage usage{};if(getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("RSS unavailable");
        // Keep first maps in RSS for both modes; reference is built only now.
        bool cpuExact=true;for(size_t i=0;i<poses.size();++i) {
            const auto roi=fpm::detail::rotatedRoi(source.view(),tw,th,poses[i].anchor,poses[i].angle);
            const auto expected=fpm::detail::ncc(roi.view(),templ,fpm::Backend::Scalar);
            cpuExact &= firstMaps[i].values.size()==expected.values.size() &&
                !std::memcmp(firstMaps[i].values.data(),expected.values.data(),expected.values.size()*sizeof(double));
        }
        std::cout<<std::setprecision(17)<<"{\"mode\":"<<mode<<",\"template\":["<<tw<<','<<th<<"],\"count\":"<<count<<",\"batch\":"<<batch
            <<",\"setup_ms\":"<<setupMs<<",\"peak_rss_bytes_before_reference\":"<<usage.ru_maxrss
            <<",\"cpu_reference_bitwise_equal\":"<<(cpuExact?"true":"false")<<",\"maps_fnv1a64\":\""<<digest(firstMaps)<<"\",\"first\":";
        print(first);std::cout<<",\"warmup_calls\":"<<warmupCalls<<",\"warmup_elapsed_ms\":"<<warmupMs
            <<",\"warmups_exact\":"<<(warmupsExact?"true":"false")<<",\"runs\":[";
        for(size_t i=0;i<runs.size();++i) {if(i)std::cout<<',';print(runs[i]);}std::cout<<"]}\n";
        bool valid=cpuExact && warmupsExact;for(const auto& r:runs)valid&=r.exact;
        return valid?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
