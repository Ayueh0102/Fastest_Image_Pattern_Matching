// SPDX-License-Identifier: BSD-2-Clause
// Standalone single-map probe. CPU reference is evaluated AFTER the RSS sample.
#include "coarse_internal.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
fpm::Image raster(int w, int h, unsigned seed) {
    fpm::Image image(w,h); std::mt19937 random(seed);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) image.row(y)[x]=uint8_t(random());
    return image;
}
struct Record { double wall; FpmCoarseMetrics metrics; bool exact; };
void printRecord(const Record& r) {
    const auto& m=r.metrics;
    std::cout << "{\"wall_ms\":" << r.wall << ",\"device_ms\":" << m.device_ms
        << ",\"upload_ms\":" << m.upload_ms << ",\"wait_ms\":" << m.wait_ms
        << ",\"normalization_ms\":" << m.normalization_ms << ",\"setup_ms\":" << m.setup_ms
        << ",\"windows\":" << m.windows << ",\"command_buffers\":" << m.command_buffers
        << ",\"integer32_maps\":" << m.integer32_maps << ",\"integer64_maps\":" << m.integer64_maps
        << ",\"gpu_bytes\":" << m.gpu_bytes << ",\"workspace_bytes\":" << m.workspace_bytes
        << ",\"exact_to_first\":" << (r.exact ? "true" : "false") << '}';
}
bool equal(const fpm::detail::ScoreMap& a, const fpm::detail::ScoreMap& b) {
    return a.width==b.width && a.height==b.height && a.values.size()==b.values.size() &&
        std::memcmp(a.values.data(),b.values.data(),a.values.size()*sizeof(double))==0;
}
std::string digest(const fpm::detail::ScoreMap& map) {
    uint64_t hash=14695981039346656037ull;
    const auto* data=reinterpret_cast<const unsigned char*>(map.values.data());
    for (size_t i=0;i<map.values.size()*sizeof(double);++i) { hash^=data[i]; hash*=1099511628211ull; }
    std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash; return out.str();
}
}
int main(int argc,char** argv) {
    try {
        if (argc!=8) throw std::invalid_argument("Usage: probe mode sw sh tw th repeats warm_ms");
        const int mode=std::stoi(argv[1]), sw=std::stoi(argv[2]), sh=std::stoi(argv[3]);
        const int tw=std::stoi(argv[4]), th=std::stoi(argv[5]), repeats=std::stoi(argv[6]);
        const double warmMs=std::stod(argv[7]);
        if (mode<0 || mode>3 || tw<1 || th<1 || sw<tw || sh<th || repeats<1 || repeats>1000 ||
            !std::isfinite(warmMs) || warmMs<0 || warmMs>60000)
            throw std::invalid_argument("Invalid probe dimensions, mode, repetitions or warmup");
        const auto setupStart=Clock::now();
        auto source=raster(sw,sh,5179);
        auto templ=fpm::detail::learnLevel(raster(tw,th,7201));
        auto backend=fpm_coarse_detail::makeMetalBackend();
        backend->setCoarseOptimization(1,mode);
        const double setupMs=elapsed(setupStart);
        backend->resetMetrics();
        const auto firstStart=Clock::now();
        auto firstMap=backend->compute(source.view(),templ);
        const Record first{elapsed(firstStart),backend->metrics(),true};
        auto run=[&]() {
            backend->resetMetrics(); const auto start=Clock::now();
            auto map=backend->compute(source.view(),templ);
            const double wall=elapsed(start);
            // Full bitwise comparison is outside the measured interval.
            return Record{wall,backend->metrics(),equal(map,firstMap)};
        };
        std::vector<Record> warmups,measured;
        const auto warmStart=Clock::now();
        do { warmups.push_back(run()); } while (elapsed(warmStart)<warmMs);
        for (int i=0;i<repeats;++i) measured.push_back(run());
        rusage usage{};
        if (getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("Cannot sample process RSS");
        const uint64_t rss=uint64_t(usage.ru_maxrss); // macOS reports bytes.
        const auto referenceStart=Clock::now();
        const auto reference=fpm::detail::ncc(source.view(),templ,fpm::Backend::Scalar);
        const double referenceMs=elapsed(referenceStart);
        const bool exact=equal(firstMap,reference);
        std::cout << std::setprecision(17) << "{\"mode\":" << mode << ",\"source\":[" << sw << ',' << sh
            << "],\"template\":[" << tw << ',' << th << "],\"setup_ms\":" << setupMs
            << ",\"peak_rss_bytes_before_cpu_reference\":" << rss << ",\"cpu_reference_ms\":" << referenceMs
            << ",\"cpu_reference_bitwise_equal\":" << (exact ? "true" : "false")
            << ",\"map_fnv1a64_native_binary64\":\"" << digest(firstMap) << "\",\"first\":";
        printRecord(first);
        std::cout << ",\"warmups\":[";
        for (size_t i=0;i<warmups.size();++i) { if (i) std::cout << ','; printRecord(warmups[i]); }
        std::cout << "],\"runs\":[";
        for (size_t i=0;i<measured.size();++i) { if (i) std::cout << ','; printRecord(measured[i]); }
        std::cout << "]}\n";
        bool valid=exact;
        for (const auto& r:warmups) valid&=r.exact;
        for (const auto& r:measured) valid&=r.exact;
        return valid ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
