// SPDX-License-Identifier: BSD-2-Clause
#include "sparse_orientation.hpp"
#ifdef FPM_SPARSE_METAL
#include "sparse_orientation_metal.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace fpm::research {
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr int orientationBins = 8;
constexpr int gradientAxisValues = 511;
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
void validateImage(ImageView image) {
    if (!image.data || image.width <= 0 || image.height <= 0 ||
        image.stride < std::size_t(image.width) ||
        image.stride > (std::size_t(PTRDIFF_MAX)-image.width)/std::size_t(image.height))
        throw std::invalid_argument("sparse orientation expects a nonempty gray8 image with a valid stride");
    if (std::size_t(image.width) > std::size_t(PTRDIFF_MAX)/std::size_t(image.height)/16)
        throw std::invalid_argument("sparse orientation image is too large");
}
void validateSettings(const SparseOrientationSettings& s) {
    if (s.featureCount < 1 || s.featureCount > 4096 || s.positionStride < 1 ||
        s.positionStride > 64 || s.spreadRadius < 0 || s.spreadRadius > 16 ||
        s.candidateCap < 1 || s.candidateCap > 1000000 || s.cpuThreads < 1 ||
        s.cpuThreads > 64 || s.responseMode < 0 || s.responseMode > 1 ||
        s.searchBackend < 0 || s.searchBackend > 1 ||
        s.cpuGradientMode < 0 || s.cpuGradientMode > 1 ||
        s.metalSubmissionMode < 0 || s.metalSubmissionMode > 1 ||
        (s.searchBackend == 1 && s.responseMode != 1) || !std::isfinite(s.angleStepDegrees) ||
        s.angleStepDegrees < .1 || s.angleStepDegrees > 180 ||
        !std::isfinite(s.minGradient) || s.minGradient <= 0 ||
        !std::isfinite(s.minScore) || s.minScore <= 0 || s.minScore > 1)
        throw std::invalid_argument("invalid sparse orientation settings");
}
double normalizeAngle(double a) {
    a = std::remainder(a,360.);
    return a <= -180 ? a+360 : a;
}
int orientationBin(double imageRadians) {
    // Canonicalize the unsigned angle before rounding. Rounding signed angles
    // away from zero would choose opposite neighbors at negative half-bin ties.
    imageRadians=std::fmod(imageRadians,pi);
    if (imageRadians<0) imageRadians+=pi;
    // A tiny bin-unit tolerance gives mathematically identical half-bin ties
    // the same direction after the floating-point modulo operation (angle+pi
    // can otherwise land a few ulps below its equivalent positive angle).
    return int(std::floor(imageRadians*orientationBins/pi+.5+1e-12))%orientationBins;
}
std::vector<std::uint8_t> makeGradientLookup(double minGradient) {
    // Gray8 central differences have only 511^2 possible signed input pairs.
    // Share this exact double-precision threshold/angle classification between
    // CPU and Metal; no approximation to atan2 or half-bin rounding is needed.
    std::vector<std::uint8_t> result(gradientAxisValues*gradientAxisValues);
    for (int dy=-255;dy<=255;++dy) for (int dx=-255;dx<=255;++dx) {
        const double gx=dx*.5,gy=dy*.5;
        if (gx*gx+gy*gy>=minGradient*minGradient)
            result[std::size_t(dy+255)*gradientAxisValues+dx+255]=
                std::uint8_t(1u<<orientationBin(std::atan2(gy,gx)));
    }
    return result;
}
std::vector<double> makeAngles(const MatchConfig& c,double step) {
    if (!std::isfinite(c.toleranceAngle) || c.toleranceAngle < 0 || c.toleranceAngle > 180)
        throw std::invalid_argument("sparse orientation toleranceAngle must be in [0,180]");
    std::vector<AngleRange> ranges = c.angleRanges;
    if (ranges.empty()) ranges.push_back({-c.toleranceAngle,c.toleranceAngle});
    std::vector<double> angles;
    for (auto r : ranges) {
        if (!std::isfinite(r.min) || !std::isfinite(r.max) || r.min < -180 ||
            r.max > 180 || r.min > r.max)
            throw std::invalid_argument("sparse orientation angle ranges must lie in [-180,180]");
        const int intervals = int(std::ceil((r.max-r.min)/step));
        if (intervals == 0) angles.push_back(normalizeAngle(r.min));
        else for (int k=0;k<=intervals;++k)
            angles.push_back(normalizeAngle(r.min+(r.max-r.min)*k/intervals));
        if (r.min <= 0 && r.max >= 0) angles.push_back(0);
    }
    std::sort(angles.begin(),angles.end());
    angles.erase(std::unique(angles.begin(),angles.end(),[](double a,double b) {
        return std::abs(a-b)<1e-9;
    }),angles.end());
    return angles;
}
struct Feature { double x,y,direction,magnitude; };
struct RotatedFeature { std::ptrdiff_t offset; int bin; };
struct Candidate { int x,y; double angle,score; };
bool better(const Candidate& a,const Candidate& b) {
    if (a.score != b.score) return a.score>b.score;
    if (a.y != b.y) return a.y<b.y;
    if (a.x != b.x) return a.x<b.x;
    return a.angle<b.angle;
}
std::pair<double,double> gradient(ImageView image,int x,int y) {
    const auto* row=image.data+std::size_t(y)*image.stride;
    return {(double(row[x+1])-row[x-1])*.5,
        (double(image.data[std::size_t(y+1)*image.stride+x])-
            image.data[std::size_t(y-1)*image.stride+x])*.5};
}
std::vector<Feature> learnFeatures(ImageView image,const SparseOrientationSettings& s) {
    std::vector<Feature> ranked;
    for (int y=1;y<image.height-1;++y) for (int x=1;x<image.width-1;++x) {
        const auto g=gradient(image,x,y);
        const double magnitude=std::hypot(g.first,g.second);
        if (magnitude >= s.minGradient)
            ranked.push_back({x-image.width*.5,y-image.height*.5,std::atan2(g.second,g.first),magnitude});
    }
    std::sort(ranked.begin(),ranked.end(),[](const Feature& a,const Feature& b) {
        if (a.magnitude!=b.magnitude) return a.magnitude>b.magnitude;
        if (a.y!=b.y) return a.y<b.y;
        return a.x<b.x;
    });
    std::vector<Feature> selected;
    selected.reserve(std::min<std::size_t>(s.featureCount,ranked.size()));
    // Greedy spatial separation prevents a single very strong edge from owning
    // the entire model. Relax the separation only when too few points survive.
    double separation=std::max(1.,std::sqrt(double(image.width)*image.height/s.featureCount)*.7);
    for (;;) {
        selected.clear();
        const double distance2=separation*separation;
        for (const auto& f : ranked) {
            bool close=false;
            for (const auto& prior : selected) {
                const double dx=f.x-prior.x,dy=f.y-prior.y;
                if (dx*dx+dy*dy<distance2) { close=true;break; }
            }
            if (!close) selected.push_back(f);
            if (selected.size()==std::size_t(s.featureCount)) break;
        }
        if (selected.size()==std::min<std::size_t>(s.featureCount,ranked.size()) || separation<=1.) break;
        separation=std::max(1.,separation*.75);
    }
    return selected;
}

// A bit mask records the nearest unsigned orientation at each strong-gradient
// pixel. OR-spreading is exactly equivalent to taking the maximum response over
// each square neighborhood for the quantized orientation response table.
using ResponseLookup = std::array<std::array<std::uint8_t,256>,orientationBins>;
const ResponseLookup& responseLookup() {
    static const ResponseLookup tables=[] {
        ResponseLookup result{};
        for (int bin=0;bin<orientationBins;++bin)
            for (int bits=1;bits<256;++bits) for (int sourceBin=0;sourceBin<orientationBins;++sourceBin)
                if (bits&(1<<sourceBin)) result[bin][bits]=std::max(result[bin][bits],
                    std::uint8_t(std::lround(255*std::abs(std::cos((bin-sourceBin)*pi/orientationBins)))));
        return result;
    }();
    return tables;
}
std::vector<std::uint8_t> responseMaps(ImageView source,const SparseOrientationSettings& s,
    const std::vector<std::uint8_t>& gradients,std::uint64_t& prepareBytes) {
    const std::size_t count=std::size_t(source.width)*source.height;
    std::vector<std::uint8_t> mask(count),horizontal(count),spread(count);
    if (s.cpuGradientMode==1) {
        for (int y=1;y<source.height-1;++y) for (int x=1;x<source.width-1;++x) {
            const auto* row=source.data+std::size_t(y)*source.stride;
            const int dx=int(row[x+1])-row[x-1];
            const int dy=int(source.data[std::size_t(y+1)*source.stride+x])-
                source.data[std::size_t(y-1)*source.stride+x];
            mask[std::size_t(y)*source.width+x]=
                gradients[std::size_t(dy+255)*gradientAxisValues+dx+255];
        }
    } else {
        for (int y=1;y<source.height-1;++y) for (int x=1;x<source.width-1;++x) {
            const auto g=gradient(source,x,y);
            if (g.first*g.first+g.second*g.second >= s.minGradient*s.minGradient)
                mask[std::size_t(y)*source.width+x]=std::uint8_t(1u<<orientationBin(std::atan2(g.second,g.first)));
        }
    }
    for (int y=0;y<source.height;++y) for (int x=0;x<source.width;++x) {
        std::uint8_t bits=0;
        for (int dx=std::max(-s.spreadRadius,-x);dx<=std::min(s.spreadRadius,source.width-1-x);++dx)
            bits|=mask[std::size_t(y)*source.width+x+dx];
        horizontal[std::size_t(y)*source.width+x]=bits;
    }
    for (int y=0;y<source.height;++y) for (int x=0;x<source.width;++x) {
        std::uint8_t bits=0;
        for (int dy=std::max(-s.spreadRadius,-y);dy<=std::min(s.spreadRadius,source.height-1-y);++dy)
            bits|=horizontal[std::size_t(y+dy)*source.width+x];
        spread[std::size_t(y)*source.width+x]=bits;
    }
    prepareBytes=mask.capacity()+horizontal.capacity()+spread.capacity();
    if (s.responseMode==1) return spread;
    std::vector<std::uint8_t> maps(count*orientationBins);
    prepareBytes+=maps.capacity();
    const auto& tables=responseLookup();
    for (int bin=0;bin<orientationBins;++bin) {
        const auto& table=tables[bin];
        auto* response=maps.data()+std::size_t(bin)*count;
        for (std::size_t i=0;i<count;++i) response[i]=table[spread[i]];
    }
    return maps;
}
Match makeMatch(const Candidate& p,int width,int height) {
    Match m;m.center={double(p.x),double(p.y)};m.angle=p.angle;m.score=p.score;
    const double c=std::cos(p.angle*pi/180),s=std::sin(p.angle*pi/180);
    const std::array<Point,4> local{{{-width*.5,-height*.5},{width*.5,-height*.5},
        {width*.5,height*.5},{-width*.5,height*.5}}};
    for (int k=0;k<4;++k)
        m.corners[k]={p.x+c*local[k].x+s*local[k].y,p.y-s*local[k].x+c*local[k].y};
    return m;
}
std::size_t tileIndex(const Candidate& p,int tileSize,int tileColumns) {
    return std::size_t(p.y/tileSize)*tileColumns+p.x/tileSize;
}
// Spatial round-robin selection limits domination by repeated peaks in one
// region. This is a bounded approximate proposal policy, not a recall guarantee.
std::vector<Candidate> diverseLimit(std::vector<Candidate> input,std::size_t cap,
    int width,int height,int tileSize) {
    std::sort(input.begin(),input.end(),better);
    if (input.size()<=cap) return input;
    const int columns=(width-1)/tileSize+1,rows=(height-1)/tileSize+1;
    std::vector<std::vector<std::size_t>> tiles(std::size_t(columns)*rows);
    for (std::size_t i=0;i<input.size();++i)
        tiles[tileIndex(input[i],tileSize,columns)].push_back(i);
    std::vector<Candidate> selected;selected.reserve(cap);
    for (std::size_t round=0;selected.size()<cap;++round) {
        std::vector<std::size_t> wave;
        for (const auto& tile : tiles) if (round<tile.size()) wave.push_back(tile[round]);
        if (wave.empty()) break;
        std::sort(wave.begin(),wave.end()); // Input is already score-sorted.
        for (auto i : wave) {
            selected.push_back(input[i]);
            if (selected.size()==cap) break;
        }
    }
    std::sort(selected.begin(),selected.end(),better);
    return selected;
}
struct AngleResult {
    std::vector<Candidate> candidates;
    std::uint64_t windows=0,evaluations=0,survivors=0,workingBytes=0;
};
}

struct SparseOrientationPattern::Impl {
    int width=0,height=0;
    SparseOrientationSettings settings;
    std::vector<Feature> features;
    std::vector<std::uint8_t> gradients; // Immutable CPU table, absent for Metal and flat models.
#ifdef FPM_SPARSE_METAL
    std::shared_ptr<detail::SparseMetalBackend> metal;
#endif
};
SparseOrientationPattern::SparseOrientationPattern(ImageView image,const SparseOrientationSettings& settings) {
    validateImage(image);validateSettings(settings);
    auto model=std::make_shared<Impl>();
    model->width=image.width;model->height=image.height;model->settings=settings;
    model->features=learnFeatures(image,settings);
    if (settings.searchBackend == 1) {
#ifdef FPM_SPARSE_METAL
      if(!model->features.empty()) {
        const auto gradients=makeGradientLookup(settings.minGradient);
        std::vector<std::uint8_t> responses;
        responses.reserve(2048);
        for(const auto& row:responseLookup()) responses.insert(responses.end(),row.begin(),row.end());
        model->metal=detail::makeSparseMetalBackend(gradients,responses);
      }
#else
        throw std::runtime_error("Metal sparse search is not compiled on this platform");
#endif
    } else if (settings.cpuGradientMode==1 && !model->features.empty())
        model->gradients=makeGradientLookup(settings.minGradient);
    impl_=std::move(model);
}
SparseOrientationPattern::~SparseOrientationPattern()=default;
int SparseOrientationPattern::featureCount() const { return int(impl_->features.size()); }
std::uint64_t SparseOrientationPattern::templateBytes() const {
    return sizeof(Impl)+impl_->features.capacity()*sizeof(Feature)+impl_->gradients.capacity();
}
int SparseOrientationPattern::width() const { return impl_->width; }
int SparseOrientationPattern::height() const { return impl_->height; }

SparseOrientationResult SparseOrientationPattern::search(ImageView source,const MatchConfig& config) const {
    validateImage(source);
    const auto& model=*impl_;const auto& settings=model.settings;
    const auto angles=makeAngles(config,settings.angleStepDegrees);
    SparseOrientationResult result;auto& metrics=result.metrics;
    metrics.templateBytes=templateBytes();metrics.featureCount=featureCount();metrics.angleCount=int(angles.size());
    metrics.responseMode=settings.responseMode;
    metrics.selectedBackend=settings.searchBackend;
    metrics.gradientMode=settings.searchBackend==1?1:settings.cpuGradientMode;
    metrics.metalSubmissionMode=settings.searchBackend==1?settings.metalSubmissionMode:0;
    if (model.features.empty()) return result;
    metrics.gradientLookupBytes=settings.searchBackend==1?
        std::uint64_t(gradientAxisValues)*gradientAxisValues:model.gradients.capacity();
    auto start=Clock::now();
    const auto& lookup=responseLookup();
    metrics.lookupBytes=sizeof(lookup);
    std::uint64_t prepareBytes=0;
    const auto maps=settings.searchBackend==0?responseMaps(source,settings,model.gradients,prepareBytes):
        std::vector<std::uint8_t>{};
    const std::size_t imageSize=std::size_t(source.width)*source.height;
    metrics.sourceBytes=maps.capacity();
    // Compact mode retains the spread mask from the three construction planes;
    // reference mode additionally materializes all eight response planes. The
    // immutable lookup table is shared across calls and model instances.
    metrics.workingBytes=prepareBytes+metrics.lookupBytes;
    metrics.sourcePrepareMs=milliseconds(start);
    start=Clock::now();
    std::vector<AngleResult> angleResults(angles.size());
    std::uint64_t gpuHostWorkspace=0;
    std::atomic<std::size_t> nextAngle{0};
    std::exception_ptr failure;std::mutex failureMutex;
    const int workers=std::min<int>(settings.cpuThreads,int(angles.size()));
    const int tileSize=std::max(32,std::max(model.width,model.height));
    const int totalScore=255*int(model.features.size());
    const auto requiredScore=std::uint32_t(std::ceil(settings.minScore*totalScore));
    auto scanAngle=[&](std::size_t index,auto compactTag) {
        constexpr bool compact=decltype(compactTag)::value;
        auto& out=angleResults[index];
        const double angle=angles[index],radians=angle*pi/180,c=std::cos(radians),s=std::sin(radians);
        // Rectangle extents are geometric width/height, with inclusive touching
        // of the source boundary. All sampled feature coordinates are interior.
        const double halfWidth=(std::abs(c)*model.width+std::abs(s)*model.height)*.5;
        const double halfHeight=(std::abs(s)*model.width+std::abs(c)*model.height)*.5;
        const int minX=int(std::ceil(halfWidth-1e-9)),maxX=int(std::floor(source.width-halfWidth+1e-9));
        const int minY=int(std::ceil(halfHeight-1e-9)),maxY=int(std::floor(source.height-halfHeight+1e-9));
        if (minX>maxX || minY>maxY) return;
        // Include both edges even when the stride does not divide the extent.
        std::vector<int> xs,ys;
        for (int x=minX;x<=maxX;) { xs.push_back(x);if(maxX-x<settings.positionStride)break;x+=settings.positionStride; }
        for (int y=minY;y<=maxY;) { ys.push_back(y);if(maxY-y<settings.positionStride)break;y+=settings.positionStride; }
        if (xs.back()!=maxX) xs.push_back(maxX);
        if (ys.back()!=maxY) ys.push_back(maxY);
        const std::size_t gridWidth=xs.size(),gridHeight=ys.size();
        std::vector<std::uint32_t> scores(gridWidth*gridHeight);
        std::vector<RotatedFeature> features;features.reserve(model.features.size());
        for (const auto& f : model.features) {
            const int x=int(std::lround(c*f.x+s*f.y)),y=int(std::lround(-s*f.x+c*f.y));
            const int bin=orientationBin(f.direction-radians);
            auto offset=std::ptrdiff_t(y)*source.width+x;
            if constexpr (!compact) offset+=std::ptrdiff_t(bin)*std::ptrdiff_t(imageSize);
            features.push_back({offset,bin});
        }
        out.windows=scores.size();
        out.workingBytes=scores.capacity()*sizeof(std::uint32_t)+features.capacity()*sizeof(RotatedFeature)+
            (xs.capacity()+ys.capacity())*sizeof(int);
        for (std::size_t iy=0;iy<gridHeight;++iy) for (std::size_t ix=0;ix<gridWidth;++ix) {
            const std::ptrdiff_t origin=std::ptrdiff_t(ys[iy])*source.width+xs[ix];
            std::uint32_t score=0;
            std::size_t j=0;
            for (;j<features.size();++j) {
                const auto response=maps[std::size_t(origin+features[j].offset)];
                if constexpr (compact) score+=lookup[features[j].bin][response];
                else score+=response;
                ++out.evaluations;
                // Exact bound for this quantized sparse score only. It does not
                // imply anything about the full-template NCC score.
                if ((j&15)==15 && score+(features.size()-j-1)*255<requiredScore) { ++j;break; }
            }
            if (j==features.size() && score>=requiredScore) {
                scores[iy*gridWidth+ix]=score;++out.survivors;
            }
        }
        for (std::size_t iy=0;iy<gridHeight;++iy) for (std::size_t ix=0;ix<gridWidth;++ix) {
            const std::size_t pos=iy*gridWidth+ix;const auto score=scores[pos];
            if (!score) continue;
            bool peak=true;
            for (int dy=-1;dy<=1 && peak;++dy) for (int dx=-1;dx<=1;++dx) {
                const auto ny=std::ptrdiff_t(iy)+dy,nx=std::ptrdiff_t(ix)+dx;
                if (ny<0 || nx<0 || ny>=std::ptrdiff_t(gridHeight) || nx>=std::ptrdiff_t(gridWidth)) continue;
                const std::size_t neighbor=std::size_t(ny)*gridWidth+std::size_t(nx);
                if (scores[neighbor]>score || (scores[neighbor]==score && neighbor<pos)) { peak=false;break; }
            }
            if (peak) out.candidates.push_back({xs[ix],ys[iy],angle,double(score)/totalScore});
        }
        out.candidates=diverseLimit(std::move(out.candidates),std::size_t(settings.candidateCap)*2,
            source.width,source.height,tileSize);
    };
    auto scanAngles=[&](auto compactTag) {
        for (;;) {
            const auto index=nextAngle.fetch_add(1);
            if(index>=angles.size()) break;
            scanAngle(index,compactTag);
        }
    };
    auto worker=[&] {
        try {
            if (settings.responseMode==1) scanAngles(std::true_type{});
            else scanAngles(std::false_type{});
        } catch (...) { std::lock_guard<std::mutex> lock(failureMutex);if(!failure) failure=std::current_exception(); }
    };
    if(settings.searchBackend==0) {
        std::vector<std::thread> threads;
        try { for (int i=1;i<workers;++i) threads.emplace_back(worker); }
        catch (...) { for(auto& thread:threads) thread.join();throw; }
        worker();for(auto& thread:threads) thread.join();
        if (failure) std::rethrow_exception(failure);
    } else {
#ifdef FPM_SPARSE_METAL
        if(imageSize>std::size_t(INT32_MAX))
            throw std::invalid_argument("Metal sparse image exceeds signed 32-bit indexing");
        std::vector<detail::SparseMetalAngle> plans(angles.size());
        for(std::size_t i=0;i<angles.size();++i) {
            auto& plan=plans[i];
            const double radians=angles[i]*pi/180,c=std::cos(radians),s=std::sin(radians);
            const double halfWidth=(std::abs(c)*model.width+std::abs(s)*model.height)*.5;
            const double halfHeight=(std::abs(s)*model.width+std::abs(c)*model.height)*.5;
            const int minX=int(std::ceil(halfWidth-1e-9)),maxX=int(std::floor(source.width-halfWidth+1e-9));
            const int minY=int(std::ceil(halfHeight-1e-9)),maxY=int(std::floor(source.height-halfHeight+1e-9));
            if(minX>maxX || minY>maxY) continue;
            for(int x=minX;x<=maxX;) {plan.xs.push_back(x);if(maxX-x<settings.positionStride)break;x+=settings.positionStride;}
            for(int y=minY;y<=maxY;) {plan.ys.push_back(y);if(maxY-y<settings.positionStride)break;y+=settings.positionStride;}
            if(plan.xs.back()!=maxX)plan.xs.push_back(maxX);
            if(plan.ys.back()!=maxY)plan.ys.push_back(maxY);
            for(const auto& f:model.features) {
                const int x=int(std::lround(c*f.x+s*f.y)),y=int(std::lround(-s*f.x+c*f.y));
                const auto offset=std::int64_t(y)*source.width+x;
                if(offset<INT32_MIN || offset>INT32_MAX)throw std::invalid_argument("Metal sparse feature offset overflow");
                plan.features.push_back({std::int32_t(offset),std::uint32_t(orientationBin(f.direction-radians))});
            }
        }
        const auto gpu=model.metal->search(source,plans,settings.spreadRadius,requiredScore,
            settings.metalSubmissionMode);
        if(gpu.angles.size()!=angles.size()) throw std::runtime_error("Metal sparse result angle count differs");
        metrics.sourcePrepareMs=gpu.metrics.sourcePrepareMs;
        metrics.sourceBytes=gpu.metrics.sourcePrepareMs>0?imageSize:0;
        metrics.gpuSetupMs=gpu.metrics.setupMs;metrics.uploadMs=gpu.metrics.uploadMs;
        metrics.encodeMs=gpu.metrics.encodeMs;metrics.waitMs=gpu.metrics.waitMs;
        metrics.deviceMs=gpu.metrics.deviceMs;metrics.readbackMs=gpu.metrics.readbackMs;
        metrics.gpuBufferBytes=gpu.metrics.gpuBufferBytes;metrics.commandBuffers=gpu.metrics.commandBuffers;
        metrics.dispatches=gpu.metrics.dispatches;metrics.cpuWaits=gpu.metrics.cpuWaits;
        metrics.peakOverflowRetries=gpu.metrics.peakOverflowRetries;
        for(std::size_t i=0;i<angles.size();++i) {
            const auto& from=gpu.angles[i];auto& out=angleResults[i];
            out.windows=from.windows;out.evaluations=from.evaluations;out.survivors=from.survivors;
            out.workingBytes=plans[i].features.capacity()*sizeof(detail::SparseMetalFeature)+
                (plans[i].xs.capacity()+plans[i].ys.capacity())*sizeof(std::int32_t)+
                from.peaks.capacity()*sizeof(detail::SparseMetalPeak);
            gpuHostWorkspace+=out.workingBytes;
            for(const auto& peak:from.peaks) {
                if(peak.xIndex>=plans[i].xs.size() || peak.yIndex>=plans[i].ys.size())
                    throw std::runtime_error("Metal sparse peak outside search grid");
                out.candidates.push_back({plans[i].xs[peak.xIndex],plans[i].ys[peak.yIndex],angles[i],double(peak.score)/totalScore});
            }
            out.candidates=diverseLimit(std::move(out.candidates),std::size_t(settings.candidateCap)*2,
                source.width,source.height,tileSize);
        }
#endif
    }
    std::vector<Candidate> candidates;
    std::uint64_t maxWorkerBytes=0,candidateBytes=0;
    for (auto& one : angleResults) {
        metrics.windows+=one.windows;metrics.featureEvaluations+=one.evaluations;metrics.thresholdSurvivors+=one.survivors;
        maxWorkerBytes=std::max(maxWorkerBytes,one.workingBytes);
        candidateBytes+=one.candidates.capacity()*sizeof(Candidate);
        candidates.insert(candidates.end(),one.candidates.begin(),one.candidates.end());
    }
    // Accounted allocation estimate, not an allocator peak or OS RSS bound.
    // Temporary selection indices and nested vector spare capacities are not
    // completely accounted here; benchmark process RSS must be measured too.
    const auto sourceAndWork=settings.searchBackend?metrics.gpuBufferBytes+gpuHostWorkspace:
        metrics.sourceBytes+maxWorkerBytes*workers;
    metrics.workingBytes=std::max(metrics.workingBytes,metrics.lookupBytes+sourceAndWork+
        candidateBytes+candidates.capacity()*sizeof(Candidate));
    metrics.searchMs=milliseconds(start)-(settings.searchBackend?metrics.sourcePrepareMs:0.);start=Clock::now();
    std::sort(candidates.begin(),candidates.end(),better);
    std::vector<Candidate> distinct;distinct.reserve(candidates.size());
    const double spatialRadius=std::max(double(settings.positionStride)*1.5,std::min(model.width,model.height)*.15);
    const double angularRadius=std::min(30.,std::max(6.,settings.angleStepDegrees*1.5));
    // Spatial bins bound NMS comparisons for large images. Neighbor cells are
    // checked across all angles; 180-degree alternatives remain distinct.
    const int cellSize=std::max(1,int(std::ceil(spatialRadius)));
    const int columns=(source.width-1)/cellSize+1,rows=(source.height-1)/cellSize+1;
    std::vector<std::vector<std::size_t>> cells(std::size_t(columns)*rows);
    for (const auto& p : candidates) {
        const int gx=p.x/cellSize,gy=p.y/cellSize;bool duplicate=false;
        for (int y=std::max(0,gy-1);y<=std::min(rows-1,gy+1) && !duplicate;++y)
            for (int x=std::max(0,gx-1);x<=std::min(columns-1,gx+1) && !duplicate;++x)
                for (auto index : cells[std::size_t(y)*columns+x]) {
                    const auto& q=distinct[index];const double dx=p.x-q.x,dy=p.y-q.y;
                    if (dx*dx+dy*dy<=spatialRadius*spatialRadius &&
                        std::abs(normalizeAngle(p.angle-q.angle))<=angularRadius) { duplicate=true;break; }
                }
        if (!duplicate) { cells[std::size_t(gy)*columns+gx].push_back(distinct.size());distinct.push_back(p); }
    }
    const auto selected=diverseLimit(std::move(distinct),settings.candidateCap,source.width,source.height,tileSize);
    result.proposals.reserve(selected.size());
    for (const auto& p : selected) result.proposals.push_back(makeMatch(p,model.width,model.height));
    metrics.workingBytes=std::max(metrics.workingBytes,metrics.lookupBytes+
        (settings.searchBackend?metrics.gpuBufferBytes:metrics.sourceBytes)+candidateBytes+
        candidates.capacity()*sizeof(Candidate)*2+cells.capacity()*sizeof(std::vector<std::size_t>)+
        candidates.size()*sizeof(std::size_t)+result.proposals.capacity()*sizeof(Match));
    metrics.selectMs=milliseconds(start);
    return result;
}
} // namespace fpm::research
