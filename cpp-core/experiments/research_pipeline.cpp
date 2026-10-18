// SPDX-License-Identifier: BSD-2-Clause
#include "research_pipeline.hpp"
#include "scale_support_refine.hpp"
#include "full_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fpm::research {
namespace {
using Clock = std::chrono::steady_clock;
constexpr double pi = 3.141592653589793238462643383279502884;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
fpm_image view(ImageView v) { return {v.data,v.width,v.height,v.stride}; }
void check(int status, const char* error) {
    if (status != FPM_OK) throw std::runtime_error(*error ? error : "Research pipeline API failed");
}
void validateImage(ImageView v) {
    if (!v.data || v.width<1 || v.height<1 || v.stride<size_t(v.width) ||
        v.stride>std::numeric_limits<size_t>::max()/size_t(v.height))
        throw std::invalid_argument("Invalid image dimensions, pixels or stride");
}
int roundedDimension(double value) {
    if (!std::isfinite(value) || value<.5 || value>32768)
        throw std::invalid_argument("Scaled template dimensions must be 1..32768");
    double lower=std::floor(value), part=value-lower;
    const int n=int(lower);
    const int rounded=n+(part>.5 || (part==.5 && (n&1)));
    if (!rounded) throw std::invalid_argument("Scale rounds template dimension to zero");
    return rounded;
}
double cross(Point a,Point b,Point p) { return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x); }
double area(const std::vector<Point>& p) {
    double sum=0; for(size_t i=1;i+1<p.size();++i) sum+=cross(p[0],p[i],p[i+1]);
    return std::abs(sum)/2;
}
double overlap(const Match& a,const Match& b) {
    auto bounds=[](const Match& m) {
        std::array<double,4> box{m.corners[0].x,m.corners[0].y,m.corners[0].x,m.corners[0].y};
        for(auto p:m.corners) { box[0]=std::min(box[0],p.x);box[1]=std::min(box[1],p.y);box[2]=std::max(box[2],p.x);box[3]=std::max(box[3],p.y); }
        return box;
    };
    auto aa=bounds(a),bb=bounds(b);
    if (aa[2]<=bb[0] || bb[2]<=aa[0] || aa[3]<=bb[1] || bb[3]<=aa[1]) return 0;
    std::vector<Point> p(a.corners.begin(),a.corners.end()),q(b.corners.begin(),b.corners.end());
    const double denominator=std::min(area(p),area(q));
    if (denominator<=0) return 0;
    for(int i=0;i<4 && !p.empty();++i) {
        std::vector<Point> next;const auto v=q[i],w=q[(i+1)%4];auto previous=p.back();double ps=cross(v,w,previous);
        for(auto current:p) {
            double side=cross(v,w,current);
            if((side>=0)!=(ps>=0)) { double t=ps/(ps-side);next.push_back({previous.x+t*(current.x-previous.x),previous.y+t*(current.y-previous.y)}); }
            if(side>=0) next.push_back(current);
            previous=current;ps=side;
        }
        p=std::move(next);
    }
    return std::clamp(area(p)/denominator,0.,1.);
}
Match makeMatch(Point center,double angle,double scale,int w,int h,double score) {
    Match m;m.center=center;m.angle=angle;m.score=score;
    double c=scale*std::cos(angle*pi/180),s=scale*std::sin(angle*pi/180);
    std::array<Point,4> local{{{-w/2.,-h/2.},{w/2.,-h/2.},{w/2.,h/2.},{-w/2.,h/2.}}};
    for(int k=0;k<4;++k) m.corners[k]={c*local[k].x+s*local[k].y+center.x,-s*local[k].x+c*local[k].y+center.y};
    return m;
}
bool allowed(double a,const MatchConfig& c) {
    // Equivalent +/-180 representations are the same physical angle.
    if(c.angleRanges.empty()) return std::abs(a)<=c.toleranceAngle+1e-5;
    for(auto r:c.angleRanges) for(double shift:{-360.,0.,360.})
        if(a+shift>=r.min-1e-5 && a+shift<=r.max+1e-5) return true;
    return false;
}
void validateOptions(const Options& o,int backend) {
    const auto& c=o.match;
    if(c.maxTargets<1 || c.maxTargets>10000 || !std::isfinite(c.minScore) || c.minScore<0 || c.minScore>1 ||
       !std::isfinite(c.maxOverlap) || c.maxOverlap<0 || c.maxOverlap>1 ||
       !std::isfinite(c.toleranceAngle) || c.toleranceAngle<0 || c.toleranceAngle>180 ||
       (c.stopLayer!=0 && c.stopLayer!=1) || !std::isfinite(o.proposalScore) || o.proposalScore<0 || o.proposalScore>1 ||
       o.nccMode<0 || o.nccMode>3 || o.fineNccMode<0 || o.fineNccMode>3 || o.fullGpuPeakMode<0 || o.fullGpuPeakMode>1 ||
       o.coarsePeakMode<0 || o.coarsePeakMode>2 || (o.coarsePeakMode==2 && (backend==0 || o.overlap)) ||
       (backend==0 && (o.nccMode || o.fineNccMode || o.fineProfiling || o.fineAngleReuse || o.fineFused || o.overlap)))
        throw std::invalid_argument("Invalid matching options or GPU option selected for CPU");
    if(c.coarseNccPruning) throw std::invalid_argument("Research frame does not support coarse pruning");
    for(auto r:c.angleRanges) if(!std::isfinite(r.min) || !std::isfinite(r.max) || r.min>r.max || r.min< -180 || r.max>180)
        throw std::invalid_argument("Angle range must be ordered within -180..180");
}
}
Image scaledTemplate(ImageView source,double scale) {
    validateImage(source);
    if(!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Scale must be finite and positive");
    const int w=roundedDimension(source.width*scale),h=roundedDimension(source.height*scale);
    if(uint64_t(w)*h>64ull*1024*1024) throw std::invalid_argument("Scaled template exceeds 64M pixels");
    if(scale==1) return Image(source);
    Image out(w,h);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        double sx=source.width/2.+(x-w/2.)/scale,sy=source.height/2.+(y-h/2.)/scale;
        int ix=int(std::floor(sx)),iy=int(std::floor(sy));double dx=sx-ix,dy=sy-iy,value=0;
        for(int oy=0;oy<2;++oy) for(int ox=0;ox<2;++ox) if(ix+ox>=0 && ix+ox<source.width && iy+oy>=0 && iy+oy<source.height)
            value+=source.data[size_t(iy+oy)*source.stride+ix+ox]*((ox?dx:1-dx)*(oy?dy:1-dy));
        out.row(y)[x]=uint8_t(std::clamp(std::floor(value+.5),0.,255.));
    }
    return out;
}
struct Session::Impl {
    Image source,templ;
    Settings settings;
    std::vector<std::unique_ptr<fpm_coarse_handle,decltype(&fpm_coarse_destroy)>> patterns;
    std::unique_ptr<fpm_frame_handle,decltype(&fpm_frame_destroy)> frame{nullptr,fpm_frame_destroy};
    std::unique_ptr<void,decltype(&fpm_pose_destroy)> pose{nullptr,fpm_pose_destroy};
    std::unique_ptr<SparseOrientationPattern> sparse;
    std::unique_ptr<FullGpuPipeline> fullGpu;
    bool poseInvert=false;
    uint64_t sourceGeneration=0;
    int refinementBackend() const { return settings.refinementBackend<0?settings.backend:settings.refinementBackend; }
    Impl(ImageView src,ImageView tmp,const Settings& s):source(src),templ(tmp),settings(s) {
        if(s.backend<0 || s.backend>1 || s.minReducedArea<1 || s.cpuThreads<1 || s.cpuThreads>256 || s.scales.empty() || s.scales.size()>64)
            throw std::invalid_argument("Invalid backend, reduced area, threads or scale bank");
        if(s.refinementBackend< -1 || s.refinementBackend>1)
            throw std::invalid_argument("Invalid refinement backend");
        if(s.residentMode<0 || s.residentMode>2 ||
            (s.residentMode && (refinementBackend()!=1 || uint64_t(tmp.width)*tmp.height>16384)))
            throw std::invalid_argument("Fused refinement requires Metal and a template with <=16384 pixels");
        if(s.poseStorageMode<0 || s.poseStorageMode>1 || (s.poseStorageMode && refinementBackend()!=1))
            throw std::invalid_argument("Shared pose storage requires Metal refinement");
        if(s.poseInputMode<0 || s.poseInputMode>1)
            throw std::invalid_argument("Invalid pose input mode");
        if(s.fullGpu) {
            if(s.backend!=1 || refinementBackend()!=1 || s.sparseOrientation)
                throw std::invalid_argument("Complete GPU pipeline requires dense Metal search and refinement");
            fullGpu=makeFullGpuPipeline(source.view(),templ.view(),s);
            return;
        }
        if(s.sparseOrientation) {
            if(s.scales.size()!=1 || s.scales.front()!=1.)
                throw std::invalid_argument("Sparse orientation requires fixed scale {1}");
            if(s.sparse.candidateCap>65536)
                throw std::invalid_argument("Sparse orientation refinement budget exceeds 65536 proposals");
            sparse=std::make_unique<SparseOrientationPattern>(templ.view(),s.sparse);
            return;
        }
        char error[1024]{};
        for(double scale:s.scales) {
            auto resized=scaledTemplate(templ.view(),scale);fpm_coarse_handle* p=nullptr;FpmCoarseMetrics metrics{};
            // Models only learn CPU pixels. The shared frame owns the sole GPU backend.
            check(fpm_coarse_create_backend(view(resized.view()),s.minReducedArea,0,&p,&metrics,error,sizeof(error)),error);
            patterns.emplace_back(p,fpm_coarse_destroy);
        }
        FpmFrameConfigV1 config{};fpm_frame_default_config(&config);config.backend=s.backend;
        config.pyramid_mode=s.backend?1:0;config.batch_angles=s.backend?4:1;config.fine_batch_size=s.backend?512:0;
        fpm_frame_handle* f=nullptr;check(fpm_frame_create(&config,sizeof(config),&f,error,sizeof(error)),error);frame.reset(f);
    }
    double ensurePose(bool invert,uint64_t& temporaryBytes) {
        temporaryBytes=0;
        if(pose && poseInvert==invert) return 0;
        const auto begin=Clock::now();
        // Release old GPU image before replacement, also after previous errors.
        pose.reset();
        char error[1024]{};
        std::vector<float> src,tmp;
        if(settings.poseInputMode==0) {
            src.resize(size_t(source.width())*source.height());tmp.resize(size_t(templ.width())*templ.height());
            for(int y=0;y<source.height();++y) for(int x=0;x<source.width();++x) src[size_t(y)*source.width()+x]=(invert?255-source.row(y)[x]:source.row(y)[x])/255.f;
            for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) tmp[size_t(y)*templ.width()+x]=templ.row(y)[x]/255.f;
            temporaryBytes=(src.capacity()+tmp.capacity())*sizeof(float);
            pose.reset(fpm_pose_create_with_storage(refinementBackend(),settings.cpuThreads,src.data(),source.width(),source.height(),tmp.data(),templ.width(),templ.height(),1,settings.poseStorageMode,error,sizeof(error)));
        } else {
            pose.reset(fpm_pose_create_u8(refinementBackend(),settings.cpuThreads,source.view().data,source.width(),source.height(),source.view().stride,
                templ.view().data,templ.width(),templ.height(),templ.view().stride,1,int(invert),settings.poseStorageMode,error,sizeof(error)));
        }
        if(!pose) throw std::runtime_error(error);
        check(fpm_pose_set_resident_mode(pose.get(),settings.residentMode,error,sizeof(error)),error);
        poseInvert=invert;
        return ms(begin);
    }
};
Session::Session(ImageView source,ImageView templ,const Settings& settings) {
    validateImage(source);validateImage(templ);impl_=std::make_unique<Impl>(source,templ,settings);
}
Session::~Session()=default;
SourceUpdate Session::updateSource(ImageView source) {
    const auto start=Clock::now();validateImage(source);
    Image replacement(source); // Complete the potentially throwing copy before mutation.
    auto& s=*impl_;SourceUpdate update;
    update.dimensionsChanged=source.width!=s.source.width() || source.height!=s.source.height();
    update.poseInvalidated=bool(s.pose);
    update.retainedModels=s.sparse?1:s.patterns.size();
    update.copiedBytes=uint64_t(source.width)*source.height;
    if(s.fullGpu) { s.fullGpu->updateSource(replacement.view()); update.retainedModels=s.settings.scales.size(); }
    s.pose.reset(); // Current API binds both images; never leave stale pixels in a pose context.
    s.source=std::move(replacement);
    update.generation=++s.sourceGeneration;update.updateMs=ms(start);return update;
}
Result Session::run(const Options& o) {
    auto& s=*impl_;validateOptions(o,s.settings.backend);Result r;char error[1024]{};const auto start=Clock::now();
    if(s.fullGpu) { r=s.fullGpu->run(o); r.sourceGeneration=s.sourceGeneration; return r; }
    if(o.fullGpuPeakMode || o.fullGpuProfiling)
        throw std::invalid_argument("Full GPU peak selection/profiling requires full_gpu=true");
    r.scaleSupportEnabled=o.scaleSupportRefine;
    r.refinementMode=s.settings.residentMode;
    r.configuredRefinementBackend=s.refinementBackend();r.sourceGeneration=s.sourceGeneration;
    r.poseStorage.mode=s.settings.poseStorageMode;
    r.poseInputMode=s.settings.poseInputMode;
    std::vector<fpm_angle_range> ranges;for(auto a:o.match.angleRanges) ranges.push_back({a.min,a.max});
    if(s.sparse) {
        // Orientation similarity is not NCC and cannot be used as a final
        // confidence. Keep all raw proposals visible for recall auditing.
        if(!o.refine || o.match.stopLayer!=0 || o.nccMode || o.fineNccMode || o.fineProfiling || o.fineAngleReuse || o.fineFused || o.overlap || o.coarsePeakMode==2)
            throw std::invalid_argument("Sparse orientation requires full NCC refinement, stopLayer=0, nccMode=0, fineNccMode=0 and overlap scheduling off");
        auto result=s.sparse->search(s.source.view(),o.match);
        r.sparseEnabled=true;r.sparse=result.metrics;
        r.proposals=std::move(result.proposals);
        r.proposalScales.assign(r.proposals.size(),1.);
        r.prepareMs=r.sparse.sourcePrepareMs;
    } else {
        if(s.settings.backend) {
            if (o.coarsePeakMode==2) {
                FpmCoarseOverlapConfigV1 off{1,sizeof(off),0,0,0,0};
                check(fpm_frame_set_coarse_overlap(s.frame.get(),&off,sizeof(off),error,sizeof(error)),error);
            }
            FpmCoarseOptimizationConfigV1 opt{1,sizeof(opt),o.coarsePeakMode,o.nccMode,0,0};
            check(fpm_frame_set_coarse_optimization(s.frame.get(),&opt,sizeof(opt),error,sizeof(error)),error);
            FpmCoarseOverlapConfigV1 optOverlap{1,sizeof(optOverlap),int(o.overlap),0,0,0};
            check(fpm_frame_set_coarse_overlap(s.frame.get(),&optOverlap,sizeof(optOverlap),error,sizeof(error)),error);
            check(fpm_frame_set_fine_ncc_mode(s.frame.get(),o.fineNccMode,error,sizeof(error)),error);
            check(fpm_frame_set_fine_fused(s.frame.get(),o.fineFused,error,sizeof(error)),error);
            check(fpm_frame_set_fine_search_options(s.frame.get(),o.fineProfiling,o.fineAngleReuse,error,sizeof(error)),error);
        }
        std::vector<fpm_coarse_handle*> models;for(auto& p:s.patterns) models.push_back(p.get());
        const auto prepare=Clock::now();
        check(fpm_frame_prepare(s.frame.get(),view(s.source.view()),models.data(),models.size(),o.match.invertSource,&r.frame,error,sizeof(error)),error);
        r.prepareMs=ms(prepare);
        fpm_config c{};c.max_targets=o.match.maxTargets;c.min_score=o.refine?std::min(o.proposalScore,o.match.minScore):o.match.minScore;
        c.max_overlap=o.match.maxOverlap;c.tolerance_angle=o.match.toleranceAngle;c.subpixel=o.match.subpixel;
        c.invert_source=o.match.invertSource;c.scalar=o.match.backend==Backend::Scalar;c.stop_layer=o.match.stopLayer;
        c.ranges=ranges.data();c.range_count=ranges.size();
        std::vector<fpm_result> output(size_t(c.max_targets));
        for(size_t i=0;i<models.size();++i) {
            size_t count=0;FpmCoarseMetrics coarse{};FpmFineMetricsV1 fine{};FpmCoarseExecutionMetricsV1 execution{};FpmCoarseOverlapMetricsV1 overlapMetrics{};
            check(fpm_frame_match(s.frame.get(),models[i],&c,output.data(),output.size(),&count,&coarse,nullptr,0,&execution,sizeof(execution),&fine,sizeof(fine),error,sizeof(error)),error);
            check(fpm_frame_coarse_overlap_metrics(s.frame.get(),&overlapMetrics,sizeof(overlapMetrics),error,sizeof(error)),error);
            FpmFineReuseMetricsV1 fineReuse{};
            check(fpm_frame_get_fine_reuse_metrics(s.frame.get(),&fineReuse,sizeof(fineReuse),error,sizeof(error)),error);
            r.fineReuse.push_back(fineReuse);
            FpmFineSearchMetricsV1 fineSearch{};
            check(fpm_frame_get_fine_search_metrics(s.frame.get(),&fineSearch,sizeof(fineSearch),error,sizeof(error)),error);
            r.fineSearch.push_back(fineSearch);
            FpmFineFusionMetricsV1 fusion{};
            check(fpm_frame_get_fine_fusion_metrics(s.frame.get(),&fusion,sizeof(fusion),error,sizeof(error)),error);
            r.fineFusion.push_back(fusion);
            r.coarse.push_back(coarse);r.fine.push_back(fine);r.overlapMetrics.push_back(overlapMetrics);
            for(size_t j=0;j<count;++j) {
                const auto& m=output[j];double scale=s.settings.scales[i];
                r.proposals.push_back(makeMatch({m.center.x,m.center.y},m.angle,scale,s.templ.width(),s.templ.height(),m.score));r.proposalScales.push_back(scale);
            }
        }
    }
    r.proposalMs=ms(start);
    auto candidates=r.proposals;auto scales=r.proposalScales;
    if(o.refine && !candidates.empty()) {
        const auto t=Clock::now();r.poseCreateMs=s.ensurePose(o.match.invertSource,r.poseConversionTemporaryBytes);
        const size_t n=candidates.size();r.refinementCandidates=n;
        if(n>65536) throw std::invalid_argument("Refinement accepts at most 65536 proposals; reduce scale bank or target count");
        std::vector<float> seeds(n*4),refined(n*4),scores(n);
        for(size_t i=0;i<n;++i) {
            seeds[4*i]=float(candidates[i].center.x-s.source.width()/2.);seeds[4*i+1]=float(s.source.height()/2.-candidates[i].center.y);
            seeds[4*i+2]=float(candidates[i].angle);seeds[4*i+3]=float(std::log(scales[i]));
        }
        FpmPoseRefineConfig config{};fpm_pose_refine_default_config(&config);
        config.optimize_scale=s.settings.scales.size()>1;config.optimize_angle=o.match.toleranceAngle!=0 || !ranges.empty();
        FpmPoseRescueConfig rescue{};fpm_pose_rescue_default_config(&rescue);rescue.backend=s.refinementBackend();
        rescue.trigger_score=std::min(rescue.trigger_score,o.match.minScore);rescue.trigger_min_score=std::min(rescue.trigger_min_score,rescue.trigger_score);
        if(fpm_pose_refine_rescue(s.pose.get(),seeds.data(),int(n),&config,&rescue,refined.data(),scores.data(),nullptr,&r.rescue,&r.pose,error,sizeof(error))) throw std::runtime_error(error);
        if(fpm_pose_last_resident_metrics(s.pose.get(),&r.resident)) throw std::runtime_error("Cannot read refinement metrics");
        r.refinementMode=fpm_pose_resident_mode(s.pose.get());
        if(fpm_pose_last_resident_kernel(s.pose.get(),&r.residentKernel)) throw std::runtime_error("Cannot read refinement kernel metrics");
        if(fpm_pose_storage_info(s.pose.get(),&r.poseStorage)) throw std::runtime_error("Cannot read pose storage metrics");
        for(size_t i=0;i<n;++i) {
            // Match NumPy float32 pose + weak Python scalar arithmetic before conversion to double.
            const float cx=refined[4*i]+float(s.source.width()/2.),cy=float(s.source.height()/2.)-refined[4*i+1];
            float a=std::fmod(refined[4*i+2]+180.f,360.f);if(a<0) a+=360.f;a-=180.f;
            scales[i]=std::exp(double(refined[4*i+3]));candidates[i]=makeMatch({cx,cy},a,scales[i],s.templ.width(),s.templ.height(),scores[i]);
        }
        if(o.scaleSupportRefine && config.optimize_scale) {
            const auto supportStart=Clock::now();
            for(size_t i=0;i<n;++i) {
                const auto update=refineScaleSupport(s.source.view(),s.templ.view(),candidates[i].center,
                    candidates[i].angle,scales[i],o.match.invertSource);
                r.scaleSupportEligible+=update.eligible;r.scaleSupportBoundary+=update.boundaryOptimum;
                r.scaleSupportEvaluations+=update.evaluations;r.scaleSupportSamples+=update.samples;
                if(update.candidate) {
                    scales[i]=update.candidate->scale;
                    candidates[i]=makeMatch(candidates[i].center,candidates[i].angle,scales[i],
                        s.templ.width(),s.templ.height(),update.candidate->score);
                    ++r.scaleSupportChanged;
                }
            }
            r.scaleSupportMs=ms(supportStart);
        }
        r.refineMs=ms(t);
    }
    const auto nms=Clock::now();std::vector<size_t> order(candidates.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t i,size_t j) {
        const auto& a=candidates[i];const auto& b=candidates[j];
        if(a.score!=b.score) return a.score>b.score;if(a.center.y!=b.center.y) return a.center.y<b.center.y;
        if(a.center.x!=b.center.x) return a.center.x<b.center.x;return a.angle<b.angle;
    });
    for(size_t i:order) {
        const auto& m=candidates[i];if(m.score<o.match.minScore || !allowed(m.angle,o.match)) continue;
        bool accept=true;for(auto& previous:r.matches) if(overlap(previous,m)>o.match.maxOverlap+1e-10) {accept=false;break;}
        if(accept) { r.matches.push_back(m);r.scales.push_back(scales[i]);if(r.matches.size()>=size_t(o.match.maxTargets)) break; }
    }
    r.nmsMs=ms(nms);
    FpmCoarseExecutionMetricsV1 prepareMetrics{};
    if(s.frame) check(fpm_frame_get_metrics(s.frame.get(),&r.frame,sizeof(r.frame),&prepareMetrics,sizeof(prepareMetrics),error,sizeof(error)),error);
    if(s.pose) {
        r.poseInputMode=fpm_pose_input_format(s.pose.get());
        if(r.poseInputMode!=s.settings.poseInputMode) throw std::runtime_error("Unexpected pose input format");
        FpmPoseMetrics memory{};
        if(fpm_pose_memory(s.pose.get(),&memory)) throw std::runtime_error("Cannot query pose memory");
        r.pose.host_buffer_bytes=memory.host_buffer_bytes;r.pose.gpu_buffer_bytes=memory.gpu_buffer_bytes;r.pose.gpu_allocated_bytes=memory.gpu_allocated_bytes;
        // A discrete-only call can still retain an earlier pose context.
        if(!r.poseStorage.host_image_bytes && fpm_pose_storage_info(s.pose.get(),&r.poseStorage))
            throw std::runtime_error("Cannot query retained pose storage");
    }
    r.totalMs=ms(start);return r;
}
} // namespace fpm::research
