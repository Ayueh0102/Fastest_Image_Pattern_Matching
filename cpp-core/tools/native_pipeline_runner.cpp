// SPDX-License-Identifier: BSD-2-Clause
// JSON job runner for the SAME native Session used by the GUI. Qt only decodes
// images and serializes audit records; all matching is C++/Metal.
#include "research_pipeline.hpp"
#include <QCoreApplication>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
using namespace fpm::research;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
fpm::Image read(const QString& path) {
    QImage image(path);image=image.convertToFormat(QImage::Format_RGB888);
    if(image.isNull()) throw std::invalid_argument("Cannot decode image");
    fpm::Image out(image.width(),image.height());
    for(int y=0;y<image.height();++y) for(int x=0;x<image.width();++x) {
        auto p=image.constScanLine(y)+3*x;out.row(y)[x]=fpm::grayFromRgb(p[0],p[1],p[2]);
    }
    return out;
}
QJsonObject memory() {
    rusage usage{};if(getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("RSS unavailable");
    double peak=usage.ru_maxrss;
#ifndef __APPLE__
    peak*=1024;
#endif
    double resident=-1;
#ifdef __APPLE__
    mach_task_basic_info info{};mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS) resident=info.resident_size;
#endif
    return {{"peak_rss_bytes",peak},{"current_rss_bytes",resident}};
}
QJsonArray matches(const std::vector<fpm::Match>& ms,const std::vector<double>& scales) {
    QJsonArray out;for(size_t i=0;i<ms.size();++i) {
        const auto& m=ms[i];QJsonArray corners;for(auto p:m.corners) corners.append(QJsonArray{p.x,p.y});
        out.append(QJsonObject{{"pos_x",m.center.x},{"pos_y",m.center.y},{"angle",m.angle},{"score",m.score},{"scale",scales[i]},{"corners",corners}});
    }return out;
}
QJsonObject record(const Result& r,bool outputs) {
    QJsonObject o{{"total_ms",r.totalMs},{"prepare_ms",r.prepareMs},{"proposal_ms",r.proposalMs},{"refine_ms",r.refineMs},{"nms_ms",r.nmsMs},
        {"generation",double(r.frame.generation)},{"models_matched",double(r.frame.models_matched)},
        {"frame_host_bytes",double(r.frame.host_pyramid_bytes)},{"frame_coarse_gpu_bytes",double(r.frame.coarse_gpu_bytes)},{"frame_fine_gpu_bytes",double(r.frame.fine_gpu_bytes)},
        {"pose_host_bytes",double(r.pose.host_buffer_bytes)},{"pose_gpu_bytes",double(r.pose.gpu_buffer_bytes)},
        {"selected_refinement_backend",r.rescue.selected_backend},{"rescued_seeds",double(r.rescue.selected_seeds)},{"memory",memory()}};
    QJsonArray work;for(size_t i=0;i<r.coarse.size();++i) {
        auto a=r.coarse[i];auto b=r.fine[i];auto c=r.overlapMetrics[i];
        work.append(QJsonObject{{"maps",double(a.score_maps)},{"windows",double(a.windows)},{"fine_maps",double(b.maps)},
            {"device_ms",a.device_ms},{"wait_ms",a.wait_ms},{"copy_ms",a.upload_ms},{"normalization_ms",a.normalization_ms},
            {"async_submitted",double(c.submitted_batches)},{"async_completed",double(c.completed_batches)},
            {"async_maps",double(c.submitted_maps)},{"async_fallback_maps",double(c.fallback_maps)},{"async_max_pending",double(c.max_pending)},
            {"async_device_ms",c.device_ms},{"async_wait_ms",c.finish_wait_ms},{"async_copy_ms",c.copy_ms}});
    }o["work"]=work;
    if(r.sparseEnabled) {
        const auto& s=r.sparse;
        o["sparse"]=QJsonObject{{"enabled",true},{"source_levels",1},{"feature_count",s.featureCount},
            {"angle_count",s.angleCount},{"candidate_count",double(r.proposals.size())},
            {"source_prepare_ms",s.sourcePrepareMs},{"search_ms",s.searchMs},{"select_ms",s.selectMs},
            {"template_bytes",double(s.templateBytes)},{"source_bytes",double(s.sourceBytes)},
            {"working_bytes",double(s.workingBytes)},{"windows",double(s.windows)},
            {"feature_evaluations",double(s.featureEvaluations)},{"threshold_survivors",double(s.thresholdSurvivors)}};
    }
    if(outputs) {o["matches"]=matches(r.matches,r.scales);o["proposals"]=matches(r.proposals,r.proposalScales);}
    return o;
}
bool equal(const Result& a,const Result& b) {
    // Serialized double precision endpoints plus exact ordered proposal fields;
    // serialization is outside every reported algorithm interval.
    return matches(a.matches,a.scales)==matches(b.matches,b.scales) && matches(a.proposals,a.proposalScales)==matches(b.proposals,b.proposalScales);
}
int main(int argc,char** argv) {
    QCoreApplication app(argc,argv);
    try {
        if(argc!=2) throw std::invalid_argument("Usage: native_pipeline_runner job.json");
        QFile file(QString::fromLocal8Bit(argv[1]));if(!file.open(QIODevice::ReadOnly)) throw std::invalid_argument("Cannot open job");
        const auto j=QJsonDocument::fromJson(file.readAll()).object();
        if(j.isEmpty() || j.contains("truth")) throw std::invalid_argument("Expected job without ground truth");
        auto src=read(j["source"].toString()),tmp=read(j["template"].toString());Settings settings;
        settings.backend=j["backend"].toInt(1);settings.minReducedArea=j["min_area"].toInt(256);settings.scales.clear();
        for(auto s:j["scales"].toArray()) settings.scales.push_back(s.toDouble());
        if(settings.scales.empty()) settings.scales={1.};
        const auto method=j["proposal_method"].toString("pyramid");
        if(method!="pyramid" && method!="sparse_orientation") throw std::invalid_argument("Unknown proposal method");
        settings.sparseOrientation=method=="sparse_orientation";
        settings.sparse.featureCount=j["sparse_features"].toInt(128);
        settings.sparse.positionStride=j["sparse_stride"].toInt(4);
        settings.sparse.angleStepDegrees=j["sparse_angle_step"].toDouble(6.);
        settings.sparse.spreadRadius=j["sparse_spread"].toInt(2);
        settings.sparse.minGradient=j["sparse_min_gradient"].toDouble(8.);
        settings.sparse.minScore=j["sparse_threshold"].toDouble(.75);
        settings.sparse.candidateCap=j["sparse_candidate_limit"].toInt(256);
        settings.cpuThreads=j["cpu_threads"].toInt(8);settings.sparse.cpuThreads=settings.cpuThreads;
        Options o;o.nccMode=j["ncc_mode"].toInt();o.overlap=j["overlap_enabled"].toBool();o.refine=j["refine"].toBool(true);
        o.match.maxTargets=j["limit"].toInt(70);o.match.minScore=j["score"].toDouble(.8);o.match.maxOverlap=j["overlap"].toDouble(.2);o.match.toleranceAngle=j["angle"].toDouble(180);
        o.match.stopLayer=j["stop_layer"].toInt(1);o.match.subpixel=j["subpixel"].toBool();o.match.invertSource=j["invert_source"].toBool();
        o.proposalScore=j["proposal_score"].toDouble(.5);
        const int repeats=j["repeats"].toInt(5);const double warm=j["warmup_seconds"].toDouble(1.),soak=j["soak_seconds"].toDouble(0);
        if(repeats<1 || repeats>10000 || !std::isfinite(warm) || warm<0 || warm>60 || !std::isfinite(soak) || soak<0 || soak>3600) throw std::invalid_argument("Invalid repetitions or duration");
        auto start=Clock::now();auto session=std::make_unique<Session>(src.view(),tmp.view(),settings);const double construct=ms(start);
        Result first=session->run(o);QJsonObject firstRecord=record(first,true);QJsonArray warmups,runs;
        bool exact=true,drained=true;start=Clock::now();
        while(ms(start)<warm*1000) {auto r=session->run(o);exact&=equal(r,first);warmups.append(r.totalMs);}
        if(soak>0) {
            // Keep a fixed reference and stream compact records; no accumulating
            // output maps or JSON records in process memory during soak.
            QFile stream(j["stream_path"].toString());if(!stream.open(QIODevice::WriteOnly|QIODevice::Truncate)) throw std::invalid_argument("Cannot open soak stream");
            const auto before=memory();const auto soakStart=Clock::now();int calls=0,recreated=0;Result reference=first;
            while(ms(soakStart)<soak*1000 || calls<repeats) {
                // Context teardown/recreation every 100 calls exposes ownership
                // and cleanup issues. Setup stays outside algorithm time.
                double recreateMs=0;
                if(calls && calls%100==0) {auto t=Clock::now();session.reset();session=std::make_unique<Session>(src.view(),tmp.view(),settings);recreateMs=ms(t);++recreated;}
                auto r=session->run(o);bool valid=equal(r,reference);exact&=valid;
                for(auto m:r.overlapMetrics) drained&=m.submitted_batches==m.completed_batches;
                auto row=record(r,false);row["exact"]=valid;row["call"]=calls;row["elapsed_ms"]=ms(soakStart);row["recreate_ms"]=recreateMs;row["context_index"]=recreated;
                auto bytes=QJsonDocument(row).toJson(QJsonDocument::Compact)+'\n';if(stream.write(bytes)!=bytes.size() || !stream.flush()) throw std::runtime_error("Cannot persist soak sample");
                ++calls;if(!exact || !drained) throw std::runtime_error("Soak output or drain mismatch");
            }
            QJsonObject summary{{"calls",calls},{"contexts_recreated",recreated},{"duration_ms",ms(soakStart)},{"before",before},{"after",memory()},{"exact",exact},{"drained",drained},{"construct_ms",construct},{"first",firstRecord}};
            std::cout<<QJsonDocument(summary).toJson(QJsonDocument::Compact).constData()<<'\n';return exact&&drained?0:1;
        }
        for(int i=0;i<repeats;++i) {auto r=session->run(o);exact&=equal(r,first);runs.append(record(r,true));}
        QJsonObject out{{"construct_ms",construct},{"first",firstRecord},{"warmups_ms",warmups},{"runs",runs},{"exact_repeat",exact},{"final_memory",memory()}};
        std::cout<<QJsonDocument(out).toJson(QJsonDocument::Compact).constData()<<'\n';return exact?0:1;
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
