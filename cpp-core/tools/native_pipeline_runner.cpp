// SPDX-License-Identifier: BSD-2-Clause
// JSON job runner for the SAME native Session used by the GUI. Qt only decodes
// images and serializes audit records; all matching is C++/Metal.
#include "research_pipeline.hpp"
#include "qt_gray_loader.hpp"
#include "run_resource_probe.hpp"
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
QJsonObject resources(const fpm::diagnostics::RunResourceRecord& r) {
    return {{"available",r.available},{"wall_ms",r.wall_ms},
        {"process_user_ms",r.process_user_ms},{"process_system_ms",r.process_system_ms},
        {"minor_faults",double(r.minor_faults)},{"major_faults",double(r.major_faults)},
        {"voluntary_switches",double(r.voluntary_switches)},{"involuntary_switches",double(r.involuntary_switches)},
        {"host_load_available",r.host_load_available},{"host_busy_ticks",double(r.host_busy_ticks)},
        {"host_idle_ticks",double(r.host_idle_ticks)}};
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
    o["scale_support_enabled"]=r.scaleSupportEnabled;
    o["scale_support_refine_ms"]=r.scaleSupportMs;
    o["scale_support_eligible"]=double(r.scaleSupportEligible);
    o["scale_support_changed"]=double(r.scaleSupportChanged);
    o["scale_support_boundary"]=double(r.scaleSupportBoundary);
    o["scale_support_evaluations"]=double(r.scaleSupportEvaluations);
    o["scale_support_samples"]=double(r.scaleSupportSamples);
    o["refinement_mode"]=r.refinementMode;
    o["configured_refinement_backend"]=r.configuredRefinementBackend;
    o["refinement_candidates"]=double(r.refinementCandidates);
    o["source_generation"]=double(r.sourceGeneration);
    o["pose_storage_mode"]=r.poseStorage.mode;
    o["pose_host_image_bytes"]=double(r.poseStorage.host_image_bytes);
    o["pose_gpu_image_bytes"]=double(r.poseStorage.gpu_image_bytes);
    o["pose_shared_image_bytes"]=double(r.poseStorage.shared_image_bytes);
    o["pose_unique_image_bytes"]=double(r.poseStorage.unique_image_bytes);
    o["pose_create_ms"]=r.poseCreateMs;
    o["pose_input_mode"]=r.poseInputMode;
    o["pose_conversion_temporary_bytes"]=double(r.poseConversionTemporaryBytes);
    o["refinement_threads"]=int(r.residentKernel.threads);
    o["refinement_simd_width"]=int(r.residentKernel.simd_width);
    o["refinement_threadgroup_bytes"]=double(r.residentKernel.threadgroup_bytes);
    o["refinement_dispatches"]=double(r.resident.encoded_dispatches);
    o["refinement_command_buffers"]=double(r.resident.command_buffers);
    o["refinement_cpu_waits"]=double(r.resident.cpu_waits);
    o["refinement_auxiliary_bytes"]=double(r.resident.auxiliary_buffer_bytes);
    o["refinement_setup_ms"]=r.resident.setup_ms;
    o["refinement_encode_ms"]=r.resident.encode_ms;
    o["refinement_wait_ms"]=r.resident.wait_ms;
    o["refinement_verify_ms"]=r.resident.cpu_verify_ms;
    o["refinement_device_ms"]=r.pose.device_ms;
    QJsonArray diagnostics;
    for(const auto& p:r.fineSearch) {
        QJsonObject d;
#define FPM_FINE_FIELD(name) d[#name]=double(p.name)
        FPM_FINE_FIELD(profiling_enabled); FPM_FINE_FIELD(angle_reuse_enabled);
        FPM_FINE_FIELD(observe_ms); FPM_FINE_FIELD(plan_ms); FPM_FINE_FIELD(descriptor_ms);
        FPM_FINE_FIELD(warp_encode_ms); FPM_FINE_FIELD(warp_wait_ms); FPM_FINE_FIELD(warp_device_ms);
        FPM_FINE_FIELD(repair_all_ms); FPM_FINE_FIELD(guard_scan_ms); FPM_FINE_FIELD(ncc_encode_ms); FPM_FINE_FIELD(ncc_wait_ms);
        FPM_FINE_FIELD(calls); FPM_FINE_FIELD(requests); FPM_FINE_FIELD(within_call_duplicates); FPM_FINE_FIELD(cross_call_duplicates);
        FPM_FINE_FIELD(untracked_call_keys); FPM_FINE_FIELD(tracked_unique_keys); FPM_FINE_FIELD(history_limit_keys);
        FPM_FINE_FIELD(history_key_payload_bytes); FPM_FINE_FIELD(history_bucket_count); FPM_FINE_FIELD(peak_call_key_capacity_bytes);
        FPM_FINE_FIELD(logical_requests); FPM_FINE_FIELD(reused_maps); FPM_FINE_FIELD(cache_peak_bytes);
        FPM_FINE_FIELD(warp_batches); FPM_FINE_FIELD(zero_repair_batches); FPM_FINE_FIELD(guard_words); FPM_FINE_FIELD(nonzero_guard_words);
#undef FPM_FINE_FIELD
        diagnostics.append(d);
    }
    o["fine_search"]=diagnostics;
    QJsonArray fusion;
    for (const auto& p:r.fineFusion) {
        QJsonObject d;
#define FPM_FUSION_FIELD(name) d[#name]=double(p.name)
        FPM_FUSION_FIELD(enabled); FPM_FUSION_FIELD(correction_ms); FPM_FUSION_FIELD(fused_wait_ms);
        FPM_FUSION_FIELD(fused_device_ms); FPM_FUSION_FIELD(fused_batches); FPM_FUSION_FIELD(fallback_batches);
        FPM_FUSION_FIELD(dense_maps); FPM_FUSION_FIELD(changed_pixels); FPM_FUSION_FIELD(corrected_windows);
#undef FPM_FUSION_FIELD
        fusion.append(d);
    }
    o["fine_fusion"]=fusion;
    QJsonArray work;for(size_t i=0;i<r.coarse.size();++i) {
        auto a=r.coarse[i];auto b=r.fine[i];auto c=r.overlapMetrics[i];
        const auto d=r.fineReuse[i];
        work.append(QJsonObject{{"maps",double(a.score_maps)},{"windows",double(a.windows)},{"fine_maps",double(b.maps)},
            {"fine_ncc_mode",d.ncc_mode},{"fine_row_bytes",double(d.row_buffer_bytes)},
            {"fine_row_dispatches",double(d.row_dispatches)},{"fine_stats_dispatches",double(d.statistics_dispatches)},
            {"fine_stats_device_ms",d.stats_device_ms},{"fine_rotation_ms",b.rotation_ms},{"fine_repair_ms",b.repair_ms},
            {"fine_ncc_ms",b.ncc_ms},{"fine_normalization_ms",b.normalization_ms},{"fine_wait_ms",b.wait_ms},
            {"fine_commands",double(b.command_buffers)},{"fine_uploads",double(b.source_uploads)},
            {"fine_upload_bytes",double(b.source_upload_bytes)},{"fine_roi_pixels",double(b.roi_pixels)},
            {"fine_repair_pixels",double(b.repair_pixels)},{"fine_peak_batch",double(b.peak_batch_maps)},
            {"device_ms",a.device_ms},{"wait_ms",a.wait_ms},{"copy_ms",a.upload_ms},{"normalization_ms",a.normalization_ms},
            {"async_submitted",double(c.submitted_batches)},{"async_completed",double(c.completed_batches)},
            {"async_maps",double(c.submitted_maps)},{"async_fallback_maps",double(c.fallback_maps)},{"async_max_pending",double(c.max_pending)},
            {"async_device_ms",c.device_ms},{"async_wait_ms",c.finish_wait_ms},{"async_copy_ms",c.copy_ms}});
    }o["work"]=work;
    if(r.sparseEnabled) {
        const auto& s=r.sparse;
        o["sparse"]=QJsonObject{{"enabled",true},{"source_levels",1},{"feature_count",s.featureCount},
            {"response_mode",s.responseMode},{"lookup_bytes",double(s.lookupBytes)},{"search_backend",s.selectedBackend},
            {"gradient_mode",s.gradientMode},{"gradient_lookup_bytes",double(s.gradientLookupBytes)},
            {"metal_submission_mode",s.metalSubmissionMode},
            {"gpu_setup_ms",s.gpuSetupMs},{"upload_ms",s.uploadMs},{"encode_ms",s.encodeMs},{"wait_ms",s.waitMs},
            {"device_ms",s.deviceMs},{"readback_ms",s.readbackMs},{"gpu_buffer_bytes",double(s.gpuBufferBytes)},
            {"command_buffers",double(s.commandBuffers)},{"dispatches",double(s.dispatches)},
            {"cpu_waits",double(s.cpuWaits)},{"peak_overflow_retries",double(s.peakOverflowRetries)},
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
        const auto loadModeName=j["gray_load_mode"].toString("direct_gray8");
        if(loadModeName!="direct_gray8" && loadModeName!="legacy_rgb") throw std::invalid_argument("Invalid gray load mode");
        const auto loadMode=loadModeName=="direct_gray8"?fpm::io::GrayLoadMode::DirectGray8:fpm::io::GrayLoadMode::LegacyRgb;
        fpm::io::GrayLoadInfo sourceLoad,templateLoad;auto loadStart=Clock::now();
        const auto sourceFrames=j["source_frames"].toArray();
        if(j.contains("source_frames") && (sourceFrames.isEmpty() || sourceFrames.size()>64))
            throw std::invalid_argument("source_frames must contain 1..64 image paths");
        for(const auto& path:sourceFrames) if(!path.isString() || path.toString().isEmpty())
            throw std::invalid_argument("source_frames paths must be nonempty strings");
        auto src=fpm::io::readGrayImage(sourceFrames.isEmpty()?j["source"].toString():sourceFrames[0].toString(),loadMode,&sourceLoad);
        auto tmp=fpm::io::readGrayImage(j["template"].toString(),loadMode,&templateLoad);
        const double decodeMs=ms(loadStart);const auto afterLoadMemory=memory();
        auto loadInfo=[](const fpm::io::GrayLoadInfo& info) {return QJsonObject{{"direct_gray8",info.directGray8},{"rgb_temporary_bytes",double(info.rgbTemporaryBytes)}};};
        const QJsonObject load{{"mode",loadModeName},{"decode_ms",decodeMs},{"source",loadInfo(sourceLoad)},{"template",loadInfo(templateLoad)}};
        Settings settings;
        settings.backend=j["backend"].toInt(1);settings.refinementBackend=j["refinement_backend"].toInt(-1);settings.minReducedArea=j["min_area"].toInt(256);settings.scales.clear();
        settings.poseStorageMode=j["pose_storage_mode"].toInt(0);
        settings.poseInputMode=j["pose_input_mode"].toInt(0);
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
        settings.sparse.responseMode=j["sparse_response_mode"].toInt(0);
        settings.sparse.searchBackend=j["sparse_search_backend"].toInt(0);
        settings.sparse.cpuGradientMode=j["sparse_cpu_gradient_mode"].toInt(0);
        settings.sparse.metalSubmissionMode=j["sparse_metal_submission_mode"].toInt(0);
        settings.residentMode=j["pose_resident_mode"].toInt(0);
        settings.cpuThreads=j["cpu_threads"].toInt(8);settings.sparse.cpuThreads=settings.cpuThreads;
        Options o;o.nccMode=j["ncc_mode"].toInt();o.overlap=j["overlap_enabled"].toBool();o.refine=j["refine"].toBool(true);
        o.fineNccMode=j["fine_ncc_mode"].toInt(0);
        o.fineFused=j["fine_fused"].toBool(false);
        o.fineProfiling=j["fine_profiling"].toBool(false); o.fineAngleReuse=j["fine_angle_reuse"].toBool(false);
        o.match.maxTargets=j["limit"].toInt(70);o.match.minScore=j["score"].toDouble(.8);o.match.maxOverlap=j["overlap"].toDouble(.2);o.match.toleranceAngle=j["angle"].toDouble(180);
        o.match.stopLayer=j["stop_layer"].toInt(1);o.match.subpixel=j["subpixel"].toBool();o.match.invertSource=j["invert_source"].toBool();
        o.proposalScore=j["proposal_score"].toDouble(.5);
        o.scaleSupportRefine=j["scale_support_refine"].toBool(false);
        const int repeats=j["repeats"].toInt(5);const double warm=j["warmup_seconds"].toDouble(1.),soak=j["soak_seconds"].toDouble(0);
        if(repeats<1 || repeats>10000 || !std::isfinite(warm) || warm<0 || warm>60 || !std::isfinite(soak) || soak<0 || soak>3600) throw std::invalid_argument("Invalid repetitions or duration");
        const bool resourceDiagnostics=j["run_resource_diagnostics"].toBool(false);
        if(resourceDiagnostics && (warm>0 || soak>0)) throw std::invalid_argument("Run resource diagnostics require first/repeat mode without warmup or soak");
        if(!sourceFrames.isEmpty()) {
            const auto mode=j["sequence_mode"].toString("reuse");
            if(mode!="reuse" && mode!="fresh") throw std::invalid_argument("sequence_mode must be reuse or fresh");
            if(warm>0 || resourceDiagnostics) throw std::invalid_argument("Source sequences require warmup_seconds=0 and resource diagnostics disabled");
            QFile stream;
            if(soak>0) {
                stream.setFileName(j["stream_path"].toString());
                if(!stream.open(QIODevice::WriteOnly|QIODevice::Truncate)) throw std::invalid_argument("Cannot open sequence soak stream");
            }
            std::unique_ptr<Session> session;
            std::vector<Result> reference(size_t(sourceFrames.size()));
            QJsonArray rows;bool exact=true,drained=true;int calls=0,cycle=0;
            const auto sequenceStart=Clock::now();
            do {
                for(int index=0;index<sourceFrames.size();++index) {
                    double decode=0;
                    if(calls) {const auto t=Clock::now();src=fpm::io::readGrayImage(sourceFrames[index].toString(),loadMode);decode=ms(t);}
                    const auto frameStart=Clock::now();
                    double release=0,construct=0;SourceUpdate update;const bool create=mode=="fresh" || !session;
                    if(create) {
                        const auto t=Clock::now();session.reset();release=ms(t);
                        const auto c=Clock::now();session=std::make_unique<Session>(src.view(),tmp.view(),settings);construct=ms(c);
                    } else update=session->updateSource(src.view());
                    auto result=session->run(o);const double frameWall=ms(frameStart);
                    bool valid=true;
                    if(cycle==0) reference[size_t(index)]=result;
                    else valid=equal(result,reference[size_t(index)]);
                    exact&=valid;
                    for(auto m:result.overlapMetrics) drained&=m.submitted_batches==m.completed_batches;
                    auto row=record(result,soak<=0);
                    row["source_index"]=index;row["sequence_cycle"]=cycle;row["call"]=calls;
                    row["source_path"]=sourceFrames[index];row["source_width"]=src.width();row["source_height"]=src.height();
                    row["source_decode_ms"]=decode;row["release_ms"]=release;row["construct_ms"]=construct;
                    row["source_update_ms"]=update.updateMs;row["source_copy_bytes"]=double(update.copiedBytes);
                    row["retained_template_models"]=double(update.retainedModels);row["pose_invalidated"]=update.poseInvalidated;
                    row["source_dimensions_changed"]=update.dimensionsChanged;row["frame_wall_ms"]=frameWall;
                    row["exact_sequence_repeat"]=valid;row["reference_cycle"]=cycle==0;
                    row["elapsed_ms"]=ms(sequenceStart);
                    if(soak>0) {
                        const auto bytes=QJsonDocument(row).toJson(QJsonDocument::Compact)+'\n';
                        if(stream.write(bytes)!=bytes.size() || !stream.flush()) throw std::runtime_error("Cannot persist source sequence sample");
                    } else rows.append(row);
                    ++calls;if(!exact || !drained) throw std::runtime_error("Source sequence output or drain mismatch");
                }
                ++cycle;
            } while(cycle<repeats || (soak>0 && ms(sequenceStart)<soak*1000));
            QJsonObject out{{"sequence_mode",mode},{"source_frames",sourceFrames},{"cycles",cycle},{"calls",calls},
                {"duration_ms",ms(sequenceStart)},{"runs",rows},{"exact_sequence_repeat",exact},{"drained",drained},
                {"load",load},{"after_load_memory",afterLoadMemory},{"final_memory",memory()},
                {"comparison_reference","first cycle per frame index; independent fresh/reuse comparison belongs to parent harness"},
                {"frame_wall_scope","source update or release+construct and run; excludes image decode, comparison, JSON and output"}};
            const auto releaseStart=Clock::now();session.reset();out["final_release_ms"]=ms(releaseStart);
            out["after_release_memory"]=memory();
            std::cout<<QJsonDocument(out).toJson(QJsonDocument::Compact).constData()<<'\n';return exact&&drained?0:1;
        }
        auto start=Clock::now();auto session=std::make_unique<Session>(src.view(),tmp.view(),settings);const double construct=ms(start);
        fpm::diagnostics::RunResourceProbe firstProbe(resourceDiagnostics);
        Result first=session->run(o);const auto firstResources=firstProbe.finish();QJsonObject firstRecord=record(first,true);QJsonArray warmups,runs;
        if(resourceDiagnostics) firstRecord["resources"]=resources(firstResources);
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
            summary["load"]=load;summary["after_load_memory"]=afterLoadMemory;
            std::cout<<QJsonDocument(summary).toJson(QJsonDocument::Compact).constData()<<'\n';return exact&&drained?0:1;
        }
        for(int i=0;i<repeats;++i) {
            fpm::diagnostics::RunResourceProbe probe(resourceDiagnostics);
            auto r=session->run(o);const auto runResources=probe.finish();exact&=equal(r,first);
            auto row=record(r,true);if(resourceDiagnostics) row["resources"]=resources(runResources);runs.append(row);
        }
        QJsonObject out{{"construct_ms",construct},{"first",firstRecord},{"warmups_ms",warmups},{"runs",runs},{"exact_repeat",exact},{"final_memory",memory()}};
        out["load"]=load;out["after_load_memory"]=afterLoadMemory;
        std::cout<<QJsonDocument(out).toJson(QJsonDocument::Compact).constData()<<'\n';return exact?0:1;
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
