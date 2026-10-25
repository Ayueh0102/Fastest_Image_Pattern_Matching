// SPDX-License-Identifier: BSD-2-Clause
// One method/backend per process. Existing pose NCC and independent chroma
// verification share supplied poses. This worker does not search or read GT.
#include "color_illuminant.hpp"
#include "color_verify.hpp"
#include "pose_ncc.h"
#include "research_pipeline.hpp"
#include "fpm/matcher.hpp"
#include <QCoreApplication>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start)
{ return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
QJsonObject memory()
{
    rusage usage{};
    if (getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("RSS unavailable");
    double peak = usage.ru_maxrss, current = -1;
#ifdef __APPLE__
    mach_task_basic_info info{}; mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),MACH_TASK_BASIC_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS)
        current = info.resident_size;
#else
    peak *= 1024;
#endif
    return {{"peak_rss_bytes",peak},{"current_rss_bytes",current}};
}
double number(const QJsonObject& job,const char* key,double fallback,double low,double high)
{
    if (!job.contains(key)) return fallback;
    const auto value = job[key];
    if (!value.isDouble() || !std::isfinite(value.toDouble()) || value.toDouble()<low || value.toDouble()>high)
        throw std::invalid_argument(std::string("Invalid ")+key);
    return value.toDouble();
}
int integer(const QJsonObject& job,const char* key,int fallback,int low,int high)
{
    const double value = number(job,key,fallback,low,high);
    if (std::floor(value)!=value) throw std::invalid_argument(std::string("Noninteger ")+key);
    return int(value);
}
bool boolean(const QJsonObject& job,const char* key,bool fallback)
{
    if (!job.contains(key)) return fallback;
    if (!job[key].isBool()) throw std::invalid_argument(std::string("Invalid boolean ")+key);
    return job[key].toBool();
}
void configureSearch(const QJsonObject& job,fpm::research::Settings& s,fpm::research::Options& o)
{
    const QStringList allowed{"backend","refinement_backend","scales","angle","limit","score","proposal_score",
        "min_area","cpu_threads","pose_resident_mode","pose_storage_mode","pose_input_mode","fine_ncc_mode",
        "fine_fused","fine_angle_reuse","ncc_mode","overlap_enabled","overlap","stop_layer","subpixel",
        "refine","scale_support_refine"};
    for(auto it=job.begin();it!=job.end();++it)if(!allowed.contains(it.key()))
        throw std::invalid_argument("Unknown search option: "+it.key().toStdString());
    s.backend=integer(job,"backend",1,0,1);
    s.refinementBackend=integer(job,"refinement_backend",s.backend,0,1);
    s.minReducedArea=integer(job,"min_area",256,1,10000000);
    s.cpuThreads=integer(job,"cpu_threads",8,1,256);
    s.residentMode=integer(job,"pose_resident_mode",s.refinementBackend?2:0,0,2);
    s.poseStorageMode=integer(job,"pose_storage_mode",s.refinementBackend?1:0,0,1);
    s.poseInputMode=integer(job,"pose_input_mode",1,0,1);
    if(job.contains("scales")) {
        if(!job["scales"].isArray() || job["scales"].toArray().isEmpty() || job["scales"].toArray().size()>64)
            throw std::invalid_argument("Expected 1..64 search scales");
        s.scales.clear();
        for(const auto& scale:job["scales"].toArray()) {
            if(!scale.isDouble() || !std::isfinite(scale.toDouble()) || scale.toDouble()<=0)
                throw std::invalid_argument("Invalid search scale");
            s.scales.push_back(scale.toDouble());
        }
    }
    o.nccMode=integer(job,"ncc_mode",0,0,3);
    o.fineNccMode=integer(job,"fine_ncc_mode",s.backend?2:0,0,3);
    o.fineFused=boolean(job,"fine_fused",s.backend!=0);
    o.fineAngleReuse=boolean(job,"fine_angle_reuse",s.backend!=0);
    o.overlap=boolean(job,"overlap_enabled",false);
    o.refine=boolean(job,"refine",true);
    o.scaleSupportRefine=boolean(job,"scale_support_refine",false);
    o.proposalScore=number(job,"proposal_score",.5,0,1);
    o.match.maxTargets=integer(job,"limit",100,1,10000);
    o.match.minScore=number(job,"score",.65,0,1);
    o.match.maxOverlap=number(job,"overlap",.2,0,1);
    o.match.toleranceAngle=number(job,"angle",180,0,180);
    o.match.stopLayer=integer(job,"stop_layer",1,0,1);
    o.match.subpixel=boolean(job,"subpixel",false);
}
QJsonArray searchMatches(const fpm::research::Result& result)
{
    QJsonArray out;
    for(std::size_t i=0;i<result.matches.size();++i) {
        const auto& match=result.matches[i];QJsonArray corners;
        for(const auto& corner:match.corners)corners.append(QJsonArray{corner.x,corner.y});
        out.append(QJsonObject{{"pos_x",match.center.x},{"pos_y",match.center.y},
            {"angle",match.angle},{"scale",result.scales[i]},{"score",match.score},{"corners",corners}});
    }
    return out;
}
QImage readRgb(const QJsonObject& job,const char* key)
{
    if (!job[key].isString()) throw std::invalid_argument(std::string("Missing path ")+key);
    QImage image(job[key].toString());
    if (image.isNull()) throw std::invalid_argument(std::string("Cannot decode ")+key);
    // Explicit research assumption: bytes are sRGB. Embedded ICC profiles are
    // not converted here; callers must normalize color-managed images first.
    return image.convertToFormat(QImage::Format_RGB888);
}
fpm::research::Rgb8View view(const QImage& image)
{ return {image.constBits(),image.width(),image.height(),std::size_t(image.bytesPerLine())}; }
std::vector<std::uint8_t> gray(const QImage& image)
{
    std::vector<std::uint8_t> values(std::size_t(image.width())*image.height());
    for (int y=0;y<image.height();++y) for(int x=0;x<image.width();++x) {
        const auto* p = image.constScanLine(y)+3*x;
        values[std::size_t(y)*image.width()+x]=fpm::grayFromRgb(p[0],p[1],p[2]);
    }
    return values;
}
QJsonObject colorJson(const fpm::research::ColorVerifyResult& result)
{
    return {{"status",result.status},{"eligible_pixels",double(result.eligiblePixels)},
        {"valid_pixels",double(result.validPixels)},{"changed_pixels",double(result.changedPixels)},
        {"valid_fraction",result.validFraction},{"mean_distance",result.meanDistance},
        {"changed_fraction",result.changedFraction}};
}
QJsonArray triple(const std::array<double,3>& values){return {values[0],values[1],values[2]};}
QJsonObject illuminantJson(const fpm::research::IlluminantChromaResult& result)
{
    return {{"status",result.status},{"eligible_pixels",double(result.eligiblePixels)},
        {"valid_pixels",double(result.validPixels)},{"changed_pixels",double(result.changedPixels)},
        {"clipped_pixels",double(result.clippedPixels)},{"valid_fraction",result.validFraction},
        {"mean_distance",result.meanDistance},{"changed_fraction",result.changedFraction},
        {"region_distance",result.regionDistance},{"uncorrected_mean_distance",result.uncorrectedMeanDistance},
        {"uncorrected_region_distance",result.uncorrectedRegionDistance},
        {"reference_eligible",double(result.referenceEligible)},{"reference_samples",double(result.referenceSamples)},
        {"reference_fraction",result.referenceFraction},{"reference_clip_fraction",triple(result.referenceClipFraction)},
        {"gain",triple(result.gain)}};
}
// Linear reference model only: the per-pixel slopes (absent for the constant model, whose
// score records stay as in round 46).
QJsonObject illuminantJson(const fpm::research::IlluminantChromaResult& result,bool linear)
{
    auto out=illuminantJson(result);
    if(linear) {
        out.insert("reference_slope_u",triple(result.referenceSlopeU));
        out.insert("reference_slope_v",triple(result.referenceSlopeV));
    }
    return out;
}
// chroma_vk options; every key is rejected for the other methods.
const QStringList illuminantKeys{"reference_mask","illuminant_reference","reference_rect","template_reference_rgb",
    "reference_clip_policy","saturation_level","max_reference_clip_fraction","min_reference_fraction","reference_model"};
fpm::research::IlluminantChromaConfig illuminantConfig(const QJsonObject& job,const fpm::research::ColorVerifyConfig& chroma)
{
    fpm::research::IlluminantChromaConfig config;
    config.chroma=chroma;
    const auto source=job["illuminant_reference"].toString("object");
    if(job.contains("illuminant_reference") && !job["illuminant_reference"].isString()) throw std::invalid_argument("Invalid illuminant_reference");
    if(source=="object") config.source=fpm::research::IlluminantSource::Object;
    else if(source=="source_rect") config.source=fpm::research::IlluminantSource::SourceRect;
    else throw std::invalid_argument("illuminant_reference must be object or source_rect");
    const auto policy=job["reference_clip_policy"].toString("rank");
    if(job.contains("reference_clip_policy") && !job["reference_clip_policy"].isString()) throw std::invalid_argument("Invalid reference_clip_policy");
    if(policy=="rank") config.clipPolicy=fpm::research::ReferenceClipPolicy::Rank;
    else if(policy=="drop") config.clipPolicy=fpm::research::ReferenceClipPolicy::Drop;
    else if(policy=="none") config.clipPolicy=fpm::research::ReferenceClipPolicy::None;
    else throw std::invalid_argument("reference_clip_policy must be rank, drop or none");
    const auto model=job["reference_model"].toString("constant");
    if(job.contains("reference_model") && !job["reference_model"].isString()) throw std::invalid_argument("Invalid reference_model");
    if(model=="constant") config.model=fpm::research::ReferenceModel::Constant;
    else if(model=="linear") config.model=fpm::research::ReferenceModel::Linear;
    else throw std::invalid_argument("reference_model must be constant or linear");
    if(config.model==fpm::research::ReferenceModel::Linear && config.source!=fpm::research::IlluminantSource::Object)
        throw std::invalid_argument("reference_model linear requires the object reference");
    config.saturationLevel=integer(job,"saturation_level",255,1,255);
    config.maxReferenceClipFraction=number(job,"max_reference_clip_fraction",.45,0,.999999);
    config.minReferenceFraction=number(job,"min_reference_fraction",.5,1e-12,1);
    const bool rect=config.source==fpm::research::IlluminantSource::SourceRect;
    if(rect!=job.contains("reference_rect") || (!rect && job.contains("template_reference_rgb")) || rect==job.contains("reference_mask"))
        throw std::invalid_argument("object needs reference_mask; source_rect needs reference_rect and may set template_reference_rgb");
    if(rect) {
        const auto values=job["reference_rect"].toArray();
        if(!job["reference_rect"].isArray() || values.size()!=4) throw std::invalid_argument("reference_rect needs [x0,y0,x1,y1]");
        for(int i=0;i<4;++i) {
            if(!values[i].isDouble() || std::floor(values[i].toDouble())!=values[i].toDouble() || std::abs(values[i].toDouble())>1e9)
                throw std::invalid_argument("reference_rect needs integers");
            config.referenceRect[i]=int(values[i].toDouble());
        }
        if(job.contains("template_reference_rgb")) {
            const auto rgb=job["template_reference_rgb"].toArray();
            if(!job["template_reference_rgb"].isArray() || rgb.size()!=3) throw std::invalid_argument("template_reference_rgb needs 3 numbers");
            for(int c=0;c<3;++c) {
                if(!rgb[c].isDouble()) throw std::invalid_argument("template_reference_rgb needs 3 numbers");
                config.templateReference[c]=rgb[c].toDouble();
            }
        }
    }
    return config;
}
QImage readMask(const QJsonObject& job,const char* key,const QImage& templ)
{
    if(!job[key].isString()) throw std::invalid_argument(std::string("Invalid mask path ")+key);
    QImage mask=QImage(job[key].toString()).convertToFormat(QImage::Format_Grayscale8);
    if(mask.isNull() || mask.size()!=templ.size()) throw std::invalid_argument("Mask must match template dimensions");
    return mask;
}
}

int main(int argc,char** argv)
{
    QCoreApplication app(argc,argv);
    try {
        if (argc!=2) throw std::invalid_argument("Usage: fpm_color_verify_runner job.json");
        QFile file(QString::fromLocal8Bit(argv[1]));
        if (!file.open(QIODevice::ReadOnly)) throw std::invalid_argument("Cannot open job");
        QJsonParseError parse{};
        const auto document = QJsonDocument::fromJson(file.readAll(),&parse);
        if (parse.error!=QJsonParseError::NoError || !document.isObject()) throw std::invalid_argument("Invalid job JSON");
        const auto job = document.object();
        if (job.contains("truth") || job.contains("ground_truth") || job.contains("labels"))
            throw std::invalid_argument("Evaluation labels must stay outside the worker");
        const auto method = job["method"].toString(), backend = job["backend"].toString("cpu");
        if (method!="gray_ncc" && method!="rgb_ncc" && method!="chroma" && method!="chroma_vk" && method!="geometry_only") throw std::invalid_argument("Invalid method");
        if (backend!="cpu" && backend!="metal") throw std::invalid_argument("Invalid backend");
        const bool chromaMethod = method=="chroma" || method=="chroma_vk";
        if (chromaMethod && backend!="cpu") throw std::invalid_argument("Chroma currently supports CPU only");
        if (!chromaMethod && job.contains("mask")) throw std::invalid_argument("Existing NCC does not support template masks");
        if (method!="chroma_vk") for (const auto& key : illuminantKeys) if (job.contains(key))
            throw std::invalid_argument("Illuminant option requires chroma_vk: "+key.toStdString());
        const int threads = integer(job,"threads",1,1,256);
        if (chromaMethod && threads!=1) throw std::invalid_argument("Chroma currently uses one CPU thread");
        const int warmups = integer(job,"warmups",2,0,10000), repeats = integer(job,"repeats",7,1,10000);
        const bool integrated=job.contains("search");
        if (integrated && (!job["search"].isObject() || job.contains("poses")))
            throw std::invalid_argument("search must be an object and cannot be combined with poses");
        if (!integrated && !job["poses"].isArray()) throw std::invalid_argument("Missing poses or search");
        if (method=="geometry_only" && (!integrated || backend!="cpu"))
            throw std::invalid_argument("geometry_only requires search; verification backend must be cpu");
        const auto poseArray = job["poses"].toArray();
        if (!integrated && (poseArray.isEmpty() || poseArray.size()>1000000)) throw std::invalid_argument("Expected 1..1000000 poses");
        std::vector<float> poses;
        poses.reserve(std::size_t(poseArray.size())*4);
        for (const auto& value : poseArray) {
            if (!value.isArray() || value.toArray().size()!=4) throw std::invalid_argument("Each pose requires four numbers");
            for (const auto& entry : value.toArray()) {
                const float number = float(entry.toDouble());
                if (!entry.isDouble() || !std::isfinite(number)) throw std::invalid_argument("Invalid pose number");
                poses.push_back(number);
            }
        }
        auto start = Clock::now();
        QImage source = readRgb(job,"source"), templ = readRgb(job,"template"), mask, referenceMask;
        const int sw=source.width(), sh=source.height(), tw=templ.width(), th=templ.height();
        if (job.contains("mask")) {
            if (!job["mask"].isString()) throw std::invalid_argument("Invalid mask path");
            mask = QImage(job["mask"].toString()).convertToFormat(QImage::Format_Grayscale8);
            if (mask.isNull() || mask.size()!=templ.size()) throw std::invalid_argument("Mask must match template dimensions");
        }
        if (job.contains("reference_mask")) referenceMask=readMask(job,"reference_mask",templ);
        const double decodeMs = milliseconds(start);
        const auto decodedMemory = memory();
        fpm::research::Settings searchSettings;
        fpm::research::Options searchOptions;
        std::unique_ptr<fpm::research::Session> session;
        double searchConstructMs=0;
        if(integrated) {
            configureSearch(job["search"].toObject(),searchSettings,searchOptions);
            start=Clock::now();
            const auto sourceGray=gray(source),templateGray=gray(templ);
            session=std::make_unique<fpm::research::Session>(
                fpm::ImageView{sourceGray.data(),sw,sh,std::size_t(sw)},
                fpm::ImageView{templateGray.data(),tw,th,std::size_t(tw)},searchSettings);
            searchConstructMs=milliseconds(start);
        }
        fpm::research::ColorVerifyConfig config;
        config.distanceThreshold=number(job,"chroma_distance_threshold",.04,0,1);
        config.minLinearSum=number(job,"min_linear_sum",.03,1e-12,3);
        config.minValidFraction=number(job,"min_valid_fraction",.95,1e-12,1);
        std::unique_ptr<fpm::research::ColorVerifier> verifier;
        std::unique_ptr<fpm::research::IlluminantChromaVerifier> illuminant;
        const auto vkConfig = method=="chroma_vk"?illuminantConfig(job,config):fpm::research::IlluminantChromaConfig{};
        std::unique_ptr<void,decltype(&fpm_pose_destroy)> handle(nullptr,&fpm_pose_destroy);
        char error[1024]{};
        start = Clock::now();
        if (method=="chroma") {
            verifier=std::make_unique<fpm::research::ColorVerifier>(view(source),view(templ),config,
                mask.isNull()?nullptr:mask.constBits(),mask.isNull()?0:std::size_t(mask.bytesPerLine()));
        } else if (method=="chroma_vk") {
            illuminant=std::make_unique<fpm::research::IlluminantChromaVerifier>(view(source),view(templ),
                mask.isNull()?nullptr:mask.constBits(),mask.isNull()?0:std::size_t(mask.bytesPerLine()),
                referenceMask.isNull()?nullptr:referenceMask.constBits(),
                referenceMask.isNull()?0:std::size_t(referenceMask.bytesPerLine()),vkConfig);
        } else if (method!="geometry_only") {
            std::vector<std::uint8_t> sourceGray,templateGray;
            const bool isGray = method=="gray_ncc";
            if (isGray) { sourceGray=gray(source); templateGray=gray(templ); }
            handle.reset(fpm_pose_create_u8(backend=="metal"?1:0,threads,
                isGray?sourceGray.data():source.constBits(),sw,sh,isGray?std::uint64_t(sw):std::uint64_t(source.bytesPerLine()),
                isGray?templateGray.data():templ.constBits(),tw,th,isGray?std::uint64_t(tw):std::uint64_t(templ.bytesPerLine()),
                isGray?1:3,0,0,error,sizeof(error)));
            if (!handle) throw std::runtime_error(error);
        }
        const double colorConstructMs = method=="geometry_only"?0:milliseconds(start);
        source=QImage();templ=QImage();mask=QImage();referenceMask=QImage(); // decode buffers released before timed scoring
        std::vector<float> scores(std::size_t(poseArray.size()));
        std::vector<fpm::research::ColorVerifyResult> colors(std::size_t(poseArray.size()));
        std::vector<fpm::research::IlluminantChromaResult> normalized(illuminant?std::size_t(poseArray.size()):0);
        auto run = [&]() {
            FpmPoseMetrics metrics{}; metrics.device_ms=-1;
            const auto begin = Clock::now();
            fpm::research::Result searchResult;
            double searchMs=0;
            if(session) {
                searchResult=session->run(searchOptions);
                searchMs=milliseconds(begin);
            }
            const auto verificationBegin=Clock::now();
            if(session && method!="geometry_only") {
                poses.resize(searchResult.matches.size()*4);
                for(std::size_t i=0;i<searchResult.matches.size();++i) {
                    const auto& match=searchResult.matches[i];
                    poses[4*i]=float(match.center.x-sw/2.);poses[4*i+1]=float(sh/2.-match.center.y);
                    poses[4*i+2]=float(match.angle);poses[4*i+3]=float(std::log(searchResult.scales[i]));
                }
                scores.resize(searchResult.matches.size());colors.resize(searchResult.matches.size());
                if (illuminant) normalized.resize(searchResult.matches.size());
            }
            if (verifier) {
                for (std::size_t i=0;i<colors.size();++i)
                    colors[i]=verifier->score({poses[4*i],poses[4*i+1],poses[4*i+2],poses[4*i+3]});
            } else if (illuminant) {
                for (std::size_t i=0;i<normalized.size();++i)
                    normalized[i]=illuminant->score({poses[4*i],poses[4*i+1],poses[4*i+2],poses[4*i+3]});
            } else if (handle && !scores.empty() && fpm_pose_score(handle.get(),poses.data(),int(scores.size()),scores.data(),&metrics,error,sizeof(error)))
                throw std::runtime_error(error);
            const double verificationMs=method=="geometry_only"?0:milliseconds(verificationBegin);
            const double wallMs = method=="geometry_only"?searchMs:milliseconds(begin);
            // Serialization and RSS queries are outside algorithm timing.
            QJsonArray out;
            if (method=="geometry_only") for(const auto& match:searchResult.matches)out.append(match.score);
            else if (verifier) for (const auto& color : colors) out.append(colorJson(color));
            else if (illuminant) for (const auto& color : normalized)
                out.append(illuminantJson(color,vkConfig.model==fpm::research::ReferenceModel::Linear));
            else for (auto score : scores) out.append(score);
            return QJsonObject{{"wall_ms",wallMs},{"pipeline_ms",wallMs},{"search_ms",searchMs},
                {"search_algorithm_ms",searchResult.totalMs},{"verification_ms",verificationMs},
                {"search_matches",searchMatches(searchResult)},
                {"search_diagnostics",QJsonObject{{"prepare_ms",searchResult.prepareMs},{"proposal_ms",searchResult.proposalMs},
                    {"refine_ms",searchResult.refineMs},{"nms_ms",searchResult.nmsMs},
                    {"frame_host_bytes",double(searchResult.frame.host_pyramid_bytes)},
                    {"frame_coarse_gpu_bytes",double(searchResult.frame.coarse_gpu_bytes)},
                    {"frame_fine_gpu_bytes",double(searchResult.frame.fine_gpu_bytes)},
                    {"pose_host_bytes",double(searchResult.pose.host_buffer_bytes)},
                    {"pose_gpu_bytes",double(searchResult.pose.gpu_buffer_bytes)}}},
                {"api_wall_ms",verifier||illuminant?verificationMs:metrics.wall_ms},
                {"device_ms",metrics.device_ms},{"scores",out},{"memory",memory()},
                {"host_buffer_bytes",double(metrics.host_buffer_bytes)},
                {"gpu_buffer_bytes",double(metrics.gpu_buffer_bytes)},
                {"gpu_allocated_bytes",double(metrics.gpu_allocated_bytes)}};
        };
        const auto first = run();
        QJsonArray warmupResults,runs;
        bool exact = true;
        for (int i=0;i<warmups;++i) { const auto result=run();exact &= result["scores"]==first["scores"] && result["search_matches"]==first["search_matches"];warmupResults.append(result["wall_ms"]); }
        for (int i=0;i<repeats;++i) { const auto result=run();exact &= result["scores"]==first["scores"] && result["search_matches"]==first["search_matches"];runs.append(result); }
        FpmPoseMetrics retained{};
        if (handle && fpm_pose_memory(handle.get(),&retained)) throw std::runtime_error("Cannot query pose memory");
        QJsonObject out{{"schema",1},{"method",method},{"backend",backend},{"threads",threads},
            {"integrated_search",integrated},{"search_config",job["search"]},
            {"pose_count",integrated?first["search_matches"].toArray().size():int(poses.size()/4)},
            {"source_size",QJsonArray{sw,sh}},{"template_size",QJsonArray{tw,th}},
            {"decode_ms",decodeMs},{"construct_ms",searchConstructMs+colorConstructMs},
            {"search_construct_ms",searchConstructMs},{"color_construct_ms",colorConstructMs},{"decoded_memory",decodedMemory},
            {"first",first},{"warmups_ms",warmupResults},{"runs",runs},{"exact_repeat",exact},
            {"retained_host_bytes",double(verifier?verifier->retainedBytes():illuminant?illuminant->retainedBytes():retained.host_buffer_bytes)},
            {"retained_gpu_bytes",double(retained.gpu_buffer_bytes)},{"final_memory",memory()},
            {"color_config",QJsonObject{{"distance_threshold",config.distanceThreshold},{"min_linear_sum",config.minLinearSum},
                {"min_valid_fraction",config.minValidFraction},{"masked",job.contains("mask")}}},
            {"scope",integrated?"gray Session full-image search then candidate color verification; no anomaly threshold training":
                "known pose verification only; no object search or anomaly threshold training"},
            {"rgb_ncc_definition","per-channel mean subtraction, joint energy normalization; channel offsets cancel"},
            {"chroma_definition","sRGB LUT to linear RGB; normalized chromaticity Euclidean distance/sqrt(2)"}};
        if (illuminant) { // existing methods keep their output keys unchanged
            out.insert("illuminant_config",QJsonObject{
                {"reference",job["illuminant_reference"].toString("object")},
                {"clip_policy",job["reference_clip_policy"].toString("rank")},{"saturation_level",vkConfig.saturationLevel},
                {"reference_model",job["reference_model"].toString("constant")},
                {"max_reference_clip_fraction",vkConfig.maxReferenceClipFraction},
                {"min_reference_fraction",vkConfig.minReferenceFraction},
                {"reference_rect",job["reference_rect"]},{"template_reference",triple(illuminant->templateReference())}});
            out.insert("illuminant_definition","per-channel median of nearest source pixels under the reference (pose-mapped "
                "template mask or fixed source rectangle); von Kries gain template/source applied before chroma; status 3 abstains; "
                "reference_model linear solves first-order slopes from the outer-third medians along template u and v, "
                "each placed at the median (u, v) behind it, and extrapolates the illuminant to each colour sample");
        }
        std::cout<<QJsonDocument(out).toJson(QJsonDocument::Compact).constData()<<'\n';
        return exact?0:1;
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
