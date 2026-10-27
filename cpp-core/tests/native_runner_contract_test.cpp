// SPDX-License-Identifier: BSD-2-Clause
// Runs the real fpm_native_pipeline_runner. --job-keys: unknown keys fail
// before image decoding and the message lists the accepted keys, while every
// accepted key passes. --fine-windows: the JSON reports mode-4 SIMD32 and
// fallback windows per level and per call. --sequence-gaps: source sequences
// decode, skip or sleep between calls and can repeat one frame without
// updating the source (CPU search, so no Metal device is needed).
// --pyramid-rule: the opt-in rule reports one decision per scale model, keeps
// the output without it, and rejects invalid or orphaned rule keys.
// --match-verify: the round-52 checks add a match_verify report only when on,
// drop a smooth ramp's border false positives, and reject invalid values.
#include "metal_device.hpp"
#include <QCoreApplication>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

namespace {
void require(bool value,const std::string& message) { if(!value) throw std::runtime_error(message); }
std::string quoted(const QString& path) {
    require(!path.contains('\''),"Test paths must not contain single quotes");
    return "'"+path.toStdString()+"'";
}
struct Output { int code=-1; QByteArray out,err; };
Output run(const QString& runner,const QTemporaryDir& dir,const QString& name,const QJsonObject& job) {
    const QString jobPath=dir.filePath(name+".json"),outPath=dir.filePath(name+".out"),errPath=dir.filePath(name+".err");
    QFile file(jobPath);require(file.open(QIODevice::WriteOnly) && file.write(QJsonDocument(job).toJson())>0,"Cannot write job");
    file.close();
    const int status=std::system((quoted(runner)+" "+quoted(jobPath)+" >"+quoted(outPath)+" 2>"+quoted(errPath)).c_str());
    Output result;result.code=status!=-1 && WIFEXITED(status) ? WEXITSTATUS(status):-1;
    QFile out(outPath),err(errPath);
    if(out.open(QIODevice::ReadOnly)) result.out=out.readAll();
    if(err.open(QIODevice::ReadOnly)) result.err=err.readAll();
    return result;
}
void jobKeys(const QString& runner,const QTemporaryDir& dir) {
    const QString missing=dir.filePath("missing.png");
    const QJsonObject base{{"source",missing},{"template",missing},{"repeats",1},{"warmup_seconds",0}};
    auto unknown=base;unknown["full_gpu_fine_mod"]=4;unknown["zz_unused"]=true;
    auto result=run(runner,dir,"unknown",unknown);
    const std::string message=result.err.toStdString();
    require(result.code==1 && result.out.isEmpty(),"Unknown job keys were not rejected: "+message);
    require(message.rfind("Unknown job key(s): full_gpu_fine_mod, zz_unused; accepted keys: angle, backend, ",0)==0,
        "Rejection does not name the unknown keys before the accepted list: "+message);
    for(const char* key:{"full_gpu_fine_mode","source_frames","sparse_min_gradient","warmup_seconds"})
        require(message.find(key)!=std::string::npos,std::string("Accepted-key list omits ")+key);
    require(message.find("Cannot decode image")==std::string::npos,"Unknown keys were checked after image decoding");
    // Every accepted key, with a value of its type: the job passes the key gate
    // and stops at decoding the missing image.
    const QJsonObject accepted{{"angle",0},{"backend",1},{"border_min_inside",0.},{"candidate_dedup_angle_steps",3.},{"candidate_dedup_radius",.5},
        {"candidate_dedup_top_k",0},{"coarse_peak_mode",1},{"cpu_threads",8},
        {"fine_angle_reuse",false},{"fine_fused",false},{"fine_ncc_mode",0},{"fine_profiling",false},{"full_gpu",false},
        {"full_gpu_fine_mode",0},{"full_gpu_peak_batch",0},{"full_gpu_peak_mode",0},{"full_gpu_profiling",false},{"gray_load_mode","direct_gray8"},
        {"invert_source",false},{"limit",4},{"min_area",256},{"ncc_mode",0},{"overlap",.2},{"overlap_enabled",false},
        {"pose_input_mode",0},{"pose_resident_mode",0},{"pose_storage_mode",0},{"proposal_method","pyramid"},
        {"proposal_score",.5},{"pyramid_rule",1},{"pyramid_rule_min_score",.5},{"refine",false},{"refinement_backend",-1},{"repeats",1},{"run_resource_diagnostics",false},
        {"scale_support_refine",false},{"scales",QJsonArray{1.}},{"score",.8},{"sequence_gap","sleep"},
        {"sequence_gap_sleep_ms",5.},{"sequence_mode","reuse"},{"small_template_area",0.},{"soak_seconds",0},
        {"source",missing},{"source_frames",QJsonArray{missing}},{"sparse_angle_step",6.},{"sparse_candidate_limit",256},
        {"sparse_cpu_gradient_mode",0},{"sparse_features",128},{"sparse_metal_submission_mode",0},{"sparse_min_gradient",8.},
        {"sparse_response_mode",0},{"sparse_search_backend",0},{"sparse_spread",2},{"sparse_stride",4},{"sparse_threshold",.75},
        {"stop_layer",1},{"stream_path",dir.filePath("stream.jsonl")},{"subpixel",false},{"template",missing},{"warmup_seconds",0}};
    // The fixture must name exactly the keys the runner itself lists, so a new
    // key cannot be added without this test passing a value of its type.
    const auto listed=QString::fromStdString(message.substr(message.find("accepted keys: ")+15)).trimmed().split(", ");
    require(listed==accepted.keys(),"Fixture keys differ from the runner's accepted-key list");
    result=run(runner,dir,"accepted",accepted);
    require(result.code==1 && result.err.trimmed()=="Cannot decode image",
        "Accepted keys did not reach image decoding: "+result.err.toStdString());
    auto truth=base;truth["truth"]=QJsonArray();
    result=run(runner,dir,"truth",truth);
    require(result.code==1 && result.err.trimmed()=="Expected job without ground truth","Ground truth was not rejected first");
}
void fineWindows(const QString& runner,const QTemporaryDir& dir) {
    // A 160x120 template has an uncached level-0 ROI (166x126 > 16 KiB) and a
    // cached 80x60 level 1, so one mode-4 search exercises both routes. The
    // offset is a multiple of 4 to stay pixel-aligned at the 40x30 top level,
    // where smoothed noise scores below 0.9*0.81; the final match scores 1.
    std::mt19937 random(370019);
    QImage templ(160,120,QImage::Format_Grayscale8),source(320,240,QImage::Format_Grayscale8);
    for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) templ.scanLine(y)[x]=uchar(random());
    source.fill(23);
    for(int y=0;y<templ.height();++y) std::copy_n(templ.constScanLine(y),templ.width(),source.scanLine(y+48)+60);
    require(templ.save(dir.filePath("template.png")) && source.save(dir.filePath("source.png")),"Cannot write runner images");
    QJsonObject job{{"source",dir.filePath("source.png")},{"template",dir.filePath("template.png")},{"full_gpu",true},
        {"backend",1},{"refinement_backend",1},{"full_gpu_peak_mode",1},{"full_gpu_fine_mode",4},{"full_gpu_profiling",true},
        {"angle",0},{"scales",QJsonArray{1.}},{"limit",4},{"score",.7},{"overlap",.2},{"stop_layer",0},{"min_area",1600},
        {"refine",false},{"repeats",1},{"warmup_seconds",0}};
    auto call=[&](const QString& name) {
        const auto result=run(runner,dir,name,job);
        require(result.code==0,name.toStdString()+": runner failed: "+result.err.toStdString());
        const auto first=QJsonDocument::fromJson(result.out).object()["first"].toObject();
        require(first["matches"].toArray().size()==1,name.toStdString()+": fixture lost its object");
        return first["full_gpu"].toObject();
    };
    auto gpu=call("mode4-profile");
    const bool simdMode=gpu["fine_effective_mode"].toInt()==4;
    qint64 simd=0,fallback=0;bool normalCache=true;
    for(const auto& value:gpu["fine_work"].toArray()) {
        const auto level=value.toObject();
        const qint64 width=level["template_width"].toInteger(),height=level["template_height"].toInteger();
        const qint64 cache=level["roi_cache_bytes"].toInteger(),angles=level["angles"].toInteger();
        const bool route=simdMode && (width+6)*(height+6)<=cache && (width*height+3)/4<=66051;
        require(level["simd32_windows"].toInteger(-1)==(route?49*angles:0) &&
            level["fallback_windows"].toInteger(-1)==(simdMode && !route?49*angles:0),
            "Per-level SIMD32/fallback windows disagree with the ROI cache and uint32 bounds");
        simd+=level["simd32_windows"].toInteger();fallback+=level["fallback_windows"].toInteger();normalCache&=cache==16384;
    }
    require(gpu["fine_work_available"].toBool() && gpu["fine_simd32_windows"].toInteger(-1)==simd &&
        gpu["fine_fallback_windows"].toInteger(-1)==fallback,"Call-level window totals differ from fine_work");
    // Shader Validation shrinks the cache to 4 KiB, which leaves level 1 uncached.
    require(!simdMode || !normalCache || (simd>0 && fallback>0),"Mode-4 fixture did not report both SIMD32 and fallback windows");
    job["full_gpu_fine_mode"]=2;gpu=call("mode2-profile");
    require(gpu["fine_simd32_windows"].toInteger(-1)==0 && gpu["fine_fallback_windows"].toInteger(-1)==0,
        "Mode 2 reported SIMD32/fallback windows");
    job["full_gpu_fine_mode"]=4;job["full_gpu_profiling"]=false;gpu=call("mode4-formal");
    require(!gpu["fine_work_available"].toBool() && gpu["fine_simd32_windows"].isNull() && gpu["fine_fallback_windows"].isNull(),
        "Unprofiled call reported window counts");
}
void sequenceGaps(const QString& runner,const QTemporaryDir& dir) {
    // One exact copy of a random template on a flat background; CPU coarse
    // search and refinement find it at score 1 without a Metal device.
    std::mt19937 random(480019);
    QImage templ(48,36,QImage::Format_Grayscale8),source(192,144,QImage::Format_Grayscale8);
    for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) templ.scanLine(y)[x]=uchar(random());
    source.fill(23);
    for(int y=0;y<templ.height();++y) std::copy_n(templ.constScanLine(y),templ.width(),source.scanLine(y+40)+64);
    require(templ.save(dir.filePath("gap-template.png")) && source.save(dir.filePath("gap-source.png")),"Cannot write runner images");
    const QString frame=dir.filePath("gap-source.png");
    const QJsonObject base{{"source_frames",QJsonArray{frame}},{"template",dir.filePath("gap-template.png")},
        {"backend",0},{"refinement_backend",0},{"refine",true},{"angle",0},{"scales",QJsonArray{1.}},{"limit",2},
        {"score",.7},{"stop_layer",0},{"repeats",4},{"warmup_seconds",0}};
    QJsonArray reference;
    auto sequence=[&](const QString& name,QJsonObject changes) {
        auto job=base;for(auto it=changes.begin();it!=changes.end();++it) job[it.key()]=it.value();
        const auto result=run(runner,dir,name,job);
        require(result.code==0,name.toStdString()+": runner failed: "+result.err.toStdString());
        const auto out=QJsonDocument::fromJson(result.out).object();
        const auto runs=out["runs"].toArray();
        require(out["exact_sequence_repeat"].toBool() && runs.size()==4,name.toStdString()+": incomplete or inexact sequence");
        for(const auto& value:runs) {
            const auto row=value.toObject();
            require(row["matches"].toArray().size()==1,name.toStdString()+": lost the object");
            if(reference.isEmpty()) reference=row["matches"].toArray();
            require(row["matches"].toArray()==reference,name.toStdString()+": output differs between gap modes");
        }
        return std::make_pair(out,runs);
    };
    struct Expect { const char* name; const char* mode; const char* gap; double sleep; };
    for(const Expect& e:{Expect{"reuse-decode","reuse","",0},Expect{"reuse-none","reuse","none",0},Expect{"reuse-sleep","reuse","sleep",25},
                         Expect{"static-decode","static","decode",0},Expect{"static-none","static","none",0},Expect{"static-sleep","static","sleep",25}}) {
        QJsonObject changes{{"sequence_mode",e.mode}};
        if(*e.gap) changes["sequence_gap"]=e.gap;
        if(e.sleep>0) changes["sequence_gap_sleep_ms"]=e.sleep;
        const auto [out,runs]=sequence(e.name,changes);
        const std::string label=e.name;const bool reuse=std::string(e.mode)=="reuse";
        const std::string gap=*e.gap ? e.gap:"decode";
        require(out["sequence_mode"].toString().toStdString()==e.mode && out["sequence_gap"].toString().toStdString()==gap,
            label+": output does not name its mode and gap");
        require(e.sleep>0 ? out["sequence_gap_sleep_ms"].toDouble()==e.sleep:out["sequence_gap_sleep_ms"].isNull(),
            label+": output does not report the sleep length");
        for(int call=0;call<4;++call) {
            const auto row=runs[call].toObject();
            const double decode=row["source_decode_ms"].toDouble(-1),wait=row["gap_ms"].toDouble(-1);
            require(row["sequence_gap"].toString().toStdString()==gap,label+": row does not name its gap");
            if(call==0) require(decode==0 && wait==0,label+": the first call has a gap");
            else if(gap=="decode") require(decode>0 && wait>=decode,label+": no decode before a later call");
            else if(gap=="sleep") require(decode==0 && wait>=e.sleep,label+": slept less than requested or decoded");
            else require(decode==0 && wait>=0 && wait<1000,label+": a no-gap call decoded or waited");
            // Reuse updates the source every later call and rebuilds the pose
            // context; static keeps both from call 0.
            require(row["source_generation"].toInt(-1)==(reuse ? call:0),label+": wrong source generation");
            require(row["pose_invalidated"].toBool()==(reuse && call>0),label+": wrong pose invalidation");
            require((row["source_copy_bytes"].toDouble()>0)==(reuse && call>0),label+": wrong source copy");
            require(reuse || call==0 || row["source_update_ms"].toDouble(-1)==0,label+": static call updated the source");
            require((row["pose_create_ms"].toDouble()>0)==(reuse || call==0),label+": wrong pose context reuse");
        }
    }
    auto reject=[&](const QString& name,QJsonObject changes,const std::string& expected) {
        auto job=base;for(auto it=changes.begin();it!=changes.end();++it) job[it.key()]=it.value();
        const auto result=run(runner,dir,name,job);
        require(result.code==1 && result.out.isEmpty() && result.err.trimmed().toStdString()==expected,
            name.toStdString()+": expected '"+expected+"', got '"+result.err.trimmed().toStdString()+"'");
    };
    const std::string gaps="sequence_gap must be decode, none or sleep";
    const std::string sleeps="sequence_gap_sleep_ms is required with, and only with, sequence_gap sleep";
    const std::string bounds="sequence_gap_sleep_ms must be a number in (0, 10000]";
    reject("bad-mode",{{"sequence_mode","sideways"}},"sequence_mode must be reuse, fresh or static");
    reject("bad-gap",{{"sequence_gap","pause"}},gaps);
    reject("numeric-gap",{{"sequence_gap",1}},gaps);
    reject("sleep-without-ms",{{"sequence_gap","sleep"}},sleeps);
    reject("ms-without-sleep",{{"sequence_gap","none"},{"sequence_gap_sleep_ms",5.}},sleeps);
    reject("ms-with-default",{{"sequence_gap_sleep_ms",5.}},sleeps);
    reject("zero-sleep",{{"sequence_gap","sleep"},{"sequence_gap_sleep_ms",0}},bounds);
    reject("long-sleep",{{"sequence_gap","sleep"},{"sequence_gap_sleep_ms",20000}},bounds);
    reject("text-sleep",{{"sequence_gap","sleep"},{"sequence_gap_sleep_ms","5"}},bounds);
    reject("static-two-frames",{{"sequence_mode","static"},{"source_frames",QJsonArray{frame,frame}}},
        "Static sequences repeat exactly one source frame");
}
void pyramidRule(const QString& runner,const QTemporaryDir& dir) {
    // Stripes with an 8 px period: depth 2 at min_area 256, aliased at the 11x8 top level.
    QImage templ(44,30,QImage::Format_Grayscale8),source(140,80,QImage::Format_Grayscale8);
    for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x)
        templ.scanLine(y)[x]=uchar(std::clamp(int(std::lround(128+70*std::sin(.8*x+.6*y)+30*std::cos(.5*x-.7*y))),0,255));
    source.fill(118);
    for(int y=0;y<templ.height();++y) std::copy_n(templ.constScanLine(y),templ.width(),source.scanLine(y+12)+10);
    require(templ.save(dir.filePath("rule-template.png")) && source.save(dir.filePath("rule-source.png")),"Cannot write runner images");
    const QJsonObject base{{"source",dir.filePath("rule-source.png")},{"template",dir.filePath("rule-template.png")},
        {"backend",0},{"refinement_backend",0},{"refine",false},{"angle",0},{"scales",QJsonArray{.95,1.}},{"limit",2},
        {"score",.8},{"stop_layer",0},{"min_area",256},{"repeats",1},{"warmup_seconds",0}};
    auto call=[&](const QString& name,QJsonObject changes,int code) {
        auto job=base;for(auto it=changes.begin();it!=changes.end();++it) job[it.key()]=it.value();
        const auto result=run(runner,dir,name,job);
        require(result.code==code,name.toStdString()+": unexpected exit, stderr "+result.err.toStdString());
        return std::make_pair(QJsonDocument::fromJson(result.out).object(),result.err.trimmed().toStdString());
    };
    const auto off=call("rule-off",{},0).first;
    require(!off.contains("pyramid_rule"),"Rule-off output gained a pyramid_rule key");
    const auto keep=call("rule-keep",{{"pyramid_rule",1},{"pyramid_rule_min_score",-1}},0).first;
    require(keep["first"].toObject()["matches"]==off["first"].toObject()["matches"],"Unreachable rule changed the matches");
    const auto on=call("rule-on",{{"pyramid_rule",1},{"pyramid_rule_min_score",.5}},0).first;
    const auto report=on["pyramid_rule"].toObject();const auto models=report["models"].toArray();
    require(report["min_score"].toDouble()==.5 && models.size()==2,"Rule report lacks its threshold or one model per scale");
    for(int i=0;i<2;++i) {
        const auto model=models[i].toObject();
        require(model["scale"].toDouble()==(i?1.:.95),"Rule models are not in scale order");
        require(model["predicted_scores"].toArray().size()==model["original_top"].toInt()+1,"Predicted scores do not cover every level");
        require(model["selected_top"].toInt()<=model["original_top"].toInt(),"Rule deepened the pyramid");
    }
    const auto upright=models[1].toObject();
    require(upright["original_top"].toInt()==2 && upright["selected_top"].toInt()<2 && upright["effective_min_area"].toInt()!=256,
        "Fine stripes kept the original top level");
    const std::string orphan="pyramid_rule_min_score requires pyramid_rule 1";
    require(call("orphan",{{"pyramid_rule_min_score",.5}},1).second==orphan,"Orphaned threshold was not rejected");
    require(call("orphan-zero",{{"pyramid_rule",0},{"pyramid_rule_min_score",.5}},1).second==orphan,"Threshold with rule 0 was not rejected");
    require(call("bad-mode",{{"pyramid_rule",2}},1).second=="pyramid_rule must be 0 or 1","Invalid rule mode was not rejected");
    require(call("bool-mode",{{"pyramid_rule",true}},1).second=="pyramid_rule must be 0 or 1","Boolean rule mode was not rejected");
    require(call("bad-score",{{"pyramid_rule",1},{"pyramid_rule_min_score",2}},1).second=="pyramid_rule_min_score must be a number in [-1, 1]",
        "Out-of-range threshold was not rejected");
}
void matchVerify(const QString& runner,const QTemporaryDir& dir) {
    // Round-50 proc50105 failure: a smooth ramp on flat noise matches the padded image border.
    QImage templ(20,14,QImage::Format_Grayscale8),source(256,256,QImage::Format_Grayscale8);
    std::mt19937 random(52010);std::normal_distribution<double> noise(0,6);
    for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) templ.scanLine(y)[x]=uchar(40+180*x/19);
    for(int y=0;y<source.height();++y) for(int x=0;x<source.width();++x) source.scanLine(y)[x]=uchar(std::clamp(128+noise(random),0.,255.));
    require(templ.save(dir.filePath("verify-template.png")) && source.save(dir.filePath("verify-source.png")),"Cannot write runner images");
    const QJsonObject base{{"source",dir.filePath("verify-source.png")},{"template",dir.filePath("verify-template.png")},
        {"backend",0},{"refinement_backend",0},{"angle",180},{"limit",100},{"score",.8},{"stop_layer",1},{"min_area",256},
        {"repeats",1},{"warmup_seconds",0}};
    auto call=[&](const QString& name,QJsonObject changes,int code) {
        auto job=base;for(auto it=changes.begin();it!=changes.end();++it) job[it.key()]=it.value();
        const auto result=run(runner,dir,name,job);
        require(result.code==code,name.toStdString()+": unexpected exit, stderr "+result.err.toStdString());
        return std::make_pair(QJsonDocument::fromJson(result.out).object(),result.err.trimmed().toStdString());
    };
    const auto off=call("verify-off",{},0).first;
    require(!off["first"].toObject().contains("match_verify"),"Verification-off output gained a match_verify key");
    require(!off["first"].toObject()["matches"].toArray().isEmpty(),"Ramp fixture lost its border false positives");
    const auto zero=call("verify-zero",{{"border_min_inside",0.},{"small_template_area",0.}},0).first;
    require(!zero["first"].toObject().contains("match_verify") && zero["first"].toObject()["matches"]==off["first"].toObject()["matches"],"Zero rules changed the output");
    const auto on=call("verify-on",{{"border_min_inside",.5}},0).first;
    const auto report=on["first"].toObject()["match_verify"].toObject();
    require(!report["after_nms"].toBool() && report["rejected_border"].toDouble()>0 && report["checked"].toDouble()>=report["border"].toDouble(),
        "Verification report is missing or wrong");
    require(on["first"].toObject()["matches"].toArray().isEmpty(),"Border false positives survived verification");
    const std::string bounds="Match verification needs border_min_inside 0..1 and small_template_area >= 0";
    require(call("verify-high",{{"border_min_inside",1.5}},1).second==bounds,"Border fraction above 1 was accepted");
    require(call("verify-negative",{{"small_template_area",-1.}},1).second==bounds,"Negative area was accepted");
    require(call("verify-text",{{"border_min_inside","0.5"}},1).second=="border_min_inside must be a number","Text border fraction was accepted");
}
}
int main(int argc,char** argv) {
    QCoreApplication app(argc,argv);
    try {
        const std::string mode=argc==3 ? argv[1]:"";
        require(mode=="--job-keys" || mode=="--fine-windows" || mode=="--sequence-gaps" || mode=="--pyramid-rule" || mode=="--match-verify",
            "Usage: native_runner_contract_test --job-keys|--fine-windows|--sequence-gaps|--pyramid-rule|--match-verify runner");
        if(mode=="--fine-windows" && !fpm_test::metalDeviceAvailable()) { std::cout<<"SKIP: no Metal device\n";return 77; }
        QTemporaryDir dir;require(dir.isValid(),"Cannot create temporary directory");
        const QString runner=QString::fromLocal8Bit(argv[2]);
        if(mode=="--job-keys") jobKeys(runner,dir); else if(mode=="--fine-windows") fineWindows(runner,dir);
        else if(mode=="--pyramid-rule") pyramidRule(runner,dir); else if(mode=="--match-verify") matchVerify(runner,dir); else sequenceGaps(runner,dir);
        std::cout<<"Native runner "<<(mode=="--job-keys" ? "rejects unknown job keys before decoding and accepts every listed key":
            mode=="--fine-windows" ? "reports mode-4 SIMD32/fallback windows per level and per call":
            mode=="--pyramid-rule" ? "reports opt-in pyramid rule decisions per scale and rejects invalid rule keys":
            mode=="--match-verify" ? "reports opt-in match verification only when on and drops ramp border false positives":
            "decodes, skips or sleeps between sequence calls and repeats a static frame without updating it")<<'\n';
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
