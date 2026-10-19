// SPDX-License-Identifier: BSD-2-Clause
// Runs the real fpm_native_pipeline_runner. --job-keys: unknown keys fail
// before image decoding and the message lists the accepted keys, while every
// accepted key passes. --fine-windows: the JSON reports mode-4 SIMD32 and
// fallback windows per level and per call.
#include "metal_device.hpp"
#include <QCoreApplication>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <algorithm>
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
    const QJsonObject accepted{{"angle",0},{"backend",1},{"coarse_peak_mode",1},{"cpu_threads",8},
        {"fine_angle_reuse",false},{"fine_fused",false},{"fine_ncc_mode",0},{"fine_profiling",false},{"full_gpu",false},
        {"full_gpu_fine_mode",0},{"full_gpu_peak_mode",0},{"full_gpu_profiling",false},{"gray_load_mode","direct_gray8"},
        {"invert_source",false},{"limit",4},{"min_area",256},{"ncc_mode",0},{"overlap",.2},{"overlap_enabled",false},
        {"pose_input_mode",0},{"pose_resident_mode",0},{"pose_storage_mode",0},{"proposal_method","pyramid"},
        {"proposal_score",.5},{"refine",false},{"refinement_backend",-1},{"repeats",1},{"run_resource_diagnostics",false},
        {"scale_support_refine",false},{"scales",QJsonArray{1.}},{"score",.8},{"sequence_mode","reuse"},{"soak_seconds",0},
        {"source",missing},{"source_frames",QJsonArray{missing}},{"sparse_angle_step",6.},{"sparse_candidate_limit",256},
        {"sparse_cpu_gradient_mode",0},{"sparse_features",128},{"sparse_metal_submission_mode",0},{"sparse_min_gradient",8.},
        {"sparse_response_mode",0},{"sparse_search_backend",0},{"sparse_spread",2},{"sparse_stride",4},{"sparse_threshold",.75},
        {"stop_layer",1},{"stream_path",dir.filePath("stream.jsonl")},{"subpixel",false},{"template",missing},{"warmup_seconds",0}};
    require(accepted.size()==51,"Fixture must list all 51 accepted keys");
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
}
int main(int argc,char** argv) {
    QCoreApplication app(argc,argv);
    try {
        const std::string mode=argc==3 ? argv[1]:"";
        require(mode=="--job-keys" || mode=="--fine-windows","Usage: native_runner_contract_test --job-keys|--fine-windows runner");
        if(mode=="--fine-windows" && !fpm_test::metalDeviceAvailable()) { std::cout<<"SKIP: no Metal device\n";return 77; }
        QTemporaryDir dir;require(dir.isValid(),"Cannot create temporary directory");
        const QString runner=QString::fromLocal8Bit(argv[2]);
        if(mode=="--job-keys") jobKeys(runner,dir); else fineWindows(runner,dir);
        std::cout<<"Native runner "<<(mode=="--job-keys" ? "rejects unknown job keys before decoding and accepts all 51 keys":
            "reports mode-4 SIMD32/fallback windows per level and per call")<<'\n';
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
