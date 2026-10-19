// SPDX-License-Identifier: BSD-2-Clause
#include "pose_ncc.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Handle=std::unique_ptr<void,decltype(&fpm_pose_destroy)>;
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void equivalent(int channels,int w,int h) {
    const int tw=17,th=19;
    std::vector<float> source(size_t(w)*h*channels),templ(size_t(tw)*th*channels);
    for(size_t i=0;i<source.size();++i) source[i]=float(.4+.18*std::sin(i*.137)+.12*std::cos(i*.037));
    for(int y=0;y<th;++y) for(int x=0;x<tw;++x) for(int c=0;c<channels;++c)
        templ[(size_t(y)*tw+x)*channels+c]=source[(size_t(y+12)*w+x+10)*channels+c];
    char error[1024]{};
    Handle copied(fpm_pose_create(1,2,source.data(),w,h,templ.data(),tw,th,channels,error,sizeof(error)),fpm_pose_destroy);
    require(bool(copied),error);
    Handle shared(fpm_pose_create_with_storage(1,2,source.data(),w,h,templ.data(),tw,th,channels,1,error,sizeof(error)),fpm_pose_destroy);
    require(bool(shared),error);
    // Both handles must own snapshots, including after caller mutation/free.
    std::fill(source.begin(),source.end(),std::numeric_limits<float>::quiet_NaN());
    std::fill(templ.begin(),templ.end(),-9.f);
    std::vector<float>().swap(source);std::vector<float>().swap(templ);
    FpmPoseStorageInfo a{},b{};
    require(fpm_pose_storage_info(copied.get(),&a)==0 && fpm_pose_storage_info(shared.get(),&b)==0,"storage query failed");
    const uint64_t logical=uint64_t(w*h+tw*th)*channels*sizeof(float);
    require(a.mode==0 && a.shared_image_bytes==0 && a.host_image_bytes==logical && a.gpu_image_bytes==logical,"copy image accounting");
    require(b.mode==1 && b.shared_image_bytes==b.host_image_bytes && b.shared_image_bytes==b.gpu_image_bytes &&
        b.unique_image_bytes==b.host_image_bytes && b.unique_image_bytes>=logical && b.unique_image_bytes-logical<131072,"shared allocation accounting");
    const float seeds[]{float(10+tw*.5-w*.5+.35),float(h*.5-12-th*.5-.2),.4f,.002f,
        10000,-10000,90,.1f,3,-5,41,-.03f};
    std::vector<float> old(3),now(3),oldPose(12),newPose(12);
    FpmPoseMetrics metrics{};FpmPoseRefineConfig config{};fpm_pose_refine_default_config(&config);config.max_iterations=3;
    for(int repeat=0;repeat<3;++repeat) {
        require(fpm_pose_score(copied.get(),seeds,3,old.data(),&metrics,error,sizeof(error))==0,error);
        require(fpm_pose_score(shared.get(),seeds,3,now.data(),&metrics,error,sizeof(error))==0,error);
        require(old==now,"no-copy GPU scorer differs");
        require(fpm_pose_refine(copied.get(),seeds,3,&config,oldPose.data(),old.data(),nullptr,&metrics,error,sizeof(error))==0,error);
        require(fpm_pose_refine(shared.get(),seeds,3,&config,newPose.data(),now.data(),nullptr,&metrics,error,sizeof(error))==0,error);
        require(old==now && oldPose==newPose,"no-copy CPU verification/refinement differs");
        require(fpm_pose_refine_accelerated(copied.get(),seeds,3,&config,1,oldPose.data(),old.data(),nullptr,&metrics,error,sizeof(error))==0,error);
        require(fpm_pose_refine_accelerated(shared.get(),seeds,3,&config,1,newPose.data(),now.data(),nullptr,&metrics,error,sizeof(error))==0,error);
        require(old==now && oldPose==newPose,"no-copy accelerated derivative differs");
    }
    FpmPoseStorageInfo after{};fpm_pose_storage_info(shared.get(),&after);
    require(after.unique_image_bytes==b.unique_image_bytes && after.shared_image_bytes==b.shared_image_bytes,"shared images replaced by workspace growth");
}
}
int main() {
    try {
        float image[16]{};char error[1024]{};
        FpmPoseStorageInfo query{1,1,1,1,1,1};
        require(fpm_pose_storage_info(nullptr,&query)==-1 && query.unique_image_bytes==0,"null storage query not reset");
        for(int mode:{-1,2}) require(!fpm_pose_create_with_storage(1,1,image,4,4,image,4,4,1,mode,error,sizeof(error)),"bad storage mode accepted");
        require(!fpm_pose_create_with_storage(0,1,image,4,4,image,4,4,1,1,error,sizeof(error)),"CPU shared storage silently accepted");
        Handle probe(fpm_pose_create(1,1,image,4,4,image,4,4,1,error,sizeof(error)),fpm_pose_destroy);
        if(!probe) {
            const std::string message(error);
            if(message=="No Metal GPU available" || message=="Metal backend was not compiled on this platform") {std::cout<<"SKIP: "<<message<<'\n';return 77;}
            throw std::runtime_error(message);
        }
        image[7]=std::numeric_limits<float>::quiet_NaN();
        require(!fpm_pose_create_with_storage(1,1,image,4,4,image,4,4,1,1,error,sizeof(error)),"shared nonfinite input accepted");
        require(!fpm_pose_create_with_storage(1,1,image,0,4,image,4,4,1,1,error,sizeof(error)),"invalid shared dimension accepted");
        for(int i=0;i<8;++i) equivalent(i%2?3:1,63+i,59+i);
        equivalent(1,4096,4096);
        std::cout<<"Shared storage: gray/RGB exact score, CPU/accelerated refine; caller lifetime, odd page tails, 4K, nine context pairs and guards passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
