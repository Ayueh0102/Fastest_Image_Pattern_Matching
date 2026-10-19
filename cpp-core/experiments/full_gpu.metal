// SPDX-License-Identifier: BSD-2-Clause
// Complete resident gray8 pipeline helpers. All image-derived decisions are
// device work. FP32 interpolation/normalization is an explicit new contract.
inline float fg_angle(float a) {
    a = fmod(a,360.f); if (a > 180.f) a -= 360.f; if (a <= -180.f) a += 360.f; return a;
}
inline bool fg_allowed(float a, device const FGRange* ranges, uint count) {
    a=fg_angle(a);
    if (!count) return true;
    for(uint i=0;i<count;++i) for(int k=-1;k<=1;++k)
        if(a+360.f*k >= ranges[i].minimum-1e-5f && a+360.f*k <= ranges[i].maximum+1e-5f) return true;
    return false;
}
inline uint fg_pixel(device const uchar* src,uint w,uint h,int x,int y,uint border) {
    return x<0 || y<0 || x>=int(w) || y>=int(h) ? border : uint(src[ulong(y)*w+uint(x)]);
}
inline uchar fg_sample(device const uchar* src,uint w,uint h,float x,float y,uint border) {
    if(!isfinite(x) || !isfinite(y) || x < -1 || y < -1 || x>=float(w) || y>=float(h)) return uchar(border);
    int ix=int(floor(x)),iy=int(floor(y)); float dx=x-ix,dy=y-iy;
    float a=float(fg_pixel(src,w,h,ix,iy,border))*(1-dx)+float(fg_pixel(src,w,h,ix+1,iy,border))*dx;
    float b=float(fg_pixel(src,w,h,ix,iy+1,border))*(1-dx)+float(fg_pixel(src,w,h,ix+1,iy+1,border))*dx;
    return uchar(clamp(floor(a*(1-dy)+b*dy+.5f),0.f,255.f));
}
inline float fg_ncc(ulong s,ulong q,ulong p,FGStats t) {
    if (!t.variance) return 1;
    ulong vi=t.area*q-s*s, a=t.area*p,b=s*t.sum;
    float n=a>=b ? float(a-b) : -float(b-a);
    float denominator=float(vi)/float(t.area)>min(.5f,10.f*FLT_EPSILON*float(q)) ? sqrt(float(vi))*sqrt(float(t.variance)):0;
    if(denominator==0) return 0;
    if(abs(n)<denominator) return n/denominator;
    return abs(n)<denominator*1.125f ? (n>0 ? 1.f:-1.f):0.f;
}
struct FGScale { uint sw,sh,w,h; float inverse; uint reserved[3]; };
kernel void fg_scale_template(device const uchar* src[[buffer(0)]],device uchar* dst[[buffer(1)]],constant FGScale& c[[buffer(2)]],uint i[[thread_position_in_grid]]) {
    if(i>=c.w*c.h) return;
    if(c.inverse==1.f && c.sw==c.w && c.sh==c.h) { dst[i]=src[i]; return; }
    dst[i]=fg_sample(src,c.sw,c.sh,float(c.sw)*.5f+(float(i%c.w)-float(c.w)*.5f)*c.inverse,
        float(c.sh)*.5f+(float(i/c.w)-float(c.h)*.5f)*c.inverse,0);
}
kernel void fg_input(device const uchar* src[[buffer(0)]],device uchar* dst[[buffer(1)]],constant uint2& c[[buffer(2)]],uint i[[thread_position_in_grid]]) {
    if(i<c.x) dst[i]=c.y ? 255-src[i] : src[i];
}
kernel void fg_learn(device const uchar* src[[buffer(0)]],device FGStats* stats[[buffer(1)]],constant uint2& c[[buffer(2)]],uint lane[[thread_index_in_threadgroup]]) {
    threadgroup ulong sums[256],squares[256];ulong s=0,q=0;
    for(uint i=lane;i<c.x;i+=256) { ulong v=src[i];s+=v;q+=v*v; }
    sums[lane]=s;squares[lane]=q;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step/=2) { if(lane<step) { sums[lane]+=sums[lane+step];squares[lane]+=squares[lane+step]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if(lane==0) stats[c.y]={sums[0],squares[0],ulong(c.x)*squares[0]-sums[0]*sums[0],c.x};
}
struct FGCoarse {
    uint sw,sh,w,h,tw,th,first,limit,statsIndex,borderIndex,windows,model;
    float cosine,sine,anchorX,anchorY,translationX,translationY,angle,threshold,overlap,scale;
    uint reserved[2];
};
kernel void fg_rotate(device const uchar* src[[buffer(0)]],device uchar* dst[[buffer(1)]],device const FGStats* stats[[buffer(2)]],constant FGCoarse& c[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=c.w*c.h) return;
    const FGStats t=stats[c.borderIndex];uint border=t.sum<128*t.area ? 255:0;
    float x=float(i%c.w),y=float(i/c.w);
    dst[i]=fg_sample(src,c.sw,c.sh,c.anchorX+c.cosine*x-c.sine*y,c.anchorY+c.sine*x+c.cosine*y,border);
}
template<typename T> inline void fg_normalize(device const T* input,device float* scores,device const FGStats* stats,constant uint4& c,uint i) {
    if(i<c.y) scores[c.x+i]=fg_ncc(input[i].sum,input[i].square,input[i].product,stats[c.z]);
}
kernel void fg_normalize32(device const Coarse32* input[[buffer(0)]],device float* scores[[buffer(1)]],device const FGStats* stats[[buffer(2)]],constant uint4& c[[buffer(3)]],uint i[[thread_position_in_grid]]) { fg_normalize(input,scores,stats,c,i); }
kernel void fg_normalize64(device const Coarse64* input[[buffer(0)]],device float* scores[[buffer(1)]],device const FGStats* stats[[buffer(2)]],constant uint4& c[[buffer(3)]],uint i[[thread_position_in_grid]]) { fg_normalize(input,scores,stats,c,i); }
inline bool fg_better(float a,uint ai,float b,uint bi) { return a>b || (a==b && ai<bi); }
kernel void fg_coarse_peaks(device float* scores[[buffer(0)]],device FGCandidate* candidates[[buffer(1)]],constant FGCoarse& c[[buffer(2)]],uint lane[[thread_index_in_threadgroup]]) {
    threadgroup float values[256];threadgroup uint indices[256];
    uint width=c.w-c.tw+1,height=c.h-c.th+1,stripe=(c.windows+255)/256;
    uint begin=min(c.windows,lane*stripe),end=min(c.windows,begin+stripe),bi=UINT_MAX;float best=-INFINITY;
    for(uint j=begin;j<end;++j) if(fg_better(scores[j],j,best,bi)){best=scores[j];bi=j;}
    for(uint iteration=0;iteration<c.limit;++iteration) {
        values[lane]=best;indices[lane]=bi;threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint step=128;step;step/=2) {
            if(lane<step && fg_better(values[lane+step],indices[lane+step],values[lane],indices[lane])) {values[lane]=values[lane+step];indices[lane]=indices[lane+step];}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score=values[0];uint index=indices[0];if(!isfinite(score)||score<c.threshold) break;
        uint x=index%width,y=index/width;
        if(lane==0) {
            float cx=float(c.sw-1)*.5f,cy=float(c.sh-1)*.5f,rx=float(x)-c.translationX-cx,ry=float(y)-c.translationY-cy;
            candidates[c.first+iteration]={cx+c.cosine*rx-c.sine*ry,cy+c.sine*rx+c.cosine*ry,c.angle,score,c.scale,score,1,c.first+iteration,c.model,0,0,0};
        }
        if(iteration+1==c.limit) break;
        float rw=c.tw*(1-c.overlap),rh=c.th*(1-c.overlap);
        int left=int(float(x)-rw),top=int(float(y)-rh);
        uint x0=uint(max(0,min(int(x),left))),x1=uint(min(int(width),max(int(x)+1,left+int(2*rw))));
        uint y0=uint(max(0,min(int(y),top))),y1=uint(min(int(height),max(int(y)+1,top+int(2*rh))));
        uint w=x1-x0,pixels=w*(y1-y0);
        for(uint j=lane;j<pixels;j+=256) scores[(y0+j/w)*width+x0+j%w]=-INFINITY;
        threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        if(bi!=UINT_MAX && bi%width>=x0 && bi%width<x1 && bi/width>=y0 && bi/width<y1) {
            best=-INFINITY;bi=UINT_MAX;
            for(uint j=begin;j<end;++j) if(fg_better(scores[j],j,best,bi)){best=scores[j];bi=j;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// Source moments are tiled; no single workgroup scans the entire full-res image.
kernel void fg_source_moments(device const uchar* src[[buffer(0)]],device FGStats* partials[[buffer(1)]],constant uint& count[[buffer(2)]],uint lane[[thread_index_in_threadgroup]],uint group[[threadgroup_position_in_grid]]) {
    threadgroup ulong sums[256];ulong s=0;
    for(uint i=group*4096+lane;i<min((group+1)*4096,count);i+=256) s+=src[i];
    sums[lane]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step/=2) {if(lane<step)sums[lane]+=sums[lane+step];threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(!lane) partials[group]={sums[0],0,0,0};
}
kernel void fg_pose_config(device const FGStats* partials[[buffer(0)]],device const FGStats* stats[[buffer(1)]],device Config& k[[buffer(2)]],constant FGFinishConfig& c[[buffer(3)]],constant uint& templateStats[[buffer(4)]],uint lane[[thread_index_in_threadgroup]]) {
    threadgroup ulong sums[256];ulong sum=0;
    uint count=(c.sw*c.sh+4095)/4096;for(uint i=lane;i<count;i+=256)sum+=partials[i].sum;
    sums[lane]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step/=2) {if(lane<step)sums[lane]+=sums[lane+step];threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(!lane) {
        k={};k.sw=c.sw;k.sh=c.sh;k.tw=c.tw;k.th=c.th;k.channels=1;k.tiles=(c.tw*c.th+1023)/1024;k.unused=c.optimizeMask;
        k.sourceMean.x=float(sums[0])/float(c.sw*c.sh)/255.f;
        k.templateSum.x=float(stats[templateStats].sum)/float(c.tw*c.th)/255.f; // temporary mean
    }
}
kernel void fg_center_source(device const uchar* src[[buffer(0)]],device float* dst[[buffer(1)]],constant Config& k[[buffer(2)]],uint i[[thread_position_in_grid]]) {
    if(i<k.sw*k.sh) dst[i]=float(src[i])/255.f-k.sourceMean.x;
}
kernel void fg_center_template(device const uchar* src[[buffer(0)]],device float* dst[[buffer(1)]],device Config& k[[buffer(2)]],uint lane[[thread_index_in_threadgroup]]) {
    threadgroup float sums[256],squares[256];float mean=k.templateSum.x,s=0,q=0;
    // All lanes latch the temporary mean before the final config write.
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    for(uint i=lane;i<k.tw*k.th;i+=256) {float v=float(src[i])/255.f-mean;dst[i]=v;s+=v;q+=v*v;}
    sums[lane]=s;squares[lane]=q;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step/=2) {if(lane<step){sums[lane]+=sums[lane+step];squares[lane]+=squares[lane+step];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(!lane) {k.templateSum=float4(sums[0],0,0,0);k.templateEnergy=max(0.f,squares[0]-sums[0]*sums[0]/float(k.tw*k.th));}
}
