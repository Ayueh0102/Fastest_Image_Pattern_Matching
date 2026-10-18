// SPDX-License-Identifier: BSD-2-Clause
// Appended after pose_fused.metal and full_gpu_types.h. Every data-dependent
// decision below stays on the GPU. Static dispatch capacities are upper bounds;
// GPU-written Config.count and FGStatus counters determine the work performed.
// This experiment has an FP32 geometry/score contract, not CPU-double parity.
static_assert(sizeof(FGCandidate)==48,"full GPU candidate ABI mismatch");
static_assert(sizeof(FGNmsConfig)==48,"full GPU NMS config ABI mismatch");
static_assert(sizeof(FGFinishConfig)==48,"full GPU finish config ABI mismatch");
static_assert(sizeof(FGStatus)==32,"full GPU status ABI mismatch");
static_assert(sizeof(FGResult)==64,"full GPU result ABI mismatch");

inline bool fg_finish_better(FGCandidate a, FGCandidate b)
{
    if (!a.valid) return false;
    if (!b.valid) return true;
    if (a.score != b.score) return a.score > b.score;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    if (a.angle != b.angle) return a.angle < b.angle;
    return a.order < b.order;
}

inline bool fg_finish_allowed(float angle, device const FGRange* ranges, uint count)
{
    if (!count) return true;
    for (uint i = 0; i < count; ++i) for (int shift = -1; shift <= 1; ++shift) {
        const float a = angle+360.f*float(shift);
        if (a >= ranges[i].minimum-1e-5f && a <= ranges[i].maximum+1e-5f) return true;
    }
    return false;
}

inline float fg_finish_cross(float2 a, float2 b, float2 p)
{
    const float2 u = b-a, v = p-a;
    return u.x*v.y-u.y*v.x;
}

inline float fg_finish_area(thread const float2* p, uint count)
{
    float sum = 0;
    for (uint i = 1; i+1 < count; ++i) sum += fg_finish_cross(p[0],p[i],p[i+1]);
    return abs(sum)*.5f;
}

inline void fg_finish_corners(FGCandidate a, float width, float height,
    thread float2* corners)
{
    const float radians = fmod(a.angle,360.f)*0.01745329251994329577f;
    const float c = cos(radians), s = sin(radians);
    const float2 horizontal = float2(c,-s)*(width*.5f);
    const float2 vertical = float2(s,c)*(height*.5f);
    const float2 center = float2(a.x,a.y);
    corners[0] = center-horizontal-vertical;
    corners[1] = center+horizontal-vertical;
    corners[2] = center+horizontal+vertical;
    corners[3] = center-horizontal+vertical;
}

// Convex rectangle intersection has at most eight distinct vertices. Twelve
// slots additionally tolerate duplicate boundary vertices; every append still
// checks its bound and marks an error rather than silently truncating geometry.
inline float fg_finish_overlap(FGCandidate a, FGCandidate b,
    constant FGNmsConfig& k, thread bool& overflow)
{
    float2 polygon[12], clip[4], next[12];
    const float sa = k.globalStage ? a.scale : 1.f;
    const float sb = k.globalStage ? b.scale : 1.f;
    fg_finish_corners(a,float(k.templateWidth)*sa,float(k.templateHeight)*sa,polygon);
    fg_finish_corners(b,float(k.templateWidth)*sb,float(k.templateHeight)*sb,clip);
    float2 amin=polygon[0],amax=polygon[0],bmin=clip[0],bmax=clip[0];
    for (uint i=1;i<4;++i) {
        amin=min(amin,polygon[i]);amax=max(amax,polygon[i]);
        bmin=min(bmin,clip[i]);bmax=max(bmax,clip[i]);
    }
    if (amax.x<=bmin.x || bmax.x<=amin.x || amax.y<=bmin.y || bmax.y<=amin.y) return 0;
    const float aa=fg_finish_area(polygon,4),ba=fg_finish_area(clip,4);
    const float denominator=k.globalStage?min(aa,ba):aa;
    if (!(denominator>0) || !isfinite(denominator)) return 0;
    uint count=4;
    for (uint edge=0;edge<4 && count;++edge) {
        uint n=0;
        const float2 v=clip[edge],w=clip[(edge+1)%4];
        float2 previous=polygon[count-1];
        float previousSide=fg_finish_cross(v,w,previous);
        for (uint i=0;i<count;++i) {
            const float2 current=polygon[i];
            const float side=fg_finish_cross(v,w,current);
            if ((side>=0)!=(previousSide>=0)) {
                if (n>=12) {overflow=true;return 0;}
                const float t=previousSide/(previousSide-side);
                next[n++]=previous+t*(current-previous);
            }
            if (side>=0) {
                if (n>=12) {overflow=true;return 0;}
                next[n++]=current;
            }
            previous=current;previousSide=side;
        }
        count=n;
        for (uint i=0;i<count;++i) polygon[i]=next[i];
    }
    return clamp(fg_finish_area(polygon,count)/denominator,0.f,1.f);
}

// Exactly one 256-thread group per dispatch. Input and output must not alias.
// Repeated reduction produces the same stable score/y/x/angle/original-order
// order as sorting all candidates, without an intermediate host sort. Each
// accepted candidate suppresses remaining inputs in parallel. Per-model NMS
// removes complete duplicates even at overlap=1; global NMS follows research
// pipeline semantics and does not add that duplicate rule.
inline void fg_finish_nms(device FGCandidate* input,device FGCandidate* output,
    device uint* modelCounts,device FGStatus& status,constant FGNmsConfig& k,
    device const FGRange* ranges,uint lane,threadgroup FGCandidate* winners,
    threadgroup uint* indices,threadgroup uint& accepted)
{
    const uint count=k.globalStage?min(k.count,status.proposals):k.count;
    if (lane==0) accepted=0;
    for (uint i=lane;i<count;i+=256) {
        device FGCandidate& a=input[k.first+i];
        if (!isfinite(a.score) || a.score<k.minScore || !isfinite(a.x) || !isfinite(a.y)
            || !isfinite(a.angle) || !isfinite(a.scale) || a.scale<=0
            || (k.globalStage && !fg_finish_allowed(a.angle,ranges,k.rangeCount))) a.valid=0;
    }
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    for (uint selection=0;selection<k.limit;++selection) {
        FGCandidate best={};uint index=0;
        for (uint i=lane;i<count;i+=256) {
            const FGCandidate a=input[k.first+i];
            if (fg_finish_better(a,best)) {best=a;index=i;}
        }
        winners[lane]=best;indices[lane]=index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step=128;step;step>>=1) {
            if (lane<step && fg_finish_better(winners[lane+step],winners[lane])) {
                winners[lane]=winners[lane+step];indices[lane]=indices[lane+step];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const FGCandidate selected=winners[0];
        const uint selectedIndex=indices[0];
        if (!selected.valid) break;
        if (lane==0) {output[k.outputFirst+accepted]=selected;++accepted;}
        for (uint i=lane;i<count;i+=256) {
            device FGCandidate& a=input[k.first+i];
            if (!a.valid) continue;
            if (i==selectedIndex) {a.valid=0;continue;}
            bool invalidGeometry=false;
            const float ratio=fg_finish_overlap(selected,a,k,invalidGeometry);
            if (invalidGeometry)
                atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(&status.overflow),1u,memory_order_relaxed);
            if (ratio>k.maxOverlap+1e-10f || (!k.globalStage && ratio>=1.f-1e-10f)) a.valid=0;
        }
        threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    }
    if (lane==0) {
        if (k.globalStage) status.matches=accepted;
        else modelCounts[k.reserved0]=accepted;
    }
}

#define FG_NMS_KERNEL(NAME) \
kernel void NAME(device FGCandidate* input [[buffer(0)]], \
    device FGCandidate* output [[buffer(1)]],device uint* counts [[buffer(2)]], \
    device FGStatus& status [[buffer(3)]],constant FGNmsConfig& k [[buffer(4)]], \
    device const FGRange* ranges [[buffer(5)]],uint lane [[thread_index_in_threadgroup]]) \
{ \
    threadgroup FGCandidate winners[256];threadgroup uint indices[256],accepted; \
    fg_finish_nms(input,output,counts,status,k,ranges,lane,winners,indices,accepted); \
}
FG_NMS_KERNEL(fg_model_nms)
FG_NMS_KERNEL(fg_global_nms)
#undef FG_NMS_KERNEL

inline ResidentState fg_finish_seed(FGCandidate p,constant FGFinishConfig& k)
{
    ResidentState s={};
    s.pose=float4(p.x-float(k.sw)*.5f,float(k.sh)*.5f-p.y,p.angle,log(p.scale));
    // A deliberately inward FP32 bound keeps retries within the original seed
    // interval. log(1.3) is rounded down here, not enlarged by FP32 conversion.
    const float4 radius=float4((k.optimizeMask&1u)?12.f:0.f,(k.optimizeMask&1u)?12.f:0.f,
        (k.optimizeMask&2u)?15.f:0.f,(k.optimizeMask&4u)?0.26236424f:0.f);
    const float4 limit=float4(1e6f,1e6f,36000.f,3.f);
    for (uint i=0;i<4;++i) {
        s.lower[i]=radius[i]>0?max(-limit[i],nextafter(s.pose[i]-radius[i],INFINITY)):s.pose[i];
        s.upper[i]=radius[i]>0?min(limit[i],nextafter(s.pose[i]+radius[i],-INFINITY)):s.pose[i];
    }
    s.packed=resident_pack(s.pose);s.score=p.score;
    return s;
}

// One thread. Per-model outputs occupy model*proposalStride slots. The compact
// order is model order then per-model NMS order, matching research_pipeline.
// Preserve a separate compact proposal snapshot before fg_publish_states.
kernel void fg_compact_prepare(device const FGCandidate* perModel [[buffer(0)]],
    device const uint* counts [[buffer(1)]],device FGCandidate* compact [[buffer(2)]],
    device ResidentState* seeds [[buffer(3)]],device ResidentState* states [[buffer(4)]],
    device Config& poseConfig [[buffer(5)]],device FGStatus& status [[buffer(6)]],
    constant FGFinishConfig& k [[buffer(7)]],uint id [[thread_position_in_grid]])
{
    if (id) return;
    status.proposals=status.matches=status.rescueParents=status.rescueStarts=status.rescueContinuations=0;
    for (uint model=0;model<k.modelCount;++model) {
        if (counts[model]>k.proposalStride) {status.overflow|=2u;continue;}
        for (uint j=0;j<counts[model];++j) {
            if (status.proposals>=k.capacity) {status.overflow|=2u;continue;}
            FGCandidate p=perModel[model*k.proposalStride+j];
            p.order=status.proposals;p.valid=1;
            compact[status.proposals]=p;
            const ResidentState seed=fg_finish_seed(p,k);
            if (k.refine && (!all(isfinite(seed.pose)) || any(abs(seed.pose)>float4(1e6f,1e6f,36000.f,3.f))
                || !all(seed.lower<=seed.pose) || !all(seed.pose<=seed.upper))) {
                status.overflow|=8u;continue;
            }
            seeds[status.proposals]=seed;states[status.proposals]=seed;
            ++status.proposals;
        }
    }
    poseConfig.count=status.proposals;poseConfig.unused=k.optimizeMask;
}

inline float fg_finish_halton(uint index,uint base)
{
    float value=0,factor=1;
    while (index) {factor/=float(base);value+=factor*float(index%base);index/=base;}
    return value;
}

// One thread; output capacities are 64 trials, 64 owners, four parents and four
// base scores. Ranking, budget, duplicate checks and Config.count stay on GPU.
kernel void fg_prepare_rescue(device const ResidentState* seeds [[buffer(0)]],
    device const ResidentState* baseline [[buffer(1)]],device ResidentState* retries [[buffer(2)]],
    device uint* owners [[buffer(3)]],device uint* parents [[buffer(4)]],
    device float* baseScores [[buffer(5)]],device Config& retryConfig [[buffer(6)]],
    device FGStatus& status [[buffer(7)]],constant FGFinishConfig& k [[buffer(8)]],
    device const Config& baselineConfig [[buffer(9)]],uint id [[thread_position_in_grid]])
{
    if (id) return;
    retryConfig=baselineConfig;retryConfig.count=0;
    status.rescueParents=status.rescueStarts=status.rescueContinuations=0;
    if (!k.refine) return;
    uint ranked[4],ranks=0;
    const float upper=min(.8f,k.minScore),lower=min(.1f,upper);
    for (uint rank=0;rank<4;++rank) {
        uint best=0;float bestScore=-INFINITY;bool found=false;
        for (uint i=0;i<status.proposals;++i) {
            const float score=baseline[i].score;
            if (!isfinite(score) || score<lower || score>=upper) continue;
            bool used=false;for (uint j=0;j<ranks;++j) used|=ranked[j]==i;
            if (!used && (!found || score>bestScore)) {best=i;bestScore=score;found=true;}
        }
        if (!found) break;
        ranked[ranks++]=best;
    }
    const ulong perParent=ulong(k.tw)*ulong(k.th)*ulong(16u*8u+30u);
    ulong remaining=ulong(512u)*1024u*1024u;
    const uint bases[4]={2,3,5,7};
    const float4 radii=float4((k.optimizeMask&1u)?3.f:0.f,(k.optimizeMask&1u)?3.f:0.f,
        (k.optimizeMask&2u)?6.f:0.f,(k.optimizeMask&4u)?.06f:0.f);
    for (uint rank=0;rank<ranks;++rank) {
        if (perParent>remaining) continue;
        const uint parent=ranked[rank],first=status.rescueStarts;
        const ResidentState original=seeds[parent];
        for (uint index=1;index<=16;++index) {
            ResidentState trial=original;
            for (uint d=0;d<4;++d)
                trial.pose[d]=clamp(original.pose[d]+(2.f*fg_finish_halton(index,bases[d])-1.f)*radii[d],
                    original.lower[d],original.upper[d]);
            if (all(trial.pose==original.pose)) continue;
            bool duplicate=false;
            for (uint previous=first;previous<status.rescueStarts;++previous)
                duplicate|=all(trial.pose==retries[previous].pose);
            if (duplicate) continue;
            trial.packed=resident_pack(trial.pose);
            if (status.rescueStarts>=64) {status.overflow|=4u;break;}
            retries[status.rescueStarts]=trial;owners[status.rescueStarts]=parent;++status.rescueStarts;
        }
        if (status.rescueStarts==first) continue;
        parents[status.rescueParents]=parent;baseScores[status.rescueParents]=baseline[parent].score;
        ++status.rescueParents;remaining-=perParent;
    }
    retryConfig.count=status.rescueStarts;
}

kernel void fg_select_continuations(device ResidentState* baseline [[buffer(0)]],
    device const ResidentState* retries [[buffer(1)]],device const uint* owners [[buffer(2)]],
    device const uint* parents [[buffer(3)]],device const float* baseScores [[buffer(4)]],
    device ResidentState* continuations [[buffer(5)]],device uint* continuationOwners [[buffer(6)]],
    device Config& continuationConfig [[buffer(7)]],device FGStatus& status [[buffer(8)]],
    device const Config& retryConfig [[buffer(9)]],uint id [[thread_position_in_grid]])
{
    if (id) return;
    for (uint i=0;i<status.rescueStarts;++i) {
        const uint owner=owners[i];
        if (isfinite(retries[i].score) && retries[i].score>baseline[owner].score) baseline[owner]=retries[i];
    }
    status.rescueContinuations=0;
    for (uint i=0;i<status.rescueParents;++i) {
        const uint parent=parents[i];
        if (!(baseline[parent].score>baseScores[i])) continue;
        continuations[status.rescueContinuations]=baseline[parent];
        continuationOwners[status.rescueContinuations]=parent;++status.rescueContinuations;
    }
    continuationConfig=retryConfig;continuationConfig.count=status.rescueContinuations;
}

kernel void fg_accept_continuations(device ResidentState* baseline [[buffer(0)]],
    device const ResidentState* continuations [[buffer(1)]],device const uint* owners [[buffer(2)]],
    device const FGStatus& status [[buffer(3)]],uint id [[thread_position_in_grid]])
{
    if (id) return;
    for (uint i=0;i<status.rescueContinuations;++i) {
        const uint owner=owners[i];
        if (isfinite(continuations[i].score) && continuations[i].score>baseline[owner].score)
            baseline[owner]=continuations[i];
    }
}

// Optional common score pass before comparisons/final publication. Uses exactly
// the fused solver's FP32 score arithmetic; no CPU score verifier participates.
kernel void fg_score_states(device const float* source [[buffer(0)]],
    device const float* templ [[buffer(1)]],device ResidentState* states [[buffer(2)]],
    constant Config& k [[buffer(3)]],uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    if (group>=k.count) return;
    threadgroup DerivativePartial scratch[64];threadgroup float score;
    fused_score(source,templ,states[group].packed,k,lane,scratch,score);
    if (lane==0) states[group].score=score;
}

kernel void fg_publish_states(device FGCandidate* candidates [[buffer(0)]],
    device const ResidentState* states [[buffer(1)]],device const FGStatus& status [[buffer(2)]],
    constant FGFinishConfig& k [[buffer(3)]],uint id [[thread_position_in_grid]])
{
    if (id>=status.proposals) return;
    const ResidentState s=states[id];
    device FGCandidate& p=candidates[id];
    p.x=s.pose.x+float(k.sw)*.5f;p.y=float(k.sh)*.5f-s.pose.y;
    float angle=fmod(s.pose.z+180.f,360.f);if (angle<0) angle+=360.f;
    p.angle=angle-180.f;p.scale=exp(s.pose.w);p.score=s.score;
    p.valid=all(isfinite(s.pose)) && isfinite(s.score) && s.score>=-1.f;
}

// Format only the declared GPU output count. kind=0 reads an immutable snapshot
// made after compaction and before refinement; kind=1 reads global NMS output.
// The CPU receives these records after the sole final wait and only serializes.
kernel void fg_write_results(device const FGCandidate* candidates [[buffer(0)]],
    device FGResult* output [[buffer(1)]],device const FGStatus& status [[buffer(2)]],
    constant FGFinishConfig& k [[buffer(3)]],constant uint& kind [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    const uint count=kind?status.matches:status.proposals;
    if (id>=count) return;
    const FGCandidate p=candidates[id];FGResult r={};
    r.x=p.x;r.y=p.y;r.angle=p.angle;r.score=p.score;r.scale=p.scale;
    float2 corners[4];
    fg_finish_corners(p,float(k.tw)*p.scale,float(k.th)*p.scale,corners);
    for (uint i=0;i<4;++i) {r.corners[2*i]=corners[i].x;r.corners[2*i+1]=corners[i].y;}
    output[id]=r;
}
