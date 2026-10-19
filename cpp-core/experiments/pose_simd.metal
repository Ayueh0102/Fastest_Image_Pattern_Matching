// SPDX-License-Identifier: BSD-2-Clause
// Opt-in mode 2: 128 physical threads preserve the original logical reduction
// trees. Host requires 32-wide SIMD groups. No fast-math or reduced precision.
// Scratch is 64*104=6656 bytes, shared between score and derivatives.
struct GrayDerivative { float mean[5], covariance[15], product[5]; uint count; };
static_assert(sizeof(GrayDerivative) == 104, "gray derivative layout mismatch");
inline void simd_merge_gray(thread GrayDerivative& a, thread const GrayDerivative& b)
{
    const uint total = a.count+b.count;
    if (!total) return;
    const float weight = float(b.count)/float(total);
    const float correction = float(a.count)*float(b.count)/float(total);
    for (uint i=0;i<5;++i) a.product[i] += b.product[i];
    for (uint i=0;i<15;++i) a.covariance[i] += b.covariance[i];
    float delta[5];
    for (uint i=0;i<5;++i) { delta[i]=b.mean[i]-a.mean[i]; a.mean[i]+=delta[i]*weight; }
    uint entry=0;
    for (uint i=0;i<5;++i) for (uint j=i;j<5;++j)
        a.covariance[entry++] += delta[i]*delta[j]*correction;
    a.count=total;
}
inline GrayDerivative simd_shift_gray(GrayDerivative a, ushort step)
{
    GrayDerivative b;
    // Every lane in the first SIMD group executes every shuffle, including
    // inactive reduction owners. Only lanes <step consume the shifted values.
    for (uint i=0;i<5;++i) { b.mean[i]=simd_shuffle_down(a.mean[i],step); b.product[i]=simd_shuffle_down(a.product[i],step); }
    for (uint i=0;i<15;++i) b.covariance[i]=simd_shuffle_down(a.covariance[i],step);
    b.count=simd_shuffle_down(a.count,step);
    return b;
}
struct GrayScore { float mean, variance, product; uint count; };
inline void simd_merge_score(thread GrayScore& a, thread const GrayScore& b)
{
    const uint total=a.count+b.count;
    const float delta=b.mean-a.mean;
    if (total) {
        a.mean += delta*(float(b.count)/float(total));
        a.variance += b.variance+delta*delta*(float(a.count)*float(b.count)/float(total));
    }
    a.product += b.product; a.count=total;
}
inline GrayScore simd_score_lane(device const float* source, device const float* templ,
    float4 p, constant Config& k, uint tile, uint logicalLane)
{
    GrayScore a={};
    for(uint pixel=tile*1024+logicalLane; pixel<min((tile+1)*1024,k.tw*k.th);pixel+=256) {
        const float u=float(pixel%k.tw)-float(k.tw)*.5f;
        const float v=float(k.th)*.5f-float(pixel/k.tw);
        const float sx=float(k.sw)*.5f+p.x+p.z*u-p.w*v;
        const float sy=float(k.sh)*.5f-p.y-(p.w*u+p.z*v);
        const int ix=int(floor(sx)),iy=int(floor(sy));
        const float dx=sx-float(ix),dy=sy-float(iy);
        const float x=load_pixel(source,ix,iy,0,k),b=load_pixel(source,ix+1,iy,0,k);
        const float e=load_pixel(source,ix,iy+1,0,k),f=load_pixel(source,ix+1,iy+1,0,k);
        const float sample=(x+(b-x)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
        a.product+=sample*templ[pixel]; ++a.count;
        const float delta=sample-a.mean;
        a.mean+=delta/float(a.count); a.variance+=delta*(sample-a.mean);
    }
    return a;
}
inline void simd_score(device const float* source, device const float* templ,
    float4 p, constant Config& k, uint lane, threadgroup GrayDerivative* scratch,
    threadgroup float& result)
{
    threadgroup GrayScore* shared=reinterpret_cast<threadgroup GrayScore*>(scratch);
    GrayScore total={};
    for(uint tile=0;tile<k.tiles;++tile) {
        GrayScore a=simd_score_lane(source,templ,p,k,tile,lane);
        const GrayScore b=simd_score_lane(source,templ,p,k,tile,lane+128);
        simd_merge_score(a,b); // original step 128, before any cross-lane merge
        if(lane>=64) shared[lane-64]=a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane<64) { const GrayScore b=shared[lane]; simd_merge_score(a,b); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane<64) shared[lane]=a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane<32) {
            const GrayScore b=shared[lane+32]; simd_merge_score(a,b);
            for(ushort step=16;step;step>>=1) {
                const GrayScore b={simd_shuffle_down(a.mean,step),simd_shuffle_down(a.variance,step),
                    simd_shuffle_down(a.product,step),simd_shuffle_down(a.count,step)};
                if(lane<step) simd_merge_score(a,b);
            }
            if(lane==0) simd_merge_score(total,a);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0) {
        const float n=float(k.tw*k.th),energy=max(0.f,total.variance);
        const float numerator=total.product-total.mean*k.templateSum.x;
        result=energy/n<=1e-12f || k.templateEnergy/n<=1e-12f ? -2.f :
            clamp(numerator/sqrt(energy*k.templateEnergy),-1.f,1.f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void simd_derivatives(device const float* source, device const float* templ,
    float4 p, constant Config& k, uint lane,
    threadgroup GrayDerivative* scratch, threadgroup GrayDerivative& total)
{
    if (lane == 0) total = {};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < k.tiles; ++tile) {
        GrayDerivative state = {};
        if (lane < 128) {
            for (uint pixel = tile*1024+lane;
                pixel < min((tile+1)*1024, k.tw*k.th); pixel += 128) {
                const float u = float(pixel%k.tw)-float(k.tw)*.5f;
                const float v = float(k.th)*.5f-float(pixel/k.tw);
                const float rx = p.z*u-p.w*v, ry = p.w*u+p.z*v;
                const float sx = float(k.sw)*.5f+p.x+p.z*u-p.w*v;
                const float sy = float(k.sh)*.5f-p.y-(p.w*u+p.z*v);
                const int ix = int(floor(sx)), iy = int(floor(sy));
                const float dx = sx-float(ix), dy = sy-float(iy);
                ++state.count;
                const float a = load_pixel(source, ix, iy, 0, k);
                const float b = load_pixel(source, ix+1, iy, 0, k);
                const float e = load_pixel(source, ix, iy+1, 0, k);
                const float f = load_pixel(source, ix+1, iy+1, 0, k);
                const float sample = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
                const float gx = (b-a)*(1-dy)+(f-e)*dy;
                const float gy = (e-a)*(1-dx)+(f-b)*dx;
                float values[5] = {sample, gx, -gy, -gx*ry-gy*rx, gx*rx-gy*ry};
                if (!(k.unused & 1u)) values[1] = values[2] = 0;
                if (!(k.unused & 2u)) values[3] = 0;
                if (!(k.unused & 4u)) values[4] = 0;
                const float t = templ[pixel];
                float delta[5];
                for (uint i = 0; i < 5; ++i) {
                    delta[i] = values[i]-state.mean[i];
                    state.mean[i] += delta[i]/float(state.count);
                    state.product[i] += values[i]*t;
                }
                uint entry = 0;
                for (uint i = 0; i < 5; ++i) for (uint j = i; j < 5; ++j)
                    state.covariance[entry++] += delta[i]*(values[j]-state.mean[j]);
            }
        }
        if (lane >= 64 && lane < 128) scratch[lane-64] = state;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 64) {
            const GrayDerivative right = scratch[lane];
            simd_merge_gray(state, right);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 64) scratch[lane] = state;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 32) {
            const GrayDerivative right=scratch[lane+32];
            simd_merge_gray(state,right); // original step 32
            for (ushort step=16;step;step>>=1) {
                const GrayDerivative right=simd_shift_gray(state,step);
                if(lane<step) simd_merge_gray(state,right);
            }
            if(lane==0) {
                GrayDerivative accumulated=total;
                simd_merge_gray(accumulated,state); total=accumulated;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


kernel void pose_refine_simd(device const float* source [[buffer(0)]],
    device const float* templ [[buffer(1)]], device ResidentState* states [[buffer(2)]],
    constant Config& k [[buffer(3)]], constant uint& maxIterations [[buffer(4)]],
    uint simdLane [[thread_index_in_simdgroup]], uint simdGroup [[simdgroup_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]])
{
    if (group >= k.count) return;
    const uint lane=simdGroup*32+simdLane;
    // 6,656 + 104 + 208 + 4 bytes before alignment; no global partial arrays.
    threadgroup GrayDerivative scratch[64], derivative;
    threadgroup ResidentState s;
    threadgroup float trialScore;
    if (lane == 0) s = states[group];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simd_score(source, templ, s.packed, k, lane, scratch, s.score);
    if (lane == 0) {
        s.damping = 1e-3f; s.iterations = 0; s.evaluations = 1;
        s.pending = s.proposed = 0;
        s.status = !isfinite(s.score) ? 3u : (s.score == -2.f ? 2u : (s.score == 1.f ? 0u : 1u));
        s.active = s.status == 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint iteration = 0; iteration < maxIterations; ++iteration) {
        const bool runIteration = s.active != 0;
        // Latch shared control before lane zero is allowed to change it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!runIteration) break;
        simd_derivatives(source, templ, s.packed, k, lane, scratch, derivative);
        if (lane == 0) {
            DerivativePartial value = {};
            value.count=derivative.count;
            for(uint i=0;i<5;++i) { value.mean[0][i]=derivative.mean[i]; value.product[i]=derivative.product[i]; }
            for(uint i=0;i<15;++i) value.covariance[i]=derivative.covariance[i];
            ++s.iterations; ++s.evaluations;
            if (!fused_assemble(value, s, k)) {
                s.active = s.pending = 0; s.status = 2;
            } else s.pending = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint trial = 0; trial < 10; ++trial) {
            const bool runTrial = s.active && s.pending;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!runTrial) break;
            if (lane == 0) {
                s.proposed = 0;
                float4 step;
                if (!fused_solve(s, step)) {
                    s.damping *= 10.f;
                } else {
                    step.z /= 0.01745329251994329577f;
                    const float ratio = max(1.f, max(max(abs(step.x)/2.f, abs(step.y)/2.f),
                        max(abs(step.z)/5.f, abs(step.w)/.1f)));
                    // Bounds were rounded inward from the seed on the host.
                    const float4 proposal = clamp(s.pose+step/ratio, s.lower, s.upper);
                    step = proposal-s.pose;
                    if (max(abs(step.x), abs(step.y)) < 1e-4f && abs(step.z) < 1e-4f
                        && abs(step.w) < 1e-6f) {
                        s.active = s.pending = 0; s.status = 0;
                    } else {
                        s.candidate = proposal; s.candidatePacked = resident_pack(proposal);
                        if (!all(isfinite(s.candidatePacked))) {
                            s.active = s.pending = 0; s.status = 3;
                        } else s.proposed = 1;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The shared predicate was published by lane zero; all 128 lanes
            // either execute the entire scorer (including barriers), or skip it.
            if (s.proposed) {
                simd_score(source, templ, s.candidatePacked, k, lane, scratch, trialScore);
                if (lane == 0) {
                    ++s.evaluations;
                    if (isfinite(trialScore) && trialScore > s.score+2.384185791015625e-7f) {
                        s.pose = s.candidate; s.packed = s.candidatePacked; s.score = trialScore;
                        s.damping = max(1e-8f, s.damping*.3f); s.pending = 0;
                    } else s.damping *= 10.f;
                    s.proposed = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0 && s.active && s.pending) {
            s.active = s.pending = 0; s.status = 3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) states[group] = s;
}
