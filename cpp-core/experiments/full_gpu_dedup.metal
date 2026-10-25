// SPDX-License-Identifier: BSD-2-Clause
// Round 43: experimental, SEMANTICS-CHANGING coarse-candidate deduplication.
// Same rule as fpm::detail::dedupKeep, in FP32: rank by (coarse score desc,
// slot order asc); a candidate is dropped when at least topK higher-ranked
// candidates of its model have centers within radius source pixels and
// wrapped search angles within angleWindow. Earlier candidates count whether
// or not they are dropped, so every slot decides independently. Each map's
// valid slots are a prefix in descending selection order. No host decisions.
inline float2 fg_dedup_center(FGCandidate c,constant FGDedupConfig& k) {
    const float radians=c.angle*0.01745329251994329577f,cs=cos(radians),sn=sin(radians);
    return float2(c.x+.5f*(float(k.templateWidth)*cs-float(k.templateHeight)*sn),
                  c.y+.5f*(float(k.templateWidth)*sn+float(k.templateHeight)*cs))*k.toSource;
}
kernel void fg_dedup_mark(device const FGCandidate* candidates[[buffer(0)]],device uint* drop[[buffer(1)]],
    device atomic_uint* counts[[buffer(2)]],constant FGDedupConfig& k[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=k.maps*k.limit) return;
    const FGCandidate own=candidates[k.first+i];
    drop[k.first+i]=0;
    if(!own.valid) return;
    const float2 center=fg_dedup_center(own,k);const float limit=k.radius*k.radius;
    uint higher=0;
    for(uint map=0;map<k.maps && higher<k.topK;++map) {
        const uint base=k.first+map*k.limit;
        const FGCandidate head=candidates[base];
        if(!head.valid || fabs(fg_angle(own.angle-head.angle))>k.angleWindow+1e-6f) continue;
        for(uint j=0;j<k.limit;++j) {
            const FGCandidate other=candidates[base+j];
            // Lower scores follow; equal scores rank by slot order.
            if(!other.valid || other.score<own.score) break;
            if(!fg_better(other.score,other.order,own.score,own.order)) continue;
            const float2 delta=fg_dedup_center(other,k)-center;
            if(dot(delta,delta)<=limit && fabs(fg_angle(own.angle-other.angle))<=k.angleWindow+1e-6f && ++higher>=k.topK) break;
        }
    }
    const bool dropped=higher>=k.topK;
    drop[k.first+i]=dropped;
    atomic_fetch_add_explicit(&counts[2*k.model],1u,memory_order_relaxed);
    if(!dropped) atomic_fetch_add_explicit(&counts[2*k.model+1],1u,memory_order_relaxed);
}
kernel void fg_dedup_apply(device FGCandidate* candidates[[buffer(0)]],device const uint* drop[[buffer(1)]],
    constant FGDedupConfig& k[[buffer(2)]],uint i[[thread_position_in_grid]]) {
    if(i<k.maps*k.limit && drop[k.first+i]) candidates[k.first+i].valid=0;
}
