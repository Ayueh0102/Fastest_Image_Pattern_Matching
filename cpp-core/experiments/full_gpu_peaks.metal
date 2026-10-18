// SPDX-License-Identifier: BSD-2-Clause
// Exact tiled counterpart to fg_coarse_peaks, with two dispatches per map.
//
// Each tile is a complete sorted run, not a truncated local top-K. The merge
// invariant is that a lane's cached winner is the best unsuppressed entry in
// all tiles owned by that lane. Suppression can only remove entries, so that
// winner remains correct until it is suppressed. Only then are its tile heads
// advanced past suppressed entries. Each cursor is monotone. Global reduction
// uses the baseline (descending FP32 score, ascending original index) order.
//
// A private score-map is still mutated to -INFINITY for the identical half-open
// suppression rectangle. It serves as the shared tombstone map; no host reads,
// global spin barriers, per-peak dispatches, or approximate pruning are needed.

kernel void fg_coarse_sort_tiles(
    device const float* scores[[buffer(0)]],
    device FGPeakEntry* entries[[buffer(1)]],
    constant FGCoarse& c[[buffer(2)]],
    device uint* heads[[buffer(3)]],
    uint lane[[thread_index_in_threadgroup]],
    uint tile[[threadgroup_position_in_grid]]) {
    threadgroup float values[FG_PEAK_TILE_SIZE];
    threadgroup uint indices[FG_PEAK_TILE_SIZE];
    const uint index=tile*FG_PEAK_TILE_SIZE+lane;
    // Baseline comparisons never select NaN. Normalizing it to -INFINITY is
    // equivalent, since non-finite winning scores terminate without an output.
    const float score=index<c.windows ? scores[index] : -INFINITY;
    values[lane]=isnan(score) ? -INFINITY:score;
    indices[lane]=index<c.windows ? index:UINT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint size=2;size<=FG_PEAK_TILE_SIZE;size*=2) {
        for(uint stride=size/2;stride;stride/=2) {
            const uint peer=lane^stride;
            const float ownValue=values[lane],peerValue=values[peer];
            const uint ownIndex=indices[lane],peerIndex=indices[peer];
            // Both lanes independently choose their side of the exchange.
            // The read barrier is required before either lane overwrites its
            // slot; SIMD scheduling alone is not a workgroup memory contract.
            const bool first=(lane&stride)==0;
            const bool descending=(lane&size)==0;
            const bool wantBetter=first==descending;
            const bool peerBetter=fg_better(peerValue,peerIndex,ownValue,ownIndex);
            const bool takePeer=wantBetter ? peerBetter:fg_better(ownValue,ownIndex,peerValue,peerIndex);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(takePeer) {values[lane]=peerValue;indices[lane]=peerIndex;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    entries[index]={values[lane],indices[lane]};
    if(!lane) heads[tile]=0;
}

kernel void fg_coarse_peaks_tiled(
    device float* scores[[buffer(0)]],
    device FGCandidate* candidates[[buffer(1)]],
    constant FGCoarse& c[[buffer(2)]],
    device const FGPeakEntry* entries[[buffer(3)]],
    device uint* heads[[buffer(4)]],
    uint lane[[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    const uint width=c.w-c.tw+1,height=c.h-c.th+1;
    const uint tiles=(c.windows+FG_PEAK_TILE_SIZE-1)/FG_PEAK_TILE_SIZE;
    const uint stripe=(tiles+255)/256;
    const uint begin=min(tiles,lane*stripe),end=min(tiles,begin+stripe);
    float best=-INFINITY;uint bi=UINT_MAX;
    for(uint tile=begin;tile<end;++tile) {
        const FGPeakEntry head=entries[tile*FG_PEAK_TILE_SIZE];
        if(fg_better(head.score,head.index,best,bi)) {best=head.score;bi=head.index;}
    }
    for(uint iteration=0;iteration<c.limit;++iteration) {
        values[lane]=best;indices[lane]=bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint step=128;step;step/=2) {
            if(lane<step && fg_better(values[lane+step],indices[lane+step],values[lane],indices[lane])) {
                values[lane]=values[lane+step];indices[lane]=indices[lane+step];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float score=values[0];const uint index=indices[0];
        if(!isfinite(score)||score<c.threshold) break;
        const uint x=index%width,y=index/width;
        if(lane==0) {
            const float cx=float(c.sw-1)*.5f,cy=float(c.sh-1)*.5f;
            const float rx=float(x)-c.translationX-cx,ry=float(y)-c.translationY-cy;
            candidates[c.first+iteration]={cx+c.cosine*rx-c.sine*ry,cy+c.sine*rx+c.cosine*ry,
                c.angle,score,c.scale,score,1,c.first+iteration,c.model,0,0,0};
        }
        // Match baseline score-map mutation: its final accepted peak is not
        // suppressed when the requested output limit has been reached.
        if(iteration+1==c.limit) break;
        const float rw=c.tw*(1-c.overlap),rh=c.th*(1-c.overlap);
        const int left=int(float(x)-rw),top=int(float(y)-rh);
        const uint x0=uint(max(0,min(int(x),left)));
        const uint x1=uint(min(int(width),max(int(x)+1,left+int(2*rw))));
        const uint y0=uint(max(0,min(int(y),top)));
        const uint y1=uint(min(int(height),max(int(y)+1,top+int(2*rh))));
        const uint w=x1-x0,pixels=w*(y1-y0);
        for(uint j=lane;j<pixels;j+=256) scores[(y0+j/w)*width+x0+j%w]=-INFINITY;
        threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        if(bi!=UINT_MAX && bi%width>=x0 && bi%width<x1 && bi/width>=y0 && bi/width<y1) {
            best=-INFINITY;bi=UINT_MAX;
            for(uint tile=begin;tile<end;++tile) {
                uint cursor=heads[tile];
                while(cursor<FG_PEAK_TILE_SIZE) {
                    const FGPeakEntry head=entries[tile*FG_PEAK_TILE_SIZE+cursor];
                    // A sorted suffix below threshold can never produce an
                    // accepted output. +INFINITY remains a winning sentinel,
                    // exactly as in the baseline; do not skip it here.
                    if(head.index==UINT_MAX || head.score<c.threshold || head.score==-INFINITY) {
                        cursor=FG_PEAK_TILE_SIZE;break;
                    }
                    if(scores[head.index]!=-INFINITY) {
                        if(fg_better(head.score,head.index,best,bi)) {best=head.score;bi=head.index;}
                        break;
                    }
                    ++cursor;
                }
                heads[tile]=cursor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
