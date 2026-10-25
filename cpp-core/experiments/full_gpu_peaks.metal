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

// Peak mode 2: exact parallel rounds, one threadgroup per map and a batch of
// maps per dispatch. Each map has its own ring slot of scores, so the maps no
// longer serialize on one shared score buffer.
//
// Candidates are finite scores not below threshold; a +INFINITY score is the
// baseline's first winner and stops it without output. Priority is the
// baseline order (fg_better). The baseline's picks G satisfy: a candidate is in
// G iff no higher-priority member of G covers it with its suppression
// rectangle. Each round accepts every live candidate that no higher-priority
// live candidate covers, then removes the live candidates that a higher-
// priority point accepted in this round covers. The rectangles are asymmetric
// (trunc(x-rw), int(2*rw)), so a point can cover a higher-priority neighbour;
// that neighbour is never removed by it. Accepted points are in G, removed
// points are not, and the best live candidate is always accepted, so every
// round progresses. Once limit points are accepted and none of the live ones
// ranks above the limit-th, the first limit members of G are final. They are
// written in priority order to the baseline's slots with its exact values.
//
// State is double-buffered: each phase reads neighbours from one array and
// writes only its own entries of the other, with barriers between phases.
// Scores are never mutated. Picking first tests each live point against its
// eight neighbours; the rest are tested by a whole SIMD group, its lanes
// splitting the neighbourhood. Settling scatters from each picked point over
// its rectangle into an atomic removal bitmask, then owners apply the bits.
#define FG_PEAK_DEAD 0
#define FG_PEAK_LIVE 1
#define FG_PEAK_PICKED 2
#define FG_PEAK_ACCEPTED 3

struct FGPeakRect { uint x0,x1,y0,y1; };
// Identical arithmetic to the baseline's suppression rectangle.
inline FGPeakRect fg_peak_rect(uint x,uint y,uint width,uint height,float rw,float rh) {
    const int left=int(float(x)-rw),top=int(float(y)-rh);
    return {uint(max(0,min(int(x),left))),uint(min(int(width),max(int(x)+1,left+int(2*rw)))),
        uint(max(0,min(int(y),top))),uint(min(int(height),max(int(y)+1,top+int(2*rh))))};
}

// Whether some q!=p with states[q]==want and higher priority covers p; called
// by a whole SIMD group with the same p, so the result is uniform. A rectangle
// reaches at most ceil(rw) columns from its centre; reach adds slack. Rows are
// visited centre first (0,-1,+1,-2,...), where covers are usually found.
inline bool fg_peak_covered(device const float* scores,device const uchar* states,uchar want,uint p,
    uint width,uint height,float rw,float rh,uint reachX,uint reachY,uint simdLane,uint simdWidth) {
    const uint px=p%width,py=p/width;const float score=scores[p];
    const uint columns=2*reachX+1,cells=columns*(2*reachY+1);
    for(uint base=0;base<cells;base+=simdWidth) {
        const uint t=base+simdLane,row=t/columns;
        const int dy=(row&1) ? -int((row+1)/2):int(row/2),dx=int(t%columns)-int(reachX);
        const int qx=int(px)+dx,qy=int(py)+dy;
        bool covers=false;
        if(t<cells && (dx||dy) && qx>=0 && qy>=0 && qx<int(width) && qy<int(height)) {
            const uint q=uint(qy)*width+uint(qx);
            if(states[q]==want && fg_better(scores[q],q,score,p)) {
                const FGPeakRect r=fg_peak_rect(uint(qx),uint(qy),width,height,rw,rh);
                covers=px>=r.x0 && px<r.x1 && py>=r.y0 && py<r.y1;
            }
        }
        if(simd_any(covers)) return true;
    }
    return false;
}

// Lane-local shortcut: whether one of p's eight neighbours with higher priority
// and states[q]==want covers p. A false result proves nothing.
inline bool fg_peak_near(device const float* scores,device const uchar* states,uchar want,uint p,
    uint width,uint height,float rw,float rh) {
    const uint px=p%width,py=p/width;const float score=scores[p];
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int qx=int(px)+dx,qy=int(py)+dy;
        if((!dx && !dy) || qx<0 || qy<0 || qx>=int(width) || qy>=int(height)) continue;
        const uint q=uint(qy)*width+uint(qx);
        if(states[q]!=want || !fg_better(scores[q],q,score,p)) continue;
        const FGPeakRect r=fg_peak_rect(uint(qx),uint(qy),width,height,rw,rh);
        if(px>=r.x0 && px<r.x1 && py>=r.y0 && py<r.y1) return true;
    }
    return false;
}

// A SIMD group marks every live point of lower priority inside q's rectangle.
inline void fg_peak_suppress(device const float* scores,device const uchar* marks,device atomic_uint* removal,uint q,
    uint width,uint height,float rw,float rh,uint simdLane,uint simdWidth) {
    const uint qx=q%width,qy=q/width;const float score=scores[q];
    const FGPeakRect r=fg_peak_rect(qx,qy,width,height,rw,rh);
    const uint w=r.x1-r.x0,cells=w*(r.y1-r.y0);
    for(uint t=simdLane;t<cells;t+=simdWidth) {
        const uint p=(r.y0+t/w)*width+r.x0+t%w;
        if(marks[p]==FG_PEAK_LIVE && fg_better(score,q,scores[p],p))
            atomic_fetch_or_explicit(&removal[p/32],1u<<(p%32),memory_order_relaxed);
    }
}

// Ascending keys follow fg_better on finite scores: score descending with
// -0 equal to +0, then index ascending.
inline ulong fg_peak_key(float score,uint index) {
    const uint bits=as_type<uint>(score==0 ? 0.f:score);
    const uint ordered=(bits&0x80000000u) ? ~bits:(bits|0x80000000u);
    return ulong(~ordered)<<32|index;
}

inline uint fg_peak_sum(uint value,threadgroup uint* scratch,uint lane,uint lanes) {
    scratch[lane]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=lanes/2;step;step/=2) {
        if(lane<step) scratch[lane]+=scratch[lane+step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint total=scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

inline void fg_peak_best(thread float& score,thread uint& index,threadgroup float* values,threadgroup uint* indices,uint lane,uint lanes) {
    values[lane]=score;indices[lane]=index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=lanes/2;step;step/=2) {
        if(lane<step && fg_better(values[lane+step],indices[lane+step],values[lane],indices[lane])) {
            values[lane]=values[lane+step];indices[lane]=indices[lane+step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    score=values[0];index=indices[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Appends flagged lanes of a SIMD group at consecutive indices of counter's
// list; called uniformly by the whole group. Returns this lane's index.
inline uint fg_peak_append(bool flag,threadgroup atomic_uint* counter,uint simdLane) {
    const simd_vote::vote_t mask=simd_vote::vote_t(simd_ballot(flag));
    uint base=0;
    if(!simdLane && mask) base=atomic_fetch_add_explicit(counter,uint(popcount(mask)),memory_order_relaxed);
    base=simd_broadcast(base,0);
    return base+uint(popcount(mask&((simd_vote::vote_t(1)<<simdLane)-1)));
}

// Dispatched with a power-of-two lane count, 256..FG_PEAK_PARALLEL_LANES.
// When a map has at most batch.listCapacity candidates, rounds visit only the
// live list and accepted list; otherwise every phase scans the whole map. Both
// visit the same points in the same phases, so their outputs are identical.
kernel void fg_coarse_peaks_parallel(
    device const float* scoreRing[[buffer(0)]],
    device FGCandidate* candidates[[buffer(1)]],
    constant FGCoarse* maps[[buffer(2)]],
    device uchar* stateRing[[buffer(3)]],
    device ulong* keyRing[[buffer(4)]],
    constant FGPeakBatch& batch[[buffer(5)]],
    device atomic_uint* removalRing[[buffer(6)]],
    device uint* listRing[[buffer(7)]],
    uint lane[[thread_index_in_threadgroup]],
    uint lanes[[threads_per_threadgroup]],
    uint slot[[threadgroup_position_in_grid]],
    uint simdLane[[thread_index_in_simdgroup]],
    uint simdWidth[[threads_per_simdgroup]],
    uint simdGroup[[simdgroup_index_in_threadgroup]],
    uint simdGroups[[simdgroups_per_threadgroup]]) {
    threadgroup float values[FG_PEAK_PARALLEL_LANES];
    threadgroup uint indices[FG_PEAK_PARALLEL_LANES];
    threadgroup atomic_uint histogram[256];
    // Live lists 0/1, the accepted list, and the gathered output keys.
    threadgroup atomic_uint counters[4];
    constant FGCoarse& c=maps[slot];
    device const float* scores=scoreRing+ulong(slot)*batch.scoreStride;
    device uchar* current=stateRing+2*ulong(slot)*batch.scoreStride;
    device uchar* marks=current+batch.scoreStride;
    device ulong* keys=keyRing+ulong(slot)*batch.keyStride;
    device atomic_uint* removal=removalRing+ulong(slot)*(batch.scoreStride/32);
    const uint capacity=batch.listCapacity;
    device uint* lists=listRing+3*ulong(slot)*capacity;
    device uint* acceptedList=lists+2*capacity;
    const uint windows=c.windows,width=c.w-c.tw+1,height=c.h-c.th+1,words=(windows+31)/32;
    if(!c.limit || !windows) return;
    if(lane<4) atomic_store_explicit(&counters[lane],0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint infinite=0;
    for(uint run=simdGroup;run<(windows+simdWidth-1)/simdWidth;run+=simdGroups) {
        const uint i=run*simdWidth+simdLane;
        const float score=i<windows ? scores[i]:0;
        infinite|=uint(i<windows && isinf(score) && score>0);
        const bool candidate=i<windows && isfinite(score) && !(score<c.threshold);
        if(i<windows) {current[i]=candidate ? FG_PEAK_LIVE:FG_PEAK_DEAD;marks[i]=current[i];}
        const uint at=fg_peak_append(candidate,&counters[0],simdLane);
        if(candidate && at<capacity) lists[at]=i;
    }
    for(uint i=lane;i<words;i+=lanes) atomic_store_explicit(&removal[i],0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    if(fg_peak_sum(infinite,indices,lane,lanes)) return;
    uint live=atomic_load_explicit(&counters[0],memory_order_relaxed);
    const bool listed=live<=capacity;
    const float rw=c.tw*(1-c.overlap),rh=c.th*(1-c.overlap);
    const uint reachX=uint(ceil(rw))+2,reachY=uint(ceil(rh))+2;
    uint accepted=0,from=0;
    for(;;) {
        // Points of this round: the live list, or every window.
        device const uint* entries=lists+from*capacity;
        const uint domain=listed ? live:windows,runs=(domain+simdWidth-1)/simdWidth;
        if(!lane) atomic_store_explicit(&counters[1-from],0,memory_order_relaxed);
        // Pick: neighbours read current; owners write their marks. Removal
        // bits are cleared once: a set bit's point is dead from then on.
        for(uint run=simdGroup;run<runs;run+=simdGroups) {
            const uint e=run*simdWidth+simdLane,p=e<domain ? (listed ? entries[e]:e):0;
            const bool open=e<domain && current[p]==FG_PEAK_LIVE && !fg_peak_near(scores,current,FG_PEAK_LIVE,p,width,height,rw,rh);
            for(simd_vote::vote_t pending=simd_vote::vote_t(simd_ballot(open));pending;pending&=pending-1) {
                const uint owner=ctz(pending),q=simd_shuffle(p,ushort(owner));
                const bool covered=fg_peak_covered(scores,current,FG_PEAK_LIVE,q,width,height,rw,rh,reachX,reachY,simdLane,simdWidth);
                if(simdLane==owner && !covered) marks[q]=FG_PEAK_PICKED;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        // Settle: neighbours read marks; picked owners accept, and each picked
        // point sets the atomic removal bits of lower-priority live points.
        uint picked=0;
        for(uint run=simdGroup;run<runs;run+=simdGroups) {
            const uint e=run*simdWidth+simdLane,p=e<domain ? (listed ? entries[e]:e):0;
            const bool chosen=e<domain && marks[p]==FG_PEAK_PICKED;
            if(chosen) {current[p]=FG_PEAK_ACCEPTED;++picked;}
            if(listed) {
                const uint at=fg_peak_append(chosen,&counters[2],simdLane);
                if(chosen) acceptedList[at]=p;
            }
            for(simd_vote::vote_t pending=simd_vote::vote_t(simd_ballot(chosen));pending;pending&=pending-1)
                fg_peak_suppress(scores,marks,removal,simd_shuffle(p,ushort(ctz(pending))),width,height,rw,rh,simdLane,simdWidth);
        }
        threadgroup_barrier(mem_flags::mem_device);
        // Owners apply removals and bring marks back to current.
        uint kept=0,bi=UINT_MAX;float best=-INFINITY;
        for(uint run=simdGroup;run<runs;run+=simdGroups) {
            const uint e=run*simdWidth+simdLane,p=e<domain ? (listed ? entries[e]:e):0;
            const uchar mark=e<domain ? marks[p]:FG_PEAK_DEAD;
            bool keep=false;
            if(mark==FG_PEAK_PICKED) marks[p]=FG_PEAK_ACCEPTED;
            else if(mark==FG_PEAK_LIVE) {
                if((atomic_load_explicit(&removal[p/32],memory_order_relaxed)>>(p%32))&1u) {current[p]=FG_PEAK_DEAD;marks[p]=FG_PEAK_DEAD;}
                else {
                    keep=true;++kept;const float score=scores[p];
                    if(fg_better(score,p,best,bi)) {best=score;bi=p;}
                }
            }
            if(listed) {
                const uint at=fg_peak_append(keep,&counters[1-from],simdLane);
                if(keep) lists[(1-from)*capacity+at]=p;
            }
        }
        threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        accepted+=fg_peak_sum(picked,indices,lane,lanes);
        live=fg_peak_sum(kept,indices,lane,lanes);
        if(!live) break;
        fg_peak_best(best,bi,values,indices,lane,lanes);
        if(accepted>=c.limit) {
            uint above=0;
            for(uint e=lane;e<(listed ? accepted:windows);e+=lanes) {
                const uint p=listed ? acceptedList[e]:e;
                above+=uint(current[p]==FG_PEAK_ACCEPTED && fg_better(scores[p],p,best,bi));
            }
            if(fg_peak_sum(above,indices,lane,lanes)>=c.limit) break;
        }
        from=1-from;
    }
    // Accepted points: the accepted list, or every window marked accepted.
    const uint domain=listed ? accepted:windows,outputs=min(accepted,c.limit);
    // Radix-select the limit-th smallest accepted key when more are accepted.
    ulong last=~0ul;
    if(accepted>c.limit) {
        ulong prefix=0;uint need=c.limit;
        for(uint pass=0;pass<8;++pass) {
            const uint shift=56-8*pass;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane<256) atomic_store_explicit(&histogram[lane],0,memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint e=lane;e<domain;e+=lanes) {
                const uint p=listed ? acceptedList[e]:e;
                if(current[p]!=FG_PEAK_ACCEPTED) continue;
                const ulong key=fg_peak_key(scores[p],p);
                if(!pass || key>>(shift+8)==prefix)
                    atomic_fetch_add_explicit(&histogram[uint(key>>shift)&255],1,memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Every lane reads the same counts, so the choice is uniform.
            uint digit=0,count=0;
            for(;digit<256;++digit) {
                count=atomic_load_explicit(&histogram[digit],memory_order_relaxed);
                if(count>=need) break;
                need-=count;
            }
            prefix=prefix<<8|digit;
            // Keys are unique, so the last pass always takes its whole bucket.
            if(count==need) {last=prefix<<shift|((1ul<<shift)-1);break;}
        }
    }
    for(uint e=lane;e<domain;e+=lanes) {
        const uint p=listed ? acceptedList[e]:e;
        if(current[p]!=FG_PEAK_ACCEPTED) continue;
        const ulong key=fg_peak_key(scores[p],p);
        if(key<=last) keys[atomic_fetch_add_explicit(&counters[3],1,memory_order_relaxed)]=key;
    }
    threadgroup_barrier(mem_flags::mem_device);
    // An output's rank among the outputs is its baseline iteration.
    for(uint e=lane;e<outputs;e+=lanes) {
        const ulong key=keys[e];uint rank=0;
        for(uint f=0;f<outputs;++f) rank+=uint(keys[f]<key);
        const uint index=uint(key),x=index%width,y=index/width;const float score=scores[index];
        const float cx=float(c.sw-1)*.5f,cy=float(c.sh-1)*.5f;
        const float rx=float(x)-c.translationX-cx,ry=float(y)-c.translationY-cy;
        candidates[c.first+rank]={cx+c.cosine*rx-c.sine*ry,cy+c.sine*rx+c.cosine*ry,
            c.angle,score,c.scale,score,1,c.first+rank,c.model,0,0,0};
    }
}
