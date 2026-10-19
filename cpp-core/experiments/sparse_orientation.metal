// SPDX-License-Identifier: BSD-2-Clause
#include <metal_stdlib>
using namespace metal;

struct SparseImageConfig { uint width, height, radius, reserved; };
struct SparseFeature { int offset; uint bin; };
struct SparseScoreConfig {
    uint sourceWidth, gridWidth, gridHeight, featureCount;
    uint requiredScore, scoreOffset, groupOffset, featureOffset;
    uint xOffset, yOffset, peakOffset, peakCapacity;
    uint counterOffset, groupThreads, reserved0, reserved1;
};
struct SparseGroupStats { uint evaluations, survivors; };
struct SparsePeak { uint xIndex, yIndex, score; };

kernel void sparse_gradient(device const uchar* source [[buffer(0)]],
    device const uchar* lookup [[buffer(1)]], device uchar* mask [[buffer(2)]],
    constant SparseImageConfig& c [[buffer(3)]], uint i [[thread_position_in_grid]])
{
    if (i >= c.width*c.height) return;
    const uint x=i%c.width, y=i/c.width;
    uchar value=0;
    if (x>0 && x+1<c.width && y>0 && y+1<c.height) {
        const int dx=int(source[i+1])-int(source[i-1]);
        const int dy=int(source[i+c.width])-int(source[i-c.width]);
        value=lookup[(dy+255)*511+(dx+255)];
    }
    mask[i]=value;
}
kernel void sparse_spread_horizontal(device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]], constant SparseImageConfig& c [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i>=c.width*c.height) return;
    const uint x=i%c.width;
    const int first=-min(int(c.radius),int(x));
    const int last=min(int(c.radius),int(c.width-1-x));
    uchar bits=0;
    for (int d=first;d<=last;++d) bits|=input[int(i)+d];
    output[i]=bits;
}
kernel void sparse_spread_vertical(device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]], constant SparseImageConfig& c [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i>=c.width*c.height) return;
    const uint y=i/c.width;
    const int first=-min(int(c.radius),int(y));
    const int last=min(int(c.radius),int(c.height-1-y));
    uchar bits=0;
    for (int d=first;d<=last;++d) bits|=input[int(i)+d*int(c.width)];
    output[i]=bits;
}

kernel void sparse_score(device const uchar* spread [[buffer(0)]],
    device const uchar* response [[buffer(1)]], device const int* xs [[buffer(2)]],
    device const int* ys [[buffer(3)]], device const SparseFeature* features [[buffer(4)]],
    device uint* scores [[buffer(5)]], device SparseGroupStats* statistics [[buffer(6)]],
    constant SparseScoreConfig& c [[buffer(7)]], uint i [[thread_position_in_grid]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]])
{
    uint score=0, evaluated=0, survived=0;
    if (i<c.gridWidth*c.gridHeight) {
        const int origin=ys[c.yOffset+i/c.gridWidth]*int(c.sourceWidth)+xs[c.xOffset+i%c.gridWidth];
        uint j=0;
        for (;j<c.featureCount;++j) {
            const SparseFeature f=features[c.featureOffset+j];
            score+=uint(response[f.bin*256+spread[origin+f.offset]]);
            ++evaluated;
            // Same feature order and exact 16-feature bound as the CPU path.
            if ((j&15)==15 && score+(c.featureCount-j-1)*255<c.requiredScore) { ++j;break; }
        }
        survived=(j==c.featureCount && score>=c.requiredScore) ? 1u : 0u;
        scores[c.scoreOffset+i]=survived ? score : 0u;
    }
    // Group statistics avoid contended per-window atomics and 32-bit totals.
    // The CPU sums these small groups into uint64 counters after completion.
    threadgroup uint evaluationSums[256], survivorSums[256];
    const uint evaluations=simd_sum(evaluated), survivors=simd_sum(survived);
    if (lane==0) { evaluationSums[simd]=evaluations;survivorSums[simd]=survivors; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local==0) {
        SparseGroupStats total={0,0};
        for (uint k=0;k<(c.groupThreads+simdWidth-1)/simdWidth;++k) {
            total.evaluations+=evaluationSums[k];total.survivors+=survivorSums[k];
        }
        statistics[c.groupOffset+group]=total;
    }
}

kernel void sparse_peaks(device const uint* scores [[buffer(0)]],
    device SparsePeak* peaks [[buffer(1)]], device atomic_uint* counters [[buffer(2)]],
    constant SparseScoreConfig& c [[buffer(3)]], uint i [[thread_position_in_grid]])
{
    if (i>=c.gridWidth*c.gridHeight) return;
    const uint score=scores[c.scoreOffset+i];
    if (!score) return;
    const int x=int(i%c.gridWidth), y=int(i/c.gridWidth);
    for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
        const int nx=x+dx, ny=y+dy;
        if (nx<0 || ny<0 || nx>=int(c.gridWidth) || ny>=int(c.gridHeight)) continue;
        const uint other=uint(ny)*c.gridWidth+uint(nx), neighbor=scores[c.scoreOffset+other];
        if (neighbor>score || (neighbor==score && other<i)) return;
    }
    // Count every peak even when output capacity is exhausted. The host never
    // consumes a truncated result: it reallocates and reruns this kernel.
    const uint destination=atomic_fetch_add_explicit(counters+c.counterOffset,1u,memory_order_relaxed);
    if (destination<c.peakCapacity)
        peaks[c.peakOffset+destination]={uint(x),uint(y),score};
}
