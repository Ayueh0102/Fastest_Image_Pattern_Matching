// SPDX-License-Identifier: BSD-2-Clause
#include <metal_stdlib>
using namespace metal;
struct Config {
    uint sw, sh, tw, th, channels, tiles, count, unused;
    float4 sourceMean, templateSum;
    float templateEnergy;
};
// 64 bytes (float4 alignment). Count is needed for the final partial tile.
struct Partial { float4 mean, m2, product; uint count; };

inline float load_pixel(device const float* source, int x, int y, uint c, constant Config& k)
{
    if (x < 0 || y < 0 || x >= int(k.sw) || y >= int(k.sh)) return -k.sourceMean[c];
    return source[(uint(y)*k.sw+uint(x))*k.channels+c];
}
// One group per pose/tile, 256 threads, four pixels per thread. Fuse bilinear
// sampling with NCC statistics: no intermediate warped image or integral map.
kernel void pose_partials(device const float* source [[buffer(0)]],
    device const float* templ [[buffer(1)]], device const float4* poses [[buffer(2)]],
    device Partial* partials [[buffer(3)]], constant Config& k [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup float4 means[256], variances[256], products[256];
    threadgroup uint counts[256];
    float4 mean = 0, m2 = 0, product = 0;
    uint count = 0;
    const float4 p = poses[group.y];
    for (uint pixel = group.x*1024+lane; pixel < min((group.x+1)*1024, k.tw*k.th); pixel += 256) {
        const float u = float(pixel%k.tw)-float(k.tw)*.5f;
        const float v = float(k.th)*.5f-float(pixel/k.tw);
        const float sx = float(k.sw)*.5f+p.x+p.z*u-p.w*v;
        const float sy = float(k.sh)*.5f-p.y-(p.w*u+p.z*v);
        const int ix = int(floor(sx)), iy = int(floor(sy));
        const float dx = sx-float(ix), dy = sy-float(iy);
        float4 sample = 0;
        for (uint c = 0; c < k.channels; ++c) {
            const float a = load_pixel(source, ix, iy, c, k), b = load_pixel(source, ix+1, iy, c, k);
            const float e = load_pixel(source, ix, iy+1, c, k), f = load_pixel(source, ix+1, iy+1, c, k);
            const float value = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
            sample[c] = value; product[c] += value*templ[pixel*k.channels+c];
        }
        // Welford avoids Q-S^2/N cancellation for constant / low-contrast ROIs.
        ++count;
        const float4 delta = sample-mean;
        mean += delta/float(count);
        m2 += delta*(sample-mean);
    }
    means[lane] = mean; variances[lane] = m2; products[lane] = product; counts[lane] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step > 0; step >>= 1) {
        if (lane < step) {
            const uint a = counts[lane], b = counts[lane+step];
            const float4 delta = means[lane+step]-means[lane];
            if (a+b > 0) {
                means[lane] += delta*(float(b)/float(a+b));
                variances[lane] += variances[lane+step]+delta*delta*(float(a)*float(b)/float(a+b));
            }
            products[lane] += products[lane+step]; counts[lane] = a+b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partials[group.y*k.tiles+group.x] = {means[0], variances[0], products[0], counts[0]};
}
kernel void pose_finish(device const Partial* partials [[buffer(0)]],
    device float* scores [[buffer(1)]], constant Config& k [[buffer(2)]], uint pose [[thread_position_in_grid]])
{
    if (pose >= k.count) return;
    float4 mean = 0, m2 = 0, product = 0;
    uint count = 0;
    for (uint i = 0; i < k.tiles; ++i) {
        Partial p = partials[pose*k.tiles+i];
        const float4 delta = p.mean-mean;
        const uint total = count+p.count;
        mean += delta*(float(p.count)/float(total));
        m2 += p.m2+delta*delta*(float(count)*float(p.count)/float(total));
        count = total; product += p.product;
    }
    const float n = float(k.tw*k.th);
    const float4 e = max(float4(0), m2);
    const float energy = e.x+e.y+e.z;
    const float4 numerator = product-mean*k.templateSum;
    scores[pose] = energy/n <= 1e-12f || k.templateEnergy/n <= 1e-12f ? -2.f :
        clamp((numerator.x+numerator.y+numerator.z)/sqrt(energy*k.templateEnergy), -1.f, 1.f);
}
