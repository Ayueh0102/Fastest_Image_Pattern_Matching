// SPDX-License-Identifier: BSD-2-Clause
// Appended to coarse_ncc.metal. Research route: exact centered integer
// moments, FP32 final division, original greedy suppression (not top-K).
struct ResidentPeakConfig {
    uint width, height, count, limit;
    ulong area, template_sum, template_variance;
    uint first, windows;
    float threshold;
    uint degenerate;
};
struct ResidentPeak { uint x, y; float score; uint reserved; };

template<typename Stats>
inline void resident_normalize(device const Stats* stats, device float* scores,
    constant ResidentPeakConfig& c, uint gid)
{
    if (gid >= c.count) return;
    float score = 1;
    if (!c.degenerate) {
        // Host proves 65025*area^2 <= UINT64_MAX before dispatch. Subtract
        // integer moments BEFORE converting to float, avoiding cancellation
        // on low-contrast images. This is not the legacy binary64 formula.
        const ulong s = stats[gid].sum, q = stats[gid].square, p = stats[gid].product;
        const ulong variance = c.area*q-s*s;
        const ulong a = c.area*p, b = s*c.template_sum;
        const float numerator = a >= b ? float(a-b) : -float(b-a);
        const float v = float(variance)/float(c.area);
        const float denominator = v > min(0.5f, 10*FLT_EPSILON*float(q)) ?
            sqrt(float(variance))*sqrt(float(c.template_variance)) : 0;
        score = 0;
        if (denominator != 0) {
            if (abs(numerator) < denominator) score = numerator/denominator;
            else if (abs(numerator) < denominator*1.125f) score = numerator > 0 ? 1 : -1;
        }
    }
    scores[c.first+gid] = score;
}
kernel void resident_normalize32(device const Coarse32* stats [[buffer(0)]],
    device float* scores [[buffer(1)]], constant ResidentPeakConfig& c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) { resident_normalize(stats, scores, c, gid); }
kernel void resident_normalize64(device const Coarse64* stats [[buffer(0)]],
    device float* scores [[buffer(1)]], constant ResidentPeakConfig& c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) { resident_normalize(stats, scores, c, gid); }

inline bool resident_better(float a, uint ai, float b, uint bi) {
    return a > b || (a == b && ai < bi);
}
kernel void resident_greedy_peaks(device float* scores [[buffer(0)]],
    device const int* bounds [[buffer(1)]], device ResidentPeak* output [[buffer(2)]],
    device uint* output_count [[buffer(3)]], constant ResidentPeakConfig& c [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]])
{
    // One workgroup owns a map. No unsupported cross-workgroup barrier or
    // CPU-mediated loop. This intentionally retains the sequential greedy
    // decisions; limited occupancy is an explicit experimental tradeoff.
    threadgroup float values[256]; threadgroup uint indices[256];
    const uint stripe = (c.windows+255)/256;
    const uint begin = min(c.windows, lane*stripe), end = min(c.windows, begin+stripe);
    float best = -INFINITY; uint best_index = UINT_MAX;
    for (uint j = begin; j < end; ++j)
        if (resident_better(scores[j], j, best, best_index)) { best = scores[j]; best_index = j; }
    if (lane == 0) *output_count = 0;
    for (uint iteration = 0; iteration < c.limit; ++iteration) {
        values[lane] = best; indices[lane] = best_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 128; step; step /= 2) {
            if (lane < step && resident_better(values[lane+step], indices[lane+step], values[lane], indices[lane])) {
                values[lane] = values[lane+step]; indices[lane] = indices[lane+step];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float score = values[0]; const uint index = indices[0];
        if (!isfinite(score) || score < c.threshold) break;
        const uint x = index % c.width, y = index / c.width;
        if (lane == 0) { output[iteration] = {x, y, score, 0}; *output_count = iteration+1; }
        if (iteration+1 == c.limit) break;
        // Host-generated endpoint tables preserve binary64/truncation rules
        // even for overlap=1 and values adjacent to integer boundaries.
        const uint x0 = uint(bounds[2*x]), x1 = uint(bounds[2*x+1]);
        const uint y0 = uint(bounds[2*c.width+2*y]), y1 = uint(bounds[2*c.width+2*y+1]);
        const uint rw = x1-x0, pixels = rw*(y1-y0);
        for (uint j = lane; j < pixels; j += 256)
            scores[(y0+j/rw)*c.width+x0+j%rw] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const uint bx = best_index % c.width, by = best_index / c.width;
        if (bx >= x0 && bx < x1 && by >= y0 && by < y1) {
            best = -INFINITY; best_index = UINT_MAX;
            for (uint j = begin; j < end; ++j)
                if (resident_better(scores[j], j, best, best_index)) { best = scores[j]; best_index = j; }
        }
        // Also prevents a fast lane overwriting values[0] before its peers
        // finish consuming this iteration's winning index and score.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
