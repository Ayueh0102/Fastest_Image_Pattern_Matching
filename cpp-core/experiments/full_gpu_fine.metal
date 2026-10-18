// SPDX-License-Identifier: BSD-2-Clause
// GPU-resident scalar-matcher control flow. Included after full_gpu_types.h
// and the shared fg_angle / fg_sample / fg_ncc / fg_allowed helpers.
// A workgroup owns one candidate throughout every layer and subpixel pass.
// FP32 coordinates/scores replace the reference CPU FP64 arithmetic; integer
// NCC moments are exact within the host-validated uint64 area bound.
constant uint FG_FINE_ANGLES = 35;
constant uint FG_FINE_WINDOWS = 49;
// Shader validation instruments threadgroup accesses and can double static
// storage. The host explicitly chooses 4096 in that mode to stay below the
// 32 KiB device limit; ordinary execution keeps its 16 KiB cache. Uncached
// pixels use exactly the same GPU sampler, with no arithmetic/CPU fallback.
#ifndef FG_FINE_ROI_CAPACITY
#define FG_FINE_ROI_CAPACITY 16384
#endif
constant uint FG_FINE_ROI_BYTES = FG_FINE_ROI_CAPACITY;
constant float FG_FINE_D2R = 0.01745329251994329577f;
constant float FG_FINE_ANGLE_EPS = 0.00004f;

struct FGFineState {
    float2 point, anchor;
    float angle, step;
    uint angleCount, nextCount, best;
    uint active, changed, maps;
    float angles[35], nextAngles[35];
    uint peakIndex[35];
    float peakScore[35];
};

inline void fg_fine_append(threadgroup float* angles, thread uint& count, float value)
{
    value = fg_angle(value);
    for (uint i = 0; i < count; ++i)
        if (fabs(fg_angle(angles[i] - value)) < 1.0e-9f) return;
    // The host enforces <=16 ranges: at most 3 + 2*16 distinct values.
    angles[count++] = value;
}

inline uint fg_fine_angles(float center, float step, device const FGRange* ranges,
    uint rangeCount, threadgroup float* angles)
{
    uint count = 0;
    for (int k = -1; k <= 1; ++k) {
        const float value = center + float(k) * step;
        if (fg_allowed(fg_angle(-value), ranges, rangeCount))
            fg_fine_append(angles, count, value);
    }
    for (uint r = 0; r < rangeCount; ++r) {
        const float values[2] = {ranges[r].minimum, ranges[r].maximum};
        for (uint e = 0; e < 2; ++e)
            if (fabs(fg_angle(-values[e] - center)) <= step + FG_FINE_ANGLE_EPS)
                fg_fine_append(angles, count, -values[e]);
    }
    // Stable insertion order matches the reference order before sorting.
    for (uint i = 1; i < count; ++i) {
        float value = angles[i];
        uint j = i;
        while (j && fg_angle(value - center) < fg_angle(angles[j - 1] - center)) {
            angles[j] = angles[j - 1]; --j;
        }
        angles[j] = value;
    }
    return count;
}

inline void fg_fine_replace_angles(threadgroup FGFineState& state)
{
    state.angleCount = state.nextCount;
    for (uint i = 0; i < state.angleCount; ++i) state.angles[i] = state.nextAngles[i];
}

inline uchar fg_fine_pixel(device const uchar* source, FGLevel level,
    float2 anchor, float c, float s, uint x, uint y)
{
    const float px = float(x) - 3.0f, py = float(y) - 3.0f;
    return fg_sample(source + level.sourceOffset, level.sw, level.sh,
        anchor.x + c * px - s * py, anchor.y + s * px + c * py, 0);
}

// 0: unchanged full-ROI cache, 1: unchanged direct sampling, 2: row tiles.
inline uint fg_fine_path(FGLevel level, uint mode)
{
    const uint roiW = level.tw + 6, roiH = level.th + 6;
    const ulong roiArea = ulong(roiW) * roiH;
    if (roiArea <= FG_FINE_ROI_BYTES) return 0;
    return mode == 1 && roiW <= FG_FINE_ROI_BYTES / 7 ? 2 : 1;
}

inline void fg_fine_angle_moments(device const uchar* source, device const uchar* templ,
    FGLevel level, float2 anchor, float c, float s, threadgroup uchar* roi,
    threadgroup ulong* partial, uint lane, uint path)
{
    const uint roiW = level.tw + 6, roiH = level.th + 6;
    const ulong roiArea = ulong(roiW) * roiH;
    const uint window = lane / 4, strip = lane % 4;
    ulong sum = 0, square = 0, dot = 0;
    if (path == 2) {
        // Each of the 49 windows sees template rows [row0,row0+rows). The
        // additional six rows are the union of all seven vertical offsets.
        // The cache is bounded regardless of template height. A width that
        // cannot fit one row plus its halo follows the original direct path.
        const uint tileRows = min(16u, FG_FINE_ROI_BYTES / roiW - 6);
        for (uint row0 = 0; row0 < level.th; row0 += tileRows) {
            const uint rows = min(tileRows, level.th - row0);
            const uint tileArea = roiW * (rows + 6);
            for (uint p = lane; p < tileArea; p += 256)
                roi[p] = fg_fine_pixel(source, level, anchor, c, s,
                    p % roiW, row0 + p / roiW);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (window < FG_FINE_WINDOWS) {
                const uint wx = window % 7, wy = window / 7;
                const uint begin = row0 * level.tw, end = (row0 + rows) * level.tw;
                // Preserve the original global p % 4 lane assignment, even
                // when an odd template width crosses a tile boundary.
                const uint first = begin + ((strip - (begin & 3u)) & 3u);
                for (uint p = first; p < end; p += 4) {
                    const uint x = p % level.tw, y = p / level.tw - row0;
                    const ulong pixel = roi[(wy + y) * roiW + wx + x];
                    const ulong t = templ[level.templateOffset + p];
                    sum += pixel; square += pixel * pixel; dot += pixel * t;
                }
            }
            // Every SIMD group finishes the tile before any group reuses it.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const bool cached = path == 0;
        if (cached) {
            for (uint p = lane; p < uint(roiArea); p += 256)
                roi[p] = fg_fine_pixel(source, level, anchor, c, s, p % roiW, p / roiW);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (window < FG_FINE_WINDOWS) {
            const uint wx = window % 7, wy = window / 7;
            const uint area = level.tw * level.th;
            for (uint p = strip; p < area; p += 4) {
                const uint x = p % level.tw, y = p / level.tw;
                const ulong pixel = cached ? roi[(wy + y) * roiW + wx + x] :
                    fg_fine_pixel(source, level, anchor, c, s, wx + x, wy + y);
                const ulong t = templ[level.templateOffset + p];
                sum += pixel; square += pixel * pixel; dot += pixel * t;
            }
        }
    }
    if (window < FG_FINE_WINDOWS) {
        partial[lane * 3] = sum;
        partial[lane * 3 + 1] = square;
        partial[lane * 3 + 2] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void fg_fine_evaluate(device const uchar* source, device const uchar* templ,
    FGLevel level, FGStats stats, threadgroup FGFineState& state,
    threadgroup uchar* roi, threadgroup ulong* partial, threadgroup float* maps,
    uint lane, constant FGFineOptions& options, device FGFineCounters& counters)
{
    const uint path = fg_fine_path(level, options.mode);
    if (lane == 0) state.best = 0xffffffffu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint a = 0; a < state.angleCount; ++a) {
        const float radians = state.angles[a] * FG_FINE_D2R;
        const float c = cos(radians), s = sin(radians);
        if (lane == 0 && options.diagnosticsEnabled) {
            if (path == 0) atomic_fetch_add_explicit(&counters.cachedAngles, 1u, memory_order_relaxed);
            else if (path == 1) atomic_fetch_add_explicit(&counters.directAngles, 1u, memory_order_relaxed);
            else atomic_fetch_add_explicit(&counters.tiledAngles, 1u, memory_order_relaxed);
        }
        fg_fine_angle_moments(source, templ, level, state.anchor, c, s, roi, partial, lane, path);
        if (lane < FG_FINE_WINDOWS) {
            ulong sum = 0, square = 0, dot = 0;
            for (uint k = 0; k < 4; ++k) {
                const uint p = (lane * 4 + k) * 3;
                sum += partial[p]; square += partial[p + 1]; dot += partial[p + 2];
            }
            maps[a * FG_FINE_WINDOWS + lane] = fg_ncc(sum, square, dot, stats);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            uint peak = 0;
            for (uint p = 1; p < FG_FINE_WINDOWS; ++p)
                if (maps[a * FG_FINE_WINDOWS + p] > maps[a * FG_FINE_WINDOWS + peak]) peak = p;
            state.peakIndex[a] = peak;
            state.peakScore[a] = maps[a * FG_FINE_WINDOWS + peak];
            if (state.best == 0xffffffffu || state.peakScore[a] > state.peakScore[state.best]) state.best = a;
            ++state.maps;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Complete the final shared loop-bound read on every SIMD group before
    // lane zero can return and replace angleCount/angles for the next search.
    // The last in-loop barrier alone does not protect that following read.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// On the 3^d Cartesian grid, {1, x_i, x_i*x_j, x_i^2-2/3}
// are orthogonal. Their squared norms are 3^d, 2*3^(d-1),
// 4*3^(d-2), and (2/3)*3^(d-1), respectively. These moment
// formulas equal the fixed-design least-squares fit in subpixel.cpp, without
// constructing its 27x10 matrix or doing QR for each candidate.
inline bool fg_fine_fit(threadgroup const float* maps, uint best, uint peak,
    uint dimension, thread float3& offset)
{
    float h[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    float g[3] = {0, 0, 0};
    const int planes = dimension == 3 ? 1 : 0;
    const float gradientDenominator = dimension == 3 ? 18.0f : 6.0f;
    const float diagonalDenominator = dimension == 3 ? 3.0f : 1.0f;
    const float crossDenominator = dimension == 3 ? 12.0f : 4.0f;
    // Subtract the center value to remove the large DC term before computing
    // curvature. Constant subtraction cannot change fitted gradient/Hessian.
    const float dc = maps[best * FG_FINE_WINDOWS + peak];
    for (int z = -planes; z <= planes; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) {
                const float value = maps[uint(int(best) + z) * FG_FINE_WINDOWS + uint(int(peak) + 7 * y + x)] - dc;
                if (!isfinite(value)) return false;
                const float v[3] = {float(x), float(y), float(z)};
                for (uint i = 0; i < dimension; ++i) {
                    g[i] += v[i] * value / gradientDenominator;
                    h[i][i] += (v[i] * v[i] - 2.0f / 3.0f) * value / diagonalDenominator;
                    for (uint j = 0; j < i; ++j) h[i][j] += v[i] * v[j] * value / crossDenominator;
                }
            }
    float lower[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (uint i = 0; i < dimension; ++i)
        for (uint j = 0; j <= i; ++j) {
            float value = -h[i][j];
            for (uint k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
            if (i == j) {
                if (!isfinite(value) || value <= 1.0e-10f) return false;
                lower[i][j] = sqrt(value);
            } else lower[i][j] = value / lower[j][j];
        }
    float intermediate[3] = {0, 0, 0}, result[3] = {0, 0, 0};
    for (uint i = 0; i < dimension; ++i) {
        float value = g[i];
        for (uint j = 0; j < i; ++j) value -= lower[i][j] * intermediate[j];
        intermediate[i] = value / lower[i][i];
    }
    for (int i = int(dimension) - 1; i >= 0; --i) {
        float value = intermediate[i];
        for (uint j = uint(i) + 1; j < dimension; ++j) value -= lower[j][i] * result[j];
        result[i] = value / lower[i][i];
        if (!isfinite(result[i]) || fabs(result[i]) > 1.0f) return false;
    }
    offset = float3(result[0], result[1], result[2]);
    return true;
}

inline void fg_fine_output(device FGCandidate& candidate, float2 point,
    float angle, float score, constant FGFineConfig& config)
{
    const float c = cos(angle * FG_FINE_D2R), s = sin(angle * FG_FINE_D2R);
    const float w = float(config.templateWidth), h = float(config.templateHeight);
    candidate.x = point.x + 0.5f * (w * c - h * s);
    candidate.y = point.y + 0.5f * (w * s + h * c);
    candidate.angle = fg_angle(-angle);
    candidate.score = score;
    candidate.scale = config.scale;
    candidate.model = config.model;
    candidate.valid = 1;
}

kernel void fg_fine_search(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device const FGLevel* levels [[buffer(2)]],
    device const FGStats* statistics [[buffer(3)]], device FGCandidate* candidates [[buffer(4)]],
    constant FGFineConfig& config [[buffer(5)]], device const FGRange* ranges [[buffer(6)]],
    constant FGFineOptions& options [[buffer(7)]], device FGFineCounters* counters [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]])
{
    if (group >= config.count) return;
    const uint index = config.first + group;
    const FGCandidate initial = candidates[index];
    if (!initial.valid) return;
    // Default: 16,384 + 4,704 + 6,860 + sizeof(FGFineState) = 28,560 B.
    // Validation cache4096: 16,272 B before instrumentation (32,540 B after).
    threadgroup uchar roi[FG_FINE_ROI_CAPACITY];
    threadgroup ulong partial[196 * 3];
    threadgroup float maps[35 * 49];
    threadgroup FGFineState state;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        state.point = float2(initial.x, initial.y);
        state.angle = initial.angle;
        state.maps = 0;
        state.active = 1;
        candidates[index].valid = 0;
        candidates[index].reserved0 = 0;
        candidates[index].reserved1 = 0;
        candidates[index].reserved2 = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (config.stop > 0 && config.top == config.stop) {
        if (lane == 0 && initial.score >= config.minScore)
            fg_fine_output(candidates[index], state.point * 2.0f, state.angle, initial.score, config);
        return;
    }
    for (int layer = max(0, int(config.top) - 1); layer >= int(config.stop); --layer) {
        const FGLevel level = levels[config.levelBase + uint(layer)];
        const FGStats stats = statistics[level.statisticsIndex];
        const uint passes = config.subpixel && layer == 0 ? 3 : 1;
        float threshold = config.minScore;
        for (int i = 0; i < layer; ++i) threshold *= 0.9f;
        if (lane == 0) {
            ++candidates[index].reserved1;
            if (options.diagnosticsEnabled)
                atomic_fetch_add_explicit(&counters[config.levelBase + uint(layer)].reachedCandidates,
                    1u, memory_order_relaxed);
        }
        for (uint pass = 0; pass < passes; ++pass) {
            if (lane == 0) {
                ++candidates[index].reserved2;
                const float scale = config.top == 0 || pass > 0 ? 1.0f : 2.0f;
                state.anchor = state.point * scale;
                state.step = atan(2.0f / float(max(level.tw, level.th))) / FG_FINE_D2R;
                state.angleCount = fg_fine_angles(state.angle, state.step, ranges, config.rangeCount, state.angles);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fg_fine_evaluate(source, templ, level, stats, state, roi, partial, maps, lane,
                options, counters[options.diagnosticsEnabled ? config.levelBase + uint(layer) : 0]);
            if (config.subpixel && layer == 0) {
                for (uint resolution = 0; resolution < 3; ++resolution) {
                    for (uint attempt = 0; attempt < 3; ++attempt) {
                        if (lane == 0) {
                            state.changed = 0;
                            if (state.best != 0xffffffffu && state.angleCount > 1 &&
                                !(state.best > 0 && state.best + 1 < state.angleCount)) {
                                state.nextCount = fg_fine_angles(state.angles[state.best], state.step, ranges,
                                    config.rangeCount, state.nextAngles);
                                state.changed = state.nextCount != state.angleCount;
                                for (uint i = 0; i < min(state.nextCount, state.angleCount); ++i)
                                    if (state.nextAngles[i] != state.angles[i]) state.changed = 1;
                                if (state.changed) fg_fine_replace_angles(state);
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (!state.changed) break;
                        fg_fine_evaluate(source, templ, level, stats, state, roi, partial, maps, lane,
                            options, counters[options.diagnosticsEnabled ? config.levelBase + uint(layer) : 0]);
                    }
                    // Latch the shared decision before lane zero replaces the
                    // angle list. A narrower resolution may reduce its count
                    // to one; no late SIMD group may see that new count here.
                    const bool endResolution = resolution == 2 || state.best == 0xffffffffu || state.angleCount < 2;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (endResolution) break;
                    if (lane == 0) {
                        state.step *= 0.5f;
                        state.nextCount = fg_fine_angles(state.angles[state.best], state.step,
                            ranges, config.rangeCount, state.nextAngles);
                        fg_fine_replace_angles(state);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    fg_fine_evaluate(source, templ, level, stats, state, roi, partial, maps, lane,
                        options, counters[options.diagnosticsEnabled ? config.levelBase + uint(layer) : 0]);
                }
            }
            if (lane == 0) {
                if (state.best == 0xffffffffu || state.peakScore[state.best] < threshold) state.active = 0;
                else {
                    const uint best = state.best, peak = state.peakIndex[best];
                    const float score = state.peakScore[best];
                    float2 location = float2(peak % 7, peak / 7);
                    state.angle = state.angles[best];
                    if (config.subpixel && layer == 0 && location.x > 0 && location.y > 0 &&
                        location.x < 6 && location.y < 6) {
                        bool fitted = false;
                        float3 offset;
                        if (best > 0 && best + 1 < state.angleCount &&
                            fabs(fg_angle(state.angle - state.angles[best - 1]) - state.step) <= FG_FINE_ANGLE_EPS &&
                            fabs(fg_angle(state.angles[best + 1] - state.angle) - state.step) <= FG_FINE_ANGLE_EPS &&
                            fg_fine_fit(maps, best, peak, 3, offset) &&
                            fg_allowed(fg_angle(-(state.angle + offset.z * state.step)), ranges, config.rangeCount)) {
                            location += offset.xy;
                            state.angle += offset.z * state.step;
                            fitted = true;
                        }
                        if (!fitted && fg_fine_fit(maps, best, peak, 2, offset)) location += offset.xy;
                    }
                    const float c = cos(state.angle * FG_FINE_D2R), s = sin(state.angle * FG_FINE_D2R);
                    const float2 delta = location - 3.0f;
                    state.point = state.anchor + float2(c * delta.x - s * delta.y, s * delta.x + c * delta.y);
                    if (pass + 1 == passes && layer == int(config.stop) && score >= config.minScore &&
                        fg_allowed(fg_angle(-state.angle), ranges, config.rangeCount))
                        fg_fine_output(candidates[index], state.point * (config.stop == 0 ? 1.0f : 2.0f),
                            state.angle, score, config);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!state.active) break;
        }
        if (!state.active) break;
    }
    if (lane == 0) candidates[index].reserved0 = state.maps;
}
