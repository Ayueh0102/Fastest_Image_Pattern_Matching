// SPDX-License-Identifier: BSD-2-Clause
#include <metal_stdlib>
using namespace metal;

struct CoarseConfig {
    uint sw, tw, th, output_width;
    ulong first_window;
    uint count, reserved;
};
// Scalar fields deliberately avoid uint3's 16-byte alignment: host uses 12B.
struct Coarse32 { uint sum, square, product; };
struct Coarse64 { ulong sum, square, product; };

struct PyramidConfig { uint width, height, output_width, output_height; };
constant uint pyramid_weights[5] = {1, 4, 6, 4, 1};

// Signed 64-bit coordinates also avoid overflow in the reflection expression
// for long, narrow images. The width/height are positive by the host contract.
inline uint pyramid_reflect101(long p, uint length)
{
    if (length == 1) return 0;
    while (p < 0 || p >= long(length))
        p = p < 0 ? -p : 2 * long(length) - p - 2;
    return uint(p);
}

// Exact equivalent of the portable separable filter: all 25 integer products
// are summed before the ONE rounding operation. Maximum sum is 255*256=65280.
kernel void pyramid_direct(device const uchar* source [[buffer(0)]],
    device uchar* output [[buffer(1)]], constant PyramidConfig& cfg [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cfg.output_width || gid.y >= cfg.output_height) return;
    uint sum = 0;
    for (int ky = -2; ky <= 2; ++ky) {
        const uint sy = pyramid_reflect101(2 * long(gid.y) + ky, cfg.height);
        for (int kx = -2; kx <= 2; ++kx) {
            const uint sx = pyramid_reflect101(2 * long(gid.x) + kx, cfg.width);
            sum += pyramid_weights[ky + 2] * pyramid_weights[kx + 2] *
                uint(source[ulong(sy) * cfg.width + sx]);
        }
    }
    output[ulong(gid.y) * cfg.output_width + gid.x] = uchar((sum + 128) >> 8);
}

kernel void pyramid_horizontal(device const uchar* source [[buffer(0)]],
    device ushort* horizontal [[buffer(1)]], constant PyramidConfig& cfg [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cfg.output_width || gid.y >= cfg.height) return;
    uint sum = 0;
    for (int k = -2; k <= 2; ++k) {
        const uint sx = pyramid_reflect101(2 * long(gid.x) + k, cfg.width);
        sum += pyramid_weights[k + 2] * uint(source[ulong(gid.y) * cfg.width + sx]);
    }
    // Horizontal maximum is 255*16=4080; do not round between the axes.
    horizontal[ulong(gid.y) * cfg.output_width + gid.x] = ushort(sum);
}

kernel void pyramid_vertical(device const ushort* horizontal [[buffer(0)]],
    device uchar* output [[buffer(1)]], constant PyramidConfig& cfg [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cfg.output_width || gid.y >= cfg.output_height) return;
    uint sum = 0;
    for (int k = -2; k <= 2; ++k) {
        const uint sy = pyramid_reflect101(2 * long(gid.y) + k, cfg.height);
        sum += pyramid_weights[k + 2] *
            uint(horizontal[ulong(sy) * cfg.output_width + gid.x]);
    }
    output[ulong(gid.y) * cfg.output_width + gid.x] = uchar((sum + 128) >> 8);
}

kernel void coarse_stats32(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse32* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.count) return;
    const ulong index = cfg.first_window + gid;
    const ulong y = index / cfg.output_width, x = index % cfg.output_width;
    uint s = 0, q = 0, p = 0;
    for (uint ty = 0; ty < cfg.th; ++ty)
        for (uint tx = 0; tx < cfg.tw; ++tx) {
            const uint v = source[(y + ty) * cfg.sw + x + tx];
            const uint t = templ[ulong(ty) * cfg.tw + tx];
            s += v; q += v * v; p += v * t;
        }
    output[gid] = {s, q, p};
}

kernel void coarse_stats64(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse64* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.count) return;
    const ulong index = cfg.first_window + gid;
    const ulong y = index / cfg.output_width, x = index % cfg.output_width;
    ulong s = 0, q = 0, p = 0;
    for (uint ty = 0; ty < cfg.th; ++ty)
        for (uint tx = 0; tx < cfg.tw; ++tx) {
            const uint v = source[(y + ty) * cfg.sw + x + tx];
            const uint t = templ[ulong(ty) * cfg.tw + tx];
            s += v; q += v * v; p += v * t;
        }
    output[gid] = {s, q, p};
}

// Four horizontally adjacent outputs per thread. The source window advances
// through a four-byte register strip: after the first pixel, only ONE new
// source byte is loaded per template column. Template values serve all four
// outputs. There is no explicit threadgroup staging or floating arithmetic.
template<typename Accum4, typename Stats>
inline void coarse_strip4(device const uchar* source, device const uchar* templ,
    device Stats* output, constant CoarseConfig& cfg, uint2 gid)
{
    const ulong x = ulong(gid.x) * 4;
    const ulong y = cfg.first_window / cfg.output_width + gid.y;
    const ulong end = cfg.first_window + cfg.count;
    if (x >= cfg.output_width || y > (end - 1) / cfg.output_width) return;
    Accum4 sum(0), square(0), product(0);
    for (uint ty = 0; ty < cfg.th; ++ty) {
        const ulong row = (y + ty) * cfg.sw;
        // Clamp only inactive right-edge lanes; valid NCC windows never need
        // padding. This also supports a source narrower than four pixels.
        uint4 values(source[row + x], source[row + min(x + 1, ulong(cfg.sw - 1))],
            source[row + min(x + 2, ulong(cfg.sw - 1))],
            source[row + min(x + 3, ulong(cfg.sw - 1))]);
        for (uint tx = 0; tx < cfg.tw; ++tx) {
            if (tx) values = uint4(values.y, values.z, values.w,
                source[row + min(x + tx + 3, ulong(cfg.sw - 1))]);
            const uint t = templ[ulong(ty) * cfg.tw + tx];
            sum += Accum4(values);
            square += Accum4(values * values);
            product += Accum4(values * t);
        }
    }
    for (uint lane = 0; lane < 4; ++lane) {
        const ulong index = y * cfg.output_width + x + lane;
        // Chunk boundaries can split a row or a strip. Only the exact original
        // [first_window, first_window + count) range may write this workspace.
        if (x + lane < cfg.output_width && index >= cfg.first_window && index < end)
            output[index - cfg.first_window] = {sum[lane], square[lane], product[lane]};
    }
}

kernel void coarse_strip4_32(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse32* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint2 gid [[thread_position_in_grid]])
{
    coarse_strip4<uint4, Coarse32>(source, templ, output, cfg, gid);
}
kernel void coarse_strip4_64(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse64* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint2 gid [[thread_position_in_grid]])
{
    coarse_strip4<ulong4, Coarse64>(source, templ, output, cfg, gid);
}

// A full 8x8 threadgroup owns an 8x8 output tile. For every <=16x16
// template tile, cooperatively load its <=23x23 source union and template.
// All 64 threads take both barriers, including inactive image/chunk edges.
template<typename Accum, typename Stats>
inline void coarse_staged8(device const uchar* source, device const uchar* templ,
    device Stats* output, constant CoarseConfig& cfg, uint2 group, uint tid,
    threadgroup uchar* sourceTile, threadgroup uchar* templateTile)
{
    const ulong baseX = ulong(group.x) * 8;
    const ulong baseY = cfg.first_window / cfg.output_width + ulong(group.y) * 8;
    const uint lx = tid % 8, ly = tid / 8;
    const ulong x = baseX + lx, y = baseY + ly;
    const ulong end = cfg.first_window + cfg.count;
    const ulong lastRow = (end - 1) / cfg.output_width;
    const ulong index = y * cfg.output_width + x;
    const bool active = x < cfg.output_width && y <= lastRow &&
        index >= cfg.first_window && index < end;
    Accum sum = 0, square = 0, product = 0;
    for (uint ty = 0; ty < cfg.th; ty += 16) {
        const uint kh = min(16u, cfg.th - ty);
        for (uint tx = 0; tx < cfg.tw; tx += 16) {
            const uint kw = min(16u, cfg.tw - tx), sw = kw + 7, sh = kh + 7;
            for (uint j = tid; j < sw * sh; j += 64) {
                const ulong sx = baseX + tx + j % sw, sy = baseY + ty + j / sw;
                // lastRow+th is at most the real source height. Zero-fill the
                // unused halo without reading outside the source or next row.
                sourceTile[j] = sx < cfg.sw && sy < lastRow + cfg.th ?
                    source[sy * cfg.sw + sx] : uchar(0);
            }
            for (uint j = tid; j < kw * kh; j += 64)
                templateTile[j] = templ[ulong(ty + j / kw) * cfg.tw + tx + j % kw];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (uint ky = 0; ky < kh; ++ky) for (uint kx = 0; kx < kw; ++kx) {
                    const uint v = sourceTile[(ly + ky) * sw + lx + kx];
                    const uint t = templateTile[ky * kw + kx];
                    sum += v; square += v * v; product += v * t;
                }
            }
            // Prevent the next tile overwriting bytes still read by peers.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) output[index - cfg.first_window] = {sum, square, product};
}

kernel void coarse_staged8_32(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse32* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uchar sourceTile[23 * 23], templateTile[16 * 16];
    coarse_staged8<uint, Coarse32>(source, templ, output, cfg, group, tid, sourceTile, templateTile);
}
kernel void coarse_staged8_64(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse64* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uchar sourceTile[23 * 23], templateTile[16 * 16];
    coarse_staged8<ulong, Coarse64>(source, templ, output, cfg, group, tid, sourceTile, templateTile);
}

struct FineWarpDescriptor {
    long anchor_x, anchor_y, cosine, sine;
    ulong rounding_guard;
    uint repair_all, reserved;
};
struct FineWarpConfig {
    uint source_width, source_height, roi_width, roi_height;
    uint roi_pixels, roi_stride, count, reserved;
};
struct FineStatsConfig {
    uint roi_width, template_width, template_height, output_width;
    uint roi_stride, windows_per_map, count, reserved;
};

inline ulong fine_pixel(device const uchar* source, long x, long y,
    constant FineWarpConfig& cfg)
{
    return x < 0 || y < 0 || x >= long(cfg.source_width) || y >= long(cfg.source_height) ?
        0 : ulong(source[ulong(y) * cfg.source_width + ulong(x)]);
}

// Fast path is exact integer Q28 sampling. The host supplies a conservative
// bound on its difference from legacy double sampling. Values near any half-
// integer threshold are flagged and repaired by the ORIGINAL double formula
// before the NCC dispatch. There is no unchecked float texture interpolation.
kernel void fine_guarded_warp(device const uchar* source [[buffer(0)]],
    device uchar* rois [[buffer(1)]], device atomic_uint* repair_bits [[buffer(2)]],
    device const FineWarpDescriptor* descriptors [[buffer(3)]],
    constant FineWarpConfig& cfg [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    const uint map = gid / cfg.roi_pixels, pixel_index = gid % cfg.roi_pixels;
    if (map >= cfg.count) return;
    const FineWarpDescriptor descriptor = descriptors[map];
    const ulong target = ulong(map) * cfg.roi_stride + pixel_index;
    if (descriptor.repair_all) {
        rois[target] = 0; // Host repairs all pixels without a dense atomic mask.
        return;
    }
    const long x = long(pixel_index % cfg.roi_width), y = long(pixel_index / cfg.roi_width);
    const long unit = long(1) << 28;
    const long sx = descriptor.anchor_x + descriptor.cosine * x - descriptor.sine * y;
    const long sy = descriptor.anchor_y + descriptor.sine * x + descriptor.cosine * y;
    ulong numerator = 0;
    if (sx >= -unit && sy >= -unit && sx < long(cfg.source_width) * unit &&
        sy < long(cfg.source_height) * unit) {
        // Signed division truncates towards zero; correct it to floor for the
        // one negative border cell. Fractional coordinates stay in [0,2^28).
        const long ix = sx < 0 ? (sx - unit + 1) / unit : sx / unit;
        const long iy = sy < 0 ? (sy - unit + 1) / unit : sy / unit;
        const ulong dx = ulong(sx - ix * unit), dy = ulong(sy - iy * unit);
        const ulong one = ulong(unit);
        const ulong top = fine_pixel(source, ix, iy, cfg) * (one - dx) +
            fine_pixel(source, ix + 1, iy, cfg) * dx;
        const ulong bottom = fine_pixel(source, ix, iy + 1, cfg) * (one - dx) +
            fine_pixel(source, ix + 1, iy + 1, cfg) * dx;
        // Nonnegative terms sum to at most 255*2^56, safely below ULONG_MAX.
        numerator = top * (one - dy) + bottom * dy;
    }
    const ulong halfUnit = ulong(1) << 55;
    const ulong remainder = numerator & ((ulong(1) << 56) - 1);
    const ulong distance = remainder > halfUnit ? remainder - halfUnit : halfUnit - remainder;
    rois[target] = uchar((numerator + halfUnit) >> 56);
    if (distance <= descriptor.rounding_guard)
        atomic_fetch_or_explicit(repair_bits + (gid >> 5), uint(1) << (gid & 31), memory_order_relaxed);
}

kernel void fine_stats32(device const uchar* rois [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse32* output [[buffer(2)]],
    constant FineStatsConfig& cfg [[buffer(3)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]])
{
    // Exactly one complete SIMD group owns each NCC window. Integer addition
    // is associative here: the host selects this path only when the COMPLETE
    // statistic fits uint32, so every nonnegative partial sum also fits.
    const uint index = gid / simdWidth;
    const uint map = index / cfg.windows_per_map, window = index % cfg.windows_per_map;
    if (map >= cfg.count) return;
    const ulong y = window / cfg.output_width, x = window % cfg.output_width;
    const ulong base = ulong(map) * cfg.roi_stride;
    const uint area = cfg.template_width * cfg.template_height;
    uint sum = 0, square = 0, product = 0;
    for (uint pixel = lane; pixel < area; pixel += simdWidth) {
        const uint ty = pixel / cfg.template_width, tx = pixel % cfg.template_width;
        const uint value = rois[base + (y + ty) * cfg.roi_width + x + tx];
        const uint t = templ[pixel];
        sum += value; square += value * value; product += value * t;
    }
    sum = simd_sum(sum); square = simd_sum(square); product = simd_sum(product);
    if (lane == 0) output[index] = {sum, square, product};
}

// Exact uint64 reduction using only widely supported uint32 SIMD reductions.
// Low/middle 16-bit limb reductions fit uint32 for SIMD widths <= 1024 (host
// checked). The sum of upper limbs cannot exceed the upper 32 bits of the
// complete statistic, which the host bounds below 2^53. Recombining in ulong
// preserves carries without floating point or a threadgroup memory barrier.
inline ulong fine_simd_sum64(ulong value)
{
    const uint low = simd_sum(uint(value & 65535));
    const uint middle = simd_sum(uint((value >> 16) & 65535));
    const uint high = simd_sum(uint(value >> 32));
    return ulong(low) + (ulong(middle) << 16) + (ulong(high) << 32);
}

kernel void fine_stats64(device const uchar* rois [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse64* output [[buffer(2)]],
    constant FineStatsConfig& cfg [[buffer(3)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]])
{
    const uint index = gid / simdWidth;
    const uint map = index / cfg.windows_per_map, window = index % cfg.windows_per_map;
    if (map >= cfg.count) return;
    const ulong y = window / cfg.output_width, x = window % cfg.output_width;
    const ulong base = ulong(map) * cfg.roi_stride;
    const uint area = cfg.template_width * cfg.template_height;
    ulong sum = 0, square = 0, product = 0;
    for (uint pixel = lane; pixel < area; pixel += simdWidth) {
        const uint ty = pixel / cfg.template_width, tx = pixel % cfg.template_width;
        const uint value = rois[base + (y + ty) * cfg.roi_width + x + tx];
        const uint t = templ[pixel];
        sum += value; square += value * value; product += value * t;
    }
    sum = fine_simd_sum64(sum); square = fine_simd_sum64(square); product = fine_simd_sum64(product);
    if (lane == 0) output[index] = {sum, square, product};
}

// Optional coarse path: a complete SIMD group cooperates on one NCC window.
// The host still uses the identical bounded statistics workspace, chunk/window
// ordering and binary64 normalization. All terms are nonnegative and the host
// selects uint32 only when 65025 * template_area <= UINT32_MAX, making both
// partial sums and the SIMD reductions exact.
kernel void coarse_parallel_stats32(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse32* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]])
{
    const uint window = gid / simdWidth;
    if (window >= cfg.count) return; // Uniform within a complete SIMD group.
    const ulong index = cfg.first_window + window;
    const ulong y = index / cfg.output_width, x = index % cfg.output_width;
    const uint area = cfg.tw * cfg.th;
    uint sum = 0, square = 0, product = 0;
    for (uint pixel = lane; pixel < area; pixel += simdWidth) {
        const uint ty = pixel / cfg.tw, tx = pixel % cfg.tw;
        const uint value = source[(y + ty) * cfg.sw + x + tx];
        const uint t = templ[pixel];
        sum += value; square += value * value; product += value * t;
    }
    sum = simd_sum(sum); square = simd_sum(square); product = simd_sum(product);
    if (lane == 0) output[window] = {sum, square, product};
}

kernel void coarse_parallel_stats64(device const uchar* source [[buffer(0)]],
    device const uchar* templ [[buffer(1)]], device Coarse64* output [[buffer(2)]],
    constant CoarseConfig& cfg [[buffer(3)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]])
{
    const uint window = gid / simdWidth;
    if (window >= cfg.count) return;
    const ulong index = cfg.first_window + window;
    const ulong y = index / cfg.output_width, x = index % cfg.output_width;
    // Keep large-template pixel/index arithmetic in 64 bits. Complete sums
    // are <= 2^53 by the same host guard as the original scalar uint64 path.
    const ulong area = ulong(cfg.tw) * cfg.th;
    ulong sum = 0, square = 0, product = 0;
    for (ulong pixel = lane; pixel < area; pixel += simdWidth) {
        const ulong ty = pixel / cfg.tw, tx = pixel % cfg.tw;
        const uint value = source[(y + ty) * cfg.sw + x + tx];
        const uint t = templ[pixel];
        sum += value; square += value * value; product += value * t;
    }
    // Reuse the proven 16/16/32-bit limb reduction from fine search; no
    // floating-point conversion or threadgroup memory synchronization.
    sum = fine_simd_sum64(sum); square = fine_simd_sum64(square); product = fine_simd_sum64(product);
    if (lane == 0) output[window] = {sum, square, product};
}
