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
