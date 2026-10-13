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
