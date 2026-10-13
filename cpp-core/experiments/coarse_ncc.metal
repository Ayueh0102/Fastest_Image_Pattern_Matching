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
