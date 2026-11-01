// Measures the m16n8k32 u8 fragment layout: one A or B byte at a time.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
__global__ void probe(const unsigned* in, unsigned* out)
{
    const int lane = threadIdx.x;
    unsigned a0 = in[lane * 6], a1 = in[lane * 6 + 1], a2 = in[lane * 6 + 2], a3 = in[lane * 6 + 3], b0 = in[lane * 6 + 4], b1 = in[lane * 6 + 5];
    unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#endif
    out[lane * 4] = c0; out[lane * 4 + 1] = c1; out[lane * 4 + 2] = c2; out[lane * 4 + 3] = c3;
}
struct Run {
    unsigned* dIn; unsigned* dOut;
    std::vector<unsigned> run(const std::vector<unsigned>& input) {
        cudaMemcpy(dIn, input.data(), 32 * 6 * 4, cudaMemcpyHostToDevice);
        probe<<<1, 32>>>(dIn, dOut);
        std::vector<unsigned> output(32 * 4);
        cudaMemcpy(output.data(), dOut, 32 * 4 * 4, cudaMemcpyDeviceToHost);
        return output;
    }
};
int main()
{
    Run r{}; cudaMalloc(&r.dIn, 32 * 6 * 4); cudaMalloc(&r.dOut, 32 * 4 * 4);
    std::vector<unsigned> ones(32 * 6, 0x01010101u), base;
    auto print = [](const char* what, int lane, int reg, int byte, const std::vector<unsigned>& d) {
        std::printf("%s %d %d %d :", what, lane, reg, byte);
        for (int i = 0; i < 128; ++i) if (d[i]) std::printf(" %d,%d", i / 4, i % 4);
        std::printf("\n");
    };
    for (int lane = 0; lane < 32; ++lane) for (int reg = 0; reg < 4; ++reg) for (int byte = 0; byte < 4; ++byte) {
        base = ones;
        for (int l = 0; l < 32; ++l) for (int k = 0; k < 4; ++k) base[l * 6 + k] = 0;
        base[lane * 6 + reg] = 1u << (8 * byte);
        print("A", lane, reg, byte, r.run(base));
    }
    for (int lane = 0; lane < 32; ++lane) for (int reg = 0; reg < 2; ++reg) for (int byte = 0; byte < 4; ++byte) {
        base = ones;
        for (int l = 0; l < 32; ++l) { base[l * 6 + 4] = 0; base[l * 6 + 5] = 0; }
        base[lane * 6 + 4 + reg] = 1u << (8 * byte);
        print("B", lane, reg, byte, r.run(base));
    }
    // Which B bytes multiply which A bytes: one A byte against one B byte.
    for (int la = 0; la < 32; ++la) for (int ra = 0; ra < 4; ++ra) for (int ba = 0; ba < 4; ++ba) {
        if (la % 4 != 0 || ra % 2 != 0) continue; // every k appears in lane%4 / reg parity scan below
        for (int lb = 0; lb < 8; ++lb) for (int rb = 0; rb < 2; ++rb) for (int bb = 0; bb < 4; ++bb) {
            std::vector<unsigned> in(32 * 6, 0);
            in[la * 6 + ra] = 1u << (8 * ba); in[lb * 6 + 4 + rb] = 1u << (8 * bb);
            const auto d = r.run(in);
            for (int i = 0; i < 128; ++i) if (d[i]) std::printf("K %d %d %d  %d %d %d : %d,%d\n", la, ra, ba, lb, rb, bb, i / 4, i % 4);
        }
    }
    return 0;
}
