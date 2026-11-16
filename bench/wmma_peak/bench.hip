// Peak matrix throughput of the WMMA flavours gfx11 offers, register-resident (no memory traffic), so the
// number is the instruction rate and not a GEMM. Tells whether the int4 flavour is worth a W4A4 prefill.
// Build and run: bench/wmma_peak/run.sh [blocks]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e = (x);                                                \
        if (e != hipSuccess) {                                             \
            printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int8v __attribute__((ext_vector_type(8)));
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int2v __attribute__((ext_vector_type(2)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

constexpr int ITER = 2048;
constexpr int THREADS = 256;  // 8 waves of 32
constexpr int UNROLL = 4;     // independent accumulator chains, so the rate is not latency-bound
constexpr int BLOCKS = 320;   // 40 CU x 8 waves

// One 16x16x16 WMMA is 16*16*16 multiply-adds = 8192 operations.
constexpr double OPS_PER_ITER = (double)UNROLL * 8192.0;

#define LOOP(END)                                                              \
    for (int i = 0; i < ITER; ++i) {                                           \
        _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) { END; }            \
    }                                                                          \
    if (threadIdx.x == 0 && blockIdx.x == 0)                                   \
        out[0] = (float)(c[0][0] + c[1][1] + c[2][2] + c[3][3]);

extern "C" __global__ void __launch_bounds__(256) peak_f16(float* out, const _Float16* in) {
    half16 a = *(const half16*)(in + threadIdx.x * 16);
    half16 b = *(const half16*)(in + 8192 + threadIdx.x * 16);
    float8 c[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) c[u] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    LOOP(c[u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(u & 1 ? b : a, u & 1 ? a : b, c[u]))
}

extern "C" __global__ void __launch_bounds__(256) peak_bf16(float* out, const _Float16* in) {
    half16 a = *(const half16*)(in + threadIdx.x * 16);
    half16 b = *(const half16*)(in + 8192 + threadIdx.x * 16);
    float8 c[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) c[u] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    LOOP(c[u] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(u & 1 ? b : a, u & 1 ? a : b, c[u]))
}

extern "C" __global__ void __launch_bounds__(256) peak_i8(float* out, const _Float16* hin) {
    const int* in = (const int*)hin;
    int4v a = *(const int4v*)(in + threadIdx.x * 4);
    int4v b = *(const int4v*)(in + 4096 + threadIdx.x * 4);
    int8v c[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) c[u] = (int8v){0, 0, 0, 0, 0, 0, 0, 0};
    LOOP(c[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, u & 1 ? b : a, true, u & 1 ? a : b, c[u], false))
}

extern "C" __global__ void __launch_bounds__(256) peak_i4(float* out, const _Float16* hin) {
    const int* in = (const int*)hin;
    int2v a = *(const int2v*)(in + threadIdx.x * 2);
    int2v b = *(const int2v*)(in + 4096 + threadIdx.x * 2);
    int8v c[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) c[u] = (int8v){0, 0, 0, 0, 0, 0, 0, 0};
    LOOP(c[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, u & 1 ? b : a, true, u & 1 ? a : b, c[u], false))
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? atoi(argv[1]) : BLOCKS;
    float* out;
    void* in;
    CK(hipMalloc(&out, 1024));
    CK(hipMalloc(&in, 1 << 20));
    CK(hipMemset(in, 0x11, 1 << 20));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    const double ops = (double)blocks * (THREADS / 32) * ITER * OPS_PER_ITER;

    struct Res {
        const char* name;
        float ms;
    };
    auto run = [&](const char* name, auto fn) {
        const _Float16* arg = (const _Float16*)in;
        float best = 1e30f;
        for (int r = 0; r < 7; ++r) {
            CK(hipEventRecord(e0, 0));
            fn<<<blocks, THREADS>>>(out, arg);
            CK(hipEventRecord(e1, 0));
            CK(hipEventSynchronize(e1));
            float ms;
            CK(hipEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        return Res{name, best};
    };
    printf("blocks %d  threads %d  iter %d  unroll %d\n", blocks, THREADS, ITER, UNROLL);
    Res r[] = {
        run("f16  16x16x16", peak_f16),
        run("bf16 16x16x16", peak_bf16),
        run("i8   16x16x16", peak_i8),
        run("i4   16x16x16", peak_i4),
    };
    for (auto& x : r) printf("%-14s %9.3f ms  %9.1f TOPS\n", x.name, x.ms, ops / (x.ms * 1e-3) / 1e12);
    printf("i8 / f16 = %.2fx    i4 / i8 = %.2fx\n", r[0].ms / r[2].ms, r[2].ms / r[3].ms);
    CK(hipFree(out));
    CK(hipFree(in));
    return 0;
}
