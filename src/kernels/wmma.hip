#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <type_traits>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// `y` of the grouped MoE prefill: the output of the down GEMM for every (token, slot) pair, read back once by
// `moe_combine`. With MALLEUS_MOE_Y_HALF it is fp16 instead of f32: the round trip is the widest intermediate of
// the prefill (755 MB a layer at pp8192, and `moe_combine` alone measured 72 % of the memory wall). fp16 and not
// bf16 because this path already computes in fp16 and the extra mantissa bits come at the same width.
template <bool YB>
struct YRow {
    // Indexed in elements from the row base, so the two widths address the same buffer with one index.
    static __device__ __forceinline__ void store(float* base, size_t i, float v, int accumulate) {
        base[i] = accumulate ? base[i] + v : v;
    }
};

template <>
struct YRow<true> {
    static __device__ __forceinline__ void store(float* base, size_t i, float v, int) {
        ((_Float16*)base)[i] = (_Float16)v;
    }
};

// Randomized Hadamard on 128-wide blocks, fp16 output (prefill activations).
extern "C" __global__ void __launch_bounds__(128) hadamard128_h(_Float16* out, const float* x, const int8_t* signs, int K) {
    __shared__ float s[128];
    const size_t base = (size_t)blockIdx.x * 128;
    const int i = threadIdx.x;
    s[i] = x[base + i] * (float)signs[i];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        const int j = i ^ h;
        const float a = s[i], b = s[j];
        __syncthreads();
        s[i] = (i & h) ? b - a : a + b;
        __syncthreads();
    }
    out[base + i] = (_Float16)(s[i] * 0.08838834764831845f);
}

#define BN 128
#define BK 64
#define LDS_STRIDE 72

union H16 {
    half16 v;
    uint4 u[2];
};

// out[T, N] (+)= xh[T, K] * dequant(W)[N, K]^T. W is MLS4 (int4 nibbles + fp16 scale per 128-group) or
// MLS8 (int8 + fp32 scale per row, is_int8 = 1). 256 threads = 8 waves as WAVES_T x (8 / WAVES_T); each wave
// owns a (BM / WAVES_T) x (BN * WAVES_T / 8) tile of 16x16 WMMA fragments. The next K tile is prefetched.
template <int BM, int WAVES_T>
__device__ __forceinline__ void gemm_wmma_impl(float* out, const _Float16* xh, const uint8_t* q, const void* sc,
                                               int T, int N, int K, int accumulate, int is_int8) {
    constexpr int WAVES_N = 8 / WAVES_T, WT = BM / WAVES_T, WN = BN / WAVES_N, FA = WT / 16, FB = WN / 16;
    constexpr int XPARTS = 256 / BM, XHALVES = BK / XPARTS, XVEC = XHALVES / 8;
    __shared__ __attribute__((aligned(16))) _Float16 xs[BM][LDS_STRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 ws[BN][LDS_STRIDE];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int t0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int wt = (wave / WAVES_N) * WT, wn = (wave % WAVES_N) * WN;
    float8 acc[FA][FB];
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b) acc[a][b] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    const int xrow = tid / XPARTS, xpart = tid % XPARTS;
    const int wrow = tid / 2, wpart = tid % 2;
    const int G = K / 128;
    const int t = t0 + xrow;
    const int n = n0 + wrow < N ? n0 + wrow : N - 1;
    uint4 xv[XVEC], wq[2];
    float wscale = 1.f;
    auto load = [&](int k0) {
        if (t < T) {
            const uint4* p = (const uint4*)(xh + (size_t)t * K + k0 + xpart * XHALVES);
#pragma unroll
            for (int i = 0; i < XVEC; ++i) xv[i] = p[i];
        } else {
#pragma unroll
            for (int i = 0; i < XVEC; ++i) xv[i] = (uint4){0, 0, 0, 0};
        }
        if (is_int8) {
            const uint4* p = (const uint4*)(q + (size_t)n * K + k0 + wpart * 32);
            wq[0] = p[0];
            wq[1] = p[1];
        } else {
            wq[0] = *(const uint4*)(q + (size_t)n * (K / 2) + (k0 + wpart * 32) / 2);
            wscale = __half2float(((const __half*)sc)[(size_t)n * G + k0 / 128]);
        }
    };
    auto store = [&]() {
        uint4* d = (uint4*)&xs[xrow][xpart * XHALVES];
#pragma unroll
        for (int i = 0; i < XVEC; ++i) d[i] = xv[i];
        H16 h[2];
        if (is_int8) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t w4[4] = {wq[c].x, wq[c].y, wq[c].z, wq[c].w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) h[c].v[i * 4 + j] = (_Float16)(float)(((int32_t)(w4[i] << (24 - 8 * j))) >> 24);
            }
        } else {
            const uint32_t w4[4] = {wq[0].x, wq[0].y, wq[0].z, wq[0].w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    h[i / 2].v[(i % 2) * 8 + j] = (_Float16)((float)(((int32_t)(w4[i] << (28 - 4 * j))) >> 28) * wscale);
        }
        uint4* e = (uint4*)&ws[wrow][wpart * 32];
        e[0] = h[0].u[0];
        e[1] = h[0].u[1];
        e[2] = h[1].u[0];
        e[3] = h[1].u[1];
    };
    load(0);
    for (int k0 = 0; k0 < K; k0 += BK) {
        store();
        __syncthreads();
        if (k0 + BK < K) load(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            H16 af[FA], bf[FB];
#pragma unroll
            for (int a = 0; a < FA; ++a) {
                const uint4* p = (const uint4*)&xs[wt + a * 16 + lane % 16][kk];
                af[a].u[0] = p[0];
                af[a].u[1] = p[1];
            }
#pragma unroll
            for (int b = 0; b < FB; ++b) {
                const uint4* p = (const uint4*)&ws[wn + b * 16 + lane % 16][kk];
                bf[b].u[0] = p[0];
                bf[b].u[1] = p[1];
            }
#pragma unroll
            for (int a = 0; a < FA; ++a)
#pragma unroll
                for (int b = 0; b < FB; ++b)
                    acc[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[a].v, bf[b].v, acc[a][b]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tt = t0 + wt + a * 16 + 2 * i + lane / 16;
                const int nn = n0 + wn + b * 16 + lane % 16;
                if (tt < T && nn < N) {
                    float v = acc[a][b][i];
                    if (is_int8) v *= ((const float*)sc)[nn];
                    float* o = out + (size_t)tt * N + nn;
                    *o = accumulate ? *o + v : v;
                }
            }
}

#define GEMM_WMMA(NAME, BM, WT)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* out, const _Float16* xh, const uint8_t* q,       \
                                                           const void* sc, int T, int N, int K, int accumulate,     \
                                                           int is_int8) {                                           \
        gemm_wmma_impl<BM, WT>(out, xh, q, sc, T, N, K, accumulate, is_int8);                                     \
    }

GEMM_WMMA(gemm_wmma, 128, 4)
GEMM_WMMA(gemm_wmma_m64, 64, 2)
GEMM_WMMA(gemm_wmma_m32, 32, 1)

typedef int int2v_t __attribute__((ext_vector_type(2)));
typedef int int4v_t __attribute__((ext_vector_type(4)));
typedef int int8v_t __attribute__((ext_vector_type(8)));

// Per-row symmetric int8 quantization of rotated activations: one 256-thread block per row.
extern "C" __global__ void __launch_bounds__(256) quant_rows_i8(int8_t* q, float* scale, const float* x, int K) {
    __shared__ float sh[8];
    const float* xr = x + (size_t)blockIdx.x * K;
    float m = 0.f;
    for (int i = threadIdx.x; i < K; i += 256) m = fmaxf(m, fabsf(xr[i]));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = m;
    __syncthreads();
    m = 0.f;
    for (int i = 0; i < 8; ++i) m = fmaxf(m, sh[i]);
    const float s = m > 0.f ? m / 127.f : 1.f;
    const float inv = 1.f / s;
    for (int i = threadIdx.x; i < K; i += 256) q[(size_t)blockIdx.x * K + i] = (int8_t)__float2int_rn(xr[i] * inv);
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
}

// Randomized Hadamard + int8 quantization per 128-wide group: one 128-thread block per (row, group).
// xsc is [T, K / 128].
extern "C" __global__ void __launch_bounds__(128) rot_quant_i8(int8_t* q, float* xsc, const float* x, const int8_t* signs,
                                                               int K) {
    __shared__ float s[128];
    __shared__ float mx[4];
    const size_t base = (size_t)blockIdx.x * 128;
    const int i = threadIdx.x;
    s[i] = x[base + i] * (float)signs[i];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        const int j = i ^ h;
        const float a = s[i], b = s[j];
        __syncthreads();
        s[i] = (i & h) ? b - a : a + b;
        __syncthreads();
    }
    const float v = s[i] * 0.08838834764831845f;
    float m = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (i % 32 == 0) mx[i / 32] = m;
    __syncthreads();
    m = fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3]));
    const float sc = m > 0.f ? m / 127.f : 1.f;
    q[base + i] = (int8_t)__float2int_rn(v / sc);
    if (i == 0) xsc[blockIdx.x] = sc;
}

// Randomized Hadamard + int4 quantization per 128-wide group, for the W4A4 MoE: as rot_quant_i8 but the scale is
// max / 7 and two elements share a byte (nibble j of byte b = element 2b + j, the MLS4 layout). q4 is [T, K / 2],
// xsc is [T, K / 128].
extern "C" __global__ void __launch_bounds__(128) rot_quant_i4(int8_t* q4, float* xsc, const float* x,
                                                               const int8_t* signs, int K) {
    __shared__ float s[128];
    __shared__ float mx[4];
    __shared__ int8_t nib[128];
    const size_t base = (size_t)blockIdx.x * 128;
    const int i = threadIdx.x;
    s[i] = x[base + i] * (float)signs[i];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        const int j = i ^ h;
        const float a = s[i], b = s[j];
        __syncthreads();
        s[i] = (i & h) ? b - a : a + b;
        __syncthreads();
    }
    const float v = s[i] * 0.08838834764831845f;
    float m = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (i % 32 == 0) mx[i / 32] = m;
    __syncthreads();
    m = fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3]));
    const float sc = m > 0.f ? m / 7.f : 1.f;
    nib[i] = (int8_t)__float2int_rn(fminf(fmaxf(v / sc, -8.f), 7.f));
    __syncthreads();
    if (i < 64) q4[base / 2 + i] = (int8_t)((nib[2 * i] & 15) | (nib[2 * i + 1] << 4));
    if (i == 0) xsc[blockIdx.x] = sc;
}

// Eight int4 (two's complement, nibble j = element j) -> eight int8 in two u32.
__device__ __forceinline__ void nib_to_s8(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t e = w & 0x0f0f0f0fu, o = (w >> 4) & 0x0f0f0f0fu;
    const uint32_t es = e | ((e & 0x08080808u) * 30u), os = o | ((o & 0x08080808u) * 30u);
    lo = __builtin_amdgcn_perm(os, es, 0x05010400u);
    hi = __builtin_amdgcn_perm(os, es, 0x07030602u);
}

// out[T, N] (+)= (xq[T, K] * xsc[T, K / 128]) . W[N, K]^T with int8 WMMA. W4: MLS4 nibbles + fp16 scale per 128-group;
// otherwise MLS8 int8 + fp32 scale per row. One 128-wide K group per step: int32 partial sums are folded into fp32
// with the group scale. 256 threads = 8 waves as WAVES_T x (8 / WAVES_T).
// A4: the activations are int4 nibbles too (xq [T, K / 2]) and both operands go to the int4 WMMA, which on gfx11 is
// double rate; it implies W4, so the MLS4 nibbles reach the WMMA unmodified.
template <int BM, int WAVES_T, bool W4, bool A4 = false>
__device__ __forceinline__ void gemm_a8_impl(float* out, const int8_t* xq, const float* xsc, const uint8_t* q,
                                             const void* sc, int T, int N, int K, int accumulate) {
    static_assert(!A4 || W4, "int4 activations need int4 weights");
    constexpr int WAVES_N = 8 / WAVES_T, WT = BM / WAVES_T, WN = BN / WAVES_N, FA = WT / 16, FB = WN / 16;
    // Bytes of one 128-wide K group held in LDS per row: 128 with the operands widened to int8, 64 when both stay
    // nibbles (A4). The 72-byte A4 stride keeps the fragment loads bank-conflict free but is only 8-byte aligned,
    // so that path moves everything 8 bytes at a time.
    constexpr int XP = 256 / BM, AB = A4 ? 64 : 128, STR = A4 ? 72 : 144, XBA = AB / XP;
    using VT = std::conditional_t<A4, uint2, uint4>;
    constexpr int VS = (int)sizeof(VT), XV = XBA / VS, XGS = AB / VS;
    constexpr int WGB = W4 ? 32 : 64, WVN = WGB / VS, WGS = 2 * WGB / VS, WOB = A4 ? 32 : 64;
    __shared__ __attribute__((aligned(16))) int8_t xs[BM][STR];
    __shared__ __attribute__((aligned(16))) int8_t ws[BN][STR];
    __shared__ float gsc[BN];
    __shared__ float xgs[BM];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    // Grouped order: runs of 4 token tiles x consecutive row tiles share weight and activation tiles in L2.
    const int nt = (T + BM - 1) / BM, nn_tiles = (N + BN - 1) / BN;
    const int gsz = min(4, nt), grp = blockIdx.x / (gsz * nn_tiles), rem = blockIdx.x % (gsz * nn_tiles);
    const int gfirst = grp * 4, gcount = min(4, nt - gfirst);
    const int tb = gfirst + rem % gcount, nb = rem / gcount;
    if (nb >= nn_tiles) return;
    const int t0 = tb * BM, n0 = nb * BN;
    const int wt = (wave / WAVES_N) * WT, wn = (wave % WAVES_N) * WN;
    float8 accf[FA][FB];
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b) accf[a][b] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    const int G = K / 128;
    const int xrow = tid / XP, xpart = tid % XP, wrow = tid / 2, wpart = tid % 2;
    const int t = min(t0 + xrow, T - 1), n = min(n0 + wrow, N - 1);
    const VT* xp = (const VT*)(xq + (size_t)t * (A4 ? K / 2 : K) + xpart * XBA);
    const VT* qp = (const VT*)(q + (size_t)n * (W4 ? K / 2 : K) + wpart * WGB);
    VT xv[XV], wv[WVN];
    float wscv = 1.f;
    auto load = [&](int g) {
#pragma unroll
        for (int i = 0; i < XV; ++i) xv[i] = xp[g * XGS + i];
#pragma unroll
        for (int i = 0; i < WVN; ++i) wv[i] = qp[g * WGS + i];
        if (W4) wscv = __half2float(((const __half*)sc)[(size_t)n * G + g]);
    };
    load(0);
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int i = 0; i < XV; ++i) ((VT*)&xs[xrow][xpart * XBA])[i] = xv[i];
        if (tid < BM) xgs[tid] = xsc[(size_t)min(t0 + tid, T - 1) * G + g];
        VT* d = (VT*)&ws[wrow][wpart * WOB];
        if constexpr (A4) {
#pragma unroll
            for (int i = 0; i < WVN; ++i) d[i] = wv[i];
        } else if constexpr (W4) {
            const uint4* w4 = (const uint4*)wv;
            const uint32_t w8[8] = {w4[0].x, w4[0].y, w4[0].z, w4[0].w, w4[1].x, w4[1].y, w4[1].z, w4[1].w};
            uint32_t o[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) nib_to_s8(w8[i], o[2 * i], o[2 * i + 1]);
#pragma unroll
            for (int i = 0; i < 4; ++i) d[i] = (uint4){o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]};
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) d[i] = wv[i];
        }
        if constexpr (W4) {
            if (wpart == 0) gsc[wrow] = wscv;
        }
        __syncthreads();
        if (g + 1 < G) load(g + 1);
        int8v_t acci[FA][FB];
#pragma unroll
        for (int a = 0; a < FA; ++a)
#pragma unroll
            for (int b = 0; b < FB; ++b) acci[a][b] = (int8v_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            if constexpr (A4) {
                int2v_t af[FA], bf[FB];
#pragma unroll
                for (int a = 0; a < FA; ++a) af[a] = *(const int2v_t*)&xs[wt + a * 16 + lane % 16][kk / 2];
#pragma unroll
                for (int b = 0; b < FB; ++b) bf[b] = *(const int2v_t*)&ws[wn + b * 16 + lane % 16][kk / 2];
#pragma unroll
                for (int a = 0; a < FA; ++a)
#pragma unroll
                    for (int b = 0; b < FB; ++b)
                        acci[a][b] =
                            __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af[a], true, bf[b], acci[a][b], false);
            } else {
                int4v_t af[FA], bf[FB];
#pragma unroll
                for (int a = 0; a < FA; ++a) af[a] = *(const int4v_t*)&xs[wt + a * 16 + lane % 16][kk];
#pragma unroll
                for (int b = 0; b < FB; ++b) bf[b] = *(const int4v_t*)&ws[wn + b * 16 + lane % 16][kk];
#pragma unroll
                for (int a = 0; a < FA; ++a)
#pragma unroll
                    for (int b = 0; b < FB; ++b)
                        acci[a][b] =
                            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[a], true, bf[b], acci[a][b], false);
            }
        }
        float rs[FA][8];
#pragma unroll
        for (int a = 0; a < FA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) rs[a][i] = xgs[wt + a * 16 + 2 * i + lane / 16];
#pragma unroll
        for (int b = 0; b < FB; ++b) {
            const float s = W4 ? gsc[wn + b * 16 + lane % 16] : 1.f;
#pragma unroll
            for (int a = 0; a < FA; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) accf[a][b][i] += (float)acci[a][b][i] * (s * rs[a][i]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tt = t0 + wt + a * 16 + 2 * i + lane / 16;
                const int nn = n0 + wn + b * 16 + lane % 16;
                if (tt < T && nn < N) {
                    float v = accf[a][b][i];
                    if (!W4) v *= ((const float*)sc)[nn];
                    float* o = out + (size_t)tt * N + nn;
                    *o = accumulate ? *o + v : v;
                }
            }
}

#define GEMM_A8(NAME, BM, WT, W4)                                                                                \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* out, const int8_t* xq, const float* xsc,         \
                                                           const uint8_t* q, const void* sc, int T, int N, int K,   \
                                                           int accumulate) {                                        \
        gemm_a8_impl<BM, WT, W4>(out, xq, xsc, q, sc, T, N, K, accumulate);                                       \
    }

GEMM_A8(gemm_w4a8, 128, 2, true)
GEMM_A8(gemm_w4a8_m64, 64, 2, true)
GEMM_A8(gemm_w4a8_m32, 32, 1, true)
GEMM_A8(gemm_w8a8, 128, 2, false)
GEMM_A8(gemm_w8a8_m64, 64, 2, false)
GEMM_A8(gemm_w8a8_m32, 32, 1, false)

// Same tiles with int4 activations as well: the int4 WMMA is double rate and the MLS4 nibbles reach it unmodified.
#define GEMM_A4(NAME, BM, WT)                                                                                      \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* out, const int8_t* xq, const float* xsc,         \
                                                           const uint8_t* q, const void* sc, int T, int N, int K,   \
                                                           int accumulate) {                                        \
        gemm_a8_impl<BM, WT, true, true>(out, xq, xsc, q, sc, T, N, K, accumulate);                                \
    }

GEMM_A4(gemm_w4a4, 128, 2)
GEMM_A4(gemm_w4a4_m64, 64, 2)
GEMM_A4(gemm_w4a4_m32, 32, 1)

// Grouped MoE GEMM on MLS4 experts (no int8 copy): rows [offs[e], offs[e + 1]) of xq [P, K] (int8 with one scale per
// 128-group, xsc [P, K / 128]) are sorted by expert and use expert e (q [E1, N, K / 2] nibbles, sc [E1, N, K / 128]
// fp16); tiles lists (e, first row) of every BM-row tile. A4: the activations are int4 nibbles too (xq [P, K / 2])
// and both operands go to the int4 WMMA, which on gfx11 is double rate and takes the weight nibbles as they are.
// Otherwise, as gemm_a8_impl: the weight group is widened to int8 in LDS,
// int32 WMMA per 128-group folded into fp32 with both scales. GLU: N = 2I, per expert I gate rows then I up rows;
// a 128-column tile takes 16-row blocks of gate and up in turn and writes silu(gate) * up to out [P, I].
template <int BM, int WAVES_T, bool GLU, bool A4, bool YB = false>
__device__ __forceinline__ void moe_a4_impl(float* out, const int8_t* xq, const float* xsc, const uint8_t* q,
                                            const __half* sc, const int* offs, const int2* tiles, const int* ntiles,
                                            int N, int K) {
    constexpr int WAVES_N = 8 / WAVES_T, WT = BM / WAVES_T, WN = BN / WAVES_N, FA = WT / 16, FB = WN / 16;
    // Bytes of one 128-wide K group held in LDS per row: 128 with the activations widened to int8, 64 when both
    // operands stay nibbles (A4). The 72-byte A4 row stride keeps the fragment loads bank-conflict free but is only
    // 8-byte aligned, so that path moves everything 8 bytes at a time; the weight group is 64 nibble bytes either
    // way, loaded by the two threads of a row at wpart * 32.
    constexpr int XP = 256 / BM, AB = A4 ? 64 : 128, STR = A4 ? 72 : 144, XBA = AB / XP;
    // Where the two threads of a weight row write: 32 nibble bytes stay 32 bytes, widened to int8 they are 64.
    constexpr int WOB = A4 ? 32 : 64;
    using VT = std::conditional_t<A4, uint2, uint4>;
    constexpr int VS = (int)sizeof(VT), XV = XBA / VS, XGS = AB / VS, WVN = 32 / VS, WGS = 64 / VS;
    // Two LDS stages: group g + 1 writes into the other stage while the waves are still multiplying g, so one
    // barrier per group instead of two and the global loads overlap the WMMA. `gemm_a8_impl` already works this way
    // (85 % of the int8 ceiling); without it, the grouped MoE sat at 15 % of the int4 one.
    constexpr int NBUF = (2 * (BM + BN) * STR + 8 * (BM + BN)) <= 40 * 1024 ? 2 : 1;
    __shared__ __attribute__((aligned(16))) int8_t xs[NBUF][BM][STR];
    __shared__ __attribute__((aligned(16))) int8_t ws[NBUF][BN][STR];
    __shared__ float gsc[NBUF][BN];
    __shared__ float xgs[NBUF][BM];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int nn_tiles = (N + BN - 1) / BN, tile = blockIdx.x / nn_tiles, nb = blockIdx.x % nn_tiles;
    if (tile >= *ntiles) return;
    const int2 td = tiles[tile];
    const int e = td.x, r0 = offs[e] + td.y, T = min(BM, offs[e + 1] - r0);
    const int G = K / 128, I = N / 2;
    const uint8_t* qe = q + (size_t)e * N * (K / 2);
    const __half* se = sc + (size_t)e * N * G;
    const int n0 = nb * BN;
    const int wt = (wave / WAVES_N) * WT, wn = (wave % WAVES_N) * WN;
    float8 accf[FA][FB];
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b) accf[a][b] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    const int xrow = tid / XP, xpart = tid % XP, wrow = tid / 2, wpart = tid % 2;
    const int t = r0 + min(xrow, T - 1);
    const int v = min(n0 + wrow, N - 1);
    const int n = GLU ? (v >> 5) * 16 + (v & 15) + ((v >> 4) & 1) * I : v;
    const VT* xp = (const VT*)(xq + (size_t)t * (A4 ? K / 2 : K) + xpart * XBA);
    const VT* qp = (const VT*)(qe + (size_t)n * (K / 2) + wpart * 32);
    VT xv[XV], wv[WVN];
    float wscv = 1.f;
    auto load = [&](int g) {
#pragma unroll
        for (int i = 0; i < XV; ++i) xv[i] = xp[g * XGS + i];
#pragma unroll
        for (int i = 0; i < WVN; ++i) wv[i] = qp[g * WGS + i];
        wscv = __half2float(se[(size_t)n * G + g]);
    };
    // Il gruppo appena letto in `xv`/`wv` va nello stadio `pb`.
    auto stage = [&](int g, int pb) {
#pragma unroll
        for (int i = 0; i < XV; ++i) ((VT*)&xs[pb][xrow][xpart * XBA])[i] = xv[i];
        if (tid < BM) xgs[pb][tid] = xsc[(size_t)(r0 + min(tid, T - 1)) * G + g];
        VT* d = (VT*)&ws[pb][wrow][wpart * WOB];
        if constexpr (A4) {
#pragma unroll
            for (int i = 0; i < WVN; ++i) d[i] = wv[i];
        } else {
            const uint4* w4 = (const uint4*)wv;
            const uint32_t w8[8] = {w4[0].x, w4[0].y, w4[0].z, w4[0].w, w4[1].x, w4[1].y, w4[1].z, w4[1].w};
            uint32_t o[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) nib_to_s8(w8[i], o[2 * i], o[2 * i + 1]);
#pragma unroll
            for (int i = 0; i < 4; ++i) d[i] = (uint4){o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]};
        }
        if (wpart == 0) gsc[pb][wrow] = wscv;
    };
    load(0);
    stage(0, 0);
    __syncthreads();
    for (int g = 0; g < G; ++g) {
        const int pb = NBUF == 2 ? (g & 1) : 0;
        if (NBUF == 2) {
            if (g + 1 < G) {
                load(g + 1);
                stage(g + 1, pb ^ 1);
            }
        } else if (g > 0) {
            // One stage: lay this group out now, behind the barrier that closed the previous one.
            load(g);
            stage(g, 0);
            __syncthreads();
        }
        int8v_t acci[FA][FB];
#pragma unroll
        for (int a = 0; a < FA; ++a)
#pragma unroll
            for (int b = 0; b < FB; ++b) acci[a][b] = (int8v_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            if constexpr (A4) {
                int2v_t af[FA], bf[FB];
#pragma unroll
                for (int a = 0; a < FA; ++a) af[a] = *(const int2v_t*)&xs[pb][wt + a * 16 + lane % 16][kk / 2];
#pragma unroll
                for (int b = 0; b < FB; ++b) bf[b] = *(const int2v_t*)&ws[pb][wn + b * 16 + lane % 16][kk / 2];
#pragma unroll
                for (int a = 0; a < FA; ++a)
#pragma unroll
                    for (int b = 0; b < FB; ++b)
                        acci[a][b] =
                            __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af[a], true, bf[b], acci[a][b], false);
            } else {
                int4v_t af[FA], bf[FB];
#pragma unroll
                for (int a = 0; a < FA; ++a) af[a] = *(const int4v_t*)&xs[pb][wt + a * 16 + lane % 16][kk];
#pragma unroll
                for (int b = 0; b < FB; ++b) bf[b] = *(const int4v_t*)&ws[pb][wn + b * 16 + lane % 16][kk];
#pragma unroll
                for (int a = 0; a < FA; ++a)
#pragma unroll
                    for (int b = 0; b < FB; ++b)
                        acci[a][b] =
                            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[a], true, bf[b], acci[a][b], false);
            }
        }
        float rs[FA][8];
#pragma unroll
        for (int a = 0; a < FA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) rs[a][i] = xgs[pb][wt + a * 16 + 2 * i + lane / 16];
#pragma unroll
        for (int b = 0; b < FB; ++b) {
            const float s = gsc[pb][wn + b * 16 + lane % 16];
#pragma unroll
            for (int a = 0; a < FA; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) accf[a][b][i] += (float)acci[a][b][i] * (s * rs[a][i]);
        }
        __syncthreads();
    }

    if (GLU) {
        static_assert(FB % 2 == 0, "gate/up pairs need an even number of column fragments");
#pragma unroll
        for (int b = 0; b < FB; b += 2) {
            const int col = (n0 + wn + b * 16) / 2 + lane % 16;
#pragma unroll
            for (int a = 0; a < FA; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int tt = wt + a * 16 + 2 * i + lane / 16;
                    if (tt < T && col < I) {
                        const float gv = accf[a][b][i], uv = accf[a][b + 1][i];
                        out[(size_t)(r0 + tt) * I + col] = gv / (1.f + __expf(-gv)) * uv;
                    }
                }
        }
        return;
    }
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tt = wt + a * 16 + 2 * i + lane / 16;
                const int nn = n0 + wn + b * 16 + lane % 16;
                if (tt < T && nn < N) YRow<YB>::store(out, (size_t)(r0 + tt) * N + nn, accf[a][b][i], 0);
            }
}

#ifndef MOE_WPE_DN
#define MOE_WPE_DN 8
#endif
#define MOE_A4(NAME, BM, WT, GLU, WPE, A4, YB)                                                                    \
    extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE)))                    \
        NAME(float* out, const int8_t* xq, const float* xsc, const uint8_t* q, const __half* sc, const int* offs,  \
             const int2* tiles, const int* ntiles, int N, int K) {                                                \
        moe_a4_impl<BM, WT, GLU, A4, YB>(out, xq, xsc, q, sc, offs, tiles, ntiles, N, K);                         \
    }

// Waves per SIMD the register allocation targets: 8 lets two blocks per CU keep weight loads in flight.
MOE_A4(moe_w4a8_glu, 64, 2, true, 8, false, false)
MOE_A4(moe_w4a8, 64, 2, false, MOE_WPE_DN, false, false)
MOE_A4(moe_w4a8_glu_m128, 128, 4, true, 8, false, false)
MOE_A4(moe_w4a8_m128, 128, 4, false, 8, false, false)
MOE_A4(moe_w4a8_glu_m32, 32, 2, true, 8, false, false)
MOE_A4(moe_w4a8_m32, 32, 2, false, 8, false, false)
// Same tiles with int4 activations: the int4 WMMA is double rate and the weight nibbles reach it unmodified.
MOE_A4(moe_w4a4_glu, 64, 2, true, 8, true, false)
MOE_A4(moe_w4a4, 64, 2, false, MOE_WPE_DN, true, false)
MOE_A4(moe_w4a4_glu_m128, 128, 4, true, 8, true, false)
MOE_A4(moe_w4a4_m128, 128, 4, false, 8, true, false)
MOE_A4(moe_w4a4_glu_m32, 32, 2, true, 8, true, false)
MOE_A4(moe_w4a4_m32, 32, 2, false, 8, true, false)
// The down GEMMs of the same tiles, writing `y` as fp16 (MALLEUS_MOE_Y_HALF): same arithmetic, half the bytes of
// the intermediate that `moe_combine` reads back.
MOE_A4(moe_w4a8_yh, 64, 2, false, MOE_WPE_DN, false, true)
MOE_A4(moe_w4a8_m128_yh, 128, 4, false, 8, false, true)
MOE_A4(moe_w4a8_m32_yh, 32, 2, false, 8, false, true)
MOE_A4(moe_w4a4_yh, 64, 2, false, MOE_WPE_DN, true, true)
MOE_A4(moe_w4a4_m128_yh, 128, 4, false, 8, true, true)
MOE_A4(moe_w4a4_m32_yh, 32, 2, false, 8, true, true)

// xg[r] = xq[perm[r] / slots] (int4 rows of K / 2 bytes) with its K / 128 group scales.
extern "C" __global__ void __launch_bounds__(128) moe_gather_g4(int8_t* xg, float* sg, const int8_t* xq,
                                                                const float* xs, const int* perm, int slots, int K) {
    const int r = blockIdx.x, t = perm[r] / slots, G = K / 128, KB = K / 2;
    for (int k = threadIdx.x * 16; k < KB; k += 128 * 16)
        *(uint4*)(xg + (size_t)r * KB + k) = *(const uint4*)(xq + (size_t)t * KB + k);
    for (int g = threadIdx.x; g < G; g += 128) sg[(size_t)r * G + g] = xs[(size_t)t * G + g];
}

// xg[r] = xq[perm[r] / slots] (int8 rows of K bytes) with its K / 128 group scales.
extern "C" __global__ void __launch_bounds__(128) moe_gather_g(int8_t* xg, float* sg, const int8_t* xq, const float* xs,
                                                               const int* perm, int slots, int K) {
    const int r = blockIdx.x, t = perm[r] / slots, G = K / 128;
    for (int k = threadIdx.x * 16; k < K; k += 128 * 16)
        *(uint4*)(xg + (size_t)r * K + k) = *(const uint4*)(xq + (size_t)t * K + k);
    for (int g = threadIdx.x; g < G; g += 128) sg[(size_t)r * G + g] = xs[(size_t)t * G + g];
}

// out[T, N] (+)= (xq[T, K] . W[N, K]^T) * xs[T] * ws[N]: int8 activations with a per-row scale, MLS8 weights.
// int32 accumulation over the whole K (|sum| < K * 127^2 < 2^31 for K < 133k). NW waves as WAVES_T x (NW / WAVES_T);
// K is staged through LDS in BK2-wide double-buffered slices; K / BK2 must be even.
#ifndef I8_PAD
#define I8_PAD 16
#endif

// Output columns: SEG = 0 splits them into out0 [T, n0], out1 [T, n1], out2 [T, N - n0 - n1] (fused projections
// sharing an input; segment sizes multiple of BN2); SEG = 1 takes rows interleaved in blocks of 16 (gate, up) and
// writes silu(gate) * up to out0 [T, N / 2]; with gluh > 0 the weight rows are not interleaved (gate rows, then up
// rows from gluh) and the blocks of 16 are addressed in place.
template <int BM, int BN2, int WAVES_T, int BK2, int NW, int GS, int SEG>
__device__ __forceinline__ void gemm_i8_impl(float* out0, float* out1, float* out2, int n0s, int n1s,
                                             const int8_t* xq, const float* xs, const int8_t* q, const float* ws,
                                             int T, int N, int K, int accumulate, int bid = -1, int gluh = 0,
                                             int ldk = 0, const float* mx = nullptr, const float* minv = nullptr,
                                             const uint16_t* mw = nullptr, int mc = 0, int mh = 0) {
    if (ldk == 0) ldk = K;
    if (bid < 0) bid = blockIdx.x;
    const auto wrow = [gluh](int v) { return gluh ? (v >> 5) * 16 + (v & 15) + ((v >> 4) & 1) * gluh : v; };
    constexpr int WAVES_N = NW / WAVES_T, WT = BM / WAVES_T, WN = BN2 / WAVES_N, FA = WT / 16, FB = WN / 16;
    constexpr int STR = BK2 + I8_PAD, ROWS = BM + BN2, CH = BK2 / 16, PER = (ROWS * CH + NW * 32 - 1) / (NW * 32);
    __shared__ __attribute__((aligned(16))) int8_t sm[2][ROWS][STR];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int nt = (T + BM - 1) / BM, nn_tiles = (N + BN2 - 1) / BN2;
    int tb, nb;
    if (GS == 0) {
        tb = bid % nt;
        nb = bid / nt;
    } else {
        const int gsz = min(GS, nt), grp = bid / (gsz * nn_tiles), rem = bid % (gsz * nn_tiles);
        const int gfirst = grp * GS, gcount = min(GS, nt - gfirst);
        tb = gfirst + rem % gcount;
        nb = rem / gcount;
    }
    if (nb >= nn_tiles) return;
    const int t0 = tb * BM, n0 = nb * BN2;
    const int wt = (wave / WAVES_N) * WT, wn = BM + (wave % WAVES_N) * WN;
    const int4v_t* src[PER];
    int dr[PER], dc[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = min(tid + i * NW * 32, ROWS * CH - 1), r = c / CH, cc = c % CH;
        dr[i] = r;
        dc[i] = cc * 16;
        src[i] = r < BM ? (const int4v_t*)(xq + (size_t)min(t0 + r, T - 1) * ldk + cc * 16)
                        : (const int4v_t*)(q + (size_t)wrow(min(n0 + r - BM, N - 1)) * ldk + cc * 16);
    }
    int8v_t acc[FA][FB];
#pragma unroll
    for (int a = 0; a < FA; ++a)
#pragma unroll
        for (int b = 0; b < FB; ++b) acc[a][b] = (int8v_t){0, 0, 0, 0, 0, 0, 0, 0};
    const int S = K / BK2;
    int4v_t v0[PER], v1[PER];
#ifndef I8_ST
#define I8_ST(st) min(st, S - 1)
#endif
#define I8_LOAD(v, st)                                                                                            \
    _Pragma("unroll") for (int i = 0; i < PER; ++i) v[i] = src[i][I8_ST(st) * (BK2 / 16)];
#define I8_STORE(buf, v) _Pragma("unroll") for (int i = 0; i < PER; ++i) *(int4v_t*)&sm[buf][dr[i]][dc[i]] = v[i];
#define I8_COMPUTE(buf)                                                                                           \
    _Pragma("unroll") for (int kk = 0; kk < BK2; kk += 16) {                                                      \
        int4v_t af[FA], bf[FB];                                                                                   \
        _Pragma("unroll") for (int a = 0; a < FA; ++a) af[a] = *(const int4v_t*)&sm[buf][wt + a * 16 + lane % 16][kk]; \
        _Pragma("unroll") for (int b = 0; b < FB; ++b) bf[b] = *(const int4v_t*)&sm[buf][wn + b * 16 + lane % 16][kk]; \
        _Pragma("unroll") for (int a = 0; a < FA; ++a) _Pragma("unroll") for (int b = 0; b < FB; ++b) acc[a][b] = \
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[a], true, bf[b], acc[a][b], false);               \
        I8_SCHED                                                                                                  \
    }
#define I8_SCHED __builtin_amdgcn_sched_barrier(0);
    I8_LOAD(v0, 0)
    I8_STORE(0, v0)
    I8_LOAD(v0, 1)
    __syncthreads();
    for (int s = 0; s < S; s += 2) {
        I8_LOAD(v1, s + 2)
        I8_COMPUTE(0)
        I8_STORE(1, v0)
        __syncthreads();
        I8_LOAD(v0, s + 3)
        I8_COMPUTE(1)
        I8_STORE(0, v1)
        __syncthreads();
    }
    if (SEG == 1) {
        static_assert(FB % 2 == 0, "gate/up pairs need an even number of column fragments");
#pragma unroll
        for (int b = 0; b < FB; b += 2) {
            const int ng = n0 + wn - BM + b * 16 + lane % 16, nu = ng + 16;
            const float sg = ws[wrow(min(ng, N - 1))], su = ws[wrow(min(nu, N - 1))];
            const int col = (n0 + wn - BM + b * 16) / 2 + lane % 16;
#pragma unroll
            for (int a = 0; a < FA; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int tt = t0 + wt + a * 16 + 2 * i + lane / 16;
                    if (tt < T && nu < N) {
                        const float x = xs[tt], gv = (float)acc[a][b][i] * sg * x, uv = (float)acc[a][b + 1][i] * su * x;
                        out0[(size_t)tt * (N / 2) + col] = gv / (1.f + __expf(-gv)) * uv;
                    }
                }
        }
        return;
    }
    if (SEG == 2) {
        // Hyper-connection mix: column n = j * mc + s holds stream s of output j (mc consecutive lanes);
        // out0[t, j] = mean_s sigmoid(u) * mx[t, s, j] * minv[t, s] * (1 + mw[s, j]).
#pragma unroll
        for (int b = 0; b < FB; ++b) {
            const int nn = n0 + wn - BM + b * 16 + lane % 16;
            const int j = nn / mc, st = nn % mc;
            const float sw = ws[min(nn, N - 1)];
            const float wsj = nn < N ? __uint_as_float(((uint32_t)mw[st * mh + j]) << 16) + 1.f : 0.f;
#pragma unroll
            for (int a = 0; a < FA; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int tt = t0 + wt + a * 16 + 2 * i + lane / 16;
                    float v = 0.f;
                    if (tt < T && nn < N) {
                        const float u = (float)acc[a][b][i] * sw * xs[tt];
                        v = mx[((size_t)tt * mc + st) * mh + j] * minv[tt * mc + st] * wsj / (1.f + __expf(-u));
                    }
                    for (int o = 1; o < mc; o <<= 1) v += __shfl_xor(v, o, 32);
                    if (tt < T && nn < N && st == 0) out0[(size_t)tt * mh + j] = v / (float)mc;
                }
        }
        return;
    }
    const int nb0 = n0 + wn - BM;
    float* out = out0;
    int ld = n0s, nbase = 0;
    if (nb0 >= n0s + n1s) {
        out = out2;
        ld = N - n0s - n1s;
        nbase = n0s + n1s;
    } else if (nb0 >= n0s) {
        out = out1;
        ld = n1s;
        nbase = n0s;
    }
#pragma unroll
    for (int b = 0; b < FB; ++b) {
        const int nn = nb0 + b * 16 + lane % 16;
        const float sw = ws[min(nn, N - 1)];
#pragma unroll
        for (int a = 0; a < FA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tt = t0 + wt + a * 16 + 2 * i + lane / 16;
                if (tt < T && nn < N) {
                    const float val = (float)acc[a][b][i] * sw * xs[tt];
                    float* o = out + (size_t)tt * ld + (nn - nbase);
                    *o = accumulate ? *o + val : val;
                }
            }
    }
}

#define GEMM_I8G(NAME, BM, BN2, WT, BK2, NW, GS, SEG)                                                            \
    extern "C" __global__ void __launch_bounds__(NW * 32)                                                         \
        NAME(float* out0, float* out1, float* out2, int n0s, int n1s, const int8_t* xq, const float* xs,           \
             const int8_t* q, const float* ws, int T, int N, int K, int accumulate) {                             \
        gemm_i8_impl<BM, BN2, WT, BK2, NW, GS, SEG>(out0, out1, out2, n0s, n1s, xq, xs, q, ws, T, N, K, accumulate); \
    }

GEMM_I8G(gemm_i8, 256, 64, 8, 64, 8, 1, 0)

// Split-K: blockIdx.y = slice of KS columns (row stride ldk); partial products to out + slice * T * N.
// 128-column tiles: this GEMM is memory bound and the activation tile (T x ldk int8, 60 MB at T = 5900) is
// re-read once per column tile, so N / 128 passes beat N / 64 at the same weight traffic.
extern "C" __global__ void __launch_bounds__(256) gemm_i8_sk(float* out, const int8_t* xq, const float* xs,
                                                            const int8_t* q, const float* ws, int T, int N, int KS,
                                                            int ldk) {
    const int sl = blockIdx.y;
    float* o = out + (size_t)sl * T * N;
    gemm_i8_impl<256, 128, 8, 64, 8, 1, 0>(o, o, o, N, 0, xq + (size_t)sl * KS, xs, q + (size_t)sl * KS, ws, T, N, KS,
                                           0, -1, 0, ldk);
}

// Hyper-connection mix_up with the mix in the epilogue (rows ordered j * C + s): out [T, H].
extern "C" __global__ void __launch_bounds__(256) gemm_i8_hcmix(float* out, const int8_t* xq, const float* xs,
                                                               const int8_t* q, const float* ws, int T, int N, int K,
                                                               const float* x, const float* inv, const uint16_t* w,
                                                               int C, int H) {
    gemm_i8_impl<128, 128, 4, 64, 8, 1, 2>(out, out, out, 0, 0, xq, xs, q, ws, T, N, K, 0, -1, 0, 0, x, inv, w, C, H);
}
GEMM_I8G(gemm_i8_m128, 128, 128, 4, 64, 8, 1, 0)
GEMM_I8G(gemm_i8_glu, 256, 64, 8, 64, 8, 1, 1)
GEMM_I8G(gemm_i8_glu_m128, 128, 128, 4, 64, 8, 1, 1)

typedef float f32x4v __attribute__((ext_vector_type(4)));

// One butterfly stage inside the wave: the partner lane is lane ^ M. Written with __shfl_xor rather than dpp:
// a dpp variant of this loop needs the ISA-encoded row_xmask literal and, in this clang, produced garbage for the
// ctrl values 0x101..0x104; the wave form below is the part that pays in any case (no LDS, no barriers).
template <int M>
__device__ __forceinline__ void hadamard_stage(float (&v)[4], int lane) {
    const bool hi = lane & M;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float o = __shfl_xor(v[j], M, 32);
        v[j] = hi ? o - v[j] : v[j] + o;
    }
}

// 128-point randomized Hadamard on one wave: lane holds elements 4 * lane + j.
__device__ __forceinline__ void wave_hadamard128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
    hadamard_stage<1>(v, lane);
    hadamard_stage<2>(v, lane);
    hadamard_stage<4>(v, lane);
    hadamard_stage<8>(v, lane);
    hadamard_stage<16>(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= 0.08838834764831845f;
}

// The two per-128-group quantizers below used to be a 128-thread block with the Hadamard in shared memory and
// fourteen __syncthreads() per group. One wave per group, four elements per lane and the dpp butterfly above:
// same arithmetic in the same order, so the quantized bytes are identical.
extern "C" __global__ void __launch_bounds__(32) rot_quant_i8_w(int8_t* q, float* xsc, const float* x,
                                                                const int8_t* signs, int K) {
    const size_t base = (size_t)blockIdx.x * 128;
    const int lane = threadIdx.x;
    const char4 sg = *(const char4*)(signs + 4 * lane);
    const float4 xv = *(const float4*)(x + base + 4 * lane);
    float v[4] = {xv.x * (float)sg.x, xv.y * (float)sg.y, xv.z * (float)sg.z, xv.w * (float)sg.w};
    wave_hadamard128(v, lane);
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float sc = m > 0.f ? m / 127.f : 1.f;
    // Divides rather than multiplies by 1 / sc: the two round differently at a boundary and the byte would change.
    char4 o = {(char)__float2int_rn(fminf(fmaxf(v[0] / sc, -128.f), 127.f)),
               (char)__float2int_rn(fminf(fmaxf(v[1] / sc, -128.f), 127.f)),
               (char)__float2int_rn(fminf(fmaxf(v[2] / sc, -128.f), 127.f)),
               (char)__float2int_rn(fminf(fmaxf(v[3] / sc, -128.f), 127.f))};
    *(char4*)(q + base + 4 * lane) = o;
    if (lane == 0) xsc[blockIdx.x] = sc;
}

extern "C" __global__ void __launch_bounds__(32) rot_quant_i4_w(int8_t* q4, float* xsc, const float* x,
                                                                const int8_t* signs, int K) {
    const size_t base = (size_t)blockIdx.x * 128;
    const int lane = threadIdx.x;
    const char4 sg = *(const char4*)(signs + 4 * lane);
    const float4 xv = *(const float4*)(x + base + 4 * lane);
    float v[4] = {xv.x * (float)sg.x, xv.y * (float)sg.y, xv.z * (float)sg.z, xv.w * (float)sg.w};
    wave_hadamard128(v, lane);
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float sc = m > 0.f ? m / 7.f : 1.f;
    int nib[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float t = fminf(fmaxf(v[j] / sc, -8.f), 7.f);
        nib[j] = (int)__float2int_rn(t) & 15;
    }
    // two elements share a byte: nibble j of byte b is element 2b + j
    char2 o = {(char)(nib[0] | (nib[1] << 4)), (char)(nib[2] | (nib[3] << 4))};
    *(char2*)(q4 + base / 2 + 2 * lane) = o;
    if (lane == 0) xsc[blockIdx.x] = sc;
}

// Rotation + quantization reading the input once, with the input transform folded in: one 256-thread block per row,
// each wave keeps its 128-chunks (c = wave, wave + 8, ...; K <= 8 * 128 * CPW) in registers. QP picks the output:
//   0    int8 with one scale per row (the W8A8 row path);
//   127  int8 with one scale per 128 group (`rot_quant_i8_w`'s layout, the W4A8 GEMMs);
//   7    int4 with one scale per 128 group (`rot_quant_i4_w`'s layout, the W4A4 GEMMs).
// The last two are what let a GEMM whose input needs a transform (o_proj, out_proj) leave the int8 weight copy: its
// weights go to 4 bits either way, and the activity format is the quality/speed dial.
template <int CPW, int QP = 0>
__device__ __forceinline__ void rot_quant_row_impl(int8_t* q, float* xs, const float* x, const float* u,
                                                   const int8_t* signs, int K, int mode, const uint16_t* nw,
                                                   float eps) {
    __shared__ float red[8], red2[8];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * K;
    const char4 sg = *(const char4*)(signs + 4 * lane);
    const int G = K / 128;
    f32x4v v[CPW];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int c = wave + 8 * i;
        if (c < G) {
            f32x4v a = *(const f32x4v*)(x + row + c * 128 + 4 * lane);
            if (mode == 1) {
                const f32x4v b = *(const f32x4v*)(u + row + c * 128 + 4 * lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) a[j] = a[j] / (1.f + __expf(-a[j])) * b[j];
            } else if (mode == 2 || mode == 4) {
                const f32x4v b = *(const f32x4v*)(u + row + c * 128 + 4 * lane);
                float sc = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
                for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor(sc, o, 32);
                const float inv = rsqrtf(sc / 128.f + eps);
                const uint2 wb = *(const uint2*)(nw + 4 * lane);
                const float w4[4] = {__uint_as_float(wb.x << 16), __uint_as_float(wb.x & 0xffff0000u),
                                     __uint_as_float(wb.y << 16), __uint_as_float(wb.y & 0xffff0000u)};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    a[j] = a[j] * inv * w4[j] * (mode == 2 ? b[j] : 1.f) / (1.f + __expf(-b[j]));
            } else if (mode == 3) {
                const f32x4v gt = *(const f32x4v*)(u + (size_t)blockIdx.x * 2 * K + (c / 2) * 512 + 256 +
                                                   (c % 2) * 128 + 4 * lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) a[j] *= 1.f / (1.f + __expf(-gt[j]));
            }
            v[i] = a;
            ss += a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
        }
    }
    if (mode == 0 && nw) {
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
        if (lane == 0) red2[wave] = ss;
        __syncthreads();
        ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += red2[i];
        const float inv = rsqrtf(ss / K + eps);
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const int c = wave + 8 * i;
            if (c < G) {
                const uint2 wb = *(const uint2*)(nw + c * 128 + 4 * lane);
                v[i][0] *= inv * (__uint_as_float(wb.x << 16) + 1.f);
                v[i][1] *= inv * (__uint_as_float(wb.x & 0xffff0000u) + 1.f);
                v[i][2] *= inv * (__uint_as_float(wb.y << 16) + 1.f);
                v[i][3] *= inv * (__uint_as_float(wb.y & 0xffff0000u) + 1.f);
            }
        }
    }
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int c = wave + 8 * i;
        if (c < G) {
            float w[4] = {v[i][0] * sg.x, v[i][1] * sg.y, v[i][2] * sg.z, v[i][3] * sg.w};
            wave_hadamard128(w, lane);
            v[i] = (f32x4v){w[0], w[1], w[2], w[3]};
            if (QP == 0)
                m = fmaxf(m, fmaxf(fmaxf(fabsf(w[0]), fabsf(w[1])), fmaxf(fabsf(w[2]), fabsf(w[3]))));
        }
    }
    if (QP != 0) {
        // One scale per 128 group. int4 packs two elements per byte (nibble j of byte b is element 2b + j), which is
        // the packing `rot_quant_i4_w` writes; int8 keeps one byte per element, as `rot_quant_i8_w` does.
        const size_t per = QP == 7 ? 64 : 128;
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const int c = wave + 8 * i;
            if (c >= G) continue;
            float cw = fmaxf(fmaxf(fabsf(v[i][0]), fabsf(v[i][1])), fmaxf(fabsf(v[i][2]), fabsf(v[i][3])));
            for (int o = 16; o > 0; o >>= 1) cw = fmaxf(cw, __shfl_xor(cw, o, 32));
            const float sc = cw > 0.f ? cw / (float)QP : 1.f, inv = 1.f / sc;
            const size_t at = (size_t)blockIdx.x * (G * per) + c * per + (QP == 7 ? 2 * lane : 4 * lane);
            if (QP == 7) {
                int nib[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) nib[j] = (int)__float2int_rn(fminf(fmaxf(v[i][j] * inv, -8.f), 7.f)) & 15;
                const char2 o2 = {(char)(nib[0] | (nib[1] << 4)), (char)(nib[2] | (nib[3] << 4))};
                *(char2*)(q + at) = o2;
            } else {
                char4 o4;
                o4.x = (int8_t)__float2int_rn(fminf(fmaxf(v[i][0] * inv, -127.f), 127.f));
                o4.y = (int8_t)__float2int_rn(fminf(fmaxf(v[i][1] * inv, -127.f), 127.f));
                o4.z = (int8_t)__float2int_rn(fminf(fmaxf(v[i][2] * inv, -127.f), 127.f));
                o4.w = (int8_t)__float2int_rn(fminf(fmaxf(v[i][3] * inv, -127.f), 127.f));
                *(char4*)(q + at) = o4;
            }
            if (lane == 0) xs[(size_t)blockIdx.x * G + c] = sc;
        }
        return;
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (lane == 0) red[wave] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int c = wave + 8 * i;
        if (c < G) {
            char4 o;
            o.x = (int8_t)__float2int_rn(v[i][0] * inv);
            o.y = (int8_t)__float2int_rn(v[i][1] * inv);
            o.z = (int8_t)__float2int_rn(v[i][2] * inv);
            o.w = (int8_t)__float2int_rn(v[i][3] * inv);
            *(char4*)(q + row + c * 128 + 4 * lane) = o;
        }
    }
    if (threadIdx.x == 0) xs[blockIdx.x] = s;
}

// Input transform before the rotation, by mode: 0 x (RMSNorm with weight 1 + nw if nw is given), 1 silu(x) * u,
// 2 per-128 RMSNorm(x) * nw * silu(u) (DeltaNet output gate), 3 x * sigmoid(gate) with the attention gate
// interleaved in u ([T][head][q 256 | gate 256]), 4 as 2 with sigmoid(u).
extern "C" __global__ void __launch_bounds__(256) rot_quant_row(int8_t* q, float* xs, const float* x, const float* u,
                                                                const int8_t* signs, int K, int mode,
                                                                const uint16_t* nw, float eps) {
    if (K <= 8 * 128 * 6)
        rot_quant_row_impl<6>(q, xs, x, u, signs, K, mode, nw, eps);
    else
        rot_quant_row_impl<17>(q, xs, x, u, signs, K, mode, nw, eps);
}

// The same folded input transform, int4 with one scale per 128 group (`MALLEUS_DENSE_W4A4=acts`): 4-bit weights and
// 4-bit activations for the projections whose input needs a transform (o_proj, out_proj).
extern "C" __global__ void __launch_bounds__(256) rot_quant_i4_row(int8_t* q, float* xs, const float* x,
                                                                 const float* u, const int8_t* signs, int K, int mode,
                                                                 const uint16_t* nw, float eps) {
    if (K <= 8 * 128 * 6)
        rot_quant_row_impl<6, 7>(q, xs, x, u, signs, K, mode, nw, eps);
    else
        rot_quant_row_impl<17, 7>(q, xs, x, u, signs, K, mode, nw, eps);
}

// The same, int8 with one scale per 128 group (`MALLEUS_DENSE_W4A4=w4a8`): the weights still go to 4 bits (the MLS8
// copies of these two tensors are what we want to lose) but the activations stay at 8 bits, which is where the
// quality of the int4 variant was lost.
extern "C" __global__ void __launch_bounds__(256) rot_quant_i8_row(int8_t* q, float* xs, const float* x,
                                                                 const float* u, const int8_t* signs, int K, int mode,
                                                                 const uint16_t* nw, float eps) {
    if (K <= 8 * 128 * 6)
        rot_quant_row_impl<6, 127>(q, xs, x, u, signs, K, mode, nw, eps);
    else
        rot_quant_row_impl<17, 127>(q, xs, x, u, signs, K, mode, nw, eps);
}

// Grouped MoE GEMM over (token, slot) pairs sorted by expert: rows [offs[e], offs[e + 1]) of xq use the int8 weights of
// expert e (q [E + 1, N, K], ws [E + 1, N]); tiles lists (e, first row) per BM-row tile, ntiles its length. SEG = 1:
// gate rows then up rows per expert, output silu(gate) * up [rows, N / 2]; SEG = 0: output [rows, N].
template <int BM, int BN2, int WT, int BK2, int NW, int SEG>
__device__ __forceinline__ void gemm_i8_moe(float* out, const int8_t* xq, const float* xs, const int8_t* q,
                                            const float* ws, const int* offs, const int2* tiles, const int* ntiles,
                                            int N, int K) {
    const int nn = (N + BN2 - 1) / BN2, tile = blockIdx.x / nn;
    if (tile >= *ntiles) return;
    const int2 td = tiles[tile];
    const int e = td.x, r0 = offs[e] + td.y, rows = min(BM, offs[e + 1] - r0);
    const int ld = SEG ? N / 2 : N;
    gemm_i8_impl<BM, BN2, WT, BK2, NW, 0, SEG>(out + (size_t)r0 * ld, out, out, ld, 0, xq + (size_t)r0 * K, xs + r0,
                                              q + (size_t)e * N * K, ws + (size_t)e * N, rows, N, K, 0,
                                              blockIdx.x % nn, SEG ? N / 2 : 0);
}

#define GEMM_I8_MOE(NAME, BM, BN2, WT, BK2, NW, SEG)                                                              \
    extern "C" __global__ void __launch_bounds__(NW * 32)                                                         \
        NAME(float* out, const int8_t* xq, const float* xs, const int8_t* q, const float* ws, const int* offs,     \
             const int2* tiles, const int* ntiles, int N, int K) {                                                \
        gemm_i8_moe<BM, BN2, WT, BK2, NW, SEG>(out, xq, xs, q, ws, offs, tiles, ntiles, N, K);                     \
    }

GEMM_I8_MOE(moe_i8_glu, 64, 128, 2, 64, 8, 1)
GEMM_I8_MOE(moe_i8, 64, 128, 2, 64, 8, 0)
