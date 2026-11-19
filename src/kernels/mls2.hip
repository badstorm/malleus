// MLS2 GEMV/GEMM: 2-bit weights, 4 codes per byte. Mirrors mls3.hip so the widths can be compared directly.
//
// q: [N, K / 4] bytes. Codes are LSB-first, 4 per byte: code i of a group sits at bits 2i..2i+2, so weight k has its
// code at bit 2k of the row. The four levels are -1.5, -0.5, 0.5, 1.5 in code order (the e2m1 shape, four levels;
// see malleus_gguf::quant::Int2Grid). A lane keeps the integer levels -3, -1, 1, 3 and folds the 0.5 into the group
// scale, which is exactly the same product for the GEMV.
// sc: [N, K / 128] fp16, one scale per group of 128 weights, exactly like MLS4/MLS3.
// xr: rotated activations [T, K] (Hadamard on 128-wide blocks, same signs as the converter).
//
// Alignment: K is a multiple of 128, so a row is K / 4 bytes, a multiple of 32; a lane covers 32 consecutive weights,
// i.e. 8 bytes at offset lane * 8. Every access is therefore 8-byte aligned and two dwords read exactly the lane's
// 64 bits, never crossing into the next lane or past the row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float mls2_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

/// The four levels of the e2m1 subset in code order, as integers whose product with `scale / 2` is the level.
__device__ __forceinline__ float mls2_level(uint32_t code) {
    return (float)((int)code * 2 - 3) * 0.5f;
}

/// Unpacks the 32 codes a lane owns, starting at weight `k` of row `n`, already multiplied by the group scale.
__device__ __forceinline__ void mls2_unpack(const uint8_t* q, const __half* sc, size_t row_bytes, int G, int n, int k,
                                            float out[32]) {
    const uint2 v = *(const uint2*)(q + (size_t)n * row_bytes + (size_t)k / 4);
    const uint32_t w[2] = {v.x, v.y};
    const float s = __half2float(sc[(size_t)n * G + k / 128]);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const uint32_t c = (w[i >> 4] >> ((i & 15) * 2)) & 3u;
        out[i] = mls2_level(c) * s;
    }
}

template <int R, int TT>
__device__ __forceinline__ void mls2_core(float* out, const float* xr, const uint8_t* q, const __half* sc, int T,
                                          int N, int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * R;
    const int t0 = blockIdx.y * TT;
    if (n0 >= N) return;
    const int G = K / 128;
    const size_t row_bytes = (size_t)K / 4;
    float acc[TT][R];
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
    for (int k = lane * 32; k < K; k += 1024) {
        float wf[R][32];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r < N ? n0 + r : N - 1;
            mls2_unpack(q, sc, row_bytes, G, n, k, wf[r]);
        }
#pragma unroll
        for (int a = 0; a < TT; ++a) {
            if (t0 + a >= T) break;
            const float4* xp = (const float4*)(xr + (size_t)(t0 + a) * K + k);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float4 xv = xp[c];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[a][r] += xv.x * wf[r][c * 4] + xv.y * wf[r][c * 4 + 1] + xv.z * wf[r][c * 4 + 2] +
                                 xv.w * wf[r][c * 4 + 3];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < TT; ++a)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = mls2_warp_sum(acc[a][r]);
            if (lane == 0 && t0 + a < T && n0 + r < N) {
                float* o = out + (size_t)(t0 + a) * N + n0 + r;
                *o = accumulate ? *o + s : s;
            }
        }
}

/// One token: grid (ceil(N / 32), 1, 1), block 256.
extern "C" __global__ void gemv_mls2(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N,
                                     int K, int accumulate) {
    mls2_core<4, 1>(out, xr, q, sc, T, N, K, accumulate);
}

/// Several tokens per block: grid (ceil(N / 16), ceil(T / 8), 1), block 256.
extern "C" __global__ void gemm_mls2(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N,
                                     int K, int accumulate) {
    mls2_core<2, 8>(out, xr, q, sc, T, N, K, accumulate);
}

// The MoE kernels are mls3.hip's with the 2-bit lane read: `xr` must already be rotated, and `moe_act` already
// rotates the activation it writes for the down projection, so only the gate/up call needs a separate hadamard.

// gate|up rows of the expert of each (token, slot) pair: q [E + 1, N, K / 4], one pair per blockIdx.y.
extern "C" __global__ void __launch_bounds__(256) moe_gate_up2(float* out, const float* xr, const uint8_t* q,
                                                               const __half* sc, const int* ids, int slots, int N,
                                                               int K) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int p = blockIdx.y, n0 = (blockIdx.x * 8 + warp) * 4;
    if (n0 >= N) return;
    const size_t e = ids[p];
    const int G = K / 128;
    const size_t row_bytes = (size_t)K / 4;
    const uint8_t* qe = q + e * N * row_bytes;
    const __half* se = sc + e * N * G;
    const float* x = xr + (size_t)(p / slots) * K;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = lane * 32; k < K; k += 1024) {
        float wf[4][32];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            mls2_unpack(qe, se, row_bytes, G, n0 + r < N ? n0 + r : N - 1, k, wf[r]);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float4 xv = *(const float4*)(x + k + 4 * c);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r] += xv.x * wf[r][c * 4] + xv.y * wf[r][c * 4 + 1] + xv.z * wf[r][c * 4 + 2] +
                          xv.w * wf[r][c * 4 + 3];
            }
        }
    }
    float* o = out + (size_t)p * N;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (n0 + r < N) o[n0 + r] = mls2_warp_sum(acc[r]);
    }
}

// out[t] = sum over the slots of w * (act of the pair . down rows of its expert), the MLS2 form of `moe_down`.
extern "C" __global__ void __launch_bounds__(256) moe_down2(float* out, const float* a, const uint8_t* q,
                                                            const __half* sc, const int* ids, const float* w,
                                                            int slots, int N, int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * 4, t = blockIdx.y;
    if (n0 >= N) return;
    const int G = K / 128;
    const size_t row_bytes = (size_t)K / 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < slots; ++s) {
        const size_t e = ids[t * slots + s];
        const float ws = w[t * slots + s];
        const uint8_t* qe = q + e * N * row_bytes;
        const __half* se = sc + e * N * G;
        const float* x = a + (size_t)(t * slots + s) * K;
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k = lane * 32; k < K; k += 1024) {
            float wf[4][32];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                mls2_unpack(qe, se, row_bytes, G, n0 + r < N ? n0 + r : N - 1, k, wf[r]);
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float4 xv = *(const float4*)(x + k + 4 * c);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    part[r] += xv.x * wf[r][c * 4] + xv.y * wf[r][c * 4 + 1] + xv.z * wf[r][c * 4 + 2] +
                               xv.w * wf[r][c * 4 + 3];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[r] += ws * mls2_warp_sum(part[r]);
    }
    if (lane != 0) return;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (n0 + r >= N) continue;
        float* o = out + (size_t)t * N + n0 + r;
        *o = accumulate ? *o + acc[r] : acc[r];
    }
}
