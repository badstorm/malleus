// MLS3 GEMV/GEMM: the 3-bit weight format of docs/model-ling3-flash.md, used for the routed gate/up of
// Ling-3.0-flash. Mirrors mls4_core in quant.hip so the two can be compared directly.
//
// q: [N, K * 3 / 8] bytes. Codes are LSB-first, 8 per 3 bytes: code i of a group sits at bit 3i, so weight k has
// its code at bit 3k of the row. A code is the level -4..3 stored as value + 4.
// sc: [N, K / 128] fp16, one scale per group of 128 weights, exactly like MLS4.
// xr: rotated activations [T, K] (Hadamard on 128-wide blocks, same signs as the converter).
//
// Alignment: K is a multiple of 128, so a row is K / 8 * 3 bytes, a multiple of 48; a lane covers 32 consecutive
// weights, i.e. 12 bytes at offset lane * 12. Every access is therefore 4-byte aligned and three dwords read
// exactly the lane's 96 bits, never crossing into the next lane or past the row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float mls3_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

/// Unpacks the 32 codes a lane owns, starting at weight `k` of row `n`, already multiplied by the group scale.
__device__ __forceinline__ void mls3_unpack(const uint8_t* q, const __half* sc, size_t row_bytes, int G, int n, int k,
                                            float out[32]) {
    const uint32_t* p = (const uint32_t*)(q + (size_t)n * row_bytes + (size_t)k * 3 / 8);
    const uint32_t w[3] = {p[0], p[1], p[2]};
    const float s = __half2float(sc[(size_t)n * G + k / 128]);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int bit = i * 3, idx = bit >> 5, sh = bit & 31;
        // The code can straddle two dwords (sh 30 and 31); the last code of the lane ends at bit 95, so the high
        // operand is only needed from w[1] and w[2] and 0 is safe past the end.
        const uint32_t hi = idx < 2 ? w[idx + 1] : 0u;
        out[i] = (float)((int)(__funnelshift_r(w[idx], hi, sh) & 7u) - 4) * s;
    }
}

template <int R, int TT>
__device__ __forceinline__ void mls3_core(float* out, const float* xr, const uint8_t* q, const __half* sc, int T,
                                          int N, int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * R;
    const int t0 = blockIdx.y * TT;
    if (n0 >= N) return;
    const int G = K / 128;
    const size_t row_bytes = (size_t)K * 3 / 8;
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
            mls3_unpack(q, sc, row_bytes, G, n, k, wf[r]);
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
            const float s = mls3_warp_sum(acc[a][r]);
            if (lane == 0 && t0 + a < T && n0 + r < N) {
                float* o = out + (size_t)(t0 + a) * N + n0 + r;
                *o = accumulate ? *o + s : s;
            }
        }
}

/// One token: grid (ceil(N / 32), 1, 1), block 256.
extern "C" __global__ void gemv_mls3(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N,
                                     int K, int accumulate) {
    mls3_core<4, 1>(out, xr, q, sc, T, N, K, accumulate);
}

/// Several tokens per block: grid (ceil(N / 16), ceil(T / 8), 1), block 256. Used when there is no MLS8 copy.
extern "C" __global__ void gemm_mls3(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N,
                                     int K, int accumulate) {
    mls3_core<2, 8>(out, xr, q, sc, T, N, K, accumulate);
}

// The MoE of Ling-3.0-flash: its routed experts are the only place MLS3 is used, and the expert kernels of
// quant.hip decode MLS4 (docs/model-ling3-flash.md, § MoE). They live here because `mls3_core` and `mls3_unpack`
// are device functions of this module: a .hip compiles to its own HSACO, and kernels do not link across modules.
//
// `xr` must already be rotated: unlike gemv4_hl, which folds the Hadamard into its staging, the MLS3 core takes
// rotated activations, so `moe` runs `hadamard128` once per layer for the gate/up pass. The down pass needs
// nothing: `moe_act` already rotates the activation it writes.

// gate|up rows of the expert of each (token, slot) pair: q [E + 1, N, K * 3 / 8], one pair per blockIdx.y.
// The body is `mls3_core`'s, not a call to it: the core takes its token index from blockIdx.y and skips the
// rows past T, which for pairs (p >= T) would leave every output at zero.
extern "C" __global__ void __launch_bounds__(256) moe_gate_up3(float* out, const float* xr, const uint8_t* q,
                                                               const __half* sc, const int* ids, int slots, int N,
                                                               int K) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int p = blockIdx.y, n0 = (blockIdx.x * 8 + warp) * 4;
    if (n0 >= N) return;
    const size_t e = ids[p];
    const int G = K / 128;
    const size_t row_bytes = (size_t)K * 3 / 8;
    const uint8_t* qe = q + e * N * row_bytes;
    const __half* se = sc + e * N * G;
    const float* x = xr + (size_t)(p / slots) * K;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = lane * 32; k < K; k += 1024) {
        float wf[4][32];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            mls3_unpack(qe, se, row_bytes, G, n0 + r < N ? n0 + r : N - 1, k, wf[r]);
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
        if (n0 + r < N) o[n0 + r] = mls3_warp_sum(acc[r]);
    }
}

// out[t] = sum over the slots of w * (act of the pair · down rows of its expert), the MLS3 form of `moe_down`.
extern "C" __global__ void __launch_bounds__(256) moe_down3(float* out, const float* a, const uint8_t* q,
                                                            const __half* sc, const int* ids, const float* w,
                                                            int slots, int N, int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * 8 + warp) * 4, t = blockIdx.y;
    if (n0 >= N) return;
    const int G = K / 128;
    const size_t row_bytes = (size_t)K * 3 / 8;
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
                mls3_unpack(qe, se, row_bytes, G, n0 + r < N ? n0 + r : N - 1, k, wf[r]);
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
        for (int r = 0; r < 4; ++r) acc[r] += ws * mls3_warp_sum(part[r]);
    }
    if (lane != 0) return;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (n0 + r >= N) continue;
        float* o = out + (size_t)t * N + n0 + r;
        *o = accumulate ? *o + acc[r] : acc[r];
    }
}
