// MLA, the attention of Ling-3.0-flash (7 of its 42 layers, plus the MTP). The cache does not hold K and V: it
// holds the compressed latent `cn` (kv_lora, 512) and the 64 rotated dims, 1152 bytes per token and layer
// instead of the 20 KiB the HF code spends on K and V (docs/model-ling3-flash.md, § MLA).
//
// Decode absorbs `kv_b_proj` into the query: the score of key j is
//     q_nope · (W_KB cn_j) + q_rot · k_rot_j = (W_KBᵀ q_nope) · cn_j + q_rot · k_rot_j
// so the 512-wide matrix is read once per token and head, not once per key, and the output is the weighted sum
// of the latents, multiplied by W_VB at the end. The two block matvecs are the whole cost of absorbing:
// 2 × 32 × 128 × 512 = 4,2 MFLOP per token and layer, against the 576 × 32 MACs per key of the dot itself.
//
// `wkb` is kv_b_proj dequantized to fp32 at load: 4,2 M parameters per layer, 16,8 MB, 118 MB for the seven
// MLA layers of Ling. Keeping it MLS4 would mean repeating the Hadamard dequantization inside this kernel for
// both matvecs, and the absorbed form is already the unusual part of the path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef unsigned short bf16_t;

__device__ __forceinline__ float h2f(__half h) {
    return __half2float(h);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// Writes the cache row of each token of the step: RMSNorm(kv_a[0:kv]) as fp16 latent, and the rotated dims
// (already rotated by the caller, with the same interleaved pairs as the query) as fp16.
extern "C" __global__ void __launch_bounds__(256) mla_cache_write(__half* ckv, __half* krot, const float* kv_a,
                                                                  const float* norm_w, float eps, int len, int t,
                                                                  int kv, int rd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int ti = i / (kv + rd);
    if (ti >= t) return;
    const int d = i % (kv + rd);
    const float* row = kv_a + (size_t)ti * (kv + rd);
    if (d < kv) {
        // The latent norm is over the kv dims only: one block covers a row when kv <= blockDim, which 512 does.
        float ss = 0.f;
        for (int c = threadIdx.x % 32; c < kv; c += 32) ss += row[c] * row[c];
        __syncwarp();
        const float inv = rsqrtf(warp_sum(ss) / (float)kv + eps);
        ckv[(size_t)(len + ti) * kv + d] = __float2half(row[d] * inv * norm_w[d]);
    } else {
        krot[(size_t)(len + ti) * rd + (d - kv)] = __float2half(row[d]);
    }
}

// Absorbed MLA decode: one warp per (token, head). Lane `l` owns the latent dims [l*16, l*16+16) of the
// absorbed query, of the running weighted sum and of each cached key, so the only cross-lane step per key is
// one warp reduction for the score.
extern "C" __global__ void __launch_bounds__(256) mla_decode(float* out, const float* q, const float* wkb,
                                                             const __half* ckv, const __half* krot, int len, int t,
                                                             int nh, int nope, int kv, int rd, float scale,
                                                             float eps) {
    __shared__ float lat[8][512];
    __shared__ float qr[8][64];
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int w = blockIdx.x * 8 + wl;
    const int ti = w / nh, h = w % nh;
    if (ti >= t) return;
    const int vis = len + ti + 1;
    const int dl = kv / 32;
    const int d0 = lane * dl;
    const float* qh = q + (size_t)ti * nh * (nope + rd) + h * (nope + rd);
    const float* wk = wkb + (size_t)h * (nope + nope) * kv;

    // The rotated part of the query goes through LDS: every lane needs all 64 values for every key.
    if (lane < 2) {
        for (int r = lane * 32; r < (lane + 1) * 32 && r < rd; ++r) qr[wl][r] = qh[nope + r];
    }
    __syncwarp();

    float qt[16], acc[16];
    for (int i = 0; i < dl; ++i) {
        qt[i] = 0.f;
        acc[i] = 0.f;
    }
    for (int n = 0; n < nope; ++n) {
        const float qn = qh[n];
        const float* row = wk + (size_t)n * kv + d0;
#pragma unroll
        for (int i = 0; i < 16; ++i) qt[i] += qn * row[i];
    }

    float m = -INFINITY, l = 0.f;
    for (int j = 0; j < vis; ++j) {
        const __half* ck = ckv + (size_t)j * kv + d0;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < dl; ++i) s += qt[i] * h2f(ck[i]);
        for (int r = lane * 2; r < (lane + 1) * 2 && r < rd; ++r) s += qr[wl][r] * h2f(krot[(size_t)j * rd + r]);
        s = warp_sum(s) * scale;
        const float mn = s > m ? s : m;
        const float f = expf(m - mn);
        const float p = expf(s - mn);
        l = l * f + p;
        m = mn;
#pragma unroll
        for (int i = 0; i < dl; ++i) acc[i] = acc[i] * f + p * h2f(ck[i]);
    }
#pragma unroll
    for (int i = 0; i < dl; ++i) lat[wl][d0 + i] = acc[i] / l;
    __syncwarp();

    for (int mv = lane; mv < nope; mv += 32) {
        const float* row = wk + (size_t)(nope + mv) * kv;
        float o = 0.f;
        for (int d = 0; d < kv; ++d) o += row[d] * lat[wl][d];
        out[(size_t)ti * nh * nope + h * nope + mv] = o;
    }
}

// Prefill/decode split: the absorbed query `qt` is computed once per (token, head) from the 128 nope dims and
// `kv_b_proj`, so the key loop does not re-read the 4.2 M-parameter matrix for every token. One warp per
// (token, head), lane l owns latent dims [l*16, l*16+16).
extern "C" __global__ void __launch_bounds__(256) mla_absorb(float* qt, const float* q, const float* wkb, int t,
                                                             int nh, int nope, int kv, int rd) {
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int w = blockIdx.x * 8 + wl;
    if (w >= t * nh) return;
    const int ti = w / nh, h = w % nh;
    const int d0 = lane * 16;
    const float* qh = q + (size_t)ti * nh * (nope + rd) + (size_t)h * (nope + rd);
    const float* wk = wkb + (size_t)h * 2 * nope * kv;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int n = 0; n < nope; ++n) {
        const float qn = qh[n];
        const float* row = wk + (size_t)n * kv + d0;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += qn * row[i];
    }
    float* dst = qt + (size_t)w * kv + d0;
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = acc[i];
}

// One split of the MLA decode: warp w is (token, head, split), 256 keys per split, lane owns 16 latent dims.
// Writes (max, sum, unnormalized latent) to part; mla_combine merges the splits and applies kv_b_proj's V half.
extern "C" __global__ void __launch_bounds__(256) mla_split(float* part, const float* qt, const float* q,
                                                            const __half* ckv, const __half* krot, int len, int t,
                                                            int nh, int nope, int kv, int rd, float scale,
                                                            int nsplit) {
    __shared__ float qr[8][64];
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int w = blockIdx.x * 8 + wl;
    const int total = t * nh * nsplit;
    if (w >= total) return;
    const int split = w % nsplit;
    const int th = w / nsplit;
    const int ti = th / nh, h = th % nh;
    const int d0 = lane * 16;
    const float* qh = q + (size_t)ti * nh * (nope + rd) + (size_t)h * (nope + rd);
    float qtv[16];
    const float* qts = qt + (size_t)th * kv + d0;
#pragma unroll
    for (int i = 0; i < 16; ++i) qtv[i] = qts[i];
    if (lane < 2) {
        for (int r = lane * 32; r < (lane + 1) * 32 && r < rd; ++r) qr[wl][r] = qh[nope + r];
    }
    __syncwarp();
    const int nkeys = len + ti + 1;
    const int k0 = split * 256, k_end = min(nkeys, k0 + 256);
    float m = -INFINITY, l = 0.f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int j = k0; j < k_end; ++j) {
        const __half* ck = ckv + (size_t)j * kv + d0;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s += qtv[i] * h2f(ck[i]);
        for (int r = lane * 2; r < (lane + 1) * 2 && r < rd; ++r) s += qr[wl][r] * h2f(krot[(size_t)j * rd + r]);
        s = warp_sum(s) * scale;
        const float mn = fmaxf(m, s);
        const float f = expf(m - mn);
        const float p = expf(s - mn);
        l = l * f + p;
        m = mn;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = acc[i] * f + p * h2f(ck[i]);
    }
    float* dst = part + (size_t)w * (kv + 2);
    if (lane == 0) {
        dst[0] = m;
        dst[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[2 + d0 + i] = acc[i];
}

// Merge the splits of one (token, head) and apply the V half of kv_b_proj: out[m] = Σ_d W_VB[m, d] · lat[d].
extern "C" __global__ void __launch_bounds__(256) mla_combine(float* out, const float* part, const float* wkb,
                                                              int nh, int nope, int kv, int nsplit) {
    __shared__ float lat[512];
    const int w = blockIdx.x;
    const int h = w % nh;
    const float* src = part + (size_t)w * nsplit * (kv + 2);
    float M = -INFINITY;
    for (int i = 0; i < nsplit; ++i) M = fmaxf(M, src[(size_t)i * (kv + 2)]);
    float L = 0.f;
    if (M != -INFINITY) {
        for (int i = 0; i < nsplit; ++i) {
            const float mi = src[(size_t)i * (kv + 2)];
            if (mi == -INFINITY) continue;
            L += src[(size_t)i * (kv + 2) + 1] * expf(mi - M);
        }
    }
    const int d = threadIdx.x;
    float a = 0.f, b = 0.f;
    if (M != -INFINITY) {
        for (int i = 0; i < nsplit; ++i) {
            const float mi = src[(size_t)i * (kv + 2)];
            if (mi == -INFINITY) continue;
            const float f = expf(mi - M);
            a += src[(size_t)i * (kv + 2) + 2 + d] * f;
            b += src[(size_t)i * (kv + 2) + 2 + d + 256] * f;
        }
    }
    lat[d] = M == -INFINITY ? 0.f : a / L;
    lat[d + 256] = M == -INFINITY ? 0.f : b / L;
    __syncthreads();
    if (threadIdx.x < nope) {
        const float* row = wkb + (size_t)h * 2 * nope * kv + (size_t)(nope + threadIdx.x) * kv;
        float sum = 0.f;
        for (int c = 0; c < kv; ++c) sum += row[c] * lat[c];
        out[(size_t)w * nope + threadIdx.x] = sum;
    }
}

// Prefill absorb, tiled over tokens: one block per (head, 64 latent dims), the W_KB tile (128 x 64 floats,
// 32 KB) loaded once in LDS and reused for every token. `qt[token, head, d] = Σ_n q[token, head, n] · W_KB[h, n, d]`.
extern "C" __global__ void __launch_bounds__(256) mla_absorb_t(float* qt, const float* q, const float* wkb, int t,
                                                               int nh, int nope, int kv, int rd) {
    __shared__ float wv[128][64];
    __shared__ float qv[4][128];
    const int h = blockIdx.x / 8, tile = blockIdx.x % 8, d0 = tile * 64;
    const float* wk = wkb + (size_t)h * 2 * nope * kv;
    for (int i = threadIdx.x; i < nope * 64; i += 256) {
        const int n = i / 64, d = i % 64;
        wv[n][d] = wk[(size_t)n * kv + d0 + d];
    }
    __syncthreads();
    for (int t0 = 0; t0 < t; t0 += 4) {
        const int tn = min(4, t - t0);
        for (int i = threadIdx.x; i < tn * nope; i += 256) {
            const int ti = i / nope, n = i % nope;
            qv[ti][n] = q[(size_t)(t0 + ti) * nh * (nope + rd) + (size_t)h * (nope + rd) + n];
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            const int ti = threadIdx.x / 64, d = threadIdx.x % 64;
            if (ti < tn) {
                float acc = 0.f;
                for (int n = 0; n < nope; ++n) acc += qv[ti][n] * wv[n][d];
                qt[(size_t)((t0 + ti) * nh + h) * kv + d0 + d] = acc;
            }
        }
        __syncthreads();
    }
}

// Prefill MLA: one warp per (token, head), absorbed query `qt` already computed, 16 latent dims per lane.
// Writes the normalized weighted latent [token, head, 512]; mla_project applies W_VB once per head.
extern "C" __global__ void __launch_bounds__(256) mla_prefill(float* lat, const float* qt, const float* q,
                                                              const __half* ckv, const __half* krot, int len, int t,
                                                              int nh, int nope, int kv, int rd, float scale) {
    __shared__ float qr[8][64];
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int w = blockIdx.x * 8 + wl;
    if (w >= t * nh) return;
    const int ti = w / nh, h = w % nh;
    const int d0 = lane * 16;
    const float* qh = q + (size_t)ti * nh * (nope + rd) + (size_t)h * (nope + rd);
    float qtv[16];
    const float* qts = qt + (size_t)w * kv + d0;
#pragma unroll
    for (int i = 0; i < 16; ++i) qtv[i] = qts[i];
    if (lane < 2) {
        for (int r = lane * 32; r < (lane + 1) * 32 && r < rd; ++r) qr[wl][r] = qh[nope + r];
    }
    __syncwarp();
    const int vis = len + ti + 1;
    float m = -INFINITY, l = 0.f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int j = 0; j < vis; ++j) {
        const __half* ck = ckv + (size_t)j * kv + d0;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s += qtv[i] * h2f(ck[i]);
        for (int r = lane * 2; r < (lane + 1) * 2 && r < rd; ++r) s += qr[wl][r] * h2f(krot[(size_t)j * rd + r]);
        s = warp_sum(s) * scale;
        const float mn = fmaxf(m, s);
        const float f = expf(m - mn);
        const float p = expf(s - mn);
        l = l * f + p;
        m = mn;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = acc[i] * f + p * h2f(ck[i]);
    }
    float* dst = lat + (size_t)w * kv + d0;
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = l > 0.f ? acc[i] / l : 0.f;
}

// Apply the V half of kv_b_proj to the prefill latent: out[token, head, m] = Σ_d W_VB[h, m, d] · lat[token, head, d].
// One block per (head, 16 output dims), W_VB tile in LDS once, latent in tiles of 8 tokens.
extern "C" __global__ void __launch_bounds__(256) mla_project(float* out, const float* lat, const float* wkb, int t,
                                                              int nh, int nope, int kv) {
    __shared__ float wv[16][512];
    __shared__ float lv[8][512];
    const int h = blockIdx.x / (nope / 16), d0 = (blockIdx.x % (nope / 16)) * 16;
    const float* wk = wkb + (size_t)h * 2 * nope * kv + (size_t)nope * kv;
    for (int i = threadIdx.x; i < 16 * kv; i += 256) {
        const int d = i / kv, c = i % kv;
        wv[d][c] = wk[(size_t)(d0 + d) * kv + c];
    }
    __syncthreads();
    for (int t0 = 0; t0 < t; t0 += 8) {
        const int tn = min(8, t - t0);
        for (int i = threadIdx.x; i < tn * kv; i += 256) {
            const int ti = i / kv, c = i % kv;
            lv[ti][c] = lat[(size_t)((t0 + ti) * nh + h) * kv + c];
        }
        __syncthreads();
        const int o = threadIdx.x;
        if (o < 128) {
            const int ti = o / 16, d = o % 16;
            if (ti < tn) {
                float acc = 0.f;
                for (int c = 0; c < kv; ++c) acc += wv[d][c] * lv[ti][c];
                out[(size_t)(t0 + ti) * nh * nope + (size_t)h * nope + d0 + d] = acc;
            }
        }
        __syncthreads();
    }
}

// Ling RoPE: interleaved pairs (2i, 2i+1) of the last `rd` dims, theta 6e6, 32 frequencies. The same rotation
// goes on the rope part of every query head and on the rope part of the kv row, which is why one kernel does
// both (and why it runs before `mla_cache_write`: the cache holds the rotated k_rot):
// `q` is [t, nh * (nope + rd)] and `kv` is [t, kvl + rd], rotated in place. `apply_rotary_pos_emb_interleave`
// de-interleaves into two halves and calls rotate_half; the same permutation on q and k_rot leaves the dot product
// unchanged, so rotating the pairs in place is equivalent and simpler (docs/model-ling3-flash.md, § MLA).
extern "C" __global__ void __launch_bounds__(256) rope_ilv(float* q, float* kv, const int* pos, int t, int nh,
                                                           int nope, int kvl, int rd, float theta) {
    const int half = rd / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * (nh + 1) * half) return;
    const int ti = i / ((nh + 1) * half);
    const int r = i % ((nh + 1) * half);
    const int h = r / half, f = r % half;
    float* p = h < nh ? q + (size_t)ti * nh * (nope + rd) + h * (nope + rd) + nope + 2 * f
                      : kv + (size_t)ti * (kvl + rd) + kvl + 2 * f;
    const float inv = 1.0f / powf(theta, (float)(2 * f) / (float)rd);
    const float ang = (float)pos[ti * 3] * inv; // the engine keeps three axes per token; Ling uses the first
    const float c = cosf(ang), s = sinf(ang);
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x1 * c + x0 * s;
}

// Head-wise output gate: o[ti][h][*] *= sigmoid(g[ti][h]), one value per head (Ling `g_proj` is [heads, hidden]).
extern "C" __global__ void mla_gate(float* o, const float* g, int n, int v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[i] *= 1.f / (1.f + expf(-g[i / v]));
}
