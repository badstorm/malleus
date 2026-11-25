#include <hip/hip_runtime.h>
#include <stdint.h>

// DeepSeek-V4.1's sliding-window attention on the device (issue #128, increment 1). The host graph
// (`crate::deepseek_text`) is the specification: every kernel here reproduces one stage of `attention()` /
// `out_projection()` on a single position, and the per-stage cosine of
// `deepseek_dev::tests::the_device_window_attention_matches_the_host_graph` is what keeps it honest.
//
// The layer this increment covers is a *window* layer (layer 0: no compressor, no indexer). A query reads the last
// `window` raw latents, one shared 512-wide key/value per position, plus a learned per-head logit that takes part in
// the softmax denominator and in nothing else.

__device__ __forceinline__ float dsv_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// RMSNorm with an f32 weight, one 256-thread block per row: DeepSeek's norms are the stored weight itself, not
// Qwen's `1 + w`, so `rmsnorm` (which folds the one and takes bf16 weights) cannot be used here.
extern "C" __global__ void __launch_bounds__(256) dsv_rmsnorm(float* out, const float* x, const float* w, int n,
                                                             float eps) {
    __shared__ float sh[8];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = out + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    ss = dsv_warp_sum(ss);
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float inv = rsqrtf(tot / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = xr[i] * inv * w[i];
}

// Interleaved RoPE on the last `rd` values of the `nh` heads of `q` and, when `nkv` is 1, of the single row of `kv`
// — the same rotation on the query heads and on the shared key the host graph applies before its cache write.
// `inv` is the table of `rd / 2` inverse frequencies (the plain base on a window layer, YaRN's on a compressed
// one), `dir` is +1 on the way in and -1 for the inverse the output projection needs.
extern "C" __global__ void __launch_bounds__(256) dsv_rope(float* q, float* kv, const float* inv, int nh, int hd,
                                                          int rd, int nkv, int pos, int dir) {
    const int half = rd / 2;
    const int lanes = nh + nkv;
    const int total = lanes * half;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int h = i / half, f = i % half;
        float* p = h < nh ? q + (size_t)h * hd + hd - rd + 2 * f : kv + hd - rd + 2 * f;
        const float ang = (float)pos * inv[f];
        const float c = cosf(ang), s = (float)dir * sinf(ang);
        const float x0 = p[0], x1 = p[1];
        p[0] = x0 * c - x1 * s;
        p[1] = x1 * c + x0 * s;
    }
}

// The window itself: one 256-thread block per head. `q` is [heads, hd], `kv` the ring of `cap` rows with the row of
// position p at slot p % cap, `sink` the learned logit per head. The query `pos` reads the last `min(cap, pos + 1)`
// positions, oldest first, and the sink enters the denominator exactly as in `sparse_attn`.
//
// Each thread owns `hd / 256` (at most DSV_V) coordinates of the head: the score of a key is a block reduction, the
// weighted sum accumulates in registers in key order, and the ring is read from L2 (256 KB, shared by the heads).
#define DSV_V 4
extern "C" __global__ void __launch_bounds__(256) dsv_win_attn(float* out, const float* q, const float* kv,
                                                              const float* sink, int heads, int hd, int cap, int pos,
                                                              float scale) {
    __shared__ float sh[8];
    __shared__ float sc[130];
    const int h = blockIdx.x;
    const int n = min(cap, pos + 1);
    const int first = pos - n + 1;
    const float* qh = q + (size_t)h * hd;
    for (int j = 0; j < n; ++j) {
        const float* row = kv + (size_t)((first + j) % cap) * hd;
        float acc = 0.f;
        for (int d = threadIdx.x; d < hd; d += 256) acc += qh[d] * row[d];
        acc = dsv_warp_sum(acc);
        __syncthreads();
        if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.f;
            for (int i = 0; i < 8; ++i) tot += sh[i];
            sc[j] = tot * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int j = 0; j < n; ++j) m = fmaxf(m, sc[j]);
        float sum = __expf(sink[h] - m);
        for (int j = 0; j < n; ++j) {
            const float e = __expf(sc[j] - m);
            sc[j] = e;
            sum += e;
        }
        sc[129] = 1.f / sum;
    }
    __syncthreads();
    float acc[DSV_V];
#pragma unroll
    for (int i = 0; i < DSV_V; ++i) acc[i] = 0.f;
    for (int j = 0; j < n; ++j) {
        const float w = sc[j];
        if (w == 0.f) continue;
        const float* row = kv + (size_t)((first + j) % cap) * hd;
        int k = 0;
        for (int d = threadIdx.x; d < hd; d += 256, ++k) acc[k] += w * row[d];
    }
    float* oh = out + (size_t)h * hd;
    const float inv = sc[129];
    int k = 0;
    for (int d = threadIdx.x; d < hd; d += 256, ++k) oh[d] = acc[k] * inv;
}

// The hyper-connections of one sub-layer, on one position. The streams are `hc_mult` copies of the residual, mixed by
// three coefficient sets: `pre` collapses them into the sub-layer's input, `post` writes the sub-layer's output back
// over all of them, and `comb` (a doubly stochastic matrix, a row softmax followed by Sinkhorn passes) says how the
// old streams carry into the new ones. The reference of all three is `deepseek_text::Hc::mixes`.
//
// The mixes are one projection of the flattened streams: `mix[r] = f[r] . h / sqrt(mean(h^2) + eps)`, one block a
// row, and every block recomputes the statistic over the same 20480 values rather than synchronise for it.
extern "C" __global__ void __launch_bounds__(256) dsv_hc_mixes(float* mix, const float* h, const float* f, int n,
                                                             int hd, float eps) {
    __shared__ float sh[8];
    __shared__ float sh2[8];
    const float* fr = f + (size_t)blockIdx.x * hd;
    float acc = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += 256) {
        const float v = h[i];
        acc += fr[i] * v;
        ss += v * v;
    }
    acc = dsv_warp_sum(acc);
    ss = dsv_warp_sum(ss);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        sh[threadIdx.x / 32] = acc;
        sh2[threadIdx.x / 32] = ss;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.f, ts = 0.f;
        for (int i = 0; i < 8; ++i) {
            ta += sh[i];
            ts += sh2[i];
        }
        mix[blockIdx.x] = ta * rsqrtf(ts / hd + eps);
    }
}

// The coefficients: `pre` and `post` are a sigmoid on their own base and scale, `comb` is a row softmax followed by
// `iters` rounds of row and column normalisation (the first column pass comes right after the softmax). One thread:
// the matrix is `hc x hc` (4x4 here) and twenty rounds of it cost less than a block reduction.
extern "C" __global__ void dsv_hc_coef(float* pre, float* post, float* comb, const float* mix, const float* base,
                                       const float* scale, int hc, float eps, int iters) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ float c[64];
    for (int j = 0; j < hc; ++j) {
        pre[j] = 1.f / (1.f + expf(-(mix[j] * scale[0] + base[j]))) + eps;
        post[j] = 2.f / (1.f + expf(-(mix[j + hc] * scale[1] + base[j + hc])));
    }
    for (int j = 0; j < hc; ++j) {
        float m = -INFINITY;
        for (int k = 0; k < hc; ++k) {
            const int idx = 2 * hc + j * hc + k;
            c[j * hc + k] = mix[idx] * scale[2] + base[idx];
            m = fmaxf(m, c[j * hc + k]);
        }
        float sum = 0.f;
        for (int k = 0; k < hc; ++k) {
            c[j * hc + k] = expf(c[j * hc + k] - m);
            sum += c[j * hc + k];
        }
        for (int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.f;
        for (int j = 0; j < hc; ++j) s += c[j * hc + k];
        for (int j = 0; j < hc; ++j) c[j * hc + k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.f;
            for (int k = 0; k < hc; ++k) s += c[j * hc + k];
            for (int k = 0; k < hc; ++k) c[j * hc + k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.f;
            for (int j = 0; j < hc; ++j) s += c[j * hc + k];
            for (int j = 0; j < hc; ++j) c[j * hc + k] /= s + eps;
        }
    }
    for (int j = 0; j < hc * hc; ++j) comb[j] = c[j];
}

// Collapse the streams: `x[d] = sum_c pre[c] * h[c][d]`.
extern "C" __global__ void dsv_hc_pre(float* x, const float* h, const float* pre, int hc, int hidden) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    float v = 0.f;
    for (int c = 0; c < hc; ++c) v += pre[c] * h[(size_t)c * hidden + d];
    x[d] = v;
}

// Write the sub-layer output back over the streams, mixing the old ones in:
// `out[e][d] = post[e] * x[d] + sum_c comb[c][e] * res[c][d]`.
extern "C" __global__ void dsv_hc_post(float* out, const float* x, const float* res, const float* post,
                                       const float* comb, int hc, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hc * hidden) return;
    const int e = i / hidden, d = i % hidden;
    float v = post[e] * x[d];
    for (int c = 0; c < hc; ++c) v += comb[c * hc + e] * res[(size_t)c * hidden + d];
    out[i] = v;
}
