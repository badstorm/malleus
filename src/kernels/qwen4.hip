//
// Qwen3.8-Flash-Next (qwen4_exp): gated hyper-connections (C residual streams of H), the PLE n-gram layer.
// Layouts: stream state [T, C * H] (stream outer); RMSNorm weights are (1 + w), one weight per element of C * H.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint16_t bf16_t;

__device__ __forceinline__ float bf2f(bf16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

__device__ __forceinline__ float block_sum256(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) r += sh[i];
    return r;
}

// Per-stream RMSNorm: out[r] = x[r] / rms(x[r]) * (1 + w[(r % C) * H ..]), r = t * C + s; one 256-thread block per row.
extern "C" __global__ void __launch_bounds__(256) hc_norm(float* out, const float* x, const bf16_t* w, int C, int H,
                                                         float eps) {
    __shared__ float sh[8];
    const size_t r = blockIdx.x;
    const float* xr = x + r * H;
    const bf16_t* wr = w + (r % C) * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum256(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += 256) out[r * H + i] = xr[i] * inv * (bf2f(wr[i]) + 1.f);
}

// Low-rank gate of the mix and the injection weights from d = mix_down(normed) [T, RP] (the sum of S split-K
// partials, S * T * RP): a[t, j] = silu(d[t, j] / C) for j < R, 0 up to RP; inj[t, s] = 2 sigmoid(d[t, R + s] / C)
// when inj is given.
extern "C" __global__ void hc_gate(float* a, float* inj, const float* d, int T, int R, int RP, int C, int S) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * RP) return;
    const int t = i / RP, j = i % RP;
    float dv = 0.f;
    for (int p = 0; p < S; ++p) dv += d[(size_t)p * T * RP + i];
    const float v = dv / (float)C;
    a[i] = j < R ? silu(v) : 0.f;
    if (inj && j >= R && j < R + C) inj[t * C + j - R] = 2.f * sigmoidf(v);
}

// mixed[t, j] = mean_s sigmoid(u[t, s, j]) * xn[t, s, j]; u columns are j * C + s when `inter` (reordered mix_up).
extern "C" __global__ void hc_mix(float* mixed, const float* u, const float* xn, int T, int C, int H, int inter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int t = i / H, j = i % H;
    const size_t b = (size_t)t * C * H + j;
    float acc = 0.f;
    for (int s = 0; s < C; ++s) {
        const float uv = inter ? u[(size_t)t * C * H + (size_t)j * C + s] : u[b + (size_t)s * H];
        acc += sigmoidf(uv) * xn[b + (size_t)s * H];
    }
    mixed[i] = acc / (float)C;
}

// Injection and the per-stream RMSNorm in one pass over x: x += o * inj, then out = norm(x) * (1 + w). The
// generic hyper-connection path otherwise launches hc_inject and hc_norm back to back.
extern "C" __global__ void __launch_bounds__(256) hc_inject_norm(float* out, float* x, const float* o, const float* inj,
                                                               const bf16_t* w, int C, int H, float eps) {
    __shared__ float sh[8];
    const size_t r = blockIdx.x;
    const int s = r % C;
    const size_t t = r / C;
    float* xr = x + r * H;
    const bf16_t* wr = w + (size_t)s * H;
    const float* orow = o + t * H;
    const float f = inj[r];
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += 256) {
        const float v = xr[i] + orow[i] * f;
        xr[i] = v;
        ss += v * v;
    }
    float vv = warp_sum(ss);
    const int wv = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[wv] = vv;
    __syncthreads();
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += sh[i];
    const float inv = rsqrtf(sum / H + eps);
    for (int i = threadIdx.x; i < H; i += 256)
        out[r * H + i] = xr[i] * inv * (bf2f(wr[i]) + 1.f);
}

// x[t, s, j] += o[t, j] * inj[t, s].
extern "C" __global__ void hc_inject(float* x, const float* o, const float* inj, int T, int C, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * C * H) return;
    const int j = i % H;
    const int ts = i / H, t = ts / C;
    x[i] += o[(size_t)t * H + j] * inj[ts];
}

// x[t, s, j] = e[t, j]: the embedding copied to every stream.
extern "C" __global__ void hc_expand(float* x, const float* e, int T, int C, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * C * H) return;
    const int j = i % H, t = i / ((size_t)C * H);
    x[i] = e[(size_t)t * H + j];
}

// PLE gate of every stream: g = <RMSNorm(key_s) (1 + wk_s), RMSNorm(x_s) (1 + wq_s)> / sqrt(H), signed square root,
// gv[t, s] = sigmoid(g) * value[t]. One 256-thread block per (t, s).
extern "C" __global__ void __launch_bounds__(256) ple_gate(float* gv, const float* key, const float* x, const float* value,
                                                          const bf16_t* wk, const bf16_t* wq, int C, int H, float eps) {
    __shared__ float sh[8];
    const size_t r = blockIdx.x;
    const int s = r % C;
    const size_t t = r / C;
    const float* kr = key + r * H;
    const float* xr = x + r * H;
    float sk = 0.f, sx = 0.f;
    for (int i = threadIdx.x; i < H; i += 256) sk += kr[i] * kr[i], sx += xr[i] * xr[i];
    const float ik = rsqrtf(block_sum256(sk, sh) / H + eps);
    const float ix = rsqrtf(block_sum256(sx, sh) / H + eps);
    float dot = 0.f;
    for (int i = threadIdx.x; i < H; i += 256)
        dot += kr[i] * ik * (bf2f(wk[s * H + i]) + 1.f) * xr[i] * ix * (bf2f(wq[s * H + i]) + 1.f);
    float g = block_sum256(dot, sh) * rsqrtf((float)H);
    g = g == 0.f ? 0.f : copysignf(sqrtf(fmaxf(fabsf(g), 1e-6f)), g);
    const float sg = sigmoidf(g);
    for (int i = threadIdx.x; i < H; i += 256) gv[r * H + i] = sg * value[t * H + i];
}

// PLE gate and the per-stream norm of its output in one pass: gv = sigmoid(gate) * value, n = norm(gv) * (1 + wn).
extern "C" __global__ void __launch_bounds__(256) ple_gate_norm(float* gv, float* n, const float* key, const float* x,
                                                              const float* value, const bf16_t* wk, const bf16_t* wq,
                                                              const bf16_t* wn, int C, int H, float eps) {
    __shared__ float sh[8];
    const size_t r = blockIdx.x;
    const int s = r % C;
    const size_t t = r / C;
    const float* kr = key + r * H;
    const float* xr = x + r * H;
    float sk = 0.f, sx = 0.f;
    for (int i = threadIdx.x; i < H; i += 256) sk += kr[i] * kr[i], sx += xr[i] * xr[i];
    const float ik = rsqrtf(block_sum256(sk, sh) / H + eps);
    const float ix = rsqrtf(block_sum256(sx, sh) / H + eps);
    float dot = 0.f;
    for (int i = threadIdx.x; i < H; i += 256)
        dot += kr[i] * ik * (bf2f(wk[s * H + i]) + 1.f) * xr[i] * ix * (bf2f(wq[s * H + i]) + 1.f);
    float g = block_sum256(dot, sh) * rsqrtf((float)H);
    g = g == 0.f ? 0.f : copysignf(sqrtf(fmaxf(fabsf(g), 1e-6f)), g);
    const float sg = sigmoidf(g);
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += 256) {
        const float v = sg * value[t * H + i];
        gv[r * H + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum256(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += 256)
        n[r * H + i] = gv[r * H + i] * inv * (bf2f(wn[s * H + i]) + 1.f);
}

// PLE short conv: x[t, c] += gv[t, c] + silu(sum_j w[c, j] * n[t - (KS - 1 - j) * DIL, c]), n = normed gv; positions
// before the input read the state (the last (KS - 1) * DIL normed rows of the previous call, oldest first).
extern "C" __global__ void ple_conv(float* x, const float* gv, const float* n, const float* state, const bf16_t* w, int T,
                                    int CH, int KS, int DIL) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * CH) return;
    const int c = i % CH, t = i / CH;
    const int SL = (KS - 1) * DIL;
    float y = 0.f;
    for (int j = 0; j < KS; ++j) {
        const int tau = t - (KS - 1 - j) * DIL;
        const float v = tau >= 0 ? n[(size_t)tau * CH + c] : state[(size_t)(SL + tau) * CH + c];
        y += bf2f(w[c * KS + j]) * v;
    }
    x[i] += gv[i] + silu(y);
}

// state <- the last SL rows of [state; n]; one thread per channel, ascending rows (the shift reads ahead).
extern "C" __global__ void ple_state_save(float* state, const float* n, int T, int CH, int SL) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    for (int i = 0; i < SL; ++i) {
        const int tau = T - SL + i;
        state[(size_t)i * CH + c] = tau >= 0 ? n[(size_t)tau * CH + c] : state[(size_t)(i + T) * CH + c];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// QSA token indexer. Keys are pooled over blocks of R consecutive tokens; a query at absolute position p sees the
// complete blocks b < (p + 1) / R and its own tail [R * ncb, p]. With more than K complete blocks it keeps the K with
// the largest score sum_h relu(q_h . k_b) / sqrt(D). Selections are bitsets over blocks, W 32-bit words per query.

// Partial M-RoPE of element d (< rot) of a head held in xs, at position (t, h, w) = p[0..3].
__device__ __forceinline__ float rope_elem(const float* xs, int d, const int* p, float theta, int rot, int sec_h,
                                           int sec_w, const float* invf) {
    const int half = rot / 2, j = d % half;
    int axis = 0;
    if (j % 3 == 1 && j < sec_h * 3) axis = 1;
    else if (j % 3 == 2 && j < sec_w * 3) axis = 2;
    const float ang = (float)p[axis] / powf(theta, (float)(2 * j) / (float)rot);
    const float c = cosf(ang), s = sinf(ang);
    return d < half ? xs[d] * c - xs[d + half] * s : xs[d] * c + xs[d - half] * s;
}

// From the projection [T, (NH + 1) * D]: query heads RMSNorm (1 + wq) and RoPE -> qi [T, NH, D]; the raw key (last
// head) -> kr [T, D]. Grid (T, NH + 1), block D (128).
extern "C" __global__ void __launch_bounds__(128) idx_prep(float* qi, float* kr, const float* proj, const bf16_t* wq,
                                                          const int* pos, int NH, float eps, float theta, int rot,
                                                          int sec_h, int sec_w, const float* invf) {
    __shared__ float sh[4], xs[128];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, D = 128;
    const float x = proj[((size_t)t * (NH + 1) + h) * D + d];
    if (h == NH) {
        kr[(size_t)t * D + d] = x;
        return;
    }
    float v = warp_sum(x * x);
    if (d % 32 == 0) sh[d / 32] = v;
    __syncthreads();
    const float inv = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / D + eps);
    xs[d] = x * inv * (bf2f(wq[d]) + 1.f);
    __syncthreads();
    qi[((size_t)t * NH + h) * D + d] =
        d < rot ? rope_elem(xs, d, pos + t * 3, theta, rot, sec_h, sec_w, invf) : xs[d];
}

// Pooled keys of the blocks completed by this step, b in [len / R, (len + T) / R): token p of a block is row
// p - len of kr, or (p < len) slot p % R of the previous tail (tk raw keys, tp positions). The key is RMSNorm (1 + wk)
// of the mean, roped at the position of the block's first token, stored fp16 in kb [blocks, D]. Grid = new blocks.
extern "C" __global__ void __launch_bounds__(128) idx_pool(_Float16* kb, const float* kr, const float* tk, const int* tp,
                                                          const int* pos, const bf16_t* wk, const int* lenp, int R,
                                                          float eps, float theta, int rot, int sec_h, int sec_w,
                                                          const float* invf) {
    __shared__ float sh[4], xs[128];
    __shared__ int p0[3];
    const int len = *lenp;
    const int b = len / R + blockIdx.x, d = threadIdx.x, D = 128;
    float m = 0.f;
    for (int r = 0; r < R; ++r) {
        const int p = b * R + r;
        m += p >= len ? kr[(size_t)(p - len) * D + d] : tk[(p % R) * D + d];
    }
    m /= (float)R;
    if (d < 3) {
        const int p = b * R;
        p0[d] = p >= len ? pos[(p - len) * 3 + d] : tp[(p % R) * 3 + d];
    }
    float v = warp_sum(m * m);
    if (d % 32 == 0) sh[d / 32] = v;
    __syncthreads();
    const float inv = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / D + eps);
    xs[d] = m * inv * (bf2f(wk[d]) + 1.f);
    __syncthreads();
    kb[(size_t)b * D + d] = (_Float16)(d < rot ? rope_elem(xs, d, p0, theta, rot, sec_h, sec_w, invf) : xs[d]);
}

// Keeps the raw keys and positions of the step's tokens that are in the last, incomplete block. One block of D.
extern "C" __global__ void __launch_bounds__(128) idx_tail(float* tk, int* tp, const float* kr, const int* pos,
                                                          const int* lenp, int T, int R) {
    const int d = threadIdx.x, D = 128;
    const int len = *lenp;
    for (int p = max(len, (len + T) / R * R); p < len + T; ++p) {
        tk[(p % R) * D + d] = kr[(size_t)(p - len) * D + d];
        if (d < 3) tp[(p % R) * 3 + d] = pos[(p - len) * 3 + d];
    }
}

// scores[i, b] (row stride ld) for queries i in [0, nq) at positions len + q0 + i, blocks b < ncb(i).
// Grid (ceil(maxblocks / 256), ceil(nq / IDX_TQ)), block 256: one block per thread, the tile's query heads in LDS.
//
// The tile is the point: with one block per query row the whole pooled key set was read once per row, and this was
// the second largest depth-scaling cost of the prefill (3,1 % at 24k tokens, 5,4x its 6k figure). A tile of IDX_TQ
// rows reads the keys once for all of them; every (query, head) accumulates over the same (c, e) in the same order,
// so the scores stay bit-identical.
constexpr int IDX_TQ = 4;
extern "C" __global__ void __launch_bounds__(256) idx_score(float* scores, const float* qi, const _Float16* kb,
                                                           const int* lenp, int q0, int NH, int R, int ld, int nq) {
    __shared__ float qs[IDX_TQ * 8 * 128];
    const int len = *lenp, D = 128;
    const int i0 = blockIdx.y * IDX_TQ;
    for (int e = threadIdx.x; e < IDX_TQ * NH * D; e += 256) {
        const int iq = e / (NH * D), r = e % (NH * D);
        qs[e] = i0 + iq < nq ? qi[(size_t)(q0 + i0 + iq) * NH * D + r] : 0.f;
    }
    __syncthreads();
    int ncb[IDX_TQ], ncmax = 0;
#pragma unroll
    for (int iq = 0; iq < IDX_TQ; ++iq) {
        ncb[iq] = i0 + iq < nq ? (len + q0 + i0 + iq + 1) / R : 0;
        ncmax = max(ncmax, ncb[iq]);
    }
    const int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= ncmax) return;
    float acc[IDX_TQ][8] = {};
    const _Float16* k = kb + (size_t)b * D;
    for (int c = 0; c < D; c += 8) {
        const uint4 raw = *(const uint4*)(k + c);
        const _Float16* kv = (const _Float16*)&raw;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float kf = (float)kv[e];
#pragma unroll
            for (int iq = 0; iq < IDX_TQ; ++iq) {
                if (b >= ncb[iq]) continue;
                for (int h = 0; h < NH; ++h) acc[iq][h] += qs[(iq * NH + h) * D + c + e] * kf;
            }
        }
    }
#pragma unroll
    for (int iq = 0; iq < IDX_TQ; ++iq) {
        if (b >= ncb[iq] || i0 + iq >= nq) continue;
        float s = 0.f;
        for (int h = 0; h < NH; ++h) s += fmaxf(acc[iq][h], 0.f);
        scores[(size_t)(i0 + iq) * ld + b] = s * rsqrtf((float)D);
    }
}

// The same scores, but a tile of 16 queries is resident while the block walks key tiles of 64 blocks: the pooled
// keys are read once per 16 query rows instead of once per 4, and q is loaded once per block. 1024 threads map to
// (query, key) pairs, each one accumulates the four heads of one score over the same (c, e) order as `idx_score`,
// so the scores are bit-identical. The host picks `kt` key tiles per block: large in prefill (reuse q), 1 in
// decode (parallelism over keys).
constexpr int IDX_Q = 16;
constexpr int IDX_KB = 64;
extern "C" __global__ void __launch_bounds__(1024) idx_score_tile(float* scores, const float* qi, const _Float16* kb,
                                                                const int* lenp, int q0, int R, int ld, int nq, int kt,
                                                                int G) {
    __shared__ __attribute__((aligned(16))) float qs[IDX_Q * 4 * 128];
    __shared__ __attribute__((aligned(16))) _Float16 ks[IDX_KB][128 + 2];
    __shared__ int nb[IDX_Q];
    __shared__ int ncmax;
    const int len = *lenp, D = 128;
    const int tid = threadIdx.x, qbase = blockIdx.y * IDX_Q;
    for (int i = tid; i < IDX_Q * 4 * D; i += 1024) {
        const int q = i / (4 * D), r = i % (4 * D);
        qs[i] = qbase + q < nq ? qi[(size_t)(q0 + qbase + q) * 4 * D + r] : 0.f;
    }
    if (tid < IDX_Q) {
        const int q = qbase + tid;
        nb[tid] = q < nq ? (len + q0 + q + 1) / R : 0;
    }
    __syncthreads();
    if (tid == 0) {
        ncmax = 0;
#pragma unroll
        for (int i = 0; i < IDX_Q; ++i) ncmax = max(ncmax, nb[i]);
    }
    __syncthreads();
    const int ntiles = (ncmax + IDX_KB - 1) / IDX_KB;
    const int t0 = blockIdx.x * kt, t1 = min(ntiles, t0 + kt);
    const int b_local = tid & (IDX_KB - 1), q_local = tid >> 6;
    const int ncb = nb[q_local];
    for (int t = t0; t < t1; ++t) {
        const int b0 = t * IDX_KB;
        const int lk = tid >> 4, ld8 = (tid & 15) * 8;
        const int b = b0 + lk;
        uint4 raw = {0, 0, 0, 0};
        if (b < ncmax) raw = *(const uint4*)(kb + (size_t)b * D + ld8);
        uint* kd = (uint*)&ks[lk][ld8];
        kd[0] = raw.x;
        kd[1] = raw.y;
        kd[2] = raw.z;
        kd[3] = raw.w;
        __syncthreads();
        const int key = b0 + b_local;
        if (key < ncb && (q_local + 1) % G == 0) {
            const _Float16* kr = ks[b_local];
            const float* qr = qs + q_local * 4 * D;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int c = 0; c < D; c += 8) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float kf = (float)kr[c + e];
#pragma unroll
                    for (int h = 0; h < 4; ++h) acc[h] += qr[h * D + c + e] * kf;
                }
            }
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < 4; ++h) sum += fmaxf(acc[h], 0.f);
            scores[(size_t)(qbase + q_local) * ld + key] = sum * rsqrtf((float)D);
        }
        __syncthreads();
    }
}

// Selection bitset of each query: all complete blocks when there are at most K, else the K largest scores (ties to
// the lower block index). Scores are >= 0, so their bit patterns order like the values: radix select on 4 bytes.
// One 256-thread block per query; sel rows of W words at query q0 + i.
extern "C" __global__ void __launch_bounds__(256) idx_select(uint32_t* sel, const float* scores, const int* lenp, int q0,
                                                            int R, int K, int ld, int W) {
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, need_s;
    __shared__ int ties_before[256];
    const int len = *lenp;
    const int i = blockIdx.x, tid = threadIdx.x;
    const int ncb = (len + q0 + i + 1) / R;
    uint32_t* srow = sel + (size_t)(q0 + i) * W;
    const uint32_t* x = (const uint32_t*)(scores + (size_t)i * ld);
    if (ncb <= K) {
        for (int w = tid; w < W; w += 256) {
            const int lo = w * 32;
            srow[w] = lo + 32 <= ncb ? 0xffffffffu : (lo >= ncb ? 0u : (1u << (ncb - lo)) - 1u);
        }
        return;
    }
    uint32_t prefix = 0, mask = 0;
    uint32_t need = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
        for (int b = tid; b < ncb; b += 256) {
            const uint32_t v = x[b];
            if ((v & mask) == prefix) atomicAdd(&hist[(v >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, bin = 255;
            for (; bin > 0; --bin) {
                if (acc + hist[bin] >= (int)need) break;
                acc += hist[bin];
            }
            prefix_s = prefix | ((uint32_t)bin << shift);
            need_s = need - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        need = need_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // Threshold value `prefix`: everything above it, plus the first `need` blocks equal to it.
    const int per = (ncb + 255) / 256, b0 = tid * per, b1 = min(ncb, b0 + per);
    int ties = 0;
    for (int b = b0; b < b1; ++b) ties += x[b] == prefix;
    ties_before[tid] = ties;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int j = 0; j < 256; ++j) {
            const int c = ties_before[j];
            ties_before[j] = acc;
            acc += c;
        }
    }
    __syncthreads();
    for (int w = tid; w < W; w += 256) srow[w] = 0u;
    __syncthreads();
    int tb = ties_before[tid];
    for (int b = b0; b < b1; ++b) {
        const uint32_t v = x[b];
        bool take = v > prefix;
        if (v == prefix) take = tb++ < (int)need;
        if (take) atomicOr(&srow[b / 32], 1u << (b % 32));
    }
}

// The same selection, with three radix passes of 12/12/8 bits instead of four of 8: the score row is read four
// times instead of five. The histogram is 4096 bins (16 KB of LDS), the tie rule and the bitset are identical.
template <int NT>
__device__ __forceinline__ void idx_select12_impl(uint32_t* sel, const float* scores, const int* lenp, int q0,
                                                  int R, int K, int ld, int W, int G, int nq) {
    __shared__ int hist[4096];
    __shared__ uint32_t prefix_s, need_s;
    __shared__ int ties_before[NT];
    const int len = *lenp;
    const int tid = threadIdx.x;
    // The representative is the last query of the group: it has the most context. Earlier rows get the same
    // bitset; the gather masks the few future keys it may name.
    const int i = min((int)blockIdx.x * G + G - 1, nq - 1);
    const int lo = max(0, i - G + 1), hi = i;
    const int ncb = (len + q0 + i + 1) / R;
    uint32_t* srow = sel + (size_t)(q0 + i) * W;
    const uint32_t* x = (const uint32_t*)(scores + (size_t)i * ld);
    if (ncb <= K) {
        for (int w = tid; w < W; w += NT) {
            const int lo_w = w * 32;
            const uint32_t bits = lo_w + 32 <= ncb ? 0xffffffffu : (lo_w >= ncb ? 0u : (1u << (ncb - lo_w)) - 1u);
            for (int j = lo; j <= hi; ++j) sel[(size_t)(q0 + j) * W + w] = bits;
        }
        return;
    }
    uint32_t prefix = 0, mask = 0;
    uint32_t need = K;
    const int shift0[3] = {20, 8, 0}, bits[3] = {12, 12, 8};
    for (int p = 0; p < 3; ++p) {
        const int shift = shift0[p], nbin = 1 << bits[p];
        for (int j = tid; j < nbin; j += NT) hist[j] = 0;
        __syncthreads();
        for (int b = tid; b < ncb; b += NT) {
            const uint32_t v = x[b];
            if ((v & mask) == prefix) atomicAdd(&hist[(v >> shift) & (nbin - 1)], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, bin = nbin - 1;
            for (; bin > 0; --bin) {
                if (acc + hist[bin] >= (int)need) break;
                acc += hist[bin];
            }
            prefix_s = prefix | ((uint32_t)bin << shift);
            need_s = need - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        need = need_s;
        mask |= (uint32_t)(nbin - 1) << shift;
        __syncthreads();
    }
    // Threshold value `prefix`: everything above it, plus the first `need` blocks equal to it.
    const int per = (ncb + 255) / NT, b0 = tid * per, b1 = min(ncb, b0 + per);
    int ties = 0;
    for (int b = b0; b < b1; ++b) ties += x[b] == prefix;
    ties_before[tid] = ties;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int j = 0; j < NT; ++j) {
            const int c = ties_before[j];
            ties_before[j] = acc;
            acc += c;
        }
    }
    __syncthreads();
    for (int w = tid; w < W; w += NT) srow[w] = 0u;
    __syncthreads();
    int tb = ties_before[tid];
    for (int b = b0; b < b1; ++b) {
        const uint32_t v = x[b];
        bool take = v > prefix;
        if (v == prefix) take = tb++ < (int)need;
        if (take) atomicOr(&srow[b / 32], 1u << (b % 32));
    }
    __syncthreads();
    for (int j = lo; j < hi; ++j)
        for (int w = tid; w < W; w += NT) sel[(size_t)(q0 + j) * W + w] = srow[w];
}

#define IDX_SELECT12(NAME, NT)                                                                                        \
    extern "C" __global__ void __launch_bounds__(NT) NAME(uint32_t* sel, const float* scores, const int* lenp, int q0, \
                                                          int R, int K, int ld, int W, int G, int nq) {              \
        idx_select12_impl<NT>(sel, scores, lenp, q0, R, K, ld, W, G, nq);                                             \
    }
IDX_SELECT12(idx_select12, 256)
IDX_SELECT12(idx_select12_1024, 1024)

// Greedy argmax of one logits row on the device: first maximum wins, ties keep the lower token id, like the host
// fold it replaces. One block, 1024 threads; the host then reads four bytes instead of the whole row.
extern "C" __global__ void __launch_bounds__(1024) logits_argmax(int* out, const float* x, int n) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float v = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float xv = x[i];
        if (xv > v || (xv == v && i < idx)) {
            v = xv;
            idx = i;
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        const float ov = __shfl_down(v, d);
        const int oi = __shfl_down(idx, d);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    if (lane == 0) {
        bv[warp] = v;
        bi[warp] = idx;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < 32 ? bv[lane] : -INFINITY;
        idx = lane < 32 ? bi[lane] : 0;
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_down(v, d);
            const int oi = __shfl_down(idx, d);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) *out = idx;
    }
}

// Union of the selections of each group of 16 queries (one prefill attention wave skips the 16-key tiles none of its
// queries uses).
extern "C" __global__ void idx_union(uint32_t* uni, const uint32_t* sel, int T, int W) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x, tile = blockIdx.y;
    if (w >= W) return;
    uint32_t u = 0;
    for (int t = tile * 16; t < min(T, tile * 16 + 16); ++t) u |= sel[(size_t)t * W + w];
    uni[(size_t)tile * W + w] = u;
}

// x[t, s, j] += e[t, j] (MTP: the embedding projection added to every stream).
extern "C" __global__ void hc_bcast_add(float* x, const float* e, int T, int C, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * C * H) return;
    const int j = i % H, t = i / ((size_t)C * H);
    x[i] += e[(size_t)t * H + j];
}

// dst[i] = src[i] for n floats (copies into buffers the caller only borrows).
extern "C" __global__ void copy_f32(float* dst, const float* src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connection mix for few tokens (decode, speculative verify; T <= 6): hc_pre_dec, hc_down_dec, hc_gate_dec,
// hc_up_mix_dec. MLS8 weights: int8 rows on the randomized-Hadamard input, fp32 row scales.

// 128-point FWHT of s[] (block of 128 threads) times 1/sqrt(128).
__device__ __forceinline__ float fwht128(float* s, int i) {
    for (int h = 1; h < 128; h <<= 1) {
        const float a = s[i], b = s[i ^ h];
        __syncthreads();
        s[i] = (i & h) ? b - a : a + b;
        __syncthreads();
    }
    return s[i] * 0.08838834764831845f;
}

// xn = per-stream RMSNorm (1 + w) of x, xr = its rotation per 128-chunk. Grid T * C * (H / 128), block 128.
extern "C" __global__ void __launch_bounds__(128) hc_pre_dec(float* xr, float* xn, const float* x, const bf16_t* w,
                                                            const int8_t* signs, int C, int H, float eps) {
    __shared__ float sh[4], s[128];
    const int nch = H / 128, row = blockIdx.x / nch, ch = blockIdx.x % nch, i = threadIdx.x;
    const float* xs = x + (size_t)row * H;
    float ss = 0.f;
    for (int k = i; k < H; k += 128) ss += xs[k] * xs[k];
    ss = warp_sum(ss);
    if (i % 32 == 0) sh[i / 32] = ss;
    __syncthreads();
    const float inv = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / H + eps);
    const int k = ch * 128 + i;
    const float n = xs[k] * inv * (bf2f(w[(row % C) * H + k]) + 1.f);
    xn[(size_t)row * H + k] = n;
    s[i] = n * (float)signs[i];
    __syncthreads();
    xr[(size_t)row * H + k] = fwht128(s, i);
}

// part[split, t, r] = sc[r] * q[r, split * KS ..] . xr[t, split * KS ..]: warp per row, KS-wide K slice per
// blockIdx.y (split-K for the few rows of mix_down). Grid (N / 8, K / KS), block 256.
extern "C" __global__ void __launch_bounds__(256) hc_down_dec(float* part, const float* xr, const int8_t* q,
                                                             const float* sc, int T, int N, int K, int KS) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * 8 + warp, split = blockIdx.y;
    if (r >= N) return;
    float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int k1 = (split + 1) * KS;
    for (int k = split * KS + lane * 16; k < k1; k += 512) {
        const uint4 v = *(const uint4*)(q + (size_t)r * K + k);
        const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
        float wf[16];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) wf[i * 4 + j] = (float)(((int32_t)(w4[i] << (24 - 8 * j))) >> 24);
#pragma unroll
        for (int a = 0; a < 6; ++a) {
            if (a >= T) break;
            const float4* xp = (const float4*)(xr + (size_t)a * K + k);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 xv = xp[c];
                acc[a] += xv.x * wf[c * 4] + xv.y * wf[c * 4 + 1] + xv.z * wf[c * 4 + 2] + xv.w * wf[c * 4 + 3];
            }
        }
    }
    const float s = sc[r];
#pragma unroll
    for (int a = 0; a < 6; ++a) {
        if (a >= T) break;
        const float v = warp_sum(acc[a]);
        if (lane == 0) part[((size_t)split * T + a) * N + r] = v * s;
    }
}

// d = sum of the split partials; ar = rotation of a (a[j] = silu(d / C) for j < R, else 0) per 128-chunk; with inj,
// inj[t, s] = 2 sigmoid(d[R + s] / C). Grid T * (RP / 128), block 128.
extern "C" __global__ void __launch_bounds__(128) hc_gate_dec(float* ar, float* inj, const float* part, int T, int R,
                                                             int RP, int C, int S, const int8_t* signs) {
    __shared__ float s[128];
    const int nch = RP / 128, t = blockIdx.x / nch, j = (blockIdx.x % nch) * 128 + threadIdx.x, i = threadIdx.x;
    float d = 0.f;
    for (int p = 0; p < S; ++p) d += part[((size_t)p * T + t) * RP + j];
    const float v = d / (float)C;
    if (inj && j >= R && j < R + C) inj[t * C + j - R] = 2.f * sigmoidf(v);
    s[i] = (j < R ? silu(v) : 0.f) * (float)signs[i];
    __syncthreads();
    ar[(size_t)t * RP + j] = fwht128(s, i);
}

// mixed[t, j] = mean_s sigmoid(sc[r] q[r] . ar[t]) * xn[t, s H + j], r = j C + s: warp per column j, the C rows of
// mix_up [H C, RP] (rows reordered by the loader). Grid H / 8, block 256.
extern "C" __global__ void __launch_bounds__(256) hc_up_mix_dec(float* mixed, const float* xn, const float* ar,
                                                               const int8_t* q, const float* sc, int T, int C, int H,
                                                               int RP) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int j = blockIdx.x * 8 + warp;
    if (j >= H) return;
    float acc[6][4];
#pragma unroll
    for (int a = 0; a < 6; ++a)
#pragma unroll
        for (int s = 0; s < 4; ++s) acc[a][s] = 0.f;
    for (int k = lane * 16; k < RP; k += 512) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= C) break;
            const uint4 v = *(const uint4*)(q + ((size_t)j * C + s) * RP + k);
            const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
            float wf[16];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) wf[i * 4 + jj] = (float)(((int32_t)(w4[i] << (24 - 8 * jj))) >> 24);
#pragma unroll
            for (int a = 0; a < 6; ++a) {
                if (a >= T) break;
                const float4* xp = (const float4*)(ar + (size_t)a * RP + k);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 xv = xp[c];
                    acc[a][s] += xv.x * wf[c * 4] + xv.y * wf[c * 4 + 1] + xv.z * wf[c * 4 + 2] + xv.w * wf[c * 4 + 3];
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 6; ++a) {
        if (a >= T) break;
        float m = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= C) break;
            const float u = warp_sum(acc[a][s]) * sc[j * C + s];
            m += sigmoidf(u) * xn[(size_t)a * C * H + (size_t)s * H + j];
        }
        if (lane == 0) mixed[(size_t)a * H + j] = m / (float)C;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connection mix for many tokens (prefill): hc_quant_row + int8 GEMMs + hc_mix_inv.

typedef float hq_f4 __attribute__((ext_vector_type(4)));

// One 256-thread block per token row of the streams x [T, C * H] (C * H <= 8 * 128 * 10): with o, first
// x += o * inj (written back); then per-stream RMSNorm (1 + w), the randomized Hadamard of every 128-chunk and int8
// quantization with one scale per row (q [T, C * H], xs [T]), as rot_quant_row; inv [T, C] keeps 1 / rms.
extern "C" __global__ void __launch_bounds__(256) hc_quant_row(int8_t* q, float* xs, float* inv, float* x,
                                                              const float* o, const float* inj, const bf16_t* w,
                                                              const int8_t* signs, int C, int H, float eps) {
    constexpr int CPW = 10;
    __shared__ float red[8][4], red2[8];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int K = C * H, G = K / 128, GS = H / 128;
    const size_t row = (size_t)blockIdx.x * K;
    hq_f4 v[CPW];
    float ss[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int c = wave + 8 * i;
        if (c < G) {
            hq_f4 a = *(const hq_f4*)(x + row + c * 128 + 4 * lane);
            if (o) {
                const float g = inj[(size_t)blockIdx.x * C + c / GS];
                const hq_f4 b = *(const hq_f4*)(o + (size_t)blockIdx.x * H + (c % GS) * 128 + 4 * lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) a[j] += b[j] * g;
                *(hq_f4*)(x + row + c * 128 + 4 * lane) = a;
            }
            v[i] = a;
            const float s2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
#pragma unroll
            for (int s = 0; s < 4; ++s)
                if (c / GS == s) ss[s] += s2;
        }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        ss[s] = warp_sum(ss[s]);
        if (lane == 0) red[wave][s] = ss[s];
    }
    __syncthreads();
    float iv[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float t = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) t += red[k][s];
        iv[s] = rsqrtf(t / H + eps);
    }
    if (threadIdx.x < C) inv[(size_t)blockIdx.x * C + threadIdx.x] = iv[threadIdx.x];
    const char4 sg = *(const char4*)(signs + 4 * lane);
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int c = wave + 8 * i;
        if (c < G) {
            const int s = c / GS;
            const float sc = s == 0 ? iv[0] : s == 1 ? iv[1] : s == 2 ? iv[2] : iv[3];
            const uint2 wb = *(const uint2*)(w + c * 128 + 4 * lane);
            const float w4[4] = {__uint_as_float(wb.x << 16), __uint_as_float(wb.x & 0xffff0000u),
                                 __uint_as_float(wb.y << 16), __uint_as_float(wb.y & 0xffff0000u)};
            float a[4] = {v[i][0] * sc * (w4[0] + 1.f) * sg.x, v[i][1] * sc * (w4[1] + 1.f) * sg.y,
                          v[i][2] * sc * (w4[2] + 1.f) * sg.z, v[i][3] * sc * (w4[3] + 1.f) * sg.w};
            // 128-point Hadamard across the wave: lane holds 4 consecutive elements.
            float b0 = a[0] + a[1], b1 = a[0] - a[1], b2 = a[2] + a[3], b3 = a[2] - a[3];
            a[0] = b0 + b2, a[1] = b1 + b3, a[2] = b0 - b2, a[3] = b1 - b3;
#pragma unroll
            for (int mm = 1; mm < 32; mm <<= 1) {
                const bool hi = lane & mm;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float ot = __shfl_xor(a[j], mm, 32);
                    a[j] = hi ? ot - a[j] : a[j] + ot;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) a[j] *= 0.08838834764831845f;
            v[i] = (hq_f4){a[0], a[1], a[2], a[3]};
            m = fmaxf(m, fmaxf(fmaxf(fabsf(a[0]), fabsf(a[1])), fmaxf(fabsf(a[2]), fabsf(a[3]))));
        }
    }
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor(m, k, 32));
    if (lane == 0) red2[wave] = m;
    __syncthreads();
    m = red2[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red2[k]);
    const float s = m > 0.f ? m / 127.f : 1.f, is = 1.f / s;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int c = wave + 8 * i;
        if (c < G) {
            char4 o4;
            o4.x = (int8_t)__float2int_rn(v[i][0] * is);
            o4.y = (int8_t)__float2int_rn(v[i][1] * is);
            o4.z = (int8_t)__float2int_rn(v[i][2] * is);
            o4.w = (int8_t)__float2int_rn(v[i][3] * is);
            *(char4*)(q + row + c * 128 + 4 * lane) = o4;
        }
    }
    if (threadIdx.x == 0) xs[blockIdx.x] = s;
}

// mixed[t, j] = mean_s sigmoid(u[t, s, j]) * x[t, s, j] * inv[t, s] * (1 + w[s, j]) (normalization of hc_quant_row).
extern "C" __global__ void hc_mix_inv(float* mixed, const float* u, const float* x, const float* inv, const bf16_t* w,
                                      int T, int C, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int t = i / H, j = i % H;
    const size_t b = (size_t)t * C * H + j;
    float acc = 0.f;
    for (int s = 0; s < C; ++s)
        acc += sigmoidf(u[b + (size_t)s * H]) * x[b + (size_t)s * H] * inv[t * C + s] * (bf2f(w[s * H + j]) + 1.f);
    mixed[i] = acc / (float)C;
}

// out[T, N] = x[T, K] . W[N, K]^T with bf16 weights on WMMA (bf16 operands, fp32 accumulation): 64 x 64 tiles,
// 8 waves of 16 x 32, K staged 32 at a time. K a multiple of 32.
typedef short bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// `ldx`/`ldo` are the row strides of the input and the output (K and N for a plain GEMM) and `blockIdx.z` walks a
// batch, each member `K` columns into x, `N` into out and `wld` rows into W plus `woff` rows: one launch for the 32
// heads of Hy4's `kv_b_proj` V half, whose rows are a contiguous block per head.
extern "C" __global__ void __launch_bounds__(256) gemm_bf16_wmma(float* out, const float* x, const bf16_t* W, int T,
                                                                int N, int K, int ldx, int ldo, int wld, int woff) {
    __shared__ __attribute__((aligned(16))) uint16_t xs[64][40];
    __shared__ __attribute__((aligned(16))) uint16_t ws[64][40];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int t0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const float* xb = x + (size_t)blockIdx.z * K;
    float* ob = out + (size_t)blockIdx.z * N;
    const bf16_t* Wb = W + (size_t)blockIdx.z * wld + woff;
    const int wt = (wave / 2) * 16, wn = (wave % 2) * 32;
    f32x8 acc[2] = {(f32x8){0, 0, 0, 0, 0, 0, 0, 0}, (f32x8){0, 0, 0, 0, 0, 0, 0, 0}};
    for (int k0 = 0; k0 < K; k0 += 32) {
        __syncthreads();
        for (int e = tid; e < 64 * 32; e += 256) {
            const int r = e / 32, c = e % 32;
            xs[r][c] = f2bf(xb[(size_t)min(t0 + r, T - 1) * ldx + k0 + c]);
            ws[r][c] = Wb[(size_t)min(n0 + r, N - 1) * K + k0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            const bf16x16 a = *(const bf16x16*)&xs[wt + lane % 16][kk];
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                const bf16x16 b = *(const bf16x16*)&ws[wn + f * 16 + lane % 16][kk];
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[f]);
            }
        }
    }
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t = t0 + wt + 2 * i + lane / 16, n = n0 + wn + f * 16 + lane % 16;
            if (t < T && n < N) ob[(size_t)t * ldo + n] = acc[f][i];
        }
}

// Sorted list of the blocks set in each query's selection bitset (list [T, L], cnt [T]); one 256-thread block per
// query, each thread owns a contiguous range of words.
extern "C" __global__ void __launch_bounds__(256) idx_compact(int* list, int* cnt, const uint32_t* sel, int W, int L) {
    __shared__ int pre[256];
    const int t = blockIdx.x, tid = threadIdx.x;
    const uint32_t* row = sel + (size_t)t * W;
    const int per = (W + 255) / 256, w0 = tid * per, w1 = min(W, w0 + per);
    int c = 0;
    for (int w = w0; w < w1; ++w) c += __popc(row[w]);
    pre[tid] = c;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < 256; ++i) {
            const int v = pre[i];
            pre[i] = acc;
            acc += v;
        }
        cnt[t] = min(acc, L);
    }
    __syncthreads();
    int o = pre[tid];
    for (int w = w0; w < w1; ++w) {
        uint32_t b = row[w];
        while (b && o < L) {
            const int i = __ffs(b) - 1;
            b &= b - 1;
            list[(size_t)t * L + o++] = w * 32 + i;
        }
    }
}

// Decode / verify attention over the selected blocks only (QSA): the virtual keys of query t are the R tokens of each
// listed block, then its tail [R * ncb, pos]. Block = (token, q head, 256 virtual keys); 8 waves of 32 keys; writes
// the split partials for attention_combine. HD = 256.
typedef unsigned int sel_u4 __attribute__((ext_vector_type(4)));
typedef unsigned int sel_u2 __attribute__((ext_vector_type(2)));

extern "C" __global__ void __launch_bounds__(256) attention_sel(float* part, const float* q, const _Float16* kc,
                                                               const _Float16* vtc, const int* list, const int* cnt,
                                                               const int* cache_pos0p, int nh, int nkv, float scale, int ctx,
                                                               int nsplit, int R, int L, const _Float16* vr) {
    constexpr int HD = 256;
    const int cache_pos0 = *cache_pos0p;
    __shared__ float qs[HD];
    __shared__ float pw[8][32];
    __shared__ int keys[8][32];
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    const int th = blockIdx.x, t = th / nh, h = th % nh, kh = h / (nh / nkv), split = blockIdx.y;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nkeys = cache_pos0 + t + 1, ncb = nkeys / R, tail0 = ncb * R;
    const int nsel = cnt[t], nv = nsel * R + (nkeys - tail0);
    qs[threadIdx.x] = q[((size_t)t * nh + h) * HD + threadIdx.x] * scale;
    const int v = split * 256 + wave * 32 + lane;
    int key = -1;
    if (v < nsel * R) key = list[(size_t)t * L + v / R] * R + v % R;
    else if (v < nv) key = tail0 + v - nsel * R;
    keys[wave][lane] = key;
    __syncthreads();
    float s = -INFINITY;
    if (key >= 0) {
        const size_t kb = (size_t)kh * ctx * HD + (size_t)(key / 32) * HD * 32 + (size_t)(key % 32) * 8;
        s = 0.f;
#pragma unroll 4
        for (int g = 0; g < HD / 8; ++g) {
            const sel_u4 kv = *(const sel_u4*)(kc + kb + (size_t)g * 256);
            const _Float16* hv = (const _Float16*)&kv;
#pragma unroll
            for (int e = 0; e < 8; ++e) s += qs[g * 8 + e] * (float)hv[e];
        }
    }
    float m = s;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float p = m == -INFINITY ? 0.f : __expf(s - m);
    float l = p;
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor(l, o, 32);
    pw[wave][lane] = p;
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    if (m != -INFINITY && vr) {
        // Row-major V copy: one coalesced 512-byte row per key, lane = 8 dims.
        for (int j = 0; j < 32; ++j) {
            const int k = keys[wave][j];
            if (k < 0) continue;
            const sel_u4 vv = *(const sel_u4*)(vr + ((size_t)k * nkv + kh) * HD + lane * 8);
            const _Float16* hv = (const _Float16*)&vv;
            const float pj = pw[wave][j];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += pj * (float)hv[i];
        }
    } else if (m != -INFINITY) {
        // Keys come in runs of R consecutive positions (one block): read R halves of V^T per run.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 32; ++j) {
                const int k = keys[wave][j];
                if (k >= 0) {
                    const size_t row = (size_t)kh * ctx * HD + (size_t)(k / 32) * HD * 32 +
                                       (size_t)((k % 32) / 8) * HD * 8 + (size_t)(lane * 8 + i) * 8 + (k % 8);
                    acc[i] += pw[wave][j] * (float)vtc[row];
                }
            }
        }
    }
    if (lane == 0) {
        wm[wave] = m;
        wl[wave] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) wacc[wave][lane * 8 + i] = acc[i];
    __syncthreads();
    const int d = threadIdx.x;
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w]);
    float Lsum = 0.f, o = 0.f;
    if (M != -INFINITY) {
        for (int w = 0; w < 8; ++w) {
            if (wm[w] == -INFINITY) continue;
            const float f = __expf(wm[w] - M);
            Lsum += wl[w] * f;
            o += wacc[w][d] * f;
        }
    }
    float* dst = part + ((size_t)th * nsplit + split) * (HD + 2);
    if (d == 0) {
        dst[0] = M;
        dst[1] = Lsum;
    }
    dst[2 + d] = o;
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connection mix for few tokens in two kernels (decode, verify; T <= 6, C <= 4, H <= 2560).

typedef _Float16 hcd_h8 __attribute__((ext_vector_type(8)));

// 128-point randomized Hadamard on one wave, lane holding 4 consecutive elements.
__device__ __forceinline__ void wave_fwht(float (&a)[4], int lane) {
    float b0 = a[0] + a[1], b1 = a[0] - a[1], b2 = a[2] + a[3], b3 = a[2] - a[3];
    a[0] = b0 + b2, a[1] = b1 + b3, a[2] = b0 - b2, a[3] = b1 - b3;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = lane & m;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(a[j], m, 32);
            a[j] = hi ? o - a[j] : a[j] + o;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) a[j] *= 0.08838834764831845f;
}

// Split-K slice s = stream s: every block takes stream s of the T inputs (x + o * inj when o is given, not written),
// RMSNorm (1 + w), rotates it into LDS (fp16) and runs 8 rows of mix_down (q [RP, C H] int8, sc [RP]) on it:
// part[s, t, r]. Blocks with blockIdx.x == 0 write 1 / rms to inv [T, C]. Grid (RP / 8, C), block 256.
extern "C" __global__ void __launch_bounds__(256) hc_down_fused(float* part, float* inv, const float* x, const float* o,
                                                               const float* inj, const bf16_t* w, const int8_t* q,
                                                               const float* sc, const int8_t* signs, int T, int RP,
                                                               int C, int H, float eps) {
    extern __shared__ __attribute__((aligned(16))) _Float16 xh_dyn[];  // [T][H], T * H * 2 bytes at launch
    __shared__ float red[8];
    const int s = blockIdx.y, tid = threadIdx.x, wave = tid / 32, lane = tid % 32, K = C * H;
    auto xh = [&](int a, int k) -> _Float16& { return xh_dyn[a * H + k]; };
    const char4 sg = *(const char4*)(signs + 4 * lane);
    for (int a = 0; a < T; ++a) {
        const float* xr = x + (size_t)a * K + (size_t)s * H;
        const float g = o ? inj[a * C + s] : 0.f;
        float ss = 0.f;
        for (int k = tid; k < H; k += 256) {
            const float v = xr[k] + (o ? o[(size_t)a * H + k] * g : 0.f);
            ss += v * v;
        }
        ss = warp_sum(ss);
        if (lane == 0) red[wave] = ss;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) tot += red[i];
        const float iv = rsqrtf(tot / H + eps);
        if (blockIdx.x == 0 && tid == 0) inv[a * C + s] = iv;
        for (int c = wave; c < H / 128; c += 8) {
            const int k0 = c * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float xv = xr[k0 + j] + (o ? o[(size_t)a * H + k0 + j] * g : 0.f);
                v[j] = xv * iv * (bf2f(w[s * H + k0 + j]) + 1.f);
            }
            v[0] *= sg.x, v[1] *= sg.y, v[2] *= sg.z, v[3] *= sg.w;
            wave_fwht(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) xh(a, k0 + j) = (_Float16)v[j];
        }
        __syncthreads();
    }
    const int r = blockIdx.x * 8 + wave;
    if (r >= RP) return;
    float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int8_t* qr = q + (size_t)r * K + (size_t)s * H;
    for (int k = lane * 16; k < H; k += 512) {
        const uint4 u = *(const uint4*)(qr + k);
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        float wf[16];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) wf[i * 4 + j] = (float)(((int32_t)(w4[i] << (24 - 8 * j))) >> 24);
#pragma unroll
        for (int a = 0; a < 6; ++a) {
            if (a >= T) break;
            const hcd_h8 x0 = *(const hcd_h8*)&xh(a, k), x1 = *(const hcd_h8*)&xh(a, k + 8);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[a] += wf[j] * (float)x0[j] + wf[8 + j] * (float)x1[j];
        }
    }
    const float scale = sc[r];
#pragma unroll
    for (int a = 0; a < 6; ++a) {
        if (a >= T) break;
        const float v = warp_sum(acc[a]);
        if (lane == 0) part[((size_t)s * T + a) * RP + r] = v * scale;
    }
}

// Gate (a = silu(d / C), rotated; inj_new = 2 sigmoid(d[R + s] / C)) recomputed by every block from the C split
// partials, then warp per output column j: u_s = mix_up row j C + s . a, mixed[t, j] = mean_s sigmoid(u_s) *
// xs * inv[t, s] * (1 + w[s, j]) with xs = x + o * inj_old, written back to the streams when o is given.
// Grid H / 8, block 256.
extern "C" __global__ void __launch_bounds__(256) hc_up_fused(float* mixed, float* x, const float* o,
                                                             const float* inj_old, float* inj_new, const float* part,
                                                             const int8_t* q, const float* sc, const bf16_t* w,
                                                             const float* inv, const int8_t* signs, int T, int R,
                                                             int RP, int C, int H, const float* ar_in) {
    __shared__ __attribute__((aligned(16))) float ar[6][384];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const char4 sg = *(const char4*)(signs + 4 * lane);
    const int nch = RP / 128;
    if (ar_in)
        for (int e = tid; e < T * RP; e += 256) ar[e / RP][e % RP] = ar_in[e];
    for (int it = ar_in ? T * nch : wave; it < T * nch; it += 8) {
        const int a = it / nch, c = it % nch;
        float v[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int j = c * 128 + 4 * lane + jj;
            float d = 0.f;
            for (int p = 0; p < C; ++p) d += part[((size_t)p * T + a) * RP + j];
            const float dv = d / (float)C;
            v[jj] = j < R ? silu(dv) : 0.f;
            if (inj_new && blockIdx.x == 0 && j >= R && j < R + C) inj_new[a * C + j - R] = 2.f * sigmoidf(dv);
        }
        v[0] *= sg.x, v[1] *= sg.y, v[2] *= sg.z, v[3] *= sg.w;
        wave_fwht(v, lane);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) ar[a][c * 128 + 4 * lane + jj] = v[jj];
    }
    __syncthreads();
    const int j = blockIdx.x * 8 + wave;
    if (j >= H) return;
    float acc[6][4];
#pragma unroll
    for (int a = 0; a < 6; ++a)
#pragma unroll
        for (int s = 0; s < 4; ++s) acc[a][s] = 0.f;
    for (int k = lane * 16; k < RP; k += 512) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= C) break;
            const uint4 u = *(const uint4*)(q + ((size_t)j * C + s) * RP + k);
            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
            float wf[16];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) wf[i * 4 + jj] = (float)(((int32_t)(w4[i] << (24 - 8 * jj))) >> 24);
#pragma unroll
            for (int a = 0; a < 6; ++a) {
                if (a >= T) break;
#pragma unroll
                for (int e = 0; e < 16; ++e) acc[a][s] += wf[e] * ar[a][k + e];
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 6; ++a) {
        if (a >= T) break;
        float m = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= C) break;
            const float u = warp_sum(acc[a][s]) * sc[j * C + s];
            const size_t xi = (size_t)a * C * H + (size_t)s * H + j;
            const float xs = x[xi] + (o ? o[(size_t)a * H + j] * inj_old[a * C + s] : 0.f);
            m += sigmoidf(u) * xs * inv[a * C + s] * (bf2f(w[s * H + j]) + 1.f);
            if (o && lane == 0) x[xi] = xs;
        }
        if (lane == 0) mixed[(size_t)a * H + j] = m / (float)C;
    }
}
