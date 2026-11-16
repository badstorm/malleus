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

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < NT / 32; ++i) r += sh[i];
    return r;
}

extern "C" __global__ void embed_bf16(float* out, const bf16_t* table, const int* ids, int H) {
    const int t = blockIdx.x;
    const bf16_t* row = table + (size_t)ids[t] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[(size_t)t * H + i] = bf2f(row[i]);
}

// out = x / rms(x) * (w + add_one); one 256-thread block per row.
extern "C" __global__ void rmsnorm(float* out, const float* x, const bf16_t* w, int n, float eps, int add_one) {
    __shared__ float sh[8];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = out + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum<256>(ss, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = xr[i] * inv * (bf2f(w[i]) + (add_one ? 1.f : 0.f));
}

// out[t, n] (+)= sum_k x[t, k] * W[n, k]; 8 warps per block, one row per warp, 8 tokens per block.y.
extern "C" __global__ void gemm_bf16(float* out, const float* x, const bf16_t* W, int T, int N, int K, int accumulate) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = blockIdx.x * 8 + warp;
    const int t0 = blockIdx.y * 8;
    if (n >= N) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const bf16_t* wr = W + (size_t)n * K;
    for (int k = lane * 8; k < K; k += 256) {
        const uint4 wv = *(const uint4*)(wr + k);
        float wf[8];
        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wf[2 * i] = __uint_as_float(ws[i] << 16);
            wf[2 * i + 1] = __uint_as_float(ws[i] & 0xffff0000u);
        }
#pragma unroll
        for (int tt = 0; tt < 8; ++tt) {
            if (t0 + tt < T) {
                const float4* xp = (const float4*)(x + (size_t)(t0 + tt) * K + k);
                const float4 a = xp[0], b = xp[1];
                acc[tt] += a.x * wf[0] + a.y * wf[1] + a.z * wf[2] + a.w * wf[3] + b.x * wf[4] + b.y * wf[5] + b.z * wf[6] + b.w * wf[7];
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < 8; ++tt) {
        const float s = warp_sum(acc[tt]);
        if (lane == 0 && t0 + tt < T) {
            float* o = out + (size_t)(t0 + tt) * N + n;
            *o = accumulate ? *o + s : s;
        }
    }
}

extern "C" __global__ void add_inplace(float* y, const float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

extern "C" __global__ void silu_mul(float* out, const float* gate, const float* up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu(gate[i]) * up[i];
}

// Per (token, head): RMSNorm(1+w) over head_dim 256, partial NeoX rope on the first `rot` dims with M-RoPE
// interleaved sections. Query heads write q_out; KV heads write k/v into the cache at row cache_pos0 + t.
// KV cache (fp16): K blocked [nkv][ctx/32][dim/8][32 keys][8 dims] and V transposed and blocked
// [nkv][ctx/32][4][256][8] (ctx = padded context length; key / 8 picks the 8-key group), so a decode wave reads
// 32 consecutive dims of K and V as coalesced 512-byte runs.
// vr (optional): V also row-major [pos][nkv][256] (gathered by the sparse attention).
extern "C" __global__ void attn_prep(float* q_out, _Float16* kc, _Float16* vt, const float* qproj, const float* kproj,
                                     const float* vproj, const bf16_t* qn_w, const bf16_t* kn_w, const int* pos,
                                     const int* cache_pos0p, int nh, int nkv, float eps, float theta, int rot, int sec_h,
                                     int sec_w, int ctx, _Float16* vr, const float* invf) {
    const int cache_pos0 = *cache_pos0p;
    __shared__ float sh[8];
    __shared__ float xs[256];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const bool is_q = h < nh;
    const int kh = h - nh;
    float x;
    if (is_q) x = qproj[(size_t)t * nh * 512 + h * 512 + d];
    else x = kproj[(size_t)t * nkv * 256 + kh * 256 + d];
    const float inv = rsqrtf(block_sum<256>(x * x, sh) / 256.f + eps);
    x = x * inv * (1.f + bf2f(is_q ? qn_w[d] : kn_w[d]));
    xs[d] = x;
    __syncthreads();
    const int half = rot / 2;
    if (d < rot) {
        const int j = d % half;
        int axis = 0;
        if (j % 3 == 1 && j < sec_h * 3) axis = 1;
        else if (j % 3 == 2 && j < sec_w * 3) axis = 2;
        const float p = (float)pos[t * 3 + axis];
        // With a scaling factor (YaRN) the host passes the blended inverse frequencies: they cannot be a closed form
        // in the kernel because the blend depends on the wavelength of each index.
        const float inv_freq = invf ? invf[j] : 1.0f / powf(theta, (float)(2 * j) / (float)rot);
        const float ang = p * inv_freq;
        const float c = cosf(ang), s = sinf(ang);
        x = d < half ? xs[d] * c - xs[d + half] * s : xs[d] * c + xs[d - half] * s;
    }
    if (is_q) {
        q_out[((size_t)t * nh + h) * 256 + d] = x;
    } else {
        const int p = cache_pos0 + t;
        kc[(size_t)kh * ctx * 256 + (size_t)(p / 32) * 256 * 32 + (size_t)(d / 8) * 256 +
           (size_t)(p % 32) * 8 + (d % 8)] = (_Float16)x;
        const _Float16 vv = (_Float16)vproj[(size_t)t * nkv * 256 + kh * 256 + d];
        vt[(size_t)kh * ctx * 256 + (size_t)(p / 32) * 256 * 32 + (size_t)((p % 32) / 8) * 256 * 8 + (size_t)d * 8 + (p % 8)] = vv;
        if (vr) vr[((size_t)p * nkv + kh) * 256 + d] = vv;
    }
}

// Causal GQA attention with online softmax. Block per (token, q head), 8 warps split the keys, lanes split
// head_dim 256 in chunks of 8. Token t attends to cache rows [0, cache_pos0 + t].
// attn[t, h*256 + d] *= sigmoid(qproj[t, h*512 + 256 + d])
extern "C" __global__ void attn_gate(float* attn, const float* qproj, int T, int nh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * nh * 256) return;
    const int t = i / (nh * 256), r = i % (nh * 256), h = r / 256, d = r % 256;
    attn[i] *= sigmoidf(qproj[(size_t)t * nh * 512 + h * 512 + 256 + d]);
}

// Depthwise causal conv (kernel 4) + SiLU over all tokens in parallel: one wave per (token, 128 channels), 4
// channels per lane. The first 2 * nk groups are the q and k heads, L2-normalized here (q also scaled by
// 1 / sqrt(128)). state holds the last 3 inputs of the previous call and is updated afterwards by conv_state_save.
typedef float cv_f4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void __launch_bounds__(256) conv_qk_norm(float* out, const float* in, const bf16_t* w,
                                                                const float* state, int T, int C, int nk) {
    const int wv = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32, groups = C / 128;
    if (wv >= T * groups) return;
    const int t = wv / groups, grp = wv % groups, c = grp * 128 + 4 * lane;
    const uint2* wp = (const uint2*)(w + (size_t)c * 4);
    cv_f4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = t - 3 + j;
        cv_f4 x;
        if (p >= 0) {
            x = *(const cv_f4*)(in + (size_t)p * C + c);
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) x[e] = state[(c + e) * 3 + p + 3];
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const uint2 wb = wp[e];
            const unsigned short wj = (j < 2 ? (j == 0 ? wb.x : wb.x >> 16) : (j == 2 ? wb.y : wb.y >> 16)) & 0xffff;
            acc[e] += __uint_as_float((unsigned)wj << 16) * x[e];
        }
    }
    cv_f4 y;
#pragma unroll
    for (int e = 0; e < 4; ++e) y[e] = silu(acc[e]);
    if (grp < 2 * nk) {
        float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
        y *= rsqrtf(ss + 1e-6f) * (grp < nk ? rsqrtf(128.f) : 1.f);
    }
    *(cv_f4*)(out + (size_t)t * C + c) = y;
}

// Keeps the last 3 inputs; with snaps, also the state after each token but the last (speculative rollback).
extern "C" __global__ void conv_state_save(const float* in, float* state, int T, int C, float* snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float o0 = state[c * 3], o1 = state[c * 3 + 1], o2 = state[c * 3 + 2];
#define CONV_AT(p) ((p) >= 0 ? in[(size_t)(p) * C + c] : ((p) == -1 ? o2 : ((p) == -2 ? o1 : o0)))
    if (snaps)
        for (int s = 0; s + 1 < T; ++s)
#pragma unroll
            for (int j = 0; j < 3; ++j) snaps[(size_t)s * C * 3 + c * 3 + j] = CONV_AT(s - 2 + j);
#pragma unroll
    for (int j = 0; j < 3; ++j) state[c * 3 + j] = CONV_AT(T - 3 + j);
#undef CONV_AT
}

// out[t] = [RMSNorm(e[t]) * (1 + we), RMSNorm(h[t]) * (1 + wh)] (MTP input); one 256-thread block per row.
extern "C" __global__ void mtp_cat(float* out, const float* e, const float* h, const bf16_t* we, const bf16_t* wh, int n,
                                   float eps) {
    __shared__ float sh[8];
    const size_t r = blockIdx.x;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const float* x = (part ? h : e) + r * n;
        const bf16_t* w = part ? wh : we;
        float ss = 0.f;
        for (int i = threadIdx.x; i < n; i += 256) ss += x[i] * x[i];
        const float inv = rsqrtf(block_sum<256>(ss, sh) / n + eps);
        for (int i = threadIdx.x; i < n; i += 256) out[r * 2 * n + part * n + i] = x[i] * inv * (bf2f(w[i]) + 1.f);
        __syncthreads();
    }
}

// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias). a and b rows have stride ld; b starts at column boff.
extern "C" __global__ void deltanet_gates(float* beta, float* g, const float* a, const float* b, const bf16_t* a_log,
                                          const bf16_t* dt_bias, int T, int H, int ld, int boff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int h = i % H, t = i / H;
    beta[i] = sigmoidf(b[(size_t)t * ld + boff + h]);
    const float x = a[(size_t)t * ld + h] + bf2f(dt_bias[h]);
    const float sp = x > 20.f ? x : log1pf(expf(x));
    g[i] = -expf(bf2f(a_log[h])) * sp;
}

// KDA gates (Ling-3.0-flash): beta = sigmoid(b_proj x) per head, and the decay g = lb * sigmoid(exp(A_log) *
// (f_proj x + dt_bias)) per key channel, with lb = kda_lower_bound (-5). Same shape as deltanet_gates except the
// decay is one value per channel instead of one per head, and A_log / dt_bias are f32 in this checkpoint (the
// converter keeps f32 tensors as they are instead of downcasting to bf16 the way it did for Qwen).
extern "C" __global__ void __launch_bounds__(256) kda_gates(float* beta, float* g, const float* f, const float* b,
                                                            const float* a_log, const float* dt_bias, int T, int H,
                                                            float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int h = i % H, t = i / H;
    beta[i] = sigmoidf(b[i]);
    const float al = expf(a_log[h]);
    const float* fp = f + (size_t)t * H * 128 + h * 128;
    const float* db = dt_bias + h * 128;
    float* gp = g + (size_t)t * H * 128 + h * 128;
#pragma unroll
    for (int c = 0; c < 128; ++c) {
        gp[c] = lb / (1.f + expf(-(al * (fp[c] + db[c]))));
    }
}

// out = x / rms(x) * w * act(z) per 128-wide row, act silu (sig = 0) or sigmoid; block of 128 threads per row.
extern "C" __global__ void rmsnorm_gated(float* out, const float* x, const float* z, const bf16_t* w, float eps, int sig) {
    __shared__ float sh[4];
    const size_t r = blockIdx.x;
    const int d = threadIdx.x;
    const float v = x[r * 128 + d];
    const float inv = rsqrtf(block_sum<128>(v * v, sh) / 128.f + eps);
    const float zz = z[r * 128 + d];
    out[r * 128 + d] = v * inv * bf2f(w[d]) * (sig ? sigmoidf(zz) : silu(zz));
}

// Gated delta rule on normalized q/k (conv_qk_norm). One wave per (value head, 16 value columns): lanes j and
// j + 16 keep rows [0, 64) and [64, 128) of state column j in registers; q/k of each token are broadcast through LDS
// and the next token is prefetched while the current one is processed.
// Speculative verify: `pin` holds np accepted rank-1 updates of the previous verify (records of k[nk*128],
// delta[nv*128], decay[nv] per token), applied while loading the state; with `pout` the state after them is stored
// and this call's updates are recorded instead of applied, so rejected tokens never touch the state.
// The state update is written once (`dn_decay`, `dn_rank1`) and used both by the scan and by the replay of the
// records: left to the compiler, the two were contracted into different multiply-adds, the replayed state came out
// a few ulp away from the scanned one, and the first token after a verify no longer matched the serial decode.
__device__ __forceinline__ float dn_decay(float s, float e) {
#pragma clang fp contract(off)
    return s * e;
}
__device__ __forceinline__ float dn_rank1(float s, float k, float d) { return __builtin_fmaf(k, d, s); }

typedef float f32x4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void __launch_bounds__(32) deltanet_scan(float* out, const float* qkv, const float* beta,
                                                               const float* g, float* state, int T, int nv, int nk,
                                                               const float* pin, int np, float* pout) {
    __shared__ __attribute__((aligned(16))) float qk[2][256];
    const int lane = threadIdx.x, h = blockIdx.x / 8, j = (blockIdx.x % 8) * 16 + lane % 16, r0 = (lane / 16) * 64;
    const int kh = h / (nv / nk), dim = nk * 256 + nv * 128;
    float* st = state + (size_t)h * 128 * 128;
    const int rec = nk * 128 + nv * 129;
    float s[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = st[(r0 + i) * 128 + j];
    for (int p = 0; p < np; ++p) {
        const float* rp = pin + (size_t)p * rec;
        const float e = rp[nk * 128 + nv * 128 + h], d = rp[nk * 128 + h * 128 + j];
        const float* kp = rp + kh * 128 + r0;
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = dn_decay(s[i], e);
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = dn_rank1(s[i], kp[i], d);
    }
    if (pout && np > 0) {
#pragma unroll
        for (int i = 0; i < 64; ++i) st[(r0 + i) * 128 + j] = s[i];
    }
    const float* row = qkv;
    f32x4 qn = *(const f32x4*)(row + kh * 128 + 4 * lane), kn = *(const f32x4*)(row + nk * 128 + kh * 128 + 4 * lane);
    float vn = row[nk * 256 + h * 128 + j], gn = g[h], bn = beta[h];
    for (int t = 0; t < T; ++t) {
        const int b = t & 1;
        *(f32x4*)&qk[b][4 * lane] = qn;
        *(f32x4*)&qk[b][128 + 4 * lane] = kn;
        const float v = vn, e = __expf(gn), bt = bn;
        if (t + 1 < T) {
            row = qkv + (size_t)(t + 1) * dim;
            qn = *(const f32x4*)(row + kh * 128 + 4 * lane);
            kn = *(const f32x4*)(row + nk * 128 + kh * 128 + 4 * lane);
            vn = row[nk * 256 + h * 128 + j];
            gn = g[(t + 1) * nv + h];
            bn = beta[(t + 1) * nv + h];
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
        const float* kp = &qk[b][128 + r0];
        const float* qp = &qk[b][r0];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 64; i += 4) {
            const f32x4 k4 = *(const f32x4*)(kp + i);
            s[i] = dn_decay(s[i], e);
            s[i + 1] = dn_decay(s[i + 1], e);
            s[i + 2] = dn_decay(s[i + 2], e);
            s[i + 3] = dn_decay(s[i + 3], e);
            kv += s[i] * k4.x + s[i + 1] * k4.y + s[i + 2] * k4.z + s[i + 3] * k4.w;
        }
        kv += __shfl_xor(kv, 16, 32);
        const float dlt = bt * (v - kv);
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 64; i += 4) {
            const f32x4 k4 = *(const f32x4*)(kp + i);
            const f32x4 q4 = *(const f32x4*)(qp + i);
            s[i] = dn_rank1(s[i], k4.x, dlt);
            s[i + 1] = dn_rank1(s[i + 1], k4.y, dlt);
            s[i + 2] = dn_rank1(s[i + 2], k4.z, dlt);
            s[i + 3] = dn_rank1(s[i + 3], k4.w, dlt);
            o += s[i] * q4.x + s[i + 1] * q4.y + s[i + 2] * q4.z + s[i + 3] * q4.w;
        }
        o += __shfl_xor(o, 16, 32);
        if (lane < 16) out[((size_t)t * nv + h) * 128 + j] = o;
        if (pout) {
            float* rp = pout + (size_t)t * rec;
            if (lane < 16) rp[nk * 128 + h * 128 + j] = dlt;
            if (blockIdx.x % 8 == 0 && h % (nv / nk) == 0) *(f32x4*)(rp + kh * 128 + 4 * lane) = *(const f32x4*)&qk[b][128 + 4 * lane];
            if (blockIdx.x % 8 == 0 && lane == 0) rp[nk * 128 + nv * 128 + h] = e;
        }
    }
    if (!pout) {
#pragma unroll
        for (int i = 0; i < 64; ++i) st[(r0 + i) * 128 + j] = s[i];
    }
}

// Greedy draft from logits: out = (argmax as float bits, softmax probability of it); one 1024-thread block.
extern "C" __global__ void __launch_bounds__(1024) logits_top1(float* out, const float* l, int n) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float m = -INFINITY;
    int mi = 0;
    for (int i = threadIdx.x; i < n; i += 1024)
        if (l[i] > m) {
            m = l[i];
            mi = i;
        }
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor(m, o, 32);
        const int oi = __shfl_xor(mi, o, 32);
        if (om > m || (om == m && oi < mi)) {
            m = om;
            mi = oi;
        }
    }
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        sv[w] = m;
        si[w] = mi;
    }
    __syncthreads();
    m = sv[lane];
    mi = si[lane];
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor(m, o, 32);
        const int oi = __shfl_xor(mi, o, 32);
        if (om > m || (om == m && oi < mi)) {
            m = om;
            mi = oi;
        }
    }
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) s += __expf(l[i] - m);
    s = block_sum<1024>(s, sv);
    if (threadIdx.x == 0) {
        out[0] = __int_as_float(mi);
        out[1] = 1.f / s;
    }
}

// out = silu(gate).clamp(max = L) * up.clamp(-L, L): the SwiGLU clamp of ADR 0004 for the shared expert of Ling,
// which is a dense FFN and so never goes through `moe_act`, where the routed experts get their clamp. `out` may
// alias `gate`.
extern "C" __global__ void swiglu_clamp(float* out, const float* gate, const float* up, int n, float lim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i], u = up[i];
    out[i] = fminf(g / (1.f + expf(-g)), lim) * fminf(fmaxf(u, -lim), lim);
}
