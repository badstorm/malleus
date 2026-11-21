#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Hy4-flash (issue #40): iHC, the interleaved M-RoPE of its gated MLA, and the attention with the learned sink.
//
// The shapes are the model's, not a special case of Ling's: the rope rotates half-split pairs (Qwen2-VL layout, not
// the consecutive pairs of `rope_ilv`), the value width differs from the nope width, and every head has a learned
// logit that joins the softmax before any key does.

// iHC pre-mix: streams [t, c, h] fp32 -> post [t, c] (the residual gate) and out [t, h] (the block input).
//   flat = streams.flatten(1);  norm = rsqrt(mean(flat^2) + eps)
//   pre  = sigmoid((fn[0:c] @ flat) * scale[0] + base[0:c]) + eps
//   post = magnitude * sigmoid((fn[c:2c] @ flat) * scale[1] + base[c:2c]) + eps
//   out  = sum_c pre[c] * streams[:, c, :]
extern "C" __global__ void __launch_bounds__(256) ihc_pre(float* out, float* post, const float* streams,
                                                          const float* fn, const float* base, const float* scale,
                                                          int t, int c, int h, float magnitude, float eps) {
    extern __shared__ float sh[];
    float* m = sh;          // [2 * c]
    float* red = sh + 2 * c; // [256]
    const int ti = blockIdx.x;
    const int tid = threadIdx.x;
    const float* flat = streams + (size_t)ti * c * h;
    const int k = c * h;
    float ss = 0.f;
    for (int i = tid; i < k; i += blockDim.x) ss += flat[i] * flat[i];
    red[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float norm = rsqrtf(red[0] / (float)k + eps);
    // The 2c mixes: one block-wide dot product each.
    for (int o = 0; o < 2 * c; ++o) {
        const float* row = fn + (size_t)o * k;
        float acc = 0.f;
        for (int i = tid; i < k; i += blockDim.x) acc += row[i] * flat[i];
        red[tid] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        m[o] = red[0] * norm;
        __syncthreads();
    }
    if (tid < c) {
        post[ti * c + tid] = magnitude * (1.f / (1.f + expf(-(m[c + tid] * scale[1] + base[c + tid])))) + eps;
        m[tid] = 1.f / (1.f + expf(-(m[tid] * scale[0] + base[tid]))) + eps;
    }
    __syncthreads();
    for (int d = tid; d < h; d += blockDim.x) {
        float acc = 0.f;
        for (int cc = 0; cc < c; ++cc) acc += m[cc] * flat[cc * h + d];
        out[(size_t)ti * h + d] = acc;
    }
}

// The residual of one block: out[ti, c, h] = post[ti, c] * x[ti, h] + res[ti, c, h], in fp32 then cast.
extern "C" __global__ void __launch_bounds__(256) ihc_res(float* out, const float* x, const float* post,
                                                          const float* res, int n, int c, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / (c * h);
    const int r = i % (c * h);
    const int ci = r / h;
    out[i] = post[t * c + ci] * x[t * h + r % h] + res[i];
}

// Interleaved 3-axis RoPE, half-split pairs. `q` is [t, nh, nope + rd] and the rotary slice of head h starts at
// nope; `kv` is [t, kvl + rd] and its rotary slice starts at kvl (one shared k_rot, as the MQA latent is).
// Frequency r (< rd/2) reads the height plane at r = 1 (mod 3) and the width plane at r = 2 (mod 3) inside the
// first `mrope_section * 3` slots, and the time plane everywhere else; the pair is (r, r + rd/2).
extern "C" __global__ void __launch_bounds__(256) hy4_rope(float* q, float* kv, const int* pos, int t, int nh,
                                                           int nope, int kvl, int rd, int sec_h, int sec_w,
                                                           float theta) {
    const int half = rd / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * (nh + 1) * half) return;
    const int ti = i / ((nh + 1) * half);
    const int r = i % ((nh + 1) * half);
    const int h = r / half, f = r % half;
    float* p = h < nh ? q + (size_t)ti * nh * (nope + rd) + h * (nope + rd) + nope
                      : kv + (size_t)ti * (kvl + rd) + kvl;
    int axis = 0;
    if (f % 3 == 1 && f < sec_h * 3) axis = 1;
    else if (f % 3 == 2 && f < sec_w * 3) axis = 2;
    // `apply_rotary_pos_emb` with the repeat-interleaved cos/sin rotates the ADJACENT pair (2f, 2f+1) with the
    // frequency f, not (f, f + half): each half of the rope dims holds the two members of the 32 interleaved pairs.
    const float inv = 1.0f / powf(theta, (float)(2 * f) / (float)rd);
    const float ang = (float)pos[ti * 3 + axis] * inv;
    const float c = cosf(ang), s = sinf(ang);
    const float x0 = p[2 * f], x1 = p[2 * f + 1];
    p[2 * f] = x0 * c - x1 * s;
    p[2 * f + 1] = x1 * c + x0 * s;
}

// Gated MLA with the learned sink. One block per (head, token): `nv` threads, dynamic shared of `nk` scores plus
// a 256-wide reduction scratch.
//   q    [t, nh, nope + rd]  the rotated query
//   kv   [nk, nh, nope + v]  kv_b applied to the latent of every visible token
//   krot [nk, rd]            the latent's rotated dims (one shared k_rot, not per head)
//   sinh [nh]                the learned sink logit, which joins the softmax as an extra key
//   gate [t, nh, v]          sigmoid() of this is the output gate
extern "C" __global__ void __launch_bounds__(256) hy4_attn(float* out, const float* q, const float* kv,
                                                           const float* krot, const float* sinh, const float* gate,
                                                           int len, int nh, int nope, int rd, int v, float scale) {
    extern __shared__ float sh[];
    float* sc = sh;
    float* red = sh + 4096;
    const int ti = blockIdx.y, h = blockIdx.x;
    const int tid = threadIdx.x;
    const int qk = nope + rd;
    const float* qh = q + (size_t)ti * nh * qk + (size_t)h * qk;
    const int nk = len + ti + 1;
    float mx = sinh[h];
    for (int j = tid; j < nk; j += blockDim.x) {
        const float* kj = kv + ((size_t)j * nh + h) * (nope + v);
        const float* rj = krot + (size_t)j * rd;
        float acc = 0.f;
        for (int d = 0; d < nope; ++d) acc += qh[d] * kj[d];
        for (int r = 0; r < rd; ++r) acc += qh[nope + r] * rj[r];
        sc[j] = acc * scale;
        mx = fmaxf(mx, sc[j]);
    }
    red[tid] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    mx = red[0];
    // The sink is one extra key, not one per thread: only the thread that reduces the sum may add it, or the
    // block-wide reduction counts it blockDim.x times.
    float sum = tid == 0 ? expf(sinh[h] - mx) : 0.f;
    for (int j = tid; j < nk; j += blockDim.x) {
        sc[j] = expf(sc[j] - mx);
        sum += sc[j];
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float den = red[0];
    const int d = tid;
    if (d < v) {
        float acc = 0.f;
        for (int j = 0; j < nk; ++j) {
            const float* vj = kv + ((size_t)j * nh + h) * (nope + v) + nope;
            acc += sc[j] * vj[d];
        }
        const size_t o = ((size_t)ti * nh + h) * v + d;
        out[o] = acc / den * (1.f / (1.f + expf(-gate[o])));
    }
}

// fp16 -> fp32 for the cached latents (the cache is fp16, the GEMMs read fp32).
extern "C" __global__ void __launch_bounds__(256) hy4_cast_f32(float* out, const __half* src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(src[i]);
}

// The final stream mix of the head: out = sum_c sigmoid((fn[c] @ flat) * norm * scale + base[c]) * streams[:, c, :].
extern "C" __global__ void __launch_bounds__(256) ihc_head(float* out, const float* streams, const float* fn,
                                                           const float* base, const float* scale, int t, int c,
                                                           int h, float eps) {
    extern __shared__ float sh[];
    float* red = sh;
    float* m = sh + 256;
    const int ti = blockIdx.x;
    const int tid = threadIdx.x;
    const float* flat = streams + (size_t)ti * c * h;
    const int k = c * h;
    float ss = 0.f;
    for (int i = tid; i < k; i += blockDim.x) ss += flat[i] * flat[i];
    red[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float norm = rsqrtf(red[0] / (float)k + eps);
    for (int o = 0; o < c; ++o) {
        const float* row = fn + (size_t)o * k;
        float acc = 0.f;
        for (int i = tid; i < k; i += blockDim.x) acc += row[i] * flat[i];
        red[tid] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) m[o] = 1.f / (1.f + expf(-(red[0] * norm * scale[0] + base[o])));
        __syncthreads();
    }
    for (int d = tid; d < h; d += blockDim.x) {
        float acc = 0.f;
        for (int cc = 0; cc < c; ++cc) acc += m[cc] * flat[cc * h + d];
        out[(size_t)ti * h + d] = acc;
    }
}

// The routed expert's SwiGLU, as the reference writes it: `silu(clamp(gate, max = L)) * clamp(up, -L, L)`. The gate is
// clamped before the silu, not after.
__device__ __forceinline__ float hy4_swiglu(float g, float u, float limit) {
    const float gc = fminf(g, limit);
    return gc / (1.f + expf(-gc)) * fmaxf(-limit, fminf(u, limit));
}

extern "C" __global__ void __launch_bounds__(256) hy4_glu(float* y, const float* gu, int inter, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    y[i] = hy4_swiglu(gu[i], gu[inter + i], limit);
}

// The same over every row of the grouped gate|up GEMM's output (`ms.gu` of the prefill: [rows, 2 * inter]) and without
// the rotation, which the int8 quantizer of the down GEMM applies.
extern "C" __global__ void __launch_bounds__(256) hy4_glu_rows(float* y, const float* gu, int inter, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= inter) return;
    const float* g = gu + (size_t)r * 2 * inter;
    y[(size_t)r * inter + i] = hy4_swiglu(g[i], g[inter + i], limit);
}

// out[i] += w * y[i]: one (token, slot) pair added into its token's latent row.
extern "C" __global__ void __launch_bounds__(256) hy4_moe_acc(float* out, const float* y, float w, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h) out[i] += w * y[i];
}

// One routed expert's GEMV on MLS4 codes: y[o] = sum_k xr[k] * (sign_extend4(nibble) * scale[o][k / 128]), with the
// input already rotated by `hadamard128` (the weights are stored in the rotated basis). One block per row.
extern "C" __global__ void __launch_bounds__(256) hy4_expert_gemv(float* y, const float* xr,
                                                                  const unsigned char* q, const __half* sc,
                                                                  int n, int k, int acc_in) {
    const int o = blockIdx.x;
    if (o >= n) return;
    const int g = k / 128;
    const int packed = k / 2;
    const unsigned char* row = q + (size_t)o * packed;
    float acc = 0.f;
    for (int i = threadIdx.x; i < packed; i += blockDim.x) {
        const float s = __half2float(sc[(size_t)o * g + (2 * i) / 128]);
        const int v0 = ((signed char)(row[i] << 4)) >> 4;
        const int v1 = ((signed char)row[i]) >> 4;
        acc += (xr[2 * i] * v0 + xr[2 * i + 1] * v1) * s;
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (acc_in) y[o] += red[0];
        else y[o] = red[0];
    }
}

// The vision tower: LayerNorm (with bias, mean subtracted), exact GELU, bias and residual adds, the full
// per-image attention of the ViT, and the 2x2 pack the patch merger's convolution reads.
__device__ __forceinline__ float hy4_bf(unsigned short v) { return __uint_as_float((unsigned)v << 16); }
__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ float warp_sum(float v) {
    // The butterfly keeps every lane in range: `__shfl_down` at o=16 asks lanes 16..31 for lane+16 (>= 32),
    // whose value is undefined on gfx11 and was observed as 0, leaving each lane with a partial sum of its own
    // 16-dim chunk and a different softmax weight.
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

extern "C" __global__ void __launch_bounds__(256) hy4_ln(float* out, const float* x, const unsigned short* w,
                                                         const unsigned short* b, int n) {
    __shared__ float sh[256];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = out + (size_t)blockIdx.x * n;
    float s = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s += xr[i];
        s2 += xr[i] * xr[i];
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    const float mean = sh[0] / n;
    __syncthreads();
    sh[threadIdx.x] = s2;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    const float var = sh[0] / n - mean * mean;
    const float inv = rsqrtf(var + 1e-5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = (xr[i] - mean) * inv * hy4_bf(w[i]) + hy4_bf(b[i]);
    }
}

extern "C" __global__ void __launch_bounds__(256) hy4_gelu(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.7071067811865476f));
}

// out[i] += bias[i % n], over rows rows: the bias of the projections the GEMM does not carry.
extern "C" __global__ void __launch_bounds__(256) hy4_bias(float* out, const unsigned short* b, int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] += hy4_bf(b[i % n]);
}

extern "C" __global__ void __launch_bounds__(256) hy4_add(float* dst, const float* src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// Full attention of the ViT: no causal mask, no sink, one block per (token, head).
extern "C" __global__ void __launch_bounds__(256) hy4_vit_attn(float* out, const float* q, const float* k,
                                                              const float* v, int t, int nh, int dim, float scale) {
    extern __shared__ float sh[];
    float* sc = sh;
    float* red = sh + 4096;
    const int ti = blockIdx.y, h = blockIdx.x;
    const int tid = threadIdx.x;
    const float* qh = q + ((size_t)ti * nh + h) * dim;
    const float* kh = k + (size_t)h * dim;
    float mx = -INFINITY;
    for (int j = tid; j < t; j += blockDim.x) {
        const float* kj = kh + (size_t)j * nh * dim;
        float acc = 0.f;
        for (int d = 0; d < dim; ++d) acc += qh[d] * kj[d];
        sc[j] = acc * scale;
        mx = fmaxf(mx, sc[j]);
    }
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.f;
    for (int j = tid; j < t; j += blockDim.x) {
        sc[j] = expf(sc[j] - mx);
        sum += sc[j];
    }
    red[tid] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float den = red[0];
    const int d = tid;
    if (d < dim) {
        float acc = 0.f;
        for (int j = 0; j < t; ++j) acc += sc[j] * v[((size_t)j * nh + h) * dim + d];
        out[((size_t)ti * nh + h) * dim + d] = acc / den;
    }
}

// Packs the 2x2 blocks the patch merger's 2x2 convolution reads: out[mp, ch * 4 + dy * 2 + dx] = x[row * 2 + dy,
// col * 2 + dx, ch], with mp = row * cols + col (the merged grid).
extern "C" __global__ void __launch_bounds__(256) hy4_merge_pack(float* out, const float* x, int hidden, int rows,
                                                                int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols * hidden * 4;
    if (i >= total) return;
    const int ch = i % hidden, rest = i / hidden;
    const int pos = rest % 4, mp = rest / 4;
    const int row = mp / cols, col = mp % cols;
    const int dy = pos / 2, dx = pos % 2;
    const int src = (((row * 2 + dy) * cols * 2) + col * 2 + dx) * hidden + ch;
    out[(size_t)mp * hidden * 4 + ch * 4 + dy * 2 + dx] = x[src];
}

// Absorbed MLA. The cache holds the latent (512) and the rotated dims (64); `kv_b_proj` is read once into the query
// instead of expanding K and V for every token, so a step costs 1.1 KiB per cached token and not `heads * (nope + v)`.
// The per-head stride of `kv_b` is `nope + v` (Hy4's V is wider than its nope), which is where Ling's kernels differ.
extern "C" __global__ void __launch_bounds__(256) hy4_absorb(float* qt, const float* q, const unsigned short* wkb,
                                                            int t, int nh, int nope, int kv, int rd, int vo) {
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int w = blockIdx.x * 8 + wl;
    if (w >= t * nh) return;
    const int ti = w / nh, h = w % nh;
    const int d0 = lane * 16;
    const float* qh = q + (size_t)ti * nh * (nope + rd) + (size_t)h * (nope + rd);
    const unsigned short* wk = wkb + (size_t)h * vo * kv;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int n = 0; n < nope; ++n) {
        const float qn = qh[n];
        const unsigned short* row = wk + (size_t)n * kv + d0;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += qn * hy4_bf(row[i]);
    }
    float* dst = qt + (size_t)w * kv + d0;
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = acc[i];
}

// One split of one (token, head): the block's eight warps split the token's keys, each with an online softmax and the
// weighted latent. The learned sink joins as one extra key, only in the first split.
extern "C" __global__ void __launch_bounds__(256) hy4_mla_split(float* part, const float* qt, const float* q,
                                                               const __half* ckv, const __half* krot,
                                                               const float* sinh, int len, int t, int nh, int nope,
                                                               int kv, int rd, float scale, int nsplit, int window) {
    __shared__ float qr[8][64];
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int th = blockIdx.x;
    const int ti = th / nh, h = th % nh;
    const int split = wl;
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
    const int per = (nkeys + nsplit - 1) / nsplit;
    // The sliding layers see only the last `window` keys, counting the query's own position.
    const int first = window > 0 ? max(0, nkeys - window) : 0;
    const int k0 = max(first, split * per), k_end = min(nkeys, k0 + per);
    float m = split == 0 ? sinh[h] : -INFINITY;
    float l = split == 0 ? 1.f : 0.f;
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
    float* dst = part + ((size_t)th * nsplit + split) * (kv + 2);
    if (lane == 0) {
        dst[0] = m;
        dst[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[2 + d0 + i] = acc[i];
}

// Merge the splits of one (token, head) into the weighted latent, in a global buffer. The sink stays in the
// denominator: the probabilities do not renormalise after it is dropped.
extern "C" __global__ void __launch_bounds__(256) hy4_mla_merge(float* lat_out, const float* part, int kv,
                                                               int nsplit) {
    const int th = blockIdx.x;
    const int d = threadIdx.x;
    const float* src = part + (size_t)th * nsplit * (kv + 2);
    float M = -INFINITY;
    for (int i = 0; i < nsplit; ++i) M = fmaxf(M, src[(size_t)i * (kv + 2)]);
    float L = 0.f;
    for (int i = 0; i < nsplit; ++i) {
        const float mi = src[(size_t)i * (kv + 2)];
        if (mi == -INFINITY) continue;
        L += src[(size_t)i * (kv + 2) + 1] * expf(mi - M);
    }
    float a = 0.f, b = 0.f;
    for (int i = 0; i < nsplit; ++i) {
        const float mi = src[(size_t)i * (kv + 2)];
        if (mi == -INFINITY) continue;
        const float f = expf(mi - M);
        a += src[(size_t)i * (kv + 2) + 2 + d] * f;
        b += src[(size_t)i * (kv + 2) + 2 + d + 256] * f;
    }
    lat_out[(size_t)th * kv + d] = L > 0.f ? a / L : 0.f;
    if (d + 256 < kv) lat_out[(size_t)th * kv + d + 256] = L > 0.f ? b / L : 0.f;
}

// The output gate of the gated MLA: out[o] *= sigmoid(gate[o]). It follows the V half of `kv_b_proj`, which is a
// GEMM now (the per-(token, head) GEMV read the 256 rows of a head one per thread, 32 cache lines per instruction).
extern "C" __global__ void __launch_bounds__(256) hy4_gate_mul(float* out, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-gate[i]));
}

// ---------------------------------------------------------------------------------------------------------------
// The compressed indexer of the indexed layers (issue #40). Every `pool` consecutive tokens become one group, so a
// query scores `groups` keys instead of `tokens` ones; the `index_topk / pool - 1` best complete groups plus the
// query's own group are the keys its attention then walks.
//
// Token keys: `token_key_projection` then its RMSNorm, cached as fp16 per layer like the KV latents. A group is
// pooled once, when its last token arrives (the reference pools the whole cache every step; the cache of a closed
// group never changes again): the two learned fp32 projections of the pooling, the per-slot `intra_group_ape`
// bias, a softmax over the slots, then `key_norm`. The query and the group keys both rotate on their trailing `rd`
// dims with the interleaved 3-D M-RoPE of the backbone, each at its own positions (the group at its first token).

// Token keys into the fp16 cache: `out = rmsnorm(x) * w`, rows `len .. len + t`, one block per token.
extern "C" __global__ void __launch_bounds__(256) hy4_ix_keys(__half* cache, const float* tk, const float* w,
                                                             int len, int t, int dim, float eps) {
    __shared__ float red[256];
    const int ti = blockIdx.x, tid = threadIdx.x;
    const float* row = tk + (size_t)ti * dim;
    float ss = 0.f;
    for (int c = tid; c < dim; c += blockDim.x) ss += row[c] * row[c];
    red[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)dim + eps);
    __half* dst = cache + (size_t)(len + ti) * dim;
    for (int c = tid; c < dim; c += blockDim.x) dst[c] = __float2half(row[c] * inv * w[c]);
}

// One pooled group `g0 + blockIdx.x`: `dim` threads, one output dim each. `xs` is the group's `pool` token keys.
//   value[p] = key_value_projection @ xs[p]        gate[p] = key_gate_projection @ xs[p] + intra_group_ape[p]
//   pooled   = sum_p softmax(gate)[p] * value[p]   out     = key_norm(pooled)
extern "C" __global__ void __launch_bounds__(128) hy4_ix_pool(__half* gk, const __half* tk, const float* wv,
                                                             const float* wg, const float* ape, const float* wn,
                                                             int g0, int ng, int dim, int pool, float eps) {
    extern __shared__ float sh[];
    float* xs = sh;              // [pool, dim]
    float* red = sh + pool * dim; // [128]
    const int g = g0 + blockIdx.x, d = threadIdx.x;
    for (int p = 0; p < pool; ++p) {
        const __half* src = tk + (size_t)(g * pool + p) * dim;
        for (int c = d; c < dim; c += blockDim.x) xs[p * dim + c] = __half2float(src[c]);
    }
    __syncthreads();
    float val[8], gate[8], mx = -INFINITY;
    for (int p = 0; p < pool; ++p) {
        const float* rv = wv + (size_t)d * dim;
        const float* rg = wg + (size_t)d * dim;
        const float* x = xs + p * dim;
        float av = 0.f, ag = 0.f;
        for (int c = 0; c < dim; ++c) {
            const float xc = x[c];
            av += rv[c] * xc;
            ag += rg[c] * xc;
        }
        val[p] = av;
        gate[p] = ag + ape[p * dim + d];
        mx = fmaxf(mx, gate[p]);
    }
    float sum = 0.f;
    for (int p = 0; p < pool; ++p) {
        gate[p] = expf(gate[p] - mx);
        sum += gate[p];
    }
    float pooled = 0.f;
    for (int p = 0; p < pool; ++p) pooled += gate[p] / sum * val[p];
    red[d] = pooled * pooled;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    gk[(size_t)g * dim + d] = __float2half(pooled * rsqrtf(red[0] / (float)dim + eps) * wn[d]);
}

// The indexer's rotary: the trailing `rd` dims of every row, the adjacent pair (2f, 2f+1) at frequency f, the plane
// chosen by `f` exactly as `hy4_rope` does it (the same M-RoPE, so the phases match the backbone's).
extern "C" __global__ void __launch_bounds__(256) hy4_ix_qrope(float* q, const int* pos, int t, int nh, int dim,
                                                              int rd, int sec_h, int sec_w, float theta) {
    const int half = rd / 2, base = dim - rd;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * nh * half) return;
    const int row = i / half, f = i % half, ti = row / nh;
    float* p = q + (size_t)row * dim + base;
    int axis = 0;
    if (f % 3 == 1 && f < sec_h * 3) axis = 1;
    else if (f % 3 == 2 && f < sec_w * 3) axis = 2;
    const float inv = 1.0f / powf(theta, (float)(2 * f) / (float)rd);
    const float ang = (float)pos[ti * 3 + axis] * inv;
    const float c = cosf(ang), s = sinf(ang);
    const float x0 = p[2 * f], x1 = p[2 * f + 1];
    p[2 * f] = x0 * c - x1 * s;
    p[2 * f + 1] = x1 * c + x0 * s;
}

// The same for the group keys, which carry the position of their first token: rows `g0 .. g0 + ng`. They live in
// the fp16 cache, and the rotation happens there like the reference's own bf16 round trip on the pooled key.
extern "C" __global__ void __launch_bounds__(256) hy4_ix_krope(__half* gk, const int* pos, int g0, int ng, int pool,
                                                              int dim, int rd, int sec_h, int sec_w, float theta) {
    const int half = rd / 2, base = dim - rd;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng * half) return;
    const int g = g0 + i / half, f = i % half;
    __half* p = gk + (size_t)g * dim + base;
    const int* gp = pos + (size_t)g * pool * 3;
    int axis = 0;
    if (f % 3 == 1 && f < sec_h * 3) axis = 1;
    else if (f % 3 == 2 && f < sec_w * 3) axis = 2;
    const float inv = 1.0f / powf(theta, (float)(2 * f) / (float)rd);
    const float ang = (float)gp[axis] * inv;
    const float c = cosf(ang), s = sinf(ang);
    const float x0 = __half2float(p[2 * f]), x1 = __half2float(p[2 * f + 1]);
    p[2 * f] = __float2half(x0 * c - x1 * s);
    p[2 * f + 1] = __float2half(x1 * c + x0 * s);
}

// The group scores of one query row: `sum_h head_w[h] * relu(q[h] . group_key)`, written order-preserving (a
// positive float compares as its bits, a negative one as their complement) so the selection can radix them, and 0
// for the groups a causal query cannot see (`g >= own`, where `own` is the query's own group).
//   one block per (query row, tile of 256 groups)
extern "C" __global__ void __launch_bounds__(256) hy4_ix_score(unsigned* keys, const float* q, const __half* gk,
                                                             const float* hw, int len, int ti0, int ncg, int nq,
                                                             int nh, int dim, int pool, float scale) {
    extern __shared__ float sh[];
    float* qs = sh;          // [nh, dim]
    float* hws = sh + nh * dim; // [nh]
    const int i = blockIdx.y, tid = threadIdx.x;
    const int ti = ti0 + i;
    for (int e = tid; e < nh * dim; e += blockDim.x) qs[e] = q[(size_t)ti * nh * dim + e];
    for (int e = tid; e < nh; e += blockDim.x) hws[e] = hw[e] * scale;
    __syncthreads();
    const int own = (len + ti) / pool;
    const int g = blockIdx.x * 256 + tid;
    if (g >= ncg) return;
    const __half* k = gk + (size_t)g * dim;
    float s = 0.f;
    for (int h = 0; h < nh; ++h) {
        const float* qh = qs + h * dim;
        float d = 0.f;
        for (int c = 0; c < dim; ++c) d += qh[c] * __half2float(k[c]);
        s += hws[h] * fmaxf(d, 0.f);
    }
    unsigned u = __float_as_uint(s);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    keys[(size_t)i * ncg + g] = g < own ? u : 0u;
}

// Keep the `nsel` best `u` of the `own` groups a query can see, then its own group; unused slots are -1. The radix
// is the one `idx_select` uses (four 8-bit passes bracket the K-th value, then one pass takes everything above it
// and enough of the ties), because a full sort of 32k groups per query is what the selection must not cost.
extern "C" __global__ void __launch_bounds__(256) hy4_ix_select(int* list, const unsigned* keys, int len, int ti0,
                                                               int pool, int ncg, int nsel, int out_len) {
    __shared__ int hist[256];
    __shared__ unsigned prefix_s, need_s, cnt_s;
    __shared__ int ties_before[256];
    const int i = blockIdx.x, tid = threadIdx.x;
    const int own = min((len + ti0 + i) / pool, ncg);
    int* row = list + (size_t)(ti0 + i) * out_len;
    const unsigned* x = keys + (size_t)i * ncg;
    for (int w = tid; w < out_len; w += blockDim.x) row[w] = -1;
    if (own <= nsel) {
        for (int g = tid; g < own; g += blockDim.x) row[g] = g;
        if (tid == 0) row[own < out_len ? own : out_len - 1] = own;
        return;
    }
    unsigned prefix = 0u, mask = 0u, need = (unsigned)nsel;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
        for (int g = tid; g < own; g += blockDim.x) {
            const unsigned v = x[g];
            if ((v & mask) == prefix) atomicAdd(&hist[(v >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, bin = 255;
            for (; bin > 0; --bin) {
                if (acc + hist[bin] >= (int)need) break;
                acc += hist[bin];
            }
            prefix_s = prefix | ((unsigned)bin << shift);
            need_s = need - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        need = need_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // Everything above the K-th value, plus the first `need` groups equal to it.
    const int per = (own + 255) / 256, g0 = tid * per, g1 = min(own, g0 + per);
    int ties = 0;
    for (int g = g0; g < g1; ++g) ties += x[g] == prefix;
    ties_before[tid] = ties;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int j = 0; j < 256; ++j) {
            const int c = ties_before[j];
            ties_before[j] = acc;
            acc += c;
        }
        cnt_s = 0u;
    }
    __syncthreads();
    int tb = ties_before[tid];
    for (int g = g0; g < g1; ++g) {
        const unsigned v = x[g];
        bool take = v > prefix;
        if (v == prefix) take = tb++ < (int)need;
        if (take) {
            const unsigned o = atomicAdd(&cnt_s, 1);
            if (o < (unsigned)out_len) row[o] = g;
        }
    }
    __syncthreads();
    if (tid == 0) row[nsel < out_len ? nsel : out_len - 1] = own;
}

// The split of the gated MLA over the groups a query recalled: `hy4_mla_split` with the keys taken from the list
// (a group is `pool` consecutive cached latents; an entry -1 is an unused slot, and the tokens past the query's own
// position are skipped, which is what makes the reference's `clamp(max = kv_length - 1)` unnecessary). Every group
// is distinct, so nothing is counted twice. The sink is still one extra key in the first split.
extern "C" __global__ void __launch_bounds__(256) hy4_mla_split_ix(float* part, const float* qt, const float* q,
                                                                 const __half* ckv, const __half* krot,
                                                                 const int* list, const float* sinh, int len,
                                                                 int t, int nh, int nope, int kv, int rd, float scale,
                                                                 int nsplit, int pool, int nsel) {
    __shared__ float qr[8][64];
    const int wl = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int th = blockIdx.x;
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
    const int* row = list + (size_t)ti * nsel;
    float m = wl == 0 ? sinh[h] : -INFINITY;
    float l = wl == 0 ? 1.f : 0.f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int e = wl; e < nsel; e += nsplit) {
        const int g = row[e];
        if (g < 0) continue;
        for (int p = 0; p < pool; ++p) {
            const int j = g * pool + p;
            if (j >= nkeys) break;
            const __half* ck = ckv + (size_t)j * kv + d0;
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) s += qtv[i] * h2f(ck[i]);
            for (int r = lane * 2; r < (lane + 1) * 2 && r < rd; ++r) s += qr[wl][r] * h2f(krot[(size_t)j * rd + r]);
            s = warp_sum(s) * scale;
            const float mn = fmaxf(m, s);
            const float f = expf(m - mn);
            const float pr = expf(s - mn);
            l = l * f + pr;
            m = mn;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = acc[i] * f + pr * h2f(ck[i]);
        }
    }
    float* dst = part + ((size_t)th * nsplit + wl) * (kv + 2);
    if (lane == 0) {
        dst[0] = m;
        dst[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[2 + d0 + i] = acc[i];
}
