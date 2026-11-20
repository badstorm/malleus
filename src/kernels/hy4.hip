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
    const float inv = 1.0f / powf(theta, (float)(2 * f) / (float)rd);
    const float ang = (float)pos[ti * 3 + axis] * inv;
    const float c = cosf(ang), s = sinf(ang);
    const float x0 = p[f], x1 = p[f + half];
    p[f] = x0 * c - x1 * s;
    p[f + half] = x1 * c + x0 * s;
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

// The routed expert's SwiGLU: y[i] = silu(gu[i]).clamp(max = L) * gu[inter + i].clamp(-L, L).
extern "C" __global__ void __launch_bounds__(256) hy4_glu(float* y, const float* gu, int inter, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float g = gu[i], u = gu[inter + i];
    const float gs = g / (1.f + expf(-g));
    y[i] = fminf(gs, limit) * fmaxf(-limit, fminf(u, limit));
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
