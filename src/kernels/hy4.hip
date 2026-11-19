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
