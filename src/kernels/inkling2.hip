#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Inkling-2's short convolution over the key and value streams (issue #62). A causal depthwise conv of kernel 4,
// with the three past tokens carried in a state buffer so decode and prefill share one path. The layer's own
// convolution (`attn_sconv`, `mlp_sconv`) is kernel 4 as well and runs through the same code.
//
// Two variants, as the reference graph has them:
//   static   the taps are the same for every token:      y = (x + conv) / sqrt(5)
//   dynamic  the taps are the static ones plus a per-token, per-head projection:  y = (x + conv) / 3
// The divisors differ because the reference divides by sqrt(conv_kernel_size + 1) in the static variant and by
// sqrt(2 * conv_kernel_size + 1) in the dynamic one; the golden comparison is what settles it.
typedef unsigned short ik_bf16;

__device__ __forceinline__ float ik_bf2f(ik_bf16 b) {
    return __uint_as_float(((unsigned)b) << 16);
}

// y[t][c] = (x[t][c] + sum_j w[c][j] * x[t-3+j][c]) / div, with x[-1..-3] from `past` ([3, C], oldest first).
// w is [C, 4] bf16 in the checkpoint's [C, 1, 4] order; dyn_w is [T, heads * 4] f32, or null in the static variant.
extern "C" __global__ void __launch_bounds__(256) inkling2_sconv(float* y, const float* x, const ik_bf16* w,
                                                                const float* dyn_w, const float* past, int T, int C,
                                                                int heads, float div) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)T * C) return;
    const int t = (int)(idx / C), c = (int)(idx % C);
    const int h = c / (C / heads);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = t - 3 + j;
        const float v = p >= 0 ? x[(size_t)p * C + c] : past[(size_t)(p + 3) * C + c];
        float wj = ik_bf2f(w[(size_t)c * 4 + j]);
        if (dyn_w) wj += dyn_w[(size_t)t * heads * 4 + (size_t)h * 4 + j];
        acc += wj * v;
    }
    y[idx] = (x[idx] + acc) / div;
}

// The per-head transforms Inkling-2 applies before and around its attention (issue #62).
//
// A "row" below is one head vector: `dim` values of one (token, head). The reference scales by a learned per-(head,
// dim) scale, clamps the L2 norm to a threshold, and (full layers) multiplies by tau, the log-scaled position
// factor `1 + alpha * log(max(1, (pos + 1) / floor))`.

// y = x * min(1, thr / (|x| + 1e-6)) * scale[i], then * tau[t] when tau is not null. Row-major [T, heads, dim].
extern "C" __global__ void __launch_bounds__(128) inkling2_l2_clip(float* y, const float* x, const float* scale,
                                                                  const float* tau, int T, int heads, int dim,
                                                                  float thr) {
    __shared__ float ss[128];
    const int row = blockIdx.x;  // token * heads + head
    const int r = threadIdx.x;
    if (r >= dim) return;
    const int t = row / heads;
    const float v = x[(size_t)row * dim + r] * scale[(size_t)(row % heads) * dim + r];
    ss[r] = v * v;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (r < o) ss[r] += ss[r + o];
        __syncthreads();
    }
    const float norm = sqrtf(ss[0]);
    const float k = min(1.f, thr / (norm + 1e-6f)) * (tau ? tau[t] : 1.f);
    y[(size_t)row * dim + r] = v * k;
}

// Unweighted RMSNorm over the last dim: y = x / sqrt(mean(x^2) + eps).
extern "C" __global__ void __launch_bounds__(128) inkling2_rms_unnorm(float* y, const float* x, int rows, int dim,
                                                                     float eps) {
    __shared__ float ss[128];
    const int row = blockIdx.x;
    const int r = threadIdx.x;
    if (r >= dim) return;
    const float v = x[(size_t)row * dim + r];
    ss[r] = v * v;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (r < o) ss[r] += ss[r + o];
        __syncthreads();
    }
    y[(size_t)row * dim + r] = v * rsqrtf(ss[0] / dim + eps);
}

// bias[head][q][s] = rel[q][head] . proj[:, s - q] when 0 <= q - s < rel_extent, else 0.
// rel is [T, heads, d_rel], proj is [d_rel, rel_extent] row-major, bias is [heads, T, T].
extern "C" __global__ void __launch_bounds__(256) inkling2_rel_bias(float* bias, const float* rel, const float* proj,
                                                                   int T, int heads, int d_rel, int extent) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)heads * T * T) return;
    const int h = (int)(idx / ((long long)T * T));
    const int q = (int)((idx % ((long long)T * T)) / T);
    const int s = (int)(idx % T);
    const int d = q - s;
    float acc = 0.f;
    if (d >= 0 && d < extent) {
        const float* rp = rel + ((size_t)q * heads + h) * d_rel;
        for (int i = 0; i < d_rel; ++i) acc += rp[i] * proj[(size_t)i * extent + d];
    }
    bias[idx] = acc;
}

// Causal attention with the relative bias, the learned sink and the output gate: one block per (token, head).
// q, k, v, out and gate are [T, heads|nkv, dim] (kh = h / (heads / nkv) for k and v); bias is [heads, T, T]; `sink`
// is [T, heads] or null. The window is the number of keys a token sees (0 = all of them, the full layers), and the
// scores live in shared memory, so a full layer with a long context needs the tiling of `attention_split` — this
// kernel is the correctness path used to check the pieces, not the long-context one.
extern "C" __global__ void __launch_bounds__(128) inkling2_attn(float* out, const float* q, const float* k,
                                                                const float* v, const float* bias, const float* sink,
                                                                const float* gate, int T, int heads, int nkv, int dim,
                                                                int window, float scale) {
    __shared__ float red[128];
    __shared__ float sc[4096];
    __shared__ float sinkv;
    const int t = blockIdx.x / heads, h = blockIdx.x % heads;
    const int kh = h / (heads / nkv);
    const int r = threadIdx.x;
    const int w0 = window > 0 ? max(0, t - window + 1) : 0;
    const int n = t - w0 + 1;
    if (r == 0) sinkv = sink ? sink[(size_t)t * heads + h] : -INFINITY;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        const int s = w0 + i;
        red[r] = q[((size_t)t * heads + h) * dim + r] * k[((size_t)s * nkv + kh) * dim + r];
        __syncthreads();
        for (int o = 64; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        if (r == 0) sc[i] = red[0] * scale + bias[((size_t)h * T + t) * T + s];
        __syncthreads();
    }
    if (r == 0) {
        float mx = sink ? sinkv : -INFINITY;
        for (int i = 0; i < n; ++i) mx = fmaxf(mx, sc[i]);
        float sum = 0.f;
        for (int i = 0; i < n; ++i) {
            sc[i] = expf(sc[i] - mx);
            sum += sc[i];
        }
        if (sink) sum += expf(sinkv - mx);
        for (int i = 0; i < n; ++i) sc[i] /= sum;
    }
    __syncthreads();
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc += sc[i] * v[((size_t)(w0 + i) * nkv + kh) * dim + r];
    const float g = gate ? gate[((size_t)t * heads + h) * dim + r] : 1.f;
    out[((size_t)t * heads + h) * dim + r] = acc / (1.f + expf(-g));
}

// The attention residual (issue #62): a learned query attends over the running stack of block states plus the
// current partial state. `x` is [T, n, hidden] (the stack of `n` states at every position), `q` is [nq, hidden]
// (1 query per layer, 3 in the full layers), and `out` is [T, nq, hidden]. Every state row is normed with an
// unweighted RMSNorm first, and the softmax is over the n states.
extern "C" __global__ void __launch_bounds__(256) inkling2_attn_res(float* out, const float* q, const float* x, int T,
                                                                   int n, int hidden, int nq, float eps) {
    __shared__ float red[256];
    __shared__ float sc[16];
    __shared__ float fac[16];
    const int t = blockIdx.x / nq, qi = blockIdx.x % nq;
    const int r = threadIdx.x;
    for (int j = 0; j < n; ++j) {
        float s = 0.f;
        for (int i = r; i < hidden; i += 256) {
            const float v = x[((size_t)t * n + j) * hidden + i];
            s += v * v;
        }
        red[r] = s;
        __syncthreads();
        for (int o = 128; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        if (r == 0) fac[j] = rsqrtf(red[0] / (float)hidden + eps);
        __syncthreads();
    }
    for (int j = 0; j < n; ++j) {
        float s = 0.f;
        for (int i = r; i < hidden; i += 256) {
            s += q[qi * hidden + i] * x[((size_t)t * n + j) * hidden + i] * fac[j];
        }
        red[r] = s;
        __syncthreads();
        for (int o = 128; o > 0; o >>= 1) {
            if (r < o) red[r] += red[r + o];
            __syncthreads();
        }
        if (r == 0) sc[j] = red[0] * rsqrtf((float)hidden);
        __syncthreads();
    }
    if (r == 0) {
        float mx = sc[0];
        for (int j = 1; j < n; ++j) mx = fmaxf(mx, sc[j]);
        float sum = 0.f;
        for (int j = 0; j < n; ++j) {
            sc[j] = expf(sc[j] - mx);
            sum += sc[j];
        }
        for (int j = 0; j < n; ++j) sc[j] /= sum;
    }
    __syncthreads();
    for (int i = r; i < hidden; i += 256) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) acc += sc[j] * x[((size_t)t * n + j) * hidden + i] * fac[j];
        out[((size_t)t * nq + qi) * hidden + i] = acc;
    }
}
