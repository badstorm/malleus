#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint16_t bf16_t;

__device__ __forceinline__ float bf2f(bf16_t v) { return __uint_as_float(((uint32_t)v) << 16); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

__device__ __forceinline__ float block_sum256(float v, float* sh) {
    v = warp_sum(v);
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < 8; ++i) r += sh[i];
    return r;
}

extern "C" __global__ void layernorm(float* out, const float* x, const bf16_t* w, const bf16_t* b, int n, float eps) {
    __shared__ float sh[8];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = out + (size_t)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) s += xr[i];
    const float mean = block_sum256(s, sh) / n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) v += (xr[i] - mean) * (xr[i] - mean);
    const float inv = rsqrtf(block_sum256(v, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = (xr[i] - mean) * inv * bf2f(w[i]) + bf2f(b[i]);
}

// act: 0 = none, 1 = gelu tanh, 2 = gelu erf. y[t, n] = act(y[t, n] + bias[n])
extern "C" __global__ void bias_act(float* y, const bf16_t* bias, int T, int N, int act) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * N) return;
    float v = y[i] + bf2f(bias[i % N]);
    if (act == 1) v = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    else if (act == 2) v = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
    y[i] = v;
}

// In-place 2D rope on q and k of packed qkv [P, 3, H, D] (D = 72): angle(d) uses h for d%(D/2) < D/4, w otherwise.
extern "C" __global__ void vision_rope(float* qkv, const int* hw, int H, int D, float theta) {
    const int p = blockIdx.x, h = blockIdx.y, which = blockIdx.z;
    const int d = threadIdx.x;
    __shared__ float xs[128];
    float* base = qkv + ((size_t)p * 3 + which) * H * D + (size_t)h * D;
    if (d < D) xs[d] = base[d];
    __syncthreads();
    if (d >= D) return;
    const int half = D / 2, quarter = D / 4;
    const int e = d % half;
    const int j = e % quarter;
    const float pos = (float)(e < quarter ? hw[p * 2] : hw[p * 2 + 1]);
    const float inv = 1.0f / powf(theta, (float)(2 * j) / (float)half);
    const float a = pos * inv, c = cosf(a), s = sinf(a);
    base[d] = d < half ? xs[d] * c - xs[d + half] * s : xs[d] * c + xs[d - half] * s;
}

// Bidirectional attention within one image. qkv [P, 3, H, D], out [P, H, D]; D <= 96.
extern "C" __global__ void vision_attention(float* out, const float* qkv, int P, int H, int D, float scale) {
    __shared__ float s_m[8], s_l[8];
    __shared__ float s_acc[8][96];
    const int p = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const size_t row = (size_t)3 * H * D;
    float q[3], acc[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; ++i) {
        const int d = lane + 32 * i;
        q[i] = d < D ? qkv[p * row + (size_t)h * D + d] * scale : 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int j = warp; j < P; j += 8) {
        const float* kr = qkv + j * row + (size_t)H * D + (size_t)h * D;
        float s = 0.f;
        for (int i = 0; i < 3; ++i) {
            const int d = lane + 32 * i;
            if (d < D) s += q[i] * kr[d];
        }
        s = warp_sum(s);
        const float mn = fmaxf(m, s), c = __expf(m - mn), pr = __expf(s - mn);
        l = l * c + pr;
        const float* vr = kr + (size_t)H * D;
        for (int i = 0; i < 3; ++i) {
            const int d = lane + 32 * i;
            acc[i] = acc[i] * c + (d < D ? pr * vr[d] : 0.f);
        }
        m = mn;
    }
    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    for (int i = 0; i < 3; ++i) {
        const int d = lane + 32 * i;
        if (d < 96) s_acc[warp][d] = acc[i];
    }
    __syncthreads();
    const int d = threadIdx.x;
    if (d >= D) return;
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, s_m[w]);
    float L = 0.f, o = 0.f;
    for (int w = 0; w < 8; ++w) {
        if (s_m[w] == -INFINITY) continue;
        const float f = __expf(s_m[w] - M);
        L += s_l[w] * f;
        o += s_acc[w][d] * f;
    }
    out[((size_t)p * H + h) * D + d] = o / L;
}
