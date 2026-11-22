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

// ---- Inkling-2's tower (issue #74) ----

// The 40x40 tiles the processor hands over become one row per native patch, in the tower's order: tile-major, then
// the `merge x merge` patches inside it. A tile is [40][40][3] row-major (channel last); a patch is [20][20][3].
extern "C" __global__ void __launch_bounds__(256) inkling2_vision_split(float* out, const float* tiles, int patches,
                                                                      int tile_side, int patch_side, int merge) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    const int dim = patch_side * patch_side * 3;
    if (i >= (long long)patches * dim) return;
    const int p = (int)(i / dim), e = (int)(i % dim);
    // Which tile, and which sub-patch of it, this row and element belong to.
    const int per = merge * merge, tile = p / per, sub = p % per;
    const int rows = patch_side * patch_side;
    const int y = (e / 3) / patch_side, x = (e / 3) % patch_side, ch = e % 3;
    const int sy = (sub / merge) * patch_side + y, sx = (sub % merge) * patch_side + x;
    out[i] = tiles[((size_t)tile * tile_side + sy) * tile_side * 3 + sx * 3 + ch];
}

// The axial RoPE of the tower: the first half of the head dims turns with the column, the second with the row, one
// pair of interleaved dims per frequency (`inv = theta^(-2j/(D/2))`). `hw` is [rows, 2] of (row, col) — 1-based for
// the patches, (0, 0) for the class token.
extern "C" __global__ void __launch_bounds__(128) inkling2_vision_rope(float* x, const int* hw, int rows, int H, int D,
                                                                     float theta) {
    __shared__ float xs[128];
    const int p = blockIdx.x, h = blockIdx.y;
    const int d = threadIdx.x;
    float* base = x + ((size_t)p * H + h) * D;
    // The two halves of a pair are read before anything is written: staging the row keeps a partner from being
    // rotated twice, which is what the in-place version did (the Qwen kernel stages it for the same reason).
    if (d < D) xs[d] = base[d];
    __syncthreads();
    if (d >= D) return;
    const int half = D / 2;
    const int axis = d < half ? 1 : 0;  // 1 = column (w), 0 = row (h)
    const int e = d % half;
    const float pos = (float)hw[p * 2 + axis];
    const float inv = 1.0f / powf(theta, (float)(2 * (e / 2)) / (float)half);
    const float a = pos * inv, co = cosf(a), si = sinf(a);
    const float x0 = xs[d & ~1], x1 = xs[d | 1];
    base[d] = (d & 1) ? x1 * co + x0 * si : x0 * co - x1 * si;
}

// Bidirectional attention of one image: out [rows, H, D] from q, k, v of the same shape. One block per (row, head),
// the eight warps splitting the keys and the partials merged at the end (the same shape as `vision_attention`, which
// needs q, k and v in one fused block; the tower keeps them apart).
extern "C" __global__ void inkling2_vision_attn(float* out, const float* q, const float* k, const float* v, int P,
                                               int H, int D, float scale) {
    __shared__ float s_m[8], s_l[8];
    __shared__ float s_acc[8][96];
    const int p = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float qv[3], acc[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; ++i) {
        const int d = lane + 32 * i;
        qv[i] = d < D ? q[((size_t)p * H + h) * D + d] * scale : 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int j = warp; j < P; j += 8) {
        const float* kr = k + ((size_t)j * H + h) * D;
        float s = 0.f;
        for (int i = 0; i < 3; ++i) {
            const int d = lane + 32 * i;
            if (d < D) s += qv[i] * kr[d];
        }
        s = warp_sum(s);
        const float mn = fmaxf(m, s), c = __expf(m - mn), pr = __expf(s - mn);
        l = l * c + pr;
        const float* vr = v + ((size_t)j * H + h) * D;
        for (int i = 0; i < 3; ++i) {
            const int d = lane + 32 * i;
            acc[i] = acc[i] * c + (d < D ? pr * vr[d] : 0.f);
        }
        m = mn;
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
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

// x[i] += lambda[i % n] * y[i]:// x[i] += lambda[i % n] * y[i]: the tower's per-channel layer scale before the residual add.
extern "C" __global__ void inkling2_vision_scale_add(float* x, const float* y, const bf16_t* lambda, long long total,
                                                     int n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i < total) x[i] += bf2f(lambda[i % n]) * y[i];
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
