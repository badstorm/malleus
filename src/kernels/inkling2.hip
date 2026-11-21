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
