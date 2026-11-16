// KDA (Kimi Delta Attention), the linear mixer of Ling-3.0-flash: 35 of its 42 layers, 1,14 GB of the 2,90 read
// per decoded token. The recurrence is the gated delta rule of DeltaNet with one difference that changes the
// kernel: the decay is a diagonal per key channel, not a scalar per head
// (docs/model-ling3-flash.md, § KDA — semantics checked against `fla-core` by `scripts/ling_ref.py selftest`).
//
//   S = S * exp(g_t)[:, None]                 g_t in (-5, 0), one value per key channel
//   S = S + outer(beta_t * k_t, v_t - k_tᵀ S)
//   o_t = q_tᵀ S
//
// q and k arrive already l2-normalized (the conv + norm pass does it, as `conv_qk_norm` does for DeltaNet) and
// `g` is the final log decay, not the raw gate.
//
// Layout, one row per token. q, k and v are three separate buffers (Ling has three projections, and nothing in
// the engine ever packs them into one row: DeltaNet does because Qwen stores one fused in_proj_qkv):
//   q, k, v: [T, nh * 128] each
//   g:     [T, nh, 128]   fp32
//   beta:  [T, nh]        fp32
//   state: [nh, 128, 128] fp32, key channel x value channel, read and written
//   out:   [T, nh, 128]   fp32
//
// Same split as `deltanet_scan`: one wave per (head, 16 value columns); lane (j, r0) keeps state rows [r0, r0+64)
// of value column j in registers, so the whole [128, 128] state of a head lives in one wave's registers and the
// two shuffles at the end of each step are the only cross-lane traffic. The decay vector goes through LDS because
// all 16 lanes of a row group need the same 64 values.
//
// Speculative verify is not wired here yet: `deltanet_scan` records its rank-1 updates instead of applying them,
// and the record of KDA has to carry 128 decay values per token and head rather than one. That is increment I8 of
// docs/ling3-engine.md; until then this kernel always applies the update to the state in place.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f32x4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void __launch_bounds__(32) kda_scan(float* out, const float* q, const float* k,
                                                          const float* v, const float* g, const float* beta,
                                                          float* state, int T, int nh) {
    __shared__ __attribute__((aligned(16))) float qk[2][256];
    __shared__ __attribute__((aligned(16))) float eg[2][128];
    const int lane = threadIdx.x;
    const int h = blockIdx.x / 8;
    const int j = (blockIdx.x % 8) * 16 + lane % 16;
    const int r0 = (lane / 16) * 64;
    const int dim = nh * 128;
    float* st = state + (size_t)h * 128 * 128;
    float s[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        s[i] = st[(r0 + i) * 128 + j];
    }

    const float* row = q;
    const float* grow = g + (size_t)h * 128;
    f32x4 qn = *(const f32x4*)(row + h * 128 + 4 * lane);
    f32x4 kn = *(const f32x4*)(k + h * 128 + 4 * lane);
    float vn = v[h * 128 + j], bn = beta[h];
    f32x4 gv = *(const f32x4*)(grow + 4 * lane);

    for (int t = 0; t < T; ++t) {
        const int b = t & 1;
        *(f32x4*)&qk[b][4 * lane] = qn;
        *(f32x4*)&qk[b][128 + 4 * lane] = kn;
        const f32x4 ev = {__expf(gv.x), __expf(gv.y), __expf(gv.z), __expf(gv.w)};
        *(f32x4*)&eg[b][4 * lane] = ev;
        const float val = vn, bt = bn;
        if (t + 1 < T) {
            const float* nx = q + (size_t)(t + 1) * dim;
            qn = *(const f32x4*)(nx + h * 128 + 4 * lane);
            kn = *(const f32x4*)(k + (size_t)(t + 1) * dim + h * 128 + 4 * lane);
            vn = v[(size_t)(t + 1) * dim + h * 128 + j];
            bn = beta[(t + 1) * nh + h];
            gv = *(const f32x4*)(g + (size_t)(t + 1) * nh * 128 + h * 128 + 4 * lane);
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        const float* kp = &qk[b][128 + r0];
        const float* qp = &qk[b][r0];
        const float* ep = &eg[b][r0];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 64; i += 4) {
            const f32x4 k4 = *(const f32x4*)(kp + i);
            const f32x4 e4 = *(const f32x4*)(ep + i);
            s[i] *= e4.x;
            s[i + 1] *= e4.y;
            s[i + 2] *= e4.z;
            s[i + 3] *= e4.w;
            kv += s[i] * k4.x + s[i + 1] * k4.y + s[i + 2] * k4.z + s[i + 3] * k4.w;
        }
        kv += __shfl_xor(kv, 16, 32);
        const float dlt = bt * (val - kv);

        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 64; i += 4) {
            const f32x4 k4 = *(const f32x4*)(kp + i);
            const f32x4 q4 = *(const f32x4*)(qp + i);
            s[i] += k4.x * dlt;
            s[i + 1] += k4.y * dlt;
            s[i + 2] += k4.z * dlt;
            s[i + 3] += k4.w * dlt;
            o += s[i] * q4.x + s[i + 1] * q4.y + s[i + 2] * q4.z + s[i + 3] * q4.w;
        }
        o += __shfl_xor(o, 16, 32);
        if (lane < 16) {
            out[((size_t)t * nh + h) * 128 + j] = o;
        }
    }

#pragma unroll
    for (int i = 0; i < 64; ++i) {
        st[(r0 + i) * 128 + j] = s[i];
    }
}

// Depthwise causal conv (kernel 4) + SiLU on the three KDA streams, then the L2 norms: q normalized and scaled by
// 1 / sqrt(128), k normalized, v left alone (docs/model-ling3-flash.md, § KDA). Same job as `conv_qk_norm` for
// DeltaNet, on three buffers instead of one packed row: cw is the three conv weights concatenated in that order,
// [3 * C, 4] bf16, and state is [3 * C, 3] fp32.
//
// The state is not updated here: `conv_state_save` runs per stream afterwards on the pre-conv inputs, the same way
// DeltaNet does it.
typedef unsigned short kd_bf16;

__device__ __forceinline__ float kd_bf2f(kd_bf16 b) {
    return __uint_as_float(((unsigned)b) << 16);
}

__device__ __forceinline__ float kd_silu(float x) {
    return x / (1.f + expf(-x));
}

extern "C" __global__ void __launch_bounds__(256) kda_conv_norm(float* oq, float* ok, float* ov, const float* iq,
                                                                const float* ik, const float* iv, const kd_bf16* cw,
                                                                const float* state, int T, int C) {
    const int wv = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const int groups = C / 128;
    const int st = wv / (T * groups);
    const int r = wv % (T * groups);
    if (st >= 3) return;
    const int t = r / groups, grp = r % groups, c = grp * 128 + 4 * lane;
    const float* in = st == 0 ? iq : (st == 1 ? ik : iv);
    float* out = st == 0 ? oq : (st == 1 ? ok : ov);
    const uint2* wp = (const uint2*)(cw + (size_t)(st * C + c) * 4);
    const float* sp = state + (size_t)(st * C + c) * 3;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = t - 3 + j;
        f32x4 x;
        if (p >= 0) {
            x = *(const f32x4*)(in + (size_t)p * C + c);
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) x[e] = sp[e * 3 + p + 3];
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const uint2 wb = wp[e];
            const unsigned short wj = (j < 2 ? (j == 0 ? wb.x : wb.x >> 16) : (j == 2 ? wb.y : wb.y >> 16)) & 0xffff;
            acc[e] += kd_bf2f(wj) * x[e];
        }
    }
    f32x4 y;
#pragma unroll
    for (int e = 0; e < 4; ++e) y[e] = kd_silu(acc[e]);
    if (st < 2) {
        float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
        y *= rsqrtf(ss + 1e-6f) * (st == 0 ? rsqrtf(128.f) : 1.f);
    }
    *(f32x4*)(out + (size_t)t * C + c) = y;
}
