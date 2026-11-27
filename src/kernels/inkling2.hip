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
    for (int lag = 0; lag < 4; ++lag) {
        const int p = t - lag;
        const float v = p >= 0 ? x[(size_t)p * C + c] : past[(size_t)(p + 3) * C + c];
        // The static variant's weight is [C, 4] with tap 0 oldest; the dynamic variant's static_w is [4, C] with
        // tap 0 the current token (the reference's `static_weight.T`). The lag runs 0 = current .. 3 = three back.
        float wj = dyn_w ? ik_bf2f(w[(size_t)lag * C + c]) : ik_bf2f(w[(size_t)c * 4 + (3 - lag)]);
        if (dyn_w) wj += dyn_w[(size_t)t * heads * 4 + (size_t)h * 4 + lag];
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
    // Lanes past `dim` (the relative-bias stream is 16 wide) contribute nothing to the tree below and write
    // nothing back: returning early used to leave their `ss` slots to whatever the previous launch left there.
    const bool live = r < dim;
    const int t = row / heads;
    const float v = live ? x[(size_t)row * dim + r] * scale[(size_t)(row % heads) * dim + r] : 0.f;
    ss[r] = v * v;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (r < o) ss[r] += ss[r + o];
        __syncthreads();
    }
    const float norm = sqrtf(ss[0]);
    const float k = min(1.f, thr / (norm + 1e-6f)) * (tau ? tau[t] : 1.f);
    if (live) y[(size_t)row * dim + r] = v * k;
}

// Unweighted RMSNorm over the last dim: y = x / sqrt(mean(x^2) + eps).
extern "C" __global__ void __launch_bounds__(128) inkling2_rms_unnorm(float* y, const float* x, int rows, int dim,
                                                                     float eps) {
    __shared__ float ss[128];
    const int row = blockIdx.x;
    const int r = threadIdx.x;
    const bool live = r < dim;
    const float v = live ? x[(size_t)row * dim + r] : 0.f;
    ss[r] = v * v;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (r < o) ss[r] += ss[r + o];
        __syncthreads();
    }
    if (live) y[(size_t)row * dim + r] = v * rsqrtf(ss[0] / dim + eps);
}

// Causal attention with the relative bias, the learned sink and the output gate.
// q, k, v, gate and rel are [T, heads|nkv, dim] for q/k/v/gate and [T, heads, d_rel] for rel; `proj` is
// [d_rel, extent] row-major and `sink` is [T, heads] or null. The window is the number of keys a token sees (0 = all
// of them, the full layers).
//
// A block is 16 query rows of one head: the four warps each own four of them, and the keys are staged in shared in
// chunks of KBATCH, so one read of K and V serves sixteen query rows. The relative bias is computed inside the block
// from the query's 16-wide `rel` row and the `proj` columns the chunk needs — the [heads, T, T] buffer it used to be
// materialised into is 103 GB at a 32K prompt, and its kernel was O(heads x T^2) on every layer.
#define IKA_KB 16
#define IKA_TQ 16
#define IKA_MAXD (IKA_KB + IKA_TQ - 1)
extern "C" __global__ void __launch_bounds__(128) inkling2_attn(float* out, const float* q, const float* k,
                                                                const float* v, const float* rel, const float* proj,
                                                                const float* sink, const float* gate, int T, int heads,
                                                                int nkv, int dim, int d_rel, int extent, int window,
                                                                float scale) {
    __shared__ float ks[IKA_KB][128];
    __shared__ float vs[IKA_KB][128];
    __shared__ float rs[IKA_TQ][16];
    __shared__ float bt[IKA_TQ][IKA_KB];
    const int h = blockIdx.x % heads, q0 = (blockIdx.x / heads) * IKA_TQ;
    const int kh = h / (heads / nkv);
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int qlast = min(T - 1, q0 + IKA_TQ - 1);
    const int k0 = window > 0 ? max(0, q0 - window + 1) : 0;
    // The query rows this warp owns: four, one set of four dims per lane.
    const int qb = q0 + wave * 4;
    float4 qv[4];
    float m[4], l[4];
    float4 acc[4];
    for (int i = 0; i < 4; ++i) {
        const int t = qb + i;
        qv[i] = t < T ? *(const float4*)(q + ((size_t)t * heads + h) * dim + 4 * lane)
                      : make_float4(0.f, 0.f, 0.f, 0.f);
        m[i] = sink && t < T ? sink[(size_t)t * heads + h] : -INFINITY;
        l[i] = sink && t < T ? 1.f : 0.f;
        acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int e = threadIdx.x; e < IKA_TQ * d_rel; e += 128) {
        rs[e / d_rel][e % d_rel] = q0 + e / d_rel <= qlast ? rel[((size_t)(q0 + e / d_rel) * heads + h) * d_rel + e % d_rel]
                                                            : 0.f;
    }
    __syncthreads();
    for (int kc = k0; kc <= qlast; kc += IKA_KB) {
        for (int e = threadIdx.x; e < IKA_KB * dim; e += 128) {
            const int s = kc + e / dim, d = e % dim;
            ks[e / dim][d] = s <= qlast ? k[((size_t)s * nkv + kh) * dim + d] : 0.f;
            vs[e / dim][d] = s <= qlast ? v[((size_t)s * nkv + kh) * dim + d] : 0.f;
        }
        // The bias tile: bt[iq][j] = rel[q0+iq] . proj[:, (q0+iq) - (kc+j)], zero outside the extent or the causal
        // window of that row. `proj` is d_rel x extent and stays hot in L1, so it is read straight from global.
        for (int e = threadIdx.x; e < IKA_TQ * IKA_KB; e += 128) {
            const int iq = e / IKA_KB, j = e % IKA_KB, s = kc + j, t = q0 + iq;
            float acc_b = 0.f;
            const int d = t - s;
            if (d >= 0 && d < extent && s <= t && t < T && (window == 0 || s >= t - window + 1)) {
                for (int i = 0; i < d_rel; ++i) acc_b += rs[iq][i] * proj[(size_t)i * extent + d];
            }
            bt[iq][j] = acc_b;
        }
        __syncthreads();
        // The softmax of a 16-key chunk in one shot: the sixteen dots of a row are independent, so they pipeline, and
        // the row is rescaled once per chunk instead of once per key. Per-key the alternative was a chain of a
        // shuffle tree, an exponential and an update, and it ran at 0.3 TFLOPS at a 32K prompt.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = qb + i;
            if (t >= T) continue;
            const int w0 = window > 0 ? max(0, t - window + 1) : 0;
            const int row = wave * 4 + i;
            float sc[IKA_KB], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < IKA_KB; ++j) {
                const int s = kc + j;
                float d = -INFINITY;
                if (s <= t && s >= w0) {
                    const float4 kv = *(const float4*)&ks[j][4 * lane];
                    d = qv[i].x * kv.x + qv[i].y * kv.y + qv[i].z * kv.z + qv[i].w * kv.w;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor(d, o, 32);
                    d = d * scale + bt[row][j];
                }
                sc[j] = d;
                mx = fmaxf(mx, d);
            }
            const float mn = fmaxf(m[i], mx);
            const float corr = mn == -INFINITY ? 1.f : expf(m[i] - mn);
            float sum = l[i] * corr;
            float4 o = {acc[i].x * corr, acc[i].y * corr, acc[i].z * corr, acc[i].w * corr};
#pragma unroll
            for (int j = 0; j < IKA_KB; ++j) {
                const float p = sc[j] == -INFINITY ? 0.f : expf(sc[j] - mn);
                const float4 vv = *(const float4*)&vs[j][4 * lane];
                sum += p;
                o.x += p * vv.x;
                o.y += p * vv.y;
                o.z += p * vv.z;
                o.w += p * vv.w;
            }
            acc[i] = o;
            l[i] = sum;
            m[i] = mn;
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int t = qb + i;
        if (t >= T) continue;
        const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        float4 o = {acc[i].x * inv, acc[i].y * inv, acc[i].z * inv, acc[i].w * inv};
        if (gate) {
            const float4 g = *(const float4*)(gate + ((size_t)t * heads + h) * dim + 4 * lane);
            o.x /= 1.f + expf(-g.x);
            o.y /= 1.f + expf(-g.y);
            o.z /= 1.f + expf(-g.z);
            o.w /= 1.f + expf(-g.w);
        }
        *(float4*)(out + ((size_t)t * heads + h) * dim + 4 * lane) = o;
    }
}

// The same attention on the tensor cores: a 16x16x16 WMMA tile instead of one fp32 dot per thread-cycle. The block
// is still 16 query rows of one head, and the four warps split the 128 dims of the score (two 16-wide chunks each)
// and own two of the eight 16-wide output tiles. Q, K and V are staged in fp16 — the precision the reference keeps
// its own cache in — and the relative bias is added in fp32 after the score tile, as the fp32 kernel does.
//
// The fragment layouts are the ones the other families use: a 16x16 A or B tile is read as sixteen contiguous halves
// from row `lane % 16`, and the fp32 accumulator holds element `i` of lane `l` at (row `2i + l / 16`, col `l % 16`).
// The score tile is (query, key), so the sixteen keys of a query row sit in the sixteen lanes of a half-warp at one
// element: the row max and the row sum are four shuffles, and the rescale of the output is per element.
typedef _Float16 ik_h16 __attribute__((ext_vector_type(16)));
typedef float ik_f8 __attribute__((ext_vector_type(8)));

// The same tile for a chunk that continues a session: the keys and values come from the cache (fp16, `rows` deep,
// position `p` at row `p % ring` when the layer slides) instead of the chunk's own buffers, and every position in the
// masks and in the bias is the session's, not the chunk's.
template <bool CACHED>
__device__ __forceinline__ void attn_wmma_core(float* out, const float* q, const float* k, const float* v,
                                               const signed char* ck, const signed char* cv, const float* cks,
                                               const float* cvs, int rows, int ring, int pos,
                                               const float* rel, const float* proj, const float* sink,
                                               const float* gate, int T, int heads, int nkv, int dim, int d_rel,
                                               int extent, int window, float scale) {
    // K rows are padded: 136 halves (272 B, still 16-byte aligned) makes the row stride 68 words, so the sixteen
    // keys a warp stages land on sixteen banks instead of one (issue #144).
    __shared__ __attribute__((aligned(16))) _Float16 ks[IKA_KB][136];
    __shared__ __attribute__((aligned(16))) _Float16 vt[128][IKA_KB];
    __shared__ __attribute__((aligned(16))) _Float16 ps[IKA_TQ][IKA_KB];
    __shared__ float rs[IKA_TQ][16];
    __shared__ float bt[IKA_TQ][IKA_KB];
    __shared__ float sp[4][IKA_TQ][IKA_KB];
    // Per row: the rescale factor of this chunk and the running sum. Four warps used to compute the softmax of all
    // sixteen rows and keep a quarter of it; each warp owns the four rows of its own O tiles now, and these two rows of
    // shared are how the others read what they need (issue #144).
    __shared__ float alpha_s[IKA_TQ];
    __shared__ float l_s[IKA_TQ];
    const int h = blockIdx.x % heads, q0 = (blockIdx.x / heads) * IKA_TQ;
    const int kh = h / (heads / nkv);
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    // `q0` is a row of this call (its buffers are [T] long); `base` is the same row in the session's positions, and
    // the keys, the masks and the bias are all in that frame.
    const int base = CACHED ? pos : 0;
    const int qlast = base + min(T - 1, q0 + IKA_TQ - 1);
    const int k0 = window > 0 ? max(0, base + q0 - window + 1) : 0;
    const int kend = base + T;
    // The row of the score tile this lane's element `i` belongs to, and the key it holds.
    const int half = lane / 16, col = lane % 16;
    // This warp's two 16-dim chunks of Q, in registers: they never change inside the loop, and keeping them out of
    // shared is 4 KB of the 18.5 the block holds — the difference between three and four blocks per CU, which is what
    // the attention of this kernel is short of (issue #144). A lane reads the row its fragment names, `lane % 16`,
    // and the sixteen dims of its chunk, scaled once.
    ik_h16 qa[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int t = q0 + lane % 16, d0 = (wave * 2 + c) * 16;
        _Float16 qh[16];
        const bool live = t < T && d0 + 16 <= dim;
        const float* qp = q + ((size_t)t * heads + h) * dim + d0;
#pragma unroll
        for (int i = 0; i < 16; ++i) qh[i] = live ? (_Float16)(qp[i] * scale) : (_Float16)0.f;
        qa[c] = *(const ik_h16*)qh;
    }
    // The rel row is a row of this call, so it is bounded by `T` and not by `qlast` (which is a session position).
    for (int e = threadIdx.x; e < IKA_TQ * d_rel; e += 128) {
        rs[e / d_rel][e % d_rel] = q0 + e / d_rel < T
                                       ? rel[((size_t)(q0 + e / d_rel) * heads + h) * d_rel + e % d_rel]
                                       : 0.f;
    }
    // The running max and sum of the four rows this warp owns (`4w + 2i + half`), and the sink as the score of a
    // virtual key with no value.
    float m[2], l[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int t = q0 + wave * 4 + 2 * i + half;
        m[i] = sink && t < T ? sink[(size_t)t * heads + h] : -INFINITY;
        l[i] = sink && t < T ? 1.f : 0.f;
    }
    ik_f8 o[2];
#pragma unroll
    for (int f = 0; f < 2; ++f) o[f] = (ik_f8){0, 0, 0, 0, 0, 0, 0, 0};
    __syncthreads();
    for (int kc = k0; kc <= qlast; kc += IKA_KB) {
        // A B fragment is one column of the product's right operand, sixteen elements along the contraction: the
        // score's is a key's sixteen dims (K as it is), the output's is a dim's sixteen keys (V transposed).
        // One thread takes sixteen contiguous dims of one key and writes them as one half vector: the same chunk of
        // shared, staged with a sixteenth of the instructions. A 16-byte `int4` of int8 codes a lane is what the cache
        // holds (issue #106, item 2); a fresh prefill's own keys and values are f32, four `float4` loads a lane. Per
        // element the old loop was a one-byte load, two converts and a multiply, and `rocprofv3` put ~40 % of this
        // kernel's time in it (issue #110, item 5).
        const int units = IKA_KB * (dim / 16);
        for (int e = threadIdx.x; e < units; e += 128) {
            // `j` on consecutive lanes: the transposed write below goes to `vt[d0 + i][j]`, whose address is
            // `(d0 + i) * IKA_KB + j` halves, so with `j` fixed across the warp every lane hits the same bank. Sixteen
            // keys across the warp instead of four spreads them over eight banks (issue #144).
            const int j = e % IKA_KB, d0 = (e / IKA_KB) * 16, s = kc + j;
            _Float16 kv16[16], vv16[16];
            if (s <= qlast) {
                if (CACHED) {
                    const int row = ring > 0 ? s % ring : s;
                    const int at = (int)((size_t)row * nkv + kh);
                    const int4 kb4 = *(const int4*)(ck + (size_t)at * dim + d0);
                    const int4 vb4 = *(const int4*)(cv + (size_t)at * dim + d0);
                    const signed char* kc8 = (const signed char*)&kb4;
                    const signed char* vc8 = (const signed char*)&vb4;
                    const float ksc = cks[at], vsc = cvs[at];
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        kv16[i] = (_Float16)((float)kc8[i] * ksc);
                        vv16[i] = (_Float16)((float)vc8[i] * vsc);
                    }
                } else {
                    const size_t at = ((size_t)s * nkv + kh) * dim + d0;
                    const float4 k4[4] = {*(const float4*)(k + at), *(const float4*)(k + at + 4),
                                          *(const float4*)(k + at + 8), *(const float4*)(k + at + 12)};
                    const float4 v4[4] = {*(const float4*)(v + at), *(const float4*)(v + at + 4),
                                          *(const float4*)(v + at + 8), *(const float4*)(v + at + 12)};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        kv16[4 * i] = (_Float16)k4[i].x;
                        kv16[4 * i + 1] = (_Float16)k4[i].y;
                        kv16[4 * i + 2] = (_Float16)k4[i].z;
                        kv16[4 * i + 3] = (_Float16)k4[i].w;
                        vv16[4 * i] = (_Float16)v4[i].x;
                        vv16[4 * i + 1] = (_Float16)v4[i].y;
                        vv16[4 * i + 2] = (_Float16)v4[i].z;
                        vv16[4 * i + 3] = (_Float16)v4[i].w;
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) kv16[i] = vv16[i] = (_Float16)0.f;
            }
            *(ik_h16*)&ks[j][d0] = *(const ik_h16*)kv16;
            // V goes into shared transposed, which is the layout the product's B fragment reads.
#pragma unroll
            for (int i = 0; i < 16; ++i) vt[d0 + i][j] = vv16[i];
        }
        // The bias tile, as in the fp32 kernel: rel[q0+iq] . proj[:, (q0+iq) - (kc+j)], zero outside the extent or
        // the causal window of that row.
        for (int e = threadIdx.x; e < IKA_TQ * IKA_KB; e += 128) {
            const int iq = e / IKA_KB, j = e % IKA_KB, s = kc + j, t = base + q0 + iq;
            float acc_b = 0.f;
            const int d = t - s;
            if (d >= 0 && d < extent && s <= t && t < kend && (window == 0 || s >= t - window + 1)) {
                for (int i = 0; i < d_rel; ++i) acc_b += rs[iq][i] * proj[(size_t)i * extent + d];
            }
            bt[iq][j] = acc_b;
        }
        __syncthreads();
        // The score: this warp's two 16-wide chunks of the 128 dims, summed in the accumulator.
        ik_f8 s8 = (ik_f8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int chunk = wave * 2 + c;
            const ik_h16 b = *(const ik_h16*)&ks[lane % 16][chunk * 16];
            s8 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[c], b, s8);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sp[wave][2 * i + half][col] = s8[i];
        __syncthreads();
        // The softmax of the 16 keys of each of this warp's four rows: the four partials, the bias, the mask, then four
        // shuffles. The shuffle tree runs inside the sixteen lanes of one row, so the result is the same as when every
        // warp computed every row.
        float p[2], alpha[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = wave * 4 + 2 * i + half, t = base + q0 + row, s = kc + col;
            float sc = sp[0][row][col] + sp[1][row][col] + sp[2][row][col] + sp[3][row][col] + bt[row][col];
            const int w0 = window > 0 ? max(0, t - window + 1) : 0;
            if (s > t || s < w0 || t >= kend) sc = -INFINITY;
            float mx = sc;
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o2, 32));
            const float mn = fmaxf(m[i], mx);
            // `expf(0)` is exactly 1, and at depth the running max is rarely raised: the branch is free and it saves
            // about half the exponentials that are left.
            alpha[i] = mn == m[i] ? 1.f : (mn == -INFINITY ? 1.f : expf(m[i] - mn));
            p[i] = sc == -INFINITY ? 0.f : expf(sc - mn);
            float sum = p[i];
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) sum += __shfl_xor(sum, o2, 32);
            l[i] = l[i] * alpha[i] + sum;
            m[i] = mn;
        }
        // P and this chunk's rescale factors: the barrier that already stood between them and the product serves both.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            ps[wave * 4 + 2 * i + half][col] = (_Float16)p[i];
            alpha_s[wave * 4 + 2 * i + half] = alpha[i];
        }
        __syncthreads();
        // `x * 1.0f` is exactly `x`: at depth the max usually does not move and the rescale is a no-op. The eight rows
        // this lane rescales are not the four it computed, so the test reads the sixteen published factors.
        float moved = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) moved = fmaxf(moved, fabsf(alpha_s[2 * i + half] - 1.f));
        if (__any(moved != 0.f)) {
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[f][i] *= alpha_s[2 * i + half];
        }
        const ik_h16 pa = *(const ik_h16*)&ps[lane % 16][0];
#pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int tile = wave * 2 + f;
            const ik_h16 b = *(const ik_h16*)&vt[tile * 16 + lane % 16][0];
            o[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[f]);
        }
        __syncthreads();
    }
    // The output: element `i` of tile `f` is (row 2i + half, dim tile*16 + col), and the sum is that row's. Every lane
    // writes rows it does not own, so the sixteen sums go through shared once.
#pragma unroll
    for (int i = 0; i < 2; ++i) l_s[wave * 4 + 2 * i + half] = l[i];
    __syncthreads();
#pragma unroll
    for (int f = 0; f < 2; ++f) {
        const int tile = wave * 2 + f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = 2 * i + half, t = q0 + row;
            if (t >= T) continue;
            const float inv = l_s[row] > 0.f ? 1.f / l_s[row] : 0.f;
            const size_t at = ((size_t)t * heads + h) * dim + tile * 16 + col;
            float val = o[f][i] * inv;
            if (gate) val /= 1.f + expf(-gate[at]);
            out[at] = val;
        }
    }
}

#define IK_ATTN_WMMA(NAME, CACHED)                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) NAME(                                                           \
        float* out, const float* q, const float* k, const float* v, const signed char* ck, const signed char* cv,      \
        const float* cks, const float* cvs, int rows, int ring, int pos, const float* rel, const float* proj,          \
        const float* sink, const float* gate, int T, int heads, int nkv, int dim, int d_rel, int extent, int window,   \
        float scale) {                                                                                                \
        attn_wmma_core<CACHED>(out, q, k, v, ck, cv, cks, cvs, rows, ring, pos, rel, proj, sink, gate, T, heads, nkv,  \
                               dim, d_rel, extent, window, scale);                                                    \
    }
IK_ATTN_WMMA(inkling2_attn_wmma, false)
IK_ATTN_WMMA(inkling2_attn_wmma_cached, true)

// The relative bias of one query row against `nk` cached keys, for the decode path: bias is [heads, nk], `rel` is
// [heads, d_rel] of the single query row. One thread per (head, key).
extern "C" __global__ void __launch_bounds__(256) inkling2_rel_bias_dec(float* bias, const float* rel,
                                                                      const float* proj, int pos, int k0, int nk,
                                                                      int heads, int d_rel, int extent) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long long)heads * nk) return;
    const int h = (int)(idx / nk), s = (int)(idx % nk);
    const int d = pos - (k0 + s);
    float acc = 0.f;
    if (d >= 0 && d < extent) {
        const float* rp = rel + (size_t)h * d_rel;
        for (int i = 0; i < d_rel; ++i) acc += rp[i] * proj[(size_t)i * extent + d];
    }
    bias[idx] = acc;
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
            const float v = x[((size_t)j * T + t) * hidden + i];
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
            s += q[qi * hidden + i] * x[((size_t)j * T + t) * hidden + i] * fac[j];
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
        for (int j = 0; j < n; ++j) acc += sc[j] * x[((size_t)j * T + t) * hidden + i] * fac[j];
        out[((size_t)t * nq + qi) * hidden + i] = acc;
    }
}

// MoE router: sigmoid scores, the bias added only for the selection, the weights renormalized over the k chosen
// scores and scaled by `route_scale`. No shared expert (issue #62). ids and w are [T, topk].
extern "C" __global__ void __launch_bounds__(256) inkling2_moe_route(int* ids, float* w, const float* logits,
                                                                    const float* bias, int E, int topk,
                                                                    float route_scale) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int chosen[64];
    const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* l = logits + (size_t)t * E;
    for (int j = 0; j < topk; ++j) {
        float v = -INFINITY;
        int idx = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += 256) {
            bool taken = false;
            for (int c = 0; c < j; ++c) taken |= chosen[c] == e;
            const float x = 1.f / (1.f + expf(-l[e])) + bias[e];
            if (!taken && (x > v || (x == v && e < idx))) v = x, idx = e;
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(idx, o, 32);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (lane == 0) bv[warp] = v, bi[warp] = idx;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < 8; ++q)
                if (bv[q] > bv[0] || (bv[q] == bv[0] && bi[q] < bi[0])) bv[0] = bv[q], bi[0] = bi[q];
            chosen[j] = bi[0];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.f, sc[64];
        for (int j = 0; j < topk; ++j) sc[j] = 1.f / (1.f + expf(-l[chosen[j]])), s += sc[j];
        for (int j = 0; j < topk; ++j) {
            ids[t * topk + j] = chosen[j];
            w[t * topk + j] = sc[j] / (s + 1e-20f) * route_scale;
        }
    }
}

// The expert activation: softcap(gate) * sigmoid(gate) * softcap(up), with softcap(x) = tanh(x / c) * c, then the
// randomized Hadamard of every 128-block like moe_act (the down weights are in the rotated basis).
//
// The checkpoint stores `w13` interleaved (`g0, u0, g1, u1, ...`): the reference deinterleaves it with
// `Interleave(dim=1)` before splitting the gate and the up. The MLS4 file keeps the checkpoint's row order, so the
// gate of element `j` is row `2j` and the up is row `2j+1`.
extern "C" __global__ void __launch_bounds__(128) inkling2_moe_act(float* out, const float* gu, const int8_t* signs,
                                                                  int I, float gate_cap, float up_cap) {
    __shared__ float s[128];
    const int nb = I / 128, p = blockIdx.x / nb, b = blockIdx.x % nb, i = threadIdx.x;
    const int pos = b * 128 + i;
    const float g = gu[(size_t)p * 2 * I + 2 * pos], u = gu[(size_t)p * 2 * I + 2 * pos + 1];
    const float gc = tanhf(g / gate_cap) * gate_cap;
    const float uc = tanhf(u / up_cap) * up_cap;
    s[i] = gc / (1.f + expf(-g)) * uc * (float)signs[i];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        const int j = i ^ h;
        const float a = s[i], c = s[j];
        __syncthreads();
        s[i] = (i & h) ? c - a : a + c;
        __syncthreads();
    }
    out[(size_t)p * I + b * 128 + i] = s[i] * 0.08838834764831845f;
}

// The loop's cross-pass controller (issue #62): the message of the previous pass is mixed into this pass's input.
//   gate   = down . concat(rms(cur), rms(message))      [rank]
//   gain   = tanh(gain_w . gate)                        [hidden]
//   forget = tanh(forget_w . gate)                      [hidden]
//   out    = cur - forget * cur + gain * message
// One block per token; the weights are bf16 like the checkpoint's, the arithmetic is fp32.
__device__ __forceinline__ void control_body(float* out, const float* cur, const float* msg, const ik_bf16* down,
                                             const ik_bf16* gain_w, const ik_bf16* forget_w, int hidden, int rank,
                                             float eps, int t) {
    __shared__ float red[256];
    __shared__ float gate[128];
    __shared__ float nc[4096];
    __shared__ float nm[4096];
    const int r = threadIdx.x;
    const float* cp = cur + (size_t)t * hidden;
    const float* mp = msg + (size_t)t * hidden;
    // rms of both inputs, then the concatenated normed pair.
    float sc = 0.f, sm = 0.f;
    for (int i = r; i < hidden; i += 256) sc += cp[i] * cp[i], sm += mp[i] * mp[i];
    red[r] = sc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (r < o) red[r] += red[r + o];
        __syncthreads();
    }
    const float fc = rsqrtf(red[0] / (float)hidden + eps);
    __syncthreads();
    red[r] = sm;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (r < o) red[r] += red[r + o];
        __syncthreads();
    }
    const float fm = rsqrtf(red[0] / (float)hidden + eps);
    for (int i = r; i < hidden; i += 256) nc[i] = cp[i] * fc, nm[i] = mp[i] * fm;
    __syncthreads();
    // gate = down . [nc | nm], one output per thread up to rank.
    for (int j = r; j < rank; j += 256) {
        const ik_bf16* w = down + (size_t)j * 2 * hidden;
        float acc = 0.f;
        for (int i = 0; i < hidden; ++i) acc += ik_bf2f(w[i]) * nc[i] + ik_bf2f(w[hidden + i]) * nm[i];
        gate[j] = acc;
    }
    __syncthreads();
    // out = cur - tanh(forget) * cur + tanh(gain) * message.
    for (int i = r; i < hidden; i += 256) {
        float g = 0.f, f = 0.f;
        for (int j = 0; j < rank; ++j) {
            g += ik_bf2f(gain_w[(size_t)i * rank + j]) * gate[j];
            f += ik_bf2f(forget_w[(size_t)i * rank + j]) * gate[j];
        }
        out[(size_t)t * hidden + i] = cp[i] * (1.f - tanhf(f)) + tanhf(g) * mp[i];
    }
}

extern "C" __global__ void __launch_bounds__(256) inkling2_control(float* out, const float* cur, const float* msg,
                                                                  const ik_bf16* down, const ik_bf16* gain_w,
                                                                  const ik_bf16* forget_w, int hidden, int rank,
                                                                  float eps) {
    control_body(out, cur, msg, down, gain_w, forget_w, hidden, rank, eps, blockIdx.x);
}

// The three attention gates of an extra pass in one launch: they share the shapes and the rank, and each was one block
// on one compute unit, so three launches with a block each were three serialised latencies. `blockIdx.y` picks the
// gate; every gate's arithmetic is `control_body`'s, unchanged, so the result is bit-identical.
extern "C" __global__ void __launch_bounds__(256) inkling2_control3(
    float* o0, const float* c0, const float* m0, const ik_bf16* d0, const ik_bf16* g0, const ik_bf16* f0, float* o1,
    const float* c1, const float* m1, const ik_bf16* d1, const ik_bf16* g1, const ik_bf16* f1, float* o2,
    const float* c2, const float* m2, const ik_bf16* d2, const ik_bf16* g2, const ik_bf16* f2, int hidden, int rank,
    float eps) {
    const int t = blockIdx.x;
    switch (blockIdx.y) {
        case 1: control_body(o1, c1, m1, d1, g1, f1, hidden, rank, eps, t); break;
        case 2: control_body(o2, c2, m2, d2, g2, f2, hidden, rank, eps, t); break;
        default: control_body(o0, c0, m0, d0, g0, f0, hidden, rank, eps, t); break;
    }
}

// int8 quantization of the rows of `x`, one scale per 128-wide group, in the layout the grouped W4A8 GEMMs read
// (`rot_quant_i8_w`) but **without** the Hadamard: the Inkling-2 expert activation leaves its rows already rotated
// (its own signs), so re-rotating here would undo it and the down GEMM would see the wrong basis.
extern "C" __global__ void __launch_bounds__(32) inkling2_quant_i8(int8_t* q, float* xsc, const float* x) {
    const size_t base = (size_t)blockIdx.x * 128;
    const int lane = threadIdx.x;
    const float4 xv = *(const float4*)(x + base + 4 * lane);
    float m = fmaxf(fmaxf(fabsf(xv.x), fabsf(xv.y)), fmaxf(fabsf(xv.z), fabsf(xv.w)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float sc = m > 0.f ? m / 127.f : 1.f;
    const char4 o = {(char)__float2int_rn(fminf(fmaxf(xv.x / sc, -128.f), 127.f)),
                     (char)__float2int_rn(fminf(fmaxf(xv.y / sc, -128.f), 127.f)),
                     (char)__float2int_rn(fminf(fmaxf(xv.z / sc, -128.f), 127.f)),
                     (char)__float2int_rn(fminf(fmaxf(xv.w / sc, -128.f), 127.f))};
    *(char4*)(q + base + 4 * lane) = o;
    if (lane == 0) xsc[blockIdx.x] = sc;
}

// y[p, i] = x[p * nq * h + qi * h + i]: the qi-th of the `nq` outputs a mix wrote, for every row. One launch instead
// of one memcpy per row, which at a long prompt was 2048 hipMemcpyAsync of h floats per layer visit.
extern "C" __global__ void __launch_bounds__(256) inkling2_copy_nq(float* y, const float* x, int nq, int qi, int h,
                                                                  int t) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)t * h) return;
    const int p = (int)(i / h), c = (int)(i % h);
    y[i] = x[((size_t)p * nq + qi) * h + c];
}

// The final logit softcap: y = tanh(x / cap) * cap. One thread per element.
extern "C" __global__ void inkling2_softcap(float* y, const float* x, long long n, float cap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i] / cap) * cap;
}

// ---- the decode path: one token at a time, against the keys and values of the slots it has already seen ----

// The short convolution of one token, with the three-row history (`past`, [3, C], oldest first) shifted in place.
// Same arithmetic as inkling2_sconv at T=1; the shift is what makes one launch per token enough.
extern "C" __global__ void __launch_bounds__(256) inkling2_sconv_dec(float* y, const float* x, const ik_bf16* w,
                                                                   const float* dyn_w, float* past, int C, int heads,
                                                                   float div) {
    const int c = (int)blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const int h = c / (C / heads);
    // Tap 0 is the current token; taps 1..3 are the three rows of the history, newest first.
    const float v[4] = {x[c], past[(size_t)2 * C + c], past[(size_t)C + c], past[c]};
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        float wj = dyn_w ? ik_bf2f(w[(size_t)lag * C + c]) : ik_bf2f(w[(size_t)c * 4 + (3 - lag)]);
        if (dyn_w) wj += dyn_w[(size_t)h * 4 + lag];
        acc += wj * v[lag];
    }
    y[c] = (x[c] + acc) / div;
    past[c] = past[(size_t)C + c];
    past[(size_t)C + c] = past[(size_t)2 * C + c];
    past[(size_t)2 * C + c] = x[c];
}

// One query against the cached keys: k and v are [max_ctx, nkv, dim] with `nk` valid rows from `k0`, q, gate and
// out are [heads, dim], bias is [heads, nk] and sink is [heads].
//
// The wall here is neither bandwidth nor arithmetic but the barrier: a block per (head) with a reduction per key ran
// at 5-6 GB/s of effective traffic where the machine does 80. Each of the four warps now owns a quarter of the keys,
// keeps its own (max, sum, output) in registers and never crosses a barrier inside the loop; the four partials are
// merged once at the end. The sink is the score of a virtual key whose value is zero, so it enters the denominator
// only, and the output gate is applied after the merge.
// A key sits at row `k0 + i`, wrapped when `ring` is not 0: a sliding layer only keeps its window, so its cache is a
// ring and the first key of the window may be anywhere in it.
extern "C" __global__ void __launch_bounds__(256) inkling2_attn_dec(float* out, const float* q, const signed char* k,
                                                                  const float* ksc, const signed char* v,
                                                                  const float* vsc, const float* bias,
                                                                  const float* sink, const float* gate, int k0, int nk,
                                                                  int heads, int nkv, int dim, int ring, float scale) {
    __shared__ float sm[8], sl[8], sa[8][128];
    const int h = blockIdx.x, kh = h / (heads / nkv);
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float4 qv = *(const float4*)(q + (size_t)h * dim + 4 * lane);
    // This warp's slice of the keys, contiguous so the reads of a warp stay in one L2 window: the per-key chain
    // measured better here than the 16-key tile the prefill kernel uses, the opposite of the prefill — at a decode
    // step there is one query row and the loads, not the exponentials, are what has to stay in flight.
    const int per = (nk + 7) / 8, i0 = wave * per, i1 = min(nk, i0 + per);
    float m = -INFINITY, l = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    // Four keys' dots at a time: each key keeps its own five-`xor` reduction in the same order, but the four chains
    // are independent, so the warp waits on one round of shuffles per four keys instead of on five per key. The
    // softmax and the value accumulation still run in key order.
    int i = i0;
    for (; i + 3 < i1; i += 4) {
        float d[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int at = i + u, row = ring > 0 ? (k0 + at >= ring ? k0 + at - ring : k0 + at) : k0 + at;
            // int8 keys with one scale per (position, KV head): four codes a lane, the same dequantization as the
            // single-key tail below.
            const float ks = ksc[(size_t)row * nkv + kh];
            const int kbits = *(const int*)(k + ((size_t)row * nkv + kh) * dim + 4 * lane);
            d[u] = qv.x * ((float)(signed char)(kbits & 0xff) * ks) +
                   qv.y * ((float)(signed char)((kbits >> 8) & 0xff) * ks) +
                   qv.z * ((float)(signed char)((kbits >> 16) & 0xff) * ks) +
                   qv.w * ((float)(signed char)((kbits >> 24) & 0xff) * ks);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (int u = 0; u < 4; ++u) d[u] += __shfl_xor(d[u], o, 32);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int at = i + u, row = ring > 0 ? (k0 + at >= ring ? k0 + at - ring : k0 + at) : k0 + at;
            const float sc = d[u] * scale + bias[(size_t)h * nk + at];
            const float mn = fmaxf(m, sc), corr = __expf(m - mn), p = __expf(sc - mn);
            const float vs = vsc[(size_t)row * nkv + kh];
            const int vbits = *(const int*)(v + ((size_t)row * nkv + kh) * dim + 4 * lane);
            acc.x = acc.x * corr + p * ((float)(signed char)(vbits & 0xff) * vs);
            acc.y = acc.y * corr + p * ((float)(signed char)((vbits >> 8) & 0xff) * vs);
            acc.z = acc.z * corr + p * ((float)(signed char)((vbits >> 16) & 0xff) * vs);
            acc.w = acc.w * corr + p * ((float)(signed char)((vbits >> 24) & 0xff) * vs);
            l = l * corr + p;
            m = mn;
        }
    }
    for (; i < i1; ++i) {
        const int row = ring > 0 ? (k0 + i >= ring ? k0 + i - ring : k0 + i) : k0 + i;
        // int8 keys with one scale per (position, KV head): four codes a lane, one scale a row.
        const float kscale = ksc[(size_t)row * nkv + kh];
        const int kbits = *(const int*)(k + ((size_t)row * nkv + kh) * dim + 4 * lane);
        const float ka = (float)(signed char)(kbits & 0xff) * kscale;
        const float kb = (float)(signed char)((kbits >> 8) & 0xff) * kscale;
        const float kc = (float)(signed char)((kbits >> 16) & 0xff) * kscale;
        const float kd = (float)(signed char)((kbits >> 24) & 0xff) * kscale;
        float d = qv.x * ka + qv.y * kb + qv.z * kc + qv.w * kd;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor(d, o, 32);
        const float sc = d * scale + bias[(size_t)h * nk + i];
        const float mn = fmaxf(m, sc), corr = __expf(m - mn), p = __expf(sc - mn);
        const float vscale = vsc[(size_t)row * nkv + kh];
        const int vbits = *(const int*)(v + ((size_t)row * nkv + kh) * dim + 4 * lane);
        acc.x = acc.x * corr + p * ((float)(signed char)(vbits & 0xff) * vscale);
        acc.y = acc.y * corr + p * ((float)(signed char)((vbits >> 8) & 0xff) * vscale);
        acc.z = acc.z * corr + p * ((float)(signed char)((vbits >> 16) & 0xff) * vscale);
        acc.w = acc.w * corr + p * ((float)(signed char)((vbits >> 24) & 0xff) * vscale);
        l = l * corr + p;
        m = mn;
    }
    if (lane == 0) sm[wave] = m, sl[wave] = l;
    sa[wave][4 * lane] = acc.x;
    sa[wave][4 * lane + 1] = acc.y;
    sa[wave][4 * lane + 2] = acc.z;
    sa[wave][4 * lane + 3] = acc.w;
    __syncthreads();
    // Merge: the sink rides in as one more split of weight exp(sink - mn) with a zero output.
    float mn = sink ? sink[h] : -INFINITY;
    for (int w = 0; w < 8; ++w) mn = fmaxf(mn, sm[w]);
    float sum = sink ? __expf(sink[h] - mn) : 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < 8; ++w) {
        const float c = sm[w] == -INFINITY ? 0.f : __expf(sm[w] - mn);
        sum += sl[w] * c;
        o[0] += sa[w][4 * lane] * c;
        o[1] += sa[w][4 * lane + 1] * c;
        o[2] += sa[w][4 * lane + 2] * c;
        o[3] += sa[w][4 * lane + 3] * c;
    }
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    float4 res = {o[0] * inv, o[1] * inv, o[2] * inv, o[3] * inv};
    if (gate) {
        const float4 g = *(const float4*)(gate + (size_t)h * dim + 4 * lane);
        res.x /= 1.f + __expf(-g.x);
        res.y /= 1.f + __expf(-g.y);
        res.z /= 1.f + __expf(-g.z);
        res.w /= 1.f + __expf(-g.w);
    }
    *(float4*)(out + (size_t)h * dim + 4 * lane) = res;
}

// MoE combine: y[t][i] += w[t] * x[t][i]. The route weight of one expert for every token, applied as the expert's
// output is added into the layer's accumulator.
extern "C" __global__ void inkling2_moe_acc(float* y, const float* x, const float* w, long long n, int h) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += w[i / h] * x[i];
}

// Stores a step's keys (or values) as fp16 at logical position `pos`, wrapping in the slot's ring: half the bytes of
// the fp32 the cache used to hold, and the same rounding the reference uses for its own cache.
extern "C" __global__ void __launch_bounds__(256) inkling2_kv_store(signed char* dst, float* dsc, const float* src,
                                                                  int pos, int t, int capacity, int ring, int c,
                                                                  int dim) {
    // One warp per (row of the step, KV head), eight of them a block: the max of the head's `dim` values, its scale,
    // then the codes. int8 with a scale a head is half the bytes of the fp16 cache and keeps eight bits of mantissa
    // where e4m3 keeps four (issue #106, item 2). Eight units a block because a chunk of 512 rows is 4 096 of them
    // and a block of 32 threads each was 4 % of the prefill.
    const int nkv = c / dim;
    const int unit = blockIdx.x * 8 + (int)(threadIdx.x / 32);
    const int r = unit / nkv, kh = unit % nkv;
    const int row = pos + r;
    const int keep = pos + t - min(t, capacity);
    if (row < keep) return;
    const int at = ring > 0 ? row % ring : row;
    const int lane = (int)(threadIdx.x % 32);
    const size_t base = (size_t)r * c + (size_t)kh * dim;
    float mx = 0.f;
    for (int i = 4 * lane; i < dim; i += 4) {
        const float4 x = *(const float4*)(src + base + i);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    const float sc = mx > 0.f ? mx / 127.f : 1.f;
    const float inv = 1.f / sc;
    signed char* dp = dst + (size_t)at * c + (size_t)kh * dim;
    for (int i = 4 * lane; i < dim; i += 4) {
        const float4 x = *(const float4*)(src + base + i);
        dp[i] = (signed char)__float2int_rn(x.x * inv);
        dp[i + 1] = (signed char)__float2int_rn(x.y * inv);
        dp[i + 2] = (signed char)__float2int_rn(x.z * inv);
        dp[i + 3] = (signed char)__float2int_rn(x.w * inv);
    }
    if (lane == 0) dsc[(size_t)at * nkv + kh] = sc;
}

// ---- the MTP head (issue #110, item 6): the fuse of the four target states and the rope the block reads ----

__device__ __forceinline__ float ik_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// The head's fuse, one stage at a time: out[r] = RMSNorm(sum_q alpha[q] * feats[q][r], eps) * w, one warp a row.
// `feats` is [4, rows, n] (the four target states this step kept), `alpha` the four weights of one row of
// `softmax(fuse_weight)`, `w` that stage's `dflare_norm` (bf16). The Markov stage is the fourth row of the same
// softmax; it runs through this kernel with its own weight.
extern "C" __global__ void __launch_bounds__(256) inkling2_draft_mix(float* out, const float* feats,
                                                                    const float* alpha, const ik_bf16* w, int rows,
                                                                    int row0, int stride, int n, float eps) {
    const int r = (int)(blockIdx.x * 8 + threadIdx.x / 32), lane = (int)(threadIdx.x % 32);
    if (r >= rows) return;
    const size_t pl = (size_t)stride * n;
    const float* f = feats + (size_t)(row0 + r) * n;
    const float a0 = alpha[0], a1 = alpha[1], a2 = alpha[2], a3 = alpha[3];
    float ss = 0.f;
    for (int i = lane; i < n; i += 32) {
        const float v = a0 * f[i] + a1 * f[pl + i] + a2 * f[2 * pl + i] + a3 * f[3 * pl + i];
        ss += v * v;
    }
    const float inv = rsqrtf(ik_warp_sum(ss) / (float)n + eps);
    float* y = out + (size_t)r * n;
    for (int i = lane; i < n; i += 32) {
        const float v = a0 * f[i] + a1 * f[pl + i] + a2 * f[2 * pl + i] + a3 * f[3 * pl + i];
        y[i] = v * inv * ik_bf2f(w[i]);
    }
}

// NeoX rope, in place, over the `dim` of every head of a [rows, nh, dim] row: the pair is (f, f + dim / 2), the
// inverse frequency theta^(-2f/dim) and the angle is the row's position.
extern "C" __global__ void __launch_bounds__(256) inkling2_rope_neox(float* x, const int* pos, int rows, int nh,
                                                                    int dim, float theta, int mode) {
    if (mode == 2) return;
    const int half = dim / 2;
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)rows * nh * half) return;
    const int r = (int)(i / (nh * half)), rest = (int)(i % (nh * half));
    const int h = rest / half, f = rest % half;
    float* p = x + ((size_t)r * nh + h) * dim;
    const float ang = (float)pos[r] / powf(theta, (float)(2 * f) / (float)dim);
    const float c = cosf(ang), s = sinf(ang);
    // mode 0: NeoX, the pair is (f, f + half). mode 1: interleaved, the pair is (2f, 2f + 1) with one angle per pair.
    const int j = mode == 0 ? f : 2 * f;
    const int k = mode == 0 ? f + half : 2 * f + 1;
    const float x0 = p[j], x1 = p[k];
    p[j] = x0 * c - x1 * s;
    p[k] = x1 * c + x0 * s;
}

// `n` bf16 values copied out as f32 at `out[off + i]`: the head's `mask_embed` row, which the block writes into
// every one of its proposals.
extern "C" __global__ void __launch_bounds__(256) inkling2_bf16_rows(float* out, const ik_bf16* src, int soff,
                                                                    int doff, int n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i < (long long)n) out[doff + i] = ik_bf2f(src[soff + i]);
}

// The three-row history a convolution input has after every row of a multi-row step: snap[r][3][c] holds the rows
// that `past` holds once row r is done (oldest first, zero before the sequence), so a rolled-back verify can put back
// the history of the row the caller keeps with one copy. Layout: [t, 3, c], `pos` is the first row's position.
extern "C" __global__ void __launch_bounds__(256) inkling2_hist_snap(float* snap, const float* past, const float* x,
                                                                    int pos, int t, int c) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)t * c) return;
    const int r = (int)(i / c), col = (int)(i % c);
    for (int k = 0; k < 3; ++k) {
        const int g = pos + r - 2 + k;
        float v = 0.f;
        if (g >= pos) v = x[(size_t)(g - pos) * c + col];
        else if (g >= 0) v = past[(size_t)(g + 3 - pos) * c + col];
        snap[((size_t)r * 3 + k) * c + col] = v;
    }
}

// ---- the Markov half of the MTP head: the walk that shapes the proposed logits ----

// One step of the walk: `h = rnn_proj @ in + bias` (5120 -> 768 = 3 x rank), then
// `gate = sigmoid(h[r]), state' = gate * state + (1 - gate) * tanh(h[r + rank]), latent = tanh(h[2r + rank])`:
// the published head splits the projection in three blocks, gate first and the latent last.
// `in` is [state | markov_w1[prev] | hidden | f_rnn], `w` is [3 * rank, 5120] bf16, one warp an output row.
extern "C" __global__ void __launch_bounds__(256) inkling2_markov_step(float* state, float* latent, const float* in,
                                                                     const ik_bf16* w, const ik_bf16* bias, int rank,
                                                                     int k) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float sh[768];
    for (int o = warp; o < 3 * rank; o += 8) {
        const ik_bf16* wr = w + (size_t)o * k;
        float acc = 0.f;
        for (int i = lane; i < k; i += 32) acc += ik_bf2f(wr[i]) * in[i];
        acc = ik_warp_sum(acc);
        if (lane == 0) sh[o] = acc + ik_bf2f(bias[o]);
    }
    __syncthreads();
    if (threadIdx.x < rank) {
        const int r = threadIdx.x;
        const float gate = 1.f / (1.f + expf(-sh[r]));
        state[r] = gate * state[r] + (1.f - gate) * tanhf(sh[rank + r]);
        latent[r] = tanhf(sh[2 * rank + r]);
    }
}

// The folded Markov bias added to a row of logits: `logits[v] += folded_bias[v] + sum_r fold[r, v] * latent[r]`.
// The fold is [rank, vocab] (the layout a GEMM of the transposed `hidden_out` against `unembed` writes), so a warp
// reads a contiguous run of vocab for one rank: 103 MB a row, which is what the walk costs.
extern "C" __global__ void __launch_bounds__(256) inkling2_markov_bias(float* logits, const ik_bf16* fold,
                                                                      const float* latent, const float* fbias,
                                                                      int vocab, int rank) {
    const int v = blockIdx.x * 256 + threadIdx.x;
    if (v >= vocab) return;
    float acc = fbias[v];
    for (int r = 0; r < rank; ++r) acc += ik_bf2f(fold[(size_t)r * vocab + v]) * latent[r];
    logits[v] += acc;
}

// [rows, cols] bf16 -> f32 transposed, i.e. out[c * rows + r] = in[r * cols + c]: the fold needs `hidden_out` as
// the rows of a GEMM input.
extern "C" __global__ void __launch_bounds__(256) inkling2_bf16_transpose(float* out, const ik_bf16* in, int rows,
                                                                         int cols) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    out[(size_t)c * rows + r] = ik_bf2f(in[i]);
}

// f32 -> bf16, n values.
extern "C" __global__ void __launch_bounds__(256) inkling2_to_bf16(ik_bf16* out, const float* in, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const unsigned u = __float_as_uint(in[i]);
    out[i] = (ik_bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
