#include <hip/hip_runtime.h>
#include <stdint.h>

#define HD 256

typedef _Float16 fa_half16 __attribute__((ext_vector_type(16)));
typedef float fa_float8 __attribute__((ext_vector_type(8)));
typedef unsigned int fa_u4 __attribute__((ext_vector_type(4)));

// Causal GQA prefill attention on fp16 WMMA (flash-attention style). Block = 64 query tokens of one q head and 8
// waves: wave w takes 16 rows (w % 4) and half of the head dims (w / 4), with its half of Q in registers; the two
// halves of each Q K^T tile are summed through LDS, then each wave keeps the softmax state and O (16 x 128) of its
// rows. 32-key tiles of K [pos][nkv][256] and V^T [nkv][chunk][4][256][8] go through LDS. Blocks of the q heads sharing a
// KV head are adjacent so their K/V reads hit L2.
// Sparse (QSA, sel != null): query t sees the key blocks of R tokens set in its bitset sel[t] (W words) and its own
// tail, keys >= R * ((cache_pos0 + t + 1) / R); the waves of 16 queries skip the key tiles none of them uses (uni,
// one union per 16 queries), the block those no wave uses.
template <int KT>
__device__ __forceinline__ void attn2_impl(float* out, const float* q, const _Float16* kc, const _Float16* vtc, int T,
                                           int cache_pos0, int nh, int nkv, float scale, int ctx, const uint32_t* sel,
                                           const uint32_t* uni, int W, int R, float2* mass, int mstride,
                                           int mtiles) {
    constexpr int NF = KT / 16;
    __shared__ __attribute__((aligned(16))) _Float16 ks[KT][HD + 16];
    __shared__ __attribute__((aligned(16))) _Float16 vt[HD][KT + 8];
    __shared__ __attribute__((aligned(16))) float sx[8][NF][8][32];
    __shared__ __attribute__((aligned(16))) _Float16 ps[8][16][KT + 8];
    __shared__ uint32_t selw[64];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, rg = wave % 4, dh = wave / 4;
    const int h = blockIdx.x % nh, t0 = (blockIdx.x / nh) * 64, kh = h / (nh / nkv);
    fa_half16 qa[8];
    {
        const int t = min(t0 + rg * 16 + lane % 16, T - 1);
        const float* qr = q + ((size_t)t * nh + h) * HD + dh * 128;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
            for (int e = 0; e < 16; ++e) qa[kk][e] = (_Float16)(qr[kk * 16 + e] * scale);
    }
    fa_float8 o[8];
#pragma unroll
    for (int f = 0; f < 8; ++f) o[f] = (fa_float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = -INFINITY, l[i] = 0.f;
    const int last_key = cache_pos0 + min(t0 + 63, T - 1);
    const int wave_last = cache_pos0 + min(t0 + rg * 16 + 15, T - 1);
    const int row0 = cache_pos0 + t0 + rg * 16 + lane / 16;
    const int tail0 = sel ? (cache_pos0 + t0 + 1) / R * R : 0;
    const int wtail0 = sel ? (cache_pos0 + min(t0 + rg * 16, T - 1) + 1) / R * R : 0;
    for (int k0 = 0; k0 <= last_key; k0 += KT) {
        bool wave_skip = false;
        if (sel && k0 + KT <= tail0) {
            const int b0 = k0 / R, nb = KT / R;
            const uint32_t mk = nb >= 32 ? 0xffffffffu : (1u << nb) - 1u;
            uint32_t any = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int g = min(t0 / 16 + w, (T - 1) / 16);
                any |= (uni[(size_t)g * W + b0 / 32] >> (b0 % 32)) & mk;
            }
            if (any == 0) continue;
        }
        if (sel && k0 + KT <= wtail0) {
            const int b0 = k0 / R, nb = KT / R;
            const uint32_t mk = nb >= 32 ? 0xffffffffu : (1u << nb) - 1u;
            const int g = min(t0 / 16 + rg, (T - 1) / 16);
            wave_skip = ((uni[(size_t)g * W + b0 / 32] >> (b0 % 32)) & mk) == 0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < KT / 8; ++i) {
            const int idx = tid + i * 256;  // KT * 32 groups of 8 halves
            const int kk = idx % KT, g = idx / KT;
            *(fa_u4*)&ks[kk][g * 8] = *(const fa_u4*)(kc + (size_t)kh * ctx * HD +
                                                      (size_t)(k0 / 32) * HD * 32 + (size_t)g * 256 + (size_t)kk * 8);
        }
        for (int i = 0; i < KT / 8; ++i) {
            const int idx = tid + i * 256;  // KT * 32 chunks of 8 halves for V^T
            const int vr = idx / (KT / 8), vc = (idx % (KT / 8)) * 8;
            *(fa_u4*)&vt[vr][vc] =
                *(const fa_u4*)(vtc + ((size_t)kh * (ctx / 32) + k0 / 32) * HD * 32 + (vc / 8) * HD * 8 + vr * 8);
        }
        if (sel && tid < 64) {
            // Selection bits of the tile's KT / R blocks for each of the 64 rows (KT / R <= 32, word-aligned).
            const int b0 = k0 / R;
            selw[tid] = sel[(size_t)min(t0 + tid, T - 1) * W + b0 / 32] >> (b0 % 32);
        }
        __syncthreads();
        const bool active = !(k0 > wave_last || wave_skip);
        fa_float8 sp[NF];
        if (active) {
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                sp[f] = (fa_float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const fa_half16 b = *(const fa_half16*)&ks[f * 16 + lane % 16][dh * 128 + kk * 16];
                    sp[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[kk], b, sp[f]);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) sx[wave][f][i][lane] = sp[f][i];
            }
        }
        __syncthreads();
        if (!active) continue;
        float cr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x[NF];
            float mx = -INFINITY;
            const int row = row0 + 2 * i;
            const uint32_t rm = sel ? selw[rg * 16 + 2 * i + lane / 16] : 0u;
            const int rtail = sel ? (row + 1) / R * R : 0;
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                const int key = k0 + f * 16 + lane % 16;
                bool vis = key <= row;
                if (sel && key < rtail) vis = vis && ((rm >> ((key - k0) / R)) & 1u);
                x[f] = vis ? sp[f][i] + sx[wave ^ 4][f][i][lane] : -INFINITY;
                mx = fmaxf(mx, x[f]);
            }
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) mx = fmaxf(mx, __shfl_xor(mx, d, 32));
            const float mn = fmaxf(m[i], mx);
            const float corr = mn == -INFINITY ? 1.f : __expf(m[i] - mn);
            float sum = 0.f;
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                const float p = mn == -INFINITY ? 0.f : __expf(x[f] - mn);
                sum += p;
                ps[wave][2 * i + lane / 16][f * 16 + lane % 16] = (_Float16)p;
            }
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) sum += __shfl_xor(sum, d, 32);
            l[i] = l[i] * corr + sum;
            m[i] = mn;
            cr[i] = corr;
            // Diagnostics (`MALLEUS_ATTN_MASS`): how much this tile of KT keys weighs for the row, as
            // (max, sum of exp) so that the exact mass is recovered afterwards. One row every `mstride`, only from
            // the `dh == 0` half of the head, which already has the full score of both halves.
            if (mass != nullptr && dh == 0 && mstride > 0 && row % mstride == mstride - 1) {
                const int sm = row / mstride, tl = k0 / KT;
                if (tl < mtiles) mass[(size_t)sm * mtiles + tl] = make_float2(mn, sum);
            }
        }
#pragma unroll
        for (int f = 0; f < 8; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[f][i] *= cr[i];
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int kf = 0; kf < NF; ++kf) {
            const fa_half16 pa = *(const fa_half16*)&ps[wave][lane % 16][kf * 16];
#pragma unroll
            for (int f = 0; f < 8; ++f) {
                const fa_half16 b = *(const fa_half16*)&vt[dh * 128 + f * 16 + lane % 16][kf * 16];
                o[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[f]);
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int t = t0 + rg * 16 + 2 * i + lane / 16;
        if (t < T) {
            const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
            float* dst = out + ((size_t)t * nh + h) * HD + dh * 128 + lane % 16;
#pragma unroll
            for (int f = 0; f < 8; ++f) dst[f * 16] = o[f][i] * inv;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) attention_wmma(float* out, const float* q, const _Float16* kc,
                                                                 const _Float16* vtc, int T, int cache_pos0, int nh,
                                                                 int nkv, float scale, int ctx, const uint32_t* sel,
                                                                 const uint32_t* uni, int W, int R, float2* mass,
                                                                 int mstride, int mtiles) {
    attn2_impl<32>(out, q, kc, vtc, T, cache_pos0, nh, nkv, scale, ctx, sel, uni, W, R, mass, mstride, mtiles);
}

// Decode / short-prompt attention, split over the keys (flash-decoding): block = (token t, q head, 256-key split),
// 8 waves of 32 keys; lane = key for the scores, and the V^T pass reads the key-vector-major layout below with
// lane = one of 32 consecutive dims (one coalesced 512-byte request per 8-key group).
// Writes the split's (max, sum, unnormalized output) to part; attention_combine merges the splits.
extern "C" __global__ void __launch_bounds__(256) attention_split(float* part, const float* q, const _Float16* kc,
                                                                  const _Float16* vtc, const int* cache_pos0p, int nh,
                                                                  int nkv, float scale, int ctx, int nsplit,
                                                                  const uint32_t* sel, int W, int R) {
    const int cache_pos0 = *cache_pos0p;
    __shared__ float qs[HD];
    __shared__ __attribute__((aligned(16))) float pw[8][32];
    __shared__ float wm[8], wl[8];
    __shared__ float wacc[8][HD];
    const int th = blockIdx.x, t = th / nh, h = th % nh, kh = h / (nh / nkv), split = blockIdx.y;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nkeys = cache_pos0 + t + 1, k_end = min(nkeys, (split + 1) * 256);
    const int tail0 = nkeys / R * R;
    if (sel && (split + 1) * 256 <= tail0) {
        // A split of 256 keys covers 256 / R blocks, whole words of the bitset: skip it when none is selected.
        const int w0 = split * 256 / R / 32, nw = 256 / R / 32;
        uint32_t any = 0;
        for (int w = 0; w < nw; ++w) any |= sel[(size_t)t * W + w0 + w];
        if (!any) {
            float* dst = part + ((size_t)th * nsplit + split) * (HD + 2);
            if (threadIdx.x == 0) dst[0] = -INFINITY, dst[1] = 0.f;
            dst[2 + threadIdx.x] = 0.f;
            return;
        }
    }
    qs[threadIdx.x] = q[((size_t)t * nh + h) * HD + threadIdx.x] * scale;
    __syncthreads();
    const int j0 = split * 256 + wave * 32, j = j0 + lane;
    float s = -INFINITY;
    bool vis = j < k_end;
    if (sel && vis && j < tail0) vis = (sel[(size_t)t * W + j / R / 32] >> ((j / R) % 32)) & 1u;
    if (vis) {
        // K is blocked as [kv head][32-key chunk][dim / 8][32 keys][8 dims]: lane = key, so a warp reads the
        // 8 dims of 32 consecutive keys as one coalesced 512-byte run, and the accumulation order over
        // dimensions is the one the old [pos][nkv][256] layout used.
        const int kchunk = split * 8 + wave;
        const _Float16* kb = kc + (size_t)kh * ctx * HD + (size_t)kchunk * HD * 32 + (size_t)lane * 8;
        s = 0.f;
#pragma unroll 4
        for (int g = 0; g < HD / 8; ++g) {
            const fa_u4 v = *(const fa_u4*)(kb + (size_t)g * 256);
            const _Float16* hv = (const _Float16*)&v;
#pragma unroll
            for (int e = 0; e < 8; ++e) s += qs[g * 8 + e] * (float)hv[e];
        }
    }
    float m = s;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float p = m == -INFINITY ? 0.f : __expf(s - m);
    float l = p;
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor(l, o, 32);
    pw[wave][lane] = p;
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();
    // V^T is stored key-vector-major as [kv head][32-key chunk][key / 8][dim][8 keys]: for one 8-key group the
    // 32 lanes read 32 consecutive dims as one coalesced 512-byte request, and lane l keeps dim pass*32+l. Four
    // loads cover a dim's 32 keys, so the accumulation order is still (key group, key in group) with no
    // reduction, unlike the old [dim][ctx] layout where every lane read a different row (32 discontiguous
    // 64-byte runs).
    const int chunk = split * 8 + wave;
    const _Float16* base = vtc + ((size_t)kh * (ctx / 32) + chunk) * HD * 32;
#pragma unroll
    for (int pass = 0; pass < 8; ++pass) {
        const int d = pass * 32 + lane;
        float acc = 0.f;
        // The last split can end inside this wave's 32 keys; the old kernel skipped the whole wave then, and the
        // chunk pointer beyond ctx/32 would leave the buffer. Zero the accumulator instead.
        if (chunk * 32 < k_end) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const fa_u4 v = *(const fa_u4*)(base + c * HD * 8 + d * 8);
                const _Float16* hv = (const _Float16*)&v;
#pragma unroll
                for (int e = 0; e < 8; ++e) acc += pw[wave][c * 8 + e] * (float)hv[e];
            }
        }
        wacc[wave][d] = acc;
    }
    if (lane == 0) {
        wm[wave] = m;
        wl[wave] = l;
    }
    __syncthreads();
    const int d = threadIdx.x;
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w]);
    float L = 0.f, o = 0.f;
    if (M != -INFINITY) {
        for (int w = 0; w < 8; ++w) {
            if (wm[w] == -INFINITY) continue;
            const float f = __expf(wm[w] - M);
            L += wl[w] * f;
            o += wacc[w][d] * f;
        }
    }
    float* dst = part + ((size_t)th * nsplit + split) * (HD + 2);
    if (d == 0) {
        dst[0] = M;
        dst[1] = L;
    }
    dst[2 + d] = o;
}

extern "C" __global__ void __launch_bounds__(256) attention_combine(float* out, const float* part, int nsplit) {
    const int th = blockIdx.x, d = threadIdx.x;
    const float* src = part + (size_t)th * nsplit * (HD + 2);
    float M = -INFINITY;
    for (int i = 0; i < nsplit; ++i) M = fmaxf(M, src[i * (HD + 2)]);
    float L = 0.f, o = 0.f;
    for (int i = 0; i < nsplit; ++i) {
        const float mi = src[i * (HD + 2)];
        if (mi == -INFINITY) continue;
        const float f = __expf(mi - M);
        L += src[i * (HD + 2) + 1] * f;
        o += src[i * (HD + 2) + 2 + d] * f;
    }
    out[(size_t)th * HD + d] = o / L;
}

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
// Two adjacent 8-dim groups of one key, as the WMMA fragment wants them.
union K16 {
    h16 v;
    fa_u4 u[2];
};
typedef unsigned int u4 __attribute__((ext_vector_type(4)));

// Sparse (QSA) attention gathering the selected keys of each query: two waves per (query token, kv head), one per
// half of the head dims; the q heads of the kv head (<= 16) are the WMMA rows. Keys: the R tokens of each block in
// list[t] (cnt[t] of them, row stride L), then the query's tail. K and V row-major [pos][nkv][256] (the V copy of
// attn_prep). Block = 4 tokens x 2 halves; grid (ceil(T / 4), nkv).
extern "C" __global__ void __launch_bounds__(256) attention_gather(float* out, const float* q, const _Float16* kc,
                                                                   const _Float16* vc, const int* list, const int* cnt,
                                                                   int T, int cache_pos0, int ctx, int nh, int nkv,
                                                                   int L, int R, float scale) {
    __shared__ __attribute__((aligned(16))) _Float16 vt[4][HD][16 + 8];
    __shared__ __attribute__((aligned(16))) _Float16 ps[8][16][16 + 8];
    __shared__ float sx[8][8][32];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, tg = wave % 4, dh = wave / 4;
    const int t = min((int)blockIdx.x * 4 + tg, T - 1), kh = blockIdx.y, gs = nh / nkv;
    const int hrow = kh * gs + min(lane % 16, gs - 1);
    h16 qa[8];
    const float* qr = q + ((size_t)t * nh + hrow) * HD + dh * 128;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int e = 0; e < 16; ++e) qa[kk][e] = (_Float16)(qr[kk * 16 + e] * scale);
    f8 o[8];
#pragma unroll
    for (int f = 0; f < 8; ++f) o[f] = (f8){0,0,0,0,0,0,0,0};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = -INFINITY, l[i] = 0.f;
    // Virtual keys of query t: the R tokens of each listed block, then the tail [R * ncb, pos].
    auto nkeys = [&](int tt) {
        const int pos = cache_pos0 + tt, tail0 = (pos + 1) / R * R;
        return cnt[tt] * R + pos + 1 - tail0;
    };
    const int n = nkeys(t), nsel = cnt[t] * R, tail0 = (cache_pos0 + t + 1) / R * R;
    const int pt = cache_pos0 + t;
    const int* kl = list + (size_t)t * L;
    int nmax = n;  // the waves of a block loop together (barriers): up to the longest list
    for (int w = 0; w < 4; ++w) nmax = max(nmax, nkeys(min((int)blockIdx.x * 4 + w, T - 1)));
    for (int k0 = 0; k0 < nmax; k0 += 16) {
        const int j = k0 + lane % 16;
        const int key = j < nsel ? kl[j / R] * R + j % R : (j < n ? tail0 + j - nsel : -1);
        {
            const int kr = key >= 0 && key <= pt ? key : 0;
            const _Float16* vr = vc + ((size_t)kr * nkv + kh) * HD + dh * 128 + (lane / 16) * 64;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const u4 v = *(const u4*)(vr + c * 8);
                const _Float16* hv = (const _Float16*)&v;
#pragma unroll
                for (int e = 0; e < 8; ++e) vt[tg][dh * 128 + (lane / 16) * 64 + c * 8 + e][lane % 16] = hv[e];
            }
        }
        f8 s = (f8){0,0,0,0,0,0,0,0};
        const int kkey = max(key, 0);
        const _Float16* kr = kc + (size_t)kh * ctx * HD + (size_t)(kkey / 32) * HD * 32 + (size_t)(kkey % 32) * 8;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            K16 b;
            b.u[0] = *(const fa_u4*)(kr + (size_t)(dh * 16 + 2 * kk) * 256);
            b.u[1] = *(const fa_u4*)(kr + (size_t)(dh * 16 + 2 * kk + 1) * 256);
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[kk], b.v, s);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sx[wave][i][lane] = s[i];
        __syncthreads();
        float cr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x = key >= 0 && key <= pt ? s[i] + sx[wave ^ 4][i][lane] : -INFINITY;
            float mx = x;
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) mx = fmaxf(mx, __shfl_xor(mx, d, 32));
            const float mn = fmaxf(m[i], mx);
            const float corr = mn == -INFINITY ? 1.f : __expf(m[i] - mn);
            const float p = mn == -INFINITY ? 0.f : __expf(x - mn);
            float sum = p;
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) sum += __shfl_xor(sum, d, 32);
            l[i] = l[i] * corr + sum; m[i] = mn; cr[i] = corr;
            ps[wave][2 * i + lane / 16][lane % 16] = (_Float16)p;
        }
#pragma unroll
        for (int f = 0; f < 8; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[f][i] *= cr[i];
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
        const h16 pa = *(const h16*)&ps[wave][lane % 16][0];
#pragma unroll
        for (int f = 0; f < 8; ++f) {
            const h16 b = *(const h16*)&vt[tg][dh * 128 + f * 16 + lane % 16][0];
            o[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[f]);
        }
        __syncthreads();
    }
    if (blockIdx.x * 4 + tg >= T) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + lane / 16;
        if (r < gs) {
            float* dst = out + ((size_t)t * nh + kh * gs + r) * HD + dh * 128 + lane % 16;
#pragma unroll
            for (int f = 0; f < 8; ++f) dst[f * 16] = o[f][i] / l[i];
        }
    }
}

// One query per block, four waves of 64 head dims each. The per-query kernel above gives a query only two waves
// (the other six serve the other three queries of its block), 244 VGPR and a 62 KB LDS footprint; here the block
// carries one query, the accumulator of a wave is half as wide, and four waves per query mean twice the threads and
// more resident work per CU. Grid (T, nkv). On gfx1151 this is 33% faster than the four-query union and 29% faster
// than the per-query kernel on a 25K-token prompt; the narrower block maps better onto 40 CUs.
extern "C" __global__ void __launch_bounds__(128) attention_gather_q1(float* out, const float* q, const _Float16* kc,
                                                                     const _Float16* vc, const int* list, const int* cnt,
                                                                     int T, int cache_pos0, int ctx, int nh, int nkv,
                                                                     int L, int R, float scale) {
    __shared__ __attribute__((aligned(16))) _Float16 vt[HD][16 + 8];
    __shared__ __attribute__((aligned(16))) _Float16 ps[4][16][16 + 8];
    __shared__ float sx[4][8][32];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, dh = wave;
    const int t = (int)blockIdx.x, kh = blockIdx.y, gs = nh / nkv;
    const int hrow = kh * gs + min(lane % 16, gs - 1);
    h16 qa[4];
    const float* qr = q + ((size_t)t * nh + hrow) * HD + dh * 64;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int e = 0; e < 16; ++e) qa[kk][e] = (_Float16)(qr[kk * 16 + e] * scale);
    f8 o[4];
#pragma unroll
    for (int f = 0; f < 4; ++f) o[f] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = -INFINITY, l[i] = 0.f;
    const int pos = cache_pos0 + t, tail0 = (pos + 1) / R * R;
    const int nsel = cnt[t] * R, n = nsel + pos + 1 - tail0;
    const int* kl = list + (size_t)t * L;
    for (int k0 = 0; k0 < n; k0 += 16) {
        const int j = k0 + lane % 16;
        const int key = j < nsel ? kl[j / R] * R + j % R : (j < n ? tail0 + j - nsel : -1);
        const int kr = key >= 0 && key <= pos ? key : 0;
        {
            const _Float16* vr = vc + ((size_t)kr * nkv + kh) * HD + dh * 64 + (lane / 16) * 32;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const u4 v = *(const u4*)(vr + c * 8);
                const _Float16* hv = (const _Float16*)&v;
#pragma unroll
                for (int e = 0; e < 8; ++e) vt[dh * 64 + (lane / 16) * 32 + c * 8 + e][lane % 16] = hv[e];
            }
        }
        f8 s = (f8){0, 0, 0, 0, 0, 0, 0, 0};
        const _Float16* krp = kc + (size_t)kh * ctx * HD + (size_t)(kr / 32) * HD * 32 + (size_t)(kr % 32) * 8;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            K16 b;
            b.u[0] = *(const fa_u4*)(krp + (size_t)(dh * 8 + 2 * kk) * 256);
            b.u[1] = *(const fa_u4*)(krp + (size_t)(dh * 8 + 2 * kk + 1) * 256);
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[kk], b.v, s);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sx[wave][i][lane] = s[i];
        __syncthreads();
        float cr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x = key >= 0 && key <= pos
                          ? s[i] + sx[wave ^ 1][i][lane] + sx[wave ^ 2][i][lane] + sx[wave ^ 3][i][lane]
                          : -INFINITY;
            float mx = x;
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) mx = fmaxf(mx, __shfl_xor(mx, d, 32));
            const float mn = fmaxf(m[i], mx);
            const float corr = mn == -INFINITY ? 1.f : __expf(m[i] - mn);
            const float p = mn == -INFINITY ? 0.f : __expf(x - mn);
            float sum = p;
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) sum += __shfl_xor(sum, d, 32);
            l[i] = l[i] * corr + sum; m[i] = mn; cr[i] = corr;
            ps[wave][2 * i + lane / 16][lane % 16] = (_Float16)p;
        }
#pragma unroll
        for (int f = 0; f < 4; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[f][i] *= cr[i];
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
        const h16 pa = *(const h16*)&ps[wave][lane % 16][0];
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const h16 b = *(const h16*)&vt[dh * 64 + f * 16 + lane % 16][0];
            o[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[f]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + lane / 16;
        if (r < gs) {
            float* dst = out + ((size_t)t * nh + kh * gs + r) * HD + dh * 64 + lane % 16;
#pragma unroll
            for (int f = 0; f < 4; ++f) dst[f * 16] = o[f][i] / l[i];
        }
    }
}

// The same sparse gather, but the four queries of a block share one key list: the union of their selected blocks is
// merged into LDS once, each query masks the blocks it did not select, and one wave per dim half stages V for all
// four, so V comes from DRAM once per block instead of once per query. Worth 6% of this kernel at a 12K prompt
// (1284 -> 1204 ms over the 13 full-attention layers) and 9% at 24K (3512 -> 3188 ms over 15), not the 4x the
// traffic accounting suggests: the union is longer than a single list, so the block runs more WMMA tiles than it
// saves, and at 12K the reads the four target groups repeated still largely hit L2. The gain grows with the prompt
// because the K/V footprint stops fitting in L2 and sharing V then removes real DRAM traffic. K is still read per
// wave. The tails are unioned too over [min tail0, max pos] and masked per query; a block another query selected
// cannot double count with a tail because the mask is per query. UMAX bounds the four lists and must match
// GATHER_UNION_MAX in qwen35.rs, which only picks this kernel when 4 * L fits: change one and the other follows.
extern "C" __global__ void __launch_bounds__(256) attention_gather_u(float* out, const float* q, const _Float16* kc,
                                                                     const _Float16* vc, const int* list,
                                                                     const int* cnt, int T, int cache_pos0, int ctx,
                                                                     int nh, int nkv, int L, int R, float scale) {
    constexpr int UMAX = 2080;
    __shared__ __attribute__((aligned(16))) _Float16 vt[HD][16 + 8];
    __shared__ __attribute__((aligned(16))) _Float16 ps[8][16][16 + 8];
    __shared__ float sx[8][8][32];
    __shared__ int sl[UMAX];
    __shared__ int un[UMAX];
    __shared__ int off[5];
    __shared__ int nun_s;
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, tg = wave % 4, dh = wave / 4;
    const int base = (int)blockIdx.x * 4;
    const int t = min(base + tg, T - 1), kh = blockIdx.y, gs = nh / nkv;
    if (tid == 0) {
        off[0] = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) off[w + 1] = off[w] + min(cnt[min(base + w, T - 1)], L);
    }
    __syncthreads();
    for (int i = tid; i < off[4]; i += 256) {
        int w = 0;
        while (i >= off[w + 1]) ++w;
        sl[i] = list[(size_t)min(base + w, T - 1) * L + (i - off[w])];
    }
    __syncthreads();
    // Four-way merge of the sorted lists into (block << 4 | mask). One thread, ~2048 steps against the ~0.3 ms of
    // key tiles this block then walks: the lists are in LDS, and every block merges its own in parallel.
    if (tid == 0) {
        const int e0 = off[1], e1 = off[2], e2 = off[3], e3 = off[4];
        int p0 = 0, p1 = e0, p2 = e1, p3 = e2, nu = 0;
        while (p0 < e0 || p1 < e1 || p2 < e2 || p3 < e3) {
            int bv = INT32_MAX;
            if (p0 < e0 && sl[p0] < bv) bv = sl[p0];
            if (p1 < e1 && sl[p1] < bv) bv = sl[p1];
            if (p2 < e2 && sl[p2] < bv) bv = sl[p2];
            if (p3 < e3 && sl[p3] < bv) bv = sl[p3];
            int m = 0;
            if (p0 < e0 && sl[p0] == bv) { m |= 1; ++p0; }
            if (p1 < e1 && sl[p1] == bv) { m |= 2; ++p1; }
            if (p2 < e2 && sl[p2] == bv) { m |= 4; ++p2; }
            if (p3 < e3 && sl[p3] == bv) { m |= 8; ++p3; }
            un[nu++] = (bv << 4) | m;
        }
        nun_s = nu;
    }
    __syncthreads();
    const int nblk = nun_s * R;
    int tl0u = INT32_MAX, pmax = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int pos = cache_pos0 + min(base + w, T - 1);
        tl0u = min(tl0u, (pos + 1) / R * R);
        pmax = max(pmax, pos);
    }
    const int ntot = nblk + (pmax >= tl0u ? pmax - tl0u + 1 : 0);
    const int pos_t = cache_pos0 + t, tl0_t = (pos_t + 1) / R * R;
    const int hrow = kh * gs + min(lane % 16, gs - 1);
    h16 qa[8];
    const float* qr = q + ((size_t)t * nh + hrow) * HD + dh * 128;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int e = 0; e < 16; ++e) qa[kk][e] = (_Float16)(qr[kk * 16 + e] * scale);
    f8 o[8];
#pragma unroll
    for (int f = 0; f < 8; ++f) o[f] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = -INFINITY, l[i] = 0.f;
    for (int k0 = 0; k0 < ntot; k0 += 16) {
        const int j = k0 + lane % 16;
        int key = -1;
        bool vis = false;
        if (j < nblk) {
            const int u = un[j / R];
            key = (u >> 4) * R + j % R;
            vis = ((u >> tg) & 1) && key <= pos_t;
        } else if (j < ntot) {
            key = tl0u + (j - nblk);
            vis = key >= tl0_t && key <= pos_t;
        }
        // The key is loaded whatever this query's mask says: the wave that stages V serves all four queries, and a
        // key one of them selected has to be in LDS even when this one skipped it. Only the score is masked.
        const int kr = key >= 0 ? key : 0;
        if (tg == 0) {
            const _Float16* vr = vc + ((size_t)kr * nkv + kh) * HD + dh * 128 + (lane / 16) * 64;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const u4 v = *(const u4*)(vr + c * 8);
                const _Float16* hv = (const _Float16*)&v;
#pragma unroll
                for (int e = 0; e < 8; ++e) vt[dh * 128 + (lane / 16) * 64 + c * 8 + e][lane % 16] = hv[e];
            }
        }
        f8 s = (f8){0, 0, 0, 0, 0, 0, 0, 0};
        const _Float16* kp =
            kc + (size_t)kh * ctx * HD + (size_t)(kr / 32) * HD * 32 + (size_t)(kr % 32) * 8;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            K16 b;
            b.u[0] = *(const fa_u4*)(kp + (size_t)(dh * 16 + 2 * kk) * 256);
            b.u[1] = *(const fa_u4*)(kp + (size_t)(dh * 16 + 2 * kk + 1) * 256);
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[kk], b.v, s);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sx[wave][i][lane] = s[i];
        __syncthreads();
        float cr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x = vis ? s[i] + sx[wave ^ 4][i][lane] : -INFINITY;
            float mx = x;
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) mx = fmaxf(mx, __shfl_xor(mx, d, 32));
            const float mn = fmaxf(m[i], mx);
            const float corr = mn == -INFINITY ? 1.f : __expf(m[i] - mn);
            const float p = mn == -INFINITY ? 0.f : __expf(x - mn);
            float sum = p;
#pragma unroll
            for (int d = 1; d < 16; d <<= 1) sum += __shfl_xor(sum, d, 32);
            l[i] = l[i] * corr + sum;
            m[i] = mn;
            cr[i] = corr;
            ps[wave][2 * i + lane / 16][lane % 16] = (_Float16)p;
        }
#pragma unroll
        for (int f = 0; f < 8; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[f][i] *= cr[i];
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
        const h16 pa = *(const h16*)&ps[wave][lane % 16][0];
#pragma unroll
        for (int f = 0; f < 8; ++f) {
            const h16 b = *(const h16*)&vt[dh * 128 + f * 16 + lane % 16][0];
            o[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, o[f]);
        }
        __syncthreads();
    }
    if (base + tg >= T) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + lane / 16;
        if (r < gs) {
            float* dst = out + ((size_t)t * nh + kh * gs + r) * HD + dh * 128 + lane % 16;
#pragma unroll
            for (int f = 0; f < 8; ++f) dst[f * 16] = o[f][i] / l[i];
        }
    }
}

// --------------------------------------------------------------------------------------------- microbenchmark
// The addressing of the old `attention_split` with no math, to tell a bandwidth problem from a compute problem
// (docs/context-degradation.md). Both read the same unique bytes; `kv_pattern` launches one block per q head
// (the old redundancy: gs blocks read the same rows), `kv_grouped` one block per kv head, doing gs heads' worth
// of accumulation per loaded value. The gap between them is the ceiling of intervention A; the production kernel
// now uses the key-vector-major V^T layout measured by `kv_v_local`.

// One block per (q head, 256-key split): the old [dim][ctx] addresses of the real kernel's K and V^T reads.
extern "C" __global__ void __launch_bounds__(256) kv_pattern(float* out, const _Float16* kc, const _Float16* vtc,
                                                             int nkv, int gs, int ctx) {
    const int h = blockIdx.x, split = blockIdx.y, kh = h / gs;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int j0 = split * 256 + wave * 32, j = j0 + lane;
    float acc = 0.f;
    const fa_u4* kr = (const fa_u4*)(kc + ((size_t)j * nkv + kh) * HD);
#pragma unroll 4
    for (int c = 0; c < HD / 8; ++c) {
        const fa_u4 v = kr[c];
        const _Float16* hv = (const _Float16*)&v;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc += (float)hv[e];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const fa_u4* vr = (const fa_u4*)(vtc + ((size_t)kh * HD + lane * 8 + i) * ctx + j0);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const fa_u4 v = vr[c];
            const _Float16* hv = (const _Float16*)&v;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc += (float)hv[e];
        }
    }
    if (acc == 12345.f) out[0] = acc;
}

// One block per (kv head, split): the same rows read once, accumulated into gs heads' worth of sums.
extern "C" __global__ void __launch_bounds__(256) kv_grouped(float* out, const _Float16* kc, const _Float16* vtc,
                                                             int nkv, int gs, int ctx) {
    const int kh = blockIdx.x, split = blockIdx.y;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int j0 = split * 256 + wave * 32, j = j0 + lane;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const fa_u4* kr = (const fa_u4*)(kc + ((size_t)j * nkv + kh) * HD);
#pragma unroll 4
    for (int c = 0; c < HD / 8; ++c) {
        const fa_u4 v = kr[c];
        const _Float16* hv = (const _Float16*)&v;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float x = (float)hv[e];
#pragma unroll
            for (int g = 0; g < 8; ++g) acc[g] += x * (float)(g + 1);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const fa_u4* vr = (const fa_u4*)(vtc + ((size_t)kh * HD + lane * 8 + i) * ctx + j0);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const fa_u4 v = vr[c];
            const _Float16* hv = (const _Float16*)&v;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float x = (float)hv[e];
#pragma unroll
                for (int g = 0; g < 8; ++g) acc[g] += x * (float)(g + 2);
            }
        }
    }
    float t = 0.f;
#pragma unroll
    for (int g = 0; g < 8; ++g) t += acc[g];
    if (t == 12345.f) out[0] = t;
}

// The same bytes, split by which of the two caches they come from, and with the V^T run made contiguous per row.
// `kv_k_only` and `kv_v_only` say which of the two halves of the pattern is the slow one; `kv_v_local` reads V^T
// the way a wave of 32 dims x 256 keys would: 512 contiguous bytes per row instead of 64.
extern "C" __global__ void __launch_bounds__(256) kv_k_only(float* out, const _Float16* kc, const _Float16* vtc,
                                                            int nkv, int gs, int ctx) {
    const int h = blockIdx.x, split = blockIdx.y, kh = h / gs;
    const int lane = threadIdx.x % 32, j = split * 256 + (threadIdx.x / 32) * 32 + lane;
    float acc = 0.f;
    const fa_u4* kr = (const fa_u4*)(kc + ((size_t)j * nkv + kh) * HD);
#pragma unroll 4
    for (int c = 0; c < HD / 8; ++c) {
        const fa_u4 v = kr[c];
        const _Float16* hv = (const _Float16*)&v;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc += (float)hv[e];
    }
    if (acc == 12345.f) out[0] = acc;
}

extern "C" __global__ void __launch_bounds__(256) kv_v_only(float* out, const _Float16* kc, const _Float16* vtc,
                                                            int nkv, int gs, int ctx) {
    const int h = blockIdx.x, split = blockIdx.y, kh = h / gs;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, j0 = split * 256 + wave * 32;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const fa_u4* vr = (const fa_u4*)(vtc + ((size_t)kh * HD + lane * 8 + i) * ctx + j0);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const fa_u4 v = vr[c];
            const _Float16* hv = (const _Float16*)&v;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc += (float)hv[e];
        }
    }
    if (acc == 12345.f) out[0] = acc;
}

extern "C" __global__ void __launch_bounds__(256) kv_v_local(float* out, const _Float16* kc, const _Float16* vtc,
                                                             int nkv, int gs, int ctx) {
    const int kh = blockIdx.x, split = blockIdx.y;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int j0 = split * 256;
    float acc = 0.f;
    // 32 dims per wave, each one a contiguous run of 256 keys (512 bytes).
#pragma unroll
    for (int d = 0; d < 32; ++d) {
        const fa_u4* vr = (const fa_u4*)(vtc + ((size_t)kh * HD + wave * 32 + d) * ctx + j0);
        // 256 keys x 2 bytes = 512 bytes per row = 32 lanes x 16 bytes, one vector each.
        const fa_u4 v = vr[lane];
        const _Float16* hv = (const _Float16*)&v;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc += (float)hv[e];
    }
    if (acc == 12345.f) out[0] = acc;
}
