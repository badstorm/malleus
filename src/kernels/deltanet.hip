#include <hip/hip_runtime.h>
#include <stdint.h>

// Chunked gated delta rule (chunks of C = 64 tokens, key/value dim 128) on q/k normalized by conv_qk_norm.
// Within a chunk with state S0 at its start, G = cumsum(g), gamma = exp(G), T = (I + A)^-1 with
// A[r][s] = beta_r exp(G_r - G_s) k_r.k_s (s < r):
//   U  = Ut0 + Wn S0,          Ut0 = T diag(beta) V,  Wn = -T diag(beta gamma) K
//   O  = Qg S0 + P U,          Qg = diag(gamma) Q,    P[r][s] = exp(G_r - G_s) q_r.k_s (s <= r)
//   S1 = gamma_C S0 + KdT U,   KdT[i][s] = exp(G_C - G_s) k_s[i]
// deltanet_chunk_prep computes the S0-independent terms for every (chunk, value head) in parallel;
// deltanet_chunk_scan walks the chunks of one head with the state in WMMA accumulators.

#define DC 64
#define DD 128

typedef _Float16 dn_half16 __attribute__((ext_vector_type(16)));
typedef float dn_float8 __attribute__((ext_vector_type(8)));

// LDS regions are reused across phases: R1 K (then V^T), R2 Q (then T in fp32), R3 A in fp32 (then K^T).
extern "C" __global__ void __launch_bounds__(256) deltanet_chunk_prep(_Float16* wn, float* ut0, _Float16* qg,
                                                                      _Float16* pm, _Float16* kdt, float* gc,
                                                                      const float* qkv, const float* beta,
                                                                      const float* g, int T, int nv, int nk) {
    __shared__ __attribute__((aligned(32))) _Float16 r1[DC * (DD + 16)];
    __shared__ __attribute__((aligned(32))) float r2[DC * (DC + 4)];
    __shared__ __attribute__((aligned(32))) float r3[DC * (DC + 8)];
    __shared__ float bs[DC], gm[DC];
    __shared__ float xb[3][16][17];
    _Float16(*kh)[DD + 16] = (_Float16(*)[DD + 16])r1;
    _Float16(*vt)[DC + 8] = (_Float16(*)[DC + 8])r1;
    _Float16(*qh)[DD + 8] = (_Float16(*)[DD + 8])r2;
    float(*tf)[DC + 4] = (float(*)[DC + 4])r2;
    float(*af)[DC + 8] = (float(*)[DC + 8])r3;
    _Float16(*kt)[DC + 8] = (_Float16(*)[DC + 8])r3;
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, col = lane % 16, rh = lane / 16;
    const int ch = blockIdx.x / nv, h = blockIdx.x % nv, khd = h / (nv / nk);
    const int t0 = ch * DC, dim = nk * 256 + nv * 128;
    const size_t blk = (size_t)ch * nv + h;
    for (int i = tid; i < DC * DD / 4; i += 256) {
        const int r = i / (DD / 4), c = (i % (DD / 4)) * 4, t = t0 + r;
        float4 k = make_float4(0.f, 0.f, 0.f, 0.f), q = k;
        if (t < T) {
            k = *(const float4*)(qkv + (size_t)t * dim + nk * 128 + khd * 128 + c);
            q = *(const float4*)(qkv + (size_t)t * dim + khd * 128 + c);
        }
        kh[r][c] = (_Float16)k.x;
        kh[r][c + 1] = (_Float16)k.y;
        kh[r][c + 2] = (_Float16)k.z;
        kh[r][c + 3] = (_Float16)k.w;
        qh[r][c] = (_Float16)q.x;
        qh[r][c + 1] = (_Float16)q.y;
        qh[r][c + 2] = (_Float16)q.z;
        qh[r][c + 3] = (_Float16)q.w;
    }
    if (tid < DC) {
        const int t = t0 + tid;
        bs[tid] = t < T ? beta[(size_t)t * nv + h] : 0.f;
        gm[tid] = t < T ? g[(size_t)t * nv + h] : 0.f;
    }
    __syncthreads();
    if (tid < 32) {
        float a0 = gm[lane], a1 = gm[lane + 32];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float x0 = __shfl_up(a0, o, 32), x1 = __shfl_up(a1, o, 32);
            if (lane >= o) {
                a0 += x0;
                a1 += x1;
            }
        }
        a1 += __shfl(a0, 31, 32);
        gm[lane] = a0;
        gm[lane + 32] = a1;
    }
    __syncthreads();
    // K K^T -> A (fp32, strictly lower, beta and decay applied); Q K^T -> P (global). Wave: row tile w % 4,
    // column tiles 2 (w / 4) and 2 (w / 4) + 1.
    {
        const int rt = wave % 4;
        dn_float8 ka[2], qa[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) ka[c] = qa[c] = (dn_float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DD; kk += 16) {
            const dn_half16 ak = *(const dn_half16*)&kh[16 * rt + col][kk];
            const dn_half16 aq = *(const dn_half16*)&qh[16 * rt + col][kk];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const dn_half16 b = *(const dn_half16*)&kh[16 * (2 * (wave / 4) + c) + col][kk];
                ka[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, b, ka[c]);
                qa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, b, qa[c]);
            }
        }
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = 16 * rt + 2 * i + rh, s = 16 * (2 * (wave / 4) + c) + col;
                const float d = __expf(gm[r] - gm[s]);
                af[r][s] = s < r ? bs[r] * d * ka[c][i] : 0.f;
                pm[(blk * DC + r) * DC + s] = (_Float16)(s <= r ? d * qa[c][i] : 0.f);
            }
    }
    for (int i = tid; i < DC * DD; i += 256) {
        const int r = i / DD, c = i % DD;
        qg[(blk * DC + r) * DD + c] = (_Float16)((float)qh[r][c] * __expf(gm[r]));
    }
    for (int i = tid; i < DD * DC; i += 256) {
        const int c = i / DC, s = i % DC;
        kdt[(blk * DD + c) * DC + s] = (_Float16)(__expf(gm[DC - 1] - gm[s]) * (float)kh[s][c]);
    }
    if (tid == 0) gc[blk] = __expf(gm[DC - 1]);
    __syncthreads();
    // T = (I + A)^-1 (fp32, over Q's region), blocked by 16: diagonal blocks by forward substitution in registers
    // (one wave each, lane = column), then T_ij = -T_ii sum_{m=j}^{i-1} A_im T_mj by increasing distance i - j.
    for (int i = tid; i < DC * DC; i += 256) tf[i / DC][i % DC] = 0.f;
    __syncthreads();
    if (wave < 4 && lane < 16) {
        const int b = 16 * wave;
        float tc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float v = r == lane ? 1.f : 0.f;
#pragma unroll
            for (int m = 0; m < r; ++m) v -= af[b + r][b + m] * tc[m];
            tc[r] = v;
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) tf[b + r][b + lane] = tc[r];
    }
    __syncthreads();
    for (int d = 1; d < 4; ++d) {
        const int nb = 4 - d;
        for (int e = tid; e < nb * 256; e += 256) {
            const int bi = e / 256 + d, bj = bi - d, r = (e % 256) / 16, c = e % 16;
            float x = 0.f;
            for (int k = 16 * bj; k < 16 * bi; ++k) x += af[16 * bi + r][k] * tf[k][16 * bj + c];
            xb[e / 256][r][c] = x;
        }
        __syncthreads();
        for (int e = tid; e < nb * 256; e += 256) {
            const int bi = e / 256 + d, bj = bi - d, r = (e % 256) / 16, c = e % 16;
            float y = 0.f;
#pragma unroll
            for (int k = 0; k < 16; ++k) y += tf[16 * bi + r][16 * bi + k] * xb[e / 256][k][c];
            tf[16 * bi + r][16 * bj + c] = -y;
        }
        __syncthreads();
    }
    // K^T over A's region
    for (int i = tid; i < DC * DD; i += 256) {
        const int s = i / DD, c = i % DD;
        kt[c][s] = kh[s][c];
    }
    __syncthreads();
    // Wn = -(T diag(beta gamma)) K and Ut0 = (T diag(beta)) V: wave: row tile w % 4, column tiles 4 (w / 4) .. +3
    const int rt = wave % 4, ct0 = (wave / 4) * 4;
    dn_half16 ta[4];
    {
        dn_float8 wa[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) wa[c] = (dn_float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DC; kk += 16) {
            dn_half16 a;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int s = kk + e;
                a[e] = (_Float16)(tf[16 * rt + col][s] * bs[s] * __expf(gm[s]));
                ta[kk / 16][e] = (_Float16)(tf[16 * rt + col][s] * bs[s]);
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const dn_half16 b = *(const dn_half16*)&kt[16 * (ct0 + c) + col][kk];
                wa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, wa[c]);
            }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                wn[(blk * DC + 16 * rt + 2 * i + rh) * DD + 16 * (ct0 + c) + col] = (_Float16)(-wa[c][i]);
    }
    __syncthreads();
    for (int i = tid; i < DC * DD / 4; i += 256) {
        const int s = i / (DD / 4), c = (i % (DD / 4)) * 4, t = t0 + s;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t < T) v = *(const float4*)(qkv + (size_t)t * dim + nk * 256 + h * 128 + c);
        vt[c][s] = (_Float16)v.x;
        vt[c + 1][s] = (_Float16)v.y;
        vt[c + 2][s] = (_Float16)v.z;
        vt[c + 3][s] = (_Float16)v.w;
    }
    __syncthreads();
    {
        dn_float8 ua[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) ua[c] = (dn_float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DC; kk += 16)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const dn_half16 b = *(const dn_half16*)&vt[16 * (ct0 + c) + col][kk];
                ua[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ta[kk / 16], b, ua[c]);
            }
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                ut0[(blk * DC + 16 * rt + 2 * i + rh) * DD + 16 * (ct0 + c) + col] = ua[c][i];
    }
}

// One block (8 waves) per value head, chunks in order. Wave w keeps state rows [16w, 16w + 16) (all 128 value
// columns) as 8 WMMA accumulators; S^T and U^T go through LDS in fp16 as WMMA B operands.
extern "C" __global__ void __launch_bounds__(256) deltanet_chunk_scan(float* out, float* state, const _Float16* wn,
                                                                      const float* ut0, const _Float16* qg,
                                                                      const _Float16* pm, const _Float16* kdt,
                                                                      const float* gc, int T, int nv) {
    __shared__ __attribute__((aligned(32))) _Float16 st[DD][DD + 16];
    __shared__ __attribute__((aligned(32))) _Float16 ut[DD][DC + 16];
    const int h = blockIdx.x, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int col = lane % 16, rh = lane / 16;
    float* sp = state + (size_t)h * DD * DD;
    dn_float8 sf[8];
#pragma unroll
    for (int ct = 0; ct < 8; ++ct)
#pragma unroll
        for (int i = 0; i < 8; ++i) sf[ct][i] = sp[(16 * wave + 2 * i + rh) * DD + 16 * ct + col];
    const int rt = wave % 4, ct0 = (wave / 4) * 4;
    const int nch = (T + DC - 1) / DC;
    for (int ch = 0; ch < nch; ++ch) {
        const size_t blk = (size_t)ch * nv + h;
#pragma unroll
        for (int ct = 0; ct < 8; ++ct)
#pragma unroll
            for (int i = 0; i < 8; ++i) st[16 * ct + col][16 * wave + 2 * i + rh] = (_Float16)sf[ct][i];
        __syncthreads();
        const _Float16* wrow = wn + (blk * DC + 16 * rt + col) * DD;
        const _Float16* qrow = qg + (blk * DC + 16 * rt + col) * DD;
        const _Float16* prow = pm + (blk * DC + 16 * rt + col) * DC;
        {
            dn_float8 ua[4];
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int i = 0; i < 8; ++i) ua[c][i] = ut0[(blk * DC + 16 * rt + 2 * i + rh) * DD + 16 * (ct0 + c) + col];
#pragma unroll
            for (int kk = 0; kk < DD; kk += 16) {
                const dn_half16 aw = *(const dn_half16*)(wrow + kk);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const dn_half16 b = *(const dn_half16*)&st[16 * (ct0 + c) + col][kk];
                    ua[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aw, b, ua[c]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int i = 0; i < 8; ++i) ut[16 * (ct0 + c) + col][16 * rt + 2 * i + rh] = (_Float16)ua[c][i];
        }
        __syncthreads();
        {
            dn_float8 oa[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) oa[c] = (dn_float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int kk = 0; kk < DD; kk += 16) {
                const dn_half16 aq = *(const dn_half16*)(qrow + kk);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const dn_half16 b = *(const dn_half16*)&st[16 * (ct0 + c) + col][kk];
                    oa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, b, oa[c]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (int kk = 0; kk < DC; kk += 16) {
                const dn_half16 ap = *(const dn_half16*)(prow + kk);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const dn_half16 b = *(const dn_half16*)&ut[16 * (ct0 + c) + col][kk];
                    oa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ap, b, oa[c]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = ch * DC + 16 * rt + 2 * i + rh;
                if (t < T) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) out[((size_t)t * nv + h) * DD + 16 * (ct0 + c) + col] = oa[c][i];
                }
            }
        }
        const float gcv = gc[blk];
        const _Float16* krow = kdt + (blk * DD + 16 * wave + col) * DC;
#pragma unroll
        for (int ct = 0; ct < 8; ++ct) sf[ct] *= gcv;
#pragma unroll
        for (int kk = 0; kk < DC; kk += 16) {
            const dn_half16 ak = *(const dn_half16*)(krow + kk);
#pragma unroll
            for (int ct = 0; ct < 8; ++ct) {
                const dn_half16 b = *(const dn_half16*)&ut[16 * ct + col][kk];
                sf[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, b, sf[ct]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    }
#pragma unroll
    for (int ct = 0; ct < 8; ++ct)
#pragma unroll
        for (int i = 0; i < 8; ++i) sp[(16 * wave + 2 * i + rh) * DD + 16 * ct + col] = sf[ct][i];
}
