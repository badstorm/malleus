#include <hip/hip_runtime.h>
#include <stdint.h>

// DeepSeek-V4.1's sliding-window attention on the device (issue #128, increment 1). The host graph
// (`crate::deepseek_text`) is the specification: every kernel here reproduces one stage of `attention()` /
// `out_projection()` on a single position, and the per-stage cosine of
// `deepseek_dev::tests::the_device_window_attention_matches_the_host_graph` is what keeps it honest.
//
// The layer this increment covers is a *window* layer (layer 0: no compressor, no indexer). A query reads the last
// `window` raw latents, one shared 512-wide key/value per position, plus a learned per-head logit that takes part in
// the softmax denominator and in nothing else.

__device__ __forceinline__ float dsv_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// RMSNorm with an f32 weight, one 256-thread block per row: DeepSeek's norms are the stored weight itself, not
// Qwen's `1 + w`, so `rmsnorm` (which folds the one and takes bf16 weights) cannot be used here.
extern "C" __global__ void __launch_bounds__(256) dsv_rmsnorm(float* out, const float* x, const float* w, int n,
                                                             float eps) {
    __shared__ float sh[8];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = out + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    ss = dsv_warp_sum(ss);
    __syncthreads();
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float inv = rsqrtf(tot / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = xr[i] * inv * w[i];
}

// Interleaved RoPE on the last `rd` values of the `nh` heads of each of `rows` rows of `q` and, when `nkv` is 1, of
// each row of `kv` — the same rotation on the query heads and on the shared key the host graph applies before its
// cache write. `inv` is the table of `rd / 2` inverse frequencies (the plain base on a window layer, YaRN's on a
// compressed one), `pos` the first row's position, `stride` what the next row adds (one position for a token, the
// ratio for a compressed latent) and `dir` is +1 on the way in and -1 for the inverse the output projection needs.
extern "C" __global__ void __launch_bounds__(256) dsv_rope(float* q, float* kv, const float* inv, int rows, int nh,
                                                          int hd, int rd, int nkv, int pos, int stride, int dir) {
    const int half = rd / 2;
    const int lanes = nh + nkv;
    const long long total = (long long)rows * lanes * half;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(i / (lanes * half));
        const int rem = (int)(i % (lanes * half));
        const int h = rem / half, f = rem % half;
        float* p = h < nh ? q + ((size_t)r * nh + h) * hd + hd - rd + 2 * f : kv + (size_t)r * hd + hd - rd + 2 * f;
        const float ang = (float)(pos + r * stride) * inv[f];
        const float c = cosf(ang), s = (float)dir * sinf(ang);
        const float x0 = p[0], x1 = p[1];
        p[0] = x0 * c - x1 * s;
        p[1] = x1 * c + x0 * s;
    }
}

// The window itself: one 256-thread block per head. `q` is [heads, hd], `kv` the ring of `cap` rows with the row of
// position p at slot p % cap, `sink` the learned logit per head. The query `pos` reads the last `min(cap, pos + 1)`
// positions, oldest first, and the sink enters the denominator exactly as in `sparse_attn`.
//
// Each thread owns `hd / 256` (at most DSV_V) coordinates of the head: the score of a key is a block reduction, the
// weighted sum accumulates in registers in key order, and the ring is read from L2 (256 KB, shared by the heads).
#define DSV_V 4
extern "C" __global__ void __launch_bounds__(256) dsv_win_attn(float* out, const float* q, const float* kv,
                                                              const float* sink, int heads, int hd, int cap, int pos,
                                                              float scale) {
    __shared__ float sh[8];
    __shared__ float sc[130];
    const int h = blockIdx.x;
    const int n = min(cap, pos + 1);
    const int first = pos - n + 1;
    const float* qh = q + (size_t)h * hd;
    for (int j = 0; j < n; ++j) {
        const float* row = kv + (size_t)((first + j) % cap) * hd;
        float acc = 0.f;
        for (int d = threadIdx.x; d < hd; d += 256) acc += qh[d] * row[d];
        acc = dsv_warp_sum(acc);
        __syncthreads();
        if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.f;
            for (int i = 0; i < 8; ++i) tot += sh[i];
            sc[j] = tot * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int j = 0; j < n; ++j) m = fmaxf(m, sc[j]);
        float sum = __expf(sink[h] - m);
        for (int j = 0; j < n; ++j) {
            const float e = __expf(sc[j] - m);
            sc[j] = e;
            sum += e;
        }
        sc[129] = 1.f / sum;
    }
    __syncthreads();
    float acc[DSV_V];
#pragma unroll
    for (int i = 0; i < DSV_V; ++i) acc[i] = 0.f;
    for (int j = 0; j < n; ++j) {
        const float w = sc[j];
        if (w == 0.f) continue;
        const float* row = kv + (size_t)((first + j) % cap) * hd;
        int k = 0;
        for (int d = threadIdx.x; d < hd; d += 256, ++k) acc[k] += w * row[d];
    }
    float* oh = out + (size_t)h * hd;
    const float inv = sc[129];
    int k = 0;
    for (int d = threadIdx.x; d < hd; d += 256, ++k) oh[d] = acc[k] * inv;
}

// The hyper-connections of one sub-layer, on one position. The streams are `hc_mult` copies of the residual, mixed by
// three coefficient sets: `pre` collapses them into the sub-layer's input, `post` writes the sub-layer's output back
// over all of them, and `comb` (a doubly stochastic matrix, a row softmax followed by Sinkhorn passes) says how the
// old streams carry into the new ones. The reference of all three is `deepseek_text::Hc::mixes`.
//
// The mixes are one projection of the flattened streams: `mix[r] = f[r] . h / sqrt(mean(h^2) + eps)`, one block a
// row, and every block recomputes the statistic over the same 20480 values rather than synchronise for it.
extern "C" __global__ void __launch_bounds__(256) dsv_hc_mixes(float* mix, const float* h, const float* f, int n,
                                                             int hd, float eps) {
    __shared__ float sh[8];
    __shared__ float sh2[8];
    const float* fr = f + (size_t)blockIdx.x * hd;
    float acc = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += 256) {
        const float v = h[i];
        acc += fr[i] * v;
        ss += v * v;
    }
    acc = dsv_warp_sum(acc);
    ss = dsv_warp_sum(ss);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        sh[threadIdx.x / 32] = acc;
        sh2[threadIdx.x / 32] = ss;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.f, ts = 0.f;
        for (int i = 0; i < 8; ++i) {
            ta += sh[i];
            ts += sh2[i];
        }
        mix[blockIdx.x] = ta * rsqrtf(ts / hd + eps);
    }
}

// The coefficients: `pre` and `post` are a sigmoid on their own base and scale, `comb` is a row softmax followed by
// `iters` rounds of row and column normalisation (the first column pass comes right after the softmax). One thread:
// the matrix is `hc x hc` (4x4 here) and twenty rounds of it cost less than a block reduction.
extern "C" __global__ void dsv_hc_coef(float* pre, float* post, float* comb, const float* mix, const float* base,
                                       const float* scale, int hc, float eps, int iters) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ float c[64];
    for (int j = 0; j < hc; ++j) {
        pre[j] = 1.f / (1.f + expf(-(mix[j] * scale[0] + base[j]))) + eps;
        post[j] = 2.f / (1.f + expf(-(mix[j + hc] * scale[1] + base[j + hc])));
    }
    for (int j = 0; j < hc; ++j) {
        float m = -INFINITY;
        for (int k = 0; k < hc; ++k) {
            const int idx = 2 * hc + j * hc + k;
            c[j * hc + k] = mix[idx] * scale[2] + base[idx];
            m = fmaxf(m, c[j * hc + k]);
        }
        float sum = 0.f;
        for (int k = 0; k < hc; ++k) {
            c[j * hc + k] = expf(c[j * hc + k] - m);
            sum += c[j * hc + k];
        }
        for (int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.f;
        for (int j = 0; j < hc; ++j) s += c[j * hc + k];
        for (int j = 0; j < hc; ++j) c[j * hc + k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.f;
            for (int k = 0; k < hc; ++k) s += c[j * hc + k];
            for (int k = 0; k < hc; ++k) c[j * hc + k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.f;
            for (int j = 0; j < hc; ++j) s += c[j * hc + k];
            for (int j = 0; j < hc; ++j) c[j * hc + k] /= s + eps;
        }
    }
    for (int j = 0; j < hc * hc; ++j) comb[j] = c[j];
}

// Collapse the streams: `x[d] = sum_c pre[c] * h[c][d]`.
extern "C" __global__ void dsv_hc_pre(float* x, const float* h, const float* pre, int hc, int hidden) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    float v = 0.f;
    for (int c = 0; c < hc; ++c) v += pre[c] * h[(size_t)c * hidden + d];
    x[d] = v;
}

// Write the sub-layer output back over the streams, mixing the old ones in:
// `out[e][d] = post[e] * x[d] + sum_c comb[c][e] * res[c][d]`.
extern "C" __global__ void dsv_hc_post(float* out, const float* x, const float* res, const float* post,
                                       const float* comb, int hc, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hc * hidden) return;
    const int e = i / hidden, d = i % hidden;
    float v = post[e] * x[d];
    for (int c = 0; c < hc; ++c) v += comb[c * hc + e] * res[(size_t)c * hidden + d];
    out[i] = v;
}

// The SwiGLU of a DeepSeek expert: the gate is clamped *before* its silu (the reference's
// `min(gate, limit)` then `silu`), the up path is clamped symmetrically, and the router's weight folds into the
// product so the down projection is one GEMV per expert and one accumulation. `limit` of 0 is no clamp at all.
extern "C" __global__ void dsv_swiglu(float* out, const float* gate, const float* up, int n, float limit,
                                      float weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fmaxf(fminf(u, limit), -limit);
    }
    out[i] = weight * (g / (1.f + expf(-g))) * u;
}

// The router of one position, one thread: `sqrt(softplus(score))` on every expert, the top-k by that score plus the
// steering bias (which does *not* enter the weights), then the optional renormalisation and the route scale. The host
// graph is `deepseek_text::moe`; the bias only steers, so the weights come from the raw scores.
extern "C" __global__ void dsv_route(int* ids, float* wts, const float* scores, const float* bias, int experts,
                                     int top_k, int renorm, float scale, int scratch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    extern __shared__ float s[];
    float* sc = s;
    int* used = (int*)(s + scratch);
    for (int e = 0; e < experts; ++e) {
        const float x = scores[e];
        // torch's softplus threshold: above 20 the exponential would only add noise (and overflow past 88).
        sc[e] = x > 20.f ? sqrtf(x) : sqrtf(log1pf(expf(x)));
        used[e] = 0;
    }
    for (int j = 0; j < top_k; ++j) {
        int best = -1;
        float bestv = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            if (used[e]) continue;
            const float v = sc[e] + (bias ? bias[e] : 0.f);
            if (v > bestv) {
                bestv = v;
                best = e;
            }
        }
        ids[j] = best;
        wts[j] = best >= 0 ? sc[best] : 0.f;
        if (best >= 0) used[best] = 1;
    }
    if (renorm && top_k > 1) {
        float sum = 1e-20f;
        for (int j = 0; j < top_k; ++j) sum += wts[j];
        for (int j = 0; j < top_k; ++j) wts[j] /= sum;
    }
    for (int j = 0; j < top_k; ++j) wts[j] *= scale;
}

// The compressor of an indexed layer: every `ratio` positions of the layer's input pool into one latent row, a
// per-dimension softmax of the gate over the group weighting the `kv` rows. `kv` and `gate` are [rows, hd], `out` is
// [rows / ratio, hd]; the norm and the rotation the host applies afterwards are separate kernels.
extern "C" __global__ void dsv_compress(float* out, const float* kv, const float* gate, int hd, int ratio) {
    const int g = blockIdx.x;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float m = -INFINITY;
        for (int j = 0; j < ratio; ++j) m = fmaxf(m, gate[(g * ratio + j) * hd + d]);
        float sum = 0.f, acc = 0.f;
        for (int j = 0; j < ratio; ++j) {
            const float w = expf(gate[(g * ratio + j) * hd + d] - m);
            sum += w;
            acc += w * kv[(g * ratio + j) * hd + d];
        }
        out[g * hd + d] = acc / sum;
    }
}

// The indexer of an indexed layer: the keys are one row per compressed position (they come from the latent, through
// `wk`, a norm and the rotation at the compressed base), and the score of a query against a key is the sum over the
// indexer's heads of `relu(q . k)` weighted by a per-head projection of the layer's input. `deepseek_text::indexer`
// is the reference: the mask of unreachable compressed rows is the caller's, and the top-k is the next kernel.
extern "C" __global__ void dsv_ix_score(float* score, const float* q, const float* k, const float* w, int t, int heads,
                                        int dim, int n_ix, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * n_ix) return;
    const int pos = i / n_ix, j = i % n_ix;
    const float* kj = k + (size_t)j * dim;
    float acc = 0.f;
    for (int h = 0; h < heads; ++h) {
        const float* qh = q + ((size_t)pos * heads + h) * dim;
        float d = 0.f;
        for (int c = 0; c < dim; ++c) d += qh[c] * kj[c];
        acc += fmaxf(d, 0.f) * w[(size_t)pos * heads + h] * scale;
    }
    score[(size_t)pos * n_ix + j] = acc;
}

// Level one of the two-level top-k: keep the best blocks, pinning the partial one the query just reached.
// Replicates `select_candidate_blocks` with t=1 on the device. `cand[b]` is set to 1 for selected blocks.
extern "C" __global__ void dsv_ix_cand(unsigned char* cand, const float* score, int n_ix, int topk_blocks,
                                         int block_size, int reach) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1) return;
    int blocks = (n_ix + block_size - 1) / block_size;
    extern __shared__ float smem[];
    float* block_max = smem;
    for (int b = 0; b < blocks; ++b) block_max[b] = -INFINITY;
    for (int j = 0; j < n_ix; ++j) {
        int b = j / block_size;
        block_max[b] = fmaxf(block_max[b], score[j]);
    }
    int last = (reach > 0 ? reach : 1) - 1;
    int last_block = last / block_size;
    if (last_block < blocks) block_max[last_block] = INFINITY;
    for (int b = 0; b < blocks; ++b) cand[b] = 0;
    for (int slot = 0; slot < topk_blocks; ++slot) {
        int best = -1;
        float bestv = -INFINITY;
        for (int b = 0; b < blocks; ++b) {
            if (cand[b]) continue;
            float v = block_max[b];
            if (v > bestv || (v == bestv && (best < 0 || b < best))) {
                bestv = v;
                best = b;
            }
        }
        if (best < 0 || bestv == -INFINITY) break;
        cand[best] = 1;
    }
}

// The selection of one position: the `topk` compressed rows that score highest, ties by ascending row, and then the
// ones the position can actually reach (`compress_lens = (pos + 1) / ratio`) — the reference takes exactly `topk`
// and keeps the reachable ones, so an unreachable row still occupies a slot. The list written is the one the
// attention reads: the window slots first (as ring slots, `at % cap`), then the compressed rows as `cap + j`, and
// `-1` where nothing is left to read.  If `cand` is non-null, rows in a zeroed block are scored as -INF: the mask
// ranks rather than cuts, so the kernel still picks exactly `topk` rows and the reach filter later turns the
// unreachable ones into `-1`.
extern "C" __global__ void dsv_ix_topk(int* sel, const float* score, int pos, int n_ix, int topk, int ratio, int cap,
                                       int window_slots, const unsigned char* cand, int block_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) return;
    extern __shared__ int used_s[];
    int* used = used_s;
    for (int j = 0; j < n_ix; ++j) used[j] = 0;
    const int reach = (pos + 1) / ratio;
    const float* sc = score;
    int* out = sel;
    for (int j = 0; j < window_slots; ++j) {
        const int at = pos - window_slots + 1 + j;
        out[j] = at >= 0 ? at % cap : -1;
    }
    // Replicate the host graph: exactly `topk` rows ordered by score (with masked rows at -INF) and then by
    // ascending index, sorted ascending, and finally filtered by reach.
    int picked[64];
    int n_picked = 0;
    for (int slot = 0; slot < topk; ++slot) {
        int best = -1;
        float bestv = -INFINITY;
        for (int j = 0; j < n_ix; ++j) {
            if (used[j]) continue;
            float v = sc[j];
            if (cand) {
                int b = j / block_size;
                if (!cand[b]) v = -INFINITY;
            }
            if (v > bestv || (v == bestv && (best < 0 || j < best))) {
                bestv = v;
                best = j;
            }
        }
        if (best < 0) break;
        used[best] = 1;
        picked[n_picked++] = best;
    }
    for (int a = 0; a < n_picked; ++a) {
        for (int b = a + 1; b < n_picked; ++b) {
            if (picked[b] < picked[a]) {
                int t = picked[a]; picked[a] = picked[b]; picked[b] = t;
            }
        }
    }
    for (int i = 0; i < n_picked; ++i) {
        int j = picked[i];
        out[window_slots + i] = (j < reach) ? (cap + j) : -1;
    }
    for (int i = n_picked; i < topk; ++i) out[window_slots + i] = -1;
}

// The sparse attention of one position over two buffers: the window's ring (`cap` rows, `at % cap`) and the
// compressed latent (`n_ix` rows, `cap + j`). `sel` is the list `dsv_ix_topk` wrote, `-1` for a slot that is not
// there; the sink joins the denominator exactly as it does in the window-only kernel.
extern "C" __global__ void __launch_bounds__(256) dsv_attn_cached(float* out, const float* q, const float* ring,
                                                                const float* ckv, const float* sink, const int* sel,
                                                                int cap, int heads, int hd, int m, float scale) {
    __shared__ float sh[8];
    extern __shared__ float sc[]; // m scores and the denominator at sc[m]
    const int h = blockIdx.x;
    const float* qh = q + (size_t)h * hd;
    const int* idx = sel;
    for (int j = 0; j < m; ++j) {
        const int at = idx[j];
        if (at < 0) {
            sc[j] = -INFINITY;
            continue;
        }
        const float* row = at < cap ? ring + (size_t)at * hd : ckv + (size_t)(at - cap) * hd;
        float acc = 0.f;
        for (int d = threadIdx.x; d < hd; d += 256) acc += qh[d] * row[d];
        acc = dsv_warp_sum(acc);
        __syncthreads();
        if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.f;
            for (int r = 0; r < 8; ++r) tot += sh[r];
            sc[j] = tot * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < m; ++j) mx = fmaxf(mx, sc[j]);
        float sum = mx == -INFINITY ? 0.f : __expf(sink[h] - mx);
        for (int j = 0; j < m; ++j) {
            const float e = sc[j] == -INFINITY ? 0.f : __expf(sc[j] - mx);
            sc[j] = e;
            sum += e;
        }
        sc[m] = sum > 0.f ? 1.f / sum : 0.f;
    }
    __syncthreads();
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = 0.f;
    for (int j = 0; j < m; ++j) {
        const float w = sc[j];
        if (w == 0.f) continue;
        const int at = idx[j];
        if (at < 0) continue;
        const float* row = at < cap ? ring + (size_t)at * hd : ckv + (size_t)(at - cap) * hd;
        int c = 0;
        for (int d = threadIdx.x; d < hd; d += 256, ++c) acc[c] += w * row[d];
    }
    float* oh = out + (size_t)h * hd;
    const float inv = sc[m];
    int c = 0;
    for (int d = threadIdx.x; d < hd; d += 256, ++c) oh[d] = acc[c] * inv;
}

// The token table's row for one position, as f32: the embedding is a lookup, and the row is the input of the first
// block's streams.
extern "C" __global__ void dsv_embed(float* out, const uint16_t* table, int id, int h) {
    const uint16_t* row = table + (size_t)id * h;
    for (int i = threadIdx.x; i < h; i += blockDim.x) out[i] = __uint_as_float(((uint32_t)row[i]) << 16);
}

// The first block's streams: the embedded row in every copy of the residual.
extern "C" __global__ void dsv_bcast(float* out, const float* x, int hc, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hc * h) return;
    out[i] = x[i % h];
}

// The Engram's gate, on one position: each copy of the residual gets `stream + gate * value`, where the value is the
// shared half of the projection of the rows the position's n-grams looked up, and the gate is a sigmoid of the signed
// square root of a normalised dot of the stream against its key. All four copies share the value; each has its own
// key, the same two per-stream weights and its own gate. `deepseek_text::Engram::forward` is the reference.
extern "C" __global__ void __launch_bounds__(256) dsv_engram(float* h, const float* kv, const float* q_w,
                                                            const float* k_w, int hidden, int hc, float eps,
                                                            float clamp) {
    const int c = blockIdx.x;
    __shared__ float a[8], b[8], d[8];
    float* stream = h + (size_t)c * hidden;
    const float* key = kv + (size_t)c * hidden;
    const float* value = kv + (size_t)hc * hidden;
    float sa = 0.f, sb = 0.f, sd = 0.f;
    for (int j = threadIdx.x; j < hidden; j += 256) {
        const float s = stream[j], k = key[j];
        sa += s * s;
        sb += k * k;
        sd += s * q_w[(size_t)c * hidden + j] * k_w[(size_t)c * hidden + j] * k;
    }
    sa = dsv_warp_sum(sa);
    sb = dsv_warp_sum(sb);
    sd = dsv_warp_sum(sd);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        const int w = threadIdx.x / 32;
        a[w] = sa;
        b[w] = sb;
        d[w] = sd;
    }
    __syncthreads();
    float ta = 0.f, tb = 0.f, td = 0.f;
    for (int i = 0; i < 8; ++i) {
        ta += a[i];
        tb += b[i];
        td += d[i];
    }
    // One rstd per copy over `hidden`, as the reference computes it, and the signed root keeps the sign of the dot.
    const float rstd = rsqrtf(ta / hidden + eps) * rsqrtf(tb / hidden + eps);
    const float dot = td * rstd / sqrtf((float)hidden);
    const float rooted = copysignf(sqrtf(fmaxf(fabsf(dot), clamp)), dot);
    const float gate = 1.f / (1.f + expf(-rooted));
    for (int j = threadIdx.x; j < hidden; j += 256) stream[j] += gate * value[j];
}
