// Decode / verify GEMV microbenchmark on the model's MLS4 shapes. Build: bench/gemv/run.sh
#include "../../src/kernels/quant.hip"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CK(x)                                                               \
    do {                                                                    \
        hipError_t e = (x);                                                 \
        if (e != hipSuccess) {                                              \
            printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Kinds of launch this bench knows: the plain MLS4 GEMV, the fused-rotation GEMV (9 args, `rs` = Hadamard signs)
// and the block GEMM core that expects already-rotated activations.
enum Kind { PLAIN, FUSED, WIDE, FUSED_R1, FUSED_R2 };

// The same core the engine uses for 1..6 tokens (`GEMV4_HL`), instantiated wider: weights read once per (row,
// chunk) and the token chunks staged in LDS as fp16 pairs, so the summation order per output element does not
// change with TT. That is what makes a wider verify bit-identical instead of merely close.
GEMV4_HL(8)
GEMV4_HL(16)

// The same core with fewer weight rows per warp. Registers per lane scale with TT*R (accumulator, partial and the
// staged x words), so TT=16 wants a small R: this is the knob that decides whether a wide verify is
// register-bound (R too big) or bandwidth-bound.
#define GEMV4_TS(TT, R)                                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        gemv_mls4_ts##TT##_##R(float* out, const float* xr, const uint8_t* q, const __half* sc, int T, int N, int K,   \
                               int acc, const int8_t* rs) {                                                           \
        gemv4_hl<TT, R>(out, xr, q, sc, T, N, K, acc, nullptr, nullptr, nullptr, 0, 0, 0, rs);                        \
    }
GEMV4_TS(8, 1)
GEMV4_TS(8, 2)
GEMV4_TS(16, 1)
GEMV4_TS(16, 2)

int main(int argc, char** argv) {
    const char* only = argc > 1 ? argv[1] : nullptr;
    struct Shape {
        const char* name;
        int n, k, count;
    } shapes[] = {{"gate/up", 17408, 5120, 128}, {"down", 5120, 17408, 64}, {"dn.qkv", 10240, 5120, 48},
                  {"dn.z", 6144, 5120, 48},     {"dn.out", 5120, 6144, 48}, {"attn.q", 12288, 5120, 16},
                  {"attn.o", 5120, 6144, 16},
                  // 35B-A3B (Ornith-1.5 / Qwen3.6): one routed expert's gate_up and down, one dense 2048->2048.
                  {"35b.exp.gu", 1024, 2048, 288}, {"35b.exp.dn", 2048, 512, 288}, {"35b.dense", 2048, 2048, 60}};
    // `wide` = the block-gemm core (grid (N/16, T/8)), the path the engine already takes for t > GEMV_MAX_T.
    // It is the candidate for a wider verify: same per-element summation order as the GEMV core, so the check
    // below proves whether the two agree bit for bit.
    struct Variant {
        const char* name;
        Kind kind;
        int t;
    } vars[] = {{"mls4", PLAIN, 1}, {"l1", FUSED, 1}, {"l2", FUSED, 2}, {"l3", FUSED, 3}, {"l4", FUSED, 4},
                {"l5", FUSED, 5}, {"l6", FUSED, 6}, {"l8", FUSED, 8}, {"l16", FUSED, 16},
                {"l8r1", FUSED_R1, 8}, {"l8r2", FUSED_R2, 8}, {"l16r1", FUSED_R1, 16}, {"l16r2", FUSED_R2, 16},
                {"g8", WIDE, 8}, {"g16", WIDE, 16}};
    // Hadamard signs the GEMV folds in and the GEMM expects applied: all +1 keeps the comparison honest and simple.
    int8_t* rs;
    CK(hipMalloc(&rs, 128));
    {
        std::vector<int8_t> h(128, 1);
        CK(hipMemcpy(rs, h.data(), 128, hipMemcpyHostToDevice));
    }
    const size_t maxq = (size_t)17408 * 5120 / 2;
    uint8_t* q;
    __half* sc;
    float *x, *o;
    CK(hipMalloc(&q, maxq * 2));
    CK(hipMalloc(&sc, maxq / 32 * 2));
    CK(hipMalloc(&x, 16 * 17408 * 4));
    float* xr;
    CK(hipMalloc(&xr, 16 * 17408 * 4));
    CK(hipMalloc(&o, 16 * 17408 * 4));
    std::vector<uint8_t> hq(maxq * 2);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = (uint8_t)(i * 2654435761u >> 13);
    CK(hipMemcpy(q, hq.data(), hq.size(), hipMemcpyHostToDevice));
    CK(hipMemset(sc, 0x20, maxq / 32 * 2));
    {
        std::vector<__half> hs(maxq / 64);
        for (size_t i = 0; i < hs.size(); ++i) hs[i] = __float2half(0.01f + 0.001f * (i % 7));
        CK(hipMemcpy(sc, hs.data(), hs.size() * 2, hipMemcpyHostToDevice));
    }
    std::vector<float> hx(16 * 17408);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = ((float)(i * 7919 % 2003) - 1001.f) / 500.f;
    CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    {
        const int n = 64, k = 5120;
        float* ref;
        CK(hipMalloc(&ref, 4 * n * 4));
        for (int a = 0; a < 4; ++a)
            hipLaunchKernelGGL(gemv_mls4, dim3(2), dim3(256), 0, 0, ref + a * n, x + a * k, q, sc, 1, n, k, 0);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        std::vector<float> r(4 * n), g(4 * n);
        CK(hipMemcpy(r.data(), ref, r.size() * 4, hipMemcpyDeviceToHost));
        printf("ref[0..3] %g %g %g %g\n", r[0], r[1], r[2], r[3]);
        {
            std::vector<__half> hs(k / 128 * 4);
            CK(hipMemcpy(hs.data(), sc, hs.size() * 2, hipMemcpyDeviceToHost));
            for (int row = 0; row < 2; ++row) {
                double cpu = 0;
                for (int kk = 0; kk < k; ++kk) {
                    const uint8_t byte = hq[(size_t)row * k / 2 + kk / 2];
                    int nib = kk % 2 ? byte >> 4 : byte & 15;
                    if (nib >= 8) nib -= 16;
                    cpu += hx[kk] * nib * (double)__half2float(hs[row * (k / 128) + kk / 128]);
                }
                printf("cpu row %d: %g\n", row, cpu);
            }
        }
        for (int t = 1; t <= 4; ++t) {
            CK(hipMemset(o, 0, 4 * n * 4));
            // The fused variants rotate inside the kernel, so the comparison against the CPU reference above is
            // exact only for `gemv_mls4` (t = 1, no rotation folded). Here the point is agreement between variants.
            switch (t) {
            case 1: hipLaunchKernelGGL(gemv_mls4, dim3(2), dim3(256), 0, 0, o, x, q, sc, 1, n, k, 0); break;
            case 2: hipLaunchKernelGGL(gemv_mls4_l2, dim3(2), dim3(256), 0, 0, o, x, q, sc, 2, n, k, 0, rs); break;
            case 3: hipLaunchKernelGGL(gemv_mls4_l3, dim3(2), dim3(256), 0, 0, o, x, q, sc, 3, n, k, 0, rs); break;
            default: hipLaunchKernelGGL(gemv_mls4_l4, dim3(2), dim3(256), 0, 0, o, x, q, sc, 4, n, k, 0, rs); break;
            }
            CK(hipGetLastError());
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(g.data(), o, t * n * 4, hipMemcpyDeviceToHost));
            if (t == 1) printf("l1[0..3] %g %g %g %g\n", g[0], g[1], g[2], g[3]);
            double err = 0, nrm = 0;
            for (int i = 0; i < t * n; ++i) {
                err = std::isfinite(g[i]) ? fmax(err, fabs(g[i] - r[i])) : INFINITY;
                nrm = fmax(nrm, fabs(r[i]));
            }
            printf("check l%d: max err %.3g (max |ref| %.3g)\n", t, err, nrm);
        }
    }
    {
        // Bit-for-bit: the wide core against the 6-row GEMV on the same weights and activations. If these disagree
        // the wide path cannot be used for a speculative verify, because the accepted tokens would not be the ones
        // the decode path would have chosen.
        const int n = 1024, k = 2048, t = 16;
        float *ow, *on;
        CK(hipMalloc(&ow, (size_t)t * n * 4));
        CK(hipMalloc(&on, (size_t)t * n * 4));
        CK(hipMemset(ow, 0, (size_t)t * n * 4));
        CK(hipMemset(on, 0, (size_t)t * n * 4));
        hipLaunchKernelGGL(gemv_mls4_l6, dim3((n + 31) / 32), dim3(256), 0, 0, on, x, q, sc, 6, n, k, 0, rs);
        hipLaunchKernelGGL(gemv_mls4_l16, dim3((n + 31) / 32), dim3(256), 0, 0, ow, x, q, sc, t, n, k, 0, rs);
        float* o1;
        CK(hipMalloc(&o1, (size_t)t * n * 4));
        CK(hipMemset(o1, 0, (size_t)t * n * 4));
        hipLaunchKernelGGL(gemv_mls4_ts16_1, dim3((n + 7) / 8), dim3(256), 0, 0, o1, x, q, sc, t, n, k, 0, rs);
        CK(hipDeviceSynchronize());
        std::vector<float> r1((size_t)t * n), w16((size_t)t * n);
        CK(hipMemcpy(r1.data(), o1, r1.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(w16.data(), ow, w16.size() * 4, hipMemcpyDeviceToHost));
        size_t d1 = 0;
        for (size_t i = 0; i < r1.size(); ++i) d1 += (memcmp(&r1[i], &w16[i], 4) != 0);
        printf("t16 r1 vs t16 r4: %zu of %zu differ%s\n", d1, r1.size(), d1 ? "  <-- NOT identical" : "  (bit-identical)");
        (void)hipFree(o1);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        std::vector<float> a6((size_t)6 * n), w8((size_t)t * n);
        CK(hipMemcpy(a6.data(), on, a6.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(w8.data(), ow, w8.size() * 4, hipMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < a6.size(); ++i) diff += (memcmp(&a6[i], &w8[i], 4) != 0);
        printf("wide vs l6 (n=%d k=%d): %zu of %zu differ%s\n", n, k, diff, a6.size(), diff ? "  <-- NOT identical" : "  (bit-identical)");
        (void)hipFree(ow);
        (void)hipFree(on);
    }
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::vector<double> tot(sizeof(vars) / sizeof(vars[0]), 0);
    for (auto& sh : shapes) {
        printf("%-8s N=%5d K=%5d |", sh.name, sh.n, sh.k);
        const double bytes = (double)sh.n * sh.k / 2 + (double)sh.n * sh.k / 128 * 2;
        for (size_t vi = 0; vi < sizeof(vars) / sizeof(vars[0]); ++vi) {
            auto& v = vars[vi];
            if (only && !strstr(v.name, only)) continue;
            // Rotate over 2 weight copies so every launch reads from DRAM.
            auto launch = [&](int i) {
                const uint8_t* qq = q + (i & 1) * maxq;
                switch (v.kind) {
                case PLAIN:
                    hipLaunchKernelGGL(gemv_mls4, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, v.t, sh.n,
                                       sh.k, 0);
                    break;
                case FUSED: {
                    void (*const f[])() = {(void (*)())gemv_mls4_l1, (void (*)())gemv_mls4_l2,
                                           (void (*)())gemv_mls4_l3, (void (*)())gemv_mls4_l4,
                                           (void (*)())gemv_mls4_l5, (void (*)())gemv_mls4_l6};
                    (void)f;
                    if (v.t == 1)
                        hipLaunchKernelGGL(gemv_mls4_l1, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 1,
                                           sh.n, sh.k, 0, rs);
                    else if (v.t == 2)
                        hipLaunchKernelGGL(gemv_mls4_l2, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 2,
                                           sh.n, sh.k, 0, rs);
                    else if (v.t == 3)
                        hipLaunchKernelGGL(gemv_mls4_l3, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 3,
                                           sh.n, sh.k, 0, rs);
                    else if (v.t == 4)
                        hipLaunchKernelGGL(gemv_mls4_l4, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 4,
                                           sh.n, sh.k, 0, rs);
                    else if (v.t == 5)
                        hipLaunchKernelGGL(gemv_mls4_l5, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 5,
                                           sh.n, sh.k, 0, rs);
                    else if (v.t == 6)
                        hipLaunchKernelGGL(gemv_mls4_l6, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 6,
                                           sh.n, sh.k, 0, rs);
                    else if (v.t == 8)
                        hipLaunchKernelGGL(gemv_mls4_l8, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 8,
                                           sh.n, sh.k, 0, rs);
                    else
                        hipLaunchKernelGGL(gemv_mls4_l16, dim3((sh.n + 31) / 32), dim3(256), 0, 0, o, x, qq, sc, 16,
                                           sh.n, sh.k, 0, rs);
                    break;
                }
                case FUSED_R1:
                case FUSED_R2: {
                    // R rows per warp: a block covers 8 warp * R rows, so the grid differs from the R=4 variants.
                    const int rows = v.kind == FUSED_R1 ? 1 : 2;
                    const dim3 g((sh.n + 8 * rows - 1) / (8 * rows));
                    if (v.t == 8) {
                        if (rows == 1)
                            hipLaunchKernelGGL(gemv_mls4_ts8_1, g, dim3(256), 0, 0, o, x, qq, sc, 8, sh.n, sh.k, 0, rs);
                        else
                            hipLaunchKernelGGL(gemv_mls4_ts8_2, g, dim3(256), 0, 0, o, x, qq, sc, 8, sh.n, sh.k, 0, rs);
                    } else {
                        if (rows == 1)
                            hipLaunchKernelGGL(gemv_mls4_ts16_1, g, dim3(256), 0, 0, o, x, qq, sc, 16, sh.n, sh.k, 0, rs);
                        else
                            hipLaunchKernelGGL(gemv_mls4_ts16_2, g, dim3(256), 0, 0, o, x, qq, sc, 16, sh.n, sh.k, 0, rs);
                    }
                    break;
                }
                case WIDE: {
                    // The engine rotates first and then calls the block core (see `Model::gemv`).
                    const int chunks = v.t * sh.k / 128;
                    hipLaunchKernelGGL(hadamard128, dim3(chunks), dim3(128), 0, 0, xr, x, rs, sh.k);
                    hipLaunchKernelGGL(gemm_mls4, dim3((sh.n + 15) / 16, (v.t + 7) / 8), dim3(256), 0, 0, o, xr, qq,
                                       sc, v.t, sh.n, sh.k, 0);
                    break;
                }
                }
            };
            launch(0);
            CK(hipDeviceSynchronize());
            const int it = 20;
            CK(hipEventRecord(e0));
            for (int i = 0; i < it; ++i) launch(i);
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            float ms;
            CK(hipEventElapsedTime(&ms, e0, e1));
            ms /= it;
            // The number that decides: milliseconds per token verified, so 8 rows for less than 8 passes shows.
            tot[vi] += ms * sh.count;
            printf(" %s %5.1f/%.2f", v.name, bytes / ms / 1e6, ms / v.t);
        }
        printf("  GB/s/ms-token\n");
    }
    // Per *verified token*: the whole point of a wider verify is this number going down. A round of T tokens that
    // costs T times a one-token pass buys nothing over T rounds.
    printf("per verified token (weighted by layer count, ms/token):");
    for (size_t vi = 0; vi < sizeof(vars) / sizeof(vars[0]); ++vi)
        if (tot[vi] > 0) printf(" %s %.2f", vars[vi].name, tot[vi] / vars[vi].t);
    printf("\n");
    return 0;
}
