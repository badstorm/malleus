// GEMM microbenchmark on the model's prefill shapes. Build: bench/gemm/run.sh
#include "../../src/kernels/wmma.hip"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CK(x)                                                                    \
    do {                                                                         \
        hipError_t e = (x);                                                      \
        if (e != hipSuccess) {                                                   \
            printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e));      \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

typedef void (*I8K)(float*, float*, float*, int, int, const int8_t*, const float*, const int8_t*, const float*, int,
                    int, int, int);

struct Var {
    const char* name;
    I8K fn;
    int bm, bn, threads;
};

static uint64_t rs = 42;
static int8_t r8() {
    rs = rs * 6364136223846793005ull + 1442695040888963407ull;
    return (int8_t)((rs >> 33) % 255 - 127);
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048;
    const char* only = argc > 2 ? argv[2] : nullptr;
    struct Shape {
        const char* name;
        int n, k, count;
    } shapes[] = {
        {"mlp.gate/up", 17408, 5120, 128}, {"mlp.down", 5120, 17408, 64}, {"dn.qkv", 10240, 5120, 48},
        {"dn.z", 6144, 5120, 48},          {"dn.out", 5120, 6144, 48},    {"attn.q", 12288, 5120, 16},
        {"attn.kv", 1024, 5120, 32},       {"attn.o", 5120, 6144, 16},
        // Qwen3.8-Flash-Next: dims are [K, N] in the file, so N/K below are the model's.
        {"fn.qkv", 10240, 2560, 96},       {"fn.z", 6144, 2560, 48},      {"fn.out", 2560, 6144, 48},
        {"fn.q", 12288, 2560, 12},         {"fn.o", 2560, 6144, 12},      {"fn.hcmix", 10240, 384, 96},
        {"fn.shared", 1280, 2560, 48},
    };
    Var vars[] = {
        {"i8", gemm_i8, 256, 64, 256},
        {"i8_m128", gemm_i8_m128, 128, 128, 256},
    };
    const int maxk = 17408, maxn = 17408;
    std::vector<int8_t> hx((size_t)T * maxk), hw((size_t)maxn * maxk);
    for (auto& v : hx) v = r8();
    for (auto& v : hw) v = r8();
    std::vector<float> hxs(T), hws(maxn), hxsc((size_t)T * (maxk / 128));
    for (int i = 0; i < T; ++i) hxs[i] = 1e-3f * (1 + i % 7);
    for (int i = 0; i < maxn; ++i) hws[i] = 1e-3f * (1 + i % 5);
    int8_t *dx, *dw;
    float *dxs, *dws, *dxsc, *o1, *o2;
    CK(hipMalloc(&dx, hx.size()));
    CK(hipMalloc(&dw, hw.size()));
    CK(hipMalloc(&dxs, T * 4));
    CK(hipMalloc(&dws, maxn * 4));
    CK(hipMalloc(&dxsc, hxsc.size() * 4));
    CK(hipMalloc(&o1, (size_t)T * maxn * 4));
    CK(hipMalloc(&o2, (size_t)T * maxn * 4));
    CK(hipMemcpy(dx, hx.data(), hx.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dxs, hxs.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dws, hws.data(), maxn * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    double tot_ref = 0, tot_best = 0;
    std::vector<double> tot_var(sizeof(vars) / sizeof(vars[0]), 0);
    for (auto& sh : shapes) {
        const int N = sh.n, K = sh.k, G = K / 128;
        for (int t = 0; t < T; ++t)
            for (int g = 0; g < G; ++g) hxsc[(size_t)t * G + g] = hxs[t];
        CK(hipMemcpy(dxsc, hxsc.data(), (size_t)T * G * 4, hipMemcpyHostToDevice));
        const double ops = 2.0 * T * N * K;
        auto timeit = [&](auto launch) {
            launch();
            CK(hipDeviceSynchronize());
            const int it = 5;
            CK(hipEventRecord(e0));
            for (int i = 0; i < it; ++i) launch();
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            float ms;
            CK(hipEventElapsedTime(&ms, e0, e1));
            return ms / it;
        };
        const int nt = (T + 127) / 128;
        const float ref_ms = timeit([&] {
            hipLaunchKernelGGL(gemm_w8a8, dim3((N + 127) / 128 * nt), dim3(256), 0, 0, o1, dx, dxsc, (const uint8_t*)dw,
                               (const void*)dws, T, N, K, 0);
        });
        tot_ref += ref_ms * sh.count;
        printf("%-12s N=%5d K=%5d  ref %7.3f ms %5.2f TOPS |", sh.name, N, K, ref_ms, ops / ref_ms / 1e9);
        std::vector<float> a((size_t)T * N), b((size_t)T * N);
        CK(hipMemcpy(a.data(), o1, a.size() * 4, hipMemcpyDeviceToHost));
        double best = ref_ms;
        for (size_t vi = 0; vi < sizeof(vars) / sizeof(vars[0]); ++vi) {
            auto& v = vars[vi];
            if (only && !strstr(v.name, only)) continue;
            const int vnt = (T + v.bm - 1) / v.bm, vnn = (N + v.bn - 1) / v.bn;
            const float ms = timeit([&] {
                hipLaunchKernelGGL(v.fn, dim3(vnt * vnn), dim3(v.threads), 0, 0, o2, o2, o2, N, 0, dx, dxs, dw, dws, T, N, K, 0);
            });
            CK(hipMemcpy(b.data(), o2, b.size() * 4, hipMemcpyDeviceToHost));
            double err = 0, nrm = 0;
            for (size_t i = 0; i < a.size(); i += 97) {
                err = fmax(err, fabs(a[i] - b[i]));
                nrm = fmax(nrm, fabs(a[i]));
            }
            tot_var[vi] += ms * sh.count;
            best = fmin(best, ms);
            printf(" %s %5.2f%s", v.name, ops / ms / 1e9, err > 1e-4 * nrm ? "(BAD)" : "");
        }
        tot_best += best * sh.count;
        printf("\n");
    }
    // MoE raggruppato: la stessa quantita' di lavoro con due raggruppamenti. (a) tutti i pair in un esperto, tile
    // piene da BM righe; (b) 512 esperti con 32 righe ciascuno, che e' il caso reale di Flash-Next (2048 token x 8
    // esperti / 512 esperti = 32 righe per esperto). Se (a) va il doppio di (b), la perdita e' nel raggruppamento e
    // non nel GEMM.
    {
        const int E = 512, IN = 640, H = 2560, TOK = 2048;  // esperti, intermedio, hidden, token del chunk
        const int P = TOK * 8;                              // pair instradati
        const int Ngu = 2 * IN;                               // righe di gate|up per esperto
        std::vector<int8_t> q((size_t)E * Ngu * H / 2), xq((size_t)P * H);
        std::vector<uint16_t> sc((size_t)E * Ngu * (H / 128));
        std::vector<float> xsc((size_t)P * (H / 128));
        for (auto& v : q) v = r8();
        for (auto& v : xq) v = r8();
        for (auto& v : sc) v = (uint16_t)(0x2400 + (v % 16));
        for (auto& v : xsc) v = 1e-3f;
        int8_t *dq, *dxq;
        uint16_t* dsc;
        float *dxsc, *dout;
        int *doffs, *dnt;
        int2* dtiles;
        CK(hipMalloc(&dq, q.size()));
        CK(hipMalloc(&dxq, xq.size()));
        CK(hipMalloc(&dsc, sc.size() * 2));
        CK(hipMalloc(&dxsc, xsc.size() * 4));
        CK(hipMalloc(&dout, (size_t)P * IN * 4));
        CK(hipMalloc(&doffs, (E + 1) * 4));
        CK(hipMalloc(&dnt, 4));
        CK(hipMalloc(&dtiles, 8192 * sizeof(int2)));
        CK(hipMemcpy(dq, q.data(), q.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dxq, xq.data(), xq.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dsc, sc.data(), sc.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dxsc, xsc.data(), xsc.size() * 4, hipMemcpyHostToDevice));
        std::vector<int> offs(E + 1);
        for (int e = 0; e <= E; ++e) offs[e] = e * (P / E);
        CK(hipMemcpy(doffs, offs.data(), offs.size() * 4, hipMemcpyHostToDevice));
        hipEvent_t a0, a1;
        CK(hipEventCreate(&a0));
        CK(hipEventCreate(&a1));
        for (int bm : {32, 64, 128}) {
            const char* name = bm == 32 ? "moe_w4a4_glu_m32" : bm == 64 ? "moe_w4a4_glu" : "moe_w4a4_glu_m128";
            for (int mode = 0; mode < 3; ++mode) {
                // (a) one expert with all the rows of the chunk, (b) the E routed experts with P/E rows each,
                // (c) the shared expert: **one** expert with the rows of the tokens only (2048), so a grid of 16 tiles
                // instead of 128. On 2026-11-15 the real model ran (b) at 9.8 TOPS and (c) at 1.35: if the microbench
                // reproduces the collapse, the loss is in the grid and not in the GEMM.
                const int shared_rows = TOK;  // the shared expert sees the tokens, not the pairs
                const int rows = mode == 0 ? P : mode == 1 ? P / E : shared_rows;
                const int exps = mode == 1 ? E : 1;
                std::vector<int2> tl;
                for (int e = 0; e < exps; ++e)
                    for (int r = 0; r + bm <= rows || r == 0; r += bm)
                        tl.push_back(make_int2(mode == 1 ? e : 0, (mode == 1 ? e * rows : 0) + r));
                const int nt = (int)tl.size();
                CK(hipMemcpy(dtiles, tl.data(), tl.size() * sizeof(int2), hipMemcpyHostToDevice));
                CK(hipMemcpy(dnt, &nt, 4, hipMemcpyHostToDevice));
                const double ops = 2.0 * rows * exps * Ngu * H;
                // The kernel has to match the rows per tile: always launching the 32-row one on 64/128 tiles
                // processes a quarter of the rows counted in `ops`, and the TOPS come out inflated by up to 4x.
                auto launch = [&](dim3 grid) {
                    if (bm == 32)
                        hipLaunchKernelGGL((moe_w4a4_glu_m32), grid, dim3(256), 0, 0, dout, dxq, dxsc,
                                           (const uint8_t*)dq, (const __half*)dsc, doffs, dtiles, dnt, Ngu, H);
                    else if (bm == 64)
                        hipLaunchKernelGGL((moe_w4a4_glu), grid, dim3(256), 0, 0, dout, dxq, dxsc, (const uint8_t*)dq,
                                           (const __half*)dsc, doffs, dtiles, dnt, Ngu, H);
                    else
                        hipLaunchKernelGGL((moe_w4a4_glu_m128), grid, dim3(256), 0, 0, dout, dxq, dxsc,
                                           (const uint8_t*)dq, (const __half*)dsc, doffs, dtiles, dnt, Ngu, H);
                };
                launch(dim3(nt * (Ngu + 127) / 128));
                CK(hipDeviceSynchronize());
                CK(hipEventRecord(a0));
                for (int i = 0; i < 3; ++i) launch(dim3(nt * (Ngu + 127) / 128));
                CK(hipEventRecord(a1));
                CK(hipEventSynchronize(a1));
                float ms;
                CK(hipEventElapsedTime(&ms, a0, a1));
                ms /= 3;
                printf("MoE gate|up %-16s BM=%-3d %-26s tiles=%5d  %7.2f ms  %5.1f TOPS\n", name, bm,
                       mode == 0 ? "1 expert, full tiles"
                       : mode == 1 ? "512 experts, 32 rows"
                                   : "1 shared expert, 2048 rows",
                       nt, ms, ops / ms / 1e9);
            }
        }
    }
    printf("per-chunk GEMM time (T=%d, 64 layers): ref %.1f ms", T, tot_ref);
    for (size_t vi = 0; vi < sizeof(vars) / sizeof(vars[0]); ++vi)
        if (tot_var[vi] > 0) printf(", %s %.1f ms", vars[vi].name, tot_var[vi]);
    printf("\n");
    return 0;
}
