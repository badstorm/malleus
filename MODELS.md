# Models

What Malleus carries, what each file costs on the device, and what the engine measures on it. One section per
model with the numbers, the date and the machine; the commands that produce them are at the end.

All numbers are single runs on the gfx1150 development machine unless a table says otherwise. The engine runs on
both Strix APUs; the two machines differ almost only in what the silicon offers (table below).

| | gfx1150 (Strix Point, development) | gfx1151 (Strix Halo) |
|---|---|---|
| GPU | Radeon 890M, 16 CU | Radeon 8060S, **40 CU** |
| Memory | 93.4 GiB of GTT reported by the driver (the GPU reads the model from GTT, not from the host) | **128 GB** shared |
| Bandwidth | ~80 GB/s measured read (`malleus bench-bw`) | nominal ~256 GB/s; **231 GB/s measured read** (`malleus bench-bw`, `amd_iommu=off`) |
| Host | LXC, 32 GB of RAM visible | LXC guest on a Proxmox host, `amd_iommu=off` |
| Stack | ROCm 7.14.1; `llama.cpp` build 12407 (`e574a17b0`) from `ghcr.io/ggml-org/llama.cpp:full-rocm` for the comparisons | same |
| Noise | ~±8 % run to run on a single run; a difference under that is not a difference | same |

The per-architecture defaults switch with the machine: int4 experts and the one-query-per-block gather, the
tiled indexer and the 12/12/8 select, and the 4-query selection groups are the gfx1151 defaults (`MALLEUS_MOE_W4A4`,
`MALLEUS_GATHER_Q1`, `MALLEUS_IDX_SCORE_TILE`, `MALLEUS_IDX_SELECT`, `MALLEUS_IDX_GROUP`); gfx1150 falls back to the
per-query kernels and the four-query union. The decode is bandwidth bound on both, so the 4-bit number per model
below is the same: the richer difference is in the prefill, where 40 CU and 2.9x the bandwidth change the answer.

## Carried models

`on disk` is the size of the `.malleus` file; `prefill copies` says whether the file also carries the MLS8 weights
for prompt processing (twice the file, better prompts) or only the 4-bit ones.

| Model | File | On disk | Prefill copies | Notes |
|---|---|---|---|---|
| Qwen3.8-27B (dense, text + images) | `qwen38-27b-mls4.malleus` | 17.6 GB | no | The reference model for the quality gates (`golden/qwen38-27b`). |
| Qwen3.6-35B-A3B (text + images) | `qwen36-35b-a3b-mls4.malleus` | 54.8 GB | **yes** (680 tensors) | Converted before `--prefill-int8` became opt-in: the name says `-mls4` but the file holds both copies. `malleus slim` rewrites it as 4-bit only. |
| Ornith-1.5-35B-A3B (text + images) | `ornith-15-35b-a3b-mls4.malleus` | 20.3 GB | no | Production model: 21.0 GB live, 5.6 GiB of session state and 5.3 GiB of working set at 262 144 tokens (28.7 GiB on a 93.4 GiB GTT). MTP head for speculative decoding. |
| Ornith-1.5-122B (text + images) | `ornith-15-122b-mls4.malleus` | 66.8 GB | no | Does not fit resident next to the rest of the host (issue #14). With `MALLEUS_EXPERT_CACHE=24` the routed experts stay on NVMe: 44.6 GiB live, prefill streams all 59.8 GB of experts per chunk (issue #14). |
| Qwen3.8-Flash-Next (text) | `qwen38-flash-next-mls2.malleus` (mixed, opt-in) | 95 GB | no | gfx1151 only. Issue #37: routed gate/up at 2 bits (MLS2), down/dense at 4/8; 50.1 GB on the device. Golden 89.45 % all-position against 95.11 % of MLS4 and decode 35.7 against 37.8 tok/s, so it stays an experiment and not a production format. |
| Ornith-1.5-35B-A3B-DFlash (draft) | `ornith-15-35b-a3b-dflash-mls4.malleus` | 0.2 GB | no | Speculative draft head (`malleus draft-info`); the MTP head inside the main file covers the common case. |
| Inkling-2-Mini (text) | `Inkling-2-Mini-mls4.malleus` | 15.3 GB | no | The looped transformer (issue #70): 24 layers, the middle 12 run three times, 128 experts / 8 active, window 1024. Session state is 393 KiB per token (48 KV slots); no MTP wiring yet. |

## Qwen3.8-Flash-Next — MLS2 (issue #37)

Measured 2026-11-19 on gfx1151 with the bf16 checkpoint, the 21-prompt `golden` set at `--ctx 8192`, greedy, and
`malleus bench --prompt 512 --decode 32 --depth 0` for speed. The MLS4 file is the one in production.

| Format | On device | All-position argmax | Last-token top-1 | Worst KL | tg32 | pp512 |
|---|---|---|---|---|---|---|
| MLS4 (production) | 73.5 GB | **95.11 %** | 19/21 | **0.225** | **37.8** | 1085 |
| MLS2 routed gate/up + MLS4 down | 50.1 GB | 89.45 % | 19/21 | 1.410 | 35.7 | 1075 |
| MLS2 every routed projection + shared | 40.0 GB | 88.47 % | 20/21 | 1.229 | 35.5 | 1085 |

Reconstruction error in the rotated space, average relative L2 per 128-group on four sampled layers (0, 3, 24, 47),
both expert tensors, 256 rows each: MLS2 uniform 0.381, MLS2 e2m1 subset 0.342, MLS3 0.194, MLS4 0.104. The e2m1
subset is -1.5, -0.5, 0.5, 1.5; the full 16-entry e2m1 table needs four bits per code, so it is not an MLS2 grid.

**Conclusion: 2 bits does not pay here.** It reads fewer bytes but not faster (the decode GEMV is instruction-bound
before it is bandwidth-bound), and both 2-bit variants lose 5.7-6.6 points of all-position argmax against MLS4. The
format and kernels stay in the tree, opt-in and off by default (`malleus convert --expert-mls2 family`, plus
`--mls2-down` for the down projection); MLS4 stays the production format.

## Qwen3.8-27B

Tokens/s, same machine and conditions. Malleus 4-bit decode + 8-bit prefill against llama.cpp Q4_K_M with
flash attention.

| Engine | pp512 | pp1024 | pp2048 | pp8192 | tg128 |
|---|---|---|---|---|---|
| llama.cpp Q4_K_M | 191.7 | 215.2 | 213.9 | 200.0 | 4.74 |
| **Malleus** | **209.3** | **220.3** | **214.6** | **203.1** | **5.02** |

Measured memory bandwidth is ~80 GB/s (`malleus bench-bw`), so the 4-bit decode of a dense 27B model is bandwidth
bound at ~5.8 tokens/s; MTP speculative decoding is what goes past that limit.

### Concurrent requests (2026-11-18)

Same machine, `malleus serve` with `MALLEUS_SLOTS=4`, `MALLEUS_CTX=16384` (4096 per session), MTP off; one client
at a time against N clients sending the same kind of request together (256-token prompt, 192 generated tokens,
greedy). The sessions share one pass over the weights, one token each, so the aggregate rate multiplies while a
single request keeps its speed.

| clients | aggregate | per-request wall (p50) |
|---|---|---|
| 1 | 4.92 tok/s | 39.0 s |
| 2 | 8.88 tok/s | 43.2 s |
| 4 | **12.82 tok/s** | 59.7 s |

`llama.cpp` (server-rocm, build 12464, `-fa on`, four slots) on the same box, same burst: 4.32 tok/s at one client
and 13.16 at four. The act of batching is worth the same to both engines; what differs is the single-request rate
and the prefill. Two sessions of 48 tokens put next to a 20 043-token prefill finish in 76 s at
`MALLEUS_PREFILL_QUANTUM=256` (they waited 120 s before the sessions were resident).

## Ornith-1.5-35B-A3B — the long-context case (2026-11-15/16)

A 35B-A3B hybrid MoE (40 layers, full attention every 4th, 256 experts / 8 active, native window 262 144), 20.2 GB
of MLS4 weights against the 20.21 GiB `Q4_K_M` GGUF of the same checkpoint. This is the model the project exists
for: **the prefill of a long prompt on one APU**.

### Engine level: prefill and decode against the depth of the session

`malleus bench --depth` against `llama-bench -d`: same synthetic tokens, prefill of 8192 tokens in one forward,
64 tokens of decode, one session grown to each depth (`-fa on -ngl 99` for llama.cpp, chunk 8192 here, MTP off on
both sides so this is kernel against kernel).

| depth | pp Malleus | pp llama.cpp | ratio | tg Malleus | tg llama.cpp | ratio |
|---|---|---|---|---|---|---|
| 0 | **968.5** | 467.6 | 2.07x | 29.08 | 28.30 | 1.03x |
| 8192 | **854.6** | 341.7 | 2.50x | 27.02 | 26.67 | 1.01x |
| 16384 | **704.8** | 270.0 | 2.61x | 24.84 | 25.09 | 0.99x |
| 32768 | **522.4** | 193.7 | 2.70x | 22.25 | 22.47 | 0.99x |
| 65536 | **341.6** | 121.2 | 2.82x | 17.78 | 18.86 | 0.94x |
| 98304 | **248.4** | 88.0 | 2.82x | 15.01 | 15.90 | 0.94x |
| 131072 | **196.1** | 69.4 | **2.83x** | 13.08 | 13.98 | 0.94x |

What is left of the shallow rate at each depth:

| | 8k | 16k | 32k | 64k | 96k | 128k |
|---|---|---|---|---|---|---|
| prefill Malleus | 88 % | 73 % | 54 % | 35 % | 26 % | 20 % |
| prefill llama.cpp | 73 % | 58 % | 41 % | 26 % | 19 % | 15 % |
| decode Malleus | 93 % | 85 % | 77 % | 61 % | 52 % | 45 % |
| decode llama.cpp | 94 % | 89 % | 79 % | 67 % | 56 % | 49 % |

The gap is not a long-context trick: it is ~2.1x already in a fresh session and grows to 2.8x, i.e. the prefill path
of this family is systematically faster here. Decode is at parity up to ~16K and ~6 % behind below that; at 4 bits a
decode step reads all the weights and is bandwidth bound, so no engine wins there. In absolute terms the cost of the
context is the same for both: the KV of this model is ~20 KB per token of context (10 full-attention layers, 2 KV
heads, head_dim 256, fp16), i.e. ~2.6 GB at 131 072 — at that depth the read of the KV is as large as the weights
themselves, and how much of that traffic becomes tokens is the whole difference above.

### In service: HTTP, real text, same prompts

`bench/api_bench.py` on both servers (`--prefill`, `--decode-ctx`), prefill with a fresh nonce (no prompt cache),
decode over `bench/spec-prompts.txt` with thinking off.

| prefill (tokens) | Malleus | llama.cpp | ratio |
|---|---|---|---|
| 411 | **748.1 tok/s** | 409.0 tok/s | 1.83x |
| 5 940 | **1025.5 tok/s** | 465.2 tok/s | 2.20x |
| 23 634 | **862.2 tok/s** | 349.6 tok/s | 2.47x |
| 94 413 | **445.7 tok/s** | — | — |

| decode (128 tokens) | Malleus | llama.cpp | ratio |
|---|---|---|---|
| ctx 0, greedy | **35.25 tok/s** | 28.95 tok/s | 1.22x |
| ctx 0, sampling (T 0.7, top_p 0.8, top_k 20) | **34.14 tok/s** | 28.79 tok/s | 1.19x |
| ctx ~24k, greedy | **29.02 tok/s** | 24.42 tok/s | 1.19x |
| ctx ~24k, sampling | **28.23 tok/s** | 24.43 tok/s | 1.16x |
| ctx ~94k, greedy | 19.66 tok/s | — | — |

Here Malleus also wins the decode, but **because of MTP** (speculative decoding with the head in the file: one
`nextn` layer, exactness checked by `malleus spec-exact`): without it the two engines are the same (29.08 against
28.30). `llama.cpp` has no MTP for this architecture — it would need a separate draft model, and there is none
matching this checkpoint.

### Long-context recall and the cost of a long prompt

`bench/needle.py`: a unique fact planted in the middle of an English corpus (Wikipedia articles, ~4.2 characters per
token), the question in English, greedy decoding, a nonce in front so no engine serves it from the prompt cache. (The
table below was measured with the earlier Italian wording of the marker and of the question: the shape of the test —
one unique fact, one question, at a given depth — is identical.)

| context | Malleus | llama.cpp |
|---|---|---|
| 6 815 tokens | recalled, 975.5 tok/s prefill, 7.4 s | recalled, 426.1 tok/s, 16.3 s |
| 27 003 tokens | recalled, 821.2 tok/s prefill, 33.2 s | recalled, 313.4 tok/s, 86.6 s |
| 109 366 tokens | recalled, 404.1 tok/s prefill, **271 s** | recalled, 143.3 tok/s, **764 s** |

Both engines find the fact at the hardest position (the middle) up to 110K tokens; what changes is the wait: 4.5
minutes against 12.7. At 131 072 tokens the same prompt takes 437 s to prefill here and ~1 600 s there.

### Quality: how far the 4-bit file is from the bf16 checkpoint

Reference: the bf16 checkpoint (`/srv/llm/models/bf16/ornith-ai/Ornith-1.5-35B-A3B`, transformers on the GPU).
20 prompts from `scripts/golden.py`; the score is the first generated token against the bf16 distribution.

| | Malleus (MLS4) | llama.cpp (Q4_K_M) |
|---|---|---|
| top-1 of the last prompt token (19 prompts both engines could run) | 13/19 | 14/19 |
| KL, mean / median / max | 0.407 / 0.162 / 1.500 | 0.228 / 0.064 / 2.144 |
| argmax over **every** position of the prompt | 50.65 % (351/693) | not measurable over HTTP |
| greedy 52 tokens against bf16 (`en-explain`) | **11/52** | 6/52 |

The two engines miss the top-1 on mostly different prompts (3 in common out of 8 misses, half of them where bf16
itself is unsure, top-1 logprob between −0.5 and −0.9), so the two quants are equally far from the checkpoint: no
quality is being traded for the prefill speed.

The KL column is not the same number: Malleus computes it over the whole vocabulary, `bench/quality_api.py` only
over the reference top-10 (the rest charged at the floor of what the engine returned), which is a lower bound by
construction. Compare the top-1 and the greedy tokens, not the KL.

## Inkling-2-Mini — the looped transformer (2026-11-21)

A 15.3 GB MLS4 file of a **looped** model: 24 physical layers of which the middle 12 (`6..18`) run three passes,
so a token costs 48 layer visits — three times the expert traffic of a stack of the same depth. Attention residuals
over a running block stack, a kernel-4 short convolution on every layer input and output, a relative-position bias
over the last 1024 positions, 128 experts with 8 active, and an 8-bit head over a 201 024-token vocabulary.
Measured on gfx1150 (Radeon 890M), `malleus bench --prompt 512,2048 --decode 32`, with the decode running against
the KV cache (48 slots, one per layer and pass).

| | value |
|---|---|
| weights on the device | 14.41 GB text + 0.64 GB vision + 0.27 GB (MTP, loaded, not used yet) |
| session state | 393 KiB per token: 48 slots × 8 KV heads × 128 × fp32 for K and V, plus the convolution histories |
| prefill `pp512` / `pp2048` | **214.1** / **120.1** tok/s |
| decode `tg32` at ctx 512 / ctx 2048 | **12.8** / **9.5** tok/s |

What the prefill costs, measured by ablation on a 2048-token chunk (70.8 s before, 17.1 s now):

| | before | now |
|---|---|---|
| MoE experts, read once per (token, slot) pair | 40.9 s | **1.7 s** — `moe_sort` groups the pairs, so each expert crosses the bus once |
| attention (`inkling2_attn`) | 24.8 s | **10.3 s** — still 60 % of the chunk, and the next wall |
| everything else (dense MLS4 GEMMs, convolutions, mixes, head) | 5.1 s | 5.1 s |

The expert traffic was the first wall and it was pure redundancy: `moe_gate_up` reads an expert once per pair, so a
512-token chunk read 1.58 GB per token — the same as a decode step — and the prefill sat on the bandwidth wall
(38.3 tok/s × 2.24 GB/token = 86 GB/s, above the 80 GB/s `bench-bw` measures). Grouping the pairs (the kernels the
other families already use: `moe_sort`, `moe_gather_g`, `moe_w4a8`, `moe_combine`, plus `inkling2_quant_i8` for the
expert activation, which is already in the rotated basis) moved the wall to the attention.

The attention is one block per (token, head): the four warps take four keys at a time and the softmax runs online,
which made it 1.6× faster, but every query row still reads all of its K and V. What is left is the query tiling of a
flash-attention: 16 rows per tile reuse one K/V read, which is where the other ~5× are. It also needs a real tile
loop to lift the `[heads, T, T]` bias buffer (6.4 GB at 8K tokens) and to allow chunked prefill — today a prompt goes
through one forward and the server uses the whole prompt.

What 1 000 tok/s would take: 8.7 GFLOP per token in 1 ms, i.e. 8.7 TFLOPS sustained over the whole graph. The
grouped MoE alone already measures ~8 TFLOPS, so the experts are not the problem; the projections run at ~0.9 (they
are 2.4 GFLOP per token and 2.7 ms of the 3.46 the non-attention path costs) and the machine's practical int8-WMMA
ceiling, measured on that same MoE, is ~9.6 TFLOPS. A tiled attention plus projections at the level the engine
already reaches for Ornith (5-6 TFLOPS) gives ~1.5-1.7 ms per token, i.e. **600-660 tok/s**: that is the honest
target on a 16-CU gfx1150. The 1 000 mark needs 40 CUs: on gfx1151 the same kernels are 2.5× the compute and 2.9×
the bandwidth, and that is where the number becomes reachable.

### What the context costs

Fresh prompt at every depth (one forward, as the graph runs today), and the same prompt with the attention kernel
removed by ablation — the rest of the graph, at the same weights:

| depth | prefill today | prefill without the attention kernel | decode today |
|---|---|---|---|
| 512 | 214.3 tok/s | **288.3** | 12.9 tok/s |
| 2 048 | 119.4 | **289.1** | 9.6 |
| 4 096 | 91.6 | — | 8.5 |
| 8 192 | 63.4 | 289.1 | 6.95 |

Everything that is not attention is **flat with depth**: 289 tok/s at 512 tokens and 289 at 8 192, i.e. 3.46 ms per
token for 8.7 GFLOP of experts and projections (2.5 TFLOPS). The degradation is one kernel: `inkling2_attn` reads the
K and V of every visible key for every query row, with no reuse across the 16 rows a tile could share, so its traffic
is `depth × 8 KiB` per slot per token — 832 MiB per token at 8 192 tokens over the 48 slots, which is 10.4 ms of
bandwidth against the 15.8 ms a token takes. Tiling the queries would divide that traffic by 16 and make the
attention compute-bound: ~0.2 ms per token at 8 K, so the projection is ~270 tok/s at 8 K (retention 95 % instead of
30 %) and ~200 at 32 K.

Decode has the same shape of problem from the other side: 78 ms per token at a 512-token context, of which 28 is the
2.24 GB of weights at the bandwidth wall, ~4 the sliding layers' 1024-key window and the rest the decode attention
kernel, which runs at ~5 GB/s of effective traffic where the machine does 80. At the wall decode would keep
~30 tok/s to 8 K and ~23 at 32 K with fp16 keys and values (halving the full slots' traffic); today the eight
full-attention slots read `32 KiB` per position per token in fp32.

**Against the production model** (Ornith-1.5-35B-A3B, same machine, `malleus bench`): prefill 214.1 against 968.5
tok/s at depth 0, decode 12.8 against 29.08 (MTP off on both sides). The prefill gap is now all in the attention;
the decode gap is the bandwidth of 2.24 GB per token against Ornith's ~0.7, i.e. the loop reading the core layers
three times.

### Quality: the gates against the bf16 checkpoint

`golden` on the 20 prompts of `golden/inkling2` (`scripts/golden.py --reasoning-effort none` on the bf16 checkpoint,
the looped forward), greedy off:

| metric | prefill (one forward) | decode (one token at a time, through the cache) |
|---|---|---|
| last-token top-1 against bf16 | **19/20** | **20/20** |
| all-position argmax | 71.39 % (519/727) | 67.81 % (493/727) |
| worst KL | 0.578 (`raw-continue`); 1e-5…1e-4 on the other 19 | 0.302 |

`raw-continue` is the one prompt of the set that is not rendered through the chat template, and the only one whose
KL is not in the 1e-5 range: it is where the file's 4 bits show. It is also the prompt the grouped prefill's int8
activations move (the W4A8 GEMM against the per-pair GEMV's fp32), and the reason the prefill column reads 19/20
while the decode path keeps 20/20: `MALLEUS_INKLING2_MOE_PAIRS=1` runs the old path, and the two differ by that one
prompt.

`eval` on the 8 documents of the eval set (four Gutenberg books, two source files, two chat continuations):

| document | ppl (ref) | top-1 | KL |
|---|---|---|---|
| it-pg56775 | 13.040 (12.723) | 90.53 % | 0.0273 |
| it-pg18456 | 9.093 (8.979) | 91.60 % | 0.0259 |
| en-pg48320 | 7.302 (6.949) | 91.80 % | 0.0485 |
| en-pg11 | 5.432 (5.257) | 91.99 % | 0.0266 |
| code-utils | 1.983 (1.912) | 96.00 % | 0.0278 |
| code-module | 2.242 (2.182) | 96.00 % | 0.0217 |
| chat-it | 55.261 (29.000) | 78.91 % | 0.3542 |
| chat-en | 45.184 (40.006) | 83.40 % | 0.1771 |
| **total** | **9.220 (8.188, +12.60 %)** | **90.03 %** | **0.0887** |

The two chat documents are the outliers (+90 % and +14 % perplexity against +1.7…+5.3 % for the six plain ones) and
they are the only documents over 1 000 tokens whose tokens are *not* plain prose: they open with the reasoning-effort
system line of the chat template. Open question, tracked in issue #70.


## How these numbers are produced

The reference tensors the quality gates compare against are published next to the model files, with the same layout
as this repo plus the logits (`golden/<model>/text/<id>/logits_last.f32`, `images/<name>/*.f32`, `eval/topk.bin`):

```sh
hf download duckpage/malleus-golden --local-dir /srv/models/malleus-golden
```

What follows generated them in the first place, and is what to re-run when a model changes: it needs the bf16
checkpoint and a GPU, so it happens once per model.

```sh
# engine level, prefill and decode at each depth (synthetic tokens)
scripts/dev.sh /target/release/malleus bench --model /models/ornith-15-35b-a3b-mls4.malleus \
  --depth 0,8192,16384,32768,65536,98304,131072 --chunk 8192 --decode 64

# the same on llama.cpp
PROMPTS=8192 GEN=64 LLAMA_ARGS="-d 0,8192,16384,32768,65536,98304,131072 -n 64" \
  bench/baseline-llamacpp.sh /srv/containers/aitools/models/gguf Ornith-1.5-35B-Q4_K_M.gguf

# in service: prefill and decode from the server's own timings
python3 bench/api_bench.py http://localhost:8090 --api-key KEY \
  --prefill 512,8192,32768,131072 --decode-ctx 0,32768,131072 --max-tokens 128

# long-context recall (the filler is any long text; 520 000 characters ~ 110 000 tokens)
python3 bench/needle.py http://localhost:8090 --api-key KEY --filler book.txt --chars 32000,131000,520000 --pos 0.5

# quality against the bf16 checkpoint: engine level, then through the API for another engine
scripts/golden.sh python3 scripts/golden.py --model /models/bf16/ornith-ai/Ornith-1.5-35B-A3B \
  --out /golden/ornith-15-35b --blob-dir /golden/ornith-15-35b --skip-images --skip-greedy
scripts/dev.sh /target/release/malleus golden --model /models/ornith-15-35b-a3b-mls4.malleus \
  --golden /golden/ornith-15-35b/golden.json --blobs /golden/ornith-15-35b --greedy
python3 bench/quality_api.py http://localhost:8091 --golden /golden/ornith-15-35b/golden.json \
  --tokenizer /srv/llm/models/bf16/ornith-ai/Ornith-1.5-35B-A3B/tokenizer.json
```

`scripts/golden.py` writes both `golden.json` and the blobs; the greedy reference needs the GPU for ~200 s per 128
tokens on this machine (bf16, transformers), so run it with `--skip-greedy` when only the last-token distribution
is needed.

## How the numbers are checked (the gates we run before a release)

Every claim above that touches output quality or the speculative draft has a gate that fails when it regresses.
They run on both architectures; the golden blobs are published next to the model files.

| Gate | What it checks | Command |
|---|---|---|
| `golden` | Last-token distribution against the bf16 reference: KL, argmax and top-1 over every golden prompt. Short prompts only (the longest entry is 4 500 tokens, below the sparse-attention threshold). | `malleus golden --model <file> --golden <golden.json> --blobs <dir>` |
| `golden-long` | An 11 000-token QSA prompt against a layer-by-layer bf16 reference: the sparse path, that the short golden never reaches. | `bench/golden-long.sh` (reference blob generated once with `scripts/ref_layerwise.py --long 11000 --attn eager`) |
| `idx-recall` | Retrieval: a fact planted in a 16K/32K/256K token stream must survive the indexer selection in the question rows (checking the `idx_select` bitsets, per layer). | `bench/idx-recall.sh` (fixtures = token ids, no tokenizer round trip; 256K fixtures generated into the golden volume) |
| `spec-exact` | The speculative verify is bit-identical to the serial step: every verified row, and every committed token at temperature 0, must match. Drafts may propose; they may not change the text. | `malleus spec-exact` |
| `eval` | KL, top-1 and accuracy over **every** position of the prompt against the bf16 top-k blobs, chunked prefill. | `malleus eval --model <file> --set <set.json> --blobs <dir>` |
| `kernel-test` | On-GPU numerical tests of the pipeline kernels with a seeded random input (Hadamard/MLS paths), independent of any model. | `malleus kernel-test` |
| `attn_equiv` | End-to-end smoke: default vs alternate gather shape (one query per block / per query) on a prompt long enough to engage the gather; catches crashes, wrong-kernel picks and gross drift (read the text diff as a smoke test only — greedy is chaotic in prefill logits). | `bench/attn_equiv.sh [repetitions]` |
| `quality_api` | Scores any OpenAI-compatible HTTP engine on the same golden prompts and reference, via `max_tokens: 1` + `top_logprobs` (the API-level counterpart of `golden`, for engines that only speak HTTP). | `bench/quality_api.py <url> --golden <golden.json> --tokenizer <tokenizer.json>` |
| `needle` | Long-context recall over HTTP, on any engine: a unique fact at 1/3 or 2/3 of a long filler, question in English, fresh nonce so nothing serves it from the prompt cache. | `bench/needle.py <url> --api-key K --filler <text> --chars 32000,131000,520000 --pos 0.5` |

Notes: `golden` and `eval` compare against the bf16 checkpoint, so they need it to be re-generated when a model
changes; the blobs are fetched with `hf download duckpage/malleus-golden`. `quality_api` needs an engine that
already supports `top_logprobs` — that is, not ours yet (see the logprobs API issue). A gate that is red on a
claim that was never tested is not a regression: `golden-long` at 11K covers the gather threshold; recall beyond
32K is covered by `idx-recall` up to 256K.

## What is worth measuring next

- **Ornith-1.5-122B with the experts on disk** (issue #14): where the engine differs structurally from llama.cpp
  is running a model that does not fit; 1.5-3 tok/s of decode is the current state, not a limit to live with.
- **QSA on Ornith** (issue #10): this checkpoint has no indexer, so the numbers above are all dense attention. With
  a selection budget the KV traffic at 128-256K drops by two orders of magnitude.
- **gfx1151**: the same curves on Strix Halo. The hardware spec is in the table at the top; the per-depth and
  in-service numbers above are gfx1150, and qwen38-flash-next (the family the indexer kernels were written for)
  is gfx1151-only — so a Strix Halo duplicate of the depth table is the open item, not the defaults (they switch
  automatically, see the top table).
