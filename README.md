# Malleus

[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-duckpage-FFD21E)](https://huggingface.co/duckpage)

An open-source LLM inference engine for AMD RDNA 3.5 APUs — **gfx1150** (Strix Point) and **gfx1151**
(Strix Halo). Rust host, hand-written HIP kernels. Developed and used in production by
[DuckPage](https://github.com/duckpage).

Strix APUs share a large pool of system memory with the GPU (96–128 GB), a cheap way to run large modern models
locally. Malleus exists for the case general-purpose engines serve worst on these chips: **a long prompt on a
machine with the memory bandwidth of a laptop**. Prompt processing is what this engine is built for — on
Ornith-1.5-35B-A3B it prefills **2.1x faster than llama.cpp in a fresh session and 2.8x across 131 072 tokens**
(196 against 69 tok/s on the same box), so a 110 000-token prompt costs 4.5 minutes instead of 12.7. Decode is at
parity, because at 4 bits a decode step reads every weight it touches; the MTP head is how we go past that. The
curves, the method and the quality numbers are in [MODELS.md](MODELS.md).

So the goal is not breadth, it is **efficiency on the two chips above**: every kernel is written for them, the model
files are quantized for their memory hierarchy, and the features around them — prefill chunking, the KV and indexer
budgets, the MTP head, the disk-backed expert cache — exist to keep a long context alive on one APU. The engine is
readable and auditable, written only from public sources: it contains no code, kernels or weight formats from any
closed engine.

## Status

| | |
|---|---|
| Models | Five families carried, with the file sizes on the device and the tokens/s measured: [MODELS.md](MODELS.md) |
| Hardware | gfx1150 (development and production), gfx1151 (supported) |
| API | OpenAI Chat Completions + Anthropic Messages, streaming, tool calls, image input, one port. No `logprobs` yet (#102) |
| Speculative decoding | MTP head + chained drafts, verified in one pass |

### Where it stands

On the model we run in production ([MODELS.md](MODELS.md) for the full sweep and the method):

- **prefill of a long prompt**: 2.07x at a fresh session, 2.83x at 131 072 tokens against llama.cpp;
- **decode**: parity kernel against kernel, ~20 % more in service thanks to the MTP head;
- **quality**: the same distance from the bf16 checkpoint as llama.cpp's `Q4_K_M` — 13/19 against 14/19 of top-1 on
  the golden prompts, and the greedy output is the closer one (11/52 against 6/52 tokens identical to bf16).

A dense model (Qwen3.8-27B) is bandwidth limited on both engines and there Malleus measures ~5 % ahead
(pp512 209.3 against 191.7 tok/s, tg128 5.02 against 4.74); the MTP head is what breaks the ~5.8 tokens/s that
~80 GB/s of measured bandwidth allows a 4-bit 27B.

## How it works

- **Weights**: `.malleus` files (a GGUF v3 container, architecture `malleus.<family>`) with **MLS4** (4-bit,
  randomized Hadamard rotation, group size 128, GPTQ-calibrated) for decode *and* prompt processing; `convert
  --prefill-int8` adds an **MLS8** (int8 per channel) copy so the prompt is read at 8 bits instead — twice the file,
  better prompts, decode unchanged. The name says which one it is: `-mls4` is one copy, `-mls4-mls8` is both.
- **Kernels**: hand-written HIP for gfx1150 and gfx1151 — GEMV for decode, fused W8A8 WMMA GEMMs for prefill, chunked
  Gated DeltaNet, flash-decoding attention, vision tower.
- **Host**: a single Rust binary (`malleus`); `unsafe` confined to the HIP bindings; no Python at runtime.
- **Security**: no outbound network connections; rootless, read-only container.

## Quick start

Requirements: a Linux host with the `amdgpu` driver, `/dev/kfd` and the GPU render node, and a GTT limit large
enough for the model.

```sh
# build the dev container (Podman or Docker)
podman build -f deploy/Containerfile --target runtime \
  --build-arg MALLEUS_GPU_TARGETS="gfx1150;gfx1151" -t malleus:dev .

# convert an upstream bf16 model to a .malleus file: one 4-bit copy, the faster file
malleus convert --model /models/Qwen3.8-27B --out /models/qwen38-27b-mls4.malleus

# the same with an 8-bit copy of the weights for prompt processing: twice the file, better prompts
malleus convert --model /models/Qwen3.8-27B --out /models/qwen38-27b-mls4-mls8.malleus --prefill-int8

# run a prompt, or serve the API on port 8090
malleus run   --model qwen38-27b-mls4.malleus --prompt "Hello!"
malleus serve --model qwen38-27b-mls4.malleus
```

Use the render node of your AMD GPU (`ls -l /dev/dri/by-path/`). A Podman quadlet and a Compose file are in
[deploy/](deploy/). Other commands (`malleus --help`): `bench` (`--depth 0,8192,…,131072` grows one session and
times prefill and decode at each depth: the measurement behind [MODELS.md](MODELS.md)), `bench-bw`, `spec-exact`
(a speculative verify must be bit-identical to the serial step, or it exits non-zero), `kernel-test`, `eval`,
`golden`, `devices`, `draft-info`, `draft-vocab`, `health`, `slim` (rewrite a `.malleus` without its MLS8 prefill
copies, without the bf16 source). `bench/needle.py` checks over HTTP — so on any engine — that a unique fact planted
in the middle of a 110 000-token text is still recalled, and reports prefill and decode at that depth;
`bench/api_bench.py` does prefill and decode over HTTP from the server's own timings. The standard `malleus golden`
stops below the sparse-attention threshold; `bench/golden-long.sh` runs the 11 000-token QSA entry against its bf16
reference (the blob is generated once with `scripts/ref_layerwise.py --long 11000 --attn eager`).
`bench/idx-recall.sh` plants a unique fact in a 16K/32K token stream and checks that its block survives the
indexer selection in the question rows; `MALLEUS_RECALL_FIXTURES` selects other fixture globs (the 256K fixtures
are generated into the golden volume, not committed).

## Environment variables

Every setting is also a flag (`malleus serve --help`); the environment is what a container uses. "Off" means the
engine behaves as if the variable were unset. Defaults that depend on the model or the GPU say so.

### Server

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_MODEL` | path (required) | The `.malleus` file (or safetensors directory) to load. |
| `MALLEUS_MODEL_NAME` | string (model file stem) | The `model` name the API answers to and reports in `/v1/models`. |
| `MALLEUS_HOST` | `0.0.0.0` | Address the API listens on. |
| `MALLEUS_PORT` | `8090` | Port for the API (Chat Completions, Messages, `/health`). |
| `MALLEUS_API_KEY` | string (off) | When set, every request needs `Authorization: Bearer <key>`. |
| `MALLEUS_CTX` | tokens, default `32768` | Maximum context. Sizes the KV state, so it is the main memory dial. |
| `MALLEUS_MAX_TOKENS` | tokens, default `8192` | Default reply budget when the request does not set `max_tokens`. |
| `MALLEUS_DEFAULT_THINKING` | `true` / `false` (default `true`) | Reasoning on/off when the request does not say. |
| `MALLEUS_VISION` | `0` turns it off | Load the vision tower when the checkpoint has one. |
| `MALLEUS_REQUEST_TIMEOUT` | seconds, default `600`, `0` disables | A request running longer than this ends the process and the unit restarts it: HIP cannot cancel a kernel already running. Give a slow model a budget larger than its worst legitimate request. |

### Loading and memory

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_PREFILL_WEIGHTS` | `none` (default: load them) | `none` skips the MLS8 prefill copies: prompt processing runs on MLS4, less memory, lower quality. |
| `MALLEUS_PREFILL_INT8_FAMILIES` | comma list of name substrings, or `none` (default: all) | Which tensors keep their MLS8 prefill copy. `none` drops all of them. |
| `MALLEUS_MEM_RESERVE_GIB` | GiB, default: estimated from the prefill chunk | Working set the memory plan counts for one forward call; set it to override the estimate. |
| `MALLEUS_HOST_RESERVE_GIB` | GiB, default `8` | RAM the memory plan leaves to the host for the page cache of a table read from disk (the n-gram table); `0` for a model without one. |
| `MALLEUS_BUF_POOL` | `0` to disable, default on | Keeps freed device buffers of up to 16 MiB (512 MiB in all) for the next allocation of the same size, so a decode step makes no `hipMalloc`/`hipFree`. |
| `MALLEUS_MEM_IGNORE` | any value except `0`/`false` | Skip the memory pre-flight and let the allocator fail instead. |
| `MALLEUS_SLOTS` | sessions, default `1` | Sessions the memory plan assumes. It is not a batch: the scheduler serves one request at a time. |
| `MALLEUS_DECODE_INT8` | `1` (default off) | Decode on the MLS8 prefill copies of the non-expert linears: quality over speed. |
| `MALLEUS_ARGMAX_GPU` | `1` (default on) | Greedy decode computes the argmax on the device and reads back 4 bytes instead of the 1 MB logits row. Output identical; the host copy is gone, speed within noise (34.8-35.0 tok/s at 256K). `0` keeps the host argmax. |
| `MALLEUS_LMHEAD_W4` | `1` (default off) | Convert `lm_head` to MLS4 at load: half its bytes per token, a few argmax points. |
| `MALLEUS_HC_W4` | `0` (default on) | Drop the hyper-connection MLS4 weights and decode them from the MLS8 copy. |
| `MALLEUS_MTP` | `0` turns it off | Load and use the MTP head (speculative decoding). |
| `MALLEUS_MTP_INT8` | `0` (default on) | The MTP head decodes on the int8 copy: more accepted drafts, more bytes. `0` keeps it at 4 bits. |
| `MALLEUS_MTP_HIDDEN` | `post` (default: last layer output) | Feed the MTP head the post-norm hidden state. |
| `MALLEUS_MTP_CHAIN` | `pre` (default `post`) | Chained drafts start from the MTP output before its final norm. |
| `MALLEUS_DRAFT_HEAD` | `int8` (default: build it) | `int8` does not build the reduced int8 draft head from the `lm_head` rows. |
| `MALLEUS_DRAFT_VOCAB` | N (default: full vocabulary, min 1024) | Score drafts on the first N token ids only. |
| `MALLEUS_DRAFT_VOCAB_FILE` | path (off) | Token ids, one per line, that drafts are restricted to (`malleus draft-vocab` writes one). |

### Prefill and kernels

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_PREFILL_CHUNK` | tokens, default `16384` with hyper-connections else `2048` | Tokens per prompt-processing chunk; the expert weights are streamed once per chunk. The working set grows with it (about 1 MiB per token on Flash-Next), and the memory plan counts it. |
| `MALLEUS_PREFILL_GEMM` | `f16` (default int8/W4A8) | Prompt GEMMs on fp16 activations instead of int8. |
| `MALLEUS_DENSE_W4A4` | `1`, `w4a8`, `acts`, `2`, `3` (default off) | `1`: int4 activations on the plain dense GEMMs. `w4a8`: also 4-bit weights for the projections whose input needs a transform (`o_proj`, `out_proj`). `acts`: int4 activations there too — faster, about −1.6 points of argmax. |
| `MALLEUS_DENSE_W4A8_ACTS` | `0` (default on) | `0` keeps those projections on int8, for an A/B of the line above. |
| `MALLEUS_MOE_W4A4` | `1` on gfx1151, `0` on gfx1150; also `glu` | Activation format of the routed experts in prefill: `1` int4, `glu` int4 on gate/up only, `0` int8. |
| `MALLEUS_EXPERT_CACHE` | GiB (default off) | Keep the routed experts on disk and cache the hot ones in per-layer slot pools; decode reads slot indices, prefill streams the whole layer (bit-identical, much slower prefill). Issue #50. |
| `MALLEUS_MOE_Y_HALF` | `1` (default off, f32) | Write the MoE intermediate `y` in fp16: half its bytes, measured +3,5 % of prefill. Ignored for checkpoints with int8 expert copies. |
| `MALLEUS_MOE_BM128` | rows per expert, default `48` | From this average rows-per-expert the grouped MoE switches to 128-row tiles. |
| `MALLEUS_MOE_SPLIT` | `0` (default on) | `0` drops the second, half-height tile list the MoE uses for the remainders of each expert. |
| `MALLEUS_W4_MAX_T` | tokens, default `0` | Inputs up to this many tokens use the 4-bit weights (W4A8). `0` never. |
| `MALLEUS_HC_FUSED` | `0` (default on) | Disable the fused hyper-connection kernels used by single-token decode. |
| `MALLEUS_MLA_PREFILL` | `1` (default off) | Opt-in tiled MLA prefill. |
| `MALLEUS_DN_SCAN` | any value (default off) | Debug: DeltaNet falls back to the token-by-token scan instead of the chunked one. |

### Attention and indexer

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_ATTN_SEL` | `0` (default on) | Disable the sparse block selection in decode and verify. |
| `MALLEUS_ATTN_GATHER` | `0` (default on) | Disable the sparse gather in prefill: attention runs dense. `1` forces the per-query gather instead of the architecture default. |
| `MALLEUS_GATHER_Q1` | `1` on gfx1151, `0` on gfx1150 | Sparse gather shape: `1` is one query per block (128 threads, four waves), `0` the four-query union. On gfx1151 `1` measured +8…+11 % of prefill at 64–98K tokens; gfx1150 keeps the union, which measured 6–9 % faster there at 12/24K. |
| `MALLEUS_GATHER_MIN` | multiple of the attention budget, default `5` | Context length from which prefill switches to the gather. Lower is faster and less exact; `3` measured +2.2 % of prefill at 24k tokens. |
| `MALLEUS_IDX_SCORE_TILE` | `1` on gfx1151 with a 4-head indexer, `0` otherwise | Score a 16-query tile while walking 64-key tiles; `0` uses the 4-query kernel. Scores are bit-identical; measured 7.95 → 5.27 s at 98K tokens. |
| `MALLEUS_IDX_SELECT` | `12` on gfx1151 from 12 288 blocks, `0` otherwise | `12` = three radix passes of 12/12/8 bits (4096 bins), `0` = four passes of 8 bits. Same bitset; the 4096-bin histogram wins only on long rows. |
| `MALLEUS_IDX_KT` | key tiles per score block, default `8` in prefill and `1` in decode | Tuning of the tiled score kernel; 8 reuses the q tile, 1 keeps the key parallelism of decode. |
| `MALLEUS_IDX_GROUP` | `4` on gfx1151 when the 12/12/8 select is active, `1` otherwise | Score and select one representative query per group of N, copy its selection to the group and mask the few future keys in the gather. 4 makes the 32K-256K curve flat (`1 453` to `1 422` tok/s at 256K); `1` restores per-query selection. Prefill only: decode/verify (`nq = 1..6`) always fall back to `1`. |
| `MALLEUS_IDX_DECODE_REUSE` | `1` (default on) | Decode: reuse the previous token's block selection until a new 4-token block completes, then recompute. At 256K serial decode 31,0 → 34,2 tok/s; at 11K teacher-forced the argmax/KL are unchanged. `0` recomputes every token. |
| `MALLEUS_IDX_STATS` | any value (default off) | Debug: print the density of the selections and of their unions; `all` prints every layer. |
| `MALLEUS_IDX_DUMP_SEL` | path (default off) | Debug: write the selection bitsets of the last sub-batch, one file per indexer layer, for `bench/idx-recall.sh`. |

### Speculative decoding and the prompt cache

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_SPEC_DRAFTS` | drafts per round, default `5` (capped at `GEMV_MAX_T - 1`) | Drafts verified per round; `0` turns speculation off. |
| `MALLEUS_DRAFT_PMIN` | f32, default `0.8` | A chain of drafts goes on while the product of their probabilities stays above this. |
| `MALLEUS_LOOKUP` | `0` (default on) | Disable the prompt-lookup drafter (copies from the request's own text). |
| `MALLEUS_LOOKUP_NGRAM` | tokens, default `3` | Length of the n-gram the prompt lookup matches. |
| `MALLEUS_LOOKUP_NORM` | `1` (default off) | Also draft from the original text of the recent messages, with line numbers and JSON escapes removed. Measured no gain at 5 drafts; it is the substrate for deeper rounds. |
| `MALLEUS_PROMPT_SNAPSHOTS` | N, default `8` | Prompt states kept for prefix reuse; `0` disables the snapshot cache. |
| `MALLEUS_WARMUP` | `0` (default on) | Skip the warm-up request at startup (it costs a chunk of prefill, it saves the first request a lot). |

### N-gram table, rope, build, debug

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_NGRAM_THREADS` | threads, default `256` | Threads that read the n-gram table for a prompt; a decode step (up to 16 tokens) reads its rows on the calling thread. |
| `MALLEUS_NGRAM_WARM` | `1` (default off) | Read the whole n-gram table at startup to warm the page cache. |
| `MALLEUS_ROPE_YARN` | factor > 1, default `1` (off) | YaRN scaling to go past the native 262144 positions (the container uses `2` at 512k). |
| `MALLEUS_ROPE_ORIG` | positions, default `262144` | Native rope length the YaRN ramp is built on. |
| `MALLEUS_GPU_TARGETS` | build time, default `gfx1150;gfx1151` | Architectures the kernels are compiled for. |
| `MALLEUS_RPATH` | build time, any value | Link the binary with `rpath=$ORIGIN/../lib`, which the container image needs. |
| `MALLEUS_DUMP_DIR` | directory (off) | Debug: write layer buffers as `{tag}.f32` for a layer-by-layer comparison. |
| `MALLEUS_EXPERT_CACHE_VERIFY` | any value (default off) | Debug: after every expert fetch, read the slot back and compare it with the file. |
| `MALLEUS_TRACE_ROUTE` | file path (off) | Debug: append the experts the router picked (`layer t row id:weight ...`), the input of `scripts/expert_cache.py`. Synchronizes once per layer, so a trace run is slower. |
| `MALLEUS_DEBUG_SNAPS` | any value (default off) | Debug: log the prompt-snapshot bookkeeping. |
| `MALLEUS_TIMING` | any value (default off) | Debug: print timings of the n-gram embedding step. |

## Documentation and contributing

- **Models and their measured performance** — the files we carry, their size on the device and the tokens/s per
  prompt length, engine against engine — are in [MODELS.md](MODELS.md).
- **Everything in the repository is in English** — code, comments, commit messages, issues, these docs. Engineering
  notes and plans are not kept here: what must not be lost goes into the issue that motivated it.
- **How to contribute** — build, workflow and the rules for changes — is in [AGENTS.md](AGENTS.md) and
  [CONTRIBUTING.md](CONTRIBUTING.md): every change goes through an issue, a dedicated branch and a pull request.

## License

**Apache License 2.0** ([LICENSE](LICENSE)): use it, modify it, ship it in your product, closed or open, as long as
you keep the [NOTICE](NOTICE) and the attribution notices — no CLA, no copyleft, and a patent grant on top. That is
deliberate: the code is not the moat, the tuning and the operations are, and a company that cannot take this code
into its stack cannot become a user of it.

| | |
|---|---|
| Code | Apache License 2.0 — [LICENSE](LICENSE) |
| Attribution and third parties | [NOTICE](NOTICE), [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |
| Names and logo | not licensed with the code — [TRADEMARKS.md](TRADEMARKS.md) |
| Contributions | Apache-2.0, signed off (`git commit -s`, [DCO](DCO)) — [CONTRIBUTING.md](CONTRIBUTING.md) |
| Model weights | keep the license of their original model |
| Redistributed in the image | ROCm runtime libraries, Debian base — licenses in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |

The software is provided **"as is"**, without warranty of any kind, express or implied, and without liability for any
damages arising from its use — see the [Apache-2.0](LICENSE) *Disclaimer of Warranty* and *Limitation of Liability*
(sections 7 and 8). The converted model weights keep the license of their original models and are not ours to warrant.
