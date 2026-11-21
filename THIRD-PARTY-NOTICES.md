# Third-party notices

Malleus is written here, from public sources: the model architecture and its upstream reference
implementation, published papers, the GGUF container specification and AMD's public GPU documentation. No source
code from another inference engine is vendored or adapted, and no file taken out of another engine's distribution
is included (the clean-room rule is in [AGENTS.md](AGENTS.md), section *Clean-room*).

This file lists what the project **redistributes** in the container image, and the MIT/Apache-licensed
implementations that were **consulted as references**, which the clean-room rule asks to record.

## Redistributed in the container image

The runtime image (`deploy/Containerfile`, target `runtime`) copies, unmodified, the shared libraries the engine
resolves at runtime. Their own licenses govern them, not this project's.

### AMD ROCm runtime libraries

`deploy/collect-runtime-libs.sh` resolves with `ldd` the ROCm libraries `malleus` links (plus `libamd_comgr.so`,
`libhsa-runtime64.so`, `librocprofiler-register.so`, which HIP loads with `dlopen`) and copies them into
`/opt/malleus/lib/`; the exact list, with a hash of each file, is written at build time to
`/opt/malleus/RUNTIME-LIBS.txt` inside the image. Copyright © Advanced Micro Devices, Inc.; the ROCm projects
are distributed under their own licenses, mostly MIT with some components under the University of Illinois/NCSA
license. Nothing in them is modified here.

### Debian runtime base

The image is built `FROM debian:trixie-slim`. The C and C++ runtime that travel with it keep their own licenses:
glibc under the GNU Lesser General Public License v2.1 or later, libstdc++ under the GNU General Public License v3
or later with the GCC Runtime Library Exception.

## Formats and specifications followed

### GGUF v3 (ggml / llama.cpp) — MIT

The `.malleus` file is a GGUF v3 container (`malleus-gguf`): same header, same metadata and tensor records, same
type numbering, with three tensor types of this project (`MLS3`, `MLS4`, `MLS8`) beyond the ggml ones. No ggml
source is copied or translated — the shape of the format is followed so that existing tooling can read the files.
Copyright © the ggml authors, MIT license.

## Implementation references (consulted, no code copied)

### transformers — Apache-2.0

The reference implementation of the model architectures, used by `scripts/golden.py` to produce the golden logits
and by `scripts/eval_ref.py`. It runs in the offline `golden` container, never in the engine and never at runtime.
Copyright © Hugging Face, Apache License 2.0.

### SGLang — Apache-2.0, DFlash — MIT

The semantic of the DSpark/DFlash speculative draft (a block of masked positions proposed in one forward and scored
with the target's `lm_head`) has no reference implementation in the checkpoint itself: SGLang is the one that ships
it, and `z-lab/dflash` documents the drafting loop. `src/crates/malleus-model/src/dflash.rs` was written here from
the checkpoint geometry and from that documented behaviour; it is cited because the clean-room rule names
MIT/Apache sources consulted as references. Copyright © the SGLang authors (Apache License 2.0) and the DFlash
authors (MIT).

### AngelSpec — Apache-2.0, vLLM — Apache-2.0

Hy4-flash's DFly2 draft (`model.dfly_layers.*`) is not in `transformers`. Its forward (slot-0 fusion, the blocks on
the target's KV cache and indexer selection, the hidden correction before the LM head) follows the training code of
`Tencent/AngelSpec` (`angelspec/models/draft/dflash.py`, `hy4_mla.py`, commit `9ee5a9de`) and the drafting loop
of vLLM's Hy4 speculator (`vllm/models/hy_v4/nvidia/dfly_speculator.py`, `dfly_inputs.py`). The draft in
`src/crates/malleus-model/src/hy4.rs` was written here from those descriptions. Copyright © Tencent (Apache
License 2.0) and the vLLM authors (Apache License 2.0).

## Programs used for measurement, not distributed

`llama.cpp` (MIT) is the baseline in `MODELS.md`: it is run as an external program from its own container image
(`bench/baseline-llamacpp.sh`), with its own defaults, and none of its code enters this repository.
`transformers` and `torch` (Apache-2.0, BSD-3-Clause) likewise run only in the offline `golden` image, to produce
the reference logits.

## Model weights

Weights are not part of this repository and keep the license of their original model. `MODELS.md` says which
checkpoint each file was converted from; the conversion changes the storage format, not the terms.

## License of Malleus itself

The **Apache License 2.0** ([LICENSE](LICENSE)), with the attribution notices in [NOTICE](NOTICE) and the names
covered by [TRADEMARKS.md](TRADEMARKS.md). The third-party components above keep their own terms, whatever this
project's license is.
