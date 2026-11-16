# Contributing to Malleus

Thanks! Bug reports, benchmarks on new hardware, kernels, new models and documentation are all welcome.

The workflow — **issue → branch → verify → PR**, with commits **without a co-author** — is described in
[AGENTS.md](AGENTS.md) and applies to both people and agents.

## Rules for changes

- Rust stable, edition 2024. `cargo fmt`, `cargo clippy -- -D warnings` and the quality gates must pass.
- `unsafe` only in `malleus-hip`, each block with a `// SAFETY:` comment.
- Kernels must build and pass the numerical checks on both **gfx1150** and **gfx1151**.
- No Python at runtime: Python only in the offline tools under `scripts/`.
- The binary opens no outbound network connections.
- One commit per logical change. Performance changes include before/after numbers (`malleus bench`) and the
  machine they were measured on; quality-relevant changes include the `malleus golden` results.
- The reference tensors the gates compare against are not in the repo (they are tens of MB per model): they live on
  the Hub, next to the `.malleus` files. `golden/<model>/` holds the prompts, the ids and the vision fixtures; the
  logits come from

  ```sh
  hf download duckpage/malleus-golden --local-dir /srv/models/malleus-golden
  malleus golden --model <model>.malleus \
    --golden golden/<model>/golden.json --blobs /srv/models/malleus-golden/golden/<model>
  ```
- Never copy, translate or disassemble code or kernels from proprietary software, and never use closed weight
  formats as a source (clean-room).

## Layout and documentation

- Sources live in `src/crates/` (Rust) and `src/kernels/` (HIP). See [AGENTS.md](AGENTS.md) for the layout.
- **English, everywhere**: code, comments, commit messages, issues, pull requests, documentation.
- Engineering notes (architecture, plans, ADRs) are not kept in this repo. To add a model: `malleus convert` it,
  measure it, add it to [MODELS.md](MODELS.md) with the numbers (version, machine, what produced them).

## License of the contributions

Malleus is distributed under the **Apache License 2.0** ([LICENSE](LICENSE)); the attribution notices are in
[NOTICE](NOTICE), the third-party components in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), and the names are
covered by [TRADEMARKS.md](TRADEMARKS.md) (the license gives you the code, not the name).

A contribution is accepted under the same license, and you keep the copyright on what you wrote. Nothing is signed
away and there is no CLA: what we ask is a **sign-off** on every commit, which is the
[Developer Certificate of Origin](DCO) 1.1 — the same rule the Linux kernel uses:

```sh
git commit -s -m "area: cosa cambia e perché"
```

That adds a `Signed-off-by: Nome <email>` line, and with it you certify that you have the right to submit the work
under Apache-2.0 (see [DCO](DCO)). A pull request whose commits are not signed off is asked to amend them; it is a
formality, not a review.

Apache-2.0 is permissive on purpose: a company can take this code, modify it and ship it in a closed product, as
long as it keeps `LICENSE`, `NOTICE` and the attribution notices. That is the trade we chose — the code is not the
moat, the tuning and the operations are — and it is why contributions are welcome from commercial users too.

The name *Malleus* is not part of that grant: a modified build distributed under another name is fine, the same
name is not.
