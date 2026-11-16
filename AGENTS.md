# Agent instructions

This file is the operating guide for every agent (and person) working on this repository.
Read it before you start.

**Everything in this repository is in English**: code, comments, documentation, commit messages,
issues and pull requests. Engineering notes (architecture, plans, ADRs, reverse-engineering) are not
kept here; the repo holds what is needed to build, run, measure and contribute.

## Workflow: issue → branch → PR

When you work on a topic, **never work directly on `main`**. Always follow these steps:

1. **Open an issue** describing the goal, what you change and how you verify it.
2. **Create a dedicated branch** linked to the issue, named `type/short-description`
   (e.g. `perf/decode-moe-w4a4`, `fix/indexer-grid`, `docs/models-readme`, `chore/...`).
   Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `chore`, `test`.
3. **Work only on that branch.** One branch = one topic.
4. **Verify** that everything is fine (build + quality gates, see below) *before* considering
   the work done.
5. **Only when everything is OK**: commit on the branch, **push it** and **open the pull request**
   against `main`, whose description **starts** with a closing reference to the issue (see below).
   Merging is the owner's call: never merge a pull request yourself, and never push to `main`.

### Pull requests

- **Start the description with `Closes #NN`** (`Fixes #NN` and `Resolves #NN` work too), on its own
  line. That is the line that closes the issue: the forge only parses those English keywords in
  front of `#NN`, so the line has to stay exactly like that — a translated keyword does nothing, and
  it has already left four issues open after their pull request merged.
- **Never close the issue by hand.** Merging the pull request closes it; closing it earlier hides
  whether the change actually landed.
- **One pull request = one topic = one issue.** When the work covers only part of the issue, write
  which part in the description **without** a closing keyword, and leave the issue open with a
  comment saying what is done and what is left (examples: #22, #23).

### Labels

One label says **where** the work is, one (optional) says **how it ends**. Never more than three, never none:
an issue without a label is an issue nobody filters for.

| label | where |
|---|---|
| `prefill` | prompt processing, long-context ingestion |
| `decode` | token generation speed and its structure |
| `context-long` | what happens as the context grows: quality, sparse selection, K/V traffic |
| `kernels` | HIP kernels and their efficiency |
| `quantization` | MLS4/MLS8, int4, calibration |
| `speculative` | MTP, drafts, verify and rollback |
| `api` | the HTTP surface, OpenAI and Anthropic compatibility |
| `serving` | scheduling, batching, sessions |
| `memory` | the memory plan, device and host budgets, expert cache |
| `chore` | maintenance, housekeeping, the build |
| `documentation` | the README, MODELS.md, AGENTS.md |

How it ends, only when it is over:

| label | meaning |
|---|---|
| `done` | closed because it is in production, not just merged |
| `experiment` | an investigation whose result is history, not a plan |
| `negative-result` | tried, measured, and it does not pay — the measurement stays |
| `merge` | validation of a merge or of an integration |

The GitHub defaults (`bug`, `enhancement`, `question`, `invalid`, `duplicate`, `wontfix`) keep their usual
meaning and go **on top of** an area label, not instead of it. `good first issue` and `help wanted` are for
work that is open to someone outside the project.

## Commits

- **Commit on your topic branch — committing is expected, do not ask for permission.** You push that
  branch (nothing else) so that the pull request can exist.
- **Never add the agent as a co-author or modify commit authorship**. Always preserve the identity
  already configured locally, without adding any agent, model, or tool attribution.
- **Every commit is signed off** (`git commit -s`): the contribution enters under the Apache License 2.0 with the
  [DCO](DCO) 1.1. There is no CLA, so no copyright is assigned; a pull request without the sign-off is asked to
  amend its commits.
- Commit messages **in English**, imperative, stating the **why** of the change, not just the what.
  The same goes for issues and pull requests: one language, English, everywhere in the project.
- The repo owner merges the pull request and owns `main`; agents push only their own topic branch.

## Repository layout

```
src/crates/      the Rust crates (engine host, API, scheduler, convert, ...)
src/kernels/     the HIP kernels (.hip)
scripts/         build and run in the container (dev.sh), golden, calibration
bench/           benchmark tools and scripts
deploy/          example quadlet/unit files for deployment
Cargo.toml       Rust workspace (at the root)
README.md        quick start
MODELS.md        the models carried and their measured performance
AGENTS.md        this file
THIRD-PARTY-NOTICES.md  what we redistribute and what we consulted (licenses)
LICENSE          Apache License 2.0
NOTICE           attribution and copyright
TRADEMARKS.md    the names and the logo are not licensed with the code
DCO              Developer Certificate of Origin, the sign-off every commit carries
```

`src/` holds **only source files**: no notes, no `.md`, no data dumps.

## Documentation

- **English, everywhere.** Code, comments, commit messages, issues, pull requests and the four files
  below.
- Keep the `.md` files in the repo to the ones that ship with the code: `README.md` (quick start),
  `MODELS.md` (the models we carry and the numbers measured on them), `CONTRIBUTING.md` and this
  file. `MODELS.md` changes together with the code that produces its numbers: it is what every
  performance claim in the README points at. `LICENSE`-style files (`LICENSE`, `NOTICE`,
  `THIRD-PARTY-NOTICES.md`, `TRADEMARKS.md`, `DCO`) are not documentation: they stay in the repo by
  nature. Anything else does not go in the repo.
- Engineering notes, plans, ADRs and reverse-engineering write-ups are **not** kept here. What must
  not be lost — a measurement, a decision, a failure — goes into the **issue** that motivated it,
  with the version, the machine and the numbers.
- Numbers published under the name are reproducible numbers: version, model file, prompt set, flags
  and machine (see `MODELS.md` for the shape of it).

## Build

- Build and run go through the container: `scripts/dev.sh <command>`
  (e.g. `scripts/dev.sh cargo build --release --bin malleus`).
- HIP kernels are compiled **offline** by `build.rs` with `hipcc` (no GPU needed to build).
- GPU targets: `gfx1150` and `gfx1151` (variable `MALLEUS_GPU_TARGETS`).
- Before opening a PR: `cargo build`, `cargo fmt --check` and `cargo clippy -- -D warnings` must pass.

## Model files

- `malleus convert` writes **one MLS4 copy** of the weights by default: it is the fastest file, and its prefill
  reads the same 4-bit weights as decode.
- `--prefill-int8` adds an **MLS8 copy for the prefill**: twice the file (~35 GB more on the device for a 35B
  MoE) and it buys prompt quality, not decode speed. Measured on Qwen3.6-35B-A3B: prefill +11…29 % without the
  copy, decode unchanged, and `eval` top-1 93.87 % (single copy) against 97.02 % (double copy).
- **The name says which one it is**: `<model>-mls4.malleus` for a single copy, `<model>-mls4-mls8.malleus` for
  the double one. The converter warns when the name and the content disagree; never hide a second copy behind a
  name that does not say so.
- Prepare model files on the big disk (`/srv/llm/models`), keep **only the model in production** on the fast one
  (`/srv/containers/aitools/models`). When swapping the production model, move the old file back to the big disk
  first.

## Quality gates (required for changes that affect output quality)

- `malleus golden` (KL / argmax / top-1 against the reference blobs). Where possible, aim for
  **bitwise-identical** and verify it.
- `malleus spec-exact` for changes to the decode kernels, the verify or the rollback: every row of a speculative
  verify, and every token committed through the speculative loop, must stay **bit-identical** to the serial
  step (it exits non-zero otherwise). Drafts only propose; the text at temperature 0 must not depend on them.
- For performance changes: measure **interleaved A/B** (off/on/off/on), remembering that on this
  machine single-run variance is ~±8% (power-limited clocks). One number before and one after,
  same machine, same thermal window.

## Working with the GPU

- Production is a user unit: `systemctl --user {stop,start,status} malleus.service`.
- **Before** a GPU test, stop production; **after**, restart it and check its health.
- **Before putting Malleus back in production**, update the checkout to the latest `main`, rebuild from
  that revision (`scripts/dev.sh cargo build --release --bin malleus`, then the image the unit runs) and run
  the quality gates. Restart the unit only from that fresh build and check `/health`: never leave an older
  binary (or an older model file) in the running unit.
- The container auto-removes on stop: use `systemctl --user`, not `podman` directly.

## Clean-room

- No references to other engines or products in the engine **code**, competitors included: not in a comment, not
  in a name, not in a constant. Reverse-engineering notes and comparisons live outside this repository, with the
  version and the conditions of what they measure, never in the sources.
- Third-party components we redistribute and MIT/Apache implementations consulted as references are listed in
  [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), with the license and what was used.

## How to reason about a change

- When you propose or close a piece of work, start from the questions: *what is the limit*
  (bandwidth / compute / memory / quality), *how many bytes or operations*, *how do we measure it*.
  A back-of-the-envelope estimate tells you immediately whether an idea is worth 2% or 50%.
