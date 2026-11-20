#!/usr/bin/env python3
"""Focused replay of the compressed indexer of one Hy4-flash indexed layer against transformers.

The indexer of issue #40 decides which `index_kpool`-token groups a query may attend to, and it only runs once the
context outgrows `index_topk` (2048 tokens), so the golden prompts (a dozen tokens) never exercise it. This tool
takes the engine's own inputs for one layer (`T<i>_in_norm.f32` and `T<i>_ix_qresid.f32` from a
`MALLEUS_HY4_TRACE=1` run) and replays the module's stages step by step, then compares them with the engine's dumps
element by element:

  token keys   `T<i>_ix_tk.f32`      the projection and its RMSNorm, per token
  groups       `IX<i>_gk.f32`        the pooling of every closed group, after `key_norm` and its rotary
  query        `T<i>_ix_q.f32`       `query_projection` of the q_a latent, after its rotary
  head weights `T<i>_ix_hw.f32`      the 16 per-head weights, before the reference's `(nh * dim) ** -0.5`
  selection    `IX<i>_list.txt`      the groups each query recalled, expanded to token slots (the meta file has
                                     `len t ncg nsel slots`)

The selection is checked against the module's own `forward` (the mask it writes), which is the point of the tool:
a pool that loses the ordering of a near-tie flips a whole group.

Run in the golden container, on a host with the engine's dumps:

    MALLEUS_GOLDEN_IMAGE=malleus-golden:hy4 scripts/golden.sh scripts/ref_indexer_replay.py \
        --model /models/hy4-flash --layer 0 --blobs /tmp/hy4-ix
"""
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (imported for parity with the other reference tools)
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ref_layerwise import KEEP_FP32, Shards, load_module  # noqa: E402

# The attention module's state-dict keys against the checkpoint's, for a load rooted at `model.layers.<i>.self_attn.`.
ATTN_RENAME = {"sinks": "learnable_sink_param", "gate_proj.weight": "linear_gate.weight"}


def load_f32(path: Path, shape=()) -> torch.Tensor:
    t = torch.from_numpy(np.fromfile(str(path), dtype=np.float32))
    return t.view(shape) if shape else t


def save_f32(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t.detach().float().cpu().contiguous().numpy().tofile(path)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float().cpu(), b.flatten().float().cpu()
    d = (a.norm() * b.norm()).clamp_min(1e-20)
    return float((a * b).sum() / d)


def report(name: str, engine: torch.Tensor, ref: torch.Tensor, rows: int) -> float:
    """Cos per row of a `[rows, width]` stage, printed as a range (the reference is bf16, the engine f32/fp16)."""
    e = engine.float().view(rows, -1).cpu()
    r = ref.float().view(rows, -1).cpu()
    cs = [cos(e[i], r[i]) for i in range(rows)]
    lo, mean = min(cs), sum(cs) / len(cs)
    rel = float((e - r).norm() / r.norm().clamp_min(1e-20))
    print(f"{name:12s} {rows:6d} rows  cos min {lo:.6f} mean {mean:.6f}  rel {rel:.4f}", flush=True)
    return lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--blobs", required=True, help="MALLEUS_DUMP_DIR of a MALLEUS_HY4_TRACE=1 engine run")
    ap.add_argument("--out", default=None, help="write the reference stages here (default: the blobs dir)")
    ap.add_argument("--cos-min", type=float, default=0.99, help="fail below this per-row cosine")
    ap.add_argument(
        "--weight-noise",
        type=float,
        default=0.0,
        help="perturb the three indexer projections to this relative L2 (the file's MLS4 error is ~0.10) and "
        "report how many selections flip: how much of the disagreement is the 4-bit weights, not the kernel",
    )
    ap.add_argument(
        "--attention-out",
        type=int,
        default=0,
        help="also run the module's own attention with the reference mask and compare the last N rows of "
        "`T<i>_attn.f32` (the layer's output after o_proj and post_proj_norm)",
    )
    args = ap.parse_args()

    import transformers
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as H

    dev = "cuda"
    blob = Path(args.blobs)
    out = Path(args.out) if args.out else blob
    li = args.layer
    cfg = AutoConfig.from_pretrained(args.model).text_config
    cfg._attn_implementation = "eager"
    torch.set_default_dtype(torch.bfloat16)

    shards = Shards(Path(args.model), dev)
    with torch.device("meta"):
        attn = H.HYV4FlashAttention(cfg, li)
    attn = attn.to_empty(device=dev)
    for name, p in attn.named_parameters():
        if any(k in name for k in KEEP_FP32):
            p.data = p.data.float()
    load_module(attn, shards, f"model.layers.{li}.self_attn.", (), ATTN_RENAME)
    attn.eval()
    idx = attn.compressed_indexer
    if idx is None:
        print(f"layer {li} is {attn.attention_type}: no compressed indexer", flush=True)
        return 2

    hn = load_f32(blob / f"T{li}_in_norm.f32").view(1, -1, cfg.hidden_size).to(dev).to(torch.bfloat16)
    qr = load_f32(blob / f"T{li}_ix_qresid.f32").view(1, -1, cfg.q_lora_rank).to(dev).to(torch.bfloat16)
    T = hn.shape[1]
    ratio, nh, dim = idx.index_kpool, idx.n_heads, idx.head_dim
    rope = idx.qk_rope_head_dim
    nope = dim - rope
    ncg = T // ratio
    print(f"T {T}, groups {ncg}, budget {idx.index_topk}, {nh} heads of {dim}", flush=True)

    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import apply_rotary_pos_emb

    # `to_empty` leaves the module's non-persistent buffers (the `*_inv_freq` tables) uninitialised, and the
    # indexer's own rotary never ran in the layer-by-layer golden (its prompts are below the budget), so it is built
    # fresh here instead of inherited from the uninitialised module.
    idx.rotary_emb = H.HYV4FlashRotaryEmbedding(cfg).to(device=dev)

    pos = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)

    def rot_q(x, p):
        """`[1, T, nh, dim]`, the trailing `rope` dims rotated (the module's own split and unsqueeze)."""
        a, b = torch.split(x, [nope, rope], dim=-1)
        c, s = idx.rotary_emb(x, p, layer_type="indexed_attention")
        b, _ = apply_rotary_pos_emb(b, b, c, s, unsqueeze_dim=2)
        return torch.cat((a, b), dim=-1)

    def rot_k(x, p):
        """`[1, groups, dim]`, rotated the way `forward` rotates the pooled keys."""
        a, b = torch.split(x.unsqueeze(2), [nope, rope], dim=-1)
        c, s = idx.rotary_emb(x, p, layer_type="indexed_attention")
        b, _ = apply_rotary_pos_emb(b, b, c, s, unsqueeze_dim=2)
        return torch.cat((a, b), dim=-1).squeeze(2)

    if args.weight_noise:
        # The indexer's three projections are MLS4 in the file (rel-L2 0.104 on this checkpoint). Perturbing the
        # reference's bf16 weights by the same amount says how much of the selection disagreement that costs.
        gen = torch.Generator(device="cpu").manual_seed(0)
        for m in (idx.query_projection, idx.token_key_projection, idx.head_weight_projection):
            w = m.weight.detach()
            n = torch.randn(w.shape, generator=gen).to(w.device)
            n *= w.norm() / n.norm() * args.weight_noise
            w.add_(n.to(w.dtype))

    fails = 0
    with torch.no_grad():
        token_keys = idx.token_key_norm(idx.token_key_projection(hn))
        group_keys = rot_k(idx.pool_groups(token_keys), pos[..., : ncg * ratio : ratio])  # bf16, as the module
        query = rot_q(idx.query_projection(qr).view(1, T, nh, dim), pos)  # [1, T, nh, dim]
        hw = idx.head_weight_projection(hn).float() * (nh**-0.5 * dim**-0.5)

        scores = torch.relu(torch.matmul(query.float(), group_keys.float().unsqueeze(1).transpose(-1, -2)))
        logits = torch.matmul(hw.unsqueeze(-2), scores).squeeze(-2)
        own = (torch.arange(T, device=dev) // ratio).unsqueeze(0)  # [1, T]
        group_ids = torch.arange(ncg, device=dev)
        logits = logits.masked_fill(group_ids >= own.unsqueeze(-1), float("-inf"))
        complete_topk = min(idx.group_topk - 1, ncg)
        recalled = torch.cat(
            (logits.topk(complete_topk, dim=-1).indices, own.unsqueeze(-1)), dim=-1
        )  # [1, T, complete_topk + 1]
        print(f"recalled {complete_topk + 1} of {ncg} groups per query", flush=True)

        save_f32(out / f"REF{li}_tk.f32", token_keys[0])
        save_f32(out / f"REF{li}_gk.f32", group_keys[0])
        save_f32(out / f"REF{li}_q.f32", query[0])
        save_f32(out / f"REF{li}_hw.f32", hw[0])
        save_f32(out / f"REF{li}_logits.f32", logits[0])

        # The allowed token slots of every query, from the reference's own selection (a set: order is free).
        slots = (recalled.unsqueeze(-1) * ratio + torch.arange(ratio, device=dev)).clamp(max=T - 1)
        allowed = torch.zeros((T, T), dtype=torch.bool, device=dev)
        allowed.scatter_(-1, slots[0].flatten(-2), True)
        allowed = torch.tril(allowed)  # the causal mask the module applies on top
        save_f32(out / f"REF{li}_allowed.f32", allowed.to(torch.float32))

        # The module's own mask: what `HYV4FlashCompressedIndexer.forward` returns for a full prompt (no cache).
        base = create_causal_mask(
            config=cfg,
            inputs_embeds=hn,
            attention_mask=None,
            cache=None,
            position_ids=pos[0],
            allow_is_causal_skip=False,
        )
        mask = idx(hn, qr, base, pos, pos, None, T)
        ref_allowed = mask[0, 0] > (torch.finfo(mask.dtype).min / 2)
        same = (ref_allowed == allowed).all().item()
        print(f"reference mask agrees with the replay: {same}", flush=True)
        if not same:
            bad = (ref_allowed != allowed).any(-1).nonzero().flatten()[:8].tolist()
            print(f"  queries that disagree: {bad}", flush=True)

        if args.attention_out:
            # The layer's attention on the reference's own selection: the indexer's effect end to end, which is the
            # number the flips at a tie boundary have to be judged by.
            pe = H.HYV4FlashRotaryEmbedding(cfg).to(dev)(hn, pos, layer_type="indexed_attention")
            ref_attn, _ = attn(
                hidden_states=hn,
                position_embeddings=pe,
                attention_mask=base,
                position_ids=pos[0],
                indexer_position_ids=pos,
                cache=None,
            )
            save_f32(out / f"REF{li}_attn.f32", ref_attn[0])
            n = min(args.attention_out, T)
            p = blob / f"T{li}_attn.f32"
            e = load_f32(p).view(T, -1)[T - n :]
            r = ref_attn[0].float().cpu()[T - n :]
            if report(f"attention (last {n})", e, r, n) < args.cos_min:
                fails += 1

    # The engine's stages against the reference's.
    stages = [
        ("token keys", f"T{li}_ix_tk.f32", token_keys[0], T),
        ("groups", f"IX{li}_gk.f32", group_keys[0], ncg),
        ("query", f"T{li}_ix_q.f32", query[0], T),
        ("head w", f"T{li}_ix_hw.f32", hw[0], T),
    ]
    for name, fn, ref, rows in stages:
        p = blob / fn
        if not p.exists():
            print(f"{name:12s} missing {p}", flush=True)
            fails += 1
            continue
        e = load_f32(p).view(rows, -1)
        if name == "head w":
            e = e * (nh**-0.5 * dim**-0.5)
        if report(name, e, ref, rows) < args.cos_min:
            fails += 1

    keys_bin = blob / f"IX{li}_keys.bin"
    if keys_bin.exists():
        raw = np.fromfile(str(keys_bin), dtype=np.uint32)
        u = torch.from_numpy(raw.astype(np.int64) & 0xFFFFFFFF).view(-1, ncg)
        # Undo the order-preserving transform the score kernel writes: a positive float is its bits with the top
        # bit set, a negative one is their complement.
        bits = torch.where(u >= 0x80000000, u ^ 0x80000000, (~u) & 0xFFFFFFFF).to(torch.int64)
        eng_logits = bits.to(torch.uint32).view(torch.float32) if hasattr(torch, "uint32") else bits.numpy().astype(np.uint32).view(np.float32).copy()
        cand = torch.arange(ncg).unsqueeze(0) < own[0].cpu().unsqueeze(-1)
        eng_logits = torch.where(cand, eng_logits, torch.full_like(eng_logits, float("-inf")))
        ref_logits = logits[0].cpu()
        finite = ref_logits.isfinite()
        rows_n = min(T, eng_logits.shape[0])
        cs = {i: cos(eng_logits[i][finite[i]], ref_logits[i][finite[i]]) for i in range(rows_n) if finite[i].any()}
        rel = float((eng_logits[finite] - ref_logits[finite]).norm() / ref_logits[finite].norm().clamp_min(1e-20))
        worst = sorted(cs, key=lambda i: cs[i])[:3]
        print(f"logits       {len(cs):6d} rows  cos min {min(cs.values()):.6f} mean {sum(cs.values())/len(cs):.6f}", end="")
        print(f"  rel {rel:.4f}  worst rows {[(i, round(cs[i], 5)) for i in worst]}", flush=True)
        save_f32(out / f"REF{li}_logits_masked.f32", ref_logits)

    meta = blob / f"IX{li}_list_meta.txt"
    lst = blob / f"IX{li}_list.txt"
    if not meta.exists() or not lst.exists():
        print(f"missing {meta} or {lst}", flush=True)
        return 1
    n_len, t, n_groups, nsel, n_slots = (int(v) for v in meta.read_text().split())
    rows = lst.read_text().split("\n")
    mismatched, extra, missing = 0, 0, 0
    for ti in range(t):
        ids = [int(v) for v in rows[ti].split()][:n_slots]
        nkeys = n_len + ti + 1
        got = set()
        for g in ids:
            if g < 0:
                continue
            for p in range(ratio):
                s = g * ratio + p
                if s < nkeys:
                    got.add(s)
        want = set(torch.nonzero(allowed[ti], as_tuple=False).flatten().tolist())
        if got != want:
            mismatched += 1
            extra += len(got - want)
            missing += len(want - got)
            if mismatched <= 4:
                # The groups that were swapped, and the reference's own gap at the selection boundary: a swap on two
                # scores that are equal (or within the bf16 noise of the reference) is a tie, not a disagreement.
                swap = sorted({s // ratio for s in (got ^ want)})
                vals = logits[0, ti]
                finite = vals.isfinite()
                top = torch.topk(vals[finite], min(nsel, int(finite.sum()))).values
                gap = float(top[-2] - top[-1]) if top.numel() > 1 else float("nan")
                print(
                    f"  query {ti}: +{len(got - want)} -{len(want - got)} slots, {len(swap)} group(s) {swap}, "
                    f"reference scores {[round(float(vals[g]), 6) for g in swap[:6]]}, "
                    f"boundary gap {gap:.6f}",
                    flush=True,
                )
    print(
        f"selection     {t:6d} rows  groups {nsel} of {n_groups} (slots {n_slots})  "
        f"{mismatched} rows differ, +{extra} -{missing} slots",
        flush=True,
    )
    if mismatched:
        fails += 1
    print("transformers", transformers.__version__, "torch", torch.__version__, flush=True)
    print("OK" if fails == 0 else f"{fails} stage(s) out of tolerance", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
