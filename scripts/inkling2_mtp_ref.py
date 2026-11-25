#!/usr/bin/env python3
"""Reference replay of Inkling-2's MTP head, stage by stage, against the engine's own dumps.

The draft head has no fixture set: the model family's reference implementation does not carry it, so the only thing a
change can be checked against is the acceptance, and the acceptance says nothing about *where* a difference comes
from. This script closes that hole: the engine (with `MALLEUS_MTP_DUMP=<dir>`) writes the four target states of the
anchor row, its draft cache, the block's input, the hidden state after every layer, the normalized output and the
Markov walk's first latent; this script recomputes each of those from the checkpoint's `mtp.safetensors` in float32
and prints the difference per stage.

Usage:
    scripts/golden.sh scripts/inkling2_mtp_ref.py --mtp /models/thinkingmachines/Inkling-2-Mini/mtp.safetensors \
        --dump <dir>
"""

import argparse
import json
import math
import os

import numpy as np
import torch


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, one weight for every row (the head's norms are per head or per token)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def mix_rmsnorm(targets: torch.Tensor, alpha: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """The fuse: `rmsnorm(sum_tap alpha[tap] * targets[tap]) * w` — one row per token in, one row out."""
    mixed = (targets.to(torch.float32) * alpha[:, None]).sum(0)
    return mixed * torch.rsqrt(mixed.pow(2).mean() + eps) * w


def rope(x: torch.Tensor, pos: int, theta: float) -> torch.Tensor:
    """NeoX rope over the last dim, in place semantics: the pair is (f, f + dim/2)."""
    dim = x.shape[-1]
    half = dim // 2
    f = torch.arange(half, dtype=torch.float32)
    ang = pos / torch.pow(torch.tensor(float(theta)), 2 * f / dim)
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1)


def load(path: str) -> dict:
    from safetensors.torch import load_file

    return {k: v.to(torch.float32) for k, v in load_file(path).items()}


def rel(got: np.ndarray, want: np.ndarray) -> str:
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scale = np.abs(want.astype(np.float64)).max() + 1e-30
    return f"max|d| {d.max():.3e}  rel {d.max() / scale:.3e}  |ref| {scale:.3f}"


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--mtp", required=True)
    a.add_argument("--dump", required=True)
    args = a.parse_args()

    meta = json.load(open(os.path.join(args.dump, "mtp.json")))
    h, heads, nkv, hd, inter = meta["h"], meta["heads"], meta["kv_heads"], meta["head_dim"], meta["inter"]
    rows, pos, eps, theta = meta["rows"], meta["pos"], meta["eps"], meta["theta"]
    kvd = nkv * hd
    local = (0, 2)
    w = load(args.mtp)
    fuse = torch.softmax(w["fuse_weight"].to(torch.float32), dim=-1)

    def f32(name, shape=None):
        v = np.fromfile(os.path.join(args.dump, name), dtype=np.float32)
        return torch.from_numpy(v if shape is None else v.reshape(shape))

    def f16(name, shape):
        v = np.fromfile(os.path.join(args.dump, name), dtype=np.float16)
        return torch.from_numpy(v.reshape(shape).astype(np.float32))

    feats = f32("anchor_feats.f32", (4, h))
    base = pos + 1  # the context holds one row per committed token, the anchor last

    def layer_ctx(s):
        """The context K/V of one draft layer: the anchor's mixture through `k_proj_ctx`/`v_proj_ctx`, the key
        through the per-head `k_norm_ctx` and the rope."""
        m = mix_rmsnorm(feats, fuse[s], w[f"layers.{s}.dflare_norm.weight"], eps)
        k = (m @ w[f"layers.{s}.self_attn.k_proj_ctx.weight"].T).reshape(nkv, hd)
        v = m @ w[f"layers.{s}.self_attn.v_proj_ctx.weight"].T
        k = rms(k, w[f"layers.{s}.self_attn.k_norm_ctx.weight"], eps)
        return torch.cat([rope(k[i], pos, theta) for i in range(nkv)]), v

    stages = []
    # The context rows the cache holds are the ones the draft actually attends to: the dump has them, so the layers
    # are compared with the same context on both sides (the anchor's own row is checked separately, against the
    # commit-time dumps, where the mix and the GEMM already agree to 2e-7).
    for s in range(3):
        if os.path.exists(os.path.join(args.dump, "commit_mix0.f32")):
            k_ref, v_ref = layer_ctx(s)
            k_got = f16(f"ctx_cache_k{s}.f16", (base, kvd))[-1]
            v_got = f16(f"ctx_cache_v{s}.f16", (base, kvd))[-1]
            stages.append((f"ctx k, anchor row, layer {s}", k_got.numpy(), k_ref.numpy()))
            stages.append((f"ctx v, anchor row, layer {s}", v_got.numpy(), v_ref.numpy()))

    # The block's input row by row: the engine dumps it at the top of the first layer (`block_in` is written
    # after the loop and holds the final hidden, so it is not the input).
    x = f32("layer_in0.f32", (rows, h))
    for s in range(3):
        p = f"layers.{s}."
        xn = rms(x, w[p + "input_layernorm.weight"], eps)
        q = xn @ w[p + "self_attn.q_proj.weight"].T
        k = xn @ w[p + "self_attn.k_proj.weight"].T
        v = xn @ w[p + "self_attn.v_proj.weight"].T
        q = rms(q.reshape(-1, hd), w[p + "self_attn.q_norm.weight"], eps).reshape(rows, heads, hd)
        k = rms(k.reshape(-1, hd), w[p + "self_attn.k_norm.weight"], eps).reshape(rows, nkv, hd)
        v = v.reshape(rows, nkv, hd)
        for r in range(rows):
            q[r] = rope(q[r], pos + r, theta)
            k[r] = rope(k[r], pos + r, theta)
        ck = f16(f"ctx_cache_k{s}.f16", (base, kvd)).reshape(base, nkv, hd)
        cv = f16(f"ctx_cache_v{s}.f16", (base, kvd)).reshape(base, nkv, hd)
        k_all = torch.cat([ck, k], 0)
        v_all = torch.cat([cv, v], 0)
        out = torch.zeros(rows, heads, hd)
        gqa = heads // nkv
        for r in range(rows):
            nk = base + r + 1
            i0 = max(0, nk - meta["window"]) if s in local else 0
            # Query head `h` reads KV head `h // gqa`: [n, nkv, hd] -> [n, heads, hd].
            kk = k_all[i0:nk].repeat_interleave(gqa, dim=1)
            vv = v_all[i0:nk].repeat_interleave(gqa, dim=1)
            sc = torch.einsum("hd,nhd->hn", q[r], kk) / math.sqrt(hd)
            prob = torch.softmax(sc, -1)
            out[r] = torch.einsum("hn,nhd->hd", prob, vv)
        o = out.reshape(rows, heads * hd) @ w[p + "self_attn.o_proj.weight"].T
        x = x + o
        xn = rms(x, w[p + "post_attention_layernorm.weight"], eps)
        gate = xn @ w[p + "mlp.gate_proj.weight"].T
        up = xn @ w[p + "mlp.up_proj.weight"].T
        act = torch.nn.functional.silu(gate) * up
        x = x + act @ w[p + "mlp.down_proj.weight"].T
        stages.append((f"block hidden, layer {s}", f32(f"block_h{s}.f32", (rows, h)).numpy(), x.numpy()))

    normed = rms(x, w["norm.weight"], eps)
    stages.append(("normed", f32("normed.f32", (rows, h)).numpy(), normed.numpy()))

    # The Markov walk of the first proposal: state 0, the anchor's own mixture as `f_rnn`.
    f_rnn = mix_rmsnorm(feats, fuse[3], w["markov_head.dflare_norm.weight"], eps)
    prev = meta["anchor"]
    proj_in = torch.cat(
        [torch.zeros(256), w["markov_head.markov_w1.weight"][prev], normed[1], f_rnn]
    )
    proj = w["markov_head.rnn_proj.weight"] @ proj_in + w["markov_head.rnn_proj.bias"]
    rank = meta["rank"]
    latent = torch.tanh(proj[2 * rank : 3 * rank])
    if os.path.exists(os.path.join(args.dump, "latent0.f32")):
        stages.append(("markov latent, row 0", f32("latent0.f32").numpy(), latent.numpy()))

    bad = 0
    for name, got, want in stages:
        line = rel(got, want)
        scale = np.abs(want).max() + 1e-30
        worst = np.abs(got.astype(np.float64) - want.astype(np.float64)).max() / scale
        bad += worst > 2e-3
        print(f"{name:26s} {line}")
    print(f"\n{len(stages) - bad}/{len(stages)} stages agree to 2e-3 (relative)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
