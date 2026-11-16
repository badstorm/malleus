#!/usr/bin/env python3
"""Layer-by-layer reference for Ling-3.0-flash (`bailing_hybrid` / `BailingMoeV3`).

238 GiB of bf16 weights fit neither in the 32 GB the container sees nor in GTT, so the reference streams one layer
at a time from the safetensors shards and pushes every sample through it before moving on (MIGRATE_MODELS.md §4.3).
Pure PyTorch: no `trust_remote_code`, no Triton and no `fla` at run time.

The KDA block transcribes `naive_recurrent_kda` of fla-core 0.6.0 (MIT); `selftest` checks it against the original,
`hfcheck` checks the whole assembly against the checkpoint's own remote code on a reduced random model.

Subcommands
    selftest   KDA, short conv and gated norm against fla-core on random tensors
    hfcheck    the whole reference against `modeling_bailing_moe_v3.py` on a small random model
    layers     per-layer hidden states of one token list, to validate the Rust engine layer by layer
    golden     prompt set: token ids, per-position argmax, last-token logits, greedy continuation
    eval       per-position reference statistics on texts disjoint from the calibration corpus
    clamp      perplexity with and without the SwiGLU clamp (docs/adr/0004-ling3-swiglu-clamp.md)
"""
import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def build(args) -> "Ling":
    return Ling(args.model, device=args.device, dtype=DTYPE[args.dtype])

# --------------------------------------------------------------------------------------------- configuration


@dataclass
class Config:
    """The config.json fields the forward pass reads; the rest is Bailing V2 legacy (docs/model-ling3-flash.md)."""

    hidden: int
    layers: int
    mtp: int
    vocab: int
    group_size: int
    dense_first: int
    eps: float
    rope_theta: float
    kda_heads: int
    kda_dim: int
    conv_k: int
    safe_gate: bool
    lower_bound: float
    heads: int
    qk_nope: int
    qk_rope: int
    qk_dim: int
    v_dim: int
    kv_lora: int
    rope_interleave: bool
    experts: int
    top_k: int
    inter: int
    shared_inter: int
    shared_experts: int
    n_group: int
    topk_group: int
    norm_topk: bool
    routed_scaling: float
    expert_bias: bool
    dense_inter: int
    expert_limit: list = field(default_factory=list)
    shared_limit: list = field(default_factory=list)
    eos: int = 0

    @staticmethod
    def load(path: Path) -> "Config":
        c = json.loads((Path(path) / "config.json").read_text())
        return Config(
            hidden=c["hidden_size"],
            layers=c["num_hidden_layers"],
            mtp=c.get("num_nextn_predict_layers", 0),
            vocab=c["vocab_size"],
            group_size=c["layer_group_size"],
            dense_first=c.get("first_k_dense_replace", 0),
            eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            kda_heads=c["num_attention_heads"],
            kda_dim=c["head_dim"],
            conv_k=c["short_conv_kernel_size"],
            safe_gate=c.get("kda_safe_gate", True),
            lower_bound=c.get("kda_lower_bound", -5.0),
            heads=c["num_attention_heads"],
            qk_nope=c["qk_nope_head_dim"],
            qk_rope=c["qk_rope_head_dim"],
            qk_dim=c["qk_head_dim"],
            v_dim=c["v_head_dim"],
            kv_lora=c["kv_lora_rank"],
            rope_interleave=c.get("rope_interleave", True),
            experts=c["num_experts"],
            top_k=c["num_experts_per_tok"],
            inter=c["moe_intermediate_size"],
            shared_inter=c["moe_shared_expert_intermediate_size"],
            shared_experts=c.get("num_shared_experts", 1),
            n_group=c["n_group"],
            topk_group=c["topk_group"],
            norm_topk=c["norm_topk_prob"],
            routed_scaling=c["routed_scaling_factor"],
            expert_bias=c.get("moe_router_enable_expert_bias", False),
            dense_inter=c["intermediate_size"],
            expert_limit=c.get("expert_swiglu_limit_list") or [],
            shared_limit=c.get("share_expert_swiglu_limit_list") or [],
            eos=c.get("eos_token_id", 0),
        )

    def is_mla(self, layer: int) -> bool:
        """The layer order of `BailingMoeV3DecoderLayer.__init__`: MLA every `layer_group_size` tokens, plus the
        tail layers of a partial group. For Ling-3.0-flash (42 layers, group 6) the second clause never fires and
        the MLA layers are 5, 11, 17, 23, 29, 35, 41; the MTP layer is always MLA (`mtp_use_kda` false)."""
        return (layer + 1) % self.group_size == 0 or layer >= self.layers // self.group_size * self.group_size

    def is_moe(self, layer: int) -> bool:
        return layer >= self.dense_first

    def limit(self, lst: list, layer: int) -> float:
        """0 and out-of-range (the MTP layer) mean no clamp, as upstream `_get_swiglu_limit` does."""
        return float(lst[layer]) if 0 <= layer < len(lst) and lst[layer] else 0.0


# --------------------------------------------------------------------------------------------- weights


class Shards:
    """Reads tensors straight to the device; the shard handles stay open, the tensors do not."""

    def __init__(self, model_dir: Path, device: str):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            from safetensors import safe_open

            f = self.dir / "model.safetensors"
            with safe_open(str(f), framework="pt", device="cpu") as h:
                self.map = {k: f.name for k in h.keys()}
        self.device = device
        self.handles: dict = {}
        self.fds: dict = {}
        self.layout: dict = {}
        self._names = sorted(self.map)

    def names(self) -> list:
        return self._names

    def prefixed(self, prefix: str) -> list:
        return [n for n in self._names if n.startswith(prefix)]

    def _data_layout(self, shard: str):
        """First byte of the data and `tensor -> (start, end)`, from the safetensors header of the shard."""
        if shard not in self.layout:
            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n))
            self.layout[shard] = (8 + n, {k: tuple(v["data_offsets"]) for k, v in head.items() if k != "__metadata__"})
        return self.layout[shard]

    def _drop(self, shard: str, name: str) -> None:
        """Drop the pages of a tensor that is already on the device.

        The reference streams 238 GiB per pass: without this the page cache fills, and then every tensor read
        goes through direct reclaim (measured: 44 MB/s against 280 MB/s). The pages are clean and the DMA pins
        them while it runs, so dropping them here is safe.
        """
        start, offsets = self._data_layout(shard)
        lo, hi = offsets[name]
        if hi - lo < (1 << 20):
            return
        fd = self.fds.get(shard)
        if fd is None:
            fd = self.fds[shard] = os.open(self.dir / shard, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        os.posix_fadvise(fd, start + lo, hi - lo, os.POSIX_FADV_DONTNEED)

    def get(self, name: str, dtype=None) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.map[name]
        h = self.handles.get(shard)
        if h is None:
            h = self.handles[shard] = safe_open(str(self.dir / shard), framework="pt", device=self.device)
        t = h.get_tensor(name)
        self._drop(shard, name)
        return t if dtype is None else t.to(dtype)


class Layer:
    """The tensors of one layer, keyed by their name without the `model.layers.N.` prefix.

    `sorted(self._names)` is lexicographic, which is the order the shards store the tensors in (`layers.10`
    before `layers.2`): reading a layer means reading one contiguous run of the shard, not 1 536 seeks.
    """

    def __init__(self, st: Shards, prefix: str, dtype):
        self.t = {n[len(prefix) :]: st.get(n, dtype) for n in st.prefixed(prefix)}

    def w(self, name: str) -> torch.Tensor:
        return self.t[name]

    def f32(self, name: str) -> torch.Tensor:
        return self.t[name].float()


# --------------------------------------------------------------------------------------------- primitives


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """`BailingMoeV3RMSNorm`: fp32 variance, weight applied after the cast back to the input dtype."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(x.dtype)


def gated_rmsnorm(x: torch.Tensor, g: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """`FusedRMSNormGated(activation="sigmoid")` over the head dim: norm, weight, then the gate.

    fla-core takes the variance of `x` alone (gate applied after the norm), which is what its Triton kernel does.
    """
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * w * torch.sigmoid(g.float()).to(x.dtype)


def short_conv(u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Causal depthwise conv1d + silu (`ShortConvolution(kernel_size, activation='silu')`).

    u [..., T, C], weight [C, 1, K] -> [..., T, C], with y[t] = silu(sum_j w[c, j] * u[t - K + 1 + j]).
    """
    k = weight.shape[2]
    t = u.shape[-2]
    uu = torch.cat([u.new_zeros(*u.shape[:-2], k - 1, u.shape[-1]), u], dim=-2)
    y = sum(uu[..., j : j + t, :] * weight[:, 0, j] for j in range(k))
    return F.silu(y)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`fla.modules.l2norm_fwd`: x / sqrt(sum(x^2) + eps) over the last dim, computed in fp32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """`silu(gate) * up`, clamped the way vLLM and SGLang do when the layer has a non-zero limit (ADR 0004)."""
    if limit > 0.0:
        return F.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def rope_interleaved(q_rot: torch.Tensor, k_rot: torch.Tensor, inv_freq: torch.Tensor, pos: torch.Tensor):
    """`apply_rotary_pos_emb_interleave`: de-interleave the pairs, then the usual rotate_half.

    q_rot [B, T, H, D], k_rot [B, T, 1, D] -> same shapes. The same permutation is applied to both, so the scores
    do not change; we keep the upstream layout so the dumps stay comparable.
    """
    d = q_rot.shape[-1]
    ang = pos.float().to(q_rot.device)[..., None] * inv_freq.to(q_rot.device).float()
    emb = torch.cat([ang, ang], dim=-1)
    cos, sin = emb.cos().unsqueeze(-2), emb.sin().unsqueeze(-2)  # [B, T, 1, D]

    def de(x):
        x = x.float().view(*x.shape[:-1], d // 2, 2)
        return torch.cat([x[..., 0], x[..., 1]], dim=-1)

    def rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    q, k = de(q_rot), de(k_rot)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


# --------------------------------------------------------------------------------------------- KDA


def kda_recurrent(q, k, v, g, beta):
    """Kimi Delta Attention, one token at a time: `fla.ops.kda.naive.naive_recurrent_kda` transcribed.

    q [B,T,H,K] already l2-normed and scaled, k [B,T,H,K], v [B,T,H,V], g [B,T,H,K] the log-decay per key
    channel, beta [B,T,H] -> o [B,T,H,V] in fp32. The state is S[b,h] = [K,V]; the decay is a diagonal over K,
    not a scalar per head as in the Gated DeltaNet of Qwen3.5.
    """
    b, t, h, kd = q.shape
    vd = v.shape[-1]
    s = torch.zeros(b, h, kd, vd, dtype=torch.float32, device=q.device)
    out = torch.empty(b, t, h, vd, dtype=torch.float32, device=q.device)
    for i in range(t):
        s = s * g[:, i].float().exp()[..., None]
        ki, vi, bi = k[:, i].float(), v[:, i].float(), beta[:, i].float()
        pred = torch.einsum("bhkv,bhk->bhv", s, ki)
        s = s + bi[..., None, None] * torch.einsum("bhk,bhv->bhkv", ki, vi - pred)
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q[:, i].float(), s)
    return out


def kda(x: torch.Tensor, ly: Layer, cfg: Config) -> torch.Tensor:
    """`BailingMoeV3KimiDeltaAttention.forward` for a batch of equal-length sequences."""
    b, t, _ = x.shape
    h, kd = cfg.kda_heads, cfg.kda_dim
    q = short_conv(x @ ly.w("attention.q_proj.weight").T, ly.w("attention.q_conv1d.weight"))
    k = short_conv(x @ ly.w("attention.k_proj.weight").T, ly.w("attention.k_conv1d.weight"))
    v = short_conv(x @ ly.w("attention.v_proj.weight").T, ly.w("attention.v_conv1d.weight"))
    q = l2norm(q.view(b, t, h, kd)).float() * (kd**-0.5)
    k = l2norm(k.view(b, t, h, kd))
    v = v.view(b, t, h, kd)

    a_log = ly.f32("attention.A_log").view(1, 1, h, 1)
    dt_bias = ly.f32("attention.dt_bias").view(1, 1, h, kd)
    raw = (x @ ly.w("attention.f_proj.weight").T).view(b, t, h, kd).float()
    g = (
        cfg.lower_bound * torch.sigmoid(a_log.exp() * (raw + dt_bias))
        if cfg.safe_gate
        else -a_log.exp() * F.softplus(raw + dt_bias)
    )
    beta = torch.sigmoid((x @ ly.w("attention.b_proj.weight").T).view(b, t, h).float())

    o = kda_recurrent(q, k, v, g, beta).to(x.dtype).view(b, t, h, kd)
    zg = (x @ ly.w("attention.g_proj.weight").T).view(b, t, h, kd)
    o = gated_rmsnorm(o, zg, ly.w("attention.o_norm.weight"), cfg.eps)
    return o.reshape(b, t, h * kd) @ ly.w("attention.o_proj.weight").T


# --------------------------------------------------------------------------------------------- MLA


def mla(x: torch.Tensor, ly: Layer, cfg: Config, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """`BailingMoeV3MultiLatentAttention.forward`, not absorbed; the scores are computed one sample at a time
    because the matrix is [H, T, T]."""
    b, t, _ = x.shape
    h, nope, rope = cfg.heads, cfg.qk_nope, cfg.qk_rope
    q = (x @ ly.w("attention.q_proj.weight").T).view(b, t, h, cfg.qk_dim)
    q_pass, q_rot = q[..., :nope], q[..., nope:]
    ckv = x @ ly.w("attention.kv_a_proj_with_mqa.weight").T
    latent, k_rot = ckv[..., : cfg.kv_lora], ckv[..., cfg.kv_lora :]
    cn = rmsnorm(latent, ly.w("attention.kv_a_layernorm.weight"), cfg.eps)
    kv = (cn @ ly.w("attention.kv_b_proj.weight").T).view(b, t, h, nope + cfg.v_dim)
    k_pass, v = kv[..., :nope], kv[..., nope:]

    q_rot, k_rot = rope_interleaved(q_rot, k_rot.unsqueeze(2), inv_freq, pos)
    qq = torch.cat([q_pass.float(), q_rot.float()], dim=-1)  # [B, T, H, qk_dim]
    kk = torch.cat([k_pass.float(), k_rot.float().expand(b, t, h, rope)], dim=-1)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
    scale = cfg.qk_dim**-0.5
    outs = []
    for i in range(b):
        s = qq[i].transpose(0, 1) @ kk[i].transpose(0, 1).transpose(-1, -2)  # [H, T, T]
        p = torch.softmax(s.masked_fill(causal, float("-inf")) * scale, dim=-1)
        outs.append((p @ v[i].float().transpose(0, 1)).transpose(0, 1))  # [T, H, V]
    o = torch.stack(outs).to(x.dtype)
    gate = torch.sigmoid((x @ ly.w("attention.g_proj.weight").T).float()).to(x.dtype)
    return (o * gate[..., None]).reshape(b, t, h * cfg.v_dim) @ ly.w("attention.dense.weight").T


# --------------------------------------------------------------------------------------------- FFN and MoE


def mlp(x: torch.Tensor, ly: Layer, prefix: str, limit: float) -> torch.Tensor:
    g = x @ ly.w(f"{prefix}gate_proj.weight").T
    u = x @ ly.w(f"{prefix}up_proj.weight").T
    return swiglu(g, u, limit) @ ly.w(f"{prefix}down_proj.weight").T


def group_topk(logits: torch.Tensor, bias: torch.Tensor, cfg: Config):
    """`noaux_tc`: sigmoid scores, top-4 of 8 groups by the top-2 sum, top-8 experts, normalized weights × 2.5.

    The bias shifts the scores used for *both* the group choice and the final top-k; the weights are then gathered
    from the unbiased scores (`BailingMoeV3Gate.forward` passes `scores_for_routing` into `group_limited_topk` and
    gathers from `scores`). Ranking the unbiased scores instead would pick a different set of experts.
    """
    t = logits.shape[0]
    scores = torch.sigmoid(logits.float())
    routed = scores + bias if bias is not None else scores
    per = cfg.experts // cfg.n_group
    gs = routed.view(t, cfg.n_group, per).topk(2, dim=-1).values.sum(-1)
    keep = torch.zeros(t, cfg.n_group, dtype=torch.bool, device=logits.device)
    keep.scatter_(1, gs.topk(cfg.topk_group, dim=-1).indices, True)
    masked = routed.masked_fill(~keep[:, :, None].expand(t, cfg.n_group, per).reshape(t, cfg.experts), float("-inf"))
    idx = masked.topk(cfg.top_k, dim=-1).indices
    w = scores.gather(1, idx)
    if cfg.norm_topk:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * cfg.routed_scaling


# Set by `route`: layer -> (idx, w) of every position, on the CPU. `moe` fills it, `cmd_route` writes the trace in the
# format `MALLEUS_TRACE_ROUTE` writes, so that scripts/expert_cache.py reads the reference and the engine alike.
ROUTE_LOG = None


def moe(x: torch.Tensor, ly: Layer, cfg: Config, layer: int, clamp: bool) -> torch.Tensor:
    """Routed experts plus the shared one, which has no gate and does not enter the weight normalization."""
    b, t, hd = x.shape
    flat = x.reshape(b * t, hd)
    bias = ly.f32("mlp.gate.expert_bias") if cfg.expert_bias else None
    idx, w = group_topk(flat @ ly.w("mlp.gate.weight").T.float(), bias, cfg)
    if ROUTE_LOG is not None:
        ROUTE_LOG[layer] = (idx.detach().cpu(), w.detach().cpu())
    out = torch.zeros(b * t, hd, dtype=torch.float32, device=x.device)

    limit = cfg.limit(cfg.expert_limit, layer) if clamp else 0.0
    order = idx.reshape(-1).argsort(stable=True)
    tid = torch.arange(b * t, device=x.device).repeat_interleave(cfg.top_k)[order]
    wid = w.reshape(-1)[order]
    counts = torch.bincount(idx.reshape(-1)[order], minlength=cfg.experts)
    starts = (counts.cumsum(0) - counts).tolist()
    counts = counts.tolist()
    for e in range(cfg.experts):
        if counts[e] == 0:
            continue
        sl = slice(starts[e], starts[e] + counts[e])
        p = f"mlp.experts.{e}."
        rows = flat[tid[sl]]
        g = rows @ ly.w(f"{p}gate_proj.weight").T
        u = rows @ ly.w(f"{p}up_proj.weight").T
        y = swiglu(g, u, limit) @ ly.w(f"{p}down_proj.weight").T
        out.index_add_(0, tid[sl], y.float() * wid[sl, None])

    if cfg.shared_experts:
        out = out + mlp(flat, ly, "mlp.shared_experts.", cfg.limit(cfg.shared_limit, layer) if clamp else 0.0).float()
    return out.view(b, t, hd)


# --------------------------------------------------------------------------------------------- the model


class Ling:
    """The whole model, one layer at a time.

    Activations and weights are fp32 by default: the reference is the most accurate computation of a model whose
    checkpoint is bf16, which is also what the engine's reference path does (ADR 0001). `--dtype bfloat16` runs it
    the way the remote HF code does, to measure that gap.
    """

    def __init__(self, model_dir, device: str = "cuda", dtype=torch.float32):
        self.dir = Path(model_dir)
        self.cfg = Config.load(self.dir)
        self.st = Shards(self.dir, device)
        self.device = device
        self.dtype = dtype
        d = self.cfg.qk_rope
        self.inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.st.get("model.word_embeddings.weight")[ids.to(self.device)].float()

    def layer_weights(self, layer: int) -> Layer:
        return Layer(self.st, f"model.layers.{layer}.", self.dtype)

    def _layer(self, x, ly, layer, pos, clamp):
        cfg = self.cfg
        h = rmsnorm(x, ly.w("input_layernorm.weight"), cfg.eps)
        x = x + (mla(h, ly, cfg, pos, self.inv_freq) if cfg.is_mla(layer) else kda(h, ly, cfg))
        h = rmsnorm(x, ly.w("post_attention_layernorm.weight"), cfg.eps)
        limit = cfg.limit(cfg.expert_limit, layer) if clamp else 0.0
        x = x + (moe(h, ly, cfg, layer, clamp) if cfg.is_moe(layer) else mlp(h, ly, "mlp.", limit))
        return x

    def run(self, ids, clamp=True, both=False, save_layers: Path | None = None):
        """Final-normed hidden states [B, T, hidden] in fp32.

        `both` carries a second copy of the batch through the same layer visit with the SwiGLU clamp disabled and
        returns the pair, so the ADR 0004 measurement costs one pass over the shards instead of two.
        """
        cfg = self.cfg
        ids = ids.to(self.device)
        b, t = ids.shape
        pos = torch.arange(t, device=self.device).expand(b, t)
        x0 = self.embed(ids)
        xs = [x0, x0.clone()] if both else [x0]
        if save_layers is not None:
            save_x0(x0, save_layers, "embed")
        for layer in range(cfg.layers):
            ly = self.layer_weights(layer)
            for j, x in enumerate(xs):
                xs[j] = self._layer(x, ly, layer, pos, clamp if not both else j == 0)
            if save_layers is not None:
                save_x0(xs[0], save_layers, f"layer{layer:02d}")
            del ly
        norm = self.st.get("model.norm.weight", self.dtype)
        out = [rmsnorm(x, norm, cfg.eps) for x in xs]
        if save_layers is not None:
            save_x0(out[0], save_layers, "final")
        return tuple(out) if both else out[0]

    def mtp(self, hidden: torch.Tensor, ids: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """The MTP head: the embedding of the *next* token plus the final-normed hidden state."""
        cfg = self.cfg
        ly = self.layer_weights(cfg.layers)
        e = rmsnorm(self.embed(torch.roll(ids.to(self.device), shifts=-1, dims=-1)), ly.w("enorm.weight"), cfg.eps)
        h = rmsnorm(hidden, ly.w("hnorm.weight"), cfg.eps)
        x = torch.cat([e, h], dim=-1) @ ly.w("eh_proj.weight").T
        pos = torch.arange(x.shape[1], device=self.device).expand(x.shape[0], -1)
        n = rmsnorm(x, ly.w("input_layernorm.weight"), cfg.eps)
        x = x + mla(n, ly, cfg, pos, self.inv_freq)
        n = rmsnorm(x, ly.w("post_attention_layernorm.weight"), cfg.eps)
        x = x + moe(n, ly, cfg, cfg.layers, clamp)
        return rmsnorm(x, ly.w("final_layernorm.weight"), cfg.eps)

    def logits(self, hidden: torch.Tensor, sample: int = 0) -> torch.Tensor:
        """[T, vocab] fp32 logits of one sample, computed on the GPU and returned on the CPU."""
        head = self.st.get("lm_head.weight", self.dtype)
        return (hidden[sample].to(self.dtype) @ head.T).float().cpu()


def save_x0(x: torch.Tensor, out: Path, name: str):
    """Hidden states of sample 0 as raw f32 [T, hidden]; `layers.json` records the shape once."""
    out.mkdir(parents=True, exist_ok=True)
    x[0].float().cpu().numpy().astype("float32").tofile(out / f"{name}.f32")
    if name == "embed":
        (out / "layers.json").write_text(json.dumps({"shape": list(x[0].shape), "dtype": "f32"}))


# --------------------------------------------------------------------------------------------- tokenizer


def tokenizer(model_dir: Path):
    """`PreTrainedTokenizerFast` on tokenizer.json: no remote code, which the checkpoint would otherwise pull in."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(Path(model_dir) / "tokenizer.json"),
        eos_token="<|role_end|>",
        pad_token="<|endoftext|>",
    )


def chat_template(model_dir: Path) -> str:
    p = Path(model_dir) / "chat_template.jinja"
    if p.exists():
        return p.read_text()
    cfg = json.loads((Path(model_dir) / "tokenizer_config.json").read_text())
    return cfg["chat_template"]


def render(tok, model_dir: Path, messages, thinking: bool = True, tools=None) -> str:
    return tok.apply_chat_template(
        messages,
        chat_template=chat_template(model_dir),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )


# --------------------------------------------------------------------------------------------- subcommands


def cmd_selftest(args):
    """Our KDA / conv / gated norm against fla-core (MIT) on random tensors."""
    dev = args.device
    torch.manual_seed(7)
    b, t, h, kd = 2, 41, 4, 32
    q = torch.randn(b, t, h, kd, device=dev)
    k = torch.randn(b, t, h, kd, device=dev)
    v = torch.randn(b, t, h, kd, device=dev)
    beta = torch.rand(b, t, h, device=dev)
    g = -5.0 * torch.rand(b, t, h, kd, device=dev).sigmoid()
    scale = kd**-0.5
    ours = kda_recurrent(q * scale, k, v, g, beta)
    try:
        from fla.ops.kda.naive import naive_recurrent_kda
    except ImportError as e:
        print(f"fla-core not installed ({e}): install it with `pip install fla-core` to run this check")
        return 1
    ref = naive_recurrent_kda(q, k, v, g, beta, scale=scale, output_final_state=False)[0].float()
    err = (ours - ref).abs().max().item()
    print(f"kda_recurrent vs fla naive_recurrent_kda: max |d| {err:.3e}")

    u = torch.randn(3, 17, 64, device=dev)
    w = torch.randn(64, 1, 4, device=dev)
    ref = F.silu(F.conv1d(F.pad(u.transpose(1, 2), (3, 0)), w, groups=64).transpose(1, 2))
    err2 = (short_conv(u, w) - ref).abs().max().item()
    print(f"short_conv vs F.conv1d:                 max |d| {err2:.3e}")

    x = torch.randn(5, 3, 32, device=dev)
    gg = torch.randn(5, 3, 32, device=dev)
    ww = torch.randn(32, device=dev)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * ww * torch.sigmoid(gg.float())).to(x.dtype)
    err3 = (gated_rmsnorm(x, gg, ww, 1e-6) - ref).abs().max().item()
    print(f"gated_rmsnorm vs the fla formula:       max |d| {err3:.3e}")
    ok = max(err, err2, err3) < 2e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_hfcheck(args):
    """Builds a reduced random model, runs it with the checkpoint's own remote code and with this reference."""
    import tempfile

    dev = args.device
    torch.manual_seed(args.seed)
    out = Path(args.out or tempfile.mkdtemp(prefix="ling-reduced-"))
    out.mkdir(parents=True, exist_ok=True)
    hidden, heads, kd = 256, 4, 32
    cfg = {
        "architectures": ["BailingMoeV3ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_bailing_moe_v3.BailingMoeV3Config",
            "AutoModelForCausalLM": "modeling_bailing_moe_v3.BailingMoeV3ForCausalLM",
        },
        "model_type": "bailing_hybrid",
        "hidden_size": hidden,
        "num_hidden_layers": 7,
        "num_nextn_predict_layers": 1,
        "vocab_size": 512,
        "layer_group_size": 6,
        "first_k_dense_replace": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 6000000,
        "rope_interleave": True,
        "rope_scaling": None,
        "max_position_embeddings": 4096,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "head_dim": kd,
        "v_head_dim": kd,
        "qk_nope_head_dim": kd,
        "qk_rope_head_dim": 16,
        "qk_head_dim": kd + 16,
        "kv_lora_rank": 64,
        "q_lora_rank": None,
        "gated_attention_proj_granularity_type": "head_wise",
        "short_conv_kernel_size": 4,
        "kda_safe_gate": True,
        "kda_lower_bound": -5.0,
        "no_kda_lora": True,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "moe_shared_expert_intermediate_size": 128,
        "num_shared_experts": 1,
        "n_group": 4,
        "topk_group": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "moe_router_enable_expert_bias": True,
        "intermediate_size": 384,
        "score_function": "sigmoid",
        "topk_method": "noaux_tc",
        "tie_word_embeddings": False,
        "use_bias": False,
        "use_qkv_bias": False,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "expert_swiglu_limit_list": [0, 0, 0, 0, 0, 0, 4],
        "share_expert_swiglu_limit_list": [0, 0, 0, 0, 0, 3, 5],
        # The keys below are legacy/unused by the reference code but the remote config class expects them.
        "mtp_use_kda": False,
        "use_qk_norm": True,
        "use_kda_lora": False,
        "num_kv_heads_for_linear_attn": 0,
        "linear_silu": True,
        "group_norm_size": 1,
        "up_proj_norm": False,
        "value_norm": False,
        "use_nGPT": False,
        "partial_rotary_factor": 0.5,
        "rotary_dim": 16,
        "max_window_layers": 4,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
        "output_dropout": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "router_dtype": "fp32",
        "seq_aux": True,
        "scale_router_input": False,
        "mtp_loss_scaling_factor": 0,
        "use_cache": True,
        "output_router_logits": False,
        "torch_dtype": "bfloat16",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ("configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"):
        (out / f).write_text((Path(args.model) / f).read_text())

    from safetensors.torch import save_file
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    # The checkpoint's remote code targets transformers 4.x (`is_torch_fx_available` and friends are gone in 5.x),
    # so this check needs `pip install 'transformers<5'` in the container.
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

    # Random weights with the upstream dtypes: A_log, dt_bias and expert_bias are f32, everything else bf16.
    # Two departures from "just random", both needed to make the comparison meaningful (see the inline notes).
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(out, trust_remote_code=True),
            trust_remote_code=True,
            **{dtype_kw: torch.bfloat16},
        )
    weights = {}
    # state_dict(), not named_parameters(): `mlp.gate.expert_bias` is a persistent buffer.
    for name, p in skeleton.state_dict().items():
        if name.endswith("expert_bias"):
            # A ramp of 0.5 per expert, not random: with random weights the group sums sit within 1e-2 of each
            # other, so the 1e-2 accumulated dtype gap between the two implementations flips the routing and the
            # end-to-end comparison measures the flip, not the math. A 0.5 margin is immune to it, and to the
            # bf16 rounding `from_pretrained` applies to buffers (~0.014 at these magnitudes).
            w = torch.arange(p.shape[0], dtype=torch.float32) * 0.5
        elif name.endswith(("A_log", "dt_bias")):
            w = (
                torch.log(torch.rand(p.shape) * 15 + 1)
                if name.endswith("A_log")
                else torch.randn(p.shape) * 0.5
            )
        elif name.endswith("mlp.gate.weight"):
            # 10x the other weights: with std 0.02 every sigmoid score lands near 0.5, the four group sums are
            # within 1e-2 of each other and the top-2 group choice becomes a coin flip on rounding noise, which
            # would swamp the end-to-end comparison. A trained router has well-separated scores.
            w = torch.randn(p.shape) * 0.2
        elif p.dim() == 1:
            w = torch.randn(p.shape) * 0.1 + 1.0
        else:
            w = torch.randn(p.shape) * 0.02
        weights[name] = w.to(torch.float32 if w.dtype == torch.float32 else torch.bfloat16)
    del skeleton
    save_file({k: v.contiguous() for k, v in weights.items()}, str(out / "model.safetensors"), {"format": "pt"})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )

    model = (
        # attn_implementation="eager" is required: with the default sdpa the model builds no 4-D mask
        # (_prepare_4d_causal_attention_mask_for_sdpa returns None) while the MLA module still calls
        # eager_attention_forward, which then runs WITHOUT causal masking.
        AutoModelForCausalLM.from_pretrained(
            out, trust_remote_code=True, attn_implementation="eager", **{dtype_kw: torch.bfloat16}
        )
        .to(dev)
        .eval()
    )
    ids = torch.randint(3, cfg["vocab_size"], (2, 23), device=dev)
    with torch.no_grad():
        hf = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        ref = hf.logits[0].float().cpu()
    # The remote code ignores the SwiGLU limit lists, so the comparison runs our reference without the clamp.
    dumps = out / "layers"
    ours = Ling(out, device=dev).run(ids, clamp=False, save_layers=dumps)
    got = Ling(out, device=dev).logits(ours, 0)
    worst = 0.0
    if args.verbose:
        import numpy as np

        hidden = cfg["hidden_size"]
        nl, gl = cfg["num_hidden_layers"], cfg["layer_group_size"]
        # The remote code appends the state *before* each layer, then the final-normed one: what is exposed is
        # embed, after layer 0 .. after layer nl-2, and the final hidden. Our dumps have one entry more.
        names = ["embed"] + [f"layer{i:02d}" for i in range(nl - 1)] + ["final"]
        print(f"{'layer':>8} {'kind':>5} {'rel rms':>10} {'max |d|':>10}")
        hs = [t[0].float().cpu() for t in hf.hidden_states]
        assert len(hs) == len(names), f"remote code exposed {len(hs)} hidden states, expected {len(names)}"
        for i, nm in enumerate(names):
            mine = torch.from_numpy(np.fromfile(dumps / f"{nm}.f32", dtype="float32").reshape(-1, hidden))
            theirs = hs[i]
            rel = ((mine - theirs).pow(2).mean().sqrt() / theirs.pow(2).mean().sqrt().clamp(min=1e-9)).item()
            worst = max(worst, rel)
            j = i - 1
            kind = "-" if nm in ("embed", "final") else ("mla" if (j + 1) % gl == 0 or j >= nl // gl * gl else "kda")
            print(f"{nm:>8} {kind:>5} {rel:10.3e} {(mine - theirs).abs().max().item():10.4f}")
    ref_rms = ref.pow(2).mean().sqrt().item()
    rel = ((got - ref).pow(2).mean().sqrt() / max(ref_rms, 1e-9)).item()
    err = (got - ref).abs().max().item()
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(
        f"reduced model in {out}: logits rms {ref_rms:.3f}, max |d| {err:.4f}, "
        f"relative rms error {rel:.2e}, argmax agreement {agree * 100:.2f}%"
    )
    # The criterion is the hidden-state error, not the argmax: our reference runs fp32 weights and activations
    # while the remote code runs bf16, and on a random model the logits are all within a few 1e-2 of each other,
    # so the argmax is a near-tie by construction. A transcription error shows up as an order-of-magnitude gap
    # in some layer, which is what `probe` then localizes.
    ok = max(worst, rel) < 0.05
    print(f"PASS (worst relative error {max(worst, rel):.2e} < 5e-2)" if ok else "FAIL")
    return 0 if ok else 1


def cmd_layers(args):
    """Per-layer hidden states of one token list, for the layer-by-layer comparison of MIGRATE_MODELS.md §5."""
    m = build(args)
    src = Path(args.ids)
    ids = json.loads(src.read_text()) if src.exists() else tokenizer(args.model)(args.ids).input_ids
    out = Path(args.out)
    t0 = time.time()
    h = m.run(torch.tensor([ids]), clamp=not args.no_clamp, save_layers=out)
    got = m.logits(h, 0)
    got[-1].numpy().astype("float32").tofile(out / "logits_last.f32")
    got.numpy().argmax(-1).astype("int32").tofile(out / "argmax.i32")
    print(f"T={len(ids)} in {time.time() - t0:.0f}s -> {out} ({m.cfg.layers + 1} dumps), argmax {got[0, -1].argmax().item()}")
    return 0


def cmd_route(args):
    """Routing trace of a text, position by position, for `scripts/expert_cache.py` (issue #50).

    One forward pass streams every layer from the bf16 shards, so the cache measurement does not need the 62 GiB
    of the converted file on the device. The output is reordered by position (all layers of position 0, then all
    of position 1, ...), which is the shape of a decode stream: the simulator groups a step every time the layer
    index goes backwards.
    """
    global ROUTE_LOG
    ids = tokenizer(Path(args.model))(Path(args.text).read_text()).input_ids[: args.tokens]
    m = build(args)
    ROUTE_LOG = {}
    t0 = time.time()
    m.run(torch.tensor([ids]), clamp=not args.no_clamp)
    dt = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in range(len(ids)):
            for layer in sorted(ROUTE_LOG):
                idx, w = ROUTE_LOG[layer]
                f.write(f"{layer} 1 {p}")
                for i, wv in zip(idx[p].tolist(), w[p].tolist()):
                    f.write(f" {i}:{wv:.6f}")
                f.write("\n")
    ROUTE_LOG = None
    print(f"T={len(ids)} in {dt:.0f}s -> {out} ({len(ids)} positions x routing)")
    return 0


def cmd_probe(args):
    """Compares one layer's attention, FFN and router against the remote code, in isolation.

    Each block is fed the exact input the remote code gave it (captured with a forward hook), so an error cannot be
    an upstream difference propagating. Needs the reduced model written by `hfcheck`.
    """
    dev = args.device
    out = Path(args.dir)
    cfgj = json.loads((out / "config.json").read_text())
    hidden = cfgj["hidden_size"]
    torch.manual_seed(args.seed)
    ids = torch.randint(3, cfgj["vocab_size"], (1, args.length), device=dev)

    import transformers
    from transformers import AutoModelForCausalLM

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = (
        # attn_implementation="eager" is required: with the default sdpa the model builds no 4-D mask
        # (_prepare_4d_causal_attention_mask_for_sdpa returns None) while the MLA module still calls
        # eager_attention_forward, which then runs WITHOUT causal masking.
        AutoModelForCausalLM.from_pretrained(
            out, trust_remote_code=True, attn_implementation="eager", **{dtype_kw: torch.bfloat16}
        )
        .to(dev)
        .eval()
    )
    cap = {}
    layer = model.model.layers[args.layer]

    def hook(name, flat=True):
        # with_kwargs: the decoder layer calls `attention(hidden_states=...)` with keywords only.
        def f(mod, a, kw, o):
            x = a[0] if a else kw["hidden_states"]
            y = o[0] if isinstance(o, tuple) else o
            cap[name] = (
                x.detach().float().cpu().reshape(-1, hidden) if flat else x.detach().float().cpu(),
                y.detach().float().cpu().reshape(-1, hidden) if flat else y.detach().float().cpu(),
            )

        return f

    layer.attention.register_forward_hook(hook("attn"), with_kwargs=True)
    layer.mlp.register_forward_hook(hook("ffn"), with_kwargs=True)
    gate = getattr(layer.mlp, "gate", None)
    if gate is not None:
        # The gate returns (topk_idx, topk_weight, logits): keep the shapes, only its input matters here.
        gate.register_forward_hook(hook("gate", flat=False), with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)

    m = Ling(out, device=dev)
    ly = m.layer_weights(args.layer)
    cfg = m.cfg
    t = ids.shape[1]
    pos = torch.arange(t, device=dev)[None]

    def cmp(name, mine, ref_out):
        """Both blocks got the remote code's own input, so any difference is the block's math."""
        mine = mine.reshape(-1, hidden).float().cpu()
        rel = ((mine - ref_out).pow(2).mean().sqrt() / ref_out.pow(2).mean().sqrt().clamp(min=1e-12)).item()
        print(
            f"{name}: output rel rms {rel:.3e}  max |d| {(mine - ref_out).abs().max().item():.4f}  "
            f"rms ours {mine.pow(2).mean().sqrt().item():.4f} vs remote {ref_out.pow(2).mean().sqrt().item():.4f}"
        )

    def dev3(cpu_flat):
        return cpu_flat.to(dev)[None]

    attn_in, attn_ref = cap["attn"]
    x = dev3(attn_in)
    attn = mla(x, ly, cfg, pos, m.inv_freq) if cfg.is_mla(args.layer) else kda(x, ly, cfg)
    cmp("attn", attn, attn_ref)

    ffn_in, ffn_ref = cap["ffn"]
    x = dev3(ffn_in)
    ffn = (
        moe(x, ly, cfg, args.layer, False)
        if cfg.is_moe(args.layer)
        else mlp(x, ly, "mlp.", cfg.limit(cfg.expert_limit, args.layer))
    )
    cmp("ffn ", ffn, ffn_ref)

    if gate is None:
        return 0
    ref_gate_in = cap["gate"][0]
    logits = dev3(ref_gate_in).reshape(-1, hidden) @ ly.w("mlp.gate.weight").T.float()
    idx, w = group_topk(logits, ly.f32("mlp.gate.expert_bias") if cfg.expert_bias else None, cfg)
    with torch.no_grad():
        tidx, tw, tlogits = gate(ref_gate_in.to(dev))
    same = (idx.cpu() == tidx.cpu()).float().mean().item()
    print(
        f"router: expert id agreement {same * 100:.2f}%, weight max |d| "
        f"{(w.cpu() - tw.float().cpu()).abs().max().item():.3e}, "
        f"logit max |d| {(logits.cpu() - tlogits.float().cpu()).abs().max().item():.3e}"
    )
    if same < 1.0:
        # A flip on a near-tie is expected: the remote code holds `expert_bias` in bf16 (from_pretrained casts the
        # buffers), we keep the checkpoint's fp32. Print the two competing biased scores to confirm.
        bias32 = ly.f32("mlp.gate.expert_bias").cpu()
        bias16 = gate.expert_bias.detach().float().cpu()
        s32 = torch.sigmoid(logits.cpu())
        s16 = torch.sigmoid(tlogits.float().cpu())
        for r in (idx.cpu() != tidx.cpu()).any(-1).nonzero()[:3].flatten().tolist():
            cand = sorted(set(idx.cpu()[r].tolist()) | set(tidx.cpu()[r].tolist()))
            print(f"  token {r}: ours {idx.cpu()[r].tolist()} w {[round(v, 5) for v in w.cpu()[r].tolist()]}")
            print(f"           hf   {tidx.cpu()[r].tolist()} w {[round(v, 5) for v in tw.cpu()[r].tolist()]}")
            for e in cand:
                print(
                    f"           expert {e}: biased score fp32 {s32[r, e] + bias32[e]:.8f} "
                    f"bf16-bias {s16[r, e] + bias16[e]:.8f}  (bias fp32 {bias32[e]:.6f} bf16 {bias16[e]:.6f})"
                )
    return 0


def cmd_trace(args):
    """Compares every MLA intermediate with the remote code's, to pinpoint a transcription error."""
    dev = args.device
    out = Path(args.dir)
    cfgj = json.loads((out / "config.json").read_text())
    torch.manual_seed(args.seed)
    ids = torch.randint(3, cfgj["vocab_size"], (1, args.length), device=dev)

    import transformers
    from transformers import AutoModelForCausalLM

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = (
        # attn_implementation="eager" is required: with the default sdpa the model builds no 4-D mask
        # (_prepare_4d_causal_attention_mask_for_sdpa returns None) while the MLA module still calls
        # eager_attention_forward, which then runs WITHOUT causal masking.
        AutoModelForCausalLM.from_pretrained(
            out, trust_remote_code=True, attn_implementation="eager", **{dtype_kw: torch.bfloat16}
        )
        .to(dev)
        .eval()
    )
    attn = model.model.layers[args.layer].attention
    cap = {}

    def hook(name):
        def f(mod, a, kw, o):
            x = a[0] if a else kw["hidden_states"]
            y = o[0] if isinstance(o, tuple) else o
            cap[name] = (x.detach().float().cpu().flatten(), y.detach().float().cpu().flatten())

        return f

    for nm in ("q_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "g_proj", "dense"):
        getattr(attn, nm).register_forward_hook(hook(nm), with_kwargs=True)
    attn.register_forward_hook(hook("attn"), with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)

    m = Ling(out, device=dev)
    ly = m.layer_weights(args.layer)
    cfg = m.cfg
    x = cap["attn"][0].reshape(1, args.length, cfg.hidden).to(dev)
    pos = torch.arange(args.length, device=dev)[None]

    def cmp(name, mine):
        mine = mine.detach().float().cpu().flatten()
        ref = cap[name][1]
        rel = ((mine - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-12)).item()
        print(f"{name:22s} rel rms {rel:.3e}  rms ours {mine.pow(2).mean().sqrt():.5f} remote {ref.pow(2).mean().sqrt():.5f}")
        return rel

    b, t, h, nope, rope = 1, args.length, cfg.heads, cfg.qk_nope, cfg.qk_rope
    cmp("q_proj", x @ ly.w("attention.q_proj.weight").T)
    cmp("kv_a_proj_with_mqa", x @ ly.w("attention.kv_a_proj_with_mqa.weight").T)
    ckv = x @ ly.w("attention.kv_a_proj_with_mqa.weight").T
    latent, k_rot0 = ckv[..., : cfg.kv_lora], ckv[..., cfg.kv_lora :]
    cn = rmsnorm(latent, ly.w("attention.kv_a_layernorm.weight"), cfg.eps)
    cmp("kv_a_layernorm", cn)
    kv = (cn @ ly.w("attention.kv_b_proj.weight").T).view(b, t, h, nope + cfg.v_dim)
    cmp("kv_b_proj", kv)
    k_pass, v = kv[..., :nope], kv[..., nope:]
    q = (x @ ly.w("attention.q_proj.weight").T).view(b, t, h, cfg.qk_dim)
    q_pass, q_rot0 = q[..., :nope], q[..., nope:]
    cmp("g_proj", x @ ly.w("attention.g_proj.weight").T)

    q_rot, k_rot = rope_interleaved(q_rot0, k_rot0.unsqueeze(2), m.inv_freq, pos)
    print(f"q_rot[0,0,0,:8]  ours {q_rot[0, 0, 0, :8].tolist()}")
    qq = torch.cat([q_pass.float(), q_rot.float()], dim=-1)
    kk = torch.cat([k_pass.float(), k_rot.float().expand(b, t, h, rope)], dim=-1)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=dev), diagonal=1)
    s = qq[0].transpose(0, 1) @ kk[0].transpose(0, 1).transpose(-1, -2)
    p = torch.softmax(s.masked_fill(causal, float("-inf")) * cfg.qk_dim**-0.5, dim=-1)
    print(f"scores row {t - 1} head 0 (ours) {s[0, -1].tolist()[:6]}")
    print(f"probs  row {t - 1} head 0 (ours) sum {p[0, -1].sum().item():.6f}")
    o = (p @ v[0].float().transpose(0, 1)).transpose(0, 1)[None]
    gate = torch.sigmoid((x @ ly.w("attention.g_proj.weight").T).float())
    gated = (o * gate[..., None]).reshape(b, t, h * cfg.v_dim)
    ref_gated = cap["dense"][0]
    mine_gated = gated.detach().float().cpu().flatten()
    print(
        f"{'gated (dense input)':22s} rel rms "
        f"{((mine_gated - ref_gated).pow(2).mean().sqrt() / ref_gated.pow(2).mean().sqrt().clamp(min=1e-12)).item():.3e}"
        f"  rms ours {mine_gated.pow(2).mean().sqrt():.5f} remote {ref_gated.pow(2).mean().sqrt():.5f}"
    )
    cmp("dense", gated @ ly.w("attention.dense.weight").T)
    return 0


def sample_docs(args, tok):
    """The evaluation documents: Gutenberg books and code, disjoint from the calibration corpus."""
    import hashlib
    import random

    sys.path.insert(0, str(Path(__file__).parent))
    from calibrate import gutenberg_body  # noqa: E402

    texts = []
    for gid, lang, sha in args.gutenberg:
        data = (Path(args.corpus) / f"pg{gid}.txt").read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            print(f"warning: pg{gid}.txt sha256 differs from the manifest", flush=True)
        texts.append((f"{lang}-pg{gid}", gutenberg_body(data.decode("utf-8", "replace"))))
    import transformers

    site = Path(transformers.__file__).parents[1]
    for rel in ("transformers/generation/utils.py", "torch/nn/modules/module.py"):
        texts.append((f"code-{Path(rel).stem}", (site / rel).read_text()))

    rng = random.Random(args.seed)
    docs = []
    for name, text in texts:
        ids = tok(text).input_ids
        off = rng.randrange(len(ids) // 4, max(len(ids) // 4 + 1, len(ids) - 2 * args.length))
        docs.append((name, ids[off : off + args.length]))
    for lang, (name, text) in (("it", texts[0]), ("en", texts[2])):
        ids = tok(text).input_ids
        off = rng.randrange(len(ids) // 2, max(len(ids) // 2 + 1, len(ids) - 2 * args.length))
        body = tok.decode(ids[off : off + args.length])
        ask = "Continua questo racconto:" if lang == "it" else "Continue this story:"
        cut = len(body) // 4
        chat = render(
            tok,
            args.model,
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{ask}\n\n{body[:cut]}"},
                {"role": "assistant", "content": body[cut:]},
            ],
            thinking=args.thinking,
        )
        docs.append((f"chat-{lang}", tok(chat).input_ids[: args.length]))
    return docs


def batch(ids_list, pad):
    """Pads to a common length; the padding is causal-after every real token, so it cannot change their outputs."""
    t = max(len(i) for i in ids_list)
    ids = torch.full((len(ids_list), t), pad, dtype=torch.long)
    for r, i in enumerate(ids_list):
        ids[r, : len(i)] = torch.tensor(i)
    return ids, torch.tensor([len(i) for i in ids_list])


def cmd_eval(args):
    """Reference for `malleus eval`: per-position top-k log-probs, argmax and NLL."""
    import numpy as np

    m = build(args)
    tok = tokenizer(args.model)
    args.gutenberg = GUTENBERG
    docs = sample_docs(args, tok)
    ids, lengths = batch([d for _, d in docs], m.cfg.eos)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = Path(args.blob_dir)
    blob.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = m.run(ids, clamp=not args.no_clamp)
    print(f"forward in {time.time() - t0:.0f}s", flush=True)
    meta = {"topk": args.topk, "docs": []}
    with open(blob / "topk.bin", "wb") as f:
        for r, (name, doc) in enumerate(docs):
            n = lengths[r].item()
            lg = m.logits(h, r)[:n]
            lp = torch.log_softmax(lg, -1)
            nxt = torch.tensor(doc[1:])
            nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
            top = torch.topk(lp, args.topk, dim=-1)
            for i in range(n):
                f.write(top.indices[i].to(torch.int32).numpy().tobytes())
                f.write(top.values[i].numpy().astype(np.float32).tobytes())
            meta["docs"].append(
                {
                    "id": name,
                    "input_ids": doc,
                    "argmax": lg.argmax(-1).tolist(),
                    "nll": [round(v, 5) for v in nll.tolist()],
                }
            )
            print(f"{name:22s} T={n} ppl {float(nll.mean().exp()):.3f}", flush=True)
    out.write_text(json.dumps(meta))
    print(f"wrote {out} and {blob / 'topk.bin'}")
    return 0


def cmd_clamp(args):
    """ADR 0004: the SwiGLU clamp is in vLLM and SGLang but not in the remote HF code. Measure which one fits better."""
    m = build(args)
    tok = tokenizer(args.model)
    args.gutenberg = GUTENBERG
    docs = sample_docs(args, tok)
    ids, lengths = batch([d for _, d in docs], m.cfg.eos)
    t0 = time.time()
    with_clamp, without = m.run(ids, both=True)
    print(f"two variants in one pass, {time.time() - t0:.0f}s", flush=True)
    tot = [0.0, 0.0, 0]
    for r, (name, doc) in enumerate(docs):
        n = lengths[r].item()
        nxt = torch.tensor(doc[1:])
        row = []
        for h in (with_clamp, without):
            lp = torch.log_softmax(m.logits(h, r)[:n], -1)
            nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
            row.append(float(nll.mean()))
            tot[len(row) - 1] += float(nll.sum())
        tot[2] += n - 1
        print(f"{name:22s} ppl clamp {row[0]:.4f}   no clamp {row[1]:.4f}   diff {row[0] - row[1]:+.4f}", flush=True)
    a, b, n = tot[0] / tot[2], tot[1] / tot[2], tot[2]
    print(f"TOTAL ppl clamp {a:.4f}  no clamp {b:.4f}  ({100 * (a / b - 1):+.3f}%)")
    print("the clamp is the right behaviour" if a < b else "the remote HF code (no clamp) is the right behaviour")
    return 0


TEXT_PROMPTS = [
    ("en-fact", "What is the capital of France? Answer in one word."),
    ("en-explain", "Explain in two sentences why the sky is blue."),
    ("en-list", "List three prime numbers greater than 100."),
    ("en-story", "Write the opening sentence of a mystery novel set in a lighthouse."),
    ("en-math", "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?"),
    ("en-reason", "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"),
    ("it-fact", "Qual è il fiume più lungo d'Italia?"),
    ("it-explain", "Spiega in due frasi cos'è la fotosintesi."),
    ("it-translate", "Traduci in inglese: 'Domani andiamo al mare se non piove.'"),
    ("it-email", "Scrivi una breve email per spostare una riunione da lunedì a martedì."),
    ("it-recipe", "Quali sono gli ingredienti della carbonara tradizionale?"),
    ("code-py", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("code-rs", "Write a Rust function that reverses a string slice and returns a String."),
    ("code-sql", "Write a SQL query that returns the 5 most recent orders from table orders(id, created_at)."),
    ("code-bug", "Find the bug:\n\ndef mean(xs):\n    return sum(xs) / len(xs) + 1\n"),
    ("code-explain", "What does `git rebase -i HEAD~3` do?"),
    ("code-regex", "Write a regex that matches an Italian fiscal code (codice fiscale)."),
    ("mixed-json", "Return a JSON object with keys name and age for a person named Marco aged 42."),
    ("tool-weather", "What is the weather in Rome today?"),
    ("long-context", "Summarize: " + " ".join(["The committee met on Monday to discuss the budget."] * 24)),
    ("raw-continue", None),
]
RAW_TEXT = "The quick brown fox jumps over the lazy dog. In 1969, humans first"
GREEDY_IDS = ["en-explain", "it-explain", "code-py", "raw-continue"]
TOOL_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        },
    }
]
GUTENBERG = [
    ("56775", "it", "31f2d1f7e0575af592559610a1bedad7bad40c15ef4af431f2b44bd154729ef2"),
    ("18456", "it", "7eed1911796c417cad328b4ee8b1978ad4fa0a040cd5c8c20cd22cb9274ca833"),
    ("48320", "en", "bf63b19ade817616aa64b5dcaf32e8685592c315fd22e336defe685baa02d419"),
    ("11", "en", "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"),
]


def cmd_golden(args):
    """Golden set: token ids, per-position argmax, last-token logits in the blob dir, greedy continuations."""
    m = build(args)
    tok = tokenizer(args.model)
    out, blob = Path(args.out), Path(args.blob_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"torch": torch.__version__, "dtype": str(m.dtype), "text": []}
    entries = []
    for pid, text in TEXT_PROMPTS:
        if text is None:
            ids = tok(RAW_TEXT).input_ids
        else:
            tools = TOOL_WEATHER if pid == "tool-weather" else None
            prompt = render(tok, args.model, [{"role": "user", "content": text}], thinking=args.thinking, tools=tools)
            ids = tok(prompt).input_ids
        entries.append((pid, ids))
    padded, lengths = batch([i for _, i in entries], m.cfg.eos)
    t0 = time.time()
    h = m.run(padded, clamp=not args.no_clamp)
    print(f"forward in {time.time() - t0:.0f}s", flush=True)
    for r, (pid, ids) in enumerate(entries):
        n = lengths[r].item()
        lg = m.logits(h, r)[:n]
        d = blob / "text" / pid
        d.mkdir(parents=True, exist_ok=True)
        lg[-1].numpy().astype("float32").tofile(d / "logits_last.f32")
        top = torch.topk(torch.log_softmax(lg[-1], -1), 10)
        e = {
            "id": pid,
            "input_ids": ids,
            "argmax": lg.argmax(-1).tolist(),
            "top10_last": {"ids": top.indices.tolist(), "logprobs": top.values.tolist()},
        }
        if pid in GREEDY_IDS and args.greedy_tokens > 0:
            e["greedy"] = greedy(m, tok, ids, args.greedy_tokens)
            e["greedy_text"] = tok.decode(e["greedy"])
            print(f"  greedy {pid}: {e['greedy_text'][:70]!r}", flush=True)
        meta["text"].append(e)
        print(f"{pid:14s} T={n} next={tok.decode([e['argmax'][-1]])!r}", flush=True)
        (out / "golden.json").write_text(json.dumps(meta, indent=1))
    (out / "golden.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / 'golden.json'} and {blob}")
    return 0


def greedy(m: Ling, tok, ids: list, max_tokens: int) -> list:
    """Greedy continuation. Every token costs a full pass over the 238 GiB of shards (~15 min), so this is for a
    handful of tokens only; the normal check is the per-position argmax of a single forward, plus scoring the
    engine's own greedy output with `layers`."""
    cur = list(ids)
    out = []
    for i in range(max_tokens):
        t0 = time.time()
        h = m.run(torch.tensor([cur]))
        nxt = int(m.logits(h, 0)[-1].argmax())
        out.append(nxt)
        print(f"    greedy token {i + 1}/{max_tokens} = {nxt} in {time.time() - t0:.0f}s", flush=True)
        if nxt == m.cfg.eos:
            break
        cur.append(nxt)
    return out


def main():
    # The reference must be the most accurate computation available: no TF32 in the fp32 matmuls.
    torch.set_float32_matmul_precision("highest")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/models/bf16/inclusionAI/Ling-3.0-flash")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=sorted(DTYPE), default="float32",
                    help="float32 (default, the most accurate) or bfloat16 (what the remote HF code does)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="KDA / conv / gated norm against fla-core")
    s.set_defaults(fn=cmd_selftest)

    s = sub.add_parser("hfcheck", help="whole reference against the remote code on a reduced model")
    s.add_argument("--out", help="where to write the reduced model (default: a temporary directory)")
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--verbose", action="store_true", help="print the per-layer relative error to locate a mismatch")
    s.set_defaults(fn=cmd_hfcheck)

    s = sub.add_parser("probe", help="compare one layer's attention, FFN and router against the remote code")
    s.add_argument("--dir", default="/golden/ling-reduced", help="reduced model written by hfcheck")
    s.add_argument("--layer", type=int, default=2)
    s.add_argument("--length", type=int, default=23)
    s.add_argument("--seed", type=int, default=3)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("trace", help="compare every MLA intermediate with the remote code")
    s.add_argument("--dir", default="/golden/ling-reduced", help="reduced model written by hfcheck")
    s.add_argument("--layer", type=int, default=5)
    s.add_argument("--length", type=int, default=23)
    s.add_argument("--seed", type=int, default=3)
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("route", help="routing trace of a text, for scripts/expert_cache.py")
    s.add_argument("--text", required=True, help="a text file; the first --tokens tokens are used")
    s.add_argument("--tokens", type=int, default=512)
    s.add_argument("--out", default="/golden/route-ling.txt")
    s.add_argument("--no-clamp", action="store_true")
    s.set_defaults(fn=cmd_route)

    s = sub.add_parser("layers", help="per-layer hidden states of one token list")
    s.add_argument("--ids", required=True, help="text to tokenize, or a JSON file with a list of token ids")
    s.add_argument("--out", default="/golden/ling3-flash/layers")
    s.add_argument("--no-clamp", action="store_true")
    s.set_defaults(fn=cmd_layers)

    for name, fn, help_ in (
        ("golden", cmd_golden, "golden set for `malleus golden`"),
        ("eval", cmd_eval, "reference statistics for `malleus eval`"),
        ("clamp", cmd_clamp, "perplexity with and without the SwiGLU clamp"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--out", default=f"golden/ling3-flash/{'golden.json' if name == 'golden' else 'eval.json'}")
        s.add_argument("--blob-dir", default="/golden/ling3-flash")
        s.add_argument("--corpus", default="/golden/calib/eval")
        s.add_argument("--length", type=int, default=1024)
        s.add_argument("--topk", type=int, default=32)
        s.add_argument("--seed", type=int, default=4321)
        s.add_argument("--thinking", action="store_true", default=True)
        s.add_argument("--no-clamp", action="store_true")
        if name == "golden":
            s.add_argument(
                "--greedy-tokens",
                type=int,
                default=0,
                help="greedy continuation; each token is a full pass over the shards (~15 min), so 0 by default",
            )
        s.set_defaults(fn=fn)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
