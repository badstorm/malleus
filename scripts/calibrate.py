#!/usr/bin/env python3
"""GPTQ calibration of the MLS4 linears in the rotated (Hadamard) space (offline tool, never shipped).

Runs the bf16 model layer by layer on a fixed calibration corpus, accumulates H = sum x~ x~^T of the rotated inputs
of every MLS4 linear, solves GPTQ with the MLS4 grid (symmetric int4, group 128, fp16 scale) and writes, per layer,
the packed nibbles and scales that `malleus convert --mls4-from` uses instead of round-to-nearest.
MoE layers: H is accumulated per expert on the tokens routed to it and GPTQ runs on all experts of a layer at once.

Checkpoints whose bf16 weights do not fit next to the Hessians (Qwen3.8-Flash-Next) are read layer by layer, on the
pattern of scripts/ref_layerwise.py; see --layerwise.
"""
import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from golden import load_on_gpu  # noqa: E402
from ref_layerwise import (  # noqa: E402
    HY4_RENAME,
    KEEP_FP32,
    HostEngramEmbedding,
    ShardedRows,
    Shards,
    load_module,
)

BLOCK = 128
HADAMARD_SEED = 0x4D4C_5334_0001
M64 = (1 << 64) - 1

GUTENBERG = [
    ("45334", "it", "4cdd962f1910a49091b945eff4233798b8c13f74b011cee75a8cccc7467d3612"),
    ("52484", "it", "b51a398da257709bfd2c0007e42980935751276cdca4178d738fa4c92f799ae6"),
    ("18457", "it", "55438a28652a1c58875b60cd7f9520b7967b04407414ce5c01404403b40a35d4"),
    ("56958", "it", "cc3e450cd8e959436fe9e63fd88677e2d79b0c6779aa30e477b8e4a4e66410fe"),
    ("61885", "it", "5f4ca693f86b641891ce4dfc89636f9428ffb7434056f17ed59c1ab0f98be446"),
    ("1000", "it", "4669dcc00ee61ceffe92d871e61ea430cec87b35cbab24f19a4c0b1c7da521b2"),
    ("1342", "en", "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"),
    ("84", "en", "2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42"),
    ("2701", "en", "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d"),
    ("500", "en", "072a84b3da860036ebf81dbd305149ef0d908064c30d513c7f640599fabeb99b"),
]
CHAT_PROMPTS = {
    "it": ["Continua il testo seguente.", "Riassumi il brano.", "Riscrivi il passaggio in italiano moderno."],
    "en": ["Continue the following text.", "Summarize the passage.", "Rewrite this passage in plain English."],
    "code": ["Explain what this code does.", "Spiega questo codice.", "Review this code and suggest improvements."],
}
MLS4_FAMILIES = {
    "linear_attention": [("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"), ("linear_attn.out_proj",)],
    "full_attention": [("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.o_proj",)],
    # Qwen3.8-Flash-Next calls its sparse-attention layer indexed_attention; the projections keep the names of
    # full_attention. self_attn.indexer.index_qk_proj is left out: it does not read the layer input, and it picks
    # which keys are read at all, so a coarser grid there changes the attention pattern and not just its values.
    "indexed_attention": [("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.o_proj",)],
}
MLP_FAMILIES = [("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)]
MOE_FAMILIES = [("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"), ("mlp.shared_expert.down_proj",)]

# Hy4-flash (`hy_v4_flash`). The attention reads the layer input in three projections at once (`q_a`, `kv_a` and the
# output gate), while `q_b` and `kv_b` read the two normed latents and `o_proj` the attention output: five Hessians.
# The latent MoE reads the full-width input in `shared_experts.gate/up` and `pre_expert_proj`, `post_expert_proj`
# reads the experts' output. The Engram's value and four key projections all read the same flattened memory.
HY4_ATTN_FAMILIES = [
    ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.gate_proj"),
    ("self_attn.q_b_proj",),
    ("self_attn.kv_b_proj",),
    ("self_attn.o_proj",),
]
HY4_DENSE_FAMILIES = [("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)]
HY4_MOE_FAMILIES = [
    ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.pre_expert_proj"),
    ("mlp.shared_experts.down_proj",),
    ("mlp.post_expert_proj",),
]
HY4_ENGRAM_FAMILIES = [
    ("engram_layer.value_proj", "engram_layer.key_projs.0", "engram_layer.key_projs.1",
     "engram_layer.key_projs.2", "engram_layer.key_projs.3"),
]
# Module name -> checkpoint name, for the tensors the converter reads by their source name.
HY4_OUT_NAME = {
    "self_attn.gate_proj.weight": "self_attn.linear_gate.weight",
    "mlp.pre_expert_proj.weight": "mlp.routed_expert_down_proj.weight",
    "mlp.post_expert_proj.weight": "mlp.routed_expert_up_proj.weight",
}
# Hyper-connection mixing, --hc. GatedResidual reads one normalized state: down and block_inject both take
# hyper_input_normed, so they share one Hessian, while up takes silu(down(...) / hc_count) and needs its own.
# They are ~0.8 GB per decode token, a large enough share of what a decode step reads that they get their own
# grid instead of sharing one with the linears next to them.
HC_FAMILIES = [
    ("attn_hyper_connection.input_mix_weight_down", "attn_hyper_connection.block_inject_weight"),
    ("attn_hyper_connection.input_mix_weight_up",),
    ("mlp_hyper_connection.input_mix_weight_down", "mlp_hyper_connection.block_inject_weight"),
    ("mlp_hyper_connection.input_mix_weight_up",),
]
# Experts with fewer routed calibration tokens than this keep round-to-nearest.
MIN_EXPERT_TOKENS = 256
# MLS2 grid(s): four levels and a scale per group of 128. "e2m1" is the symmetric subset of the e2m1 table that a
# two-bit code can carry: -1.5, -0.5, 0.5, 1.5 (the full 16-entry table needs four bits per code).
MLS2_GRIDS = {
    "uniform": [-2.0, -1.0, 0.0, 1.0],
    "e2m1": [-1.5, -0.5, 0.5, 1.5],
}
# Above this checkpoint size the bf16 model cannot be resident with the Hessians and the GPTQ working set, so the
# layers are built one at a time. Qwen3.8-27B is 54 GB and loads whole; Qwen3.8-Flash-Next is 336 GB and cannot.
LAYERWISE_BYTES = 100 << 30


def hadamard_signs(seed=HADAMARD_SEED):
    s, out = seed, []
    for _ in range(BLOCK):
        s = (s + 0x9E37_79B9_7F4A_7C15) & M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & M64
        z ^= z >> 31
        out.append(-1.0 if z & 1 else 1.0)
    return out


SIGNS_MASK = 0x98913798D17E8D905C16E345695A0124


def rotation_matrix(device):
    """M with x~ = M x per 128-block, identical to malleus_gguf::quant::rotate_block."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < BLOCK:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = hadamard_signs()
    assert sum(1 << i for i, v in enumerate(signs) if v < 0) == SIGNS_MASK
    d = torch.tensor(signs, dtype=torch.float64)
    return (h * d[None, :] / math.sqrt(BLOCK)).float().to(device)


def rotate(x, m):
    """Rotates the last dimension of x (fp32) block by block."""
    sh = x.shape
    return (x.reshape(-1, sh[-1] // BLOCK, BLOCK) @ m.T).reshape(sh)


def unrotate(x, m):
    sh = x.shape
    return (x.reshape(-1, sh[-1] // BLOCK, BLOCK) @ m).reshape(sh)


def best_scale(w):
    """Per-row error-minimizing scale for groups w [..., N, 128], same search as quantize_group_int4, fp16-rounded."""
    amax = w.abs().amax(-1).clamp_min(1e-12)
    best_err = torch.full_like(amax, float("inf"))
    best_s = amax / 7.0
    for step in range(41):
        s = amax / (7.0 + step * 0.025)
        q = (w / s[..., None]).round().clamp(-8, 7)
        err = ((w - q * s[..., None]) ** 2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
    # fp16 cannot represent a scale below its smallest normal: without this clamp a group whose weights are tiny
    # gets scale 0 and the dequantizer reads the group as zeros.
    return best_s.half().clamp_min(torch.finfo(torch.float16).tiny).float()


def rtn(w):
    q = torch.empty_like(w)
    s = torch.empty(*w.shape[:-1], w.shape[-1] // BLOCK, device=w.device)
    for g in range(w.shape[-1] // BLOCK):
        sl = slice(g * BLOCK, (g + 1) * BLOCK)
        s[..., g] = best_scale(w[..., sl])
        q[..., sl] = (w[..., sl] / s[..., g:g + 1]).round().clamp(-8, 7)
    return q, s


def nearest2(w, s, levels):
    """Dequantized MLS2 weights and the code indices: a 4-level grid, nearest level per weight."""
    v = w / s[..., None]
    best = torch.full_like(v, levels[0])
    err = (v - levels[0]).abs()
    for level in levels[1:]:
        e = (v - level).abs()
        better = e < err
        best = torch.where(better, torch.full_like(v, level), best)
        err = torch.where(better, e, err)
    return best, best


def best_scale2(w, levels):
    """Per-row error-minimizing scale for groups w [..., N, 128] on a four-level grid, fp16-rounded."""
    amax = w.abs().amax(-1).clamp_min(1e-12)
    best_err = torch.full_like(amax, float("inf"))
    best_s = amax
    for step in range(81):
        s = amax / (1.0 + step * 0.025)
        q, _ = nearest2(w, s, levels)
        err = ((w - q * s[..., None]) ** 2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
    return best_s.half().clamp_min(torch.finfo(torch.float16).tiny).float()


def rtn2(w, levels):
    q = torch.empty_like(w)
    s = torch.empty(*w.shape[:-1], w.shape[-1] // BLOCK, device=w.device)
    for g in range(w.shape[-1] // BLOCK):
        sl = slice(g * BLOCK, (g + 1) * BLOCK)
        s[..., g] = best_scale2(w[..., sl], levels)
        q[..., sl], _ = nearest2(w[..., sl], s[..., g:g + 1], levels)
    return q, s


def gptq2(w, hinv, dead, levels):
    """GPTQ on a four-level MLS2 grid; same batched shape as `gptq`."""
    if w.dim() == 2:
        q, s = gptq2(w[None], hinv[None], dead[None], levels)
        return q[0], s[0]
    w = w.clone()
    b, n, k = w.shape
    w.masked_fill_(dead[:, None, :], 0.0)
    q = torch.zeros_like(w)
    s = torch.zeros(b, n, k // BLOCK, device=w.device)
    for g in range(k // BLOCK):
        i1, i2 = g * BLOCK, (g + 1) * BLOCK
        w1 = w[:, :, i1:i2].clone()
        e1 = torch.zeros_like(w1)
        hi = hinv[:, i1:i2, i1:i2]
        sc = best_scale2(w1, levels)
        s[:, :, g] = sc
        for i in range(BLOCK):
            col = w1[:, :, i]
            qc, _ = nearest2(col, sc, levels)
            q[:, :, i1 + i] = qc
            err = (col - qc * sc) / hi[:, i, i][:, None]
            w1[:, :, i:] -= err[:, :, None] * hi[:, i, i:][:, None, :]
            e1[:, :, i] = err
        w[:, :, i2:] -= e1 @ hinv[:, i1:i2, i2:]
    return q, s


def inverse_factor(h, damp):
    """Upper Cholesky factor of H^-1 (dampened), shared by the linears that read the same input."""
    dead = h.diag() == 0
    h = (h + h.T) / 2
    h[dead, dead] = 1.0
    mean = h.diag().mean()
    for _ in range(6):
        hd = h.clone()
        hd.diagonal().add_(damp * mean)
        l, info = torch.linalg.cholesky_ex(hd)
        if info == 0:
            break
        damp *= 4
    else:
        raise RuntimeError("H is not positive definite even with large damping")
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(l), upper=True)
    return hinv, dead


def gptq(w, hinv, dead):
    """GPTQ over the MLS4 grid; groups are the 128-wide blocks and the scale is chosen on the updated weights.
    Batched over experts when w is [B, N, K], hinv [B, K, K] and dead [B, K]."""
    if w.dim() == 2:
        q, s = gptq(w[None], hinv[None], dead[None])
        return q[0], s[0]
    w = w.clone()
    b, n, k = w.shape
    w.masked_fill_(dead[:, None, :], 0.0)
    q = torch.zeros_like(w)
    s = torch.zeros(b, n, k // BLOCK, device=w.device)
    for g in range(k // BLOCK):
        i1, i2 = g * BLOCK, (g + 1) * BLOCK
        w1 = w[:, :, i1:i2].clone()
        e1 = torch.zeros_like(w1)
        hi = hinv[:, i1:i2, i1:i2]
        sc = best_scale(w1)
        s[:, :, g] = sc
        for i in range(BLOCK):
            col = w1[:, :, i]
            qc = (col / sc).round().clamp(-8, 7)
            q[:, :, i1 + i] = qc
            err = (col - qc * sc) / hi[:, i, i][:, None]
            w1[:, :, i:] -= err[:, :, None] * hi[:, i, i:][:, None, :]
            e1[:, :, i] = err
        w[:, :, i2:] -= e1 @ hinv[:, i1:i2, i2:]
    return q, s


def proxy_loss(dw, h):
    return float(((dw @ h) * dw).sum())


def pack(q, s):
    qi = q.to(torch.int8).cpu()
    lo, hi = qi[..., 0::2] & 0xF, qi[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.uint8).contiguous(), s.half().cpu().contiguous()


def gutenberg_body(t):
    a, b = t.find("*** START OF"), t.find("*** END OF")
    if a >= 0:
        t = t[t.find("\n", a) + 1:]
    if b >= 0:
        t = t[:t.rfind("\n", 0, t.find("*** END OF"))]
    return t.replace("\r\n", "\n").strip()


def load_sources(corpus: Path, repo: Path):
    src = {"it": [], "en": [], "code": []}
    for gid, lang, sha in GUTENBERG:
        p = corpus / f"pg{gid}.txt"
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            print(f"warning: {p.name} sha256 differs from the manifest", flush=True)
        src[lang].append(gutenberg_body(data.decode("utf-8", "replace")))
    for p in sorted(repo.rglob("*")):
        if p.suffix in (".md",) and "golden" not in p.parts and p.is_file():
            src["it"].append(p.read_text(errors="replace"))
        elif p.suffix in (".rs", ".hip", ".py", ".sh", ".toml") and p.is_file() and "target" not in p.parts:
            src["code"].append(f"// {p.relative_to(repo)}\n" + p.read_text(errors="replace"))
    stdlib = Path(json.__file__).parents[1]
    for p in sorted(stdlib.glob("*.py")):
        src["code"].append(f"# {p.name}\n" + p.read_text(errors="replace"))
    return src


def build_samples(tok, src, n, length, seed):
    rng = random.Random(seed)
    mix = ["it"] * 3 + ["en"] * 3 + ["code"] * 4
    ids_by = {k: tok("\n\n".join(v)).input_ids for k, v in src.items()}
    samples = []
    for i in range(n):
        kind = mix[i % len(mix)]
        ids = ids_by[kind]
        if i % 2 == 0:
            off = rng.randrange(0, len(ids) - length)
            samples.append(ids[off:off + length])
            continue
        body_len = length
        off = rng.randrange(0, len(ids) - body_len)
        text = tok.decode(ids[off:off + body_len])
        cut = len(text) // 3
        msgs = [{"role": "user", "content": rng.choice(CHAT_PROMPTS[kind]) + "\n\n" + text[:cut]},
                {"role": "assistant", "content": text[cut:]}]
        chat = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        samples.append(tok(chat).input_ids[:length])
    return samples


class Stop(Exception):
    pass


@contextlib.contextmanager
def default_dtype(dt):
    """The transformers modules build their parameters at the default dtype, so bf16 has to be set while they are
    constructed; it must not leak, because the Hessians and the whole GPTQ solve are float32 by design."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class Whole:
    """Backend for a checkpoint whose bf16 weights fit on the GPU (Qwen3.8-27B): load it all and run the real
    forward, capturing the layer inputs with hooks."""

    def __init__(self, model_dir, dev):
        self.dev = dev
        self.model = load_on_gpu(model_dir)
        lm = self.model.model.language_model
        self.layers = lm.layers
        self.types = lm.config.layer_types

    def n_layers(self):
        return len(self.layers)

    def get(self, li):
        return self.layers[li]

    def free(self):
        """The whole model stays resident, so there is nothing to release between layers."""

    def capture(self, samples):
        """Runs the forward up to the first layer of every type and returns (hidden, kw): the input of layer 0 per
        sample, and the call arguments per sample and layer type. Those arguments depend only on the positions and
        the mask, so the ones seen at the first layer of a type are valid for every layer of that type."""
        first = {t: self.types.index(t) for t in set(self.types)}
        stop_at = max(first.values())
        hidden, kw = [], [{} for _ in samples]

        def hook_at(i):
            def hook(_m, a, k):
                if i == 0:
                    hidden.append(a[0].detach())
                for t, li in first.items():
                    if li == i:
                        kw[len(hidden) - 1][t] = {x: v for x, v in k.items() if x != "past_key_values"}
                if i == stop_at:
                    raise Stop
            return hook

        hs = [self.layers[i].register_forward_pre_hook(hook_at(i), with_kwargs=True) for i in range(stop_at + 1)]
        with torch.no_grad():
            for ids in samples:
                try:
                    self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
                except Stop:
                    pass
        for h in hs:
            h.remove()
        return hidden, kw


class Layerwise:
    """Backend for a checkpoint that does not fit (Qwen3.8-Flash-Next: 161 GB of experts alone against 124 GB of
    GTT, so load_on_gpu dies at 116.9 GB allocated). Builds one decoder layer at a time from the safetensors
    shards exactly like scripts/ref_layerwise.py, whose output is validated against the full-model golden, and
    frees it before building the next. The hashed n-gram table of a PLE layer is never loaded: the rows each
    sequence needs are read from the shards."""

    def __init__(self, model_dir, dev):
        from transformers import AutoConfig
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M

        self.M, self.dev = M, dev
        self.cfg = AutoConfig.from_pretrained(model_dir).text_config
        if not hasattr(self.M, "Qwen4ExpTextDecoderLayer"):
            raise NotImplementedError("layer-by-layer calibration supports qwen4_exp only")
        # Not sdpa: on torch 2.13.0+rocm7.14.0 the sdpa path of the indexed_attention layers dies with
        # hipErrorInvalidValue at `attn_output * torch.sigmoid(gate)`, while eager runs the same layer and
        # produces finite output. Only 12 of the 48 layers are indexed_attention, and eager costs a T x T
        # intermediate per head, which at 2048 tokens is ~1.5 GB transient -- worth it for a working run.
        self.cfg._attn_implementation = "eager"
        self.shards = Shards(Path(model_dir), dev)
        self.prefix = "model.language_model."
        self.types = self.cfg.layer_types
        self.cur = (-1, None)

    def n_layers(self):
        return self.cfg.num_hidden_layers

    def capture(self, samples):
        """The input of layer 0 is the token embedding repeated over the hyper-connection streams, so no decoder
        layer has to run to obtain it: only embed_tokens and the rotary embedding are loaded."""
        with default_dtype(torch.bfloat16), torch.no_grad():
            emb = self.shards.get(self.prefix + "embed_tokens.weight").to(self.dev)
            rot = self.M.Qwen4ExpTextRotaryEmbedding(self.cfg).to(self.dev)
            hidden, kw, types = [], [], set(self.types)
            for ids in samples:
                x = torch.tensor([ids], device=self.dev)
                h = emb[x]
                n = len(ids)
                pos = torch.arange(n, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
                call = {
                    "position_embeddings": rot(h, pos),
                    "attention_mask": torch.ones(n, n, dtype=torch.bool, device=self.dev).tril().view(1, 1, n, n),
                    "conv_mask": None,
                    "ple_input_ids": x,
                }
                hidden.append(h.repeat(1, 1, self.cfg.hc_count))
                kw.append({t: call for t in types})
            del emb, rot
        torch.cuda.empty_cache()
        return hidden, kw

    def get(self, li):
        if self.cur[0] == li:
            return self.cur[1]
        self.free()
        with default_dtype(torch.bfloat16):
            with torch.device("meta"):
                layer = self.M.Qwen4ExpTextDecoderLayer(self.cfg, li)
            lp = f"{self.prefix}layers.{li}."
            skip = ()
            if layer.ple is not None:
                ng = ShardedRows(self.shards, lp + "ple.ple_embedding.ngram_embedding.", self.dev)
                layer.ple.ple_embedding.ngram_embedding = ng
                skip = ("ngram_embedding",)
            layer = layer.to_empty(device=self.dev)
            load_module(layer, self.shards, lp, skip)
        layer.eval()
        self.cur = (li, layer)
        return layer

    def free(self):
        self.cur = (-1, None)
        torch.cuda.empty_cache()


class Hy4Layerwise:
    """Layer-by-layer backend for Hy4-flash, whose 411 GB checkpoint never fits.

    The layer, its Engram table and its non-persistent buffers are built exactly as scripts/ref_layerwise.py
    builds them, so the captured inputs are the ones `malleus golden` compares against.
    """

    def __init__(self, model_dir, dev, no_engram=False):
        from transformers import AutoConfig
        from transformers.models.hy_v4_flash import modeling_hy_v4_flash as H

        self.H, self.dev, self.no_engram = H, dev, no_engram
        self.cfg = AutoConfig.from_pretrained(model_dir).text_config
        self.cfg._attn_implementation = "eager"
        self.shards = Shards(Path(model_dir), dev)
        self.prefix = "model."
        self.types = self.cfg.layer_types
        self.cur = (-1, None)
        self.engram_layers = set(self.cfg.engram_layer_ids or ())

    def n_layers(self):
        return self.cfg.num_hidden_layers

    def capture(self, samples):
        """Input of layer 0 plus the per-type call arguments (masks, positions, n-grams), batched.

        Every calibration sample has the same length, so one batch is valid and makes the expert statistics loop
        once over the 512 experts instead of once per sample (which would take hours on 411 GB).
        """
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        cfg, dev = self.cfg, self.dev
        pad = cfg.engram_pad_id
        n = {len(s) for s in samples}
        assert len(n) == 1, f"batched capture needs equal lengths, got {sorted(n)}"
        with default_dtype(torch.bfloat16), torch.no_grad():
            emb = self.shards.get(self.prefix + "embed_tokens.weight").to(dev)
            rot = self.H.HYV4FlashRotaryEmbedding(cfg).to(dev).eval()
            x = torch.tensor(samples, device=dev)
            b, t = x.shape
            e = emb[x]
            pos = torch.arange(t, device=dev).view(1, 1, -1).expand(3, b, -1)
            pe = {lt: rot(e, position_ids=pos, layer_type=lt) for lt in rot.layer_types}
            mask_kwargs = {
                "config": cfg,
                "inputs_embeds": e,
                "attention_mask": None,
                "cache": None,
                "position_ids": pos[0],
                "allow_is_causal_skip": False,
            }
            masks = {
                "indexed_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
            shifted = [x]
            offset_of = torch.arange(t, device=dev).unsqueeze(0)
            for o in range(1, cfg.engram_max_ngram_size):
                prev = torch.nn.functional.pad(x, (o, 0), value=pad)[:, :t]
                shifted.append(torch.where(offset_of < o, torch.tensor(pad, device=dev), prev))
            kw = {
                k: {
                    "attention_mask": masks[k],
                    "position_ids": pos,
                    "cache": None,
                    "position_embeddings": pe[k],
                    "indexer_position_ids": pos,
                    "ngram_tokens": torch.stack(shifted, dim=-1),
                    "engram_mask": x != pad,
                }
                for k in set(self.types)
            }
            hidden = e.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
            del emb, rot
        torch.cuda.empty_cache()
        return hidden, kw

    def get(self, li):
        if self.cur[0] == li:
            return self.cur[1]
        self.free()
        H, cfg, dev = self.H, self.cfg, self.dev
        with default_dtype(torch.bfloat16):
            with torch.device("meta"):
                layer = H.HYV4FlashDecoderLayer(cfg, li)
            layer = layer.to_empty(device=dev)
            for name, p in layer.named_parameters():
                if any(k in name for k in KEEP_FP32):
                    p.data = p.data.float()
            prefix = f"{self.prefix}layers.{li}."
            skip = ()
            el = layer.engram_layer
            if el is not None:
                sizes = [s for order in cfg.engram_layer_head_vocab_sizes[str(li + 1)] for s in order]
                name = prefix + "engram_layer.multi_head_embedding.embedding.weight"
                el.multi_head_embedding = HostEngramEmbedding(
                    self.shards, name, sizes, cfg.engram_n_embed_per_ngram // cfg.engram_n_head_per_ngram, dev
                )
                skip = ("multi_head_embedding.embedding.weight",)
            load_module(layer, self.shards, prefix, skip, HY4_RENAME)
            if el is not None:
                # Non-persistent buffers are computed in `__init__`, which ran on the meta device: rebuild them
                # from the loaded map exactly as `_init_weights` does, or every Engram lookup reads a wrong row.
                el.head_vocab_sizes = torch.tensor(sizes, device=dev)
                el.multi_head_embedding.offsets = (
                    torch.tensor([0] + sizes[:-1], device=dev).cumsum(0).to(torch.long)
                )
                el.multi_head_embedding.head_vocab_sizes = el.head_vocab_sizes
                el.lookup_table = self.shards.get(prefix + "engram_layer.lookup_table").to(dev)
                el.hash_multipliers = el.draw_hash_multipliers(int(el.lookup_table.max()) + 1).to(dev)
        layer.eval()
        if self.no_engram:
            # The hashed table costs one random read per (token, head, n-gram): a 65k-token corpus touches ~1M
            # rows of a 31 GB tensor and takes tens of minutes per layer. The memory adds to the hidden state
            # but its own projections then stay round-to-nearest; the MLA and MoE still see real text.
            layer.engram_layer = None
        self.cur = (li, layer)
        return layer

    def free(self):
        self.cur = (-1, None)
        torch.cuda.empty_cache()

    def out_name(self, key):
        for module, checkpoint in HY4_OUT_NAME.items():
            if key.endswith(module):
                return key[: -len(module)] + checkpoint
        return key


# The families a layer is calibrated on, given its attention type and whether it carries routed experts.
def families_for(backend, li, layer, experts, args):
    if isinstance(backend, Hy4Layerwise):
        groups = () if args.only_routed_experts else (
            tuple(HY4_ATTN_FAMILIES)
            + (tuple(HY4_ENGRAM_FAMILIES) if layer.engram_layer is not None else ())
            + tuple(HY4_MOE_FAMILIES if experts is not None else HY4_DENSE_FAMILIES)
        )
        return groups
    return () if args.only_routed_experts else (
        MLS4_FAMILIES[backend.types[li]]
        + (HC_FAMILIES if args.hc else [])
        + (MOE_FAMILIES if experts is not None else MLP_FAMILIES)
    )



def engram_ngrams(samples, cfg, dev):
    """The shifted token windows the Engram hashes; the same thing Hy4Layerwise.capture builds."""
    x = torch.tensor(samples, device=dev)
    pad = cfg.engram_pad_id
    b, t = x.shape
    shifted = [x]
    offset_of = torch.arange(t, device=dev).unsqueeze(0)
    for o in range(1, cfg.engram_max_ngram_size):
        prev = torch.nn.functional.pad(x, (o, 0), value=pad)[:, :t]
        shifted.append(torch.where(offset_of < o, torch.tensor(pad, device=dev), prev))
    return torch.stack(shifted, dim=-1)


def calibrate_engram(args, backend, samples, out, rot):
    """GPTQ for the Engram projections, whose input is the hashed memory and not the transformer state.

    The value and the four key projections all read the same normalized rows, so one Hessian serves the five. The
    table is the expensive part: each token touches 16 rows per n-gram order, which is why this pass defaults to a
    small corpus and writes `engram_XX.safetensors` next to the layer files (the converter merges the directory).
    """
    from safetensors.torch import save_file

    cfg, dev = backend.cfg, backend.dev
    ngram = engram_ngrams(samples, cfg, dev)
    print(f"engram: {ngram.shape[0]} samples x {ngram.shape[1]} tokens", flush=True)
    report = []
    for li in sorted(x - 1 for x in backend.engram_layers):
        t0 = time.time()
        layer = backend.get(li)
        el = layer.engram_layer
        if el is None:
            print(f"layer {li} has no Engram", flush=True)
            backend.free()
            continue
        width = el.value_proj.weight.shape[1]
        with torch.no_grad():
            hash_ids = el.hash_ngrams(ngram)
            emb = el.multi_head_embedding(hash_ids).to(torch.bfloat16)
            # `hy4::engram` does not round-trip the rows through fp8 (that branch is disabled), so the Hessian is
            # accumulated on the same input the engine will see.
            normed = torch.stack([n(emb[:, :, i]) for i, n in enumerate(el.lookup_per_head_norms)], dim=2)
            flat = normed.flatten(start_dim=-2).reshape(-1, width).float()
        x = rotate(flat, rot)
        c = (16384.0 / (x * x).sum(0).max().clamp_min(1e-20)).sqrt()
        xh = (x * c).half()
        h = (xh.T @ xh).float() / (c * c)
        hinv, dead = inverse_factor(h, args.damp)
        tensors = {}
        for name in HY4_ENGRAM_FAMILIES[0]:
            lin = el.get_submodule(name.removeprefix("engram_layer."))
            w = lin.weight.float()
            wt = rotate(w, rot)
            base = proxy_loss(wt, h)
            qr, sr = rtn(wt)
            l_rtn = proxy_loss(wt - qr * sr.repeat_interleave(BLOCK, -1), h)
            qg, sg = gptq(wt, hinv, dead)
            deq = qg * sg.repeat_interleave(BLOCK, -1)
            l_gptq = proxy_loss(wt - deq, h)
            lin.weight.data.copy_(unrotate(deq, rot).reshape(lin.weight.shape).to(lin.weight.dtype))
            key = f"model.layers.{li}.{name}.weight"
            tensors[key], tensors[key + ".scales"] = pack(qg, sg)
            report.append({"tensor": key, "rtn": l_rtn / base, "gptq": l_gptq / base})
            print(
                f"  {li:2d} {name:28s} rtn {l_rtn / base:.3e} gptq {l_gptq / base:.3e} "
                f"({l_rtn / max(l_gptq, 1e-30):.2f}x)",
                flush=True,
            )
        save_file(tensors, str(out / f"engram_{li:02d}.safetensors"), metadata={"format": "mls4-gptq"})
        json.dump(
            {"samples": len(samples), "length": args.length, "damp": args.damp, "tensors": report},
            open(out / "engram_report.json", "w"),
            indent=1,
        )
        print(f"engram layer {li} done in {time.time() - t0:.0f}s", flush=True)
        layer = el = None
        backend.free()
        torch.cuda.empty_cache()

def checkpoint_bytes(model_dir):
    idx = Path(model_dir) / "model.safetensors.index.json"
    if idx.exists():
        names = set(json.loads(idx.read_text())["weight_map"].values())
    else:
        names = {p.name for p in Path(model_dir).glob("*.safetensors")}
    return sum((Path(model_dir) / n).stat().st_size for n in names)


class expert_stats:
    """Replaces the forward of a stacked-experts module to accumulate, per expert, H of the rotated gate_up inputs
    (tokens routed to it) and of the rotated down inputs; quantize() then runs batched GPTQ on all experts."""

    def __init__(self, mod, rot, mls2_levels=None):
        self.mod, self.rot = mod, rot
        self.mls2_levels = mls2_levels
        e, n2, k = mod.gate_up_proj.shape
        dev = mod.gate_up_proj.device
        self.h_gu = torch.zeros(e, k, k, device=dev)
        self.h_dn = torch.zeros(e, n2 // 2, n2 // 2, device=dev)
        self.count = torch.zeros(e, dtype=torch.long)
        self.accumulate = True
        self.orig = mod.forward
        mod.forward = self.forward

    def detach(self):
        self.mod.forward = self.orig

    def forward(self, hidden_states, top_k_index, top_k_weights):
        m = self.mod
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            tok_idx, pos = torch.where(top_k_index == e)
            x = hidden_states[tok_idx]
            gu = torch.nn.functional.linear(x, m.gate_up_proj[e])
            if hasattr(m, "_apply_gate"):  # Hy4 clamps its SwiGLU; the module owns the exact gate.
                a = m._apply_gate(gu)
            else:
                gate, up = gu.chunk(2, dim=-1)
                a = m.act_fn(gate) * up
            if self.accumulate:
                xr = rotate(x.float(), self.rot)
                self.h_gu[e] += xr.T @ xr
                ar = rotate(a.float(), self.rot)
                self.h_dn[e] += ar.T @ ar
                self.count[e] += x.shape[0]
            y = torch.nn.functional.linear(a, m.down_proj[e]) * top_k_weights[tok_idx, pos, None]
            out.index_add_(0, tok_idx, y.to(out.dtype))
        return out

    def quantize(self, damp, batch=64):
        """Yields (name, packed (nibbles, scales), losses) and writes the dequantized weights back."""
        m, rot = self.mod, self.rot
        rare = self.count < MIN_EXPERT_TOKENS
        for name, hm in (("gate_up_proj", "h_gu"), ("down_proj", "h_dn")):
            p = getattr(m, name)
            h = getattr(self, hm)
            h /= self.count.clamp_min(1).to(h.device)[:, None, None].float()
            n_exp = h.shape[0]
            loss = {"rtn": 0.0, "gptq": 0.0, "rtn_experts": int(rare.sum())}
            qs, ss = [], []
            for e0 in range(0, n_exp, batch):
                sl = slice(e0, min(e0 + batch, n_exp))
                hb = h[sl]
                hinv = torch.empty_like(hb)
                dead = torch.zeros(hb.shape[:2], dtype=torch.bool, device=h.device)
                for j in range(hb.shape[0]):
                    if rare[e0 + j]:
                        hinv[j] = torch.eye(hb.shape[1], device=h.device)
                    else:
                        hinv[j], dead[j] = inverse_factor(hb[j], damp)
                wt = rotate(p[sl].float(), rot)
                qg, sg = gptq(wt, hinv, dead)
                loss2 = None
                if self.mls2_levels is not None:
                    q2, s2 = rtn2(wt, self.mls2_levels)
                    lg2, sg2 = gptq2(wt, hinv, dead, self.mls2_levels)
                    loss2 = (q2, s2, lg2, sg2)
                # A scale that is not finite poisons the whole expert: the packed scales reach the converter as
                # they are, and a NaN there is a NaN in the model. Fall back to RTN for the experts it happens to
                # and say so, rather than writing a model nobody can load.
                ok = torch.isfinite(sg).all(-1).all(-1)
                if not bool(ok.all()):
                    qr0, sr0 = rtn(wt[~ok])
                    qg[~ok], sg[~ok] = qr0, sr0
                    print(f"    {name}: {int((~ok).sum())} experts with a non-finite GPTQ scale, kept RTN",
                          flush=True)
                del hinv
                qr, sr = rtn(wt)
                r = rare[sl].to(h.device)
                qg[r], sg[r] = qr[r], sr[r]
                deq = qg * sg.repeat_interleave(BLOCK, -1)
                if loss2 is not None:
                    q2, s2, lg2, sg2 = loss2
                    d2 = q2 * s2.repeat_interleave(BLOCK, -1)
                    dg2 = lg2 * sg2.repeat_interleave(BLOCK, -1)
                    loss["rtn2"] = loss.get("rtn2", 0.0)
                    loss["gptq2"] = loss.get("gptq2", 0.0)
                for j in range(hb.shape[0]):
                    base = max(proxy_loss(wt[j], hb[j]), 1e-30)
                    dr = wt[j] - qr[j] * sr[j].repeat_interleave(BLOCK, -1)
                    loss["rtn"] += proxy_loss(dr, hb[j]) / base / n_exp
                    loss["gptq"] += proxy_loss(wt[j] - deq[j], hb[j]) / base / n_exp
                    if loss2 is not None:
                        loss["rtn2"] += proxy_loss(wt[j] - d2[j], hb[j]) / base / n_exp
                        loss["gptq2"] += proxy_loss(wt[j] - dg2[j], hb[j]) / base / n_exp
                p.data[sl] = unrotate(deq, rot).reshape(deq.shape).to(p.dtype)
                tq, ts = pack(qg, sg)
                qs.append(tq)
                ss.append(ts)
                del wt, qg, sg, qr, sr, deq
            setattr(self, hm, None)
            del h
            yield name, (torch.cat(qs), torch.cat(ss)), loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="/golden/calib/texts")
    ap.add_argument("--out", default="/golden/calib/qwen38-27b")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (0 = all)")
    ap.add_argument("--start", type=int, default=0,
                    help="resume: propagate bf16 through this many layers, then calibrate from there")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--holdout", type=int, default=8, help="every n-th sample only measures the error (0 = none)")
    ap.add_argument("--layerwise", default="auto", choices=["auto", "yes", "no"],
                    help="build one decoder layer at a time instead of loading the whole model; auto does it when "
                         f"the checkpoint exceeds {LAYERWISE_BYTES >> 30} GB")
    ap.add_argument("--no-routed-experts", action="store_true",
                    help="calibrate the dense families and the shared expert only, and propagate the hidden "
                         "states through the bf16 routed experts")
    ap.add_argument("--only-routed-experts", action="store_true",
                    help="calibrate the routed experts only, leaving every other family as the converter ships "
                         "it (round-to-nearest): isolates the effect of the experts on the model's quality, "
                         "which is what a checkpoint without their int8 prefill copy pays for")
    ap.add_argument("--mls2-report", action="store_true",
                    help="also report the reconstruction error of a four-level MLS2 grid next to rtn/gptq; the "
                         "output stays MLS4, this is a screen and not a conversion")
    ap.add_argument("--mls2-grid", default="e2m1", choices=sorted(MLS2_GRIDS),
                    help="MLS2 grid for --mls2-report")
    ap.add_argument("--engram-only", action="store_true",
                    help="hy4: calibrate only the Engram projections; adds engram_XX.safetensors to --out")
    ap.add_argument("--no-engram", action="store_true",
                    help="hy4: skip the Engram memory during calibration (it stays round-to-nearest) instead of "
                         "reading ~1M random table rows per layer")
    ap.add_argument("--hc", action="store_true",
                    help="also calibrate the hyper-connection mixing (input_mix_weight_down/up, "
                         "block_inject_weight) of every layer, but not the final hyper_connection_mixer")
    args = ap.parse_args()
    if args.only_routed_experts and args.no_routed_experts:
        ap.error("--only-routed-experts and --no-routed-experts are mutually exclusive")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    from transformers import AutoTokenizer

    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    src = load_sources(Path(args.corpus), Path(__file__).resolve().parents[1])
    samples = build_samples(tok, src, args.samples, args.length, args.seed)
    print(f"{len(samples)} samples, {sum(map(len, samples))} tokens", flush=True)

    t0 = time.time()
    size = checkpoint_bytes(args.model)
    layerwise = args.layerwise == "yes" or (args.layerwise == "auto" and size > LAYERWISE_BYTES)
    family = json.loads((Path(args.model) / "config.json").read_text()).get("model_type", "")
    is_hy4 = family.startswith("hy_v4")
    if is_hy4 and not layerwise:
        ap.error("hy4-flash is 411 GB: it has to be calibrated layer by layer (--layerwise yes)")
    if args.engram_only and not is_hy4:
        ap.error("--engram-only is Hy4's Engram pass")
    backend = (
        Hy4Layerwise(args.model, dev, args.no_engram and not args.engram_only)
        if is_hy4
        else Layerwise(args.model, dev)
        if layerwise
        else Whole(args.model, dev)
    )
    print(f"{size >> 30} GB checkpoint, {'layer by layer' if layerwise else 'whole model'}", flush=True)
    batched = is_hy4  # one forward for the whole calibration set (see Hy4Layerwise.capture)
    types = backend.types
    rot = rotation_matrix(dev)
    mls2_levels = torch.tensor(MLS2_GRIDS[args.mls2_grid], device=dev)
    prefix = "model.layers" if is_hy4 else "model.language_model.layers"

    if args.engram_only:
        calibrate_engram(args, backend, samples, out, rot)
        return
    hidden, kw = backend.capture(samples)
    print(f"captured inputs in {time.time() - t0:.0f}s", flush=True)

    report = []
    n_layers = args.layers or backend.n_layers()
    if args.start:
        if not batched:
            ap.error("--start resumes a hy4 calibration only")
        # The prefix only has to produce the inputs of the first calibrated layer: run it with bf16 weights and
        # release each layer, so a resumed run does not keep the Hessians of the layers it already wrote.
        for li in range(args.start):
            layer = backend.get(li)
            with torch.no_grad():
                hidden = layer(hidden, **kw[types[li]])
            backend.free()
            print(f"propagated layer {li}", flush=True)
    for li in range(args.start, n_layers):
        tl = time.time()
        torch.cuda.empty_cache()
        layer = backend.get(li)
        experts = getattr(layer.mlp, "experts", None)
        # The 512 routed experts of a layer are already MLS4 in the shipped model, so GPTQ there buys quality but
        # no traffic, and it is ~95% of both the runtime and the output size. --no-routed-experts leaves them in
        # bf16 for the propagation and calibrates only the dense families and the shared expert.
        routed = None if args.no_routed_experts else experts
        groups = families_for(backend, li, layer, experts, args)
        if routed is not None:
            moe = expert_stats(routed, rot, mls2_levels if args.mls2_report else None)
        mods = {name: layer.get_submodule(name) for g in groups for name in g}
        # K rounded up to the 128-group grid: `input_mix_weight_up` is [10240, 320] and the converter stores it
        # padded to 384 columns, so the rotated grid and the Hessian have to agree with that. The padded columns
        # get a zero diagonal in H, which inverse_factor marks dead and gptq zeroes, so they stay zero.
        kpad = {g[0]: -(-mods[g[0]].weight.shape[1] // BLOCK) * BLOCK for g in groups}
        hmat = {(g[0], v): None for g in groups for v in (0, 1)}
        count = {k: 0 for k in hmat}
        cur = [0]

        def acc(name):
            def hook(_m, a):
                key = (name, int(args.holdout > 0 and cur[0] % args.holdout == args.holdout - 1))
                x = a[0].reshape(-1, a[0].shape[-1]).float()
                if (pad := kpad[name] - x.shape[1]):
                    x = torch.cat([x, x.new_zeros(x.shape[0], pad)], 1)
                x = rotate(x, rot)
                c = (16384.0 / (x * x).sum(0).max().clamp_min(1e-20)).sqrt()
                xh = (x * c).half()
                hh = (xh.T @ xh).float() / (c * c)
                hmat[key] = hh if hmat[key] is None else hmat[key] + hh
                count[key] += x.shape[0]
            return hook

        hooks = [mods[g[0]].register_forward_pre_hook(acc(g[0])) for g in groups]
        with torch.no_grad():
            if batched:
                layer(hidden, **kw[types[li]])
            else:
                for j, x in enumerate(hidden):
                    cur[0] = j
                    layer(x, **kw[j][types[li]])
        for h in hooks:
            h.remove()
        t_h = time.time() - tl

        with torch.no_grad():
            tensors = {}
            for g in groups:
                hh = hmat[(g[0], 0)] / count[(g[0], 0)]
                hv = hh if hmat[(g[0], 1)] is None else hmat[(g[0], 1)] / count[(g[0], 1)]
                hinv, dead = inverse_factor(hh, args.damp)
                del hmat[(g[0], 0)], hmat[(g[0], 1)], hh
                for name in g:
                    lin = mods[name]
                    k0 = lin.weight.shape[1]
                    w = lin.weight.float()
                    if k0 < kpad[g[0]]:
                        w = torch.cat([w, w.new_zeros(w.shape[0], kpad[g[0]] - k0)], 1)
                    wt = rotate(w, rot)
                    base = proxy_loss(wt, hv)
                    qr, sr = rtn(wt)
                    l_rtn = proxy_loss(wt - qr * sr.repeat_interleave(BLOCK, 1), hv)
                    qg, sg = gptq(wt, hinv, dead)
                    deq = qg * sg.repeat_interleave(BLOCK, 1)
                    l_gptq = proxy_loss(wt - deq, hv)
                    extra = {}
                    if args.mls2_report:
                        q2, s2 = rtn2(wt, mls2_levels)
                        l2 = proxy_loss(wt - q2 * s2.repeat_interleave(BLOCK, 1), hv)
                        qg2, sg2 = gptq2(wt, hinv, dead, mls2_levels)
                        lg2 = proxy_loss(wt - qg2 * sg2.repeat_interleave(BLOCK, 1), hv)
                        extra = {"rtn2": l2 / base, "gptq2": lg2 / base}
                    # Drop the padding again: the propagation runs the real module, whose K is unpadded.
                    lin.weight.copy_(unrotate(deq, rot)[:, :k0].to(lin.weight.dtype))
                    key = backend.out_name(f"{prefix}.{li}.{name}.weight")
                    tensors[key], tensors[key + ".scales"] = pack(qg, sg)
                    report.append({"tensor": key, "rtn": l_rtn / base, "gptq": l_gptq / base, **extra})
                    print(f"  {li:2d} {name:28s} rtn {l_rtn / base:.3e} gptq {l_gptq / base:.3e} "
                          f"({l_rtn / max(l_gptq, 1e-30):.2f}x)", flush=True)
                del hinv
                torch.cuda.empty_cache()
            if routed is not None:
                key = f"{prefix}.{li}.mlp.experts"
                report.append({"tensor": key, "tokens_per_expert": moe.count.tolist()})
                for name, (tq, ts), loss in moe.quantize(args.damp):
                    tensors[f"{key}.{name}"], tensors[f"{key}.{name}.scales"] = tq, ts
                    report.append({"tensor": f"{key}.{name}", **loss})
                    line = (f"  {li:2d} experts.{name:19s} rtn {loss['rtn']:.3e} gptq {loss['gptq']:.3e} "
                            f"({loss['rtn'] / max(loss['gptq'], 1e-30):.2f}x), {loss['rtn_experts']} experts RTN")
                    if "rtn2" in loss:
                        line += f" | rtn2 {loss['rtn2']:.3e} gptq2 {loss['gptq2']:.3e}"
                    print(line, flush=True)
                    torch.cuda.empty_cache()
                # The Hessians are done: the same fast loop now just propagates the dequantized weights, instead of
                # the module's one_hot over 513 experts on a 65k-token batch.
                moe.accumulate = False
            t_q = time.time() - tl - t_h
            from safetensors.torch import save_file
            save_file(tensors, str(out / f"layer_{li:02d}.safetensors"), metadata={"format": "mls4-gptq"})

            if batched:
                hidden = layer(hidden, **kw[types[li]])
            else:
                for j, x in enumerate(hidden):
                    hidden[j] = layer(x, **kw[j][types[li]])
            if routed is not None:
                moe.detach()
        meta = {"samples": len(samples), "length": args.length, "damp": args.damp, "holdout": args.holdout}
        json.dump({**meta, "tensors": report}, open(out / "report.json", "w"), indent=1)
        print(f"layer {li} done in {time.time() - tl:.0f}s (stats {t_h:.0f}s, gptq {t_q:.0f}s, "
              f"total {time.time() - t0:.0f}s)", flush=True)
        # Drop every reference to the layer before the next one is built. get() releases the previous layer, but
        # these locals would keep it alive through the build, and two Flash-Next layers are 20 GB of weights on
        # top of the Hessians; tensors holds ~2 GB of packed nibbles in host RAM.
        layer = experts = routed = mods = hooks = tensors = moe = None
        backend.free()


if __name__ == "__main__":
    main()
