#!/usr/bin/env python3
"""A tiny DeepSeek-V4.1 that runs on a CPU, and the activations of every stage of it.

The release is 510 GB and its reference implementation calls five GPU kernels, so neither runs here. This builds the
same graph at a scale that fits a laptop (4 layers, 8 experts, 256 hidden), with the two kernels the graph cannot do
without — sparse attention with its sink, and the Sinkhorn split of the hyper-connections — reimplemented in plain
torch, and with the quantization left as identity so that the comparison is about the graph and not about storage.

It writes:
  <out>/model.safetensors   the weights, under the release's own tensor names (bf16)
  <out>/config.json         the matching config.json
  <out>/manifest.json       the shapes of the dumps
  <out>/*.f32               per-stage activations: the embedding, every block, every attention and FFN output,
                            every indexer's top-k list, the final logits

`malleus deepseek-ref` compares a Malleus forward against these files.

    python3 scripts/deepseek_ref.py --reference /models/deepseekv41-flash/inference --out /golden/deepseek-ref/tiny
"""

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------
# The two kernels the graph needs, in torch.
# ----------------------------------------------------------------------------------------------------------------


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b, s, h, d], kv [b, n, d], topk_idxs [b, s, m] (-1 = empty slot).

    The sink is one extra score per head that goes into the denominator only; a row with no valid index comes out
    at zero (the training kernel's convention).
    """
    b, s, h, d = q.size()
    out = q.new_zeros(b, s, h, d)
    for bi in range(b):
        for i in range(s):
            idx = topk_idxs[bi, i]
            keep = idx >= 0
            if not keep.any():
                continue
            sel = kv[bi, idx[keep].long()]  # [t, d]
            scores = torch.einsum("hd,td->ht", q[bi, i].float(), sel.float()) * softmax_scale
            scores = torch.cat([scores, attn_sink.float().unsqueeze(-1)], dim=-1)
            p = torch.softmax(scores, dim=-1)[:, :-1]
            out[bi, i] = torch.einsum("ht,td->hd", p, sel.float()).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """pre / post / comb of the hyper-connections, `comb` made doubly stochastic."""
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb = comb.unflatten(-1, (hc_mult, hc_mult))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def install_kernels() -> types.ModuleType:
    """A `kernel` module with the same names as the release's, so `model.py` imports this one."""
    mod = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not inplace:
            raise NotImplementedError("the tiny reference runs without quantization")
        return x

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        if not inplace:
            raise NotImplementedError("the tiny reference runs without quantization")
        return x

    def fp8_gemm(*a, **k):
        raise NotImplementedError("the tiny reference runs without quantization")

    def fp4_gemm(*a, **k):
        raise NotImplementedError("the tiny reference runs without quantization")

    mod.act_quant = act_quant
    mod.fp4_act_quant = fp4_act_quant
    mod.fp8_gemm = fp8_gemm
    mod.fp4_gemm = fp4_gemm
    mod.sparse_attn = sparse_attn
    mod.hc_split_sinkhorn = hc_split_sinkhorn
    sys.modules["kernel"] = mod
    return mod


# ----------------------------------------------------------------------------------------------------------------
# The tiny model: all the structure, none of the size.
# ----------------------------------------------------------------------------------------------------------------

ARGS = dict(
    max_batch_size=1,
    max_seq_len=48,
    temperature=0,
    dtype="bf16",  # only decides the storage dtype; the run is float32
    expert_dtype=None,
    vocab_size=512,
    dim=256,
    moe_inter_dim=128,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=8,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    norm_topk_prob=True,
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=256,
    head_dim=128,
    rope_head_dim=32,
    norm_eps=1e-6,
    o_groups=4,
    o_lora_rank=128,
    window_size=8,
    # Two window-only layers, two that compress 2:1, and two at 1:1 where the last reads the candidate blocks of
    # the one before it: the same shape the release has, scaled down.
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 3, 4, 5),
    compress_rope_theta=40000.0,
    original_seq_len=0,
    rope_theta=10000.0,
    rope_factor=40,
    index_n_heads=2,
    index_head_dim=64,
    index_topk=4,
    candidate_source_layer=4,
    # Every block of a 16-token prompt: a cut mask leaves fewer candidates than `index_topk`, and the reference
    # then fills the list with positions it had masked out, whose pick is whatever torch's `topk` returns on ties.
    # The mask itself is verified (`cand4`) with a cut of 2 blocks by the same script, run with that value.
    candidate_topk_blocks=8,
    candidate_block_size=2,
    hc_mult=4,
    hc_sinkhorn_iters=3,
    hc_eps=1e-6,
    engram_layer_ids=(),
    engram_num_embeddings=(),
    vision_n_layers=0,
    image_token_id=0,
    dspark_block_size=0,
)


def init_weights(model: torch.nn.Module) -> None:
    """Deterministic, non-degenerate weights: `torch.empty` leaves the biases and the HC bases garbage otherwise."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                p.fill_(1.0)
            elif name.endswith("q_weight") or name.endswith("k_weight"):
                p.fill_(1.0)
            elif "hc_" in name and name.endswith("_scale"):
                p.normal_(0.0, 0.1)
            else:
                p.normal_(0.0, 0.05)


def export_safetensors(model, tokenizer_free_args, out: Path) -> dict[str, "torch.Tensor"]:
    """Writes the module's parameters under the release's names (the same names `deepseek::expected` lists)."""
    from safetensors.torch import save_file

    sd: dict[str, torch.Tensor] = {}
    t = model

    def add(name, tensor):
        sd[name] = tensor.detach().to(torch.bfloat16).contiguous()

    add("embed.weight", t.embed.weight)
    add("head.weight", t.head.weight)
    add("norm.weight", t.norm.weight)
    for i, layer in enumerate(t.layers):
        p = f"layers.{i}."
        add(p + "attn_norm.weight", layer.attn_norm.weight)
        add(p + "ffn_norm.weight", layer.ffn_norm.weight)
        for which in ("hc_attn", "hc_ffn"):
            add(p + f"{which}_fn", getattr(layer, f"{which}_fn"))
            add(p + f"{which}_base", getattr(layer, f"{which}_base"))
            add(p + f"{which}_scale", getattr(layer, f"{which}_scale"))
        a = layer.attn
        add(p + "attn.wq_a.weight", a.wq_a.weight)
        add(p + "attn.wq_b.weight", a.wq_b.weight)
        add(p + "attn.wkv.weight", a.wkv.weight)
        add(p + "attn.wo_a.weight", a.wo_a.weight)
        add(p + "attn.wo_b.weight", a.wo_b.weight)
        add(p + "attn.attn_sink", a.attn_sink)
        add(p + "attn.q_norm.weight", a.q_norm.weight)
        add(p + "attn.kv_norm.weight", a.kv_norm.weight)
        if a.compressor is not None:
            c = a.compressor
            add(p + "attn.compressor.norm.weight", c.norm.weight)
            add(p + "attn.compressor.wkv.weight", c.wkv.weight)
            if hasattr(c, "wgate"):
                add(p + "attn.compressor.wgate.weight", c.wgate.weight)
        if a.indexer is not None:
            ix = a.indexer
            add(p + "attn.indexer.wq_b.weight", ix.wq_b.weight)
            add(p + "attn.indexer.weights_proj.weight", ix.weights_proj.weight)
            if ix.owns_k:
                add(p + "attn.indexer.wk.weight", ix.wk.weight)
                add(p + "attn.indexer.k_norm.weight", ix.k_norm.weight)
        f = layer.ffn
        add(p + "ffn.gate.weight", f.gate.weight)
        add(p + "ffn.gate.bias", f.gate.bias)
        if f.gate.bias_vl is not None:
            add(p + "ffn.gate.bias_vl", f.gate.bias_vl)
        for part in ("w1", "w2", "w3"):
            add(p + f"ffn.shared_experts.{part}.weight", getattr(f.shared_experts, part).weight)
        for e, expert in enumerate(f.experts):
            if expert is None:
                continue
            for part in ("w1", "w2", "w3"):
                add(p + f"ffn.experts.{e}.{part}.weight", getattr(expert, part).weight)

    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"))
    cfg = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "image_token_id": 0,
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": ARGS["vocab_size"],
            "hidden_size": ARGS["dim"],
            "moe_intermediate_size": ARGS["moe_inter_dim"],
            "num_hidden_layers": ARGS["n_layers"],
            "num_attention_heads": ARGS["n_heads"],
            "num_key_value_heads": 1,
            "head_dim": ARGS["head_dim"],
            "qk_rope_head_dim": ARGS["rope_head_dim"],
            "q_lora_rank": ARGS["q_lora_rank"],
            "o_lora_rank": ARGS["o_lora_rank"],
            "o_groups": ARGS["o_groups"],
            "swiglu_limit": ARGS["swiglu_limit"],
            "rms_norm_eps": ARGS["norm_eps"],
            "rope_theta": ARGS["rope_theta"],
            "rope_scaling": {"rope_type": "linear", "factor": 1.0, "original_max_position_embeddings": 0},
            "compress_rope_theta": ARGS["compress_rope_theta"],
            "n_routed_experts": ARGS["n_routed_experts"],
            "n_shared_experts": 1,
            "num_experts_per_tok": ARGS["n_activated_experts"],
            "scoring_func": ARGS["score_func"],
            "topk_method": "noaux_tc",
            "norm_topk_prob": ARGS["norm_topk_prob"],
            "routed_scaling_factor": ARGS["route_scale"],
            "sliding_window": ARGS["window_size"],
            "compress_ratios": list(ARGS["compress_ratios"]),
            "kv_source_layer_ids": list(ARGS["kv_source_layers"]),
            "index_source_layer_ids": list(ARGS["index_source_layers"]),
            "index_n_heads": ARGS["index_n_heads"],
            "index_head_dim": ARGS["index_head_dim"],
            "index_topk": ARGS["index_topk"],
            "candidate_source_layer_id": ARGS["candidate_source_layer"],
            "candidate_topk_blocks": ARGS["candidate_topk_blocks"],
            "candidate_block_size": ARGS["candidate_block_size"],
            "hc_mult": ARGS["hc_mult"],
            "hc_sinkhorn_iters": ARGS["hc_sinkhorn_iters"],
            "hc_eps": ARGS["hc_eps"],
            "engram_layer_ids": [],
            "engram_num_embeddings": [],
            "num_nextn_predict_layers": 0,
            "eos_token_id": 1,
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path, help="the release's inference directory (model.py)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    install_kernels()
    sys.path.insert(0, str(a.reference))
    import model as ref

    torch.manual_seed(a.seed)
    torch.set_default_dtype(torch.float32)
    args = ref.ModelArgs(**ARGS)
    model = ref.Transformer(args)
    model = model.float()
    init_weights(model)
    model.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    sd = export_safetensors(model, args, a.out)
    # The engine reads the file this script just wrote, and that file is bf16: run the reference on the same
    # rounded values. Otherwise the comparison below measures the storage, not the graph — on a random model a
    # near-tie in the indexer then flips a slot and the logits land at 0.997 instead of at f32 noise.
    unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False).unexpected_keys
    assert not unexpected, unexpected

    # The prompt: a fixed sequence, so the Rust side reads the same token ids from the manifest.
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, ARGS["vocab_size"], (1, a.tokens), generator=g, dtype=torch.int64)

    dumps = {}

    def dump(name, tensor):
        dumps[name] = tensor.detach().float().reshape(-1).contiguous()

    def hook(name, pick=lambda out: out):
        def f(_module, _inputs, output):
            dump(name, pick(output))

        return f

    final = {}
    handles = [
        model.embed.register_forward_hook(hook("embed")),
        model.norm.register_forward_hook(lambda _m, _i, out: final.setdefault("h", out.detach().float())),
    ]
    def attn_hook(i):
        def f(_m, _inp, _out):
            from model import shared_attn as sa
            ratio = ARGS["compress_ratios"][i]
            if ratio and sa.compress_kv is not None:
                dump(f"ckv{i}", sa.compress_kv[0, : ids.size(1) // ratio])
            if sa.index_k is not None and i in ARGS["kv_source_layers"]:
                dump(f"ixk{i}", sa.index_k[0, : ids.size(1) // max(ratio, 1)])
            if sa.topk_idxs is not None:
                dump(f"topk{i}", sa.topk_idxs.float())
            if i == ARGS["candidate_source_layer"] and sa.candidates is not None:
                dump(f"cand{i}", sa.candidates.float())

        return f

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(hook(f"block{i}", lambda out: out[0])))
        handles.append(layer.attn.register_forward_hook(hook(f"attn{i}")))
        handles.append(layer.attn.register_forward_hook(attn_hook(i)))
        handles.append(layer.ffn.register_forward_hook(hook(f"ffn{i}")))
    tops = {}
    with torch.inference_mode():
        output_ids, logits, _ = model(ids, 0)
        logits_all = model.head(final["h"], full_logits=True)
        # `shared_attn` keeps one slot per kind, so it has to be read after each layer rather than through a hook.
        tops = {"topk": model_topk_after(model)}
    for h in handles:
        h.remove()
    dump("logits", logits)
    dump("logits_all", logits_all)
    dump("ids", ids)
    dump("topk", tops["topk"].float())

    for name, tensor in dumps.items():
        tensor.numpy().astype("<f4").tofile(str(a.out / f"{name}.f32"))
    manifest = {
        "tokens": ids.tolist(),
        "vocab": ARGS["vocab_size"],
        "hidden": ARGS["dim"],
        "layers": ARGS["n_layers"],
        "window": ARGS["window_size"],
        "shapes": {k: list(v.shape) for k, v in dumps.items()},
        "dumps": sorted(dumps),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest["shapes"][k] for k in ("logits",)}, indent=0))


def model_topk_after(model):
    """The indexer list the last source published, for the layers that reuse it (shape and ids)."""
    from model import shared_attn

    return shared_attn.topk_idxs


if __name__ == "__main__":
    main()
