#!/usr/bin/env python3
"""Focused replay of the gated MLA of one Hy4-flash layer against transformers.

The absorbed-MLA port of issue #40 re-derives the reference's key/value expansion by hand (`qt` folds the QK half
of `kv_b_proj` into the query, V is `kv_b_v @ merged latent`). The reference's own V differs from that formula by
~15 % somewhere, so this tool replays one `HYV4FlashAttention` step by step — exactly the calls the module's
`forward` makes — and dumps every stage in the layout the engine's `MALLEUS_HY4_TRACE` dumps use, so the two can
be compared element by element.

The stepwise trace is asserted equal to the module's own `self_attn(...)` call: if this tool and the reference
disagree, it is this tool that is wrong, not the engine.

Output (per layer): `--out/<layer>/q.f32`, `k_pass.f32`, `k_rot.f32`, `qt.f32`, `value.f32`, `probs.f32`,
`latmerged.f32`, `attn_raw.f32`, `gate.f32`, `attn_gated.f32`, all [T, heads * dim] f32 row-major like the
engine's dumps, plus `sink.txt` with the learned sink logits.

Run in the golden container: `scripts/golden.sh scripts/ref_mla_replay.py --model /models/hy4-flash --layer 0`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (imported for parity with the other reference tools)
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ref_layerwise import HY4_RENAME, KEEP_FP32, Shards, load_module  # noqa: E402


def save_f32(path: Path, t: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    t.detach().float().cpu().contiguous().numpy().tofile(path)


def hy4_attn_trace(a, x, mask, pe, blobs: Path, tag: str):
    """The gated MLA step by step, mirroring `HYV4FlashAttention.forward` (issue #40 handoff, step 3)."""
    import torch.nn.functional as F
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import apply_rotary_pos_emb

    B, T, _ = x.shape
    nh, nope, rope = a.num_heads, a.qk_nope_head_dim, a.qk_rope_head_dim
    v, kv = a.v_head_dim, a.kv_lora_rank
    qk = nope + rope

    def save(name, t):
        save_f32(blobs / f"{tag}_{name}.f32", t)

    with torch.no_grad():
        gate_states = a.gate_proj(x).view(B, T, nh, v)
        q_resid = a.q_a_layernorm(a.q_a_proj(x))
        q_states = a.q_b_proj(q_resid).view(B, T, nh, qk).transpose(1, 2)  # [B, nh, T, qk]
        q_pass, q_rot = torch.split(q_states, [nope, rope], dim=-1)
        q_rot_raw = q_rot.clone()  # the query's rope dims before the rotary (what the engine dumps)
        compressed_kv = a.kv_a_proj_with_mqa(x)
        kv_pass, k_rot = torch.split(compressed_kv, [kv, rope], dim=-1)
        k_pass = a.kv_a_layernorm(kv_pass).view(B, 1, T, kv)
        k_rot = k_rot.view(B, 1, T, rope)
        cos, sin = pe
        print("shapes: cos", tuple(cos.shape), "sin", tuple(sin.shape), "q_rot_in", tuple(q_rot.shape), flush=True)
        import os as _os
        if _os.environ.get("DUMP_ROT"):
            save_f32(blobs / f"{tag}_cos.f32", cos)
            save_f32(blobs / f"{tag}_sin.f32", sin)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        # The engine dumps its query before the rotation: keep that stage too, to compare like with like.
        q_unrot = torch.cat((q_pass, q_rot_raw), dim=-1)  # [B, nh, T, qk]
        save("q_unrot", q_unrot.transpose(1, 2).reshape(B, T, nh * qk))
        query_states = torch.cat((q_pass, q_rot), dim=-1)  # [B, nh, T, qk]
        save("q", query_states.transpose(1, 2).reshape(B, T, nh * qk))
        save("kpass", k_pass.expand(B, nh, T, kv).transpose(1, 2).reshape(B, T, nh * kv))
        save("krot", k_rot.expand(B, nh, T, rope).transpose(1, 2).reshape(B, T, nh * rope))

        # expand_kv: kv_b_proj maps the latent to [heads * (nope + v)], split into k_nope and the value.
        kv_flat = a.kv_b_proj(k_pass).view(B, T, nh, nope + v).transpose(1, 2)  # [B, nh, T, nope+v]
        k_nope, value_states = torch.split(kv_flat, [nope, v], dim=-1)
        save("value", value_states.transpose(1, 2).reshape(B, T, nh * v))
        key_states = torch.cat((k_nope, k_rot.expand(-1, nh, -1, -1)), dim=-1)

        # The absorbed query of hy4_absorb: q_pass folded against the QK rows of kv_b_proj.
        wq = a.kv_b_proj.weight.view(nh, nope + v, kv)[:, :nope, :]  # [nh, nope, kv]
        qt = torch.einsum("bhtd,hdg->bhtg", q_pass, wq)  # [B, nh, T, kv]
        save("qt", qt.transpose(1, 2).reshape(B, T, nh * kv))

        # Eager attention with the learned sink, on the reference's own numbers (causal mask). The sink is an fp32
        # parameter, so the concatenated logits run through softmax in fp32; the module then casts the probabilities
        # back to the value dtype (`dropout(..., training=False).to(value_states.dtype)`, p=0 is the identity).
        print("dtypes: q", query_states.dtype, "k", key_states.dtype, "v", value_states.dtype,
              "mask", mask.dtype, "sinks", a.sinks.dtype, flush=True)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * a.scaling
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        sinks = a.sinks.reshape(1, -1, 1, 1).expand(B, nh, T, 1)
        scores_s = torch.cat((scores, sinks), dim=-1)
        print("dtypes after cat:", scores_s.dtype, flush=True)
        scores_s = scores_s - scores_s.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores_s, dim=-1, dtype=scores_s.dtype)[..., :-1]
        p = probs.to(x.dtype)  # the module's dropout(p=0).to(value_states.dtype)
        save("probs", probs.transpose(1, 2).reshape(B, T, nh * T))

        # The weighted latent, exactly what the absorbed path's split+merge should produce.
        latmerged = torch.matmul(p, k_pass.expand(B, nh, T, kv).to(x.dtype))  # [B, nh, T, kv]
        save("latmerged", latmerged.transpose(1, 2).reshape(B, T, nh * kv))

        attn = torch.matmul(p, value_states)  # [B, nh, T, v]
        save("attn_raw", attn.transpose(1, 2).reshape(B, T, nh * v))
        attn = attn * torch.sigmoid(gate_states.transpose(1, 2))
        # gate_states is [B, T, nh, V] (not [B, nh, T, V] like the query space): no transpose, or the file is
        # head-major and every comparison reads the wrong (token, head) pair.
        save("gate", gate_states.reshape(B, T, nh * v))
        save("attn_gated", attn.transpose(1, 2).reshape(B, T, nh * v))
        attn = attn.transpose(1, 2).reshape(B, T, -1).contiguous()
        return a.post_proj_norm(a.o_proj(attn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/hy4-flash", help="checkpoint dir (safetensors shards)")
    ap.add_argument("--layer", type=int, default=0, help="0-based layer whose attention to replay")
    ap.add_argument("--ids", type=int, nargs="+", default=[3462, 341, 269, 8778, 299, 12749, 30, 6144, 297, 936, 4962, 13])
    ap.add_argument("--out", default="/golden/hy4-flash/mla", help="blob dir for the dumps")
    args = ap.parse_args()

    sys.path.insert(0, "/src/scripts")
    sys.path.insert(0, str(Path(__file__).parent))
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as H
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash

    cfg = AutoConfig.from_pretrained(args.model).text_config
    cfg._attn_implementation = "eager"
    dev = "cuda"
    torch.set_default_dtype(torch.bfloat16)
    shards = Shards(Path(args.model), dev)

    li = args.layer
    x = torch.tensor([args.ids], device=dev)
    T = len(args.ids)
    emb = shards.get("model.embed_tokens.weight").to(dev)
    e = emb[x]
    h = e.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    pos = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
    with torch.no_grad():
        rot = H.HYV4FlashRotaryEmbedding(cfg).to(dev).eval()
        pe = {lt: rot(e, position_ids=pos, layer_type=lt) for lt in rot.layer_types}
        mk = {"config": cfg, "inputs_embeds": e, "attention_mask": None, "cache": None,
              "position_ids": pos[0], "allow_is_causal_skip": False}
        masks = {
            "indexed_attention": create_causal_mask(**mk),
            "sliding_attention": create_sliding_window_causal_mask(**mk),
        }

    with torch.device("meta"):
        layer = H.HYV4FlashDecoderLayer(cfg, li)
    layer = layer.to_empty(device=dev)
    for name, p in layer.named_parameters():
        if any(k in name for k in KEEP_FP32):
            p.data = p.data.float()
    prefix = f"model.layers.{li}."
    load_module(layer, shards, prefix, (), HY4_RENAME)
    layer.eval()

    # The traced attention must equal the module's own call: build the layer input like the layer does.
    with torch.no_grad():
        # iHC pre-mix on the streams, as the module forward does (modular_hy_v4_flash.py, step 1).
        flat = h.flatten(2).float()
        mixes = torch.nn.functional.linear(flat, layer.attn_hc.fn.float()) * layer.attn_hc.input_norm(flat)
        pre_b, post_b = layer.attn_hc.base.split(layer.attn_hc.hc_mult, dim=-1)
        pre_s, post_s = layer.attn_hc.scale.unbind(0)
        pre_l, post_l = mixes.split(layer.attn_hc.hc_mult, dim=-1)
        pre = torch.sigmoid(pre_l * pre_s + pre_b) + layer.attn_hc.hc_eps
        post = layer.attn_hc.hc_post_magnitude * torch.sigmoid(post_l * post_s + post_b) + layer.attn_hc.hc_eps
        x_attn = torch.sum(pre.unsqueeze(-1) * h, dim=2).to(h.dtype)
        hn = layer.input_layernorm(x_attn)
        save_f32(Path(args.out) / f"T{li}_in_norm.f32", hn[0])

        at = layer.attention_type
        out_t = hy4_attn_trace(layer.self_attn, hn, masks[at], pe[at], Path(args.out), f"T{li}")
        ref = layer.self_attn(
            hidden_states=hn,
            attention_mask=masks[at],
            position_embeddings=pe[at],
            position_ids=pos,
            indexer_position_ids=pos,
        )
        if not torch.equal(out_t, ref[0]):
            d = (out_t.float() - ref[0].float()).abs().max()
            raise SystemExit(
                f"traced attention diverges from the module's own call (max abs {float(d):.6g}, "
                f"dtypes {out_t[0].dtype} vs {ref[0].dtype}, shapes {tuple(out_t.shape)} vs {tuple(ref[0].shape)})"
            )

    print(f"layer {li} ({at}): attention trace == module forward, T={T} heads={layer.self_attn.num_heads} "
          f"nope={layer.self_attn.qk_nope_head_dim} rope={layer.self_attn.qk_rope_head_dim} "
          f"v={layer.self_attn.v_head_dim} kv={layer.self_attn.kv_lora_rank} sink={layer.self_attn.sinks.tolist()}")
    print("dumps ->", Path(args.out) / f"T{li}_*.f32")


if __name__ == "__main__":
    main()
