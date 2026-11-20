#!/usr/bin/env python3
"""Layer-by-layer bf16 reference for models that do not fit in memory (offline tool, never shipped).

Builds one decoder layer at a time from the transformers modules of the model family, loads its tensors from the
safetensors shards, runs every sequence through it (teacher forcing, no cache) and frees it. The hashed n-gram
table of the PLE layer is never loaded: the rows each sequence needs are read from the shards.

Writes golden/<name>/golden.json (input ids, argmax at every position, top-10 of the last position) in the
format of scripts/golden.py, the last-position logits to --blob-dir/text/<id>/logits_last.f32 and, for --dump,
the hyper-connection state after every layer to --blob-dir/layers/<id>/L<i>.f32 ([T, hc * hidden] f32).

Supported: qwen4_exp (Qwen3.8-Flash-Next) and hy_v4_flash (Hy4-flash, text path: no image or video input yet).
Run in an image with torch (ROCm) and transformers >= 5.17 (hy4 needs the `models/hy_v4_flash` of #50903).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from golden import RAW_TEXT, TEXT_PROMPTS  # noqa: E402


def save_f32(path: Path, t: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    t.detach().float().cpu().contiguous().numpy().tofile(path)


class Shards:
    """Tensor name -> shard file. Files are opened per read: open handles keep every page read mapped."""

    def __init__(self, model_dir: Path, device: str):
        idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.where = idx
        self.dir = model_dir
        self.device = device

    def _open(self, name: str):
        from safetensors import safe_open

        return safe_open(str(self.dir / self.where[name]), framework="pt", device="cpu")

    def get(self, name: str) -> torch.Tensor:
        with self._open(name) as f:
            return f.get_tensor(name).clone()

    def shape(self, name: str):
        with self._open(name) as f:
            return f.get_slice(name).get_shape()

    def rows(self, name: str, idx: list[int]) -> torch.Tensor:
        with self._open(name) as f:
            s = f.get_slice(name)
            return torch.cat([s[i : i + 1].clone() for i in idx]) if idx else None


class ShardedRows(nn.Module):
    """Stand-in for the PLE n-gram nn.Embedding: reads only the requested rows from the checkpoint shards."""

    def __init__(self, shards: Shards, prefix: str, device: str):
        super().__init__()
        names = sorted(
            (n for n in shards.where if n.startswith(prefix + "shard_")),
            key=lambda n: int(n[len(prefix) + len("shard_") :].split(".")[0]),
        )
        self.shards, self.names, self.starts = shards, names, []
        o = 0
        for n in names:
            self.starts.append(o)
            o += shards.shape(n)[0]
        self.total = o
        self.weight = torch.empty(0, device=device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.flatten().cpu()
        uniq, inv = torch.unique(flat, return_inverse=True)
        groups = {}
        si = 0
        for u in uniq.tolist():
            while si + 1 < len(self.starts) and self.starts[si + 1] <= u:
                si += 1
            groups.setdefault(si, []).append(u - self.starts[si])
        out = [self.shards.rows(self.names[si], rows) for si, rows in sorted(groups.items())]
        table = torch.cat(out).to(self.weight.device)
        return table[inv.to(self.weight.device)].view(*ids.shape, -1)


def load_module(mod: nn.Module, shards: Shards, prefix: str, skip=(), rename=None):
    """Copies checkpoint tensors `prefix + <param name>` into the module (parameters and persistent buffers).

    `rename` maps a state-dict key to the checkpoint's spelling where the two disagree (`sinks` against
    `learnable_sink_param`, the iHC bases against `hc_pre.hc_*`).
    """
    sd = mod.state_dict()
    for k in sd:
        if any(s in k for s in skip):
            continue
        t = shards.get(prefix + (rename or {}).get(k, k))
        if t.shape != sd[k].shape:
            raise ValueError(f"{prefix + k}: checkpoint {tuple(t.shape)}, module {tuple(sd[k].shape)}")
        with torch.no_grad():
            sd[k].copy_(t)


def sequences(tok, args, chat=True):
    seqs = []
    keep = set(args.only.split(",")) if args.only else None
    for pid, text in TEXT_PROMPTS:
        if text is None or not chat:
            ids = tok(RAW_TEXT if text is None else text).input_ids
        else:
            msgs = [{"role": "user", "content": text}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
        seqs.append((pid, list(ids)))
    if args.long:
        import transformers

        src = Path(transformers.__file__).parent / "models" / "qwen4_exp" / "modeling_qwen4_exp.py"
        ids = tok(src.read_text()).input_ids[: args.long]
        seqs.append((f"long-{args.long}", ids))
    return [s for s in seqs if keep is None or s[0] in keep]


class HostEngramEmbedding(nn.Module):
    """Stand-in for the 32 GB Engram table: reads only the rows a sequence asks for from the shards."""

    def __init__(self, shards: Shards, name: str, head_vocab_sizes: list[int], dim: int, device: str):
        super().__init__()
        self.shards, self.name, self.dim = shards, name, dim
        self.head_vocab_sizes = head_vocab_sizes
        self.offsets = torch.tensor([0] + list(head_vocab_sizes[:-1])).cumsum(0).to(device)

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        # Each hash head has its own slice of the shared table: the offsets are what makes the ids global.
        flat = (hash_ids.clamp_min(0).cpu() + self.offsets.cpu()).flatten()
        # One read per distinct row: a prompt of a few hundred tokens touches at most that many.
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = self.shards.rows(self.name, uniq.tolist()).to(hash_ids.device).to(torch.bfloat16)
        out = rows[inv.to(hash_ids.device)]
        return out.view(*hash_ids.shape, self.dim).masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)


# The parameters the checkpoint keeps in fp32 (see `_keep_in_fp32_modules_strict`): loading them into the bf16
# module would round them twice.
KEEP_FP32 = (
    "e_score_correction_bias",
    "attn_hc.fn",
    "attn_hc.scale",
    "attn_hc.base",
    "ffn_hc.fn",
    "ffn_hc.scale",
    "ffn_hc.base",
    "sinks",
    "key_value_projection",
    "key_gate_projection",
    "intra_group_ape",
)

# State-dict key -> the checkpoint's spelling of the same tensor.
HY4_RENAME = {
    "attn_hc.fn": "hc_attn_layer.hc_pre.hc_fn",
    "attn_hc.base": "hc_attn_layer.hc_pre.hc_base",
    "attn_hc.scale": "hc_attn_layer.hc_pre.hc_scale",
    "ffn_hc.fn": "hc_mlp_layer.hc_pre.hc_fn",
    "ffn_hc.base": "hc_mlp_layer.hc_pre.hc_base",
    "ffn_hc.scale": "hc_mlp_layer.hc_pre.hc_scale",
    "self_attn.sinks": "self_attn.learnable_sink_param",
    "self_attn.gate_proj.weight": "self_attn.linear_gate.weight",
    "mlp.pre_expert_proj.weight": "mlp.routed_expert_down_proj.weight",
    "mlp.post_expert_proj.weight": "mlp.routed_expert_up_proj.weight",
    # The dense layer's sandwich norm ships under the MLP's old name (`sandwich_norm` in the checkpoint).
    "post_mlp_layernorm.weight": "mlp.post_fc2_norm.weight",
}


def hy4_trace(layer, h, s, blobs: Path, tag: str, ngram, engram_mask):
    """The layer's own forward, step by step, so that the engine can be compared at every boundary.

    The steps are the ones in `HYV4FlashDecoderLayer.forward`; the result is checked against the module's own
    forward, so this cannot drift from what is being reproduced.
    """
    import torch.nn.functional as F

    def save(name, t):
        save_f32(blobs / f"{tag}_{name}.f32", t[0])

    dtype = h.dtype
    if layer.engram_layer is not None and ngram is not None:
        # The Engram steps, dumped: hashes, the rows after the fp8 round trip and the per-head norms, the memory
        # value, the per-stream gates and the result. The final line asserts the steps equal the module's forward.
        el = layer.engram_layer
        hash_ids = el.hash_ngrams(ngram)
        emb = el.multi_head_embedding(hash_ids).to(h.dtype)
        if el.fp8_embedding:
            import transformers.models.hy_v4_flash.modeling_hy_v4_flash as _M

            emb = _M._fp8_round_trip(emb)
        save("e_hash", hash_ids.to(torch.float32))
        save("e_emb_fp8", emb)
        normed = torch.stack([n(emb[:, :, i]) for i, n in enumerate(el.lookup_per_head_norms)], dim=2)
        flat = normed.flatten(start_dim=-2)
        save("e_emb_norm", flat)
        main = el.unified_pre_gamma * el.unified_rmsnorm(el.value_proj(flat))
        save("e_main", main)
        gates = []
        for c in range(el.hc_mult):
            key = el.norm_key[c](el.key_projs[c](flat))
            query = el.norm_query[c](h[:, :, c])
            gate = (key * query).sum(dim=-1) * el.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gs = torch.stack(gates, dim=2)
        save("e_gate", gs)
        e = gs.unsqueeze(-1) * main.unsqueeze(2) + el.unified_per_hc_scales(main)
        save("engram_out", e.to(torch.float32))
        h = h + e
    save("layer_in", h)
    residual = h
    flat = h.flatten(2).float()
    mixes = F.linear(flat, layer.attn_hc.fn.float()) * layer.attn_hc.input_norm(flat)
    pre_b, post_b = layer.attn_hc.base.split(layer.attn_hc.hc_mult, dim=-1)
    pre_s, post_s = layer.attn_hc.scale.unbind(0)
    pre_l, post_l = mixes.split(layer.attn_hc.hc_mult, dim=-1)
    pre = torch.sigmoid(pre_l * pre_s + pre_b) + layer.attn_hc.hc_eps
    post = layer.attn_hc.hc_post_magnitude * torch.sigmoid(post_l * post_s + post_b) + layer.attn_hc.hc_eps
    x = torch.sum(pre.unsqueeze(-1) * h, dim=2).to(dtype)
    save("ihc_pre", pre.to(dtype))
    save("ihc_post", post.to(dtype))
    save("ihc_out", x)
    hn = layer.input_layernorm(x)
    save("in_norm", hn)
    a, _ = layer.self_attn(
        hidden_states=hn,
        attention_mask=s["masks"][layer.attention_type],
        position_embeddings=s["pe"][layer.attention_type],
        position_ids=s["pos"],
        indexer_position_ids=s["pos"],
    )
    save("attn", a)
    x = (post.float().unsqueeze(-1) * a.float().unsqueeze(-2) + residual.float()).to(dtype)
    save("streams_attn", x)
    residual = x
    flat = x.flatten(2).float()
    mixes = F.linear(flat, layer.ffn_hc.fn.float()) * layer.ffn_hc.input_norm(flat)
    pre_b, post_b = layer.ffn_hc.base.split(layer.ffn_hc.hc_mult, dim=-1)
    pre_s, post_s = layer.ffn_hc.scale.unbind(0)
    pre_l, post_l = mixes.split(layer.ffn_hc.hc_mult, dim=-1)
    pre = torch.sigmoid(pre_l * pre_s + pre_b) + layer.ffn_hc.hc_eps
    post = layer.ffn_hc.hc_post_magnitude * torch.sigmoid(post_l * post_s + post_b) + layer.ffn_hc.hc_eps
    x = torch.sum(pre.unsqueeze(-1) * residual, dim=2).to(dtype)
    save("ffn_pre", pre.to(dtype))
    save("ffn_post", post.to(dtype))
    hf = layer.post_attention_layernorm(x)
    save("ffn_in", hf)
    m = layer.post_mlp_layernorm(layer.mlp(hf))
    save("ffn_out", m)
    out = (post.float().unsqueeze(-1) * m.float().unsqueeze(-2) + residual.float()).to(dtype)
    return out


VISION_RENAME = {
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
    "mlp.fc1": "mlp.dense_h_to_4h",
    "mlp.fc2": "mlp.dense_4h_to_h",
    "patch_merger.before_rms": "perceive.norm",
    "patch_merger.proj_conv": "perceive.proj.0",
    "patch_merger.proj_out": "perceive.proj.2",
    "patch_merger.mlp": "perceive.mlp",
    "patch_merger.image_newline": "perceive.image_newline",
}


def hy4_vision(args, cfg, model_dir, src_dev):
    """Hy4's vision tower on one image: the golden the engine's vision path is compared against."""
    import torch
    from transformers import AutoProcessor
    from transformers.models.hy_v4_vl import modeling_hy_v4_vl as V

    img = Path(args.vision)
    from transformers import AutoConfig

    vcfg = AutoConfig.from_pretrained(args.model).vision_config
    vcfg._attn_implementation = "eager"
    proc = AutoProcessor.from_pretrained(args.model)
    from PIL import Image

    im = Image.open(img).convert("RGB")
    # A small image: the tower is what is being checked here, not the anyres policy.
    im.thumbnail((512, 512))
    inputs = proc(images=im, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(src_dev).to(torch.bfloat16)
    grid = inputs["image_grid_thw"].to(src_dev)
    print("image", img.name, "pixel_values", tuple(pixel_values.shape), "grid", grid.tolist(), flush=True)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        tower = V.HYV4VLVisionModel(vcfg)
    tower = tower.to_empty(device=src_dev)
    tower.eval()
    shards = Shards(model_dir, src_dev)
    # The module's names against the checkpoint's: the tower is HunYuan's ViT with Hy4's merger.
    sd = tower.state_dict()
    for k in sd:
        name = k
        for a, b in VISION_RENAME.items():
            name = name.replace(a, b)
        t = shards.get("vit." + name)
        if t.shape != sd[k].shape:
            raise ValueError(f"vit.{name}: checkpoint {tuple(t.shape)}, module {tuple(sd[k].shape)}")
        with torch.no_grad():
            sd[k].copy_(t)
    with torch.no_grad():
        out = tower(pixel_values, grid)
    tower_out = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    merged = out.pooler_output if hasattr(out, "pooler_output") else out[1]
    blob = Path(args.blob_dir)
    save_f32(blob / "vision" / f"{img.stem}-tower.f32", tower_out[0].float())
    save_f32(blob / "vision" / f"{img.stem}-merged.f32", merged[0].float())
    print(
        "vision tower %s rms %.5f | merged %s rms %.5f"
        % (
            tuple(tower_out.shape),
            float(tower_out.float().pow(2).mean().sqrt()),
            tuple(merged.shape),
            float(merged.float().pow(2).mean().sqrt()),
        ),
        flush=True,
    )


def hy4(args, cfg, tok, shards, seqs, out, blob, dump, dev):
    """Hy4-flash: the text path of issue #40, layer by layer.

    Everything the layer does (iHC, Engram, gated MLA with the learned sink, the compressed indexer, latent MoE)
    comes from the transformers modules, so this tool says what the engine has to reproduce, not how.
    """
    import transformers
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4HyperHead
    from transformers.models.hy_v4_vl import modeling_hy_v4_vl as V

    pre = "model."
    pad = cfg.engram_pad_id
    t0 = time.time()
    with torch.no_grad():
        emb = shards.get(pre + "embed_tokens.weight").to(dev)
        rot = H.HYV4FlashRotaryEmbedding(cfg).to(dev).eval()
        state = {}
        for pid, ids in seqs:
            x = torch.tensor([ids], device=dev)
            T = len(ids)
            e = emb[x]
            h = e.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
            pos = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
            # Text-only: one position plane, so the three M-RoPE axes read the same positions.
            pe = {lt: rot(e, position_ids=pos, layer_type=lt) for lt in rot.layer_types}
            mask_kwargs = {
                "config": cfg,
                "inputs_embeds": e,
                "attention_mask": None,
                "cache": None,
                "position_ids": pos[0],
                "allow_is_causal_skip": False,  # the indexer's selection needs a mask to write into
            }
            masks = {
                "indexed_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
            # `build_ngram_tokens` / `build_engram_inputs` for a prompt of tokens: the shifted ids and the mask
            # of the positions that own an n-gram (no image or video token in a text prompt).
            shifted = [x]
            offset_of = torch.arange(T, device=dev).unsqueeze(0)
            for o in range(1, cfg.engram_max_ngram_size):
                prev = torch.nn.functional.pad(x, (o, 0), value=pad)[:, :T]
                shifted.append(torch.where(offset_of < o, torch.tensor(pad, device=dev), prev))
            state[pid] = {
                "ids": x,
                "h": h,
                "pe": pe,
                "masks": masks,
                "pos": pos,
                "ngram": torch.stack(shifted, dim=-1),
                "engram_mask": x != pad,
            }
        del emb
        n_layers = args.layers or cfg.num_hidden_layers
        for i in range(n_layers):
            with torch.device("meta"):
                layer = H.HYV4FlashDecoderLayer(cfg, i)
            layer = layer.to_empty(device=dev)
            for name, p in layer.named_parameters():
                if any(k in name for k in KEEP_FP32):
                    p.data = p.data.float()
            prefix = f"{pre}layers.{i}."
            skip = ()
            el = layer.engram_layer
            if el is not None:
                sizes = [s for order in cfg.engram_layer_head_vocab_sizes[str(i + 1)] for s in order]
                name = prefix + "engram_layer.multi_head_embedding.embedding.weight"
                el.multi_head_embedding = HostEngramEmbedding(
                    shards, name, sizes, cfg.engram_n_embed_per_ngram // cfg.engram_n_head_per_ngram, dev
                )
                skip = ("multi_head_embedding.embedding.weight",)
            load_module(layer, shards, prefix, skip, HY4_RENAME)
            if el is not None:
                # The module's non-persistent buffers are built in `__init__`, which ran on the meta device, so
                # `to_empty` leaves them uninitialised: the table's offsets, the head sizes and the hash
                # multipliers must be rebuilt from the loaded map exactly as `_init_weights` does, or every lookup
                # reads the wrong row.
                # Non-persistent buffers: the table's sizes, the frozen vocabulary map and the hash multipliers
                # drawn from it are computed, not stored.
                el.head_vocab_sizes = torch.tensor(sizes, device=dev)
                el.multi_head_embedding.offsets = (
                    torch.tensor([0] + sizes[:-1], device=dev).cumsum(0).to(torch.long)
                )
                el.multi_head_embedding.head_vocab_sizes = el.head_vocab_sizes
                el.lookup_table = shards.get(prefix + "engram_layer.lookup_table").to(dev)
                el.hash_multipliers = el.draw_hash_multipliers(int(el.lookup_table.max()) + 1).to(dev)
                print(
                    f"layer {i}: Engram, {len(sizes)} heads, table rows {sum(sizes)}, "
                    f"compressed vocab {int(el.lookup_table.max()) + 1}",
                    flush=True,
                )
            layer.eval()
            at = layer.attention_type
            import os as _os

            trace = int(_os.environ.get("HY4_TRACE_LAYER", "-1"))
            for pid, s in state.items():
                if pid in dump and i == trace:
                    out_t = hy4_trace(layer, s["h"], s, blob / "layers" / pid, f"T{i}", s["ngram"], s["engram_mask"])
                    ref = layer(
                        s["h"],
                        attention_mask=s["masks"][at],
                        position_embeddings=s["pe"][at],
                        position_ids=s["pos"],
                        cache=None,
                        indexer_position_ids=s["pos"],
                        ngram_tokens=s["ngram"],
                        engram_mask=s["engram_mask"],
                    )
                    assert torch.equal(out_t, ref), "the traced steps diverge from the module's forward"
                    s["h"] = ref
                else:
                    s["h"] = layer(
                        s["h"],
                        attention_mask=s["masks"][at],
                        position_embeddings=s["pe"][at],
                        position_ids=s["pos"],
                        cache=None,
                        indexer_position_ids=s["pos"],
                        ngram_tokens=s["ngram"],
                        engram_mask=s["engram_mask"],
                    )
                if pid in dump:
                    save_f32(blob / "layers" / pid / f"L{i}.f32", s["h"][0])
            del layer
            torch.cuda.empty_cache()
            rss = int(open("/proc/self/statm").read().split()[1]) * 4096 >> 30
            print(f"layer {i} done ({at}), {time.time() - t0:.0f}s, rss {rss} GiB", flush=True)

        with torch.device("meta"):
            head = HYV4HyperHead(cfg)
            norm = H.HYV4FlashRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        head, norm = head.to_empty(device=dev), norm.to_empty(device=dev)
        for p in head.parameters():
            p.data = p.data.float()
        head_rename = {"hc_fn": "hc_head_fn", "hc_scale": "hc_head_scale", "hc_base": "hc_head_base"}
        load_module(head, shards, pre + "hc_head.", rename=head_rename)
        load_module(norm, shards, pre, rename={"weight": "norm.weight"})
        head.eval()
        lm = shards.get("lm_head.weight").to(dev).float()
        meta = {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "reference": "layer-by-layer (scripts/ref_layerwise.py)",
            "layers": n_layers,
            "text": [],
        }
        with torch.no_grad():
            for pid, ids in seqs:
                s = state[pid]
                hn = norm(head(s["h"]))
                if pid in dump:
                    save_f32(blob / "layers" / pid / "head.f32", hn[0])
                logits = hn[0].float() @ lm.T
                save_f32(blob / "text" / pid / "logits_last.f32", logits[-1])
                lp = torch.log_softmax(logits, -1)
                top = torch.topk(lp[-1], 10)
                meta["text"].append(
                    {
                        "id": pid,
                        "input_ids": ids,
                        "argmax": logits.argmax(-1).tolist(),
                        "top10_last": {"ids": top.indices.tolist(), "logprobs": top.values.tolist()},
                    }
                )
                print(pid, len(ids), repr(tok.decode([int(logits[-1].argmax())])), flush=True)
    json.dump(meta, open(out / "golden.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blob-dir", required=True)
    ap.add_argument("--long", type=int, default=0, help="add a raw-text sequence of this many tokens (QSA indexer)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="attention implementation; the QSA sparse layers fail in sdpa above ~10K tokens, so the "
                         "long entry is generated with eager; hy4 runs with eager for the same reason (its sdpa "
                         "path fails on this stack with the learned sink)")
    ap.add_argument("--dump", default="en-fact", help="comma-separated ids whose per-layer states are saved")
    ap.add_argument("--layers", type=int, default=0, help="run only the first N layers (debug)")
    ap.add_argument("--only", default="", help="comma-separated prompt ids to run (default: all)")
    ap.add_argument("--vision", default=None, help="run only the vision tower on this image (hy4)")
    args = ap.parse_args()

    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    dev = "cuda"
    model_dir = Path(args.model)
    out, blob = Path(args.out), Path(args.blob_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model).text_config
    cfg._attn_implementation = args.attn
    tok = AutoTokenizer.from_pretrained(args.model)
    shards = Shards(model_dir, dev)
    pre = "model.language_model."
    dump = set(args.dump.split(",")) if args.dump else set()

    # Hy4-flash keeps its tensors under `model.` and reads its prompts as plain text (no chat template of the
    # Qwen family, and the image path is not part of this tool yet).
    family_hy4 = cfg.model_type.startswith("hy_v4")
    seqs = sequences(tok, args, chat=not family_hy4)
    torch.set_default_dtype(torch.bfloat16)
    if family_hy4:
        if args.vision:
            hy4_vision(args, cfg, model_dir, dev)
            return
        hy4(args, cfg, tok, shards, seqs, out, blob, dump, dev)
        return
    t0 = time.time()
    with torch.no_grad():
        emb = shards.get(pre + "embed_tokens.weight").to(dev)
        rot = M.Qwen4ExpTextRotaryEmbedding(cfg).to(dev)
        state = {}
        for pid, ids in seqs:
            x = torch.tensor([ids], device=dev)
            h = emb[x]
            T = len(ids)
            pos = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
            pe = rot(h, pos)
            mask = torch.ones(T, T, dtype=torch.bool, device=dev).tril().view(1, 1, T, T)
            state[pid] = {"ids": x, "h": h.repeat(1, 1, cfg.hc_count), "pe": pe, "mask": mask}
        del emb
        n_layers = args.layers or cfg.num_hidden_layers
        for i in range(n_layers):
            with torch.device("meta"):
                layer = M.Qwen4ExpTextDecoderLayer(cfg, i)
            lp = f"{pre}layers.{i}."
            skip = ()
            if layer.ple is not None:
                ng = ShardedRows(shards, lp + "ple.ple_embedding.ngram_embedding.", dev)
                layer.ple.ple_embedding.ngram_embedding = ng
                skip = ("ngram_embedding",)
            layer = layer.to_empty(device=dev)
            if layer.ple is not None:
                want = M._build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, layer.ple.ple_embedding.ple_layer_index, cfg.seed)
            load_module(layer, shards, lp, skip)
            layer.eval()
            if layer.ple is not None:
                pe_ = layer.ple.ple_embedding
                assert torch.equal(pe_.layer_multipliers.cpu(), want), "n-gram multipliers differ from the formula"
                assert torch.equal(pe_.ngram_heads_vocab_sizes.cpu(), torch.tensor(pe_.head_vocab_sizes)), "head sizes"
                assert torch.equal(pe_.ngram_heads_offsets.cpu(), torch.tensor(pe_.head_offsets)), "head offsets"
                print(f"layer {i}: PLE, table rows {ng.total}, multipliers {want.tolist()}", flush=True)
            for pid, s in state.items():
                s["h"] = layer(s["h"], position_embeddings=s["pe"], attention_mask=s["mask"], conv_mask=None,
                               past_key_values=None, ple_input_ids=s["ids"])
                if pid in dump:
                    save_f32(blob / "layers" / pid / f"L{i}.f32", s["h"][0])
            del layer
            torch.cuda.empty_cache()
            rss = int(open("/proc/self/statm").read().split()[1]) * 4096 >> 30
            print(f"layer {i} done, {time.time() - t0:.0f}s, rss {rss} GiB", flush=True)

        with torch.device("meta"):
            mixer = M.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        mixer = mixer.to_empty(device=dev)
        load_module(mixer, shards, pre + "hyper_connection_mixer.")
        head = shards.get("lm_head.weight").to(dev)
        meta = {"transformers": transformers.__version__, "torch": torch.__version__, "dtype": "bfloat16",
                "reference": "layer-by-layer (scripts/ref_layerwise.py)", "layers": n_layers, "text": []}
        for pid, ids in seqs:
            s = state[pid]
            hn = mixer(s["h"])
            if pid in dump:
                save_f32(blob / "layers" / pid / "mixed.f32", hn[0])
            logits = (hn[0] @ head.T).float()
            save_f32(blob / "text" / pid / "logits_last.f32", logits[-1])
            lp = torch.log_softmax(logits, -1)
            top = torch.topk(lp[-1], 10)
            meta["text"].append({"id": pid, "input_ids": ids, "argmax": logits.argmax(-1).tolist(),
                                 "top10_last": {"ids": top.indices.tolist(), "logprobs": top.values.tolist()}})
            print(pid, len(ids), repr(tok.decode([int(logits[-1].argmax())])), flush=True)
        json.dump(meta, open(out / "golden.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
