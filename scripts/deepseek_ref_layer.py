#!/usr/bin/env python3
"""The reference implementation, run on one real layer of a DeepSeek-V4.1 checkpoint.

The tiny reference (`deepseek_ref.py`) checks the graph's shape on random weights. This one checks the numbers
on the checkpoint's own weights: it builds a one-layer model with the released geometry, loads layer 0 (attention,
hyper-connections, 384 routed experts and the shared one), and dumps every stage so the engine's host forward can
be compared against it.

Layer 0 is a window layer: compress ratio 0, no compressor, no indexer, no Engram, no DSpark. That is exactly the
part of the graph that has to be right before any of the rest can be trusted.

    python3 scripts/deepseek_ref_layer.py --checkpoint /models/deepseekv41-flash --reference /models/deepseekv41-flash/inference \
        --out /golden/deepseek-ref/layer0 --prompt "The capital of France is"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


class Checkpoint:
    """Random access to the checkpoint's tensors, with the two block formats decoded to f32."""

    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self._handles = {}

    def _read(self, name: str):
        shard = self.index[name]
        if shard not in self._handles:
            f = open(self.root / shard, "rb")
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
            self._handles[shard] = (f, header, 8 + n)
        f, header, base = self._handles[shard]
        e = header[name]
        a, b = e["data_offsets"]
        f.seek(base + a)
        return f.read(b - a), e["dtype"], e["shape"]

    def f32(self, name: str) -> np.ndarray:
        raw, dtype, shape = self._read(name)
        if dtype == "BF16":
            return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().numpy().reshape(shape)
        if dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        if dtype == "F8_E4M3":
            q = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).float().numpy().reshape(shape)
            sraw, _, sshape = self._read(name.replace(".weight", ".scale"))
            # e8m0: the byte is the biased exponent, one scale per 32x32 block.
            sc = np.ldexp(np.ones(sshape, np.float32), np.frombuffer(sraw, np.uint8).astype(np.int32).reshape(sshape) - 127)
            blocks = np.repeat(np.repeat(sc, 32, axis=0), 32, axis=1)
            return (q * blocks[: shape[0], : shape[1]]).astype(np.float32)
        if dtype == "I8":
            # e2m1 pairs, low nibble first, one e8m0 scale per row and 32 columns.
            b = np.frombuffer(raw, np.uint8).reshape(shape[0], shape[1])
            out = np.empty((shape[0], shape[1] * 2), np.float32)
            for half, nib in ((0, b & 0x0F), (1, b >> 4)):
                out[:, half::2] = np.where(nib & 8, -MAG[nib & 7], MAG[nib & 7])
            sraw, _, sshape = self._read(name.replace(".weight", ".scale"))
            sc = np.ldexp(np.ones(sshape, np.float32), np.frombuffer(sraw, np.uint8).astype(np.int32).reshape(sshape) - 127)
            return (out * np.repeat(sc, 32, axis=1)).astype(np.float32)
        raise SystemExit(f"{name}: {dtype} is not a weight this script can read")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, help="the directory holding model.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompt", default="The capital of France is")
    a = ap.parse_args()

    # The graph's two kernels, in torch, exactly as `deepseek_ref.py` does for the tiny model.
    from deepseek_ref import install_kernels

    install_kernels()
    sys.path.insert(0, str(a.reference))
    import model as ref

    cfg = json.loads((a.checkpoint / "config.json").read_text())
    t = cfg["text_config"]
    vocab = t["vocab_size"]
    dim = t["hidden_size"]
    args = ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=512,
        temperature=0,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=vocab,
        dim=dim,
        moe_inter_dim=t["moe_intermediate_size"],
        n_layers=1,
        n_mtp_layers=0,
        n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"],
        n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["num_experts_per_tok"],
        score_func=t["scoring_func"],
        norm_topk_prob=t["norm_topk_prob"],
        route_scale=t["routed_scaling_factor"],
        swiglu_limit=t["swiglu_limit"],
        q_lora_rank=t["q_lora_rank"],
        head_dim=t["head_dim"],
        rope_head_dim=t["qk_rope_head_dim"],
        norm_eps=t["rms_norm_eps"],
        o_groups=t["o_groups"],
        o_lora_rank=t["o_lora_rank"],
        window_size=t["sliding_window"],
        compress_ratios=(0,),
        kv_source_layers=(),
        index_source_layers=(),
        compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=0,
        rope_theta=t["rope_theta"],
        index_n_heads=t["index_n_heads"],
        index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"],
        candidate_source_layer=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        hc_mult=t["hc_mult"],
        hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"],
        engram_layer_ids=(),
        engram_num_embeddings=(),
        vision_n_layers=0,
        image_token_id=0,
        dspark_block_size=0,
    )

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    # The reference builds its linears in bf16 whatever the default dtype says, so widen the module before
    # loading: the weights then land in f32 exactly, and the comparison is about the graph, not about storage.
    model = ref.Transformer(args).float()

    ck = Checkpoint(a.checkpoint)
    state = model.state_dict()
    loaded = 0
    for name in list(state):
        if name.startswith("layers.0.") or name in ("embed.weight", "norm.weight", "head.weight"):
            src = ck.f32(name)
            if tuple(src.shape) != tuple(state[name].shape):
                raise SystemExit(f"{name}: checkpoint {src.shape} vs module {tuple(state[name].shape)}")
            state[name].copy_(torch.from_numpy(src.reshape(state[name].shape)))
            loaded += 1
    print(f"loaded {loaded} of {len(state)} tensors, {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters", file=sys.stderr)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(a.checkpoint / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.int64)

    dumps = {}

    def hook(name, pick=lambda out: out):
        def f(_m, _i, out):
            dumps[name] = pick(out).detach().float().reshape(-1).numpy()

        return f

    handles = [model.embed.register_forward_hook(hook("embed"))]
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(hook(f"block{i}", lambda o: o[0])))
        handles.append(layer.attn.register_forward_hook(hook(f"attn{i}")))
        handles.append(layer.ffn.register_forward_hook(hook(f"ffn{i}")))
        handles.append(layer.ffn_norm.register_forward_hook(hook(f"ffn_in{i}")))
        handles.append(
            layer.ffn_norm.register_forward_hook(
                lambda _m, inp, _o, i=i: dumps.__setitem__(
                    f"ffn_premix{i}", inp[0].detach().float().reshape(-1).numpy()
                )
            )
        )
    final = {}
    handles.append(model.norm.register_forward_hook(lambda _m, _i, o: final.setdefault("h", o.detach().float())))
    with torch.inference_mode():
        _ids, logits, _ = model(x, 0)
        logits_all = model.head(final["h"], full_logits=True)
    for h in handles:
        h.remove()
    dumps["logits_all"] = logits_all.detach().float().reshape(-1).numpy()
    dumps["ids"] = np.array(ids, np.float32)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, arr in dumps.items():
        arr.astype("<f4").tofile(str(a.out / f"{name}.f32"))
    (a.out / "manifest.json").write_text(
        json.dumps(
            {
                "ids": ids,
                "prompt": a.prompt,
                "hidden": dim,
                "vocab": vocab,
                "shapes": {k: list(v.shape) for k, v in dumps.items()},
            },
            indent=1,
        )
        + "\n"
    )
    print(json.dumps({"ids": ids, "logits": list(dumps["logits_all"].shape)}))


if __name__ == "__main__":
    main()
