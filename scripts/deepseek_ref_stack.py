#!/usr/bin/env python3
"""The release's own blocks, run end to end from a prompt: embed -> blocks -> norm -> head.

`deepseek_ref_layer.py` checks one block in isolation; this checks that the *handoff* between blocks is right — the
streams the hyper-connections carry, the compressed KV and the indexer list a source publishes for the layers after
it, and the Engram lookup. It builds only the blocks the caller names, in bf16 (the release's storage dtype), so a
two- or four-block stack fits: 27 GB a block against 54 GB in f32.

    python3 scripts/deepseek_ref_stack.py --checkpoint /mnt/usb/deepseekv41-flash \
        --reference /mnt/usb/deepseekv41-flash/inference --layers 0,1 --out /golden/deepseek-ref/stack01 \
        --prompt "The capital of France is"

The comparison is at bf16 precision: a structural error is what it is here to find, not a 1e-6 difference.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, help="the directory holding model.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", default="0,1", help="block indices to run, in order")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--bos", action="store_true", help="prepend the begin-of-sentence token")
    ap.add_argument(
        "--random-x",
        type=int,
        metavar="TOKENS",
        help="do not embed a prompt: start the window from a random stream of this many positions (dumped as x.f32)",
    )
    ap.add_argument("--f32", action="store_true", help="keep the weights in f32 (one block only: 54 GB each)")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="build, load and run one block at a time and drop it, so a window deep in the stack fits (27 GB a block)",
    )
    ap.add_argument(
        "--table",
        type=Path,
        help="the companion file with the Engram tables: the rows come from there, as the engine reads them, "
        "instead of the checkpoint's 98 GB fp8 table (which does not fit)",
    )
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from deepseek_ref import install_kernels
    from deepseek_ref_engram import Table

    install_kernels()
    from deepseek_ref_layer import Checkpoint

    sys.path.insert(0, str(a.reference))
    import model as ref
    from engram import EngramLayout, NgramHashState
    from tokenizers import Tokenizer

    cfg = json.loads((a.checkpoint / "config.json").read_text())
    t = cfg["text_config"]
    layers = [int(x) for x in a.layers.split(",")]
    dtype = torch.float32 if a.f32 else torch.bfloat16

    tcfg = dict(
        max_batch_size=1,
        max_seq_len=4096,
        temperature=0,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=t["vocab_size"],
        dim=t["hidden_size"],
        moe_inter_dim=t["moe_intermediate_size"],
        n_layers=max(layers) + 1,
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
        compress_ratios=tuple(t["compress_ratios"][: max(layers) + 1]),
        kv_source_layers=tuple(x for x in t["kv_source_layer_ids"] if x <= max(layers)),
        index_source_layers=tuple(x for x in t["index_source_layer_ids"] if x <= max(layers)),
        compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=t["rope_scaling"]["original_max_position_embeddings"],
        rope_theta=t["rope_theta"],
        rope_factor=t["rope_scaling"]["factor"],
        beta_fast=t["rope_scaling"]["beta_fast"],
        beta_slow=t["rope_scaling"]["beta_slow"],
        index_n_heads=t["index_n_heads"],
        index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"],
        candidate_source_layer=(
            t["candidate_source_layer_id"] if t["candidate_source_layer_id"] <= max(layers) else -1
        ),
        candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"],
        hc_mult=t["hc_mult"],
        hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"],
        engram_layer_ids=tuple(x for x in t["engram_layer_ids"] if x <= max(layers)),
        engram_num_embeddings=tuple(
            n for i, n in zip(t["engram_layer_ids"], t["engram_num_embeddings"]) if i <= max(layers)
        ),
        engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_vocab_size=t["engram_vocab_size"],
        engram_n_heads=t["engram_n_heads"],
        engram_head_dim=t["engram_head_dim"],
        engram_pad_id=t["engram_pad_token_id"],
        engram_compressed_vocab_size=t["engram_compressed_vocab_size"],
        vision_n_layers=0,
        image_token_id=0,
        dspark_block_size=0,
    )
    args = ref.ModelArgs(**tcfg)

    class Shim:
        """`build_compressed_token_map` wants the transformers wrapper; it uses these two things."""

        def __init__(self, tok):
            self.backend_tokenizer = tok

        def __len__(self):
            return tok.get_vocab_size()

    tok = Tokenizer.from_file(str(a.checkpoint / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    if a.bos:
        ids.insert(0, tok.token_to_id("<｜begin▁of▁sentence｜>"))
    tokens = a.random_x or len(ids)

    torch.set_default_dtype(dtype)
    ref.default_dtype = dtype
    ck = Checkpoint(a.checkpoint)
    embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
    norm = ref.RMSNorm(args.dim, args.norm_eps)
    head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    class NoTable(torch.nn.Module):
        """A 98 GB table per Engram layer is not something to allocate: the rows are fetched after the build."""

        def __init__(self, num_embeddings: int, dim: int):
            super().__init__()

        def forward(self, ids):
            raise RuntimeError("the Engram table was replaced after the block was built")

    ref.ParallelEngramEmbedding = NoTable
    layout = EngramLayout.from_args(args)
    hashstate = NgramHashState(args, layout, Shim(tok)) if layout is not None else None

    def load(module, prefix: str) -> None:
        state = module.state_dict()
        for name in state:
            src = ck.f32(prefix + name)
            if tuple(src.shape) != tuple(state[name].shape):
                raise SystemExit(f"{prefix}{name}: checkpoint {src.shape} vs module {tuple(state[name].shape)}")
            state[name].copy_(torch.from_numpy(src.reshape(state[name].shape)))

    load(embed, "embed.")
    load(norm, "norm.")
    load(head, "head.")
    if hashstate is not None:
        # DeepSeek hashes the ids one lookback at a time, over the compressed token map; image spans would be DEAD.
        hashes = hashstate(torch.tensor([ids], dtype=torch.int64), 0, None)
        # The Engram table is 98 GB in fp8: take the rows from our own file, which is what the engine reads.
        table = Table(a.table)

    def block_at(i):
        """One block, loaded from the checkpoint; with `--stream` the caller drops it before the next one."""
        block = ref.Block(i, args, layout)
        if dtype == torch.float32:
            # `wo_a` is built in bf16 whatever the default is. Widen it in place: `block.float()` would hold a
            # second copy of 27 GB while it converts.
            for module in block.modules():
                for name, param in list(module.named_parameters(recurse=False)):
                    if param.dtype != torch.float32:
                        setattr(module, name, torch.nn.Parameter(param.data.float()))
        load(block, f"layers.{i}.")
        if block.engram is not None:
            name = f"layers.{i}.engram.embed.weight"
            dim = t["engram_head_dim"]
            own = table

            class Rows(torch.nn.Module):
                def forward(self, ids):
                    rows = own.rows(name, ids.reshape(-1).numpy(), dim)
                    return torch.from_numpy(rows).to(dtype).view(*ids.shape, dim)

            block.engram.embed = Rows()
        return block

    dumps = {}
    params = 0
    with torch.inference_mode():
        if a.random_x:
            g = torch.Generator().manual_seed(1234)
            h = (torch.randn(1, a.random_x, args.hc_mult, args.dim, generator=g) * 0.02).to(dtype)
            dumps["x"] = h
        else:
            h = embed(torch.tensor([ids], dtype=torch.int64))
            h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
        held = {} if a.stream else {i: block_at(i) for i in layers}
        for i in layers:
            block = block_at(i) if a.stream else held[i]
            params = sum(p.numel() for p in block.parameters())
            if block.engram is not None:
                h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
            h, pre_mix = block(h, 0, pre_mix, None)
            dumps[f"block{i}"] = h
            if a.stream:
                # 27 GB a block (54 in f32): drop this one before the next is built.
                del block
                gc.collect()
        # `hc_pre` uses no weights, so the last block does not have to stay for it.
        x = torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=2).to(dtype)
        out = head(norm(x), full_logits=True)
    dumps["logits_all"] = out
    print(f"{len(layers)} blocks, {params/1e9:.1f}B params a block", file=sys.stderr)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, tensor in dumps.items():
        tensor.detach().float().reshape(-1).contiguous().numpy().astype("<f4").tofile(str(a.out / f"{name}.f32"))
    (a.out / "manifest.json").write_text(
        json.dumps(
            {
                "layer": layers[-1],
                "layers": layers,
                "ids": ids,
                "tokens": tokens,
                "random_x": bool(a.random_x),
                "hidden": t["hidden_size"],
                "hc": t["hc_mult"],
                "dtype": "f32" if a.f32 else "bf16",
                "shapes": {k: list(v.shape) for k, v in dumps.items()},
            },
            indent=1,
        )
        + "\n"
    )
    print(json.dumps({"layers": layers, "ids": ids, "dumps": sorted(dumps)}))


if __name__ == "__main__":
    main()
