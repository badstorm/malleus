#!/usr/bin/env python3
"""Dump the DeepSeek-V4.1 vision reference for one image: the patches the processor makes, the ViT output and the
aligner's merged tokens.

`malleus inspect --image` and the `deepseek` tests compare against these files. The tower and the processor are the
only part of this model that can be checked on one machine today: they need 1 GB of weights, not 510.

    python3 scripts/deepseek_vision_ref.py --checkpoint /models/deepseekv41-flash \
        --reference /models/deepseekv41-flash/inference --image photo.jpg --out /tmp/vision-ref

Writes `<out>.json` (shapes) and `<out>-{patches,vit,merged}.f32` (little-endian float32).
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def read_tensor(ckpt: Path, name: str) -> torch.Tensor:
    """One tensor of the checkpoint, by name, from whichever shard holds it."""
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(ckpt / index[name]), framework="pt") as f:
        return f.get_tensor(name)


def load_state(ckpt: Path, prefix: str, module: torch.nn.Module) -> None:
    """Copy `prefix<param>` of the checkpoint into `module`, shard by shard, from the safetensors index."""
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    state = module.state_dict()
    files: dict[str, list[str]] = {}
    for name in state:
        full = f"{prefix}{name}"
        if full not in index:
            raise SystemExit(f"the checkpoint has no {full}")
        files.setdefault(index[full], []).append(name)
    for shard, names in files.items():
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for name in names:
                full = f"{prefix}{name}"
                src = f.get_tensor(full)
                if tuple(src.shape) != tuple(state[name].shape):
                    raise SystemExit(f"{full}: {tuple(src.shape)} against {tuple(state[name].shape)}")
                state[name].copy_(src)
    module.load_state_dict(state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, help="the directory holding vision.py")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--blocks", action="store_true", help="also dump the embedding and every block's output")
    ap.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="compute in fp32 (default: the gate compares the math, not the storage) or bf16 as the release runs",
    )
    a = ap.parse_args()

    cfg = json.loads((a.checkpoint / "config.json").read_text())
    v, t = cfg["vision_config"], cfg["text_config"]
    args = types.SimpleNamespace(
        vision_enabled=True,
        vision_patch_size=v["patch_size"],
        vision_dim=v["hidden_size"],
        vision_n_heads=v["num_attention_heads"],
        vision_inter_dim=v["intermediate_size"],
        vision_rope_theta=float(v["rope_theta"]),
        vision_n_layers=v["num_hidden_layers"],
        vision_downsample_ratio=v["downsample_ratio"],
        vision_max_n_token=v["max_image_tokens"],
        vision_min_pixels=v["min_pixels"],
        vision_max_wh_ratio=v.get("max_wh_ratio"),
        dim=t["hidden_size"],
    )

    sys.path.insert(0, str(a.reference))
    from image_processor import load_image  # noqa: E402
    from vision import Aligner, ViT  # noqa: E402

    torch.set_default_dtype(torch.float32 if a.dtype == "fp32" else torch.bfloat16)
    torch.manual_seed(0)
    vit, aligner = ViT(args), Aligner(args)
    load_state(a.checkpoint, "vision.", vit)
    load_state(a.checkpoint, "aligner.", aligner)
    vit.eval()
    aligner.eval()

    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"url": a.image}, args)
    # The processor always hands the tower bf16 patches; with a float32 model they are widened (exactly).
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    patches = patches.to(dtype)
    steps = []

    def hook(_module, _inputs, output):
        steps.append(output.detach().float().reshape(-1).numpy())

    handles = []
    if a.blocks:
        handles.append(vit.patch_embed.register_forward_hook(hook))
        for block in vit.blocks:
            handles.append(block.register_forward_hook(hook))
    with torch.inference_mode():
        vit_out = vit(patches, n_vit_h, n_vit_w)
        merged = aligner(vit_out, n_vit_h, n_vit_w)
    for h in handles:
        h.remove()

    # The span the LLM sees: the three learned delimiters and the aligner rows in reading order, laid out as
    # `[START] ([IMAGE] * w + [NEW_LINE]) * h [END]` — what `Transformer.merge_image_embeddings` writes into the
    # prompt embeddings. It is what `deepseek_vision::span_embeddings` has to reproduce.
    from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, image_token_types

    rows = merged.reshape(n_llm_h * n_llm_w, t["hidden_size"]).float()
    delimiter = {name: read_tensor(a.checkpoint, f"image_{name}").float() for name in ("start", "end", "newline")}
    take = iter(range(rows.shape[0]))
    span = []
    for kind in image_token_types(n_llm_h, n_llm_w).reshape(-1).tolist():
        if kind == IMAGE_START:
            span.append(delimiter["start"])
        elif kind == IMAGE_END:
            span.append(delimiter["end"])
        elif kind == IMAGE_NEW_LINE:
            span.append(delimiter["newline"])
        else:
            span.append(rows[next(take)])
    span = torch.stack(span).reshape(-1)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "patches": patches.float().numpy().reshape(-1),
        "vit": vit_out.float().numpy().reshape(-1),
        "merged": merged.float().numpy().reshape(-1),
        "span": span.float().numpy().reshape(-1),
    }
    for name, arr in arrays.items():
        Path(f"{a.out}-{name}.f32").write_bytes(arr.astype("<f4").tobytes())
    if steps:
        for i, step in enumerate(steps):
            Path(f"{a.out}-step{i:02d}.f32").write_bytes(step.astype("<f4").tobytes())
        manifest_steps = len(steps)
    else:
        manifest_steps = 0
    manifest = {
        "steps": manifest_steps,
        "image": str(a.image),
        "n_vit_h": n_vit_h,
        "n_vit_w": n_vit_w,
        "n_llm_h": n_llm_h,
        "n_llm_w": n_llm_w,
        "dtype": a.dtype,
        "patch": v["patch_size"],
        "hidden": v["hidden_size"],
        "dim": t["hidden_size"],
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    Path(f"{a.out}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
