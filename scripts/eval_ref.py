#!/usr/bin/env python3
"""Reference for `malleus eval`: per-position bf16 statistics on texts disjoint from the calibration corpus.

Writes eval.json (token ids, reference argmax and NLL per position) next to the golden set and the top-k log-probs
of every position to --blob-dir/eval/topk.bin (k int32 ids then k f32 log-probs per position).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import gutenberg_body  # noqa: E402
from golden import load_on_gpu  # noqa: E402

GUTENBERG = [
    ("56775", "it", "31f2d1f7e0575af592559610a1bedad7bad40c15ef4af431f2b44bd154729ef2"),
    ("18456", "it", "7eed1911796c417cad328b4ee8b1978ad4fa0a040cd5c8c20cd22cb9274ca833"),
    ("48320", "en", "bf63b19ade817616aa64b5dcaf32e8685592c315fd22e336defe685baa02d419"),
    ("11", "en", "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"),
]
CODE = ["transformers/generation/utils.py", "torch/nn/modules/module.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="/golden/calib/eval")
    ap.add_argument("--out", default="golden/qwen38-27b/eval.json")
    ap.add_argument("--blob-dir", default="/golden/qwen38-27b")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(4321)
    texts = []
    for gid, lang, sha in GUTENBERG:
        data = (Path(args.corpus) / f"pg{gid}.txt").read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            print(f"warning: pg{gid}.txt sha256 differs from the manifest", flush=True)
        texts.append((f"{lang}-pg{gid}", gutenberg_body(data.decode("utf-8", "replace"))))
    site = Path(transformers.__file__).parents[1]
    for rel in CODE:
        texts.append((f"code-{Path(rel).stem}", (site / rel).read_text()))

    docs = []
    for name, text in texts:
        ids = tok(text).input_ids
        off = rng.randrange(len(ids) // 4, len(ids) - 2 * args.length)
        docs.append((name, ids[off:off + args.length]))
    for lang, (name, text) in (("it", texts[0]), ("en", texts[2])):
        ids = tok(text).input_ids
        off = rng.randrange(len(ids) // 2, len(ids) - 2 * args.length)
        body = tok.decode(ids[off:off + args.length])
        ask = "Continua questo racconto:" if lang == "it" else "Continue this story:"
        cut = len(body) // 4
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{ask}\n\n{body[:cut]}"},
                {"role": "assistant", "content": body[cut:]}]
        chat = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        docs.append((f"chat-{lang}", tok(chat).input_ids[:args.length]))

    model = load_on_gpu(args.model)
    blob = Path(args.blob_dir) / "eval"
    blob.mkdir(parents=True, exist_ok=True)
    out = {"topk": args.topk, "docs": []}
    with open(blob / "topk.bin", "wb") as f, torch.no_grad():
        for name, ids in docs:
            logits = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits, -1)
            nxt = torch.tensor(ids[1:], device="cuda")
            nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
            top = torch.topk(lp, args.topk, dim=-1)
            for i in range(len(ids)):
                f.write(top.indices[i].to(torch.int32).cpu().numpy().tobytes())
                f.write(top.values[i].cpu().numpy().astype(np.float32).tobytes())
            out["docs"].append({"id": name, "input_ids": ids, "argmax": logits.argmax(-1).tolist(),
                                "nll": [round(v, 5) for v in nll.tolist()]})
            print(f"{name:22s} T={len(ids)} ppl {float(nll.mean().exp()):.3f}", flush=True)
    json.dump(out, open(args.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
