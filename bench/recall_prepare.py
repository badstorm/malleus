#!/usr/bin/env python3
"""Build a needle-in-a-haystack token fixture for the indexer recall gate.

Run in the golden container (needs transformers + the model tokenizer):

  scripts/golden.sh python3 bench/recall_prepare.py --model /models/extra/Qwen3.8-Flash-Next \
    --filler-json /golden/long-11000/golden.json --out /src/bench/recall/16k.json \
    --tokens 16000 --needle-fraction 0.33

The filler is the token ids of the `long-11000` reference (raw source text, no chat template), repeated if
needed. The needle and the question are tokenized on their own and inserted as ids, so their block positions are
exact and no detokenize/re-tokenize round trip is involved.
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

NEEDLE = " The secret access code is ZX-9174-QQ."
QUESTION = "\n\nWhat is the secret access code? Answer with the code only."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler-json", required=True, help="golden.json with a long-<n> text entry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--needle-fraction", type=float, default=0.33)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    g = json.loads(Path(args.filler_json).read_text())
    entry = next(e for e in g["text"] if e["id"].startswith("long-"))
    filler = list(entry["input_ids"])
    needle = tok.encode(NEEDLE, add_special_tokens=False)
    question = tok.encode(QUESTION, add_special_tokens=False)
    body = args.tokens - len(needle) - len(question)
    if body <= 0:
        raise SystemExit("target shorter than needle + question")
    while len(filler) < body:
        filler += filler
    filler = filler[:body]
    at = int(body * args.needle_fraction) // 4 * 4
    ids = filler[:at] + needle + filler[at:]
    ids += question
    needle_blocks = [at // 4, (at + len(needle) - 1) // 4]
    out = {
        "name": args.name or f"recall-{args.tokens // 1000}k-{int(args.needle_fraction * 100)}",
        "tokens": len(ids),
        "needle_at": at,
        "needle_len": len(needle),
        "needle_blocks": needle_blocks,
        "query_start": len(ids) - len(question),
        "query_end": len(ids),
        "ids": ids,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{args.out}: {len(ids)} tokens, needle at {at} blocks {needle_blocks}, "
          f"query {out['query_start']}..{out['query_end']}")


if __name__ == "__main__":
    main()
