#!/usr/bin/env python3
"""The Engram hash state of a DeepSeek-V4.1 checkpoint, computed once.

The n-gram table is addressed by a compressed token map (tokens that normalise alike share an id), by one prime
per (layer, n-gram order, head) and by one multiplier per (layer, lookback). None of that depends on a prompt, and
recomputing it at load time would need the tokenizer's normalizers and numpy's generator. So it is written once,
next to the tables, and the engine reads it.

    python3 scripts/deepseek_engram_state.py --checkpoint /models/deepseekv41-flash --out /golden/deepseek-engram

Writes `<out>-state.json` (primes, offsets, multipliers, sizes) and `<out>-token-map.i32` (one compressed id per
token of the tokenizer's vocabulary).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--reference", type=Path, help="the release's inference directory (default: <checkpoint>/inference)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ref = a.reference or (a.checkpoint / "inference")
    sys.path.insert(0, str(ref))

    import torch
    from engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers
    from tokenizers import Tokenizer

    t = json.loads((a.checkpoint / "config.json").read_text())["text_config"]
    tok = Tokenizer.from_file(str(a.checkpoint / "tokenizer.json"))

    class Shim:
        """`build_compressed_token_map` expects the transformers wrapper; it only needs these two things."""

        backend_tokenizer = tok

        def __len__(self):
            return tok.get_vocab_size()

    token_map, vocab_size = build_compressed_token_map(Shim())
    assert vocab_size == t["engram_compressed_vocab_size"], (vocab_size, t["engram_compressed_vocab_size"])

    layout = EngramLayout(
        max_ngram_size=t["engram_max_ngram_size"],
        layer_ids=tuple(t["engram_layer_ids"]),
        num_embeddings=tuple(t["engram_num_embeddings"]),
        primes=(),
        n_heads=t["engram_n_heads"],
        head_dim=t["engram_head_dim"],
    )
    # The same prime/offset search `EngramLayout.from_args` runs, from the config's bucket size.
    primes, seen = [], set()
    for _ in layout.layer_ids:
        per_ngram = []
        for _ in range(t["engram_max_ngram_size"] - 1):
            sizes, current = [], t["engram_vocab_size"] - 1
            for _ in range(t["engram_n_heads"]):
                current = _next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat]
    # The multipliers derive from the *compressed* vocabulary, not from the tokenizer's own size.
    mult = compute_hash_multipliers(tuple(t["engram_layer_ids"]), t["engram_max_ngram_size"], vocab_size)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}-token-map.i32").write_bytes(np.array(token_map, dtype="<i4").tobytes())
    state = {
        "layers": list(layout.layer_ids),
        "num_embeddings": list(layout.num_embeddings),
        "max_ngram": t["engram_max_ngram_size"],
        "n_heads": t["engram_n_heads"],
        "head_dim": t["engram_head_dim"],
        "pad": t["engram_pad_token_id"],
        "vocab": int(tok.get_vocab_size()),
        "primes": primes,
        "offsets": offsets,
        "multipliers": mult.tolist(),
        "token_map": len(token_map),
    }
    Path(f"{a.out}-state.json").write_text(json.dumps(state) + "\n")
    print(json.dumps({"compressed_vocab": vocab_size, "layers": state["layers"], "primes0": primes[0][0][:3]}))


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while c in seen or not _is_prime(c):
        c += 1
    return c


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


if __name__ == "__main__":
    main()
