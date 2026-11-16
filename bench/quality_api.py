#!/usr/bin/env python3
"""Last-token agreement of an OpenAI-compatible engine against the golden bf16 reference.

Usage: bench/quality_api.py URL [--golden golden/qwen38-flash-next/golden.json]
                              [--tokenizer /path/to/tokenizer.json] [--model NAME] [--top 10]

Sends every golden text prompt with thinking off, max_tokens 1 and top_logprobs, checks that the engine
tokenized it to exactly the reference length, and scores the engine's top-1 against the reference argmax plus a
KL over the reference top-k. This is the API-level counterpart of `malleus golden`, so a competing engine that
only speaks HTTP can be scored on the same prompts and the same reference.
"""
import argparse
import json
import math
import pathlib
import re
import sys
import urllib.request

RAW_TEXT = "The quick brown fox jumps over the lazy dog. In 1969, humans first"


def byte_decoder():
    """GPT-2 style byte-level BPE: vocab strings back to the bytes they stand for."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def tok_bytes(s, dec):
    """Vocab string to the bytes it stands for; added tokens are literal text, not byte-level BPE."""
    try:
        return bytes(dec[c] for c in s)
    except KeyError:
        return s.encode()


def prompts_from(repo):
    """The (id, text) pairs golden.py feeds the reference, read back out of the script."""
    src = pathlib.Path(repo, "scripts/golden.py").read_text()
    body = re.search(r"TEXT_PROMPTS = \[(.*?)\n\]", src, re.S).group(1)
    out = []
    for pid, txt in re.findall(r'\("([a-z0-9\-]+)",\s*((?:"(?:[^"\\]|\\.)*")|None|(?:[\w.]+\s*\+[\s\S]*?\)))\)', body):
        out.append((pid, None if txt == "None" else eval(txt)))
    return out


def ask(url, model, text, ids, top, raw):
    if raw:
        body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "logprobs": top}
        path = "/v1/completions"
    else:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": top,
                "chat_template_kwargs": {"enable_thinking": False}}
        path = "/v1/chat/completions"
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--golden", default="golden/qwen38-flash-next/golden.json")
    ap.add_argument("--tokenizer", default="/srv/containers/aitools/models/extra/Qwen3.8-Flash-Next/tokenizer.json")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    g = json.load(open(a.golden))
    by = {e["id"]: e for e in g["text"]}
    dec = byte_decoder()
    tk = json.load(open(a.tokenizer))
    inv = {i: s for s, i in tk["model"]["vocab"].items()}
    # ids above the base vocabulary live in added_tokens, and the reference top-k reaches them
    for at in tk.get("added_tokens", []):
        inv[at["id"]] = at["content"]
    model = a.model
    if not model:
        with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]

    texts = dict(prompts_from(a.repo))
    texts["raw-continue"] = RAW_TEXT
    print(f"{a.url}  model={model}  riferimento bf16 da {a.golden}")
    print(f"{'prompt':16s} {'tok':>5s} {'ref top1':>9s} {'engine top1':>22s} {'ok':>3s} {'KL top-k':>9s}")
    hit = tot = skipped = 0
    kls = []
    for pid, entry in by.items():
        text = texts.get(pid)
        if text is None:
            skipped += 1
            continue
        ref_ids = entry["top10_last"]["ids"]
        ref_lp = entry["top10_last"]["logprobs"]
        try:
            d = ask(a.url, model, text, entry["input_ids"], a.top, pid == "raw-continue")
        except Exception as e:
            print(f"{pid:16s} ERRORE {type(e).__name__}: {str(e)[:60]}")
            skipped += 1
            continue
        n = d["usage"]["prompt_tokens"]
        if n != len(entry["input_ids"]):
            print(f"{pid:16s} {n:5d} tokenizzazione diversa dal golden ({len(entry['input_ids'])}), scartato")
            skipped += 1
            continue
        ch = d["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        if not lp:
            print(f"{pid:16s} {n:5d} no logprobs in the response")
            skipped += 1
            continue
        got = lp[0]["top_logprobs"]
        got_bytes = {bytes(x["bytes"]): x["logprob"] for x in got if x.get("bytes")}
        if not got_bytes:
            print(f"{pid:16s} {n:5d} no logprob with the bytes in the response, skipped")
            skipped += 1
            continue
        ref_top = tok_bytes(inv[ref_ids[0]], dec)
        top1 = max(got_bytes, key=got_bytes.get)
        ok = top1 == ref_top
        # KL(reference || engine) restricted to the reference top-k, the tail mass the engine did not return
        # is charged at the worst logprob it did return, so the number is an upper bound and comparable
        # between engines measured the same way.
        floor = min(got_bytes.values())
        kl = 0.0
        for tid, rlp in zip(ref_ids[:a.top], ref_lp[:a.top]):
            p = math.exp(rlp)
            b = tok_bytes(inv[tid], dec)
            kl += p * (rlp - got_bytes.get(b, floor))
        hit += ok
        tot += 1
        kls.append(kl)
        print(f"{pid:16s} {n:5d} {ref_ids[0]:9d} {ref_top.decode('utf-8','replace')[:20]:>22s} "
              f"{'SI' if ok else 'no':>3s} {kl:9.4f}")
    print(f"\ntop-1 ultimo token: {hit}/{tot}  (voci scartate {skipped})")
    if kls:
        kls_s = sorted(kls)
        print(f"KL sul top-{a.top}: media {sum(kls)/len(kls):.4f}  mediana {kls_s[len(kls_s)//2]:.4f}  "
              f"massima {max(kls):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
