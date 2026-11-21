#!/usr/bin/env python3
"""Long-context recall of an OpenAI-compatible server: does a fact planted in the middle of a long text survive?

The same text is used for every engine (same tokenizer), so `prompt_tokens` must agree between them: the test
measures the engine on a prompt of N tokens, not the tokenizer. The answer is scored by the fact, not by the logits,
and the request carries a nonce in front so no engine can serve it from its prompt cache.

Usage: bench/needle.py URL [--api-key K] [--filler TEXT] [--chars 131000,520000] [--pos 0.1,0.5,0.9]
                           [--out rows.json]

Preflight and decode rates come from the server's `timings` (llama-server and `malleus serve` both report them).
"""
import argparse
import json
import random
import sys
import time
import urllib.request

NEEDLE = "\n\nConfidential note: the access code of the Vallechiara safe is ORNITH-4712.\n\n"
QUESTION = "\n\nWhat is the access code of the Vallechiara safe? Answer with the code only."
FACT = ("ORNITH-4712", "ORNITH 4712")
NEUTRAL = ("\n\nLibrary note: the catalogue number of the Vallechiara herbarium is ORNITH-4712.\n\n",
           "\n\nWhat is the catalogue number of the Vallechiara herbarium? Answer with the number only.")


def ask(url, model, text, key, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--filler", default="", help="long text file (a book, a corpus); see --filler-help")
    ap.add_argument("--chars", default="32000,131000,520000", help="haystack sizes, in characters")
    ap.add_argument("--pos", default="0.1,0.5,0.9", help="where to plant the fact, as a fraction of the text")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default="", help="write one row of JSON per request")
    ap.add_argument("--neutral", action="store_true",
                    help="plant a catalogue number instead of a safe's access code (some models refuse to give one)")
    a = ap.parse_args()
    if not a.filler:
        ap.error("--filler is required: paste the text to search (an English corpus of ~4.2 characters/token; "
                 "520 000 characters is about 110 000 tokens)")

    book = open(a.filler, encoding="utf-8", errors="replace").read()
    model = a.model
    if not model:
        headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
        with urllib.request.urlopen(urllib.request.Request(a.url.rstrip("/") + "/v1/models", headers=headers),
                                    timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    print(f"{a.url}  model={model}  filler {len(book)} characters")
    print(f"{'chars':>8s} {'pos':>4s} {'tokens':>7s} {'wall s':>7s} {'pp tok/s':>8s} {'tg tok/s':>8s} "
          f"{'ok':>3s} answer")
    rng = random.Random(time.time_ns())
    rows = []
    for chars in [int(x) for x in a.chars.split(",")]:
        hay = (book * (chars // len(book) + 1))[:chars]
        for pos in [float(x) for x in a.pos.split(",")]:
            at = int(len(hay) * pos)
            needle, question = (NEUTRAL if a.neutral else (NEEDLE, QUESTION))
            text = f"[{rng.random()}] " + hay[:at] + needle + hay[at:] + question
            out, wall = ask(a.url, model, text, a.api_key, a.max_tokens)
            t = out.get("timings") or {}
            answer = (out["choices"][0]["message"].get("content") or "").strip().replace("\n", " ")
            ok = any(f in answer for f in FACT)
            pp = t.get("prompt_n", 0) / t.get("prompt_ms", 1) * 1000
            tg = t.get("predicted_n", 0) / t.get("predicted_ms", 1) * 1000 if t.get("predicted_ms") else 0
            print(f"{chars:8d} {pos:4.1f} {out['usage']['prompt_tokens']:7d} {wall:7.1f} {pp:8.1f} {tg:8.1f} "
                  f"{'SI' if ok else 'no':>3s} {answer[:60]}", flush=True)
            rows.append({"chars": chars, "pos": pos, "tokens": out["usage"]["prompt_tokens"], "wall": wall,
                         "pp": pp, "tg": tg, "ok": ok, "answer": answer})
            if a.out:
                json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nrecalled {sum(1 for r in rows if r['ok'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
