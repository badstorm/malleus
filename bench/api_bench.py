#!/usr/bin/env python3
"""Prefill and decode speed of an OpenAI-compatible server that reports llama-server `timings`.

Usage: bench/api_bench.py URL [--api-key K] [--prefill 512,2048,8192,32768] [--decode-ctx 0,32768] [--max-tokens 256] [--runs 1]

Prefill: prompts of about N tokens of pseudo-random words with a fresh nonce (no prompt-cache hits), max_tokens 1;
reports prompt_n / prompt_ms. Decode: the prompts of bench/spec-prompts.txt (thinking off), greedy and with the
talkbot sampling (temperature 0.7, top_p 0.8, top_k 20), optionally after a filler context of N tokens.

--api-key (or $MALLEUS_API_KEY) sends an `Authorization: Bearer` header, which a server started with a key requires.
"""
import json
import pathlib
import random
import sys
import time
import urllib.request

argv = sys.argv[1:]
url = argv[0].rstrip("/")


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


prefill = [int(x) for x in opt("--prefill", "512,2048,8192,32768").split(",") if x]
decode_ctx = [int(x) for x in opt("--decode-ctx", "0").split(",") if x != ""]
max_tokens = int(opt("--max-tokens", "256"))
runs = int(opt("--runs", "1"))
api_key = opt("--api-key", "")

HEADERS = {"Content-Type": "application/json"}
if api_key:
    HEADERS["Authorization"] = f"Bearer {api_key}"
prompts = [
    l.replace("\\n", "\n")
    for l in (pathlib.Path(__file__).parent / "spec-prompts.txt").read_text().splitlines()
    if l.strip()
]
WORDS = (
    "the of and to in is that for it as with was on be by at this from are or have an they which one you were her "
    "all she there would their we him been has when who will more no if out so said what up its about into than them "
    "can only other new some could time these two may then do first any my now such like our over man me even most "
    "made after also did many before must through back years where much your way well down should because each just "
    "those people how too little state good very make world still own see men work long get here between both life "
    "being under never day same another know while last might us great old year off come since against go came right "
    "used take three casa tempo anno giorno uomo cosa volta lavoro parte vita mano città paese storia acqua sistema "
    "fn let mut struct impl return match if else for while loop break continue pub use mod const static enum trait"
).split()


def filler(n_words, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def chat(messages, **kw):
    body = {"messages": messages, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    return r, time.time() - t0


chat([{"role": "user", "content": "Ciao"}], max_tokens=8)
rng = random.Random(time.time_ns())
for n in prefill:
    for _ in range(runs):
        text = f"[{rng.random()}] " + filler(int(n * 0.72), rng) + "\nHow many words are above? Answer briefly."
        r, wall = chat([{"role": "user", "content": text}], max_tokens=1, temperature=0)
        t = r["timings"]
        print(f"prefill {t['prompt_n']:6d} tok  {t['prompt_n'] / t['prompt_ms'] * 1000:8.1f} tok/s  "
              f"(wall {wall:.2f}s, cached {t.get('cache_n', 0)})", flush=True)

for ctx in decode_ctx:
    head = f"[{rng.random()}] " + filler(int(ctx * 0.72), rng) + "\n\n" if ctx else ""
    for mode, kw in (("greedy", {"temperature": 0}), ("talkbot", dict(temperature=0.7, top_p=0.8, top_k=20))):
        n = ms = 0
        for p in prompts:
            # the filler is a separate earlier turn so its prefix stays cacheable across prompts
            msgs = [{"role": "user", "content": head + "Ignore the text above."}, {"role": "assistant", "content": "OK."},
                    {"role": "user", "content": p}] if ctx else [{"role": "user", "content": p}]
            r, _ = chat(msgs, **kw)
            t = r["timings"]
            n += t["predicted_n"]
            ms += t["predicted_ms"]
        print(f"decode ctx {ctx:6d} {mode:8s} {n:5d} tok  {n / ms * 1000:7.2f} tok/s", flush=True)
