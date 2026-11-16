#!/usr/bin/env python3
"""Decode speed through the chat API on bench/spec-prompts.txt (Malleus and llama-server both report timings).

Usage: bench/spec_api.py URL [API_KEY] [--greedy | --talkbot] [--max-tokens N]
--talkbot sends the sampling Malleus applies to talkbot requests (temperature 0.7, top_p 0.8, top_k 20).
"""
import json
import pathlib
import sys
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
url = args[0].rstrip("/")
key = args[1] if len(args) > 1 else ""
greedy = "--greedy" in sys.argv
talkbot = "--talkbot" in sys.argv
max_tokens = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else 160
prompts = [
    l.replace("\\n", "\n")
    for l in (pathlib.Path(__file__).parent / "spec-prompts.txt").read_text().splitlines()
    if l.strip()
]


def chat(prompt):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if greedy:
        body["temperature"] = 0
    if talkbot:
        body.update(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    return json.load(urllib.request.urlopen(req, timeout=1800))


chat("Ciao")
n = ms = 0
for p in prompts:
    t = chat(p)["timings"]
    n += t["predicted_n"]
    ms += t["predicted_ms"]
    print(f"{t['predicted_n']:4d} tok  {t['predicted_per_second']:6.2f} tok/s")
print(f"total {n} tok at {n / ms * 1000:.2f} tok/s ({'greedy' if greedy else 'talkbot sampling' if talkbot else 'server default sampling'})")
