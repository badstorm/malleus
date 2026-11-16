#!/usr/bin/env python3
"""Decode speed on agent turns: the reply copies a tool result the prompt shows with line numbers (or JSON escapes).

The point of this fixture is the workload where a coding agent reads a file and writes it back: on such a turn
about half of the generated tokens are copies of text that is in the conversation, but never verbatim in the
prompt's own tokens (the tool result carries `NNN→` prefixes or `\\n` escapes). See bench/agent-prompts.jsonl.

Usage: bench/agent_turn.py URL [--max-tokens N] [--greedy] [--quiet]
"""
import json
import pathlib
import sys
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
url = args[0].rstrip("/")
max_tokens = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else 256
greedy = "--greedy" in sys.argv
quiet = "--quiet" in sys.argv
turns = [
    json.loads(l)
    for l in (pathlib.Path(__file__).parent / "agent-prompts.jsonl").read_text().splitlines()
    if l.strip()
]


def chat(t):
    body = {"messages": t["messages"], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if greedy:
        body["temperature"] = 0
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=1800))


rates = []
for t in turns:
    try:
        d = chat(t)
    except Exception as e:  # noqa: BLE001 - a fixture that fails to load should say which one
        print(f"{t['id']:16s} ERROR {e}")
        continue
    tm = d.get("timings") or {}
    r = tm.get("predicted_per_second") or 0.0
    rates.append(r)
    if not quiet:
        print(f"{t['id']:16s} {r:7.1f} tok/s  {d['usage']['completion_tokens']:4d} token")
if rates:
    print(f"{'media':16s} {sum(rates)/len(rates):7.1f} tok/s")
