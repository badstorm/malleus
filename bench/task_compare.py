#!/usr/bin/env python3
"""Two models, the same tasks, a score each — the measurement issue #134 asks for.

Top-1 agreement with a quantized model's own reference says how far a file drifted; it does not say whether the
model is still good enough to use. This harness asks the models themselves: the same prompts, then a checker per
task (JSON equality, a Python function against unit tests, an exact string, the shape of the answer).

The inference is slow (a 763B model on one APU answers a task in minutes), so running and scoring are two steps:
answers are saved, and the score is computed later, against as many files as were collected.

    bench/task_compare.py run  http://localhost:8090 --out answers-qwen.json [--max-tokens 512] [--greedy]
    bench/task_compare.py score answers-qwen.json answers-deepseek-2bit.json answers-deepseek-1bit.json

`run` uses the chat endpoint with thinking off unless `--think`, and stores the reply and the seconds it took.
`score` prints one column per file: pass/fail per task, the total, and the mean seconds an answer took.
"""
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

TASKS = pathlib.Path(__file__).parent / "task-set.jsonl"


def tasks():
    return [json.loads(l) for l in TASKS.read_text().splitlines() if l.strip()]


def chat(url, messages, max_tokens, greedy, think):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if greedy:
        body["temperature"] = 0
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def code_blocks(text):
    """The fenced blocks of a reply, or the whole reply when it is not fenced."""
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m if m else [text]


def norm(s):
    return re.sub(r"[\s`'\".,;:!?()]+", "", s).lower()


def check(kind, c, reply):
    blocks = code_blocks(reply)
    if kind == "json_equal":
        for b in [reply] + blocks:
            for m in re.findall(r"(\[.*?\]|\{.*?\})", b, re.S):
                try:
                    got = json.loads(m)
                except json.JSONDecodeError:
                    continue
                if got == c["want"]:
                    return True
        return False
    if kind == "python":
        code = blocks[0]
        try:
            out = subprocess.run(
                [sys.executable, "-c", code + "\n" + c["test"]],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return False
        return out.returncode == 0
    if kind == "exact":
        return norm(c["want"]) in norm(reply)
    if kind == "exact_lines":
        got = [norm(l) for l in reply.splitlines() if l.strip()]
        return [norm(w) for w in c["want"]] == got[: len(c["want"])]
    if kind == "contains_all":
        return all(norm(w) in norm(reply) for w in c["want"])
    if kind == "boxed":
        m = re.search(r"\\boxed\{([^}]*)\}", reply)
        if not m:
            return False
        return re.sub(r"[^0-9.\-]", "", m.group(1)) == c["want"]
    if kind == "bullets":
        lines = [l.strip() for l in reply.splitlines() if l.strip().startswith("- ")]
        if len(lines) != c["count"]:
            return False
        return all(len(l.split()) <= c["max_words"] for l in lines)
    raise SystemExit(f"unknown check {kind}")


def run(url, out, max_tokens, greedy, think):
    got = []
    for t in tasks():
        limit = min(max_tokens, t.get("max_tokens", max_tokens))
        t0 = time.time()
        try:
            reply = chat(url, [{"role": "user", "content": t["prompt"]}], limit, greedy, think)
        except Exception as e:  # a timeout or a dead server is a failed answer, not a crash
            reply = f"<error: {e}>"
        dt = time.time() - t0
        ok = check(t["check"]["kind"], t["check"], reply)
        print(f"{t['id']:18} {'pass' if ok else 'FAIL'} {dt:7.1f}s", flush=True)
        got.append({"id": t["id"], "reply": reply, "seconds": dt})
    pathlib.Path(out).write_text(json.dumps({"url": url, "answers": got}, indent=1))
    print(f"wrote {out}: {len(got)} answers")



def score(paths):
    ts = {t["id"]: t for t in tasks()}
    cols = []
    for p in paths:
        d = json.load(open(p))
        ans = {a["id"]: a for a in d["answers"]}
        cols.append((pathlib.Path(p).stem, ans, d.get("url", "")))
    print(f"{'task':18} " + " ".join(f"{c[0][:22]:>22}" for c in cols))
    total = [0] * len(cols)
    secs = [[] for _ in cols]
    for tid, t in ts.items():
        row = []
        for i, (_, ans, _) in enumerate(cols):
            a = ans.get(tid)
            ok = a is not None and check(t["check"]["kind"], t["check"], a["reply"])
            total[i] += bool(ok)
            if a:
                secs[i].append(a["seconds"])
            row.append("pass" if ok else ("--" if a is None else "FAIL"))
        print(f"{tid:18} " + " ".join(f"{r:>22}" for r in row))
    n = len(ts)
    print(f"{'total':18} " + " ".join(f"{t}/{n}".rjust(22) for t in total))
    print(f"{'mean seconds':18} " + " ".join(f"{sum(s) / len(s):.0f}".rjust(22) for s in secs if s))


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cmd = argv[0]
    if cmd == "run":
        url = argv[1]
        out = argv[argv.index("--out") + 1] if "--out" in argv else "answers.json"
        max_tokens = int(argv[argv.index("--max-tokens") + 1]) if "--max-tokens" in argv else 512
        run(url, out, max_tokens, "--greedy" in argv, "--think" in argv)
    elif cmd == "score":
        score([a for a in argv[1:] if not a.startswith("--")])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
