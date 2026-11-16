#!/usr/bin/env python3
"""Fixed cost and scaling of prompt processing, measured through the HTTP API.

Usage: bench/api_overhead.py URL [--sizes 8,64,256,1024,4096] [--reps 3] [--model NAME]

Sends max_tokens=1 completions so the wall time is prompt processing plus whatever the server and the client add
around it, then fits t = a + b*n over the sizes: `a` is the fixed cost per request (tokenization, chat template,
queueing, session and n-gram work) and 1/b is the marginal prefill rate. Two engines compared with this script are
compared on the same axis, which is what tells a short-prompt deficit in the engine from one in the fixed cost.
"""
import argparse
import json
import time
import urllib.request


def once(url, model, n_words, nonce, max_tokens=1):
    filler = " ".join(f"w{(i * 7919) % 100000}" for i in range(n_words))
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"{filler}\n\nNonce {nonce}: reply with one word."}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    u = d.get("usage", {})
    return wall, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--sizes", default="8,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    model = a.model
    if not model:
        with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    print(f"{a.url}  model={model}  reps={a.reps}")
    pts = []
    print(f"{'words':>7s} {'prompt tok':>11s} {'best wall':>10s} {'tok/s':>9s} {'ms/token':>9s}")
    for i, n in enumerate(sizes):
        best = None
        for r in range(a.reps):
            wall, ptok, _ = once(a.url, model, n, f"{n}-{r}-{time.time_ns()}")
            if best is None or wall < best[0]:
                best = (wall, ptok)
        wall, ptok = best
        pts.append((ptok, wall))
        print(f"{n:7d} {ptok:11d} {wall*1000:9.1f}ms {ptok/wall:9.1f} {wall/ptok*1000:9.3f}")
    # Per-segment marginal rate and the intercept it implies. A single straight line over all the sizes is
    # misleading here: the curve is convex, because attention grows with the context while the expert weights are
    # streamed once per chunk whatever the length. The global fit is printed last and only for reference.
    print(f"\n{'segment':>17s} {'ms/token':>9s} {'tok/s':>8s} {'implied fixed':>14s}")
    for (n0, t0), (n1, t1) in zip(pts, pts[1:]):
        b = (t1 - t0) / (n1 - n0)
        print(f"{n0:7d}-{n1:<9d} {b*1000:9.3f} {1/b:8.1f} {(t0 - b*n0)*1000:11.1f} ms")
    k = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = k * sxx - sx * sx
    if den:
        b = (k * sxy - sx * sy) / den
        intercept = (sy - b * sx) / k
        first = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        last = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
        print(f"\nglobal fit (reference only): fixed {intercept*1000:.1f} ms, marginal {1/b:.1f} tok/s")
        if last > first * 1.1:
            print(f"  the curve is convex: {first*1000:.3f} ms/token on the first segment against "
                  f"{last*1000:.3f} on the last, so read the per-segment rows, not this one")


if __name__ == "__main__":
    main()
