#!/usr/bin/env python3
"""Summarise a rocprofv3 kernel trace: per-kernel GPU time, launch gaps and per-token cost.

Usage: bench/ktrace.py TRACE.csv [--decode N] [--top 30] [--phase decode|prefill|all] [--gap-us 300]

`rocprofv3 --kernel-trace --output-format csv -d DIR -- malleus bench ...` writes one row per dispatch.
Decode tokens all run the same kernel sequence, so the decode phase is what follows the last
prompt-processing-only kernel; the prefill phase is what follows the weight upload. Reports busy time, the
idle time left inside the phase by launch gaps, and the per-kernel breakdown, so that a slow kernel and a
too-short one (whose cost is mostly its gap) are told apart.
"""
import argparse
import collections
import csv
import re
import sys

PREFILL_ONLY = (
    r"^(gemm_i8|gemm_wmma|gemm_w4a8|gemm_w8a8|gemm_mls|moe_w4a8|moe_i8|hc_quant_row"
    r"|hc_mix_inv|rot_quant_row|deltanet_chunk|attention_wmma|attention_gather|idx_union)"
)


def blocks(r):
    """Workgroups of a dispatch. rocprofv3 writes Grid_Size as the number of work-*items* (blocks x workgroup),
    so dividing by the workgroup size is what gives a number to compare with the engine's launch geometry."""
    grid = int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"])
    wg = int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])
    return grid // wg if wg else grid


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["Kind"] != "KERNEL_DISPATCH":
                continue
            rows.append(
                dict(
                    name=r["Kernel_Name"],
                    start=int(r["Start_Timestamp"]),
                    end=int(r["End_Timestamp"]),
                    grid=blocks(r),
                    vgpr=int(r["VGPR_Count"]),
                    scratch=int(r["Scratch_Size"]),
                    lds=int(r["LDS_Block_Size"]),
                )
            )
    rows.sort(key=lambda r: r["start"])
    return rows


def split_tokens(rows, min_gap_ns):
    """Splits dispatches into tokens at idle gaps of at least `min_gap_ns`."""
    out, cur, prev_end = [], [], None
    for r in rows:
        if prev_end is not None and r["start"] - prev_end >= min_gap_ns and cur:
            out.append(cur)
            cur = []
        cur.append(r)
        prev_end = max(prev_end or 0, r["end"])
    if cur:
        out.append(cur)
    return out


def report(rows, label, per):
    busy = sum(r["end"] - r["start"] for r in rows)
    span = rows[-1]["end"] - rows[0]["start"] if rows else 0
    idle = span - busy
    print(f"\n=== {label} ===")
    print(f"dispatches {len(rows)}   busy {busy/1e6:.2f} ms   span {span/1e6:.2f} ms   idle {idle/1e6:.2f} ms "
          f"({100*idle/span if span else 0:.1f}%)")
    if per:
        print(f"per token: busy {busy/per/1e6:.3f} ms  idle {idle/per/1e6:.3f} ms  wall {span/per/1e6:.3f} ms  "
              f"({per/span*1e9 if span else 0:.2f} tok/s)")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # count, ns, max grid, max vgpr
    for r in rows:
        a = agg[r["name"]]
        a[0] += 1
        a[1] += r["end"] - r["start"]
        a[2] = max(a[2], r["grid"])
        a[3] = max(a[3], r["vgpr"])
    return agg, busy, idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--decode", type=int, default=0, help="decode tokens to take from the tail of the trace")
    ap.add_argument("--gap-us", type=float, default=300.0, help="idle gap that separates two tokens")
    ap.add_argument("--top", type=int, default=28)
    ap.add_argument("--phase", default="all", choices=["all", "decode", "prefill"])
    ap.add_argument("--regex", default="", help="only report kernels whose name contains this")
    ap.add_argument("--decode-after", default="", help=f"regex marking prefill-only kernels (default {PREFILL_ONLY})")
    a = ap.parse_args()

    rows = load(a.trace)
    if not rows:
        sys.exit("no dispatches")
    print(f"{len(rows)} dispatches, span {(rows[-1]['end']-rows[0]['start'])/1e9:.2f} s")

    mark = re.compile(a.decode_after or PREFILL_ONLY)
    # Prompt-processing kernels only run in the forward pass, so the first one ends the weight load and the
    # last one ends the prefill; decode is whatever follows.
    fwd = [r for r in rows if mark.match(r["name"])]
    pre_start = fwd[0]["start"] if fwd else rows[0]["start"]
    pre_end = fwd[-1]["end"] if fwd else rows[-1]["end"]
    pre = [r for r in rows if pre_start <= r["start"] <= pre_end]
    dec = [r for r in rows if r["start"] > pre_end]
    if a.decode:
        # sanity check: `deltanet_scan` runs once per linear layer, `moe_route_w` once per MoE layer
        for probe, per_token in (("deltanet_scan", 36), ("moe_route_w", 49)):
            n = sum(1 for r in dec if r["name"] == probe)
            if n and abs(n / per_token - a.decode) > 0.5:
                print(f"warning: {probe} x{n} -> {n/per_token:.2f} tokens, asked {a.decode}", file=sys.stderr)

    for label, sel, per in (("decode", dec, a.decode), ("prefill", pre, None)):
        if not sel or (a.phase != "all" and a.phase != label):
            continue
        agg, busy, idle = report(sel, label, per)
        tot = sum(v[1] for v in agg.values())
        print(f"{'kernel':34s} {'n':>7s} {'total ms':>10s} {'%':>6s} {'avg us':>9s} {'maxblk':>9s} {'vgpr':>5s}")
        items = sorted(agg.items(), key=lambda kv: -kv[1][1])
        shown, shown_ns = 0, 0
        for name, (n, ns, g, vg) in items:
            if a.regex and a.regex not in name:
                continue
            print(f"{name[:34]:34s} {n/per if per else n:7.1f} {ns/1e6:10.3f} {100*ns/tot:6.2f} "
                  f"{ns/n/1e3:9.2f} {g:9d} {vg:5d}")
            shown += 1
            shown_ns += ns
            if shown >= a.top:
                break
        rest = tot - shown_ns
        if rest:
            print(f"{'-- other kernels':34s} {'':7s} {rest/1e6:10.3f} {100*rest/tot:6.2f}")
        if per:
            print(f"{'-- idle (launch gaps)':34s} {'':7s} {idle/1e6:10.3f} {100*idle/(busy+idle):6.2f} "
                  f"{idle/per/1e3:9.2f}")


if __name__ == "__main__":
    main()
