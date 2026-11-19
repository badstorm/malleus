#!/usr/bin/env python3
"""What expert streaming would cost, before writing the streaming code (issue #14).

A MoE token touches `top_k` experts per layer, and every one of them that is not in the cache is read from the
model file. The decode rate is therefore `min(compute, disk_bandwidth / bytes_from_disk_per_token)`: a byte count,
not a kernel problem. This script supplies the three inputs on the real file and the real routing:

  --model FILE     the `.malleus` file: per-expert slice offsets and sizes, from the GGUF header (no tensor read)
  --trace FILE     the routing trace written by `MALLEUS_TRACE_ROUTE=<path> malleus run|bench`
  --cache-gib G    simulate caches of G GiB (repeatable) under LRU, LFU, LFU with aging and an oracle
  --disk-check N   really `pread` the misses of the first N steps, `--threads` in parallel, O_DIRECT

Examples
    scripts/expert_cache.py --model /srv/models/malleus/ling3-flash-mls4.malleus

    MALLEUS_TRACE_ROUTE=/tmp/route.txt malleus bench --model ling3-flash-mls4.malleus --decode 128
    scripts/expert_cache.py --model ling3-flash-mls4.malleus --trace /tmp/route.txt \\
        --cache-gib 8,16,24,32,48,60 --disk-check 8 --threads 16
"""
import argparse
import mmap
import os
import struct
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

GIB = 1 << 30
ALIGN = 4096
DTYPE = {
    0: ("f32", 4),
    1: ("f16", 2),
    24: ("i8", 1),
    26: ("i32", 4),
    27: ("i64", 8),
    30: ("bf16", 2),
    0x4D4C0002: ("mls2", 2 / 8),
    0x4D4C0003: ("mls3", 3 / 8),
    0x4D4C0004: ("mls4", 1 / 2),
    0x4D4C0008: ("mls8", 1),
}


class Reader:
    def __init__(self, f):
        self.f = f

    def bytes(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit("truncated GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self.bytes(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.bytes(8))[0]

    def string(self):
        return self.bytes(self.u64()).decode()

    def value(self, ty, depth=0):
        if ty == 0:
            return self.bytes(1)[0]
        if ty == 1:
            return struct.unpack("<b", self.bytes(1))[0]
        if ty == 2:
            return struct.unpack("<H", self.bytes(2))[0]
        if ty == 3:
            return struct.unpack("<h", self.bytes(2))[0]
        if ty == 4:
            return self.u32()
        if ty == 5:
            return struct.unpack("<i", self.bytes(4))[0]
        if ty == 6:
            return struct.unpack("<f", self.bytes(4))[0]
        if ty == 7:
            return self.bytes(1)[0] != 0
        if ty == 8:
            return self.string()
        if ty == 9:
            ety, n = self.u32(), self.u64()
            return [self.value(ety, depth + 1) for _ in range(n)]
        if ty == 10:
            return self.u64()
        if ty == 11:
            return struct.unpack("<q", self.bytes(8))[0]
        if ty == 12:
            return struct.unpack("<d", self.bytes(8))[0]
        raise SystemExit(f"bad metadata value type {ty}")


def read_header(path):
    """Metadata and tensor name -> (dtype, dims in file order, absolute data offset), as `Gguf::open` reads them."""
    with open(path, "rb") as f:
        r = Reader(f)
        if r.bytes(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        if r.u32() != 3:
            raise SystemExit(f"{path}: not a GGUF v3 file")
        n_tensors, n_kv = r.u64(), r.u64()
        meta = {}
        for _ in range(n_kv):
            k = r.string()
            meta[k] = r.value(r.u32())
        raw = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            dtype, off = r.u32(), r.u64()
            raw.append((name, dims, dtype, off))
        align = meta.get("general.alignment") or 32
        base = (f.tell() + align - 1) // align * align
    tensors = {n: (DTYPE.get(d, (hex(d), 0))[0], dims, base + o) for n, dims, d, o in raw}
    return meta, tensors


def expert_geometry(path):
    """Layers, experts, bytes per expert and, per layer, the four (offset, block, experts) of one expert.

    `gate_up_proj` is stored with the expert as the outermost row dim, so one expert is a contiguous block and
    streaming it is a `pread` (4 with the scales), no repacking. The scales are not 4 KiB multiples: see `read`.
    """
    meta, tensors = read_header(path)
    # The prefix depends on the family (`model.` for Ling, `model.language_model.` for Qwen): key the layers by
    # their index and rebuild the names from what the file actually has.
    names = {}
    for name in tensors:
        if ".mlp.experts." not in name:
            continue
        head, tail = name.split(".mlp.experts.", 1)
        if tail not in ("gate_up_proj", "gate_up_proj.scales", "down_proj", "down_proj.scales"):
            continue
        layer = int(head.split("layers.")[1].split(".")[0])
        names.setdefault(layer, {})[tail] = name
    layers = sorted(names)
    if not layers:
        raise SystemExit(f"{path}: no routed experts (dense model?)")
    per_layer = {}
    for layer in layers:
        blocks = []
        for suffix in ("gate_up_proj", "gate_up_proj.scales", "down_proj", "down_proj.scales"):
            name = names[layer].get(suffix)
            if name is None:
                raise SystemExit(f"{path}: layer {layer}: .mlp.experts.{suffix} missing")
            dtype, dims, off = tensors[name]
            size = 1
            for d in dims[:-1]:
                size *= d
            size = int(size * dict((v[0], v[1]) for v in DTYPE.values())[dtype])
            blocks.append((off, size, dims[-1]))
        counts = {b[2] for b in blocks}
        if len(counts) != 1:
            raise SystemExit(f"{path}: layer {layer} has inconsistent expert counts {counts}")
        per_layer[layer] = counts.pop(), blocks
    experts = {c for c, _ in per_layer.values()}
    if len(experts) != 1:
        raise SystemExit(f"{path}: inconsistent expert counts {experts}")
    per_expert = sum(b[1] for b in per_layer[layers[0]][1])
    return meta, layers, experts.pop(), per_expert, per_layer


def slice_bounds(blocks, expert):
    """Absolute (offset, size) of one expert's blocks."""
    return [(off + expert * size, size) for off, size, _ in blocks]


def parse_trace(path, keep="decode"):
    """[(layer, [ids])] in file order: one line per MoE call and token (`layer t row id:weight ...`).

    `keep="decode"` takes the calls with one token, `keep="prefill"` the others (the chunk that warms the pools),
    `keep="all"` both. A prefill call carries one row per token, so its lines are not a step of the decode.
    """
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            one = int(p[1]) == 1
            if keep == "decode" and not one:
                continue
            if keep == "prefill" and one:
                continue
            out.append((int(p[0]), [int(x.split(":")[0]) for x in p[3:]]))
    if not out:
        raise SystemExit(f"{path}: empty routing trace")
    return out


def steps(events, experts):
    """Forward passes: a new step starts when the layer index goes backwards.

    The ids past the last routed expert are the shared expert (Ling appends it as slot `top_k`): it is resident
    like the dense weights and never streams, so it is dropped here.
    """
    out, cur, last = [], [], None
    for layer, ids in events:
        if last is not None and layer <= last:
            out.append(cur)
            cur = []
        cur.append((layer, [i for i in ids if i < experts]))
        last = layer
    if cur:
        out.append(cur)
    return out


def simulate(st, per_layer, policy="lru", warm=None, age=None):
    """Per step, the (layer, expert) pairs that miss, with `per_layer` slots on every layer.

    The engine has one pool per layer (the kernel indexes it with that layer's stride), so the simulation does too;
    a global budget would be a different design. Every policy starts from the same **warmed** pools, because the
    engine fills them from the prefill chunk: without that, the comparison would just measure who got the warm.

    Policies: `lru` (what the engine does), `lfu` (counts, least used evicted, oldest on ties), `lfu-age` (the
    counts halved every `age` steps, so an expert that stops being used loses its protection) and `static` (the
    most used pairs of the whole trace, pinned: an oracle that knows the future).
    """
    layers = sorted({layer for step in st for layer, _ in step})
    counts = defaultdict(Counter)
    if warm:
        for layer, ids in warm:
            counts[layer].update(ids)
    if policy == "static":
        for step in st:
            for layer, ids in step:
                counts[layer].update(ids)
        pinned = {l: {e for e, _ in counts[l].most_common(per_layer)} for l in layers}
        return [[(layer, e) for layer, ids in step for e in ids if e not in pinned[layer]] for step in st]
    cache = {l: OrderedDict() for l in layers}
    for l in layers:
        for e, _ in counts[l].most_common(per_layer):
            cache[l][e] = None
    misses = []
    for i, step in enumerate(st):
        step_misses = []
        for layer, ids in step:
            c, f = cache[layer], counts[layer]
            for e in ids:
                if e in c:
                    f[e] += 1
                    if policy == "lru":
                        c.move_to_end(e)
                    continue
                step_misses.append((layer, e))
                f[e] += 1
                if len(c) >= per_layer:
                    victim = next(iter(c)) if policy == "lru" else min(c, key=lambda x: (f[x], 0))
                    del c[victim]
                c[e] = None
        misses.append(step_misses)
        if age and policy == "lfu-age" and i % age == age - 1:
            for l in layers:
                for e in list(counts[l]):
                    counts[l][e] = max(1, counts[l][e] // 2)
    return misses


def coverage(picks):
    """Cache size (pairs, hence GiB) that covers a fraction of the picks: the skew, measured."""
    total = sum(picks.values())
    cum, out = 0, {}
    for i, (_, n) in enumerate(picks.most_common(), 1):
        cum += n
        for frac in (0.5, 0.8, 0.9, 0.95, 0.99):
            if frac not in out and cum >= frac * total:
                out[frac] = i
    return out


def read(fd, off, size):
    """O_DIRECT read windowed to 4 KiB: an expert's `down` scales (30 720 B) are not a multiple of it."""
    start = off // ALIGN * ALIGN
    end = (off + size + ALIGN - 1) // ALIGN * ALIGN
    buf = mmap.mmap(-1, end - start)
    try:
        os.preadv(fd, [memoryview(buf)], start)
    finally:
        buf.close()


def disk_check(path, per_layer, misses, threads):
    """Time the misses step by step with real reads.

    A step is the 40 layers in sequence: within a layer the reads of its experts (4 blocks each) go in parallel,
    across layers they cannot, because the router of layer L+1 needs the output of layer L. At the hit rates
    measured on a real generation the misses are few, so the number shows the *latency* of a miss more than the
    bandwidth of the disk.
    """
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(threads) as ex:
            for step in misses:
                blocks = [b for le in step for b in slice_bounds(per_layer[le[0]][1], le[1])]
                list(ex.map(lambda b: read(fd, b[0], b[1]), blocks))
    finally:
        os.close(fd)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a .malleus file with routed experts")
    ap.add_argument("--trace", help="routing trace written by MALLEUS_TRACE_ROUTE")
    ap.add_argument("--cache-gib", default="8,16,24,32,48,60", help="cache sizes to simulate")
    ap.add_argument("--disk-check", type=int, default=0, metavar="STEPS", help="time the real reads of N steps")
    ap.add_argument("--threads", type=int, default=16, help="parallel preads per layer in --disk-check")
    ap.add_argument("--bandwidth", type=float, default=291.0, help="disk MB/s for the arithmetic (measured: HDD)")
    ap.add_argument("--compute-tok-s", type=float, default=18.0, help="decode tok/s with everything resident")
    args = ap.parse_args()

    meta, layers, experts, per_expert, per_layer = expert_geometry(args.model)
    size = os.path.getsize(args.model)
    print(f"model   : {args.model}")
    print(f"file    : {size / 1e9:.2f} GB ({size / GIB:.2f} GiB), {meta.get('general.architecture', '?')}")
    print(f"experts : {len(layers)} MoE layers x {experts} esperti, {per_expert / 1e6:.3f} MB per expert, "
          f"{len(layers) * experts * per_expert / 1e9:.2f} GB routed in the file")
    if not args.trace:
        print("\nno --trace: only the file geometry. A trace comes from "
              "`MALLEUS_TRACE_ROUTE=/tmp/route.txt malleus run ...`")
        return 0

    events = parse_trace(args.trace)
    st = steps(events, experts)
    # The engine warms the pools from the prefill chunk (a device-to-device copy, not a second read): the
    # simulation starts every policy from the same state.
    warm = [(layer, [e for e in ids if e < experts]) for layer, ids in parse_trace(args.trace, keep="prefill")]
    picks = Counter((layer, e) for step in st for layer, ids in step for e in ids)
    every = [len(ids) for step in st for _, ids in step]
    top_k = max(1, round(sum(every) / len(every)) if every else 1)
    used = len({layer for step in st for layer, _ in step})
    per_token = top_k * used * per_expert
    print(f"\ntrace   : {len(events)} lines, {len(st)} steps, {sum(picks.values())} picks, top_k {top_k}, "
          f"{used} layer MoE{' (the MTP layer is not in the run)' if used < len(layers) else ''}")
    print(f"          one token asks {per_token / 1e9:.3f} GB of experts; the pairs used are "
          f"{len(picks)} = {len(picks) * per_expert / 1e9:.2f} GB")
    mean = sum(picks.values()) / len(picks)
    top = picks.most_common(1)[0][1]
    print(f"          picks per pair: mean {mean:.1f}, max {top} ({top / mean:.1f}x), "
          f"seen once {sum(1 for v in picks.values() if v == 1)}")
    print("          coverage: " + ", ".join(
        f"{int(f * 100)}% in {n} pairs ({n * per_expert / GIB:.1f} GiB)" for f, n in sorted(coverage(picks).items())))
    if len(st) > 1:
        ov = sorted(len({e for _, ids in a for e in ids} & {e for _, ids in b for e in ids})
                    / max(1, len({e for _, ids in a for e in ids}))
                    for a, b in zip(st, st[1:]))
        print(f"          overlap with the previous step: median {ov[len(ov) // 2] * 100:.0f}%, "
              f"p10 {ov[len(ov) // 10] * 100:.0f}%, p90 {ov[9 * len(ov) // 10] * 100:.0f}%")

    n_layers = len({layer for step in st for layer, _ in step})
    print(f"\n{'cache':>8} {'policy':>7} {'hit':>8} {'miss/step':>10} {'MB/step':>9} "
          f"{'tok/s (BW)':>11} {'tok/s (I/O)':>12}")
    print(f"{'':>8} {'':>7} {'':>8} {'':>10} {'':>9} {'prefetch':>11} {'sync reads':>12}")
    for g in [float(x) for x in args.cache_gib.split(",") if x]:
        cap = max(1, int(g * GIB / per_expert) // n_layers)
        for policy in ("lru", "lfu", "lfu-age", "static"):
            age = 32 if policy == "lfu-age" else None
            misses = simulate(st, cap, policy, warm, age)
            total = len(st) * top_k * len(layers)
            hit = 1 - sum(len(m) for m in misses) / total
            per = sum(len(m) for m in misses) / len(misses)
            mb = per * per_expert / 1e6
            by_bw = min(args.compute_tok_s, args.bandwidth / mb) if mb else args.compute_tok_s
            io = ""
            if args.disk_check:
                n = min(args.disk_check, len(st))
                dt = disk_check(args.model, per_layer, misses[:n], args.threads)
                io = f"{n / dt:.2f}"
            print(f"{g:7.1f}G {policy:>7} {hit * 100:7.2f}% {per:10.2f} {mb:9.1f} "
                  f"{by_bw:11.2f} {io:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
