#!/usr/bin/env python3
"""Rewrites a DeepSeek-V4.1 per-tensor allocation (the released build's, from ISTA-DASLab) at a leaner budget.

The published map is the result of an optimisation at 2.00 bpw, and it is the input our converter reads
(`MALLEUS_DEEPSEEK_ALLOC`). To ask "what does 1.25 bpw cost", the same map is taken down one rung: the routed
experts, which are 89 % of the backbone's bytes, move to `IQ1_S` (our MLS1: one sign per value, an e4m3 scale per
32) while everything else stays where the optimisation put it.

    scripts/deepseek_alloc_1bit.py /models/deepseek-v41-iq2-allocation.txt > /tmp/alloc-1bit.txt

`--exps` picks the type for the experts. `--keep-experts` leaves a comma-separated list of `blk.N` alone, which is
how a layer-by-layer experiment (keep the first and last layers at 2 bits) is expressed.
"""
import sys

EXPS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    path = args[0]
    kind = "IQ1_S"
    keep = set()
    if "--exps" in sys.argv:
        kind = sys.argv[sys.argv.index("--exps") + 1]
    if "--keep-experts" in sys.argv:
        keep = set(sys.argv[sys.argv.index("--keep-experts") + 1].split(","))
    routed_only = "--routed-only" in sys.argv

    out, moved = [], 0
    for line in open(path, encoding="utf-8").read().splitlines():
        if not line or line.startswith("#") or ":" not in line:
            out.append(line)
            continue
        name, _, dtype = line.partition(":")
        name, dtype = name.strip(), dtype.strip()
        block = name.split(".")[0]
        experts = any(f".{e}." in name for e in EXPS)
        if routed_only:
            experts = experts and "_shexp" not in name
        if experts and block not in keep:
            if dtype != kind:
                moved += 1
            dtype = kind
        out.append(f"{name}: {dtype}")
    print("\n".join(out))
    print(f"# {moved} expert tensors moved to {kind}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
