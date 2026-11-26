#!/usr/bin/env python3
"""How much of a layer's routing a fixed, calibrated set of experts covers — measured on text it was not chosen on.

DeepSeek-V4.1-Flash is 384 experts a layer and a token takes 6 of them (issue #134). If a subset chosen from the
router's own distribution covers most of the traffic, the backbone can be served with the hot set resident and the
rest on disk; if it does not, the experts have to be smaller instead. The first number one computes — the coverage
of the same text the set was chosen on — always looks good (at 192 slots no set can miss itself), so this script
holds one histogram out and evaluates the set chosen on the others.

    bench/router_coverage.py holdout.json fit.json [fit2.json ...]

Each file is `malleus deepseek-run --router <file>`, i.e. `{"counts": [layer][expert]}`. Prints, per resident
budget, how many experts a layer keeps and the median share of the held-out slots they take.
"""
import json
import sys

LAYERS = 40
EXPERTS = 384
TOP_K = 6
# Bytes one routed expert takes at 2 bits (MLS2 with its fp16 scales), the file in use today.
BYTES_PER_EXPERT = 9.4e6


def load(path):
    return json.load(open(path))["counts"]


def share(counts, order, per_layer):
    """The share of a layer's slots the top `per_layer` experts take, per layer, median over layers."""
    out = []
    for layer in range(min(LAYERS, len(counts))):
        slots = sum(counts[layer])
        if slots == 0:
            continue
        out.append(sum(counts[layer][e] for e in order[layer][:per_layer]) / slots)
    out.sort()
    return out[len(out) // 2] if out else 0.0


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    held = load(argv[0])
    fit = load(argv[1])
    for extra in argv[2:]:
        other = load(extra)
        fit = [[a + b for a, b in zip(la, lb)] for la, lb in zip(fit, other)]
    # The set: the most used experts of every layer, as the fit text saw them.
    order = [sorted(range(EXPERTS), key=lambda e: -fit[layer][e]) for layer in range(len(fit))]
    print(f"fit on {sum(map(sum, fit)) // TOP_K} positions, held out {sum(map(sum, held)) // TOP_K}")
    print("  per layer  resident GB  held-out slots covered")
    for per_layer in (32, 64, 96, 128, 160, 192, 232, 256, 320, 384):
        gb = per_layer * BYTES_PER_EXPERT * LAYERS / 1e9
        print(f"  {per_layer:9}  {gb:10.1}  {share(held, order, per_layer) * 100:5.1f} %")
    # The same set scored on the text it was chosen on, for the size of the overfit.
    print(f"  (on the fit text itself: {share(fit, order, 64) * 100:5.1f} % at 64, "
          f"{share(fit, order, 128) * 100:5.1f} % at 128)")


if __name__ == "__main__":
    main(sys.argv[1:])
