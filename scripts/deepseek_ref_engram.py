#!/usr/bin/env python3
"""The Engram forward of one layer, on the release's own module and our own table.

The release runs its 510 GB fp8 tables; our `.malleus` holds the same rows at 4 bits (RowI4, one f16 scale per
row). So this reads the rows from our companion file — decoded here, independently of the loader — feeds them to
the release's `Engram`, and writes the input and the output of the gate. The comparison is then about the graph
(`wkv`, the per-copy normalisation, the signed sqrt and the sigmoid), not about the table's precision.

    python3 scripts/deepseek_ref_engram.py --checkpoint /mnt/usb/deepseekv41-flash \
        --reference /mnt/usb/deepseekv41-flash/inference \
        --table /srv/containers/aitools/models/deepseek-v41-flash-engram.malleus \
        --hash /srv/containers/aitools/golden/deepseek-engram-hash --layer 1 --out /golden/deepseek-ref/engram1
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


class Table:
    """Random access to the Engram tables of a `.malleus` file: the rows are 4-bit nibbles, one f16 scale each."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", self.f.read(24))
        if magic != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        align = 32
        for _ in range(n_kv):
            key = self._str()
            kind = struct.unpack("<I", self.f.read(4))[0]
            value = self._value(kind)
            if key == "general.alignment":
                align = int(value)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            dims = struct.unpack("<I", self.f.read(4))[0]
            shape = [struct.unpack("<Q", self.f.read(8))[0] for _ in range(dims)]
            kind = struct.unpack("<I", self.f.read(4))[0]
            offset = struct.unpack("<Q", self.f.read(8))[0]
            # GGUF writes the dimensions fastest-varying first, so `[k, rows]` is a `[rows, k]` tensor.
            self.tensors[name] = (list(reversed(shape)), kind, offset)
        head = self.f.tell()
        self.data = head + (-head % align)

    def _str(self) -> str:
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode()

    def _value(self, kind: int):
        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if kind == 8:
            return self._str()
        if kind == 9:
            inner = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            return [self._value(inner) for _ in range(n)]
        return struct.unpack(scalar[kind], self.f.read(struct.calcsize(scalar[kind])))[0]

    def rows(self, name: str, rows: np.ndarray, k: int) -> np.ndarray:
        """The rows as f32, `[len(rows), k]`: `k/2` bytes of nibbles at the row's offset, the scale after them."""
        shape, kind, offset = self.tensors[name]
        if shape != [shape[0], k] or kind != 0x4D4C0014:
            raise SystemExit(f"{name}: {shape} type {kind:#x}, expected a RowI4 [{shape[0]}, {k}]")
        n, packed = shape[0], (k + 1) // 2
        scale_at, _, scale_off = self.tensors[f"{name}.scales"]
        out = np.empty((len(rows), k), np.float32)
        for i, r in enumerate(rows):
            if not 0 <= r < n:
                raise SystemExit(f"{name}: row {r} outside the table's {n}")
            self.f.seek(self.data + offset + int(r) * packed)
            nib = np.frombuffer(self.f.read(packed), np.uint8)
            self.f.seek(self.data + scale_off + int(r) * 2)
            scale = np.frombuffer(self.f.read(2), "<f2")[0].astype(np.float32)
            # Low nibble first, stored as level + 8 (offset binary, not two's complement).
            vals = np.empty(2 * packed, np.float32)
            vals[0::2] = (nib & 0x0F).astype(np.float32) - 8.0
            vals[1::2] = (nib >> 4).astype(np.float32) - 8.0
            out[i] = vals[:k] * scale
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, help="the directory holding model.py")
    ap.add_argument("--table", required=True, type=Path, help="the companion file with the Engram tables")
    ap.add_argument("--hash", required=True, type=Path, help="the `<out>` of deepseek_engram_state.py plus its hash dump")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from deepseek_ref import install_kernels

    # The release's `kernel.py` calls tilelang, which is not installed and not needed: the two kernels the graph
    # touches are the torch versions the tiny reference already carries.
    install_kernels()
    from deepseek_ref_layer import Checkpoint

    sys.path.insert(0, str(a.reference))
    import model as ref
    from engram import EngramLayout

    t = json.loads((a.checkpoint / "config.json").read_text())["text_config"]
    dim, hc = t["hidden_size"], t["hc_mult"]
    layer_ids = tuple(t["engram_layer_ids"])
    position = layer_ids.index(a.layer)

    manifest = json.loads(Path(f"{a.hash}.json").read_text())
    ids = manifest["ids"]
    shape = manifest["shape"]
    cols = shape[3]
    # The dump is written table after table (that is how the gate reads it), not position after position.
    hashes = np.fromfile(f"{a.hash}.i64", dtype="<i8").reshape(shape[2], shape[1], cols)[position]
    tokens, _ = hashes.shape

    table = Table(a.table)
    values = table.rows(
        f"layers.{a.layer}.engram.embed.weight", hashes.reshape(-1), t["engram_head_dim"]
    ).reshape(1, tokens, cols, t["engram_head_dim"])

    args = ref.ModelArgs(
        dim=dim,
        hc_mult=hc,
        norm_eps=t["rms_norm_eps"],
        engram_layer_ids=layer_ids,
        engram_num_embeddings=tuple(t["engram_num_embeddings"]),
        engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_n_heads=t["engram_n_heads"],
        engram_head_dim=t["engram_head_dim"],
    )
    layout = EngramLayout(
        max_ngram_size=t["engram_max_ngram_size"],
        layer_ids=layer_ids,
        num_embeddings=tuple(t["engram_num_embeddings"]),
        primes=(),
        n_heads=t["engram_n_heads"],
        head_dim=t["engram_head_dim"],
    )

    # A 98 GB table per layer is not something to allocate: the module only ever calls `embed` for the rows, and
    # those come from our file.
    class Rows(torch.nn.Module):
        def __init__(self, num_embeddings: int, dim: int):
            super().__init__()
            self.rows = None

        def forward(self, indices):
            return self.rows

    ref.ParallelEngramEmbedding = Rows
    torch.set_default_dtype(torch.float32)
    e = ref.Engram(args, a.layer, layout).float()
    ck = Checkpoint(a.checkpoint)
    state = e.state_dict()
    for name in ("wkv.weight", "q_weight", "k_weight"):
        src = ck.f32(f"layers.{a.layer}.engram.{name}")
        if tuple(src.shape) != tuple(state[name].shape):
            raise SystemExit(f"{name}: checkpoint {src.shape} vs module {tuple(state[name].shape)}")
        state[name].copy_(torch.from_numpy(src.reshape(state[name].shape)))
    e.embed.rows = torch.from_numpy(values)

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(1, tokens, hc, dim, generator=g) * 0.02
    dumps = {}
    e.embed.register_forward_hook(lambda _m, _i, o: dumps.setdefault("rows", o))
    e.wkv.register_forward_hook(lambda _m, _i, o: dumps.setdefault("kv", o))
    with torch.inference_mode():
        out = e(x, torch.from_numpy(hashes))

    dumps["x"], dumps["out"] = x, out
    a.out.mkdir(parents=True, exist_ok=True)
    for name, arr in dumps.items():
        arr.detach().float().reshape(-1).numpy().astype("<f4").tofile(str(a.out / f"{name}.f32"))
    (a.out / "manifest.json").write_text(
        json.dumps(
            {
                "layer": a.layer,
                "position": position,
                "ids": ids,
                "hidden": dim,
                "hc": hc,
                "tokens": tokens,
                "shapes": {k: list(v.shape) for k, v in dumps.items()},
            },
            indent=1,
        )
        + "\n"
    )
    print(json.dumps({"layer": a.layer, "tokens": tokens, "cols": cols, "table": str(a.table)}))


if __name__ == "__main__":
    main()
