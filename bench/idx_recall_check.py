#!/usr/bin/env python3
"""Read the MALLEUS_IDX_DUMP_SEL files of one recall fixture and report whether the needle was selected."""
import json
import sys
from pathlib import Path

fixture = json.loads(Path(sys.argv[1]).read_text())
dump = Path(sys.argv[2])
needle_blocks = fixture["needle_blocks"]
q0, q1 = fixture["query_start"], fixture["query_end"]
files = sorted(dump.glob("L*.txt"))
layers_hit = rows_hit = rows = last_hit = 0
for f in files:
    lines = f.read_text().splitlines()
    if not lines:
        continue
    base, qbase, nq, w = map(int, lines[0].split())
    hit = False
    for i, line in enumerate(lines[1:]):
        pos = base + qbase + i
        if not (q0 <= pos < q1):
            continue
        words = [int(x, 16) for x in line.split()]
        ok = any((words[b // 32] >> (b % 32)) & 1 for b in needle_blocks)
        rows += 1
        rows_hit += ok
        hit |= ok
        if pos == q1 - 1:
            last_hit = int(ok)
    layers_hit += hit
print(f"{fixture['name']}: {fixture['tokens']} tok, needle blocks {needle_blocks}, "
      f"query rows {q0}..{q1}")
print(f"  layers selecting the needle: {layers_hit}/{len(files)}")
print(f"  query rows selecting it: {rows_hit}/{rows} ({100.0 * rows_hit / max(rows, 1):.1f}%)")
print(f"  last query row: {'hit' if last_hit else 'MISS'}")
