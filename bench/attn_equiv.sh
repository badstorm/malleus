#!/bin/sh
# Equivalence and speed of the two prefill attention paths on a prompt long enough to engage the gather kernel.
#
#   bench/attn_equiv.sh [repetitions]
#
# The gather path kicks in at len + t >= 5 * indexer budget (10240 tokens for Qwen3.8-Flash-Next), and no golden
# entry reaches it: the longest is 4500 tokens. The numerical gate for these kernels was a kernel-vs-CPU test
# (removed from src/ and tracked in #4); this script is the end-to-end half. It compares the default one-query-per-
# block kernel (`attention_gather_q1`) with the per-query kernel, which differ only in accumulation order.
#
# Read the text diff as a smoke test only. Greedy decoding is chaotic in the prefill logits: two provably
# equivalent gather kernels, differing by 1e-4 per attention element, produced 8 differing words out of 39 here.
# So a handful of words proves nothing either way -- run a numerical test for that. What this script does catch
# is a path that crashes, picks the wrong kernel, or drifts far enough to stop making sense.
#
# MALLEUS_GATHER_Q1=1: one query per block (default on gfx1151); MALLEUS_ATTN_GATHER=1: per-query kernel.
set -e
cd "$(dirname "$0")/.."
REP=${1:-430}
CTX=16384
TOK=48
# The prompt has to be visible inside the dev container, and the only thing mounted read-write there is the repo,
# so it is written next to the sources and removed on the way out.
PROMPT=.attn-equiv-prompt.txt
trap 'rm -f "$PROMPT"' EXIT INT TERM

python3 - "$PROMPT" "$REP" <<'PY'
import pathlib, sys
base = [
    "The committee met on Monday to discuss the budget for the new library, and the minutes were circulated on Friday.",
    "Sparse attention selects blocks of keys with a learned indexer, so the query only reads a fraction of the cache.",
    "In 1969 humans first landed on the Moon, and the guidance computer had less memory than a modern thumbnail image.",
    "def mean(xs): return sum(xs) / len(xs)  # the arithmetic mean of a non-empty sequence of numbers",
    "Il codice fiscale italiano è composto da sedici caratteri alfanumerici derivati dai dati anagrafici della persona.",
    "A quantized weight tensor keeps a scale per group, and the group size trades accuracy against memory traffic.",
]
out = [f"[{i}] {base[i % len(base)]}" for i in range(int(sys.argv[2]))]
pathlib.Path(sys.argv[1]).write_text(
    " ".join(out) + "\n\nQuestion: summarise the passage above in one sentence.")
PY

# dev.sh only forwards CARGO_TARGET_DIR, so the switch has to be set inside the container with env(1); setting it
# in front of dev.sh leaves it in the host shell and both arms silently run the same path.
seen=$(scripts/dev.sh env MALLEUS_GATHER_Q1=probe sh -c 'echo "$MALLEUS_GATHER_Q1"')
[ "$seen" = probe ] || { echo "MALLEUS_GATHER_Q1 does not reach the container"; exit 1; }

for g in q1 per; do
    if [ "$g" = q1 ]; then envs="MALLEUS_GATHER_Q1=1"; else envs="MALLEUS_ATTN_GATHER=1 MALLEUS_GATHER_Q1=0"; fi
    printf 'gather=%s: ' "$g"
    scripts/dev.sh env $envs sh -c \
        "/target/release/malleus run --model \"\$(ls /models/*.malleus | head -1)\" --ctx $CTX \
         --max-tokens $TOK --prompt \"\$(cat /src/$PROMPT)\" 2>&1 | tail -2" \
        | tee "/tmp/attn_equiv_$g.txt" | grep -o 'TTFT.*' || true
done

echo "--- testo generato ---"
python3 - /tmp/attn_equiv_q1.txt /tmp/attn_equiv_per.txt "$TOK" <<'PY2'
import difflib, sys
a = open(sys.argv[1]).readline().split()
b = open(sys.argv[2]).readline().split()
n = int(sys.argv[3])
same = sum(1 for op in difflib.SequenceMatcher(None, a, b).get_matching_blocks() for _ in range(op.size))
print(f"q1 {len(a)} parole, per-query {len(b)}, in comune {same} su {n} passi greedy")
if same == len(a) == len(b):
    print("identici")
else:
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            print(f"  {tag}: q1 {' '.join(a[i1:i2])!r} | per-query {' '.join(b[j1:j2])!r}")
    print("Atteso: le due strade accumulano in ordine diverso e il greedy amplifica. Se il testo resta")
    print("plausible but concludes nothing -- the numeric gate is the kernel-vs-CPU test tracked in #4.")
PY2
