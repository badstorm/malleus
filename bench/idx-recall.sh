#!/bin/sh
# Indexer recall gate: does the block containing a planted fact survive the sparse selection at 16K/32K/256K?
#
#   bench/idx-recall.sh
#
# The fixtures are token ids (no tokenizer round trip): bench/recall/*.json by default, or the files named by
# MALLEUS_RECALL_FIXTURES. They were generated with bench/recall_prepare.py in the golden container; the filler is
# the same raw source text as the long golden. The context is derived from each fixture's token count; the model
# prefills in chunks of 32768. The model is run with --prompt-ids and MALLEUS_IDX_DUMP_SEL, then the dumped bitsets
# are checked for the needle blocks in the question rows. This is a retrieval test, not a logit/KL test.
#
# The 256K fixtures are not committed (3.7 MB): generate them into the golden volume with
#
#   scripts/golden.sh python3 bench/recall_prepare.py --model /models/extra/Qwen3.8-Flash-Next \
#     --filler-json /golden/long-11000/golden.json --out /golden/recall256/256k.json \
#     --tokens 256000 --needle-fraction 0.33 --name recall-256k
#
# then run with MALLEUS_RECALL_FIXTURES=/srv/containers/aitools/golden/recall256/*.json.
# MALLEUS_IDX_GROUP=N stresses the selection-reuse A/B without changing the fixtures.
set -e
cd "$(dirname "$0")/.."
MODELS=${MALLEUS_MODELS:-/srv/llm/models}
MODEL=${MALLEUS_RECALL_MODEL:-$(ls "$MODELS"/*.malleus "$MODELS"/*/*.malleus 2>/dev/null | head -1)}
[ -n "$MODEL" ] || { echo "no .malleus model under $MODELS; set MALLEUS_RECALL_MODEL" >&2; exit 1; }
case "$MODEL" in "$MODELS"/*) MODEL=/models/${MODEL#"$MODELS"/};; esac
FIXTURES=${MALLEUS_RECALL_FIXTURES:-bench/recall/*.json}
rm -rf .recall-sel
for f in $FIXTURES; do
    TOK=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' "$f")
    CTX=$((TOK + 4096))
    python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
open('.recall-ids', 'w').write(' '.join(map(str, d['ids'])))
PY
    rm -rf .recall-sel
    scripts/dev.sh sh -c "MALLEUS_IDX_GROUP=${MALLEUS_IDX_GROUP:-1} \
      MALLEUS_PREFILL_CHUNK=${MALLEUS_RECALL_CHUNK:-32768} \
      MALLEUS_ROPE_YARN='${MALLEUS_ROPE_YARN:-}' MALLEUS_IDX_DUMP_SEL=/src/.recall-sel \
      /target/release/malleus run --model '$MODEL' --prompt x --prompt-ids /src/.recall-ids \
      --ctx $CTX --max-tokens 1" 2>&1 | grep -E '^prompt|Error' || true
    python3 bench/idx_recall_check.py "$f" .recall-sel
done
rm -f .recall-ids
rm -rf .recall-sel
