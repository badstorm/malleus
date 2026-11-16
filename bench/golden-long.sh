#!/bin/sh
# Long-context quality gate: an 11 000-token QSA prompt against a bf16 layer-by-layer reference.
#
# The reference blob is not in the repo (like the short golden blobs): generate it once with
#
#   MALLEUS_MODELS=/srv/containers/aitools/models MALLEUS_GOLDEN=/srv/containers/aitools/golden \
#   MALLEUS_GOLDEN_IMAGE=malleus-golden:gfx1151 \
#   scripts/golden.sh python3 scripts/ref_layerwise.py --model /models/extra/Qwen3.8-Flash-Next \
#     --out /golden/long-11000 --blob-dir /golden/long-11000/blobs --long 11000 --attn eager --dump ""
#
# then copy blobs/text/long-11000/logits_last.f32 into the golden blobs tree
# and merge the long-11000 entry into golden/qwen38-flash-next/golden-long.json.
#
#   bench/golden-long.sh
#
# The gate fails on KL/argmax drift like `malleus golden`; the golden runner prints the per-position agreement.
set -e
cd "$(dirname "$0")/.."
MODELS=${MALLEUS_MODELS:-/srv/llm/models}
MODEL=${MALLEUS_LONG_MODEL:-$(ls "$MODELS"/*.malleus "$MODELS"/*/*.malleus 2>/dev/null | head -1)}
[ -n "$MODEL" ] || { echo "no .malleus model under $MODELS; set MALLEUS_LONG_MODEL" >&2; exit 1; }
case "$MODEL" in "$MODELS"/*) MODEL=/models/${MODEL#"$MODELS"/};; esac
scripts/dev.sh sh -c "MALLEUS_DENSE_W4A4=1 /target/release/malleus golden \
  --model '$MODEL' \
  --golden /src/golden/qwen38-flash-next/golden-long.json \
  --blobs /golden/qwen38-flash-next --ctx 16384"
