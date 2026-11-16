#!/bin/sh
# Runs an offline reference tool (golden / eval / calibration) in the golden container: torch + transformers on the
# GPU, models read-only in /models and blobs writable in /golden. Usage: scripts/golden.sh scripts/golden.py ...
set -e
cd "$(dirname "$0")/.."
RENDER_NODE="${MALLEUS_RENDER_NODE:-/dev/dri/renderD129}"
MODELS="${MALLEUS_MODELS:-/srv/llm/models}"
GOLDEN="${MALLEUS_GOLDEN:-/srv/llm/golden}"
IMAGE="${MALLEUS_GOLDEN_IMAGE:-malleus-golden:latest}"
# Command to run before the tool, e.g. to install an optional reference dependency in the ephemeral container.
SETUP="${MALLEUS_GOLDEN_SETUP:-}"
mkdir -p "$GOLDEN"
exec ${MALLEUS_SUDO-sudo} podman run --rm -i --device /dev/kfd --device "$RENDER_NODE" --group-add keep-groups \
  --security-opt label=disable \
  -v "$PWD":/src -v "$MODELS":/models:ro -v "$GOLDEN":/golden \
  -w /src "$IMAGE" sh -c "${SETUP:+$SETUP && }exec \"\$@\"" sh "$@"
