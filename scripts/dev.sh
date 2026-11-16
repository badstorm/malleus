#!/bin/sh
# Runs a command in the dev container with the GPU. Usage: scripts/dev.sh cargo build --release
set -e
cd "$(dirname "$0")/.."
RENDER_NODE="${MALLEUS_RENDER_NODE:-/dev/dri/renderD129}"
MODELS="${MALLEUS_MODELS:-/srv/llm/models}"
GOLDEN="${MALLEUS_GOLDEN:-/srv/llm/golden}"
IMAGE="${MALLEUS_DEV_IMAGE:-malleus-dev:latest}"
mkdir -p "$GOLDEN"
exec ${MALLEUS_SUDO-sudo} podman run --rm -i --device /dev/kfd --device "$RENDER_NODE" --group-add keep-groups \
  --security-opt label=disable \
  -v "$PWD":/src -v malleus-cargo:/root/.cargo/registry -v malleus-target:/target \
  -v "$MODELS":/models:ro -v "$GOLDEN":/golden \
  -e CARGO_TARGET_DIR=/target -w /src "$IMAGE" "$@"
