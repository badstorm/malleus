#!/bin/sh
# llama.cpp baseline on the local GPU (needs the GPU free). Usage: bench/baseline-llamacpp.sh <dir-with-gguf> <gguf-name>
# Env: PROMPTS, GEN, LLAMA_ARGS (extra llama-bench flags, e.g. "-d 0,131072").
set -e
DIR="$1"; GGUF="$2"
RENDER_NODE="${MALLEUS_RENDER_NODE:-/dev/dri/renderD129}"
IMAGE="${LLAMACPP_IMAGE:-ghcr.io/stew675/llama-cpp-rdna-boosts:latest}"
sudo podman run --rm --device /dev/kfd --device "$RENDER_NODE" --group-add keep-groups --security-opt label=disable \
  -v "$DIR":/m:ro --entrypoint /app/llama "$IMAGE" bench -m "/m/$GGUF" -ngl 99 -fa 1 \
  -p "${PROMPTS:-512,1024,8192}" -n "${GEN:-128}" -r 1 -o md ${LLAMA_ARGS}
