#!/bin/sh
# The task set against a DeepSeek-V4.1 file, served from this box by the host graph (issue #134).
#
# The host graph answers one position every ~6 s, so this is an overnight job: the whole set at 256 tokens a task
# is about three and a half hours. Nothing else can use the machine meanwhile — the graph holds 32.8 GB of blocks
# and 45 GB of expert cache, and the production unit has to be stopped.
#
#   bench/deepseek_task_run.sh /models/deepseek-v41-flash-mls2.malleus answers-deepseek-2bit.json
#   bench/deepseek_task_run.sh /models/deepseek-1bit.malleus         answers-deepseek-1bit.json
#
# The Engram state is the one `scripts/deepseek_engram_state.py` wrote once (`/golden/deepseek-engram`); the tables
# are the model's `-engram.malleus` sibling unless MALLEUS_DEEPSEEK_ENGRAM says otherwise.
set -e
MODEL="${1:?usage: $0 <model.malleus> <answers.json> [max tokens]}"
OUT="${2:?usage: $0 <model.malleus> <answers.json> [max tokens]}"
TOKENS="${3:-256}"
DEEPSEEK_FILE="${MALLEUS_DEEPSEEK_FILE:-$MODEL}"
PORT=8092

systemctl --user stop malleus.service || true
podman rm -f malleus-task 2>/dev/null || true
podman run -d --rm --name malleus-task \
  --device /dev/kfd --device /dev/dri/renderD128 --group-add keep-groups --security-opt label=disable \
  -p "$PORT:8092" --ipc=host --pids-limit=-1 \
  -v /srv/containers/aitools/models:/models:ro -v "${MALLEUS_GOLDEN:-$HOME/golden}:/golden" \
  -e MALLEUS_MODELS_DIR=/models -e MALLEUS_DEEPSEEK_ENGRAM_STATE=/golden/deepseek-engram \
  ghcr.io/duckpage/malleus:v0.1.3 serve --models-dir /models --host 0.0.0.0 --port 8092 --ctx 4096

# The host graph reads 26 GB of blocks and dequantises them before it answers: a minute or two.
for _ in $(seq 60); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { podman logs malleus-task | tail -20; exit 1; }

python3 bench/task_compare.py run "http://localhost:$PORT" --out "$OUT" --max-tokens "$TOKENS" --greedy
podman stop malleus-task >/dev/null 2>&1 || true
systemctl --user start malleus.service || true
echo "answers in $OUT"
