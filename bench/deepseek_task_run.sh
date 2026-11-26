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
#
# The container is the *dev* one with the binary this tree built (`MALLEUS_BIN`, extracted from the build volume and
# mounted: the ROCm libraries live in the image, the binary does not): the release image is older than the file
# formats these experiments produce — it does not read MLS2 at all — and the point of the run is the code in review.
set -e
MODEL="${1:?usage: $0 <model.malleus> <answers.json> [max tokens]}"
OUT="${2:?usage: $0 <model.malleus> <answers.json> [max tokens]}"
OUTNAME="$(basename "$OUT")"
TOKENS="${3:-256}"
PORT=8092
# The catalog is the model's own directory and the model is named by its file stem: two DeepSeek files in one
# directory would leave the server choosing, and the point of the run is which file answered.
DIR="$(cd "$(dirname "$MODEL")" && pwd)"
NAME="$(basename "$MODEL" .malleus)"

# Whatever happens — a failed load, an interrupted run — production comes back.
trap 'systemctl --user start malleus.service || true' EXIT
systemctl --user stop malleus.service || true

# The whole run lives in one container: the server, the wait for it, and the harness. Publishing a port and driving
# both sides from the host cost an afternoon of orchestration that fails in ways the log does not show.
podman rm -f malleus-task 2>/dev/null || true
podman run --rm --name malleus-task \
  --device /dev/kfd --device /dev/dri/renderD128 --group-add keep-groups --security-opt label=disable \
  --ipc=host --pids-limit=-1 \
  -v "$DIR":/models:ro -v /srv/containers/aitools/models:/pool:ro -v "$PWD":/src:ro \
  -v "${MALLEUS_GOLDEN:-$HOME/golden}:/golden" -v "${MALLEUS_BIN:-/tmp/malleus-bin}:/malleus:ro" \
  -e MALLEUS_MODELS_DIR=/models -e MALLEUS_DEFAULT_MODEL="$NAME" \
  ${MALLEUS_DEEPSEEK_ENGRAM:+-e MALLEUS_DEEPSEEK_ENGRAM=$MALLEUS_DEEPSEEK_ENGRAM} \
  -e MALLEUS_DEEPSEEK_ENGRAM_STATE=/golden/deepseek-engram \
  -w /src "${MALLEUS_DEV_IMAGE:-malleus-dev:latest}" sh -c "
    /malleus serve --models-dir /models --host 127.0.0.1 --port 8092 --ctx 4096 > /tmp/serve.log 2>&1 &
    for i in \$(seq 240); do
      curl -sf http://127.0.0.1:8092/health >/dev/null 2>&1 && break
      sleep 10
    done
    curl -sf http://127.0.0.1:8092/health >/dev/null || { tail -20 /tmp/serve.log; exit 1; }
    python3 bench/task_compare.py run http://127.0.0.1:8092 --out /golden/$OUTNAME --max-tokens $TOKENS --greedy
  "
echo "answers in $OUT"
