#!/bin/sh
# Repeatable GPU baseline (run before/after host changes). Usage: bench/gpu-baseline.sh > out.txt
set -e
cd "$(dirname "$0")/.."
Q4=/srv/containers/aitools/models/hf/disabled/models--unsloth--Qwen3.8-27B-GGUF/blobs/322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482
echo "# $(date -Is) $(hostname) $(uname -r)"
echo "# cmdline: $(cat /proc/cmdline)"
scripts/dev.sh /target/release/malleus bench-bw --gb 4
PROMPTS=512 bench/baseline-llamacpp.sh "$(dirname $Q4)" "$(basename $Q4)" 2>&1 | grep -E '^\| qwen'
scripts/dev.sh sh -c 'MALLEUS_PREFILL_WEIGHTS=none MALLEUS_VISION=0 /target/release/malleus bench --model /models/malleus/qwen38-27b-mls4.malleus --prompt 512 --decode 64' 2>&1 | grep '^pp'
