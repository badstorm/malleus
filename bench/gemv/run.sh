#!/bin/sh
# Builds and runs the GEMV microbenchmark in the dev container. Usage: bench/gemv/run.sh [variant filter]
set -e
cd "$(dirname "$0")/../.."
exec scripts/dev.sh sh -c "hipcc -O3 --offload-arch=\${MALLEUS_ARCH:-gfx1150} -std=c++17 -o /target/gemv_bench bench/gemv/bench.hip && /target/gemv_bench $*"
