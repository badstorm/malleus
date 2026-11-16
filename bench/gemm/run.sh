#!/bin/sh
# Builds and runs the GEMM microbenchmark in the dev container. Usage: bench/gemm/run.sh [T] [variant filter]
set -e
cd "$(dirname "$0")/../.."
exec scripts/dev.sh sh -c "hipcc -O3 --offload-arch=\${MALLEUS_ARCH:-gfx1150} -std=c++17 -o /target/gemm_bench bench/gemm/bench.hip && /target/gemm_bench $*"
