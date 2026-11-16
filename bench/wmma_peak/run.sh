#!/bin/sh
# Builds and runs the WMMA peak-rate microbenchmark in the dev container. Usage: bench/wmma_peak/run.sh [blocks]
set -e
cd "$(dirname "$0")/../.."
exec scripts/dev.sh sh -c "hipcc -O3 --offload-arch=\${MALLEUS_ARCH:-gfx1151} -std=c++17 -o /target/wmma_peak \
  bench/wmma_peak/bench.hip && /target/wmma_peak $*"
