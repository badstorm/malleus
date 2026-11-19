# Golden reference for Hy4-flash (issue #40): the `malleus-golden` image plus the `transformers` release that carries
# `hy_v4_flash`, pinned to the commit instead of resolved from the branch every run. Build with
#
#   podman build -t malleus-golden:hy4 -f deploy/golden-hy4.Containerfile deploy
#
# and point scripts/golden.sh at it: MALLEUS_GOLDEN_IMAGE=malleus-golden:hy4.
#
# The commit is the head of huggingface/transformers#50903, which adds `models/hy_v4_flash` on top of `hy_v4` and
# `hy_v4_vl`. Bump it when Hy4-flash lands upstream, and drop this file then: one `malleus-golden` is enough once the
# released `transformers` knows the model.
FROM localhost/malleus-golden:latest
ARG HY4_TRANSFORMERS_COMMIT=b865850d05326cf20e306b31f14b0027de802da3
RUN pip install --no-cache-dir \
      "transformers @ git+https://github.com/huggingface/transformers.git@${HY4_TRANSFORMERS_COMMIT}"
