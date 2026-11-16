#!/bin/sh
# Copies the shared libraries the binary needs from the ROCm SDK into $2/lib, plus those HIP loads with dlopen.
set -eu
BIN="$1"; OUT="$2"
mkdir -p "$OUT/lib"
: > "$OUT/RUNTIME-LIBS.txt"
EXTRA="libamd_comgr.so libhsa-runtime64.so librocprofiler-register.so"
list() {
  ldd "$1" | awk '/=>/ { print $3 }' | grep -E '^/opt/rocm|_rocm_sdk|rocm_sysdeps' || true
}
{
  list "$BIN"
  for e in $EXTRA; do
    f=$(find -L /opt/rocm/lib -maxdepth 1 -name "$e*" | head -1)
    [ -n "$f" ] && { echo "$f"; list "$f"; }
  done
} | sort -u | while read -r f; do
  [ -z "$f" ] && continue
  real=$(readlink -f "$f")
  cp -L "$real" "$OUT/lib/$(basename "$f")"
  [ "$(basename "$real")" != "$(basename "$f")" ] && cp -L "$real" "$OUT/lib/$(basename "$real")"
  echo "$(basename "$f") $(sha256sum "$real" | cut -c1-16)" >> "$OUT/RUNTIME-LIBS.txt"
done
for f in "$OUT"/lib/*.so*; do
  LD_LIBRARY_PATH="$OUT/lib" ldd "$f" | grep -q "not found" && { echo "missing deps for $f"; LD_LIBRARY_PATH="$OUT/lib" ldd "$f" | grep "not found"; exit 1; }
done
LD_LIBRARY_PATH="$OUT/lib" ldd "$BIN" | grep -q "not found" && { echo "missing deps for $BIN"; exit 1; }
du -sh "$OUT/lib"
