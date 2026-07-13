#!/usr/bin/env bash
# Build the vendored emulator core (libgb.so). Run once before using the
# package barebone; the Dockerfile runs the same step inside the image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
make -C "$HERE/vendor/rlgb" ${ARCH:+ARCH="$ARCH"}
echo "built: $HERE/vendor/rlgb/rlgb/libgb.so"
