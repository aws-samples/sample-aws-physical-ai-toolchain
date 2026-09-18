#!/usr/bin/env bash
# Shared by the full image and the incremental decoder repair.
set -euo pipefail
: "${FFMPEG_PREFIX:?FFMPEG_PREFIX is required}"
build_dir="$(mktemp -d /tmp/vla-ffmpeg.XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT
git clone --depth 1 --branch n7.0.2 https://git.ffmpeg.org/ffmpeg.git "$build_dir/ffmpeg"
cd "$build_dir/ffmpeg"
./configure --prefix="$FFMPEG_PREFIX" \
  --enable-shared --disable-static \
  --enable-gpl --enable-libx264 --enable-libx265 --enable-libdav1d \
  --disable-doc --disable-programs
make -j"$(nproc)"
make install
