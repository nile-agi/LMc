#!/usr/bin/env bash
# build.sh — Cross-compilation helper for lmc.
#
# Usage:
#   ./build.sh              (native, auto-detect)
#   ./build.sh native       (x86-64 or ARM native)
#   ./build.sh arm64        (aarch64, cross from x86 Linux)
#   ./build.sh arm32        (armv7, e.g. Raspberry Pi 32-bit)
#   ./build.sh windows-mingw  (x86-64 Windows via MinGW)
#   ./build.sh android      (ARM64 Android via NDK)

set -euo pipefail

SRCDIR="src"
SRCS="$SRCDIR/utils.c $SRCDIR/models.c $SRCDIR/quant.c \
      $SRCDIR/gguf.c  $SRCDIR/ops.c    $SRCDIR/tokenizer.c $SRCDIR/lmc.c"

BASE_FLAGS="-std=c99 -O3 -ffast-math -funroll-loops -Wall -I$SRCDIR"
LIBS="-lm"

TARGET="${1:-native}"
OUT="lmc"

case "$TARGET" in
  native)
    echo "[build] Native ($(uname -m))"
    CC="${CC:-gcc}"
    FLAGS="$BASE_FLAGS -march=native"
    ;;

  arm64|aarch64)
    echo "[build] Cross: aarch64 (ARM64 Linux / Raspberry Pi 4)"
    CC="${CC:-aarch64-linux-gnu-gcc}"
    FLAGS="$BASE_FLAGS -march=armv8-a+fp+simd"
    OUT="lmc-arm64"
    ;;

  arm32|armv7)
    echo "[build] Cross: armv7 (Raspberry Pi 2/3, Android Termux 32-bit)"
    CC="${CC:-arm-linux-gnueabihf-gcc}"
    FLAGS="$BASE_FLAGS -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
    OUT="lmc-arm32"
    ;;

  windows-mingw)
    echo "[build] Cross: x86-64 Windows (MinGW)"
    CC="${CC:-x86_64-w64-mingw32-gcc}"
    FLAGS="$BASE_FLAGS -march=x86-64"
    OUT="lmc.exe"
    LIBS="$LIBS -lws2_32"
    ;;

  android)
    # Requires Android NDK in PATH. Set ANDROID_NDK env var.
    ANDROID_NDK="${ANDROID_NDK:-$HOME/android-ndk}"
    TOOLCHAIN="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64"
    echo "[build] Android ARM64 (API 21+)"
    CC="$TOOLCHAIN/bin/aarch64-linux-android21-clang"
    FLAGS="$BASE_FLAGS -march=armv8-a"
    OUT="lmc-android-arm64"
    ;;

  omp)
    echo "[build] Native + OpenMP"
    CC="${CC:-gcc}"
    FLAGS="$BASE_FLAGS -march=native -fopenmp"
    LIBS="$LIBS"
    OUT="lmc_omp"
    ;;

  *)
    echo "Unknown target: $TARGET"
    echo "Valid: native arm64 arm32 windows-mingw android omp"
    exit 1
    ;;
esac

echo "[build] CC=$CC"
echo "[build] FLAGS=$FLAGS"
echo "[build] OUT=$OUT"
set -x
# shellcheck disable=SC2086
$CC $FLAGS $SRCS -o "$OUT" $LIBS
set +x
echo "[build] ✓  $OUT"
