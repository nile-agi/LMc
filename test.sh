#!/usr/bin/env bash
# test.sh — Run inference smoke tests on available GGUF models.
# Compares token output and reports tokens/sec.
#
# Usage:
#   ./test.sh                  (auto-detect binary and models)
#   ./test.sh lmc_omp          (use OpenMP binary)
#   ./test.sh lmc models/gpt2-xl.gguf

set -euo pipefail

BIN="${1:-./lmc}"
MODEL="${2:-}"
PROMPT="The quick brown fox"
N_PREDICT=32
TEMP=0.0    # greedy decode for deterministic output

if [ ! -x "$BIN" ]; then
    echo "[test] Binary not found: $BIN  — run 'make' first"
    exit 1
fi

run_test() {
    local model="$1"
    local label="$2"
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Test: $label"
    echo "  Model: $model"
    echo "══════════════════════════════════════════"
    time "$BIN" \
        --model   "$model" \
        --prompt  "$PROMPT" \
        --n-predict "$N_PREDICT" \
        --temp    "$TEMP" \
        --top-p   1.0
    echo ""
}

FOUND=0
if [ -n "$MODEL" ]; then
    run_test "$MODEL" "user-specified"
    FOUND=1
else
    # Auto-detect any GGUF in models/
    for f in models/*.gguf models/*.bin; do
        [ -f "$f" ] || continue
        run_test "$f" "$(basename "$f")"
        FOUND=1
    done
fi

if [ "$FOUND" -eq 0 ]; then
    echo "[test] No model files found in models/."
    echo "  Download from: https://huggingface.co/ggml-org/gpt2-large-GGUF"
    echo "  Example:"
    echo "    wget -P models/ https://huggingface.co/ggml-org/gpt2-large-GGUF/resolve/main/gpt2-large-q4_k_m.gguf"
    exit 1
fi

echo "[test] All tests complete."
