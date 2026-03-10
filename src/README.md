# lmc — Lightweight Model Engine

A fast, portable C99 LLM inference engine.  
GGUF-native · zero external dependencies · runs on x86, ARM, Android, Windows.

```
./lmc --model models/gpt2-xl.gguf \
      --prompt "The meaning of life is" \
      --n-predict 200 --temp 0.8 --threads 4
```

---

## Features

| | |
|---|---|
| **Architectures** | GPT-2 (Small / Medium / Large / XL), LLaMA / Mistral / Qwen family |
| **File format** | GGUF v1–v3 (native); custom `.bin` (backward-compat) |
| **Quantisation** | F32, F16, Q2\_K, Q3\_K, Q4\_0, Q4\_1, Q4\_K, Q5\_0, Q5\_1, Q5\_K, Q6\_K, Q8\_0, IQ3\_XXS, IQ3\_S, IQ4\_XS |
| **Parallelism** | Optional OpenMP; single-threaded build also available |
| **Platforms** | Linux, macOS, Windows (MinGW), Raspberry Pi, Android (NDK) |
| **Dependencies** | None beyond `-lm`; no BLAS, no Python |

---

## Quick Start

### 1. Build

```bash
make            # single-threaded, optimised
make omp        # OpenMP multi-threaded (recommended for desktop)
make help       # list all targets
```

macOS: `brew install libomp` is required for the `omp` target.  
Windows: use an MSYS2 / MinGW64 terminal.

### 2. Get a model

```bash
# GPT-2 Large Q4_K_M (~520 MB) from HuggingFace
wget -P models/ \
  https://huggingface.co/ggml-org/gpt2-large-GGUF/resolve/main/gpt2-large-q4_k_m.gguf

# Required tokenizer files for GPT-2
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
```

LLaMA / Mistral GGUF files are available from [HuggingFace](https://huggingface.co/models?library=gguf).

### 3. Run

```bash
./lmc_omp \
  --model   models/gpt2-large-q4_k_m.gguf \
  --prompt  "Once upon a time" \
  --n-predict 200 --temp 0.8 --threads 4
```

lmc auto-detects the first `.gguf` file in `models/` if `--model` is omitted.

---

## Command-Line Reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | auto-detect | Path to `.gguf` or `.bin` model file |
| `--prompt` | `-p` | `"Hello, world!"` | Input text prompt |
| `--n-predict` | `-n` | `128` | Maximum tokens to generate |
| `--temp` | `-t` | `0.7` | Sampling temperature (0 = greedy) |
| `--top-p` | | `0.9` | Nucleus sampling probability cutoff |
| `--threads` | `-j` | all cores | OpenMP thread count |
| `--encoder` | | `encoder.json` | GPT-2 encoder vocab path |
| `--bpe` | | `vocab.bpe` | GPT-2 merge rules path |
| `--seed` | | `42` | RNG seed for reproducibility |
| `--help` | `-h` | | Print usage and exit |

---

## Model Support

### GPT-2 family

| Variant | Params | Layers | Heads | Dim  |
|---------|--------|--------|-------|------|
| Small   | 124 M  | 12     | 12    | 768  |
| Medium  | 345 M  | 24     | 16    | 1024 |
| Large   | 774 M  | 36     | 20    | 1280 |
| XL      | 1.5 B  | 48     | 25    | 1600 |

Tokenizer: OpenAI BPE (`encoder.json` + `vocab.bpe`, 50 257 tokens).

### LLaMA / Mistral family (via ARCH\_LLAMA)

Models with `general.architecture = llama | mistral | qwen2` in their GGUF
metadata are loaded automatically. The engine supports:

- **LLaMA-2** 7B / 13B / 70B
- **LLaMA-3** 8B / 70B
- **Mistral** 7B / 8×7B (MoE layers treated as dense for now)
- **Qwen-2** 0.5B / 1.5B / 7B

Features active for ARCH\_LLAMA: RMSNorm, RoPE, grouped-query attention (GQA),
SwiGLU feed-forward network.

---

## Performance

Numbers are approximate; measured with `make bench` on single-socket hardware.

| Model | Quant | Device | Threads | t/s |
|-------|-------|--------|---------|-----|
| GPT-2 Small 124 M | F16 | i7-1165G7 | 8 | ~120 |
| GPT-2 Large 774 M | Q4\_K\_M | i7-1165G7 | 8 | ~28 |
| GPT-2 XL 1.5 B | Q4\_K\_M | i7-1165G7 | 8 | ~14 |
| GPT-2 Small 124 M | Q4\_K\_M | Raspberry Pi 4 | 4 | ~8 |
| GPT-2 Large 774 M | Q4\_K\_M | Raspberry Pi 4 | 4 | ~2.1 |
| LLaMA-2 7 B | Q4\_K\_M | i7-1165G7 | 8 | ~7 |
| Mistral 7 B | Q4\_K\_M | i7-1165G7 | 8 | ~7 |

For edge/embedded optimisation notes see [docs/edge_optimizations.md](docs/edge_optimizations.md).

---

## Project Layout

```
lmc/
├── src/
│   ├── lmc.c           Main entry point, arg parsing, generation loop, sampling
│   ├── ops.h / ops.c   Kernels: matmul, gelu/silu, layer_norm/rms_norm,
│   │                   RoPE, attention (GPT-2 + LLaMA/GQA), transformer forward
│   ├── models.h / .c   ModelConfig (arch, dims, GQA), LayerWeights, KVCache,
│   │                   Activations, weight-pointer assignment, RoPE cache init
│   ├── quant.h / .c    All 13 dequantisation kernels + codebook tables
│   ├── gguf.h / .c     GGUF v1–3 parser, .bin loader, format detection,
│   │                   tensor name → weight-pointer dispatch (GPT-2 + LLaMA)
│   ├── tokenizer.h/.c  GPT-2 BPE tokenizer (encoder.json + vocab.bpe)
│   └── utils.h / .c    Arena allocator, logging macros, portability helpers
├── benchmarks/
│   ├── bench.c         Token/s and peak-RAM benchmark binary
│   └── results.md      Hardware comparison table
├── docs/
│   ├── adding_new_model.md    How to add LLaMA, Phi, Gemma, …
│   ├── adding_new_quant.md    How to add a new quantisation type
│   └── edge_optimizations.md  ARM NEON, Android NDK, battery-aware tips
├── models/             (gitignored) — place .gguf files here
├── Makefile            Linux / macOS / Windows (MinGW)
├── build.sh            Cross-compile: arm64, arm32, windows-mingw, android
└── test.sh             Smoke tests across all .gguf files in models/
```

### Module dependency graph

```
utils ← models ← ops ← lmc
  ↑              ↑
quant ← gguf ──┘
tokenizer ← lmc
```

No circular dependencies.  Every module can be unit-tested in isolation.

---

## Architecture Details

### GPT-2 block (ARCH\_GPT2)

```
x → LayerNorm → [QKV combined] → causal attention → proj → + x
                                                              ↓
  → LayerNorm → FC1(GELU) → FC2 ──────────────────────────→ + x
```

- QKV is a single `[3D × D]` matmul.
- KV cache is head-major: `[layer][head][pos][head_dim]`.

### LLaMA block (ARCH\_LLAMA)

```
x → RMSNorm → Q·Wq, K·Wk, V·Wv → RoPE(Q,K) → GQA attn → Wo → + x
                                                                   ↓
  → RMSNorm → gate·Wg, up·Wu → silu(gate) ⊙ up → down·Wd ──→ + x
```

- Q/K/V are separate projections; K and V use `n_kv_heads ≤ n_heads` (GQA).
- RoPE is applied to Q and K using a precomputed cos/sin table.
- FFN uses SwiGLU: `out = down(silu(gate(x)) ⊙ up(x))`.

---

## Building for Other Platforms

```bash
./build.sh arm64          # AArch64 Linux (cross-compile from x86)
./build.sh arm32          # ARMv7 with NEON
./build.sh android        # Android NDK arm64-v8a
./build.sh windows-mingw  # Windows .exe via MinGW-w64
```

OpenMP is disabled for cross-compile targets by default.

---

## Extending lmc

| Task | Guide |
|------|-------|
| Add a new model architecture (Phi, Gemma, …) | [docs/adding_new_model.md](docs/adding_new_model.md) |
| Add a new quantisation type | [docs/adding_new_quant.md](docs/adding_new_quant.md) |
| Optimise for edge / low-memory hardware | [docs/edge_optimizations.md](docs/edge_optimizations.md) |

### At a glance: adding a new architecture

1. Add `ARCH_*` to `ModelArch` in `models.h`.
2. Add the relevant weight fields to `LayerWeights` / `ModelWeights`.
3. Add metadata key constants to `gguf.h` and parse them in `gguf.c`.
4. Add tensor name mappings in `gguf_name_to_ptr()`.
5. Implement a `<arch>_block_forward()` in `ops.c` and dispatch from
   `transformer_block_forward()` on `CFG_ARCH`.

No existing GPT-2 or LLaMA code needs to change.

---

## License

MIT — see [LICENSE](LICENSE).  Maximum adoption encouraged.
