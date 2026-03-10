# lmc — Lightweight Model Engine

A single-file-origin, modular C99 LLM inference engine.  
GGUF-native. Zero external dependencies. Runs on x86, ARM, Android, Windows.

```
./lmc --model models/gpt2-xl.gguf \
      --prompt "The meaning of life is" \
      --n-predict 128 --temp 0.7 --threads 4
```

---

## Features

- **GGUF-native** — reads models directly from the llama.cpp / HuggingFace GGUF format
- **All GPT-2 variants** — Small (124M), Medium (345M), Large (774M), XL (1.5B)
- **Rich quantisation support** — F32, F16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ3_S, IQ4_XS
- **Optimised kernels** — 16-wide matmul unroll, head-major KV cache, fast GELU, OpenMP parallelism
- **Edge-friendly** — Raspberry Pi 4 at 8 t/s (GPT-2 Small Q4_K_M, 4 threads)
- **Portable** — Pure C99, no BLAS, no external libs beyond `-lm`

---

## Quick Start

### 1. Build

```bash
make            # single-threaded
make omp        # OpenMP multi-threaded (recommended)
make help       # all targets
```

macOS requires `brew install libomp` for OpenMP.  
Windows: use MSYS2/MinGW64 terminal.

### 2. Download a model (GGUF)

```bash
# GPT-2 Large, Q4_K_M (~520 MB)
wget -P models/ \
  https://huggingface.co/ggml-org/gpt2-large-GGUF/resolve/main/gpt2-large-q4_k_m.gguf

# GPT-2 tokenizer files (required)
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
```

### 3. Run

```bash
./lmc_omp \
  --model   models/gpt2-large-q4_k_m.gguf \
  --prompt  "Once upon a time" \
  --n-predict 200 --temp 0.8 --threads 4
```

---

## Command-Line Reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | auto-detect | Path to `.gguf` or `.bin` model file |
| `--prompt` | `-p` | `"Hello, world!"` | Input text prompt |
| `--n-predict` | `-n` | `128` | Maximum tokens to generate |
| `--temp` | `-t` | `0.7` | Temperature (0 = greedy, 1 = random) |
| `--top-p` | | `0.9` | Nucleus sampling cutoff |
| `--threads` | `-j` | all | Number of OpenMP threads |
| `--encoder` | | `encoder.json` | Path to GPT-2 encoder.json |
| `--bpe` | | `vocab.bpe` | Path to GPT-2 vocab.bpe |
| `--help` | `-h` | | Show usage |

---

## Project Structure

```
lmc/
├── src/
│   ├── lmc.c          Main entry, arg parsing, generation loop, sampling
│   ├── ops.h/c        matmul, gelu, silu, layer_norm, attention, forward pass
│   ├── models.h/c     Config structs, weight pointers, KV cache, activations
│   ├── quant.h/c      Dequantisation kernels for all GGUF quant types
│   ├── gguf.h/c       GGUF parser, .bin loader, format detection
│   ├── tokenizer.h/c  GPT-2 BPE tokenizer (encoder.json + vocab.bpe)
│   └── utils.h/c      Arena allocator, logging macros
├── benchmarks/
│   ├── bench.c        Token/s and RAM benchmark binary
│   └── results.md     Hardware comparison table
├── docs/
│   ├── adding_new_model.md   How to add LLaMA, Mistral, Phi, etc.
│   ├── adding_new_quant.md   How to add new quantisation types
│   └── edge_optimizations.md ARM, Android, Pi, low-memory tips
├── models/            (gitignored) — put your GGUF files here
├── Makefile           Linux / macOS / Windows (MinGW)
├── build.sh           Cross-compile: arm64, arm32, windows-mingw, android
└── test.sh            Smoke tests across all models in models/
```

---

## Performance

| Model             | Quant   | Device            | Threads | t/s   |
|-------------------|---------|-------------------|---------|-------|
| GPT-2 Small 124M  | F16     | i7-1165G7 (x86)   | 8       | ~120  |
| GPT-2 Large 774M  | Q4_K_M  | i7-1165G7 (x86)   | 8       | ~28   |
| GPT-2 XL  1.5B   | Q4_K_M  | i7-1165G7 (x86)   | 8       | ~14   |
| GPT-2 Small 124M  | Q4_K_M  | Raspberry Pi 4    | 4       | ~8    |
| GPT-2 Large 774M  | Q4_K_M  | Raspberry Pi 4    | 4       | ~2.1  |

---

## Extending lmc

- **New architecture** (LLaMA, Mistral, Phi): see [docs/adding_new_model.md](docs/adding_new_model.md)
- **New quant type**: see [docs/adding_new_quant.md](docs/adding_new_quant.md)
- **Edge tuning** (ARM, Android, battery): see [docs/edge_optimizations.md](docs/edge_optimizations.md)

---

## License

MIT — maximum adoption encouraged. See [LICENSE](LICENSE).
