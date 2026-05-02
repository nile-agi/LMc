<div align="center">
  <img src="./imgs/lm.c logo.png" alt="LMc" style = "width: 100px; heigth: 150px;">
</div>

<div align="center">
  <p style="margin: 8px 0 16px 0; font-size: 18px; color: #8b949e; font-weight: 500;">
    Local Model computing
  </p>
  <p style="margin: 0; font-size: 14px; color: #8b949e; line-height: 1.5;">
    ← Local Machine learning Models computation on your low-end edge device ➝ by custom research
  </p>
</div>

---

<div align="center">

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![C99](https://img.shields.io/badge/C-C99-blue.svg)](https://en.wikipedia.org/wiki/C99)
  [![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20iOS%20%7C%20Windows%20%7C%20Android%20%7C%20RPi-green.svg)](#platform-support)

</div>

---

<div align="center">

ᯓ➤ [📨 Email Us](mailto:info@nileagi.com)   ᯓ➤ [🌐. Visit](https://lmc.nileagi.com)  ᯓ➤ [[in] lınkedln](https://www.linkedin.com/posts/nile-agi/)

</div>

---

LMc is an AI inference engine written in pure C99. It runs machine learning models locally on **any device** — no cloud, no GPU required.

GGUF-native. Zero external dependencies. Runs on x86, ARM, Android, Windows.

LMc is designed for the reality of computing in Africa and the Global South: low-spec phones, aging laptops, shared computers, and limited bandwidth. Where llama.cpp is a toolkit, LMc is a standard — the **FFmpeg of AI inference**.

**Current status:** Proof of concept — GPT-2 124M working end-to-end. Architecture and extension points are production-ready. New models and hardware backends plug in cleanly.

---

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

There are a lot of available source you can use to get the weight from, I find the following to be simple sources.

You will probably find a lof of models version (I mean quantized version).

|Models| Source| Available Variant|
| :--- | :----:  | :---: |
| GPT-2| Hugging Face -> [QuantFactory](https://huggingface.co/QuantFactory/gpt2-GGUF) | [gpt2-large-GGUF (346-898MB)](https://huggingface.co/QuantFactory/gpt2-large-GGUF), [gpt2-GGUF (81.2 - 178 MB) ](https://huggingface.co/QuantFactory/gpt2-GGUF/tree/main) |
| | Hugging Face -> [mradermacher](https://huggingface.co/mradermacher)| [Small 0.2-0.4GB](https://huggingface.co/mradermacher/gpt2-GGUF), [Medium, 0.5-0.9GB](https://huggingface.co/mradermacher/gpt2-medium-GGUF), [Large, 0.4-1.8 GB](https://huggingface.co/mradermacher/gpt2-large-GGUF), [XLarge 1.0-3.4GB](https://huggingface.co/mradermacher/gpt2-xl-GGUF) |


More sources will be added...

```bash
# GPT-2 Large, Q4_K_M (~520 MB)
# Download any variant, lets use the large version (not large enough though)
wget -P models/ \
  https://huggingface.co/QuantFactory/gpt2-large-GGUF/resolve/main/gpt2-large.Q4_K_M.gguf

# GPT-2 tokenizer files (required)
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
```

### 3. Run

You only need to be so keen with the model name match the store name, you can be responsible human

```bash
./lmc_omp \
  --model   models/gpt2-large-Q4_K_M.gguf \
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

## FAQ

**Q: Why C99 and not C++ or Rust?**
C99 compiles on everything — old Android toolchains, embedded systems, RISC-V boards, 20-year-old GCC versions. It is the lingua franca of systems programming. C++ and Rust add value but also add friction for contributors and targets.

**Q: How does this compare to llama.cpp?**
llama.cpp is a mature, excellent project optimized for performance across many models and backends. LMc is optimized for *simplicity, portability, and accessibility*. The goal is not to beat llama.cpp on benchmarks but to be the easiest way to run AI on any hardware, with code that any developer can read and modify.

**Q: Why focus on Africa?**
Africa has the fastest-growing developer community in the world and the least access to cloud AI infrastructure. Most AI inference tools assume GPU access or reliable internet. LMc assumes neither.

**Q: Can I use LMc in my project?**
Yes. MIT license. Use it in commercial products, embed it in apps, fork it, do whatever you want. Attribution appreciated but not required.

---

## License

MIT — maximum adoption encouraged. See [LICENSE](LICENSE).
