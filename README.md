<!-- <div align="center"> -->

<!-- ![lmc logo](https://placehold.co/100x100/ffffff/000000/WebP?text=LMc&css={"borderRadius":"30px"})  -->
<!-- 
![LMx](https://placehold.co/100x100/ffffff/000000.WebP?text=LMc&css={"border-radius":"100px","fontSize":"120px","fontWeight":"bold","textAlign":"center"}) -->

<!-- <img src="https://placehold.co/100x100/EEEEEE/000000/?text=LMc" alt="LMc" style="border-radius: 20px; overflow: hidden; display: block;" > -->

<!-- </div> -->

<!-- <div style="background: #0d1117; padding: 20px; border-radius: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #c9d1d9; max-width: 800px; margin: 0 auto;"> -->

<!-- <table style="border: none; width: 100%; background: transparent;">
  <tr>
    <td style="width: 100px; vertical-align: top;">
         <img src="https://placehold.co/100x100/eeeeee/000000/WebP?text=LMc" alt="LMc" style="border-radius: 16px; padding: 2px; margin: 0 auto; overflow: hidden; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); width: 100px; height: 100px; font-weight: bold; font-size: 100px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    </td>
    <td style="vertical-align: top;">
      <p style="margin: 8px 0 16px 0; font-size: 18px; color: #8b949e; font-weight: 500;">
        Local Model compute
      </p>
      <p style="margin: 0; font-size: 14px; color: #8b949e; line-height: 1.5;">
       ← Local Machine learning Models computation on your laptop or in a data center - by custom research
      </p>
    </td>
  </tr>
</table> -->

<div style="display: flex; align-items: flex-start; gap: 20px; margin-bottom: 20px; border-radius: 16px;">
  <img src="https://placehold.co/100x100/eeeeee/000000/WebP?text=LMc" alt="LMc" style="border-radius: 16px; padding: 2px; overflow: hidden; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); width: 100px; height: 100px; font-weight: bold; font-size: 100px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <div>
    <p style="margin: 8px 0 16px 0; font-size: 18px; color: #8b949e; font-weight: 500;">
      Local Model compute
    </p>
    <p style="margin: 0; font-size: 14px; color: #8b949e; line-height: 1.5;">
      ← Local Machine learning Models computation on your laptop or in a data center - by custom research
    </p>
  </div>
</div>

<!-- </div> -->

<!-- # Local Model Compute -->

<!-- > **Edge-first AI inference for everyone.**
> Pure C99 · No dependencies · Runs on any CPU · Built for Africa and the Global South. -->

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C99](https://img.shields.io/badge/C-C99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Android%20%7C%20RPi-green.svg)](#platform-support)

---

## What is LMc?

LMc is an AI inference engine written in pure C99. It runs machine learning models locally on **any device** — no cloud, no GPU required.

LMc is designed for the reality of computing in Africa and the Global South: low-spec phones, aging laptops, shared computers, and limited bandwidth. Where llama.cpp is a toolkit, LMc is a standard — the **FFmpeg of AI inference**.

**Current status:** Proof of concept — GPT-2 124M working end-to-end. Architecture and extension points are production-ready. New models and hardware backends plug in cleanly.

---

## Why LMc?

| Problem | LMc's Answer |
|---|---|
| Africa is the most GPU-poor region in the world | Optimized for CPU-only, low-RAM edge devices |
| llama.cpp, MLX, llamafile are complex / GPU-focused | Single-purpose, zero-dependency, one binary |
| AI inference requires internet / cloud | Fully offline, works without connectivity |
| Hard to add new models or hardware backends | Clean extension points, documented interfaces |
| No universal standard for edge AI | LMc aims to be that standard |

---

## Features

- **Pure C99** — compiles on any C compiler, any OS, any CPU architecture
- **Zero runtime dependencies** — only the C standard library and `libm`
- **Multiple model formats**
  - `.bin` — LMc custom float32 binary (fast load, simple format)
  - `.gguf` — GGUF format (llama.cpp / Ollama / HuggingFace compatible)
- **Multiple quantization types** — F32, F16, Q8_0, Q5_K (S+M), Q6_K
- **OpenMP multi-threading** — compile once, scale to all CPU cores
- **KV cache** — incremental decoding, no re-computation of prompt
- **Clean extension points** — add new models, formats, and hardware backends with minimal code
- **Structured error handling** — no `exit(1)` hidden in library code
- **Verbose logging** — structured, level-controlled output to stderr

---

## Supported Platforms

| Platform | CPU | Status |
|---|---|---|
| Linux x86-64 | Any | ✅ Production |
| Linux ARM64 | Raspberry Pi 4/5, Jetson | ✅ Production |
| Linux ARM32 | Raspberry Pi 2/3, phones | ✅ Production |
| macOS (Intel) | x86-64 | ✅ Production |
| macOS (Apple Silicon) | ARM64 | ✅ Production |
| Windows (MinGW/MSVC) | Any | ✅ Production |
| Android (Termux) | ARM64/ARM32 | ✅ Production |
| RISC-V | Any | ✅ (generic C) |
| GPU (CUDA, Metal, OpenCL) | — | 🚧 Planned |
| NPU (NNAPI, CoreML, QNN) | — | 🚧 Planned |

---

## Quick Start

### 1. Build

```bash
git clone https://github.com/nile-agi/LMc.git
cd LMc
make
```

For multi-threaded (faster on multi-core CPUs):
```bash
make openmp
```

### 2. Get the tokenizer files

```bash
python3 scripts/get_tokenizer.py
```

This downloads `encoder.json` and `vocab.bpe` (~1 MB total) from OpenAI's servers.

### 3. Get a model

**Option A — Convert from HuggingFace** (requires PyTorch):
```bash
pip install torch transformers
python3 scripts/converter.py
# → generates gpt2_124m.bin (~500 MB, float32)
```

**Option B — Download GGUF** (no PyTorch needed):
```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download ggml-org/gpt2-GGUF gpt2.Q8_0.gguf

# Or using wget
wget https://huggingface.co/ggml-org/gpt2-GGUF/resolve/main/gpt2.Q8_0.gguf
```

### 4. Run

```bash
./build/lmc "The future of AI in Africa is"
```

```
╔══════════════════════════════════════════════════════╗
║  LMc v0.1.0 — Local Model Compute                    ║
║  Edge-first AI inference for everyone                ║
╚══════════════════════════════════════════════════════╝

lmc [INFO]  Model file  : gpt2.Q8_0.gguf
lmc [INFO]  Format      : GGUF (.gguf)
lmc [INFO]  Parameters  : 124439808  (474.7 MB float32)
lmc [INFO]  KV cache    : 36.0 MB
lmc [INFO]  Vocabulary  : 50257 tokens
lmc [INFO]  BPE merges  : 50000 rules
lmc [INFO]  Prompt tokens: 8

The future of AI in Africa is a story of access and opportunity...
```

---

## Usage

```
lmc "prompt" [options]

Options:
  --model    <path>   Model file (.bin or .gguf)
  --encoder  <path>   encoder.json path   (default: ./encoder.json)
  --bpe      <path>   vocab.bpe path      (default: ./vocab.bpe)
  --tokens   <n>      Max new tokens       (default: 128)
  --temp     <f>      Temperature 0.0-2.0  (default: 0.7)
  --topp     <f>      Top-p threshold 0-1  (default: 0.9)
  --seed     <n>      RNG seed             (default: random)
  --verbose           Enable verbose logging
  --quiet             Suppress info logs
  --version           Print version and exit
  --help              Print this help and exit
```

**Examples:**

```bash
# Basic generation
./build/lmc "Once upon a time in Lagos"

# More tokens, higher creativity
./build/lmc "def fibonacci(n):" --tokens 100 --temp 0.9

# Deterministic output (reproducible)
./build/lmc "Hello, world" --seed 42

# Custom model path
./build/lmc "Your prompt" --model /path/to/gpt2.Q6_K.gguf

# Quiet mode (output only, no info logs)
./build/lmc "Tell me about Kenya" --quiet
```

---

## Model File Support

| Format | Extension | Quant Types | Notes |
|---|---|---|---|
| LMc binary | `.bin` | F32 | Generated by `scripts/converter.py` |
| GGUF | `.gguf` | F32, F16, Q8_0, Q5_K, Q6_K | llama.cpp compatible |

Model auto-detection checks these paths in order:
1. `gpt2_124m.bin`
2. `gpt2.f16.gguf`
3. `gpt2.Q8_0.gguf`
4. `gpt2.Q6_K.gguf`
5. `gpt2.gguf`

---

## Memory Requirements

| Format | Disk Size | RAM (inference) |
|---|---|---|
| F32 (.bin) | ~500 MB | ~560 MB |
| F16 (.gguf) | ~250 MB | ~560 MB\* |
| Q8_0 (.gguf) | ~130 MB | ~560 MB\* |
| Q5_K (.gguf) | ~85 MB | ~560 MB\* |
| Q6_K (.gguf) | ~100 MB | ~560 MB\* |

\* Quantized formats are dequantized to float32 at load time. In-memory representation is always float32. Future work: quantized-compute inference (stay quantized during matmul).

**Additional memory:**
- KV cache: ~36 MB (for 1024 token context)
- Activation buffers: ~1 MB

**Minimum system:** 640 MB RAM, any CPU from the last 20 years.

---

## Project Structure

```
lmc/
├── include/
│   ├── lmc.h              Public API — use this to embed LMc in your app
│   └── lmc_internal.h     Internal shared types and declarations
│
├── src/
│   ├── main.c             CLI entry point
│   ├── lmc.c              Core API (context lifecycle, model/tokenizer load, generate)
│   ├── arena.c            Memory arena (bump-pointer allocator for weights)
│   ├── math_ops.c         Core math: matmul, layer_norm, softmax, gelu, f16→f32
│   ├── quantization.c     Dequantization: F16, Q8_0, Q5_K, Q6_K
│   ├── sampling.c         Token sampling: greedy, top-p nucleus
│   ├── tokenizer.c        GPT-2 BPE tokenizer (encoder.json + vocab.bpe)
│   │
│   ├── models/
│   │   ├── gpt2_weights.c     GPT-2 weight layout and arena assignment
│   │   └── gpt2_inference.c   GPT-2 forward pass, attention, KV cache
│   │
│   └── backends/
│       ├── bin_loader.c       LMc custom .bin format loader
│       └── gguf_loader.c      GGUF format loader (F32/F16/Q8_0/Q5_K/Q6_K)
│
├── scripts/
│   ├── converter.py       Convert HuggingFace GPT-2 → gpt2_124m.bin
│   └── get_tokenizer.py   Download encoder.json and vocab.bpe
│
├── Makefile
├── LICENSE
└── README.md
```

---

## Architecture Deep Dive

### The Public API (`include/lmc.h`)

LMc exposes a clean C API for embedding in other projects:

```c
#include "lmc.h"

// Create context
LmcContext *ctx = lmc_ctx_new();

// Load model (auto-detects format)
lmc_load_model(ctx, "gpt2.Q8_0.gguf");

// Load tokenizer
lmc_load_tokenizer(ctx, "encoder.json", "vocab.bpe");

// Init KV cache and activations
lmc_kv_cache_init(ctx);
lmc_activations_init(ctx);

// Generate
LmcGenConfig cfg;
lmc_gen_config_default(&cfg);
cfg.max_new_tokens = 200;
cfg.temperature    = 0.8f;
lmc_generate(ctx, "The capital of Tanzania is", &cfg);

// Cleanup
lmc_ctx_free(ctx);
```

### Adding a New Model

1. Create `src/models/<model>_weights.c`:
   - Implement `lmc_<model>_param_count()` returning total float32 params
   - Implement `lmc_<model>_assign_weight_ptrs()` to map arena floats to struct fields

2. Create `src/models/<model>_inference.c`:
   - Implement `lmc_<model>_forward(ctx, token_id, pos)` returning logit pointer

3. Add tensor name mapping in `src/backends/gguf_loader.c`:
   - Extend `gguf_name_to_ptr()` with the new model's GGUF tensor names

4. Add a model-type enum to `include/lmc.h` and dispatch in `src/lmc.c`

### Adding a New Hardware Backend

The compute bottleneck is `lmc_matmul_vec()` in `src/math_ops.c`. To accelerate:

1. Add a compile-time flag (e.g., `-DLMC_BACKEND_CUDA`)
2. Implement the backend in `src/backends/<backend>.c`
3. Conditionally replace `lmc_matmul_vec()` with your implementation
4. Add detection and selection logic to `lmc_available_backends()`

The interface is deliberately minimal: all backends implement the same `matmul_vec(out, weight, bias, in, M, K)` signature.

### Adding a New Quantization Type

1. Define block size and bytes-per-block constants in `src/quantization.c`
2. Implement `lmc_dequant_<type>(src, dst, n_elements)`
3. Add a `GGUF_TYPE_<n>` constant in `src/backends/gguf_loader.c`
4. Add a `case` in the tensor load loop in `lmc_load_gguf()`
5. Add the `LmcQuantType` enum value in `include/lmc.h`

---

## Build System

```bash
make              # Optimized build, auto-detects CPU arch
make openmp       # Multi-threaded build
make debug        # Debug build with AddressSanitizer + UBSan
make clean        # Remove build/
make install      # Install to /usr/local/bin
make test         # Smoke test (requires a model file)
make help         # Full target list
```

**Cross-compilation examples:**

```bash
# Raspberry Pi (ARM64) from x86-64
make CC=aarch64-linux-gnu-gcc EXTRA_CFLAGS="-march=armv8-a"

# Android via NDK
make CC=$NDK/bin/aarch64-linux-android29-clang EXTRA_CFLAGS="-march=armv8-a"

# Generic / conservative (no march=native)
make EXTRA_CFLAGS=""
```

---

## Roadmap

### v0.2 — Performance
- [ ] Quantized-compute matmul (stay in Q8_0/Q4 during inference, no dequant step)
- [ ] SIMD acceleration: AVX2 (x86-64), NEON (ARM64), RVV (RISC-V)
- [ ] Batch prefill for faster prompt processing
- [ ] Memory-mapped weight loading (mmap) — zero copy on Linux/macOS

### v0.3 — More Models
- [ ] LLaMA / Mistral / Qwen architecture support
- [ ] Phi-2 / Phi-3 (efficient on low-RAM devices)
- [ ] TinyLlama 1.1B
- [ ] RWKV (linear attention, very low memory)

### v0.4 — Hardware Backends
- [ ] OpenCL backend (runs on any GPU including integrated Intel/AMD)
- [ ] Vulkan compute backend
- [ ] Apple Metal backend
- [ ] Android NNAPI / NPU backend (Qualcomm, MediaTek)

### v0.5 — Developer Experience
- [ ] Streaming callback API
- [ ] Python bindings (`ctypes`)
- [ ] REST API server mode
- [ ] Chat / conversation mode (stateful context)
- [ ] GGUF writing (quantize your own models)

### v1.0 — Production
- [ ] All major open LLMs supported
- [ ] Benchmark suite vs llama.cpp, MLX
- [ ] Pre-built binaries for major platforms
- [ ] Mobile SDK (iOS, Android)

---

## Contributing

LMc welcomes contributions from everyone, especially developers from Africa and the Global South.

### Where to start

- **Good first issues:** Quantization improvements, SIMD for a new arch, Python bindings
- **Wanted:** Testing on low-spec devices (phones, RPi, old laptops), performance profiling
- **Architecture:** Adding a new model (Phi-2 would be a great first model addition)

### Code style

- Pure C99, no extensions unless clearly documented and guarded
- Every function has a block comment explaining what it does
- New files get a header comment with purpose, references, and SPDX license tag
- Error handling: always propagate `LmcError`, never call `exit()` in library code
- No global mutable state outside of `LmcContext`

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

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

- **GPT-2** by OpenAI — the model that started it all
- **llama.cpp** by Georgi Gerganov — inspiration for GGUF format support and quantization techniques
- **The African developer community** — the reason this exists

---

*LMc — Because intelligence should be local.*
