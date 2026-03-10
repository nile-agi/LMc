# Edge Optimisation Guide

This guide covers squeezing more performance out of lmc on constrained
hardware: ARM single-board computers, Android phones, 32-bit microcontrollers,
and battery-powered devices.

---

## Know your bottleneck first

Before tuning anything, run the bench binary to see where time is spent:

```bash
make bench
./lmc_bench --model models/gpt2-large-q4_k_m.gguf --prompt "Hello" --n-predict 64
```

Almost all inference time is `matmul_vec()`.  The remaining time is:

| Function | % of wall time (GPT-2 Large, 4 threads) |
|----------|-----------------------------------------|
| `matmul_vec` (lm_head + per-layer) | ~85 % |
| `attention_forward` (QK dot + V acc) | ~10 % |
| `softmax`, `layer_norm`, `rms_norm` | ~3 % |
| File I/O, dequantisation (load only) | — |

Optimise `matmul_vec` first.  Everything else is noise.

---

## Compiler flags

The Makefile already uses `-O3 -march=native -ffast-math -funroll-loops`.
On cross-compile targets you must set `-march` explicitly:

| Target | Recommended flags |
|--------|-------------------|
| ARMv8 (Pi 4, M1 Mac rosetta) | `-march=armv8-a+crc+crypto -mtune=cortex-a72` |
| ARMv8.2 (Pi 5, Cortex-A76) | `-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76` |
| ARMv7 with NEON | `-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard` |
| Android arm64-v8a | `-march=armv8-a -target aarch64-linux-android21` |
| Android armeabi-v7a | `-march=armv7-a -mfpu=neon -mfloat-abi=softfp` |
| RISC-V (SiFive U74) | `-march=rv64gcv -mabi=lp64d` |

Use `build.sh` for cross-compilation — it sets these flags automatically:

```bash
./build.sh arm64       # AArch64 Linux
./build.sh arm32       # ARMv7 with NEON
./build.sh android     # Android NDK arm64-v8a
```

---

## ARM NEON intrinsics in `matmul_vec`

The compiler auto-vectorises the existing 16-wide loop with `-march=native` on
ARM, but you can guarantee good vector code by writing explicit NEON intrinsics.
Replace the inner loop in `matmul_vec()` (`src/ops.c`) with:

```c
#ifdef __ARM_NEON
#include <arm_neon.h>

/* Drop-in replacement for the inner accumulation loop.
 * Processes 16 floats per iteration using four float32x4_t lanes.
 * Reduces to scalar for the tail (Dh=64 has no tail).             */
static float dot_neon(const float *restrict w, const float *restrict x, int K) {
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k <= K - 16; k += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(w+k+ 0), vld1q_f32(x+k+ 0));
        acc1 = vmlaq_f32(acc1, vld1q_f32(w+k+ 4), vld1q_f32(x+k+ 4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(w+k+ 8), vld1q_f32(x+k+ 8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(w+k+12), vld1q_f32(x+k+12));
    }
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    float r = vaddvq_f32(acc0);
    for (; k < K; k++) r += w[k] * x[k];   /* scalar tail */
    return r;
}
#endif
```

Then in `matmul_vec()`, replace:

```c
float acc = bias ? bias[m] : 0.0f;
int k = 0;
for (; k <= K - 16; k += 16)
    acc += w[k+0]*in[k+0] + /* … */ + w[k+15]*in[k+15];
for (; k < K; k++) acc += w[k]*in[k];
out[m] = acc;
```

with:

```c
#ifdef __ARM_NEON
float acc = dot_neon(w, in, K) + (bias ? bias[m] : 0.0f);
#else
float acc = bias ? bias[m] : 0.0f;
int k = 0;
for (; k <= K - 16; k += 16)
    acc += w[k+0]*in[k+0] + /* … */ + w[k+15]*in[k+15];
for (; k < K; k++) acc += w[k]*in[k];
#endif
out[m] = acc;
```

### ARM dotprod (ARMv8.2+)

On Cortex-A76 / A78 / X1 / Apple M-series, the `sdot` instruction computes
four int8 multiply-accumulates per cycle.  This is relevant only if you
switch to on-the-fly Q8_0 dequantisation (multiply by the block scale inside
the dot product loop rather than materialising float32 weights).  To check
support at runtime:

```c
#if defined(__ARM_FEATURE_DOTPROD)
    /* use vdotq_s32 / vdotq_u32 */
#endif
```

### fp16 arithmetic (ARMv8.2+)

`-march=armv8.2-a+fp16` enables `float16x8_t` NEON vectors.  Using fp16
arithmetic cuts memory bandwidth in half at a small accuracy cost:

```c
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>

static float dot_fp16(const float16_t *w, const float16_t *x, int K) {
    float32x4_t acc = vdupq_n_f32(0.f);
    for (int k = 0; k <= K - 8; k += 8) {
        float16x8_t wv = vld1q_f16(w + k);
        float16x8_t xv = vld1q_f16(x + k);
        /* widen and accumulate */
        acc = vmlaq_f32(acc, vcvt_f32_f16(vget_low_f16(wv)),
                             vcvt_f32_f16(vget_low_f16(xv)));
        acc = vmlaq_f32(acc, vcvt_f32_f16(vget_high_f16(wv)),
                             vcvt_f32_f16(vget_high_f16(xv)));
    }
    return vaddvq_f32(acc);
}
#endif
```

This requires storing weights as fp16 in the arena, which means changing
`Arena` to use `uint16_t` (or a separate fp16 arena) and converting during
`assign_weight_ptrs()`.  Significant refactor; only worth it if you are
specifically targeting Cortex-A76+ with fp16-quantised models.

---

## Android NDK

### Build

```bash
# Set NDK path
export ANDROID_NDK=$HOME/Android/Sdk/ndk/26.3.11579264

./build.sh android
# Produces lmc_android (arm64-v8a) and lmc_android_omp (with OpenMP)
```

### Deployment

```bash
# Push to device (requires adb)
adb push lmc_android /data/local/tmp/
adb push models/llama-3.2-1b-q4_k_m.gguf /data/local/tmp/models/
adb push encoder.json vocab.bpe /data/local/tmp/

# Run
adb shell "cd /data/local/tmp && chmod +x lmc_android && \
           ./lmc_android --model models/llama-3.2-1b-q4_k_m.gguf \
           --prompt 'Hello' --n-predict 50 --temp 0.7"
```

### OpenMP on Android

The NDK ships `libomp.so` starting from NDK r21.  Link it statically to avoid
deployment issues:

```makefile
# In build.sh android target
LDFLAGS += -static-openmp -fopenmp
```

### Memory limits

Android kills processes that exceed the per-app memory limit (typically 512 MB
on low-end devices, 2–4 GB on flagships).  Choose quantisation accordingly:

| Model | Quant | Arena (float32) | KV cache | Total |
|-------|-------|-----------------|----------|-------|
| LLaMA-3.2 1B | Q4_K_M | ~700 MB | ~90 MB | ~800 MB |
| LLaMA-3.2 3B | Q4_K_M | ~2.1 GB | ~180 MB | ~2.3 GB |
| Mistral 7B | Q4_K_M | ~4.2 GB | ~400 MB | ~4.6 GB |

For devices under 2 GB, use 1B models with Q2_K or Q3_K.

### Reduce context length

The KV cache is `L × Hkv × seq_len × head_dim × 2 × 4` bytes.  Halving
`seq_len` from 8192 to 4096 saves ~50% of KV memory.  Add a `--ctx` flag or
hard-limit `g_cfg.seq_len` at load time:

```c
/* In load_model_gguf(), after resolving g_cfg: */
if (g_cfg.seq_len > 2048 && getenv("LMC_MAX_CTX"))
    g_cfg.seq_len = atoi(getenv("LMC_MAX_CTX"));
```

---

## Raspberry Pi

### Pi 4 (Cortex-A72, 4× ARMv8)

```bash
./build.sh arm64     # cross-compile from x86, or compile natively on the Pi
```

Native compile on Pi 4 OS (64-bit):

```bash
gcc -std=c99 -O3 -march=armv8-a+crc -mtune=cortex-a72 \
    -ffast-math -funroll-loops \
    -Wall -Isrc src/*.c -o lmc -lm
```

With OpenMP:

```bash
sudo apt install libomp-dev
gcc -std=c99 -O3 -march=armv8-a+crc -mtune=cortex-a72 \
    -ffast-math -funroll-loops -fopenmp \
    -Wall -Isrc src/*.c -o lmc_omp -lm
```

Expected throughput (Pi 4, 4 threads):

| Model | Quant | t/s |
|-------|-------|-----|
| GPT-2 Small | Q4_K_M | ~8 |
| GPT-2 Large | Q4_K_M | ~2 |
| LLaMA-3.2 1B | Q4_K_M | ~1.5 |

### Pi 5 (Cortex-A76, 4× ARMv8.2)

The A76 supports the `dotprod` extension:

```bash
gcc -std=c99 -O3 -march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76 \
    -ffast-math -funroll-loops -fopenmp \
    -Wall -Isrc src/*.c -o lmc_omp -lm
```

Expected throughput (Pi 5, 4 threads):

| Model | Quant | t/s |
|-------|-------|-----|
| GPT-2 Small | Q4_K_M | ~25 |
| GPT-2 Large | Q4_K_M | ~6 |
| LLaMA-3.2 1B | Q4_K_M | ~4 |

### Thermal throttling

The Pi 4 throttles to 600 MHz when the SoC exceeds 80 °C.  A heatsink and
fan keeps it at full 1.8 GHz indefinitely:

```bash
# Monitor temperature and clock while running
watch -n1 "vcgencmd measure_temp; vcgencmd measure_clock arm"
```

Set a governor that sustains performance:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Battery and power

### Thread count vs. battery life

More threads → faster completion but higher peak power.  On mobile:

- **1–2 threads**: low power, good for background generation or streaming
- **4 threads**: balanced; typical for on-device chatbots
- **all cores**: fastest, but drains battery quickly; avoid on battery for
  generation > 200 tokens

### Clock scaling

On Linux (Pi, Android root):

```bash
# Lock to a lower frequency for sustained low-power generation
echo 1200000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
```

### Memory bandwidth

On RAM-constrained devices, quantisation format matters more than compute.

| Quant | Size (7B) | Bandwidth (GB/s needed at 10 t/s) |
|-------|-----------|----------------------------------|
| F32 | 26 GB | 260 GB/s (impossible on mobile) |
| F16 | 13 GB | 130 GB/s |
| Q8_0 | 7 GB | 70 GB/s |
| Q4_K_M | 4.2 GB | 42 GB/s |
| Q2_K | 2.7 GB | 27 GB/s |

The Pi 4's LPDDR4 delivers ~12 GB/s; GPT-2 Small Q4_K_M at 8 t/s is just
within reach.  LLaMA 7B requires a Pi 5 or a device with faster RAM.

### Reduce activation buffer allocations

`init_activations()` calls `malloc()` for each buffer.  On systems where
`malloc` is slow (some microcontroller RTOSes), allocate all activations from
the arena instead:

```c
/* In init_activations(): replace malloc with arena_alloc */
g_act.x        = arena_alloc(D);
g_act.x_norm   = arena_alloc(D);
/* … etc … */
```

This requires calling `init_activations()` before `arena_init()` is sealed,
which means computing activation sizes as part of `gpt2_total_params()` /
`llama_total_params()`.

---

## Windows (MinGW / MSYS2)

```bash
# In MSYS2 MinGW64 terminal
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp

./build.sh windows-mingw
# Produces lmc.exe and lmc_omp.exe
```

Or manually:

```bash
x86_64-w64-mingw32-gcc -std=c99 -O3 -march=native -ffast-math \
    -funroll-loops -fopenmp -Wall -Isrc src/*.c -o lmc_omp.exe \
    -lm -static-libgcc -static-libstdc++
```

The `-static-libgcc` flag avoids a dependency on `libgcc_s_seh-1.dll` on
machines without MSYS2 installed.

---

## 32-bit ARMv7 (armeabi-v7a, Raspberry Pi OS 32-bit)

```bash
gcc -std=c99 -O3 -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard \
    -ffast-math -funroll-loops -Wall -Isrc src/*.c -o lmc -lm
```

**Pointer size warning**: on 32-bit systems, `size_t` is 4 bytes.  The arena
uses `size_t` throughout, so the maximum slab is ~4 GB — large enough for any
model that fits in RAM.  However, `(size_t)V * D` where `V=50257` and
`D=1600` overflows a 32-bit `int` (80 M > 2^26) but not `size_t`.  All
`arena_alloc()` calls and `gpt2_total_params()` already use `size_t`
arithmetic, so 32-bit builds are safe.

Model size is still the practical limit: GPT-2 XL F16 is 3 GB, which exceeds
the default 32-bit virtual address space on some OSes.  Use Q4_K models.

---

## Summary: quick-reference settings by device

| Device | `--threads` | Quant | Expected t/s |
|--------|------------|-------|-------------|
| MacBook Air M2 | 8 | Q4_K_M (7B) | ~25 |
| Intel i7-1165G7 | 8 | Q4_K_M (7B) | ~7 |
| Raspberry Pi 5 | 4 | Q4_K_M (1B) | ~4 |
| Raspberry Pi 4 | 4 | Q4_K_M (1B) | ~1.5 |
| Android Pixel 7 | 4 | Q4_K_M (1B) | ~3 |
| Android low-end | 2 | Q3_K (1B) | ~0.8 |
