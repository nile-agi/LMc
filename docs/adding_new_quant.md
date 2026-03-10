# Adding a New Quantisation Type to lmc

lmc dequantises all weights to float32 at load time.  Adding a new quant type
means writing one dequantisation function and wiring it into the loader.
No other code changes.

---

## How quantisation works in lmc

```
GGUF file
  └─ tensor data (packed quantised bytes)
       │
       │  LOAD_QUANT(BLOCK_SIZE, BYTES_PER_BLOCK, dequant_fn)   ← gguf.c
       │
       ▼
  float32 weights in the arena slab    ← read by all kernels
```

The macro `LOAD_QUANT` in `gguf.c` handles all the boilerplate:

```c
#define LOAD_QUANT(BS, BPB, FN)  do { \
    if (t->n_elements % (BS) != 0) \
        LMC_FATAL("…"); \
    size_t _rb = (t->n_elements / (BS)) * (BPB); \
    uint8_t *_tmp = (uint8_t *)malloc(_rb); \
    if (fread(_tmp, 1, _rb, f) != _rb) LMC_FATAL("…"); \
    (FN)(_tmp, dst, t->n_elements); \
    free(_tmp); \
} while(0)
```

Your job is to supply the three values `BS`, `BPB`, and `FN`.

---

## The four steps

### Step 1 — `quant.h`: add constants and declare the function

```c
/* ── Your new type ─────────────────────────────────────────────── */
#define GGUF_TYPE_Q3_K_XS       99    /* example: hypothetical new type ID */

#define Q3_K_XS_BLOCK_SIZE     256    /* elements per block                */
#define Q3_K_XS_BYTES_PER_BLOCK 88    /* serialised bytes per block        */

/* declaration */
void dequant_q3k_xs(const uint8_t *src, float *dst, size_t n_elements);
```

**Finding the real type ID**: open any GGUF that uses this quant type in a hex
editor and look at the 4-byte `type` field in the tensor info section, or run:

```bash
python3 -c "
import gguf
r = gguf.GGUFReader('model.gguf')
for t in r.tensors:
    print(t.name, t.tensor_type)
"
```

The integer printed is the `GGUF_TYPE_*` constant.

---

### Step 2 — `quant.c`: implement `dequant_<name>()`

#### Function signature (must match exactly)

```c
void dequant_q3k_xs(const uint8_t *src, float *dst, size_t n_elements);
```

- `src` — raw bytes from the GGUF file, no alignment guarantee
- `dst` — caller-allocated float32 array, `n_elements` elements
- `n_elements` — guaranteed to be a multiple of `Q3K_XS_BLOCK_SIZE`

#### General block structure

GGUF quant types are organised in **blocks**.  Each block encodes
`BLOCK_SIZE` weights.  The common layout is:

```
[fp16 super-scale d] [fp16 super-min m (if present)] [quantised values]
```

Read `d` and `m` with `f16_to_f32()` (already defined in `quant.c`):

```c
uint16_t raw_d; memcpy(&raw_d, src, 2); float d = f16_to_f32(raw_d);
```

#### Worked example: Q4_0

Q4_0 is the simplest non-trivial type: one fp16 scale per 32 values, and two
4-bit signed integers packed per byte.

```c
void dequant_q4_0(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q4_0_BLOCK_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * Q4_0_BYTES_PER_BLOCK;

        /* Bytes 0–1: fp16 scale */
        uint16_t raw_d; memcpy(&raw_d, block, 2);
        float d = f16_to_f32(raw_d);

        /* Bytes 2–17: 16 bytes encoding 32 × 4-bit values */
        const uint8_t *qs = block + 2;
        float *out = dst + b * Q4_0_BLOCK_SIZE;
        for (int i = 0; i < 16; i++) {
            out[i]    = ((int)(qs[i] & 0x0F) - 8) * d;  /* lower nibble */
            out[i+16] = ((int)(qs[i] >>   4) - 8) * d;  /* upper nibble */
        }
    }
}
```

#### Block geometry reference

| Type | Block size | Bytes/block | Structure summary |
|------|-----------|-------------|-------------------|
| Q4_0 | 32 | 18 | fp16 d + 16 bytes of 4-bit nibbles |
| Q4_1 | 32 | 20 | fp16 d + fp16 m + 16 bytes nibbles |
| Q5_0 | 32 | 22 | fp16 d + 4-byte high-bit mask + 16 bytes nibbles |
| Q5_1 | 32 | 24 | fp16 d + fp16 m + 4-byte mask + 16 bytes nibbles |
| Q8_0 | 32 | 34 | fp16 d + 32 × int8 values |
| Q2_K | 256 | 84 | 16B scales + 16B mins + 64B 2-bit qs + fp16 d + fp16 dmin |
| Q3_K | 256 | 110 | 32B high-bit mask + 64B 2-bit lows + 12B scale + fp16 d |
| Q4_K | 256 | 144 | fp16 d + fp16 dmin + 12B 6-bit scales + 128B nibbles |
| Q5_K | 256 | 176 | fp16 d + fp16 dmin + 12B scales + 32B high-bit + 128B nibbles |
| Q6_K | 256 | 210 | 128B lower-4-bits + 64B upper-2-bits + 16B int8 scales + fp16 d |
| Q8_0 | 32 | 34 | fp16 d + 32 int8 |

For the K-quants (Q2_K through Q6_K), the authoritative byte layout is in
`ggml-common.h` in the llama.cpp repository.  Match it exactly.

#### Sub-scale decode helper (K-quants)

`Q4_K` and `Q5_K` use the same 6-bit-packed scale field.  The helper
`get_scale_min_k4()` is already in `quant.c` and can be reused:

```c
static void get_scale_min_k4(int j, const uint8_t *scales,
                              uint8_t *out_sc, uint8_t *out_m);
```

#### I-quant codebooks

`IQ3_XXS` and `IQ3_S` use static lookup tables (`iq3xxs_grid[]`,
`iq3s_grid[]`, `ksigns_iq2xs[]`) defined at the top of `quant.c`.  For any
new i-quant type, add its codebook the same way — copy the values directly
from `ggml-quants.h` in the llama.cpp source.

---

### Step 3 — `gguf.c`: wire the dispatch

Inside `load_model_gguf()`, in the `switch (t->type)` block:

```c
case GGUF_TYPE_Q3_K_XS:
    LOAD_QUANT(Q3_K_XS_BLOCK_SIZE, Q3_K_XS_BYTES_PER_BLOCK, dequant_q3k_xs);
    break;
```

Place it in numerical order with the other cases.

Update the error message in the `default:` case to list the new type:

```c
default:
    LMC_FATAL("Unsupported tensor type %u for '%s'\n"
              "  Supported: F32 F16 Q2_K Q3_K Q4_0 Q4_1 Q4_K "
              "Q5_0 Q5_1 Q5_K Q6_K Q8_0 IQ3_XXS IQ3_S IQ4_XS Q3_K_XS",
              t->type, t->name);
```

---

### Step 4 — verify correctness

A wrong dequantisation is silent: the model loads and runs but outputs
gibberish.  Use this two-step check before committing:

**4a. Round-trip test against llama.cpp**

```bash
# Convert any model to your new quant type with llama.cpp's quantize tool
./quantize model.gguf model-newtype.gguf Q3_K_XS

# Run lmc on the new file
./lmc_omp --model model-newtype.gguf --prompt "The capital of France is"

# Run llama.cpp on the same file
./llama-cli -m model-newtype.gguf -p "The capital of France is" -n 5
```

The first ~5 tokens should be identical (given the same temperature=0 / greedy
settings).

**4b. Perplexity spot-check**

```bash
# Use bench.c to check per-token entropy looks sane
./lmc_bench --model model-newtype.gguf --prompt "..." --n-predict 100
```

If perplexity is orders of magnitude higher than the F16 baseline, the
dequantisation is wrong.

---

## Checklist

- [ ] `GGUF_TYPE_*`, `BLOCK_SIZE`, `BYTES_PER_BLOCK` added to `quant.h`
- [ ] `dequant_<name>()` declared in `quant.h` and implemented in `quant.c`
- [ ] `case GGUF_TYPE_*: LOAD_QUANT(…); break;` added to `gguf.c`
- [ ] Default error message updated to list the new type
- [ ] Output matches llama.cpp on the same model (greedy, same prompt)
- [ ] `make` and `make omp` compile with zero warnings
