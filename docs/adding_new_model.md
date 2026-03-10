# Adding a New Model Architecture to lmc

This guide walks through adding a brand-new transformer architecture — covering
every file that must change and every decision that must be made.  The GPT-2 and
LLaMA paths are already implemented; use them as worked examples.

The guide also includes architecture-specific notes for **Phi**, **Gemma**,
**Qwen**, **BitNet b1.58**, and **Whisper**.

---

## How the loader works (read this first)

```
load_model(path)
  └─ load_model_gguf(path)
       1. Parse GGUF header (magic, version, tensor count, KV count)
       2. Loop over KV metadata → populate g_cfg (arch, dims, rope params …)
       3. Resolve g_cfg.arch from general.architecture string
       4. Call llama_total_params() or gpt2_total_params()
          → arena_init() with the result
       5. assign_weight_ptrs()    — slice arena into g_weights fields
       6. init_rope_cache()       — LLaMA only; no-op for GPT-2
       7. Loop over tensor descriptors → fseek + dequant → g_weights slots
          (gguf_name_to_ptr() maps tensor names to float** pointers)
```

Adding a new arch means touching exactly five places, in this order:

| # | File | What to do |
|---|------|------------|
| 1 | `models.h` | Add `ARCH_*` constant and weight fields |
| 2 | `models.c` | Add `<arch>_total_params()` and `assign_weight_ptrs()` branch |
| 3 | `ops.h / ops.c` | Add normalisation / attention / FFN kernels + dispatch |
| 4 | `gguf.h / gguf.c` | Add metadata key constants + KV parsing + tensor name map |
| 5 | `lmc.c` | Update `find_default_model()` candidate list |

---

## Step 1 — `models.h`: declare the architecture

### 1a. Add an ARCH_ constant

```c
typedef enum {
    ARCH_UNKNOWN = 0,
    ARCH_GPT2,
    ARCH_LLAMA,
    ARCH_PHI3,    /* ← new */
} ModelArch;
```

### 1b. Extend ModelConfig if needed

Most new architectures fit in the existing fields.  Add a field only if there
is no existing field for the concept:

```c
typedef struct {
    /* existing fields … */
    int partial_rotary_factor;  /* Phi-3: only first N dims get RoPE */
} ModelConfig;
```

And a CFG_ accessor:

```c
#define CFG_PARTIAL_ROPE  g_cfg.partial_rotary_factor
```

### 1c. Add weight fields to LayerWeights

Only add fields that are not already present.  Fields unused by any given
arch stay NULL (the struct is zero-initialised by `calloc`).

```c
typedef struct {
    /* existing … */

    /* Phi-3: shared QKV projection (single [3D × D] weight like GPT-2),
     * but with RoPE and RMSNorm instead of LayerNorm.              */
    float *phi_qkv_weight;   /* [3D × D]  */
} LayerWeights;
```

In practice Phi-3 uses the same `qkv_weight` field that GPT-2 uses, so no
new field is needed in that case — just reuse it.

---

## Step 2 — `models.c`: parameter count and weight allocation

### 2a. Add `<arch>_total_params()`

This function is called to size the arena slab.  It must count **every float**
that `assign_weight_ptrs()` will `arena_alloc()`.

```c
size_t phi3_total_params(void) {
    const int D = CFG_D, V = CFG_V, L = CFG_L, F = CFG_F;
    const int H = CFG_H, Dh = CFG_Dh;    /* Phi-3 uses MHA, Hkv == H */
    size_t n = (size_t)V * D;             /* token_embd */
    for (int l = 0; l < L; l++) {
        n += D;                            /* rms_attn_weight */
        n += (size_t)3 * H * Dh * D;      /* qkv_weight (fused) */
        n += (size_t)D * D;                /* attn_proj_weight */
        n += D;                            /* rms_ffn_weight */
        n += (size_t)F * D;                /* ffn_gate_weight */
        n += (size_t)F * D;                /* ffn_up_weight */
        n += (size_t)D * F;                /* ffn_down_weight */
    }
    n += D;                /* final rms weight */
    n += (size_t)V * D;    /* lm_head */
    return n;
}
```

### 2b. Add a branch in `assign_weight_ptrs()`

```c
} else if (g_cfg.arch == ARCH_PHI3) {

    g_weights.wte = arena_alloc((size_t)V * D);
    g_weights.wpe = NULL;

    for (int l = 0; l < L; l++) {
        LayerWeights *lw = &g_weights.layers[l];
        lw->rms_attn_weight = arena_alloc(D);
        lw->qkv_weight      = arena_alloc((size_t)3 * H * Dh * D);
        lw->attn_proj_weight= arena_alloc((size_t)D * D);
        lw->rms_ffn_weight  = arena_alloc(D);
        lw->gate_weight     = arena_alloc((size_t)F * D);
        lw->up_weight       = arena_alloc((size_t)F * D);
        lw->down_weight     = arena_alloc((size_t)D * F);
    }
    g_weights.rms_f_weight = arena_alloc(D);
    g_weights.lm_head      = arena_alloc((size_t)V * D);
}
```

### 2c. Extend `init_kv_cache()` if needed

The existing implementation uses `CFG_Hkv`, which is correct for both MHA and
GQA.  Nothing to change unless your architecture has a fundamentally different
cache layout.

### 2d. Extend `init_rope_cache()` if needed

The existing implementation stores `[seq_len × head_dim/2]` tables.  Phi-3's
partial RoPE rotates only the first `partial_rotary_factor` dimensions; in that
case pass `g_cfg.rope_dim / 2` instead of `CFG_Dh / 2` when indexing.

---

## Step 3 — `ops.h / ops.c`: computation kernels

### 3a. Reuse what you can

| Architecture need | Existing function |
|-------------------|-------------------|
| LayerNorm (μ, σ) | `layer_norm()` |
| RMSNorm | `rms_norm()` |
| Causal MHA + full KV cache | `attention_forward()` (GPT-2) |
| Causal MHA/GQA + RoPE + KV cache | `llama_attention_forward()` |
| Two-layer GELU MLP | GPT-2 branch in `transformer_block_forward()` |
| SwiGLU (gate · up → down) | LLaMA branch in `transformer_block_forward()` |

### 3b. Add only what is genuinely new

**GeGLU** (Gemma / PaLM): replace `silu()` with `gelu()` in the gate product:

```c
/* In ops.c — add next to swiglu_ffn_forward */
static void geglu_ffn_forward(float *restrict out, const float *restrict x,
                               const LayerWeights *lw,
                               float *restrict gate_buf, float *restrict up_buf)
{
    const int D = CFG_D, F = CFG_F;
    matmul_vec(gate_buf, lw->gate_weight, NULL, x, F, D);
    matmul_vec(up_buf,   lw->up_weight,   NULL, x, F, D);
    for (int i = 0; i < F; i++)
        gate_buf[i] = gelu(gate_buf[i]) * up_buf[i];   /* GELU instead of SiLU */
    matmul_vec(out, lw->down_weight, NULL, gate_buf, D, F);
}
```

### 3c. Dispatch in `transformer_block_forward()`

Add a branch alongside the existing `ARCH_LLAMA` / else branches:

```c
} else if (g_cfg.arch == ARCH_PHI3) {

    /* Pre-attention RMSNorm */
    rms_norm(scratch_norm, x, lw->rms_attn_weight, D, CFG_EPS > 0 ? CFG_EPS : 1e-5f);

    /* QKV is fused; Phi-3 uses full MHA with RoPE — reuse llama path */
    float *q_buf = scratch_qkv;
    float *k_buf = scratch_qkv + D;
    float *v_buf = scratch_qkv + D + CFG_Dh * CFG_Hkv;
    matmul_vec(scratch_qkv, lw->qkv_weight, NULL, scratch_norm, 3 * D, D);
    {
        const float *cr = g_weights.rope_cos + (size_t)pos * (CFG_Dh / 2);
        const float *sr = g_weights.rope_sin + (size_t)pos * (CFG_Dh / 2);
        for (int h = 0; h < CFG_H;   h++) rope_apply(q_buf + h*CFG_Dh, cr, sr, CFG_Dh);
        for (int h = 0; h < CFG_Hkv; h++) rope_apply(k_buf + h*CFG_Dh, cr, sr, CFG_Dh);
    }
    /* ... store KV, compute attention, residual, FFN (SwiGLU) ... */
}
```

### 3d. Update `model_forward()`

Add the final normalisation and embedding lookup for the new arch:

```c
if (g_cfg.arch == ARCH_LLAMA || g_cfg.arch == ARCH_PHI3) {
    /* No position embedding — RoPE handles it */
    memcpy(x, tok_emb, (size_t)D * sizeof(float));
} else {
    /* GPT-2: token + absolute position */
    const float *pos_emb = g_weights.wpe + (size_t)pos * D;
    for (int i = 0; i < D; i++) x[i] = tok_emb[i] + pos_emb[i];
}
```

---

## Step 4 — `gguf.h / gguf.c`: file loading

### 4a. Add metadata key constants in `gguf.h`

```c
/* ── Phi-3 metadata keys ──────────────────────────────────────────────── */
#define GGUF_KEY_PHI3_LAYERS    "phi3.block_count"
#define GGUF_KEY_PHI3_HEADS     "phi3.attention.head_count"
#define GGUF_KEY_PHI3_KV_HEADS  "phi3.attention.head_count_kv"
#define GGUF_KEY_PHI3_EMBED     "phi3.embedding_length"
#define GGUF_KEY_PHI3_FFN       "phi3.feed_forward_length"
#define GGUF_KEY_PHI3_CTX       "phi3.context_length"
#define GGUF_KEY_PHI3_ROPE      "phi3.rope.freq_base"
#define GGUF_KEY_PHI3_NORM_EPS  "phi3.attention.layer_norm_rms_epsilon"
```

### 4b. Parse them in the KV loop in `load_model_gguf()` in `gguf.c`

Add new `else if` branches in the `GGUF_MTYPE_UINT32` block:

```c
} else if (vtype == GGUF_MTYPE_UINT32) {
    uint32_t val = gguf_u32(f);
    /* … existing gpt2.*, llama.*, mistral.* … */

    /* Phi-3 */
    else if (!strcmp(key,"phi3.block_count"))              g_cfg.n_layers   = (int)val;
    else if (!strcmp(key,"phi3.attention.head_count"))     g_cfg.n_heads    = (int)val;
    else if (!strcmp(key,"phi3.attention.head_count_kv"))  g_cfg.n_kv_heads = (int)val;
    else if (!strcmp(key,"phi3.embedding_length"))         g_cfg.embed_dim  = (int)val;
    else if (!strcmp(key,"phi3.feed_forward_length"))      g_cfg.ffn_dim    = (int)val;
    else if (!strcmp(key,"phi3.context_length"))           g_cfg.seq_len    = (int)val;
```

And the `GGUF_MTYPE_FLOAT32` block:

```c
    else if (!strcmp(key,"phi3.rope.freq_base"))           g_cfg.rope_theta = val;
```

### 4c. Resolve the architecture string

In the `arch_str` resolution block after the KV loop:

```c
if (!strcmp(arch_str,"llama") || !strcmp(arch_str,"mistral")) {
    g_cfg.arch = ARCH_LLAMA;
    /* … */
} else if (!strcmp(arch_str,"phi3")) {
    g_cfg.arch = ARCH_PHI3;
    if (g_cfg.n_kv_heads == 0)  g_cfg.n_kv_heads = g_cfg.n_heads;
    if (g_cfg.rope_theta == 0.0f) g_cfg.rope_theta = 10000.0f;
    if (g_cfg.seq_len > MAX_SEQ_LEN) g_cfg.seq_len = MAX_SEQ_LEN;
} else {
    g_cfg.arch = ARCH_GPT2;
    /* … */
}
```

### 4d. Add tensor name mappings in `gguf_name_to_ptr()`

Add a branch inside the `if (strncmp(name,"blk.",4) == 0)` block:

```c
} else if (g_cfg.arch == ARCH_PHI3) {
    if (!strcmp(rest,"attn_norm.weight"))   return &lw->rms_attn_weight;
    if (!strcmp(rest,"attn_qkv.weight"))    return &lw->qkv_weight;   /* reuse */
    if (!strcmp(rest,"attn_output.weight")) return &lw->attn_proj_weight;
    if (!strcmp(rest,"ffn_norm.weight"))    return &lw->rms_ffn_weight;
    if (!strcmp(rest,"ffn_gate.weight"))    return &lw->gate_weight;
    if (!strcmp(rest,"ffn_up.weight"))      return &lw->up_weight;
    if (!strcmp(rest,"ffn_down.weight"))    return &lw->down_weight;
}
```

> **Finding the real tensor names**: dump them with
> `python3 -c "import gguf; r=gguf.GGUFReader('model.gguf'); [print(t.name) for t in r.tensors]"`
> or with the `gguf-dump` CLI tool from the llama.cpp repo.

### 4e. Update the tensor count sanity check

The `max_tensors` guard in `load_model_gguf()` uses `GGUF_GPT2_TENSORS_PER_LAYER`.
Either increase the constant or add an arch-aware calculation:

```c
int tensors_per_layer = (g_cfg.arch == ARCH_LLAMA || g_cfg.arch == ARCH_PHI3)
                      ? GGUF_LLAMA_TENSORS_PER_LAYER
                      : GGUF_GPT2_TENSORS_PER_LAYER;
int tensors_global    = (g_cfg.arch == ARCH_LLAMA || g_cfg.arch == ARCH_PHI3)
                      ? GGUF_LLAMA_TENSORS_GLOBAL
                      : GGUF_GPT2_TENSORS_GLOBAL;
const int max_tensors = CFG_L * tensors_per_layer + tensors_global + 4;
```

### 4f. Add a variant log line

```c
} else if (g_cfg.arch == ARCH_PHI3) {
    LMC_INFO("Architecture: Phi-3");
    if      (CFG_L==32 && CFG_D==3072) LMC_INFO("Variant: Phi-3 Mini (3.8B)");
    else if (CFG_L==40 && CFG_D==5120) LMC_INFO("Variant: Phi-3 Medium (14B)");
    else                               LMC_INFO("Variant: Phi-3 custom (%dL/%dD)", CFG_L, CFG_D);
}
```

---

## Step 5 — `lmc.c`: auto-detection

Add the model's common GGUF filenames to `candidates[]` in `find_default_model()`:

```c
static const char *candidates[] = {
    /* … existing … */
    "phi-3-mini.gguf",  "phi-3-mini-instruct.gguf",
    "phi-3-medium.gguf","phi-3-medium-instruct.gguf",
    NULL
};
```

---

## Architecture-specific notes

### Gemma / Gemma-2

- `general.architecture` = `"gemma"` / `"gemma2"`
- Uses RMSNorm, RoPE, GQA, and **GeGLU** (GELU gate rather than SiLU)
- Gemma pre-normalises both input and output of the attention block
  (pre-norm + post-norm); add a second `rms_post_attn_weight` field to
  `LayerWeights` and apply it after the residual add
- Tensor name prefix: `blk.N.*` (same as LLaMA)
- Gemma-2 adds a sliding-window attention pattern (every other layer uses a
  4096-token local window); implement by masking `scores[t] = -INFINITY`
  for `t < pos - window_size` before the softmax

### Qwen-2 / Qwen-2.5

- `general.architecture` = `"qwen2"`
- Structurally identical to LLaMA-2 with GQA; the existing `ARCH_LLAMA`
  code path handles it if you add the `qwen2.*` KV keys (same pattern as
  `mistral.*` in the existing code)
- Add to the arch-string resolution block: `|| !strcmp(arch_str,"qwen2")`

### Phi-3 / Phi-3.5

- `general.architecture` = `"phi3"`
- Uses fused QKV projection (one `[3D × D]` matrix, like GPT-2)
  but with RMSNorm and RoPE (like LLaMA) — mix the two paths
- Partial RoPE: only the first `partial_rotary_factor * head_dim` dimensions
  are rotated; the remainder are passed through unchanged.  Store
  `g_cfg.rope_dim = partial_rotary_factor * head_dim` and use it in
  `rope_apply()` as the loop bound instead of `head_dim`

### BitNet b1.58

- `general.architecture` = `"bitnet"`
- Weights are ternary `{-1, 0, +1}`, packed as 2-bit values
- The dequantisation step is trivial (no scale, no bias), but you need a new
  `GGUF_TYPE_Q1_5` (or similar) constant in `quant.h` and a `dequant_q1_5()`
  in `quant.c` that maps `00→0, 01→+1, 10→-1`
- All other code (attention, FFN, normalisation) is standard LLaMA SwiGLU;
  set `g_cfg.arch = ARCH_LLAMA` and let the existing kernels run on the
  dequantised floats — correct, though not as fast as a native int8 kernel

### Whisper (encoder-decoder)

Whisper is an encoder-decoder architecture; the decode loop in `lmc.c` assumes
decoder-only autoregression and would need significant restructuring:

1. Add `ARCH_WHISPER` and a second model-weights struct `WhisperWeights`
   (or extend `ModelWeights` with encoder-specific fields)
2. The **encoder** runs a single full forward pass over the 80-band mel
   spectrogram (shape `[1500 × 80]`) through convolutional feature extraction
   and then standard transformer blocks
3. The **decoder** uses cross-attention: K and V come from the encoder output,
   not the decoder's own KV cache.  Add `cross_k_weight` / `cross_v_weight`
   fields to `LayerWeights` and a second attention call in the decoder block
4. The generation loop must first run the encoder, store its output, then
   autoregressively decode tokens using that encoder output as the cross-KV
   source

> Whisper is architecturally far from a decoder-only model; treat it as a
> larger refactor rather than a drop-in addition.

---

## Checklist

Before opening a PR or calling the port "done":

- [ ] `make` and `make omp` both compile with zero warnings
- [ ] `./lmc_omp --model <new_model>.gguf --prompt "Hello" --n-predict 20` produces
       coherent output
- [ ] Variant detection log line shows the correct model size
- [ ] KV cache MB reported matches `L × Hkv × S × Dh × 2 × 4 bytes`
- [ ] Parameters reported matches independent count (e.g. from `gguf-dump`)
- [ ] Add the new model filenames to `candidates[]` in `lmc.c`
- [ ] Add the new `ARCH_*` constant to the architecture table in `README.md`
