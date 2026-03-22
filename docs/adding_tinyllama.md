# Adding TinyLlama to LMc: A Deep-Dive Guide

> **Who this is for** — You understand what a neural network is and want to
> go from "I know GPT-2 runs" to "I understand every tensor in TinyLlama and
> can add any Llama-family model to LMc myself."

---

## Table of Contents

1. [What Is TinyLlama?](#1-what-is-tinyllama)
2. [LMc's Architecture — The Existing GPT-2 Path](#2-lmcs-architecture)
3. [GPT-2 vs TinyLlama — Side-by-Side](#3-gpt-2-vs-tinyllama)
4. [Every Tensor in TinyLlama — Shapes & Purpose](#4-every-tensor-in-tinyllama)
5. [The Four New Operations](#5-the-four-new-operations)
6. [Quantisation in GGUF — All 11 Formats Explained](#6-quantisation-in-gguf)
7. [GGUF Tensor Names — What LMc Looks Up](#7-gguf-tensor-names)
8. [The SentencePiece Tokenizer vs GPT-2 BPE](#8-tokenizer-comparison)
9. [Files We Added and Why](#9-files-we-added)
10. [How to Download and Run](#10-how-to-download-and-run)
11. [Performance Expectations](#11-performance-expectations)

---

## 1. What Is TinyLlama?

TinyLlama-1.1B-Chat is a **1.1 billion parameter** causal language model from
the [TinyLlama project](https://github.com/jzhang38/TinyLlama).

Key facts:
- Architecture: **Llama 2** (same family as Meta's LLaMA / Mistral)
- Parameters: ~1.1B  (roughly 9× GPT-2 Small)
- Trained on: 3 trillion tokens
- Chat model: fine-tuned on OpenAssistant conversations with SFT + DPO
- GGUF name: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`

Why it matters for edge computing:
- At Q4_K_M it is ~670 MB — fits on a Raspberry Pi 4 (4 GB)
- It produces coherent multi-turn chat; GPT-2 at the same size cannot
- It uses the same GGUF format LMc already knows how to parse

---

## 2. LMc's Architecture

LMc is structured around a clean separation of concerns:

```
gguf.h/c        ← GGUF file parser (reads tensors by name, returns raw data)
quant.h/c       ← Dequantisation kernels (Q2_K → float, Q8_0 → float, etc.)
models.h/c      ← Struct definitions: config, weights, KV-cache, activations
ops.h/c         ← Math kernels: matmul, LayerNorm, GELU, softmax, attention
tokenizer.h/c   ← GPT-2 BPE tokenizer (encoder.json + vocab.bpe)
utils.h/c       ← Arena allocator, logging
lmc.c           ← main(): arg parsing → gguf_open → model_load → generate loop
```

The quantisation story is simple: every weight stays in its quantised format
on disk (mmap'd into memory). When we need to multiply a weight matrix by an
activation vector, we call `dequant_row(weight_data, type, row, float_buf, cols)`
to get one row as floats, then do a dot product. This is the standard
"on-the-fly dequantisation" approach used by llama.cpp.

**Quant types LMc already supports** (from `quant.h`):
`F32 · F16 · Q2_K · Q3_K · Q4_0 · Q4_1 · Q4_K · Q5_0 · Q5_1 · Q5_K ·
Q6_K · Q8_0 · IQ3_XXS · IQ3_S · IQ4_XS`

All 11 TinyLlama quant levels sit inside that list. **We do not need to add
any new dequantisation code.** The only additions are:

1. A new model struct (`TLModel`) for TinyLlama's different hyper-params
2. A new forward pass that uses different ops (RMSNorm, RoPE, GQA, SwiGLU)
3. A new tokenizer (`SpmCtx`) for the 32k-token SentencePiece vocabulary

---

## 3. GPT-2 vs TinyLlama — Side-by-Side

This is the most important table in the document. Every difference here
corresponds to a block of code we had to add.

| Feature | GPT-2 (existing) | TinyLlama (new) |
|---------|-----------------|-----------------|
| **Normalisation** | LayerNorm (mean + variance) | RMSNorm (root-mean-square only) |
| **Positional encoding** | Learned lookup table `wpe[pos]` | RoPE (Rotary, applied to Q & K) |
| **Attention type** | Standard MHA (n_head == n_kv_head) | GQA — 32 Q heads, **4** KV heads |
| **MLP non-linearity** | GELU(x) | SwiGLU: SiLU(gate) ⊙ up |
| **MLP structure** | 2 matrices: W_fc → W_proj | 3 matrices: W_gate, W_up, W_down |
| **Linear bias** | Yes (all projections have bias) | No (zero bias everywhere) |
| **Vocabulary** | 50 257 (GPT-2 BPE) | 32 000 (SentencePiece BPE) |
| **Tokenizer source** | `encoder.json` + `vocab.bpe` files | Embedded in GGUF metadata |
| **Context length** | 1 024 tokens | 2 048 tokens |
| **Number of layers** | 12 (Small) to 48 (XL) | 22 |
| **Hidden dim** | 768 (Small) to 1600 (XL) | 2 048 |
| **Head dim** | 64 | 64 (same!) |
| **Weight tying** | `lm_head = wte` (shared) | Separate `output.weight` |

### Why does GQA exist?

In standard MHA (GPT-2), every attention head has its own Q, K, V. For 32
heads × 64 dims × 2048 ctx, the KV cache for one layer takes:

```
2 × 32 × 2048 × 64 × 4 bytes = 33 MB per layer
× 22 layers = 726 MB  just for KV cache
```

GQA (Grouped Query Attention) solves this by sharing KV heads:

```
2 × 4 × 2048 × 64 × 4 bytes = 4 MB per layer
× 22 layers = 91 MB   (8× smaller KV cache!)
```

The 32 Q heads are divided into 4 groups of 8. Each group shares one KV head.
Q can still be distinct per head (capturing different relationships), but K/V
are amortised across the group. This is the primary memory saving in TinyLlama
relative to a naive Llama 2 7B port.

### Why RMSNorm instead of LayerNorm?

LayerNorm:
```c
mean = sum(x) / n
var  = sum((x - mean)^2) / n
y    = (x - mean) / sqrt(var + eps) * gamma + beta
```

RMSNorm:
```c
rms = sqrt(sum(x^2) / n + eps)
y   = (x / rms) * w
```

RMSNorm drops the mean-centring step and the `beta` bias. It is about **7%
faster** in practice and empirically just as effective. It also has **half
the parameters** (no beta per layer).

### Why SwiGLU instead of GELU?

GPT-2 MLP (2 matrices):
```
h = GELU(x @ W_fc + b_fc)
y = h @ W_proj + b_proj
```

TinyLlama MLP / SwiGLU (3 matrices, no bias):
```
gate = W_gate @ x          # shape [n_ff]
up   = W_up   @ x          # shape [n_ff]
h    = SiLU(gate) * up     # element-wise gate
y    = W_down @ h          # shape [n_embd]
```

`SiLU(x) = x * sigmoid(x)` — a smooth, self-gated activation. The "gate"
tensor learns which features to pass through; the "up" tensor contains the
actual values. This is empirically better than plain GELU at the same
parameter count, which is why all modern Llama-family models use it.

### Why RoPE instead of learned positional embeddings?

GPT-2 learns a lookup table `wpe[pos]` of shape `[1024, 768]`. This:
- Wastes parameters (768K extra weights)
- Cannot extrapolate beyond training context (1024 tokens hard limit)

RoPE applies a rotation in the complex plane to each pair of Q/K dimensions
based on the token position. The rotation angle for pair `(d, d+1)` is:

```
θ_d = pos / (freq_base ^ (2d / head_dim))
```

This is computed during the forward pass (no extra parameters), and the
relative position between any two tokens is naturally encoded in the dot
product after rotation. RoPE enables context extension without retraining.

---

## 4. Every Tensor in TinyLlama — Shapes & Purpose

### Token Embedding

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `token_embd.weight` | `[32000, 2048]` | Maps token ID → 2048-dim vector |

**Compare GPT-2 Small:**
- `model.wte` = `[50257, 768]` — larger vocab, smaller embedding
- `model.wpe` = `[1024, 768]` — positional table (TinyLlama has no equivalent!)

### Per-Layer Tensors (×22 layers)

Let:
- `E = 2048` (embedding / hidden dim)
- `H = 32` (Q attention heads)
- `Hk = 4` (KV attention heads)
- `D = 64` (head dim = E / H)
- `F = 5632` (FFN hidden dim; = 11/4 × E, rounded up to multiple of 256)

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `blk.L.attn_norm.weight` | `[E]` = `[2048]` | RMSNorm scale before attention |
| `blk.L.attn_q.weight` | `[H×D, E]` = `[2048, 2048]` | Query projection |
| `blk.L.attn_k.weight` | `[Hk×D, E]` = `[256, 2048]` | Key projection (4 heads only!) |
| `blk.L.attn_v.weight` | `[Hk×D, E]` = `[256, 2048]` | Value projection (4 heads only!) |
| `blk.L.attn_output.weight` | `[E, H×D]` = `[2048, 2048]` | Project attention output back |
| `blk.L.ffn_norm.weight` | `[E]` = `[2048]` | RMSNorm scale before FFN |
| `blk.L.ffn_gate.weight` | `[F, E]` = `[5632, 2048]` | SwiGLU gate projection |
| `blk.L.ffn_up.weight` | `[F, E]` = `[5632, 2048]` | SwiGLU up projection |
| `blk.L.ffn_down.weight` | `[E, F]` = `[2048, 5632]` | SwiGLU down projection |

**Compare GPT-2 Small (12 layers, per layer):**

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `h.L.ln_1.weight` | `[768]` | LayerNorm γ |
| `h.L.ln_1.bias` | `[768]` | LayerNorm β (no equivalent in TinyLlama!) |
| `h.L.attn.c_attn.weight` | `[768, 2304]` | Q+K+V fused (768×3) |
| `h.L.attn.c_attn.bias` | `[2304]` | Bias (no equivalent in TinyLlama!) |
| `h.L.attn.c_proj.weight` | `[768, 768]` | Output projection |
| `h.L.mlp.c_fc.weight` | `[768, 3072]` | MLP up (4×) |
| `h.L.mlp.c_proj.weight` | `[3072, 768]` | MLP down |

### Final Tensors

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `output_norm.weight` | `[2048]` | Final RMSNorm scale |
| `output.weight` | `[32000, 2048]` | Language-model head (un-tied) |

**GPT-2 comparison:** GPT-2 _ties_ the LM head to `wte` — it reuses the same
weight matrix for both embedding and output. TinyLlama keeps them separate,
trading a small memory cost for better fine-tuning flexibility.

### Total Parameter Count

**TinyLlama 1.1B:**
```
token_embd:      32000 × 2048                     =   65.5M
22 layers:
  attn_norm:     22 × 2048                         =    0.09M
  wq:            22 × 2048 × 2048                  =   92.3M
  wk:            22 × 256  × 2048                  =   11.5M
  wv:            22 × 256  × 2048                  =   11.5M
  wo:            22 × 2048 × 2048                  =   92.3M
  ffn_norm:      22 × 2048                         =    0.09M
  w_gate:        22 × 5632 × 2048                  =  253.7M
  w_up:          22 × 5632 × 2048                  =  253.7M
  w_down:        22 × 2048 × 5632                  =  253.7M
output_norm:     2048                              =    0.002M
lm_head:         32000 × 2048                      =   65.5M
─────────────────────────────────────────────────────────────
TOTAL:                                             ≈ 1100M (1.1B) ✓
```

---

## 5. The Four New Operations

### 5.1 RMSNorm

```c
// y_i = w_i * x_i / sqrt(mean(x^2) + eps)
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (ss * x[i]);
}
```

Parameters: `eps = 1e-5` (from GGUF: `llama.attention.layer_norm_rms_epsilon`)

### 5.2 RoPE

```c
// For each pair of dimensions in each head:
// [v0, v1] → [v0*cos(θ) - v1*sin(θ),  v0*sin(θ) + v1*cos(θ)]
// θ = pos / (freq_base ^ (2d / head_dim))
static void rope_apply(float *qk, int n_heads, int head_dim, int pos, float freq_base) {
    for (int h = 0; h < n_heads; h++) {
        float *v = qk + h * head_dim;
        for (int d = 0; d < head_dim / 2; d++) {
            float theta = pos / powf(freq_base, 2.0f * d / head_dim);
            float c = cosf(theta), s = sinf(theta);
            float v0 = v[2*d], v1 = v[2*d+1];
            v[2*d]   = v0*c - v1*s;
            v[2*d+1] = v0*s + v1*c;
        }
    }
}
```

This is applied to Q (32 heads) **and** K (4 heads) after their projections,
before storing K into the KV cache and computing attention scores.

### 5.3 Grouped-Query Attention

The core loop (pseudocode):
```
for each query head h in [0..31]:
    kv_head = h / 8          # which of the 4 KV heads to use
    
    for t in [0..pos]:       # over all past + current tokens
        score[t] = dot(Q[h], K_cache[kv_head][t]) / sqrt(64)
    
    softmax(score[0..pos])
    
    out[h] = sum over t:  score[t] * V_cache[kv_head][t]
```

The key insight: `K_cache` and `V_cache` only have 4 entries per layer position,
but the 32 Q heads index into them with `kv_head = h / 8`.

### 5.4 SwiGLU FFN

```c
gate = W_gate @ x              // [5632] = [5632×2048] @ [2048]
up   = W_up   @ x              // [5632]
for i in range(5632):
    h[i] = silu(gate[i]) * up[i]   // SiLU(gate) element-wise * up
y    = W_down @ h              // [2048] = [2048×5632] @ [5632]
```

`SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`

The "gate" decides what information flows through; "up" carries the content.
This is sometimes called a "gated MLP" or "GLU (Gated Linear Unit)".

---

## 6. Quantisation in GGUF — All 11 Formats Explained

All these formats are already handled by LMc's `quant.h`. You just need to
know what they mean to pick the right one for your hardware.

### How GGUF quantisation works

Instead of storing each weight as a 32-bit float (4 bytes), we store groups
of weights in compressed form with shared scale factors. The size reduction
is roughly `bits / 32`.

### The formats

| Format | Bits/weight | Size (TinyLlama) | Method |
|--------|-------------|-----------------|--------|
| `Q2_K` | 2.63 | ~366 MB | 2-bit with 4-bit super-scales |
| `Q3_K_S` | 3.00 | ~418 MB | 3-bit, small groups |
| `Q3_K_M` | 3.35 | ~467 MB | 3-bit, medium groups |
| `Q3_K_L` | 3.60 | ~502 MB | 3-bit, large groups |
| `Q4_K_S` | 4.37 | ~609 MB | 4-bit K-quant, small |
| `Q4_0` | 4.50 | ~628 MB | 4-bit, 32-element blocks |
| `Q4_K_M` | 4.85 | ~676 MB | **Best balance — recommended** |
| `Q5_K_S` | 5.21 | ~727 MB | 5-bit K-quant, small |
| `Q5_0` | 5.00 | ~698 MB | 5-bit, 32-element blocks |
| `Q5_K_M` | 5.68 | ~792 MB | 5-bit K-quant, medium |
| `Q6_K` | 6.57 | ~916 MB | 6-bit K-quant |
| `Q8_0` | 8.50 | ~1185 MB | 8-bit, near lossless |
| `F16` | 16.00 | ~2230 MB | Half-float, reference |

### What does "K" mean? (K-quants)

"K-quants" (Q2_K, Q3_K, Q4_K etc.) use a **hierarchical scaling** approach:

```
block (256 weights):
   super-block scale (6-bit)
   min value (6-bit)
   sub-blocks (e.g. 8 × 32 weights):
       per-sub-block scale (4-bit or 6-bit)
       quantised values (2-bit, 3-bit, 4-bit, etc.)
```

The "S/M/L" suffix (small/medium/large) refers to the precision of the
sub-block scales: more bits → higher quality → larger file.

### What does "0" mean? (Q4_0, Q5_0, Q8_0)

"Baseline" quants: one scale per 32-weight block, no minimum value stored.
Simpler to decode (fewer operations), slightly lower quality than K-quants
at the same bit rate. Q8_0 is so close to F32 quality that it is used as the
reference for benchmarking.

### Which to use?

| Scenario | Recommendation |
|----------|----------------|
| < 1 GB RAM | Q2_K (~366 MB) |
| 1–2 GB RAM | Q4_K_M (~676 MB) — sweet spot |
| 2–4 GB RAM | Q6_K (~916 MB) or Q8_0 |
| Speed priority | Q4_0 (simpler decode) |
| Quality priority | Q8_0 |

---

## 7. GGUF Tensor Names — What LMc Looks Up

When `tinyllama_load()` calls `gguf_get_tensor(ctx, name, &info)`, these are
the exact string names it expects in the GGUF file:

```
token_embd.weight            ← embedding table

blk.0.attn_norm.weight       ← layer 0 RMSNorm (attention)
blk.0.attn_q.weight          ← layer 0 Q projection
blk.0.attn_k.weight          ← layer 0 K projection
blk.0.attn_v.weight          ← layer 0 V projection
blk.0.attn_output.weight     ← layer 0 output projection
blk.0.ffn_norm.weight        ← layer 0 RMSNorm (FFN)
blk.0.ffn_gate.weight        ← layer 0 SwiGLU gate
blk.0.ffn_up.weight          ← layer 0 SwiGLU up
blk.0.ffn_down.weight        ← layer 0 SwiGLU down

... (repeat for blk.1 through blk.21)

output_norm.weight           ← final RMSNorm
output.weight                ← language-model head
```

You can verify these with `llama.cpp`'s `gguf-dump` tool:
```bash
./gguf-dump tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | grep tensor
```

---

## 8. Tokenizer Comparison

### GPT-2 BPE (existing `tokenizer.h`)

- Vocabulary: **50 257 tokens**
- Source: external files `encoder.json` (id→string) + `vocab.bpe` (merge rules)
- Algorithm: Byte-Pair Encoding, learns merges bottom-up from corpus
- Word boundary: none (uses byte-level representation)
- Special tokens: `<|endoftext|>` = 50256

### TinyLlama SentencePiece BPE (`spm_tok.h`)

- Vocabulary: **32 000 tokens**
- Source: **embedded in GGUF** under `tokenizer.ggml.tokens` (string array)
  and `tokenizer.ggml.scores` (float array)
- Algorithm: Unigram-BPE hybrid (SentencePiece), merges ranked by log-prob
- Word boundary: `▁` (U+2581) prepended to words (instead of GPT-2 `Ġ`)
- Special tokens: BOS=1, EOS=2, UNK=0, PAD=32000

### How SentencePiece BPE works

1. Start with every unicode character as its own symbol
2. Prepend `▁` (thin space) to the first character of each word
3. Scan all adjacent symbol pairs and look up their merged form in the vocab
4. Merge the pair with the **highest score** (= highest log-probability)
5. Repeat until no more merges are possible
6. Any character not in the vocabulary → byte fallback `<0xNN>`

Example encoding of `"Hello world"`:
```
Input:   Hello world
Step 1:  ▁H e l l o ▁w o r l d
Merge:   ▁He l l o ▁w o r l d       (▁He exists in vocab, high score)
Merge:   ▁Hell o ▁w o r l d         (▁Hell in vocab)
Merge:   ▁Hello ▁w o r l d          (▁Hello in vocab)
Merge:   ▁Hello ▁world              (▁world in vocab)
IDs:     [15043, 3186]
```

The `spm_tok.c` implementation uses a symbol linked-list and a max-heap of
candidate bigrams, giving O(n log n) time.

---

## 9. Files We Added and Why

```
src/tinyllama.h          ← Struct definitions (TLConfig, TLWeights, TLKVCache,
                            TLActivations, TLModel). Clean separation of data
                            from logic, following LMc's existing style.

src/tinyllama.c          ← The actual forward pass:
                            - rmsnorm()          § 5.1
                            - rope_apply()       § 5.2
                            - gqa_attention()    § 5.3
                            - swiglu_ffn()       § 5.4
                            - matvec()           handles all quant types via
                                                 dequant_row() from quant.h
                            - tinyllama_load()   reads GGUF metadata + tensors
                            - tinyllama_forward() ties it all together

src/spm_tok.h            ← SentencePiece tokenizer interface.
                            Separate from tokenizer.h (GPT-2) because the
                            two have incompatible APIs and data sources.

src/spm_tok.c            ← BPE merge algorithm, byte-fallback, GGUF vocab
                            reader.

src/lmc_tinyllama_integration.patch.c
                         ← Exact diff showing what to add to models.h and
                            lmc.c (arch detection, dispatch, generation loop).
```

### Why not modify ops.h?

`ops.h` already has `gelu`, `layer_norm`, `matmul`, `softmax`. The new ops
(RMSNorm, SwiGLU, RoPE, GQA) are specific to Llama-family and we keep them in
`tinyllama.c` to avoid polluting the shared ops with model-specific logic.
If you later add Mistral or Phi (also Llama-family), you can move them to
`ops.h` and share across models.

### Why does `TLTensor` wrap the existing GGUF tensor?

LMc's `gguf.h` returns a `GgufTensorInfo` with `data`, `type`, `ne[]`. We
wrap this in `TLTensor` which holds `{data, type, rows, cols}` — a thin
convenience wrapper. The underlying data is still mmap'd from disk; no copy.

---

## 10. How to Download and Run

### Step 1: Get a GGUF file

From HuggingFace (TheBloke's TinyLlama collection):

```bash
mkdir -p models

# 2-bit (tiny, low quality)
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/\
resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# 4-bit K-quant medium (RECOMMENDED — best quality/size balance)
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/\
resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 8-bit (near-lossless, ~1.2 GB)
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/\
resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

### Step 2: Build LMc with TinyLlama support

```bash
# Apply the integration patch (see lmc_tinyllama_integration.patch.c)
# Then rebuild:
make omp
```

### Step 3: Run

TinyLlama-Chat uses this prompt format:
```
<|system|>
You are a helpful assistant.
<|user|>
{your question}
<|assistant|>

```

```bash
./lmc_omp \
  --model   models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --prompt  "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello! How are you?\n<|assistant|>\n" \
  --n-predict 256 \
  --temp    0.7 \
  --top-p   0.9 \
  --threads 4
```

---

## 11. Performance Expectations

Based on the LMc benchmark methodology and TinyLlama's parameter count:

| Hardware | Quant | Expected t/s |
|----------|-------|-------------|
| i7-1165G7 (laptop, 8 threads) | Q4_K_M | ~18–25 t/s |
| i7-1165G7 (laptop, 8 threads) | Q8_0 | ~10–14 t/s |
| Raspberry Pi 4 (4 threads) | Q4_K_M | ~2–4 t/s |
| Raspberry Pi 4 (4 threads) | Q2_K | ~4–6 t/s |
| Android (4 threads) | Q4_K_M | ~3–6 t/s |

**Memory usage (RSS peak):**

| Quant | Model size | KV cache (2048 ctx) | Total |
|-------|-----------|---------------------|-------|
| Q2_K | 366 MB | 91 MB | ~460 MB |
| Q4_K_M | 676 MB | 91 MB | ~770 MB |
| Q8_0 | 1185 MB | 91 MB | ~1280 MB |

KV cache = `2 layers × 4 KV_heads × 2048 ctx × 64 head_dim × 4 bytes × 22 layers`
         = `2 × 4 × 2048 × 64 × 4 × 22 ≈ 91 MB`

---

## Quick Reference: Adding Any Llama-Family Model

The pattern established here works for any Llama 2 variant (Mistral 7B,
Llama 3, Phi-2, etc.). The only things that change are the config values
in `tinyllama_load()`:

| Model | n_embd | n_layer | n_head | n_kv_head | n_ff |
|-------|--------|---------|--------|-----------|------|
| TinyLlama 1.1B | 2048 | 22 | 32 | 4 | 5632 |
| Mistral 7B | 4096 | 32 | 32 | 8 | 14336 |
| Llama 3.2 1B | 2048 | 16 | 32 | 8 | 8192 |
| Phi-2 2.7B | 2560 | 32 | 32 | 32 | 10240 |

All read these values from GGUF metadata automatically — `tinyllama_load()`
already uses `meta_int()` with correct key names. Point it at any of these
GGUFs and it will configure itself. The forward pass code is identical.
