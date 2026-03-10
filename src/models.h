/* models.h — Model configuration, weight structures, KV cache, and activations.
 *
 * Supports GPT-2 (Small / Medium / Large / XL) and LLaMA / Mistral family
 * models through a unified ModelArch dispatch flag.  Adding a new architecture
 * requires only:
 *   1. Adding an ARCH_* constant to ModelArch.
 *   2. Populating the relevant fields in ModelConfig and LayerWeights.
 *   3. Adding a configure_<arch>() function in models.c.
 *   4. Wiring dispatch in ops.c (transformer_block_forward / model_forward).
 *
 * No existing GPT-2 kernel code changes when a new arch is added.
 */
#ifndef LMC_MODELS_H
#define LMC_MODELS_H

#include <stddef.h>
#include <stdint.h>
#include "utils.h"

/* ── Static compile-time limits ───────────────────────────────────────────── *
 * Used only for static array sizing.  All heap structures are runtime-sized. *
 * Raise MAX_N_LAYERS if you need LLaMA-3 70B (80 layers).                   */
#define MAX_N_LAYERS    80     /* GPT-2 XL = 48, LLaMA-3 70B = 80            */
#define MAX_VOCAB_SIZE  128256 /* GPT-2 = 50257, LLaMA-3 = 128256            */
#define MAX_SEQ_LEN     8192   /* GPT-2 = 1024,  LLaMA-3 8B = 8192          */

/* ── Architecture selector ────────────────────────────────────────────────── */
typedef enum {
    ARCH_UNKNOWN = 0,
    ARCH_GPT2,   /* LayerNorm, learned absolute pos emb, combined QKV, GELU  */
    ARCH_LLAMA,  /* RMSNorm, RoPE, separate Q/K/V, GQA, SwiGLU (gate×up→down)*/
} ModelArch;

/* ── Runtime architecture configuration ──────────────────────────────────── *
 * Fully populated by the file loader (gguf.c) before any computation.       *
 * All kernels access fields through CFG_* macros — adding a new field here  *
 * never causes a compile error in kernel files.                              */
typedef struct {
    /* --- Universal (all architectures) --- */
    ModelArch arch;       /* selects which transformer variant to run         */
    int vocab_size;       /* 50257 (GPT-2) / 32000 (LLaMA-2) / 128256 (LLaMA-3)*/
    int seq_len;          /* maximum context / sequence length                */
    int n_layers;         /* number of transformer blocks                     */
    int n_heads;          /* query head count (H)                             */
    int embed_dim;        /* residual stream width D                          */
    int ffn_dim;          /* FFN hidden dimension F; typically 4×D            */
    int head_dim;         /* D / n_heads  (always 64 for GPT-2)               */

    /* --- LLaMA / GQA extensions (zero / unused for GPT-2) --- */
    int   n_kv_heads;     /* key-value head count; equals n_heads for MHA     */
    int   n_kv_groups;    /* n_heads / n_kv_heads (1 for MHA / all GPT-2)     */
    float rope_theta;     /* RoPE base frequency (default 10000.0)            */
    float norm_eps;       /* ε for RMSNorm / LayerNorm (default 1e-5)         */
    int   rope_dim;       /* number of dims rotated by RoPE (= head_dim)      */
} ModelConfig;

extern ModelConfig g_cfg;

/* ── Convenience accessors ────────────────────────────────────────────────── */
#define CFG_V     g_cfg.vocab_size
#define CFG_S     g_cfg.seq_len
#define CFG_L     g_cfg.n_layers
#define CFG_H     g_cfg.n_heads
#define CFG_D     g_cfg.embed_dim
#define CFG_F     g_cfg.ffn_dim
#define CFG_Dh    g_cfg.head_dim
#define CFG_Hkv   g_cfg.n_kv_heads   /* GQA: ≤ CFG_H; equals CFG_H for GPT-2 */
#define CFG_KVG   g_cfg.n_kv_groups  /* = CFG_H / CFG_Hkv                     */
#define CFG_ROPE  g_cfg.rope_theta
#define CFG_EPS   g_cfg.norm_eps
#define CFG_ARCH  g_cfg.arch

/* ── Per-layer weights ────────────────────────────────────────────────────── *
 * Unified struct for all supported architectures.  Fields unused by the      *
 * active architecture are NULL; kernels check CFG_ARCH before dereferencing. *
 *                                                                             *
 *  GPT-2  uses: ln1_*, ln2_*, qkv_*, attn_proj_*, ffn_fc_*, ffn_proj_*     *
 *  LLaMA  uses: rms_*, q_*, k_*, v_*, attn_proj_*, gate_*, up_*, down_*    */
typedef struct {

    /* --- Pre-attention normalisation --- */
    float *ln1_weight;        /* [D]          GPT-2: LayerNorm 1 weight       */
    float *ln1_bias;          /* [D]          GPT-2: LayerNorm 1 bias         */
    float *rms_attn_weight;   /* [D]          LLaMA: RMSNorm before attention */

    /* --- Pre-FFN normalisation --- */
    float *ln2_weight;        /* [D]          GPT-2: LayerNorm 2 weight       */
    float *ln2_bias;          /* [D]          GPT-2: LayerNorm 2 bias         */
    float *rms_ffn_weight;    /* [D]          LLaMA: RMSNorm before FFN       */

    /* --- Attention — GPT-2 (combined QKV projection) --- */
    float *qkv_weight;        /* [3D × D]     fused Q, K, V projection        */
    float *qkv_bias;          /* [3D]                                          */

    /* --- Attention — LLaMA (separate, GQA-aware projections) --- */
    float *q_weight;          /* [D × D]          Q projection  (n_heads × Dh)*/
    float *k_weight;          /* [Hkv×Dh × D]     K projection  (n_kv_heads)  */
    float *v_weight;          /* [Hkv×Dh × D]     V projection  (n_kv_heads)  */

    /* --- Attention output projection (both architectures) --- */
    float *attn_proj_weight;  /* [D × D]                                       */
    float *attn_proj_bias;    /* [D]   NULL for LLaMA (no bias on projections) */

    /* --- FFN — GPT-2: two-layer MLP with GELU --- */
    float *ffn_fc_weight;     /* [F × D]      FC1 (expand)                     */
    float *ffn_fc_bias;       /* [F]                                            */
    float *ffn_proj_weight;   /* [D × F]      FC2 (contract)                   */
    float *ffn_proj_bias;     /* [D]                                            */

    /* --- FFN — LLaMA: SwiGLU gate network ---
     *   h = silu(gate(x)) ⊙ up(x)
     *   out = down(h)                                                          */
    float *gate_weight;       /* [F × D]      gate projection                  */
    float *up_weight;         /* [F × D]      up projection                    */
    float *down_weight;       /* [D × F]      down projection                  */

} LayerWeights;

/* ── Full model weights ───────────────────────────────────────────────────── */
typedef struct {
    /* --- Embeddings --- */
    float        *wte;          /* [V × D]   token embeddings (all archs)      */
    float        *wpe;          /* [S × D]   position embeddings (GPT-2 only)  */

    /* --- Transformer layers --- */
    LayerWeights *layers;       /* heap-allocated array [n_layers]              */

    /* --- Final normalisation --- */
    float        *ln_f_weight;  /* [D]   GPT-2:  final LayerNorm weight         */
    float        *ln_f_bias;    /* [D]   GPT-2:  final LayerNorm bias           */
    float        *rms_f_weight; /* [D]   LLaMA:  final RMSNorm weight           */

    /* --- Language model head --- */
    float        *lm_head;      /* [V × D]
                                 * .bin / GPT-2 tied weights: aliased to wte.
                                 * GGUF: "output.weight" tensor if present,
                                 *       otherwise falls back to wte.           */

    /* --- Rotary position embedding cache (LLaMA; NULL for GPT-2) ---
     * Precomputed at load time by init_rope_cache().
     * Layout: [seq_len × rope_dim/2], where rope_dim = head_dim.
     * cos[pos][i] = cos(pos / rope_theta^(2i/rope_dim))                       */
    float        *rope_cos;
    float        *rope_sin;
} ModelWeights;

extern ModelWeights g_weights;

/* ── KV cache ─────────────────────────────────────────────────────────────── *
 * Head-major layout within each layer:                                       *
 *   ptr = base_for_layer + kv_head * seq_len * head_dim + pos * head_dim     *
 *                                                                             *
 * For GQA (n_kv_heads < n_heads):                                            *
 *   kv_head = query_head % n_kv_heads                                        *
 *   Cache is sized by n_kv_heads, not n_heads — saves memory proportionally. *
 *                                                                             *
 * For MHA (n_kv_heads == n_heads): behaves identically to the GPT-2 cache.  */
typedef struct {
    float *k_cache;   /* [n_layers × n_kv_heads × seq_len × head_dim]         */
    float *v_cache;   /* same shape                                            */
    int    seq_len;   /* tokens currently cached; 0 = empty                   */
} KVCache;

extern KVCache g_kv_cache;

/* ── Per-forward-pass activation / scratch buffers ───────────────────────── *
 * One allocation per field; sized for the maximum dimension across all       *
 * layers.  LLaMA SwiGLU needs two FFN-dim buffers (gate + up); the extra    *
 * ffn_up field is NULL for GPT-2.                                            */
typedef struct {
    /* residual stream */
    float *x;           /* [D]          residual, modified in-place            */
    float *x_norm;      /* [D]          normalised copy (read-only kernel input)*/

    /* attention intermediates */
    float *qkv;         /* [3D]         GPT-2: fused QKV matmul output         */
    float *q;           /* [H × Dh]     LLaMA: Q after RoPE applied            */
    float *k_cur;       /* [Hkv × Dh]   LLaMA: K for current position          */
    float *v_cur;       /* [Hkv × Dh]   LLaMA: V for current position          */
    float *attn_out;    /* [D]          attention weighted-sum output           */
    float *proj_out;    /* [D]          attention output projection             */

    /* FFN intermediates */
    float *ffn_hidden;  /* [F]          GPT-2: FC1 output / LLaMA: gate·silu   */
    float *ffn_up;      /* [F]          LLaMA: up(x) buffer (NULL for GPT-2)   */
    float *ffn_out;     /* [D]          FFN block output                        */

    /* final output */
    float *logits;      /* [V]          un-normalised next-token logits         */

    /* per-layer scratch */
    float *attn_scores; /* [n_heads × seq_len]  softmax input/output            */

    /* top-p sampling scratch */
    void  *sorted_buf;  /* ProbIdx[V]: (float prob, int idx) pairs for sort     */
} Activations;

extern Activations g_act;

/* ── Model lifecycle ──────────────────────────────────────────────────────── */

/* Total parameter count (in float32 equivalents); used to size the arena.   */
size_t gpt2_total_params (void);
size_t llama_total_params(void);

/* Arena-allocate all weight slices and point g_weights fields at them.
 * Must be called after g_cfg has been fully populated by the loader.        */
void assign_weight_ptrs(void);

/* Allocate KV cache (heap, calloc).  Uses g_cfg.n_kv_heads for GQA sizing. */
void init_kv_cache(void);

/* Allocate per-forward-pass activation scratch buffers.                     */
void init_activations(void);
void free_activations(void);

/* Precompute RoPE cos/sin tables and store in g_weights.rope_{cos,sin}.
 * Allocates from the arena.  Must be called after assign_weight_ptrs().
 * No-op for GPT-2 (ARCH_GPT2).                                              */
void init_rope_cache(void);

#endif /* LMC_MODELS_H */
