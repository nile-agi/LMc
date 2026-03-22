/* =============================================================================
 * tinyllama.h  —  TinyLlama-1.1B inference for LMc
 *
 * Architecture: Llama-2 family (GQA · RoPE · RMSNorm · SwiGLU)
 * Supported quants: Q2_K · Q3_K_S/M/L · Q4_K_S/M · Q4_0 · Q5_K_S/M · Q5_0
 *                   Q6_K · Q8_0  (dequant via existing quant.h kernels)
 *
 * C99 — no C++ — no external libs beyond -lm
 * =============================================================================
 */
#ifndef TINYLLAMA_H
#define TINYLLAMA_H

#include <stdint.h>
#include <stddef.h>

/* ── forward declarations for types defined in the rest of LMc ─────────────
 *    Declared here as incomplete struct types so tinyllama.h is self-contained
 *    without pulling in gguf.h / utils.h at header inclusion time.
 *    tinyllama.c includes gguf.h and utils.h directly.                      */
#ifndef GGUF_CTX_DECLARED
#define GGUF_CTX_DECLARED
struct GgufCtx;
typedef struct GgufCtx GgufCtx;
#endif

#ifndef ARENA_DECLARED
#define ARENA_DECLARED
struct Arena;
typedef struct Arena Arena;
#endif

/* ──────────────────────────────────────────────────────────────────────────
 * 1.  Model hyper-parameters
 *     Populated from GGUF metadata keys (llama.*)
 * ────────────────────────────────────────────────────────────────────────── */
typedef struct {
    int     n_vocab;        /* 32 000  — tokenizer.ggml.* determines exact  */
    int     n_embd;         /* 2 048   — llama.embedding_length              */
    int     n_layer;        /* 22      — llama.block_count                   */
    int     n_head;         /* 32      — llama.attention.head_count          */
    int     n_kv_head;      /* 4       — llama.attention.head_count_kv       */
    int     n_ff;           /* 5 632   — llama.feed_forward_length           */
    int     n_ctx;          /* 2 048   — llama.context_length                */
    int     head_dim;       /* n_embd / n_head  (computed, = 64)             */
    int     kv_groups;      /* n_head  / n_kv_head (= 8)                    */
    float   rope_freq_base; /* 10 000.0                                      */
    float   rms_eps;        /* 1e-5                                          */
    int     bos_id;         /* 1                                             */
    int     eos_id;         /* 2                                             */
} TLConfig;

/* ──────────────────────────────────────────────────────────────────────────
 * 2.  Weight tensors
 *     Each pointer is to raw GGUF tensor data (may be quantised).
 *     We pair each weight with its GGML type so callers know the stride.
 * ────────────────────────────────────────────────────────────────────────── */

/* Opaque handle for a (possibly-quantised) tensor */
typedef struct {
    void   *data;      /* raw block pointer from GGUF mmap                  */
    int     type;      /* GGML_TYPE_* (F32=0, F16=1, Q4_0=2, etc.)         */
    int     rows;
    int     cols;
} TLTensor;

#define TL_MAX_LAYERS 32   /* enough for any Llama-family variant            */

typedef struct {
    /* token embedding table: [n_vocab, n_embd] */
    TLTensor tok_embd;

    /* per-layer weights */
    TLTensor rms_attn_w [TL_MAX_LAYERS];   /* RMSNorm before attention  [n_embd] */
    TLTensor wq         [TL_MAX_LAYERS];   /* query    [n_head*head_dim, n_embd] */
    TLTensor wk         [TL_MAX_LAYERS];   /* key      [n_kv_head*head_dim,n_embd]*/
    TLTensor wv         [TL_MAX_LAYERS];   /* value    [n_kv_head*head_dim,n_embd]*/
    TLTensor wo         [TL_MAX_LAYERS];   /* output   [n_embd, n_head*head_dim] */

    TLTensor rms_ffn_w  [TL_MAX_LAYERS];   /* RMSNorm before FFN        [n_embd] */
    TLTensor w_gate     [TL_MAX_LAYERS];   /* SwiGLU gate  [n_ff, n_embd]        */
    TLTensor w_up       [TL_MAX_LAYERS];   /* SwiGLU up    [n_ff, n_embd]        */
    TLTensor w_down     [TL_MAX_LAYERS];   /* SwiGLU down  [n_embd, n_ff]        */

    /* final RMSNorm + output projection */
    TLTensor rms_final_w;                  /* [n_embd]                           */
    TLTensor lm_head;                      /* [n_vocab, n_embd]  (not tied)      */
} TLWeights;

/* ──────────────────────────────────────────────────────────────────────────
 * 3.  KV-cache  (head-major layout for cache-friendly decode)
 *     Shape: [n_layer][n_kv_head][n_ctx][head_dim]
 * ────────────────────────────────────────────────────────────────────────── */
typedef struct {
    float *k;   /* [n_layer * n_kv_head * n_ctx * head_dim] */
    float *v;   /* [n_layer * n_kv_head * n_ctx * head_dim] */
    int    n_ctx_max;
} TLKVCache;

/* ──────────────────────────────────────────────────────────────────────────
 * 4.  Per-forward-pass activations  (arena-allocated scratch buffers)
 * ────────────────────────────────────────────────────────────────────────── */
typedef struct {
    float *x;           /* current hidden state              [n_embd]        */
    float *xb;          /* residual buffer                   [n_embd]        */
    float *xb2;         /* second residual / attn input      [n_embd]        */

    float *q;           /* query  (dequant+rope'd)   [n_head*head_dim]       */
    float *k_cur;       /* key    (current token)    [n_kv_head*head_dim]    */
    float *v_cur;       /* value  (current token)    [n_kv_head*head_dim]    */

    float *attn_score;  /* attention logits          [n_head * n_ctx]        */
    float *attn_out;    /* weighted sum of V's       [n_embd]                */

    float *gate;        /* SwiGLU gate projection    [n_ff]                  */
    float *up;          /* SwiGLU up   projection    [n_ff]                  */
    float *hb;          /* hidden FFN buffer         [n_ff]                  */

    float *logits;      /* final output over vocab   [n_vocab]               */
} TLActivations;

/* ──────────────────────────────────────────────────────────────────────────
 * 5.  Top-level model handle
 * ────────────────────────────────────────────────────────────────────────── */
typedef struct {
    TLConfig     cfg;
    TLWeights    w;
    TLKVCache    kv;
    TLActivations act;
} TLModel;

/* ──────────────────────────────────────────────────────────────────────────
 * 6.  Public API
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * tinyllama_load  — populate TLModel from an open GgufCtx.
 *
 * @param model   caller-allocated TLModel (zeroed)
 * @param ctx     GGUF context already parsed (gguf_open)
 * @param arena   arena for activations / KV-cache
 * @returns 0 on success, non-zero on error
 */
int tinyllama_load(TLModel *model, GgufCtx *ctx, Arena *arena);

/**
 * tinyllama_forward  — single-token forward pass.
 *
 * @param model   loaded model
 * @param token   input token id
 * @param pos     current sequence position (0-based)
 * @returns pointer to logits buffer [n_vocab]  (owned by model->act)
 */
float *tinyllama_forward(TLModel *model, int token, int pos);

/**
 * tinyllama_free  — release KV-cache / activations (NOT the GGUF mmap).
 */
void tinyllama_free(TLModel *model);

#endif /* TINYLLAMA_H */
