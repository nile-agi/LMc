/* =============================================================================
 * tinyllama.c  —  TinyLlama-1.1B forward pass
 *
 * Key differences from the GPT-2 path already in LMc:
 *   · RMSNorm   instead of LayerNorm
 *   · RoPE      instead of learned positional embeddings
 *   · GQA       instead of standard MHA  (32 Q heads, 4 KV heads)
 *   · SwiGLU    instead of GELU MLP
 *   · No bias   on any linear projection
 *   · Quantised weights dequantised row-by-row via quant.h
 *
 * C99 — no C++ — no external libs beyond -lm
 * =============================================================================
 */

#include "tinyllama.h"
#include "quant.h"      /* dequant_row()                                     */
#include "gguf.h"       /* gguf_get_tensor(), gguf_get_meta_*()              */
#include "utils.h"      /* arena_alloc(), lmc_log(), LMC_ASSERT()           */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Scalar primitives
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float silu(float x) {
    /* SiLU(x) = x * σ(x)  =  x / (1 + e^{-x}) */
    return x / (1.0f + expf(-x));
}

/* ─── RMSNorm ────────────────────────────────────────────────────────────── */
/* y_i = x_i / RMS(x) * w_i,   RMS(x) = sqrt( mean(x^2) + eps )            */
static void rmsnorm(float * restrict out,
                    const float * restrict x,
                    const float * restrict w,
                    int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (ss * x[i]);
}

/* ─── Softmax (in-place) ─────────────────────────────────────────────────── */
static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Quantised matrix-vector multiply
 *     out[rows] = mat[rows][cols] · vec[cols]
 *     mat may be F32 or any GGML quantised type; vec is always F32.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void matvec(float * restrict out,
                   const TLTensor *mat,
                   const float   * restrict vec,
                   int rows, int cols)
{
    /* dequant_row() from quant.h fills a scratch float[cols] for one row   */
    float row_buf[8192];  /* max cols in TinyLlama is 5632 (n_ff)           */

    if (mat->type == 0 /* GGML_TYPE_F32 */) {
        const float *m = (const float *)mat->data;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < rows; r++) {
            float s = 0.0f;
            const float *row = m + (size_t)r * cols;
            for (int c = 0; c < cols; c++) s += row[c] * vec[c];
            out[r] = s;
        }
    } else {
        /* quantised path: dequantise one row at a time                     */
        for (int r = 0; r < rows; r++) {
            dequant_row(mat->data, mat->type, r, row_buf, cols);
            float s = 0.0f;
            for (int c = 0; c < cols; c++) s += row_buf[c] * vec[c];
            out[r] = s;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3  RoPE  (Rotary Positional Embedding)
 *
 *  For each pair of dimensions (2d, 2d+1) in a head:
 *    θ_d  = pos / (rope_freq_base ^ (2d / head_dim))
 *    q'[2d]   = q[2d]   * cos(θ_d) - q[2d+1] * sin(θ_d)
 *    q'[2d+1] = q[2d+1] * cos(θ_d) + q[2d]   * sin(θ_d)
 *
 *  Applied in-place to both Q [n_head * head_dim]
 *                         and K [n_kv_head * head_dim].
 * ═══════════════════════════════════════════════════════════════════════════ */
static void rope_apply(float *qk,   /* q or k buffer, already decoded F32    */
                       int    n_heads,
                       int    head_dim,
                       int    pos,
                       float  freq_base)
{
    for (int h = 0; h < n_heads; h++) {
        float *v = qk + h * head_dim;
        for (int d = 0; d < head_dim / 2; d++) {
            float theta = (float)pos /
                          powf(freq_base,
                               2.0f * (float)d / (float)head_dim);
            float c = cosf(theta), s = sinf(theta);
            float v0 = v[2*d], v1 = v[2*d+1];
            v[2*d]   = v0 * c - v1 * s;
            v[2*d+1] = v0 * s + v1 * c;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §4  Grouped-Query Attention
 *
 *  n_head    = 32   (query heads)
 *  n_kv_head = 4    (key/value heads)
 *  kv_groups = 8    (query heads per KV head)
 *
 *  For query head h:
 *    kv_h = h / kv_groups   ← shared KV head index
 *    score[t] = dot(Q_h, K_{kv_h, t}) / sqrt(head_dim)
 *    out_h    = Σ_t softmax(score[t]) * V_{kv_h, t}
 * ═══════════════════════════════════════════════════════════════════════════ */
static void gqa_attention(TLActivations *act,
                          TLKVCache     *kv,
                          const TLConfig *cfg,
                          int layer,
                          int pos)
{
    int   H  = cfg->n_head;
    int   Hk = cfg->n_kv_head;
    int   D  = cfg->head_dim;
    int   G  = cfg->kv_groups;          /* = H / Hk */
    float scale = 1.0f / sqrtf((float)D);

    /* KV-cache strides: [n_layer][n_kv_head][n_ctx][head_dim] */
    size_t layer_stride = (size_t)Hk * cfg->n_ctx * D;
    float *k_layer = kv->k + (size_t)layer * layer_stride;
    float *v_layer = kv->v + (size_t)layer * layer_stride;

    /* Write current K, V into cache */
    for (int kh = 0; kh < Hk; kh++) {
        float *k_slot = k_layer + (size_t)kh * cfg->n_ctx * D + (size_t)pos * D;
        float *v_slot = v_layer + (size_t)kh * cfg->n_ctx * D + (size_t)pos * D;
        memcpy(k_slot, act->k_cur + kh * D, (size_t)D * sizeof(float));
        memcpy(v_slot, act->v_cur + kh * D, (size_t)D * sizeof(float));
    }

    /* Multi-head attention with GQA */
    memset(act->attn_out, 0, (size_t)H * D * sizeof(float));

    for (int h = 0; h < H; h++) {
        int  kh     = h / G;            /* which KV head this Q head uses    */
        float *q_h  = act->q  + h  * D;
        float *out_h = act->attn_out + h * D;

        float *scores = act->attn_score + h * cfg->n_ctx;

        /* Compute attention scores for all past + current tokens */
        for (int t = 0; t <= pos; t++) {
            float *k_t = k_layer + (size_t)kh * cfg->n_ctx * D + (size_t)t * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
        }

        softmax(scores, pos + 1);

        /* Weighted sum over V */
        for (int t = 0; t <= pos; t++) {
            float *v_t = v_layer + (size_t)kh * cfg->n_ctx * D + (size_t)t * D;
            float w = scores[t];
            for (int d = 0; d < D; d++) out_h[d] += w * v_t[d];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5  SwiGLU FFN
 *
 *  gate = SiLU(W_gate · x)
 *  up   = W_up   · x
 *  out  = W_down · (gate ⊙ up)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void swiglu_ffn(TLActivations *act,
                       const TLWeights *w,
                       const TLConfig  *cfg,
                       int layer)
{
    int E = cfg->n_embd, F = cfg->n_ff;

    matvec(act->gate, &w->w_gate[layer], act->xb,  F, E);
    matvec(act->up,   &w->w_up  [layer], act->xb,  F, E);

    /* fused: hb[i] = SiLU(gate[i]) * up[i] */
    for (int i = 0; i < F; i++)
        act->hb[i] = silu(act->gate[i]) * act->up[i];

    matvec(act->xb2, &w->w_down[layer], act->hb, E, F);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §6  Embedding lookup   (always F32 in TinyLlama's GGUF)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void embed(float *out, const TLTensor *embd_table, int token_id, int n_embd) {
    if (embd_table->type == 0 /* F32 */) {
        const float *row = (const float *)embd_table->data +
                           (size_t)token_id * n_embd;
        memcpy(out, row, (size_t)n_embd * sizeof(float));
    } else {
        dequant_row(embd_table->data, embd_table->type, token_id, out, n_embd);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §7  RMSNorm from quantised weight  (weight tensors are always F32/F16)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void rmsnorm_from_tensor(float       *out,
                                const float *x,
                                const TLTensor *wt,
                                int n, float eps)
{
    /* RMSNorm weights are always stored F32 in these GGUF files */
    rmsnorm(out, x, (const float *)wt->data, n, eps);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §8  Full forward pass
 * ═══════════════════════════════════════════════════════════════════════════ */
float *tinyllama_forward(TLModel *model, int token, int pos)
{
    const TLConfig  *cfg = &model->cfg;
    const TLWeights *w   = &model->w;
    TLActivations   *act = &model->act;
    TLKVCache       *kv  = &model->kv;

    int E  = cfg->n_embd;
    int H  = cfg->n_head;
    int Hk = cfg->n_kv_head;
    int D  = cfg->head_dim;   /* = E / H */

    /* ── Step 1: Token embedding ───────────────────────────────────────── */
    embed(act->x, &w->tok_embd, token, E);

    /* ── Step 2: Transformer layers ────────────────────────────────────── */
    for (int l = 0; l < cfg->n_layer; l++) {

        /* 2a. Attention pre-norm */
        rmsnorm_from_tensor(act->xb, act->x, &w->rms_attn_w[l],
                            E, cfg->rms_eps);

        /* 2b. Q, K, V projections (quantised matmul → F32 scratch) */
        matvec(act->q,     &w->wq[l], act->xb,  H  * D, E);
        matvec(act->k_cur, &w->wk[l], act->xb,  Hk * D, E);
        matvec(act->v_cur, &w->wv[l], act->xb,  Hk * D, E);

        /* 2c. RoPE — applied per-head to both Q and K */
        rope_apply(act->q,     H,  D, pos, cfg->rope_freq_base);
        rope_apply(act->k_cur, Hk, D, pos, cfg->rope_freq_base);

        /* 2d. Grouped-Query Attention (writes into act->attn_out) */
        gqa_attention(act, kv, cfg, l, pos);

        /* 2e. Output projection  (attn_out is H*D = E floats) */
        matvec(act->xb2, &w->wo[l], act->attn_out, E, H * D);

        /* 2f. Residual connection */
        for (int i = 0; i < E; i++) act->x[i] += act->xb2[i];

        /* 2g. FFN pre-norm */
        rmsnorm_from_tensor(act->xb, act->x, &w->rms_ffn_w[l],
                            E, cfg->rms_eps);

        /* 2h. SwiGLU feed-forward  (result in act->xb2) */
        swiglu_ffn(act, w, cfg, l);

        /* 2i. Residual connection */
        for (int i = 0; i < E; i++) act->x[i] += act->xb2[i];
    }

    /* ── Step 3: Final RMSNorm ─────────────────────────────────────────── */
    rmsnorm_from_tensor(act->xb, act->x, &w->rms_final_w, E, cfg->rms_eps);

    /* ── Step 4: Language-model head → logits ──────────────────────────── */
    matvec(act->logits, &w->lm_head, act->xb, cfg->n_vocab, E);

    return act->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §9  Model loading from GgufCtx
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Helper: look up a tensor by name and fill in the TLTensor wrapper.        */
static int load_tensor(TLTensor *out, GgufCtx *ctx, const char *name)
{
    GgufTensorInfo info;
    if (!gguf_get_tensor(ctx, name, &info)) {
        fprintf(stderr, "[tinyllama] missing tensor: %s\n", name);
        return -1;
    }
    out->data = info.data;
    out->type = info.type;
    out->rows = (int)info.ne[1];
    out->cols = (int)info.ne[0];
    return 0;
}

/* Helper: read one int metadata value with a fallback default.              */
static int meta_int(GgufCtx *ctx, const char *key, int def) {
    int64_t v;
    if (gguf_get_meta_int(ctx, key, &v)) return (int)v;
    return def;
}

static float meta_float(GgufCtx *ctx, const char *key, float def) {
    double v;
    if (gguf_get_meta_float(ctx, key, &v)) return (float)v;
    return def;
}

int tinyllama_load(TLModel *model, GgufCtx *ctx, Arena *arena)
{
    TLConfig  *cfg = &model->cfg;
    TLWeights *w   = &model->w;

    /* ── Read hyper-parameters from GGUF metadata ──────────────────────── */
    cfg->n_embd    = meta_int(ctx, "llama.embedding_length",    2048);
    cfg->n_layer   = meta_int(ctx, "llama.block_count",          22);
    cfg->n_head    = meta_int(ctx, "llama.attention.head_count", 32);
    cfg->n_kv_head = meta_int(ctx, "llama.attention.head_count_kv", 4);
    cfg->n_ff      = meta_int(ctx, "llama.feed_forward_length",  5632);
    cfg->n_ctx     = meta_int(ctx, "llama.context_length",       2048);
    cfg->n_vocab   = meta_int(ctx, "llama.vocab_size",           32000);

    cfg->rms_eps        = meta_float(ctx, "llama.attention.layer_norm_rms_epsilon",
                                     1e-5f);
    cfg->rope_freq_base = meta_float(ctx, "llama.rope.freq_base", 10000.0f);

    cfg->bos_id  = meta_int(ctx, "tokenizer.ggml.bos_token_id", 1);
    cfg->eos_id  = meta_int(ctx, "tokenizer.ggml.eos_token_id", 2);

    /* Derived */
    cfg->head_dim  = cfg->n_embd / cfg->n_head;
    cfg->kv_groups = cfg->n_head / cfg->n_kv_head;

    LMC_ASSERT(cfg->n_layer <= TL_MAX_LAYERS,
               "n_layer exceeds TL_MAX_LAYERS — rebuild with larger value");

    /* ── Load tensors ───────────────────────────────────────────────────── */
    if (load_tensor(&w->tok_embd, ctx, "token_embd.weight") != 0) return -1;

    char name[128];
    for (int l = 0; l < cfg->n_layer; l++) {
#define LOAD_LAYER(field, fmt) \
        snprintf(name, sizeof(name), fmt, l); \
        if (load_tensor(&w->field[l], ctx, name) != 0) return -1;

        LOAD_LAYER(rms_attn_w, "blk.%d.attn_norm.weight")
        LOAD_LAYER(wq,         "blk.%d.attn_q.weight")
        LOAD_LAYER(wk,         "blk.%d.attn_k.weight")
        LOAD_LAYER(wv,         "blk.%d.attn_v.weight")
        LOAD_LAYER(wo,         "blk.%d.attn_output.weight")
        LOAD_LAYER(rms_ffn_w,  "blk.%d.ffn_norm.weight")
        LOAD_LAYER(w_gate,     "blk.%d.ffn_gate.weight")
        LOAD_LAYER(w_up,       "blk.%d.ffn_up.weight")
        LOAD_LAYER(w_down,     "blk.%d.ffn_down.weight")
#undef LOAD_LAYER
    }

    if (load_tensor(&w->rms_final_w, ctx, "output_norm.weight")  != 0) return -1;
    if (load_tensor(&w->lm_head,     ctx, "output.weight")       != 0) return -1;

    /* ── Allocate KV-cache ──────────────────────────────────────────────── */
    TLKVCache *kv = &model->kv;
    kv->n_ctx_max = cfg->n_ctx;
    size_t kv_size = (size_t)cfg->n_layer * cfg->n_kv_head *
                     cfg->n_ctx * cfg->head_dim;
    kv->k = (float *)arena_alloc(arena, kv_size * sizeof(float));
    kv->v = (float *)arena_alloc(arena, kv_size * sizeof(float));
    if (!kv->k || !kv->v) {
        fprintf(stderr, "[tinyllama] OOM allocating KV cache\n");
        return -1;
    }
    memset(kv->k, 0, kv_size * sizeof(float));
    memset(kv->v, 0, kv_size * sizeof(float));

    /* ── Allocate activation buffers ────────────────────────────────────── */
    TLActivations *act = &model->act;
    int E = cfg->n_embd, H = cfg->n_head, Hk = cfg->n_kv_head;
    int D = cfg->head_dim, F = cfg->n_ff, V = cfg->n_vocab;

#define ACT_ALLOC(field, n) \
    act->field = (float *)arena_alloc(arena, (size_t)(n) * sizeof(float)); \
    if (!act->field) { fprintf(stderr, "[tinyllama] OOM: " #field "\n"); return -1; }

    ACT_ALLOC(x,          E)
    ACT_ALLOC(xb,         E)
    ACT_ALLOC(xb2,        E)
    ACT_ALLOC(q,          H  * D)
    ACT_ALLOC(k_cur,      Hk * D)
    ACT_ALLOC(v_cur,      Hk * D)
    ACT_ALLOC(attn_score, H  * cfg->n_ctx)
    ACT_ALLOC(attn_out,   H  * D)
    ACT_ALLOC(gate,       F)
    ACT_ALLOC(up,         F)
    ACT_ALLOC(hb,         F)
    ACT_ALLOC(logits,     V)
#undef ACT_ALLOC

    lmc_log("[tinyllama] loaded  layers=%d  n_embd=%d  n_head=%d  n_kv_head=%d  "
            "n_ff=%d  head_dim=%d\n",
            cfg->n_layer, E, H, Hk, F, D);
    return 0;
}

void tinyllama_free(TLModel *model)
{
    /* KV-cache and activation buffers belong to the arena.
     * The arena itself is freed by the caller.
     * Nothing extra to do here unless we switch to malloc.           */
    (void)model;
}
