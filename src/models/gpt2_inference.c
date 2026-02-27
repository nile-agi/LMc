/*
 * src/models/gpt2_inference.c — GPT-2 Forward Pass
 *
 * Implements:
 *   - Multi-head causal self-attention with KV cache
 *   - Transformer block (pre-norm architecture)
 *   - Full forward pass: embed → N×block → ln_f → lm_head → logits
 *   - KV cache and activation buffer lifecycle
 *
 * Architecture (GPT-2 124M):
 *   - 12 layers, 12 heads, embed_dim=768, ffn_dim=3072
 *   - Pre-LayerNorm (norm before each sub-layer)
 *   - GELU activation in FFN
 *   - Causal (left-to-right) attention with additive mask via KV cache
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * KV CACHE LIFECYCLE
 * ============================================================ */
LmcError lmc_kv_cache_init(LmcContext *ctx) {
    const size_t cache_floats = (size_t)GPT2_N_LAYERS
                              * GPT2_SEQ_LEN
                              * GPT2_N_HEADS
                              * GPT2_HEAD_DIM;

    ctx->kv_cache.k_cache = (float*)calloc(cache_floats, sizeof(float));
    ctx->kv_cache.v_cache = (float*)calloc(cache_floats, sizeof(float));
    ctx->kv_cache.seq_len = 0;

    if (!ctx->kv_cache.k_cache || !ctx->kv_cache.v_cache) {
        LMC_ERROR("Failed to allocate KV cache (%.1f MB)",
                  cache_floats * 2 * sizeof(float) / (1024.0*1024.0));
        lmc_kv_cache_free(ctx);
        return LMC_ERR_OOM;
    }

    LMC_INFO("KV cache     : %.1f MB",
             cache_floats * 2 * sizeof(float) / (1024.0*1024.0));
    return LMC_OK;
}

void lmc_kv_cache_free(LmcContext *ctx) {
    if (!ctx) return;
    free(ctx->kv_cache.k_cache); ctx->kv_cache.k_cache = NULL;
    free(ctx->kv_cache.v_cache); ctx->kv_cache.v_cache = NULL;
    ctx->kv_cache.seq_len = 0;
}

/* ============================================================
 * ACTIVATION BUFFERS LIFECYCLE
 * ============================================================ */
LmcError lmc_activations_init(LmcContext *ctx) {
    LmcActivations *a = &ctx->act;
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;
    const int F = GPT2_FFN_DIM;
    const int H = GPT2_N_HEADS;
    const int S = GPT2_SEQ_LEN;

#define ACT_ALLOC(field, n) \
    do { \
        a->field = (float*)malloc((n) * sizeof(float)); \
        if (!a->field) { \
            LMC_ERROR("Failed to allocate activation buffer: " #field); \
            lmc_activations_free(ctx); \
            return LMC_ERR_OOM; \
        } \
    } while(0)

    ACT_ALLOC(x,           D);
    ACT_ALLOC(x_norm,      D);
    ACT_ALLOC(qkv,      3 * D);
    ACT_ALLOC(attn_out,    D);
    ACT_ALLOC(proj_out,    D);
    ACT_ALLOC(ffn_hidden,  F);
    ACT_ALLOC(ffn_out,     D);
    ACT_ALLOC(logits,      V);
    ACT_ALLOC(attn_scores, H * S);

#undef ACT_ALLOC

    /* Top-p sorted buffer: ProbIdx (float + int = 8 bytes) per vocab token */
    a->sorted_buf = malloc((size_t)V * (sizeof(float) + sizeof(int)));
    if (!a->sorted_buf) {
        LMC_ERROR("Failed to allocate sampling buffer");
        lmc_activations_free(ctx);
        return LMC_ERR_OOM;
    }

    return LMC_OK;
}

void lmc_activations_free(LmcContext *ctx) {
    if (!ctx) return;
    LmcActivations *a = &ctx->act;
    free(a->x);           a->x           = NULL;
    free(a->x_norm);      a->x_norm      = NULL;
    free(a->qkv);         a->qkv         = NULL;
    free(a->attn_out);    a->attn_out    = NULL;
    free(a->proj_out);    a->proj_out    = NULL;
    free(a->ffn_hidden);  a->ffn_hidden  = NULL;
    free(a->ffn_out);     a->ffn_out     = NULL;
    free(a->logits);      a->logits      = NULL;
    free(a->attn_scores); a->attn_scores = NULL;
    free(a->sorted_buf);  a->sorted_buf  = NULL;
}

/* ============================================================
 * MULTI-HEAD CAUSAL SELF-ATTENTION
 *
 * Uses the incremental KV cache: for position `pos`, the cache
 * already holds keys/values for positions 0..pos-1. We append
 * the new K,V and attend over all 0..pos.
 *
 *   QKV = x_norm · W_qkv + b_qkv    [3*D]
 *   For each head h:
 *     scores[t] = (Q_h · K_h_t) / sqrt(d_h)   for t = 0..pos
 *     softmax(scores)
 *     out_h = Σ_t scores[t] * V_h_t
 * ============================================================ */
static void attention_forward(
    float *out,             /* [D] output                          */
    const float *x_norm,    /* [D] layer-normed input              */
    const LmcLayerWeights *lw,
    float *k_cache,         /* [SEQ_LEN, H, Dh] layer K cache      */
    float *v_cache,         /* [SEQ_LEN, H, Dh] layer V cache      */
    int pos,                /* current position (0-indexed)        */
    float *qkv_buf,         /* [3*D] scratch                       */
    float *scores_buf)      /* [H * SEQ_LEN] scratch               */
{
    const int D  = GPT2_EMBED_DIM;
    const int H  = GPT2_N_HEADS;
    const int Dh = GPT2_HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)Dh);

    /* Project to Q, K, V */
    lmc_matmul_vec(qkv_buf, lw->qkv_weight, lw->qkv_bias, x_norm, 3*D, D);

    const float *q_vec = qkv_buf;
    const float *k_vec = qkv_buf + D;
    const float *v_vec = qkv_buf + 2*D;

    /* Write K, V into the cache at position `pos` */
    float *k_dest = k_cache + (size_t)pos * H * Dh;
    float *v_dest = v_cache + (size_t)pos * H * Dh;
    memcpy(k_dest, k_vec, (size_t)H * Dh * sizeof(float));
    memcpy(v_dest, v_vec, (size_t)H * Dh * sizeof(float));

    int ctx_len = pos + 1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        const float *q_h  = q_vec + h * Dh;
        float *scores = scores_buf + h * GPT2_SEQ_LEN;

        /* Attention scores: Q_h · K_h_t for all t in [0, ctx_len) */
        for (int t = 0; t < ctx_len; t++) {
            const float *k_t = k_cache + (size_t)t * H * Dh + h * Dh;
            float dot = 0.0f;
            int d = 0;
            for (; d <= Dh - 8; d += 8) {
                dot += q_h[d+0]*k_t[d+0] + q_h[d+1]*k_t[d+1]
                     + q_h[d+2]*k_t[d+2] + q_h[d+3]*k_t[d+3]
                     + q_h[d+4]*k_t[d+4] + q_h[d+5]*k_t[d+5]
                     + q_h[d+6]*k_t[d+6] + q_h[d+7]*k_t[d+7];
            }
            for (; d < Dh; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
        }

        lmc_softmax(scores, ctx_len);

        /* Weighted sum of V vectors */
        float *out_h = out + h * Dh;
        memset(out_h, 0, Dh * sizeof(float));
        for (int t = 0; t < ctx_len; t++) {
            const float *v_t = v_cache + (size_t)t * H * Dh + h * Dh;
            float s = scores[t];
            for (int d = 0; d < Dh; d++) out_h[d] += s * v_t[d];
        }
    }
}

/* ============================================================
 * TRANSFORMER BLOCK
 *
 * GPT-2 pre-norm layout:
 *   x = x + attn_proj(MHA(LN1(x)))
 *   x = x + ffn_proj(GELU(ffn_fc(LN2(x))))
 * ============================================================ */
static void transformer_block(
    float *x,                   /* [D] in/out hidden state    */
    const LmcLayerWeights *lw,
    float *k_cache,
    float *v_cache,
    int pos,
    /* scratch buffers (all heap-allocated, no stack VLAs) */
    float *s_norm,
    float *s_qkv,
    float *s_attn,
    float *s_scores,
    float *s_ffn,
    float *s_proj,
    float *s_ffnout)
{
    const int D = GPT2_EMBED_DIM;
    const int F = GPT2_FFN_DIM;

    /* --- Sub-layer 1: MHA --- */
    lmc_layer_norm(s_norm, x, lw->ln1_weight, lw->ln1_bias, D);
    attention_forward(s_attn, s_norm, lw,
                      k_cache, v_cache, pos,
                      s_qkv, s_scores);
    lmc_matmul_vec(s_proj, lw->attn_proj_weight, lw->attn_proj_bias,
                   s_attn, D, D);
    for (int i = 0; i < D; i++) x[i] += s_proj[i];

    /* --- Sub-layer 2: FFN --- */
    lmc_layer_norm(s_norm, x, lw->ln2_weight, lw->ln2_bias, D);
    lmc_matmul_vec(s_ffn, lw->ffn_fc_weight, lw->ffn_fc_bias,
                   s_norm, F, D);
    for (int i = 0; i < F; i++) s_ffn[i] = lmc_gelu(s_ffn[i]);
    lmc_matmul_vec(s_ffnout, lw->ffn_proj_weight, lw->ffn_proj_bias,
                   s_ffn, D, F);
    for (int i = 0; i < D; i++) x[i] += s_ffnout[i];
}

/* ============================================================
 * FULL GPT-2 FORWARD PASS
 *
 * Given one token and its position in the sequence, computes
 * the full forward pass and returns a pointer to the logit
 * buffer [VOCAB_SIZE]. The KV cache is updated in place.
 *
 * token_id : vocab index of input token
 * pos      : 0-indexed position in the sequence
 * returns  : pointer to logits[VOCAB_SIZE] (valid until next call)
 * ============================================================ */
float* lmc_gpt2_forward(LmcContext *ctx, int token_id, int pos) {
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;
    const int L = GPT2_N_LAYERS;

    const LmcModelWeights *mw = &ctx->weights;
    LmcActivations        *a  = &ctx->act;
    LmcKVCache            *kv = &ctx->kv_cache;

    /* --- Input embedding: token + position --- */
    float *x = a->x;
    const float *tok_emb = mw->wte + (size_t)token_id * D;
    const float *pos_emb = mw->wpe + (size_t)pos * D;
    for (int i = 0; i < D; i++) x[i] = tok_emb[i] + pos_emb[i];

    /* --- Transformer layers --- */
    const size_t layer_stride = (size_t)GPT2_SEQ_LEN * GPT2_N_HEADS * GPT2_HEAD_DIM;

    for (int l = 0; l < L; l++) {
        float *k_cache_l = kv->k_cache + l * layer_stride;
        float *v_cache_l = kv->v_cache + l * layer_stride;

        transformer_block(
            x, &mw->layers[l],
            k_cache_l, v_cache_l, pos,
            a->x_norm,
            a->qkv,
            a->attn_out,
            a->attn_scores,
            a->ffn_hidden,
            a->proj_out,
            a->ffn_out
        );
    }

    /* --- Final layer norm --- */
    lmc_layer_norm(a->x_norm, x, mw->ln_f_weight, mw->ln_f_bias, D);

    /* --- LM head: project to vocabulary --- */
    lmc_matmul_vec(a->logits, mw->lm_head, NULL, a->x_norm, V, D);

    return a->logits;
}
