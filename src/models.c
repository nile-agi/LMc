/* models.c — Global model state, weight pointer assignment, cache/activation init. */
#include "models.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Global state definitions ─────────────────────────────────────────────── */
ModelConfig  g_cfg      = {0};
ModelWeights g_weights  = {0};
KVCache      g_kv_cache = {NULL, NULL, 0};
Activations  g_act      = {0};

/* ── Parameter count (float32 equivalents) ────────────────────────────────── */
size_t gpt2_total_params(void) {
    const int D = CFG_D, V = CFG_V, S = CFG_S, L = CFG_L, F = CFG_F;
    size_t n = (size_t)V * D + (size_t)S * D;   /* wte + wpe */
    for (int l = 0; l < L; l++) {
        n += 2*D;                    /* ln1  weight + bias  */
        n += (size_t)3*D*D + 3*D;   /* qkv  weight + bias  */
        n += (size_t)D*D + D;        /* attn_proj            */
        n += 2*D;                    /* ln2  weight + bias  */
        n += (size_t)F*D + F;        /* ffn_fc               */
        n += (size_t)D*F + D;        /* ffn_proj             */
    }
    n += 2*D;   /* ln_f weight + bias */
    return n;
}

size_t llama_total_params(void) {
    const int D = CFG_D, V = CFG_V, L = CFG_L, F = CFG_F;
    const int Hkv = (CFG_Hkv > 0) ? CFG_Hkv : CFG_H, Dh = CFG_Dh;
    size_t n = (size_t)V * D;
    for (int l = 0; l < L; l++) {
        n += D;                           /* rms_attn_weight               */
        n += (size_t)D   * D;             /* q_weight                      */
        n += (size_t)Hkv * Dh * D;        /* k_weight                      */
        n += (size_t)Hkv * Dh * D;        /* v_weight                      */
        n += (size_t)D   * D;             /* attn_proj_weight               */
        n += D;                           /* rms_ffn_weight                 */
        n += (size_t)F   * D;             /* gate_weight                    */
        n += (size_t)F   * D;             /* up_weight                      */
        n += (size_t)D   * F;             /* down_weight                    */
    }
    n += D;                               /* rms_f_weight                   */
    n += (size_t)V * D;                   /* lm_head (not weight-tied)      */
    n += (size_t)2 * CFG_S * (Dh / 2);   /* rope_cos + rope_sin tables     */
    return n;
}

/* ── Assign arena slices to weight pointers — architecture-aware ──────────── */
void assign_weight_ptrs(void) {
    const int D = CFG_D, V = CFG_V, L = CFG_L, F = CFG_F;
    if (g_cfg.arch == ARCH_LLAMA) {
        const int Hkv = (CFG_Hkv > 0) ? CFG_Hkv : CFG_H, Dh = CFG_Dh;
        g_weights.wte  = arena_alloc((size_t)V * D);
        g_weights.wpe  = NULL;
        g_weights.layers = (LayerWeights *)calloc((size_t)L, sizeof(LayerWeights));
        if (!g_weights.layers) LMC_FATAL("Cannot allocate LayerWeights (%d)", L);
        for (int l = 0; l < L; l++) {
            LayerWeights *lw = &g_weights.layers[l];
            lw->rms_attn_weight  = arena_alloc(D);
            lw->q_weight         = arena_alloc((size_t)D   * D);
            lw->k_weight         = arena_alloc((size_t)Hkv * Dh * D);
            lw->v_weight         = arena_alloc((size_t)Hkv * Dh * D);
            lw->attn_proj_weight = arena_alloc((size_t)D   * D);
            lw->rms_ffn_weight   = arena_alloc(D);
            lw->gate_weight      = arena_alloc((size_t)F   * D);
            lw->up_weight        = arena_alloc((size_t)F   * D);
            lw->down_weight      = arena_alloc((size_t)D   * F);
        }
        g_weights.rms_f_weight = arena_alloc(D);
        g_weights.lm_head      = arena_alloc((size_t)V * D);
        g_weights.rope_cos = g_weights.rope_sin = NULL;
    } else {
        const int S = CFG_S;
        g_weights.wte = arena_alloc((size_t)V * D);
        g_weights.wpe = arena_alloc((size_t)S * D);
        g_weights.layers = (LayerWeights *)calloc((size_t)L, sizeof(LayerWeights));
        if (!g_weights.layers) LMC_FATAL("Cannot allocate LayerWeights (%d)", L);
        for (int l = 0; l < L; l++) {
            LayerWeights *lw = &g_weights.layers[l];
            lw->ln1_weight       = arena_alloc(D);
            lw->ln1_bias         = arena_alloc(D);
            lw->qkv_weight       = arena_alloc((size_t)3 * D * D);
            lw->qkv_bias         = arena_alloc(3 * D);
            lw->attn_proj_weight = arena_alloc((size_t)D * D);
            lw->attn_proj_bias   = arena_alloc(D);
            lw->ln2_weight       = arena_alloc(D);
            lw->ln2_bias         = arena_alloc(D);
            lw->ffn_fc_weight    = arena_alloc((size_t)F * D);
            lw->ffn_fc_bias      = arena_alloc(F);
            lw->ffn_proj_weight  = arena_alloc((size_t)D * F);
            lw->ffn_proj_bias    = arena_alloc(D);
        }
        g_weights.ln_f_weight = arena_alloc(D);
        g_weights.ln_f_bias   = arena_alloc(D);
    }
}

void init_rope_cache(void) {
    if (g_cfg.arch != ARCH_LLAMA) return;
    const int S = CFG_S, Dh = CFG_Dh;
    const float theta = (g_cfg.rope_theta > 0.0f) ? g_cfg.rope_theta : 10000.0f;
    g_weights.rope_cos = arena_alloc((size_t)S * (Dh / 2));
    g_weights.rope_sin = arena_alloc((size_t)S * (Dh / 2));
    for (int pos = 0; pos < S; pos++) {
        float *c = g_weights.rope_cos + pos * (Dh / 2);
        float *s = g_weights.rope_sin + pos * (Dh / 2);
        for (int d = 0; d < Dh / 2; d++) {
            float angle = (float)pos / powf(theta, 2.0f*(float)d/(float)Dh);
            c[d] = cosf(angle); s[d] = sinf(angle);
        }
    }
    LMC_INFO("RoPE cache: %.1f MB  (S=%d Dh/2=%d theta=%.0f)",
             (double)S*Dh*2*4.0/(1024*1024), S, Dh/2, (double)theta);
}

/* ── KV cache — GQA-aware ────────────────────────────────────────────────── *
 * Uses n_kv_heads (not n_heads) for LLaMA GQA sizing.                      *
 * TinyLlama: 4 KV heads vs 32 Q heads → 8× smaller KV cache.               */
void init_kv_cache(void) {
    const int Hkv = (g_cfg.n_kv_heads > 0) ? g_cfg.n_kv_heads : g_cfg.n_heads;
    const size_t sz = (size_t)CFG_L * Hkv * CFG_S * CFG_Dh;
    g_kv_cache.k_cache = (float *)calloc(sz, sizeof(float));
    g_kv_cache.v_cache = (float *)calloc(sz, sizeof(float));
    g_kv_cache.seq_len = 0;
    if (!g_kv_cache.k_cache || !g_kv_cache.v_cache)
        LMC_FATAL("Cannot allocate KV cache (%.1f MB)",
                  sz * 2 * 4.0 / (1024.0 * 1024.0));
    LMC_INFO("KV cache: %.1f MB  (L=%d Hkv=%d S=%d Dh=%d, head-major)",
             sz * 2 * 4.0 / (1024.0 * 1024.0), CFG_L, Hkv, CFG_S, CFG_Dh);
}

/* ── Activation scratch buffers ───────────────────────────────────────────── */
void init_activations(void) {
    const int D = CFG_D, V = CFG_V, F = CFG_F, H = CFG_H, S = CFG_S;

#define ACT_ALLOC(field, n) \
    do { g_act.field = (float *)malloc((size_t)(n) * sizeof(float)); \
         if (!g_act.field) LMC_FATAL("OOM activation: " #field); } while(0)

    ACT_ALLOC(x,           D);
    ACT_ALLOC(x_norm,      D);
    ACT_ALLOC(qkv,      3 * D);
    ACT_ALLOC(attn_out,    D);
    ACT_ALLOC(proj_out,    D);
    ACT_ALLOC(ffn_hidden,  F);
    ACT_ALLOC(ffn_out,     D);
    ACT_ALLOC(logits,      V);
    ACT_ALLOC(attn_scores, H * S);
    /* LLaMA / GQA buffers (free(NULL) is safe — harmless for GPT-2) */
    { const int Hkv=(g_cfg.n_kv_heads>0)?g_cfg.n_kv_heads:H;
      const int Dh =(g_cfg.head_dim  >0)?g_cfg.head_dim  :D/H;
      ACT_ALLOC(q,     H   * Dh);
      ACT_ALLOC(k_cur, Hkv * Dh);
      ACT_ALLOC(v_cur, Hkv * Dh); }
    ACT_ALLOC(ffn_up, F);
#undef ACT_ALLOC

    /* ProbIdx sorting buffer for top-p sampling — size = vocab * (float+int) */
    g_act.sorted_buf = malloc((size_t)V * (sizeof(float) + sizeof(int)));
    if (!g_act.sorted_buf) LMC_FATAL("OOM sorted_buf");
}

void free_activations(void) {
    free(g_act.x);        free(g_act.x_norm);    free(g_act.qkv);
    free(g_act.attn_out); free(g_act.proj_out);  free(g_act.ffn_hidden);
    free(g_act.ffn_out);  free(g_act.logits);    free(g_act.attn_scores);
    free(g_act.q);   free(g_act.k_cur); free(g_act.v_cur); free(g_act.ffn_up);
    free(g_act.sorted_buf);
    memset(&g_act, 0, sizeof(g_act));
}
