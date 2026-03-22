/* ops_llama.c — LLaMA kernel implementations.
 *
 * Implements the functions declared in ops.h for the LLaMA path:
 *   rms_norm()                  — RMSNorm (no mean subtraction, no beta)
 *   rope_apply()                — RoPE for one head vector
 *   llama_attention_forward()   — GQA attention with RoPE and KV cache
 *   swiglu_ffn_forward()        — SwiGLU FFN: down(silu(gate) ⊙ up)
 *   llama_model_forward()       — full forward pass, called from model_forward()
 *
 * Compiled separately; linked via Makefile SRCS.
 * model_forward() in ops.c dispatches here when g_cfg.arch == ARCH_LLAMA.
 */
#include "ops.h"
#include "models.h"
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── RMSNorm ──────────────────────────────────────────────────────────────── *
 * out_i = weight_i * x_i / sqrt(mean(x²) + eps)                            */
void rms_norm(float *restrict out,
              const float *restrict x,
              const float *restrict weight,
              int dim, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = weight[i] * ss * x[i];
}

/* ── RoPE — single head vector ────────────────────────────────────────────── *
 * Rotates adjacent pairs of vec[0..head_dim-1] using precomputed cos/sin.  *
 * cos_row and sin_row each have head_dim/2 elements.                        *
 * Call once per Q head (H times) and once per K head (Hkv times).          */
void rope_apply(float *vec,
                const float *cos_row, const float *sin_row,
                int head_dim)
{
    for (int d = 0; d < head_dim / 2; d++) {
        float v0 = vec[2*d], v1 = vec[2*d + 1];
        vec[2*d]     = v0 * cos_row[d] - v1 * sin_row[d];
        vec[2*d + 1] = v0 * sin_row[d] + v1 * cos_row[d];
    }
}

/* ── LLaMA GQA attention ──────────────────────────────────────────────────── *
 * TinyLlama: H=32 Q heads, Hkv=4 KV heads, kv_group=8.                     *
 * Caller supplies scratch buffers q_buf[H*Dh], k_buf[Hkv*Dh], v_buf[Hkv*Dh]*
 * and scores_buf[H*S].                                                       */
void llama_attention_forward(
    float *restrict out,
    const float *restrict x_norm,
    const LayerWeights *lw,
    float *restrict k_cache, float *restrict v_cache,
    int pos,
    float *restrict q_buf,
    float *restrict k_buf,
    float *restrict v_buf,
    float *restrict scores_buf)
{
    const int H = CFG_H, Hkv = CFG_Hkv, Dh = CFG_Dh;
    const int D = CFG_D, G = CFG_KVG, S = CFG_S;
    const float scale = 1.0f / sqrtf((float)Dh);

    /* Project Q, K, V (no bias in LLaMA) */
    matmul_vec(q_buf, lw->q_weight, NULL, x_norm, H   * Dh, D);
    matmul_vec(k_buf, lw->k_weight, NULL, x_norm, Hkv * Dh, D);
    matmul_vec(v_buf, lw->v_weight, NULL, x_norm, Hkv * Dh, D);

    /* Apply RoPE — one call per head, sharing the same position row */
    const float *cos_row = g_weights.rope_cos + (size_t)pos * (Dh / 2);
    const float *sin_row = g_weights.rope_sin + (size_t)pos * (Dh / 2);
    for (int h = 0; h < H;   h++) rope_apply(q_buf + h * Dh, cos_row, sin_row, Dh);
    for (int h = 0; h < Hkv; h++) rope_apply(k_buf + h * Dh, cos_row, sin_row, Dh);

    /* Store current K, V in KV cache (head-major: [Hkv][S][Dh]) */
    for (int kh = 0; kh < Hkv; kh++) {
        float *ks = k_cache + ((size_t)kh * S + pos) * Dh;
        float *vs = v_cache + ((size_t)kh * S + pos) * Dh;
        memcpy(ks, k_buf + kh * Dh, (size_t)Dh * sizeof(float));
        memcpy(vs, v_buf + kh * Dh, (size_t)Dh * sizeof(float));
    }

    const int ctx = pos + 1;
    memset(out, 0, (size_t)H * Dh * sizeof(float));

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        const int    kh     = h / G;
        const float *q_h    = q_buf + h * Dh;
        float       *scores = scores_buf + h * S;
        const float *k_base = k_cache + (size_t)kh * S * Dh;
        const float *v_base = v_cache + (size_t)kh * S * Dh;
        float       *out_h  = out + h * Dh;

        /* QK dot products — 16-wide unroll (Dh=64 → 4 iters, no tail) */
        for (int t = 0; t < ctx; t++) {
            const float *k_t = k_base + (size_t)t * Dh;
            float dot = 0.0f;
            int d = 0;
            for (; d <= Dh - 16; d += 16)
                dot += q_h[d+ 0]*k_t[d+ 0] + q_h[d+ 1]*k_t[d+ 1]
                     + q_h[d+ 2]*k_t[d+ 2] + q_h[d+ 3]*k_t[d+ 3]
                     + q_h[d+ 4]*k_t[d+ 4] + q_h[d+ 5]*k_t[d+ 5]
                     + q_h[d+ 6]*k_t[d+ 6] + q_h[d+ 7]*k_t[d+ 7]
                     + q_h[d+ 8]*k_t[d+ 8] + q_h[d+ 9]*k_t[d+ 9]
                     + q_h[d+10]*k_t[d+10] + q_h[d+11]*k_t[d+11]
                     + q_h[d+12]*k_t[d+12] + q_h[d+13]*k_t[d+13]
                     + q_h[d+14]*k_t[d+14] + q_h[d+15]*k_t[d+15];
            for (; d < Dh; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
        }
        softmax(scores, ctx);

        /* V accumulation — 8-wide unroll */
        for (int t = 0; t < ctx; t++) {
            const float *v_t = v_base + (size_t)t * Dh;
            float s = scores[t];
            int d = 0;
            for (; d <= Dh - 8; d += 8) {
                out_h[d+0] += s*v_t[d+0]; out_h[d+1] += s*v_t[d+1];
                out_h[d+2] += s*v_t[d+2]; out_h[d+3] += s*v_t[d+3];
                out_h[d+4] += s*v_t[d+4]; out_h[d+5] += s*v_t[d+5];
                out_h[d+6] += s*v_t[d+6]; out_h[d+7] += s*v_t[d+7];
            }
            for (; d < Dh; d++) out_h[d] += s * v_t[d];
        }
    }
}

/* ── SwiGLU FFN ───────────────────────────────────────────────────────────── *
 * out[D] = W_down · (silu(W_gate · x) ⊙ W_up · x)                         *
 * gate_buf and up_buf are caller-supplied scratch of size [F].              */
void swiglu_ffn_forward(
    float *restrict out,
    const float *restrict x,
    const LayerWeights *lw,
    float *restrict gate_buf,
    float *restrict up_buf)
{
    const int D = CFG_D, F = CFG_F;
    matmul_vec(gate_buf, lw->gate_weight, NULL, x, F, D);
    matmul_vec(up_buf,   lw->up_weight,   NULL, x, F, D);
    for (int i = 0; i < F; i++)
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
    matmul_vec(out, lw->down_weight, NULL, gate_buf, D, F);
}

/* ── llama_model_forward ──────────────────────────────────────────────────── *
 * Called from model_forward() in ops.c when g_cfg.arch == ARCH_LLAMA.      */
float *llama_model_forward(int token_id, int pos)
{
    const int D = CFG_D, V = CFG_V;
    float *x = g_act.x;

    /* Token embedding only — RoPE handles position, no wpe table */
    memcpy(x, g_weights.wte + (size_t)token_id * D, (size_t)D * sizeof(float));

    for (int l = 0; l < CFG_L; l++) {
        const LayerWeights *lw  = &g_weights.layers[l];
        size_t off = (size_t)l * CFG_Hkv * CFG_S * CFG_Dh;

        /* ── Attention sub-layer ────────────────────────────────────────── */
        rms_norm(g_act.x_norm, x, lw->rms_attn_weight, D, CFG_EPS);
        llama_attention_forward(
            g_act.attn_out, g_act.x_norm, lw,
            g_kv_cache.k_cache + off,
            g_kv_cache.v_cache + off,
            pos,
            g_act.q, g_act.k_cur, g_act.v_cur,
            g_act.attn_scores);
        matmul_vec(g_act.proj_out, lw->attn_proj_weight, NULL,
                   g_act.attn_out, D, D);
        for (int i = 0; i < D; i++) x[i] += g_act.proj_out[i];

        /* ── FFN sub-layer ──────────────────────────────────────────────── */
        rms_norm(g_act.x_norm, x, lw->rms_ffn_weight, D, CFG_EPS);
        swiglu_ffn_forward(g_act.ffn_out, g_act.x_norm, lw,
                            g_act.ffn_hidden, g_act.ffn_up);
        for (int i = 0; i < D; i++) x[i] += g_act.ffn_out[i];
    }

    /* Final RMSNorm + LM head */
    rms_norm(g_act.x_norm, x, g_weights.rms_f_weight, D, CFG_EPS);
    matmul_vec(g_act.logits, g_weights.lm_head, NULL, g_act.x_norm, V, D);
    return g_act.logits;
}