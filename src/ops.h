/* ops.h — Neural network operators: activations, normalisation, matmul,
 *          attention, and full transformer forward pass.
 *
 * All functions are pure (no global writes except through explicit pointer
 * arguments) except for model_forward(), which writes to g_act and returns
 * a pointer into it.
 *
 * Architecture dispatch:
 *   transformer_block_forward() and model_forward() branch on CFG_ARCH.
 *   GPT-2 path:  layer_norm → attention_forward   → gelu FFN
 *   LLaMA path:  rms_norm   → llama_attention_forward → swiglu_ffn_forward
 */
#ifndef LMC_OPS_H
#define LMC_OPS_H

#include "models.h"

/* ── Scalar activations ───────────────────────────────────────────────────── */

/* GELU approximation: x·σ(1.702x).  Max error < 0.001 vs exact tanh form.
 * ~2× faster than tanhf(); used by GPT-2 FFN.                               */
float gelu(float x);

/* Sigmoid-linear unit: x·σ(x).  Used by LLaMA SwiGLU gate.                */
float silu(float x);

/* ── Vector operations ────────────────────────────────────────────────────── */

/* Numerically stable softmax in-place over x[0..n-1].
 * Falls back to uniform distribution if max(x) - min(x) < 1e-9.            */
void softmax(float *x, int n);

/* GPT-2-style LayerNorm: out = (x − μ) / √(σ² + ε) * weight + bias.
 * All pointers must be non-aliasing (restrict).  ε = 1e-5.                 */
void layer_norm(float *restrict out,
                const float *restrict x,
                const float *restrict weight,
                const float *restrict bias,
                int dim);

/* LLaMA-style RMSNorm: out = x / rms(x) * weight.  No mean subtraction.
 * ε taken from CFG_EPS.  All pointers must be non-aliasing.                */
void rms_norm(float *restrict out,
              const float *restrict x,
              const float *restrict weight,
              int dim, float eps);

/* Matrix-vector multiply: out[M] = weight[M × K] · in[K] + bias[M].
 * bias may be NULL (treated as zero vector).
 * OpenMP-parallelised across rows when M ≥ 256.
 * 16-wide manual unroll + __builtin_prefetch for cache warmup.              */
void matmul_vec(float *restrict out,
                const float *restrict weight,
                const float *restrict bias,
                const float *restrict in,
                int M, int K);

/* ── Rotary position embeddings (LLaMA) ──────────────────────────────────── *
 * Applies RoPE in-place to a [n_heads × head_dim] buffer.                   *
 * cos_row and sin_row point to g_weights.rope_cos[pos] and rope_sin[pos],   *
 * each of length rope_dim/2.                                                 *
 *                                                                             *
 * For GQA the caller passes n_heads = n_kv_heads when rotating K.           */
/* Apply RoPE in-place to a single head vector of length head_dim.
 * cos_row and sin_row each have head_dim/2 elements (one row of the cache).
 * Call once per head for Q (H times) and once per head for K (Hkv times).  */
void rope_apply(float *vec,
                const float *cos_row, const float *sin_row,
                int head_dim);

/* ── Attention ────────────────────────────────────────────────────────────── */

/* GPT-2 causal self-attention.
 *   - Uses combined qkv_weight / qkv_bias from lw.
 *   - Writes K/V for position pos into k_cache / v_cache.
 *   - Reads back all positions [0..pos] from the cache.
 *   - out[D] receives the concatenated, projected head outputs.             */
void attention_forward(
    float *restrict out,
    const float *restrict x_norm,
    const LayerWeights *lw,
    float *restrict k_cache, float *restrict v_cache,
    int pos,
    float *restrict qkv_buf,
    float *restrict scores_buf);

/* LLaMA causal self-attention with GQA and RoPE.
 *   - Uses separate q_weight, k_weight, v_weight from lw.
 *   - Applies RoPE to Q and K via precomputed g_weights.rope_{cos,sin}.
 *   - GQA broadcast: kv_head = query_head % n_kv_heads.
 *   - out[D] receives the projected head outputs.
 *   Scratch buffers (q_buf, k_buf, v_buf, scores_buf) are caller-supplied
 *   to keep the function free of hidden allocation.                         */
void llama_attention_forward(
    float *restrict out,
    const float *restrict x_norm,
    const LayerWeights *lw,
    float *restrict k_cache, float *restrict v_cache,
    int pos,
    float *restrict q_buf,
    float *restrict k_buf,
    float *restrict v_buf,
    float *restrict scores_buf);

/* ── Feed-forward networks ───────────────────────────────────────────────── */

/* LLaMA SwiGLU FFN: out[D] = down(silu(gate(x)) ⊙ up(x)).
 * gate_buf and up_buf are caller-supplied scratch of size [F].              */
void swiglu_ffn_forward(
    float *restrict out,
    const float *restrict x,
    const LayerWeights *lw,
    float *restrict gate_buf,
    float *restrict up_buf);

/* ── Full transformer ─────────────────────────────────────────────────────── */

/* Single transformer block.  Dispatches to GPT-2 or LLaMA kernels based on
 * CFG_ARCH.  x[D] is the residual stream; modified in-place.               */
void transformer_block_forward(
    float *x,
    const LayerWeights *lw,
    float *k_cache, float *v_cache,
    int pos,
    float *scratch_norm,
    float *scratch_qkv,
    float *scratch_attn,
    float *scratch_scores,
    float *scratch_ffn,
    float *scratch_ffn2,
    float *scratch_ffnout);

/* Run a full forward pass for one token at position pos.
 * Reads from g_cfg, g_weights, g_kv_cache; writes to g_act.
 * Returns g_act.logits — a pointer into the activation buffer.             */
float *model_forward(int token_id, int pos);

#endif /* LMC_OPS_H */
