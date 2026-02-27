/*
 * src/models/gpt2_weights.c — GPT-2 Weight Layout
 *
 * Defines how GPT-2 124M weights are laid out in the arena,
 * and assigns pointers from the arena into the weight struct.
 *
 * This is the single source of truth for the weight memory map.
 * Both the .bin loader and the GGUF loader use these pointers.
 *
 * Adding a new model:
 *   1. Create src/models/<model>_weights.c with its own
 *      param_count() and assign_weight_ptrs() functions.
 *   2. Add a model-type enum to lmc.h.
 *   3. Dispatch in lmc_load_model() based on architecture metadata.
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * PARAMETER COUNT
 *
 * GPT-2 124M total parameters (float32):
 *   wte:          50257 * 768    =  38,597,376
 *   wpe:           1024 * 768    =     786,432
 *   Per layer (12):
 *     ln1:          2 * 768      =       1,536
 *     qkv:  3*768*768 + 3*768   =   1,771,776
 *     attn: 768*768 + 768        =     590,592
 *     ln2:          2 * 768      =       1,536
 *     ffn_fc: 3072*768 + 3072   =   2,362,368
 *     ffn_pr: 768*3072 + 768    =   2,360,064
 *   ln_f:          2 * 768       =       1,536
 *   Total: ~124 M
 * ============================================================ */
size_t lmc_gpt2_param_count(void) {
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;
    const int S = GPT2_SEQ_LEN;
    const int L = GPT2_N_LAYERS;
    const int F = GPT2_FFN_DIM;

    size_t n = 0;

    /* Embeddings */
    n += (size_t)V * D;  /* wte */
    n += (size_t)S * D;  /* wpe */

    /* Transformer layers */
    for (int l = 0; l < L; l++) {
        n += 2 * D;                    /* ln1: weight + bias */
        n += (size_t)3*D*D + 3*D;     /* qkv: weight + bias */
        n += (size_t)D*D + D;          /* attn_proj: weight + bias */
        n += 2 * D;                    /* ln2: weight + bias */
        n += (size_t)F*D + F;          /* ffn_fc: weight + bias */
        n += (size_t)D*F + D;          /* ffn_proj: weight + bias */
    }

    /* Final layer norm */
    n += 2 * D;

    return n;
}

/* ============================================================
 * ASSIGN WEIGHT POINTERS FROM ARENA
 *
 * Order MUST match the binary layout in gpt2_124m.bin.
 * GGUF loader ignores this order (it maps by tensor name).
 * ============================================================ */
LmcError lmc_gpt2_assign_weight_ptrs(LmcContext *ctx) {
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;
    const int S = GPT2_SEQ_LEN;
    const int L = GPT2_N_LAYERS;
    const int F = GPT2_FFN_DIM;

    LmcArena        *a  = &ctx->arena;
    LmcModelWeights *mw = &ctx->weights;

    mw->wte = lmc_arena_alloc(a, (size_t)V * D);
    mw->wpe = lmc_arena_alloc(a, (size_t)S * D);

    for (int l = 0; l < L; l++) {
        LmcLayerWeights *lw = &mw->layers[l];

        lw->ln1_weight       = lmc_arena_alloc(a, D);
        lw->ln1_bias         = lmc_arena_alloc(a, D);
        lw->qkv_weight       = lmc_arena_alloc(a, (size_t)3*D*D);
        lw->qkv_bias         = lmc_arena_alloc(a, 3*D);
        lw->attn_proj_weight = lmc_arena_alloc(a, (size_t)D*D);
        lw->attn_proj_bias   = lmc_arena_alloc(a, D);
        lw->ln2_weight       = lmc_arena_alloc(a, D);
        lw->ln2_bias         = lmc_arena_alloc(a, D);
        lw->ffn_fc_weight    = lmc_arena_alloc(a, (size_t)F*D);
        lw->ffn_fc_bias      = lmc_arena_alloc(a, F);
        lw->ffn_proj_weight  = lmc_arena_alloc(a, (size_t)D*F);
        lw->ffn_proj_bias    = lmc_arena_alloc(a, D);
    }

    mw->ln_f_weight = lmc_arena_alloc(a, D);
    mw->ln_f_bias   = lmc_arena_alloc(a, D);

    /* lm_head defaults to tied weights (== wte).
     * GGUF loader may override this with output.weight if present. */
    mw->lm_head = mw->wte;

    return LMC_OK;
}
