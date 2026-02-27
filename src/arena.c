/*
 * src/arena.c — LMc Memory Arena
 *
 * A simple bump-pointer allocator for model weights.
 * Weights are loaded once and never freed individually,
 * so a single large malloc + pointer bumping is ideal:
 *   - Cache-friendly (contiguous memory)
 *   - Zero fragmentation
 *   - O(1) allocation
 *   - Single free() to release all weights
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

LmcError lmc_arena_init(LmcArena *a, size_t n_floats) {
    if (!a || n_floats == 0) return LMC_ERR_INVALID_ARG;

    a->data = (float*)malloc(n_floats * sizeof(float));
    if (!a->data) {
        LMC_ERROR("Arena OOM: failed to allocate %.2f MB",
                  n_floats * sizeof(float) / (1024.0 * 1024.0));
        return LMC_ERR_OOM;
    }

    a->capacity = n_floats;
    a->used     = 0;
    return LMC_OK;
}

void lmc_arena_free(LmcArena *a) {
    if (!a) return;
    free(a->data);
    a->data     = NULL;
    a->capacity = 0;
    a->used     = 0;
}

float* lmc_arena_alloc(LmcArena *a, size_t n_floats) {
    if (!a || !a->data) {
        LMC_FATAL("Arena not initialized");
    }
    if (a->used + n_floats > a->capacity) {
        LMC_FATAL("Arena OOM: need %zu floats, only %zu remaining (capacity %zu)",
                  n_floats, a->capacity - a->used, a->capacity);
    }
    float *ptr = a->data + a->used;
    a->used += n_floats;
    return ptr;
}
