/* utils.c — Arena allocator implementation */
#include "utils.h"

Arena g_arena = {NULL, 0, 0};

void arena_init(size_t n_floats) {
    g_arena.capacity = n_floats;
    g_arena.used     = 0;
    g_arena.data     = (float*)malloc(n_floats * sizeof(float));
    if (!g_arena.data)
        LMC_FATAL("Cannot allocate %.1f MB for weight arena",
                  n_floats * 4.0 / (1024.0 * 1024.0));
}

float *arena_alloc(size_t n_floats) {
    if (g_arena.used + n_floats > g_arena.capacity)
        LMC_FATAL("Arena OOM: need %zu floats, only %zu remain",
                  n_floats, g_arena.capacity - g_arena.used);
    float *ptr = g_arena.data + g_arena.used;
    g_arena.used += n_floats;
    return ptr;
}

void arena_free(void) {
    free(g_arena.data);
    g_arena.data     = NULL;
    g_arena.capacity = 0;
    g_arena.used     = 0;
}
