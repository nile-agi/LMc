/*
 * src/sampling.c — LMc Token Sampling
 *
 * Implements sampling strategies for next-token selection:
 *   - Greedy (temperature=0): argmax
 *   - Top-p (nucleus) sampling: sample from smallest vocabulary
 *     subset whose cumulative probability ≥ top_p
 *
 * Adding new sampling strategies:
 *   1. Add a new function lmc_sample_<strategy>(...)
 *   2. Expose it in lmc_internal.h
 *   3. Wire it into lmc_generate() via LmcGenConfig
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * XORSHIFT64 PRNG
 * Fast, non-cryptographic, period 2^64-1.
 * Sufficient for token sampling.
 * ============================================================ */
static uint64_t g_rng_state = 0x853c49e6748fea9bULL;

void lmc_rng_seed(uint64_t seed) {
    g_rng_state = seed ^ 0xdeadbeefcafe1234ULL;
    if (g_rng_state == 0) g_rng_state = 1; /* avoid degenerate zero state */
    /* Warm up the generator */
    for (int i = 0; i < 8; i++) {
        g_rng_state ^= g_rng_state << 13;
        g_rng_state ^= g_rng_state >> 7;
        g_rng_state ^= g_rng_state << 17;
    }
}

static uint64_t rng_u64(void) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 7;
    g_rng_state ^= g_rng_state << 17;
    return g_rng_state;
}

/* Uniform float in [0, 1) with 53-bit precision */
static float rng_float(void) {
    return (float)(rng_u64() >> 11) / (float)(1ULL << 53);
}

/* ============================================================
 * COMPARISON HELPER FOR QSORT
 * Sort in descending probability order.
 * ============================================================ */
typedef struct { float prob; int idx; } ProbIdx;

static int cmp_prob_desc(const void *a, const void *b) {
    const ProbIdx *pa = (const ProbIdx*)a;
    const ProbIdx *pb = (const ProbIdx*)b;
    if (pb->prob > pa->prob) return  1;
    if (pb->prob < pa->prob) return -1;
    return 0;
}

/* ============================================================
 * TOP-P (NUCLEUS) SAMPLING
 *
 * Algorithm:
 *   1. Divide logits by temperature (higher temp = more uniform).
 *   2. Apply softmax to get a probability distribution.
 *   3. Sort tokens by probability (descending).
 *   4. Find the smallest "nucleus" of tokens whose cumulative
 *      probability mass ≥ top_p.
 *   5. Sample uniformly from within that nucleus.
 *
 * Special cases:
 *   - temperature ≤ 0:  greedy (argmax)
 *   - top_p ≥ 1.0:      sample from the full distribution
 *
 * Parameters:
 *   logits      [VOCAB_SIZE]  raw logit scores (modified in place)
 *   temperature  sampling temperature
 *   top_p        nucleus threshold
 *   sorted_buf   caller-allocated ProbIdx[VOCAB_SIZE] scratch buffer
 *
 * Returns: the selected token id
 * ============================================================ */
int lmc_sample_top_p(float *logits, float temperature, float top_p,
                     void *sorted_buf) {
    const int V = GPT2_VOCAB_SIZE;

    /* --- Greedy path (temperature ≈ 0) --- */
    if (temperature < 1e-6f) {
        int best = 0;
        for (int i = 1; i < V; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    /* --- Apply temperature and softmax --- */
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_temp;
    lmc_softmax(logits, V);

    /* --- Build sorted nucleus --- */
    ProbIdx *sorted = (ProbIdx*)sorted_buf;
    for (int i = 0; i < V; i++) {
        sorted[i].prob = logits[i];
        sorted[i].idx  = i;
    }
    qsort(sorted, V, sizeof(ProbIdx), cmp_prob_desc);

    /* --- Find nucleus boundary --- */
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < V; i++) {
        cumsum += sorted[i].prob;
        nucleus_size = i + 1;
        if (cumsum >= top_p) break;
    }

    /* --- Renormalize within nucleus --- */
    float nucleus_sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) nucleus_sum += sorted[i].prob;
    float inv_ns = 1.0f / nucleus_sum;

    /* --- Sample --- */
    float r = rng_float(), cdf = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        cdf += sorted[i].prob * inv_ns;
        if (r < cdf) return sorted[i].idx;
    }

    /* Fallback to last token in nucleus (floating-point edge case) */
    return sorted[nucleus_size - 1].idx;
}
