/*
 * lmc_internal.h — LMc Internal Shared Definitions
 *
 * NOT part of the public API. Used only by LMc source files.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LMC_INTERNAL_H
#define LMC_INTERNAL_H

#include "lmc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* ============================================================
 * LOGGING MACROS
 * ============================================================ */
/* Internal logging — always goes to stderr.
 * Using the GNU ##__VA_ARGS__ extension for zero-argument calls.
 * This is supported by GCC and Clang on all target platforms. */
#define LMC_LOG(level, fmt, ...) \
    do { \
        if ((level) <= lmc_get_log_level()) { \
            static const char *const _pfx[] = { \
                "", "[ERROR]", "[WARN] ", "[INFO] ", "[DEBUG]", "[VERB] " \
            }; \
            fprintf(stderr, "lmc %s " fmt "\n", _pfx[(level)], ##__VA_ARGS__); \
        } \
    } while(0)

#define LMC_ERROR(fmt, ...)   LMC_LOG(LMC_LOG_ERROR,   fmt, ##__VA_ARGS__)
#define LMC_WARN(fmt, ...)    LMC_LOG(LMC_LOG_WARN,    fmt, ##__VA_ARGS__)
#define LMC_INFO(fmt, ...)    LMC_LOG(LMC_LOG_INFO,    fmt, ##__VA_ARGS__)
#define LMC_DEBUG(fmt, ...)   LMC_LOG(LMC_LOG_DEBUG,   fmt, ##__VA_ARGS__)
#define LMC_VERBOSE(fmt, ...) LMC_LOG(LMC_LOG_VERBOSE, fmt, ##__VA_ARGS__)

/* Fatal: print and abort */
#define LMC_FATAL(fmt, ...) \
    do { \
        fprintf(stderr, "lmc [FATAL] " fmt "\n", ##__VA_ARGS__); \
        exit(1); \
    } while(0)

/* ============================================================
 * GPT-2 124M ARCHITECTURE CONSTANTS
 * ============================================================ */
#define GPT2_VOCAB_SIZE   50257
#define GPT2_SEQ_LEN      1024
#define GPT2_N_LAYERS     12
#define GPT2_N_HEADS      12
#define GPT2_EMBED_DIM    768
#define GPT2_FFN_DIM      3072   /* 4 * EMBED_DIM          */
#define GPT2_HEAD_DIM     64     /* EMBED_DIM / N_HEADS    */
#define GPT2_EOS_TOKEN    50256

/* ============================================================
 * MEMORY ARENA
 * Weights live in a single contiguous allocation for cache
 * friendliness and simple lifecycle management.
 * ============================================================ */
typedef struct {
    float  *data;
    size_t  capacity; /* in floats */
    size_t  used;     /* in floats */
} LmcArena;

LmcError lmc_arena_init(LmcArena *a, size_t n_floats);
void     lmc_arena_free(LmcArena *a);
float*   lmc_arena_alloc(LmcArena *a, size_t n_floats);

/* ============================================================
 * WEIGHT STRUCTURES
 * ============================================================ */
typedef struct {
    float *ln1_weight;        /* [EMBED_DIM]               */
    float *ln1_bias;          /* [EMBED_DIM]               */
    float *qkv_weight;        /* [3*EMBED_DIM, EMBED_DIM]  */
    float *qkv_bias;          /* [3*EMBED_DIM]             */
    float *attn_proj_weight;  /* [EMBED_DIM, EMBED_DIM]    */
    float *attn_proj_bias;    /* [EMBED_DIM]               */
    float *ln2_weight;        /* [EMBED_DIM]               */
    float *ln2_bias;          /* [EMBED_DIM]               */
    float *ffn_fc_weight;     /* [FFN_DIM, EMBED_DIM]      */
    float *ffn_fc_bias;       /* [FFN_DIM]                 */
    float *ffn_proj_weight;   /* [EMBED_DIM, FFN_DIM]      */
    float *ffn_proj_bias;     /* [EMBED_DIM]               */
} LmcLayerWeights;

typedef struct {
    float          *wte;                    /* [VOCAB_SIZE, EMBED_DIM] token embeddings   */
    float          *wpe;                    /* [SEQ_LEN, EMBED_DIM]    position embeddings */
    LmcLayerWeights layers[GPT2_N_LAYERS];
    float          *ln_f_weight;            /* [EMBED_DIM]             */
    float          *ln_f_bias;              /* [EMBED_DIM]             */
    float          *lm_head;               /* [VOCAB_SIZE, EMBED_DIM]
                                             * .bin:  tied, == wte
                                             * .gguf: output.weight or wte fallback */
} LmcModelWeights;

/* ============================================================
 * KV CACHE
 * ============================================================ */
typedef struct {
    float *k_cache;  /* [N_LAYERS, SEQ_LEN, N_HEADS, HEAD_DIM] */
    float *v_cache;  /* [N_LAYERS, SEQ_LEN, N_HEADS, HEAD_DIM] */
    int    seq_len;  /* tokens currently in cache               */
} LmcKVCache;

/* ============================================================
 * ACTIVATION BUFFERS (one set, reused per token)
 * ============================================================ */
typedef struct {
    float *x;            /* [EMBED_DIM]         current hidden state */
    float *x_norm;       /* [EMBED_DIM]         layer-normed         */
    float *qkv;          /* [3 * EMBED_DIM]     Q, K, V projections  */
    float *attn_out;     /* [EMBED_DIM]         attention output      */
    float *proj_out;     /* [EMBED_DIM]         after attn projection */
    float *ffn_hidden;   /* [FFN_DIM]           FFN intermediate      */
    float *ffn_out;      /* [EMBED_DIM]         FFN output            */
    float *logits;       /* [VOCAB_SIZE]        final logits          */
    float *attn_scores;  /* [N_HEADS, SEQ_LEN]  per-head scores       */
    void  *sorted_buf;   /* [VOCAB_SIZE] ProbIdx for top-p sampling   */
} LmcActivations;

/* ============================================================
 * TOKENIZER
 * ============================================================ */
#define BPE_MAX_VOCAB      50257
#define BPE_MAX_MERGES     50000
#define BPE_TOKEN_MAX_LEN  256
#define VOCAB_HASH_SIZE    131072

typedef struct {
    int left, right, result;
} LmcBPEMerge;

typedef struct {
    uint8_t bytes[BPE_TOKEN_MAX_LEN];
    int     len;
} LmcVocabEntry;

typedef struct {
    LmcVocabEntry vocab[BPE_MAX_VOCAB];
    int           vocab_size;
    LmcBPEMerge   merges[BPE_MAX_MERGES];
    int           n_merges;
    char          byte_encoder[256][8];
    int           byte_decoder[0x400];
    int           vocab_hash[VOCAB_HASH_SIZE];
    int           vocab_hash_next[BPE_MAX_VOCAB];
} LmcTokenizer;

/* ============================================================
 * MAIN CONTEXT (opaque, allocated on heap)
 * ============================================================ */
struct LmcContext {
    LmcArena        arena;
    LmcModelWeights weights;
    LmcKVCache      kv_cache;
    LmcActivations  act;
    LmcTokenizer    tokenizer;
    LmcModelFormat  format;
    char            model_path[512];

    /* state flags */
    int model_loaded;
    int tokenizer_loaded;
};

/* ============================================================
 * INTERNAL FUNCTION DECLARATIONS
 * (implemented across multiple .c files)
 * ============================================================ */

/* --- math ops (math_ops.c) --- */
float  lmc_f16_to_f32(uint16_t h);
float  lmc_gelu(float x);
void   lmc_softmax(float *x, int n);
void   lmc_layer_norm(float *out, const float *x,
                      const float *w, const float *b, int dim);
void   lmc_matmul_vec(float *out, const float *weight, const float *bias,
                      const float *in, int M, int K);

/* --- quantization (quantization.c) --- */
void lmc_dequant_f16(const uint8_t *src, float *dst, size_t n);
void lmc_dequant_q8_0(const uint8_t *src, float *dst, size_t n);
void lmc_dequant_q5k(const uint8_t *src, float *dst, size_t n);
void lmc_dequant_q6k(const uint8_t *src, float *dst, size_t n);

/* --- weights (models/gpt2_weights.c) --- */
size_t   lmc_gpt2_param_count(void);
LmcError lmc_gpt2_assign_weight_ptrs(LmcContext *ctx);

/* --- loaders (bin_loader.c, gguf_loader.c) --- */
LmcError lmc_load_bin(LmcContext *ctx, const char *path);
LmcError lmc_load_gguf(LmcContext *ctx, const char *path);

/* --- inference (models/gpt2_inference.c) --- */
float*   lmc_gpt2_forward(LmcContext *ctx, int token_id, int pos);
LmcError lmc_kv_cache_init(LmcContext *ctx);
void     lmc_kv_cache_free(LmcContext *ctx);
LmcError lmc_activations_init(LmcContext *ctx);
void     lmc_activations_free(LmcContext *ctx);

/* --- tokenizer (tokenizer.c) --- */
LmcError lmc_tokenizer_init_byte_encoder(LmcTokenizer *tok);
LmcError lmc_tokenizer_load_encoder(LmcTokenizer *tok, const char *path);
LmcError lmc_tokenizer_load_bpe(LmcTokenizer *tok, const char *path);
int      lmc_tokenize(const LmcTokenizer *tok, const char *text,
                      int *out_ids, int max_tokens);
int      lmc_detokenize_token(const LmcTokenizer *tok, int token_id,
                               char *out_buf, int buf_size);

/* --- sampling (sampling.c) --- */
void lmc_rng_seed(uint64_t seed);
int  lmc_sample_top_p(float *logits, float temperature, float top_p,
                      void *sorted_buf);

#endif /* LMC_INTERNAL_H */
