/*
 * src/lmc.c — LMc Core API Implementation
 *
 * Implements: context lifecycle, format detection, model/tokenizer loading,
 * generation entry point, logging, system info.
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"
#include <time.h>

/* ============================================================
 * GLOBAL LOG LEVEL
 * ============================================================ */
static LmcLogLevel g_log_level = LMC_LOG_INFO;

void lmc_set_log_level(LmcLogLevel level) { g_log_level = level; }
LmcLogLevel lmc_get_log_level(void)       { return g_log_level; }

/* ============================================================
 * ERROR STRINGS
 * ============================================================ */
const char* lmc_error_str(LmcError err) {
    switch (err) {
        case LMC_OK:              return "OK";
        case LMC_ERR_IO:          return "I/O error";
        case LMC_ERR_OOM:         return "Out of memory";
        case LMC_ERR_FORMAT:      return "Unknown or unsupported file format";
        case LMC_ERR_ARCH:        return "Architecture mismatch";
        case LMC_ERR_QUANT:       return "Unsupported quantization type";
        case LMC_ERR_TOKENIZER:   return "Tokenizer error";
        case LMC_ERR_CONTEXT:     return "Context window exceeded";
        case LMC_ERR_INVALID_ARG: return "Invalid argument";
        default:                  return "Unknown error";
    }
}

/* ============================================================
 * QUANT / FORMAT NAMES
 * ============================================================ */
const char* lmc_quant_name(LmcQuantType q) {
    switch (q) {
        case LMC_QUANT_F32:  return "F32";
        case LMC_QUANT_F16:  return "F16";
        case LMC_QUANT_Q8_0: return "Q8_0";
        case LMC_QUANT_Q5_K: return "Q5_K";
        case LMC_QUANT_Q6_K: return "Q6_K";
        default:             return "UNKNOWN";
    }
}

const char* lmc_format_name(LmcModelFormat fmt) {
    switch (fmt) {
        case LMC_FORMAT_BIN:  return "LMc binary (.bin)";
        case LMC_FORMAT_GGUF: return "GGUF (.gguf)";
        default:              return "Unknown";
    }
}

/* ============================================================
 * BACKEND INFO
 * ============================================================ */
LmcBackendFlags lmc_available_backends(void) {
    LmcBackendFlags flags = LMC_BACKEND_CPU;
#ifdef _OPENMP
    flags |= LMC_BACKEND_OPENMP;
#endif
    /* Future: probe CUDA, Metal, OpenCL, etc. */
    return flags;
}

const char* lmc_backend_name(LmcBackendFlags flag) {
    switch (flag) {
        case LMC_BACKEND_CPU:    return "CPU";
        case LMC_BACKEND_OPENMP: return "OpenMP";
        case LMC_BACKEND_CUDA:   return "CUDA";
        case LMC_BACKEND_OPENCL: return "OpenCL";
        case LMC_BACKEND_METAL:  return "Metal";
        case LMC_BACKEND_VULKAN: return "Vulkan";
        case LMC_BACKEND_NNAPI:  return "NNAPI";
        case LMC_BACKEND_COREML: return "CoreML";
        default:                 return "Unknown";
    }
}

/* ============================================================
 * BUILD / SYSTEM INFO
 * ============================================================ */
void lmc_print_build_info(void) {
    fprintf(stderr,
        "lmc [INFO] Version    : " LMC_VERSION_STR "\n"
        "lmc [INFO] Compiled   : " __DATE__ " " __TIME__ "\n"
#ifdef __clang__
        "lmc [INFO] Compiler   : Clang " __clang_version__ "\n"
#elif defined(__GNUC__)
        "lmc [INFO] Compiler   : GCC " __VERSION__ "\n"
#else
        "lmc [INFO] Compiler   : Unknown\n"
#endif
#ifdef __x86_64__
        "lmc [INFO] Arch       : x86-64\n"
#elif defined(__aarch64__)
        "lmc [INFO] Arch       : ARM64 (AArch64)\n"
#elif defined(__arm__)
        "lmc [INFO] Arch       : ARM32\n"
#elif defined(__riscv)
        "lmc [INFO] Arch       : RISC-V\n"
#else
        "lmc [INFO] Arch       : Generic\n"
#endif
#ifdef _OPENMP
        "lmc [INFO] OpenMP     : enabled\n"
#else
        "lmc [INFO] OpenMP     : disabled\n"
#endif
    );
}

void lmc_print_system_info(void) {
    lmc_print_build_info();

    LmcBackendFlags avail = lmc_available_backends();
    fprintf(stderr, "lmc [INFO] Backends   :");
    static const LmcBackendFlags all_flags[] = {
        LMC_BACKEND_CPU, LMC_BACKEND_OPENMP, LMC_BACKEND_CUDA,
        LMC_BACKEND_OPENCL, LMC_BACKEND_METAL, LMC_BACKEND_VULKAN,
        LMC_BACKEND_NNAPI, LMC_BACKEND_COREML
    };
    for (size_t i = 0; i < sizeof(all_flags)/sizeof(all_flags[0]); i++) {
        if (avail & all_flags[i])
            fprintf(stderr, " %s", lmc_backend_name(all_flags[i]));
    }
    fprintf(stderr, "\n");

#ifdef _OPENMP
    fprintf(stderr, "lmc [INFO] CPU Threads: %d\n", omp_get_max_threads());
#endif
}

/* ============================================================
 * FORMAT DETECTION
 * ============================================================ */
#define BIN_MAGIC  0x47505432U   /* "GPT2" */
#define GGUF_MAGIC 0x46554747U   /* "GGUF" */

LmcModelFormat lmc_detect_format(const char *path) {
    /* Fast path: file extension */
    size_t len = strlen(path);
    if (len >= 5 && strcmp(path + len - 5, ".gguf") == 0) return LMC_FORMAT_GGUF;
    if (len >= 4 && strcmp(path + len - 4, ".bin")  == 0) return LMC_FORMAT_BIN;

    /* Peek at magic bytes */
    FILE *f = fopen(path, "rb");
    if (!f) return LMC_FORMAT_UNKNOWN;
    uint32_t magic = 0;
    size_t n = fread(&magic, sizeof(uint32_t), 1, f);
    fclose(f);
    (void)n; /* we check magic value; partial read gives 0 which matches nothing */
    if (magic == BIN_MAGIC)  return LMC_FORMAT_BIN;
    if (magic == GGUF_MAGIC) return LMC_FORMAT_GGUF;
    return LMC_FORMAT_UNKNOWN;
}

static const char* const k_default_model_paths[] = {
    "gpt2_124m.bin",
    "gpt2.f16.gguf",
    "gpt2.Q8_0.gguf",
    "gpt2.Q6_K.gguf",
    "gpt2.gguf",
    NULL
};

const char* lmc_find_default_model(void) {
    for (int i = 0; k_default_model_paths[i]; i++) {
        FILE *f = fopen(k_default_model_paths[i], "rb");
        if (f) { fclose(f); return k_default_model_paths[i]; }
    }
    return NULL;
}

/* ============================================================
 * CONTEXT LIFECYCLE
 * ============================================================ */
LmcContext* lmc_ctx_new(void) {
    LmcContext *ctx = (LmcContext*)calloc(1, sizeof(LmcContext));
    if (!ctx) {
        LMC_ERROR("Failed to allocate LmcContext (%zu bytes)", sizeof(LmcContext));
        return NULL;
    }
    return ctx;
}

void lmc_ctx_free(LmcContext *ctx) {
    if (!ctx) return;
    lmc_arena_free(&ctx->arena);
    lmc_kv_cache_free(ctx);
    lmc_activations_free(ctx);
    free(ctx);
}

/* ============================================================
 * GEN CONFIG DEFAULTS
 * ============================================================ */
void lmc_gen_config_default(LmcGenConfig *cfg) {
    if (!cfg) return;
    cfg->max_new_tokens = 128;
    cfg->temperature    = 0.7f;
    cfg->top_p          = 0.9f;
    cfg->seed           = -1;  /* use time() */
}

/* ============================================================
 * LOAD MODEL
 * ============================================================ */
LmcError lmc_load_model(LmcContext *ctx, const char *path) {
    if (!ctx || !path) return LMC_ERR_INVALID_ARG;

    LmcModelFormat fmt = lmc_detect_format(path);
    LMC_INFO("Model file  : %s", path);
    LMC_INFO("Format      : %s", lmc_format_name(fmt));

    ctx->format = fmt;
    snprintf(ctx->model_path, sizeof(ctx->model_path), "%s", path);

    LmcError err;
    switch (fmt) {
        case LMC_FORMAT_BIN:
            err = lmc_load_bin(ctx, path);
            break;
        case LMC_FORMAT_GGUF:
            err = lmc_load_gguf(ctx, path);
            break;
        default:
            LMC_ERROR("Cannot determine format of: %s", path);
            LMC_ERROR("Supported: *.bin (LMc float32) | *.gguf (GGUF F32/F16/Q8_0/Q5_K/Q6_K)");
            return LMC_ERR_FORMAT;
    }

    if (err != LMC_OK) return err;
    ctx->model_loaded = 1;
    return LMC_OK;
}

/* ============================================================
 * LOAD TOKENIZER
 * ============================================================ */
LmcError lmc_load_tokenizer(LmcContext *ctx,
                              const char *encoder_path,
                              const char *bpe_path) {
    if (!ctx || !encoder_path || !bpe_path) return LMC_ERR_INVALID_ARG;

    LmcError err;

    err = lmc_tokenizer_init_byte_encoder(&ctx->tokenizer);
    if (err != LMC_OK) return err;

    err = lmc_tokenizer_load_encoder(&ctx->tokenizer, encoder_path);
    if (err != LMC_OK) return err;

    err = lmc_tokenizer_load_bpe(&ctx->tokenizer, bpe_path);
    if (err != LMC_OK) return err;

    ctx->tokenizer_loaded = 1;
    return LMC_OK;
}

/* ============================================================
 * GENERATION
 * ============================================================ */
LmcError lmc_generate(LmcContext *ctx,
                       const char *prompt,
                       const LmcGenConfig *cfg) {
    if (!ctx)   return LMC_ERR_INVALID_ARG;
    if (!prompt) return LMC_ERR_INVALID_ARG;

    if (!ctx->model_loaded) {
        LMC_ERROR("No model loaded. Call lmc_load_model() first.");
        return LMC_ERR_INVALID_ARG;
    }
    if (!ctx->tokenizer_loaded) {
        LMC_ERROR("No tokenizer loaded. Call lmc_load_tokenizer() first.");
        return LMC_ERR_INVALID_ARG;
    }

    LmcGenConfig default_cfg;
    if (!cfg) {
        lmc_gen_config_default(&default_cfg);
        cfg = &default_cfg;
    }

    /* Clamp config values */
    int   max_new = cfg->max_new_tokens;
    float temp    = cfg->temperature;
    float top_p   = cfg->top_p;

    if (max_new <= 0)              max_new = 128;
    if (max_new > GPT2_SEQ_LEN-10) max_new = GPT2_SEQ_LEN-10;
    if (temp < 0.0f)               temp    = 0.7f;
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 0.9f;

    /* Seed RNG */
    uint64_t seed = (cfg->seed < 0)
                    ? (uint64_t)time(NULL)
                    : (uint64_t)cfg->seed;
    lmc_rng_seed(seed);

    /* Tokenize prompt */
    int prompt_ids[GPT2_SEQ_LEN];
    int n_prompt = lmc_tokenize(&ctx->tokenizer, prompt,
                                prompt_ids, GPT2_SEQ_LEN);
    if (n_prompt == 0) {
        LMC_ERROR("Empty prompt after tokenization.");
        return LMC_ERR_TOKENIZER;
    }

    LMC_INFO("Prompt tokens: %d", n_prompt);
    fprintf(stdout, "%s", prompt);
    fflush(stdout);

    /* Reset KV cache */
    ctx->kv_cache.seq_len = 0;

    /* Prefill: run model over all prompt tokens */
    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        logits = lmc_gpt2_forward(ctx, prompt_ids[i], i);
        ctx->kv_cache.seq_len = i + 1;
    }

    /* Decode: autoregressive generation */
    int pos = n_prompt;
    char decode_buf[64];

    for (int step = 0; step < max_new; step++) {
        if (pos >= GPT2_SEQ_LEN) {
            fprintf(stdout, "\n[context window full]\n");
            break;
        }

        int next_token = lmc_sample_top_p(logits, temp, top_p,
                                          ctx->act.sorted_buf);

        if (next_token == GPT2_EOS_TOKEN) {
            fprintf(stdout, "\n[EOS]\n");
            break;
        }

        int dec_len = lmc_detokenize_token(&ctx->tokenizer, next_token,
                                           decode_buf, sizeof(decode_buf));
        if (dec_len > 0) {
            fwrite(decode_buf, 1, (size_t)dec_len, stdout);
            fflush(stdout);
        }

        logits = lmc_gpt2_forward(ctx, next_token, pos);
        ctx->kv_cache.seq_len = pos + 1;
        pos++;
    }

    fprintf(stdout, "\n");
    LMC_INFO("Generated %d tokens.", pos - n_prompt);
    return LMC_OK;
}
