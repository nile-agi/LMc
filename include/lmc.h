/*
 * lmc.h — LMc (Local Model Compute) — Public API
 *
 * LMc is an open-source, edge-first AI inference engine written in pure C99.
 * Designed to run AI models efficiently on ALL devices: phones, laptops,
 * desktops, tablets — with no GPU required.
 *
 * Targets:
 *   - All CPUs  (x86-64, ARM64, ARM32, RISC-V, MIPS, PowerPC, ...)
 *   - All GPUs  (via future backends: CUDA, OpenCL, Metal, Vulkan, ...)
 *   - All NPUs  (via future backends: CoreML, NNAPI, QNN, ...)
 *   - Edge devices in Africa and low-resource environments globally
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2024 LMc Contributors
 */

#ifndef LMC_H
#define LMC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* ============================================================
 * VERSION
 * ============================================================ */
#define LMC_VERSION_MAJOR 0
#define LMC_VERSION_MINOR 1
#define LMC_VERSION_PATCH 0
#define LMC_VERSION_STR   "0.1.0"

/* ============================================================
 * COMPILE-TIME SANITY
 * ============================================================ */
typedef char lmc_assert_float32_is_4bytes[(sizeof(float) == 4) ? 1 : -1];
typedef char lmc_assert_uint8_is_1byte[(sizeof(uint8_t) == 1) ? 1 : -1];

/* ============================================================
 * ERROR CODES
 * ============================================================ */
typedef enum {
    LMC_OK              =  0,
    LMC_ERR_IO          = -1,   /* File read/write error          */
    LMC_ERR_OOM         = -2,   /* Out of memory                  */
    LMC_ERR_FORMAT      = -3,   /* Unknown or unsupported format  */
    LMC_ERR_ARCH        = -4,   /* Architecture mismatch          */
    LMC_ERR_QUANT       = -5,   /* Unsupported quantization type  */
    LMC_ERR_TOKENIZER   = -6,   /* Tokenizer error                */
    LMC_ERR_CONTEXT     = -7,   /* Context window exceeded        */
    LMC_ERR_INVALID_ARG = -8,   /* Bad argument                   */
} LmcError;

const char* lmc_error_str(LmcError err);

/* ============================================================
 * LOGGING
 * ============================================================ */
typedef enum {
    LMC_LOG_SILENT  = 0,
    LMC_LOG_ERROR   = 1,
    LMC_LOG_WARN    = 2,
    LMC_LOG_INFO    = 3,
    LMC_LOG_DEBUG   = 4,
    LMC_LOG_VERBOSE = 5,
} LmcLogLevel;

/* Set global log level (default: LMC_LOG_INFO) */
void lmc_set_log_level(LmcLogLevel level);
LmcLogLevel lmc_get_log_level(void);

/* ============================================================
 * HARDWARE BACKEND FLAGS
 * (future: select compute backend per inference call)
 * ============================================================ */
typedef enum {
    LMC_BACKEND_CPU      = (1 << 0),   /* Pure C, always available       */
    LMC_BACKEND_OPENMP   = (1 << 1),   /* CPU multi-threading via OpenMP */
    LMC_BACKEND_CUDA     = (1 << 2),   /* NVIDIA CUDA (future)           */
    LMC_BACKEND_OPENCL   = (1 << 3),   /* OpenCL (future)                */
    LMC_BACKEND_METAL    = (1 << 4),   /* Apple Metal (future)           */
    LMC_BACKEND_VULKAN   = (1 << 5),   /* Vulkan compute (future)        */
    LMC_BACKEND_NNAPI    = (1 << 6),   /* Android NNAPI / NPU (future)   */
    LMC_BACKEND_COREML   = (1 << 7),   /* Apple CoreML / ANE (future)    */
    LMC_BACKEND_AUTO     = (int)0x7FFFFFFF, /* Auto-select best available     */
} LmcBackendFlags;

/* Probe which backends are compiled in and available at runtime */
LmcBackendFlags lmc_available_backends(void);
const char*     lmc_backend_name(LmcBackendFlags flag);

/* ============================================================
 * QUANTIZATION TYPES
 * ============================================================ */
typedef enum {
    LMC_QUANT_F32   = 0,  /* 32-bit float            */
    LMC_QUANT_F16   = 1,  /* 16-bit float            */
    LMC_QUANT_Q8_0  = 8,  /* 8-bit symmetric, block  */
    LMC_QUANT_Q5_K  = 13, /* 5-bit K-quant           */
    LMC_QUANT_Q6_K  = 14, /* 6-bit K-quant           */
    LMC_QUANT_UNKNOWN = -1,
} LmcQuantType;

const char* lmc_quant_name(LmcQuantType q);

/* ============================================================
 * MODEL FILE FORMAT
 * ============================================================ */
typedef enum {
    LMC_FORMAT_UNKNOWN = 0,
    LMC_FORMAT_BIN     = 1,  /* LMc custom float32 binary  */
    LMC_FORMAT_GGUF    = 2,  /* GGUF (llama.cpp compatible) */
} LmcModelFormat;

const char* lmc_format_name(LmcModelFormat fmt);
LmcModelFormat lmc_detect_format(const char *path);
const char*    lmc_find_default_model(void);

/* ============================================================
 * GENERATION CONFIG
 * ============================================================ */
typedef struct {
    int   max_new_tokens; /* Max tokens to generate (default: 128)        */
    float temperature;    /* Sampling temperature, 0 = greedy (default: 0.7) */
    float top_p;          /* Nucleus sampling threshold (default: 0.9)    */
    int   seed;           /* RNG seed, -1 = use time() (default: -1)      */
    /* Future: top_k, repetition_penalty, stop_tokens[], stream_callback  */
} LmcGenConfig;

/* Fill config with sensible defaults */
void lmc_gen_config_default(LmcGenConfig *cfg);

/* ============================================================
 * MAIN CONTEXT (opaque — allocate with lmc_ctx_new)
 * ============================================================ */
typedef struct LmcContext LmcContext;

LmcContext* lmc_ctx_new(void);
void        lmc_ctx_free(LmcContext *ctx);

/* Load model from file (auto-detects format) */
LmcError lmc_load_model(LmcContext *ctx, const char *model_path);

/* Load tokenizer files */
LmcError lmc_load_tokenizer(LmcContext *ctx,
                             const char *encoder_json_path,
                             const char *vocab_bpe_path);

/* Run text generation. Writes to stdout by default (streaming). */
LmcError lmc_generate(LmcContext *ctx,
                       const char *prompt,
                       const LmcGenConfig *cfg);

/* ============================================================
 * SYSTEM INFO
 * ============================================================ */
void lmc_print_system_info(void);
void lmc_print_build_info(void);

#ifdef __cplusplus
}
#endif

#endif /* LMC_H */
