/*
 * src/backends/gguf_loader.c — LMc GGUF Format Loader
 *
 * Loads model weights from GGUF files (llama.cpp / Ollama / HuggingFace
 * compatible format). Supported GGUF versions: 1, 2, 3.
 *
 * Supported tensor types:
 *   F32   (type 0)  — 32-bit float, no conversion
 *   F16   (type 1)  — 16-bit float, dequantized to F32
 *   Q8_0  (type 8)  — 8-bit symmetric quant, block size 32
 *   Q5_K  (type 13) — 5-bit K-quant (S and M), super-block 256
 *   Q6_K  (type 14) — 6-bit K-quant, super-block 256
 *
 * Adding support for a new GGUF quantization type:
 *   1. Add a new GGUF_TYPE_<n> constant below.
 *   2. Implement lmc_dequant_<n>() in src/quantization.c.
 *   3. Add a case in the tensor load loop in lmc_load_gguf().
 *   4. Update the supported types error message.
 *
 * GGUF tensor name → LMc weight pointer mapping is in gguf_name_to_ptr().
 * Adding a new model: add its tensor names there.
 *
 * References:
 *   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * GGUF FORMAT CONSTANTS
 * ============================================================ */
#define GGUF_MAGIC       0x46554747U   /* "GGUF" LE */
#define GGUF_VERSION_MIN 1
#define GGUF_VERSION_MAX 3
#define GGUF_ALIGNMENT   32            /* data section alignment in bytes */

/* Tensor type IDs (GGUF spec) */
#define GGUF_TYPE_F32    0
#define GGUF_TYPE_F16    1
#define GGUF_TYPE_Q8_0   8
#define GGUF_TYPE_Q5_K   13
#define GGUF_TYPE_Q6_K   14

/* Metadata value type IDs */
#define GGUF_MTYPE_UINT8    0
#define GGUF_MTYPE_INT8     1
#define GGUF_MTYPE_UINT16   2
#define GGUF_MTYPE_INT16    3
#define GGUF_MTYPE_UINT32   4
#define GGUF_MTYPE_INT32    5
#define GGUF_MTYPE_FLOAT32  6
#define GGUF_MTYPE_BOOL     7
#define GGUF_MTYPE_STRING   8
#define GGUF_MTYPE_ARRAY    9
#define GGUF_MTYPE_UINT64   10
#define GGUF_MTYPE_INT64    11
#define GGUF_MTYPE_FLOAT64  12

/* Raw bytes per block for each supported quantization */
#define Q8_0_BLOCK_SIZE       32
#define Q8_0_BYTES_PER_BLOCK  34

#define Q5_K_BLOCK_SIZE      256
#define Q5_K_BYTES_PER_BLOCK 176

#define Q6_K_BLOCK_SIZE      256
#define Q6_K_BYTES_PER_BLOCK 210

#define GGUF_MAX_TENSORS 512
#define GGUF_MAX_DIMS      4

/* ============================================================
 * LOW-LEVEL BINARY READERS (all little-endian)
 * ============================================================ */
#define DEF_READER(name, type) \
    static type gguf_##name(FILE *f) { \
        type v = 0; \
        if (fread(&v, sizeof(v), 1, f) != 1) { \
            LMC_FATAL("Unexpected EOF reading GGUF " #name); \
        } \
        return v; \
    }

DEF_READER(u8,  uint8_t)
DEF_READER(u16, uint16_t)
DEF_READER(u32, uint32_t)
DEF_READER(u64, uint64_t)

#undef DEF_READER

/* Read a GGUF string: uint64 length + raw bytes, NOT null-terminated in file */
static char* gguf_read_string(FILE *f) {
    uint64_t len = gguf_u64(f);
    char *s = (char*)malloc(len + 1);
    if (!s) LMC_FATAL("OOM reading GGUF string of length %llu", (unsigned long long)len);
    if (fread(s, 1, len, f) != len) LMC_FATAL("EOF reading GGUF string");
    s[len] = '\0';
    return s;
}

/* ============================================================
 * METADATA SKIPPING
 * Metadata is read for version/arch detection (future use).
 * For now we skip all key-value pairs.
 * ============================================================ */
static void gguf_skip_value(FILE *f, uint32_t vtype);

static void gguf_skip_array(FILE *f) {
    uint32_t elem_type = gguf_u32(f);
    uint64_t count     = gguf_u64(f);
    for (uint64_t i = 0; i < count; i++) gguf_skip_value(f, elem_type);
}

static void gguf_skip_value(FILE *f, uint32_t vtype) {
    uint8_t  u8v;
    uint16_t u16v;
    uint32_t u32v;
    uint64_t u64v;

    switch (vtype) {
        case GGUF_MTYPE_UINT8:
        case GGUF_MTYPE_INT8:
        case GGUF_MTYPE_BOOL:
            u8v = gguf_u8(f); (void)u8v; break;
        case GGUF_MTYPE_UINT16:
        case GGUF_MTYPE_INT16:
            u16v = gguf_u16(f); (void)u16v; break;
        case GGUF_MTYPE_UINT32:
        case GGUF_MTYPE_INT32:
        case GGUF_MTYPE_FLOAT32:
            u32v = gguf_u32(f); (void)u32v; break;
        case GGUF_MTYPE_UINT64:
        case GGUF_MTYPE_INT64:
        case GGUF_MTYPE_FLOAT64:
            u64v = gguf_u64(f); (void)u64v; break;
        case GGUF_MTYPE_STRING: {
            char *s = gguf_read_string(f);
            free(s);
            break;
        }
        case GGUF_MTYPE_ARRAY:   gguf_skip_array(f); break;
        default:
            LMC_FATAL("Unknown GGUF metadata type %u", vtype);
    }
}

/* ============================================================
 * TENSOR DESCRIPTOR
 * ============================================================ */
typedef struct {
    char     name[256];
    uint32_t type;          /* GGUF_TYPE_* */
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint64_t offset;        /* byte offset from data section start */
    size_t   n_elements;
} GGUFTensor;

/* ============================================================
 * TENSOR NAME → WEIGHT POINTER MAPPING (GPT-2)
 *
 * Maps GGUF tensor names to pointers in LmcModelWeights.
 * Returns NULL for tensors we don't need (will be skipped).
 *
 * To add a new model:
 *   - Add its tensor names here and map them to weight pointers.
 * ============================================================ */
static float** gguf_name_to_ptr(LmcContext *ctx, const char *name) {
    LmcModelWeights *mw = &ctx->weights;

    /* Global tensors */
    if (strcmp(name, "token_embd.weight")  == 0) return &mw->wte;
    if (strcmp(name, "position_embd.weight") == 0) return &mw->wpe;
    if (strcmp(name, "output_norm.weight") == 0) return &mw->ln_f_weight;
    if (strcmp(name, "output_norm.bias")   == 0) return &mw->ln_f_bias;
    if (strcmp(name, "output.weight")      == 0) return &mw->lm_head;

    /* Per-layer tensors: "blk.N.xxx" */
    if (strncmp(name, "blk.", 4) != 0) return NULL;

    int layer = atoi(name + 4);
    if (layer < 0 || layer >= GPT2_N_LAYERS) return NULL;

    const char *dot = strchr(name + 4, '.');
    if (!dot) return NULL;
    const char *rest = dot + 1;

    LmcLayerWeights *lw = &mw->layers[layer];

    if (strcmp(rest, "attn_norm.weight")   == 0) return &lw->ln1_weight;
    if (strcmp(rest, "attn_norm.bias")     == 0) return &lw->ln1_bias;
    if (strcmp(rest, "attn_qkv.weight")    == 0) return &lw->qkv_weight;
    if (strcmp(rest, "attn_qkv.bias")      == 0) return &lw->qkv_bias;
    if (strcmp(rest, "attn_output.weight") == 0) return &lw->attn_proj_weight;
    if (strcmp(rest, "attn_output.bias")   == 0) return &lw->attn_proj_bias;
    if (strcmp(rest, "ffn_norm.weight")    == 0) return &lw->ln2_weight;
    if (strcmp(rest, "ffn_norm.bias")      == 0) return &lw->ln2_bias;
    if (strcmp(rest, "ffn_up.weight")      == 0) return &lw->ffn_fc_weight;
    if (strcmp(rest, "ffn_up.bias")        == 0) return &lw->ffn_fc_bias;
    if (strcmp(rest, "ffn_down.weight")    == 0) return &lw->ffn_proj_weight;
    if (strcmp(rest, "ffn_down.bias")      == 0) return &lw->ffn_proj_bias;

    return NULL;
}

static const char* gguf_type_name(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_F32:  return "F32";
        case GGUF_TYPE_F16:  return "F16";
        case GGUF_TYPE_Q8_0: return "Q8_0";
        case GGUF_TYPE_Q5_K: return "Q5_K";
        case GGUF_TYPE_Q6_K: return "Q6_K";
        default:             return "UNKNOWN";
    }
}

/* ============================================================
 * MAIN GGUF LOADER
 * ============================================================ */
LmcError lmc_load_gguf(LmcContext *ctx, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        LMC_ERROR("Cannot open GGUF file: %s", path);
        return LMC_ERR_IO;
    }

    /* ---- Parse header ---- */
    uint32_t magic   = gguf_u32(f);
    uint32_t version = gguf_u32(f);

    if (magic != GGUF_MAGIC) {
        LMC_ERROR("Not a GGUF file: bad magic 0x%08X in %s", magic, path);
        fclose(f);
        return LMC_ERR_FORMAT;
    }
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        LMC_ERROR("Unsupported GGUF version %u (supported: %u-%u)",
                  version, GGUF_VERSION_MIN, GGUF_VERSION_MAX);
        fclose(f);
        return LMC_ERR_FORMAT;
    }

    uint64_t n_tensors = gguf_u64(f);
    uint64_t n_kv      = gguf_u64(f);

    LMC_INFO("GGUF version : %u", version);
    LMC_INFO("Tensors      : %llu", (unsigned long long)n_tensors);
    LMC_INFO("Metadata KVs : %llu", (unsigned long long)n_kv);

    /* ---- Skip metadata ---- */
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key  = gguf_read_string(f);
        uint32_t t = gguf_u32(f);
        gguf_skip_value(f, t);
        free(key);
    }

    /* ---- Read tensor info ---- */
    if (n_tensors > GGUF_MAX_TENSORS) {
        LMC_ERROR("Too many tensors: %llu > %d",
                  (unsigned long long)n_tensors, GGUF_MAX_TENSORS);
        fclose(f);
        return LMC_ERR_FORMAT;
    }

    GGUFTensor *tensors = (GGUFTensor*)calloc((size_t)n_tensors, sizeof(GGUFTensor));
    if (!tensors) { LMC_ERROR("OOM for tensor table"); fclose(f); return LMC_ERR_OOM; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &tensors[i];
        char *name = gguf_read_string(f);
        strncpy(t->name, name, sizeof(t->name) - 1);
        free(name);

        t->n_dims      = gguf_u32(f);
        t->n_elements  = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d]     = gguf_u64(f);
            t->n_elements *= (size_t)t->dims[d];
        }
        t->type   = gguf_u32(f);
        t->offset = gguf_u64(f);
    }

    /* ---- Locate data section (aligned to GGUF_ALIGNMENT bytes) ---- */
    long header_end = ftell(f);
    long data_start = ((long)header_end + GGUF_ALIGNMENT - 1)
                      / GGUF_ALIGNMENT * GGUF_ALIGNMENT;

    /* ---- Allocate arena ---- */
    size_t n_params   = lmc_gpt2_param_count();
    /* Reserve an extra wte-sized block for a possible separate output.weight */
    size_t lm_head_sz = (size_t)GPT2_VOCAB_SIZE * GPT2_EMBED_DIM;

    LMC_INFO("Parameters   : %zu  (%.1f MB float32)",
             n_params, n_params * 4.0 / (1024*1024));

    LmcError err = lmc_arena_init(&ctx->arena, n_params + lm_head_sz);
    if (err != LMC_OK) { free(tensors); fclose(f); return err; }

    err = lmc_gpt2_assign_weight_ptrs(ctx);
    if (err != LMC_OK) { free(tensors); fclose(f); return err; }

    /*
     * Allocate lm_head from arena and pre-populate with wte copy.
     * If "output.weight" tensor exists it will overwrite this.
     * If it doesn't (tied-weight GGUF), the wte copy is correct.
     */
    ctx->weights.lm_head = lmc_arena_alloc(&ctx->arena, lm_head_sz);
    memcpy(ctx->weights.lm_head, ctx->weights.wte,
           lm_head_sz * sizeof(float));

    /* ---- Load tensors ---- */
    int n_loaded = 0;

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &tensors[i];

        float **dst_ptr = gguf_name_to_ptr(ctx, t->name);
        if (!dst_ptr) {
            LMC_DEBUG("Skip %-50s  %s  n=%zu  (unmapped)",
                      t->name, gguf_type_name(t->type), t->n_elements);
            continue;
        }
        float *dst = *dst_ptr;

        /* Seek to tensor data */
        long tensor_pos = data_start + (long)t->offset;
        if (fseek(f, tensor_pos, SEEK_SET) != 0) {
            LMC_ERROR("Cannot seek to tensor %s at offset %ld", t->name, tensor_pos);
            free(tensors); fclose(f); return LMC_ERR_IO;
        }

        /* Load and dequantize based on type */
        switch (t->type) {

            case GGUF_TYPE_F32: {
                size_t n = fread(dst, sizeof(float), t->n_elements, f);
                if (n != t->n_elements) {
                    LMC_ERROR("Short read on F32 tensor %s (%zu/%zu)",
                              t->name, n, t->n_elements);
                    free(tensors); fclose(f); return LMC_ERR_IO;
                }
                break;
            }

            case GGUF_TYPE_F16: {
                size_t raw_bytes = t->n_elements * sizeof(uint16_t);
                uint8_t *tmp = (uint8_t*)malloc(raw_bytes);
                if (!tmp) { LMC_ERROR("OOM for F16 buffer"); free(tensors); fclose(f); return LMC_ERR_OOM; }
                if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                    LMC_ERROR("Short read on F16 tensor %s", t->name);
                    free(tmp); free(tensors); fclose(f); return LMC_ERR_IO;
                }
                lmc_dequant_f16(tmp, dst, t->n_elements);
                free(tmp);
                break;
            }

            case GGUF_TYPE_Q8_0: {
                if (t->n_elements % Q8_0_BLOCK_SIZE != 0) {
                    LMC_ERROR("Q8_0 tensor %s: %zu elements not multiple of %d",
                              t->name, t->n_elements, Q8_0_BLOCK_SIZE);
                    free(tensors); fclose(f); return LMC_ERR_QUANT;
                }
                size_t raw_bytes = (t->n_elements / Q8_0_BLOCK_SIZE) * Q8_0_BYTES_PER_BLOCK;
                uint8_t *tmp = (uint8_t*)malloc(raw_bytes);
                if (!tmp) { LMC_ERROR("OOM for Q8_0 buffer"); free(tensors); fclose(f); return LMC_ERR_OOM; }
                if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                    LMC_ERROR("Short read on Q8_0 tensor %s", t->name);
                    free(tmp); free(tensors); fclose(f); return LMC_ERR_IO;
                }
                lmc_dequant_q8_0(tmp, dst, t->n_elements);
                free(tmp);
                break;
            }

            case GGUF_TYPE_Q5_K: {
                if (t->n_elements % Q5_K_BLOCK_SIZE != 0) {
                    LMC_ERROR("Q5_K tensor %s: %zu elements not multiple of %d",
                              t->name, t->n_elements, Q5_K_BLOCK_SIZE);
                    free(tensors); fclose(f); return LMC_ERR_QUANT;
                }
                size_t raw_bytes = (t->n_elements / Q5_K_BLOCK_SIZE) * Q5_K_BYTES_PER_BLOCK;
                uint8_t *tmp = (uint8_t*)malloc(raw_bytes);
                if (!tmp) { LMC_ERROR("OOM for Q5_K buffer"); free(tensors); fclose(f); return LMC_ERR_OOM; }
                if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                    LMC_ERROR("Short read on Q5_K tensor %s", t->name);
                    free(tmp); free(tensors); fclose(f); return LMC_ERR_IO;
                }
                lmc_dequant_q5k(tmp, dst, t->n_elements);
                free(tmp);
                break;
            }

            case GGUF_TYPE_Q6_K: {
                if (t->n_elements % Q6_K_BLOCK_SIZE != 0) {
                    LMC_ERROR("Q6_K tensor %s: %zu elements not multiple of %d",
                              t->name, t->n_elements, Q6_K_BLOCK_SIZE);
                    free(tensors); fclose(f); return LMC_ERR_QUANT;
                }
                size_t raw_bytes = (t->n_elements / Q6_K_BLOCK_SIZE) * Q6_K_BYTES_PER_BLOCK;
                uint8_t *tmp = (uint8_t*)malloc(raw_bytes);
                if (!tmp) { LMC_ERROR("OOM for Q6_K buffer"); free(tensors); fclose(f); return LMC_ERR_OOM; }
                if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                    LMC_ERROR("Short read on Q6_K tensor %s", t->name);
                    free(tmp); free(tensors); fclose(f); return LMC_ERR_IO;
                }
                lmc_dequant_q6k(tmp, dst, t->n_elements);
                free(tmp);
                break;
            }

            default:
                LMC_ERROR("Unsupported tensor type %u (%s) for tensor: %s",
                          t->type, gguf_type_name(t->type), t->name);
                LMC_ERROR("Supported: F32(0) F16(1) Q8_0(8) Q5_K(13) Q6_K(14)");
                LMC_ERROR("Convert with: ./quantize model.gguf out.gguf Q5_K_M");
                free(tensors); fclose(f);
                return LMC_ERR_QUANT;
        }

        LMC_VERBOSE("Loaded %-50s  %s  n=%zu",
                    t->name, gguf_type_name(t->type), t->n_elements);
        n_loaded++;
    }

    free(tensors);
    fclose(f);

    LMC_INFO("Tensors loaded: %d", n_loaded);
    LMC_INFO("Loaded        : %s (GGUF)", path);
    return LMC_OK;
}
