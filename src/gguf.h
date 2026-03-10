/* gguf.h — GGUF format constants, metadata key strings, tensor descriptor,
 *           and the public model-loading API.
 *
 * GGUF is the binary model container format used by llama.cpp and exported
 * by HuggingFace.  A GGUF file contains:
 *   1. A fixed header   (magic, version, tensor count, metadata KV count)
 *   2. Metadata KV pairs (architecture name, hyperparameters, tokenizer info)
 *   3. Tensor info list  (name, shape, type, file offset)
 *   4. Tensor data       (aligned to 32 bytes; quantised or float)
 *
 * Supported architectures:    gpt2, llama (also covers mistral, qwen, phi-3)
 * Supported GGUF versions:    1, 2, 3
 * Supported tensor types:     see quant.h (F32, F16, Q2_K … IQ4_XS)
 *
 * To add a new architecture:
 *   1. Add the GGUF metadata key constants below.
 *   2. Parse them in load_model_gguf() in gguf.c.
 *   3. Add tensor name mappings in gguf_name_to_ptr().
 *   4. Add configure_<arch>() in models.c.
 *   See docs/adding_new_model.md for a full walkthrough.
 */
#ifndef LMC_GGUF_H
#define LMC_GGUF_H

#include <stddef.h>
#include <stdint.h>

/* ── File format identifiers ──────────────────────────────────────────────── */
#define GGUF_MAGIC          0x46554747U   /* little-endian "GGUF"             */
#define GGUF_VERSION_MIN    1
#define GGUF_VERSION_MAX    3

/* Legacy custom binary format written by lmc itself (backward-compat only). */
#define MODEL_MAGIC         0x47505432U   /* "GPT2"                           */
#define MODEL_VERSION       1

/* ── Metadata value types ─────────────────────────────────────────────────── *
 * GGUF_MTYPE_* values identify the C type of a metadata value field.        */
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

/* ── Universal metadata keys (all architectures) ─────────────────────────── */
#define GGUF_KEY_ARCH           "general.architecture"  /* "gpt2" | "llama" … */
#define GGUF_KEY_NAME           "general.name"
#define GGUF_KEY_DESCRIPTION    "general.description"
#define GGUF_KEY_QUANT_VERSION  "general.quantization_version"
#define GGUF_KEY_ALIGNMENT      "general.alignment"     /* file alignment bytes */

/* ── GPT-2 metadata keys ──────────────────────────────────────────────────── */
#define GGUF_KEY_GPT2_LAYERS    "gpt2.block_count"
#define GGUF_KEY_GPT2_HEADS     "gpt2.attention.head_count"
#define GGUF_KEY_GPT2_EMBED     "gpt2.embedding_length"
#define GGUF_KEY_GPT2_FFN       "gpt2.feed_forward_length"
#define GGUF_KEY_GPT2_CTX       "gpt2.context_length"
#define GGUF_KEY_GPT2_EPS       "gpt2.attention.layer_norm_epsilon"

/* ── LLaMA / Mistral / Qwen metadata keys ────────────────────────────────── *
 * Also used by models that set general.architecture = "mistral" or "qwen2"; *
 * those architectures are handled as ARCH_LLAMA internally.                  */
#define GGUF_KEY_LLAMA_LAYERS   "llama.block_count"
#define GGUF_KEY_LLAMA_HEADS    "llama.attention.head_count"
#define GGUF_KEY_LLAMA_KV_HEADS "llama.attention.head_count_kv"
#define GGUF_KEY_LLAMA_EMBED    "llama.embedding_length"
#define GGUF_KEY_LLAMA_FFN      "llama.feed_forward_length"
#define GGUF_KEY_LLAMA_CTX      "llama.context_length"
#define GGUF_KEY_LLAMA_ROPE     "llama.rope.freq_base"
#define GGUF_KEY_LLAMA_ROPE_DIM "llama.rope.dimension_count"
#define GGUF_KEY_LLAMA_NORM_EPS "llama.attention.layer_norm_rms_epsilon"

/* Tokenizer keys (both architectures share these) */
#define GGUF_KEY_TOK_MODEL      "tokenizer.ggml.model"   /* "gpt2" | "llama" */
#define GGUF_KEY_TOK_VOCAB      "tokenizer.ggml.tokens"
#define GGUF_KEY_TOK_SCORES     "tokenizer.ggml.scores"
#define GGUF_KEY_TOK_MERGES     "tokenizer.ggml.merges"
#define GGUF_KEY_TOK_BOS        "tokenizer.ggml.bos_token_id"
#define GGUF_KEY_TOK_EOS        "tokenizer.ggml.eos_token_id"
#define GGUF_KEY_TOK_VOCAB_SIZE "tokenizer.ggml.vocab_size"

/* ── Tensor descriptor ────────────────────────────────────────────────────── */
#define GGUF_MAX_DIMS      4
#define GGUF_MAX_NAME    256

typedef struct {
    char     name[GGUF_MAX_NAME]; /* null-terminated tensor name              */
    uint32_t type;                /* GGUF_TYPE_* quantisation type            */
    uint32_t n_dims;              /* number of dimensions in use              */
    uint64_t dims[GGUF_MAX_DIMS]; /* shape: dims[0] is innermost (columns)   */
    uint64_t offset;              /* byte offset from start of tensor data    */
    size_t   n_elements;          /* total element count (product of dims)    */
} GGUFTensor;

/* Expected tensor counts per architecture.
 * Used as a sanity check after loading; mismatches warn but don't abort.   */
#define GGUF_GPT2_TENSORS_PER_LAYER   12  /* ln1×2, qkv×2, proj×2, ln2×2, fc×2, proj×2 */
#define GGUF_GPT2_TENSORS_GLOBAL       5  /* wte, wpe, ln_f×2, lm_head                  */
#define GGUF_LLAMA_TENSORS_PER_LAYER   9  /* rms_attn, q, k, v, o, rms_ffn, gate, up, down */
#define GGUF_LLAMA_TENSORS_GLOBAL      3  /* token_embd, output_norm, output             */

/* Backward-compatible aliases (used in gguf.c before the rename). */
#define GGUF_TENSORS_PER_LAYER  GGUF_GPT2_TENSORS_PER_LAYER
#define GGUF_TENSORS_GLOBAL     GGUF_GPT2_TENSORS_GLOBAL

/* ── Public API ───────────────────────────────────────────────────────────── */

typedef enum { FORMAT_UNKNOWN, FORMAT_BIN, FORMAT_GGUF } ModelFormat;

/* Inspect the first 4 bytes to identify the file type without fully parsing.*/
ModelFormat detect_format(const char *path);

/* Scan the models/ directory for the first recognisable model file.
 * Returns a static string (do not free); returns NULL if none found.       */
const char *find_default_model(void);

/* Main entry point: detect format, parse the file, populate g_cfg,
 * g_weights, g_kv_cache, and g_act.  Calls LMC_FATAL on hard errors.      */
void load_model(const char *path);

#endif /* LMC_GGUF_H */
