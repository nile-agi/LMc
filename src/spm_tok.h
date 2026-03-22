/* =============================================================================
 * spm_tok.h  —  SentencePiece BPE tokenizer for TinyLlama / Llama-family
 *
 * The vocabulary (32 000 tokens) is embedded in the GGUF file itself under
 * metadata keys:
 *   tokenizer.ggml.model   = "llama"    (BPE / SPM)
 *   tokenizer.ggml.tokens  = string[]   (vocab pieces)
 *   tokenizer.ggml.scores  = float[]    (log-prob merge scores)
 *   tokenizer.ggml.token_type = int[]   (NORMAL=1, UNKNOWN=2, CONTROL=3,
 *                                        UNUSED=4, BYTE=6)
 *
 * BOS = 1,  EOS = 2,  UNK = 0
 * Word-boundary marker: ▁  (U+2581, UTF-8: 0xE2 0x96 0x81)
 *
 * C99 — no external libs
 * =============================================================================
 */
#ifndef SPM_TOK_H
#define SPM_TOK_H

#include <stdint.h>

/* forward-declare to avoid circular include                                 */
#ifndef GGUF_CTX_DECLARED
#define GGUF_CTX_DECLARED
struct GgufCtx;
typedef struct GgufCtx GgufCtx;
#endif

/* ─── Token-type flags (mirrors GGUF tokenizer.ggml.token_type) ──────────*/
#define SPM_TOK_NORMAL    1
#define SPM_TOK_UNKNOWN   2
#define SPM_TOK_CONTROL   3
#define SPM_TOK_UNUSED    4
#define SPM_TOK_BYTE      6

/* ─── Maximum token-string length (bytes, including NUL) ─────────────────*/
#define SPM_MAX_TOKEN_LEN 64

/* ─── Vocabulary entry ───────────────────────────────────────────────────*/
typedef struct {
    char    text[SPM_MAX_TOKEN_LEN];   /* piece string (UTF-8)              */
    float   score;                     /* log-prob; higher = merge earlier  */
    int     type;                      /* SPM_TOK_* above                   */
} SpmToken;

/* ─── Tokenizer context ──────────────────────────────────────────────────*/
typedef struct {
    SpmToken *vocab;         /* array[n_vocab]                              */
    int       n_vocab;
    int       bos_id;        /* 1                                           */
    int       eos_id;        /* 2                                           */
    int       unk_id;        /* 0                                           */

    /* internal: sorted merge-pair lookup built during init                 */
    void     *_lookup;       /* opaque; freed by spm_free()                */
} SpmCtx;

/* ── API ─────────────────────────────────────────────────────────────────*/

/**
 * spm_init_from_gguf  — read vocab from an open GGUF context.
 * Returns 0 on success, -1 on failure.
 */
int  spm_init_from_gguf(SpmCtx *spm, GgufCtx *ctx);

/**
 * spm_encode  — tokenise a UTF-8 string into token IDs.
 *
 * @param spm         tokeniser context
 * @param text        NUL-terminated input string
 * @param add_bos     prepend BOS token (1) if non-zero
 * @param out_ids     caller-allocated output array
 * @param max_tokens  capacity of out_ids
 * @returns number of tokens written, or -1 on overflow
 */
int  spm_encode(const SpmCtx *spm,
                const char   *text,
                int           add_bos,
                int          *out_ids,
                int           max_tokens);

/**
 * spm_decode_token  — return the piece string for a token ID.
 *                     Returns "<unk>" for out-of-range IDs.
 */
const char *spm_decode_token(const SpmCtx *spm, int token_id);

/**
 * spm_free  — release lookup table (vocab memory is from GGUF arena).
 */
void spm_free(SpmCtx *spm);

#endif /* SPM_TOK_H */
