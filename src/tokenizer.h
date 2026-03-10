/* tokenizer.h — Text tokenisation and detokenisation.
 *
 * Currently implements GPT-2 Byte-Pair Encoding (BPE) loaded from the
 * standard OpenAI encoder.json + vocab.bpe file pair.
 *
 * Architecture notes
 * ──────────────────
 *  • Vocabulary is stored as a flat array of VocabEntry (byte sequences).
 *  • A hash map (open addressing, power-of-two size) accelerates lookup by
 *    byte sequence → token ID; used during BPE merge candidate search.
 *  • Merge rules are loaded in priority order (first rule = highest priority)
 *    and applied with a priority-queue-style O(n log n) BPE encoder.
 *  • byte_encoder[] maps raw bytes 0–255 to their GPT-2 Unicode string
 *    representation (e.g. byte 0x20 → "Ġ"); byte_decoder[] is the reverse.
 *
 * Adding a new tokenizer type (e.g. SentencePiece for LLaMA)
 * ──────────────────────────────────────────────────────────
 *  1. Add TOKENIZER_* constant to TokenizerType enum.
 *  2. Implement load_spm_tokenizer() and the encode/decode variants.
 *  3. Dispatch on g_tokenizer.type in tokenize() / detokenize_token().
 *  The struct has space for a vocab_size field that both types share.
 */
#ifndef LMC_TOKENIZER_H
#define LMC_TOKENIZER_H

#include <stdint.h>

/* ── Tokenizer type ───────────────────────────────────────────────────────── */
typedef enum {
    TOKENIZER_UNKNOWN = 0,
    TOKENIZER_GPT2_BPE,    /* OpenAI BPE: encoder.json + vocab.bpe            */
    TOKENIZER_SPM,         /* SentencePiece unigram / BPE (LLaMA, LLaMA-2)   */
    TOKENIZER_TIKTOKEN,    /* tiktoken cl100k_base / o200k_base (GPT-4 etc.)  */
} TokenizerType;

/* ── Capacity constants ───────────────────────────────────────────────────── */
#define BPE_MAX_VOCAB       50257    /* GPT-2 vocabulary size                  */
#define BPE_MAX_MERGES      50000    /* GPT-2 merge rule count                 */
#define BPE_TOKEN_MAX_LEN     256    /* max byte length of a single token      */
#define UNICODE_BYTE_RANGE    256    /* number of distinct raw byte values     */

/* Hash table parameters (open addressing, linear probing).
 * VOCAB_HASH_SIZE must be a power of two ≥ 2 × BPE_MAX_VOCAB.              */
#define VOCAB_HASH_SIZE   131072     /* 2^17; load factor ≤ 0.39 for GPT-2   */

/* ── Data types ───────────────────────────────────────────────────────────── */

/* A single BPE merge rule: token[left] ⊕ token[right] → token[result].    */
typedef struct {
    int left;
    int right;
    int result;
} BPEMerge;

/* A token's canonical byte representation (may include null bytes).        */
typedef struct {
    uint8_t bytes[BPE_TOKEN_MAX_LEN];
    int     len;
} VocabEntry;

/* ── Tokenizer state ──────────────────────────────────────────────────────── *
 * A single global instance (g_tokenizer) is populated by load_tokenizer(). *
 * All fields after 'type' are initialised to zero/empty by load_tokenizer; *
 * fields irrelevant to the active tokenizer type are left zero.             */
typedef struct {
    TokenizerType type;

    /* --- Vocabulary --- */
    VocabEntry vocab[BPE_MAX_VOCAB];
    int        vocab_size;          /* actual number of tokens loaded          */

    /* --- Merge rules (GPT-2 BPE) --- */
    BPEMerge   merges[BPE_MAX_MERGES];
    int        n_merges;

    /* --- Byte ↔ Unicode mapping (GPT-2 BPE) ---
     * byte_encoder[b] = the GPT-2 Unicode representation of raw byte b
     *   (UTF-8 string, NUL-terminated, length ≤ 4 bytes for U+0000..U+00FF)
     * byte_decoder[codepoint] = raw byte value (-1 if not a byte-level token) */
    char       byte_encoder[UNICODE_BYTE_RANGE][8];
    int        byte_decoder[0x400];     /* covers U+0000..U+03FF              */

    /* --- Vocabulary hash map (token bytes → token ID) ---
     * Open-addressing hash: vocab_hash[slot] = token_id (−1 = empty).
     * Collision chain: vocab_hash_next[token_id] = next token_id in chain.  */
    int        vocab_hash     [VOCAB_HASH_SIZE];
    int        vocab_hash_next[BPE_MAX_VOCAB];

    /* --- Special tokens --- */
    int        bos_id;          /* beginning-of-sequence token ID (or -1)     */
    int        eos_id;          /* end-of-sequence token ID (or -1)            */
    int        pad_id;          /* padding token ID (or -1)                    */
} Tokenizer;

extern Tokenizer g_tokenizer;

/* ── API ──────────────────────────────────────────────────────────────────── */

/* Load a GPT-2 BPE tokenizer from encoder.json and vocab.bpe files.
 * Sets g_tokenizer.type = TOKENIZER_GPT2_BPE.
 * Calls LMC_FATAL on any parse error.                                       */
void load_tokenizer(const char *encoder_path, const char *bpe_path);

/* Encode a UTF-8 string into token IDs using the active tokenizer.
 * out_ids  — caller-allocated array of at least max_tokens ints.
 * Returns  — the number of tokens written (≤ max_tokens).
 * Excess tokens beyond max_tokens are silently dropped.                     */
int tokenize(const Tokenizer *tok, const char *text,
             int *out_ids, int max_tokens);

/* Decode a single token ID to its UTF-8 byte string.
 * out_buf  — caller-allocated buffer of at least buf_size bytes.
 * Returns  — number of bytes written (not including NUL terminator);
 *            0 if the token ID is out of range.                             */
int detokenize_token(const Tokenizer *tok, int token_id,
                     char *out_buf, int buf_size);

#endif /* LMC_TOKENIZER_H */
