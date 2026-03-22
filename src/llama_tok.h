/* llama_tok.h — SentencePiece BPE tokenizer for LLaMA-family models.
 *
 * The 32 000-entry vocabulary is embedded in the GGUF file itself.
 * gguf.c populates g_llama_vocab[] during load_model_gguf(); call
 * llama_tok_init() afterwards to build the hash-table lookup.
 *
 * C99 — no external libs.
 */
#ifndef LLAMA_TOK_H
#define LLAMA_TOK_H

/* Maximum vocab entries (32 768 > TinyLlama's 32 000) */
#define LLAMA_TOK_MAX_VOCAB  32768
/* Maximum UTF-8 bytes in one piece, including NUL */
#define LLAMA_TOK_MAX_PIECE  64

/* Token type flags (mirrors tokenizer.ggml.token_type in GGUF) */
#define LLAMA_TOK_NORMAL   1
#define LLAMA_TOK_UNKNOWN  2
#define LLAMA_TOK_CONTROL  3
#define LLAMA_TOK_BYTE     6

typedef struct {
    char  text[LLAMA_TOK_MAX_PIECE];
    float score;   /* log-prob: higher = merge earlier */
    int   type;    /* LLAMA_TOK_* flags */
} LlamaVocabEntry;

/* Globals written by gguf.c during load_model_gguf() */
extern LlamaVocabEntry g_llama_vocab[LLAMA_TOK_MAX_VOCAB];
extern int             g_llama_vocab_n;
extern int             g_llama_bos_id;
extern int             g_llama_eos_id;

/* Build hash lookup after vocab is populated. Returns 0 OK, -1 empty. */
int llama_tok_init(void);

/* Tokenize text → ids. Returns token count, -1 on overflow. */
int llama_tok_encode(const char *text, int add_bos,
                     int *out_ids, int max_ids);

/* Raw piece string for token id (▁ is returned verbatim). */
const char *llama_tok_piece(int id);

#endif /* LLAMA_TOK_H */