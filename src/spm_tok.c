/* =============================================================================
 * spm_tok.c  —  SentencePiece BPE tokenizer for TinyLlama
 *
 * Algorithm (BPE with SPM conventions):
 *   1. Split text on whitespace; prepend ▁ to each word-piece.
 *   2. Represent each piece as a linked list of symbol nodes.
 *   3. Build a priority queue of symbol bigrams scored by vocab score.
 *   4. Greedily merge highest-scoring bigrams until no more merges exist.
 *   5. Output the resulting token IDs.
 *
 * For byte-fallback: unknown characters are encoded as <0xNN> byte tokens.
 *
 * C99 — no external libs
 * =============================================================================
 */

#include "spm_tok.h"
#include "gguf.h"     /* gguf_get_meta_str_arr(), gguf_get_meta_float_arr() */
#include "utils.h"    /* arena_alloc(), lmc_log()                           */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Internal symbol list (doubly-linked list of pieces)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define SPM_MAX_INPUT 8192    /* maximum input string bytes                  */

typedef struct Symbol {
    int           id;         /* vocab id for this piece (-1 = unknown)     */
    char          text[SPM_MAX_TOKEN_LEN];
    int           len;        /* byte length of text[]                       */
    struct Symbol *prev;
    struct Symbol *next;
} Symbol;

/* ─── Bigram priority-queue node (min-heap on -score → max score first) ─*/
typedef struct {
    Symbol  *left;
    float    score;
    int      seq;   /* tie-breaker: earlier position wins                    */
} Bigram;

typedef struct {
    Bigram *heap;
    int     size;
    int     cap;
} BigramHeap;

static void heap_push(BigramHeap *h, Bigram b) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 64;
        h->heap = (Bigram *)realloc(h->heap, (size_t)h->cap * sizeof(Bigram));
    }
    /* append + sift-up (max-heap on score) */
    int i = h->size++;
    h->heap[i] = b;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->heap[parent].score >= h->heap[i].score) break;
        Bigram tmp = h->heap[parent];
        h->heap[parent] = h->heap[i];
        h->heap[i] = tmp;
        i = parent;
    }
}

static Bigram heap_pop(BigramHeap *h) {
    Bigram top = h->heap[0];
    h->heap[0] = h->heap[--h->size];
    /* sift-down */
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, best = i;
        if (l < h->size && h->heap[l].score > h->heap[best].score) best = l;
        if (r < h->size && h->heap[r].score > h->heap[best].score) best = r;
        if (best == i) break;
        Bigram tmp = h->heap[best]; h->heap[best] = h->heap[i]; h->heap[i] = tmp;
        i = best;
    }
    return top;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Vocab string → token ID lookup  (linear scan; good enough for 32k)
 * ═══════════════════════════════════════════════════════════════════════════ */
static int vocab_lookup(const SpmCtx *spm, const char *text) {
    for (int i = 0; i < spm->n_vocab; i++) {
        if (strcmp(spm->vocab[i].text, text) == 0) return i;
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3  UTF-8 helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Returns byte-length of the next UTF-8 character starting at *s.          */
static int utf8_char_len(const unsigned char *s) {
    if      (*s < 0x80) return 1;
    else if (*s < 0xE0) return 2;
    else if (*s < 0xF0) return 3;
    else                return 4;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §4  Try to merge two adjacent symbols into one
 * ═══════════════════════════════════════════════════════════════════════════ */
static void try_add_bigram(BigramHeap *heap, const SpmCtx *spm,
                           Symbol *left, Symbol *right, int seq)
{
    if (!left || !right) return;

    /* Build the merged piece string */
    char merged[SPM_MAX_TOKEN_LEN * 2];
    int  total = left->len + right->len;
    if (total >= (int)sizeof(merged) - 1) return;
    memcpy(merged,              left->text,  (size_t)left->len);
    memcpy(merged + left->len,  right->text, (size_t)right->len);
    merged[total] = '\0';

    int id = vocab_lookup(spm, merged);
    if (id < 0) return;   /* pair not in vocab, cannot merge */

    Bigram b;
    b.left  = left;
    b.score = spm->vocab[id].score;
    b.seq   = seq;
    heap_push(heap, b);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5  Core BPE encode
 * ═══════════════════════════════════════════════════════════════════════════ */
static int encode_piece(const SpmCtx *spm,
                        const char   *piece,   /* may start with ▁          */
                        int          *out,
                        int           max,
                        int          *n_out)
{
    /* Build initial symbol list: one node per UTF-8 character              */
    Symbol nodes[SPM_MAX_INPUT];
    int    n_nodes = 0;

    const unsigned char *p = (const unsigned char *)piece;
    int seq = 0;
    while (*p) {
        int cl = utf8_char_len(p);
        Symbol *s = &nodes[n_nodes];
        if (cl >= SPM_MAX_TOKEN_LEN) cl = SPM_MAX_TOKEN_LEN - 1;
        memcpy(s->text, p, (size_t)cl);
        s->text[cl] = '\0';
        s->len  = cl;
        s->id   = vocab_lookup(spm, s->text);
        s->prev = n_nodes > 0 ? &nodes[n_nodes-1] : NULL;
        s->next = NULL;
        if (n_nodes > 0) nodes[n_nodes-1].next = s;
        n_nodes++;
        p += cl;
        seq++;
    }

    if (n_nodes == 0) return 0;

    /* Build initial bigram heap                                             */
    BigramHeap heap = {NULL, 0, 0};
    for (int i = 0; i < n_nodes - 1; i++)
        try_add_bigram(&heap, spm, &nodes[i], nodes[i].next, i);

    /* Greedy BPE merges                                                     */
    while (heap.size > 0) {
        Bigram b = heap_pop(&heap);
        Symbol *left  = b.left;
        Symbol *right = left->next;

        /* Validate: symbols may have been merged already (check lengths)   */
        if (!right) continue;
        char merged[SPM_MAX_TOKEN_LEN * 2];
        int  total = left->len + right->len;
        if (total >= (int)sizeof(merged) - 1) continue;
        memcpy(merged,             left->text,  (size_t)left->len);
        memcpy(merged + left->len, right->text, (size_t)right->len);
        merged[total] = '\0';

        int id = vocab_lookup(spm, merged);
        if (id < 0) continue;
        if (spm->vocab[id].score != b.score) continue; /* stale entry      */

        /* Merge right into left                                             */
        memcpy(left->text, merged, (size_t)total + 1);
        left->len = total;
        left->id  = id;
        left->next = right->next;
        if (right->next) right->next->prev = left;

        /* Add new candidate bigrams around the merged symbol               */
        try_add_bigram(&heap, spm, left->prev, left, b.seq - 1);
        try_add_bigram(&heap, spm, left, left->next,  b.seq);
    }

    free(heap.heap);

    /* Emit tokens; apply byte-fallback for unknown pieces                  */
    for (Symbol *s = &nodes[0]; s; s = s->next) {
        if (*n_out >= max) return -1;   /* overflow                         */
        if (s->id >= 0) {
            out[(*n_out)++] = s->id;
        } else {
            /* Byte-fallback: encode each byte as <0xNN>                    */
            const unsigned char *bp = (const unsigned char *)s->text;
            for (int i = 0; i < s->len; i++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", bp[i]);
                int bid = vocab_lookup(spm, byte_tok);
                if (bid >= 0 && *n_out < max) out[(*n_out)++] = bid;
                else if (*n_out >= max)       return -1;
                else                          out[(*n_out)++] = spm->unk_id;
            }
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §6  Public: spm_encode
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ▁ in UTF-8 */
static const char WORD_BOUNDARY[] = "\xe2\x96\x81";

int spm_encode(const SpmCtx *spm,
               const char   *text,
               int           add_bos,
               int          *out_ids,
               int           max_tokens)
{
    int n = 0;
    if (add_bos) {
        if (n >= max_tokens) return -1;
        out_ids[n++] = spm->bos_id;
    }

    /* Walk the input and emit one piece per "word" (split on space/NL/TAB) */
    const char *p = text;
    char piece[SPM_MAX_INPUT];

    while (*p) {
        /* Skip leading whitespace; start a new word with ▁ prefix         */
        int is_first_word = (p == text);
        /* collect a word */
        const char *word_start = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
        int wlen = (int)(p - word_start);

        if (wlen > 0) {
            int prefix_len = 3;   /* ▁ is 3 bytes in UTF-8                  */
            int total = prefix_len + wlen;
            if (total >= (int)sizeof(piece) - 1) total = (int)sizeof(piece) - 2;
            memcpy(piece, WORD_BOUNDARY, (size_t)prefix_len);
            memcpy(piece + prefix_len, word_start,
                   (size_t)(wlen < total - prefix_len ? wlen : total - prefix_len));
            piece[total] = '\0';

            if (encode_piece(spm, piece, out_ids, max_tokens, &n) != 0)
                return -1;
        }

        /* skip whitespace (spaces become part of the next ▁ prefix)       */
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        (void)is_first_word;
    }

    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §7  spm_decode_token
 * ═══════════════════════════════════════════════════════════════════════════ */
const char *spm_decode_token(const SpmCtx *spm, int token_id) {
    if (token_id < 0 || token_id >= spm->n_vocab) return "<unk>";
    const char *t = spm->vocab[token_id].text;
    /* Replace ▁ with a space for display                                    */
    return t;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §8  spm_init_from_gguf  — reads vocab from GGUF metadata
 * ═══════════════════════════════════════════════════════════════════════════ */
int spm_init_from_gguf(SpmCtx *spm, GgufCtx *ctx)
{
    memset(spm, 0, sizeof(*spm));

    /* Read vocab size from metadata, fall back to counting tokens           */
    int64_t n_vocab = 0;
    if (!gguf_get_meta_int(ctx, "llama.vocab_size", &n_vocab)) {
        /* count strings in tokenizer.ggml.tokens array                     */
        n_vocab = gguf_get_meta_str_arr_count(ctx, "tokenizer.ggml.tokens");
    }
    if (n_vocab <= 0) {
        fprintf(stderr, "[spm_tok] cannot determine vocab size\n");
        return -1;
    }

    spm->n_vocab = (int)n_vocab;
    spm->vocab   = (SpmToken *)calloc((size_t)n_vocab, sizeof(SpmToken));
    if (!spm->vocab) {
        fprintf(stderr, "[spm_tok] OOM allocating vocab\n");
        return -1;
    }

    /* Populate text strings                                                 */
    for (int i = 0; i < spm->n_vocab; i++) {
        const char *s = gguf_get_meta_str_arr(ctx, "tokenizer.ggml.tokens", i);
        if (s) {
            strncpy(spm->vocab[i].text, s, SPM_MAX_TOKEN_LEN - 1);
        } else {
            snprintf(spm->vocab[i].text, SPM_MAX_TOKEN_LEN, "<tok_%d>", i);
        }
    }

    /* Populate scores                                                       */
    for (int i = 0; i < spm->n_vocab; i++) {
        float score = gguf_get_meta_float_arr(ctx, "tokenizer.ggml.scores", i);
        spm->vocab[i].score = score;
    }

    /* Populate types                                                        */
    for (int i = 0; i < spm->n_vocab; i++) {
        int t = (int)gguf_get_meta_int_arr(ctx, "tokenizer.ggml.token_type", i);
        spm->vocab[i].type = t ? t : SPM_TOK_NORMAL;
    }

    int64_t bos = 1, eos = 2;
    gguf_get_meta_int(ctx, "tokenizer.ggml.bos_token_id", &bos);
    gguf_get_meta_int(ctx, "tokenizer.ggml.eos_token_id", &eos);
    spm->bos_id = (int)bos;
    spm->eos_id = (int)eos;
    spm->unk_id = 0;

    lmc_log("[spm_tok] loaded %d tokens  bos=%d  eos=%d\n",
            spm->n_vocab, spm->bos_id, spm->eos_id);
    return 0;
}

void spm_free(SpmCtx *spm) {
    free(spm->vocab);
    spm->vocab = NULL;
}
