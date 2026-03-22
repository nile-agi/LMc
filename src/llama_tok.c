/* llama_tok.c — SentencePiece BPE tokenizer for LLaMA / TinyLlama.
 *
 * Algorithm:
 *   1. Split text on whitespace; prepend U+2581 (▁) to each word.
 *   2. Represent each word as a linked symbol list (one node per UTF-8 char).
 *   3. Build a max-heap of candidate bigrams keyed on vocab score.
 *   4. Greedily merge highest-score pair until none remain.
 *   5. Emit token IDs; byte-fallback <0xNN> for any unknown character.
 *
 * Vocab lookup: FNV-1a open-addressing hash table — O(1) average.
 *
 * C99 — no external libs.
 */
#include "llama_tok.h"
#include "utils.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* ── Globals (written by gguf.c) ─────────────────────────────────────────── */
LlamaVocabEntry g_llama_vocab[LLAMA_TOK_MAX_VOCAB];
int             g_llama_vocab_n = 0;
int             g_llama_bos_id  = 1;
int             g_llama_eos_id  = 2;

/* ── Hash table (built by llama_tok_init) ────────────────────────────────── */
#define HT_SIZE 65536   /* must be power-of-2, > 2 × MAX_VOCAB */
static int g_ht[HT_SIZE]; /* -1 = empty slot; else vocab index */

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)(*s++); h *= 16777619u; }
    return h;
}

static int ht_find(const char *text) {
    uint32_t h = fnv1a(text) & (HT_SIZE - 1);
    int probes = 0;
    while (g_ht[h] != -1 && probes < HT_SIZE) {
        if (strcmp(g_llama_vocab[g_ht[h]].text, text) == 0)
            return g_ht[h];
        h = (h + 1) & (HT_SIZE - 1);
        probes++;
    }
    return -1;
}

int llama_tok_init(void) {
    if (g_llama_vocab_n <= 0) return -1;
    memset(g_ht, -1, sizeof(g_ht));
    for (int i = 0; i < g_llama_vocab_n; i++) {
        uint32_t h = fnv1a(g_llama_vocab[i].text) & (HT_SIZE - 1);
        int probes = 0;
        while (g_ht[h] != -1 && probes < HT_SIZE) {
            h = (h + 1) & (HT_SIZE - 1);
            probes++;
        }
        if (probes < HT_SIZE) g_ht[h] = i;
    }
    LMC_INFO("[llama_tok] vocab=%d  bos=%d  eos=%d",
            g_llama_vocab_n, g_llama_bos_id, g_llama_eos_id);
    return 0;
}

/* ── UTF-8 helpers ───────────────────────────────────────────────────────── */
static int utf8_len(const unsigned char *s) {
    if      (*s < 0x80) return 1;
    else if (*s < 0xE0) return 2;
    else if (*s < 0xF0) return 3;
    else                return 4;
}

/* ── Symbol linked-list (one per word being tokenised) ───────────────────── */
#define MAX_WORD_SYMS 512   /* max symbols in one word (≥ max UTF-8 chars)   */

typedef struct Sym {
    char        text[LLAMA_TOK_MAX_PIECE];
    int         len;   /* byte length of text (excluding NUL) */
    int         id;    /* vocab id, or -1 if unknown */
    struct Sym *prev;
    struct Sym *next;
} Sym;

/* ── Bigram max-heap (keyed on score) ────────────────────────────────────── */
typedef struct {
    Sym   *left;
    float  score;
} Bigram;

typedef struct {
    Bigram *data;
    int     n;
    int     cap;
} Heap;

static void heap_push(Heap *h, Bigram b) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 64;
        h->data = (Bigram *)realloc(h->data, (size_t)h->cap * sizeof(Bigram));
    }
    int i = h->n++;
    h->data[i] = b;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].score >= h->data[i].score) break;
        Bigram tmp = h->data[p]; h->data[p] = h->data[i]; h->data[i] = tmp;
        i = p;
    }
}

static Bigram heap_pop(Heap *h) {
    Bigram top = h->data[0];
    h->data[0] = h->data[--h->n];
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, best = i;
        if (l < h->n && h->data[l].score > h->data[best].score) best = l;
        if (r < h->n && h->data[r].score > h->data[best].score) best = r;
        if (best == i) break;
        Bigram tmp = h->data[best]; h->data[best] = h->data[i]; h->data[i] = tmp;
        i = best;
    }
    return top;
}

/* ── Try to add bigram (left, right) to heap ─────────────────────────────── */
static void try_bigram(Heap *h, Sym *left) {
    Sym *right = left->next;
    if (!right) return;
    char merged[LLAMA_TOK_MAX_PIECE * 2];
    int  total = left->len + right->len;
    if (total >= (int)sizeof(merged) - 1) return;
    memcpy(merged,            left->text,  (size_t)left->len);
    memcpy(merged + left->len, right->text, (size_t)right->len);
    merged[total] = '\0';
    int id = ht_find(merged);
    if (id < 0) return;
    Bigram b;
    b.left  = left;
    b.score = g_llama_vocab[id].score;
    heap_push(h, b);
}

/* ── Tokenise one piece (with leading ▁ already prepended) ─────────────── */
static int tokenise_piece(const char *piece, int *out, int max, int *n) {
    /* Build symbol list: one node per UTF-8 character */
    static Sym nodes[MAX_WORD_SYMS];
    int        n_nodes = 0;
    const unsigned char *p = (const unsigned char *)piece;

    while (*p && n_nodes < MAX_WORD_SYMS) {
        int cl = utf8_len(p);
        if (cl > (int)sizeof(nodes[0].text) - 1) cl = (int)sizeof(nodes[0].text) - 1;
        Sym *s = &nodes[n_nodes];
        memcpy(s->text, p, (size_t)cl);
        s->text[cl] = '\0';
        s->len  = cl;
        s->id   = ht_find(s->text);
        s->prev = n_nodes > 0 ? &nodes[n_nodes - 1] : NULL;
        s->next = NULL;
        if (n_nodes > 0) nodes[n_nodes - 1].next = s;
        n_nodes++;
        p += cl;
    }
    if (n_nodes == 0) return 0;

    /* Build initial bigram heap */
    Heap heap = {NULL, 0, 0};
    for (int i = 0; i < n_nodes - 1; i++)
        try_bigram(&heap, &nodes[i]);

    /* Greedy BPE merges */
    while (heap.n > 0) {
        Bigram b  = heap_pop(&heap);
        Sym   *left  = b.left;
        Sym   *right = left->next;
        if (!right) continue;

        /* Validate the pair still represents the scored bigram */
        char merged[LLAMA_TOK_MAX_PIECE * 2];
        int  total = left->len + right->len;
        if (total >= (int)sizeof(merged) - 1) continue;
        memcpy(merged,            left->text,  (size_t)left->len);
        memcpy(merged + left->len, right->text, (size_t)right->len);
        merged[total] = '\0';
        int id = ht_find(merged);
        if (id < 0) continue;
        if (g_llama_vocab[id].score != b.score) continue; /* stale entry */

        /* Merge right into left */
        memcpy(left->text, merged, (size_t)(total + 1));
        left->len  = total;
        left->id   = id;
        left->next = right->next;
        if (right->next) right->next->prev = left;

        /* Add new candidate bigrams around the merged symbol */
        if (left->prev) try_bigram(&heap, left->prev);
        try_bigram(&heap, left);
    }
    free(heap.data);

    /* Emit tokens; byte-fallback for any remaining unknown piece */
    for (Sym *s = &nodes[0]; s; s = s->next) {
        if (*n >= max) return -1; /* overflow */
        if (s->id >= 0) {
            out[(*n)++] = s->id;
        } else {
            /* Encode each byte as <0xNN> */
            const unsigned char *bp = (const unsigned char *)s->text;
            for (int i = 0; i < s->len; i++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", bp[i]);
                int bid = ht_find(byte_tok);
                if (*n >= max) return -1;
                out[(*n)++] = (bid >= 0) ? bid : 0; /* 0 = UNK fallback */
            }
        }
    }
    return 0;
}

/* ── Public: llama_tok_encode ────────────────────────────────────────────── */
/* UTF-8 word-boundary marker: ▁ = U+2581 = 0xE2 0x96 0x81 */
static const char WORD_SEP[] = "\xe2\x96\x81";

int llama_tok_encode(const char *text, int add_bos,
                     int *out_ids, int max_ids)
{
    if (g_llama_vocab_n <= 0) return -1;
    int n = 0;
    if (add_bos) {
        if (n >= max_ids) return -1;
        out_ids[n++] = g_llama_bos_id;
    }

    const char *p = text;
    char piece[LLAMA_TOK_MAX_PIECE + 4]; /* +4 for ▁ prefix (3 bytes) + NUL */

    while (*p) {
        /* collect one whitespace-delimited word */
        const char *ws = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
        int wlen = (int)(p - ws);

        if (wlen > 0) {
            /* prepend ▁ */
            int plen = 3 + wlen;
            if (plen >= (int)sizeof(piece)) plen = (int)sizeof(piece) - 1;
            memcpy(piece, WORD_SEP, 3);
            memcpy(piece + 3, ws, (size_t)(plen - 3));
            piece[plen] = '\0';
            if (tokenise_piece(piece, out_ids, max_ids, &n) != 0) return -1;
        }

        /* skip whitespace */
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    }
    return n;
}

/* ── Public: llama_tok_piece ─────────────────────────────────────────────── */
const char *llama_tok_piece(int id) {
    if (id < 0 || id >= g_llama_vocab_n) return "<unk>";
    return g_llama_vocab[id].text;
}