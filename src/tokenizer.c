/*
 * src/tokenizer.c — LMc GPT-2 BPE Tokenizer
 *
 * Implements the GPT-2 Byte-Pair Encoding (BPE) tokenizer:
 *   1. Byte encoding: maps raw bytes to Unicode codepoints so that
 *      all 256 byte values have valid string representations.
 *   2. Vocabulary: loaded from encoder.json (token → id mapping).
 *   3. Merge rules: loaded from vocab.bpe (BPE merge priority list).
 *   4. Encoding: splits text into pre-tokenization chunks,
 *      applies byte encoding, then applies BPE merges.
 *   5. Decoding: converts token ids back to UTF-8 bytes.
 *
 * The tokenizer is fully self-contained and requires no external
 * libraries. encoder.json and vocab.bpe must be downloaded from
 * HuggingFace or the OpenAI GPT-2 repository.
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * BYTE ENCODER INITIALIZATION
 *
 * GPT-2 defines a mapping from raw byte values (0-255) to
 * Unicode codepoints such that all 256 values have printable
 * representations. This avoids control characters in the vocab.
 *
 * Printable ASCII ranges (33-126, 161-172, 174-255) map to
 * themselves. The remaining 256 - 188 = 68 bytes map to
 * codepoints starting at 256.
 * ============================================================ */
LmcError lmc_tokenizer_init_byte_encoder(LmcTokenizer *tok) {
    int bs[256], cs[256], n_bs = 0;

    /* Printable ASCII and Latin-1 supplement */
    for (int b = 33;  b <= 126; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
    for (int b = 161; b <= 172; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
    for (int b = 174; b <= 255; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }

    /* Remaining bytes → codepoints starting at 256 */
    int extra = 256;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < n_bs; i++) {
            if (bs[i] == b) { found = 1; break; }
        }
        if (!found) { bs[n_bs] = b; cs[n_bs] = extra++; n_bs++; }
    }

    /* Build byte_decoder: codepoint → raw byte */
    for (int i = 0; i < 256; i++) tok->byte_decoder[cs[i]] = bs[i];

    /* Build byte_encoder: raw byte → UTF-8 string representation */
    for (int i = 0; i < 256; i++) {
        int cp  = cs[i];
        char *o = tok->byte_encoder[bs[i]];

        if (cp < 0x80) {
            o[0] = (char)cp; o[1] = '\0';
        } else if (cp < 0x800) {
            o[0] = (char)(0xC0 | (cp >> 6));
            o[1] = (char)(0x80 | (cp & 0x3F));
            o[2] = '\0';
        } else {
            o[0] = (char)(0xE0 | (cp >> 12));
            o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            o[2] = (char)(0x80 | (cp & 0x3F));
            o[3] = '\0';
        }
    }

    return LMC_OK;
}

/* ============================================================
 * HASH TABLE FOR VOCABULARY LOOKUP
 * FNV-1a hash — fast, simple, good distribution.
 * ============================================================ */
static uint32_t str_hash(const uint8_t *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= s[i]; h *= 16777619u; }
    return h;
}

static void vocab_hash_insert(LmcTokenizer *tok, int id) {
    uint32_t slot              = str_hash(tok->vocab[id].bytes, tok->vocab[id].len)
                                 % VOCAB_HASH_SIZE;
    tok->vocab_hash_next[id]   = tok->vocab_hash[slot];
    tok->vocab_hash[slot]      = id;
}

static int vocab_lookup(const LmcTokenizer *tok, const uint8_t *s, int len) {
    uint32_t slot = str_hash(s, len) % VOCAB_HASH_SIZE;
    int id = tok->vocab_hash[slot];
    while (id != -1) {
        if (tok->vocab[id].len == len &&
            memcmp(tok->vocab[id].bytes, s, (size_t)len) == 0) return id;
        id = tok->vocab_hash_next[id];
    }
    return -1;
}

/* ============================================================
 * LOAD encoder.json
 *
 * JSON format: { "token_string": token_id, ... }
 * We parse this with a hand-rolled minimal JSON reader
 * (no external dependencies).
 * ============================================================ */
LmcError lmc_tokenizer_load_encoder(LmcTokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        LMC_ERROR("Cannot open encoder.json: %s", path);
        return LMC_ERR_IO;
    }

    /* Init hash tables */
    memset(tok->vocab_hash,      -1, sizeof(tok->vocab_hash));
    memset(tok->vocab_hash_next, -1, sizeof(tok->vocab_hash_next));
    tok->vocab_size = 0;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char*)malloc((size_t)fsize + 1);
    if (!buf) {
        LMC_ERROR("OOM loading encoder.json (%ld bytes)", fsize);
        fclose(f);
        return LMC_ERR_OOM;
    }

    if ((long)fread(buf, 1, (size_t)fsize, f) != fsize) {
        LMC_ERROR("Short read on encoder.json");
        free(buf); fclose(f);
        return LMC_ERR_IO;
    }
    buf[fsize] = '\0';
    fclose(f);

    /* Parse JSON object: skip to '{', then read "key": value pairs */
    char *p = buf;
    while (*p && *p != '{') p++;
    if (*p) p++;

    while (*p) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == '}' || *p == '\0') break;
        if (*p != '"') { p++; continue; }

        p++;  /* skip opening '"' */
        uint8_t key[BPE_TOKEN_MAX_LEN];
        int key_len = 0;

        /* Read string key with escape handling */
        while (*p && *p != '"' && key_len < BPE_TOKEN_MAX_LEN - 4) {
            if (*p == '\\') {
                p++;
                switch (*p) {
                    case '"':  key[key_len++] = '"';  p++; break;
                    case '\\': key[key_len++] = '\\'; p++; break;
                    case '/':  key[key_len++] = '/';  p++; break;
                    case 'n':  key[key_len++] = '\n'; p++; break;
                    case 'r':  key[key_len++] = '\r'; p++; break;
                    case 't':  key[key_len++] = '\t'; p++; break;
                    case 'b':  key[key_len++] = '\b'; p++; break;
                    case 'f':  key[key_len++] = '\f'; p++; break;
                    case 'u': {
                        p++;
                        char hex[5] = {0};
                        for (int hi = 0; hi < 4 && *p; hi++) hex[hi] = *p++;
                        int cp = (int)strtol(hex, NULL, 16);
                        if (cp < 0x80) {
                            key[key_len++] = (uint8_t)cp;
                        } else if (cp < 0x800) {
                            key[key_len++] = (uint8_t)(0xC0 | (cp >> 6));
                            key[key_len++] = (uint8_t)(0x80 | (cp & 0x3F));
                        } else {
                            key[key_len++] = (uint8_t)(0xE0 | (cp >> 12));
                            key[key_len++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                            key[key_len++] = (uint8_t)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: key[key_len++] = (uint8_t)*p++; break;
                }
            } else {
                key[key_len++] = (uint8_t)*p++;
            }
        }
        if (*p == '"') p++;

        /* Skip ": " */
        while (*p == ' ' || *p == ':' || *p == '\t') p++;
        if (*p < '0' || *p > '9') continue;

        /* Read integer token id */
        int token_id = 0;
        while (*p >= '0' && *p <= '9') { token_id = token_id * 10 + (*p++ - '0'); }

        if (token_id < BPE_MAX_VOCAB) {
            memcpy(tok->vocab[token_id].bytes, key, (size_t)key_len);
            tok->vocab[token_id].len = key_len;
            vocab_hash_insert(tok, token_id);
            if (token_id + 1 > tok->vocab_size) tok->vocab_size = token_id + 1;
        }
    }

    free(buf);
    LMC_INFO("Vocabulary   : %d tokens", tok->vocab_size);
    return LMC_OK;
}

/* ============================================================
 * LOAD vocab.bpe
 *
 * Format: one merge rule per line: "token_a token_b"
 * First line is a version header (skipped).
 * ============================================================ */
LmcError lmc_tokenizer_load_bpe(LmcTokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        LMC_ERROR("Cannot open vocab.bpe: %s", path);
        return LMC_ERR_IO;
    }

    tok->n_merges = 0;
    char line[1024];

    /* Skip version header line (e.g., "#version: 0.2") */
    if (!fgets(line, sizeof(line), f)) { fclose(f); return LMC_OK; }

    while (fgets(line, sizeof(line), f) && tok->n_merges < BPE_MAX_MERGES) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        char *space = strchr(line, ' ');
        if (!space) continue;
        *space = '\0';

        const char *left_str  = line;
        const char *right_str = space + 1;

        int left_id  = vocab_lookup(tok, (const uint8_t*)left_str,
                                    (int)strlen(left_str));
        int right_id = vocab_lookup(tok, (const uint8_t*)right_str,
                                    (int)strlen(right_str));
        if (left_id == -1 || right_id == -1) continue;

        int ll = (int)strlen(left_str);
        int rl = (int)strlen(right_str);
        if (ll + rl >= BPE_TOKEN_MAX_LEN) continue;

        uint8_t merged[BPE_TOKEN_MAX_LEN];
        memcpy(merged, left_str, (size_t)ll);
        memcpy(merged + ll, right_str, (size_t)rl);
        int result_id = vocab_lookup(tok, merged, ll + rl);
        if (result_id == -1) continue;

        tok->merges[tok->n_merges].left   = left_id;
        tok->merges[tok->n_merges].right  = right_id;
        tok->merges[tok->n_merges].result = result_id;
        tok->n_merges++;
    }

    fclose(f);
    LMC_INFO("BPE merges   : %d rules", tok->n_merges);
    return LMC_OK;
}

/* ============================================================
 * BPE ENCODING (APPLY MERGE RULES)
 *
 * Greedy bottom-up merge: find the highest-priority (lowest index)
 * adjacent pair and merge it. Repeat until no merges apply.
 * ============================================================ */
#define MAX_WORD_LEN    128
#define MAX_WORD_TOKENS (MAX_WORD_LEN * 4)

typedef struct { int ids[MAX_WORD_TOKENS]; int len; } TokenSeq;

static void bpe_apply_merges(TokenSeq *seq, const LmcTokenizer *tok) {
    while (seq->len >= 2) {
        int best_merge = tok->n_merges;
        int best_pos   = -1;

        for (int i = 0; i < seq->len - 1; i++) {
            int a = seq->ids[i], b = seq->ids[i+1];
            for (int m = 0; m < tok->n_merges; m++) {
                if (tok->merges[m].left == a && tok->merges[m].right == b) {
                    if (m < best_merge) { best_merge = m; best_pos = i; }
                    break;
                }
            }
        }

        if (best_pos == -1) break;

        /* Apply merge: replace pair at best_pos with result */
        seq->ids[best_pos] = tok->merges[best_merge].result;
        for (int i = best_pos + 1; i < seq->len - 1; i++)
            seq->ids[i] = seq->ids[i+1];
        seq->len--;
    }
}

/* Encode a single pre-tokenized word (as byte sequence) into token ids */
static int encode_word(const LmcTokenizer *tok,
                       const uint8_t *word_bytes, int word_len,
                       int *out_ids) {
    TokenSeq seq = {.len = 0};

    for (int i = 0; i < word_len && seq.len < MAX_WORD_TOKENS; i++) {
        const char *encoded = tok->byte_encoder[word_bytes[i]];
        int tid = vocab_lookup(tok, (const uint8_t*)encoded, (int)strlen(encoded));
        seq.ids[seq.len++] = (tid == -1) ? (int)word_bytes[i] : tid;
    }

    bpe_apply_merges(&seq, tok);

    for (int i = 0; i < seq.len; i++) out_ids[i] = seq.ids[i];
    return seq.len;
}

/* ============================================================
 * TOKENIZE TEXT
 *
 * GPT-2 pre-tokenization: split on word boundaries,
 * treating a leading space as part of the word.
 * Then apply BPE within each chunk.
 * ============================================================ */
int lmc_tokenize(const LmcTokenizer *tok, const char *text,
                 int *out_ids, int max_tokens) {
    int n_tokens = 0;
    const uint8_t *p = (const uint8_t*)text;
    int text_len = (int)strlen(text);
    int i = 0;

    while (i < text_len && n_tokens < max_tokens) {
        uint8_t word[MAX_WORD_LEN];
        int wlen = 0;

        /* Optionally consume a leading space as part of the word */
        if (p[i] == ' ' && i + 1 < text_len) word[wlen++] = p[i++];

        if (i >= text_len) {
            if (wlen > 0) {
                int word_ids[MAX_WORD_TOKENS];
                int n = encode_word(tok, word, wlen, word_ids);
                for (int j = 0; j < n && n_tokens < max_tokens; j++)
                    out_ids[n_tokens++] = word_ids[j];
            }
            break;
        }

        uint8_t c = p[i];
        /* Alphanumeric / multibyte: collect a run */
        if ((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c>=0x80) {
            while (i < text_len && wlen < MAX_WORD_LEN - 1) {
                uint8_t cc = p[i];
                if ((cc>='A'&&cc<='Z')||(cc>='a'&&cc<='z')||
                    (cc>='0'&&cc<='9')||cc>=0x80)
                    word[wlen++] = p[i++];
                else break;
            }
        } else {
            /* Single non-alphanumeric character */
            word[wlen++] = p[i++];
        }

        if (wlen > 0) {
            int word_ids[MAX_WORD_TOKENS];
            int n = encode_word(tok, word, wlen, word_ids);
            for (int j = 0; j < n && n_tokens < max_tokens; j++)
                out_ids[n_tokens++] = word_ids[j];
        }
    }

    return n_tokens;
}

/* ============================================================
 * DETOKENIZE ONE TOKEN
 *
 * Converts a single token id back to its raw byte string.
 * Uses byte_decoder to reverse the byte encoding step.
 * ============================================================ */

/* Read one UTF-8 codepoint from *s, advancing *s */
static int utf8_decode_char(const char **s) {
    unsigned char c = (unsigned char)**s;
    int cp;
    if (c < 0x80) {
        cp = c; (*s)++;
    } else if ((c & 0xE0) == 0xC0) {
        cp  = (c & 0x1F) << 6; (*s)++;
        cp |= ((unsigned char)**s & 0x3F); (*s)++;
    } else if ((c & 0xF0) == 0xE0) {
        cp  = (c & 0x0F) << 12; (*s)++;
        cp |= ((unsigned char)**s & 0x3F) << 6; (*s)++;
        cp |= ((unsigned char)**s & 0x3F); (*s)++;
    } else {
        cp = '?'; (*s)++;
    }
    return cp;
}

int lmc_detokenize_token(const LmcTokenizer *tok, int token_id,
                          char *out_buf, int buf_size) {
    if (token_id < 0 || token_id >= tok->vocab_size) return 0;

    const LmcVocabEntry *ve = &tok->vocab[token_id];
    const char *s   = (const char*)ve->bytes;
    const char *end = s + ve->len;
    int out_len = 0;

    while (s < end && out_len < buf_size - 1) {
        int cp = utf8_decode_char(&s);
        if (cp >= 0 && cp < 0x400) {
            out_buf[out_len++] = (char)tok->byte_decoder[cp];
        }
    }
    out_buf[out_len] = '\0';
    return out_len;
}
