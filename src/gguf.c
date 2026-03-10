/* gguf.c — GGUF and custom-binary model loaders. */
#include "gguf.h"
#include "models.h"
#include "quant.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Low-level GGUF byte readers ──────────────────────────────────────────── */
static void gguf_read_bytes(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) LMC_FATAL("Unexpected EOF in GGUF file");
}
static uint8_t  gguf_u8 (FILE *f){uint8_t  v;gguf_read_bytes(f,&v,1);return v;}
static uint16_t gguf_u16(FILE *f){uint16_t v;gguf_read_bytes(f,&v,2);return v;}
static uint32_t gguf_u32(FILE *f){uint32_t v;gguf_read_bytes(f,&v,4);return v;}
static uint64_t gguf_u64(FILE *f){uint64_t v;gguf_read_bytes(f,&v,8);return v;}
static int32_t  __attribute__((unused)) gguf_i32(FILE *f){int32_t  v;gguf_read_bytes(f,&v,4);return v;}
static int64_t  __attribute__((unused)) gguf_i64(FILE *f){int64_t  v;gguf_read_bytes(f,&v,8);return v;}
static float    gguf_f32(FILE *f){float    v;gguf_read_bytes(f,&v,4);return v;}
static double   gguf_f64(FILE *f){double   v;gguf_read_bytes(f,&v,8);return v;}

static char *gguf_read_string(FILE *f, uint64_t *out_len) {
    uint64_t len = gguf_u64(f);
    char *s = (char *)malloc(len + 1);
    if (!s) LMC_FATAL("OOM gguf_read_string");
    gguf_read_bytes(f, s, len);
    s[len] = '\0';
    if (out_len) *out_len = len;
    return s;
}

static void gguf_skip_value(FILE *f, uint32_t vtype);
static void gguf_skip_array(FILE *f) {
    uint32_t et = gguf_u32(f); uint64_t cnt = gguf_u64(f);
    for (uint64_t i = 0; i < cnt; i++) gguf_skip_value(f, et);
}
static void gguf_skip_value(FILE *f, uint32_t vtype) {
    switch (vtype) {
        case GGUF_MTYPE_UINT8: case GGUF_MTYPE_INT8: case GGUF_MTYPE_BOOL:
            gguf_u8(f); break;
        case GGUF_MTYPE_UINT16: case GGUF_MTYPE_INT16: gguf_u16(f); break;
        case GGUF_MTYPE_UINT32: case GGUF_MTYPE_INT32: gguf_u32(f); break;
        case GGUF_MTYPE_FLOAT32: gguf_f32(f); break;
        case GGUF_MTYPE_UINT64: case GGUF_MTYPE_INT64: gguf_u64(f); break;
        case GGUF_MTYPE_FLOAT64: gguf_f64(f); break;
        case GGUF_MTYPE_STRING: { char *s = gguf_read_string(f,NULL); free(s); break; }
        case GGUF_MTYPE_ARRAY:  gguf_skip_array(f); break;
        default: LMC_FATAL("Unknown GGUF metadata type %u", vtype);
    }
}

/* ── Tensor name → weight pointer ─────────────────────────────────────────── */
static float **gguf_name_to_ptr(const char *name) {
    if (!strcmp(name,"token_embd.weight"))   return &g_weights.wte;
    if (!strcmp(name,"position_embd.weight"))return &g_weights.wpe;
    if (!strcmp(name,"output_norm.weight"))  return &g_weights.ln_f_weight;
    if (!strcmp(name,"output_norm.bias"))    return &g_weights.ln_f_bias;
    if (!strcmp(name,"output.weight"))       return &g_weights.lm_head;
    if (strncmp(name,"blk.",4)==0) {
        int layer = atoi(name+4);
        if (layer < 0 || layer >= CFG_L) return NULL;
        const char *rest = strchr(name+4,'.'); if (!rest) return NULL; rest++;
        LayerWeights *lw = &g_weights.layers[layer];
        if (!strcmp(rest,"attn_norm.weight"))   return &lw->ln1_weight;
        if (!strcmp(rest,"attn_norm.bias"))     return &lw->ln1_bias;
        if (!strcmp(rest,"attn_qkv.weight"))    return &lw->qkv_weight;
        if (!strcmp(rest,"attn_qkv.bias"))      return &lw->qkv_bias;
        if (!strcmp(rest,"attn_output.weight")) return &lw->attn_proj_weight;
        if (!strcmp(rest,"attn_output.bias"))   return &lw->attn_proj_bias;
        if (!strcmp(rest,"ffn_norm.weight"))    return &lw->ln2_weight;
        if (!strcmp(rest,"ffn_norm.bias"))      return &lw->ln2_bias;
        if (!strcmp(rest,"ffn_up.weight"))      return &lw->ffn_fc_weight;
        if (!strcmp(rest,"ffn_up.bias"))        return &lw->ffn_fc_bias;
        if (!strcmp(rest,"ffn_down.weight"))    return &lw->ffn_proj_weight;
        if (!strcmp(rest,"ffn_down.bias"))      return &lw->ffn_proj_bias;
    }
    return NULL;
}

/* ── Load quantised tensor dispatch macro ─────────────────────────────────── */
#define LOAD_QUANT(BS, BPB, FN)  do { \
    if (t->n_elements % (BS) != 0) \
        LMC_FATAL("Tensor %s: n_elements=%zu not multiple of %d", \
                  t->name, t->n_elements, (BS)); \
    size_t _rb = (t->n_elements / (BS)) * (BPB); \
    uint8_t *_tmp = (uint8_t *)malloc(_rb); \
    if (!_tmp) LMC_FATAL("OOM quant buf %s", t->name); \
    if (fread(_tmp, 1, _rb, f) != _rb) \
        LMC_FATAL("Short read tensor %s", t->name); \
    (FN)(_tmp, dst, t->n_elements); \
    free(_tmp); \
} while(0)

/* ── GGUF loader ──────────────────────────────────────────────────────────── */
static void load_model_gguf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) LMC_FATAL("Cannot open GGUF: %s", path);

    uint32_t magic   = gguf_u32(f);
    uint32_t version = gguf_u32(f);
    if (magic   != GGUF_MAGIC) LMC_FATAL("Not a GGUF file (magic=0x%08X)", magic);
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX)
        LMC_FATAL("Unsupported GGUF version %u", version);
    LMC_INFO("GGUF version %u", version);

    uint64_t n_tensors = gguf_u64(f);
    uint64_t n_kv      = gguf_u64(f);
    LMC_INFO("Tensors: %llu   KV pairs: %llu",
             (unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* Defaults — overridden by metadata keys below */
    g_cfg.vocab_size = MAX_VOCAB_SIZE;
    g_cfg.seq_len    = MAX_SEQ_LEN;
    g_cfg.n_layers   = 0;
    g_cfg.n_heads    = 0;
    g_cfg.embed_dim  = 0;
    g_cfg.ffn_dim    = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = gguf_read_string(f, NULL);
        uint32_t vtype = gguf_u32(f);
        if (vtype == GGUF_MTYPE_UINT32) {
            uint32_t val = gguf_u32(f);
            if      (!strcmp(key,"gpt2.block_count"))          g_cfg.n_layers = (int)val;
            else if (!strcmp(key,"gpt2.attention.head_count")) g_cfg.n_heads  = (int)val;
            else if (!strcmp(key,"gpt2.embedding_length"))     g_cfg.embed_dim= (int)val;
            else if (!strcmp(key,"gpt2.feed_forward_length"))  g_cfg.ffn_dim  = (int)val;
            else if (!strcmp(key,"gpt2.context_length"))       g_cfg.seq_len  = (int)val;
        } else {
            gguf_skip_value(f, vtype);
        }
        free(key);
    }

    if (g_cfg.n_layers==0 || g_cfg.n_heads==0 || g_cfg.embed_dim==0)
        LMC_FATAL("GGUF missing required arch metadata (L=%d H=%d D=%d)",
                  CFG_L, CFG_H, CFG_D);
    if (g_cfg.ffn_dim == 0) g_cfg.ffn_dim = 4 * g_cfg.embed_dim;
    g_cfg.head_dim = g_cfg.embed_dim / g_cfg.n_heads;

    LMC_INFO("Architecture: L=%d H=%d D=%d F=%d Dh=%d V=%d S=%d",
             CFG_L,CFG_H,CFG_D,CFG_F,CFG_Dh,CFG_V,CFG_S);
    if      (CFG_L==12&&CFG_D== 768) LMC_INFO("Variant: GPT-2 Small  (124M)");
    else if (CFG_L==24&&CFG_D==1024) LMC_INFO("Variant: GPT-2 Medium (345M)");
    else if (CFG_L==36&&CFG_D==1280) LMC_INFO("Variant: GPT-2 Large  (774M)");
    else if (CFG_L==48&&CFG_D==1600) LMC_INFO("Variant: GPT-2 XL    (1.5B)");
    else                              LMC_INFO("Variant: Custom (%dL/%dD)",CFG_L,CFG_D);

    /* Read tensor descriptors */
    const int max_tensors = CFG_L * GGUF_TENSORS_PER_LAYER + GGUF_TENSORS_GLOBAL + 4;
    if ((int)n_tensors > max_tensors)
        LMC_FATAL("Too many tensors (%llu > %d expected for %dL)",
                  (unsigned long long)n_tensors, max_tensors, CFG_L);

    GGUFTensor *tensors = (GGUFTensor *)calloc(n_tensors, sizeof(GGUFTensor));
    if (!tensors) LMC_FATAL("OOM tensor table");

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &tensors[i];
        uint64_t nl; char *nm = gguf_read_string(f, &nl);
        strncpy(t->name, nm, sizeof(t->name)-1); free(nm);
        t->n_dims = gguf_u32(f);
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d]     = gguf_u64(f);
            t->n_elements *= (size_t)t->dims[d];
        }
        t->type   = gguf_u32(f);
        t->offset = gguf_u64(f);
    }

    /* Align to data section (32-byte boundary) */
    long header_end = ftell(f);
    long data_start = (header_end + 31) / 32 * 32;
    fseek(f, data_start, SEEK_SET);

    /* Allocate arena + weight pointers */
    size_t total = gpt2_total_params();
    LMC_INFO("Parameters: %zu  (%.1f MB float32)", total, total*4.0/(1024*1024));
    const size_t lm_head_sz = (size_t)CFG_V * CFG_D;
    arena_init(total + lm_head_sz);
    assign_weight_ptrs();
    g_weights.lm_head = arena_alloc(lm_head_sz);
    memcpy(g_weights.lm_head, g_weights.wte, lm_head_sz * sizeof(float));

    /* Load tensors */
    int loaded = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &tensors[i];
        float **dst_ptr = gguf_name_to_ptr(t->name);
        if (!dst_ptr) { LMC_INFO("Skip %-48s n=%zu", t->name, t->n_elements); continue; }
        float *dst = *dst_ptr;
        fseek(f, data_start + (long)t->offset, SEEK_SET);

        switch (t->type) {
        case GGUF_TYPE_F32:
            if (fread(dst,sizeof(float),t->n_elements,f)!=t->n_elements)
                LMC_FATAL("Short read F32 %s",t->name);
            break;
        case GGUF_TYPE_F16: {
            uint16_t *tmp=(uint16_t*)malloc(t->n_elements*2);
            if (!tmp) LMC_FATAL("OOM F16 %s",t->name);
            if (fread(tmp,2,t->n_elements,f)!=t->n_elements) LMC_FATAL("Short read F16 %s",t->name);
            for (size_t e=0;e<t->n_elements;e++) dst[e]=f16_to_f32(tmp[e]);
            free(tmp); break;
        }
        case GGUF_TYPE_Q2_K:    LOAD_QUANT(Q2_K_BLOCK_SIZE,   Q2_K_BYTES_PER_BLOCK,   dequant_q2k);    break;
        case GGUF_TYPE_Q3_K:    LOAD_QUANT(Q3_K_BLOCK_SIZE,   Q3_K_BYTES_PER_BLOCK,   dequant_q3k);    break;
        case GGUF_TYPE_Q4_0:    LOAD_QUANT(Q4_0_BLOCK_SIZE,   Q4_0_BYTES_PER_BLOCK,   dequant_q4_0);   break;
        case GGUF_TYPE_Q4_1:    LOAD_QUANT(Q4_1_BLOCK_SIZE,   Q4_1_BYTES_PER_BLOCK,   dequant_q4_1);   break;
        case GGUF_TYPE_Q4_K:    LOAD_QUANT(Q4_K_BLOCK_SIZE,   Q4_K_BYTES_PER_BLOCK,   dequant_q4k);    break;
        case GGUF_TYPE_Q5_0:    LOAD_QUANT(Q5_0_BLOCK_SIZE,   Q5_0_BYTES_PER_BLOCK,   dequant_q5_0);   break;
        case GGUF_TYPE_Q5_1:    LOAD_QUANT(Q5_1_BLOCK_SIZE,   Q5_1_BYTES_PER_BLOCK,   dequant_q5_1);   break;
        case GGUF_TYPE_Q5_K:    LOAD_QUANT(Q5_K_BLOCK_SIZE,   Q5_K_BYTES_PER_BLOCK,   dequant_q5k);    break;
        case GGUF_TYPE_Q6_K:    LOAD_QUANT(Q6_K_BLOCK_SIZE,   Q6_K_BYTES_PER_BLOCK,   dequant_q6k);    break;
        case GGUF_TYPE_Q8_0:    LOAD_QUANT(Q8_0_BLOCK_SIZE,   Q8_0_BYTES_PER_BLOCK,   dequant_q8_0);   break;
        case GGUF_TYPE_IQ3_XXS: LOAD_QUANT(IQ3_XXS_BLOCK_SIZE,IQ3_XXS_BYTES_PER_BLOCK,dequant_iq3xxs); break;
        case GGUF_TYPE_IQ3_S:   LOAD_QUANT(IQ3_S_BLOCK_SIZE,  IQ3_S_BYTES_PER_BLOCK,  dequant_iq3s);   break;
        case GGUF_TYPE_IQ4_XS:  LOAD_QUANT(IQ4_XS_BLOCK_SIZE, IQ4_XS_BYTES_PER_BLOCK, dequant_iq4_xs); break;
        default:
            LMC_FATAL("Unsupported tensor type %u for '%s'\n"
                      "  Supported: F32 F16 Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0 IQ3_XXS IQ3_S IQ4_XS",
                      t->type, t->name);
        }
        loaded++;
    }
#undef LOAD_QUANT

    free(tensors); fclose(f);
    LMC_INFO("Tensors loaded: %d", loaded);
    LMC_INFO("Loaded (GGUF): %s", path);
}

/* ── Custom .bin loader ────────────────────────────────────────────────────── */
static void load_model_bin(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) LMC_FATAL("Cannot open model: %s", path);

    uint32_t magic,version,vocab_size,seq_len,n_layers,n_heads,embed_dim;
#define R1(v) do{if(fread(&(v),sizeof(v),1,f)!=1)LMC_FATAL("Truncated header: %s",path);}while(0)
    R1(magic);R1(version);R1(vocab_size);R1(seq_len);R1(n_layers);R1(n_heads);R1(embed_dim);
#undef R1
    if (magic   != MODEL_MAGIC)   LMC_FATAL("Bad magic 0x%08X", magic);
    if (version != MODEL_VERSION) LMC_FATAL("Version mismatch: got %u", version);

    g_cfg.vocab_size = (int)vocab_size;
    g_cfg.seq_len    = (int)seq_len;
    g_cfg.n_layers   = (int)n_layers;
    g_cfg.n_heads    = (int)n_heads;
    g_cfg.embed_dim  = (int)embed_dim;
    g_cfg.ffn_dim    = 4 * (int)embed_dim;
    g_cfg.head_dim   = (int)embed_dim / (int)n_heads;

    LMC_INFO("Architecture: L=%d H=%d D=%d F=%d Dh=%d V=%d S=%d",
             CFG_L,CFG_H,CFG_D,CFG_F,CFG_Dh,CFG_V,CFG_S);

    size_t total = gpt2_total_params();
    LMC_INFO("Parameters: %zu  (%.1f MB)", total, total*4.0/(1024*1024));
    arena_init(total);
    assign_weight_ptrs();

    if (fread(g_arena.data, sizeof(float), total, f) != total)
        LMC_FATAL("Truncated weights: %s", path);
    g_weights.lm_head = g_weights.wte;   /* tied weights */
    fclose(f);
    LMC_INFO("Loaded (.bin): %s", path);
}

/* ── Format detection ─────────────────────────────────────────────────────── */
ModelFormat detect_format(const char *path) {
    size_t len = strlen(path);
    if (len>=5 && !strcmp(path+len-5,".gguf")) return FORMAT_GGUF;
    if (len>=4 && !strcmp(path+len-4,".bin"))  return FORMAT_BIN;
    FILE *f = fopen(path,"rb"); if (!f) return FORMAT_UNKNOWN;
    uint32_t magic=0; if(fread(&magic,4,1,f)!=1)magic=0; fclose(f);
    if (magic==MODEL_MAGIC) return FORMAT_BIN;
    if (magic==GGUF_MAGIC)  return FORMAT_GGUF;
    return FORMAT_UNKNOWN;
}

static const char *candidates[] = {
    "gpt2_124m.bin","gpt2_medium.bin","gpt2_large.bin","gpt2_xl.bin",
    "gpt2.f16.gguf","gpt2.gguf",
    "gpt2-medium.f16.gguf","gpt2-medium.gguf",
    "gpt2-large.f16.gguf", "gpt2-large.gguf",
    "gpt2-xl.f16.gguf",    "gpt2-xl.gguf",
    NULL
};

const char *find_default_model(void) {
    for (int i = 0; candidates[i]; i++) {
        FILE *f = fopen(candidates[i],"rb");
        if (f) { fclose(f); return candidates[i]; }
    }
    return NULL;
}

void load_model(const char *path) {
    ModelFormat fmt = detect_format(path);
    switch (fmt) {
        case FORMAT_BIN:  LMC_INFO("Format: custom float32 .bin"); load_model_bin (path); break;
        case FORMAT_GGUF: LMC_INFO("Format: GGUF");                load_model_gguf(path); break;
        default: LMC_FATAL("Cannot determine format: %s", path);
    }
}
