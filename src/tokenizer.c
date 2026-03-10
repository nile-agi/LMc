/* tokenizer.c — GPT-2 BPE tokenizer implementation. */
#include "tokenizer.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tokenizer g_tokenizer = {0};

/* ── Byte encoder/decoder initialisation ─────────────────────────────────── */
static void init_byte_encoder(Tokenizer *tok) {
    int bs[256], cs[256], n_bs = 0;
    for (int b=33;  b<=126; b++) { bs[n_bs]=b; cs[n_bs]=b; n_bs++; }
    for (int b=161; b<=172; b++) { bs[n_bs]=b; cs[n_bs]=b; n_bs++; }
    for (int b=174; b<=255; b++) { bs[n_bs]=b; cs[n_bs]=b; n_bs++; }
    int extra = 256;
    for (int b=0; b<256; b++) {
        int found=0;
        for (int i=0;i<n_bs;i++) if(bs[i]==b){found=1;break;}
        if (!found) { bs[n_bs]=b; cs[n_bs]=extra++; n_bs++; }
    }
    for (int i=0;i<256;i++) tok->byte_decoder[cs[i]]=bs[i];
    for (int i=0;i<256;i++) {
        int cp=cs[i]; char *out=tok->byte_encoder[bs[i]];
        if (cp<0x80)  { out[0]=(char)cp; out[1]='\0'; }
        else if (cp<0x800) {
            out[0]=(char)(0xC0|(cp>>6)); out[1]=(char)(0x80|(cp&0x3F)); out[2]='\0';
        } else {
            out[0]=(char)(0xE0|(cp>>12)); out[1]=(char)(0x80|((cp>>6)&0x3F));
            out[2]=(char)(0x80|(cp&0x3F)); out[3]='\0';
        }
    }
}

/* ── UTF-8 decoder ────────────────────────────────────────────────────────── */
static int utf8_decode(const char **s) {
    unsigned char c = (unsigned char)**s;
    int cp;
    if      (c<0x80)       { cp=c; (*s)++; }
    else if ((c&0xE0)==0xC0){ cp=(c&0x1F)<<6; (*s)++; cp|=((unsigned char)**s&0x3F); (*s)++; }
    else if ((c&0xF0)==0xE0){
        cp=(c&0x0F)<<12; (*s)++;
        cp|=((unsigned char)**s&0x3F)<<6; (*s)++;
        cp|=((unsigned char)**s&0x3F);    (*s)++;
    } else { cp='?'; (*s)++; }
    return cp;
}

/* ── Vocabulary hash table ────────────────────────────────────────────────── */
static uint32_t str_hash(const uint8_t *s, int len) {
    uint32_t h=2166136261u;
    for (int i=0;i<len;i++){h^=s[i];h*=16777619u;}
    return h;
}
static void vocab_hash_insert(Tokenizer *tok, int tid) {
    uint32_t slot = str_hash(tok->vocab[tid].bytes, tok->vocab[tid].len) % VOCAB_HASH_SIZE;
    tok->vocab_hash_next[tid] = tok->vocab_hash[slot];
    tok->vocab_hash[slot]     = tid;
}
static int vocab_lookup(const Tokenizer *tok, const uint8_t *s, int len) {
    uint32_t slot = str_hash(s,len) % VOCAB_HASH_SIZE;
    int id = tok->vocab_hash[slot];
    while (id != -1) {
        if (tok->vocab[id].len==len && memcmp(tok->vocab[id].bytes,s,len)==0) return id;
        id = tok->vocab_hash_next[id];
    }
    return -1;
}

/* ── encoder.json loader ──────────────────────────────────────────────────── */
static void load_encoder_json(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path,"r");
    if (!f) LMC_FATAL(
        "Cannot open encoder.json: %s\n"
        "[FATAL] GPT-2 tokenizer files are required.  Download them with:\n"
        "[FATAL]   wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json\n"
        "[FATAL]   wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe\n"
        "[FATAL] Then either place them next to the model file, or pass:\n"
        "[FATAL]   --encoder /path/to/encoder.json --bpe /path/to/vocab.bpe",
        path);
    memset(tok->vocab_hash,     -1, sizeof(tok->vocab_hash));
    memset(tok->vocab_hash_next,-1, sizeof(tok->vocab_hash_next));
    tok->vocab_size = 0;
    fseek(f,0,SEEK_END); long fsize=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=(char*)malloc((size_t)fsize+1);
    if (!buf) LMC_FATAL("OOM encoder.json");
    if (fread(buf,1,(size_t)fsize,f)!=(size_t)fsize) LMC_FATAL("Short read encoder.json");
    buf[fsize]='\0'; fclose(f);
    char *p=buf;
    while(*p&&*p!='{')p++;
    if(*p)p++;
    while(*p) {
        while(*p&&(*p==' '||*p=='\n'||*p=='\r'||*p=='\t'||*p==','))p++;
        if(*p=='}')break;
        if(*p!='"'){p++;continue;}
        p++;
        uint8_t key[BPE_TOKEN_MAX_LEN]; int key_len=0;
        while(*p&&*p!='"'&&key_len<BPE_TOKEN_MAX_LEN-1) {
            if(*p=='\\') {
                p++;
                switch(*p){
                    case '"':  key[key_len++]='"';  p++;break;
                    case '\\': key[key_len++]='\\'; p++;break;
                    case '/':  key[key_len++]='/';  p++;break;
                    case 'n':  key[key_len++]='\n'; p++;break;
                    case 'r':  key[key_len++]='\r'; p++;break;
                    case 't':  key[key_len++]='\t'; p++;break;
                    case 'b':  key[key_len++]='\b'; p++;break;
                    case 'f':  key[key_len++]='\f'; p++;break;
                    case 'u':{
                        p++; char hex[5]={0};
                        for(int hi=0;hi<4&&*p;hi++)hex[hi]=*p++;
                        int cp=(int)strtol(hex,NULL,16);
                        if(cp<0x80) key[key_len++]=(uint8_t)cp;
                        else if(cp<0x800){
                            key[key_len++]=(uint8_t)(0xC0|(cp>>6));
                            key[key_len++]=(uint8_t)(0x80|(cp&0x3F));
                        }else{
                            key[key_len++]=(uint8_t)(0xE0|(cp>>12));
                            key[key_len++]=(uint8_t)(0x80|((cp>>6)&0x3F));
                            key[key_len++]=(uint8_t)(0x80|(cp&0x3F));
                        }
                        break;
                    }
                    default: key[key_len++]=(uint8_t)*p++;break;
                }
            } else { key[key_len++]=(uint8_t)*p++; }
        }
        if(*p=='"')p++;
        while(*p&&(*p==' '||*p==':'||*p=='\t'))p++;
        if(*p<'0'||*p>'9')continue;
        int token_id=0;
        while(*p>='0'&&*p<='9'){token_id=token_id*10+(*p-'0');p++;}
        if(token_id<BPE_MAX_VOCAB){
            memcpy(tok->vocab[token_id].bytes,key,(size_t)key_len);
            tok->vocab[token_id].len=key_len;
            vocab_hash_insert(tok,token_id);
            if(token_id+1>tok->vocab_size)tok->vocab_size=token_id+1;
        }
    }
    free(buf);
    LMC_INFO("Vocabulary: %d tokens", tok->vocab_size);
}

/* ── vocab.bpe loader ─────────────────────────────────────────────────────── */
static void load_vocab_bpe(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path,"r");
    if (!f) LMC_FATAL(
        "Cannot open vocab.bpe: %s\n"
        "[FATAL] Download: wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        path);
    tok->n_merges=0;
    char line[1024];
    if (!fgets(line,sizeof(line),f)) { fclose(f); return; }   /* skip header */
    while (fgets(line,sizeof(line),f) && tok->n_merges < BPE_MAX_MERGES) {
        int len=(int)strlen(line);
        while(len>0&&(line[len-1]=='\n'||line[len-1]=='\r'))line[--len]='\0';
        if(!len)continue;
        char *space=strchr(line,' '); if(!space)continue; *space='\0';
        char *l_str=line, *r_str=space+1;
        int li=vocab_lookup(tok,(const uint8_t*)l_str,(int)strlen(l_str));
        int ri=vocab_lookup(tok,(const uint8_t*)r_str,(int)strlen(r_str));
        if(li==-1||ri==-1)continue;
        int ll=(int)strlen(l_str),rl=(int)strlen(r_str);
        if(ll+rl>=BPE_TOKEN_MAX_LEN)continue;
        uint8_t merged[BPE_TOKEN_MAX_LEN];
        memcpy(merged,l_str,(size_t)ll); memcpy(merged+ll,r_str,(size_t)rl);
        int res=vocab_lookup(tok,merged,ll+rl); if(res==-1)continue;
        tok->merges[tok->n_merges].left  =li;
        tok->merges[tok->n_merges].right =ri;
        tok->merges[tok->n_merges].result=res;
        tok->n_merges++;
    }
    fclose(f);
    LMC_INFO("BPE merges: %d rules", tok->n_merges);
}

void load_tokenizer(const char *ep, const char *bp) {
    init_byte_encoder(&g_tokenizer);
    load_encoder_json(&g_tokenizer, ep);
    load_vocab_bpe   (&g_tokenizer, bp);
}

/* ── BPE encoding ─────────────────────────────────────────────────────────── */
#define MAX_WORD_LEN    128
#define MAX_WORD_TOKENS (MAX_WORD_LEN * 4)
typedef struct { int ids[MAX_WORD_TOKENS]; int len; } TokenSeq;

static void bpe_apply_merges(TokenSeq *seq, const Tokenizer *tok) {
    while (seq->len >= 2) {
        int best_merge=tok->n_merges, best_pos=-1;
        for (int i=0;i<seq->len-1;i++) {
            int a=seq->ids[i], b=seq->ids[i+1];
            for (int m=0;m<tok->n_merges;m++) {
                if (tok->merges[m].left==a && tok->merges[m].right==b) {
                    if (m<best_merge){best_merge=m;best_pos=i;}
                    break;
                }
            }
        }
        if(best_pos==-1)break;
        seq->ids[best_pos]=tok->merges[best_merge].result;
        for(int i=best_pos+1;i<seq->len-1;i++) seq->ids[i]=seq->ids[i+1];
        seq->len--;
    }
}

static int encode_word(const Tokenizer *tok, const uint8_t *word_bytes,
                       int word_len, int *out_ids) {
    TokenSeq seq={.len=0};
    for(int i=0;i<word_len&&seq.len<MAX_WORD_TOKENS;i++){
        uint8_t b=word_bytes[i];
        const char *enc=tok->byte_encoder[b];
        int tid=vocab_lookup(tok,(const uint8_t*)enc,(int)strlen(enc));
        seq.ids[seq.len++]=(tid==-1)?(int)b:tid;
    }
    bpe_apply_merges(&seq,tok);
    for(int i=0;i<seq.len;i++) out_ids[i]=seq.ids[i];
    return seq.len;
}

int tokenize(const Tokenizer *tok, const char *text, int *out_ids, int max_tokens) {
    int n_tokens=0;
    const uint8_t *p=(const uint8_t*)text;
    int text_len=(int)strlen(text), i=0;
    while(i<text_len && n_tokens<max_tokens){
        uint8_t word[MAX_WORD_LEN]; int wlen=0;
        if(p[i]==' '&&i+1<text_len) word[wlen++]=p[i++];
        if(i>=text_len){ if(wlen>0){int ids[MAX_WORD_TOKENS]; int n=encode_word(tok,word,wlen,ids); for(int j=0;j<n&&n_tokens<max_tokens;j++)out_ids[n_tokens++]=ids[j];} break; }
        uint8_t c=p[i];
        if((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c>=0x80){
            while(i<text_len&&wlen<MAX_WORD_LEN-1){
                uint8_t cc=p[i];
                if((cc>='A'&&cc<='Z')||(cc>='a'&&cc<='z')||(cc>='0'&&cc<='9')||cc>=0x80) word[wlen++]=p[i++];
                else break;
            }
        } else { word[wlen++]=p[i++]; }
        if(wlen>0){int ids[MAX_WORD_TOKENS]; int n=encode_word(tok,word,wlen,ids); for(int j=0;j<n&&n_tokens<max_tokens;j++)out_ids[n_tokens++]=ids[j];}
    }
    return n_tokens;
}

int detokenize_token(const Tokenizer *tok, int token_id, char *out_buf, int buf_size) {
    if(token_id<0||token_id>=tok->vocab_size)return 0;
    const VocabEntry *ve=&tok->vocab[token_id];
    const char *s=(const char*)ve->bytes, *end=s+ve->len;
    int out_len=0;
    while(s<end&&out_len<buf_size-1){
        int cp=utf8_decode(&s);
        if(cp>=0&&cp<0x400) out_buf[out_len++]=(char)tok->byte_decoder[cp];
    }
    out_buf[out_len]='\0';
    return out_len;
}
