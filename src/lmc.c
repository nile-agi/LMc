/* lmc.c — Main entry point, argument parsing, sampling, generation loop.
 *
 * Usage:
 *   ./lmc --model models/gpt2-xl.gguf --prompt "Hello!" --n-predict 128
 *         --temp 0.7 --top-p 0.9 --threads 4 --encoder encoder.json --bpe vocab.bpe
 */
#include "utils.h"
#include "models.h"
#include "quant.h"
#include "gguf.h"
#include "ops.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* compile-time float32 sanity check */
typedef char assert_float32_is_4_bytes[(sizeof(float)==4)?1:-1];

/* ── RNG (xorshift64) ─────────────────────────────────────────────────────── */
static uint64_t g_rng_state = 0;
static void     rng_seed (uint64_t seed) { g_rng_state = seed ^ 0xdeadbeefcafeULL; if(!g_rng_state)g_rng_state=1; }
static uint64_t rng_u64  (void) { g_rng_state^=g_rng_state<<13; g_rng_state^=g_rng_state>>7; g_rng_state^=g_rng_state<<17; return g_rng_state; }
static float    rng_float(void) { return (float)(rng_u64()>>11)/(float)(1ULL<<53); }

/* ── Top-p (nucleus) sampling ─────────────────────────────────────────────── */
typedef struct { float prob; int idx; } ProbIdx;
static int cmp_prob_desc(const void *a, const void *b) {
    const ProbIdx *pa=(const ProbIdx*)a, *pb=(const ProbIdx*)b;
    return (pb->prob>pa->prob)?1:(pb->prob<pa->prob)?-1:0;
}

static int sample_top_p(float *logits, float temperature, float top_p) {
    const int V = CFG_V;
    ProbIdx *sorted = (ProbIdx *)g_act.sorted_buf;
    if (temperature < 1e-6f) {
        int best=0; for(int i=1;i<V;i++) if(logits[i]>logits[best])best=i; return best;
    }
    float inv_temp = 1.0f / temperature;
    for (int i=0;i<V;i++) logits[i] *= inv_temp;
    softmax(logits, V);
    for (int i=0;i<V;i++){sorted[i].prob=logits[i];sorted[i].idx=i;}
    qsort(sorted, V, sizeof(ProbIdx), cmp_prob_desc);
    float cumsum=0.0f; int nucleus=0;
    for(int i=0;i<V;i++){cumsum+=sorted[i].prob;nucleus=i+1;if(cumsum>=top_p)break;}
    float ns=0.0f; for(int i=0;i<nucleus;i++) ns+=sorted[i].prob;
    float inv_ns=1.0f/ns, r=rng_float(), cdf=0.0f;
    for(int i=0;i<nucleus;i++){cdf+=sorted[i].prob*inv_ns;if(r<cdf)return sorted[i].idx;}
    return sorted[nucleus-1].idx;
}

/* ── Generation loop ──────────────────────────────────────────────────────── */
static void generate(const char *prompt, int max_new_tokens,
                     float temperature, float top_p) {
    int prompt_tokens[MAX_SEQ_LEN];
    int n_prompt = tokenize(&g_tokenizer, prompt, prompt_tokens, CFG_S);
    if (n_prompt == 0) { LMC_ERROR("Empty prompt"); return; }
    LMC_INFO("Prompt tokens: %d", n_prompt);
    printf("\n--- Generated Text ---\n%s", prompt); fflush(stdout);

    g_kv_cache.seq_len = 0;
    float *logits = NULL;
    clock_t t0 = clock();

    printf("\n--- Generated Text ---\n"); fflush(stdout);

    /* Print the prompt back through the tokenizer so byte encoding is consistent */
    {
        char dbuf[64];
        for (int i = 0; i < n_prompt; i++) {
            int dlen = detokenize_token(&g_tokenizer, prompt_tokens[i], dbuf, sizeof(dbuf));
            if (dlen > 0) fwrite(dbuf, 1, (size_t)dlen, stdout);
        }
        fflush(stdout);
    }

    /* Prefill */
    for (int i = 0; i < n_prompt; i++) {
        logits = model_forward(prompt_tokens[i], i);
        g_kv_cache.seq_len = i + 1;
    }

    /* EOS token: use tokenizer's eos_id if set, fall back to GPT-2's 50256 */
    int eos_id = (g_tokenizer.eos_id > 0) ? g_tokenizer.eos_id : 50256;

    /* Decode */
    int pos = n_prompt, tokens_gen = 0;
    char decode_buf[64];
    for (int step = 0; step < max_new_tokens; step++) {
        if (pos >= CFG_S) { printf("\n[Context window full]\n"); break; }
        int next = sample_top_p(logits, temperature, top_p);
        if (next == eos_id) { printf("\n[EOS]\n"); break; }
        int dec_len = detokenize_token(&g_tokenizer, next, decode_buf, sizeof(decode_buf));
        if (dec_len > 0) { fwrite(decode_buf, 1, (size_t)dec_len, stdout); fflush(stdout); }
        logits = model_forward(next, pos);
        g_kv_cache.seq_len = pos + 1;
        pos++; tokens_gen++;
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\n--- Done: %d tokens in %.2fs  (%.1f t/s) ---\n",
           tokens_gen, elapsed,
           elapsed > 0 ? (double)tokens_gen / elapsed : 0.0);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
/* ── Tokenizer file auto-detection ───────────────────────────────────────── *
 * Search for 'filename' in several candidate directories:
 *   1. Current working directory
 *   2. Same directory as the model file (if model_path is known)
 *   3. A 'tokenizer/' sub-directory of the model's directory
 *   4. The executable's own directory (argv[0])
 * Returns a pointer to a static buffer on success, or NULL if not found.   */
/* find_tokenizer_file() may be called twice (encoder + bpe), so each call
 * gets its own static buffer — the caller keeps the returned pointer alive
 * across both calls and into load_tokenizer().                              */
static const char *find_tokenizer_file(const char *filename,
                                        const char *model_path,
                                        const char *argv0)
{
    /* Two slots: slot 0 = first call (encoder.json), slot 1 = second (vocab.bpe). */
    static char bufs[2][1024];
    static int  slot = 0;
    char *buf = bufs[slot & 1];
    slot++;
    FILE *f;

#define TOK_BUF_SZ 1024
    /* 1. Current directory */
    snprintf(buf, TOK_BUF_SZ, "%s", filename);
    if ((f = fopen(buf, "r"))) { fclose(f); return buf; }

    /* 2. Same directory as the model */
    if (model_path) {
        const char *slash = strrchr(model_path, '/');
        if (!slash) slash = strrchr(model_path, '\\');
        if (slash) {
            int dir_len = (int)(slash - model_path) + 1;
            snprintf(buf, TOK_BUF_SZ, "%.*s%s", dir_len, model_path, filename);
            if ((f = fopen(buf, "r"))) { fclose(f); return buf; }
            /* 3. tokenizer/ sub-directory next to the model */
            snprintf(buf, TOK_BUF_SZ, "%.*stokenizer/%s", dir_len, model_path, filename);
            if ((f = fopen(buf, "r"))) { fclose(f); return buf; }
        }
    }

    /* 4. Same directory as the executable */
    if (argv0) {
        const char *slash = strrchr(argv0, '/');
        if (!slash) slash = strrchr(argv0, '\\');
        if (slash) {
            int dir_len = (int)(slash - argv0) + 1;
            snprintf(buf, TOK_BUF_SZ, "%.*s%s", dir_len, argv0, filename);
            if ((f = fopen(buf, "r"))) { fclose(f); return buf; }
        }
    }
#undef TOK_BUF_SZ

    return NULL;
}

int main(int argc, char *argv[]) {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  lmc — lightweight model engine  (C99, GGUF)    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    const char *prompt       = "Hello, world!";
    const char *model_path   = NULL;
    const char *encoder_path = NULL;   /* resolved below after arg parsing  */
    const char *bpe_path     = NULL;
    int         max_tokens   = 128;
    float       temperature  = 0.7f;
    float       top_p        = 0.9f;
    int         n_threads    = 0;   /* 0 = use all available */

    /* ── Argument parsing ─────────────────────────────────────────────────── */
    for (int i = 1; i < argc; i++) {
#define NEXTARG(dst, conv) \
    do { if(i+1>=argc){LMC_ERROR("Missing value for %s",argv[i]);return 1;} \
         (dst)=conv(argv[++i]); } while(0)
        if      (!strcmp(argv[i],"--model")   ||!strcmp(argv[i],"-m"))  { if(i+1<argc)model_path=argv[++i]; }
        else if (!strcmp(argv[i],"--prompt")  ||!strcmp(argv[i],"-p"))  { if(i+1<argc)prompt=argv[++i]; }
        else if (!strcmp(argv[i],"--n-predict")||!strcmp(argv[i],"-n")) NEXTARG(max_tokens,(int)atoi);
        else if (!strcmp(argv[i],"--temp")    ||!strcmp(argv[i],"-t"))  NEXTARG(temperature,(float)atof);
        else if (!strcmp(argv[i],"--top-p"))                            NEXTARG(top_p,(float)atof);
        else if (!strcmp(argv[i],"--threads") ||!strcmp(argv[i],"-j"))  NEXTARG(n_threads,(int)atoi);
        else if (!strcmp(argv[i],"--encoder"))                          { if(i+1<argc)encoder_path=argv[++i]; }
        else if (!strcmp(argv[i],"--bpe"))                              { if(i+1<argc)bpe_path=argv[++i]; }
        else if (!strcmp(argv[i],"--help")    ||!strcmp(argv[i],"-h")) {
            printf("Usage: lmc [options]\n"
                   "  --model   / -m  PATH     Model file (.gguf or .bin)\n"
                   "  --prompt  / -p  TEXT     Input prompt (default: \"Hello, world!\")\n"
                   "  --n-predict/-n  N        Tokens to generate (default: 128)\n"
                   "  --temp    / -t  FLOAT    Temperature (default: 0.7)\n"
                   "  --top-p         FLOAT    Top-p nucleus (default: 0.9)\n"
                   "  --threads / -j  N        Number of threads (default: all)\n"
                   "  --encoder       PATH     encoder.json path (default: encoder.json)\n"
                   "  --bpe           PATH     vocab.bpe path    (default: vocab.bpe)\n");
            return 0;
        }
#undef NEXTARG
    }

    /* ── Validation ───────────────────────────────────────────────────────── */
    if (max_tokens <= 0)                        max_tokens   = 128;
    if (max_tokens > MAX_SEQ_LEN - 10)          max_tokens   = MAX_SEQ_LEN - 10;
    if (temperature < 0.0f)                     temperature  = 0.7f;
    if (top_p <= 0.0f || top_p > 1.0f)         top_p        = 0.9f;

    /* ── Thread setup ─────────────────────────────────────────────────────── */
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    LMC_INFO("OpenMP: %d thread(s)", omp_get_max_threads());
#else
    LMC_INFO("Single-threaded (compile with -fopenmp for multi-thread)");
    (void)n_threads;
#endif

    /* ── Tokenizer file resolution ───────────────────────────────────────── *
     * If the user didn't pass explicit --encoder / --bpe paths, probe common
     * locations automatically before falling back to bare filenames.        */
    if (!encoder_path) {
        encoder_path = find_tokenizer_file("encoder.json", model_path, argv[0]);
        if (!encoder_path) encoder_path = "encoder.json"; /* will error clearly */
    }
    if (!bpe_path) {
        bpe_path = find_tokenizer_file("vocab.bpe", model_path, argv[0]);
        if (!bpe_path) bpe_path = "vocab.bpe";
    }

    /* ── Model auto-detection ─────────────────────────────────────────────── */
    if (!model_path) {
        model_path = find_default_model();
        if (!model_path) {
            LMC_ERROR("No model file found. Pass --model <path.gguf>");
            LMC_ERROR("Download: huggingface.co/ggml-org/gpt2-* (GGUF)");
            return 1;
        }
    }

    /* ── Config printout ──────────────────────────────────────────────────── */
    LMC_INFO("Model:       %s", model_path);
    LMC_INFO("Prompt:      \"%s\"", prompt);
    LMC_INFO("Max tokens:  %d", max_tokens);
    LMC_INFO("Temperature: %.2f", temperature);
    LMC_INFO("Top-p:       %.2f\n", top_p);

    rng_seed((uint64_t)time(NULL));

    /* ── Load and run ─────────────────────────────────────────────────────── */
    load_model    (model_path);
    load_tokenizer(encoder_path, bpe_path);
    init_kv_cache ();
    init_activations();

    generate(prompt, max_tokens, temperature, top_p);

    /* ── Cleanup ──────────────────────────────────────────────────────────── */
    arena_free();
    free(g_weights.layers);
    free(g_kv_cache.k_cache);
    free(g_kv_cache.v_cache);
    free_activations();
    return 0;
}
