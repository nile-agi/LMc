/* chat_templates.h — Chat prompt templates for instruct-tuned models.
 *
 * Usage:
 *   const ChatTemplate *t = detect_template(model_path);
 *   char buf[8192];
 *   build_prompt(buf, sizeof(buf), t, system_msg, user_msg);
 *   // buf now contains the full prompt ready for tokenisation
 *
 * To add a new model: append an entry to g_templates[] before the sentinel.
 * The pattern field is matched as a case-insensitive substring of the model
 * filename, so "llama-3" matches "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf".
 */
#ifndef LMC_CHAT_TEMPLATES_H
#define LMC_CHAT_TEMPLATES_H

#include <string.h>
#include <stddef.h>

typedef struct {
    const char *id;           /* short human-readable name                   */
    const char *pattern;      /* lowercase substring to match in model path  */
    const char *sys_pre;      /* text before system message                  */
    const char *sys_suf;      /* text after  system message                  */
    const char *user_pre;     /* text before user message                    */
    const char *user_suf;     /* text after  user message                    */
    const char *asst_pre;     /* text before assistant turn (model generates) */
    const char *default_sys;  /* default system message (NULL = omit turn)   */
} ChatTemplate;

/* ── Template table ─────────────────────────────────────────────────────── *
 * Entries are checked in order; first match wins.                           */
static const ChatTemplate g_templates[] = {

    /* TinyLlama-Chat / Zephyr (HuggingFace chat-ml variant) */
    {
        "zephyr",   "tinyllama",
        "<|system|>\n",    "\n",
        "<|user|>\n",      "\n",
        "<|assistant|>\n",
        "You are a helpful assistant."
    },
    /* Zephyr-beta and other Zephyr models */
    {
        "zephyr",   "zephyr",
        "<|system|>\n",    "\n",
        "<|user|>\n",      "\n",
        "<|assistant|>\n",
        "You are a helpful assistant."
    },
    /* LLaMA-3 Instruct */
    {
        "llama3",   "llama-3",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "You are a helpful assistant."
    },
    /* LLaMA-2 Chat */
    {
        "llama2",   "llama-2",
        "[INST] <<SYS>>\n",   "\n<</SYS>>\n\n",
        "",                   " [/INST] ",
        "",
        "You are a helpful assistant."
    },
    /* Mistral / Mixtral Instruct (no system turn in base template) */
    {
        "mistral",  "mistral",
        "", "",
        "[INST] ", " [/INST]",
        "",
        NULL
    },
    /* Phi-2 */
    {
        "phi2",     "phi-2",
        "", "",
        "Instruct: ", "\n",
        "Output: ",
        NULL
    },
    /* Phi-3 */
    {
        "phi3",     "phi-3",
        "<|system|>\n",    "<|end|>\n",
        "<|user|>\n",      "<|end|>\n",
        "<|assistant|>\n",
        "You are a helpful assistant."
    },
    /* Gemma Instruct */
    {
        "gemma",    "gemma",
        "", "",
        "<start_of_turn>user\n", "<end_of_turn>\n",
        "<start_of_turn>model\n",
        NULL
    },
    /* ChatML (general — matches last as fallback for "chat" models) */
    {
        "chatml",   "chat",
        "<|im_start|>system\n",  "<|im_end|>\n",
        "<|im_start|>user\n",    "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "You are a helpful assistant."
    },
    /* Sentinel — must be last */
    { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL }
};

/* ── detect_template ─────────────────────────────────────────────────────── *
 * Returns the first template whose pattern appears in model_path            *
 * (case-insensitive), or NULL if none matched (raw prompt pass-through).   */
static const ChatTemplate *detect_template(const char *model_path) {
    if (!model_path) return NULL;
    char lower[1024];
    int i;
    for (i = 0; model_path[i] && i < 1023; i++) {
        char c = model_path[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }
    lower[i] = '\0';
    for (const ChatTemplate *t = g_templates; t->id; t++)
        if (strstr(lower, t->pattern)) return t;
    return NULL;
}

/* ── build_prompt ────────────────────────────────────────────────────────── *
 * Assembles:  sys_pre + system_msg + sys_suf                                *
 *           + user_pre + user_msg + user_suf                                *
 *           + asst_pre                                                       *
 * into buf[buf_size].  Returns byte count written (excl. NUL), -1 on OOB.  *
 * Pass system_msg=NULL to use the template default (may omit the turn).    */
static int build_prompt(char *buf, int buf_size,
                        const ChatTemplate *tmpl,
                        const char *system_msg,
                        const char *user_msg)
{
    if (!tmpl) {
        /* No template — pass user message through unchanged */
        int len = (int)strlen(user_msg);
        if (len >= buf_size) return -1;
        memcpy(buf, user_msg, (size_t)len + 1);
        return len;
    }

    int n = 0;
#define APND(s) do { \
    if (s) { int _l=(int)strlen(s); if(n+_l>=buf_size) return -1; \
             memcpy(buf+n,s,(size_t)_l); n+=_l; } } while(0)

    /* System turn */
    const char *sys = system_msg ? system_msg : tmpl->default_sys;
    if (sys && *sys && tmpl->sys_pre && *tmpl->sys_pre) {
        APND(tmpl->sys_pre);
        APND(sys);
        APND(tmpl->sys_suf);
    }
    /* User turn */
    APND(tmpl->user_pre);
    APND(user_msg);
    APND(tmpl->user_suf);
    /* Assistant prefix — model generates from here */
    APND(tmpl->asst_pre);
#undef APND

    buf[n] = '\0';
    return n;
}

#endif /* LMC_CHAT_TEMPLATES_H */