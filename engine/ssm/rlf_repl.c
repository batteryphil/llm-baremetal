/* rlf_repl.c — REPL command handlers for RLF engine
 * Freestanding C, no libc. Hooks into existing SomaMind REPL dispatch.
 * Add to Makefile and call rlf_repl_dispatch() from the main command router.
 */
#include "rlf_engine.h"
#include "../gguf/gguf_loader.h"

/* External REPL helpers provided by the main REPL layer */
extern void repl_print(const char *s);
extern void repl_print_line(const char *s);
extern void *repl_get_backbone(void);       /* currently loaded GGUF model  */
extern const void *repl_map_file(const char *path, unsigned long *size_out);

/* Module-level state */
static RlfWeights g_rlf_weights;
static int        g_rlf_loaded = 0;
static RlfCtx     g_rlf_ctx;   /* static scratch — 1 inference at a time */

/* ── /rlf_load <file.ooss> ──────────────────────────────────────────────── */
static void cmd_rlf_load(const char *arg) {
    if (!arg || !arg[0]) {
        repl_print("[RLF] usage: /rlf_load <file.ooss>\r\n"); return;
    }
    unsigned long sz = 0;
    const void *data = repl_map_file(arg, &sz);
    if (!data) {
        repl_print("[RLF] ERROR: could not open "); repl_print(arg);
        repl_print("\r\n"); return;
    }
    int rc = rlf_ooss_load(data, sz, &g_rlf_weights);
    if (rc != 0) {
        repl_print("[RLF] ERROR: ooss parse failed (code=");
        char tmp[8]; tmp[0]='0'-rc; tmp[1]='\0'; repl_print(tmp);
        repl_print(")\r\n"); return;
    }
    g_rlf_loaded = 1;
    repl_print("[RLF] loaded: "); repl_print(arg); repl_print("\r\n");
    rlf_status_print(&g_rlf_weights);
}

/* ── /rlf_status ────────────────────────────────────────────────────────── */
static void cmd_rlf_status(void) {
    rlf_status_print(g_rlf_loaded ? &g_rlf_weights : (void*)0);
}

/* ── /rlf_preprocess <prompt> ───────────────────────────────────────────── */
static void cmd_rlf_preprocess(const char *prompt) {
    if (!prompt || !prompt[0]) {
        repl_print("[RLF] usage: /rlf_preprocess <prompt>\r\n"); return;
    }
    rlf_preprocess(prompt, &g_rlf_ctx);
    repl_print("[RLF] preprocessed: ");
    repl_print(g_rlf_ctx.prompt_pp);
    repl_print("\r\n[RLF] bindings:\r\n");
    for (int i = 0; i < g_rlf_ctx.env.n; i++) {
        RlfBinding *b = &g_rlf_ctx.env.bindings[i];
        repl_print("  "); repl_print(b->name); repl_print(" = ");
        if (b->resolved) {
            char buf[32]; rlf_ftoa(b->numeric, buf);
            repl_print(buf);
        } else {
            repl_print(b->value); repl_print(" (unresolved)");
        }
        repl_print("\r\n");
    }
}

/* ── /rlf_infer <prompt> ────────────────────────────────────────────────── */
static void cmd_rlf_infer(const char *prompt) {
    if (!prompt || !prompt[0]) {
        repl_print("[RLF] usage: /rlf_infer <prompt>\r\n"); return;
    }
    void *backbone = repl_get_backbone();
    if (!backbone) {
        repl_print("[RLF] ERROR: no backbone loaded. Use /core_load first.\r\n");
        return;
    }
    if (!g_rlf_loaded) {
        repl_print("[RLF] ERROR: no RLF weights. Use /rlf_load first.\r\n");
        return;
    }
    char answer[64] = {0};
    int n_loops = 0;
    repl_print("[RLF] reasoning...\r\n");
    int rc = rlf_infer(prompt, backbone, &g_rlf_weights,
                       &g_rlf_ctx, answer, sizeof(answer), &n_loops);
    repl_print("[RLF] answer: "); repl_print(answer[0] ? answer : "(empty)");
    repl_print("\r\n[RLF] loops: ");
    char lbuf[4]; lbuf[0]='0'+n_loops; lbuf[1]='\0'; repl_print(lbuf);
    repl_print(rc==0 ? " (HALT)\r\n" : " (max loops)\r\n");
}

/* ── Public dispatcher ──────────────────────────────────────────────────── */
/* Call this from the main REPL command router after all existing commands. */
int rlf_repl_dispatch(const char *cmd, const char *arg) {
    if (!cmd) return 0;
    /* simple strcmp without libc */
    #define MATCH(s) (cmd[0]==s[1]&&cmd[1]==s[2]&&cmd[2]==s[3]&&\
                      cmd[3]==s[4]&&cmd[4]==s[5]&&cmd[5]==s[6]&&\
                      cmd[6]==s[7]&&cmd[7]==s[8]&&cmd[8]==s[9])
    /* use strncmp-style character comparison */
    int match_load   = (cmd[0]=='r'&&cmd[1]=='l'&&cmd[2]=='f'&&cmd[3]=='_'&&
                        cmd[4]=='l'&&cmd[5]=='o'&&cmd[6]=='a'&&cmd[7]=='d'&&!cmd[8]);
    int match_status = (cmd[0]=='r'&&cmd[1]=='l'&&cmd[2]=='f'&&cmd[3]=='_'&&
                        cmd[4]=='s'&&cmd[5]=='t'&&!cmd[6]);
    int match_infer  = (cmd[0]=='r'&&cmd[1]=='l'&&cmd[2]=='f'&&cmd[3]=='_'&&
                        cmd[4]=='i'&&cmd[5]=='n'&&!cmd[6]);
    int match_pp     = (cmd[0]=='r'&&cmd[1]=='l'&&cmd[2]=='f'&&cmd[3]=='_'&&
                        cmd[4]=='p'&&cmd[5]=='r'&&!cmd[6]);

    if (match_load)   { cmd_rlf_load(arg);         return 1; }
    if (match_status) { cmd_rlf_status();           return 1; }
    if (match_infer)  { cmd_rlf_infer(arg);         return 1; }
    if (match_pp)     { cmd_rlf_preprocess(arg);    return 1; }
    return 0; /* not an RLF command */
}
