/* rlf_backbone_bridge.c — Wires rlf_engine.c extern stubs to OOSI v3 runtime
 * ==========================================================================
 * This file is #included inside llama2_efi_final.c AFTER soma_loader.c
 * (which defines g_oosi_v3_ctx, g_oosi_v3_weights, g_oosi_v3_valid) and
 * AFTER rlf_engine.c is compiled as a separate object (linked at build time).
 *
 * The four functions below satisfy the `extern` declarations in rlf_engine.c:
 *   extern int  mamba_forward(void*, const uint32_t*, int, rlf_f32*, int);
 *   extern int  mamba_lm_head(void*, const rlf_f32*, uint32_t*, rlf_f32*);
 *   extern int  mamba_tokenize(void*, const char*, uint32_t*, int);
 *   extern void mamba_detokenize(void*, uint32_t, char*, int);
 *
 * Design notes:
 *  - `backbone` pointer is ignored — we use the global g_oosi_v3_ctx.
 *  - Forward runs N tokens sequentially via oosi_v3_forward_one(); the
 *    last hidden state is extracted from ctx->scratch (x_out slice).
 *  - lm_head performs argmax over ctx->logits[vocab_size].
 *  - Tokenize delegates to llmk_oo_infer_tokenize().
 *  - Detokenize delegates to llmk_oo_infer_decode_token().
 */

#include "../ssm/rlf_engine.h"
#include "../ssm/oosi_v3_infer.h"
#include "llmk_oo_infer.h"

/* These statics are defined in soma_loader.c (same TU after unity build) */
/* extern declarations so the compiler sees them before soma_loader include */
/* NOTE: no extern needed — same TU, already visible after soma_loader.c   */

/* ── mamba_forward ──────────────────────────────────────────────────────── */
/* Run prompt_ids[0..n_tokens-1] through the OOSI v3 backbone one token at
 * a time.  out_hidden receives the last-token hidden state (d_model floats).
 * The hidden state lives in ctx->scratch at offset 0 (x_out is the first
 * d_model floats written by oosi_v3_forward_one for the current token).
 */
int mamba_forward(
    void           *backbone  __attribute__((unused)),
    const uint32_t *tokens,
    int             n_tokens,
    rlf_f32        *out_hidden,
    int             d_model)
{
    if (!g_oosi_v3_valid || n_tokens <= 0 || !tokens || !out_hidden)
        return -1;

    /* Reset recurrent state so this is a fresh encode of the full sequence */
    oosi_v3_gen_ctx_reset(&g_oosi_v3_ctx);

    for (int t = 0; t < n_tokens; t++) {
        oosi_v3_forward_one(&g_oosi_v3_ctx, (int)tokens[t]);
    }

    /* x_out (last token hidden, post-norm, pre-lm_head) is at scratch[0..D-1]
     * The scratch layout per oosi_v3_infer.c:
     *   scratch[0        .. D-1]      = x_norm  (reused as x_out at end)
     *   scratch[D        .. D+2Di-1]  = x_and_z
     *   etc.
     * After forward_one completes, scratch[0..D-1] holds the output embedding.
     */
    const ssm_f32 *src = g_oosi_v3_ctx.scratch;
    for (int i = 0; i < d_model && i < g_oosi_v3_weights.d_model; i++)
        out_hidden[i] = src[i];

    return 0;
}

/* ── mamba_lm_head ──────────────────────────────────────────────────────── */
/* Argmax over ctx->logits (already populated by the last forward_one call).
 * Also returns the winning logit value.
 */
int mamba_lm_head(
    void           *backbone  __attribute__((unused)),
    const rlf_f32  *hidden    __attribute__((unused)),  /* logits already in ctx */
    uint32_t       *top1_token,
    rlf_f32        *top1_logit)
{
    if (!g_oosi_v3_valid || !top1_token || !top1_logit)
        return -1;

    const ssm_f32 *logits = g_oosi_v3_ctx.logits;
    int            vocab  = g_oosi_v3_weights.vocab_size;
    if (!logits || vocab <= 0) return -2;

    int best = 0;
    ssm_f32 best_val = logits[0];
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    *top1_token  = (uint32_t)best;
    *top1_logit  = best_val;
    return 0;
}

/* ── mamba_tokenize ─────────────────────────────────────────────────────── */
int mamba_tokenize(
    void       *backbone  __attribute__((unused)),
    const char *text,
    uint32_t   *ids,
    int         max_ids)
{
    if (!text || !ids || max_ids <= 0) return 0;
    /* llmk_oo_infer_tokenize returns int[] but we need uint32_t[].
     * GPT-NeoX token ids fit in 17 bits — safe to cast via int buffer. */
    static int tmp[RLF_MAX_PROMPT];
    int cap = (max_ids < RLF_MAX_PROMPT) ? max_ids : RLF_MAX_PROMPT;
    int n = llmk_oo_infer_tokenize(text, tmp, cap);
    if (n < 0) n = 0;
    for (int i = 0; i < n; i++) ids[i] = (uint32_t)tmp[i];
    return n;
}

/* ── mamba_detokenize ───────────────────────────────────────────────────── */
void mamba_detokenize(
    void     *backbone  __attribute__((unused)),
    uint32_t  id,
    char     *buf,
    int       len)
{
    if (!buf || len <= 0) return;
    buf[0] = '\0';
    llmk_oo_infer_decode_token((int)id, buf, len);
}
